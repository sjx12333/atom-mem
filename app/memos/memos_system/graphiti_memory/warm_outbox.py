"""Role-level durable outbox for Warm Layer writes.

The Warm Layer must have a body for every complete turn **at the end of that
turn**, and ``record_turn`` must never report success while the body only exists
in memory. Two writes are involved (the session state file and the Warm Layer
journal) and they can fail independently, so the order is:

1. persist the session events *and* the pending warm record together,
2. attempt the idempotent ``WarmStore.append``,
3. drop the outbox record only on success; keep it otherwise,
4. report the turn as recorded only once the body is on disk or durably queued.

The sweeper (``sweep_warm_outbox``) retries whatever is left here, so a crashed
or failed write is repaired instead of silently producing "memory starts
mid-conversation".

Layout: ``<root>/warm_outbox/<role_uuid>.json`` — one JSON array per role, since
the unit of work is small (one turn) and the file is rewritten only when a turn
fails or is repaired.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .domain import replace_file, utc_timestamp

OUTBOX_DIR = "warm_outbox"


class WarmOutbox:
    """File-backed retry queue for Warm Layer writes, one file per role."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, role_uuid: str) -> Path:
        value = str(role_uuid or "")
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("invalid role UUID")
        return self.root / OUTBOX_DIR / f"{value}.json"

    # -------------------------------------------------------------------- io

    def _read(self, role_uuid: str) -> List[Dict[str, Any]]:
        path = self._path(role_uuid)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):
            records = data.get("records")
            return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _write(self, role_uuid: str, records: List[Dict[str, Any]]) -> None:
        path = self._path(role_uuid)
        if not records:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------- api

    @staticmethod
    def _key(record: Dict[str, Any]) -> str:
        return str(record.get("turn_key") or "")

    def save(self, role_uuid: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Persist one turn's warm work (idempotent on ``turn_key``)."""
        key = self._key(record)
        if not key:
            raise ValueError("warm outbox record needs a turn_key")
        records = self._read(role_uuid)
        for index, current in enumerate(records):
            if self._key(current) == key:
                merged = {**current, **record}
                merged["attempts"] = int(current.get("attempts") or 0)
                records[index] = merged
                self._write(role_uuid, records)
                return merged
        record = {**record, "attempts": int(record.get("attempts") or 0)}
        records.append(record)
        self._write(role_uuid, records)
        return record

    def remove(self, role_uuid: str, key: str) -> bool:
        records = self._read(role_uuid)
        kept = [item for item in records if self._key(item) != str(key)]
        if len(kept) == len(records):
            return False
        self._write(role_uuid, kept)
        return True

    def list(self, role_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Pending warm work, oldest first (stable replay order)."""
        base = self.root / OUTBOX_DIR
        if not base.exists():
            return []
        if role_uuid is not None:
            roles = [str(role_uuid)]
        else:
            roles = [path.stem for path in sorted(base.glob("*.json"))]
        records: List[Dict[str, Any]] = []
        for role in roles:
            records.extend(self._read(role))
        records.sort(key=lambda item: (str(item.get("created_at") or ""), self._key(item)))
        return records

    def pending_batches(self, role_uuid: str) -> Dict[str, int]:
        """Round counts per batch for the turns still waiting in this outbox.

        ``round_no`` must be allocated by the Warm Layer registry rather than by
        each store, so the next number has to account for queued-but-unwritten
        turns as well: otherwise two turns written after one failure would both
        claim the same round.
        """
        counts: Dict[str, int] = {}
        for record in self._read(role_uuid):
            batch_id = str(record.get("batch_id") or "")
            if batch_id:
                counts[batch_id] = counts.get(batch_id, 0) + 1
        return counts

    def retry(
        self,
        role_uuid: str,
        append: Callable[[List[Dict[str, Any]]], Any],
        *,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Replay queued turns through ``append``; drop each one on success."""
        repaired: List[str] = []
        failed: List[Dict[str, Any]] = []
        for record in self._read(role_uuid)[: max(0, int(limit))]:
            entries = [item for item in (record.get("entries") or []) if isinstance(item, dict)]
            if not entries:
                self.remove(role_uuid, self._key(record))
                continue
            try:
                append(entries)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the sweep
                attempts = int(record.get("attempts") or 0) + 1
                self.save(
                    role_uuid,
                    {
                        **record,
                        "attempts": attempts,
                        "last_error": str(exc),
                        "last_attempt_at": utc_timestamp(),
                    },
                )
                failed.append({"turn_key": self._key(record), "error": str(exc), "attempts": attempts})
                continue
            self.remove(role_uuid, self._key(record))
            repaired.append(self._key(record))
        return {"repaired": repaired, "failed": failed}

    def clear(self, role_uuid: str) -> None:
        self._write(role_uuid, [])


def outbox_turn_key(role_uuid: str, batch_id: str, round_no: int, event_ids: Iterable[str]) -> str:
    """Stable identity of one turn's warm work (same turn ⇒ same key)."""
    joined = ",".join(str(item) for item in event_ids)
    return f"{role_uuid}:{batch_id}:{round_no}:{joined}"


__all__ = ["WarmOutbox", "OUTBOX_DIR", "outbox_turn_key"]
