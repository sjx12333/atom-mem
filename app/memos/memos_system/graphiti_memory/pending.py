"""Durable pending-publication records: freeze first, extract later, clean up after.

A sealed conversation batch is written here **before** it is handed to the
background extraction queue. That record is the only thing that lets a batch
survive a backend restart between "the conversation was closed" and "the
extraction was published": the queue lives in memory, and the archived
conversation content is only deleted after a successful publication
(ADR-0128, ADR-0147).

Layout: ``<root>/pending_publications/<role_uuid>/<batch_id>.json`` holding
the frozen snapshot exactly as the saga consumes it (including its
``operation_id``), so a replay re-runs the same publication instead of creating a
second one.

The record is keyed by **batch identity** (``batch_id``), not by conversation:
several chat batches share one conversation (the pet keeps the conversation
across the 1-hour batch boundary), and a conversation-keyed file made the second
batch overwrite the first — one record for two frozen publications, so a restart
between freeze and publish could only replay one of them. A snapshot without a
``batch_id`` (a legacy session close) keeps the old conversation-scoped name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .authority import AuthorityStore

PENDING_DIR = "pending_publications"


class PendingPublicationStore:
    """File-backed record of batches sealed but not yet published."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.authority = AuthorityStore(self.root)

    # ------------------------------------------------------------------ paths

    @staticmethod
    def key_of(snapshot: Dict[str, Any]) -> str:
        """The record's file key: the batch when there is one, else the conversation."""
        return str(snapshot.get("batch_id") or snapshot.get("conversation_id") or "").strip()

    @staticmethod
    def _name(role_uuid: str, key: str) -> str:
        return f"{PENDING_DIR}/{role_uuid}/{key}.json"

    def path(self, role_uuid: str, key: str) -> Path:
        return self.root / self._name(role_uuid, key)

    def path_for(self, snapshot: Dict[str, Any]) -> Path:
        return self.path(str(snapshot.get("role_uuid") or ""), self.key_of(snapshot))

    # ------------------------------------------------------------------- io

    def save(self, snapshot: Dict[str, Any]) -> str:
        """Write (or overwrite) the frozen batch; returns its root-relative name."""
        role_uuid = str(snapshot.get("role_uuid") or "")
        conversation_id = str(snapshot.get("conversation_id") or "")
        key = self.key_of(snapshot)
        if not role_uuid or not key:
            raise ValueError("pending publication needs role_uuid and a batch/conversation id")
        name = self._name(role_uuid, key)
        self.authority.save_root_file(name, snapshot)
        return name

    def load(self, role_uuid: str, key: str) -> Dict[str, Any]:
        record = self.authority.load_root_file(self._name(role_uuid, key), {})
        return record if isinstance(record, dict) else {}

    def exists(self, role_uuid: str, key: str) -> bool:
        return bool(self.load(role_uuid, key))

    def delete(self, role_uuid: str, key: str) -> bool:
        path = self.path(role_uuid, key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------- snapshot-shaped access
    #
    # Callers hold the snapshot, not its key: these spell "the record of *this*
    # frozen batch" once instead of every call site re-deriving the identity
    # (which is how a batch got deleted through another batch's conversation).

    def load_for(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return self.load(str(snapshot.get("role_uuid") or ""), self.key_of(snapshot))

    def exists_for(self, snapshot: Dict[str, Any]) -> bool:
        return bool(self.load_for(snapshot))

    def delete_for(self, snapshot: Dict[str, Any]) -> bool:
        return self.delete(str(snapshot.get("role_uuid") or ""), self.key_of(snapshot))

    def list(self, role_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every pending batch, oldest first (stable by name for replay order)."""
        base = self.root / PENDING_DIR
        if not base.exists():
            return []
        pattern = f"{role_uuid}/*.json" if role_uuid else "*/*.json"
        records: List[Dict[str, Any]] = []
        for path in sorted(base.glob(pattern)):
            relative = path.relative_to(self.root).as_posix()
            record = self.authority.load_root_file(relative, {})
            if isinstance(record, dict) and (
                record.get("conversation_id") or record.get("batch_id")
            ):
                records.append(record)
        return records
