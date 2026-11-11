"""Role-level Warm Layer: the complete chat transcript, retained by time only.

Three layers hold memory (see ``memoryreworkplan.md``):

* **Live Context** — the current turn's session events (still in the coordinator).
* **Warm Layer** — this module. Every complete, observable turn of the role, as
  full text, kept for a TTL (default 7 days) so recent conversation can be
  recalled verbatim.
* **Cold Layer** — knowledge/memory nodes and relations, extracted from a whole
  conversation batch once the batch's TTL has expired.

Invariants this module exists to guarantee:

* A turn has a Warm Layer body **at the end of the turn** (never "at the next
  2-minute checkpoint", never "when the pet closes the conversation").
* The Warm Layer belongs to the **role**, not to a session: ``close()``, a pet
  stop, a client disconnect, a role switch or a service restart must not delete
  un-evicted bodies.
* Retention is **time only**. There is no entry-count cap and no
  "drop the oldest when full" path anywhere in here.
* Eviction is per **complete ``batch_id``**: one batch is one extraction unit, so
  a conversation is never split across two extractions. A batch whose newest
  entry is still inside the TTL is not evictable at all.
* Bodies are deleted only after a confirmed publication (knowledge publish plus
  the diary/memory pass reached a terminal state). Any failure keeps the text
  and records ``retryable_failed``.

Storage layout, per role::

    data/graphiti/warm/<role_uuid>.jsonl        append-only entry journal
    data/graphiti/warm/<role_uuid>.vec.jsonl    append-only vector sidecar
    data/graphiti/warm/<role_uuid>.state.json   batch registry + extraction state

The journal is append-only with idempotent ids; the sidecar is a cache and never
authority (a missing vector falls back to lexical scoring, it never hides a
body). Compaction rewrites the journal without already-evicted batches and keeps
a deletion audit.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import replace_file, score_text, text_hash, utc_timestamp
from .warm_rag import search_windows
from .warm_entries import (
    _cosine,
    _embed,
    batch_cosine,
    normalize,
    pack_vector,
    unpack_vector,
)

#: Batch state machine. ``open`` collects turns, ``sealed`` is closed but still
#: retained, ``extraction_pending`` has a durable extraction job, ``published``
#: finished both passes, ``retryable_failed`` finished neither and must be retried.
BATCH_OPEN = "open"
BATCH_SEALED = "sealed"
BATCH_EXTRACTION_PENDING = "extraction_pending"
BATCH_PUBLISHED = "published"
BATCH_RETRYABLE_FAILED = "retryable_failed"

#: Per-entry extraction lifecycle inside the Warm Layer (``warm`` = not yet part
#: of any extraction, which is the normal state for a retained transcript).
EXTRACTION_WARM = "warm"
EXTRACTION_PENDING = "pending"
EXTRACTION_PUBLISHED = "published"
EXTRACTION_FAILED = "retryable_failed"

#: Sidecar vector state.
VECTOR_READY = "ready"
VECTOR_PENDING = "pending"
VECTOR_FAILED = "failed"
VECTOR_ABSENT = "absent"

#: Default TTL: a conversation batch is evicted seven days after its newest
#: entry. This is a *context-integrity* cost: a batch that is still being
#: appended to waits for its last entry too (see ``evictable_batches``).
DEFAULT_TTL_SECONDS = 7 * 24 * 3600.0

#: An ``open`` batch whose last output is this old is sealed by the sweeper. The
#: 1-hour window defines the batch boundary only; sealing never extracts.
DEFAULT_IDLE_SECONDS = 3600.0

#: Embedding budget for the synchronous path. ``embed_timeout_seconds`` is the
#: plan's ~300 ms; anything slower is written with ``vector_status=pending`` and
#: computed later by the background backfill, so a slow endpoint can never delay
#: a pet turn.
DEFAULT_EMBED_BUDGET_SECONDS = 0.3


class WarmStoreError(RuntimeError):
    """A Warm Layer write could not be made durable."""


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _append_line(path: Path, payload: Dict[str, Any]) -> None:
    """Append one durable line: write, flush and fsync before returning.

    A turn is only reported as recorded once its body is on disk, so the append
    must not be buffered: a crash right after ``record_turn`` must still find the
    transcript.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A torn trailing line is the only expected damage after a
                    # crash during append; skipping it keeps the rest readable.
                    continue
                if isinstance(row, dict) and row.get("id"):
                    rows.append(row)
    except OSError:
        return []
    return rows


def entry_id_for(role_uuid: str, event_id: str) -> str:
    """The durable Warm Layer id of one source event (idempotent by design)."""
    return f"warm:{role_uuid}:{event_id}"


class WarmStore:
    """Role-scoped durable transcript with batch identity and TTL eviction.

    One instance per role. Every public method is thread-safe: the service
    records turns from HTTP worker threads while the sweeper and the vector
    backfill run in their own threads.
    """

    def __init__(
        self,
        root: Path | str,
        role_uuid: str,
        *,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        reranker: Optional[Any] = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        embed_budget_seconds: float = DEFAULT_EMBED_BUDGET_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root) / "warm"
        self.role_uuid = self._check_role(role_uuid)
        self.embedder = embedder
        self.reranker = reranker
        self.ttl_seconds = float(ttl_seconds)
        self.idle_seconds = float(idle_seconds)
        self.embed_budget_seconds = max(0.0, float(embed_budget_seconds))
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._vectors: Dict[str, List[float]] = {}
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._state_version = 0
        self._loaded = False
        #: Source event ids already stored, so a retried turn is idempotent.
        self._by_event: Dict[str, str] = {}

    # ------------------------------------------------------------------ paths

    @staticmethod
    def _check_role(role_uuid: str) -> str:
        value = str(role_uuid or "")
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("invalid role UUID")
        return value

    @property
    def journal_path(self) -> Path:
        return self.root / f"{self.role_uuid}.jsonl"

    @property
    def vector_path(self) -> Path:
        return self.root / f"{self.role_uuid}.vec.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / f"{self.role_uuid}.state.json"

    @property
    def deletion_audit_path(self) -> Path:
        return self.root / f"{self.role_uuid}.evictions.jsonl"

    # ------------------------------------------------------------- lazy load

    def _load(self) -> None:
        """Lazy-load the role's files into memory (idempotent, once per store).

        Nothing here is on a hot path: a role with no retained transcript pays a
        single missing-file check.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            entries: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            by_event: Dict[str, str] = {}
            for index, row in enumerate(_read_jsonl(self.journal_path)):
                entry_id = str(row["id"])
                if entry_id in entries:
                    continue
                # ``_seq`` is the turn's write order. It is what makes "the batch in
                # conversation order" exact: several entries share one ``round_no``
                # (a user message, its assistant reply and its tool observations),
                # and ordering those by id would put the assistant reply first.
                row["_seq"] = int(row.get("_seq") or index)
                entries[entry_id] = row
                order.append(entry_id)
                event_id = str(row.get("event_id") or "")
                if event_id:
                    by_event[event_id] = entry_id
            vectors: Dict[str, List[float]] = {}
            for row in _read_jsonl(self.vector_path):
                try:
                    vectors[str(row["id"])] = unpack_vector(str(row.get("v") or ""))
                except (ValueError, TypeError):
                    continue
            self._entries = entries
            self._order = order
            self._by_event = by_event
            self._vectors = vectors
            state = self._read_state()
            self._batches = {
                str(key): dict(value)
                for key, value in (state.get("batches") or {}).items()
                if isinstance(value, dict)
            }
            self._state_version = int(state.get("version") or 0)
            self._loaded = True
            self._repair_vectors()

    def _read_state(self) -> Dict[str, Any]:
        path = self.state_path
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            # A corrupt registry must not hide the transcripts: the journal is
            # authority for bodies, the registry only for batch bookkeeping.
            return {}
        if not isinstance(value, dict):
            return {}
        if value.get("role_uuid") not in (None, self.role_uuid):
            # A file swapped between roles would mislabel every batch.
            raise WarmStoreError(
                f"warm state belongs to another role: {value.get('role_uuid')!r}"
            )
        return value

    def _repair_vectors(self) -> None:
        """Reconcile entry vector status with the sidecar (never drops a body)."""
        for entry_id, entry in self._entries.items():
            vector = self._vectors.get(entry_id)
            if vector:
                if entry.get("vector_status") != VECTOR_READY:
                    entry["vector_status"] = VECTOR_READY
            elif entry.get("vector_status") == VECTOR_READY:
                # Declared ready but the sidecar line is gone (truncated file):
                # mark it for backfill instead of trusting the claim.
                entry["vector_status"] = VECTOR_PENDING

    # ------------------------------------------------------------------ state

    def _save_state(self) -> None:
        self._state_version += 1
        payload = {
            "role_uuid": self.role_uuid,
            "version": self._state_version,
            "updated_at": utc_timestamp(),
            "batches": self._batches,
        }
        _atomic_write_json(self.state_path, payload)

    def _register_batch(
        self,
        batch_id: str,
        *,
        conversation_id: str,
        opened_at: str,
        last_output_at: str,
    ) -> Dict[str, Any]:
        batch = self._batches.get(batch_id)
        if batch is None:
            batch = {
                "batch_id": batch_id,
                "role_uuid": self.role_uuid,
                "conversation_ids": [],
                "opened_at": opened_at,
                "last_output_at": last_output_at,
                "sealed_at": None,
                "seal_reason": None,
                "status": BATCH_OPEN,
                "suspended_at": None,
                "extraction": None,
            }
            self._batches[batch_id] = batch
        conversations = batch.setdefault("conversation_ids", [])
        if conversation_id and conversation_id not in conversations:
            conversations.append(conversation_id)
        batch["last_output_at"] = last_output_at
        return batch

    # ----------------------------------------------------------------- append

    def _normalize_entries(self, entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            content = str(raw.get("content") or "")
            if not content.strip():
                continue
            event_id = str(raw.get("event_id") or raw.get("id") or "")
            if not event_id:
                event_id = uuid.uuid4().hex
            batch_id = str(raw.get("batch_id") or "")
            if not batch_id:
                raise WarmStoreError("a warm entry needs a batch_id")
            item = {
                "id": entry_id_for(self.role_uuid, event_id),
                "event_id": event_id,
                "role_uuid": self.role_uuid,
                "batch_id": batch_id,
                "conversation_id": str(raw.get("conversation_id") or ""),
                "round_no": int(raw.get("round_no") or 0),
                "role": str(raw.get("role") or "user"),
                "content": content,
                "created_at": str(raw.get("created_at") or utc_timestamp()),
                "content_hash": str(raw.get("content_hash") or text_hash(content)),
                "extracted_at": None,
                "extraction_status": EXTRACTION_WARM,
            }
            if raw.get("tool_name"):
                item["tool_name"] = str(raw["tool_name"])
            if isinstance(raw.get("source"), dict):
                item["source"] = dict(raw["source"])
            if isinstance(raw.get("metadata"), dict) and raw["metadata"]:
                item["metadata"] = dict(raw["metadata"])
            #: 发布侧凭据（工作记忆 message id / 批次 id / 层）逐条冻结：封批时要用
            #: 它把整批正文与它引用的来源对应起来，否则 cutover/归档都对不上。
            source_ids = [str(value) for value in (raw.get("source_ids") or []) if value]
            if source_ids:
                item["source_ids"] = source_ids
            if raw.get("source_batch_id"):
                item["source_batch_id"] = str(raw["source_batch_id"])
            if raw.get("source_layer"):
                item["source_layer"] = str(raw["source_layer"])
            normalized.append(item)
        return normalized

    def append(self, entries: Iterable[Dict[str, Any]]) -> List[str]:
        """Durably append complete entries; returns their Warm Layer ids.

        Idempotent on ``event_id``: re-appending the same source event returns
        the existing id and writes nothing. Raises when the body could not be
        made durable, so ``record_turn`` can keep its retry record instead of
        claiming success.
        """
        pending = self._normalize_entries(entries)
        if not pending:
            return []
        stored: List[str] = []
        with self._lock:
            self._load()
            for item in pending:
                existing = self._by_event.get(item["event_id"])
                if existing is not None:
                    stored.append(existing)
                    continue
                vector, vector_status = self._embed_bounded(item["content"])
                item["vector"] = None
                item["vector_status"] = vector_status
                item["_seq"] = len(self._order)
                try:
                    _append_line(self.journal_path, _body_row(item))
                except OSError as exc:
                    raise WarmStoreError(f"warm append failed: {exc}") from exc
                if vector is not None:
                    try:
                        _append_line(
                            self.vector_path,
                            {"id": item["id"], "d": len(vector), "v": pack_vector(vector)},
                        )
                    except OSError:
                        # The body is already durable; a missing vector line is a
                        # precision loss the backfill repairs, never a lost turn.
                        item["vector_status"] = VECTOR_PENDING
                        vector = None
                self._entries[item["id"]] = item
                self._order.append(item["id"])
                self._by_event[item["event_id"]] = item["id"]
                if vector is not None:
                    self._vectors[item["id"]] = vector
                self._register_batch(
                    item["batch_id"],
                    conversation_id=item["conversation_id"],
                    opened_at=item["created_at"],
                    last_output_at=item["created_at"],
                )
                stored.append(item["id"])
            self._save_state()
        return stored

    def _embed_bounded(self, text: str) -> Tuple[Optional[List[float]], str]:
        """Embedding under the write budget; returns ``(vector, status)``.

        Slow or failing embeddings are recorded as ``pending``/``failed`` next to
        the body so a recoverable background task can backfill them. The budget
        is advisory (a blocking HTTP call cannot be pre-empted from Python), so a
        slow endpoint may still exceed it; what the status guarantees is that an
        *outcome* (ready or pending) is always written with the body.
        """
        if self.embedder is None:
            return None, VECTOR_ABSENT
        started = time.monotonic()
        vector = _embed(self.embedder, text)
        elapsed = time.monotonic() - started
        if vector is None:
            return None, VECTOR_FAILED
        if elapsed > self.embed_budget_seconds > 0:
            # Still usable — we keep it, the status only records that the budget
            # was exceeded so diagnostics can see slow endpoints.
            return vector, VECTOR_READY
        return vector, VECTOR_READY

    # -------------------------------------------------------------- maintenance

    def pending_vectors(self, limit: int = 64) -> List[Dict[str, Any]]:
        """Entries whose vector must be (re)computed, oldest first."""
        with self._lock:
            self._load()
            rows = [
                entry
                for entry in self._order_entries()
                if entry.get("vector_status") in {VECTOR_PENDING, VECTOR_FAILED}
            ]
        return rows[: max(0, int(limit))]

    def fill_vector(self, entry_id: str, vector: Optional[Sequence[float]]) -> bool:
        """Attach a backfilled vector to an existing entry; body is untouched."""
        with self._lock:
            self._load()
            entry = self._entries.get(str(entry_id))
            if entry is None:
                return False
            if vector is None:
                entry["vector_status"] = VECTOR_FAILED
                self._save_state()
                return False
            values = [float(item) for item in vector]
            try:
                _append_line(
                    self.vector_path,
                    {"id": entry["id"], "d": len(values), "v": pack_vector(values)},
                )
            except OSError:
                entry["vector_status"] = VECTOR_PENDING
                self._save_state()
                return False
            self._vectors[entry["id"]] = values
            entry["vector_status"] = VECTOR_READY
            self._save_state()
            return True

    # ------------------------------------------------------------------ batch

    def batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._load()
            value = self._batches.get(str(batch_id))
            return dict(value) if value is not None else None

    def batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(value) for value in self._batches.values()]

    def open_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [
                dict(value)
                for value in self._batches.values()
                if value.get("status") == BATCH_OPEN
            ]

    def retained_batches(self) -> List[Dict[str, Any]]:
        """Batches whose transcript is still here (everything except published).

        A published batch keeps its registry row as the durable record that the
        extraction happened; it holds no bodies, so it is not "retained".
        """
        with self._lock:
            self._load()
            return [
                dict(value)
                for value in self._batches.values()
                if value.get("status") != BATCH_PUBLISHED
            ]

    def seal_batch(
        self,
        batch_id: str,
        *,
        sealed_at: Optional[str] = None,
        reason: str = "idle_1h",
    ) -> Optional[Dict[str, Any]]:
        """Seal one batch. Sealing **never** extracts and never deletes text."""
        with self._lock:
            self._load()
            batch = self._batches.get(str(batch_id))
            if batch is None:
                return None
            if batch.get("status") == BATCH_OPEN:
                batch["status"] = BATCH_SEALED
                batch["sealed_at"] = sealed_at or utc_timestamp()
                batch["seal_reason"] = str(reason)
                self._save_state()
            elif batch.get("seal_reason") is None:
                batch["seal_reason"] = str(reason)
                self._save_state()
            return dict(batch)

    def suspend_batch(self, batch_id: str, *, at: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Record that the pet/client went away. Never seals, extracts or deletes."""
        with self._lock:
            self._load()
            batch = self._batches.get(str(batch_id))
            if batch is None:
                return None
            batch["suspended_at"] = at or utc_timestamp()
            self._save_state()
            return dict(batch)

    def batch_rounds(self, batch_id: str) -> List[Dict[str, Any]]:
        """The batch's bodies in conversation order (the extraction transcript)."""
        with self._lock:
            self._load()
            rows = [
                entry
                for entry in self._order_entries()
                if entry.get("batch_id") == str(batch_id)
            ]
        rows.sort(key=lambda item: (int(item.get("round_no") or 0), int(item.get("_seq") or 0)))
        return rows

    def _order_entries(self) -> List[Dict[str, Any]]:
        return [self._entries[entry_id] for entry_id in self._order if entry_id in self._entries]

    # ------------------------------------------------------------- extraction

    def mark_extraction(
        self,
        batch_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Advance a batch's extraction state and stamp its entries.

        Entry-level ``extraction_status`` matters because eviction is only legal
        after a confirmed publication; leaving the entries at ``warm`` while the
        batch is ``extraction_pending`` is what makes a crash mid-publish
        recoverable.
        """
        with self._lock:
            self._load()
            batch = self._batches.get(str(batch_id))
            if batch is None:
                return None
            batch["status"] = str(status)
            record = dict(batch.get("extraction") or {})
            record.update(metadata or {})
            record["status"] = str(status)
            record["updated_at"] = utc_timestamp()
            batch["extraction"] = record
            if status == BATCH_EXTRACTION_PENDING:
                entry_status = EXTRACTION_PENDING
            elif status == BATCH_PUBLISHED:
                entry_status = EXTRACTION_PUBLISHED
            elif status == BATCH_RETRYABLE_FAILED:
                entry_status = EXTRACTION_FAILED
            else:
                entry_status = None
            if entry_status is not None:
                stamp = utc_timestamp()
                for entry in self._entries.values():
                    if entry.get("batch_id") != str(batch_id):
                        continue
                    entry["extraction_status"] = entry_status
                    if entry_status == EXTRACTION_PUBLISHED:
                        entry["extracted_at"] = stamp
            self._save_state()
            return dict(batch)

    def failed_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [
                dict(value)
                for value in self._batches.values()
                if value.get("status") == BATCH_RETRYABLE_FAILED
            ]

    def pending_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [
                dict(value)
                for value in self._batches.values()
                if value.get("status") == BATCH_EXTRACTION_PENDING
            ]

    def published_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [
                dict(value)
                for value in self._batches.values()
                if value.get("status") == BATCH_PUBLISHED
            ]

    def evictable_batches(
        self, now: Optional[float] = None, ttl: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Sealed batches whose **newest** entry is older than the TTL.

        The newest-entry rule is the whole batch-integrity cost: a batch that is
        still receiving turns is never evictable, and a conversation is never
        split into two extractions to make the TTL line up.
        """
        moment = self.clock() if now is None else float(now)
        window = self.ttl_seconds if ttl is None else float(ttl)
        with self._lock:
            self._load()
            newest: Dict[str, float] = {}
            for entry in self._order_entries():
                batch_id = str(entry.get("batch_id") or "")
                stamp = self._epoch(entry.get("created_at"))
                if stamp is None:
                    continue
                if stamp > newest.get(batch_id, float("-inf")):
                    newest[batch_id] = stamp
            due: List[Dict[str, Any]] = []
            for batch_id, batch in self._batches.items():
                if batch.get("status") not in {BATCH_SEALED, BATCH_RETRYABLE_FAILED}:
                    continue
                last = newest.get(batch_id)
                if last is None:
                    continue
                if moment - last >= window:
                    due.append({**dict(batch), "newest_entry_at": last, "age_seconds": moment - last})
        due.sort(key=lambda item: (item.get("newest_entry_at") or 0.0, str(item.get("batch_id"))))
        return due

    def idle_open_batches(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Open batches whose last output is older than the 1-hour window."""
        moment = self.clock() if now is None else float(now)
        with self._lock:
            self._load()
            last_seen: Dict[str, float] = {}
            for entry in self._order_entries():
                batch_id = str(entry.get("batch_id") or "")
                stamp = self._epoch(entry.get("created_at"))
                if stamp is None:
                    continue
                if stamp > last_seen.get(batch_id, float("-inf")):
                    last_seen[batch_id] = stamp
            due: List[Dict[str, Any]] = []
            for batch_id, batch in self._batches.items():
                if batch.get("status") != BATCH_OPEN:
                    continue
                stamp = last_seen.get(batch_id)
                if stamp is None:
                    continue
                if moment - stamp >= self.idle_seconds:
                    due.append({**dict(batch), "last_output_epoch": stamp, "idle_seconds": moment - stamp})
        due.sort(key=lambda item: (item.get("last_output_epoch") or 0.0, str(item.get("batch_id"))))
        return due

    def evict(self, batch_id: str, *, reason: str = "ttl_expired") -> List[str]:
        """Remove a published batch's bodies and compact the journal.

        Callers must have confirmed the publication first (``mark_extraction``
        with ``published``); this method only enforces the state condition and
        writes the deletion audit.
        """
        with self._lock:
            self._load()
            batch = self._batches.get(str(batch_id))
            if batch is None:
                return []
            if batch.get("status") != BATCH_PUBLISHED:
                raise WarmStoreError(
                    f"refusing to evict {batch_id}: status={batch.get('status')!r}"
                )
            removed = [
                entry["id"]
                for entry in self._order_entries()
                if entry.get("batch_id") == str(batch_id)
            ]
            if not removed:
                return []
            dropped = set(removed)
            self._entries = {
                key: value for key, value in self._entries.items() if key not in dropped
            }
            self._order = [key for key in self._order if key not in dropped]
            self._by_event = {
                key: value for key, value in self._by_event.items() if value not in dropped
            }
            self._vectors = {
                key: value for key, value in self._vectors.items() if key not in dropped
            }
            _atomic_write_json(
                self.state_path,
                {
                    "role_uuid": self.role_uuid,
                    "version": self._state_version + 1,
                    "updated_at": utc_timestamp(),
                    "batches": self._batches,
                },
            )
            self._state_version += 1
            self._compact(removed, reason=reason)
            return removed

    def _compact(self, removed: Sequence[str], *, reason: str) -> None:
        """Rewrite journal + sidecar without evicted ids; keep a deletion audit.

        Bodies are appended for a whole TTL and deleted once, so a full rewrite at
        deletion time is cheaper than any incremental scheme and keeps the
        crash-safety story simple: the rewrite is atomic (temp + replace), and a
        crash before the replace leaves the old journal intact — the batch is
        still ``published``, so the next sweep re-runs the deletion.
        """
        dropped = set(removed)
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.journal_path.name}.", suffix=".tmp", dir=str(self.root)
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for entry_id in self._order:
                    entry = self._entries.get(entry_id)
                    if entry is None:
                        continue
                    handle.write(json.dumps(_body_row(entry), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, self.journal_path)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.vector_path.name}.", suffix=".tmp", dir=str(self.root)
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for entry_id in self._order:
                    vector = self._vectors.get(entry_id)
                    if not vector:
                        continue
                    handle.write(
                        json.dumps(
                            {"id": entry_id, "d": len(vector), "v": pack_vector(vector)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, self.vector_path)
            _append_line(
                self.deletion_audit_path,
                {
                    "at": utc_timestamp(),
                    "role_uuid": self.role_uuid,
                    "batch_id": None,
                    "reason": reason,
                    "removed_ids": list(removed),
                    "removed": len(removed),
                },
            )
        except OSError as exc:
            raise WarmStoreError(f"warm compaction failed: {exc}") from exc

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        gate: float = 0.15,
        relative_floor_ratio: float = 0.0,
        current_batch_id: Optional[str] = None,
        current_round_no: Optional[int] = None,
        include_assistant: bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
        query_vector: Optional[Sequence[float]] = None,
        raw_rag: bool = False,
        rerank: bool = False,
        chunk_tokens: int = 256,
        chunk_overlap: int = 32,
        dense_top_k: int = 128,
        rerank_top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Recall retained bodies with the Warm Layer's own similarity floor.

        Admission is this layer's floor (a vector cosine or its lexical
        fallback), so the layer cannot be admitted by an empty graph score, and
        a body without a vector still recalls lexically instead of vanishing.

        ``relative_floor_ratio`` cuts the tail *inside* this layer: only fragments
        scoring at least that fraction of the layer's best hit are returned. The
        best hit itself can never be cut (``ratio <= 1``), so tightening the layer
        cannot turn "something relevant was found" into zero recall.
        """
        text = str(query or "").strip()
        with self._lock:
            self._load()
            entries = [dict(entry) for entry in self._order_entries()]
            vectors = dict(self._vectors)
        if not text or not entries:
            return []
        floor = max(0.0, float(gate))
        if query_vector is None:
            query_vector = _embed(self.embedder, text)
        if query_vector is not None:
            query_vector = normalize([float(item) for item in query_vector])
        if raw_rag:
            eligible: List[Dict[str, Any]] = []
            for entry in entries:
                if not include_assistant and str(entry.get("role") or "") == "assistant":
                    continue
                if current_batch_id and str(entry.get("batch_id") or "") == str(current_batch_id):
                    if current_round_no is None or int(entry.get("round_no") or 0) == int(current_round_no):
                        continue
                eligible.append(entry)
            ranked_windows = search_windows(
                text,
                entries=eligible,
                vectors=vectors,
                query_vector=query_vector,
                gate=floor,
                top_k=top_k,
                relative_floor_ratio=relative_floor_ratio,
                reranker=self.reranker if rerank else None,
                dense_top_k=dense_top_k,
                rerank_top_k=rerank_top_k,
                chunk_tokens=chunk_tokens,
                overlap_tokens=chunk_overlap,
                diagnostics=diagnostics,
            )
            for item in ranked_windows:
                item["created_epoch"] = self._epoch(item.get("created_at"))
            return self._dedupe(ranked_windows, top_k)
        scores = (
            batch_cosine(query_vector, [vectors.get(entry["id"]) or [] for entry in entries])
            if query_vector is not None
            else [0.0 for _ in entries]
        )
        ranked: List[Dict[str, Any]] = []
        for position, entry in enumerate(entries):
            if not include_assistant and str(entry.get("role") or "") == "assistant":
                continue
            if current_batch_id and str(entry.get("batch_id") or "") == str(current_batch_id):
                if current_round_no is None or int(entry.get("round_no") or 0) == int(current_round_no):
                    continue
            lexical = float(score_text(text, entry.get("content", "")) or 0.0)
            semantic = float(scores[position] or 0.0)
            if vectors.get(entry["id"]):
                similarity = max(semantic, lexical)
                source = "warm_embedding" if semantic >= lexical else "warm_lexical"
            else:
                # No vector: the lexical score *is* the similarity. Reporting it
                # as such is what keeps the fallback explainable.
                similarity = lexical
                source = "warm_lexical_fallback"
            if similarity < floor:
                continue
            ranked.append(
                {
                    "id": entry["id"],
                    "layer": "WarmLayer",
                    "content": entry.get("content", ""),
                    "source": "warm_layer",
                    "source_type": "warm_layer",
                    "match_type": "direct",
                    "similarity": round(similarity, 6),
                    "final_score": round(similarity, 6),
                    "score_source": source,
                    "semantic_score": round(semantic, 6),
                    "lexical_score": round(lexical, 6),
                    "rerank_score": None,
                    "content_hash": entry.get("content_hash"),
                    "batch_id": entry.get("batch_id"),
                    "conversation_id": entry.get("conversation_id"),
                    "round_no": entry.get("round_no"),
                    "role": entry.get("role"),
                    "created_at": entry.get("created_at"),
                    "created_epoch": self._epoch(entry.get("created_at")),
                    "token_estimate": len(str(entry.get("content") or "")),
                }
            )
        ranked = self._apply_relative_floor(ranked, relative_floor_ratio, diagnostics)
        return self._dedupe(ranked, top_k)

    @staticmethod
    def _apply_relative_floor(
        ranked: List[Dict[str, Any]],
        relative_floor_ratio: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Keep only hits close to this layer's best one (never drops the best).

        This is the "stricter recent-chat recall" knob: an absolute floor alone
        admits everything above it, which on a long transcript is a long tail of
        barely-related lines. The best hit always survives, so the tightening can
        never be the reason a query that found *something* gets nothing.
        """
        ratio = float(relative_floor_ratio or 0.0)
        if ratio <= 0.0 or not ranked:
            return ranked
        ratio = min(ratio, 1.0)
        best = max(float(item.get("similarity") or 0.0) for item in ranked)
        cutoff = best * ratio
        kept = [item for item in ranked if float(item.get("similarity") or 0.0) >= cutoff]
        if diagnostics is not None:
            diagnostics["relative_floor_ratio"] = ratio
            diagnostics["relative_floor_cutoff"] = round(cutoff, 6)
            diagnostics["relative_floor_dropped"] = len(ranked) - len(kept)
        return kept

    @staticmethod
    def _dedupe(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """One hit per ``(batch_id, round_no)``, then one per ``content_hash``."""
        best: Dict[Any, Dict[str, Any]] = {}
        order: List[Any] = []
        for item in items:
            key: Any = (item.get("batch_id"), item.get("round_no"))
            if key == (None, None):
                key = item.get("id")
            existing = best.get(key)
            if existing is None:
                best[key] = item
                order.append(key)
            elif float(item.get("final_score") or 0.0) > float(existing.get("final_score") or 0.0):
                best[key] = item
        by_hash: Dict[str, Dict[str, Any]] = {}
        hash_order: List[str] = []
        for key in order:
            item = best[key]
            digest = str(item.get("content_hash") or item.get("id"))
            existing = by_hash.get(digest)
            if existing is None:
                by_hash[digest] = item
                hash_order.append(digest)
            elif float(item.get("final_score") or 0.0) > float(existing.get("final_score") or 0.0):
                by_hash[digest] = item
        selected = [by_hash[digest] for digest in hash_order]
        selected.sort(
            key=lambda item: (-float(item.get("final_score") or 0.0), str(item.get("created_at") or ""), str(item.get("id")))
        )
        return selected[: max(0, int(top_k))]

    # ------------------------------------------------------------------ stats

    def stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        moment = self.clock() if now is None else float(now)
        with self._lock:
            self._load()
            entries = self._order_entries()
            oldest = None
            for entry in entries:
                stamp = self._epoch(entry.get("created_at"))
                if stamp is None:
                    continue
                if oldest is None or stamp < oldest:
                    oldest = stamp
            counts: Dict[str, int] = {}
            for batch in self._batches.values():
                key = str(batch.get("status") or "unknown")
                counts[key] = counts.get(key, 0) + 1
            return {
                "role_uuid": self.role_uuid,
                "entries": len(entries),
                "oldest_age_seconds": None if oldest is None else round(moment - oldest, 3),
                "missing_vectors": sum(
                    1 for entry in entries if not self._vectors.get(entry["id"])
                ),
                "pending_vectors": sum(
                    1
                    for entry in entries
                    if entry.get("vector_status") in {VECTOR_PENDING, VECTOR_FAILED}
                ),
                "batches": len(self._batches),
                "batch_states": counts,
                "pending_batches": counts.get(BATCH_EXTRACTION_PENDING, 0),
                "open_batches": counts.get(BATCH_OPEN, 0),
                "sealed_batches": counts.get(BATCH_SEALED, 0),
                "published_batches": counts.get(BATCH_PUBLISHED, 0),
                "failed_batches": counts.get(BATCH_RETRYABLE_FAILED, 0),
                "ttl_seconds": self.ttl_seconds,
                "idle_seconds": self.idle_seconds,
            }

    def entries_for_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [
                dict(entry)
                for entry in self._order_entries()
                if entry.get("batch_id") == str(batch_id)
            ]

    def all_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(entry) for entry in self._order_entries()]

    def snapshot(self) -> Dict[str, Any]:
        """Everything the management UI needs, with vectors stripped."""
        with self._lock:
            self._load()
            entries = [
                {key: value for key, value in entry.items() if key not in {"vector", "_seq"}}
                for entry in self._order_entries()
            ]
            counts: Dict[str, int] = {}
            rounds: Dict[str, set] = {}
            for entry in entries:
                batch_id = str(entry.get("batch_id") or "")
                counts[batch_id] = counts.get(batch_id, 0) + 1
                rounds.setdefault(batch_id, set()).add(int(entry.get("round_no") or 0))
            batches = []
            for value in self._batches.values():
                batch = dict(value)
                batch_id = str(batch.get("batch_id") or "")
                batch["entry_count"] = counts.get(batch_id, 0)
                batch["round_count"] = len(rounds.get(batch_id, set()))
                batches.append(batch)
            return {
                "kind": "warm_layer",
                "role_uuid": self.role_uuid,
                "entries": entries,
                "batches": batches,
                "stats": self.stats(),
            }

    # ------------------------------------------------------------------ misc

    @staticmethod
    def _epoch(value: Any) -> Optional[float]:
        """Parse a persisted UTC ISO-8601 timestamp into a Unix epoch."""
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            from datetime import datetime, timezone

            moment = datetime.fromisoformat(normalized)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()

    def clear(self) -> None:
        """Delete the role's whole warm store. Test/maintenance helper only.

        This is deliberately not reachable from any session lifecycle path: the
        invariant is that closing a session can never clear the Warm Layer.
        """
        with self._lock:
            for path in (
                self.journal_path,
                self.vector_path,
                self.state_path,
                self.deletion_audit_path,
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
            self._entries = {}
            self._order = []
            self._vectors = {}
            self._by_event = {}
            self._batches = {}
            self._loaded = True


def _body_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The persisted row: everything except the vector (which has a sidecar)."""
    return {key: value for key, value in entry.items() if key != "vector"}


def build_warm_embedder(embedder: Optional[Callable[[str], Sequence[float]]]) -> Any:
    """Adapter kept for symmetry with ``build_temp_graph_extractor``."""
    return embedder


__all__ = [
    "WarmStore",
    "WarmStoreError",
    "BATCH_OPEN",
    "BATCH_SEALED",
    "BATCH_EXTRACTION_PENDING",
    "BATCH_PUBLISHED",
    "BATCH_RETRYABLE_FAILED",
    "EXTRACTION_WARM",
    "EXTRACTION_PENDING",
    "EXTRACTION_PUBLISHED",
    "EXTRACTION_FAILED",
    "VECTOR_READY",
    "VECTOR_PENDING",
    "VECTOR_FAILED",
    "VECTOR_ABSENT",
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_IDLE_SECONDS",
    "DEFAULT_EMBED_BUDGET_SECONDS",
]
