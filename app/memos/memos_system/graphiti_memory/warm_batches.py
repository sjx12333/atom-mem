"""Chat-batch identity and the 7-day TTL extraction path.

``conversation_id`` is a client session identifier and cannot be a chat-batch
identity: the pet mints a new one on restart, inherits an old one, and the
backend may reassign it mid-turn. This module owns the real batch identity
(``batch_id``) plus the only two transitions that touch the Cold Layer:

* **seal** — the 1-hour idle window (or a hard boundary such as an explicit end,
  a role switch or a service restart) closes a batch. Sealing never extracts and
  never deletes a body.
* **sweep** — a sealed batch whose newest body is older than the TTL is frozen
  into a ``ConversationSnapshot`` and handed to the existing
  ``ExtractionPipeline.publish`` and diary path. Only a confirmed publication
  makes the batch evictable.

The extraction itself is *not* reimplemented here: ``extract_batch`` is the one
call site that reuses the service's existing publish/diary implementation, so the
eviction path cannot drift from the formal knowledge path.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from .domain import ConversationSnapshot, utc_timestamp
from .warm_store import (
    BATCH_EXTRACTION_PENDING,
    BATCH_OPEN,
    BATCH_PUBLISHED,
    BATCH_RETRYABLE_FAILED,
    BATCH_SEALED,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_TTL_SECONDS,
    WarmStore,
)

#: Terminal outcomes of the two extraction passes. A batch may only be evicted
#: when the knowledge pass *and* the diary/memory pass reached one of these.
#: ``skipped`` is the diary path's explicit "configuration disabled this pass"
#: terminal state (``diary.py``), which is exactly the plan's "明确的配置禁用终态";
#: ``failed`` is deliberately absent — a failed diary keeps the transcript.
PUBLISH_OK = {"published", "empty", "disabled", "skipped", "written"}


class WarmBatchRegistry:
    """Mint and advance ``batch_id``\\ s for one role.

    A batch is opened by the first complete turn and collects every following
    turn until a boundary seals it. The 1-hour idle window is one such boundary;
    it defines *where* a batch ends, never what happens to the transcript.
    """

    def __init__(
        self,
        store: WarmStore,
        *,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.idle_seconds = float(idle_seconds)
        self.clock = clock
        self._lock = threading.RLock()
        self._current: Optional[str] = None
        self._last_output_epoch: float = 0.0

    # ------------------------------------------------------------------ state

    def _restore(self) -> None:
        """Adopt the role's newest open batch after a restart.

        The batch belongs to the role, so a service restart must not orphan it:
        the next turn continues the open batch if its idle window has not
        elapsed, and seals it first when it has (plan 阶段 2.3).
        """
        if self._current is not None:
            return
        candidates = self.store.open_batches()
        if not candidates:
            return
        newest = max(
            candidates,
            key=lambda item: (str(item.get("last_output_at") or ""), str(item.get("batch_id"))),
        )
        self._current = str(newest.get("batch_id"))
        self._last_output_epoch = self._epoch(newest.get("last_output_at")) or 0.0

    def _epoch(self, value: Any) -> Optional[float]:
        return WarmStore._epoch(value)

    def current_batch_id(self) -> Optional[str]:
        with self._lock:
            self._restore()
            return self._current

    def adopt(self, batch_id: str) -> Optional[str]:
        """Claim an existing batch as the current one (a caller-supplied claim).

        Used when the same complete turn reaches the backend twice through
        different channels: the turn must land in the batch it belongs to instead
        of silently opening a second one.
        """
        value = str(batch_id or "").strip()
        if not value:
            return None
        with self._lock:
            batch = self.store.batch(value)
            if batch is None:
                return None
            if batch.get("status") != BATCH_OPEN:
                return None
            self._current = value
            stamp = self._epoch(batch.get("last_output_at"))
            if stamp is not None:
                self._last_output_epoch = max(self._last_output_epoch, stamp)
            return value

    # -------------------------------------------------------------- allocation

    def batch_for_turn(
        self,
        conversation_id: str,
        *,
        now: Optional[float] = None,
        reason_on_reopen: str = "reopen_after_idle",
    ) -> Dict[str, Any]:
        """The batch this turn belongs to, sealing a stale one first.

        Returns ``{"batch_id", "opened", "sealed", "seal_reason"}`` so the caller
        can log the boundary without re-deriving it.
        """
        moment = self.clock() if now is None else float(now)
        with self._lock:
            self._restore()
            sealed: Optional[str] = None
            seal_reason: Optional[str] = None
            if self._current is not None:
                batch = self.store.batch(self._current)
                idle = moment - float(self._last_output_epoch or 0.0)
                if batch is not None and batch.get("status") != BATCH_OPEN:
                    # The batch is known and no longer open (sealed/published):
                    # the next turn opens a new one.
                    self._current = None
                elif batch is None and self.store.batches():
                    # The registry knows other batches but not this one — the
                    # current id was never persisted, so it is not usable.
                    self._current = None
                elif self._last_output_epoch and idle >= self.idle_seconds:
                    self.store.seal_batch(self._current, reason=reason_on_reopen)
                    sealed = self._current
                    seal_reason = reason_on_reopen
                    self._current = None
            opened = False
            if self._current is None:
                self._current = f"batch_{uuid.uuid4().hex}"
                opened = True
            self._last_output_epoch = moment
            return {
                "batch_id": self._current,
                "opened": opened,
                "sealed": sealed,
                "seal_reason": seal_reason,
                "conversation_id": str(conversation_id or ""),
            }

    def note_output(self, now: Optional[float] = None) -> None:
        """Re-anchor the idle window (called when the reply finished speaking)."""
        with self._lock:
            self._last_output_epoch = self.clock() if now is None else float(now)

    def seal_current(self, reason: str) -> Optional[str]:
        """Seal the current batch and start a new one on the next turn."""
        with self._lock:
            self._restore()
            if self._current is None:
                return None
            self.store.seal_batch(self._current, reason=reason)
            sealed = self._current
            self._current = None
            return sealed


class WarmBatchExtractor:
    """The 7-day TTL sweep: seal idle batches, extract expired ones, evict on success.

    Holds no transcript of its own — ``WarmStore`` is the only body store — and
    delegates the actual extraction to an injected ``extract`` callable, which is
    the service's existing publish + diary path.
    """

    def __init__(
        self,
        store: WarmStore,
        *,
        extract: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        audit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.store = store
        self.extract = extract
        self.ttl_seconds = float(ttl_seconds)
        self.clock = clock
        self.audit = audit
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()

    # ------------------------------------------------------------------ sweep

    def sweep(self, *, now: Optional[float] = None, limit: int = 8) -> Dict[str, Any]:
        """Advance every due batch for this role once.

        Only the TTL branch extracts. Sealing the 1-hour boundary is deliberately
        separate (``seal_idle``), so an idle conversation can be sealed without
        ever being sent for extraction.
        """
        moment = self.clock() if now is None else float(now)
        published: List[Dict[str, Any]] = []
        requeued: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        skipped: List[str] = []

        # Already-failed batches are retried here: the failure recorded a reason
        # and left the bodies in place, and the next sweep is the retry.
        for batch in self.store.failed_batches():
            if len(published) + len(requeued) + len(failed) >= max(1, int(limit)):
                break
            outcome = self._process(str(batch.get("batch_id")), moment)
            if outcome is None:
                skipped.append(str(batch.get("batch_id")))
            elif outcome.get("status") == "published":
                published.append(outcome)
            elif outcome.get("status") == "failed":
                failed.append(outcome)
            else:
                requeued.append(outcome)

        for batch in self.store.evictable_batches(now=moment, ttl=self.ttl_seconds):
            if len(published) + len(requeued) + len(failed) >= max(1, int(limit)):
                break
            batch_id = str(batch.get("batch_id"))
            if self.store.batch(batch_id) is None:
                continue
            if self.store.batch(batch_id).get("status") == BATCH_EXTRACTION_PENDING:
                # A pending batch was already frozen; do not re-freeze it here.
                requeued.append({"batch_id": batch_id, "status": "pending"})
                continue
            outcome = self._process(batch_id, moment)
            if outcome is None:
                skipped.append(batch_id)
            elif outcome.get("status") == "published":
                published.append(outcome)
            elif outcome.get("status") == "failed":
                failed.append(outcome)
            else:
                requeued.append(outcome)
        return {
            "role_uuid": self.store.role_uuid,
            "published": published,
            "requeued": requeued,
            "failed": failed,
            "skipped": skipped,
        }

    def seal_idle(self, *, now: Optional[float] = None) -> List[str]:
        """Seal open batches past the idle window. Never extracts, never deletes."""
        sealed: List[str] = []
        for batch in self.store.idle_open_batches(now=now):
            batch_id = str(batch.get("batch_id"))
            result = self.store.seal_batch(batch_id, reason="idle_1h")
            if result is not None:
                sealed.append(batch_id)
        return sealed

    # -------------------------------------------------------------- one batch

    def build_snapshot(self, batch_id: str, *, reason: str = "ttl_expired") -> Dict[str, Any]:
        """Freeze a batch's whole transcript into the existing snapshot shape."""
        entries = self.store.batch_rounds(batch_id)
        events = [
            {
                "event_id": str(entry.get("event_id") or entry.get("id")),
                "role": str(entry.get("role") or "user"),
                "content": str(entry.get("content") or ""),
                "timestamp": str(entry.get("created_at") or utc_timestamp()),
                "complete": True,
                "observable": True,
                "tool_name": entry.get("tool_name"),
                "metadata": dict(entry.get("metadata") or {}),
                "source": dict(entry["source"]) if isinstance(entry.get("source"), dict) else None,
                "round_no": int(entry.get("round_no") or 0),
                "batch_id": batch_id,
            }
            for entry in entries
        ]
        batch = self.store.batch(batch_id) or {}
        conversations = [str(item) for item in (batch.get("conversation_ids") or []) if item]
        snapshot = ConversationSnapshot(
            role_uuid=self.store.role_uuid,
            conversation_id=conversations[0] if conversations else batch_id,
            reason=reason,
            events=events,
            created_at=utc_timestamp(),
        ).to_dict()
        snapshot["batch_id"] = batch_id
        snapshot["conversation_ids"] = conversations
        snapshot["seal_reason"] = batch.get("seal_reason")
        snapshot["source_layer"] = "WarmLayer"
        # 发布侧凭据从批次自己的正文里恢复（写入时逐轮冻结）：没有它们，抽取就
        # 无法把这一批正文与它引用的工作记忆/来源归档对应起来。
        source_ids: List[str] = []
        source_batch_id: Optional[str] = None
        source_layer: Optional[str] = None
        for entry in entries:
            for item in entry.get("source_ids") or []:
                value = str(item)
                if value and value not in source_ids:
                    source_ids.append(value)
            source_batch_id = source_batch_id or entry.get("source_batch_id")
            source_layer = source_layer or entry.get("source_layer")
        snapshot["memos_source_ids"] = source_ids
        if source_batch_id:
            snapshot["source_batch_id"] = str(source_batch_id)
        if source_layer:
            snapshot["source_layer"] = str(source_layer)
        # The external-source registrar reads this key; a warm batch may contain
        # turns from several channels, and the per-entry ``source`` above is what
        # keeps "who said it" exact.
        declared = [
            dict(entry["source"])
            for entry in entries
            if isinstance(entry.get("source"), dict) and entry.get("source")
        ]
        if declared:
            snapshot["external_sources"] = declared
        return snapshot

    def _process(self, batch_id: str, moment: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            if batch_id in self._in_flight:
                return None
            self._in_flight.add(batch_id)
        try:
            return self.extract_batch(batch_id)
        finally:
            with self._lock:
                self._in_flight.discard(batch_id)

    def is_running(self, batch_id: str) -> bool:
        """True while a sweep/retry is extracting this batch right now.

        Callers that can be triggered by hand (the management checkpoint) ask
        before freezing, so a sweep that is already publishing a batch cannot be
        started a second time by a double-click.
        """
        with self._lock:
            return str(batch_id) in self._in_flight

    def extract_batch(self, batch_id: str, *, reason: str = "ttl_expired") -> Dict[str, Any]:
        """Freeze → publish → evict, keeping the bodies on any failure.

        The order is the whole safety story: the snapshot and the
        ``extraction_pending`` state are durable *before* the LLM runs, and the
        bodies are deleted only after a confirmed publication, so a crash at any
        point leaves either a retryable job or intact text.
        """
        snapshot = self.build_snapshot(batch_id, reason=reason)
        self.store.mark_extraction(
            batch_id,
            BATCH_EXTRACTION_PENDING,
            {"reason": reason, "frozen_at": utc_timestamp(), "events": len(snapshot.get("events") or [])},
        )
        if not snapshot.get("events"):
            # Nothing extractable is a legal terminal state: there is no knowledge
            # to lose, so the batch does not have to hold its (empty) text.
            self.store.mark_extraction(
                batch_id, BATCH_PUBLISHED, {"status": "empty", "reason": "no_events"}
            )
            removed = self.store.evict(batch_id, reason="empty_batch")
            self._audit(
                "warm_batch_evicted",
                {"batch_id": batch_id, "reason": "empty_batch", "removed": len(removed)},
            )
            return {"batch_id": batch_id, "status": "published", "reason": "empty_batch"}
        if self.extract is None:
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {"error": "no extractor configured", "attempt_at": utc_timestamp()},
            )
            return {"batch_id": batch_id, "status": "failed", "error": "no extractor configured"}
        try:
            result = self.extract(snapshot) or {}
        except Exception as exc:  # noqa: BLE001 - any failure keeps the transcript
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {"error": f"{type(exc).__name__}: {exc}", "attempt_at": utc_timestamp()},
            )
            self._audit(
                "warm_batch_extraction_failed",
                {"batch_id": batch_id, "error": f"{type(exc).__name__}: {exc}"},
            )
            return {"batch_id": batch_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        knowledge = str((result.get("manifest") or {}).get("status") or result.get("status") or "")
        diary_state = _diary_state(result.get("diary"))
        if knowledge == "queued":
            # Enqueued to the background extraction queue. The transcript stays
            # until the **worker** confirms, so finalising is deferred to it:
            # marking the batch published here would evict the bodies before the
            # graph write has actually happened.
            self.store.mark_extraction(
                batch_id,
                BATCH_EXTRACTION_PENDING,
                {"queued_at": utc_timestamp(), "diary": result.get("diary")},
            )
            return {
                "batch_id": batch_id,
                "status": "queued",
                "diary": diary_state,
                "snapshot": snapshot,
                "deferred": True,
            }
        if knowledge not in PUBLISH_OK:
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {
                    "error": f"knowledge publish returned {knowledge or 'unknown'}",
                    "attempt_at": utc_timestamp(),
                },
            )
            return {
                "batch_id": batch_id,
                "status": "failed",
                "error": f"knowledge publish returned {knowledge or 'unknown'}",
            }
        if diary_state not in PUBLISH_OK:
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {
                    "error": f"diary/memory pass returned {diary_state}",
                    "attempt_at": utc_timestamp(),
                    "diary": result.get("diary"),
                },
            )
            return {
                "batch_id": batch_id,
                "status": "failed",
                "error": f"diary/memory pass returned {diary_state}",
            }
        self.store.mark_extraction(
            batch_id,
            BATCH_PUBLISHED,
            {
                "published_at": utc_timestamp(),
                "knowledge": knowledge,
                "diary": diary_state,
                "operation_id": snapshot.get("operation_id"),
            },
        )
        removed = self.store.evict(batch_id, reason=reason)
        self._audit(
            "warm_batch_evicted",
            {"batch_id": batch_id, "reason": reason, "removed": len(removed)},
        )
        return {
            "batch_id": batch_id,
            "status": "published",
            "removed": len(removed),
            "diary": diary_state,
        }

    # ------------------------------------------------------------------- misc

    def finalise(self, batch_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Confirm a **queued** extraction once the background worker finished it.

        Reached from the service's queued-extraction wrapper. Until this runs the
        batch stays ``extraction_pending`` and keeps its transcript, so the
        ordering is "worker confirms publish → then delete", never the reverse.
        Every failure keeps the text and records ``retryable_failed``.
        """
        knowledge = str((result.get("manifest") or {}).get("status") or result.get("status") or "")
        diary_state = _diary_state(result.get("diary"))
        if knowledge not in PUBLISH_OK:
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {
                    "error": f"knowledge publish returned {knowledge or 'unknown'}",
                    "attempt_at": utc_timestamp(),
                },
            )
            return {"batch_id": batch_id, "status": "failed", "error": knowledge}
        if diary_state not in PUBLISH_OK:
            self.store.mark_extraction(
                batch_id,
                BATCH_RETRYABLE_FAILED,
                {
                    "error": f"diary/memory pass returned {diary_state}",
                    "attempt_at": utc_timestamp(),
                    "diary": result.get("diary"),
                },
            )
            return {"batch_id": batch_id, "status": "failed", "error": diary_state}
        self.store.mark_extraction(
            batch_id,
            BATCH_PUBLISHED,
            {
                "published_at": utc_timestamp(),
                "knowledge": knowledge,
                "diary": diary_state,
                "operation_id": (result.get("manifest") or {}).get("operation_id"),
            },
        )
        removed = self.store.evict(batch_id, reason="queued_extraction_published")
        self._audit(
            "warm_batch_evicted",
            {"batch_id": batch_id, "reason": "queued_extraction_published", "removed": len(removed)},
        )
        return {
            "batch_id": batch_id,
            "status": "published",
            "removed": len(removed),
            "diary": diary_state,
        }

    def recover(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """Startup recovery: re-run pending and failed batches.

        Called once per role at startup so a crash between "knowledge published"
        and "bodies deleted" is repaired rather than leaving the transcript
        stranded forever. The service's own publish path is idempotent on the
        batch's operation id, so a partially published batch is not published
        twice.
        """
        recovered: List[Dict[str, Any]] = []
        for batch in self.store.pending_batches() + self.store.failed_batches():
            batch_id = str(batch.get("batch_id"))
            outcome = self._process(batch_id, self.clock() if now is None else float(now))
            if outcome is not None:
                recovered.append(outcome)
        return {"role_uuid": self.store.role_uuid, "recovered": recovered}

    def status(self) -> Dict[str, Any]:
        stats = self.store.stats()
        return {
            "role_uuid": self.store.role_uuid,
            "ttl_seconds": self.ttl_seconds,
            "batches": [
                {
                    "batch_id": batch.get("batch_id"),
                    "status": batch.get("status"),
                    "seal_reason": batch.get("seal_reason"),
                    "sealed_at": batch.get("sealed_at"),
                    "last_output_at": batch.get("last_output_at"),
                    "conversation_ids": batch.get("conversation_ids"),
                    "extraction": batch.get("extraction"),
                }
                for batch in self.store.batches()
            ],
            "stats": stats,
            "evictable": [
                {
                    "batch_id": item.get("batch_id"),
                    "age_seconds": item.get("age_seconds"),
                }
                for item in self.store.evictable_batches(now=self.clock(), ttl=self.ttl_seconds)
            ],
        }

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit(action, details)
        except Exception:  # noqa: BLE001 - audit never fails the sweep
            pass


def _diary_state(value: Any) -> str:
    """Normalize the diary pass outcome into a terminal-state token.

    ``DiaryWriter.write`` reports ``{"status": "written"|"empty"|"skipped"}``, and
    ``MemoryExtractor.extract`` reports ``{"status": "ok", ...}``. The second pass
    returns both merged (``{**extraction, "published", "diary_id"}``), so reading
    ``status`` first would classify a perfectly healthy pass as ``ok`` — a token
    the extractor's terminal set does not contain, which would mark every batch
    retryable_failed and keep its transcript forever.
    """
    if value is None:
        return "disabled"
    if isinstance(value, str):
        return value or "unknown"
    if isinstance(value, dict):
        if value.get("diary_id") or value.get("entry"):
            # The diary itself was written; whether the *memory extraction* found
            # anything is reported separately by ``memories``/``published``.
            return "written"
        status = value.get("status")
        if status:
            return str(status)
        return "published"
    return "unknown"


__all__ = [
    "WarmBatchRegistry",
    "WarmBatchExtractor",
    "PUBLISH_OK",
]
