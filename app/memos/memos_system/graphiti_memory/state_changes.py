"""State-change hub: the single push channel the management WebUI subscribes to.

The management panel used to poll whole payloads on a fixed timer
(``/graphiti/chatlog`` and ``/graphiti/extraction/queue`` every 3 s per open
page). That burns LLM-adjacent work for nothing: an idle memory service was
still answering hundreds of identical requests per hour.

This module turns "refresh on a timer" into "refresh on a state transition".
One in-process hub holds a monotonically increasing version per observable
topic:

* ``chatlog``    - a conversation round was recorded (user input + AI reply +
                   the memories recalled for it).
* ``extraction`` - the background extraction queue changed: a job was queued,
                   moved stage, finished or failed.
* ``session``    - a role conversation changed lifecycle state, including the
                   1-minute temporary-graph checkpoint and the 1-hour formal
                   close (both of which are what actually trigger extraction).
* ``graph``      - shared topology / private cognition changed.
* ``imports``    - a role-import job advanced (queued, stage, done, failed).

Transitions are discovered by comparing a compact fingerprint on a short
internal tick. A tick with an unchanged fingerprint publishes nothing, so
subscribers are woken by real changes only. Producers that already know the
exact moment of a transition (``record_turn``) mark the topic directly and the
change is visible immediately, without waiting for the tick.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Observable topics. Keep this list stable: the WebUI maps view -> topics.
TOPIC_CHATLOG = "chatlog"
TOPIC_EXTRACTION = "extraction"
TOPIC_SESSION = "session"
TOPIC_GRAPH = "graph"
TOPIC_IMPORTS = "imports"
TOPICS: Tuple[str, ...] = (
    TOPIC_CHATLOG,
    TOPIC_EXTRACTION,
    TOPIC_SESSION,
    TOPIC_GRAPH,
    TOPIC_IMPORTS,
)

SCHEMA_VERSION = 1


class StateChangeHub:
    """Version counters + subscriber notification for management-UI state.

    The hub never owns memory data. It only records "topic X changed" so a
    client can decide whether re-reading a payload is worth a request.
    """

    def __init__(self, tick_seconds: float = 1.0, history: int = 64):
        self._lock = threading.Condition()
        self._versions: Dict[str, int] = {topic: 0 for topic in TOPICS}
        self._origin: Dict[str, str] = {topic: "startup" for topic in TOPICS}
        self._history: List[Dict[str, Any]] = []
        self._history_limit = max(0, int(history))
        self._seq = 0
        self._tick_seconds = max(0.05, float(tick_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fingerprints: Dict[str, str] = {}
        self._observer: Optional[Callable[[], Dict[str, str]]] = None
        self._deadline_provider: Optional[Callable[[], Any]] = None
        # Monotonic timestamps at which a session will transition on its own
        # (the 2-minute temporary-graph checkpoint and the 1-hour formal close
        # are both pure functions of "time since last output"). Without these a
        # subscriber would only learn about an idle checkpoint on its next
        # keep-alive, so the push channel would lag the state it reports.
        self._deadlines: Tuple[float, ...] = ()
        # Optional idle executor: the pet no longer closes its conversation on
        # quit, so a backend that keeps running must advance an idle session
        # itself (checkpoint at 2 minutes, formal close at 1 hour).
        self._sweeper: Optional[Callable[[], Any]] = None
        self._due_provider: Optional[Callable[[], bool]] = None
        self._sweeper_running = False
        self._last_sweep_at = 0.0
        self._sweep_min_interval = 5.0
        self._subscribers: Set[Any] = set()

    # ------------------------------------------------------------------ wiring

    def attach_observer(
        self,
        observer: Callable[[], Dict[str, str]],
        deadline_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Register the fingerprint producer (one ``topic -> fingerprint`` map).

        Called once by the service. Keeping the producer outside the hub is what
        lets the hub stay ignorant of memory internals. ``deadline_provider``
        optionally supplies the monotonic instants at which a session will
        transition on its own (see :meth:`set_deadlines`).
        """
        self._observer = observer
        self._deadline_provider = deadline_provider

    def attach_sweeper(
        self,
        sweeper: Optional[Callable[[], Any]],
        due_provider: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Register the idle executor (called when a published deadline is due).

        The hub only knows *when* an idle window elapses; the service knows what
        to do about it. Keeping the executor outside the hub mirrors
        ``attach_observer`` and lets tests run without one.

        ``due_provider`` answers "has a window already elapsed?". It cannot be
        derived from ``_deadlines``: that list holds only future instants (so
        subscribers never spin on an elapsed one), which would leave a
        deadline-driven sweep with nothing to fire on. Without a provider the hub
        falls back to the deadline list, which is only correct for a caller that
        publishes already-elapsed deadlines.
        """
        self._sweeper = sweeper
        self._due_provider = due_provider

    def set_deadlines(self, deadlines: Any) -> None:
        """Publish monotonic timestamps at which a session transitions by itself.

        Producers that compute idle windows (``checkpoint_seconds`` and
        ``formal_idle_seconds``) hand their absolute deadlines over so
        ``wait_for_change`` can wake subscribers exactly when the window
        actually elapses, instead of waiting for the next keep-alive.
        """
        cleaned: List[float] = []
        for value in deadlines or ():
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        deadlines_tuple = tuple(sorted(cleaned))
        with self._lock:
            if deadlines_tuple != self._deadlines:
                self._deadlines = deadlines_tuple
                self._lock.notify_all()

    def start(self) -> None:
        """Start the internal tick thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._pump, name="graphiti-state-change", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the tick thread. Safe to call from a shutdown hook."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._lock.notify_all()

    def _pump(self) -> None:
        while not self._stop.wait(self._tick_seconds):
            try:
                self.poll_once()
            except Exception:
                # A failing probe must never kill the pump: the next tick retries
                # with the previous fingerprints intact.
                continue

    def poll_once(self) -> None:
        """Compare fingerprints once and publish any topic that moved."""
        if self._deadline_provider is not None:
            try:
                self.set_deadlines(self._deadline_provider())
            except Exception:
                pass
        self._maybe_sweep()
        observer = self._observer
        if observer is None:
            return
        try:
            observed = observer() or {}
        except Exception:
            return
        for topic in TOPICS:
            if topic not in observed:
                continue
            fingerprint = str(observed[topic])
            if self._fingerprints.get(topic) == fingerprint:
                continue
            # First observation only establishes the baseline: a service that
            # just started has no subscribed client to notify yet, and treating
            # the initial values as "changed" would only produce noise.
            first = topic not in self._fingerprints
            self._fingerprints[topic] = fingerprint
            if not first:
                self.mark(topic, origin="observed")

    # --------------------------------------------------------------- producers

    def _maybe_sweep(self) -> None:
        """Run the idle executor once a published deadline is due.

        Runs in its own thread: a checkpoint performs LLM calls, and blocking the
        pump would stall the push channel it exists to serve. At most one sweep
        is in flight and consecutive sweeps are spaced out, so a deadline that
        does not clear immediately cannot turn into a busy loop.
        """
        sweeper = self._sweeper
        if sweeper is None:
            return
        now = time.monotonic()
        with self._lock:
            if self._sweeper_running or (now - self._last_sweep_at) < self._sweep_min_interval:
                return
            if self._due_provider is not None:
                try:
                    due = bool(self._due_provider())
                except Exception:
                    due = False
                if not due:
                    return
            elif not any(value <= now for value in self._deadlines):
                return
            self._sweeper_running = True
            self._last_sweep_at = now

        def run() -> None:
            try:
                sweeper()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._sweeper_running = False

        threading.Thread(target=run, name="graphiti-idle-sweep", daemon=True).start()

    def mark(self, topic: str, *, origin: str = "producer", detail: Optional[Dict[str, Any]] = None) -> int:
        """Record a transition for ``topic`` and wake every subscriber."""
        if topic not in self._versions:
            raise ValueError(f"unknown state-change topic: {topic}")
        with self._lock:
            self._versions[topic] += 1
            self._origin[topic] = origin
            self._seq += 1
            event = {
                "seq": self._seq,
                "topic": topic,
                "version": self._versions[topic],
                "origin": origin,
                "at": time.time(),
            }
            if detail:
                event["detail"] = detail
            self._history.append(event)
            if self._history_limit:
                self._history = self._history[-self._history_limit:]
            else:
                self._history = []
            self._lock.notify_all()
            return self._versions[topic]

    # -------------------------------------------------------------- subscribers

    def snapshot(self) -> Dict[str, Any]:
        """Current versions: what a client needs to diff after (re)connecting."""
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "seq": self._seq,
                "versions": dict(self._versions),
                "updates": {topic: self._origin.get(topic, "startup") for topic in TOPICS},
            }

    def version_of(self, topic: str) -> int:
        with self._lock:
            return self._versions.get(topic, 0)

    def wait_for_change(
        self,
        known: Dict[str, int],
        timeout: float = 15.0,
    ) -> Optional[Dict[str, Any]]:
        """Block until any topic passes its known version, or ``timeout``.

        Returns the transition payload, or ``None`` on timeout so the caller can
        emit a keep-alive comment without inventing a fake change. A pending
        idle deadline shortens the wait: the state will transition on its own at
        that instant, so the client should not have to wait for the keep-alive.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            while True:
                changed = {
                    topic: self._versions[topic]
                    for topic in TOPICS
                    if self._versions[topic] > int(known.get(topic, 0))
                }
                if changed:
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "seq": self._seq,
                        "changed": changed,
                        "versions": dict(self._versions),
                        "updates": {topic: self._origin.get(topic, "startup") for topic in changed},
                    }
                now = time.monotonic()
                remaining = deadline - now
                for pending in self._deadlines:
                    if pending <= now:
                        remaining = 0.0
                        break
                    remaining = min(remaining, pending - now)
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)

    def history(self, topic: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = [event for event in self._history if topic is None or event["topic"] == topic]
            return [dict(item) for item in items[-max(0, int(limit)):]]


def observe_service(service: Any) -> Callable[[], Dict[str, str]]:
    """Build the fingerprint producer for a ``GraphitiMemoryService``.

    Every probe is O(open sessions + queue length) and reads in-memory state
    only. Fingerprints are deliberately coarse: they must change exactly when
    the corresponding WebUI payload would change, and never on a bare tick.
    """

    def observe() -> Dict[str, str]:
        return {
            TOPIC_CHATLOG: _chatlog_fingerprint(service),
            TOPIC_EXTRACTION: _extraction_fingerprint(service),
            TOPIC_SESSION: _session_fingerprint(service),
            TOPIC_GRAPH: _graph_fingerprint(service),
            TOPIC_IMPORTS: _imports_fingerprint(service),
        }

    return observe


def _idle_windows(service: Any) -> Tuple[List[float], bool]:
    """``(future deadlines, something already due)`` for every active session.

    Two different consumers need two different answers from the same computation:

    * ``wait_for_change`` needs only **future** instants — a deadline that has
      already passed would make every wait return immediately (a hot loop of
      keep-alives in the SSE stream);
    * the idle sweeper needs to know whether a window has **already** elapsed,
      which is exactly the case a future-only list can never report.

    Reading them from one place keeps the sweep and the wake-up honest about the
    same two windows (2-minute checkpoint, 1-hour formal close).
    """
    deadlines: List[float] = []
    due = False
    clock = time.monotonic
    now = clock()
    try:
        sessions = dict(getattr(service, "sessions", {}) or {})
    except Exception:
        return deadlines, due
    for session in sessions.values():
        if getattr(session, "state", None) != "active":
            continue
        try:
            last_output = float(getattr(session, "_last_output_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not last_output:
            continue
        try:
            checkpoint_seconds = float(getattr(session, "checkpoint_seconds", 120.0) or 120.0)
            formal_idle_seconds = float(getattr(session, "formal_idle_seconds", 3600.0) or 3600.0)
        except (TypeError, ValueError):
            continue
        checkpointed = bool(getattr(session, "_checkpointed", False))
        events = getattr(session, "_events", None) or []
        # A checkpoint needs fresh 上文, so an already-checkpointed session with
        # no new events only has the formal-close deadline pending.
        windows = []
        if not checkpointed and events:
            windows.append(last_output + checkpoint_seconds)
        windows.append(last_output + formal_idle_seconds)
        for value in windows:
            if value <= now:
                due = True
            else:
                deadlines.append(value)
    # `_last_output_at` is captured from the same monotonic clock the hub uses
    # (`time.monotonic`), so the offsets are directly comparable.
    return sorted(deadlines), due


def session_deadlines(service: Any) -> List[float]:
    """Monotonic instants at which an idle session transitions by itself.

    The 2-minute temporary-graph checkpoint and the 1-hour formal close are pure
    functions of "time since the last completed output", so the hub can be told
    exactly when to wake subscribers instead of letting the WebUI poll to find
    out. Deadlines are for open sessions only; a closed conversation has none.

    Only **future** instants are returned: an elapsed deadline has to be cleared
    by the sweeper, and reporting it here would turn every subscriber's wait into
    an immediate return.
    """
    return _idle_windows(service)[0]


def session_windows_due(service: Any) -> bool:
    """True when at least one active session's idle window has already elapsed.

    This is the sweeper's trigger. It cannot be expressed as "a published deadline
    is now in the past": :func:`session_deadlines` deliberately drops elapsed
    values, so a sweeper that only looked at the published list would never run
    and the 2-minute checkpoint / 1-hour formal close would silently never
    happen once the desktop pet stopped polling ``check-idle`` (ADR-0146).
    """
    return _idle_windows(service)[1]


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _chatlog_fingerprint(service: Any) -> str:
    root = getattr(getattr(service, "chatlog", None), "root", None)
    if root is None:
        return "none"
    try:
        directory = Path(root)
        if not directory.exists():
            return "empty"
        parts = []
        for path in sorted(directory.glob("*.json")):
            try:
                stat = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return _stable(parts)
    except Exception:
        return "error"


def _extraction_fingerprint(service: Any) -> str:
    try:
        status = service.extraction_queue_status()
    except Exception:
        return "error"
    jobs = []
    for group in ("running", "pending", "retry"):
        for item in status.get(group) or []:
            jobs.append({
                "group": group,
                "role_uuid": item.get("role_uuid"),
                "conversation_id": item.get("conversation_id"),
                "state": item.get("state"),
                "progress": item.get("progress"),
                "attempts": item.get("attempts"),
                "queue_position": item.get("queue_position"),
            })
    return _stable(jobs)


def _session_fingerprint(service: Any) -> str:
    try:
        sessions = dict(getattr(service, "sessions", {}) or {})
        states = []
        for (role_uuid, conversation_id), session in sessions.items():
            checkpoint = getattr(session, "_checkpoint", None)
            temporary = getattr(session, "temporary_graph", None)
            entries = len(getattr(temporary, "entries", []) or [])
            states.append({
                "role_uuid": role_uuid,
                "conversation_id": conversation_id,
                "state": getattr(session, "state", None),
                "checkpointed": bool(getattr(session, "_checkpointed", False)),
                "checkpoint_at": (checkpoint or {}).get("created_at"),
                "temporary_entries": entries,
                "turns": len(getattr(session, "_events", []) or []),
            })
        if states:
            return _stable(states)
        # No live session in memory (fresh process): fall back to persisted
        # session states so a checkpoint made by another process still lands.
        persisted = []
        for item in service.authority.list_sessions():
            persisted.append({
                "role_uuid": item.get("role_uuid"),
                "conversation_id": item.get("conversation_id"),
                "state": item.get("state"),
                "updated_at": item.get("updated_at"),
            })
        return _stable(persisted)
    except Exception:
        return "error"


def _graph_fingerprint(service: Any) -> str:
    try:
        revision = service.graph_revision("global").get("revision")
        return str(revision or "none")
    except Exception:
        return "error"


def _imports_fingerprint(service: Any) -> str:
    """Role-import jobs: only the fields the import page renders.

    Progress logs are excluded on purpose - they carry timestamps that change on
    every read and would turn this topic into a de-facto poll.
    """
    try:
        coordinator = getattr(service, "role_imports", None)
        if coordinator is None:
            return "none"
        jobs = coordinator.list_jobs()
    except Exception:
        return "error"
    compact = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        compact.append({
            "job_id": job.get("job_id") or job.get("id"),
            "status": job.get("status"),
            "role_uuid": job.get("role_uuid"),
            "completed_chars": job.get("completed_chars"),
            "source_chars": job.get("source_chars"),
            "completed_chunks": job.get("completed_chunks"),
            "total_chunks": job.get("total_chunks"),
            "current_chunk": job.get("current_chunk"),
            "message": job.get("message"),
            "error": job.get("error"),
            "warning": job.get("warning"),
            "updated_at": job.get("updated_at"),
        })
    return _stable(compact)
