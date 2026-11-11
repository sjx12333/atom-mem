"""Background async extraction queue with per-role FIFO and dual-key concurrency.

One ``ExtractionScheduler`` instance drives every closed-session extraction for a
``GraphitiMemoryService``. Jobs for the SAME role run strictly in enqueue order
(FIFO); jobs for DIFFERENT roles may run concurrently, up to the configured
concurrency (one job on the primary extraction key + one on the backup key).
Failed jobs are surfaced as ``retry_locked`` with a bounded attempt count so a
client (WebUI / desktop pet) can observe and retry them.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable, Dict, List, Optional

from .coordinator import EXTRACTION_STAGES
from .domain import utc_timestamp


class ExtractionScheduler:
    def __init__(self, concurrency: int = 1, allow_backup: bool = True):
        self._cond = threading.Condition(threading.Lock())
        self._jobs: List[Dict[str, Any]] = []
        self._concurrency = max(1, int(concurrency))
        # False when no usable backup credential is configured: a job must never
        # be labelled "backup" and then silently run on the primary key.
        self._allow_backup = bool(allow_backup)
        self._started = False
        self._stopped = False
        self._workers: List[threading.Thread] = []
        self._seq = 0
        # Last retryable error per (role, conversation) so retry can re-drive it
        # and a re-enqueue after a failure can switch credentials.
        self._retryable: Dict[tuple[str, str], Dict[str, Any]] = {}

    # --- lifecycle ---------------------------------------------------------

    def set_allow_backup(self, allowed: bool) -> None:
        """Update whether the backup credential may be used (config reload).

        Called when the runtime config is saved, so enabling or rotating the
        backup key takes effect without restarting the service.
        """
        with self._cond:
            self._allow_backup = bool(allowed)

    def start(self) -> "ExtractionScheduler":
        with self._cond:
            if self._started:
                return self
            self._started = True
        for _ in range(self._concurrency):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
        return self

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    # --- enqueue -----------------------------------------------------------

    @staticmethod
    def _failover_key(job: Dict[str, Any]) -> tuple[str, str]:
        """The failover bucket of one job: the batch when it has one.

        Several chat batches share a conversation id, so a conversation-keyed
        bucket let one batch's failure push an unrelated sibling batch onto the
        backup credential (and let the sibling clear the failure record).
        """
        snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), dict) else {}
        batch_id = str((snapshot or {}).get("batch_id") or "").strip()
        return (str(job.get("role_uuid") or ""), batch_id or str(job.get("conversation_id") or ""))

    def enqueue(
        self,
        role_uuid: str,
        conversation_id: str,
        run: Callable[[str], Dict[str, Any]],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Queue a job.

        ``run`` is a callable that receives the chosen key name (``"primary"``
        or ``"backup"``) and performs the extraction + publish, returning the
        published result. It must be safe to run from a worker thread.
        """
        with self._cond:
            self._prune_completed()
            self._seq += 1
            job: Dict[str, Any] = {
                "id": self._seq,
                "role_uuid": str(role_uuid),
                "conversation_id": str(conversation_id),
                "snapshot": snapshot,
                "run": run,
                "state": "queued",
                "enqueued_at": utc_timestamp(),
                "started_at": None,
                "elapsed_seconds": 0,
                "used_key": None,
                "attempts": 0,
                "message": "等待抽取",
                "error": None,
                "progress": {},
                "current_stage": None,
                "progress_log": [],
            }
            self._jobs.append(job)
            self._cond.notify()
            return self._public_job(job)

    # --- status ------------------------------------------------------------

    @staticmethod
    def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role_uuid": job.get("role_uuid"),
            "conversation_id": job.get("conversation_id"),
            "state": job.get("state"),
            "started_at": job.get("started_at") or job.get("enqueued_at"),
            "enqueued_at": job.get("enqueued_at"),
            "elapsed_seconds": job.get("elapsed_seconds", 0),
            "used_key": job.get("used_key"),
            "message": job.get("message", ""),
            "progress": job.get("progress", {}),
            "current_stage": job.get("current_stage"),
            "progress_log": list(job.get("progress_log", [])),
            "attempts": job.get("attempts", 0),
            "error": job.get("error"),
        }

    def queue_status(self) -> Dict[str, Any]:
        with self._cond:
            running = [self._public_job(j) for j in self._jobs if j["state"] in ("extracting", "publishing")]
            pending = []
            pending_index = 1
            for j in self._jobs:
                if j["state"] == "queued":
                    item = self._public_job(j)
                    item["queue_position"] = pending_index
                    pending.append(item)
                    pending_index += 1
            retry = [self._public_job(j) for j in self._jobs if j["state"] == "retry_locked"]
            return {"running": running, "pending": pending, "retry": retry}

    def role_status(self, role_uuid: str) -> Dict[str, Any]:
        with self._cond:
            for j in self._jobs:
                if j["role_uuid"] == role_uuid and j["state"] in ("extracting", "publishing"):
                    has_pending = any(x["role_uuid"] == role_uuid and x["state"] == "queued" for x in self._jobs)
                    return {
                        "role_uuid": role_uuid,
                        "extracting": True,
                        "has_pending": has_pending,
                        "state": j["state"],
                        "message": j.get("message", ""),
                    }
            has_pending = any(x["role_uuid"] == role_uuid and x["state"] == "queued" for x in self._jobs)
            return {
                "role_uuid": role_uuid,
                "extracting": False,
                "has_pending": has_pending,
                "state": "queued" if has_pending else "clear",
                "message": "等待抽取" if has_pending else "",
            }

    def retry(
        self, role_uuid: str, conversation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Re-queue a ``retry_locked`` job in place.

        Re-queuing (instead of enqueuing a new job) keeps ``attempts`` and
        ``used_key`` so ``_choose_key`` fails the job over to the OTHER
        credential on the next run; without that, every retry restarted on the
        key that had just rejected the request.
        """
        with self._cond:
            for job in self._jobs:
                if job["state"] != "retry_locked" or job["role_uuid"] != str(role_uuid):
                    continue
                if conversation_id is not None and job["conversation_id"] != str(conversation_id):
                    continue
                job["state"] = "queued"
                job["message"] = "等待重试"
                job["error"] = None
                job["started_at"] = None
                job["elapsed_seconds"] = 0
                job["progress"] = {}
                job["current_stage"] = None
                job["progress_log"].append({
                    "timestamp": utc_timestamp(),
                    "stage": "queued",
                    "message": f"重试（第 {int(job.get('attempts', 0)) + 1} 次尝试）",
                    "completed": 0,
                    "total": len(EXTRACTION_STAGES),
                })
                job["progress_log"] = job["progress_log"][-60:]
                self._cond.notify()
                return self._public_job(job)
            return None

    def _prune_completed(self) -> None:
        """Drop finished jobs so the queue stays lean (caller holds the lock)."""
        self._jobs = [j for j in self._jobs if j["state"] != "completed"]

    # --- worker ------------------------------------------------------------

    def _pick_next(self) -> Optional[Dict[str, Any]]:
        """Pick the earliest queued job whose role is not already running."""
        running_roles = {j["role_uuid"] for j in self._jobs if j["state"] in ("extracting", "publishing")}
        for j in self._jobs:
            if j["state"] == "queued" and j["role_uuid"] not in running_roles:
                j["state"] = "extracting"
                j["started_at"] = utc_timestamp()
                j["message"] = "抽取中"
                return j
        return None

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stopped:
                        return
                    job = self._pick_next()
                    if job is not None:
                        break
                    self._cond.wait()
                # Assign a key: alternate primary/backup so different roles use
                # different extraction credentials when both are available.
                job["used_key"] = self._choose_key(job)

            try:
                progress = self._make_progress(job)
                if self._accepts_progress(job["run"]):
                    result = job["run"](job["used_key"], progress)
                else:
                    result = job["run"](job["used_key"])
                with self._cond:
                    job["state"] = "completed"
                    job["message"] = "抽取完成"
                    job["elapsed_seconds"] = self._elapsed(job)
                    job["current_stage"] = "embed_and_save"
                    job["progress"] = {"done": True, "completed": len(EXTRACTION_STAGES), "total": len(EXTRACTION_STAGES), "message": "抽取完成", "current_stage": "embed_and_save", "stages": [{"key": key, "label": label} for key, label in EXTRACTION_STAGES]}
                    job["progress_log"].append({"timestamp": utc_timestamp(), "stage": "completed", "message": "抽取完成", "completed": len(EXTRACTION_STAGES), "total": len(EXTRACTION_STAGES)})
                    job["progress_log"] = job["progress_log"][-60:]
                    # A published batch is no longer a failover source.
                    self._retryable.pop(self._failover_key(job), None)
            except Exception as exc:  # noqa: BLE001
                with self._cond:
                    job["attempts"] += 1
                    job["error"] = str(exc)
                    job["state"] = "retry_locked"
                    job["message"] = "抽取失败，等待重试"
                    job["elapsed_seconds"] = self._elapsed(job)
                    job["progress"] = {"done": False, "completed": 0, "total": 1, "message": str(exc)}
                    self._retryable[self._failover_key(job)] = dict(job)
                with self._cond:
                    self._cond.notify_all()

    def _choose_key(self, job: Dict[str, Any]) -> str:
        keys = ("primary", "backup")
        # A job that already failed once — or a fresh enqueue for a (role,
        # conversation) whose previous attempt failed — must try the OTHER
        # credential: a rejected/limited key can never recover by re-running on
        # itself. This is what makes the backup key take over on retry.
        last_failure = self._retryable.get(self._failover_key(job)) or {}
        previous = job.get("used_key") or last_failure.get("used_key")
        attempts = job.get("attempts", 0) or last_failure.get("attempts", 0)
        if attempts and previous in keys:
            candidate = "backup" if previous == "primary" else "primary"
            if candidate != "backup" or self._allow_backup:
                return candidate
        # Otherwise prefer the key currently less busy.
        busy: Dict[str, int] = {}
        for other in self._jobs:
            if other["state"] in ("extracting", "publishing") and other.get("used_key"):
                busy[other["used_key"]] = busy.get(other["used_key"], 0) + 1
        if self._allow_backup and self._concurrency >= 2 and busy.get("primary", 0) > busy.get("backup", 0):
            return "backup"
        return "primary"

    @staticmethod
    def _accepts_progress(run: Callable) -> bool:
        """True if run can accept a trailing progress argument."""
        try:
            parameters = list(inspect.signature(run).parameters.values())
        except (TypeError, ValueError):
            return True
        positional = [
            p for p in parameters
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return (
            len(positional) >= 2
            or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
        )

    def _make_progress(self, job: Dict[str, Any]):
        """Return a progress reporter bound to job.

        The reporter upserts the job's per-stage progress and records a
        timestamped progress-log entry so the WebUI can surface extraction
        progress in real time (mirrors the staged reporting in E:/demo_app).
        """
        def progress(stage_key: str, message: str, completed: Optional[int] = None) -> None:
            with self._cond:
                index = next((i for i, (key, _) in enumerate(EXTRACTION_STAGES) if key == stage_key), 0)
                total = len(EXTRACTION_STAGES)
                # completed 至少与当前阶段索引一致，避免进度数字与七段圆点错位
                # （例如数字显示 5/7 但只有 4 个圆点亮起）。
                requested = int(completed) if completed is not None else index
                done = min(max(0, requested, index), total)
                job["message"] = message
                job["current_stage"] = stage_key
                job["progress"] = {
                    "done": False,
                    "completed": done,
                    "total": total,
                    "message": message,
                    "current_stage": stage_key,
                    "stage_index": index,
                    "stages": [{"key": key, "label": label} for key, label in EXTRACTION_STAGES],
                }
                job["progress_log"].append({
                    "timestamp": utc_timestamp(),
                    "stage": stage_key,
                    "message": message,
                    "completed": done,
                    "total": total,
                })
                job["progress_log"] = job["progress_log"][-60:]
        return progress

    @staticmethod
    def _elapsed(job: Dict[str, Any]) -> int:
        if not job.get("started_at"):
            return 0
        try:
            from datetime import datetime, timezone
            started = job["started_at"].replace("Z", "+00:00")
            return max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()))
        except Exception:  # noqa: BLE001
            return 0

    def report_done(self, role_uuid: str, conversation_id: str) -> None:
        """Drop a completed job so the queue stays lean (no-op harmless otherwise)."""
        with self._cond:
            self._jobs = [
                j for j in self._jobs
                if not (j["role_uuid"] == role_uuid and j["conversation_id"] == conversation_id and j["state"] == "completed")
            ]
