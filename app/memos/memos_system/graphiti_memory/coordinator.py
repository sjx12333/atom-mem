"""Single owner of the live context: turn intake, batch boundaries and sealing.

Three layers hold memory, and this module owns the boundary between the first two:

* **Live Context** — ``_events``: the messages of the current, not-yet-archived
  turn window. Only this module may trim it.
* **Warm Layer** — written by ``record_turn`` through a callback, and *never*
  cleared here. ``close()``, a pet stop, a client disconnect, a role switch and a
  service restart all leave un-evicted warm bodies exactly where they are.
* **Cold Layer** — reached only by the TTL sweeper, which freezes a sealed batch
  into a snapshot and hands it to the existing publication path.

The 2-minute checkpoint is a **Live Context** operation: it drops events that are
already durable in the Warm Layer so the session's in-memory window stays small.
It does not extract, does not publish, and cannot delete a body.

The 1-hour idle window defines **where a chat batch ends**. Sealing a batch at
that boundary does not extract either; a batch is extracted only when its newest
body passes the TTL (see ``warm_batches.py``).

The old "close ⇒ build snapshot ⇒ publish" path is gone on purpose: it coupled
conversation length to extraction latency and lost the transcript whenever the
process died between the two.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .domain import ConversationEvent, typed_id, utc_timestamp

logger = logging.getLogger("graphiti")


class SessionLockedError(RuntimeError):
    pass


SESSION_STATES = frozenset(
    {
        "active",
        "checkpointing",
        "closing",
        "extracting",
        "publishing",
        "cleanup_pending",
        "retry_locked",
        "completed",
    }
)

# Ordered extraction stages shared by the session progress projection and the
# background extraction-job progress reporter. Mirrors the REAL pipeline that
# reports progress: engine steps (extract/resolve nodes/edges), snapshot
# embed/save, then saga writes (relations, attributes, temporal). The old
# demo-only slots (人物状态栏/可知性判定) were never reported by any call and
# are removed.
EXTRACTION_STAGES = [
    ("extract_nodes", "提取实体"),
    ("resolve_nodes", "去重实体"),
    ("resolve_edges", "解析关系"),
    ("extract_edges", "提取关系"),
    ("embed_and_save", "向量化保存"),
    ("extract_attributes", "写入实体属性"),
    ("resolve_temporal", "时间区间判定与保存"),
]
NON_TERMINAL_STATES = SESSION_STATES - {"completed"}
RECOVERABLE_STATES = {"closing", "extracting", "publishing", "cleanup_pending", "checkpointing"}

#: 空闲窗口默认值：2 分钟裁剪 Live Context，1 小时定义聊天批次边界（封存）。
#: 两者都不抽取；抽取只由 7 天 TTL 巡检触发（见 ``warm_batches.py``）。
#: 来源单一，运行时诊断（``service.runtime_status``）与测试都引用这里。
DEFAULT_CHECKPOINT_SECONDS = 120.0
DEFAULT_FORMAL_IDLE_SECONDS = 3600.0

#: Boundaries that only record that the writer went away. A pet stop shorter than
#: the idle window must not seal, extract or delete anything.
SUSPEND_REASONS = frozenset({"client_close", "pet_stop", "process_exit"})


class ConversationSessionCoordinator:
    VALID_STATES = SESSION_STATES

    def __init__(
        self,
        role_uuid: str,
        conversation_id: str,
        publisher: Optional[Callable[[Dict[str, Any]], Any]] = None,
        checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
        formal_idle_seconds: float = DEFAULT_FORMAL_IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        checkpoint_handler: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
        archive_handler: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
        cleanup_handler: Optional[Callable[[Dict[str, Any]], Any]] = None,
        hot_zone_snapshot: Optional[Dict[str, Any]] = None,
        prompt_additions: Optional[Dict[str, Any]] = None,
        state_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
        restored_state: Optional[Dict[str, Any]] = None,
        owner_epoch: Optional[str] = None,
        inherit_idle_seconds: Optional[float] = None,
        warm_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
        batch_registry: Any = None,
    ):
        self.role_uuid = role_uuid
        self.conversation_id = conversation_id
        #: Deprecated: the coordinator no longer publishes. Kept so an existing
        #: caller (and the offline test harness) can still be constructed; the
        #: attribute is retained for inspection but never invoked.
        self.publisher = publisher
        self.checkpoint_seconds = checkpoint_seconds
        self.formal_idle_seconds = formal_idle_seconds
        # 桌宠重启但 memos 服务未重启时，上一段对话在本窗口内继续沿用（继承）；
        # 服务重启（owner_epoch 变化）或超窗则按"对话已结束"封批。
        self.inherit_idle_seconds = (
            float(inherit_idle_seconds)
            if inherit_idle_seconds is not None
            else float(formal_idle_seconds)
        )
        # 写入本会话的进程实例标识：与会话属主的 epoch 相同才允许继承。
        self.owner_epoch = owner_epoch
        self.clock = clock
        #: Deprecated: checkpoints no longer build a temporary graph. Retained
        #: for signature compatibility with existing callers.
        self.checkpoint_handler = checkpoint_handler
        self.archive_handler = archive_handler or checkpoint_handler
        self.cleanup_handler = cleanup_handler
        self.state_handler = state_handler
        #: Called once per complete turn to make the turn durable in the Warm
        #: Layer. The sink owns the batch registry, the warm outbox and the
        #: WarmStore; a failure is raised so ``record_turn`` can report it.
        self.warm_sink = warm_sink
        #: Mints/seals ``batch_id``\\ s. Shared with the sink so the batch in the
        #: session state and the batch in the Warm Layer are the same object.
        self.batch_registry = batch_registry
        self.state = "active"
        # Wall-clock timestamp is persisted so a UI can keep the timer stable
        # across refreshes and process restarts (monotonic time is used only
        # for idle/checkpoint thresholds).
        self.started_at = utc_timestamp()
        self._progress_log: List[Dict[str, Any]] = []
        self._events: List[ConversationEvent] = []
        self._last_output_at = clock()
        # Wall-clock timestamp of the last completed AI output. This drives the
        # WebUI "idle" timer that surfaces the 2-minute checkpoint and the 1-hour
        # batch-seal windows. It is a wall-clock value (unlike ``_last_output_at``,
        # which is monotonic) so a browser can compute an accurate elapsed value
        # across polls/refreshes.
        self._last_output_at_wall = utc_timestamp()
        self._checkpoint: Optional[Dict[str, Any]] = None
        self._publication_succeeded = False
        self._source_ids: set[str] = set()
        self._source_batch_id: Optional[str] = None
        self._source_layer: str = "LongTermMemory"
        #: 本会话当前聊天批次（角色级注册表分配的 id）。
        self._batch_id: Optional[str] = None
        #: 下一次 ``record_turn`` 会得到的轮次号。召回用它精确排除"当前这一轮"
        #: （``current_round_no``），否则搜索无法把刚记完的一轮与下一轮区分开。
        self._round_no: int = 0
        #: 已成功写入 Warm Layer 的回合 id，保证重试不会重复追加。
        self._warm_flushed: set[str] = set()
        self._warm_failures: List[Dict[str, Any]] = []
        # 用户对当前角色的称呼（AI 在对话中如何称呼用户，默认"主人"）。
        # 由桌宠在记录回合时随 turn 提交，抽取时用它代替 "user:" 标签，
        # 使抽取出的用户实体名是真实称呼而不是 "user"。
        self._user_appellation: Optional[str] = None
        # 外部渠道来源（QQ 私聊/群聊）元数据：本会话最近一次外部来源回合。
        # 抽取时据此把来源当作外部联系人，而不是本机主人。
        self._external_source: Optional[Dict[str, Any]] = None
        self.temporary_graph: Any = None
        # True after a checkpoint trims the live context; checkpoints must not run
        # again until a new turn produces fresh 上文.
        self._checkpointed = False
        self._hot_zone_snapshot = copy.deepcopy(hot_zone_snapshot or {"objects": []})
        #: Session-frozen persona overlay + hot-zone renderings. Read once at
        #: session start so the system prefix cannot change mid-session.
        self._prompt_additions = copy.deepcopy(prompt_additions or {"persona_overlay": [], "hot_zone": []})
        if restored_state:
            self._restore(restored_state)
        self._append_progress_log(self.state, initial=True)
        self._persist()

    @property
    def events(self) -> List[ConversationEvent]:
        return list(self._events)

    @property
    def checkpoint_snapshot(self) -> Optional[Dict[str, Any]]:
        return self._checkpoint

    @property
    def batch_id(self) -> Optional[str]:
        return self._batch_id

    @property
    def current_round_no(self) -> int:
        """The round number the *next* recorded turn will get.

        Recall excludes ``(batch_id, current_round_no)`` so a turn never has its
        own text injected back into its own prompt. Zero means "no turn recorded
        yet in this session": nothing to exclude.
        """
        return self._round_no

    def export_state(self) -> Dict[str, Any]:
        progress = self.progress()
        try:
            elapsed_seconds = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))).total_seconds()))
        except (TypeError, ValueError):
            elapsed_seconds = 0
        return {
            "role_uuid": self.role_uuid,
            "conversation_id": self.conversation_id,
            "state": self.state,
            "started_at": self.started_at,
            "elapsed_seconds": elapsed_seconds,
            "progress": progress,
            "progress_log": copy.deepcopy(self._progress_log[-100:]),
            "events": [event.to_dict() for event in self._events],
            # Live Context only. The transcript itself lives in the Warm Layer and
            # is served by ``/graphiti/ui/warm-layer``; ``entries`` stays for the
            # management UI's compatibility shape.
            "entries": [],
            "checkpointed": self._checkpointed,
            "last_output_at": self._last_output_at,
            "last_output_at_wall": self._last_output_at_wall or self.started_at,
            "last_snapshot": None,
            "checkpoint": self._checkpoint,
            "publication_succeeded": self._publication_succeeded,
            "memos_source_ids": sorted(self._source_ids),
            "source_batch_id": self._source_batch_id,
            "source_layer": self._source_layer,
            "batch_id": self._batch_id,
            "round_no": self._round_no,
            "user_appellation": self._user_appellation,
            # 外部渠道来源（QQ 等）：随会话冻结，抽取时据此把来源当作外部联系人。
            "external_source": copy.deepcopy(self._external_source),
            "hot_zone_snapshot": copy.deepcopy(self._hot_zone_snapshot),
            "prompt_additions": copy.deepcopy(self._prompt_additions),
            # 写入本会话的进程实例标识；新进程据此判断是"桌宠重启（继承）"还是
            # "memos 服务重启（按对话已结束抽取）"。
            "owner_epoch": self.owner_epoch,
            "warm_flushed": sorted(self._warm_flushed),
            "warm_failures": copy.deepcopy(self._warm_failures[-20:]),
            "updated_at": utc_timestamp(),
        }

    def _restore(self, state: Dict[str, Any]) -> None:
        if state.get("role_uuid") != self.role_uuid or state.get("conversation_id") != self.conversation_id:
            raise ValueError("session state belongs to another conversation")
        restored = str(state.get("state") or "active")
        # A process cannot safely resume a partially closed batch in-place. The
        # old design locked the role in ``retry_locked`` until a publication could
        # be finished; extraction now belongs to the batch TTL path, so an
        # interrupted lifecycle simply returns to ``active`` with its preserved
        # live events and lets the batch seal normally.
        if restored in RECOVERABLE_STATES or restored not in self.VALID_STATES:
            self.state = "active"
        elif restored == "retry_locked":
            self.state = "active"
        else:
            self.state = restored
        self.started_at = str(state.get("started_at") or utc_timestamp())
        self._progress_log = [
            dict(item) for item in state.get("progress_log", [])
            if isinstance(item, dict) and item.get("state")
        ][-100:]
        self._events = [
            ConversationEvent(
                event_id=str(event["event_id"]),
                role=str(event["role"]),
                content=str(event["content"]),
                timestamp=str(event["timestamp"]),
                complete=bool(event.get("complete", True)),
                observable=bool(event.get("observable", True)),
                tool_name=event.get("tool_name"),
                metadata=dict(event.get("metadata", {})),
                # 逐回合来源必须随会话一起持久化：恢复后抽取仍要知道每句话是谁说的。
                source=copy.deepcopy(event.get("source")) if isinstance(event.get("source"), dict) else None,
            )
            for event in state.get("events", [])
            if isinstance(event, dict) and event.get("event_id") and event.get("role")
        ]
        self._last_output_at = self._idle_origin(
            state.get("last_output_at_wall") or self.started_at,
            float(state.get("last_output_at", self.clock())),
        )
        self._last_output_at_wall = str(state.get("last_output_at_wall") or self.started_at)
        self._checkpoint = state.get("checkpoint")
        self._checkpointed = bool(state.get("checkpointed", False))
        self._publication_succeeded = bool(state.get("publication_succeeded", False))
        self._source_ids = {str(item) for item in state.get("memos_source_ids", []) if item}
        self._source_batch_id = state.get("source_batch_id")
        self._source_layer = str(state.get("source_layer") or "LongTermMemory")
        self._batch_id = state.get("batch_id")
        self._round_no = int(state.get("round_no") or 0)
        self._warm_flushed = {str(item) for item in state.get("warm_flushed", []) if item}
        self._warm_failures = [
            dict(item) for item in state.get("warm_failures", []) if isinstance(item, dict)
        ]
        self._user_appellation = state.get("user_appellation")
        self._external_source = copy.deepcopy(state.get("external_source"))
        restored_epoch = state.get("owner_epoch")
        if isinstance(restored_epoch, str) and restored_epoch:
            # 会话属主的 epoch 覆盖构造参数：磁盘上的会话属于写它的那个进程实例，
            # 不能因为本次进程也传了 epoch 就被当成"同一进程"。
            self.owner_epoch = restored_epoch
        if isinstance(state.get("hot_zone_snapshot"), dict):
            self._hot_zone_snapshot = copy.deepcopy(state["hot_zone_snapshot"])
        if isinstance(state.get("prompt_additions"), dict):
            self._prompt_additions = copy.deepcopy(state["prompt_additions"])
        if self.state == "completed":
            # The conversation lifecycle is over; its transcript is in the Warm
            # Layer, so nothing is rebuilt from leftover 上文.
            self._events = []
            self._checkpoint = None
            self._checkpointed = False

    def _idle_origin(self, wall: Any, fallback: float) -> float:
        """Map a persisted wall-clock instant onto this process's clock.

        ``last_output_at`` is monotonic and therefore meaningless in a new
        process, so a restored session used to fire its 1-hour formal-idle close
        immediately (or never) depending on the old clock value. Deriving the
        origin from ``last_output_at_wall`` keeps the 2-minute checkpoint and
        1-hour idle windows at their real elapsed time across restarts.
        """
        try:
            moment = datetime.fromisoformat(str(wall).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return fallback
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds())
        return self.clock() - age

    @property
    def hot_zone_snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self._hot_zone_snapshot)

    @property
    def prompt_additions(self) -> Dict[str, Any]:
        return copy.deepcopy(self._prompt_additions)

    @property
    def last_output_at_wall(self) -> str:
        return self._last_output_at_wall or self.started_at

    def inherits_conversation(self, current_epoch: Optional[str]) -> bool:
        """Whether a new client session may continue this conversation.

        A pet restart while the memos service keeps running continues the
        conversation (same ``owner_epoch``, still inside the inherit window); a
        service restart, an expired window or any non-active state ends it, and
        the batch is sealed instead.

        A conversation whose **batch is no longer open** also ends here. Its body
        either moved to the Cold Layer or is waiting for that extraction, so
        continuing it would reopen a session that the batch registry has already
        closed and (in the evicted case) whose transcript no longer exists. That
        is exactly how a *new* conversation could be handed back an old
        ``conversation_id`` by ``_latest_active_conversation``, and then all of its
        turns were filed under the finished conversation's batch.
        """
        if self.state != "active":
            return False
        if not current_epoch or not self.owner_epoch or self.owner_epoch != current_epoch:
            return False
        if not self._batch_is_open():
            return False
        return (self.clock() - self._last_output_at) <= self.inherit_idle_seconds

    def _batch_is_open(self) -> bool:
        registry = self.batch_registry
        if registry is None or not self._batch_id:
            return True
        try:
            batch = registry.store.batch(self._batch_id)
        except Exception:  # noqa: BLE001 - an unreadable registry must not block a turn
            return True
        if batch is None:
            return True
        return str(batch.get("status") or "") in {"", "open"}

    def mark_utterance_end(self) -> Dict[str, Any]:
        """Restart the idle windows at the instant the reply finished speaking.

        ``record_turn`` fires as soon as the text exists — before TTS has played
        it — so the checkpoint window used to elapse while the pet was still
        talking. The client calls this from its TTS-end hook, which makes both
        the 2-minute checkpoint and the 1-hour batch boundary count from real
        end-of-speech.
        """
        if self.state != "active":
            return self.export_state()
        self._last_output_at = self.clock()
        self._last_output_at_wall = utc_timestamp()
        if self.batch_registry is not None:
            try:
                self.batch_registry.note_output()
            except Exception:  # noqa: BLE001 - the window is best-effort
                pass
        self._persist()
        return self.export_state()

    def _persist(self) -> None:
        if self.state_handler:
            self.state_handler(self.export_state())

    def _set_state(self, state: str) -> None:
        if state not in self.VALID_STATES:
            raise ValueError(f"invalid session state: {state}")
        self.state = state
        self._append_progress_log(state)
        self._persist()

    def _append_progress_log(self, state: str, *, initial: bool = False) -> None:
        if self._progress_log and self._progress_log[-1].get("state") == state:
            return
        labels = {
            "active": "对话进行中",
            "checkpointing": "正在裁剪本地上文",
            "closing": "正在封存对话批次",
            "extracting": "正在抽取记忆与实体",
            "publishing": "正在写入 Graphiti 图谱",
            "cleanup_pending": "正在清理临时数据",
            "completed": "批次已封存",
            "retry_locked": "抽取失败，等待重试",
        }
        self._progress_log.append({
            "timestamp": utc_timestamp(),
            "state": state,
            "message": labels.get(state, state),
        })
        self._progress_log = self._progress_log[-100:]

    def progress(self) -> Dict[str, Any]:
        """Expose a stable, UI-friendly projection of the lifecycle state."""
        stages = list(EXTRACTION_STAGES)
        mapping = {
            "active": ("extract_nodes", 0, "对话进行中；抽取在批次到期后进行"),
            "checkpointing": ("extract_nodes", 0, "裁剪已进入温层的本地上文"),
            "closing": ("extract_nodes", 0, "封存本段对话批次"),
            "extracting": ("extract_nodes", 1, "调用 Graphiti 抽取实体、关系和属性"),
            "publishing": ("embed_and_save", 4, "写入实体、关系和向量"),
            "cleanup_pending": ("embed_and_save", 4, "清理临时数据"),
            "completed": ("embed_and_save", 7, "批次已封存，正文保留在温层"),
            "retry_locked": ("extract_nodes", 0, "抽取失败，可重试"),
        }
        key, completed, message = mapping.get(self.state, ("extract_nodes", 0, self.state))
        return {
            "stages": [{"key": item[0], "label": item[1]} for item in stages],
            "current_stage": key,
            "stage_index": next((i for i, item in enumerate(stages) if item[0] == key), 0),
            "stage_total": len(stages),
            "completed": completed,
            "total": len(stages),
            "message": message,
            "active": self.state not in {"completed", "retry_locked"},
            "done": self.state == "completed",
        }

    def _ensure_writable(self) -> None:
        if self.state != "active":
            raise SessionLockedError(f"conversation is {self.state}")

    # -------------------------------------------------------------- turn intake

    def record_turn(
        self,
        user_content: str,
        assistant_content: str,
        tool_events: Optional[List[Dict[str, Any]]] = None,
        complete: bool = True,
        source_ids: Optional[List[str]] = None,
        source_batch_id: Optional[str] = None,
        source_layer: str = "LongTermMemory",
        user_appellation: Optional[str] = None,
        external_source: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record one observable turn, or report why it was refused.

        The refusal path used to ``return`` silently: a conversation whose 上文
        never reached extraction looked like "memory starts mid-conversation"
        with nothing to diagnose. Every refusal now carries a reason that the
        caller logs.

        A refused turn is **not** written to the Warm Layer (it is not a complete,
        observable turn); an accepted turn is durable there before this method
        returns.
        """
        self._ensure_writable()
        tools = list(tool_events or [])
        drop_reason = self._turn_drop_reason(user_content, assistant_content, tools, complete)
        if drop_reason:
            logger.warning(
                "dropped conversation turn (%s): role=%s conversation=%s",
                drop_reason,
                self.role_uuid,
                self.conversation_id,
            )
            return {"recorded": False, "reason": drop_reason}
        turn_id = typed_id("conversation")
        timestamp = utc_timestamp()
        # 逐回合来源：这一轮如果来自 QQ 等外部渠道，就把它钉在**用户那条**事件上，
        # 抽取时按条标注说话人。会话里可以来回换人，所以不能再只靠整批一个"最近
        # 一次外部来源"。
        turn_source = (
            copy.deepcopy(external_source)
            if isinstance(external_source, dict) and str(external_source.get("channel") or "").strip()
            else None
        )
        self._events.append(
            ConversationEvent(turn_id + ":user", "user", user_content, timestamp, source=turn_source)
        )
        self._events.append(ConversationEvent(turn_id + ":assistant", "assistant", assistant_content, timestamp))
        tool_index = 0
        for tool in tools:
            if tool.get("observable", True) and str(tool.get("content", "")).strip():
                tool_index += 1
                # Tool observations are events of **this** turn, so their ids are
                # derived from the turn id. A caller-supplied id would break the
                # "which events belong to this turn" question that the Warm Layer
                # flush and the Live Context trim both ask.
                self._events.append(
                    ConversationEvent(
                        f"{turn_id}:tool:{tool_index}",
                        "tool",
                        str(tool["content"]),
                        tool.get("timestamp", timestamp),
                        tool_name=tool.get("tool_name"),
                    )
                )
        self._last_output_at = self.clock()
        self._last_output_at_wall = timestamp
        # A fresh turn means new 上文 is available for the next checkpoint.
        self._checkpointed = False
        self._source_ids.update(str(item) for item in (source_ids or []) if item)
        if source_batch_id:
            self._source_batch_id = str(source_batch_id)
        if source_layer in {"LongTermMemory", "UserMemory", "WarmLayer"}:
            self._source_layer = source_layer
        if user_appellation and str(user_appellation).strip():
            self._user_appellation = str(user_appellation).strip()[:32]
        if isinstance(external_source, dict) and str(external_source.get("channel") or "").strip():
            self._external_source = copy.deepcopy(external_source)
        warm_status: Dict[str, Any] = {}
        if self.warm_sink is not None:
            warm_status = self._flush_warm(turn_id, turn_source, batch_id=batch_id)
        self._persist()
        result = {"recorded": True, "turn_id": turn_id, "batch_id": self._batch_id}
        result.update(warm_status)
        return result

    def _flush_warm(
        self,
        turn_id: str,
        turn_source: Optional[Dict[str, Any]],
        *,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hand this turn to the Warm Layer sink; raise when it is not durable.

        ``record_turn`` must not answer "recorded" while the body exists only in
        memory, so a sink failure propagates: the caller reports the failure and
        the sweeper repairs it from the sink's own outbox.

        ``batch_id`` is the caller's explicit batch claim. It exists for the one
        case where the batch is not the coordinator's to choose: the **same**
        complete turn delivered by two channels (the desktop pet's own
        ``/sessions/turn`` and, say, a QQ bridge) must not become two batches.
        """
        if turn_id in self._warm_flushed:
            return {"warm": "already_flushed", "batch_id": self._batch_id}
        registry = self.batch_registry
        if registry is None:
            return {"warm": "no_registry", "batch_id": self._batch_id}
        claimed = str(batch_id or "").strip()
        if claimed and self._batch_id is None:
            try:
                adopted = registry.adopt(claimed)
            except Exception:  # noqa: BLE001 - adoption is best-effort bookkeeping
                adopted = None
            if adopted:
                self._batch_id = adopted
        assignment = registry.batch_for_turn(self.conversation_id)
        self._batch_id = assignment.get("batch_id")
        assignment["batch_id"] = self._batch_id
        events: List[Dict[str, Any]] = []
        for event in self._events:
            # Only **this** turn's events. The live window may still hold earlier
            # turns that no checkpoint has trimmed yet, and re-freezing them would
            # give every later turn the same source-event set (so its outbox key
            # would collide and it would claim the earlier turn's round number).
            if not str(event.event_id).startswith(str(turn_id)):
                continue
            if not event.content.strip():
                continue
            events.append(
                {
                    "event_id": event.event_id,
                    "role": event.role,
                    "content": event.content,
                    "timestamp": event.timestamp,
                    "created_at": event.timestamp,
                    "tool_name": event.tool_name,
                    "source": copy.deepcopy(event.source) if isinstance(event.source, dict) else None,
                    "metadata": dict(event.metadata or {}),
                    # 发布侧凭据逐轮冻结在正文旁边，封批时才能恢复成快照字段。
                    "source_ids": sorted(self._source_ids),
                    "source_batch_id": self._source_batch_id,
                    "source_layer": self._source_layer,
                }
            )
        payload = {
            "turn_id": turn_id,
            "role_uuid": self.role_uuid,
            "conversation_id": self.conversation_id,
            "batch_id": self._batch_id,
            "sealed_before": assignment.get("sealed"),
            "events": events,
            # A turn's user input and its tool observations are extractable; the
            # assistant reply is Warm-Layer-only (it is an input to the knowledge
            # pass only through the batch transcript, not a fact of its own).
            "extractable_event_ids": [
                event.event_id for event in self._events
                if str(event.event_id).startswith(str(turn_id))
                and event.role != "assistant"
                and event.content.strip()
            ],
            # 发布侧凭据随这一轮一起冻结（工作记忆 message id / 批次 id / 层）。
            "source_ids": sorted(self._source_ids),
            "source_batch_id": self._source_batch_id,
            "source_layer": self._source_layer,
            "external_source": copy.deepcopy(turn_source),
        }
        try:
            outcome = self.warm_sink(payload) or {}
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never swallowed
            self._warm_failures.append(
                {"turn_id": turn_id, "at": utc_timestamp(), "error": f"{type(exc).__name__}: {exc}"}
            )
            raise
        # Only events that actually reached the journal count as durable. A turn the
        # sink had to queue in the warm outbox is *durable* too (the outbox is on
        # disk) but its entries are not in the Warm Layer yet, so a later search
        # cannot see them and the checkpoint must not treat them as trimmed.
        durable = {str(item) for item in (outcome.get("durable_event_ids") or []) if item}
        if durable:
            self._warm_flushed.update(durable)
        if isinstance(outcome.get("round_no"), int):
            self._round_no = int(outcome["round_no"])
        if assignment.get("sealed"):
            self._append_progress_log("active")
        return {
            "warm": "stored" if not outcome.get("queued") else "queued",
            "batch_id": self._batch_id,
            "round_no": outcome.get("round_no"),
            "sealed_batch": assignment.get("sealed"),
        }

    @staticmethod
    def _turn_drop_reason(
        user_content: Any,
        assistant_content: Any,
        tools: List[Any],
        complete: bool,
    ) -> Optional[str]:
        """The single authoritative rule for "this turn is not extractable"."""
        if not complete:
            return "turn is marked incomplete"
        if not str(user_content or "").strip():
            return "user content is empty"
        if not str(assistant_content or "").strip():
            return "assistant content is empty"
        for tool in tools:
            if not isinstance(tool, dict):
                return "tool observation is not an object"
            if not tool.get("complete", True):
                name = tool.get("tool_name") or "unknown_tool"
                return f"tool observation is incomplete: {name}"
        return None

    # -------------------------------------------------------------- checkpoints

    def checkpoint(self, now: Optional[float] = None, force: bool = False) -> Optional[Dict[str, Any]]:
        """Trim the Live Context. Never extracts, publishes or deletes a body.

        Only events that are already durable in the Warm Layer may be dropped:
        an unflushed turn stays in the live window, because a checkpoint failure
        (or a process death straight after one) must not be able to lose text.
        """
        self._ensure_writable()
        current = self.clock() if now is None else now
        if not force and current - self._last_output_at < self.checkpoint_seconds:
            return None
        if self._checkpointed or not self._events:
            return None
        self._set_state("checkpointing")
        pending = list(self._events)
        trimmed = [event for event in pending if self.is_event_durable(event.event_id)]
        kept = [event for event in pending if not self.is_event_durable(event.event_id)]
        checkpoint: Dict[str, Any] = {
            "role_uuid": self.role_uuid,
            "conversation_id": self.conversation_id,
            "batch_id": self._batch_id,
            "kind": "live_context_trim",
            "trimmed_event_ids": [event.event_id for event in trimmed],
            "kept_event_ids": [event.event_id for event in kept],
            "created_at": utc_timestamp(),
        }
        self._checkpoint = checkpoint
        self._events = kept
        self._checkpointed = not kept
        self._set_state("active")
        return checkpoint

    def is_event_durable(self, event_id: str) -> bool:
        """True when this event already reached the Warm Layer journal."""
        return str(event_id) in self._warm_flushed

    def check_idle(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        # check-idle is a read-mostly probe (the desktop pet polls it every
        # minute). It must never fail with an error just because the session is
        # mid-lifecycle. It advances the two idle windows: the 2-minute Live
        # Context trim, and the 1-hour batch seal (which never extracts).
        if self.state != "active":
            return None
        current = self.clock() if now is None else now
        elapsed = current - self._last_output_at
        if elapsed >= self.formal_idle_seconds:
            return self.seal_batch("idle_1h")
        if elapsed >= self.checkpoint_seconds and not self._checkpointed:
            return self.checkpoint(current)
        return None

    # ------------------------------------------------------------------ sealing

    def seal_batch(self, reason: str = "idle_1h") -> Dict[str, Any]:
        """Seal the current chat batch. Text stays in the Warm Layer.

        A seal ends the batch *boundary*, not the transcript's life: only the TTL
        sweep may extract, and only a confirmed publication may delete.
        """
        if self.state not in {"active"}:
            return self.export_state()
        self._set_state("closing")
        sealed: Optional[str] = None
        if self.batch_registry is not None and self._batch_id:
            try:
                sealed = self.batch_registry.seal_current(reason)
            except Exception as exc:  # noqa: BLE001 - a failed seal keeps the batch open
                logger.warning("batch seal failed (%s): %s", reason, exc)
        self._clear_transient_state()
        self._set_state("completed")
        return {
            "status": "sealed",
            "reason": reason,
            "batch_id": sealed or self._batch_id,
            "conversation_id": self.conversation_id,
            "warm_retained": True,
            "session": self.export_state(),
        }

    def suspend(self, reason: str = "pet_stop") -> Dict[str, Any]:
        """Record that the writer went away. Never seals, extracts or deletes.

        The pet stopping (or a client disconnecting) inside the idle window is not
        a batch boundary: the next turn continues the same batch, which is what
        keeps "stop the pet and reopen it" from splitting a conversation in two.
        """
        if self.state != "active":
            return {"status": "inactive", "reason": reason, "conversation_id": self.conversation_id}
        suspended_at = utc_timestamp()
        if self.batch_registry is not None and self._batch_id:
            try:
                self.batch_registry.store.suspend_batch(self._batch_id, at=suspended_at)
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch suspend failed (%s): %s", reason, exc)
        return {
            "status": "suspended",
            "reason": reason,
            "batch_id": self._batch_id,
            "conversation_id": self.conversation_id,
            "suspended_at": suspended_at,
            "warm_retained": True,
        }

    def _clear_transient_state(self) -> None:
        """Drop session-local references only.

        There is deliberately no ``WarmStore.clear`` (or equivalent) anywhere in
        this module: clearing the live window must be unable to delete a body.
        """
        self.temporary_graph = None
        self._checkpoint = None
        self._events = []
        self._checkpointed = False

    # ------------------------------------------------------------------ ending

    def close(self, reason: str = "explicit_end") -> Dict[str, Any]:
        """End the session lifecycle at a batch boundary.

        ``explicit_end`` / ``new_conversation`` / ``role_switch`` /
        ``service_restart`` are hard boundaries; ``client_close`` / ``pet_stop`` /
        ``process_exit`` only suspend. Both keep the transcript.
        """
        if self.state != "active":
            return {
                "status": "inactive",
                "reason": reason,
                "state": self.state,
                "conversation_id": self.conversation_id,
                "batch_id": self._batch_id,
                "warm_retained": True,
            }
        if reason in SUSPEND_REASONS:
            return self.suspend(reason)
        return self.seal_batch(reason)

    def end_conversation(self) -> Dict[str, Any]:
        return self.close("explicit_end")

    def close_for_new_conversation(self) -> Dict[str, Any]:
        return self.close("new_conversation")

    def on_client_close(self) -> Dict[str, Any]:
        return self.close("client_close")

    def on_pet_stop(self) -> Dict[str, Any]:
        return self.close("pet_stop")

    def on_process_exit(self) -> Dict[str, Any]:
        return self.close("process_exit")
