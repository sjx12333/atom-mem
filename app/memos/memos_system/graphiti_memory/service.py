"""Application-facing facade that composes the Graphiti memory boundaries."""

from __future__ import annotations

import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .activation import ActivationStore
from .async_loop import AsyncLoopRunner
from .authority import AuthorityStore
from .calendar import CalendarStore, parse_moment, render_item as render_calendar_item
from .chatlog import ChatLogStore
from .coordinator import EXTRACTION_STAGES, NON_TERMINAL_STATES as ACTIVE_SESSION_STATES
from .coordinator import SessionLockedError
from .coordinator import (
    DEFAULT_CHECKPOINT_SECONDS,
    DEFAULT_FORMAL_IDLE_SECONDS,
)
from .extraction_engine import EmptyExtractionError, build_extracted_payload, run_graphiti_extraction
from graphiti_core.llm_client.errors import OutputTruncatedError
from .events_protocol import EXTRACTION_RESTRICTIONS
from .diary import DiaryWriter
from .llm_chat import chat_client_from_config
from .memory_extraction import (
    MAX_MEMORIES_PER_SOURCE_BLOCK,
    SOURCE_MEMORY_KIND,
    MemoryExtractor,
    entity_name_lookup,
    normalize_memories,
    resolve_entity_links,
)
from .domain import publication_key, same_publication, text_hash, typed_id, utc_timestamp
from .identity import foreign_identity_scope, qq_node_id, role_name_sets, user_node_id as user_node_uuid
from .extraction import ExtractionPipeline, ExplicitMemoryService
from .extraction_queue import ExtractionScheduler
from .gateway import GraphitiMemoryGateway
from .hot_zone import HotZoneBuilder, HotZoneStore
from .pending import PendingPublicationStore
from .memory_layer import build_memory_layer
from .memory_ops import MemoryOperations
from .neo4j_projection import Neo4jEventProjection
from .operations import AuditLogger, BackupManager, CutoverManager, MaintenanceBarrier, ProjectionStore
from .prompt import RequestSnapshotFactory
from .reflection import OverlayStore, ReflectionScheduler
from .prompt_sections import split_prompt
from .retrieval import RetrievalPolicy, apply_budget
from .roles import RoleAuthority
from .topology import SharedTopology
from .temporary_graph import build_temp_graph_extractor
from .warm_batches import WarmBatchExtractor, WarmBatchRegistry
from .warm_outbox import WarmOutbox, outbox_turn_key
from .warm_store import (
    BATCH_EXTRACTION_PENDING,
    DEFAULT_TTL_SECONDS as WARM_DEFAULT_TTL_SECONDS,
    WarmStore,
    WarmStoreError,
)
from .neo4j_bridge import GraphitiNeo4jBridge
from .role_import import RoleImportCoordinator
from .role_knowledge_import import RoleKnowledgeImportCoordinator
from .identity import source_names
from .contacts import (
    ExternalSourceRegistrar,
    QQ_SOURCE_MARKER,
    declared_external_sources,
    same_named_plain_entities,
)
from .state_changes import (
    TOPIC_CHATLOG,
    TOPIC_EXTRACTION,
    TOPIC_GRAPH,
    TOPIC_SESSION,
    StateChangeHub,
    observe_service,
    session_deadlines,
    session_windows_due,
)
try:
    from graphiti_factory import create_graphiti_client
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from graphiti_factory import create_graphiti_client
from .runtime import event, load_config, load_model_adapters, load_retrieval_policy, neo4j_process_status, public_config, save_config, start_neo4j, stop_neo4j


def event_source_label(item: Dict[str, Any]) -> Optional[str]:
    """一条事件的说话人标签：本机主人/AI 用固定称呼，外部渠道用联系人身份。

    The extractor reads a flat transcript, so "谁说的"只能靠前缀。QQ 轮次以前整批
    只挂一个"最近一次外部来源"，多人来回换人时必然记错人；现在每条用户消息带着
    自己那轮的来源，这里据此写成 `昵称(QQ 号码)`。
    """
    source = item.get("source") if isinstance(item, dict) else None
    if not isinstance(source, dict):
        return None
    number = str(source.get("number") or "").strip()
    display = str(source.get("display_name") or number).strip()
    if not number or not display:
        return None
    if str(source.get("kind") or "private") == "group":
        group = str(source.get("group_name") or source.get("group_number") or "").strip()
        return f"{display}(QQ {number}，群 {group})" if group else f"{display}(QQ {number})"
    return f"{display}(QQ {number})"


def external_sources_in(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """整批里出现过的外部来源，按首次出现去重。

    优先按**逐条事件**的来源统计；旧快照没有逐条来源时退回整批的
    ``external_source``（兼容）。实现见 ``contacts.declared_external_sources``：
    发布侧（``saga``）必须用**同一份**判据决定"这批声明的联系人是谁"，两边各写
    一份就会让抽取把知识挂到新建节点上。
    """
    return declared_external_sources(snapshot)


def external_source_hint(snapshot: Dict[str, Any], user_label: str) -> str:
    """Warn the extractor that this batch contains external-channel turns.

    An inbound QQ turn reaches the pipeline as a ``user`` role message, so without
    this the extractor reads the contact's statements as the owner's facts. The
    hint names the external sender(s) and forbids attributing their statements to
    the fixed user identity. When several contacts appear in one batch it lists
    them all and forbids mixing them with each other as well.

    Module-level on purpose: the caller may build the instructions without holding
    a service instance.
    """
    sources = external_sources_in(snapshot)
    if not sources:
        return ""

    # 单来源：保持原有文案（含"Extract ... under the name 'X'"，抽取侧靠它定实体名）。
    if len(sources) == 1:
        source = sources[0]
        number = str(source.get("number") or "").strip()
        display = str(source.get("display_name") or number).strip()
        if str(source.get("kind") or "private") == "group":
            group_name = str(source.get("group_name") or source.get("group_number") or "").strip()
            where = (
                f"This conversation turn came from a QQ GROUP ({group_name}), from the group member "
                f"'{display}' (QQ {number})."
            )
        else:
            where = (
                f"This conversation turn came from a QQ PRIVATE CHAT with the contact '{display}' (QQ {number})."
            )
        return (
            f" {where} This contact is an EXTERNAL person, NOT the local owner: never attribute this person's "
            f"statements, preferences or identity to '{user_label}', and never merge them into the owner node. "
            f"Extract the contact as its own entity under the name '{display}'. Relations the conversation states "
            f"between the contact and the AI role are relations of the AI's own cognition; do not invent a "
            f"relation between this contact and the owner."
        )

    # 多来源：一段会话里来回换人。逐条标注已经是"谁说的"的权威依据，这里再把
    # "不要张冠李戴"讲清楚。
    described: List[str] = []
    for source in sources:
        number = str(source.get("number") or "").strip()
        display = str(source.get("display_name") or number).strip()
        if str(source.get("kind") or "private") == "group":
            group_name = str(source.get("group_name") or source.get("group_number") or "").strip()
            described.append(
                f"'{display}' (QQ {number}) in QQ GROUP ({group_name})" if group_name
                else f"'{display}' (QQ {number}) in a QQ group"
            )
        else:
            described.append(f"'{display}' (QQ {number})")
    return (
        " This conversation contains turns from SEVERAL external QQ contacts: "
        + ", ".join(described)
        + f". Each user line is prefixed with the contact it came from. These contacts are EXTERNAL people, "
        f"NOT the local owner: never attribute any of their statements, preferences or identity to "
        f"'{user_label}', never merge them into the owner node, and never attribute one contact's statements "
        "to another contact. Extract each contact as its own entity under the exact name shown in its source "
        "label. Relations the conversation states between a contact and the AI role are relations of the AI's "
        "own cognition; do not invent a relation between a contact and the owner."
    )


class GraphitiMemoryService:
    NON_TERMINAL_SESSION_STATES = ACTIVE_SESSION_STATES
    def __init__(
        self,
        root: Path | str,
        extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        retrieval_policy: Optional[RetrievalPolicy | Dict[str, Any]] = None,
        embedder=None,
        reranker=None,
        graphiti_client=None,
        role_import_connection_probe: Optional[Callable[[Dict[str, Any]], None]] = None,
        memory_layer: Any = None,
        vector_store: Any = None,
        memory_chat_client: Any = None,
        neo4j_graph: Any = None,
    ):
        self.root = Path(root)
        self.runtime_config = load_config()
        # The graph bridge is resolved **before** the memory layer, because the
        # layer's vector store runs on the same connection. An injected bridge (a
        # test, or a host that owns the driver) has to be the one the layer uses as
        # well: otherwise a service built on a temporary data root still writes its
        # vectors to whatever database the config points at.
        self.neo4j_graph = (
            neo4j_graph if neo4j_graph is not None else GraphitiNeo4jBridge(self.runtime_config)
        )
        #: An injected bridge is the caller's, so a later config refresh must not
        #: swap it for one built from the file (the same rule the injected memory
        #: chat client follows).
        self._neo4j_graph_injected = neo4j_graph is not None
        self.neo4j_events = Neo4jEventProjection(self.neo4j_graph)
        # A role's memory lives in the shared topology's object attributes; the
        # vectors live inside Neo4j next to the objects they describe, and SQLite
        # keeps only the rebuildable bookkeeping (events, working memory, the
        # embedding fingerprint cache). There is no separate vector database.
        self.memory_layer = memory_layer or build_memory_layer(
            self.root,
            self.runtime_config,
            vector_store=vector_store,
            neo4j_bridge=self.neo4j_graph,
        )
        self.sqlite = self.memory_layer.store
        # The vendored graphiti_core Neo4j driver is async and must be driven
        # from ONE event loop. Everything that touches ``self.graphiti`` (and
        # the backup client) is routed through this single-loop runner so the
        # shared driver is never used from several threads/loops at once —
        # otherwise every query after the first fails with
        # "'NoneType' object has no attribute 'send'".
        self._graphiti_runner = AsyncLoopRunner()
        configured_embedder, configured_reranker, self.model_status = load_model_adapters(self.runtime_config)
        # The retrieval policy is loaded once per service start (ADR-0034: the
        # session captures the policy version, role config may never override it).
        # An explicitly injected policy still wins so tests can drive exact gates.
        effective_policy = retrieval_policy if retrieval_policy is not None else load_retrieval_policy(self.runtime_config)
        self.gateway = GraphitiMemoryGateway(
            self.root,
            policy=effective_policy,
            embedder=embedder or configured_embedder,
            reranker=reranker or configured_reranker,
            memory_layer=self.memory_layer,
        )
        self.authority: AuthorityStore = self.gateway.authority
        self.topology: SharedTopology = self.gateway.topology
        hot_zone_config = (self.runtime_config.get("hot_zone", {}) or {})
        self.activations = ActivationStore(
            self.authority,
            policy=self.gateway.node_recall.policy,
            window_days=float(hot_zone_config.get("hit_window_days", 7) or 7),
        )
        # Recall reads weights; it never writes them. `service.search` records the
        # hit after the final result set is known (one atomic write per request).
        self.gateway.node_recall.activation = self.activations
        embedding_section = self.runtime_config.get("embedding", {}) or {}
        self.embedding_model = str(embedding_section.get("model") or "")
        self.embedding_version = str(
            embedding_section.get("dimension") or embedding_section.get("version") or ""
        )
        # 记忆节点只有日记这一条来源（见 `_write_diary_and_memories`）：知识通路
        # 不再抽取事件，「事件抽取提示词」已废弃且没有读取方，所以这里不再构建
        # 事件抽取器。
        # The post-conversation pass: the AI writes a diary, and the memory nodes
        # are extracted from that diary (never from the raw conversation, so the
        # AI's own viewpoint is what survives). Both use the chat model; see
        # `_memory_chat_client` for which credential the pass prefers.
        diary_section = (self.runtime_config.get("diary", {}) or {})
        diary_client = memory_chat_client or chat_client_from_config(
            self.runtime_config, section=diary_section.get("key_section") or "chat"
        )
        self.diary = DiaryWriter(
            self.root,
            client=diary_client,
            enabled=bool(diary_section.get("enabled", True)),
        )
        #: A caller-supplied chat client (tests, or a host that owns the credential)
        #: takes precedence over the config section on every later config refresh.
        self._memory_chat_client = memory_chat_client
        self.memory_extractor = MemoryExtractor(
            client=diary_client,
            enabled=bool(diary_section.get("enabled", True) and diary_section.get("memory_extraction", True)),
        )
        self.graphiti = graphiti_client or create_graphiti_client(self.runtime_config)

        self.graphiti_enabled = bool(graphiti_client is not None or (self.runtime_config.get("chat", {}).get("enabled") and self.runtime_config.get("chat", {}).get("api_key")))
        # Backup extraction client (a different key) enabling concurrent
        # extraction of two different roles without sharing credentials.
        self._configure_backup_client(allow_create=graphiti_client is None)
        # Background async extraction queue: per-role FIFO, dual-key concurrency.
        self.extraction = ExtractionScheduler(
            concurrency=2 if self.backup_enabled else 1,
            allow_backup=self.graphiti_backup is not None,
        ).start()
        self.roles = RoleAuthority(self.root, graph_identity_callback=self._ensure_ai_node)
        # State-change push channel for the management WebUI: the panel refreshes
        # when one of these topics actually transitions instead of polling whole
        # payloads on a timer (see state_changes.py).
        self.state_changes = StateChangeHub()
        self.state_changes.attach_observer(observe_service(self), session_deadlines)
        # 空闲巡检：桌宠退出不再封批，运行中的后端必须自己把空闲会话推进到
        # 2 分钟检查点 / 1 小时正式结束（原先由桌宠轮询 check-idle 驱动）。
        # 触发条件是"窗口已经到期"（session_windows_due），不能看发布出去的
        # deadline 列表 —— 那里只放未来时刻（避免订阅端空转），已到期的窗口永远
        # 不会出现在里面，巡检线程就会永远不触发。
        self.state_changes.attach_sweeper(
            self.sweep_background_work,
            due_provider=lambda: self._background_work_due(),
        )
        self.state_changes.start()
        # Backfill the default user <-> AI companion edge for any role created
        # before auto-edges existed, so the graph is never left with a pair of
        # isolated identity nodes. Idempotent and cheap when already present.
        try:
            self.roles.ensure_identity_edges()
        except Exception:
            pass
        self.pipeline = ExtractionPipeline(
            self.authority,
            self.topology,
            memory_layer=self.memory_layer,
            embedder=self.gateway.node_recall.embedder,
            neo4j_projection=self.neo4j_events,
            embedding_model=self.embedding_model,
            embedding_version=self.embedding_version,
        )
        self.explicit = ExplicitMemoryService(self.authority)
        self.external_sources = ExternalSourceRegistrar(self.topology, self.authority)
        self.backups = BackupManager(self.authority)
        self.projections = ProjectionStore(self.root, self.topology)
        self.cutover = CutoverManager(self.authority)
        self.maintenance = MaintenanceBarrier(self.authority)
        self.audit = AuditLogger(self.root / "audit" / "retrieval.jsonl")
        self.management_log = AuditLogger(self.root / "audit" / "management.jsonl")
        self.hot_zones = HotZoneStore(self.authority, self.root)
        self.hot_zone_builder = HotZoneBuilder(
            self.authority,
            self.topology,
            self.activations,
            store=self.hot_zones,
            config=(self.runtime_config.get("hot_zone", {}) or {}),
            policy=self.gateway.node_recall.policy,
            token_counter=self.gateway.node_recall.token_counter,
            renderer=self._render_hot_zone_item,
        )
        # 待抽取发布记录：封批时先落盘（文件 + publication_operations pending 行），
        # 发布成功后删除；启动时从这里重放，队列本身只在内存里。
        self.pending = PendingPublicationStore(self.root)
        # 聊天记录查看窗口（每轮用户输入 / 召回记忆 / AI 回复）+ 召回语料
        # （ADR-0163：同一批回合的两份投影，`.json` 给 WebUI 看，`.recall.jsonl`
        # 给召回读；后者 append-only，不受前者的滚动截断影响）。视图窗口保持
        # 原有大小（整文件重写，含每轮的召回结果负载），语料窗口才由策略决定。
        self.chatlog = ChatLogStore(
            self.root,
            recall_max_entries=int(getattr(effective_policy, "chatlog_max_entries", 2000)),
            embedder=self.gateway.node_recall.embedder,
        )
        # 日历视图（ADR-0178）：时间表达句子的按天侧表，纯规则生成、不抽取。
        # 它是**视图**：不进入事实图，重建只需重放回合正文，删角色时一起删除。
        self.calendar = CalendarStore(self.root)
        # 热区记忆开关（插件栏「插件配置」→ hot_zone_memory 同步而来，默认关闭）。
        # 关闭时不再进行热区记忆的轮替：recalculate_hot_zone 直接返回 disabled，
        # 会话启动时也冻结为空热区，热区内容不会进入任何提示词上下文。
        self.hot_zone_enabled = bool(self.authority.load_root_file("hot_zone_preference.json", {}).get("enabled", False))
        self.overlays = OverlayStore(self.authority)
        self.reflections = ReflectionScheduler(self, overlays=self.overlays)
        self.request_snapshots = RequestSnapshotFactory(self.gateway)
        self.extractor = extractor
        # 临时图谱抽取器：与正式抽取共用 chat 配置；未配置时临时图谱退化为
        # 纯文本归档（只向量化，不做节点/边抽取）。
        self._temp_extractor = build_temp_graph_extractor(self.runtime_config)
        self.sessions: Dict[tuple[str, str], Any] = {}
        # ------------------------------------------------ role-level Warm Layer
        # 温层属于**角色**，不属于 session：每个角色一个 WarmStore，正文按 7 天
        # TTL 保留，逐出单位是完整聊天批次。它独立于 session 生命周期，任何
        # close/桌宠停止/角色切换/进程退出都不能删除未被逐出的正文。
        self.warm_outbox = WarmOutbox(self.root)
        self._warm_stores: Dict[str, WarmStore] = {}
        self._warm_registries: Dict[str, WarmBatchRegistry] = {}
        self._warm_extractors: Dict[str, WarmBatchExtractor] = {}
        self.warm_ttl_seconds = WARM_DEFAULT_TTL_SECONDS
        self.log_info: Dict[str, str] = {}
        self.role_import_connection_probe = role_import_connection_probe or (lambda _config: None)
        self.role_imports = RoleImportCoordinator(self)
        # 给已有角色补充世界知识（多份 txt，各自独立可撤销）。与 role_imports 并存：
        # 那条通路造角色，这条通路只往已有角色里加知识。
        self.role_knowledge_imports = RoleKnowledgeImportCoordinator(self)
        # 进程实例标识（epoch）：会话跨重启的"继承 vs 按结束抽取"判据。同一 epoch
        # 表示 memos 服务没有重启，桌宠重启可以继续上一段对话；epoch 变化说明服务
        # 重启过，旧会话按"对话已结束"封批抽取。
        self.runtime_epoch = self._write_runtime_epoch()

    # ------------------------------------------------------- Warm Layer wiring

    def warm_store(self, role_uuid: str) -> WarmStore:
        """The role's Warm Layer (lazily built, one instance per role)."""
        role = str(role_uuid or "")
        if not role:
            raise ValueError("warm store needs a role_uuid")
        store = self._warm_stores.get(role)
        if store is None:
            store = WarmStore(
                self.root,
                role,
                embedder=self.gateway.node_recall.embedder,
                reranker=self.gateway.node_recall.reranker,
            )
            self._warm_stores[role] = store
        return store

    def warm_registry(self, role_uuid: str) -> WarmBatchRegistry:
        """The role's chat-batch registry (mints and seals ``batch_id``\\ s)."""
        role = str(role_uuid or "")
        registry = self._warm_registries.get(role)
        if registry is None:
            registry = WarmBatchRegistry(self.warm_store(role))
            self._warm_registries[role] = registry
        return registry

    def warm_extractor(self, role_uuid: str) -> WarmBatchExtractor:
        """The role's 7-day TTL extractor, wired to the existing publish path."""
        role = str(role_uuid or "")
        extractor = self._warm_extractors.get(role)
        if extractor is None:
            extractor = WarmBatchExtractor(
                self.warm_store(role),
                extract=self._extract_frozen_warm_batch,
                ttl_seconds=self.warm_ttl_seconds,
                audit=lambda action, details: self._log_event(
                    action, details={**details, "role_uuid": role}
                ),
            )
            self._warm_extractors[role] = extractor
        return extractor

    def _warm_roles(self) -> List[str]:
        """Every role that could own a batch: registered, live, or queued."""
        roles: set[str] = set()
        try:
            roles.update(str(item) for item in self.authority.role_ids() if item)
        except Exception:  # noqa: BLE001 - diagnostics must not break a sweep
            pass
        roles.update(str(session.role_uuid) for session in self.sessions.values())
        roles.update(str(key) for key in self._warm_stores)
        roles.update(str(item.get("role_uuid")) for item in self.warm_outbox.list())
        roles.discard("")
        roles.discard("None")
        return sorted(roles)

    def _store_warm_turn(self, role_uuid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make one complete turn durable in the Warm Layer (or in its outbox).

        Order matters (plan 阶段 2.1): the session events are already persisted when
        this runs, so the record is (re)written together with the warm outbox
        record *before* the journal append is attempted. Only after the append
        succeeds is the outbox record dropped, so a failure can never leave the
        turn "recorded" with no durable body anywhere.
        """
        role = str(role_uuid or "")
        store = self.warm_store(role)
        registry = self.warm_registry(role)
        batch_id = str(payload.get("batch_id") or "")
        events = [dict(item) for item in (payload.get("events") or []) if isinstance(item, dict)]
        if not events:
            return {"stored": True, "round_no": None, "entries": 0, "durable_event_ids": []}
        # The registry is the batch authority, but an append that already failed
        # leaves it without a record of the batch it just minted. The coordinator's
        # assignment is the same decision, so adopt it back: otherwise a second
        # failed turn mints *another* batch and every retry claims round 1.
        if batch_id:
            try:
                registry.adopt(batch_id)
            except Exception:  # noqa: BLE001 - adoption is best-effort bookkeeping
                pass
        round_no = self._next_round_no(role, batch_id)
        entries = [
            {
                **event,
                "role_uuid": role,
                "batch_id": batch_id,
                "conversation_id": str(payload.get("conversation_id") or ""),
                "round_no": round_no,
                "created_at": str(event.get("created_at") or event.get("timestamp") or utc_timestamp()),
            }
            for event in events
        ]
        record = {
            "turn_key": outbox_turn_key(
                role, batch_id, round_no, [str(item.get("event_id") or "") for item in entries]
            ),
            "role_uuid": role,
            "batch_id": batch_id,
            "conversation_id": str(payload.get("conversation_id") or ""),
            "round_no": round_no,
            "turn_id": str(payload.get("turn_id") or ""),
            "created_at": utc_timestamp(),
            # 发布侧的来源凭据（工作记忆 message id / 批次 id / 层）随正文一起冻结：
            # 抽取时要用它把这一批的正文和会话归档对应起来，重放也不能丢。
            "source_ids": [str(item) for item in (payload.get("source_ids") or []) if item],
            "source_batch_id": payload.get("source_batch_id"),
            "source_layer": payload.get("source_layer"),
            "entries": entries,
        }
        self.warm_outbox.save(role, record)
        try:
            stored = store.append(entries)
        except Exception as exc:  # noqa: BLE001 - the outbox is the durable fallback
            self._log_event(
                "warm_append",
                status="queued",
                message=str(exc),
                details={"role_uuid": role, "batch_id": batch_id, "round_no": round_no},
            )
            return {
                "stored": False,
                "queued": True,
                "round_no": round_no,
                "entries": len(entries),
                "durable_event_ids": [],
                "error": str(exc),
            }
        self.warm_outbox.remove(role, record["turn_key"])
        # The registry tracked the same batch; keep its idle anchor honest even
        # when the turn arrived through a restored session.
        try:
            registry.note_output()
        except Exception:  # noqa: BLE001
            pass
        return {
            "stored": True,
            "round_no": round_no,
            "entries": len(stored),
            # Real event ids (not the turn id): the trust window and the Live
            # Context trim both work on event identity.
            "durable_event_ids": [
                str(item.get("event_id") or "") for item in entries if item.get("event_id")
            ],
        }

    def _next_round_no(self, role_uuid: str, batch_id: str) -> int:
        """Allocate the next ``round_no`` for a batch.

        The Warm Layer registry is the single allocator (never ChatLogStore, and
        never a per-call counter): the number accounts for bodies already on disk
        *and* turns still waiting in the outbox, so a repaired write cannot claim
        a round number twice.
        """
        store = self.warm_store(role_uuid)
        highest = 0
        for entry in store.entries_for_batch(batch_id):
            highest = max(highest, int(entry.get("round_no") or 0))
        for pending in self.warm_outbox.list(role_uuid):
            if str(pending.get("batch_id") or "") == str(batch_id):
                highest = max(highest, int(pending.get("round_no") or 0))
        return highest + 1

    #: How many times one transcript may be halved before the batch is left for a
    #: plain retry: two levels turn one oversized batch into at most four parts.
    MAX_BISECT_DEPTH = 2

    @staticmethod
    def _publishable_events(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The events a publication actually sends to the extractor."""
        return [
            event
            for event in snapshot.get("events", [])
            if event.get("complete")
            and event.get("observable")
            and str(event.get("content", "")).strip()
        ]

    def _bisect_publication_snapshot(
        self, snapshot: Dict[str, Any], events: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split a frozen batch whose extraction overshot the output budget.

        run_graphiti_extraction sends the whole batch as one prompt, so a long
        batch can exhaust the model's completion budget (OutputTruncatedError)
        and an unchanged retry reproduces the failure. The document importer already
        answers this with bisect_document_chunk; this is the same remedy for a
        chat batch: two snapshots split on message boundaries (never inside one
        message), each with its own batch_id so each publishes as its own episode
        instead of being mistaken for a replay of the other.
        """
        depth = int(snapshot.get("bisect_depth") or 0)
        if len(events) < 2 or depth >= self.MAX_BISECT_DEPTH:
            return []
        total = sum(len(str(event.get("content") or "")) for event in events)
        target, seen, cut = total / 2, 0, 1
        for index, event in enumerate(events[:-1], 1):
            seen += len(str(event.get("content") or ""))
            if seen >= target:
                cut = index
                break
        base = str(snapshot.get("batch_id") or snapshot.get("snapshot_id") or "batch")
        halves: List[Dict[str, Any]] = []
        for suffix, chunk in (("a", events[:cut]), ("b", events[cut:])):
            if not chunk:
                continue
            piece = dict(snapshot)
            piece["events"] = list(chunk)
            piece["batch_id"] = f"{base}.bisect{depth + 1}{suffix}"
            piece["bisect_depth"] = depth + 1
            piece["bisected_from"] = base
            piece["operation_id"] = typed_id("operation")
            piece["publication_operation_id"] = piece["operation_id"]
            halves.append(piece)
        return halves

    def _extract_frozen_warm_batch(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """The Warm Layer's extraction entry point.

        It deliberately delegates to the existing frozen-snapshot publication
        (``_dispatch_publish`` → ``ExtractionPipeline.publish`` → diary/memory
        pass) instead of growing a second extraction implementation on the
        eviction path (plan 阶段 3.2).

        When the publication is **queued** (the real, asynchronous path), the
        batch must not be confirmed here: the graph write has not happened yet.
        The queue job is wrapped so the batch is finalised (published → bodies
        evicted) only after the worker reported success, and marked retryable
        otherwise. Without this wrapper a batch stayed ``extraction_pending``
        forever — the transcript was safe, but nothing ever deleted it and the
        UI showed a permanent "抽取中".
        """
        frozen = dict(snapshot)
        batch_id = str(frozen.get("batch_id") or "")
        if not batch_id or not (self.graphiti_enabled and self.extraction is not None):
            return self._dispatch_publish(frozen)
        role = str(frozen.get("role_uuid") or "")
        extractor = self.warm_extractor(role)

        def run(key, progress=None):
            def publish_part(snap, depth, complete):
                # Publish one snapshot, halving it again if the model still truncates.
                # A single half can still exceed the completion budget on a long
                # session (reasoning tokens count against the output cap), and the old
                # flat two-way split then failed the whole batch forever. Recursive
                # bisection up to depth 2 makes each part small enough.
                try:
                    return [self._extract_and_publish(snap, key, progress, complete=complete)]
                except (OutputTruncatedError, EmptyExtractionError):
                    if depth >= 2:
                        raise
                    halves = self._bisect_publication_snapshot(snap, self._publishable_events(snap))
                    if not halves:
                        raise
                    published = []
                    for half in halves:
                        published.extend(publish_part(half, depth + 1, False))
                    return published

            try:
                outcomes = publish_part(frozen, 0, True)
                if len(outcomes) > 1:
                    self._complete_pending(frozen)
                    outcome = {
                        "status": "published",
                        "bisected": True,
                        "parts": len(outcomes),
                        "manifest": {**(outcomes[-1].get("manifest") or {}), "status": "published"},
                        "diary": outcomes[-1].get("diary"),
                    }
                else:
                    outcome = outcomes[0]
            except Exception as exc:  # noqa: BLE001 - record, keep the transcript
                extractor.store.mark_extraction(
                    batch_id,
                    "retryable_failed",
                    {"error": f"{type(exc).__name__}: {exc}", "attempt_at": utc_timestamp()},
                )
                raise
            return extractor.finalise(batch_id, outcome)

        frozen["operation_id"] = self._publication_operation_id(frozen)
        frozen.setdefault("publication_operation_id", frozen["operation_id"])
        frozen.setdefault("pending_since", frozen.get("created_at") or utc_timestamp())
        self.pending.save(frozen)
        store = getattr(self.memory_layer, "store", None)
        if store is not None:
            store.begin_publication(
                frozen["operation_id"],
                role,
                conversation_id=str(frozen.get("conversation_id") or ""),
                manifest={
                    "status": "pending",
                    "conversation_id": frozen.get("conversation_id"),
                    "batch_id": batch_id,
                    "pending_since": frozen.get("pending_since"),
                },
            )
        job = self.extraction.enqueue(
            role, str(frozen.get("conversation_id") or ""), run, frozen
        )
        return {
            "status": "queued",
            "queue": job,
            "manifest": {"status": "queued", "conversation_id": frozen.get("conversation_id")},
        }

    def _write_runtime_epoch(self) -> str:
        """Stamp this process instance and return its id."""
        epoch = uuid.uuid4().hex
        try:
            self.authority.save_root_file(
                "runtime_epoch.json",
                {"epoch": epoch, "started_at": utc_timestamp()},
            )
        except Exception as exc:  # noqa: BLE001 - identity is best-effort
            self._log_event("runtime_epoch", status="failed", message=str(exc))
        return epoch

    def _configure_backup_client(self, *, allow_create: bool = True) -> None:
        """(Re)build the backup extraction client from the current config.

        Runs at construction and on every runtime-config save so that enabling,
        rotating or disabling the backup key ("抽取备用模型（另一把 key）") takes
        effect without a restart. The scheduler's concurrency is fixed at
        construction, but key failover also works at concurrency 1.
        """
        backup_cfg = self.runtime_config.get("chat_backup", {}) or {}
        self.graphiti_backup = None
        self.backup_enabled = bool(
            backup_cfg.get("enabled")
            and str(backup_cfg.get("api_key") or "").strip()
            and str(backup_cfg.get("model") or "").strip()
        )
        if self.backup_enabled and allow_create:
            try:
                self.graphiti_backup = create_graphiti_client(self.runtime_config, section="chat_backup")
            except Exception:
                self.graphiti_backup = None
                self.backup_enabled = False
        scheduler = getattr(self, "extraction", None)
        if scheduler is not None:
            # Gate on the client that actually exists: a job labelled "backup"
            # would otherwise fall back to the primary key silently.
            scheduler.set_allow_backup(self.graphiti_backup is not None)

    def _ensure_ai_node(self, role_uuid: str, display_name: str) -> None:
        try:
            self.neo4j_graph.ensure_ai_node(role_uuid, display_name)
        except Exception as exc:
            self._log_event("ai_node_sync", status="failed", message=str(exc), details={"role_uuid": role_uuid})

    def sync_identity_nodes(self) -> Dict[str, Any]:
        """Ensure every role's partition has its user node and AI role node.

        One pair per role (ADR-0153): the AI node keyed by the role UUID and the
        user node keyed by ``user:<role_uuid>``, both in the role's own ``group_id``.
        """
        result = {"users": [], "roles": [], "errors": []}
        for role_uuid in self.authority.role_ids():
            config = self.authority.load_role_file(role_uuid, "config.json", {})
            display_name = config.get("display_name", role_uuid)
            appellation = self._role_user_appellation(role_uuid)
            try:
                result["users"].append(self.neo4j_graph.ensure_user_node(role_uuid, appellation))
            except Exception as exc:
                result["errors"].append(f"{role_uuid} user: {exc}")
            try:
                result["roles"].append(self.neo4j_graph.ensure_ai_node(role_uuid, display_name))
            except Exception as exc:
                result["errors"].append(f"{role_uuid}: {exc}")
        return result

    def _role_user_appellation(self, role_uuid: str) -> str:
        """The name this role's user node carries (its own appellation, or 主人)."""
        identity = self.authority.load_role_file(role_uuid, "identity.json", {})
        appellation = str(identity.get("user_appellation") or "").strip()
        return appellation or RoleAuthority.USER_NODE_NAME

    def _remember_user_appellation(self, role_uuid: str, appellation) -> str:
        """Persist how this role addresses its user, so every reader agrees on it.

        The turn carries the appellation the client uses right now, but the readers
        that name the user side look at ``identity.json`` instead: the diary prompt
        (``_speaker_labels``) and the identity node the publication saga keeps
        stable. Left unwritten, a role created without an imported identity keeps
        the default 主人 while the caller says something else, and every fact about
        the user lands under a name the question never mentions (a LoCoMo run that
        asks about "Caroline" then cannot match any of them). The role, not the
        conversation, owns the appellation, so this is role-scoped state and the
        first assignment is what the node keeps.
        """
        role = str(role_uuid or "").strip()
        name = str(appellation or "").strip()[:32]
        if not role or not name:
            return name
        identity = self.authority.load_role_file(role, "identity.json", {}) or {}
        if str(identity.get("user_appellation") or "").strip() == name:
            return name
        identity["user_appellation"] = name
        self.authority.save_role_file(role, "identity.json", identity)
        return name

    def _log_event(
        self,
        action: str,
        *,
        status: str = "info",
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record one structured event (JSONL) plus one text log line.

        ``duration_ms`` is the server-side cost of the stage the event describes.
        The pet measures the same calls from the outside (turn_latency.jsonl), so
        the two numbers together separate "网络/客户端等待" from "后端自己算的".
        """
        if self.log_info:
            event(self.log_info, action, status=status, message=message, details=details, duration_ms=duration_ms)

    def runtime_status(self) -> Dict[str, Any]:
        return {
            "config": public_config(self.runtime_config),
            "model_status": self.model_status,
            "neo4j_status": neo4j_process_status(self.runtime_config),
            "logs": self.log_info,
            "restart_required": False,
            # 会话生命周期诊断：epoch 用于判断"继承还是按结束封批"；窗口用于确认
            # 2 分钟 Live Context 裁剪与 1 小时批次边界是否按预期生效；温层 TTL
            # 是批次真正被抽取（并随后删除正文）的时刻。
            "runtime_epoch": self.runtime_epoch,
            "session_windows": {
                "checkpoint_seconds": DEFAULT_CHECKPOINT_SECONDS,
                "formal_idle_seconds": DEFAULT_FORMAL_IDLE_SECONDS,
                "inherit_idle_seconds": DEFAULT_FORMAL_IDLE_SECONDS,
            },
            "warm_layer": {
                "ttl_seconds": self.warm_ttl_seconds,
                "outbox_pending": len(self.warm_outbox.list()),
                "roles": [
                    {
                        "role_uuid": role,
                        "entries": self.warm_store(role).stats().get("entries"),
                        "batches": self.warm_store(role).stats().get("batches"),
                        "batch_states": self.warm_store(role).stats().get("batch_states"),
                    }
                    for role in self._warm_roles()
                ],
            },
        }

    def check_neo4j(self) -> Dict[str, Any]:
        result = neo4j_process_status(self.runtime_config)
        if result.get("status") == "connected":
            try:
                self.neo4j_graph.ensure_schema()
                identities = self.sync_identity_nodes()
                result["graphiti"] = {"status": "ready", "identities": identities}
            except Exception as exc:
                result["graphiti"] = {"status": "error", "message": str(exc)}
        return result

    def start_neo4j(self) -> Dict[str, Any]:
        return start_neo4j(self.runtime_config)

    def stop_neo4j(self) -> Dict[str, Any]:
        return stop_neo4j(self.runtime_config)

    def update_runtime_config(self, value: Dict[str, Any]) -> Dict[str, Any]:
        previous = self.runtime_config
        self.runtime_config = save_config(value)
        old_graphiti = self.graphiti
        self.graphiti = create_graphiti_client(self.runtime_config)
        self.graphiti_enabled = bool(self.runtime_config.get("chat", {}).get("enabled") and self.runtime_config.get("chat", {}).get("api_key"))
        # A rotated/enabled backup key must take effect now, not after a restart.
        self._configure_backup_client()
        try:
            self._graphiti_runner.run(old_graphiti.driver.close())
        except Exception:
            pass
        if not self._neo4j_graph_injected:
            self.neo4j_graph.close()
            self.neo4j_graph = GraphitiNeo4jBridge(self.runtime_config)
            self.neo4j_events = Neo4jEventProjection(self.neo4j_graph)
        embedding_section = self.runtime_config.get("embedding", {}) or {}
        self.embedding_model = str(embedding_section.get("model") or "")
        self.embedding_version = str(
            embedding_section.get("dimension") or embedding_section.get("version") or ""
        )
        embedder, reranker, self.model_status = load_model_adapters(self.runtime_config)
        self.gateway.node_recall.embedder = embedder
        self.gateway.node_recall.reranker = reranker
        diary_section = (self.runtime_config.get("diary", {}) or {})
        # A caller-supplied client survives a config refresh; otherwise rebuild from
        # the configured credential section.
        if getattr(self, "_memory_chat_client", None) is None:
            diary_client = chat_client_from_config(
                self.runtime_config, section=diary_section.get("key_section") or "chat"
            )
            self.diary.client = diary_client
            self.memory_extractor.client = diary_client
        self.diary.enabled = bool(diary_section.get("enabled", True))
        self.memory_extractor.enabled = bool(
            diary_section.get("enabled", True) and diary_section.get("memory_extraction", True)
        )
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.embedder = embedder
            self.pipeline.saga.embedder = embedder
        if self.log_info:
            event(self.log_info, "runtime_config_updated", details=self.model_status)
        # A changed Neo4j endpoint must be validated immediately.  This makes
        # Save behave like the reference WebUI: the persisted values are also
        # the values used by the next request, and connection errors are
        # visible without requiring a process restart.
        if self.runtime_config.get("database") != previous.get("database"):
            self._last_config_connection = neo4j_process_status(self.runtime_config)
        return self.runtime_status()

    def _build_session(self, role_uuid: str, conversation_id: str):
        """Create/restore the coordinator for one conversation (no lock checks)."""
        key = (role_uuid, conversation_id)
        state = self.authority.load_session(role_uuid, conversation_id)
        role = str(role_uuid)
        hot_zone_snapshot = (
            self.hot_zones.freeze(role_uuid, renderer=self._render_hot_zone_item)
            if self.hot_zone_enabled
            else {"role_uuid": role_uuid, "objects": [], "status": "disabled", "rendered_texts": []}
        )
        overlay_text = self.overlays.active_text(role_uuid) if hasattr(self, "overlays") else ""
        prompt_additions = {
            "persona_overlay": [
                line for line in overlay_text.splitlines() if line.strip()
            ],
            "hot_zone": list(hot_zone_snapshot.get("rendered_texts") or []),
        }
        session = self.gateway.new_session(
            role_uuid,
            conversation_id,
            # The coordinator no longer publishes: a batch is frozen and extracted
            # by the Warm Layer's TTL sweep. The publisher is kept as a no-op slot
            # for callers that still inspect it.
            self._publish_snapshot,
            hot_zone_snapshot=hot_zone_snapshot,
            prompt_additions=prompt_additions,
            # Temporary graphs are no longer the Warm Layer's carrier and no
            # longer call an LLM per entry (plan 阶段 6.2). The class stays for the
            # migration reader and the management UI's compatibility shape.
            temp_extractor=None,
            state_handler=self.authority.save_session,
            restored_state=state or None,
            owner_epoch=self.runtime_epoch,
            warm_sink=lambda payload: self._store_warm_turn(role, payload),
            batch_registry=self.warm_registry(role),
        )
        self.sessions[key] = session
        return session

    def _session_for(self, role_uuid: str, conversation_id: str):
        """Live session for the conversation, restored from disk when absent."""
        session = self.sessions.get((role_uuid, conversation_id))
        if session is None:
            session = self._build_session(role_uuid, conversation_id)
        return session

    def _conflicting_conversations(self, role_uuid: str, conversation_id: str) -> list[tuple[str, str]]:
        """Other non-terminal conversations of the role as ``(id, state)`` pairs.

        A role runs one conversation at a time (CONTEXT "封闭批次不可变性"), so any
        other non-terminal conversation keeps it occupied until it is settled.
        """
        conflicts: Dict[str, str] = {}
        for known in self.sessions.values():
            if (
                known.role_uuid == role_uuid
                and known.conversation_id != conversation_id
                and known.state in self.NON_TERMINAL_SESSION_STATES
            ):
                conflicts[str(known.conversation_id)] = str(known.state)
        for persisted in self.authority.list_sessions():
            if (
                persisted.get("role_uuid") == role_uuid
                and str(persisted.get("conversation_id")) != str(conversation_id)
                and persisted.get("state") in self.NON_TERMINAL_SESSION_STATES
            ):
                conflicts.setdefault(
                    str(persisted.get("conversation_id")), str(persisted.get("state"))
                )
        return sorted(conflicts.items())

    def _settle_superseded_session(self, role_uuid: str, conversation_id: str) -> None:
        """Seal or finish a superseded conversation so it stops holding the role.

        CONTEXT "切换时批次封闭": the batch is sealed and handed to its own
        background extraction queue instead of blocking the next conversation
        (a restart used to leave an un-retryable lock that blocked the role
        forever). Only ``active`` (never sealed) and ``retry_locked`` (sealed,
        unpublished) conversations are settled; a closing/extracting/publishing
        session belongs to another worker and is left untouched.
        """
        session = self._session_for(role_uuid, conversation_id)
        if session.state == "active":
            session.close("new_conversation")
        elif session.state == "retry_locked":
            session.retry()

    def _inheritable_conversation(
        self, role_uuid: str, conflicts: list[tuple[str, str]]
    ) -> Optional[str]:
        """The conflicting conversation this request should continue instead.

        规则（会话生命周期）：桌宠重启但 memos 服务没有重启时，上一段对话继续
        沿用；memos 服务重启（epoch 变化）、超出继承窗口或会话已非 active 时，
        按"对话已结束"封批抽取。只在**恰好一个**冲突会话时才继承——存在多段
        遗留会话时情况本身是歧义的，交给既有的封批路径更安全。
        """
        if len(conflicts) != 1:
            return None
        conflict_id = conflicts[0][0]
        try:
            session = self._session_for(role_uuid, conflict_id)
        except Exception:  # noqa: BLE001 - an unreadable session cannot be inherited
            return None
        return conflict_id if session.inherits_conversation(self.runtime_epoch) else None

    def start_session(
        self,
        role_uuid: str,
        conversation_id: str,
        *,
        settle_superseded: bool = False,
    ):
        if self.maintenance.status().get("status") != "clear":
            raise SessionLockedError("memory maintenance is active")
        key = (role_uuid, conversation_id)
        if key in self.sessions:
            return self.sessions[key]
        conflicts = self._conflicting_conversations(role_uuid, conversation_id)
        inherited = self._inheritable_conversation(role_uuid, conflicts)
        if inherited is not None:
            # 桌宠换了会话 id 但服务没重启：这是同一段对话的延续，不封批也不抽取，
            # 临时图谱与上文继续可用（返回的会话 id 即权威 id，客户端会采纳）。
            return self._session_for(role_uuid, inherited)
        if conflicts and settle_superseded:
            for conflict_id, _state in conflicts:
                try:
                    self._settle_superseded_session(role_uuid, conflict_id)
                except Exception as exc:  # noqa: BLE001
                    # The old batch must publish before the role is released; keep
                    # the barrier, but say which conversation is holding it.
                    raise SessionLockedError(
                        f"role {role_uuid} has an unpublished conversation: {conflict_id} ({exc})"
                    ) from exc
            conflicts = self._conflicting_conversations(role_uuid, conversation_id)
        if conflicts:
            raise SessionLockedError(
                f"role {role_uuid} has an unfinished conversation: {conflicts[0][0]}"
            )
        return self._build_session(role_uuid, conversation_id)

    def _run_graphiti_async(self, awaitable):
        """Run a graphiti_core coroutine on the single dedicated async loop.

        graphiti_core's async driver (Neo4jDriver) must be used from one event
        loop. Spawning a fresh ``asyncio.run`` per call (the previous
        ``_run_async``) used the shared driver from many different loops, which
        corrupted its connection pool and surfaced as
        ``'NoneType' object has no attribute 'send'`` on every query. All
        calls funnel through ``self._graphiti_runner`` so the driver is always
        driven from the same loop.
        """
        return self._graphiti_runner.run(awaitable)

    def _extract_and_publish(self, snapshot: Dict[str, Any], key: Optional[str] = None, progress: Optional[Callable[..., None]] = None, *, complete: bool = True) -> Dict[str, Any]:
        """Run one extraction + publish for a frozen snapshot.

        ``key`` selects the extraction client: ``"backup"`` uses the backup key
        (different credential) when available, anything else uses the primary.
        When Graphiti is disabled (offline/test) the configured ``extractor`` is
        used synchronously.
        """
        events = [event for event in snapshot.get("events", []) if event.get("complete") and event.get("observable") and str(event.get("content", "")).strip()]
        if not events:
            self._discard_pending(snapshot)
            return {"status": "published", "manifest": {"status": "published", "conversation_id": snapshot.get("conversation_id"), "published_ids": []}}
        if self._publication_is_published(snapshot):
            # 重放一个已经发布的批次：只做清理，绝不重复抽取（ADR-0141）。
            self._complete_pending(snapshot)
            return {
                "status": "published",
                "manifest": self.authority.load_role_file(
                    str(snapshot.get("role_uuid") or ""), "publication_manifest.json", {}
                ),
            }
        extracted = None
        # Speaker labels are needed by both the extraction instructions and the
        # diary body, so they are resolved before either branch can run.
        user_label, assistant_label = self._speaker_labels(snapshot)
        if self.graphiti_enabled:
            client = self.graphiti_backup if key == "backup" and self.graphiti_backup else self.graphiti
            body = self._render_transcript(events, user_label, assistant_label)
            # Deterministic episode id so re-processing the same frozen snapshot
            # reuses the same episode. We do NOT hand it to graphiti's
            # ``add_episode`` (which treats a supplied uuid as already-existing
            # and raises ``EpisodicNode.get_by_uuid`` -> NodeNotFoundError on a
            # first run). Instead we drive the low-level extract/resolve/save
            # pipeline (the reference project's first three steps) which builds
            # the episode directly.
            #
            # The batch id is part of the identity: two batches of one conversation
            # would otherwise share an episode uuid, and the second batch would be
            # extracting into an episode that already exists.
            episode_seed = f"graphiti:{snapshot['role_uuid']}:{snapshot['conversation_id']}"
            if str(snapshot.get("batch_id") or "").strip():
                episode_seed = f"{episode_seed}:{snapshot['batch_id']}"
            episode_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, episode_seed))
            pipeline_result = self._run_graphiti_async(run_graphiti_extraction(
                client,
                episode_uuid=episode_uuid,
                body=body,
                role_uuid=str(snapshot["role_uuid"]),
                conversation_id=str(snapshot["conversation_id"]),
                progress=progress,
                custom_extraction_instructions=self._extraction_instructions(snapshot, user_label, assistant_label),
                assistant_label=assistant_label,
            ))
            extracted = build_extracted_payload(
                {**pipeline_result, "raw_text": body}
            )
            if not any(
                (
                    extracted.get("nodes"),
                    extracted.get("relations"),
                    extracted.get("memories"),
                    extracted.get("perspective"),
                    extracted.get("events"),
                )
            ) and not str(pipeline_result.get("raw_text") or "").strip():
                # The extractor produced nothing **and** returned no raw output at
                # all over a non-empty batch: that is the observable signature of a
                # reasoning model that spent its whole token budget and answered
                # with empty content (``finish_reason=length``), not a conversation
                # with nothing to extract. Publishing it would silently drop this
                # conversation's knowledge, so surface it as a retryable failure. A
                # model that answered with a parseable "nothing here" is a legal
                # empty result and publishes as such.
                raise EmptyExtractionError(
                    f"extraction returned no content at all for conversation "
                    f"{snapshot.get('conversation_id')} ({len(events)} messages)"
                )
        if extracted is None:
            if self.extractor is None:
                # Offline installations without a configured chat endpoint
                # retain the legacy authority-only lifecycle. A configured
                # Graphiti client never takes this path: extraction failures
                # propagate and leave the session retry-locked.
                if self.runtime_config.get("chat", {}).get("api_key"):
                    raise RuntimeError("Graphiti extractor is unavailable")
                extracted = {"memories": []}
            else:
                extracted = self.extractor("\n".join(item["content"] for item in events)) or {}
        result = self.pipeline.publish(snapshot, lambda _text: extracted, progress)
        # 发布成功后才删除归档正文与待抽取记录（ADR-0128 / ADR-0147）。
        # A bisected publication defers this to its caller, which completes the
        # original batch only once every half has published.
        if complete:
            self._complete_pending(snapshot)
        # QQ 会话的来源节点与默认边：抽取这一趟一定会跑完，而"收到消息即硬加入"
        # 的请求可能整趟都没到（网关未起、重试耗尽、进程重启）。按这一批冻结正文里
        # 的真实来源补一次，缺节点就建、缺边就补，并把抽取出的同名联系人实体也连到
        # 角色节点上。纯属修复，失败只记日志，绝不影响已经成功的发布。
        try:
            sources = external_sources_in(snapshot)
            if sources:
                result["external_sources"] = self.reconcile_external_sources(
                    str(snapshot.get("role_uuid") or ""), sources=sources
                )
        except Exception as exc:  # noqa: BLE001 - repair must never fail a publish
            self._log_event(
                "external_sources_reconciled",
                status="failed",
                message=str(exc),
                details={
                    "role_uuid": snapshot.get("role_uuid"),
                    "conversation_id": snapshot.get("conversation_id"),
                },
            )
        # The second pass runs after the knowledge pass has committed, so a memory
        # node can link to an entity the same conversation produced. It never fails
        # the batch: knowledge is already published and a diary is best-effort, and
        # an unconfigured deployment simply reports "skipped".
        try:
            body = self._render_transcript(events, user_label, assistant_label)
            diary = self._write_diary_and_memories(snapshot, body=body, progress=progress)
        except Exception as exc:  # noqa: BLE001 - the diary must not fail a publish
            self._log_event(
                "diary_failed",
                status="failed",
                message=str(exc),
                details={"role_uuid": snapshot.get("role_uuid"), "conversation_id": snapshot.get("conversation_id")},
            )
            diary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        # The pass runs after the manifest was written, so its outcome is attached
        # to the manifest: without that, a diary that was skipped or failed would
        # leave no durable trace at all.
        manifest_on_disk = self.authority.load_role_file(
            str(snapshot.get("role_uuid") or ""), "publication_manifest.json", {}
        )
        if same_publication(manifest_on_disk, snapshot):
            manifest_on_disk["diary"] = diary
            self.authority.save_role_file(
                str(snapshot.get("role_uuid") or ""), "publication_manifest.json", manifest_on_disk
            )
        result["diary"] = diary
        return result

    # ------------------------------------------------------- diary + memory pass

    @staticmethod
    def _date_label() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _write_diary_and_memories(
        self,
        snapshot: Dict[str, Any],
        *,
        body: str,
        progress: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """Write the AI's diary for one finished conversation, then mine it.

        Order matters and is the whole point: the conversation archive is already
        durable before this runs (the frozen snapshot exists and the raw text is
        only deleted after a successful knowledge publish), so a crash here loses
        the diary at worst, never the conversation.
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        if not role_uuid:
            return {"status": "skipped", "reason": "no role"}
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        display_name = str(config.get("display_name") or "").strip()
        appellation = self._role_user_appellation(role_uuid)
        persona = str(self._management_prompt(role_uuid).get("persona") or "").strip()
        if progress:
            progress("diary", "AI 写日记", 8)
        written = self.diary.write(
            role_uuid=role_uuid,
            conversation_id=str(snapshot.get("conversation_id") or ""),
            batch_id=str(snapshot.get("batch_id") or ""),
            conversation=body,
            persona=persona,
            appellation=appellation,
            display_name=display_name,
            date_label=self._date_label(),
            source_message_ids=self._source_message_ids(snapshot),
        )
        if written.get("status") != "written":
            return written
        entry = written["entry"]
        if progress:
            progress("memory_extraction", "从日记抽取记忆节点", 9)
        extraction = self.memory_extractor.extract(
            diary=str(entry.get("text") or ""),
            appellation=appellation,
            display_name=display_name,
        )
        published = self._publish_memory_nodes(
            role_uuid=role_uuid,
            conversation_id=str(snapshot.get("conversation_id") or ""),
            diary=entry,
            memories=extraction.get("memories") or [],
        )
        return {**extraction, "published": published, "diary_id": entry.get("id")}

    def _publish_memory_nodes(
        self,
        *,
        role_uuid: str,
        conversation_id: str,
        diary: Dict[str, Any],
        memories: List[Dict[str, Any]],
        source_kind: str = "conversation_happened",
        extra_name_lookup: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Publish each memory candidate as a memory node plus its entity links.

        这是 `kind="event"` 记忆节点的**唯一生产者**：知识通路不抽取事件（见
        `PublicationSaga` 与 `ExtractionPipeline.publish` 的说明），临时图谱在
        会话内抽到的事件也不进入发布。每个记忆节点带自己的日记总结正文，并以
        `memory_link` 连到日记提到的、本角色分区内真实存在的知识节点。

        ``source_kind`` names where the memory came from: a real conversation this
        service witnessed, or (文档创建角色) one block of the source document. It is
        an attribute of the memory node, so it has to be honest.

        ``extra_name_lookup`` adds name→node mappings on top of the partition's
        knowledge nodes. The document-import pass needs it for the source aliases
        ("原文角色名"/"原文用户名"): those resolve onto the role's own identity
        nodes, whose *display* names may still be placeholders while the import is
        unconfirmed, so a lookup built from node names alone would miss them.
        """
        if not memories:
            return []
        self.topology.reload()
        # Identity names are ambiguous across partitions: every role's user node is
        # named after the same appellation ("主人"), so a lookup built from *all*
        # identity nodes resolved "主人" to whichever node happened to be inserted
        # first — measurably another role's user node. Only this role's own two
        # identity nodes may be addressed by name; every other candidate must be an
        # object this role was actually granted.
        own_identity = {str(role_uuid), user_node_uuid(str(role_uuid))}
        knowledge_nodes = [
            node
            for node in self.topology.data.get("nodes", {}).values()
            if str(node.get("id") or "") in own_identity
            or (
                str(node.get("kind") or "") != "event"
                and self.topology.is_allowed(role_uuid, str(node.get("id") or ""))
            )
        ]
        lookup = entity_name_lookup(knowledge_nodes)
        if extra_name_lookup:
            # Aliases resolve *onto* the identity nodes, so they may legitimately
            # override a plain-name hit (a knowledge node that happens to be named
            # like the source alias is not this role's own identity).
            lookup.update({str(name).casefold(): str(node) for name, node in extra_name_lookup.items()})
        # A grant that has since been proven to come from the contaminated lookup is
        # not repaired by itself: an already-linked object stays readable until it is
        # revoked, so never link to (or grant) an identity node of another partition.
        own_user = str(user_node_uuid(str(role_uuid)))
        foreign_identity = {
            str(node_id)
            for node_id in self.topology.data.get("whitelist", {}).get(str(role_uuid), {}).get("nodes", [])
            if str(node_id).startswith("user:") and str(node_id) != own_user
        }
        if foreign_identity:
            for node_id in sorted(foreign_identity):
                self.topology.revoke_node(str(role_uuid), node_id)
        published: List[Dict[str, Any]] = []
        for index, memory in enumerate(memories):
            summary = str(memory.get("summary") or "").strip()
            if not summary:
                continue
            event_id = self._diary_memory_id(diary, index, summary)
            try:
                self.topology.create_event_node(
                    role_uuid,
                    event_id,
                    summary,
                    occurred_at=memory.get("occurred_at") or diary.get("date"),
                    source_kind=str(source_kind or "conversation_happened"),
                    source_conversation_id=conversation_id,
                    content_hash=text_hash(summary),
                )
                self.topology.grant_node(role_uuid, event_id)
            except Exception as exc:  # noqa: BLE001 - one bad memory must not stop the rest
                published.append({"id": event_id, "status": "failed", "error": str(exc)})
                continue
            links = [node_id for node_id in resolve_entity_links(memory.get("related_entities") or [], lookup)]
            for node_id in links:
                try:
                    relation = self.topology.upsert_relation(
                        event_id, node_id, kind=self.topology.MEMORY_LINK_KIND
                    )
                    self.topology.grant_relation(role_uuid, relation["id"])
                except Exception:  # noqa: BLE001 - a link is optional, the memory is not
                    continue
            self.pipeline.saga._index_object(role_uuid, event_id)
            published.append({"id": event_id, "status": "published", "links": links})
        return published

    @staticmethod
    def _diary_memory_id(diary: Dict[str, Any], index: int, summary: str) -> str:
        """Deterministic memory-node id so a replayed batch never duplicates one.

        The batch id is part of the seed for the same reason it is part of the
        episode: two batches of one conversation must not mint the same node for
        two different memories (the second would silently overwrite the first).
        A diary without a batch id keeps the original seed.
        """
        batch = str(diary.get("batch_id") or "")
        seed = "diary-memory:{role}:{conversation}:{batch}:{index}:{text}".format(
            role=diary.get("role_uuid"),
            conversation=diary.get("conversation_id"),
            batch=batch,
            index=index,
            text=text_hash(summary),
        )
        return f"e_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    def publish_import_memories(
        self,
        *,
        role_uuid: str,
        conversation_id: str,
        memories: Sequence[Dict[str, Any]],
        appellation: str = "",
        display_name: str = "",
        source_role_name: str = "",
        source_user_name: str = "",
    ) -> Dict[str, Any]:
        """Publish one document-import **memory batch** (the import's second pass).

        文档创建角色 对每个原文分块分两批发布：第一批是知识节点（走
        ``ExtractionPipeline.publish`` → ``PublicationSaga``），第二批就是这里 ——
        该分块抽出的记忆写成 ``kind="event"`` 记忆节点并 ``memory_link`` 到相关实体。

        它单独记一条 ``publication_operations`` 行（有自己的 ``conversation_id``），
        所以"两批"在账本上各自可审计：知识批的 manifest 不会被记忆批覆盖，
        记忆批失败也不回滚已经发布的知识。摘要为空则整批为空，是合法结果。
        """
        role_uuid = str(role_uuid or "").strip()
        conversation_id = str(conversation_id or "").strip()
        if not role_uuid:
            raise ValueError("role_uuid is required")
        if not conversation_id:
            raise ValueError("conversation_id is required")
        items = normalize_memories(list(memories or []), limit=MAX_MEMORIES_PER_SOURCE_BLOCK)
        identity = self.authority.load_role_file(role_uuid, "identity.json", {})
        resolver, user_names, ai_names = self.pipeline.saga._identity_resolver(
            role_uuid,
            str(display_name or role_uuid),
            str(appellation or "").strip() or "主人",
            str(source_user_name or identity.get("source_user_name") or ""),
            str(source_role_name or identity.get("source_role_name") or ""),
        )
        alias_lookup: Dict[str, str] = {}
        for name in list(user_names) + list(ai_names):
            target = resolver(name)
            if target:
                alias_lookup[str(name).casefold()] = str(target)
        manifest: Dict[str, Any] = {
            "kind": "import_memory_batch",
            "role_uuid": role_uuid,
            "conversation_id": conversation_id,
            "source_kind": SOURCE_MEMORY_KIND,
            "candidates": len(items),
            "status": "publishing",
            "created_at": utc_timestamp(),
        }
        store = getattr(self.memory_layer, "store", None)
        operation_id = self._publication_operation_id(
            {
                "role_uuid": role_uuid,
                "conversation_id": conversation_id,
                "snapshot_id": f"import-memory:{text_hash(conversation_id)}",
            }
        )
        if store is not None:
            store.begin_publication(
                operation_id, role_uuid, conversation_id=conversation_id, manifest=manifest
            )
        try:
            published = self._publish_memory_nodes(
                role_uuid=role_uuid,
                conversation_id=conversation_id,
                diary={"role_uuid": role_uuid, "conversation_id": conversation_id},
                memories=items,
                source_kind=SOURCE_MEMORY_KIND,
                extra_name_lookup=alias_lookup,
            )
            published_ids = [str(item.get("id")) for item in published if item.get("status") == "published"]
            failed = [item for item in published if item.get("status") != "published"]
            manifest.update({
                "status": "published",
                "published_ids": published_ids,
                "published": len(published_ids),
                "failed": failed,
                "published_at": utc_timestamp(),
            })
            revision = store.bump_revision() if store is not None else None
            if store is not None:
                manifest["sqlite_revision"] = revision
                store.complete_publication(
                    operation_id,
                    published_ids=published_ids,
                    source_message_ids=[],
                    embedding_model=self.embedding_model or None,
                    embedding_version=self.embedding_version or None,
                    manifest=manifest,
                )
            return {
                "status": "published",
                "operation_id": operation_id,
                "role_uuid": role_uuid,
                "conversation_id": conversation_id,
                "published_ids": published_ids,
                "published": published,
                "candidates": len(items),
                "revision": revision,
            }
        except Exception as exc:
            manifest.update({"status": "retryable", "error": f"{type(exc).__name__}: {exc}"})
            if store is not None:
                store.fail_publication(operation_id, str(exc))
            raise


    @staticmethod
    def _source_message_ids(snapshot: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        for event in snapshot.get("events", []):
            value = str(event.get("event_id") or event.get("id") or event.get("message_id") or "").strip()
            if value and value not in ids:
                ids.append(value)
        return ids

    # 知识类实体规则：抽取只有两类内容——可被描述的知识实体与事件。
    # 比喻/类比中被比较的对象不是实体（"她温柔得像天使" 里的"天使" 不抽取）。
    # 解析层会再次执行同一条规则（graphiti_memory.knowledge），不只依赖提示词。
    _KNOWLEDGE_CLASS_EXCLUSION = (
        "Explicit extraction classes: output only knowledge entities and events. "
        "A knowledge entity's name must be a real thing the dialogue discusses and can describe "
        "(e.g. \"主人养了一只叫大橘的猫，大橘很黏人\" -> entity \"大橘\"). "
        "Metaphor and analogy exclusion: NEVER extract an entity that appears only as the target of "
        "a comparison, metaphor, simile, or analogy (for example, in \"she is as gentle as an angel\" "
        "do NOT extract \"angel\"; in \"he is strong as an ox\" do NOT extract \"ox\"; in \"他壮得像头牛\" "
        "do NOT extract \"牛\"; in \"她的歌声像夜莺\" do NOT extract \"夜莺\"; in \"脸红的像苹果\" do NOT "
        "extract \"苹果\"). Only extract real, concrete entities that actually exist in the conversation. "
        "Entities used purely as a figure of speech must be skipped even if they look like a proper noun."
    )

    #: Kept as an alias: external tooling and older tests read this name.
    _METAPHOR_EXCLUSION = _KNOWLEDGE_CLASS_EXCLUSION

    def _speaker_labels(self, snapshot: Dict[str, Any]) -> tuple[str, str]:
        """真实发言角色名称：用户侧用当前角色的称呼（默认"主人"），AI 侧用角色显示名。

        抽取文本不再使用 "user:"/"assistant:" 标签，否则 LLM 会把 "user"/"assistant"
        当作实体抽取进图谱（用户节点重复、AI 变成 assistant 节点）。为多角色协同
        做准备：每个发言角色都记录其真实名称。
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        # The frozen snapshot only carries the appellation when the warm batch
        # recorded it; the role identity file is the durable answer (see
        # _remember_user_appellation), so an older snapshot still gets named.
        appellation = (
            str(snapshot.get("user_appellation") or "").strip()
            or (self._role_user_appellation(role_uuid) if role_uuid else "")
            or "主人"
        )
        display_name = ""
        if role_uuid:
            config = self.authority.load_role_file(role_uuid, "config.json", {})
            display_name = str(config.get("display_name") or "").strip()
        return appellation, (display_name or "assistant")

    def _extraction_instructions(self, snapshot: Dict[str, Any], user_label: str, assistant_label: str) -> str:
        """注入抽取 LLM 的附加指令：统一限制词 + 称呼映射。

        限制词来自 ``events_protocol.EXTRACTION_RESTRICTIONS``，与临时图谱抽取
        共用同一份文本（阶段 3：统一正式抽取和临时抽取的限制词），事件字段在
        解析层再次校验，不只依赖 LLM 遵守提示词。
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        identity_hint = (
            f"In this conversation the speaker labelled 'user' is called '{user_label}' and the speaker "
            f"labelled 'assistant' is called '{assistant_label}'. When you extract an entity that refers to "
            f"the user or this AI, use exactly '{user_label}' / '{assistant_label}' as the entity name, never "
            f"'user' or 'assistant'. "
            if role_uuid
            else ""
        )
        identity = self.authority.load_role_file(role_uuid, "identity.json", {}) if role_uuid else {}
        source_role = source_names(snapshot.get("source_role_name") or identity.get("source_role_name") or "")
        source_user = source_names(snapshot.get("source_user_name") or identity.get("source_user_name") or "")
        alias_hint = ""
        if source_role or source_user:
            alias_hint = (
                f" In source text, these names ALL refer to the SAME role character: {source_role or [assistant_label]}. "
                f"These names ALL refer to the SAME user: {source_user or [user_label]}. "
                "The '/' separator lists aliases, not separate people. Resolve every alias to its fixed AI/user "
                "identity; merge their attributes and relationships and do not create duplicate identity nodes."
            )
        return f"{EXTRACTION_RESTRICTIONS}\n{identity_hint}{alias_hint}{external_source_hint(snapshot, user_label)}".strip()

    # ------------------------------------------------- 待抽取发布记录（ADR-0147）

    def _publication_operation_id(self, snapshot: Dict[str, Any]) -> str:
        """Stable operation id for one frozen batch (a replay must not duplicate).

        A warm batch is keyed by its ``batch_id`` so that *retrying* the same batch
        reuses the same operation (``build_snapshot`` mints a fresh ``snapshot_id``
        per call, which would otherwise make every retry look like a new
        publication and could publish a half-finished batch twice). A legacy
        session snapshot keeps the original seed.
        """
        existing = str(snapshot.get("operation_id") or "").strip()
        if existing:
            return existing
        batch_id = str(snapshot.get("batch_id") or "").strip()
        if batch_id:
            seed = "publication-batch:{role}:{batch}".format(
                role=snapshot.get("role_uuid"), batch=batch_id
            )
        else:
            seed = "publication:{role}:{conversation}:{snapshot}".format(
                role=snapshot.get("role_uuid"),
                conversation=snapshot.get("conversation_id"),
                snapshot=snapshot.get("snapshot_id"),
            )
        return f"o_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    def _persist_pending(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """先落盘再入队：文件（正文）+ ``publication_operations`` pending 行。

        队列是内存的；封批与抽取之间重启会让任务消失，而归档正文在发布成功前
        不会被删除。落盘失败必须抛出：否则这一批就退化成"只在内存里"，与旧行为
        一样会静默丢失。
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        conversation_id = str(snapshot.get("conversation_id") or "")
        operation_id = self._publication_operation_id(snapshot)
        snapshot["operation_id"] = operation_id
        snapshot.setdefault("publication_operation_id", operation_id)
        snapshot.setdefault("pending_since", snapshot.get("created_at") or utc_timestamp())
        self.pending.save(snapshot)
        store = getattr(self.memory_layer, "store", None)
        if store is not None:
            store.begin_publication(
                operation_id,
                role_uuid,
                conversation_id=conversation_id,
                manifest={
                    "status": "pending",
                    "conversation_id": conversation_id,
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "pending_since": snapshot.get("pending_since"),
                },
            )
        return snapshot

    def _complete_pending(self, snapshot: Dict[str, Any]) -> None:
        """发布成功后：清理归档正文，再删除待抽取记录。

        顺序固定：先 stub 归档（ADR-0128 的"发布成功后才删正文"），再删记录。
        但"清理失败"不能连记录一起删掉——那样归档正文就再也没有任何东西指向它，
        既不会重试也永远不会被清理。清理失败时保留记录，重放时 saga 见已发布会直接
        返回，这里再清理一次（幂等）。
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        conversation_id = str(snapshot.get("conversation_id") or "")
        cleaned = True
        try:
            self.pipeline.cleanup(snapshot)
        except Exception as exc:  # noqa: BLE001 - cleanup is retryable, never fatal
            cleaned = False
            self._log_event(
                "pending_publication",
                status="failed",
                message=f"archive cleanup failed: {exc}",
                details={"role_uuid": role_uuid, "conversation_id": conversation_id},
            )
        if not cleaned:
            # Keep the record so the next replay/startup retries the cleanup. It
            # already carries the publication's operation id, so the retry can only
            # clean up — never extract twice.
            return
        try:
            self.pending.delete_for(snapshot)
        except OSError as exc:  # noqa: BLE001
            self._log_event(
                "pending_publication",
                status="failed",
                message=f"pending record cleanup failed: {exc}",
                details={"role_uuid": role_uuid, "conversation_id": conversation_id},
            )

    def _discard_pending(self, snapshot: Dict[str, Any]) -> None:
        """A batch with nothing extractable leaves no pending record behind."""
        try:
            self.pending.delete_for(snapshot)
        except OSError:
            pass

    def _publication_is_published(self, snapshot: Dict[str, Any]) -> bool:
        """True when this frozen batch is already published (replay must not re-extract).

        Matched by **batch identity** (``domain.same_publication``): two batches of
        one conversation share the ``conversation_id``, and the old
        conversation-only comparison made the second batch look published — its
        knowledge was never extracted and its transcript was still deleted.
        """
        role_uuid = str(snapshot.get("role_uuid") or "")
        manifest = self.authority.load_role_file(role_uuid, "publication_manifest.json", {})
        if same_publication(manifest, snapshot) and manifest.get("status") == "published":
            return True
        store = getattr(self.memory_layer, "store", None)
        operation_id = str(snapshot.get("operation_id") or "")
        if store is not None and operation_id:
            row = store.get_publication(operation_id)
            if row and row.get("status") == "done":
                return True
        return False

    def _dispatch_publish(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """落盘后交给后台队列（离线/测试环境同步执行）。"""
        snapshot = self._persist_pending(snapshot)
        # Real Graphiti extraction is asynchronous: enqueue a job to the
        # background scheduler so two different roles can be extracted
        # concurrently (primary + backup key) without blocking the conversation.
        if self.graphiti_enabled and self.extraction is not None:
            def run(key, progress=None):
                return self._extract_and_publish(snapshot, key, progress)
            job = self.extraction.enqueue(
                str(snapshot.get("role_uuid")),
                str(snapshot.get("conversation_id")),
                run,
                snapshot,
            )
            return {"status": "queued", "queue": job, "manifest": {"status": "queued", "conversation_id": snapshot.get("conversation_id")}}
        # Offline/test extractor path stays synchronous.
        return self._extract_and_publish(snapshot, None, None)

    def _publish_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        events = [event for event in snapshot.get("events", []) if event.get("complete") and event.get("observable") and str(event.get("content", "")).strip()]
        if not events:
            self._discard_pending(snapshot)
            return {"status": "published", "manifest": {"status": "published", "conversation_id": snapshot.get("conversation_id"), "published_ids": []}}
        return self._dispatch_publish(snapshot)

    def recover_pending_publications(self) -> Dict[str, Any]:
        """启动重放：把上次进程遗留的待抽取批次重新交给抽取队列。

        队列只在内存里，而归档正文在发布成功前不删除；因此"关掉后端/整机后再
        启动"必须从这里恢复，否则那批对话永远不会被抽取而 UI 显示全绿。
        已发布的批次只做清理（不会重复抽取），失败的重放留在盘上等下一次。
        """
        try:
            records = self.pending.list()
        except Exception as exc:  # noqa: BLE001 - startup must not die on a bad record
            self._log_event("pending_replay", status="failed", message=str(exc))
            return {"records": 0, "requeued": 0, "published": 0, "failed": [{"error": str(exc)}]}

        requeued: List[Dict[str, Any]] = []
        published: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for snapshot in records:
            role_uuid = str(snapshot.get("role_uuid") or "")
            conversation_id = str(snapshot.get("conversation_id") or "")
            if not role_uuid or not conversation_id:
                continue
            try:
                if self._publication_is_published(snapshot):
                    self._complete_pending(snapshot)
                    published.append({"role_uuid": role_uuid, "conversation_id": conversation_id})
                    continue
                self._dispatch_publish(snapshot)
                requeued.append({"role_uuid": role_uuid, "conversation_id": conversation_id})
            except Exception as exc:  # noqa: BLE001 - one bad batch must not stop the replay
                failed.append({"role_uuid": role_uuid, "conversation_id": conversation_id, "error": str(exc)})
        if records:
            self._log_event(
                "pending_replay",
                status="failed" if failed else "info",
                message=f"{len(records)} pending batch(es) at startup",
                details={"requeued": len(requeued), "published": len(published), "failed": len(failed)},
            )
        return {
            "records": len(records),
            "requeued": len(requeued),
            "published": len(published),
            "failed": failed,
        }

    def start_selected_session(self, role_uuid: str, conversation_id: str):
        self.roles.select(role_uuid)
        previous_roles = {
            session.role_uuid
            for session in self.sessions.values()
            if session.role_uuid != role_uuid and session.state in self.NON_TERMINAL_SESSION_STATES
        }
        previous_roles.update(
            state["role_uuid"]
            for state in self.authority.list_sessions()
            if state.get("role_uuid") != role_uuid
            and state.get("state") in self.NON_TERMINAL_SESSION_STATES
            and isinstance(state.get("role_uuid"), str)
        )
        for previous_role in sorted(previous_roles):
            self.close_active_sessions("role_switch", role_uuid=previous_role)
        return self.start_session(role_uuid, conversation_id)

    def select_role(self, role_uuid: str) -> Dict[str, Any]:
        return self.roles.select(role_uuid)

    def _restored_from_another_epoch(self, session: Any) -> bool:
        """True when this live session was restored from a dead process instance.

        ``owner_epoch`` 与当前进程不符，说明这段对话是上一个 memos 进程留下的：
        按会话继承规则，它应当被封批抽取，而不是被本次进程继续往下写。
        """
        epoch = getattr(session, "owner_epoch", None)
        return bool(epoch) and bool(self.runtime_epoch) and epoch != self.runtime_epoch

    def _prepare_turn_session(self, session: Any, role_uuid: str):
        """Return a writable session for this turn, minting a new one when needed.

        三种必须换会话的情况：
        - 服务重启过（epoch 不同）而会话仍是 ``active``：旧对话按"已结束"封批；
        - 会话已经 ``completed``（例如 1 小时空闲窗口已经封批）；
        - 会话 ``retry_locked``：这是旧设计留下的状态，恢复时已经归一为
          ``active``，这里只需要把残留状态放回可写。

        封批与抽取已经解耦：这里封存的是**批次边界**，正文继续留在温层，等 TTL
        巡检再抽取。新会话 id 由后端生成并随响应交回，客户端会采纳。
        """
        if session.state == "retry_locked":
            # Legacy state: extraction is no longer tied to the session, so the
            # lock only has to stop being a lock.
            session.state = "active"
        if session.state == "active" and not self._restored_from_another_epoch(session):
            return session
        if session.state == "active" and self._restored_from_another_epoch(session):
            session.close("service_restart")
        elif session.state == "active":
            # Not an epoch change: this is the role's *previous* conversation and
            # the caller already asked for a different one, so the batch boundary
            # is "a new conversation", not "a restart".
            session.close("new_conversation")
        return self._build_session(role_uuid, f"c_{uuid.uuid4().hex}")

    def record_turn(self, role_uuid: str, conversation_id: str, **turn: Any) -> Dict[str, Any]:
        try:
            # 耗时口径：方法入口 → 事件点（会话结算 + 回合记录本身）。后面的
            # chatlog 写入与向量投递是后台任务，不计入这一步。
            started = time.perf_counter()
            # 面板记账用的召回凭据不属于回合本身：会话协调器不认识它，必须在
            # 交给会话之前取出，否则 ``session.record_turn(**turn)`` 会因未知参数
            # 直接报错（ADR-0166）。
            recall_token = turn.pop("recall_token", None)
            # 历史重放（导入/评测）的回合时间是**发生时间**，不是写库时间：
            # 日历视图里的「今天/明天」必须按它解析，否则 2023 年的对话会被
            # 解析到今天的日期上（ADR-0178）。
            calendar_at = parse_moment(turn.pop("chat_time", None))
            # A turn must never be lost because a superseded (for example
            # restart-interrupted) conversation of the same role still occupies
            # it: seal that batch into the background queue and continue.
            session = self.start_session(role_uuid, conversation_id, settle_superseded=True)
            session = self._prepare_turn_session(session, role_uuid)
            # Learn the appellation before anything names the user side: the warm
            # batch is extracted later, and both the diary prompt and the identity
            # node read the role identity file at that point.
            self._remember_user_appellation(role_uuid, turn.get("user_appellation"))
            # 后端是会话 id 的权威：继承时这里返回的是上一段对话的 id，客户端据此
            # 采纳（否则同一段对话会被记到两个 conversation 下）。
            effective_conversation_id = session.conversation_id
            status = session.record_turn(**turn)
            # 被拒绝的回合必须留痕：宠物端看到的现象是"记忆从对话中途开始"，
            # 而后端以前对丢弃这条路径完全静默，无法定位。
            self._log_event(
                "session_turn",
                status="info" if status.get("recorded") else "dropped",
                message=str(status.get("reason") or ""),
                details={"role_uuid": role_uuid, "conversation_id": effective_conversation_id, "complete": turn.get("complete", True)},
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            # 聊天记录窗口：仅记录完整可观察回合（与会话协调器相同的门槛）。
            # 纯查看用途，不影响会话事件与后续抽取。
            recorded = bool(
                turn.get("complete", True)
                and str(turn.get("user_content", "")).strip()
                and str(turn.get("assistant_content", "")).strip()
            )
            if recorded:
                self.chatlog.record_turn(
                    role_uuid,
                    effective_conversation_id,
                    str(turn.get("user_content", "")),
                    str(turn.get("assistant_content", "")),
                    recall_token=recall_token,
                )
                # 日历视图：同一批正文里**带时间表达**的句子进侧表（纯规则，
                # 零 LLM 成本）。写失败不影响本轮记忆：视图缺一条不是记忆丢了。
                try:
                    self.calendar.record_turn(
                        role_uuid,
                        effective_conversation_id,
                        str(turn.get("user_content", "")),
                        str(turn.get("assistant_content", "")),
                        at=calendar_at,
                    )
                except Exception as exc:  # noqa: BLE001 - a derived view never fails a turn
                    self._log_event(
                        "calendar_write",
                        status="failed",
                        message=str(exc),
                        details={"role_uuid": role_uuid, "conversation_id": effective_conversation_id},
                    )
                # A turn is the transition the chatlog view exists to show.
                self.state_changes.mark(
                    TOPIC_CHATLOG,
                    origin="turn_recorded",
                    detail={"role_uuid": role_uuid, "conversation_id": effective_conversation_id},
                )
            # The turn also moves the session's 2-minute / 1-hour idle windows.
            self.state_changes.mark(
                TOPIC_SESSION,
                origin="turn_recorded",
                detail={"role_uuid": role_uuid, "conversation_id": effective_conversation_id},
            )
            return status
        except Exception as exc:
            self._log_event("session_turn", status="failed", message=str(exc), details={"role_uuid": role_uuid, "conversation_id": conversation_id}, duration_ms=(time.perf_counter() - started) * 1000)
            raise

    def checkpoint(self, role_uuid: str, conversation_id: str, now: Optional[float] = None, force: bool = False):
        """Trim the Live Context (2-minute checkpoint).

        The checkpoint no longer builds a temporary graph and no longer creates an
        extraction source: the warm body was already written at the end of the
        turn, so all that is left for this signal is dropping the events that are
        safely on disk from the session's in-memory window.
        """
        result = self.start_session(role_uuid, conversation_id).checkpoint(now, force=force)
        if result is not None:
            self.state_changes.mark(
                TOPIC_SESSION,
                origin="live_context_trim",
                detail={"role_uuid": role_uuid, "conversation_id": conversation_id},
            )
        return result

    def check_idle(self, role_uuid: str, conversation_id: str, now: Optional[float] = None):
        return self.start_session(role_uuid, conversation_id).check_idle(now)

    def effective_conversation_id(self, role_uuid: str, conversation_id: str) -> str:
        """后端认可的会话 id（继承时是上一段对话的 id）。

        只有"能挂上会话"时才返回别的 id：存在未结算的旧会话（服务重启/超窗）时
        必须保留客户端自己的 id，否则下一次 record_turn 会把旧会话当成同一段继续，
        而不是封批抽取它。
        """
        if not conversation_id:
            return conversation_id
        try:
            return self.start_session(role_uuid, conversation_id).conversation_id
        except SessionLockedError:
            return conversation_id

    def mark_utterance_end(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        """把空闲窗口的起点挪到"这轮真的说完了"。

        record_turn 在文本生成完就发生（早于 TTS 播放），所以倒计时原先会在 AI
        说话期间就走；桌宠在 TTS 结束时调用本方法重新锚定 2 分钟检查点与 1 小时
        正式结束窗口。
        """
        try:
            session = self.start_session(role_uuid, conversation_id)
        except SessionLockedError:
            return {"status": "degraded", "role_uuid": role_uuid, "conversation_id": conversation_id}
        state = session.mark_utterance_end()
        self.state_changes.mark(
            TOPIC_SESSION,
            origin="utterance_end",
            detail={"role_uuid": role_uuid, "conversation_id": session.conversation_id},
        )
        return {"status": "ok", "conversation_id": session.conversation_id, "session": state}

    def _settlement_work_due(self) -> bool:
        """True when a hot zone or reflection window may have elapsed."""

        now = time.time()
        try:
            roles = self._warm_roles()
        except Exception:  # noqa: BLE001
            return False
        if getattr(self, "hot_zone_enabled", False) and getattr(self, "hot_zones", None) is not None:
            quiet = float((self.runtime_config.get("hot_zone", {}) or {}).get("settle_quiet_seconds", 3600) or 3600)
            for role in roles:
                last = self._role_last_output_epoch(role)
                if last is None or now - last <= quiet:
                    continue
                current = self.hot_zones.load(role)
                if not current.get("recalculated_at") or current.get("last_output_at") != last:
                    return True
        reflection = self.runtime_config.get("reflection", {}) or {}
        if bool(reflection.get("enabled", True)) and getattr(self, "reflections", None) is not None:
            for role in roles:
                try:
                    state = self.reflections._load_state(role)
                except Exception:  # noqa: BLE001
                    continue
                if float(state.get("next_due_at") or 0.0) <= now and not self._role_has_active_session(role):
                    return True
        return False

    def _background_work_due(self) -> bool:
        """True when the composite sweeper has something to do right now.

        Cannot be derived from the hub's published deadlines: those hold only
        *future* instants, while the Warm Layer's TTL, its outbox and its missing
        vectors are all conditions that are either true now or not at all. Warm
        Layer state is checked first and independently of any session, which is
        the whole point of not hanging eviction off "sweep the active sessions"
        (plan 阶段 3.1).
        """
        if self._settlement_work_due():
            return True
        if session_windows_due(self):
            return True
        try:
            if self.warm_outbox.list():
                return True
        except Exception:  # noqa: BLE001 - a broken outbox must not stall the sweep
            return True
        for role in self._warm_roles():
            store = self.warm_store(role)
            try:
                if store.pending_vectors(limit=1):
                    return True
                if store.failed_batches() or store.pending_batches():
                    return True
                if store.idle_open_batches() or store.evictable_batches(ttl=self.warm_ttl_seconds):
                    return True
            except Exception:  # noqa: BLE001
                return True
        return False

    def sweep_idle_sessions(self) -> Dict[str, Any]:
        """推进空闲会话：2 分钟裁剪 Live Context / 1 小时封存聊天批次。

        两者都不抽取。桌宠运行时由它轮询 ``check-idle``；桌宠退出后不再封批，
        所以仍在运行的 memos 服务必须自己把空闲会话推到裁剪或封批（由
        StateChangeHub 的巡检线程调用，见 ``state_changes.attach_sweeper``）。
        """
        advanced: List[Dict[str, Any]] = []
        for (role_uuid, conversation_id), session in list(self.sessions.items()):
            if getattr(session, "state", None) != "active":
                continue
            try:
                result = session.check_idle()
            except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
                self._log_event(
                    "idle_sweep",
                    status="failed",
                    message=str(exc),
                    details={"role_uuid": role_uuid, "conversation_id": conversation_id},
                )
                continue
            if result is None:
                continue
            advanced.append({"role_uuid": role_uuid, "conversation_id": conversation_id})
            self.state_changes.mark(
                TOPIC_SESSION,
                origin="idle_sweep",
                detail={"role_uuid": role_uuid, "conversation_id": conversation_id},
            )
        return {"advanced": advanced}

    def sweep_warm_outbox(self) -> Dict[str, Any]:
        """补写失败的温层正文（角色级 outbox），与 session 状态完全无关。"""
        repaired: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for record in self.warm_outbox.list():
            role = str(record.get("role_uuid") or "")
            if not role:
                continue
            store = self.warm_store(role)
            key = str(record.get("turn_key") or "")
            entries = [dict(item) for item in (record.get("entries") or []) if isinstance(item, dict)]
            if not entries:
                self.warm_outbox.remove(role, key)
                continue
            try:
                store.append(entries)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the sweep
                attempts = int(record.get("attempts") or 0) + 1
                self.warm_outbox.save(
                    role,
                    {
                        **record,
                        "attempts": attempts,
                        "last_error": str(exc),
                        "last_attempt_at": utc_timestamp(),
                    },
                )
                failed.append({"role_uuid": role, "turn_key": key, "error": str(exc), "attempts": attempts})
                continue
            self.warm_outbox.remove(role, key)
            repaired.append({"role_uuid": role, "turn_key": key, "entries": len(entries)})
            self.state_changes.mark(
                TOPIC_SESSION, origin="warm_outbox", detail={"role_uuid": role}
            )
        return {"repaired": repaired, "failed": failed}

    def sweep_warm_vectors(self) -> Dict[str, Any]:
        """补算缺失/失败的温层向量。正文永远保留，缺向量只走词法回退。"""
        filled: List[Dict[str, Any]] = []
        still_missing = 0
        embedder = self.gateway.node_recall.embedder
        for role in self._warm_roles():
            store = self.warm_store(role)
            for entry in store.pending_vectors(limit=16):
                vector = None
                if embedder is not None:
                    try:
                        vector = embedder(str(entry.get("content") or ""))
                    except Exception:  # noqa: BLE001 - stays pending for the next sweep
                        vector = None
                if store.fill_vector(str(entry.get("id")), vector):
                    filled.append({"role_uuid": role, "id": entry.get("id")})
                else:
                    still_missing += 1
        return {"filled": filled, "still_missing": still_missing}

    def sweep_warm_evictions(self) -> Dict[str, Any]:
        """独立巡检修剪：遍历所有角色的 sealed batch，与 session 状态无关。

        批次的**最新**正文超过 TTL 才可抽取；抽取成功（知识发布 + 日记/记忆通路
        都到终态）后才删除正文，任何异常都保留正文并记 ``retryable_failed``。
        """
        published: List[Dict[str, Any]] = []
        requeued: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        sealed: List[Dict[str, Any]] = []
        for role in self._warm_roles():
            store = self.warm_store(role)
            extractor = self.warm_extractor(role)
            try:
                idle = extractor.seal_idle()
                for batch_id in idle:
                    sealed.append({"role_uuid": role, "batch_id": batch_id})
                outcome = extractor.sweep()
            except Exception as exc:  # noqa: BLE001 - one role must not stop the sweep
                self._log_event(
                    "warm_sweep", status="failed", message=str(exc), details={"role_uuid": role}
                )
                continue
            published.extend({"role_uuid": role, **item} for item in outcome.get("published", []))
            requeued.extend({"role_uuid": role, **item} for item in outcome.get("requeued", []))
            failed.extend({"role_uuid": role, **item} for item in outcome.get("failed", []))
            del store
        if published or requeued or failed or sealed:
            self.state_changes.mark(
                TOPIC_EXTRACTION,
                origin="warm_sweep",
                detail={
                    "published": len(published),
                    "requeued": len(requeued),
                    "failed": len(failed),
                    "sealed": len(sealed),
                },
            )
        return {
            "published": published,
            "requeued": requeued,
            "failed": failed,
            "sealed": sealed,
        }

    def sweep_hot_zones(self) -> Dict[str, Any]:
        """Rebuild each quiet role's hot zone; never touch a live session."""

        built: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for role in self._warm_roles():
            try:
                result = self.build_hot_zone(role)
            except Exception as exc:  # noqa: BLE001 - one role must not stop the sweep
                failed.append({"role_uuid": role, "error": str(exc)})
                continue
            status = str(result.get("status") or "")
            if status == "updated":
                built.append({"role_uuid": role, "objects": len(result.get("objects") or [])})
            elif status in {"unchanged", "disabled"}:
                unchanged.append({"role_uuid": role, "reason": result.get("reason")})
            else:
                failed.append({"role_uuid": role, "status": status, "reason": result.get("reason")})
        if built:
            self.state_changes.mark(
                TOPIC_GRAPH,
                origin="hot_zone_settled",
                detail={"roles": [item["role_uuid"] for item in built]},
            )
        return {"built": built, "unchanged": unchanged, "failed": failed}

    def sweep_reflections(self) -> Dict[str, Any]:
        """Generate at most one idle reflection per sweep (see reflection.py)."""

        scheduler = getattr(self, "reflections", None)
        if scheduler is None:
            return {"status": "disabled", "generated": []}
        try:
            return scheduler.sweep()
        except Exception as exc:  # noqa: BLE001 - reflection must never break chat
            self._log_event("reflection_sweep", status="failed", message=str(exc))
            return {"status": "failed", "error": str(exc), "generated": []}

    def sweep_background_work(self) -> Dict[str, Any]:
        """The composite idle executor registered with the state-change hub.

        Three independent sweeps, none of which depends on a session being
        active: warm outbox repair, warm vector backfill, and the TTL eviction of
        sealed batches. ``sweep_idle_sessions`` stays separate because it is the
        only one that advances live sessions.
        """
        result: Dict[str, Any] = {}
        for name, sweep in (
            ("sessions", self.sweep_idle_sessions),
            ("outbox", self.sweep_warm_outbox),
            ("vectors", self.sweep_warm_vectors),
            ("evictions", self.sweep_warm_evictions),
            ("hot_zone", self.sweep_hot_zones),
            ("reflection", self.sweep_reflections),
        ):
            try:
                result[name] = sweep()
            except Exception as exc:  # noqa: BLE001 - one sweep must not stop the others
                result[name] = {"error": str(exc)}
        return result

    def warm_layer_status(self, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Warm Layer diagnostics: counts, batch states and due batches."""
        roles = [str(role_uuid)] if role_uuid else self._warm_roles()
        return {
            "ttl_seconds": self.warm_ttl_seconds,
            "outbox": len(self.warm_outbox.list(role_uuid)),
            "roles": [
                {
                    "role_uuid": role,
                    **self.warm_extractor(role).status(),
                }
                for role in roles
            ],
        }

    def warm_layer_snapshot(self, role_uuid: str) -> Dict[str, Any]:
        """Every retained batch/body of a role, for the management UI.

        The batch rows are the management shape (``_warm_batch_records``) rather than
        the store's raw registry rows: the 温层 page renders ``entries`` and
        ``checkpoint_ready`` per batch, and the store's own rows have neither.
        """
        snapshot = self.warm_store(role_uuid).snapshot()
        snapshot["batches"] = self._warm_batch_records(role_uuid)
        return snapshot

    def trigger_warm_batch_extraction(
        self, role_uuid: str, batch_ids: Sequence[str], *, reason: str = "manual_checkpoint"
    ) -> Dict[str, Any]:
        """Send the selected batches to extraction now instead of at the TTL.

        This is the manual form of the 7-day checkpoint: the batch must be sealed
        first (a still-open batch would be split mid-conversation), then it is
        frozen and extracted through the same path the TTL sweep uses. Bodies are
        deleted only if that extraction confirms publication.

        Every selected batch is independent: one batch's failure is reported in its
        own result row and never stops the others (multi-select is a batch loop, not
        an all-or-nothing transaction). A batch that is already frozen/queued is
        reported as such instead of being re-frozen, so a double click cannot give
        one batch two publication jobs.
        """
        store = self.warm_store(role_uuid)
        extractor = self.warm_extractor(role_uuid)
        results: List[Dict[str, Any]] = []
        for batch_id in [str(item) for item in (batch_ids or []) if item]:
            batch = store.batch(batch_id)
            if batch is None:
                results.append({"batch_id": batch_id, "status": "not_found"})
                continue
            if batch.get("status") == BATCH_EXTRACTION_PENDING or extractor.is_running(batch_id):
                # Already frozen and waiting for the worker: it is *not* a failure,
                # and re-freezing it would enqueue the same batch twice.
                results.append(
                    {
                        "batch_id": batch_id,
                        "status": "queued",
                        "reason": "already_in_flight",
                        "deferred": True,
                    }
                )
                continue
            if batch.get("status") == "open":
                store.seal_batch(batch_id, reason=reason)
            try:
                results.append(extractor.extract_batch(batch_id, reason=reason))
            except Exception as exc:  # noqa: BLE001 - report, keep the transcript
                results.append(
                    {"batch_id": batch_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
        self.state_changes.mark(
            TOPIC_EXTRACTION,
            origin="manual_warm_checkpoint",
            detail={"role_uuid": role_uuid, "batches": len(results)},
        )
        return {"role_uuid": role_uuid, "results": results}

    def seal_warm_batch(self, role_uuid: str, batch_ids: Sequence[str], *, reason: str = "manual_seal") -> Dict[str, Any]:
        """Seal the selected batches (封存对话批次) without extracting them."""
        store = self.warm_store(role_uuid)
        sealed: List[Dict[str, Any]] = []
        for batch_id in [str(item) for item in (batch_ids or []) if item]:
            result = store.seal_batch(batch_id, reason=reason)
            if result is None:
                sealed.append({"batch_id": batch_id, "status": "not_found"})
            else:
                sealed.append({"batch_id": batch_id, "status": result.get("status"), "reason": result.get("seal_reason")})
        if sealed:
            self.state_changes.mark(
                TOPIC_SESSION,
                origin="manual_seal",
                detail={"role_uuid": role_uuid, "batches": len(sealed)},
            )
        return {"role_uuid": role_uuid, "batches": sealed}

    def recover_warm_layer(self) -> Dict[str, Any]:
        """Startup recovery for the three durable Warm Layer job classes.

        ``warm outbox`` (bodies that never reached the journal), ``pending``
        batches (frozen and possibly mid-publish) and ``failed`` batches (both
        passes did not reach a terminal state) are all re-driven here. The
        publication is idempotent on the batch's operation id, so a crash after
        the knowledge pass does not publish twice.
        """
        outbox = self.sweep_warm_outbox()
        recovered: List[Dict[str, Any]] = []
        for role in self._warm_roles():
            try:
                recovered.extend(self.warm_extractor(role).recover().get("recovered", []))
            except Exception as exc:  # noqa: BLE001 - startup must never die here
                recovered.append({"role_uuid": role, "error": str(exc)})
        if outbox.get("repaired") or recovered:
            self._log_event(
                "warm_recovery",
                details={"repaired": len(outbox.get("repaired") or []), "recovered": len(recovered)},
            )
        return {"outbox": outbox, "recovered": recovered}


    def _render_hot_zone_item(self, role_uuid: str, item: Dict[str, Any]) -> str:
        object_id = str(item.get("id") or "")
        if not object_id:
            return str(item.get("text") or "")
        try:
            self.topology.reload()
            record = self.topology.find_record(object_id)
            if record is None:
                return ""
            return str(self.gateway.node_recall.compact_render(record, role_uuid) or "")
        except Exception:  # noqa: BLE001 - rendering must not break a session
            return ""

    @staticmethod
    def _iso_epoch(value: Any) -> Optional[float]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()

    def _role_last_output_epoch(self, role_uuid: str) -> Optional[float]:
        moments: list[float] = []
        for session in self.sessions.values():
            if str(getattr(session, "role_uuid", "")) != role_uuid:
                continue
            moment = self._iso_epoch(getattr(session, "last_output_at_wall", None))
            if moment is not None:
                moments.append(moment)
        for state in self.authority.list_sessions():
            if str(state.get("role_uuid") or "") != role_uuid:
                continue
            moment = self._iso_epoch(state.get("last_output_at_wall"))
            if moment is not None:
                moments.append(moment)
        if moments:
            return max(moments)
        current = self.hot_zones.load(role_uuid).get("last_output_at")
        try:
            return float(current) if current is not None else None
        except (TypeError, ValueError):
            return None

    def _role_has_active_session(self, role_uuid: str) -> bool:
        if any(
            str(getattr(session, "role_uuid", "")) == role_uuid
            and getattr(session, "state", None) in self.NON_TERMINAL_SESSION_STATES
            for session in self.sessions.values()
        ):
            return True
        return any(
            str(state.get("role_uuid") or "") == role_uuid
            and state.get("state") in self.NON_TERMINAL_SESSION_STATES
            for state in self.authority.list_sessions()
        )

    def build_hot_zone(
        self,
        role_uuid: str,
        *,
        now: Optional[float] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not self.hot_zone_enabled:
            current = self.hot_zones.load(role_uuid)
            return {
                **current,
                "status": "disabled",
                "reason": "hot_zone_memory_disabled",
                "snapshot": current,
            }
        return self.hot_zone_builder.build(
            role_uuid,
            now=now,
            last_output_at=self._role_last_output_epoch(role_uuid),
            session_active=self._role_has_active_session(role_uuid),
            force=force,
        )

    def recalculate_hot_zone(
        self,
        role_uuid: str,
        objects: list[Dict[str, Any]],
        *,
        last_output_at: float,
        now: float,
    ) -> Dict[str, Any]:
        if not self.hot_zone_enabled:
            current = self.hot_zones.load(role_uuid)
            return {
                **current,
                "status": "disabled",
                "reason": "hot_zone_memory_disabled",
                "snapshot": current,
            }
        session_active = any(
            session.role_uuid == role_uuid and session.state in self.NON_TERMINAL_SESSION_STATES
            for session in self.sessions.values()
        )
        session_active = session_active or any(
            state.get("role_uuid") == role_uuid
            and state.get("state") in self.NON_TERMINAL_SESSION_STATES
            for state in self.authority.list_sessions()
        )
        return self.hot_zones.recalculate(
            role_uuid,
            objects,
            last_output_at=last_output_at,
            now=now,
            session_active=session_active,
        )

    def set_hot_zone_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Switch hot-zone memory rotation on/off (persisted across restarts).

        Called from the plugin bar (memos 插件配置 → 热区记忆) via
        ``POST /graphiti/ui/hot-zone``. When disabled, no hot-zone rotation runs
        and sessions freeze an empty hot zone.
        """
        self.hot_zone_enabled = bool(enabled)
        self.authority.save_root_file(
            "hot_zone_preference.json", {"enabled": self.hot_zone_enabled}
        )
        self._log_event("hot_zone_preference", details={"enabled": self.hot_zone_enabled})
        return self.hot_zone_status()

    def rebuild_hot_zone(self, role_uuid: str) -> Dict[str, Any]:
        return self.build_hot_zone(role_uuid, force=True)

    def hot_zone_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.hot_zone_enabled,
            "rotation": self.hot_zone_enabled,
            "reason": "enabled" if self.hot_zone_enabled else "hot_zone_memory_disabled",
        }

    def session_prompt(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        """Return the session-frozen prompt additions (overlay + hot zone).

        This never creates a session: a prompt addition that is not frozen yet
        would break the "stable prefix for the whole conversation" contract.
        """

        key = (str(role_uuid), str(conversation_id))
        session = self.sessions.get(key)
        if session is None:
            raise SessionLockedError("session has not been started")
        return {
            "role_uuid": session.role_uuid,
            "conversation_id": session.conversation_id,
            **session.prompt_additions,
            "frozen_at": session.started_at,
        }

    def reflection_status(self, role_uuid: str) -> Dict[str, Any]:
        pointer = self.overlays.pointer(role_uuid)
        state = self.reflections._load_state(role_uuid)
        versions = self.overlays.versions(role_uuid)
        return {
            "role_uuid": role_uuid,
            "pointer": pointer,
            "state": state,
            "active_text": self.overlays.active_text(role_uuid),
            "versions": [
                {
                    "overlay_version_id": item.get("overlay_version_id"),
                    "created_by": item.get("created_by"),
                    "created_at": item.get("created_at"),
                    "items": len(item.get("items") or []),
                    "token_count": item.get("token_count"),
                    "base_prompt_version_id": item.get("base_prompt_version_id"),
                }
                for item in versions[-20:]
            ],
        }

    def run_reflection(self, role_uuid: str, *, force: bool = False) -> Dict[str, Any]:
        return self.reflections.run(role_uuid, force=force)

    def set_overlay_paused(self, role_uuid: str, paused: bool) -> Dict[str, Any]:
        return self.overlays.set_paused(role_uuid, paused)

    def activate_overlay(self, role_uuid: str, overlay_id: str) -> Dict[str, Any]:
        return self.overlays.activate(role_uuid, overlay_id)

    def edit_overlay(self, role_uuid: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self._management_prompt(role_uuid)
        return self.overlays.edit(
            role_uuid,
            items,
            base_prompt_version_id=str(
                prompt.get("current_version_id") or prompt.get("prompt_version_id") or ""
            ),
        )

    def lock_overlay_item(self, role_uuid: str, item_id: str, locked: bool) -> Dict[str, Any]:
        return self.overlays.set_locked(role_uuid, item_id, locked)

    def activation_status(self, role_uuid: str) -> Dict[str, Any]:
        return self.activations.status(role_uuid)

    def reset_activation(self, role_uuid: str, *, actor: str = "local_management") -> Dict[str, Any]:
        result = self.activations.reset(role_uuid, actor=actor)
        self.management_log.log(
            {
                "role_uuid": role_uuid,
                "actor": actor,
                "operation": "activation_reset",
                "at": utc_timestamp(),
            }
        )
        return result

    @staticmethod
    def _render_transcript(
        events: List[Dict[str, Any]], user_label: str, assistant_label: str
    ) -> str:
        """冻结批次 -> 抽取器读的纯文本。

        每条用户消息的说话人标签按**这一条自己的来源**决定：本机输入用主人称呼，
        QQ 轮次用联系人身份（``昵称(QQ 号码)``）。整批只挂一个"最近一次外部来源"
        时，一段来回换人的会话必然把 A 说的话算到 B 头上。
        """
        lines: List[str] = []
        for item in events:
            role = str(item.get("role") or "")
            if role == "user":
                speaker = event_source_label(item) or user_label
            elif role == "assistant":
                speaker = assistant_label
            else:
                speaker = role or "tool"
            lines.append(f"{speaker}: {item.get('content', '')}")
        return "\n".join(lines)

    def _focus_object_ids(self, focus_contact: Any) -> List[str]:
        """当前聊天对象对应的图谱节点 id（召回排序偏好用）。

        QQ 联系人/群的节点 id 由 ``identity.qq_node_id`` 按号码确定性派生，和
        硬加入时用的是同一个函数，所以这里不需要宠物端知道 UUID。节点可能已经被
        合并过，因此再解析一次规范主 id。解析失败（号码缺失/非法）时返回空列表：
        加分只是排序偏好，拿不到就当作没有，绝不影响召回本身。
        """
        if not isinstance(focus_contact, dict):
            return []
        number = str(focus_contact.get("number") or "").strip()
        if not number:
            return []
        kind = "group" if str(focus_contact.get("kind") or "") == "group" else "user"
        try:
            candidates = [qq_node_id(kind, number)]
            group_number = str(focus_contact.get("group_number") or "").strip()
            if group_number:
                # 群消息：发送者与所属群都算"与本次聊天有关"。
                candidates.append(qq_node_id("group", group_number))
        except ValueError:
            return []
        self.topology.reload()
        resolved: List[str] = []
        for candidate in candidates:
            canonical = str(self.topology.resolve_node(candidate) or candidate)
            if canonical not in resolved:
                resolved.append(canonical)
        return resolved

    def graph_query(
        self,
        role_uuid: str,
        query: str,
        *,
        operator: Optional[str] = None,
        anchors: Optional[List[str]] = None,
        predicate: str = "",
        target: str = "",
        top_k: int = 12,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Run one deterministic graph query for the tool (no embedding/rerank)."""
        return self.gateway.graph_query(
            role_uuid,
            query,
            operator=operator,
            anchors=anchors,
            predicate=predicate,
            target=target,
            top_k=top_k,
            max_depth=max_depth,
        )

    def search(self, role_uuid: str, user_id: str, conversation_id: Optional[str], query: str, **options: Any) -> Dict[str, Any]:
        try:
            started = time.perf_counter()
            if self.maintenance.status().get("status") != "clear":
                raise SessionLockedError("memory maintenance is active")
            # The Warm Layer is role-scoped, so recall no longer needs a live
            # session to find recent transcript. The session (when there is one) is
            # consulted only to learn which ``(batch_id, round_no)`` is *this*
            # turn, which must never be injected back into its own context.
            session = None
            effective_conversation_id = conversation_id
            if conversation_id:
                try:
                    session = self.start_session(role_uuid, conversation_id)
                    # 继承时它返回上一段对话：召回的工作记忆与响应里的会话 id 都必须
                    # 按这个 id 取，响应也把它交回给客户端采纳。
                    effective_conversation_id = session.conversation_id
                except SessionLockedError:
                    session = None
                    self._log_event(
                        "memory_search",
                        status="degraded",
                        message="conversation session locked; searching without live session",
                        details={"role_uuid": role_uuid},
                    )
            # 与本次聊天对象有关的记忆加分：把 focus_contact（QQ 渠道+号码）解析成
            # 该联系人在这段角色分区里的节点 id，交给召回层做排序偏好。
            focus_object_ids = self._focus_object_ids(options.pop("focus_contact", None))
            # 当前轮由 (batch_id, round_no) 精确定位；调用方可以显式覆盖（工具查询
            # 就属于"我已经看过的那些片段"之外的查询）。
            current_batch_id = options.pop("current_batch_id", None)
            if current_batch_id is None:
                current_batch_id = getattr(session, "batch_id", None)
            current_round_no = options.pop("current_round_no", None)
            if current_round_no is None:
                session_round = int(getattr(session, "current_round_no", 0) or 0)
                current_round_no = session_round + 1 if session_round else None
            include_assistant = bool(options.pop("include_assistant", False))
            # 日历视图的参照时间（历史重放用）：只是查询侧解析「上周三」的锚点，
            # 不是召回参数，必须在交给 gateway 之前取走（ADR-0178）。
            calendar_reference = options.pop("calendar_reference", None)
            count_activation = bool(options.pop("count_activation", True))
            result = self.gateway.search(
                role_uuid,
                user_id,
                effective_conversation_id,
                query,
                warm_store=self.warm_store(role_uuid),
                current_batch_id=current_batch_id,
                current_round_no=current_round_no,
                include_assistant=include_assistant,
                focus_object_ids=focus_object_ids,
                **options,
            )
            result["conversation_id"] = effective_conversation_id
            # Graphiti Core's own hybrid search is **off by default**. It is a full
            # BM25 + cosine + BFS + community search followed by a cross-encoder over
            # every candidate, and this gateway then refuses all of its results because
            # they carry no calibrated score (merge_core_memories(admit_unscored=False)).
            # Measured on the live audit: 52 candidates, 0 admitted, while recall took
            # 4.1-4.6 s. Paying seconds for a source that can never be injected is the
            # largest avoidable latency in the turn. Operators can turn it back on for
            # diagnostics or to evaluate a future RRF-only recipe.
            core_search_enabled = bool(
                (self.runtime_config.get("retrieval", {}) or {}).get("core_search_enabled", False)
            )
            if self.graphiti_enabled and core_search_enabled and hasattr(self.graphiti, "search_"):
                # Graphiti Core's own search is a supplementary source. It must be
                # folded into the gated result set, never replace it: replacing the
                # whole result here is how unrelated nodes/edges (all hardcoded to
                # score 1.0) bypassed every threshold and reached the prompt.
                try:
                    search_kwargs = {"group_ids": [role_uuid]}
                    if options.get("search_config") is not None:
                        search_kwargs["config"] = options["search_config"]
                    graphiti_result = self._run_graphiti_async(self.graphiti.search_(query, **search_kwargs))
                    core_memories = self._graphiti_memories(graphiti_result, role_uuid)
                    merged = self.gateway.merge_core_memories(
                        result.get("memories", []),
                        core_memories,
                        top_k=int(options.get("top_k", 5)),
                        token_budget=options.get("token_budget"),
                    )
                    result["memories"] = merged["memories"]
                    result["count"] = len(merged["memories"])
                    result["token_estimate"] = merged["token_estimate"]
                    result.setdefault("retrieval_diagnostics", {})["graphiti_core"] = {
                        "available": True,
                        "candidates": len(core_memories),
                        "admitted": merged["admitted"],
                        "dropped_unscored": merged["dropped_unscored"],
                    }
                except Exception as exc:
                    result.setdefault("retrieval_diagnostics", {})["graphiti_core"] = "unavailable"
                    result.setdefault("retrieval_diagnostics", {})["graphiti_error"] = str(exc)
            elif self.graphiti_enabled:
                result.setdefault("retrieval_diagnostics", {})["graphiti_core"] = {
                    "available": True,
                    "status": "disabled_by_policy",
                }
            # ADR-0164：临时图谱是**本场未结算的上文**，其中的回合点名了一些
            # 图谱这一轮没看到（甚至还没进图谱）的实体。命中临时图谱回合时，用这些
            # 回合的正文**一次性**补做一次实体召回（多条命中合成一次调用，不是每条
            # 一次），把被点名的实体本身捞回来；只补一次，不做递归。
            policy = self.gateway.node_recall.policy
            if bool(getattr(policy, "temp_hit_entity_recall", False)):
                self._recall_entities_for_warm_hits(role_uuid, result, options)
            # 日历视图（ADR-0178）：查询里有明确日期，或查询本身是时间问句时，
            # 用命中的时间表达句子补召回。它们**占用** top_k 名额（挤掉分最低的
            # 图谱条目），不是额外加塞——同一预算下的对比才成立。
            if bool(getattr(policy, "calendar_enabled", False)):
                self._inject_calendar(role_uuid, result, query, options, reference=calendar_reference)
            # 激活量写入：只记图谱层的直接命中；工具/测试查询通过
            # `count_activation=False` 跳过，避免同一轮重复查询刷高权重。
            batch_id = str(current_batch_id or f"unbatched:{effective_conversation_id or 'unknown'}")
            try:
                activation_stats = self.activations.record_recall(
                    role_uuid,
                    result.get("memories") or [],
                    batch_id=batch_id,
                    traversed_relations=(
                        result.get("retrieval_diagnostics", {}) or {}
                    ).get("traversed_relations")
                    or [],
                    count=count_activation,
                )
                result.setdefault("retrieval_diagnostics", {})["activation"] = activation_stats
            except Exception as exc:  # noqa: BLE001 - a stats write must not fail a turn
                result.setdefault("retrieval_diagnostics", {})["activation"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                self._log_event(
                    "memory_activation",
                    status="failed",
                    message=str(exc),
                    details={"role_uuid": role_uuid, "batch_id": batch_id},
                )
            result.setdefault("retrieval_diagnostics", {}).setdefault("timings", {})["total_ms"] = round(
                (time.perf_counter() - started) * 1000.0, 1
            )
            self.audit.log({"role_uuid": role_uuid, "user_id": user_id, "conversation_id": effective_conversation_id, "query": query, "policy_version": result.get("policy_version"), "result_ids": [item.get("id") for item in result.get("memories", [])], "result_count": result.get("count", 0), "token_estimate": result.get("token_estimate", 0), "retrieval": result.get("retrieval_diagnostics", {})})
            self._log_event("memory_search", details={"role_uuid": role_uuid, "result_count": result.get("count", 0), "query_length": len(query)}, duration_ms=(time.perf_counter() - started) * 1000)
            # 聊天记录窗口：把本次召回结果挂到待定回合上（仅查看用途，
            # 不进入任何抽取来源）。返回的凭据由客户端在回合记录时原样带回——
            # 面板的 memories 只能靠它认领：搜索跑在 LLM 之前、回合落在回复之后，
            # 中间本机输入会被加上来源前缀、后端也可能换掉会话 id，所以"用文本
            # 或会话 id 相等去认领"必然失配（ADR-0166）。没有会话 id 时同样发凭据：
            # 它不再依赖会话，否则全新安装（还没有会话 id）第一轮就永远显示 0 条。
            token = self.chatlog.record_search(
                role_uuid, effective_conversation_id, query, result.get("memories", [])
            )
            if token:
                result["recall_token"] = token
            return result
        except Exception as exc:
            self._log_event("memory_search", status="failed", message=str(exc), details={"role_uuid": role_uuid}, duration_ms=(time.perf_counter() - started) * 1000)
            raise

    def _recall_entities_for_warm_hits(
        self, role_uuid: str, result: Dict[str, Any], options: Dict[str, Any]
    ) -> None:
        """One batched entity recall for the warm turns we just recalled.

        一次追问之所以"问到了但图谱里没有"，往往是因为被点名的那个人/物只在**近期
        正文**（温层）里出现过、还没进长期图谱。这里把这些回合的正文合成**一次**
        查询做实体召回，而不是每条命中各做一次，也不是递归展开。
        """
        diagnostics_key = "warm_hit_entity_recall"
        memories = list(result.get("memories") or [])
        temp_items = [item for item in memories if str(item.get("layer") or "") == "WarmLayer"]
        diagnostics: Dict[str, Any] = {"warm_hits": len(temp_items), "status": "skipped"}
        if not temp_items:
            result.setdefault("retrieval_diagnostics", {})[diagnostics_key] = diagnostics
            return
        # 多条命中一次性召回：正文按原顺序拼接并设上限，避免一次追问把整段上文
        # 变成一个超长查询。
        query = "\n".join(str(item.get("content") or "") for item in temp_items).strip()[:1200]
        if not query:
            diagnostics["status"] = "empty_query"
            result.setdefault("retrieval_diagnostics", {})[diagnostics_key] = diagnostics
            return

        policy = self.gateway.node_recall.policy
        try:
            follow_up = self.gateway.search_entities(
                role_uuid,
                query,
                top_k=int(getattr(policy, "temp_hit_entity_top_k", 5)),
                token_budget=options.get("token_budget"),
                similarity_threshold=float(options.get("similarity_threshold", 0.0) or 0.0),
            )
        except Exception as exc:  # noqa: BLE001 - a supplementary pass never fails the turn
            diagnostics.update({"status": "failed", "error": str(exc)})
            result.setdefault("retrieval_diagnostics", {})[diagnostics_key] = diagnostics
            return

        known = {str(item.get("id")) for item in memories}
        added = [item for item in follow_up.get("memories", []) if str(item.get("id")) not in known]
        diagnostics.update(
            {
                "status": "ok",
                "batched_query_chars": len(query),
                "entities": len(added),
                "entity_ids": [item.get("id") for item in added],
            }
        )
        if added:
            merged = memories + [
                {**item, "layer": "GraphitiMemory", "warm_hit_entity_recall": True} for item in added
            ]
            merged = self.gateway._drop_duplicate_content(merged)
            merged.sort(
                key=lambda item: (
                    -float(item.get("final_score") or 0.0),
                    str(item.get("created_at") or ""),
                    str(item.get("id") or ""),
                )
            )
            budget = (
                policy.token_budget
                if options.get("token_budget") is None
                else max(0, int(options.get("token_budget")))
            )
            selected = apply_budget(merged, int(options.get("top_k", 5) or 5), budget)
            result["memories"] = selected
            result["count"] = len(selected)
            result["token_estimate"] = sum(
                int(item.get("token_estimate", len(str(item.get("content", "")))))
                for item in selected
            )
        result.setdefault("retrieval_diagnostics", {})[diagnostics_key] = diagnostics

    def _inject_calendar(
        self,
        role_uuid: str,
        result: Dict[str, Any],
        query: str,
        options: Dict[str, Any],
        *,
        reference: Any = None,
    ) -> None:
        """Fold this query's calendar rows into the result, inside the budget.

        两种读法（ADR-0178）：查询自带日期 → 按天取回；查询是时间问句 →
        按词汇覆盖率取回，句子前面挂着它解析出的日期。两者都只是**视图**，
        没有一条进入事实图；命中的条目占掉 `top_k` 名额，不会让模型多看东西。
        """
        policy = self.gateway.node_recall.policy
        diagnostics: Dict[str, Any] = {"status": "skipped"}
        result.setdefault("retrieval_diagnostics", {})["calendar"] = diagnostics
        try:
            found = self.calendar.search(
                role_uuid,
                query,
                reference=parse_moment(reference),
                limit=int(getattr(policy, "calendar_top_k", 3)),
                lexical_floor=float(getattr(policy, "calendar_lexical_floor", 0.30)),
            )
        except Exception as exc:  # noqa: BLE001 - a view never fails a turn
            diagnostics.update({"status": "failed", "error": str(exc)})
            return
        diagnostics.update(
            {key: found.get(key) for key in ("status", "mode", "days", "candidates")}
        )
        items = [render_calendar_item(row) for row in found.get("items") or []]
        memories = list(result.get("memories") or [])
        known = {str(item.get("id")) for item in memories}
        items = [item for item in items if str(item.get("id")) not in known]
        if not items:
            diagnostics["status"] = "duplicate" if found.get("items") else "empty"
            return
        top_k = int(options.get("top_k", 5) or 5)
        budget = (
            policy.token_budget
            if options.get("token_budget") is None
            else max(0, int(options.get("token_budget")))
        )
        # 名额先给日历：天数约束是硬条件，而图谱条目之间仍按原分数排序。
        keep = apply_budget(memories, max(0, top_k - len(items)), budget)
        merged = items + keep

        def tokens(rows: Sequence[Dict[str, Any]]) -> int:
            return sum(
                int(row.get("token_estimate", len(str(row.get("content", "")))))
                for row in rows
            )

        # 日历条目已经占住名额，这里只收 token：先丢分最低的非日历条目，
        # 实在放不下才丢日历条目自己的尾巴。
        while merged and tokens(merged) > budget:
            base = [row for row in merged if row.get("layer") != "Calendar"]
            victim = (
                min(base, key=lambda row: (float(row.get("final_score") or 0.0), str(row.get("id"))))
                if base
                else merged[-1]
            )
            merged.remove(victim)
            diagnostics.setdefault("evicted_ids", []).append(victim.get("id"))
        result["memories"] = merged
        result["count"] = len(merged)
        result["token_estimate"] = tokens(merged)
        diagnostics.update({"status": "injected", "injected_ids": [row["id"] for row in items]})
    @staticmethod
    def _graphiti_memories(search_result: Any, role_uuid: str) -> list[Dict[str, Any]]:
        """Convert Graphiti Core nodes/edges to MemOS result units (diagnostics-only).

        Graphiti Core returns ranked objects without a calibrated relevance score.
        The score is therefore reported as unknown (``None``) instead of being
        fabricated as ``1.0``: that hardcoded score is what made every core result
        look perfectly relevant, outrank real memories and bypass the gates.

        **Episodes are not converted at all.** An episode is the frozen conversation
        body — evidence, not a memory unit — and publication deliberately deletes that
        body afterwards. Returning it made the raw transcript (junk narration included)
        an injectable "memory".
        """
        memories: list[Dict[str, Any]] = []
        for collection, source_type in ((getattr(search_result, "edges", []), "EntityEdge"), (getattr(search_result, "nodes", []), "EntityNode")):
            for item in collection or []:
                group_id = getattr(item, "group_id", role_uuid)
                if group_id != role_uuid:
                    continue
                content = getattr(item, "fact", None) or getattr(item, "summary", None) or getattr(item, "content", None) or getattr(item, "name", None)
                if not content:
                    continue
                record_id = str(getattr(item, "uuid", ""))
                memories.append({"id": record_id, "content": str(content), "source": source_type, "source_type": source_type, "group_id": role_uuid, "similarity": None, "rerank_score": None, "final_score": None, "token_estimate": len(str(content)), "match_type": "direct", "score_source": "graphiti_core_unscored"})
        return memories

    def register_external_source(
        self,
        role_uuid: str,
        *,
        kind: str,
        number: str,
        display_name: Optional[str] = None,
        group_number: Optional[str] = None,
        group_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hard-register an inbound external source (ADR-0152), then self-heal.

        Called for every accepted QQ message before the turn is answered. It only
        writes the node and the role grant: no relation, no description. A failure
        is raised so the caller can retry; it never blocks the answer itself.

        The registration is followed by a role-scoped reconcile, because this call
        is the one piece of the QQ path the desktop pet can lose entirely (gateway
        not up, retry budget exhausted, restart mid-flight) and a lost call used to
        leave a contact node with no edge to its AI role node forever. Reconcile is
        cheap — it walks this role's QQ nodes, of which there are a handful — and it
        needs no new input: the next message from any contact repairs every earlier
        gap. A reconcile failure is reported inside the response, never raised: the
        registration itself already succeeded and must not be reported as failed.
        """
        try:
            self.topology.reload()
            if role_uuid not in self.topology.data.get("nodes", {}):
                raise ValueError("role is not available")
            result = self.external_sources.register(
                role_uuid,
                kind=kind,
                number=number,
                display_name=display_name,
                group_number=group_number,
                group_name=group_name,
            )
            try:
                result["reconciled"] = self._reconcile_contacts(role_uuid)
            except Exception as exc:  # noqa: BLE001 - repair is best-effort
                result["reconciled"] = {"status": "failed", "error": str(exc)}
            self._log_event(
                "external_source_registered",
                details={
                    "role_uuid": role_uuid,
                    "kind": kind,
                    "number": number,
                    "group_number": group_number,
                    "nodes": result["nodes"],
                    "reconciled": result["reconciled"].get("repaired"),
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001 - the caller retries; the log records it
            self._log_event(
                "external_source_registered",
                status="failed",
                message=str(exc),
                details={"role_uuid": role_uuid, "kind": kind, "number": number},
            )
            raise

    def reconcile_external_sources(
        self,
        role_uuid: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Verify and repair the QQ contact nodes of one role (self-healing).

        ``sources`` optionally names contacts the caller knows about
        (``{"kind", "number", "display_name", "group_number", "group_name"}``); each
        one is (re-)registered, and the contact's knowledge is re-unified: a duplicate
        entity an older batch created under the extractor's uuid is absorbed into the
        number-keyed node once its provenance confirms it (see
        :meth:`_absorb_duplicate_contact_entities`). Omit it to only repair the
        contacts this role already holds.
        """
        try:
            self.topology.reload()
            if role_uuid not in self.topology.data.get("nodes", {}):
                raise ValueError("role is not available")
            result = self._reconcile_contacts(role_uuid, sources=sources)
            self._log_event(
                "external_sources_reconciled",
                details={
                    "role_uuid": role_uuid,
                    "checked": result.get("checked"),
                    "changed": result.get("changed"),
                    "repaired": result.get("repaired"),
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001
            self._log_event(
                "external_sources_reconciled",
                status="failed",
                message=str(exc),
                details={"role_uuid": role_uuid},
            )
            raise

    def _reconcile_contacts(
        self,
        role_uuid: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Registrar reconcile plus the duplicate-entity repair of one role."""
        result = self.external_sources.reconcile(role_uuid, sources=sources)
        absorbed = self.absorb_duplicate_contact_entities(role_uuid)
        if absorbed:
            result.setdefault("repaired", {}).setdefault("entities_absorbed", []).extend(absorbed)
            result["changed"] = True
            self._log_event(
                "contact_entities_absorbed",
                details={"role_uuid": role_uuid, "absorbed": absorbed},
            )
        return result

    def absorb_duplicate_contact_entities(
        self,
        role_uuid: str,
        *,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Move knowledge off duplicates onto the number-keyed contact node.

        The duplicate is the shape a defect left behind: extraction published an entity
        named like the contact under the extractor's own uuid, because nothing mapped
        that name onto the deterministic QQ identity — so the contact node sat empty
        while its knowledge hung on a second node with the same name. Publication now
        resolves the name onto the number-keyed node (``saga``), and this is the repair
        for what earlier batches already wrote.

        The merge is **evidence-gated, never name-only**: every recorded provenance of
        the candidate's knowledge — its fact nodes and its own memory elements, for
        every role that holds them — must be a conversation whose external source is
        this contact. A same-named entity the owner talked about locally carries no
        such provenance and is left exactly where it is (CONTEXT 同名不代表同一实体).

        ``dry_run`` reports the same decisions without writing anything, so an operator
        can see what a repair would do first.
        """
        self.topology.reload()
        candidates: List[Tuple[str, str]] = []
        for contact_node_id in list(self.topology.data.get("nodes", {})):
            contact_id = str(self.topology.resolve_node(contact_node_id) or contact_node_id)
            node = self.topology.data.get("nodes", {}).get(contact_id)
            if node is None or str(node.get("external_source") or "") != QQ_SOURCE_MARKER:
                continue
            if not self.topology.is_allowed(role_uuid, contact_id):
                continue
            for member_id in same_named_plain_entities(self.topology, role_uuid, contact_id):
                if (contact_id, member_id) not in candidates:
                    candidates.append((contact_id, member_id))
        if not candidates:
            return []

        # One pass over this role's sessions: conversation_id -> the contact number
        # that conversation came from. A conversation the role has no session for is
        # unknown provenance, and unknown provenance never authorises a merge.
        conversation_numbers: Dict[str, str] = {}
        for session in self.authority.list_sessions():
            if str(session.get("role_uuid") or "") != str(role_uuid):
                continue
            number = str((session.get("external_source") or {}).get("number") or "").strip()
            conversation_id = str(session.get("conversation_id") or "")
            if number and conversation_id:
                conversation_numbers[conversation_id] = number

        absorbed: List[Dict[str, Any]] = []
        for contact_id, member_id in candidates:
            node = self.topology.data.get("nodes", {}).get(contact_id)
            if node is None:
                continue
            number = str(node.get("external_number") or "").strip()
            provenance = self._knowledge_provenance(member_id)
            if not self._provenance_comes_from_contact(provenance, number, conversation_numbers):
                continue
            record = {
                "contact_id": contact_id,
                "member_id": member_id,
                "number": number,
                "provenance": sorted(set(provenance)),
            }
            if dry_run:
                absorbed.append({**record, "applied": False})
                continue
            try:
                merged = self.merge_nodes(contact_id, member_id)
            except (KeyError, ValueError):
                # Already merged by an earlier candidate, or not a mergeable entity:
                # the repair reports what it did, it never fails the caller.
                continue
            absorbed.append(
                {**record, "applied": True, "operation_id": merged.get("operation_id")}
            )
        return absorbed

    def _knowledge_provenance(self, node_id: str) -> List[str]:
        """Every ``source_conversation_id`` recorded on one node's knowledge.

        Two places carry it: the node's own memory elements (role-keyed) and its fact
        nodes. A missing reference comes back as an empty string, which the gate treats
        as "not from this contact".
        """
        from . import node_memory

        self.topology.reload()
        canonical = str(self.topology.resolve_node(node_id) or node_id)
        record = self.topology.find_record(canonical)
        if record is None:
            return []
        refs: List[str] = []
        for role_uuid in node_memory.roles_with_memory(record):
            for element in node_memory.elements_for_role(record, role_uuid):
                refs.append(str(element.get("source_conversation_id") or ""))
        for candidate in self.topology.data.get("nodes", {}).values():
            if str(candidate.get("kind") or "") != "fact":
                continue
            parent = str(candidate.get("parent_id") or "")
            if not parent:
                continue
            if str(self.topology.resolve_node(parent) or parent) != canonical:
                continue
            refs.append(str(candidate.get("source_conversation_id") or ""))
        return refs

    @staticmethod
    def _provenance_comes_from_contact(
        provenance: List[str],
        number: str,
        conversation_numbers: Dict[str, str],
    ) -> bool:
        """True when every provenance reference is a known batch from this contact."""
        if not provenance or not number:
            return False
        return all(
            bool(reference) and conversation_numbers.get(reference) == number
            for reference in provenance
        )

    def search_working_memory(
        self,
        role_uuid: str,
        user_id: str,
        conversation_id: Optional[str],
        query: str,
        **options: Any,
    ) -> Dict[str, Any]:        return self.gateway.search_working_memory(
            role_uuid, user_id, conversation_id, query, **options
        )

    def chatlog_list(
        self,
        role_uuid: str,
        conversation_id: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Return the view-only chat recall log (rounds, oldest first)."""
        rounds = self.chatlog.list(role_uuid, conversation_id, limit)
        return {
            "role_uuid": role_uuid,
            "conversation_id": conversation_id,
            "rounds": rounds,
            "count": len(rounds),
            "view_only": True,
        }

    def create_transient_memory_snapshot(
        self,
        request_id: str,
        role_uuid: str,
        user_id: str,
        conversation_id: str,
        query: str,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        return self.request_snapshots.create(
            request_id,
            role_uuid,
            user_id,
            conversation_id,
            query,
            top_k=top_k,
        )

    def assemble_request_context(self, request_id: str, **context: Any) -> list[Dict[str, Any]]:
        return self.request_snapshots.assemble(request_id, **context)

    def discard_transient_memory_snapshot(self, request_id: str) -> None:
        self.request_snapshots.discard(request_id)

    def close_session(self, role_uuid: str, conversation_id: str, reason: str = "explicit_end") -> Dict[str, Any]:
        """End the session lifecycle at a batch boundary.

        This deliberately does **not** extract: sealing a batch only defines where
        the conversation ends, and the transcript keeps living in the Warm Layer
        until its TTL expires (or an operator triggers the batch explicitly).
        Extraction still runs through the same publication path, just later.
        """
        try:
            # Resolve the role's real active conversation (like session_status)
            # so "结束并封存" seals the conversation the desktop pet is actually
            # recording events into, instead of an unrelated/stale id whose
            # session may have no events.
            target = self._latest_active_conversation(role_uuid) or conversation_id
            result = self.start_session(role_uuid, target).close(reason)
            self._log_event("session_close", details={"role_uuid": role_uuid, "conversation_id": target, "reason": reason})
            return result
        except Exception as exc:
            self._log_event("session_close", status="failed", message=str(exc), details={"role_uuid": role_uuid, "conversation_id": conversation_id})
            raise

    def end_conversation(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        return self.close_session(role_uuid, conversation_id, "explicit_end")

    def close_for_new_conversation(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        return self.close_session(role_uuid, conversation_id, "new_conversation")

    def retry_session(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        """Retry the pending extraction of one role conversation.

        With Graphiti enabled the real work runs in the background queue, so the
        lock a client sees may live there while the session coordinator is
        already ``completed`` (``_overlay_extraction_state`` projects the queue's
        ``retry_locked`` onto it). Re-drive that job first; only when no job
        holds the conversation does the session-level publication need retrying.
        """
        job = self._extraction_job_for(role_uuid, conversation_id)
        if job is not None and self.extraction is not None:
            if job.get("state") == "retry_locked":
                retried = self.extraction.retry(role_uuid, conversation_id)
                if retried is not None:
                    self._log_event("extraction_retry", details={"role_uuid": role_uuid, "conversation_id": conversation_id, "source": "queue"})
                    return {
                        "status": "queued",
                        "role_uuid": str(role_uuid),
                        "conversation_id": str(conversation_id),
                        "queue": retried,
                    }
            # Already queued/extracting: there is nothing to re-drive.
            return {
                "status": str(job.get("state") or "queued"),
                "role_uuid": str(role_uuid),
                "conversation_id": str(conversation_id),
                "queue": job,
            }
        return self.start_session(role_uuid, conversation_id).retry()

    def extraction_queue_status(self) -> Dict[str, Any]:
        """Snapshot of the background extraction queue (running/pending/retry)."""
        if self.extraction is None:
            return {"running": [], "pending": [], "retry": []}
        return self.extraction.queue_status()

    def pending_publications_status(self) -> Dict[str, Any]:
        """Batches sealed but not yet published (the durable replay input)."""
        try:
            records = self.pending.list()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return {"pending_publications": [], "count": 0, "error": str(exc)}
        return {
            "role_uuid": None,
            "count": len(records),
            "pending_publications": [
                {
                    "role_uuid": item.get("role_uuid"),
                    "conversation_id": item.get("conversation_id"),
                    "batch_id": item.get("batch_id"),
                    "operation_id": item.get("operation_id"),
                    "snapshot_id": item.get("snapshot_id"),
                    "pending_since": item.get("pending_since"),
                    "reason": item.get("reason"),
                    "event_count": len(item.get("events") or []),
                }
                for item in records
            ],
        }

    def role_extraction_status(self, role_uuid: str) -> Dict[str, Any]:
        """Whether the given role's memory is currently being extracted or queued."""
        if self.extraction is None:
            return {"role_uuid": role_uuid, "extracting": False, "has_pending": False, "state": "clear", "message": ""}
        return self.extraction.role_status(role_uuid)

    @staticmethod
    def _session_updated(state: Dict[str, Any]) -> str:
        return str(state.get("updated_at") or state.get("started_at") or "")

    def _latest_active_conversation(self, role_uuid: str) -> Optional[str]:
        """Return the conversation id this role is currently running, or None.

        The WebUI watch and the desktop pet both drive the same role; a pet owns
        the live conversation under its own id, while the WebUI may still be
        polling a different (stale or newly guessed) id. To keep a read-only
        status in sync with reality we resolve the newest non-terminal session
        for the role (live in-memory first, then persisted) instead of trusting
        the id supplied by the caller.
        """
        best: Optional[tuple[str, str]] = None
        for session in self.sessions.values():
            if session.role_uuid != role_uuid or session.state not in self.NON_TERMINAL_SESSION_STATES:
                continue
            updated = self._session_updated(session.export_state())
            if best is None or updated > best[0]:
                best = (updated, session.conversation_id)
        for state in self.authority.list_sessions():
            if (
                state.get("role_uuid") != role_uuid
                or state.get("state") not in self.NON_TERMINAL_SESSION_STATES
            ):
                continue
            updated = self._session_updated(state)
            if best is None or updated > best[0]:
                best = (updated, str(state.get("conversation_id")))
        return best[1] if best else None

    def session_status(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        """Read-only session status for the WebUI watch.

        A polling status must reflect reality rather than reject it: it must
        never raise ``SessionLockedError`` just because the requested
        ``conversation_id`` does not match the conversation the role is actually
        running (e.g. the desktop pet owns the live session under its own id).
        If the role has a non-terminal conversation we adopt it and report that,
        so the frontend can sync its own ``conversation_id`` from the response.
        """
        target = self._latest_active_conversation(role_uuid) or conversation_id
        try:
            return self._overlay_extraction_state(self.start_session(role_uuid, target).export_state())
        except SessionLockedError:
            # A second active conversation can still race under an inconsistent
            # state. Fall back to a read-only report of the persisted active
            # session instead of surfacing a 500 to a monitoring poll.
            active = self._latest_active_conversation(role_uuid)
            if not active:
                raise
            state = next(
                (item for item in self.authority.list_sessions()
                 if item.get("role_uuid") == role_uuid and item.get("conversation_id") == active),
                None,
            )
            if state is None:
                raise
            return {
                "role_uuid": role_uuid,
                "conversation_id": state.get("conversation_id"),
                "state": state.get("state", "unknown"),
                "started_at": state.get("started_at"),
                "elapsed_seconds": state.get("elapsed_seconds", 0),
                "events": state.get("events", []),
                "entries": state.get("entries", []),
                "checkpointed": state.get("checkpointed", False),
                "last_output_at_wall": state.get("last_output_at_wall") or state.get("started_at"),
                "updated_at": state.get("updated_at"),
                "progress": state.get("progress", {}),
                "progress_log": state.get("progress_log", []),
            }

    def _extraction_job_for(self, role_uuid: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Return the running/pending/retry queue job for a role conversation."""
        if self.extraction is None:
            return None
        status = self.extraction.queue_status()
        for bucket in ("running", "pending", "retry"):
            for job in status[bucket]:
                if (
                    str(job.get("role_uuid")) == str(role_uuid)
                    and str(job.get("conversation_id")) == str(conversation_id)
                ):
                    return job
        return None

    def _overlay_extraction_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect a background extraction result on an optimistically-completed session.

        The session coordinator advances to ``completed`` (all-green progress)
        as soon as the immutable snapshot is handed to the background extraction
        queue, but the real graph publication happens asynchronously in that
        queue. If that publication failed, the session would keep reporting a
        misleading all-green ``completed`` status while the queue held a
        ``retry_locked`` job. Overlay the queue result so the WebUI can judge
        the extraction situation instead of showing green for a failed run.

        While the job is queued/running, the session reports the job's staged
        progress (1/7 -> 7/7) instead of an instant "completed" — this is what
        drives the seven-segment progress display in the conversation view.
        """
        if (
            self.extraction is None
            or state.get("state") != "completed"
        ):
            return state
        job = self._extraction_job_for(
            state.get("role_uuid"), state.get("conversation_id")
        )
        if job is None:
            return state
        if job["state"] == "retry_locked":
            updated = dict(state)
            updated["state"] = "retry_locked"
            progress = dict(state.get("progress") or {})
            progress["done"] = False
            progress["completed"] = 0
            progress["message"] = f"抽取失败，等待重试：{job.get('error') or job.get('message')}"
            updated["progress"] = progress
            updated["progress_log"] = list(state.get("progress_log") or []) + [{
                "timestamp": utc_timestamp(),
                "state": "retry_locked",
                "message": "抽取失败，等待重试",
            }]
            return updated
        if job["state"] in ("queued", "extracting", "publishing"):
            job_progress = job.get("progress") or {}
            stages = job_progress.get("stages") or [
                {"key": key, "label": label} for key, label in EXTRACTION_STAGES
            ]
            stage_index = min(
                max(0, int(job_progress.get("stage_index") or 0)), len(stages) - 1
            ) if stages else 0
            current_stage = job_progress.get("current_stage") or (
                stages[stage_index]["key"] if stages else ""
            )
            if job_progress.get("done"):
                return state
            updated = dict(state)
            updated["state"] = job["state"]
            updated["progress"] = {
                "stages": stages,
                "current_stage": current_stage,
                "stage_index": stage_index,
                "stage_total": len(stages),
                "completed": max(0, int(job_progress.get("completed") or 0)),
                "total": max(1, int(job_progress.get("total") or len(stages) or 1)),
                "message": (
                    job_progress.get("message")
                    or job.get("message")
                    or ("等待抽取" if job["state"] == "queued" else "抽取中")
                ),
                "active": True,
                "done": False,
            }
            merged_log = list(state.get("progress_log") or []) + list(job.get("progress_log") or [])
            updated["progress_log"] = self._dedupe_progress_log(merged_log)[-100:]
            return updated
        return state

    @staticmethod
    def _dedupe_progress_log(entries: list) -> list:
        """Collapse adjacent identical progress-log entries (session + job merge)."""
        result: list = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if result and (
                result[-1].get("state") == entry.get("state")
                and result[-1].get("stage") == entry.get("stage")
                and result[-1].get("message") == entry.get("message")
            ):
                continue
            result.append(dict(entry))
        return result

    def close_active_sessions(self, reason: str, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Drive every open session through one batch boundary.

        ``retry_locked`` used to mean "sealed but unpublished, still holding the
        role". Extraction now belongs to the batch TTL path, so a session can no
        longer hold a role hostage on a failed publication; the field stays in the
        result shape for callers that read it, always empty.
        """
        closed: List[Dict[str, Any]] = []
        sealed: List[Dict[str, Any]] = []
        suspended: List[Dict[str, Any]] = []
        keys: set[tuple[str, str]] = {
            key for key, session in self.sessions.items()
            if role_uuid is None or session.role_uuid == role_uuid
        }
        for state in self.authority.list_sessions():
            persisted_role_uuid = state.get("role_uuid")
            conversation_id = state.get("conversation_id")
            if (
                isinstance(persisted_role_uuid, str)
                and isinstance(conversation_id, str)
                and (role_uuid is None or persisted_role_uuid == role_uuid)
                and state.get("state") in ACTIVE_SESSION_STATES
            ):
                keys.add((persisted_role_uuid, conversation_id))
        for session_role_uuid, conversation_id in sorted(keys):
            if not session_role_uuid or not conversation_id:
                continue
            session = self.start_session(session_role_uuid, conversation_id)
            if session.state != "active":
                continue
            try:
                result = session.close(reason)
            except Exception as exc:  # noqa: BLE001 - one session must not stop the sweep
                self._log_event(
                    "session_close",
                    status="failed",
                    message=str(exc),
                    details={"role_uuid": session_role_uuid, "conversation_id": conversation_id},
                )
                continue
            entry = {"role_uuid": session_role_uuid, "conversation_id": conversation_id}
            closed.append(entry)
            if str(result.get("status")) == "sealed":
                sealed.append(entry)
            elif str(result.get("status")) == "suspended":
                suspended.append(entry)
        return {
            "closed": closed,
            "sealed": sealed,
            "suspended": suspended,
            "retry_locked": [],
        }

    def on_client_close(self) -> Dict[str, Any]:
        if self.extraction is not None:
            self.extraction.stop()
        return self.close_active_sessions("client_close")

    def on_pet_stop(self) -> Dict[str, Any]:
        return self.close_active_sessions("pet_stop")

    def on_process_exit(self) -> Dict[str, Any]:
        result = self.close_active_sessions("process_exit")
        if self.extraction is not None:
            self.extraction.stop()
        # Stop the state-change tick before tearing down the stores it observes,
        # so a late tick cannot touch a closed database.
        try:
            self.state_changes.stop()
        except Exception:
            pass
        try:
            self._graphiti_runner.run(self.graphiti.driver.close())
        except Exception:
            pass
        self.neo4j_graph.close()
        self._graphiti_runner.shutdown()
        return result

    def startup_backup(self, role_uuid: str):
        return self.backups.create_async(role_uuid)

    def startup_backups(self) -> list[Any]:
        """Start one non-blocking authority backup per registered role."""
        return [self.backups.create_async(role_uuid) for role_uuid in self.authority.role_ids()]

    def create_backup(self, role_uuid: str, kind: str = "manual", reason: Optional[str] = None) -> Dict[str, Any]:
        path = self.backups.create(role_uuid, kind=kind, reason=reason)
        return {"status": "created", "path": str(path.relative_to(self.root)).replace("\\", "/"), "backups": self.backups.list(role_uuid, kind)}

    def restore_backup(self, role_uuid: str, path: str) -> Dict[str, Any]:
        barrier = self.maintenance.enter("memory_restore_maintenance")
        try:
            protection = self.backups.create(role_uuid, kind="restore_protection", reason="before_restore")
            result = self.backups.restore(self.backups.resolve_registered(path), role_uuid)
            self.topology.data = self.topology._load()
            projection = self.projections.rebuild(role_uuid, self.authority)
            self.projections.validate(role_uuid)
            result.update({"protection_backup": str(protection), "projection": projection, "barrier_generation": barrier["generation"]})
            self.maintenance.complete("memory_restore_maintenance")
            return result
        except Exception as exc:
            self.maintenance.complete("memory_restore_maintenance", error=str(exc))
            raise

    def list_backups(self, role_uuid: Optional[str] = None, kind: Optional[str] = None) -> list[Dict[str, Any]]:
        return self.backups.list(role_uuid, kind)

    def rebuild_projection(self, role_uuid: str) -> Dict[str, Any]:
        projection = self.projections.rebuild(role_uuid, self.authority)
        self.projections.validate(role_uuid)
        return {"status": "rebuilt", "projection": projection}

    # ------------------------------------------------- SQLite operations surface

    def memory_operations(self) -> MemoryOperations:
        """Lazily-built operations command surface for the running service."""
        operations = getattr(self, "_memory_operations", None)
        if operations is None:
            operations = MemoryOperations(
                self.root,
                memory_layer=self.memory_layer,
                topology=self.topology,
                neo4j_projection=self.neo4j_events,
                authority=self.authority,
                config=self.runtime_config,
                embedder=self.gateway.node_recall.embedder,
            )
            self._memory_operations = operations
        operations.embedder = self.gateway.node_recall.embedder
        return operations

    def memory_status(self) -> Dict[str, Any]:
        """Storage status: SQLite path/revision, vector index and orphan events."""
        return self.memory_layer.status()

    def memory_diagnose(self, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Counts, revisions, missing vectors and orphan events for every store."""
        return self.memory_operations().diagnose(role_uuid)

    def rebuild_vector_index(self, *, role_uuid: Optional[str] = None, drop: bool = True) -> Dict[str, Any]:
        """Re-embed every object's memory text into the Neo4j vector index."""
        return self.memory_operations().rebuild_vectors(drop=drop, role_uuid=role_uuid)

    def validate_graph_defaults(self) -> Dict[str, Any]:
        """Validate that the default user/AI nodes and companion edges exist."""
        return self.memory_operations().validate_neo4j_defaults()

    def export_role_memory(self, role_uuid: str, target: Optional[str] = None) -> Dict[str, Any]:
        """Write one role's SQLite authority out as a JSON backup/export."""
        return self.memory_operations().export_json(role_uuid, target=target)

    def restore_role_memory(self, role_uuid: str) -> Dict[str, Any]:
        """Load a JSON export back into the store.

        Only the event bodies and the explicit "remember this" submissions are
        restored: a role's memory about an object is not exportable state any more,
        because it is the graph itself. Re-importing it would mean writing node
        attributes from a file, which is what made the old JSON mirror drift from
        the online store in the first place.
        """
        restored = self.authority.restore_memory_from_json(role_uuid, self.sqlite)
        self.sqlite.bump_revision()
        return {"status": "restored", "role_uuid": role_uuid, "restored": restored}

    def memory_vector_freshness(self, role_uuid: str) -> Dict[str, Any]:
        """Missing/queued/failed vectors for one role."""
        return self.gateway.node_recall.vector_freshness(role_uuid)

    def claim_user_memory_batch(
        self,
        role_uuid: str,
        batch_id: str,
        source_ids: list[str],
        *,
        layer: str = "UserMemory",
    ) -> Dict[str, Any]:
        return self.cutover.claim_batch(role_uuid, batch_id, source_ids, layer=layer)

    def clean_cutover(
        self,
        role_uuid: str,
        operation_id: str,
        source_ids: list[str],
        *,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        if not self.projections.validate(role_uuid):
            raise RuntimeError("role projection is not valid")
        manifest = self.authority.load_role_file(role_uuid, "publication_manifest.json", {})
        if manifest.get("status") != "published":
            raise RuntimeError("legacy cleanup requires a published Graphiti manifest")
        barrier = self.maintenance.enter("memory_cutover_maintenance")
        try:
            result = self.cutover.clean_cutover(role_uuid, operation_id, source_ids, confirm=confirm)
            result["barrier_generation"] = barrier["generation"]
            self.maintenance.complete("memory_cutover_maintenance")
            return result
        except Exception as exc:
            self.maintenance.complete("memory_cutover_maintenance", error=str(exc))
            raise

    def cutover_status(self) -> Dict[str, Any]:
        return self.authority.load_root_file("legacy/cutover.json", {"status": "not_started"})

    def maintenance_status(self) -> Dict[str, Any]:
        return self.maintenance.status()

    def ui_formation(self) -> Dict[str, Any]:
        """Return safe formation diagnostics without role secrets.

        Memory counts are counted from the graph itself now: every element a role
        holds lives on the object it is about, so the role's **projected** topology
        is the only correct source for "how much does this role remember".
        """
        from . import node_memory

        role_items = []
        for role in self.roles.list_safe():
            role_uuid = role.get("role_uuid")
            if not role_uuid:
                continue
            groups = self.authority.load_role_file(role_uuid, "relation_groups.json", [])
            topology = self._management_topology(role_uuid)
            self.topology.reload()
            elements = 0
            for record in self.topology.memory_records_for_role(role_uuid):
                elements += len(node_memory.active_elements(record, role_uuid))
            sessions = [
                {
                    key: state[key]
                    for key in ("conversation_id", "state", "updated_at", "source_batch_id")
                    if key in state
                }
                for state in self.authority.list_sessions()
                if state.get("role_uuid") == role_uuid
            ]
            role_items.append({
                "role": role,
                "state": self.authority.role_state(role_uuid),
                "counts": {
                    "memory_elements": elements,
                    "relation_groups": len([item for item in groups if item.get("status", "active") == "active"]),
                    "nodes": len(topology["nodes"]),
                    "relations": len(topology["relations"]),
                    "sessions": len(sessions),
                },
                "relation_groups": [
                    {
                        key: item[key]
                        for key in ("id", "content", "index_brief", "status", "source", "source_id", "manual_protected")
                        if key in item
                    }
                    for item in groups
                ],
                "topology": topology,
                "sessions": sessions,
                "publication_manifest": self.authority.load_role_file(role_uuid, "publication_manifest.json", {}),
            })
        return {
            "service": "graphiti",
            "maintenance": self.maintenance_status(),
            "cutover": self.cutover_status(),
            "roles": role_items,
            "data_root": "data/graphiti",
            "runtime": self.runtime_status(),
            "pad": {"enabled": False, "reason": "PAD is isolated and not part of the Graphiti runtime"},
        }

    @staticmethod
    def _safe_node(node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: node[key]
            for key in ("id", "kind", "name", "created_at", "updated_at")
            if key in node
        }

    @staticmethod
    def _safe_relation(relation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: relation[key]
            for key in ("id", "kind", "endpoints", "created_at", "updated_at")
            if key in relation
        }

    def _management_topology(self, role_uuid: str) -> Dict[str, Any]:
        """This role's objects with the memory it holds about each of them."""
        from . import node_memory

        topology = self._scoped_role_topology(role_uuid)
        nodes = []
        for node in topology["nodes"]:
            nodes.append({
                **self._safe_node(node),
                "memory": node_memory.active_elements(node, role_uuid),
            })
        relations = []
        for relation in topology["relations"]:
            relations.append({
                **self._safe_relation(relation),
                "memory": node_memory.active_elements(relation, role_uuid),
            })
        return {"nodes": nodes, "relations": relations}

    def _scoped_role_topology(self, role_uuid: str) -> Dict[str, Any]:
        """``topology_for_role`` with every other role's identity removed.

        A role's graph — whether it is the main graph's 角色视角 mode or the
        role-scoped management view (``/graphiti/roles/{id}/topology``) — must not
        render another AI. Two sources put one there: a whitelist contaminated
        before the grant rule existed, and an entity the conversation re-created
        from the other AI's name. Both are filtered here once, and both keep their
        memory and their recall: only this view is scoped.
        """
        self.topology.reload()
        topology = self.topology.topology_for_role(role_uuid)
        hidden_ids, hidden_names = self._foreign_identity_scope(role_uuid)

        def visible(record: Dict[str, Any]) -> bool:
            object_id = str(record.get("id") or "")
            if object_id in hidden_ids:
                return False
            name = str(record.get("name") or "").strip().casefold()
            return not (name and name in hidden_names)

        nodes = [node for node in topology["nodes"] if visible(node)]
        relations = [
            relation
            for relation in topology["relations"]
            if visible(relation)
            and not any(str(endpoint) in hidden_ids for endpoint in relation.get("endpoints") or [])
        ]
        return {**topology, "nodes": nodes, "relations": relations}

    def global_topology(self) -> Dict[str, Any]:
        """Return the single shared graph without applying a role whitelist."""
        self.topology.reload()
        return {
            "nodes": [self._safe_node(node) for node in self.topology.data.get("nodes", {}).values()],
            "relations": [self._safe_relation(relation) for relation in self.topology.data.get("relations", {}).values()],
        }

    def graph_display(self, mode: str = "global", role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Return a WebUI main-graph payload for one of two display modes.

        - ``mode == "global"``: the whole shared graph. Each node carries its
          stable id, kind, and display name; the auto-created user node and the
          AI role node therefore render as their real names (e.g. "用户" and the
          role's display name) instead of UUID-only labels.
        - ``mode == "role"``: only the subgraph this role can see (whitelist),
          with that role's private cognition attached to each node and edge.
        """
        if mode == "role":
            if not role_uuid:
                raise ValueError("role_uuid is required for role mode")
            return self._role_graph_display(role_uuid)
        return self._global_graph_display()

    def _global_graph_display(self) -> Dict[str, Any]:
        self.topology.reload()
        nodes = self.topology.data.get("nodes", {})
        relations = self.topology.data.get("relations", {})
        payload = {
            "mode": "global",
            "nodes": [
                {
                    "id": node_id,
                    "kind": node.get("kind"),
                    "name": node.get("name"),
                    # A fact carries the entity it is mounted on (节点细化): the graph
                    # shows it attached, not floating as an entity of its own.
                    "parent_id": node.get("parent_id"),
                }
                for node_id, node in nodes.items()
            ],
            "relations": [
                {
                    "id": relation_id,
                    "kind": relation.get("kind"),
                    "name": relation.get("name"),
                    "endpoints": list(relation.get("endpoints", [])),
                }
                for relation_id, relation in relations.items()
            ],
        }
        payload["revision"] = self._payload_revision(payload)
        return payload

    def _role_graph_display(self, role_uuid: str) -> Dict[str, Any]:
        """This role's view of the graph, each object carrying its own memory.

        There is nothing to group: the projected records *are* the role's memory,
        because every element lives on the object it is about.

        A **fact** is rendered as its own node (节点细化) so the graph shows exactly
        what recall can select: one entity, its identity, and the individual
        knowledge items mounted on it via ``has_fact``.

        Another role's identity is never part of this view. A whitelist that was
        contaminated before the grant rule existed (or an entity re-created from
        the other AI's name) would otherwise render another AI inside this role's
        graph, which is exactly what 角色视角 promises not to show. Hidden nodes
        keep their memory and their recall; only this payload is scoped.
        """
        from . import node_memory

        topology = self._scoped_role_topology(role_uuid)
        nodes = []
        for node in list(topology["nodes"]) + list(topology.get("items") or []):
            nodes.append({
                "id": node.get("id"),
                "kind": node.get("kind"),
                "name": node.get("name"),
                "parent_id": node.get("parent_id"),
                "memory": [
                    str(item.get("text") or "")
                    for item in node_memory.active_elements(node, role_uuid)
                    if str(item.get("text") or "").strip()
                ],
                "summary": str(node.get("summary") or ""),
                "text": node_memory.fact_body(node) if node_memory.is_fact_record(node) else "",
            })
        relations = []
        for relation in topology["relations"]:
            relations.append({
                "id": relation.get("id"),
                "endpoints": relation.get("endpoints"),
                "kind": relation.get("kind"),
                "name": relation.get("name"),
                "memory": [
                    str(item.get("text") or "")
                    for item in node_memory.active_elements(relation, role_uuid)
                    if str(item.get("text") or "").strip()
                ],
            })
        payload = {"mode": "role", "role_uuid": role_uuid, "nodes": nodes, "relations": relations}
        payload["revision"] = self._payload_revision(payload)
        return payload

    def _foreign_identity_scope(self, role_uuid: str) -> Tuple[Set[str], Set[str]]:
        """Other roles' identity nodes and names, as this role's view must hide them."""
        nodes = self.topology.data.get("nodes", {})
        configs = [
            {"role_uuid": other, **self.authority.load_role_file(other, "config.json", {})}
            for other in self.authority.role_ids()
        ]
        # A whitelisted record whose role has no config any more (deleted or
        # imported role) still has to be recognised as foreign by its node kind,
        # which `foreign_identity_scope` does from the topology alone.
        return foreign_identity_scope(role_uuid, nodes, role_name_sets(configs))

    @staticmethod
    def _payload_revision(payload: Dict[str, Any]) -> str:
        """Stable content fingerprint of a graph display payload.

        The WebUI polls ``/graphiti/graph/revision`` (a tiny JSON) instead of the
        full graph and only re-fetches + re-draws when this value changes, so an
        idle graph view stops refreshing the canvas on every poll.
        """
        # Never fold the already-computed revision key into the hash, otherwise
        # the value returned by /graph and the one returned by /graph/revision
        # (which hashes the full display payload) would never match.
        view = {key: value for key, value in payload.items() if key != "revision"}
        canonical = json.dumps(view, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def graph_revision(self, mode: str = "global", role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Lightweight graph-change probe for the WebUI (no full payload).

        Same content the display endpoints serve, hashed deterministically so
        the frontend can compare revisions cheaply on every poll.
        """
        if mode == "role":
            if not role_uuid:
                raise ValueError("role_uuid is required for role mode")
            revision = self._payload_revision(self._role_graph_display(role_uuid))
            return {"mode": mode, "role_uuid": role_uuid, "revision": revision}
        return {"mode": "global", "role_uuid": None, "revision": self._payload_revision(self._global_graph_display())}

    def _warm_batch_records(self, role_uuid: str) -> List[Dict[str, Any]]:
        """The management view of one role's chat batches — one shape, two callers.

        Both the chat-batch view (``temporary_graphs``) and the 温层 page
        (``GET /ui/warm-layer/{role}``) render these rows, so they must come from the
        same builder. They did not: the page fetched ``WarmStore.snapshot()``, whose
        batch rows carry only ``entry_count``/``round_count``, while the template read
        ``batch.entries.length`` — a render-time error that blanked the whole page.
        """
        store = self.warm_store(role_uuid)
        records: List[Dict[str, Any]] = []
        for batch in store.batches():
            batch_id = str(batch.get("batch_id"))
            entries = store.entries_for_batch(batch_id)
            rounds = sorted({int(item.get("round_no") or 0) for item in entries})
            records.append({
                "role_uuid": role_uuid,
                "batch_id": batch_id,
                "conversation_ids": list(batch.get("conversation_ids") or []),
                "status": batch.get("status"),
                "opened_at": batch.get("opened_at"),
                "last_output_at": batch.get("last_output_at"),
                "sealed_at": batch.get("sealed_at"),
                "seal_reason": batch.get("seal_reason"),
                "suspended_at": batch.get("suspended_at"),
                "extraction": batch.get("extraction"),
                "entry_count": len(entries),
                "round_count": len(rounds),
                "rounds": rounds,
                "entries": [
                    {
                        "event_id": item.get("event_id"),
                        "role": item.get("role"),
                        "round_no": item.get("round_no"),
                        "content": str(item.get("content") or "")[:400],
                        "created_at": item.get("created_at"),
                    }
                    for item in entries
                ],
                # An operator-triggered checkpoint acts on a *sealed* batch: a
                # still-open batch is sealed first so the conversation is never
                # split across two extractions.
                "checkpoint_ready": batch.get("status") in {"sealed", "retryable_failed"},
            })
        records.sort(key=lambda item: str(item.get("opened_at") or ""))
        return records

    def temporary_graphs(self) -> Dict[str, Any]:
        """Chat-batch view for the management UI, plus the legacy session shape.

        The temporary graph is no longer the Warm Layer's carrier (plan 阶段 6.2),
        so this endpoint's subject changed: what an operator needs to see is the
        role's **chat batches** — which ones are open, sealed or already extracted,
        and how much transcript each still holds — because that is what the manual
        checkpoint acts on. ``sessions`` stays in the payload for the legacy
        conversation view and for scripts that still read the temporary graph.
        """
        batches: List[Dict[str, Any]] = []
        for role in self._warm_roles():
            batches.extend(self._warm_batch_records(role))
        batches.sort(key=lambda item: (str(item.get("role_uuid")), str(item.get("opened_at") or "")))

        persisted = {
            (str(item.get("role_uuid")), str(item.get("conversation_id"))): item
            for item in self.authority.list_sessions()
            if item.get("role_uuid") and item.get("conversation_id")
        }
        for key, session in self.sessions.items():
            if key not in persisted:
                persisted[key] = session.export_state()
        sessions = []
        for item in persisted.values():
            checkpoint = item.get("checkpoint") if isinstance(item, dict) else None
            temporary = checkpoint.get("temporary_graph") if isinstance(checkpoint, dict) else None
            entries = temporary.get("entries", []) if isinstance(temporary, dict) else []
            nodes = temporary.get("nodes", []) if isinstance(temporary, dict) else []
            node_edges = temporary.get("node_edges", []) if isinstance(temporary, dict) else []
            entry_edges = temporary.get("entry_edges", []) if isinstance(temporary, dict) else []
            sessions.append({
                "role_uuid": item.get("role_uuid"),
                "conversation_id": item.get("conversation_id"),
                "batch_id": item.get("batch_id"),
                "state": item.get("state", "unknown"),
                "started_at": item.get("started_at"),
                "elapsed_seconds": item.get("elapsed_seconds", 0),
                "progress": item.get("progress", {}),
                "progress_log": item.get("progress_log", []),
                "updated_at": item.get("updated_at"),
                "created_at": checkpoint.get("created_at") if isinstance(checkpoint, dict) else None,
                "entries": [dict(entry) for entry in entries if isinstance(entry, dict)],
                "count": len(entries),
                "nodes": [dict(node) for node in nodes if isinstance(node, dict)],
                "node_edges": [dict(edge) for edge in node_edges if isinstance(edge, dict)],
                "entry_edges": [dict(edge) for edge in entry_edges if isinstance(edge, dict)],
            })
        sessions.sort(key=lambda value: (str(value.get("role_uuid")), str(value.get("conversation_id"))))
        return {
            "batches": batches,
            "batch_count": len(batches),
            "sessions": sessions,
            "count": len(sessions),
            "ttl_seconds": self.warm_ttl_seconds,
        }

    def _management_memory(self, role_uuid: str) -> list[Dict[str, Any]]:
        """Every memory element this role holds, flattened with its owner object.

        The list shape is kept (rather than returning the object tree) because the
        management UI shows a flat, filterable table; the object identity travels
        on each row so a row can be traced back to the node it lives on.
        """
        from . import node_memory

        self.topology.reload()
        rows: list[Dict[str, Any]] = []
        for object_type in ("node", "relation", "event"):
            topology = self.topology.topology_for_role(role_uuid)
            source = topology["nodes"] if object_type != "relation" else topology["relations"]
            for record in source:
                object_id = str(record.get("id") or "")
                if node_memory.object_type_of(record) != object_type:
                    continue
                for element in node_memory.active_elements(record, role_uuid):
                    rows.append({
                        **element,
                        "id": str(element.get("id") or ""),
                        "object_id": object_id,
                        "object_type": object_type,
                        "object_name": str(record.get("name") or object_id),
                    })
        rows.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or "")))
        return rows

    def _management_history(self, role_uuid: str) -> list[Dict[str, Any]]:
        """Invalidated elements of this role, as read-only history.

        Never mixed into recall (ADR-0072/ADR-0121): the recall path reads
        ``active_elements`` only, and this view exists solely for the audit UI.
        """
        self.topology.reload()
        return self.topology.memory_history_for_role(role_uuid)

    def _management_prompt(self, role_uuid: str) -> Dict[str, Any]:
        current = self.authority.load_role_file(role_uuid, "prompt/current.json", {})
        current_id = current.get("current_version_id")
        pending_id = current.get("pending_version_id")
        version_id = pending_id or current_id
        version = self.authority.load_role_file(
            role_uuid,
            f"prompt/versions/{version_id}.json" if version_id else "prompt/missing.json",
            {},
        )
        content = version.get("content", "")
        # The editor exposes the persona and the 用语示例 list as two edit areas and
        # compiles them back into the single authoritative body (ADR-0118/ADR-0144).
        persona, examples = split_prompt(content)
        return {
            "prompt_version_id": version.get("prompt_version_id", version_id),
            "current_version_id": current_id,
            "pending_version_id": pending_id,
            "effective_boundary": current.get("effective_boundary", "next_session"),
            "content": content,
            "persona": persona,
            "examples": examples,
        }

    def role_runtime_config(self, role_uuid: str) -> Dict[str, Any]:
        """Loopback runtime config for the local desktop pet.

        This is the only surface where the role's API key and base URL are
        returned, and it is gated behind management (loopback-only) access: the
        desktop pet is the local owner that legitimately needs the chat
        credentials to make its LLM calls. It must never be exposed through the
        browser WebUI, which is deliberately kept free of keys.
        """
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        prompt = self._management_prompt(role_uuid)
        return {
            "role_uuid": role_uuid,
            "display_name": config.get("display_name", role_uuid),
            "provider": config.get("provider", "openai_compat"),
            "base_url": config.get("base_url"),
            "model": config.get("model"),
            "api_key": config.get("api_key", ""),
            "prompt": prompt.get("content", ""),
            "persona": prompt.get("persona", ""),
            "examples": prompt.get("examples", []),
            "runtime_options": dict(config.get("runtime_options") or {}),
            "temperature": (config.get("runtime_options") or {}).get("temperature"),
        }

    def management_projection(self, role_uuid: str) -> Dict[str, Any]:
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        state = self.authority.role_state(role_uuid)
        return {
            "role": {
                key: config[key]
                for key in ("role_uuid", "display_name", "provider", "model", "avatar")
                if key in config
            },
            "state": {
                key: state[key]
                for key in ("role_uuid", "lifecycle", "memory_generation", "updated_at")
                if key in state
            },
            "prompt": self._management_prompt(role_uuid),
            "topology": self._management_topology(role_uuid),
            "memory": self._management_memory(role_uuid),
            "memory_history": self._management_history(role_uuid),
        }

    def add_management_cognition(
        self,
        role_uuid: str,
        text: str,
        anchor_id: str,
        kind: str = "fact",
        *,
        actor: str = "local_management",
    ) -> Dict[str, Any]:
        """Attach one memory element to an object of this role's partition."""
        from . import node_memory

        self.topology.reload()
        if not self.topology.is_allowed(role_uuid, anchor_id):
            raise PermissionError("memory anchor is outside the role whitelist")
        stored = self.topology.find_record(anchor_id)
        if stored is None:
            raise KeyError(anchor_id)
        content = str(text or "").strip()
        if not content:
            raise ValueError("memory text is required")
        result = node_memory.upsert_elements(stored, role_uuid, [content], kind=kind)
        saved = self.topology.save_record_memory(anchor_id, stored)
        added = result["added"][0] if result["added"] else {}
        self.pipeline.saga._index_object(role_uuid, anchor_id)
        self.management_log.log({
            "role_uuid": role_uuid,
            "actor": actor,
            "operation": "memory_create",
            "operation_id": None,
            "memory_id": added.get("id"),
            "object_id": anchor_id,
            "at": utc_timestamp(),
        })
        return {**added, "object_id": anchor_id, "object_type": node_memory.object_type_of(saved)}

    def update_management_prompt(
        self,
        role_uuid: str,
        content: str,
        *,
        expected_version_id: Optional[str] = None,
        actor: str = "local_management",
    ) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip() or len(content.strip()) > 100_000:
            raise ValueError("prompt must be non-empty and at most 100000 characters")
        current = self.authority.load_role_file(role_uuid, "prompt/current.json", {})
        current_id = current.get("current_version_id")
        if expected_version_id is not None and expected_version_id != current_id:
            raise ValueError("prompt changed since it was loaded")
        version_id = typed_id("prompt")
        self.authority.save_role_file(
            role_uuid,
            f"prompt/versions/{version_id}.json",
            {
                "prompt_version_id": version_id,
                "role_uuid": role_uuid,
                "content": content.strip(),
                "parent_version_id": current_id,
                "created_at": utc_timestamp(),
            },
        )
        self.authority.save_role_file(
            role_uuid,
            "prompt/current.json",
            {
                "current_version_id": current_id,
                "pending_version_id": version_id,
                "effective_boundary": "next_session",
            },
        )
        operation = {
            "role_uuid": role_uuid,
            "actor": actor,
            "operation": "prompt_update",
            "operation_id": typed_id("operation"),
            "previous_version_id": current_id,
            "new_version_id": version_id,
            "at": utc_timestamp(),
        }
        self.management_log.log(operation)
        return {"prompt": self._management_prompt(role_uuid), "audit": operation}

    def update_management_cognition(
        self,
        role_uuid: str,
        cognition_id: str,
        text: str,
        *,
        anchor_id: Optional[str] = None,
        kind: Optional[str] = None,
        reason: str,
        actor: str = "local_management",
    ) -> Dict[str, Any]:
        """Correct one memory element: invalidate the old one, add the new one.

        The invalidated element stays on the object as read-only history, so the
        audit trail lives next to the memory it describes instead of in a second
        file that could drift from it.
        """
        from . import node_memory

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("correction reason is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text is required")
        located = self._find_memory(role_uuid, cognition_id)
        if located is None:
            raise KeyError(cognition_id)
        object_id, old = located
        if not self.topology.is_allowed(role_uuid, object_id):
            raise PermissionError("memory object is outside the role whitelist")
        new_kind = kind or str(old.get("kind") or node_memory.DEFAULT_MEMORY_KIND)
        if new_kind not in node_memory.MEMORY_KINDS:
            raise ValueError("unsupported memory kind")
        stored = self.topology.find_record(object_id)
        if stored is None:
            raise KeyError(object_id)
        operation_id = typed_id("operation")
        node_memory.invalidate_element(
            stored,
            role_uuid,
            cognition_id,
            reason=reason.strip(),
            replacement_text=text.strip(),
        )
        if new_kind != str(old.get("kind") or ""):
            for element in node_memory.elements_for_role(stored, role_uuid):
                if element.get("status") == "active" and node_memory.normalize_text(element.get("text")) == node_memory.normalize_text(text):
                    element["kind"] = new_kind
        self.topology.save_record_memory(object_id, stored)
        self.pipeline.saga._index_object(role_uuid, object_id)
        elements = node_memory.elements_for_role(stored, role_uuid)
        replacement = next(
            (item for item in elements if item.get("status") == "active"
             and node_memory.normalize_text(item.get("text")) == node_memory.normalize_text(text)),
            {},
        )
        audit = {
            "role_uuid": role_uuid,
            "actor": actor,
            "operation": "memory_correction",
            "operation_id": operation_id,
            "previous_memory_id": cognition_id,
            "replacement_memory_id": replacement.get("id"),
            "object_id": object_id,
            "at": utc_timestamp(),
        }
        self.management_log.log(audit)
        return {
            "memory": {**replacement, "object_id": object_id},
            "previous_memory_id": cognition_id,
            "audit": audit,
        }

    def invalidate_management_cognition(
        self,
        role_uuid: str,
        cognition_id: str,
        *,
        reason: str,
        actor: str = "local_management",
    ) -> Dict[str, Any]:
        """Invalidate one memory element, keeping it as read-only history."""
        from . import node_memory

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalidation reason is required")
        located = self._find_memory(role_uuid, cognition_id)
        if located is None:
            raise KeyError(cognition_id)
        object_id, _ = located
        if not self.topology.is_allowed(role_uuid, object_id):
            raise PermissionError("memory object is outside the role whitelist")
        stored = self.topology.find_record(object_id)
        if stored is None:
            raise KeyError(object_id)
        operation_id = typed_id("operation")
        node_memory.invalidate_element(stored, role_uuid, cognition_id, reason=reason.strip())
        self.topology.save_record_memory(object_id, stored)
        self.pipeline.saga._index_object(role_uuid, object_id)
        audit = {
            "role_uuid": role_uuid,
            "actor": actor,
            "operation": "memory_invalidate",
            "operation_id": operation_id,
            "memory_id": cognition_id,
            "object_id": object_id,
            "at": utc_timestamp(),
        }
        self.management_log.log(audit)
        return {"memory_id": cognition_id, "object_id": object_id, "audit": audit}

    def _find_memory(self, role_uuid: str, element_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Locate one memory element by id anywhere in this role's partition."""
        from . import node_memory

        self.topology.reload()
        for record in self.topology.memory_records_for_role(role_uuid):
            for element in node_memory.elements_for_role(record, role_uuid):
                if str(element.get("id")) == str(element_id):
                    return str(record.get("id") or ""), element
        return None

    def management_audit(self, role_uuid: str) -> list[Dict[str, Any]]:
        if not self.management_log.path.exists():
            return []
        entries = []
        allowed_keys = {
            "role_uuid", "actor", "operation", "operation_id", "previous_version_id", "new_version_id",
            "previous_memory_id", "replacement_memory_id", "memory_id", "object_id", "at",
        }
        with self.management_log.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("role_uuid") == role_uuid:
                    entries.append({key: event[key] for key in allowed_keys if key in event})
        return entries

    def purge_role(self, role_uuid: str, *, remove_directory: bool = True, allow_active: bool = False) -> Dict[str, Any]:
        """Delete one role's whole partition, everywhere it exists.

        Cancelling or losing a document import used to delete only the role directory and
        the registry entry, which left the published graph behind: the shared document
        kept the two unconfirmed identity nodes (rendered in the global graph as
        「未命名角色」) plus every fact, hook, predicate, event and relation that partition
        had published, its vectors and its SQLite bookkeeping — all unreachable, because
        the partition they belonged to no longer existed.

        Order matters: the topology objects go first because the ids they return are what
        the vector and SQLite deletions are keyed by. The derived stores are best-effort
        (each failure is reported instead of aborting the purge) but the topology rewrite
        is authoritative — if it raises, nothing else has happened yet.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required to purge a role")
        state = self.authority.role_state(role)
        lifecycle = str((state or {}).get("lifecycle") or "")
        if lifecycle == "active" and not allow_active:
            raise ValueError(f"role {role} is active; pass allow_active to purge a live role")
        report: Dict[str, Any] = {
            "role_uuid": role,
            "lifecycle": lifecycle,
            "display_name": str(self.authority.load_role_file(role, "config.json", {}).get("display_name") or ""),
        }

        self.topology.reload()
        purge = self.topology.purge_role(role)
        report["topology"] = {
            "removed_nodes": len(purge["removed_nodes"]),
            "removed_relations": len(purge["removed_relations"]),
            "removed_event_links": len(purge["removed_event_links"]),
            "kept_shared_nodes": len(purge["kept_shared_nodes"]),
            "kept_shared_relations": len(purge["kept_shared_relations"]),
            "had_partition": purge["had_partition"],
        }
        object_ids = [*purge["removed_nodes"], *purge["removed_relations"], *purge["removed_event_links"]]

        vectors = getattr(self.memory_layer, "vectors", None)
        if vectors is not None and getattr(self.memory_layer, "vectors_available", False):
            try:
                report["vectors"] = vectors.delete_role(role)
            except Exception as exc:  # noqa: BLE001 - report, never abort the purge
                report["vectors"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            report["vectors"] = {"status": "unavailable"}

        try:
            report["sqlite"] = self.sqlite.delete_role(role, object_ids=object_ids)
        except Exception as exc:  # noqa: BLE001
            report["sqlite"] = {"error": f"{type(exc).__name__}: {exc}"}

        try:
            report["neo4j"] = self._purge_role_neo4j(role)
        except Exception as exc:  # noqa: BLE001
            report["neo4j"] = {"error": f"{type(exc).__name__}: {exc}"}
        # 日历视图是派生数据（可由回合正文重放），但角色没了它就是垃圾。
        try:
            report["calendar"] = self.calendar.purge(role)
        except Exception as exc:  # noqa: BLE001
            report["calendar"] = {"error": f"{type(exc).__name__}: {exc}"}

        projection = self.root / "projections" / f"{role}.json"
        try:
            if projection.exists():
                projection.unlink()
                report["projection_removed"] = True
        except OSError as exc:
            report["projection_removed"] = f"{type(exc).__name__}: {exc}"

        if remove_directory:
            directory = self.authority.characters_root / role
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
            report["directory_removed"] = not directory.exists()

        registry = self.authority.role_registry()
        if role in (registry.get("roles") or {}):
            registry["roles"].pop(role, None)
            self.authority.save_role_registry(registry)
            report["registry_removed"] = True
        else:
            report["registry_removed"] = False
        report["status"] = "purged"
        return report

    def _purge_role_neo4j(self, role_uuid: str) -> Dict[str, Any]:
        """Delete one role's identity nodes and its projected events from Neo4j."""
        bridge = getattr(self.neo4j_graph, "execute", None)
        if bridge is None:
            return {"status": "unavailable"}
        removed: Dict[str, Any] = {}
        for label, uuid in (("ai_node", f"ai:{role_uuid}"), ("user_node", f"user:{role_uuid}")):
            result = self.neo4j_graph.execute(
                "MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n RETURN count(*) AS removed",
                {"uuid": uuid},
            )
            records = getattr(result, "records", None) or []
            removed[label] = int(records[0].data().get("removed", 0)) if records else 0
        result = self.neo4j_graph.execute(
            "MATCH (n:Event) WHERE n.group_id = $role OR n.role_uuid = $role_node "
            "DETACH DELETE n RETURN count(*) AS removed",
            {"role": role_uuid, "role_node": f"ai:{role_uuid}"},
        )
        records = getattr(result, "records", None) or []
        removed["events"] = int(records[0].data().get("removed", 0)) if records else 0
        return removed

    def forget_role_relation(self, role_uuid: str, relation_id: str) -> Dict[str, Any]:
        """Revoke one relation and drop this role's memory about it.

        The tombstone is written **before** the invalidation so the role's memory
        history for that relation stays reachable after the whitelist entry is gone.
        """
        relation_id = self.topology.resolve_relation(relation_id)
        if self.topology.get_relation(relation_id) is None:
            raise KeyError(relation_id)
        self.topology.revoke_relation(role_uuid, relation_id)
        self.authority.add_tombstone(role_uuid, "relation", relation_id, "role_forget")
        removed = self._forget_object_memory(role_uuid, relation_id)
        return {"role_uuid": role_uuid, "relation_id": relation_id, "removed_memory_ids": removed}

    def forget_role_node(self, role_uuid: str, node_id: str) -> Dict[str, Any]:
        node_id = self.topology.resolve_node(node_id)
        removed_relations = self.topology.revoke_node(role_uuid, node_id)
        self.authority.add_tombstone(role_uuid, "node", node_id, "role_forget")
        removed_cognitions = self._forget_object_memory(role_uuid, node_id)
        for relation_id in removed_relations:
            self.authority.add_tombstone(role_uuid, "relation", relation_id, "role_forget")
            removed_cognitions.extend(self._forget_object_memory(role_uuid, relation_id))
        return {
            "role_uuid": role_uuid,
            "node_id": node_id,
            "removed_relation_ids": removed_relations,
            "removed_memory_ids": removed_cognitions,
        }

    def _forget_object_memory(self, role_uuid: str, object_id: str, *, reason: str = "role_forget") -> List[str]:
        """Invalidate every element this role holds about one object.

        Nothing is physically deleted: the elements stay on the object as
        read-only history so a later audit can still see what the role used to
        believe, and recall stops seeing them because it reads active elements only.
        """
        from . import node_memory

        self.topology.reload()
        stored = self.topology.find_record(object_id)
        if stored is None:
            return []
        removed: List[str] = []
        for element in node_memory.elements_for_role(stored, role_uuid):
            if element.get("status", "active") != "active":
                continue
            node_memory.invalidate_element(stored, role_uuid, str(element.get("id") or ""), reason=reason)
            removed.append(str(element.get("id") or ""))
        if removed:
            self.topology.save_record_memory(object_id, stored)
            self.pipeline.saga._index_object(role_uuid, object_id)
        return removed

    def delete_shared_relation(self, relation_id: str) -> Dict[str, Any]:
        """Delete one relation for everyone, keeping each role's memory as history.

        Every role's elements about the relation are invalidated **before** the
        record is removed, so the deletion leaves a readable history instead of
        silently discarding what each role used to believe.
        """
        relation_id = self.topology.resolve_relation(relation_id)
        # Everyone who could read it: the role registry plus any role that was
        # actually granted it (a role created outside the registry, as in a
        # SQLite-only authority, still owns memory that must become history).
        self.topology.reload()
        granted = [
            role_uuid
            for role_uuid, entry in (self.topology.data.get("whitelist") or {}).items()
            if relation_id in (entry or {}).get("relations", [])
        ]
        removed: Dict[str, List[str]] = {}
        for role_uuid in sorted(set(self.authority.role_ids()) | set(granted)):
            removed[role_uuid] = self._forget_object_memory(
                role_uuid, relation_id, reason="explicit_delete"
            )
        result = self.topology.delete_relation(relation_id, reason="explicit_delete")
        if result["status"] == "already_deleted":
            return result
        for role_uuid in removed:
            self.authority.add_tombstone(role_uuid, "relation", relation_id, "explicit_delete")
        result["removed_memories"] = removed
        return result

    def merge_nodes(self, primary_id: str, member_id: str, operation_id: Optional[str] = None) -> Dict[str, Any]:
        """Merge two nodes and fold the member's memory into the primary node.

        Memory elements are role-keyed, so each role's slice is merged separately:
        the member's elements move to the primary node and any element the primary
        already holds with the same text is dropped instead of being duplicated.
        """
        from . import node_memory

        self.topology.reload()
        member_snapshot = self.topology.find_record(member_id)
        member_memory = {
            role: node_memory.elements_for_role(member_snapshot, role)
            for role in node_memory.roles_with_memory(member_snapshot)
        } if member_snapshot is not None else {}
        result = self.topology.merge_nodes(primary_id, member_id, operation_id=operation_id)
        primary_id = result["primary_uuid"]
        migrated: Dict[str, List[str]] = {}
        for role_uuid, elements in member_memory.items():
            self.topology.reload()
            primary = self.topology.find_record(primary_id)
            if primary is None:
                continue
            moved: List[str] = []
            for element in elements:
                if element.get("status", "active") != "active":
                    continue
                text = str(element.get("text") or "").strip()
                if not text or node_memory.contains_text(primary, role_uuid, text) is not None:
                    continue
                attached = node_memory.upsert_elements(
                    primary,
                    role_uuid,
                    [text],
                    kind=str(element.get("kind") or node_memory.DEFAULT_MEMORY_KIND),
                    source_conversation_id=element.get("source_conversation_id"),
                )
                moved.extend(str(item.get("id") or "") for item in attached["added"])
            if moved:
                self.topology.save_record_memory(primary_id, primary)
                self.pipeline.saga._index_object(role_uuid, primary_id)
            migrated[role_uuid] = moved
        result["migrated_memory_ids"] = migrated
        return result
