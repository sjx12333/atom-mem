"""Loopback-safe Graphiti gateway routes with a MemOS-compatible response shape."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from ..graphiti_memory import GraphitiMemoryService
    from ..graphiti_memory.api_management import ManagementAccessError
    from ..graphiti_memory.prompt_sections import compile_prompt
    from ..graphiti_memory.roles import ConnectionTestError, RoleLifecycleError
    from .management_auth import management_access_dependency, record_management_failure
    from .knowledge_import_routes import create_knowledge_import_routes
except ImportError:
    from graphiti_memory import GraphitiMemoryService
    from graphiti_memory.api_management import ManagementAccessError
    from graphiti_memory.prompt_sections import compile_prompt
    from graphiti_memory.roles import ConnectionTestError, RoleLifecycleError
    from api.management_auth import management_access_dependency, record_management_failure
    from api.knowledge_import_routes import create_knowledge_import_routes


def _compiled_prompt(persona: str, examples: Optional[List[str]]) -> str:
    """Compile the persona + 用语示例 edit areas into the single authoritative body."""
    if examples is None:
        return persona
    try:
        return compile_prompt(persona, examples)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _runtime_options_from_form(runtime_temperature: Optional[str]) -> Dict[str, Any]:
    """Build runtime options from the import form (empty string = temperature disabled)."""
    if runtime_temperature is None or not str(runtime_temperature).strip():
        return {}
    try:
        return {"temperature": float(str(runtime_temperature).strip())}
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="temperature must be a number, or empty to disable it"
        ) from exc


class FocusContact(BaseModel):
    """External contact the current turn is talking to (QQ 等)。

    只用于**排序偏好**：与该联系人有关的记忆在召回里加分往前放，不改变任何
    召回门槛，也不新增白名单访问。
    """

    channel: str = "qq"
    kind: str = "user"
    number: str
    group_number: Optional[str] = None


class GraphitiSearchRequest(BaseModel):
    query: str
    role_uuid: str
    user_id: str = "feiniu_default"
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=0, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_contact: Optional[FocusContact] = None
    # --- 工具退避（阶段 5.2）------------------------------------------------
    # 工具查询必须能说"这些片段系统已经自动给过了"，否则同一段内容会在自动片段
    # 和工具结果里各出现一次（模型读成两份证据）。三种排除按顺序生效：精确 id →
    # ``(batch_id, round_no)`` → 内容哈希。
    exclude_ids: List[str] = Field(default_factory=list)
    exclude_batch_rounds: List[List[Any]] = Field(default_factory=list)
    exclude_hashes: List[str] = Field(default_factory=list)
    #: 只按正文排除（调用方拿不到哈希时）：排除本次排序最靠前的 N 条的哈希。
    exclude_top_hashes: int = Field(default=0, ge=0, le=50)
    #: 当前轮的 (batch_id, round_no)：当前轮绝不自引用。
    current_batch_id: Optional[str] = None
    current_round_no: Optional[int] = None
    #: 默认排除 role=assistant 的正文（AI 的自述不是主人的事实）。
    include_assistant: bool = False
    #: 工具二次查询/搜索测试/管理探测应传 false：命中仍返回，但不计入激活量。
    count_activation: bool = True
    #: 查询问的是"哪一天"时的参照时间（ISO 8601）。历史重放里「上周三」必须
    #: 相对当时解析，不能相对现在（ADR-0178）。缺省 = 现在。
    calendar_reference: Optional[str] = None


class GraphQueryRequest(BaseModel):
    """One deterministic graph query from the LLM-facing memory tool.

    The model may supply anchors and an operator explicitly, or only a natural
    language query; the backend resolves anchors against the role whitelist and
    infers the operator when it is omitted. It never answers, only returns
    evidence with stable fact/entity ids.
    """

    query: str = ""
    role_uuid: str
    operator: Optional[str] = None
    anchors: List[str] = Field(default_factory=list)
    predicate: str = ""
    target: str = ""
    top_k: int = Field(default=12, ge=1, le=40)
    max_depth: int = Field(default=2, ge=1, le=4)


class GraphitiAddRequest(BaseModel):
    role_uuid: str
    user_id: str = "feiniu_default"
    conversation_id: str
    messages: List[Dict[str, Any]]


class GraphitiEmbedRequest(BaseModel):
    """Texts to embed with the role's configured embedding model (ADR-0163/0165).

    The desktop pet archives chat transcripts it owns (QQ history) and needs
    vectors for them. Embedding lives here so the credential and the model
    fingerprint stay in exactly one place; the caller only stores the result.
    """

    texts: List[str] = Field(min_length=1, max_length=64)
    role_uuid: Optional[str] = None


class ExternalSourceRequest(BaseModel):
    """One inbound message source to hard-register (ADR-0152)."""

    role_uuid: str
    kind: str = Field(description="private or group")
    number: str = Field(description="sender QQ number")
    display_name: Optional[str] = None
    group_number: Optional[str] = None
    group_name: Optional[str] = None


class ExternalSourceSpec(BaseModel):
    """One contact the caller knows about, used to repair a missing node/edge."""

    kind: str = "private"
    number: str
    display_name: Optional[str] = None
    group_number: Optional[str] = None
    group_name: Optional[str] = None


class ExternalSourceReconcileRequest(BaseModel):
    """Verify (and repair) the QQ nodes and role edges of one role.

    ``sources`` is optional: without it the role's own QQ nodes are checked and a
    missing edge to the AI role node is written. With it, a contact whose node was
    never created at all is registered too, and the knowledge entity extracted under
    the same display name is linked to the role node.
    """

    role_uuid: str
    sources: Optional[List[ExternalSourceSpec]] = None


class ExplicitRememberRequest(BaseModel):
    role_uuid: str
    content: str
    decision: Dict[str, Any]


class MemoryElementRequest(BaseModel):
    """One memory element to attach to an object of a role's partition."""

    text: str
    anchor_id: str
    kind: str = "fact"


class PromptUpdateRequest(BaseModel):
    content: str
    expected_version_id: Optional[str] = None


class MemoryElementUpdateRequest(BaseModel):
    text: str
    reason: str
    anchor_id: Optional[str] = None
    kind: Optional[str] = None


class MemoryElementInvalidateRequest(BaseModel):
    reason: str


class SessionTurnRequest(BaseModel):
    role_uuid: str
    conversation_id: str
    user_content: str
    assistant_content: str
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)
    complete: bool = True
    source_ids: List[str] = Field(default_factory=list)
    source_batch_id: Optional[str] = None
    source_layer: str = "LongTermMemory"
    # 用户对当前角色的称呼（AI 对用户的称呼，默认"主人"）。抽取时用它代替
    # "user:" 标签，使抽取出的用户实体名是真实称呼而不是 "user"。
    user_appellation: Optional[str] = None
    # 这一轮**发生**的时间（ISO 8601 或 epoch 秒）。实时对话不需要它；历史重放
    # （导入、离线评测）必须带上，否则正文里的「今天/明天」会按写库那一刻解析，
    # 2023 年的对话会被记成本周（ADR-0178）。缺省 = 现在。
    chat_time: Optional[str] = None
    # 外部渠道来源（QQ 私聊/群聊）。存在时本轮不是本机主人输入，抽取必须把来源
    # 当外部联系人处理，不能记成主人的事实。
    external_source: Optional[Dict[str, Any]] = None
    # 面板记账凭据：``/graphiti/search`` 返回它，回合记录原样带回，后端据此把
    # "这一轮实际召回了什么"写进聊天记录窗口。缺失时该轮只记 ``recall_linked:false``，
    # 绝不猜测（ADR-0166）。
    recall_token: Optional[str] = None


class SessionRequest(BaseModel):
    role_uuid: str
    conversation_id: str
    reason: str = "explicit_end"
    # force: 宠物端触发裁剪时跳过 2 分钟检查点时间门槛，强制归档+临时图谱化。
    force: bool = False


class UtteranceEndRequest(BaseModel):
    role_uuid: str
    conversation_id: str


class WarmBatchRequest(BaseModel):
    """Select chat batches and act on them.

    ``batch_ids`` empty means "every batch of this role", which is what the
    management UI's "seal everything" / "extract everything" affordance sends
    when the operator selects nothing.
    """

    role_uuid: str
    batch_ids: List[str] = Field(default_factory=list)
    reason: str = "manual_checkpoint"


class BackupRequest(BaseModel):
    role_uuid: str
    kind: str = "manual"
    reason: Optional[str] = None


class BatchClaimRequest(BaseModel):
    role_uuid: str
    batch_id: str
    source_ids: List[str]
    layer: str = "UserMemory"


class CutoverRequest(BaseModel):
    role_uuid: str
    operation_id: str
    source_ids: List[str]
    confirm: bool = False


class RestoreRequest(BaseModel):
    role_uuid: str
    path: str


class VectorRebuildRequest(BaseModel):
    role_uuid: Optional[str] = None
    drop: bool = True


class MemoryExportRequest(BaseModel):
    target: Optional[str] = None


class RuntimeConfigRequest(BaseModel):
    config: Dict[str, Any]


class HotZoneRequest(BaseModel):
    enabled: bool


class ReflectionRunRequest(BaseModel):
    role_uuid: str
    force: bool = False


class OverlayPauseRequest(BaseModel):
    role_uuid: str
    paused: bool


class OverlayActivateRequest(BaseModel):
    role_uuid: str
    overlay_id: str


class OverlayEditRequest(BaseModel):
    role_uuid: str
    items: List[Dict[str, Any]]


class OverlayLockRequest(BaseModel):
    role_uuid: str
    item_id: str
    locked: bool


class ActivationResetRequest(BaseModel):
    role_uuid: str


class HotZoneRebuildRequest(BaseModel):
    role_uuid: str


class RoleCreateRequest(BaseModel):
    display_name: str
    provider: str = "openai_compat"
    base_url: Optional[str] = None
    model: str
    api_key: str
    prompt: str
    #: 用语示例 edit area. When provided, the server compiles it into the single
    #: authoritative prompt body (ADR-0118/ADR-0144); ``None`` keeps ``prompt`` verbatim
    #: so older clients that edit the full body still work.
    examples: Optional[List[str]] = None
    avatar: Optional[str] = None
    runtime_options: Dict[str, Any] = Field(default_factory=dict)


class RoleConnectionTestRequest(BaseModel):
    provider: str = "openai_compat"
    base_url: Optional[str] = None
    model: str
    api_key: str


class RoleImportSpec(BaseModel):
    display_name: str
    source_role_name: str
    source_user_name: str
    user_appellation: str
    provider: str = "openai_compat"
    base_url: Optional[str] = None
    model: str
    api_key: str
    avatar: Optional[str] = None
    runtime_options: Dict[str, Any] = Field(default_factory=dict)


class RoleImportConfirmRequest(BaseModel):
    prompt: str
    examples: Optional[List[str]] = None


def _default_llm_probe(config: Dict[str, Any]) -> None:
    """Side-effect-free minimal LLM connection probe (no persisted memory).

    Raises on any connection/authentication failure so the role wizard can
    classify the error (ADR-0093) before committing a new role.
    """
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    model = str(config.get("model") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    if not base_url or not model or not api_key:
        raise ValueError("base_url, model, and api_key are required")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def _sse(event: str, payload: Dict[str, Any]) -> str:
    """Serialize one server-sent event (one JSON line, no multi-line frames)."""
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def create_router(
    root: Path,
    management_token: Optional[str] = None,
    llm_probe: Optional[Callable[[Dict[str, Any]], None]] = None,
    service: Optional[GraphitiMemoryService] = None,
) -> APIRouter:
    router = APIRouter(prefix="/graphiti", tags=["graphiti-memory"])
    # An injected service is how a test keeps the whole route surface off the real
    # database: a temporary data root does not sandbox the Neo4j target, so building
    # the service here from the file would let a route test write to production.
    service = service if service is not None else GraphitiMemoryService(root)
    router.graphiti_memory_service = service
    gateway = service.gateway
    explicit = service.explicit
    roles = service.roles
    probe = llm_probe or _default_llm_probe
    service.role_import_connection_probe = probe
    configured_management_token = management_token or os.getenv("GRAPHITI_MANAGEMENT_TOKEN")

    require_management_access = management_access_dependency(root, configured_management_token)

    def _record_management_failure(request: Request, status: int, detail: Any) -> None:
        record_management_failure(root, request, status, detail)

    def require_management_role(role_uuid: str) -> None:
        registry = roles.authority.role_registry().get("roles", {})
        if role_uuid not in registry or registry[role_uuid].get("status") == "deleted":
            raise HTTPException(status_code=404, detail="role is not available")

    def require_startable(role_uuid: str) -> None:
        try:
            roles.assert_startable(role_uuid)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/search")
    async def search(request: GraphitiSearchRequest):
        require_startable(request.role_uuid)
        # 召回是同步阻塞的重活：向量检索、reranker（LLM）以及等待 graphiti 的
        # 单一异步循环。宠物端每一轮都会调它，直接在事件循环里跑会让整个网关
        # （这一轮自己的后续请求、WebUI 轮询、SSE 心跳）一起停住。
        return await asyncio.to_thread(
            service.search,
            request.role_uuid,
            request.user_id,
            request.conversation_id,
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            focus_contact=(
                request.focus_contact.model_dump() if request.focus_contact is not None else None
            ),
            # 工具退避：把"系统已经自动提供过的片段"排除掉，并让响应报告省略了多少
            # 条（``retrieval_diagnostics.suppressed``），模型才不会把空结果读成
            # "没有记忆"。
            exclude_ids=list(request.exclude_ids or []),
            exclude_batch_rounds=[list(item) for item in (request.exclude_batch_rounds or [])],
            exclude_hashes=list(request.exclude_hashes or []),
            exclude_top_hashes=int(request.exclude_top_hashes or 0),
            current_batch_id=request.current_batch_id,
            current_round_no=request.current_round_no,
            include_assistant=bool(request.include_assistant),
            count_activation=bool(request.count_activation),
            calendar_reference=request.calendar_reference,
        )

    @router.post("/query")
    async def graph_query(request: GraphQueryRequest):
        require_startable(request.role_uuid)
        return await asyncio.to_thread(
            service.graph_query,
            request.role_uuid,
            request.query,
            operator=request.operator,
            anchors=list(request.anchors or []),
            predicate=request.predicate,
            target=request.target,
            top_k=request.top_k,
            max_depth=request.max_depth,
        )

    @router.post("/embed")
    async def embed(request: GraphitiEmbedRequest):
        """Embed caller-supplied text with the role's configured model.

        Deliberately **not** behind the management token: it returns a vector for
        text the caller already holds and reads no memory content. It exists so a
        client that archives its own transcripts (the QQ history store, ADR-0165)
        can precompute vectors without duplicating the credential or the model
        fingerprint. A missing or failing embedder is reported, never guessed.
        """
        embedder = gateway.node_recall.embedder
        if embedder is None:
            raise HTTPException(status_code=503, detail="embedding model is not configured")
        texts = [str(text or "") for text in request.texts]
        if any(not text.strip() for text in texts):
            raise HTTPException(status_code=400, detail="texts must not be blank")

        def _embed() -> List[List[float]]:
            vectors: List[List[float]] = []
            for text in texts:
                vector = embedder(text)
                if not vector:
                    raise RuntimeError("embedding model returned no vector")
                vectors.append([float(item) for item in vector])
            return vectors

        try:
            vectors = await asyncio.to_thread(_embed)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as 503
            raise HTTPException(status_code=503, detail=f"embedding failed: {exc}") from exc
        return {"dimension": len(vectors[0]) if vectors else 0, "vectors": vectors}

    @router.post("/add")
    async def add(request: GraphitiAddRequest):
        require_startable(request.role_uuid)
        return gateway.add_working_memory(
            request.role_uuid,
            request.user_id,
            request.conversation_id,
            request.messages,
        )

    @router.post("/ingest/external-source")
    async def register_external_source(request: ExternalSourceRequest):
        """Hard-register the sender (and group) of an accepted inbound message.

        The desktop pet calls this the moment a message is accepted, before it
        answers, so a contact exists in the graph even if the conversation never
        produces an extractable statement. It is idempotent and writes only the
        node plus the role grant (ADR-0152).
        """
        try:
            require_startable(request.role_uuid)
            return service.register_external_source(
                request.role_uuid,
                kind=request.kind,
                number=request.number,
                display_name=request.display_name,
                group_number=request.group_number,
                group_name=request.group_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/ingest/external-source/reconcile")
    async def reconcile_external_sources(request: ExternalSourceReconcileRequest):
        """Verify and repair QQ sources: the node and its default edge to the role.

        The self-healing entry point behind the desktop pet's hard join. It closes
        the gap the fire-and-forget registration cannot: a contact whose node or
        edge was never written (request lost, retries exhausted, restart) becomes
        complete again on the next call, with no manual graph surgery.
        """
        try:
            require_startable(request.role_uuid)
            return service.reconcile_external_sources(
                request.role_uuid,
                sources=[item.model_dump() for item in (request.sources or [])],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/turn")
    async def record_turn(request: SessionTurnRequest):
        try:
            require_startable(request.role_uuid)
            # 记录一个回合**可能同步跑完整抽取**：长时间静默后再说话，旧会话
            # 往往需要先封批（``_settle_superseded_session``）或重发
            # （``retry_locked``），两条路径都会在这一调用里做 LLM 抽取与图写入。
            # 以前这里直接在事件循环里执行，于是「静默后第一句话」会把整个网关
            # 卡住几十秒：这一轮自己的 memos 调用（5 秒超时）与其它请求全部超时，
            # 表现出来就是"长时间静默后再次对话无响应"。放到线程池执行。
            await asyncio.to_thread(
                service.record_turn,
                request.role_uuid,
                request.conversation_id,
                user_content=request.user_content,
                assistant_content=request.assistant_content,
                tool_events=request.tool_events,
                complete=request.complete,
                source_ids=request.source_ids,
                source_batch_id=request.source_batch_id,
                source_layer=request.source_layer,
                user_appellation=request.user_appellation,
                chat_time=request.chat_time,
                external_source=request.external_source,
                recall_token=request.recall_token,
            )
            return service.session_status(request.role_uuid, request.conversation_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/checkpoint")
    async def checkpoint_session(request: SessionRequest):
        try:
            require_startable(request.role_uuid)
            # 临时图谱的 LLM 抽取是同步阻塞调用；在事件循环里直接执行会卡死整个
            # 服务（所有轮询/请求都无响应）。放到线程池执行，使 "checkpointing"
            # 状态可被前端轮询看到，同时不阻塞其它请求。
            checkpoint = await asyncio.to_thread(
                service.checkpoint, request.role_uuid, request.conversation_id, None, request.force
            )
            return {"checkpoint": checkpoint, "session": service.session_status(request.role_uuid, request.conversation_id)}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/check-idle")
    async def check_session_idle(request: SessionRequest):
        try:
            require_startable(request.role_uuid)
            result = await asyncio.to_thread(service.check_idle, request.role_uuid, request.conversation_id)
            return {"result": result, "session": service.session_status(request.role_uuid, request.conversation_id)}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/utterance-end")
    async def mark_utterance_end(request: UtteranceEndRequest):
        """TTS 播完：空闲窗口从"真的说完"开始计时（只改时钟，不触发抽取）。"""
        try:
            require_startable(request.role_uuid)
            # 本身很轻（改计时 + 落盘），但它挂在 TTS 结束钩子上：绝不能因为事件
            # 循环正忙而让说话结束的收尾卡住，所以与其它会话操作一致走线程池。
            return await asyncio.to_thread(
                service.mark_utterance_end, request.role_uuid, request.conversation_id
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/close")
    async def close_session(request: SessionRequest):
        try:
            require_startable(request.role_uuid)
            # close 内部会同步跑正式知识图谱抽取（LLM 调用可能持续数十秒）；
            # 与 checkpoint 保持一致放到线程池执行，否则整个事件循环被阻塞，
            # 宠物端每分钟的 check-idle 与 WebUI 的状态轮询全部卡死。
            result = await asyncio.to_thread(
                service.close_session, request.role_uuid, request.conversation_id, request.reason
            )
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/sessions/retry")
    async def retry_session(request: SessionRequest):
        try:
            require_startable(request.role_uuid)
            # 没有后台任务接手时，重试会同步执行整批抽取与发布。
            return await asyncio.to_thread(
                service.retry_session, request.role_uuid, request.conversation_id
            )
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/sessions/{role_uuid}/{conversation_id}")
    async def session_status(role_uuid: str, conversation_id: str):
        # Inspection/monitoring only: a session view must stay available even if
        # the role is not currently startable (e.g. LLM key temporarily absent),
        # so the WebUI synced view does not fail with a 409 on every poll.
        require_management_role(role_uuid)
        return service.session_status(role_uuid, conversation_id)

    @router.post("/remember")
    async def remember(request: ExplicitRememberRequest):
        try:
            require_startable(request.role_uuid)
            # 显式记忆变更会立即走高优先级发布队列（判定重复、写图、写索引），
            # 属于重活，同样不能占用事件循环。
            return await asyncio.to_thread(
                explicit.remember, request.role_uuid, request.content, request.decision
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/operations/backups", dependencies=[Depends(require_management_access)])
    async def list_backups(role_uuid: Optional[str] = None, kind: Optional[str] = None):
        return {"backups": service.list_backups(role_uuid, kind)}

    @router.post("/operations/backups", dependencies=[Depends(require_management_access)])
    async def create_backup(request: BackupRequest):
        try:
            require_management_role(request.role_uuid)
            return service.create_backup(request.role_uuid, request.kind, request.reason)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/operations/projections/{role_uuid}/rebuild", dependencies=[Depends(require_management_access)])
    async def rebuild_projection(role_uuid: str):
        try:
            require_management_role(role_uuid)
            return service.rebuild_projection(role_uuid)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/operations/user-memory/claim", dependencies=[Depends(require_management_access)])
    async def claim_user_memory_batch(request: BatchClaimRequest):
        try:
            require_management_role(request.role_uuid)
            return service.claim_user_memory_batch(request.role_uuid, request.batch_id, request.source_ids, layer=request.layer)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/operations/cutover", dependencies=[Depends(require_management_access)])
    async def clean_cutover(request: CutoverRequest):
        try:
            require_management_role(request.role_uuid)
            return service.clean_cutover(request.role_uuid, request.operation_id, request.source_ids, confirm=request.confirm)
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/operations/cutover", dependencies=[Depends(require_management_access)])
    async def cutover_status():
        return service.cutover_status()

    @router.get("/operations/maintenance", dependencies=[Depends(require_management_access)])
    async def maintenance_status():
        return service.maintenance_status()

    # --- SQLite memory authority operations (阶段 6/7) -------------------------

    @router.get("/operations/memory/status", dependencies=[Depends(require_management_access)])
    async def memory_status():
        return service.memory_status()

    @router.get("/operations/memory/diagnose", dependencies=[Depends(require_management_access)])
    async def memory_diagnose(role_uuid: Optional[str] = None):
        return service.memory_diagnose(role_uuid)

    @router.get("/operations/memory/vectors/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def memory_vectors(role_uuid: str):
        require_management_role(role_uuid)
        return service.memory_vector_freshness(role_uuid)

    @router.post("/operations/memory/vectors/rebuild", dependencies=[Depends(require_management_access)])
    async def rebuild_vectors(request: VectorRebuildRequest):
        try:
            if request.role_uuid:
                require_management_role(request.role_uuid)
            return service.rebuild_vector_index(role_uuid=request.role_uuid, drop=request.drop)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/operations/memory/defaults", dependencies=[Depends(require_management_access)])
    async def validate_defaults():
        return service.validate_graph_defaults()

    @router.post("/operations/memory/export/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def export_memory(role_uuid: str, request: MemoryExportRequest):
        try:
            require_management_role(role_uuid)
            return service.export_role_memory(role_uuid, request.target)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/operations/memory/restore/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def restore_memory(role_uuid: str):
        try:
            require_management_role(role_uuid)
            return service.restore_role_memory(role_uuid)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/operations/restore", dependencies=[Depends(require_management_access)])
    async def restore_backup(request: RestoreRequest):
        try:
            require_management_role(request.role_uuid)
            return service.restore_backup(request.role_uuid, request.path)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/roles", dependencies=[Depends(require_management_access)])
    async def list_roles():
        return {"roles": roles.list_safe()}

    @router.post("/roles/connection-test", dependencies=[Depends(require_management_access)])
    async def role_connection_test(request: RoleConnectionTestRequest):
        config = {
            "provider": request.provider,
            "base_url": request.base_url,
            "model": request.model,
            "api_key": request.api_key,
        }
        try:
            result = roles.test_connection(config, probe)
        except ConnectionTestError as exc:
            raise HTTPException(status_code=400, detail=f"connection test failed: {exc.category}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"connection_test": result}

    @router.post("/roles", dependencies=[Depends(require_management_access)])
    async def create_role(request: RoleCreateRequest):
        try:
            config = roles.create(
                None,
                request.display_name,
                request.provider,
                request.model,
                request.api_key,
                prompt=_compiled_prompt(request.prompt, request.examples),
                runtime_options=request.runtime_options,
                avatar=request.avatar,
                base_url=request.base_url,
                connection_test=probe,
            )
        except ConnectionTestError as exc:
            raise HTTPException(status_code=400, detail=f"connection test failed: {exc.category}") from exc
        except (RoleLifecycleError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"role": roles.safe_config(config["role_uuid"])}

    @router.post("/role-imports", dependencies=[Depends(require_management_access)])
    async def start_role_import(
        request: Request,
        file: UploadFile = File(...), display_name: str = Form(...),
        source_role_name: str = Form(...), source_user_name: str = Form(...),
        user_appellation: str = Form(...), provider: str = Form("openai_compat"),
        base_url: Optional[str] = Form(None), model: str = Form(...), api_key: str = Form(...),
        avatar: Optional[str] = Form(None), runtime_temperature: Optional[str] = Form(None),
    ):
        if not (file.filename or "").lower().endswith(".txt"):
            _record_management_failure(request, 400, "only UTF-8 .txt files are supported")
            raise HTTPException(status_code=400, detail="only UTF-8 .txt files are supported")
        try:
            content = (await file.read()).decode("utf-8")
            spec = {
                "display_name": display_name, "source_role_name": source_role_name,
                "source_user_name": source_user_name, "user_appellation": user_appellation,
                "provider": provider, "base_url": base_url, "model": model,
                "api_key": api_key, "avatar": avatar,
                "runtime_options": _runtime_options_from_form(runtime_temperature),
            }
            return service.role_imports.start(spec, content)
        except UnicodeDecodeError as exc:
            _record_management_failure(request, 400, "source file must be UTF-8 text")
            raise HTTPException(status_code=400, detail="source file must be UTF-8 text") from exc
        except ValueError as exc:
            _record_management_failure(request, 400, str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/role-imports", dependencies=[Depends(require_management_access)])
    async def list_role_imports(role_uuid: Optional[str] = None):
        return {"jobs": service.role_imports.list_jobs(role_uuid)}

    @router.get("/role-imports/{job_id}", dependencies=[Depends(require_management_access)])
    async def role_import_status(job_id: str):
        try:
            return service.role_imports.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/role-imports/{job_id}/confirm", dependencies=[Depends(require_management_access)])
    async def confirm_role_import(request: Request, job_id: str, request_body: RoleImportConfirmRequest):
        try:
            prompt = _compiled_prompt(request_body.prompt, request_body.examples)
            return {"job": service.role_imports.confirm(job_id, prompt)}
        except KeyError as exc:
            _record_management_failure(request, 404, exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RoleLifecycleError as exc:
            # Activating an imported role can fail on its own configuration (no successful
            # connection test, prompt reference mismatch). That used to escape as a 500
            # with an opaque "Internal Server Error" body, which the wizard could only
            # show as "创建角色失败: Internal Server Error".
            _record_management_failure(request, 409, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            _record_management_failure(request, 409, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/role-imports/{job_id}/cancel", dependencies=[Depends(require_management_access)])
    async def cancel_role_import(request: Request, job_id: str):
        try:
            return {"job": service.role_imports.cancel(job_id)}
        except KeyError as exc:
            _record_management_failure(request, 404, exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            _record_management_failure(request, 409, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/roles/{role_uuid}/purge", dependencies=[Depends(require_management_access)])
    async def purge_role(request: Request, role_uuid: str, confirm: bool = False):
        """Delete one role's partition: graph objects, vectors, bookkeeping and files.

        An unconfirmed (``initializing``) partition is an import that never became a
        role, so it is purged freely; an **active** role needs ``?confirm=true``, because
        that deletes memory the user is living with.
        """
        registry = roles.authority.role_registry().get("roles", {})
        if role_uuid not in registry and not (roles.authority.characters_root / role_uuid).exists():
            _record_management_failure(request, 404, "role is not available")
            raise HTTPException(status_code=404, detail="role is not available")
        try:
            report = service.role_imports.purge_role(role_uuid, allow_active=bool(confirm))
        except ValueError as exc:
            _record_management_failure(request, 409, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"purge": report}

    @router.put("/roles/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def update_role(role_uuid: str, request: RoleCreateRequest):
        try:
            require_management_role(role_uuid)
            config = roles.update(
                role_uuid,
                request.display_name,
                request.provider,
                request.model,
                request.api_key,
                prompt=_compiled_prompt(request.prompt, request.examples),
                runtime_options=request.runtime_options,
                avatar=request.avatar,
                base_url=request.base_url,
                connection_test=probe,
            )
        except ConnectionTestError as exc:
            raise HTTPException(status_code=400, detail=f"connection test failed: {exc.category}") from exc
        except (RoleLifecycleError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"role": roles.safe_config(config["role_uuid"])}

    @router.get("/roles/{role_uuid}/runtime", dependencies=[Depends(require_management_access)])
    async def role_runtime(role_uuid: str):
        require_management_role(role_uuid)
        return service.role_runtime_config(role_uuid)

    @router.post("/roles/{role_uuid}/select", dependencies=[Depends(require_management_access)])
    async def select_role(role_uuid: str):
        try:
            return {"role": roles.select(role_uuid)}
        except RoleLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/ui/formation")
    async def ui_formation():
        return service.ui_formation()

    @router.get("/ui/session")
    async def ui_session(request: Request):
        host = request.client.host if request.client else None
        if host not in ManagementAccessError.LOOPBACK_HOSTS:
            raise HTTPException(status_code=401, detail="local WebUI only")
        return {"token": ManagementAccessError.issue(configured_management_token, ttl_seconds=3600)}

    @router.get("/ui/runtime", dependencies=[Depends(require_management_access)])
    async def ui_runtime():
        return service.runtime_status()

    @router.put("/ui/runtime", dependencies=[Depends(require_management_access)])
    async def update_ui_runtime(request: RuntimeConfigRequest):
        try:
            return service.update_runtime_config(request.config)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/ui/hot-zone", dependencies=[Depends(require_management_access)])
    async def hot_zone_status():
        return service.hot_zone_status()

    @router.post("/ui/hot-zone", dependencies=[Depends(require_management_access)])
    async def set_hot_zone(request: HotZoneRequest):
        return service.set_hot_zone_enabled(request.enabled)

    @router.post("/ui/hot-zone/rebuild", dependencies=[Depends(require_management_access)])
    async def hot_zone_rebuild(request: HotZoneRebuildRequest):
        require_management_role(request.role_uuid)
        return await asyncio.to_thread(service.rebuild_hot_zone, request.role_uuid)

    @router.get("/session-prompt")
    async def session_prompt(role_uuid: str, conversation_id: str):
        require_startable(role_uuid)
        try:
            return service.session_prompt(role_uuid, conversation_id)
        except Exception as exc:  # noqa: BLE001 - no session means "read the session response"
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/ui/activation", dependencies=[Depends(require_management_access)])
    async def activation_status(role_uuid: str):
        require_management_role(role_uuid)
        return service.activation_status(role_uuid)

    @router.post("/ui/activation/reset", dependencies=[Depends(require_management_access)])
    async def activation_reset(request: ActivationResetRequest):
        require_management_role(request.role_uuid)
        return service.reset_activation(request.role_uuid)

    @router.get("/ui/reflection", dependencies=[Depends(require_management_access)])
    async def reflection_status(role_uuid: str):
        require_management_role(role_uuid)
        return service.reflection_status(role_uuid)

    @router.post("/ui/reflection/run", dependencies=[Depends(require_management_access)])
    async def reflection_run(request: ReflectionRunRequest):
        require_management_role(request.role_uuid)
        return await asyncio.to_thread(
            service.run_reflection, request.role_uuid, force=request.force
        )

    @router.post("/ui/reflection/overlay/activate", dependencies=[Depends(require_management_access)])
    async def overlay_activate(request: OverlayActivateRequest):
        require_management_role(request.role_uuid)
        try:
            return service.activate_overlay(request.role_uuid, request.overlay_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="overlay not found") from exc

    @router.post("/ui/reflection/overlay/pause", dependencies=[Depends(require_management_access)])
    async def overlay_pause(request: OverlayPauseRequest):
        require_management_role(request.role_uuid)
        return service.set_overlay_paused(request.role_uuid, request.paused)

    @router.post("/ui/reflection/overlay/edit", dependencies=[Depends(require_management_access)])
    async def overlay_edit(request: OverlayEditRequest):
        require_management_role(request.role_uuid)
        return service.edit_overlay(request.role_uuid, request.items)

    @router.post("/ui/reflection/overlay/lock", dependencies=[Depends(require_management_access)])
    async def overlay_lock(request: OverlayLockRequest):
        require_management_role(request.role_uuid)
        try:
            return service.lock_overlay_item(request.role_uuid, request.item_id, request.locked)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="overlay item not found") from exc

    @router.get("/chatlog", dependencies=[Depends(require_management_access)])
    async def chatlog(
        role_uuid: str,
        conversation_id: Optional[str] = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        require_management_role(role_uuid)
        return service.chatlog_list(role_uuid, conversation_id, limit)

    @router.get("/ui/neo4j/status", dependencies=[Depends(require_management_access)])
    async def neo4j_status():
        return service.check_neo4j()

    @router.post("/ui/neo4j/connect", dependencies=[Depends(require_management_access)])
    async def neo4j_connect():
        return service.check_neo4j()

    @router.post("/ui/neo4j/start", dependencies=[Depends(require_management_access)])
    async def neo4j_start():
        return service.start_neo4j()

    @router.post("/ui/neo4j/stop", dependencies=[Depends(require_management_access)])
    async def neo4j_stop():
        return service.stop_neo4j()

    @router.get("/ui/topology", dependencies=[Depends(require_management_access)])
    async def ui_topology():
        return service.global_topology()

    @router.get("/graph", dependencies=[Depends(require_management_access)])
    async def graph_display(mode: str = "global", role_uuid: Optional[str] = None):
        try:
            return service.graph_display(mode, role_uuid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/graph/revision", dependencies=[Depends(require_management_access)])
    async def graph_revision(mode: str = "global", role_uuid: Optional[str] = None):
        """轻量图谱变更探针：仅返回内容指纹，供 WebUI 轮询判断是否重拉全量图谱。"""
        try:
            return service.graph_revision(mode, role_uuid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/extraction/queue", dependencies=[Depends(require_management_access)])
    async def extraction_queue():
        return service.extraction_queue_status()

    @router.get("/extraction/status/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def extraction_status(role_uuid: str):
        return service.role_extraction_status(role_uuid)

    @router.get("/extraction/pending", dependencies=[Depends(require_management_access)])
    async def extraction_pending():
        """已封批但尚未发布的批次（落盘记录；重启后由它重放，见 ADR-0147）。"""
        return service.pending_publications_status()

    # ------------------------------------------------------------------
    # State-change push channel (see graphiti_memory/state_changes.py).
    #
    # The management panel used to re-read whole payloads on a 3 s timer.
    # Instead it subscribes here once and re-reads a payload only when the
    # topic it displays actually transitioned: a conversation round was
    # recorded, the extraction queue moved, a session changed lifecycle state
    # (including the 1-minute temporary-graph checkpoint and the 1-hour formal
    # close that trigger extraction), or the graph changed.
    # ------------------------------------------------------------------

    @router.get("/events/state", dependencies=[Depends(require_management_access)])
    async def state_versions():
        """One-shot version snapshot: what a client diffs after a reconnect."""
        return service.state_changes.snapshot()

    @router.get("/events", dependencies=[Depends(require_management_access)])
    async def state_stream(request: Request, topics: Optional[str] = None):
        """Server-sent events carrying state transitions only.

        No polling and no payload duplication: each event is a few dozen bytes,
        and an idle memory service sends nothing at all (a keep-alive comment
        every 15 s keeps the connection observable and detects a dead client).
        ``topics`` optionally narrows the stream to a comma-separated subset.
        """
        hub = service.state_changes
        initial = hub.snapshot()
        available = set(initial["versions"].keys())
        wanted: Optional[set] = None
        if topics:
            wanted = {part.strip() for part in str(topics).split(",") if part.strip()}
            unknown = wanted - available
            if unknown:
                raise HTTPException(
                    status_code=400, detail=f"unknown state topic: {sorted(unknown)}"
                )

        async def stream():
            known = dict(initial["versions"])
            # Tell the client its baseline so it can decide whether what it has
            # already rendered is still current.
            yield _sse("ready", {
                "schema_version": initial["schema_version"],
                "versions": known,
                "topics": sorted(available),
            })
            while True:
                if await request.is_disconnected():
                    return
                # wait_for_change blocks, so run it off the event loop: the pet's
                # own /graphiti calls share this process.
                payload = await asyncio.to_thread(hub.wait_for_change, dict(known), 15.0)
                if payload is None:
                    yield ": keep-alive\n\n"
                    continue
                known.update(payload["versions"])
                if wanted is not None:
                    changed = {
                        topic: version
                        for topic, version in payload["changed"].items()
                        if topic in wanted
                    }
                    if not changed:
                        # A transition outside this subscriber's topics still
                        # advanced the baseline above; emit nothing for it.
                        continue
                    payload["changed"] = changed
                    payload["updates"] = {
                        topic: origin
                        for topic, origin in payload["updates"].items()
                        if topic in changed
                    }
                yield _sse("change", payload)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/ui/temporary-graphs", dependencies=[Depends(require_management_access)])
    async def ui_temporary_graphs():
        return service.temporary_graphs()

    @router.get("/ui/warm-layer", dependencies=[Depends(require_management_access)])
    async def ui_warm_layer(role_uuid: Optional[str] = None):
        """Warm Layer diagnostics: retained batches, bodies, vector/outbox state."""
        return service.warm_layer_status(role_uuid)

    @router.get(
        "/ui/warm-layer/{role_uuid}", dependencies=[Depends(require_management_access)]
    )
    async def ui_warm_layer_role(role_uuid: str):
        require_management_role(role_uuid)
        return service.warm_layer_snapshot(role_uuid)

    @router.post(
        "/ui/warm-layer/checkpoint", dependencies=[Depends(require_management_access)]
    )
    async def warm_layer_checkpoint(request: WarmBatchRequest):
        """提前触发选中批次的 7 天检查点（封存后送去抽取）。

        This is the manual form of the TTL sweep. An open batch is sealed first so
        a conversation is never split across two extractions, then the batch is
        frozen and published through the same path the TTL uses. The bodies are
        deleted only if that extraction confirms publication.
        """
        require_management_role(request.role_uuid)
        batch_ids = list(request.batch_ids or [])
        if not batch_ids:
            batch_ids = [
                str(batch.get("batch_id"))
                for batch in service.warm_store(request.role_uuid).batches()
                if batch.get("status") != "published"
            ]
        return service.trigger_warm_batch_extraction(
            request.role_uuid, batch_ids, reason=request.reason or "manual_checkpoint"
        )

    @router.post("/ui/warm-layer/seal", dependencies=[Depends(require_management_access)])
    async def warm_layer_seal(request: WarmBatchRequest):
        """只封存选中的对话批次（1 小时检查点的语义），不抽取、不删正文。"""
        require_management_role(request.role_uuid)
        batch_ids = list(request.batch_ids or [])
        if not batch_ids:
            batch_ids = [
                str(batch.get("batch_id"))
                for batch in service.warm_store(request.role_uuid).open_batches()
            ]
        return service.seal_warm_batch(
            request.role_uuid, batch_ids, reason=request.reason or "manual_seal"
        )

    @router.post("/ui/warm-layer/sweep", dependencies=[Depends(require_management_access)])
    async def warm_layer_sweep():
        """Run the composite warm sweeps once (outbox repair, vectors, TTL eviction)."""
        return service.sweep_background_work()

    @router.get("/roles/{role_uuid}/topology", dependencies=[Depends(require_management_access)])
    async def topology(role_uuid: str):
        require_management_role(role_uuid)
        return service.management_projection(role_uuid)["topology"]

    @router.get("/roles/{role_uuid}/memory", dependencies=[Depends(require_management_access)])
    async def memory(role_uuid: str):
        """Every memory element this role holds, with the object it lives on."""
        require_management_role(role_uuid)
        return {"memory": service.management_projection(role_uuid)["memory"]}

    @router.get("/roles/{role_uuid}/cognition", dependencies=[Depends(require_management_access)])
    async def cognition(role_uuid: str):
        """Compatibility alias for `/memory` (the atom store is retired)."""
        require_management_role(role_uuid)
        projection = service.management_projection(role_uuid)
        return {"memory": projection["memory"], "cognitions": projection["memory"]}

    @router.post("/roles/{role_uuid}/memory", dependencies=[Depends(require_management_access)])
    async def add_memory(role_uuid: str, request: MemoryElementRequest):
        try:
            require_management_role(role_uuid)
            return service.add_management_cognition(
                role_uuid, request.text, request.anchor_id, request.kind
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/roles/{role_uuid}/cognition", dependencies=[Depends(require_management_access)])
    async def add_cognition(role_uuid: str, request: MemoryElementRequest):
        return await add_memory(role_uuid, request)

    @router.get("/management/roles/{role_uuid}", dependencies=[Depends(require_management_access)])
    async def management_projection(role_uuid: str):
        require_management_role(role_uuid)
        return service.management_projection(role_uuid)

    @router.get("/management/roles/{role_uuid}/audit", dependencies=[Depends(require_management_access)])
    async def management_audit(role_uuid: str):
        require_management_role(role_uuid)
        return {"events": service.management_audit(role_uuid)}

    @router.put("/management/roles/{role_uuid}/prompt", dependencies=[Depends(require_management_access)])
    async def update_prompt(role_uuid: str, request: PromptUpdateRequest):
        try:
            require_management_role(role_uuid)
            return service.update_management_prompt(
                role_uuid,
                request.content,
                expected_version_id=request.expected_version_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.put("/management/roles/{role_uuid}/memory/{memory_id}", dependencies=[Depends(require_management_access)])
    async def update_memory(role_uuid: str, memory_id: str, request: MemoryElementUpdateRequest):
        try:
            require_management_role(role_uuid)
            return service.update_management_cognition(
                role_uuid,
                memory_id,
                request.text,
                anchor_id=request.anchor_id,
                kind=request.kind,
                reason=request.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put("/management/roles/{role_uuid}/cognitions/{cognition_id}", dependencies=[Depends(require_management_access)])
    async def update_cognition(role_uuid: str, cognition_id: str, request: MemoryElementUpdateRequest):
        return await update_memory(role_uuid, cognition_id, request)

    @router.post("/management/roles/{role_uuid}/memory/{memory_id}/invalidate", dependencies=[Depends(require_management_access)])
    async def invalidate_memory(role_uuid: str, memory_id: str, request: MemoryElementInvalidateRequest):
        try:
            require_management_role(role_uuid)
            return service.invalidate_management_cognition(role_uuid, memory_id, reason=request.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/management/roles/{role_uuid}/cognitions/{cognition_id}/invalidate", dependencies=[Depends(require_management_access)])
    async def invalidate_cognition(role_uuid: str, cognition_id: str, request: MemoryElementInvalidateRequest):
        return await invalidate_memory(role_uuid, cognition_id, request)

    # 补充世界知识（多份 txt，独立可撤销）。单独一个模块注册，改它不会动本文件的
    # 路由表；这里只有这一行是为它加的。
    create_knowledge_import_routes(router, service, root, configured_management_token)

    return router


def install_graphiti_routes(
    app,
    root: Path,
    management_token: Optional[str] = None,
    service: Optional[GraphitiMemoryService] = None,
) -> GraphitiMemoryService:
    router = create_router(root, management_token=management_token, service=service)
    app.include_router(router)
    service = router.graphiti_memory_service
    app.state.graphiti_memory_service = service
    # 上次进程遗留的待抽取批次：封批与抽取之间重启（关掉后端/整机）会让内存队列
    # 里的任务消失，必须在这里重放，否则那批对话永远不会被抽取（ADR-0147）。
    try:
        service.recover_pending_publications()
    except Exception:  # noqa: BLE001 - replay is best-effort at startup, never fatal
        pass
    # Warm Layer recovery (plan 阶段 3.4): warm outbox writes that never reached
    # the journal, batches frozen but not published, and batches whose extraction
    # failed are all re-driven before the service answers its first request.
    try:
        service.recover_warm_layer()
    except Exception:  # noqa: BLE001 - recovery is best-effort, never fatal
        pass
    # Startup backup is best-effort and intentionally does not delay service
    # readiness or role selection.
    service.startup_backups()
    return service
