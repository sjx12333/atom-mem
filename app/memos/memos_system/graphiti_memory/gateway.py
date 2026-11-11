"""MemOS-compatible gateway with role-scoped long-term extensions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import node_memory
from .authority import AuthorityStore
from .domain import normalized_text, score_text, typed_id, utc_timestamp
from .retrieval import RetrievalPolicy, apply_budget
from .node_recall import NodeMemoryRecall
from .topology import SharedTopology
from .temporary_graph import TemporaryGraph
from .vector_store import VectorIndexError
from .coordinator import ConversationSessionCoordinator


#: A working-memory record is one chat turn, so anything past this is not a turn.
#: Production had a single 825,207-character record — a base64 screenshot that
#: ``str(message["content"])`` had stringified. Such a payload is dropped (not
#: truncated): keeping the first 20 KB of a base64 blob in short-term memory helps
#: nothing and still costs a full-file rewrite on the old JSON store.
WORKING_MEMORY_MAX_CHARS = 20_000

#: Ranking bonus for memories about the contact this turn is talking to (QQ 等)。
#: 排序偏好，不参与准入门槛；上限由 1.0 封顶（见 ``_apply_focus_boost``）。
FOCUS_CONTACT_BOOST = 0.25

#: 联想加成（hook）的强度。当某个实体检索到的**联想节点**与本次查询高度相似时，给它
#: 一个很小的排序加分。刻意取小值（对比 focus 的 0.25）：
#:
#: * 它衡量的是"模型预测的关联"而不是"对话说过的话"，因此只该在两条候选**本来就很接近**
#:   时决定谁先被注入——实测重排分在 0.30–0.50 的窄带里漂移，0.08 足以改变顺序；
#: * 它与 ``FOCUS_CONTACT_BOOST`` 叠加后由 1.0 封顶，不会把分数推出门槛标定的量纲。
HOOK_ASSOCIATION_BOOST = 0.08

#: 一个实体的联想要有资格加分，其最强的 hook 余弦至少要到这里。低于它说明这次查询和
#: 该联想无关，此时**不加分**（而不是给 0 分惩罚：hook 没命中不等于实体不相关）。
HOOK_ASSOCIATION_FLOOR = 0.35


def _source_counts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Per-source tally of the injected set, for the retrieval audit.

    Recall now has four sources (working memory, graph atoms, the chat-log
    corpus, the session's temporary graph). Without this tally a zero-recall
    report cannot say *which* of them was empty, and "the transcript had nothing"
    reads identically to "the graph had nothing".
    """
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item.get("layer") or item.get("source_type") or item.get("source") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def working_memory_text(content: Any) -> str:
    """Text-only view of a message body.

    ``str(content)`` on a multimodal body stringifies the whole parts list —
    base64 image payload included — which is exactly how an 825 KB screenshot
    became a working-memory record. Only ``type == "text"`` parts are text;
    an opaque body yields ``""`` and the caller drops it.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        # Production's 825 KB record is not a list but the *stringified* parts
        # array (``[{"type": "text", ...}, {"type": "image_url", ...}]``), so a
        # plain string check would keep the base64. Only re-parse when the text
        # actually looks like a content-parts payload.
        if stripped[:1] in "[{" and '"type"' in stripped:
            try:
                parsed = json.loads(stripped)
            except ValueError:
                return stripped
            if isinstance(parsed, (list, dict)):
                return working_memory_text(parsed)
        return stripped
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and str(part.get("type") or "") == "text":
                pieces.append(str(part.get("text") or ""))
        return " ".join(piece for piece in pieces if piece).strip()
    if isinstance(content, dict):
        if str(content.get("type") or "") == "text":
            return str(content.get("text") or "").strip()
        if "text" in content:
            return str(content["text"]).strip()
        return ""
    return str(content).strip()


def prepare_working_memory_content(content: Any, record_id: Any = None) -> tuple[str, Dict[str, Any] | None]:
    """Return ``(text, skip_reason)`` for one message body.

    Shared by the live write path and by the JSON import so a legacy file cannot
    smuggle an oversized or opaque record into SQLite past the same rules.
    """
    text = working_memory_text(content)
    if not text:
        return "", None
    if len(text) > WORKING_MEMORY_MAX_CHARS:
        # 丢弃而不是截断：被截断的 blob 留在短期记忆里毫无用处，
        # 但会一直占着每一轮的读写成本。
        return "", {"reason": "oversized", "length": len(text), "message_id": record_id}
    return text, None


class WorkingMemoryStore:
    """Live MemOS-compatible working memory, SQLite-backed when available.

    SQLite is the online store (ADR-0148). The previous JSON store read and
    rewrote the *entire* file on every add and re-parsed it on every search, so
    cost grew with total history: production's file reached 899 KB / 142 records
    and every single turn paid for all of it.
    """

    def __init__(self, root: Path, store: Any = None):
        self.path = root / "working_memory.json"
        self.store = store
        if store is None:
            # Only the JSON fallback needs the file's directory to exist.
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    def _write(self, records: List[Dict[str, Any]]) -> None:
        descriptor, temp_name = tempfile.mkstemp(prefix=".working.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def migrate_json_into_sqlite(self) -> Dict[str, Any]:
        """Import the legacy JSON file into SQLite, then retire the file.

        Renamed rather than deleted: the file is the only copy of short-term
        history until this import succeeds, and the rename is what makes the
        migration idempotent — a second run finds no file and skips. Re-running
        against a file that reappeared is still safe, because inserts de-duplicate
        on the same identity key the JSON store used.
        """
        if self.store is None or not self.path.exists():
            return {"status": "skipped", "reason": "no_json_file"}
        try:
            records = self._read()
        except (OSError, ValueError) as exc:
            return {"status": "failed", "reason": "unreadable", "error": str(exc)}
        # 走与在线写入完全相同的正文规则：旧文件里那条 825 KB 的 base64 记录
        # 不能因为"来自迁移"就被原样搬进 SQLite。
        sanitized: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            text, skip = prepare_working_memory_content(record.get("content"), record.get("id"))
            if skip is not None:
                skipped.append(skip)
                continue
            if not text:
                continue
            item = dict(record)
            item["content"] = text
            sanitized.append(item)
        added = self.store.save_working_memory(sanitized)
        backup = self.path.parent / f"{self.path.name}.migrated.bak"
        try:
            os.replace(self.path, backup)
        except OSError as exc:  # pragma: no cover - filesystem guard
            return {
                "status": "partial",
                "reason": "retire_failed",
                "error": str(exc),
                "imported": len(added),
                "records": len(records),
            }
        report: Dict[str, Any] = {
            "status": "migrated",
            "records": len(records),
            "imported": len(added),
            "backup": str(backup),
        }
        if skipped:
            report["skipped"] = skipped
        return report

    def add(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.store is not None:
            return self.store.save_working_memory(list(records))
        current = self._read()
        existing_keys = {
            (record.get("role_uuid"), record.get("user_id"), record.get("conversation_id"), record.get("id"))
            for record in current
            if record.get("id")
        }
        added = []
        for record in records:
            item = dict(record)
            item.setdefault("id", typed_id("memory"))
            item.setdefault("created_at", utc_timestamp())
            item.setdefault("layer", "WorkingMemory")
            identity = (item.get("role_uuid"), item.get("user_id"), item.get("conversation_id"), item["id"])
            if identity in existing_keys:
                continue
            current.append(item)
            existing_keys.add(identity)
            added.append(item)
        self._write(current)
        return added

    def search(self, role_uuid: str, user_id: str, conversation_id: Optional[str], query: str) -> List[Dict[str, Any]]:
        records = (
            self.store.list_working_memory(role_uuid, user_id, conversation_id)
            if self.store is not None
            else self._read()
        )
        results = []
        for record in records:
            if record.get("role_uuid") != role_uuid or record.get("user_id") != user_id:
                continue
            if conversation_id and record.get("conversation_id") != conversation_id:
                continue
            score = score_text(query, record.get("content", ""))
            if score > 0:
                results.append((score, record))
        results.sort(key=lambda item: (-item[0], item[1].get("created_at", ""), item[1].get("id", "")))
        return [{**record, "layer": "WorkingMemory", "similarity": round(score, 6), "final_score": round(score, 6)} for score, record in results]


class GraphitiMemoryGateway:
    """Stable boundary for existing MemOS callers and new Graphiti services."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        policy: Optional[RetrievalPolicy | Dict[str, Any]] = None,
        embedder=None,
        reranker=None,
        memory_layer: Any = None,
        activation: Any = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = AuthorityStore(self.root)
        # Working memory lives in SQLite whenever a memory layer is attached; the
        # JSON file is only the fallback for stores built without one. The legacy
        # file is imported once here and retired (ADR-0148).
        self.working = WorkingMemoryStore(
            self.root, store=getattr(memory_layer, "store", None) if memory_layer is not None else None
        )
        self.working_migration = (
            self.working.migrate_json_into_sqlite() if self.working.store is not None else {"status": "skipped"}
        )
        self.topology = SharedTopology(self.root / "shared" / "topology.json")
        self.memory_layer = memory_layer
        # One recall implementation: a candidate is a whitelisted node, relation or
        # memory event, a hit returns that object's complete memory, and there is no
        # second store that could answer differently.
        self.node_recall = NodeMemoryRecall(
            self.topology,
            memory_layer=memory_layer,
            policy=policy,
            embedder=embedder,
            reranker=reranker,
            activation=activation,
        )

    def add_working_memory(
        self,
        role_uuid: str,
        user_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        records = []
        skipped: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            # 只取文本部分：多模态消息直接 str() 会把 base64 截图整段变成
            # 工作记忆（生产库里 142 条里有 1 条 825,207 字符就是这么来的）。
            content, skip = prepare_working_memory_content(
                message.get("content"), message.get("id") or message.get("message_id")
            )
            if skip is not None:
                skipped.append(skip)
                continue
            if not content:
                continue
            records.append(
                {
                    "id": message.get("id") or message.get("message_id") or typed_id("memory"),
                    "role_uuid": role_uuid,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "role": message.get("role", "user"),
                    "content": content,
                    "source": "memos_working_memory",
                    "created_at": message.get("created_at", utc_timestamp()),
                    "layer": "WorkingMemory",
                }
            )
        added = self.working.add(records) if records else []
        result: Dict[str, Any] = {"status": "success", "added": len(added), "memories": added}
        if skipped:
            result["skipped"] = skipped
        return result

    def add_long_term_memory(self, role_uuid: str, content: str, **metadata: Any) -> Dict[str, Any]:
        return self.authority.save_memory(role_uuid, content, **metadata)

    def search_working_memory(
        self,
        role_uuid: str,
        user_id: str,
        conversation_id: Optional[str],
        query: str,
        top_k: Optional[int] = 5,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Search the live MemOS-compatible layer without requiring Graphiti."""
        results = self.working.search(role_uuid, user_id, conversation_id, query)
        results = [
            item for item in results if item.get("similarity", 0.0) >= similarity_threshold
        ]
        if top_k is not None:
            results = results[: max(0, int(top_k))]
        return {
            "query": query,
            "memories": results,
            "count": len(results),
            "token_estimate": sum(
                int(item.get("token_estimate", len(str(item.get("content", "")))))
                for item in results
            ),
            "policy_version": self.node_recall.policy.version,
            "retrieval_diagnostics": {"source": "WorkingMemory"},
        }

    def search(
        self,
        role_uuid: str,
        user_id: str,
        conversation_id: Optional[str],
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        token_budget: Optional[int] = None,
        warm_store: Any = None,
        current_batch_id: Optional[str] = None,
        current_round_no: Optional[int] = None,
        include_assistant: bool = False,
        focus_object_ids: Optional[Sequence[str]] = None,
        chatlog: Any = None,
        exclude_ids: Optional[Sequence[str]] = None,
        exclude_batch_rounds: Optional[Sequence[Any]] = None,
        exclude_hashes: Optional[Sequence[str]] = None,
        exclude_top_hashes: Optional[int] = 0,
    ) -> Dict[str, Any]:
        results = self.working.search(role_uuid, user_id, conversation_id, query)
        # Candidates are the role's whitelisted nodes/relations plus its memory
        # events, and a hit returns that object's complete memory. There is exactly
        # one implementation: a second read path could only ever disagree with it.
        policy = self.node_recall.policy
        # One query, one embedding. The graph pass and the Warm Layer would
        # otherwise embed the same text independently; computing it here and
        # handing the vector to both keeps the recall identical while halving the
        # per-turn embedding calls.
        query_vector: Optional[List[float]] = None
        if (
            str(query or "").strip()
            and int(top_k) > 0
            and self.node_recall._embeddings_configured()
        ):
            query_vector = self.node_recall._query_vector(query)
        graphiti = self.node_recall.search(
            role_uuid,
            query,
            top_k=max(top_k, policy.direct_top_k),
            token_budget=token_budget,
            similarity_threshold=similarity_threshold,
            query_vector=query_vector,
        )
        if graphiti.get("index_failure"):
            # A broken vector index is a recall component failure: the caller must
            # abort the request instead of silently generating without long-term
            # memory (ADR-0032 / CONTEXT 召回组件错误).
            raise VectorIndexError(
                str(graphiti.get("diagnostics", {}).get("error") or "vector index failure")
            )
        results.extend({**item, "layer": "GraphitiMemory"} for item in graphiti["memories"])
        # 温层是本轮召回的**近期正文**来源（role 级、7 天 TTL）。它有自己的
        # 余弦/词法地板（``warm_similarity_threshold``）和层内相对地板
        # （``warm_relative_floor_ratio``），不要求实时精排，也不允许靠"没有
        # rerank_score"绕过任何门槛。
        warm_result: Dict[str, Any] = {"memories": [], "diagnostics": {"source": "WarmLayer", "status": "disabled"}}
        if warm_store is not None:
            gate = float(getattr(policy, "warm_similarity_threshold", 0.35))
            ratio = float(getattr(policy, "warm_relative_floor_ratio", 0.55))
            warm_diagnostics: Dict[str, Any] = {}
            try:
                warm_memories = warm_store.search(
                    query,
                    top_k=max(top_k, int(getattr(policy, "warm_top_k", 8))),
                    gate=gate,
                    relative_floor_ratio=ratio,
                    current_batch_id=current_batch_id,
                    current_round_no=current_round_no,
                    include_assistant=bool(include_assistant),
                    diagnostics=warm_diagnostics,
                    query_vector=query_vector,
                    raw_rag=bool(getattr(policy, "warm_rag_enabled", False)),
                    rerank=True,
                    chunk_tokens=int(getattr(policy, "warm_rag_chunk_tokens", 256)),
                    chunk_overlap=int(getattr(policy, "warm_rag_chunk_overlap", 32)),
                    dense_top_k=int(getattr(policy, "warm_rag_dense_top_k", 128)),
                    rerank_top_k=int(getattr(policy, "warm_rag_rerank_top_k", 15)),
                )
                warm_result = {
                    "memories": warm_memories,
                    "diagnostics": {
                        "source": "WarmLayer",
                        "status": "ok",
                        "threshold": gate,
                        "admitted": len(warm_memories),
                        "current_batch_id": current_batch_id,
                        "current_round_no": current_round_no,
                        "include_assistant": bool(include_assistant),
                        **warm_diagnostics,
                    },
                }
            except Exception as exc:  # noqa: BLE001 - a transcript must never kill a turn
                warm_result = {
                    "memories": [],
                    "diagnostics": {"source": "WarmLayer", "status": "failed", "error": str(exc)},
                }
            results.extend(warm_result["memories"])
        # 聊天记录继续保留 WebUI 展示与回合审计的写入路径，但**不再**作为召回来源
        # （阶段 6.1：温层覆盖不变量通过后退役 ChatLog 召回）。旧的 ``chatlog_*``
        # policy 字段标记 deprecated，暂不删除。
        chatlog_result: Dict[str, Any] = {
            "memories": [],
            "diagnostics": {"source": "ChatLog", "status": "retired"},
        }
        unique: Dict[Any, Dict[str, Any]] = {}
        for item in results:
            unique.setdefault(item.get("id"), item)
        results = list(unique.values())
        # 各层先**独立准入**，再统一排序与 token budget。
        #
        # 旧实现里 ``item.get("rerank_score") is None`` 会整体豁免门槛，等价于
        # "缺分数 ⇒ 自动通过"：图谱候选一旦拿不到标定分数就绕过长期层门槛进入提示词。
        # 现在长期层必须有可用的标定分数才能通过；温层用自己已经算出的
        # ``similarity``（余弦或词法回退）作为它的地板，两者都不是空值豁免。
        effective_gate = max(float(similarity_threshold), float(policy.direct_rerank_threshold))
        admitted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for item in results:
            layer = str(item.get("layer") or "")
            if layer == "WarmLayer":
                score = item.get("similarity")
                source = str(item.get("score_source") or "warm_similarity")
                ok = score is not None and float(score) >= float(
                    getattr(policy, "warm_similarity_threshold", 0.35)
                )
            elif layer == "WorkingMemory" or str(item.get("source") or "") == "memos_working_memory":
                # Working memory is the live short-term store, not a graph atom: it
                # has its own lexical score, so it is admitted by that score rather
                # than by the graph's calibrated rerank gate it can never satisfy.
                # It is the same "recent conversation" class as the Warm Layer, so
                # it shares that layer's (tightened) floor.
                score = item.get("similarity")
                source = "working_memory_lexical"
                ok = score is not None and float(score) >= float(
                    getattr(policy, "warm_similarity_threshold", 0.35)
                )
            elif str(item.get("source_type") or "") == "chatlog" or layer == "ChatLog":
                score = item.get("similarity")
                source = "chatlog_recall"
                ok = score is not None and float(score) >= float(
                    getattr(policy, "chatlog_semantic_threshold", 0.15)
                )
            else:
                score = item.get("rerank_score")
                source = str(item.get("score_source") or "graph_rerank")
                ok = score is not None and float(score) >= effective_gate
            marked = {**item, "score_source": source, "gate_score": score}
            (admitted if ok else rejected).append(marked)
        results = admitted
        # 与当前聊天对象有关的记忆往前放。刻意放在门槛之后、排序之前：
        # 它是**排序偏好**，不参与准入判定，也不会抬高相对首名地板把别的记忆挤掉。
        if focus_object_ids:
            results = self._apply_focus_boost(results, focus_object_ids)
        # 联想加成（hook）：同样放在全部准入判定之后。一个实体检索到的联想节点与本次查询
        # 越像，它越容易被想起——但**绝不放行**低于门槛的记忆，也不改任何准入门槛。
        hook_boost = self._apply_hook_boost(results)
        results = hook_boost["items"]
        # 两路独立准入之后再融合：RRF 只看名次，不把温层的余弦和图谱的 rerank
        # 混进同一个分数尺度，也不会放行任何低于各自门槛的记忆。查询没有点名
        # 实体时，正文/温层权重更高；点名实体时，图谱事实权重更高。
        results, fusion_diagnostics = self._apply_layer_fusion(
            results, role_uuid=role_uuid, query=query, policy=policy
        )
        results.sort(
            key=lambda item: (
                -float(item.get("fusion_score", item.get("final_score", 0.0))),
                item.get("created_at", ""),
                item.get("id", ""),
            )
        )
        # 工具退避（阶段 5.2）：先按精确 id 和 (batch_id, round_no) 删，再按内容哈希
        # 和高相似度删。省略条数与原因必须显式报告，否则模型会把"空结果"读成
        # "没有记忆"。
        suppressed = self._suppress_duplicates(
            results,
            exclude_ids=exclude_ids,
            exclude_batch_rounds=exclude_batch_rounds,
            exclude_hashes=exclude_hashes,
            exclude_top_hashes=int(exclude_top_hashes or 0),
        )
        results = suppressed["kept"]
        budget = policy.token_budget if token_budget is None else max(0, int(token_budget))
        selected = apply_budget(results, top_k, budget)
        selected = self._drop_duplicate_content(selected)
        used_tokens = sum(
            int(item.get("token_estimate", len(str(item.get("content", "")))))
            for item in selected
        )
        diagnostics = dict(graphiti.get("diagnostics", {}))
        diagnostics["warm_layer"] = warm_result.get("diagnostics", {})
        diagnostics["chatlog"] = chatlog_result.get("diagnostics", {})
        diagnostics["source_counts"] = _source_counts(selected)
        diagnostics["gate"] = {
            "effective_gate": effective_gate,
            "warm_threshold": float(getattr(policy, "warm_similarity_threshold", 0.35)),
            "warm_relative_floor_ratio": float(
                getattr(policy, "warm_relative_floor_ratio", 0.55)
            ),
            "rejected_unscored_or_below_gate": len(rejected),
            "rejected_ids": [item.get("id") for item in rejected][:20],
        }
        diagnostics["suppressed"] = suppressed["diagnostics"]
        diagnostics["hook_boost"] = hook_boost["diagnostics"]
        diagnostics["fusion"] = fusion_diagnostics
        return {
            "query": query,
            "memories": selected,
            "count": len(selected),
            "token_estimate": used_tokens,
            "policy_version": self.node_recall.policy.version,
            "retrieval_diagnostics": diagnostics,
        }

    def _query_mentions_entity(self, role_uuid: str, query: str) -> bool:
        """True when the query names a whitelisted entity (or a two-char alias).

        This is deliberately a cheap lexical check, not entity linking: it only
        decides which recall *channel* gets more RRF weight. A wrong "mentioned"
        answer can change ordering, never admission, so a miss is bounded.
        """
        text = normalized_text(query)
        if not text:
            return False
        for record in self.topology.memory_records_for_role(role_uuid):
            if node_memory.object_type_of(record) != "node":
                continue
            name = normalized_text(record.get("name"))
            if len(name) >= 2 and name in text:
                return True
        return False

    @staticmethod
    def _layer_rank_score(item: Dict[str, Any]) -> float:
        for key in ("final_score", "similarity", "rerank_score", "gate_score"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _apply_layer_fusion(
        self,
        items: List[Dict[str, Any]],
        *,
        role_uuid: str,
        query: str,
        policy: RetrievalPolicy,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fuse already-admitted layers by rank, not by raw score.

        The graph layer's qwen3-rerank score and the Warm Layer's embedding cosine
        live on different scales; adding or comparing them directly is meaningless.
        Reciprocal Rank Fusion only needs each layer's ordering, so it can prefer a
        transcript turn for an associative query without weakening the graph gate.

        Admission happens before this method. RRF can reorder accepted memories; it
        cannot admit a rejected one.
        """
        enabled = bool(getattr(policy, "rrf_fusion_enabled", False))
        routing = bool(getattr(policy, "query_routing_enabled", False))
        diagnostics: Dict[str, Any] = {
            "enabled": enabled,
            "routing_enabled": routing,
            "status": "disabled" if not enabled else "ok",
            "k": max(1, int(getattr(policy, "rrf_k", 60))),
            "entity_mentioned": None,
            "weights": {},
            "layers": {},
        }
        if not enabled or len(items) < 2:
            diagnostics["status"] = "single_layer" if items else "empty"
            return items, diagnostics

        mentioned = self._query_mentions_entity(role_uuid, query) if routing else False
        if mentioned:
            weights = {"GraphitiMemory": 1.0, "WarmLayer": 0.55, "WorkingMemory": 0.45}
        else:
            weights = {"GraphitiMemory": 0.65, "WarmLayer": 1.0, "WorkingMemory": 0.55}
        diagnostics["entity_mentioned"] = mentioned
        diagnostics["weights"] = weights

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            layer = str(item.get("layer") or item.get("source_type") or "unknown")
            groups.setdefault(layer, []).append(item)
        diagnostics["layers"] = {key: len(value) for key, value in groups.items()}
        if len(groups) < 2:
            diagnostics["status"] = "single_layer"
            return items, diagnostics

        k = diagnostics["k"]
        fused: Dict[str, float] = {}
        for layer, group in groups.items():
            group.sort(
                key=lambda item: (
                    -self._layer_rank_score(item),
                    item.get("created_at", ""),
                    item.get("id", ""),
                )
            )
            weight = float(weights.get(layer, 0.5))
            for rank, item in enumerate(group, start=1):
                key = str(item.get("id") or id(item))
                fused[key] = fused.get(key, 0.0) + weight / (k + rank)
        for item in items:
            key = str(item.get("id") or id(item))
            item["fusion_score"] = round(fused.get(key, 0.0), 8)
        diagnostics["status"] = "fused"
        return (
            sorted(
                items,
                key=lambda item: (
                    -float(item.get("fusion_score", 0.0)),
                    -self._layer_rank_score(item),
                    item.get("id", ""),
                ),
            ),
            diagnostics,
        )

    @staticmethod
    def _suppress_duplicates(
        items: List[Dict[str, Any]],
        *,
        exclude_ids: Optional[Sequence[str]] = None,
        exclude_batch_rounds: Optional[Sequence[Any]] = None,
        exclude_hashes: Optional[Sequence[str]] = None,
        exclude_top_hashes: int = 0,
    ) -> Dict[str, Any]:
        """Drop what the caller already has (a tool result must never repeat it).

        Three passes, in the plan's order: exact ids, then ``(batch_id,
        round_no)``, then content hashes. ``exclude_top_hashes`` additionally
        suppresses the strongest N hashes so a tool that only knows the *text* it
        was shown can abstain without echoing a hash.
        """
        ids = {str(item) for item in (exclude_ids or []) if item}
        pairs = {
            (str(item[0]), int(item[1]))
            for item in (exclude_batch_rounds or [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        hashes = {str(item) for item in (exclude_hashes or []) if item}
        if exclude_top_hashes > 0:
            ranked = sorted(
                items, key=lambda row: -float(row.get("final_score") or 0.0)
            )[:exclude_top_hashes]
            hashes.update(
                str(row.get("content_hash")) for row in ranked if row.get("content_hash")
            )
        omitted: Dict[str, int] = {"id": 0, "batch_round": 0, "hash": 0}
        kept: List[Dict[str, Any]] = []
        for item in items:
            if str(item.get("id")) in ids:
                omitted["id"] += 1
                continue
            pair = (str(item.get("batch_id")), int(item.get("round_no") or 0))
            if item.get("batch_id") is not None and pair in pairs:
                omitted["batch_round"] += 1
                continue
            digest = str(item.get("content_hash") or "")
            if digest and digest in hashes:
                omitted["hash"] += 1
                continue
            kept.append(item)
        total = sum(omitted.values())
        diagnostics = {
            "omitted": total,
            "omitted_by": omitted,
            "message": (
                f"有 {total} 条与系统自动提供的片段重复，已省略"
                if total
                else "无重复省略"
            ),
        }
        return {"kept": kept, "diagnostics": diagnostics}

    def graph_query(
        self,
        role_uuid: str,
        query: str,
        *,
        operator: Optional[str] = None,
        anchors: Optional[Sequence[str]] = None,
        predicate: str = "",
        target: str = "",
        top_k: int = 12,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Execute one deterministic graph-query operator for the tool.

        This is deliberately not the recall path: it does not embed, rerank or
        admit anything into the automatic context. The model asks a structural
        question (two anchors, a shared predicate, a path, a count) and gets
        whitelist-filtered evidence back. The normal search remains the fallback.
        """
        from .graph_query import GraphQueryEngine

        return GraphQueryEngine(self.topology, role_uuid).execute(
            query=query,
            operator=operator,
            anchors=anchors,
            predicate=predicate,
            target=target,
            top_k=top_k,
            max_depth=max_depth,
        )

    def search_entities(
        self,
        role_uuid: str,
        query: str,
        *,
        top_k: int = 5,
        token_budget: Optional[int] = None,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Entity-only recall: whitelisted nodes and relations, no events or facts.

        Used by the "a temporary-graph turn was recalled" follow-up (ADR-0164):
        those turns name entities the graph pass never saw, and what recovers them
        is the entity objects themselves, not another round of transcript.
        """
        policy = self.node_recall.policy
        # Entity-only recall deliberately stays on the dossier path: its caller
        # (ADR-0164 temp-graph follow-up) needs the entity objects themselves, not
        # the fact-first answer units. The policy may enable fact-first globally.
        recall = (
            self.node_recall._search_entity_first
            if bool(getattr(policy, "fact_first_enabled", False))
            else self.node_recall.search
        )
        result = recall(
            role_uuid,
            query,
            top_k=max(top_k, int(getattr(policy, "direct_top_k", 20))),
            token_budget=token_budget,
            similarity_threshold=similarity_threshold,
        )
        if result.get("index_failure"):
            raise VectorIndexError(
                str(result.get("diagnostics", {}).get("error") or "vector index failure")
            )
        entities = [
            item
            for item in result.get("memories", [])
            if str(item.get("object_type") or "") in {"node", "relation"}
        ]
        budget = policy.token_budget if token_budget is None else max(0, int(token_budget))
        selected = apply_budget(entities, top_k, budget)
        diagnostics = dict(result.get("diagnostics", {}))
        diagnostics["entity_only"] = True
        return {
            "memories": selected,
            "count": len(selected),
            "token_estimate": sum(
                int(item.get("token_estimate", len(str(item.get("content", "")))))
                for item in selected
            ),
            "retrieval_diagnostics": diagnostics,
        }

    def _apply_focus_boost(
        self, items: List[Dict[str, Any]], focus_object_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Give memories about the current contact a bounded ranking bonus.

        Requirement: when the pet is talking to someone on QQ, memories related to
        *that* contact should score higher. The bonus is deliberately applied after
        the admission gates so it can never admit a below-threshold memory or move
        the relative floor for the others; it only decides order (and therefore who
        gets the limited Top-K slots). Score is capped at 1.0 so the scale stays the
        one the gates were calibrated against, and each boosted row is marked for
        diagnostics.
        """
        boosted: List[Dict[str, Any]] = []
        for item in items:
            if not self.node_recall.touches_focus(item, focus_object_ids):
                boosted.append(item)
                continue
            current = float(item.get("final_score") or 0.0)
            marked = {**item, "focus_boost": True}
            marked["final_score"] = round(min(1.0, current + FOCUS_CONTACT_BOOST), 6)
            boosted.append(marked)
        return boosted

    @staticmethod
    def _apply_hook_boost(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """联想加成：命中实体的联想节点越像本次查询，这条记忆排得越前。

        位置和 ``_apply_focus_boost`` 一样，刻意放在**所有准入判定之后、排序之前**：

        * 它**不能**放行任何东西。``items`` 传进来时已经过完图谱重排门槛、相对地板、
          层内地板与层间门槛，所以这里加多少分都不会让一条本该被砍掉的记忆进提示词。
          hook 的**准入**只发生在 ``node_recall._nominate_hook_parents`` 的逐字命中
          路径上，且提名之后仍要过同一道重排门槛；
        * 它**不能**抬高相对首名地板。相对地板在 ``node_recall._gate`` 里、比这里更早，
          因此加分不会把别的记忆挤掉——这正是"排序偏好"与"准入条件"的分界。

        ``hook_semantic`` 由 ``node_recall._attach_hook_resemblance`` 写入：它是该实体所有
        联想节点里与查询最相似的那一个的**余弦**。低于 ``HOOK_ASSOCIATION_FLOOR`` 时
        **不加分也不扣分**——联想没命中不等于这个实体不相关。

        每个被加分的条目都打上 ``hook_boost: True`` 并在 diagnostics 里留 id，所以
        "这次排序是不是联想造成的"永远可查。
        """
        boosted: List[Dict[str, Any]] = []
        marked: List[Dict[str, Any]] = []
        for item in items:
            raw = item.get("hook_semantic")
            if raw is None:
                boosted.append(item)
                continue
            try:
                semantic = float(raw)
            except (TypeError, ValueError):
                boosted.append(item)
                continue
            if semantic < HOOK_ASSOCIATION_FLOOR:
                boosted.append(item)
                continue
            current = float(item.get("final_score") or 0.0)
            updated = {**item, "hook_boost": True, "hook_semantic": round(semantic, 6)}
            updated["final_score"] = round(
                min(1.0, current + HOOK_ASSOCIATION_BOOST), 6
            )
            boosted.append(updated)
            marked.append(
                {
                    "id": item.get("id"),
                    "hook_semantic": round(semantic, 6),
                    "final_score_before": round(current, 6),
                    "final_score_after": updated["final_score"],
                }
            )
        return {
            "items": boosted,
            "diagnostics": {
                "boost": HOOK_ASSOCIATION_BOOST,
                "floor": HOOK_ASSOCIATION_FLOOR,
                "boosted": marked,
                "boosted_count": len(marked),
            },
        }

    @staticmethod
    def _drop_duplicate_content(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the best-scoring copy of each distinct sentence across sources."""
        best: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for item in items:
            key = str(item.get("content_hash") or "")
            if not key:
                key = f"id:{item.get('id')}"
            existing = best.get(key)
            if existing is None:
                best[key] = item
                order.append(key)
                continue
            if float(item.get("final_score") or 0.0) > float(existing.get("final_score") or 0.0):
                best[key] = item
        return [best[key] for key in order]

    def merge_core_memories(
        self,
        gated: List[Dict[str, Any]],
        core: List[Dict[str, Any]],
        *,
        top_k: int,
        token_budget: Optional[int] = None,
        admit_unscored: bool = False,
    ) -> Dict[str, Any]:
        """Fold unscored Graphiti Core results into the already-gated set.

        Core results carry no calibrated score, so **they are not injectable**: a
        memory that cannot be scored cannot pass the semantic gate, the rerank gate or
        the relative floor, and the three gates exist precisely because content that
        skipped them reached the prompt. They are reported in diagnostics
        (``dropped_unscored``) instead.

        The previous behaviour admitted them last, "as supplementary text, while the
        budget has room". Measured end-to-end, that is how the **raw conversation
        transcript** (the Core search returns the episode) and bare entity *names* got
        injected: with three memories passing the gates and ``top_k = 5``, the two free
        slots were filled by text the gates had just rejected — the opposite of
        "不足条数就少给，零条也是正常结果".

        ``admit_unscored=True`` restores the old padding for an operator who wants to
        measure it; nothing calls it that way by default.
        """
        policy = self.node_recall.policy
        scored = [item for item in gated]
        unscored = [
            item
            for item in core
            if item.get("content") and str(item.get("content")).strip()
        ]
        if not admit_unscored:
            budget = policy.token_budget if token_budget is None else max(0, int(token_budget))
            selected = apply_budget(list(scored), top_k, budget)
            return {
                "memories": selected,
                "token_estimate": sum(
                    int(item.get("token_estimate", len(str(item.get("content", "")))))
                    for item in selected
                ),
                "admitted": 0,
                "dropped_unscored": len(unscored),
            }
        merged = scored + unscored
        merged = self._drop_duplicate_content(merged)
        # Ranked, not merely concatenated: ``apply_budget`` compares
        # ``final_score`` when something has to go, so the supplementary unscored
        # text must sit behind every scored memory instead of tying at 0.0.
        merged.sort(
            key=lambda item: (
                1 if item.get("score_source") == "graphiti_core_unscored" else 0,
                -float(item.get("final_score") or 0.0),
                str(item.get("id") or ""),
            )
        )
        budget = policy.token_budget if token_budget is None else max(0, int(token_budget))
        selected = apply_budget(merged, top_k, budget)
        admitted = sum(1 for item in selected if item.get("score_source") == "graphiti_core_unscored")
        return {
            "memories": selected,
            "token_estimate": sum(
                int(item.get("token_estimate", len(str(item.get("content", "")))))
                for item in selected
            ),
            "admitted": admitted,
            "dropped_unscored": max(0, len(unscored) - admitted),
        }

    def list_role_memories(self, role_uuid: str) -> Dict[str, Any]:
        memories = self.authority.list_memories(role_uuid)
        return {"memories": memories, "count": len(memories)}

    def new_session(
        self,
        role_uuid: str,
        conversation_id: str,
        publisher,
        hot_zone_snapshot: Optional[Dict[str, Any]] = None,
        prompt_additions: Optional[Dict[str, Any]] = None,
        temp_extractor=None,
        **coordinator_options: Any,
    ):
        temporary_graph = TemporaryGraph(embedder=self.node_recall.embedder, extractor=temp_extractor)
        coordinator = ConversationSessionCoordinator(
            role_uuid=role_uuid,
            conversation_id=conversation_id,
            publisher=publisher,
            checkpoint_handler=temporary_graph.checkpoint,
            archive_handler=temporary_graph.archive,
            hot_zone_snapshot=hot_zone_snapshot,
            prompt_additions=prompt_additions,
            **coordinator_options,
        )
        coordinator.temporary_graph = temporary_graph
        temporary_graph.restore(coordinator.checkpoint_snapshot and coordinator.checkpoint_snapshot.get("temporary_graph"))
        return coordinator
