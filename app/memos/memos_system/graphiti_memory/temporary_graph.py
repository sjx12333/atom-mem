"""Session-local 临时图谱：上文存档 + 增量抽取的图结构。

记忆分三类：上文（当前对话聊天记录）、临时图谱召回上文（本存档召回）、
知识图谱召回内容（正式图谱召回）。本模块实现第二类的数据载体：

- ``entries``   有序对话条目存档（临时图谱的查询范围）。一轮 = 一次用户输入
                （打断后聚合算一次）或一次 LLM 输出；按对话发生顺序追加到末尾。
- ``nodes``     抽取出的实体节点（名称向量；实体与条目的“出现”关系跨条承载于
                ``entry_ids``）。
- ``node_edges`` 实体-实体关系（可跨条；按 source/target/name 去重合并）。
- ``entry_edges`` 条目-条目关系（严格按对话发生顺序：新增条目与其前一条建立
                顺序承接边）。

``checkpoint``：归档新增条目并增量临时图谱化 —— 逐条向量化 + LLM 抽取实体/
关系（允许返回空）。任何一步失败即整体抛异常（调用方不提交、不裁剪）。

``archive``：仅归档新增条目（向量化，不做 LLM 抽取）—— close 时把剩余上文
并入存档后对整个存档做正式知识图谱抽取。

``search``：向量匹配；命中节点则注入其关联的全部条目，命中条目则召回单条目；
结果结构与记忆召回一致（layer=TemporaryGraph），走记忆召回流程/接口。
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .domain import score_text, utc_timestamp
from .events_protocol import (
    build_extraction_prompt,
    parse_extraction_payload,
)

# The restriction wording lives in events_protocol so the temporary and formal
# extraction paths cannot drift (plan 阶段 3: 统一正式抽取和临时抽取的限制词).
EXTRACT_SYSTEM_PROMPT = build_extraction_prompt(conversation="")



def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _entry_key(entry: Dict[str, Any]) -> Optional[str]:
    return str(entry.get("event_id") or entry.get("id") or "")


def _norm_name(name: Any) -> str:
    cleaned = str(name or "").strip().strip("。，！？、；：\"'《》「」()（）-—_").strip()
    return cleaned[:64]


def _parse_extraction(content: str) -> Dict[str, Any]:
    """Parse the LLM JSON output leniently; empty on failure (allowed empty result)."""
    return parse_extraction_payload(content)


def build_temp_graph_extractor(config: Dict[str, Any]) -> Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]]:
    """Build a synchronous LLM extractor for temporary-graph entries.

    Uses the same chat config as formal extraction (``chat`` section). Returns
    ``None`` when the chat endpoint is not configured, in which case the
    temporary graph degrades to text-only archiving (no node/edge extraction).
    """
    chat = (config or {}).get("chat", {}) or {}
    base_url = str(chat.get("base_url") or "").rstrip("/")
    api_key = str(chat.get("api_key") or "").strip()
    model = str(chat.get("model") or "").strip()
    if not (bool(chat.get("enabled")) and api_key and base_url and model):
        return None

    def extract(entry: Dict[str, Any], prior_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        prior = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prior_entries[-5:]
        )
        current = f"{entry.get('role', 'user')}: {entry.get('content', '')}"
        user_prompt = build_extraction_prompt(conversation=current, history=prior)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        content = (json.loads(raw).get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _parse_extraction(content)

    return extract


class TemporaryGraph:
    def __init__(self, embedder: Optional[Callable[[str], Sequence[float]]] = None, extractor=None) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.nodes: Dict[str, Dict[str, Any]] = {}  # key = normalized name
        self.node_edges: List[Dict[str, Any]] = []
        self.entry_edges: List[Dict[str, Any]] = []
        # 会话内抽取到的事件候选：临时图谱只做索引/观察，它不是已发布的记忆。
        # 记忆节点（kind="event"）的唯一生产者是日记通路
        # （`GraphitiMemoryService._publish_memory_nodes`），这些候选不进入发布；
        # 它们仍随 `/graphiti/ui/temporary-graphs` 可查。
        self.events: List[Dict[str, Any]] = []
        self.embedder = embedder
        self.extractor = extractor
        self._seen: set[str] = set()

    # ---------------------------------------------------------------- helpers

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.embedder is None:
            return None
        try:
            vector = self.embedder(text)
            return list(vector) if vector is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _new_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(entry)
        key = _entry_key(item)
        item["id"] = key or f"entry_{uuid.uuid4().hex[:12]}"
        item["vector"] = self._embed(str(item.get("content", "")))
        item["created_at"] = item.get("created_at") or utc_timestamp()
        return item

    def _node(self, name: str) -> Dict[str, Any]:
        node = self.nodes.get(name)
        if node is None:
            node = {
                "id": f"node_{uuid.uuid4().hex[:12]}",
                "name": name,
                "vector": self._embed(name),
                "entry_ids": [],
                "created_at": utc_timestamp(),
            }
            self.nodes[name] = node
        return node

    # ---------------------------------------------------------------- ingest

    def archive(self, entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """仅归档（close 时把剩余上文并入存档）：去重、向量化、追加，不做抽取。"""
        for entry in entries:
            key = _entry_key(entry)
            if key and key in self._seen:
                continue
            item = self._new_entry(entry)
            self.entries.append(item)
            if key:
                self._seen.add(key)
        return self._snapshot()

    def checkpoint(self, entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """归档新增条目并增量临时图谱化。

        Raises:
            Exception: 向量化/LLM 抽取失败 —— 调用方不提交任何变更（本次不裁剪）。
        """
        pending = []
        for entry in entries:
            key = _entry_key(entry)
            if key and key in self._seen:
                continue
            item = self._new_entry(entry)  # 向量化失败会抛异常（不提交）
            pending.append(item)
        if not pending:
            return self._snapshot()
        if self.extractor is not None:
            for item in pending:
                result = self.extractor(item, self.entries)
                self._apply_extraction(item, result)
        # 提交：条目-条目关系严格按对话发生顺序（新增条目与前一条承接）
        for item in pending:
            if self.entries:
                self.entry_edges.append(
                    {
                        "source": self.entries[-1]["id"],
                        "target": item["id"],
                        "relation": "顺序承接",
                        "kind": "entry_order",
                    }
                )
            self.entries.append(item)
            key = _entry_key(item)
            if key:
                self._seen.add(key)
        return self._snapshot()

    def _apply_extraction(self, item: Dict[str, Any], result: Dict[str, Any]) -> None:
        entry_id = item["id"]
        for event in result.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            summary = str(event.get("summary") or "").strip()
            if not summary:
                continue
            self.events.append(
                {
                    "summary": summary,
                    "source_kind": event.get("source_kind"),
                    "occurred_at": event.get("occurred_at"),
                    "source_event_ids": list(event.get("source_event_ids") or []),
                    "entry_id": entry_id,
                    "created_at": utc_timestamp(),
                }
            )
        for entity in result.get("entities", []):
            name = _norm_name(entity.get("name") if isinstance(entity, dict) else entity)
            if not name:
                continue
            node = self._node(name)
            if entry_id not in node["entry_ids"]:
                node["entry_ids"].append(entry_id)
        for rel in result.get("relations", []):
            if not isinstance(rel, dict):
                continue
            source, target = _norm_name(rel.get("source")), _norm_name(rel.get("target"))
            rname = str(rel.get("relation") or "").strip()[:32]
            if not source or not target or not rname or source == target:
                continue
            src, tgt = self._node(source), self._node(target)
            for node, node_id in ((src, entry_id), (tgt, entry_id)):
                if entry_id not in node["entry_ids"]:
                    node["entry_ids"].append(entry_id)
            existing = next(
                (
                    edge
                    for edge in self.node_edges
                    if edge["source"] == src["id"] and edge["target"] == tgt["id"] and edge["name"] == rname
                ),
                None,
            )
            if existing is not None:
                if entry_id not in existing["entry_ids"]:
                    existing["entry_ids"].append(entry_id)
            else:
                self.node_edges.append(
                    {
                        "id": f"edge_{uuid.uuid4().hex[:12]}",
                        "source": src["id"],
                        "target": tgt["id"],
                        "name": rname,
                        "entry_ids": [entry_id],
                        "created_at": utc_timestamp(),
                    }
                )

    # ---------------------------------------------------------------- recall

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """向量匹配召回（走记忆召回流程/接口）。

        - 命中条目 -> 召回单条目；
        - 命中节点 -> 注入该节点关联的全部条目；
        条目去重后按相似度降序，返回与记忆召回一致的结构。
        """
        query_vector = self._embed(query)
        entry_by_id = {entry["id"]: entry for entry in self.entries}
        ranked: List[tuple[float, Dict[str, Any], str]] = []

        for entry in self.entries:
            if query_vector is not None:
                vector = entry.get("vector")
                if vector is None:
                    vector = self._embed(str(entry.get("content", "")))
                    if vector is None:
                        continue
                    entry["vector"] = vector
                similarity = _cosine(query_vector, vector)
            else:
                similarity = score_text(query, str(entry.get("content", "")))
            if similarity > 0:
                ranked.append((similarity, entry, "entry"))

        node_hits: List[tuple[float, Dict[str, Any]]] = []
        for node in self.nodes.values():
            if query_vector is not None:
                vector = node.get("vector")
                if vector is None:
                    vector = self._embed(node["name"])
                    if vector is None:
                        continue
                    node["vector"] = vector
                similarity = _cosine(query_vector, vector)
            else:
                similarity = score_text(query, node["name"])
            if similarity > 0:
                node_hits.append((similarity, node))
        node_hits.sort(key=lambda item: -item[0])
        for similarity, node in node_hits[: max(1, int(top_k))]:
            for entry_id in node.get("entry_ids", []):
                entry = entry_by_id.get(entry_id)
                if entry is not None:
                    ranked.append((similarity, entry, "node"))

        best: Dict[str, tuple[float, Dict[str, Any], str]] = {}
        for similarity, entry, kind in ranked:
            key = entry["id"]
            previous = best.get(key)
            if previous is None or similarity > previous[0]:
                best[key] = (similarity, entry, kind)

        results: List[Dict[str, Any]] = []
        for key, (similarity, entry, kind) in sorted(best.items(), key=lambda item: -item[1][0]):
            out = dict(entry)
            out.pop("vector", None)
            out["similarity"] = round(similarity, 6)
            out["final_score"] = round(similarity, 6)
            out["match"] = kind
            results.append(out)
            if len(results) >= max(1, int(top_k)):
                break
        return results

    # ---------------------------------------------------------------- state

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "temporary_graph_checkpoint",
            "entries": [{k: v for k, v in entry.items() if k != "vector"} for entry in self.entries],
            "nodes": [{k: v for k, v in node.items() if k != "vector"} for node in self.nodes.values()],
            "node_edges": [dict(edge) for edge in self.node_edges],
            "entry_edges": [dict(edge) for edge in self.entry_edges],
            "events": [dict(event) for event in self.events],
        }

    def restore(self, checkpoint: Dict[str, Any] | None) -> None:
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "temporary_graph_checkpoint":
            return
        self.entries = [dict(entry) for entry in checkpoint.get("entries", []) if isinstance(entry, dict)]
        self.nodes = {}
        for node in checkpoint.get("nodes", []):
            if isinstance(node, dict) and node.get("name"):
                node = dict(node)
                node.pop("vector", None)
                self.nodes[str(node["name"])] = node
        self.node_edges = [dict(edge) for edge in checkpoint.get("node_edges", []) if isinstance(edge, dict)]
        self.entry_edges = [dict(edge) for edge in checkpoint.get("entry_edges", []) if isinstance(edge, dict)]
        self.events = [dict(event) for event in checkpoint.get("events", []) if isinstance(event, dict)]
        self._seen = {key for entry in self.entries if (key := _entry_key(entry))}

    def clear(self) -> None:
        self.entries.clear()
        self.nodes.clear()
        self.node_edges.clear()
        self.entry_edges.clear()
        self.events.clear()
        self._seen.clear()
