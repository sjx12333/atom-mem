"""Formal and explicit memory extraction adapters.

抽取只产出两类节点：**知识节点**（实体/关系，由知识通路发布）与**记忆节点**
（`kind="event"`，只由日记通路 ``_publish_memory_nodes`` 创建）。这个模块里的
知识通路 `publish` 因此不把任何 events 交给发布 Saga，见其内部注释。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .authority import AuthorityStore
from .saga import PublicationSaga
from .topology import SharedTopology


class ExplicitMemoryService:
    def __init__(self, authority: AuthorityStore):
        self.authority = authority

    def remember(self, role_uuid: str, source_text: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        persist_as = decision.get("persist_as")
        if persist_as not in {"memory", "relation_group"}:
            raise ValueError("persist_as must be memory or relation_group")
        if not source_text.strip():
            raise ValueError("remembered content cannot be empty")
        submission_id = decision.get("submission_id")
        if not submission_id:
            raise ValueError("submission_id is required")
        existing = self.authority.load_role_file(role_uuid, "explicit_submissions.json", {})
        if submission_id in existing:
            return existing[submission_id]
        if persist_as == "memory":
            record = self.authority.save_memory(
                role_uuid,
                decision.get("content", source_text),
                kind="memory",
                source="explicit_remember",
                source_id=submission_id,
                manual_protected=True,
            )
        else:
            groups = self.authority.load_role_file(role_uuid, "relation_groups.json", [])
            record = {
                "id": decision.get("id"),
                "kind": "relation_group",
                "role_uuid": role_uuid,
                "content": decision.get("content", source_text),
                "index_brief": decision.get("index_brief", source_text),
                "node_refs": list(decision.get("node_refs", [])),
                "relation_refs": list(decision.get("relation_refs", [])),
                "status": "active",
                "source": "explicit_remember",
                "source_id": submission_id,
                "manual_protected": True,
            }
            if not record["id"]:
                from .domain import typed_id

                record["id"] = typed_id("relation_group")
            replaced = False
            for index, current in enumerate(groups):
                if current.get("id") == record["id"] or current.get("source_id") == submission_id:
                    groups[index] = record
                    replaced = True
                    break
            if not replaced:
                groups.append(record)
            self.authority.save_role_file(role_uuid, "relation_groups.json", groups)
        existing[submission_id] = record
        self.authority.save_role_file(role_uuid, "explicit_submissions.json", existing)
        return record


class ExtractionPipeline:
    def __init__(
        self,
        authority: AuthorityStore,
        topology: SharedTopology,
        memory_layer: Any = None,
        embedder: Any = None,
        neo4j_projection: Any = None,
        embedding_model: str = "",
        embedding_version: str = "",
    ):
        self.authority = authority
        self.topology = topology
        self.memory_layer = memory_layer
        self.embedder = embedder
        self.neo4j_projection = neo4j_projection
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.saga = PublicationSaga(
            authority,
            topology,
            memory_layer=memory_layer,
            embedder=embedder,
            neo4j_projection=neo4j_projection,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )

    def publish(self, snapshot: Dict[str, Any], extractor: Callable[[str], Dict[str, Any]], progress: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
        valid_events = [
            event
            for event in snapshot.get("events", [])
            if event.get("complete") and event.get("observable") and str(event.get("content", "")).strip()
        ]
        frozen_snapshot = dict(snapshot)
        frozen_snapshot["events"] = valid_events
        if progress:
            progress("embed_and_save", "保存对话快照并向量化", 4)
        self.authority.save_archive(frozen_snapshot)
        text = "\n".join(event["content"] for event in valid_events)
        extracted = extractor(text) or {}
        extracted.setdefault("nodes", [])
        extracted.setdefault("relations", [])
        extracted.setdefault("memories", [])
        extracted.setdefault("relation_groups", [])
        # 抽取只产出两类节点：知识节点（实体/关系）与记忆节点（日记 event 节点）。
        # 这条知识通路的产物里不含记忆节点，所以任何 events 一律不进入发布：
        # 「事件抽取提示词」已废弃（``build_event_extractor`` 恒返回 None），
        # 临时图谱在会话内抽到的事件只是该检查点自己的观察数据（仍随
        # ``/graphiti/ui/temporary-graphs`` 可查），不构成已发布的记忆。
        # 数量交给 saga 记进 manifest：有事件被交进来时要看得见，而不是静默吞掉。
        discarded = len([item for item in (extracted.get("events") or []) if isinstance(item, dict)])
        extracted["events"] = []
        extracted["discarded_event_candidates"] = discarded
        for group in extracted["relation_groups"]:
            extracted["memories"].append(
                {
                    **group,
                    "content": group.get("promoted_summary") or group.get("index_brief", ""),
                    "kind": "relation_group",
                }
            )
        result = self.saga.publish(frozen_snapshot, extracted, progress)
        return result

    def cleanup(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return self.authority.cleanup_archive(snapshot["role_uuid"], snapshot["conversation_id"])

    def run(self, snapshot: Dict[str, Any], extractor: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        result = self.publish(snapshot, extractor)
        self.cleanup(snapshot)
        return result
