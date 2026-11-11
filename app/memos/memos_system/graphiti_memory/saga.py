"""Deterministic publication pipeline and single visibility manifest.

The saga is the only cross-store publication boundary (CONTEXT 跨存储发布 Saga):

1. shared topology nodes/relations are upserted and granted to the role;
2. **a role's memory about an object is written into that object's attributes**
   (``node["memory"][role_uuid]``), so there is one place a fact lives and no
   second store to keep in sync;
3. the projection is rebuilt and validated before any legacy MemOS source is
   hidden;
4. a failure at any step restores the pre-operation snapshot and keeps a
   retryable operation row in SQLite.

Extraction produces exactly **two kinds of node** and this saga publishes only
those two:

* **knowledge nodes** — entities/relations the conversation established (step 1);
* **memory nodes** (``kind="event"``) — written by the diary pass, which is the
  single producer (``GraphitiMemoryService._publish_memory_nodes``).

The knowledge pass no longer extracts memory nodes: the old per-conversation
event array is retired (``events_protocol.build_event_extractor`` is deprecated
and returns ``None``), and the temporary graph's in-conversation candidates stay
an inspection detail of that checkpoint instead of becoming published memory.
A memory node never enters entity merge or ordinary relation upsert. Its semantic
edges to the entities it mentions are ordinary relations, because those edges are
what make the memory reachable; its ownership edge to the owning AI role node
stays non-semantic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .authority import AuthorityStore
from .contacts import (
    ExternalSourceRegistrar,
    contact_identity_index,
    declared_external_sources,
    resolve_contact_identity,
)
from .domain import normalized_text, publication_key, same_publication, text_hash, typed_id, utc_timestamp
from .operations import CutoverManager, ProjectionStore
from .roles import RoleAuthority
from .topology import SharedTopology
from .identity import (
    PENDING_USER_NODE_NAME,
    foreign_identity_scope,
    role_name_sets,
    source_names,
    user_node_id,
)

# 固定身份节点（与 roles.py / neo4j_bridge.py 保持一致）：每个角色分区各有自己的
# 用户保留主体（``user:<role_uuid>``）与 AI 保留主体（角色自身）。用户不再是全局
# 单例，否则两个角色会共享同一段"人"的记忆（ADR-0153）。
DEFAULT_APPELLATION = "主人"


def _norm_name(value: Any) -> str:
    return str(value or "").strip().lower()


class PublicationSaga:
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
        self.cutover = CutoverManager(authority)
        self.projections = ProjectionStore(authority.root, topology)

    def _identity_resolver(self, role_uuid: str, display_name: str, user_appellation: str, source_user_name: str = "", source_role_name: str = ""):
        """Return (resolve, user_names, ai_names) for this role's fixed identities.

        CONTEXT.md 语义实体解析：用户和 AI 保留主体按固定身份直接解析。抽取出的
        "user"/"assistant"/称呼/角色名 节点必须复用本角色的固定身份节点，而不是创建
        带随机 UUID 的新节点（否则图谱出现重复用户节点、assistant 伪节点）。

        The user node is scoped to the role (``user:<role_uuid>``): each partition
        has exactly one user identity, and no role can resolve an entity onto
        another role's user node.
        """
        self.topology.reload()
        role_user_id = user_node_id(role_uuid)
        current_user_name = str(
            self.topology.data.get("nodes", {}).get(role_user_id, {}).get("name") or "用户"
        )
        user_names = {
            _norm_name(name)
            for name in ("user", "用户", current_user_name, user_appellation, *source_names(source_user_name))
            if _norm_name(name)
        }
        ai_names = {
            _norm_name(name)
            for name in ("assistant", "ai", display_name, *source_names(source_role_name))
            if _norm_name(name)
        }

        def resolve(name: Any) -> Optional[str]:
            lowered = _norm_name(name)
            if lowered in user_names:
                return role_user_id
            if lowered in ai_names:
                return role_uuid
            return None

        return resolve, user_names, ai_names

    def _ensure_declared_contacts(
        self,
        role_uuid: str,
        sources: List[Dict[str, Any]],
        manifest: Dict[str, Any],
    ) -> None:
        """Write this batch's declared external contacts before its knowledge.

        The contact identity is derived from its number (ADR-0152), and the node the
        ordinary extraction pipeline hangs knowledge on is *that* identity — so it has
        to exist before the node loop runs. The hard join is a fire-and-forget request
        the pet can lose (gateway not up, retry budget exhausted, restart mid-flight),
        which is exactly the case handled here rather than only in the self-healing
        reconcile pass. A failure is recorded on the manifest and never fatal: the
        knowledge still publishes.
        """
        registrar = ExternalSourceRegistrar(self.topology, self.authority)
        for source in sources:
            try:
                registrar.register(
                    role_uuid,
                    kind="group" if str(source.get("kind") or "") == "group" else "private",
                    number=str(source.get("number") or ""),
                    display_name=source.get("display_name"),
                    group_number=source.get("group_number"),
                    group_name=source.get("group_name"),
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                manifest.setdefault("contact_identity_failures", []).append(
                    {"number": str(source.get("number") or ""), "error": str(exc)}
                )
        self.topology.reload()

    def _foreign_identity_scope(self, role_uuid: str) -> Tuple[Set[str], Set[str]]:
        """Another role's identity nodes and names, which this publication must not claim.

        Every role's user node is named after the same appellation and its AI node
        after a display name, so both a name lookup and a re-extracted node can land
        on another partition. Publishing that reference would grant the other AI
        into this role's whitelist and show it in 角色视角.

        Only a role that actually went live owns a name here. A document-import
        partition that never reached confirmation is still ``initializing`` and its
        identity node still carries the placeholder name (ADR-0167: 角色名在确认时
        才下发), so its configured display name answers to nothing — counting it made a
        *retry* of the same character look like a reference to another role, and the
        whole first chunk was rejected. The failed import's leftover partition is
        exactly this case.
        """
        self.topology.reload()
        configs = []
        for other in self.authority.role_ids():
            if str(other) == str(role_uuid):
                continue
            lifecycle = str(self.authority.role_state(other).get("lifecycle") or "")
            if lifecycle not in RoleAuthority.STARTABLE_LIFECYCLES:
                continue
            configs.append({"role_uuid": other, **self.authority.load_role_file(other, "config.json", {})})
        return foreign_identity_scope(role_uuid, self.topology.data.get("nodes", {}), role_name_sets(configs))

    @staticmethod
    def _is_foreign_identity_reference(
        node_id: str,
        name: str,
        foreign_ids: Set[str],
        foreign_names: Set[str],
    ) -> bool:
        """True when a published node is (or reads as) another role's identity."""
        if str(node_id) in foreign_ids:
            return True
        target = _norm_name(name)
        return bool(target) and target in foreign_names

    # ------------------------------------------------------------------ memory
    # Extraction publishes exactly two node kinds: knowledge nodes (above) and
    # **memory nodes** (``kind="event"``), which only the diary pass creates
    # (``GraphitiMemoryService._publish_memory_nodes``). The per-conversation
    # event array is retired — ``events_protocol.build_event_extractor`` is
    # deprecated and returns ``None``, and the temporary graph's in-conversation
    # candidates stay an inspection detail of that checkpoint — so the knowledge
    # pass produces no memory node and the event branch, its deterministic event
    # ids, its event vectorisation and its Neo4j event projection are all gone.

    @staticmethod
    def _source_message_ids(snapshot: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        for event in snapshot.get("events", []):
            for key in ("event_id", "id", "message_id"):
                value = str(event.get(key) or "").strip()
                if value and value not in ids:
                    ids.append(value)
                    break
        return ids

    def _merge_node_description(self, role_uuid: str, node_id: str, incoming: str) -> Dict[str, Any]:
        """Accumulate one role's view of an entity's description on that node.

        A node's description is the union of what the conversation has stated about
        it, in first-seen order, **per role**: the node is shared, so each role's
        description is its own field inside a role-keyed map and one role's view can
        never overwrite another's. Re-extracting the same sentence adds nothing,
        which keeps the node from growing without bound.
        """
        from .knowledge import merge_descriptions

        node = self.topology.data.get("nodes", {}).get(node_id)
        if node is None:
            return {"node_id": node_id, "changed": False, "reason": "node_missing"}
        current = node.get("description")
        current_view = str(current.get(role_uuid) or "") if isinstance(current, dict) else str(current or "")
        merged = merge_descriptions(current_view, incoming)
        if not merged["changed"]:
            return {"node_id": node_id, "changed": False}
        self.topology.set_node_description(node_id, role_uuid, merged["description"])
        return {"node_id": node_id, "changed": True, "added": merged["added"]}

    # -------------------------------------------------------------- publication

    def publish(self, snapshot: Dict[str, Any], extracted: Dict[str, Any], progress: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
        role_uuid = snapshot["role_uuid"]
        previous_manifest = self.authority.load_role_file(role_uuid, "publication_manifest.json", {})
        if previous_manifest.get("status") == "published" and same_publication(previous_manifest, snapshot):
            # Replay of *this* frozen snapshot (matched by batch identity, not by
            # conversation): the second batch of a conversation must still extract.
            return {"status": "published", "manifest": previous_manifest}
        operation_id = (
            snapshot.get("operation_id")
            or snapshot.get("publication_operation_id")
            or (previous_manifest.get("operation_id") if same_publication(previous_manifest, snapshot) else None)
            or typed_id("operation")
        )
        role_snapshot = self.authority.snapshot_role(role_uuid)
        # The rollback point must be the **persisted** pre-publication graph. The
        # saga's topology is a different instance from the one role creation writes
        # through, so without this reload its in-memory copy can predate nodes this
        # operation must not lose: a failure would then restore that stale copy over
        # the file and delete every identity node, which is how a role disappeared
        # from the graph after an extraction step failed.
        self.topology.reload()
        topology_snapshot = self.topology.snapshot()
        visibility_snapshot = self.authority.load_root_file("legacy/visibility.json", None)
        cutover_snapshot = self.authority.load_root_file("legacy/cutover.json", None)
        manifest = {
            "operation_id": operation_id,
            "role_uuid": role_uuid,
            "conversation_id": snapshot["conversation_id"],
            # The batch is the extraction unit: without this key a second batch of
            # the same conversation matches this manifest and never extracts.
            "batch_id": snapshot.get("batch_id"),
            "publication_key": publication_key(snapshot),
            "status": "publishing",
            "source_hidden": False,
            "hidden_source_ids": [],
            "source_batch_id": snapshot.get("source_batch_id"),
            "created_at": utc_timestamp(),
            "published_ids": [],
        }
        self.authority.save_role_file(role_uuid, "publication_manifest.json", manifest)
        store = getattr(self.memory_layer, "store", None)
        if store is not None:
            store.begin_publication(
                operation_id,
                role_uuid,
                conversation_id=snapshot.get("conversation_id"),
                manifest=manifest,
            )
        try:
            config = self.authority.load_role_file(role_uuid, "config.json", {})
            identity = self.authority.load_role_file(role_uuid, "identity.json", {})
            display_name = str(config.get("display_name") or "").strip() or role_uuid
            user_appellation = str(snapshot.get("user_appellation") or identity.get("user_appellation") or "").strip() or DEFAULT_APPELLATION
            # 文档创建角色的第一段（抽取期）**不给身份节点赋最终名字**：角色要在用户
            # 确认提示词后才正式拥有名字，所以这段时间身份节点留在占位名下，抽取出的
            # 别名照旧解析到这两个身份（映射来自 identity.json，与节点怎么叫无关）。
            # 确认时由 ``RoleAuthority.assign_identity_names`` 一次性赋名并重新索引。
            identity_name_pending = bool(snapshot.get("identity_name_pending"))
            # The partition's identity nodes exist before the first publication so
            # the extractor's "user"/AI entities always resolve onto them.
            role_user_id = user_node_id(role_uuid)
            if role_user_id not in self.topology.data.get("nodes", {}):
                self.topology.upsert_node(
                    role_user_id,
                    PENDING_USER_NODE_NAME if identity_name_pending else user_appellation,
                    kind="user",
                )
                self.topology.grant_node(role_uuid, role_user_id)
            resolve_identity, user_names, ai_names = self._identity_resolver(
                role_uuid, display_name, user_appellation,
                str(snapshot.get("source_user_name") or identity.get("source_user_name") or ""),
                str(snapshot.get("source_role_name") or identity.get("source_role_name") or ""),
            )
            # The batch's own external contacts, resolved the same way as the fixed
            # identities: a contact entity must land on the number-keyed node instead
            # of a second node under the extractor's uuid. Only names this batch
            # declared as its sources are in the index, so a same-named entity the
            # batch did not declare stays an ordinary entity.
            declared_sources = declared_external_sources(snapshot)
            contact_ids = contact_identity_index(declared_sources)
            if declared_sources:
                self._ensure_declared_contacts(role_uuid, declared_sources, manifest)

            # --- 节点：身份解析后 upsert 到共享拓扑，并授权给当前角色白名单 ---
            if progress:
                progress("extract_attributes", "写入共享图谱节点与关系", 5)
            foreign_ids, foreign_names = self._foreign_identity_scope(role_uuid)
            node_id_map: Dict[str, str] = {}
            node_descriptions: Dict[str, str] = {}
            foreign_node_refs: Dict[str, str] = {}
            for node in extracted.get("nodes", []):
                original_id = str(node.get("id") or typed_id("node"))
                name = str(node.get("name") or original_id).strip()
                identity_id = resolve_identity(name)
                contact_id = None if identity_id else resolve_contact_identity(name, contact_ids)
                target_id = identity_id or contact_id or original_id
                if (
                    identity_id is None
                    and contact_id is None
                    and self._is_foreign_identity_reference(target_id, name, foreign_ids, foreign_names)
                ):
                    # The conversation mentioned another AI (or resolved a shared
                    # appellation onto another partition's user node). That is not
                    # this role's graph: publishing it would put the other AI inside
                    # this role's 角色视角 and hand it this role's memory slice.
                    # Recorded, not silently forgotten — including which memories
                    # were anchored to it, so a reviewer can still read what was held
                    # back instead of finding a bare node id.
                    #
                    # Two references are never foreign:
                    # * one that resolved to **this** role's fixed identity — the
                    #   resolution is authoritative, and a name this role declares as
                    #   its own source alias must not be taken away by the heuristic
                    #   just because another partition also answers to it (a leftover
                    #   unconfirmed import of the same character used to reject the
                    #   retry's own identity node and abort the chunk);
                    # * a resolved contact, whose identity comes from this batch's own
                    #   declared source rather than from a name lookup.
                    foreign_node_refs[original_id] = name
                    manifest.setdefault("rejected_foreign_identity_nodes", []).append(
                        {"id": original_id, "name": name}
                    )
                    continue
                node_id_map[original_id] = target_id
                description = str(node.get("description") or "").strip()
                existing = self.topology.data.get("nodes", {}).get(target_id)
                if existing is None:
                    # A description never lands as a plain node field: it is one
                    # role's memory element, written below with the other memories,
                    # so the node cannot carry one role's prose to another.
                    self.topology.upsert_node(target_id, name, kind="node")
                elif identity_id is not None and not identity_name_pending:
                    # An identity node keeps its stable display name: a role's
                    # appellation belongs to that role's own user node, and the AI
                    # node keeps the role's display name. While an import is still
                    # unconfirmed (``identity_name_pending``) the two nodes stay on
                    # their placeholder names and confirmation assigns the real ones.
                    desired = display_name if identity_id == role_uuid else user_appellation
                    if desired and existing.get("name") != desired:
                        self.topology.upsert_node(
                            target_id, desired, kind=existing.get("kind", "node")
                        )
                elif identity_id is not None:
                    manifest.setdefault("identity_names_pending", []).append(
                        {"id": target_id, "name": existing.get("name")}
                    )
                if contact_id is not None:
                    # The contact node is named by its registration (a rename updates
                    # the same number-keyed node, ADR-0152); the extractor's spelling
                    # of that identity never renames it. The entity contributed its
                    # knowledge to the contact, which is a mapping, not a node write.
                    manifest.setdefault("contact_identity_nodes", []).append(
                        {"id": original_id, "name": name, "node_id": target_id}
                    )
                if description:
                    node_descriptions[target_id] = description
                self.topology.grant_node(role_uuid, target_id)
                manifest["published_ids"].append(target_id)

            # --- 关系：端点重映射到身份节点后 upsert + 授权 ---
            #
            # 这是"联想召回"的根因修复点。此前的写法是
            # ``upsert_relation(source_id, target_id)`` —— 抽取层明明给出了关系的名字
            # （``extraction_engine`` 的 ``name``），却在这里被丢掉，于是图谱里
            # **每一个实体之间的边都是无名的**（实测 53 条边里 29 条 has_fact、
            # 15 条 memory_link、2 条 companion，只剩 7 条无名 relation）。
            # 结果：即使给 node_recall 加上多跳，也没有一条带语义的边可走。
            #
            # 现在做两件事：
            #   1. 把关系的名字真正写进共享边（``name``）；
            #   2. 用关系名建一个**共享谓词节点**，两个实体各自连到它。这才是"有关节点
            #      天然建立图谱连接"：交会点由谓词文本决定，跨会话、跨实体自动合并，
            #      不是"因为共同出现过所以硬连一条边"。
            relation_id_by_edge: Dict[str, str] = {}
            collapsed_edges = set()
            predicate_links: List[Dict[str, Any]] = []
            for relation in extracted.get("relations", []):
                source_id = node_id_map.get(str(relation["source_id"]), str(relation["source_id"]))
                target_id = node_id_map.get(str(relation["target_id"]), str(relation["target_id"]))
                missing = [
                    endpoint
                    for endpoint in (source_id, target_id)
                    if endpoint not in self.topology.data["nodes"]
                ]
                if missing:
                    # An edge may name an object that was never published: a node the
                    # knowledge filter dropped, another role's identity (held back
                    # above), or an id the model reused from context. One such edge
                    # must not roll back the whole chunk — it is recorded like every
                    # other dropped item and the rest of the batch publishes.
                    manifest.setdefault("dropped_relation_endpoints", []).append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "missing": missing,
                            "name": str(relation.get("name") or ""),
                        }
                    )
                    continue
                if source_id == target_id:
                    # Explicit aliases can collapse an extracted same-person edge.
                    # Preserve the mapping for audit, never publish a self-loop or
                    # turn that relationship into an entity attribute.
                    collapsed_edges.add(str(relation.get("edge_uuid") or ""))
                    manifest.setdefault("collapsed_identity_relations", []).append(dict(relation))
                    continue
                predicate = str(relation.get("predicate") or "").strip()
                try:
                    relation_record = self.topology.upsert_relation(
                        source_id,
                        target_id,
                        # 关系名不再丢弃。没有谓词时保持无名（不编造一个标签）。
                        **({"name": predicate} if predicate else {}),
                    )
                    self.topology.grant_relation(role_uuid, relation_record["id"])
                    edge_uuid = str(relation.get("edge_uuid") or "")
                    if edge_uuid:
                        relation_id_by_edge[edge_uuid] = relation_record["id"]
                    manifest["published_ids"].append(relation_record["id"])
                    if predicate:
                        # 关系是**独立检索对象**：谓词名 + 两端名字就是它的索引文本，
                        # 命中它就等于命中"A 因为 <谓词> 与 B 有关"。发布路径过去只索引
                        # 谓词/事实/hook，关系对象永远没有向量，于是 33 次真实召回里
                        # relation 候选数为 0（关系事实上不可召回）。
                        self._index_object(role_uuid, str(relation_record["id"]))
                        # 谓词节点：两个实体共同连到的交会点。共享边只保留谓词名作为
                        # 检索标签，两端的**理解**仍留在各自私有的 fact 里；这里新增的是
                        # 共享对象，不是共享判断。
                        predicate_node = self.topology.create_predicate_node(
                            role_uuid,
                            predicate,
                            source_conversation_id=snapshot.get("conversation_id"),
                        )
                        manifest["published_ids"].append(str(predicate_node["id"]))
                        for endpoint_id in (source_id, target_id):
                            link = self.topology.link_entity_to_predicate(
                                role_uuid, endpoint_id, str(predicate_node["id"])
                            )
                            predicate_links.append(
                                {
                                    "predicate_id": str(predicate_node["id"]),
                                    "entity_id": endpoint_id,
                                    "relation_id": str(link["id"]),
                                }
                            )
                        self._index_object(role_uuid, str(predicate_node["id"]))
                except (KeyError, ValueError) as exc:
                    # Same rule as the hooks below: an object the topology refuses
                    # (an event endpoint, a node already tombstoned) is recorded
                    # instead of failing the batch.
                    manifest.setdefault("dropped_relations", []).append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "name": str(relation.get("name") or ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
            if predicate_links:
                manifest["predicate_nodes"] = predicate_links

            # --- AI 视角：自述（无实体）与看法（AI <-> 被描述实体的中继） ---
            #
            # These are fact-like objects, but their anchor is the speaker/target
            # pair, not a parent entity. Keeping them out of the ordinary memory
            # loop is what prevents "温兆伦不火" from being filed as knowledge of
            # 温兆伦 with no record that the AI said it.
            published_by_name: Dict[str, str] = {}
            for node in extracted.get("nodes", []):
                original = str(node.get("id") or "")
                mapped = node_id_map.get(original, original)
                name_key = normalized_text(node.get("name"))
                if name_key and mapped in self.topology.data.get("nodes", {}):
                    published_by_name[name_key] = mapped
            perspective_published: List[Dict[str, Any]] = []
            perspective_texts: set[str] = set()
            for item in extracted.get("perspective", []) or []:
                if not isinstance(item, dict):
                    continue
                text = " ".join(str(item.get("text") or "").split())
                kind = str(item.get("kind") or "").strip()
                speaker_name = str(item.get("speaker_name") or "").strip()
                if not text or kind not in ("self_statement", "opinion") or not speaker_name:
                    continue
                speaker_id = resolve_identity(speaker_name)
                if not speaker_id:
                    manifest.setdefault("unresolved_perspective_speakers", []).append(
                        {"speaker": speaker_name, "text": text}
                    )
                    continue
                target_id = ""
                if kind == "opinion":
                    target_name = str(item.get("target_name") or "").strip()
                    target_id = resolve_identity(target_name)
                    if not target_id:
                        target_id = published_by_name.get(normalized_text(target_name), "")
                    if not target_id:
                        for node_id, node in self.topology.data.get("nodes", {}).items():
                            if normalized_text(node.get("name")) == normalized_text(target_name):
                                target_id = node_id
                                break
                    if not target_id:
                        manifest.setdefault("unresolved_opinion_targets", []).append(
                            {"target": target_name, "text": text}
                        )
                        continue
                try:
                    record = self.topology.create_perspective_node(
                        role_uuid,
                        speaker_id,
                        text,
                        fact_kind=kind,
                        target_id=target_id or None,
                        source_conversation_id=snapshot.get("conversation_id"),
                    )
                except (KeyError, ValueError) as exc:
                    manifest.setdefault("dropped_perspective_nodes", []).append(
                        {"kind": kind, "text": text, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                written = str(record.get("id") or "")
                manifest["published_ids"].append(written)
                self._index_object(role_uuid, written)
                perspective_published.append(
                    {
                        "id": written,
                        "kind": kind,
                        "speaker_id": speaker_id,
                        "target_id": target_id or "",
                    }
                )
                perspective_texts.add(normalized_text(text))
            if perspective_published:
                manifest["perspective_nodes"] = perspective_published

            # --- 角色认知：抽取出的节点/边事实写进「它所描述的那个对象的属性」 ---
            if progress:
                progress("resolve_temporal", "时间区间判定与保存", 6)
            for memory in extracted.get("memories", []):
                content = str(memory.get("content") or "").strip()
                if not content:
                    continue
                if (
                    normalized_text(content) in perspective_texts
                    and not memory.get("node_description")
                ):
                    # The same sentence arrived through both channels. The
                    # perspective node is the one that keeps the speaker, so the
                    # duplicate entity fact is recorded and skipped instead of
                    # producing two copies of one opinion.
                    manifest.setdefault("perspective_duplicate_memories", []).append(content)
                    continue
                original_relations = [str(ref) for ref in memory.get("relation_refs", [])]
                # A statement that came from an edge stays out of the graph when that
                # edge collapsed onto one identity (an alias statement such as
                # "小雨又名雨姐"): its endpoints are the same node, so it is not a
                # relation and must not become that node's knowledge either. The
                # extraction layer re-anchors such a sentence onto the named entity,
                # so the check has to follow the claim back to its source edge.
                original_relations += [
                    str(ref) for ref in memory.get("source_relation_refs", []) if str(ref or "")
                ]
                if original_relations and all(ref in collapsed_edges for ref in original_relations):
                    continue
                node_refs = [
                    node_id_map.get(str(ref), str(ref))
                    for ref in memory.get("node_refs", [])
                    if str(ref or "").strip()
                ]
                relation_refs = [
                    relation_id_by_edge.get(str(ref), str(ref))
                    for ref in memory.get("relation_refs", [])
                    if str(ref or "").strip() and str(ref) not in collapsed_edges
                ]
                if memory.get("node_description") and node_refs:
                    if any(ref in foreign_node_refs for ref in node_refs):
                        manifest.setdefault("rejected_foreign_identity_memories", []).append(content)
                        continue
                    # A node's own description (ADR-0150/0153). It is one memory
                    # element of kind ``description`` **on that node**, for this
                    # role only, and the node's ``description`` map is the derived,
                    # human-readable view of those elements. There is no second
                    # store to keep in sync any more.
                    for ref in node_refs:
                        self.topology.reload()
                        records = self._attach_memory(
                            role_uuid, ref, content, kind="description",
                            source_conversation_id=snapshot.get("conversation_id"),
                        )
                        if not records:
                            manifest.setdefault("unknown_memory_anchors", []).append(ref)
                            continue
                        manifest["published_ids"].append(ref)
                        self._merge_node_description(role_uuid, ref, content)
                        self._index_object(role_uuid, ref)
                    continue
                anchors: Dict[str, str] = {}  # anchor -> element kind
                for ref in node_refs:
                    anchors.setdefault(ref, "fact")
                for ref in relation_refs:
                    anchors.setdefault(ref, "relation")
                anchors = {ref: kind for ref, kind in anchors.items() if ref not in foreign_node_refs}
                if (
                    not anchors
                    and memory.get("node_refs")
                    and all(str(ref) in foreign_node_refs for ref in memory.get("node_refs"))
                    and not relation_refs
                ):
                    # The only object this memory was about is another AI's identity:
                    # hold it back with that node (recorded, not silently lost).
                    manifest.setdefault("rejected_foreign_identity_memories", []).append(content)
                    continue
                if not anchors:
                    # A fact with no anchor has no object to live on. It used to
                    # fall back to a standalone "plain memory" row; that store is
                    # retired, so record the drop instead of inventing an object.
                    manifest.setdefault("unanchored_memories", []).append(content)
                    continue
                for anchor, kind in anchors.items():
                    records = self._attach_memory(
                        role_uuid, anchor, content, kind=kind,
                        source_conversation_id=snapshot.get("conversation_id"),
                    )
                    if not records:
                        manifest.setdefault("unknown_memory_anchors", []).append(anchor)
                        continue
                    # A non-description memory now lives on its **own fact node**
                    # (节点细化), so that node is what needs a vector — indexing the
                    # anchor would leave the fact unsearchable and, worse, would
                    # re-index an entity whose text did not change. A statement about
                    # a relationship is stored on both of its endpoints, so it can be
                    # more than one node.
                    for record in records:
                        written = str(record.get("id") or anchor)
                        manifest["published_ids"].append(written)
                        self._index_object(role_uuid, written)

            # Graphiti already decided these older edges are contradicted. The new
            # assertion is published above; now retire the old *fact node* that
            # carries the same sentence, so recall cannot answer with the stale one.
            invalidated_facts = self._invalidate_superseded_facts(
                role_uuid, extracted.get("invalidated_edges") or []
            )
            if invalidated_facts:
                manifest["invalidated_facts"] = invalidated_facts

            # --- 联想节点（hook）：模型对一个实体**可能**的关联所做的预测 ---
            #
            # 这一步的产物不是知识。两条硬边界：
            #
            # 1. 每个 hook 带 ``basis="model_predicted"``（见 ``topology.create_hook_node``
            #    与 ``node_recall.render_record``）：一旦失去这个标记，"我预测的关联"就会
            #    被模型读成"已确认的记忆"，与 ADR-0150 的禁止推断契约直接冲突；
            # 2. hook **不直接参与准入**。它给已过门槛的实体提供排序加成
            #    （``gateway._apply_hook_boost``）；唯一例外是查询逐字包含某个 hook、
            #    且该文本只指向一个母实体时，允许它**提名**该实体——母实体仍要过
            #    重排门槛，见 ``node_recall._nominate_hook_parents``。
            #
            # 它的价值：谓词节点只能连**语料里出现过**的关联，hook 能连从未被一起提到过的
            # 东西——猫和开罐器可能永远不在同一句话里，但"开罐头的声音"是两者共同的 hook。
            hooks_published: List[Dict[str, Any]] = []
            for hook in extracted.get("hooks", []) or []:
                if not isinstance(hook, dict):
                    continue
                text = " ".join(str(hook.get("text") or "").split())
                parent_id = node_id_map.get(str(hook.get("parent_id")), str(hook.get("parent_id") or ""))
                if not text or not parent_id:
                    continue
                if parent_id in foreign_node_refs:
                    # 只挂在别人身份节点上的联想不是这个角色的图谱。
                    manifest.setdefault("rejected_foreign_identity_hooks", []).append(
                        {"parent_id": parent_id, "text": text}
                    )
                    continue
                self.topology.reload()
                try:
                    record = self.topology.create_hook_node(
                        role_uuid,
                        parent_id,
                        text,
                        basis=str(hook.get("basis") or "model_predicted"),
                        confidence=hook.get("confidence"),
                        source_conversation_id=snapshot.get("conversation_id"),
                    )
                except (KeyError, ValueError) as exc:
                    # 父节点不存在或不可挂载：记录而不是静默丢。
                    manifest.setdefault("unattachable_hooks", []).append(
                        {"parent_id": parent_id, "text": text, "error": str(exc)}
                    )
                    continue
                written = str(record.get("id") or "")
                manifest["published_ids"].append(written)
                self._index_object(role_uuid, written)
                hooks_published.append({"id": written, "parent_id": parent_id, "text": text})
            if hooks_published:
                manifest["hook_nodes"] = hooks_published

            # No memory node is published here. Extraction produces exactly two
            # node kinds, and the knowledge pass only produces knowledge nodes;
            # memory nodes come from the diary pass (see the module docstring).
            # If a caller still hands this saga an event array, record the count
            # instead of publishing a memory node the architecture no longer
            # sources from here — a regression stays visible.
            ignored_events = extracted.pop("discarded_event_candidates", None)
            if ignored_events is None:
                ignored_events = len(
                    [item for item in (extracted.get("events") or []) if isinstance(item, dict)]
                )
            if ignored_events:
                manifest["ignored_event_candidates"] = ignored_events

            # 抽取层拒收的陈述与丢弃的重复句：进 manifest，改动效果可审计。
            # 和 unanchored_memories 一样，是"记录而不是静默丢"。
            # ``dropped_open_predicates``：关系名是开放谓词（"相关"）因此没有建成共享
            # 谓词节点——这条边留成无名边，但"为什么它没能连上"是可查的。
            # ``dropped_hooks``：模型为一个没能通过知识过滤的实体预测了联想。
            for key in (
                "dropped_non_knowledge_statements",
                "duplicate_memories",
                "dropped_open_predicates",
                "dropped_hooks",
            ):
                value = extracted.get(key)
                if value:
                    manifest[key] = value

            # The projection is rebuilt before any legacy source is hidden.
            projection = self.projections.rebuild(role_uuid, self.authority)
            self.projections.validate(role_uuid)
            source_ids = list(dict.fromkeys(
                str(item)
                for item in (
                    snapshot.get("memos_source_ids", [])
                    + snapshot.get("source_ids", [])
                    + extracted.get("memos_source_ids", [])
                )
                if item
            ))
            if source_ids:
                self.cutover.register_sources(role_uuid, source_ids, layer=snapshot.get("source_layer", "LongTermMemory"))
                hidden = self.cutover.hide_sources(role_uuid, operation_id, source_ids)
                manifest["source_hidden"] = hidden["status"] == "hidden"
                manifest["hidden_source_ids"] = source_ids
            else:
                manifest["source_hidden"] = True
            manifest["projection_version"] = projection["version"]
            manifest["status"] = "published"
            manifest["published_at"] = utc_timestamp()
            self.authority.save_role_file(role_uuid, "publication_manifest.json", manifest)
            self.authority.save_role_state(role_uuid, memory_generation=self.authority.role_state(role_uuid)["memory_generation"] + 1)
            if store is not None:
                revision = store.bump_revision()
                manifest["sqlite_revision"] = revision
                store.complete_publication(
                    operation_id,
                    published_ids=manifest["published_ids"],
                    source_message_ids=self._source_message_ids(snapshot),
                    embedding_model=self.embedding_model or None,
                    embedding_version=self.embedding_version or None,
                    manifest=manifest,
                )
                self.authority.save_role_file(role_uuid, "publication_manifest.json", manifest)
            return {"status": "published", "manifest": manifest}
        except Exception as exc:
            self.authority.restore_role(role_uuid, role_snapshot)
            self.topology.restore(topology_snapshot)
            if visibility_snapshot is None:
                visibility_path = self.authority.root / "legacy" / "visibility.json"
                if visibility_path.exists():
                    visibility_path.unlink()
            else:
                self.authority.save_root_file("legacy/visibility.json", visibility_snapshot)
            if cutover_snapshot is None:
                cutover_path = self.authority.root / "legacy" / "cutover.json"
                if cutover_path.exists():
                    cutover_path.unlink()
            else:
                self.authority.save_root_file("legacy/cutover.json", cutover_snapshot)
            manifest["status"] = "retry_locked"
            self.authority.save_role_file(role_uuid, "publication_manifest.json", manifest)
            if store is not None:
                store.fail_publication(operation_id, str(exc))
            raise

    # --------------------------------------------------------- memory elements

    def _invalidate_superseded_facts(
        self, role_uuid: str, invalidated_edges: Any
    ) -> List[Dict[str, Any]]:
        """Retire project fact nodes whose sentence Graphiti just invalidated.

        Matching is by exact normalized sentence and role, not by slot inference:
        Graphiti's edge resolution is the authority on what was contradicted, and
        the old fact keeps its own node as read-only history. Invalidation happens
        after the new facts are published, so a correction that replaces an old
        assertion never leaves the graph with neither side.
        """
        from . import node_memory

        if not isinstance(invalidated_edges, list):
            return []
        wanted: Dict[str, str] = {}
        for item in invalidated_edges:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("fact") or "").split())
            if text:
                wanted.setdefault(text, str(item.get("edge_uuid") or ""))
        if not wanted:
            return []

        self.topology.reload()
        retired: List[Dict[str, Any]] = []
        for record in self.topology.fact_records_for_role(role_uuid):
            if str(record.get("status") or "active") != "active":
                continue
            body = " ".join(node_memory.fact_body(record).split())
            if body not in wanted:
                continue
            fact_id = str(record.get("id") or "")
            if not fact_id:
                continue
            updated = self.topology.invalidate_fact_node(
                fact_id, reason="superseded_by_new_assertion"
            )
            retired.append(
                {
                    "fact_id": fact_id,
                    "text": body,
                    "invalidated_at": updated.get("invalidated_at"),
                    "source_edge_uuid": wanted[body],
                }
            )
        return retired

    def _attach_memory(
        self,
        role_uuid: str,
        object_id: str,
        text: str,
        *,
        kind: str,
        source_conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Attach one knowledge/memory item to the object(s) it is about, for this role.

        Two shapes, by kind:

        * ``description`` stays an **element on the entity**: it is the entity's own
          identity ("这个实体是谁"), it travels with every hit of that entity, and it
          is not a separately selectable item;
        * everything else becomes a **fact node of its own** (节点细化). A fact carries
          its own text, its own status and its own vector, which is what makes it
          possible to gate, count, age and invalidate one sentence without touching
          its entity — the thing that was impossible while sentences were elements
          glued to the entity's single vector.

        Returns the objects actually written; an empty list means the anchor cannot
        carry the sentence, which the caller records instead of losing it silently.

        ``kind="relation"`` is the one anchor that is not an entity: a statement that
        names no entity is knowledge about a *relationship*, and a fact node can only
        hang off an entity. Such a sentence is therefore attached to **both endpoints**
        — the two subjects it is about — rather than making
        :meth:`topology.SharedTopology.create_fact_node` reject the relation as a
        parent, which used to abort the whole batch with
        ``KeyError('fact parent not found: r_…')``.
        """
        from . import node_memory

        self.topology.reload()
        stored = self.topology.find_record(object_id)
        if stored is None:
            return []
        if kind != "description":
            parents = [object_id]
            if kind == "relation":
                relation = self.topology.get_relation(object_id)
                parents = [str(item) for item in ((relation or {}).get("endpoints") or []) if str(item)]
            records: List[Dict[str, Any]] = []
            for parent in parents:
                # Facts hang off entities; a fact, event or unknown object is not a
                # parent, and one unanchored sentence must never fail the batch.
                if not self.topology.is_attachable_node(parent):
                    continue
                records.append(self.topology.create_fact_node(
                    role_uuid,
                    parent,
                    text,
                    kind=kind,
                    source_conversation_id=source_conversation_id,
                ))
            return records
        node_memory.upsert_elements(
            stored,
            role_uuid,
            [text],
            kind=kind,
            source_conversation_id=source_conversation_id,
        )
        return [self.topology.save_record_memory(object_id, stored)]

    def _index_object(self, role_uuid: str, object_id: str) -> None:
        """Refresh the vector for one object's memory as this role holds it.

        The vector is derived: its text is the role's active element text for the
        object, so any write that changed the memory produces a different content
        hash and therefore a fresh embedding.
        """
        from . import node_memory

        store = getattr(self.memory_layer, "store", None)
        record = self.topology.find_record(object_id)
        if record is None:
            return
        endpoint_names = {
            str(node_id): str((node or {}).get("name") or node_id)
            for node_id, node in (self.topology.data.get("nodes") or {}).items()
        }
        text = node_memory.index_text(record, role_uuid, endpoint_names=endpoint_names)
        if not text:
            return
        object_type = node_memory.object_type_of(record)
        content_hash = node_memory.index_hash(record, role_uuid, endpoint_names=endpoint_names)
        if store is not None and self._vector_is_fresh(store, role_uuid, object_id, object_type, content_hash):
            return
        self._vectorize_object(store, role_uuid, object_id, object_type, text, content_hash)

    def _vector_is_fresh(
        self,
        store: Any,
        role_uuid: str,
        object_id: str,
        object_type: str,
        content_hash: str,
    ) -> bool:
        """True when this role's stored vector already matches the object's text."""
        try:
            existing = store.get_embedding(object_id, role_uuid=role_uuid)
        except Exception:  # noqa: BLE001 - a cache miss must never block publication
            return False
        if not existing:
            return False
        return (
            str(existing.get("content_hash") or "") == content_hash
            and str(existing.get("object_type") or "") == object_type
            and str(existing.get("model") or "") == str(self.embedding_model or "")
            and str(existing.get("version") or "") == str(self.embedding_version or "")
        )

    def _vectorize_object(
        self,
        store: Any,
        role_uuid: str,
        object_id: str,
        object_type: str,
        text: str,
        content_hash: str,
    ) -> None:
        """Embed one object's memory text and store the vector inside Neo4j."""
        if self.memory_layer is None:
            return
        if self.embedder is None:
            if store is not None:
                store.enqueue_embedding_rebuild(
                    object_id, object_type, reason="no_embedder", role_uuid=role_uuid
                )
            return
        vector = self.embedder(text)
        if not vector:
            if store is not None:
                store.enqueue_embedding_rebuild(
                    object_id, object_type, reason="no_vector", role_uuid=role_uuid
                )
            return
        vectors = getattr(self.memory_layer, "vectors", None)
        if vectors is None or not getattr(self.memory_layer, "vectors_available", False):
            if store is not None:
                store.enqueue_embedding_rebuild(
                    object_id, object_type, reason="vector_index_unavailable", role_uuid=role_uuid
                )
            return
        operation = None
        if store is not None:
            operation = store.log_vector_operation(
                object_id, object_type, "upsert", payload={"content_hash": content_hash}
            )
        try:
            result = vectors.upsert(
                object_id,
                vector,
                object_type=object_type,
                role_uuid=role_uuid,
                content_hash=content_hash,
                status="active",
                content=text,
            )
        except Exception as exc:  # noqa: BLE001
            if store is not None and operation is not None:
                store.mark_vector_operation(operation["operation_id"], status="failed", error=str(exc))
            raise
        if store is None:
            return
        if operation is not None:
            store.mark_vector_operation(operation["operation_id"], status="done")
        store.record_embedding(
            object_id,
            object_type,
            role_uuid=role_uuid,
            content_hash=content_hash,
            model=self.embedding_model or None,
            version=self.embedding_version or None,
            dimension=len(vector),
            vector_point_id=result["point_id"],
        )
        store.complete_embedding_rebuild(object_id, role_uuid=role_uuid)
