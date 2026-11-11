"""Shared identity-only topology and role whitelist.

Nodes are identity-only (``kind`` in ``user``/``ai_role``/``node``/``event``/``fact``)
and the shared edge is an undirected, semantic-free logical link.

Two node kinds are *not* entities and are kept out of entity rendering and multi-hop
adjacency:

**Events** (plan 阶段 2 / 最终不变式):

- an event is never merged into another node;
- an event may own exactly one semantic-free link, and only to its AI role node;
- that link is edited only through :meth:`SharedTopology.create_event_node` and
  :meth:`SharedTopology.link_event_to_role`, never through the ordinary
  relation interface;
- events are excluded from entity multi-hop adjacency.

**Facts** (节点细化):

- a fact is one knowledge/memory item that belongs to exactly one entity, so it
  carries its own ``text`` and its own vector and can be gated, counted, aged and
  invalidated **individually** instead of riding along with whatever the parent
  entity's text happened to be;
- it is attached with one structural ``has_fact`` edge to its parent, plus the
  parent recorded as ``parent_id``; the edge is bookkeeping, not semantics;
- like events, facts are excluded from entity multi-hop adjacency (otherwise every
  sentence would become a path waypoint), and they are never merged.
"""

from __future__ import annotations

import json
import os
import tempfile
import copy
import functools
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain import normalized_text, replace_file, stable_pair, text_hash, typed_id, utc_timestamp
from .identity import is_foreign_user_node_id, user_node_id
from .node_memory import project_for_role


NODE_KINDS = ("node", "user", "ai_role", "event", "fact", "predicate", "hook")
EVENT_NODE_KIND = "event"
EVENT_LINK_KIND = "event_link"
FACT_NODE_KIND = "fact"
#: Fact-like objects that are spoken by the AI rather than stated about a
#: subject entity. They keep kind="fact" so the vector index, whitelist and
#: recall gates need no new object type, but their fact_kind changes how they
#: are anchored:
#:
#: * self_statement - the AI's own durable self-description; it has no parent
#:   entity (the AI is the speaker, not the subject).
#: * opinion - the AI's durable evaluation of one entity; it is a relay whose
#:   speaker_id is the AI and target_id is the described entity.
SELF_STATEMENT_KIND = "self_statement"
OPINION_KIND = "opinion"
PERSPECTIVE_KINDS = (SELF_STATEMENT_KIND, OPINION_KIND)
#: Structural edge from an entity to one of its facts. Deliberately *not* a semantic
#: relation: the meaning of "这个实体有这条知识" is the fact node itself.
PARENT_LINK_KIND = "has_fact"
#: 谓词节点（predicate）：把一条**有名字的关系**拆成两个实体共同连到的交会点，于是
#: 「年糕」和「开罐头的声音」之间有了真正可走的路径。节点名就是归一化后的谓词短语，
#: 因此同一句谓词跨会话、跨实体自动合并成同一个节点——交会点是数据自己长出来的，
#: 不是靠共现硬连的。
PREDICATE_NODE_KIND = "predicate"
#: 实体 → 谓词节点的结构边。无语义、不参与关系簿记，与 ``has_fact`` 同级。
PREDICATE_LINK_KIND = "predicate_link"
#: 联想节点（hook）：由 chat 模型对一个实体**可能**的用法/关联做出的预测，例如
#: 「猫」→「开罐头的声音」。两个实体因此可以通过同一个 hook 文本交会。
#:
#: ⚠️ 与知识节点严格区分：hook 是**预测**，不是对话陈述过的事实
#: （``extraction_engine`` 的实体陈述规则第 5 条明确禁止推断）。它带
#: ``basis="model_predicted"``，只用于召回排序加成，绝不能被渲染成"已确认的知识"。
HOOK_NODE_KIND = "hook"
HOOK_LINK_KIND = "hook_link"
#: hook 的 provenance 取值。当前只有"模型预测"一种；留成闭集是为了将来若真能从对话里
#: **观察到**某个关联时，两者不会混成同一类东西。
HOOK_BASES = ("model_predicted",)
DEFAULT_HOOK_BASIS = "model_predicted"
#: 谓词短语的字数上限。开放谓词（"相关""有关"）谈不上是知识，超过这个长度的多半是一句
#: 话而不是一个谓词，两者都在 ``knowledge.normalize_predicate`` 被拒收。
MAX_PREDICATE_CHARS = 24

#: Process-wide relation creation counter (see ``_next_relation_sequence``).
_RELATION_SEQUENCE = 0
_RELATION_SEQUENCE_LOCK = threading.Lock()

#: One lock per topology file, shared by every ``SharedTopology`` instance in this
#: process. The store is a read-modify-write JSON document: ``_save`` writes the whole
#: in-memory document and ``reload`` replaces it, while several writers hold their own
#: instance over the same file — the service's ``pipeline``/``gateway`` topology,
#: ``RoleAuthority``'s (:class:`roles.RoleAuthority`), the management WebUI and the
#: background sweeper. Without serialisation two of them interleave their file work and
#: erase each other's objects: during a document import a chunk lost the fact nodes it
#: had just created, and ``create_fact_node`` then raised ``KeyError('<fact id>')``,
#: aborting the whole import. ``_merge_disk_view`` is the other half of the fix: a save
#: adopts whatever another writer added since this instance last read the file, so a
#: stale instance can no longer delete it.
_TOPOLOGY_LOCKS: Dict[str, threading.RLock] = {}
_TOPOLOGY_LOCKS_GUARD = threading.Lock()


def topology_lock(path: os.PathLike[str] | str) -> threading.RLock:
    """The process-wide lock guarding one topology file."""
    key = os.path.abspath(str(path))
    with _TOPOLOGY_LOCKS_GUARD:
        lock = _TOPOLOGY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _TOPOLOGY_LOCKS[key] = lock
        return lock


def event_link_id(event_id: str) -> str:
    """Deterministic id so re-publishing an event never adds a second link."""
    return f"{EVENT_LINK_KIND}:{event_id}"


def _atomic(method):
    """Run one document operation under the file lock, from start to finish.

    A single mutation is already safe on its own (it ends in ``_save``), but a compound
    operation decides *and* acts on the document several times — write a fact, then
    grant it — and a ``reload``/``restore`` from another thread landing in between is
    what made the fact vanish and the grant raise ``KeyError(<fact id>)``. The lock is
    reentrant, so the nested ``_save``/``grant_node`` calls below it are unaffected.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class SharedTopology:
    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = topology_lock(self.path)
        #: File stamp this instance last read or wrote, so ``_save`` only has to
        #: re-read the document when somebody else actually changed it.
        self._known_stamp: Optional[tuple] = None
        self.data = self._load()

    def _file_stamp(self) -> Optional[tuple]:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            data = {
                "nodes": {},
                "relations": {},
                "event_links": {},
                "merges": {"nodes": {}, "relations": {}, "operations": {}},
                "tombstones": [],
                "deleted_objects": {},
            }
            self._known_stamp = None
            return data
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("nodes", {})
        data.setdefault("relations", {})
        data.setdefault("event_links", {})
        merges = data.setdefault("merges", {})
        merges.setdefault("nodes", {})
        merges.setdefault("relations", {})
        merges.setdefault("operations", {})
        data.setdefault("tombstones", [])
        data.setdefault("deleted_objects", {})
        self._known_stamp = self._file_stamp()
        return data

    def _merge_disk_view(self, disk: Dict[str, Any]) -> None:
        """Reconcile this instance's document with the one on disk before saving.

        Objects another writer **added** are adopted, so a stale instance's save can no
        longer delete knowledge that was published in the meantime (see
        :data:`_TOPOLOGY_LOCKS`). An object this instance already holds keeps its own
        copy, because this instance's version is the one being saved.

        Two things stay authoritative in the other direction, so adoption cannot undo a
        deliberate act: an object tombstoned on disk, or one another instance merged
        away, is removed here rather than put back. A role's whitelist is deliberately
        left alone — a grant removed by an explicit forget must not come back just
        because another instance still had it.
        """
        disk_merges = disk.get("merges") or {}
        merges = self.data.setdefault("merges", {})
        # A tombstone counts whichever view holds it: an object this instance just
        # deleted is already gone from its own document but is still on disk, and it
        # must not be adopted back.
        deleted = {
            **(disk.get("deleted_objects") or {}),
            **(self.data.get("deleted_objects") or {}),
        }
        # Merged-away members are deliberately absent from the active document; the
        # merge mapping is what keeps their history addressable.
        retired = {
            "node": set(merges.get("nodes") or {}) | set(disk_merges.get("nodes") or {}),
            "relation": set(merges.get("relations") or {}) | set(disk_merges.get("relations") or {}),
            "event_link": set(),
        }
        for collection, object_type in (
            ("nodes", "node"),
            ("relations", "relation"),
            ("event_links", "event_link"),
        ):
            target = self.data.setdefault(collection, {})
            # An explicit delete or a completed merge on disk is authoritative: this
            # instance's older copy must not put the object back.
            for object_id in list(target):
                if (
                    f"{object_type}:{object_id}" in (disk.get("deleted_objects") or {})
                    or object_id in retired[object_type]
                ):
                    del target[object_id]
            for object_id, record in (disk.get(collection) or {}).items():
                if (
                    object_id in target
                    or object_id in retired[object_type]
                    or f"{object_type}:{object_id}" in deleted
                ):
                    continue
                target[object_id] = record
        for bucket in ("nodes", "relations", "operations"):
            target = merges.setdefault(bucket, {})
            for key, value in (disk_merges.get(bucket) or {}).items():
                target.setdefault(key, value)
        own_deleted = self.data.setdefault("deleted_objects", {})
        for key, value in deleted.items():
            own_deleted.setdefault(key, value)
        tombstones = self.data.setdefault("tombstones", [])
        known = {
            (item.get("object_type"), item.get("object_id"), item.get("at"))
            for item in tombstones
            if isinstance(item, dict)
        }
        for item in disk.get("tombstones") or []:
            if not isinstance(item, dict):
                continue
            key = (item.get("object_type"), item.get("object_id"), item.get("at"))
            if key in known:
                continue
            tombstones.append(item)
            known.add(key)

    def _save(self, *, merge: bool = True) -> None:
        with self._lock:
            stamp = self._file_stamp()
            if merge and stamp is not None and stamp != self._known_stamp:
                # Somebody else wrote the file since this instance last read it.
                # Adopt their additions before overwriting, or this save would
                # silently delete objects this instance never saw.
                self._merge_disk_view(self._load())
            descriptor, temp_name = tempfile.mkstemp(prefix=".topology.", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                replace_file(temp_name, self.path)
            except Exception:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
            self._known_stamp = self._file_stamp()

    def reload(self) -> None:
        """Re-read the topology from disk.

        Role creation writes through a separate SharedTopology instance (the
        role authority) than the read paths used by the graph display (the
        gateway), so call this before serving the main graph to pick up identity
        nodes and whitelist grants persisted by role bootstrap in the same run.
        """
        with self._lock:
            self.data = self._load()

    def resolve_node(self, node_id: str) -> str:
        current = node_id
        seen = set()
        while current in self.data["merges"]["nodes"] and current not in seen:
            seen.add(current)
            current = self.data["merges"]["nodes"][current]["primary_uuid"]
        return current

    def resolve_relation(self, relation_id: str) -> str:
        current = relation_id
        seen = set()
        while current in self.data["merges"]["relations"] and current not in seen:
            seen.add(current)
            current = self.data["merges"]["relations"][current]["primary_id"]
        return current

    def is_event_node(self, node_id: str) -> bool:
        """True when the id resolves to an event node (never a mergeable entity)."""
        node = self.data["nodes"].get(self.resolve_node(node_id))
        return bool(node and node.get("kind") == EVENT_NODE_KIND)

    def event_node_ids(self) -> List[str]:
        return sorted(
            node_id
            for node_id, node in self.data["nodes"].items()
            if node.get("kind") == EVENT_NODE_KIND
        )

    @_atomic
    def upsert_node(self, node_id: str, name: str, **metadata: Any) -> Dict[str, Any]:
        node_id = self.resolve_node(node_id)
        if node_id in self.data["nodes"] and self.data["nodes"][node_id].get("status") == "deleted":
            raise ValueError(f"node has been permanently deleted: {node_id}")
        kind = str(metadata.get("kind") or "")
        if kind and kind not in NODE_KINDS:
            raise ValueError(f"unsupported node kind: {kind}")
        existing = self.data["nodes"].get(node_id)
        if existing is not None and existing.get("kind") == EVENT_NODE_KIND and kind and kind != EVENT_NODE_KIND:
            raise ValueError(f"event {node_id} cannot be rewritten as a {kind} node")
        node = existing or {"id": node_id, "kind": "node", "created_at": utc_timestamp()}
        node.setdefault("member_uuids", [node_id])
        node.update({"name": name, "updated_at": utc_timestamp(), **metadata})
        self.data["nodes"][node_id] = node
        self._save()
        # The **live** record is returned, not a copy: a caller that adds memory
        # elements to the returned dict must be mutating the stored record, or its
        # elements would be silently dropped on the next save.
        return node

    @_atomic
    def set_node_description(self, node_id: str, role_uuid: str, description: str) -> Dict[str, Any]:
        """Set **one role's** description view of a node (ADR-0150 / ADR-0153).

        The node's ``description`` field is a role-keyed map: the node is shared, so
        a role's understanding of it must not overwrite another role's. Recall reads
        the elements (``node_memory.description_for_role``); this merged view exists
        so the graph display does not have to reassemble them.
        """
        node_id = self.resolve_node(node_id)
        node = self.data["nodes"].get(node_id)
        if node is None:
            raise KeyError(node_id)
        views = node.get("description")
        if not isinstance(views, dict):
            views = {}
            node["description"] = views
        hashes = node.get("description_hash")
        if not isinstance(hashes, dict):
            hashes = {}
            node["description_hash"] = hashes
        stamps = node.get("description_updated_at")
        if not isinstance(stamps, dict):
            stamps = {}
            node["description_updated_at"] = stamps
        views[str(role_uuid)] = str(description or "")
        hashes[str(role_uuid)] = text_hash(views[str(role_uuid)])
        stamps[str(role_uuid)] = utc_timestamp()
        node["updated_at"] = stamps[str(role_uuid)]
        self._save()
        return dict(node)

    # ------------------------------------------------------- node-attribute memory

    def find_record(self, object_id: str) -> Optional[Dict[str, Any]]:
        """The topology record (node or relation) carrying ``object_id``."""
        resolved_node = self.resolve_node(object_id)
        node = self.data["nodes"].get(resolved_node)
        if node is not None:
            return node
        resolved_relation = self.resolve_relation(object_id)
        relation = self.data["relations"].get(resolved_relation)
        if relation is not None:
            return relation
        link = self.data["event_links"].get(resolved_relation)
        return link if link is not None else None

    def memory_records_for_role(self, role_uuid: str) -> List[Dict[str, Any]]:
        """Every whitelisted record that carries memory for this role, projected.

        This is the candidate source for node-attribute recall: the role's
        whitelist is the access boundary, and each record is projected down to this
        role's slice so another role's prose cannot be read. A **memory event**
        qualifies through its own summary body even when the role holds no separate
        element about it.
        """
        from .node_memory import index_text, project_for_role

        records: List[Dict[str, Any]] = []
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"nodes": [], "relations": []})
        for node_id in whitelist.get("nodes", []):
            node = self.data["nodes"].get(self.resolve_node(node_id))
            if node is None or node.get("status") in ("deleted", "invalidated"):
                continue
            projected = project_for_role(node, role_uuid)
            if index_text(projected, role_uuid):
                records.append(projected)
        for relation_id in whitelist.get("relations", []):
            relation = self.data["relations"].get(self.resolve_relation(relation_id))
            if relation is None or relation.get("status") == "deleted":
                continue
            projected = project_for_role(relation, role_uuid)
            if index_text(projected, role_uuid):
                records.append(projected)
        return records

    @_atomic
    def save_record_memory(self, object_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a record whose ``memory`` changed, keeping other roles' slices.

        The incoming record may be a role projection. It is merged back onto the
        stored record per role, so saving one role's slice can never drop another
        role's elements.
        """
        resolved_node = self.resolve_node(object_id)
        target = self.data["nodes"].get(resolved_node)
        collection = self.data["nodes"]
        if target is None:
            resolved_relation = self.resolve_relation(object_id)
            target = self.data["relations"].get(resolved_relation)
            collection = self.data["relations"]
            resolved_node = resolved_relation
        if target is None:
            raise KeyError(object_id)

        incoming_memory = record.get("memory")
        role_keys: List[str] = []
        for field in ("description", "description_hash", "description_updated_at"):
            view = record.get(field)
            if not isinstance(view, dict):
                continue
            for key in view:
                if str(key) not in role_keys:
                    role_keys.append(str(key))
        if isinstance(incoming_memory, list):
            # A projected record: a flat element list belonging to one role. The
            # role travels on the projection (`memory_role`), so the flat list goes
            # back under the right key instead of being merged into another role's.
            role_uuid = str(record.get("memory_role") or "")
            if not role_uuid and role_keys:
                role_uuid = role_keys[0]
            if not role_uuid:
                raise ValueError(
                    "cannot save a flat memory list without a role: use the role-keyed shape"
                )
            stored_memory = target.get("memory")
            if not isinstance(stored_memory, dict):
                stored_memory = {}
            if incoming_memory:
                # An empty list is a projection of a role with nothing to say about
                # this object; writing it would wipe that role's existing slice.
                stored_memory[role_uuid] = incoming_memory
            target["memory"] = stored_memory if stored_memory else target.get("memory")
        elif isinstance(incoming_memory, dict):
            stored_memory = target.get("memory")
            if not isinstance(stored_memory, dict):
                stored_memory = {}
            for role_uuid, elements in incoming_memory.items():
                stored_memory[str(role_uuid)] = elements
            target["memory"] = stored_memory
        # Role-keyed description views are merged the same way.
        for field in ("description", "description_hash", "description_updated_at"):
            incoming = record.get(field)
            if not isinstance(incoming, dict):
                continue
            stored = target.get(field)
            if not isinstance(stored, dict):
                stored = {}
            stored.update({str(key): value for key, value in incoming.items()})
            target[field] = stored
        target["updated_at"] = utc_timestamp()
        collection[resolved_node] = target
        self._save()
        return dict(target)

    def memory_history_for_role(self, role_uuid: str) -> List[Dict[str, Any]]:
        """Invalidated elements of one role, as read-only history.

        Returned as a separate list (never mixed into recall) so "失效认知历史不参与
        召回" keeps holding after the store change, and never containing another
        role's slice.

        The candidate set is the role's whitelist **plus** the objects it holds a
        tombstone for: a relation that was revoked or explicitly deleted is no
        longer in the whitelist, but the role's memory about it is precisely the
        history this view exists to show.
        """
        from .node_memory import elements_for_role

        history: List[Dict[str, Any]] = []
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"nodes": [], "relations": []})
        node_ids: List[str] = list(whitelist.get("nodes", []))
        relation_ids: List[str] = list(whitelist.get("relations", []))
        # A relation the role revoked (or a relation whose node it revoked) is no
        # longer whitelisted, but the role's invalidated memory about it is exactly
        # the history this view exists to show.
        relation_ids += list(whitelist.get("revoked_relations", []))
        for tombstone in self.data.get("tombstones", []):
            # A tombstone may be role-authored (one role revoked its own access) or
            # shared (the object was deleted for everyone). Only the former is
            # filtered; a shared deletion is history for every role that held it.
            author = tombstone.get("role_uuid")
            if author and str(author) != str(role_uuid):
                continue
            kind = str(tombstone.get("object_type") or "")
            if kind in ("node", "event"):
                node_ids.append(str(tombstone.get("object_id") or ""))
            elif kind in ("relation", "event_link"):
                relation_ids.append(str(tombstone.get("object_id") or ""))
        candidates = [self.data["nodes"].get(self.resolve_node(item)) for item in node_ids]
        candidates += [self.data["relations"].get(self.resolve_relation(item)) for item in relation_ids]
        # A relation deleted for everyone is no longer in any whitelist, so its
        # tombstoned record is the only place its memory can still be read from.
        for tombstone in self.data.get("tombstones", []):
            kind = str(tombstone.get("object_type") or "")
            if kind not in ("relation", "event_link", "node", "event"):
                continue
            record = self.deleted_record(kind, str(tombstone.get("object_id") or ""))
            if record is not None:
                candidates.append(record)
        for record in candidates:
            if record is None:
                continue
            for element in elements_for_role(record, role_uuid):
                if element.get("status") != "invalidated":
                    continue
                history.append(
                    {
                        "object_id": record.get("id"),
                        "object_type": "relation" if "endpoints" in record else "node",
                        "object_name": record.get("name", ""),
                        **element,
                    }
                )
        history.sort(
            key=lambda item: (str(item.get("invalidated_at") or ""), str(item.get("id") or "")),
            reverse=True,
        )
        return history

    @staticmethod
    def _next_relation_sequence() -> int:
        """A process-wide monotonic counter for relation creation order."""
        global _RELATION_SEQUENCE
        with _RELATION_SEQUENCE_LOCK:
            _RELATION_SEQUENCE += 1
            return _RELATION_SEQUENCE

    #: Relation kind that links a memory node to an entity it mentions. It is the
    #: only way an event node may take part in an ordinary relation: the event is a
    #: memory unit, and this edge is what makes the entity reachable from it.
    MEMORY_LINK_KIND = "memory_link"

    @_atomic
    def upsert_relation(self, source_id: str, target_id: str, **metadata: Any) -> Dict[str, Any]:
        source_id = self.resolve_node(source_id)
        target_id = self.resolve_node(target_id)
        if source_id == target_id:
            raise ValueError("self relations are not supported")
        kind = str(metadata.get("kind") or "relation")
        for endpoint in (source_id, target_id):
            if self.is_event_node(endpoint) and kind != self.MEMORY_LINK_KIND:
                # A memory event's ownership edge is created by
                # ``link_event_to_role`` and carries no semantics; its only other
                # allowed edge is an explicit memory link to an entity.
                raise ValueError(
                    f"event {endpoint} cannot participate in an ordinary relation; "
                    "use link_event_to_role or a memory_link"
                )
        pair = stable_pair(source_id, target_id)
        # Identity of an edge:
        #
        # * structural kinds (has_fact / predicate_link / hook_link / memory_link /
        #   companion / qq_contact / event_link) stay **pair-unique** — their meaning
        #   is "these two objects are linked", and a second copy would only duplicate
        #   bookkeeping;
        # * a semantic relation (kind="relation") is identified by
        #   (pair, predicate). Two entities can be related in more than one way
        #   ("主人-养-年糕" and "主人-喜欢-年糕"), and collapsing them onto the first
        #   edge silently lost the second predicate. The predicate node still exists
        #   for cross-entity merging ("同一句谓词跨实体自动合并"), so the graph now
        #   keeps both the multi-edge facts and the shared 交会点.
        semantic = kind == "relation"
        name_key = normalized_text(metadata.get("name") or "") if semantic else ""
        for relation in self.data["relations"].values():
            if tuple(relation["endpoints"]) != pair:
                continue
            if str(relation.get("kind") or "relation") != kind:
                continue
            if semantic and normalized_text(relation.get("name") or "") != name_key:
                continue
            return dict(relation)
        relation = {
            "id": typed_id("relation"),
            "kind": kind,
            "endpoints": list(pair),
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            # A monotonic creation order. `created_at` only has second resolution,
            # so two relations created in the same second would otherwise be
            # indistinguishable; a merge that collapses them onto one pair has to
            # pick a survivor deterministically, and "the oldest edge" is the one a
            # role is most likely to have accumulated memory on.
            "sequence": self._next_relation_sequence(),
            **metadata,
        }
        self.data["relations"][relation["id"]] = relation
        self._save()
        return dict(relation)

    # -------------------------------------------------------------------- facts

    def is_fact_node(self, node_id: str) -> bool:
        """True when the id resolves to a fact node (never a mergeable entity)."""
        node = self.data["nodes"].get(self.resolve_node(node_id))
        return bool(node and node.get("kind") == FACT_NODE_KIND)

    def fact_node_ids(self) -> List[str]:
        return sorted(
            node_id
            for node_id, node in self.data["nodes"].items()
            if node.get("kind") == FACT_NODE_KIND
        )

    def fact_records_for_role(
        self, role_uuid: str, *, include_invalidated: bool = True
    ) -> List[Dict[str, Any]]:
        """Every fact node this role may read, projected for that role.

        ``include_invalidated`` defaults to True for management/diagnostics. Recall
        must pass False: an invalidated fact is history, not an answer.
        """
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"nodes": [], "relations": []})
        records = []
        for node_id in whitelist.get("nodes", []):
            node = self.data["nodes"].get(self.resolve_node(node_id))
            if node is None or node.get("status") == "deleted":
                continue
            if node.get("kind") != FACT_NODE_KIND:
                continue
            if (
                not include_invalidated
                and str(node.get("status") or "active") != "active"
            ):
                continue
            records.append(project_for_role(node, role_uuid))
        return records

    def fact_id(self, role_uuid: str, parent_id: str, text: str) -> str:
        """Deterministic fact id so replaying a batch never duplicates one.

        Keyed by the role, the parent entity and the statement text: the same
        sentence about the same entity from the same role is one fact, and
        re-publishing a frozen batch must be a no-op (ADR-0141).
        """
        seed = "fact:{role}:{parent}:{text}".format(
            role=role_uuid, parent=self.resolve_node(parent_id), text=text_hash(text)
        )
        return f"{'f_'}{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    @_atomic
    def create_fact_node(
        self,
        role_uuid: str,
        parent_id: str,
        text: str,
        *,
        kind: str = "fact",
        source_conversation_id: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (or reuse) one fact node, its parent link and both grants.

        Idempotent by ``fact_id``: an existing fact keeps its creation stamp and its
        status, so a replay cannot resurrect an invalidated fact or fork the same
        sentence into two nodes.
        """
        if not role_uuid:
            raise ValueError("fact role_uuid is required")
        statement = " ".join(str(text or "").split())
        if not statement:
            raise ValueError("fact text is required")
        parent_id = self.resolve_node(parent_id)
        parent = self.data["nodes"].get(parent_id)
        if parent is None:
            raise KeyError(f"fact parent not found: {parent_id}")
        if parent.get("kind") == FACT_NODE_KIND:
            raise ValueError(f"a fact cannot be attached to another fact: {parent_id}")
        identifier = fact_id or self.fact_id(role_uuid, parent_id, statement)
        now = utc_timestamp()
        node = self.data["nodes"].get(identifier)
        if node is None:
            node = self.upsert_node(
                identifier,
                statement,
                kind=FACT_NODE_KIND,
                # The node kind is ``fact`` (it is a non-entity object); the kind of
                # the *statement* (fact / preference / correction / relation) is kept
                # separately so the management view and recall can still tell them
                # apart after 节点细化.
                fact_kind=str(kind or "fact"),
                text=statement,
                parent_id=parent_id,
                role_uuid=str(role_uuid),
                status="active",
                content_hash=text_hash(statement),
                source_conversation_id=source_conversation_id,
                created_at=now,
            )
            link = self.upsert_relation(parent_id, identifier, kind=PARENT_LINK_KIND)
            self.data["relations"][link["id"]]["structural"] = True
            self._save()
        else:
            link = self._existing_relation(parent_id, identifier)
            if link is None:
                created = self.upsert_relation(parent_id, identifier, kind=PARENT_LINK_KIND)
                self.data["relations"][created["id"]]["structural"] = True
                link = created
            else:
                link = dict(link)
        self.grant_node(role_uuid, identifier)
        self.grant_node(role_uuid, parent_id)
        self.grant_relation(role_uuid, str(link["id"]))
        # The node this call wrote is returned directly instead of being read back out
        # of the document: that re-read was the one place a whole-document swap could
        # turn a successful write into a bare ``KeyError(<fact id>)``, which aborted an
        # entire document-import chunk (see :data:`_TOPOLOGY_LOCKS`).
        return dict(node)

    @_atomic
    def create_perspective_node(
        self,
        role_uuid: str,
        speaker_id: str,
        text: str,
        *,
        fact_kind: str = SELF_STATEMENT_KIND,
        target_id: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an AI-spoken self-statement or opinion.

        These are deliberately not ordinary facts: a fact's parent_id is the
        entity the sentence is about, which loses the speaker. A self statement
        has no entity subject at all; an opinion is a relay between the AI
        (speaker_id) and the entity it evaluates (target_id). Both keep
        kind="fact" so they are indexed and gated exactly like a fact, while
        fact_kind tells recall how to render and navigate them.

        No has_fact edge is written. A self statement has no parent; an opinion's
        relay is expressed by its fields and by the category graph's
        speaker--target edge, not by pretending the target owns the AI's voice.
        """
        if not role_uuid:
            raise ValueError("perspective role_uuid is required")
        kind = str(fact_kind or "").strip()
        if kind not in PERSPECTIVE_KINDS:
            raise ValueError(f"unsupported perspective kind: {kind}")
        statement = " ".join(str(text or "").split())
        if not statement:
            raise ValueError("perspective text is required")
        speaker_id = self.resolve_node(speaker_id)
        speaker = self.data["nodes"].get(speaker_id)
        if speaker is None:
            raise KeyError(f"perspective speaker not found: {speaker_id}")
        if str(speaker.get("kind") or "") != "ai_role":
            raise ValueError(f"perspective speaker must be an ai_role: {speaker_id}")
        target_id = self.resolve_node(target_id) if target_id else ""
        if kind == OPINION_KIND:
            if not target_id:
                raise ValueError("an opinion needs a target entity")
            target = self.data["nodes"].get(target_id)
            if target is None:
                raise KeyError(f"opinion target not found: {target_id}")
            if str(target.get("kind") or "") in (FACT_NODE_KIND, "predicate", "hook", "event"):
                raise ValueError(f"opinion target must be an entity: {target_id}")
        else:
            target_id = ""
        seed = "perspective:{role}:{kind}:{speaker}:{target}:{text}".format(
            role=role_uuid,
            kind=kind,
            speaker=speaker_id,
            target=target_id,
            text=text_hash(statement),
        )
        identifier = f"f_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"
        now = utc_timestamp()
        node = self.data["nodes"].get(identifier)
        if node is None:
            node = self.upsert_node(
                identifier,
                statement,
                kind=FACT_NODE_KIND,
                fact_kind=kind,
                text=statement,
                parent_id="",
                speaker_id=speaker_id,
                **({"target_id": target_id} if target_id else {}),
                role_uuid=str(role_uuid),
                status="active",
                content_hash=text_hash(statement),
                source_conversation_id=source_conversation_id,
                created_at=now,
            )
        self.grant_node(role_uuid, identifier)
        self.grant_node(role_uuid, speaker_id)
        if target_id:
            self.grant_node(role_uuid, target_id)
        return dict(node)

    def _existing_relation(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        pair = stable_pair(str(source_id), str(target_id))
        for relation in self.data["relations"].values():
            if tuple(relation.get("endpoints") or []) == pair:
                return relation
        return None

    @_atomic
    def invalidate_fact_node(self, fact_id: str, *, reason: str = "") -> Dict[str, Any]:
        """Mark one fact invalidated; it stays as read-only history, never recalled."""
        fact_id = self.resolve_node(fact_id)
        node = self.data["nodes"].get(fact_id)
        if node is None:
            raise KeyError(fact_id)
        if node.get("kind") != FACT_NODE_KIND:
            raise ValueError(f"{fact_id} is not a fact node")
        node["status"] = "invalidated"
        node["invalidated_at"] = utc_timestamp()
        node["updated_at"] = node["invalidated_at"]
        if reason:
            node["invalidated_reason"] = str(reason)
        self._save()
        return dict(node)

    # -------------------------------------------------------------- predicates
    #
    # 谓词节点的全部意义：让"有关"的两个实体有**一条真的走得通的路径**。做法是把一条
    # 有名字的关系拆成两个实体共同连到的交会点，而不是给两个实体直接加一条边——直接加
    # 边在 ADR-0047/0049（共享边无语义、每对节点只有一条边）下既表达不了谓词，也留不住
    # "它们**因为什么**有关"。节点名就是归一化后的谓词文本，所以同一句谓词跨会话自动
    # 合并成同一个节点，交会点是数据自己长出来的。

    def is_predicate_node(self, node_id: str) -> bool:
        """True when the id resolves to a predicate node (a relation's shared subject)."""
        node = self.data["nodes"].get(self.resolve_node(node_id))
        return bool(node and node.get("kind") == PREDICATE_NODE_KIND)

    def is_hook_node(self, node_id: str) -> bool:
        """True when the id resolves to an associative hook node (model-predicted)."""
        node = self.data["nodes"].get(self.resolve_node(node_id))
        return bool(node and node.get("kind") == HOOK_NODE_KIND)

    def is_attachable_node(self, node_id: str) -> bool:
        """True when a fact/predicate/hook may hang off this node.

        The one thing that must never gain children is a fact: a fact about a fact has
        no entity to be recalled from. Predicates and hooks are deliberately allowed,
        so "开罐头的声音" can carry its own knowledge later.
        """
        resolved = self.resolve_node(node_id)
        return self.is_fact_node(resolved) is False and resolved in self.data["nodes"]

    @staticmethod
    def predicate_key(text: Any) -> str:
        """The normalized text that *is* a predicate node's identity.

        Two relations spelled differently by the model ("开罐头的声音" / " 开罐头的声音 ")
        must land on one node, or the交会点 silently forks and the link is lost. Only
        whitespace is normalized: any more aggressive rewrite would merge predicates
        that a role may genuinely need to tell apart.
        """
        return " ".join(str(text or "").split())

    def find_predicate_node(self, text: Any) -> Optional[Dict[str, Any]]:
        """The existing predicate node whose text equals ``text`` (normalized)."""
        key = self.predicate_key(text)
        if not key:
            return None
        for node in self.data["nodes"].values():
            if node.get("kind") != PREDICATE_NODE_KIND:
                continue
            if self.predicate_key(node.get("name")) == key:
                return node
        return None

    def predicate_node_id(self, text: Any) -> str:
        """Deterministic id for a predicate text, so re-publishing never forks it."""
        key = self.predicate_key(text)
        if not key:
            raise ValueError("predicate text is required")
        seed = "predicate:{text}".format(text=text_hash(key))
        return f"p_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    @_atomic
    def create_predicate_node(
        self,
        role_uuid: str,
        text: Any,
        *,
        source_conversation_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (or reuse) the shared predicate node for one relation's text.

        Idempotent by normalized text: the same predicate published twice is one node,
        which is exactly what makes two entities meet there. Returns the **existing**
        node when the text was already published, so a replay is a no-op.
        """
        if not role_uuid:
            raise ValueError("predicate role_uuid is required")
        key = self.predicate_key(text)
        if not key:
            raise ValueError("predicate text is required")
        identifier = self.predicate_node_id(key)
        existing = self.find_predicate_node(key)
        if existing is not None:
            identifier = str(existing.get("id") or identifier)
        node = self.data["nodes"].get(identifier)
        if node is None:
            node = self.upsert_node(
                identifier,
                str(name or key),
                kind=PREDICATE_NODE_KIND,
                text=key,
                status="active",
                content_hash=text_hash(key),
                source_conversation_id=source_conversation_id,
                created_at=utc_timestamp(),
            )
        self.grant_node(role_uuid, identifier)
        return dict(node)

    @_atomic
    def link_entity_to_predicate(
        self,
        role_uuid: str,
        entity_id: str,
        predicate_id: str,
    ) -> Dict[str, Any]:
        """Link one entity to a predicate node and put both in the role's whitelist.

        Idempotent: the pair carries at most one edge (ADR-0049), so a replay reuses
        the existing link instead of creating a duplicate.
        """
        self.grant_node(role_uuid, entity_id)
        self.grant_node(role_uuid, predicate_id)
        link = self._existing_relation(entity_id, predicate_id)
        if link is None:
            link = self.upsert_relation(entity_id, predicate_id, kind=PREDICATE_LINK_KIND)
            relation = self.data["relations"].get(str(link["id"]))
            if relation is not None:
                relation["structural"] = False
                self._save()
        self.grant_relation(role_uuid, str(link["id"]))
        return dict(link)

    # ------------------------------------------------------------------- hooks

    def hook_id(
        self,
        role_uuid: str,
        parent_id: str,
        text: Any,
        *,
        basis: str = DEFAULT_HOOK_BASIS,
    ) -> str:
        """Deterministic hook id so re-publishing a batch never duplicates a hook.

        Keyed by role + parent + normalized hook text + basis. Two entities that share
        the *same hook text* still get two distinct hook nodes — they meet through the
        text, and merging them would fuse two entities' associations into one object.
        """
        key = self.predicate_key(text)
        seed = "hook:{role}:{parent}:{text}:{basis}".format(
            role=role_uuid,
            parent=self.resolve_node(parent_id),
            text=text_hash(key),
            basis=str(basis or DEFAULT_HOOK_BASIS),
        )
        return f"{'h_'}{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    @_atomic
    def create_hook_node(
        self,
        role_uuid: str,
        parent_id: str,
        text: Any,
        *,
        basis: str = DEFAULT_HOOK_BASIS,
        confidence: Optional[float] = None,
        source_conversation_id: Optional[str] = None,
        hook_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (or reuse) one model-predicted associative hook of one entity.

        The hook hangs off its parent entity with a ``hook_link`` edge and records
        ``parent_id``, exactly like a fact — that is what lets recall score it and
        attribute the hit back to the entity it belongs to.

        ``basis`` is a closed set and defaults to ``model_predicted``: a hook is a
        prediction, and this field is the only thing that keeps it from being read as
        a statement the conversation made. See the module docstring warning.
        """
        if not role_uuid:
            raise ValueError("hook role_uuid is required")
        key = self.predicate_key(text)
        if not key:
            raise ValueError("hook text is required")
        normalized_basis = str(basis or DEFAULT_HOOK_BASIS)
        if normalized_basis not in HOOK_BASES:
            raise ValueError(f"unsupported hook basis: {basis}")
        parent_id = self.resolve_node(parent_id)
        parent = self.data["nodes"].get(parent_id)
        if parent is None:
            raise KeyError(f"hook parent not found: {parent_id}")
        if not self.is_attachable_node(parent_id):
            raise ValueError(f"a hook cannot be attached to this node: {parent_id}")
        identifier = hook_id or self.hook_id(
            role_uuid, parent_id, key, basis=normalized_basis
        )
        if identifier not in self.data["nodes"]:
            self.upsert_node(
                identifier,
                key,
                kind=HOOK_NODE_KIND,
                text=key,
                parent_id=parent_id,
                role_uuid=str(role_uuid),
                basis=normalized_basis,
                confidence=confidence,
                status="active",
                content_hash=text_hash(key),
                source_conversation_id=source_conversation_id,
                created_at=utc_timestamp(),
            )
        node = self.data["nodes"].get(identifier)
        if node is None:
            # Only reachable if the document was replaced from under this call; the
            # hook is honestly reported as unattachable instead of raising a bare id.
            raise KeyError(f"hook node not found after write: {identifier}")
        link = self._existing_relation(parent_id, identifier)
        if link is None:
            created = self.upsert_relation(parent_id, identifier, kind=HOOK_LINK_KIND)
            relation = self.data["relations"].get(str(created["id"]))
            if relation is not None:
                relation["structural"] = False
                self._save()
            link = created
        self.grant_node(role_uuid, identifier)
        self.grant_node(role_uuid, parent_id)
        self.grant_relation(role_uuid, str(link["id"]))
        return dict(node)

    def hook_records_for_role(self, role_uuid: str) -> List[Dict[str, Any]]:
        """Every hook node this role may read, projected for that role."""
        return [
            project_for_role(node, role_uuid)
            for node in self.data["nodes"].values()
            if node.get("kind") == HOOK_NODE_KIND
            and self.is_allowed(role_uuid, str(node.get("id") or ""))
        ]

    # ------------------------------------------------------------------- events

    def create_event_node(
        self,
        role_uuid: str,
        event_id: str,
        summary: str,
        *,
        occurred_at: Optional[str] = None,
        formed_at: Optional[str] = None,
        source_kind: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create the event node and its single role link as one write.

        The event node carries only stable identity plus summary/time/vector
        attributes. It is deliberately not created through
        :meth:`upsert_relation`: the ownership edge has no semantic field and is
        not part of ordinary relation bookkeeping.
        """
        if not role_uuid:
            raise ValueError("event role_uuid is required")
        if not event_id:
            raise ValueError("event id is required")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("event summary cannot be empty")
        if role_uuid not in self.data["nodes"] or self.data["nodes"][role_uuid].get("kind") != "ai_role":
            raise KeyError(f"event role node not found: {role_uuid}")
        metadata: Dict[str, Any] = {
            "kind": EVENT_NODE_KIND,
            "role_uuid": role_uuid,
            "summary": summary.strip(),
            "occurred_at": occurred_at,
            "formed_at": formed_at or utc_timestamp(),
            "source_kind": source_kind,
            "source_conversation_id": source_conversation_id,
            "content_hash": content_hash,
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
        }
        node = self.upsert_node(event_id, summary.strip(), **metadata)
        self.link_event_to_role(role_uuid, event_id)
        return node

    @_atomic
    def link_event_to_role(self, role_uuid: str, event_id: str) -> Dict[str, Any]:
        """Idempotently ensure the event's one semantic-free link to its role."""
        event_id = self.resolve_node(event_id)
        role_uuid = self.resolve_node(role_uuid)
        node = self.data["nodes"].get(event_id)
        if node is None:
            raise KeyError(event_id)
        if node.get("kind") != EVENT_NODE_KIND:
            raise ValueError(f"{event_id} is not an event node")
        owner = node.get("role_uuid")
        if owner and owner != role_uuid:
            raise ValueError(f"event {event_id} already belongs to role {owner}")
        node["role_uuid"] = role_uuid
        link_id = event_link_id(event_id)
        link = self.data["event_links"].get(link_id)
        if link is None:
            # Repair a legacy id instead of creating a second link for one event.
            existing = [
                item for item in self.data["event_links"].values()
                if event_id in item.get("endpoints", [])
            ]
            if existing:
                link = existing[0]
            else:
                link = {
                    "id": link_id,
                    "kind": EVENT_LINK_KIND,
                    "endpoints": [event_id, role_uuid],
                    "created_at": utc_timestamp(),
                    "updated_at": utc_timestamp(),
                }
                self.data["event_links"][link_id] = link
        self._save()
        return dict(link)

    def event_link_for(self, event_id: str) -> Optional[Dict[str, Any]]:
        event_id = self.resolve_node(event_id)
        for link in self.data["event_links"].values():
            if event_id in link.get("endpoints", []):
                return dict(link)
        return None

    def event_links_for_role(self, role_uuid: str) -> List[Dict[str, Any]]:
        return [
            dict(link)
            for link in self.data["event_links"].values()
            if role_uuid in link.get("endpoints", [])
        ]

    def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        relation = self.data["relations"].get(self.resolve_relation(relation_id))
        if relation is not None:
            return dict(relation)
        link = self.data["event_links"].get(relation_id)
        return dict(link) if link else None


    @_atomic
    def grant_relation(self, role_uuid: str, relation_id: str) -> None:
        relation_id = self.resolve_relation(relation_id)
        relation = self.get_relation(relation_id)
        if not relation:
            raise KeyError(relation_id)
        whitelist = self.data.setdefault("whitelist", {}).setdefault(role_uuid, {"nodes": [], "relations": []})
        if relation_id not in whitelist["relations"]:
            whitelist["relations"].append(relation_id)
        for node_id in relation["endpoints"]:
            if node_id not in whitelist["nodes"]:
                whitelist["nodes"].append(node_id)
        self._save()

    @_atomic
    def grant_node(self, role_uuid: str, node_id: str) -> None:
        node_id = self.resolve_node(node_id)
        node = self.data["nodes"].get(node_id)
        if node is None:
            raise KeyError(node_id)
        if self._is_foreign_identity(role_uuid, node_id, node):
            # Another partition's identity is not this role's knowledge. A name
            # lookup can resolve a shared appellation ("主人") or a role name onto
            # the other role's identity node; granting it would render that other
            # AI inside this role's 角色视角 graph (and read its memory slice).
            # Dropping the grant keeps the whitelist a pure access closure.
            return
        whitelist = self.data.setdefault("whitelist", {}).setdefault(role_uuid, {"nodes": [], "relations": []})
        if node_id not in whitelist["nodes"]:
            whitelist["nodes"].append(node_id)
        self._save()

    @staticmethod
    def _is_foreign_identity(role_uuid: str, node_id: str, node: Dict[str, Any]) -> bool:
        """True when a node is another role's fixed identity (AI node or user node)."""
        role = str(role_uuid or "")
        candidate = str(node_id)
        if str(node.get("kind") or "") == "ai_role":
            return bool(role) and candidate != role
        return is_foreign_user_node_id(role, candidate)

    def is_allowed(self, role_uuid: str, object_id: str) -> bool:
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"nodes": [], "relations": []})
        node_id = self.resolve_node(object_id)
        if node_id in whitelist.get("nodes", []):
            return True
        relation_id = self.resolve_relation(object_id)
        if relation_id not in whitelist.get("relations", []):
            return False
        relation = self.data["relations"].get(relation_id)
        if relation is None:
            # Event links are role-owned rather than whitelist-closure based;
            # the event node itself must also be granted.
            link = self.data["event_links"].get(relation_id)
            if link is None:
                return False
            event_id = next(
                (item for item in link.get("endpoints", []) if item != role_uuid), None
            )
            return bool(event_id and event_id in whitelist.get("nodes", []))
        return bool(relation and set(relation["endpoints"]).issubset(set(whitelist.get("nodes", []))))


    def relation_ids_for_role(self, role_uuid: str) -> List[str]:
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"relations": []})
        return [relation_id for relation_id in whitelist.get("relations", []) if self.is_allowed(role_uuid, relation_id)]

    def nodes_for_relation(self, relation_id: str) -> List[str]:
        relation = self.get_relation(relation_id)
        return list(relation.get("endpoints", [])) if relation else []

    @_atomic
    def revoke_relation(self, role_uuid: str, relation_id: str) -> None:
        """Drop a relation from the role's whitelist, keeping it as forgotten history.

        The id moves to ``revoked_relations`` rather than disappearing, so the role's
        invalidated memory about it stays readable through
        :meth:`memory_history_for_role` while recall (which reads the whitelist) can
        no longer reach it.
        """
        relation_id = self.resolve_relation(relation_id)
        whitelist = self.data.get("whitelist", {}).get(role_uuid)
        if not whitelist:
            return
        whitelist["relations"] = [item for item in whitelist.get("relations", []) if item != relation_id]
        revoked = whitelist.setdefault("revoked_relations", [])
        if relation_id not in revoked:
            revoked.append(relation_id)
        self._save()

    @_atomic
    def revoke_node(self, role_uuid: str, node_id: str) -> List[str]:
        node_id = self.resolve_node(node_id)
        whitelist = self.data.get("whitelist", {}).get(role_uuid)
        if not whitelist:
            return []
        removed_relations = []
        for relation_id in list(whitelist.get("relations", [])):
            relation = self.get_relation(relation_id)
            if relation and node_id in relation["endpoints"]:
                removed_relations.append(relation_id)
        whitelist["relations"] = [item for item in whitelist.get("relations", []) if item not in removed_relations]
        whitelist["nodes"] = [item for item in whitelist.get("nodes", []) if item != node_id]
        revoked = whitelist.setdefault("revoked_relations", [])
        for relation_id in removed_relations:
            if relation_id not in revoked:
                revoked.append(relation_id)
        self._save()
        return removed_relations

    @_atomic
    def delete_relation(self, relation_id: str, reason: str = "explicit_delete", operation_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete one relation from every role, keeping its record as history.

        The record moves to ``deleted_objects`` instead of being dropped: a role's
        memory about the relation stays readable through
        :meth:`memory_history_for_role` (the data is tiny and a deletion is
        irreversible, so keeping the tombstoned text is the choice that cannot be
        regretted), while ordinary reads — the whitelist, ``topology_for_role`` and
        recall — can no longer reach it.
        """
        canonical_id = self.resolve_relation(relation_id)
        stored = self.data["relations"].pop(canonical_id, None)
        if stored is None:
            link = self.data["event_links"].pop(canonical_id, None)
            if link is None:
                return {"relation_id": canonical_id, "status": "already_deleted"}
            for whitelist in self.data.get("whitelist", {}).values():
                whitelist["relations"] = [item for item in whitelist.get("relations", []) if item != canonical_id]
            self._store_deleted("event_link", canonical_id, link, reason, operation_id)
            return {
                "relation_id": canonical_id,
                "status": "deleted",
                "relation": link,
                "tombstone": self.data["tombstones"][-1],
            }
        for whitelist in self.data.get("whitelist", {}).values():
            whitelist["relations"] = [item for item in whitelist.get("relations", []) if item != canonical_id]
        self._store_deleted("relation", canonical_id, stored, reason, operation_id)
        return {
            "relation_id": canonical_id,
            "status": "deleted",
            "relation": stored,
            "tombstone": self.data["tombstones"][-1],
        }

    @_atomic
    def _store_deleted(
        self,
        object_type: str,
        object_id: str,
        record: Dict[str, Any],
        reason: str,
        operation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a tombstoned object and keep its last content addressable."""
        tombstones = [
            item
            for item in self.data.get("tombstones", [])
            if not (item.get("object_type") == object_type and item.get("object_id") == object_id)
        ]
        tombstone = {
            "object_type": object_type,
            "object_id": object_id,
            "reason": reason,
            "operation_id": operation_id,
            "at": utc_timestamp(),
        }
        tombstones.append(tombstone)
        self.data["tombstones"] = tombstones
        deleted = self.data.setdefault("deleted_objects", {})
        deleted[f"{object_type}:{object_id}"] = copy.deepcopy(record)
        self._save()
        return tombstone

    def deleted_record(self, object_type: str, object_id: str) -> Optional[Dict[str, Any]]:
        """The last stored content of a tombstoned object, for history views."""
        record = (self.data.get("deleted_objects") or {}).get(f"{object_type}:{object_id}")
        return copy.deepcopy(record) if record is not None else None

    @_atomic
    def purge_role(self, role_uuid: str) -> Dict[str, Any]:
        """Remove one role's partition from the shared graph, in one document write.

        Deleting a role used to mean deleting its directory and its registry entry,
        which left **every object it had published** behind in the shared document: a
        document import that failed (or was cancelled before confirmation) kept
        rendering its two identity nodes as 「未命名角色」 in the global graph together
        with all of its facts, hooks, predicates and relations, and nothing could reach
        them any more because the partition they belonged to was gone.

        Everything this role owned is removed and tombstoned in the same write, so a
        stale in-process instance of the document cannot merge the objects back (see
        :meth:`_merge_disk_view`). Anything another role also holds is only un-granted
        from this role: the shared graph stays shared, and an object another partition
        can still read is never deleted on its behalf.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required to purge a partition")
        nodes = self.data.setdefault("nodes", {})
        relations = self.data.setdefault("relations", {})
        links = self.data.setdefault("event_links", {})
        whitelist = self.data.setdefault("whitelist", {})
        had_partition = role in whitelist
        entry = whitelist.pop(role, None)
        entry = entry if isinstance(entry, dict) else {"nodes": [], "relations": []}
        others = [item for other, item in whitelist.items() if other != role and isinstance(item, dict)]
        held_nodes = {str(item) for other in others for item in other.get("nodes", [])}
        held_relations = {str(item) for other in others for item in other.get("relations", [])}

        owned_nodes = {str(item) for item in (entry.get("nodes") or [])}
        owned_nodes |= {role, user_node_id(role)}
        # A fact, predicate or hook whose parent entity is going away has no anchor
        # left: it would render as an orphan object with a dangling ``parent_id``.
        candidates = set(owned_nodes)
        while True:
            grown = {
                str(node_id)
                for node_id, node in nodes.items()
                if str(node.get("parent_id") or "") in candidates
            }
            if grown <= candidates:
                break
            candidates |= grown
        removed_nodes = {str(item) for item in candidates if str(item) in nodes and str(item) not in held_nodes}

        relation_candidates = {str(item) for item in (entry.get("relations") or [])}
        for relation_id, relation in relations.items():
            if removed_nodes & {str(item) for item in relation.get("endpoints", [])}:
                relation_candidates.add(str(relation_id))
        removed_relations = {
            str(item) for item in relation_candidates
            if str(item) in relations and str(item) not in held_relations
        }
        owned_links = {
            str(link_id) for link_id, link in links.items()
            if role in [str(item) for item in link.get("endpoints", [])]
        }
        removed_links = {str(item) for item in owned_links if str(item) not in held_relations}

        reason = "role_purge"
        deleted = self.data.setdefault("deleted_objects", {})
        purge_keys = (
            {f"node:{item}" for item in removed_nodes}
            | {f"relation:{item}" for item in removed_relations | removed_links}
            | {f"event_link:{item}" for item in removed_links}
        )
        self.data["tombstones"] = [
            item for item in self.data.get("tombstones", [])
            if f"{item.get('object_type')}:{item.get('object_id')}" not in purge_keys
        ]
        tombstones = self.data["tombstones"]
        for node_id in sorted(removed_nodes):
            deleted[f"node:{node_id}"] = copy.deepcopy(nodes.pop(node_id))
            tombstones.append({"object_type": "node", "object_id": node_id, "reason": reason,
                               "operation_id": None, "at": utc_timestamp()})
        for relation_id in sorted(removed_relations):
            deleted[f"relation:{relation_id}"] = copy.deepcopy(relations.pop(relation_id))
            tombstones.append({"object_type": "relation", "object_id": relation_id, "reason": reason,
                               "operation_id": None, "at": utc_timestamp()})
        for link_id in sorted(removed_links):
            record = copy.deepcopy(links.pop(link_id))
            # Addressable both ways: relation bookkeeping resolves an event link by
            # relation id, event bookkeeping by its own id.
            deleted[f"event_link:{link_id}"] = copy.deepcopy(record)
            deleted[f"relation:{link_id}"] = record
            tombstones.append({"object_type": "relation", "object_id": link_id, "reason": reason,
                               "operation_id": None, "at": utc_timestamp()})

        gone = removed_nodes | removed_relations | removed_links
        merges = self.data.setdefault("merges", {"nodes": {}, "relations": {}, "operations": {}})
        merges["nodes"] = {
            member: record for member, record in (merges.get("nodes") or {}).items()
            if member not in gone and str((record or {}).get("primary_uuid") or "") not in gone
        }
        merges["relations"] = {
            member: record for member, record in (merges.get("relations") or {}).items()
            if member not in gone and str((record or {}).get("primary_id") or "") not in gone
        }
        for node in nodes.values():
            members = [item for item in node.get("member_uuids") or [] if str(item) not in gone]
            if members:
                node["member_uuids"] = members
            else:
                node.pop("member_uuids", None)
        # Other roles may still list a purged id they never held (a stale grant is
        # possible for a relation whose endpoints changed). Dropping those keeps every
        # whitelist a closure over objects that exist.
        for other, item in whitelist.items():
            if other == role or not isinstance(item, dict):
                continue
            item["nodes"] = [i for i in item.get("nodes", []) if str(i) not in gone]
            item["relations"] = [i for i in item.get("relations", []) if str(i) not in gone]
        self._save()
        return {
            "role_uuid": role,
            "removed_nodes": sorted(removed_nodes),
            "removed_relations": sorted(removed_relations),
            "removed_event_links": sorted(removed_links),
            "kept_shared_nodes": sorted(str(item) for item in candidates if str(item) in held_nodes),
            "kept_shared_relations": sorted(
                str(item) for item in relation_candidates | owned_links if str(item) in held_relations
            ),
            "had_partition": had_partition,
        }

    @_atomic
    def merge_nodes(self, primary_id: str, member_id: str, operation_id: Optional[str] = None) -> Dict[str, Any]:
        primary_id = self.resolve_node(primary_id)
        member_id = self.resolve_node(member_id)
        if primary_id == member_id:
            return dict(self.data["nodes"].get(primary_id, {"id": primary_id, "member_uuids": [primary_id]}))
        for node_id in (primary_id, member_id):
            if self.is_event_node(node_id):
                raise ValueError(
                    f"event node {node_id} cannot be merged: events are immutable "
                    "independent memory units"
                )
            node = self.data["nodes"].get(node_id) or {}
            if str(node.get("kind") or "") in ("ai_role", "user"):
                # An identity node is this partition's fixed subject: the role's own
                # AI node anchors its events, its whitelist and its Neo4j node, and a
                # user node is scoped to exactly one role. Merging one away silently
                # deletes that identity (and any role that depended on it), so the
                # semantic-merge operation refuses it instead.
                raise ValueError(
                    f"identity node {node_id} cannot be merged: role and user "
                    "identities are fixed"
                )
        if primary_id not in self.data["nodes"] or member_id not in self.data["nodes"]:
            raise KeyError("both merge nodes must exist")

        merge_id = operation_id or typed_id("operation")

        primary = self.data["nodes"][primary_id]
        members = list(dict.fromkeys(primary.get("member_uuids", [primary_id]) + [member_id]))
        primary["member_uuids"] = members
        primary["updated_at"] = utc_timestamp()
        self.data["nodes"][primary_id] = primary
        del self.data["nodes"][member_id]
        self.data["merges"]["nodes"][member_id] = {
            "primary_uuid": primary_id,
            "merged_at": utc_timestamp(),
            "operation_id": merge_id,
        }

        # One survivor per **edge identity**, not per node pair: a semantic relation
        # is identified by (pair, predicate) so merging two entities keeps both
        # "A-养-猫" and "A-喜欢-猫" instead of deleting one of them. Structural
        # kinds stay pair-unique (a second has_fact / companion edge is bookkeeping
        # duplication, not meaning).
        relation_by_identity = {}
        removed_self_loops = []
        # Deterministic order: when a merge collapses two edges onto the same pair,
        # the survivor must be decided by the data, not by dict insertion order.
        # Oldest first, so the edge the role has known longest (and has accumulated
        # memory on) is the one that survives and the newer duplicate is redirected
        # to it. `created_at` alone can tie, so the id breaks the tie.
        ordered_relations = sorted(
            self.data["relations"].items(),
            key=lambda item: (
                int(item[1].get("sequence") or 0),
                str(item[1].get("created_at") or ""),
                str(item[0]),
            ),
        )
        for relation_id, relation in ordered_relations:
            if relation_id not in self.data["relations"]:
                # Already removed as a duplicate/self-loop earlier in this loop.
                continue
            endpoints = [self.resolve_node(endpoint) for endpoint in relation["endpoints"]]
            if endpoints[0] == endpoints[1]:
                removed_self_loops.append(relation_id)
                del self.data["relations"][relation_id]
                continue
            pair = stable_pair(*endpoints)
            relation_kind = str(relation.get("kind") or "relation")
            identity = (
                pair,
                relation_kind,
                normalized_text(relation.get("name") or "") if relation_kind == "relation" else "",
            )
            winner = relation_by_identity.get(identity)
            if winner is not None:
                self.data["merges"]["relations"][relation_id] = {
                    "primary_id": winner,
                    "merged_at": utc_timestamp(),
                    "operation_id": merge_id,
                }
                del self.data["relations"][relation_id]
                for whitelist in self.data.get("whitelist", {}).values():
                    whitelist["relations"] = [winner if item == relation_id else item for item in whitelist.get("relations", [])]
                continue
            relation["endpoints"] = list(pair)
            relation["updated_at"] = utc_timestamp()
            relation_by_identity[identity] = relation_id

        for whitelist in self.data.get("whitelist", {}).values():
            whitelist["nodes"] = list(dict.fromkeys(primary_id if item == member_id else item for item in whitelist.get("nodes", [])))
            # Event links are addressed by the event node, not by the relation
            # whitelist, so they must never be pruned by relation bookkeeping.
            whitelist["relations"] = [
                item for item in dict.fromkeys(whitelist.get("relations", []))
                if item in self.data["relations"] or item in self.data["event_links"]
            ]
            whitelist["relations"] = [item for item in whitelist["relations"] if item not in removed_self_loops]

        # A fact names the entity it is about in its own ``parent_id`` attribute, and
        # readers resolve that id directly (recall display, the retrievable text of the
        # fact). Once the entity is merged away the attribute must name the surviving
        # node, or those readers see an id that no longer exists. The member's own
        # redirect is already recorded above, so "resolves to the primary" is exactly
        # "was this entity (or an earlier merge of it)".
        for node in self.data["nodes"].values():
            if str(node.get("kind") or "") != FACT_NODE_KIND:
                continue
            parent = str(node.get("parent_id") or "")
            if not parent or str(self.resolve_node(parent) or parent) != primary_id:
                continue
            if parent == primary_id:
                continue
            node["parent_id"] = primary_id
            node["updated_at"] = utc_timestamp()

        self.data["merges"]["operations"][merge_id] = {
            "operation_id": merge_id,
            "primary_uuid": primary_id,
            "member_uuid": member_id,
            "removed_self_loop_relation_ids": removed_self_loops,
            "created_at": utc_timestamp(),
        }
        self._save()
        return {
            "primary_uuid": primary_id,
            "member_uuids": members,
            "operation_id": merge_id,
            "removed_self_loop_relation_ids": removed_self_loops,
        }

    def topology_for_role(self, role_uuid: str) -> Dict[str, Any]:
        whitelist = self.data.get("whitelist", {}).get(role_uuid, {"nodes": [], "relations": []})
        nodes = []
        events = []
        items = []
        for node_id in whitelist.get("nodes", []):
            node = self.data["nodes"].get(node_id)
            if node is None:
                continue
            if node.get("kind") == EVENT_NODE_KIND:
                # Events are read as their own memory units; they never enter
                # ordinary node/relation rendering or multi-hop adjacency.
                events.append({**project_for_role(node, role_uuid), "object_type": "event"})
            elif node.get("kind") == FACT_NODE_KIND:
                # A fact is one knowledge item of one entity: its own object, shown as
                # its own node, never an entity and never a multi-hop waypoint.
                items.append({**project_for_role(node, role_uuid), "object_type": "fact"})
            else:
                nodes.append({**project_for_role(node, role_uuid), "object_type": "node"})
        relations = []
        for relation_id in self.relation_ids_for_role(role_uuid):
            relation = self.data["relations"].get(relation_id)
            if relation is not None:
                relations.append(project_for_role(relation, role_uuid))
            else:
                link = self.data["event_links"].get(relation_id)
                if link is not None:
                    events.extend(
                        {
                            **project_for_role(self.data["nodes"][event_id], role_uuid),
                            "object_type": "event",
                            "event_link_id": link["id"],
                        }
                        for event_id in link.get("endpoints", [])
                        if event_id in self.data["nodes"]
                        and event_id != role_uuid
                        and self.data["nodes"][event_id].get("kind") == EVENT_NODE_KIND
                    )
        deduplicated: Dict[str, Dict[str, Any]] = {}
        for event in events:
            deduplicated.setdefault(str(event.get("id")), event)
        return {
            "nodes": nodes,
            "relations": relations,
            "events": list(deduplicated.values()),
            "items": items,
        }

    def events_for_role(self, role_uuid: str) -> List[Dict[str, Any]]:
        """Every event node that this role may read, with ``object_type=event``."""
        return self.topology_for_role(role_uuid)["events"]


    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data)

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Roll the document back to ``snapshot`` exactly, without merging.

        A rollback must be able to remove what the failed transaction wrote, so it is
        the one write that deliberately does not adopt objects another writer added in
        the meantime (see :meth:`_save`).
        """
        with self._lock:
            self.data = copy.deepcopy(snapshot)
            self._save(merge=False)
