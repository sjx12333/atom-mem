"""Fact-first category graph: entities navigate, facts answer.

ADR-0175 changes the recall unit. An entity is no longer an answer candidate; it is
a **category / navigation node** that groups facts and connects to other entities.
This module builds that small, read-only view of one role's topology:

* facts are grouped by their parent entity;
* semantic relations and shared predicates become undirected entity-to-entity
  adjacency;
* a bounded BFS or DFS walks from seed entities and collects facts, with a path
  label and a hop-decayed strength.

The walk itself never decides admission. It only produces candidates; rerank and
the calibrated gates still decide what may enter the prompt.
"""

from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import node_memory


ENTITY_KINDS = {"node", "user", "ai_role"}


def _entity_canonical_map(names: Dict[str, str]) -> Dict[str, str]:
    """One navigation hub per concept, even when extraction split the concept.

    Two shapes are merged, both of them observed in the production graph:

    * **the same name twice** — 海燕-2 was published under two node ids, so its
      alias fact hung off one and its loadout facts off the other. The query could
      only ever reach one half, which is why that set recalled nothing.
    * **an aspect named as "X的Y"** — "F-15EX的主流挂载" is a facet of F-15EX, not a
      new thing. Filing the loadout under the facet kept it out of F-15EX's
      neighbourhood: asking "545 怎么带" could reach the alias fact but never the
      sentence that answers it.

    Merging is a *navigation* decision only: facts keep their own record and parent,
    entities keep their own name for rendering. A name is only folded into a prefix
    when the remainder starts with "的", so "F-15EX" and "F-15EX2" stay apart.
    """
    normalized = {str(key): " ".join(str(value or "").split()) for key, value in names.items()}
    groups: Dict[str, List[str]] = {}
    for entity_id, name in normalized.items():
        groups.setdefault(name, []).append(entity_id)

    canonical: Dict[str, str] = {}
    for ids in groups.values():
        base = sorted(ids)[0]
        for entity_id in ids:
            canonical[entity_id] = base

    ordered = sorted(groups, key=len, reverse=True)
    for name in ordered:
        if len(name) < 3:
            continue
        prefixes = [
            other
            for other in ordered
            if other != name
            and len(other) >= 2
            and name.startswith(other)
            and name[len(other):].startswith("的")
        ]
        if not prefixes:
            continue
        base_name = max(prefixes, key=len)
        base = sorted(groups[base_name])[0]
        for entity_id in groups[name]:
            canonical[entity_id] = base
    return canonical


@dataclass(frozen=True)
class WalkedFact:
    """One fact reached from a seed entity, with the path that reached it."""

    record: Dict[str, Any]
    parent_id: str
    seed_id: str
    depth: int
    path: str
    strength: float
    hub_factor: float
    #: Additive, depth-decayed boost applied on top of the fact's own cosine.
    depth_weight: float = 0.0
    #: The path's own evidence: seed strength * decay^hop * hub damping, **without**
    #: the fact's cosine and without the additive bonus. Recall keeps this apart from
    #: the text score so "reachable through an edge" can admit a fact without
    #: pretending it is textually similar to the query (and without outranking one
    #: that is).
    path_score: float = 0.0


#: Whether a self statement (an AI's durable statement about itself) is part of the
#: recall corpus. It was always published but always skipped by the graph, so the
#: AI's own words could be stored and never recalled. The record carries no
#: parent_id; its speaker_id (the ai_role node) is the anchor.
_SELF_STATEMENT_VIEW: Optional[bool] = None


def set_self_statement_view(enabled: Optional[bool]) -> None:
    """Pin the view on/off; None restores the config-driven default."""

    global _SELF_STATEMENT_VIEW
    _SELF_STATEMENT_VIEW = enabled


def self_statement_view_enabled() -> bool:
    global _SELF_STATEMENT_VIEW
    if _SELF_STATEMENT_VIEW is None:
        try:
            from .runtime import load_config

            retrieval = (load_config() or {}).get("retrieval") or {}
            _SELF_STATEMENT_VIEW = bool(retrieval.get("self_statement_recall", True))
        except Exception:  # noqa: BLE001 - the view must never break recall
            _SELF_STATEMENT_VIEW = True
    return bool(_SELF_STATEMENT_VIEW)


class FactCategoryGraph:
    """A role-scoped, in-memory view of facts -> entities -> other entities."""

    def __init__(
        self,
        *,
        role_uuid: str,
        facts_by_entity: Dict[str, List[Dict[str, Any]]],
        entity_names: Dict[str, str],
        adjacency: Dict[str, List[Tuple[str, str, str]]],
        degree: Dict[str, int],
        canonical: Optional[Dict[str, str]] = None,
    ):
        self.role_uuid = role_uuid
        self.facts_by_entity = facts_by_entity
        self.entity_names = entity_names
        self.adjacency = adjacency
        self.degree = degree
        #: Original entity id -> navigation id. Duplicate nodes and "X 的 Y" aspects
        #: share one hub, so a fact filed under any of them is reachable from the
        #: hub the query actually names.
        self.canonical = canonical or {}

    def hub_of(self, entity_id: str) -> str:
        """The navigation id an entity belongs to (itself when not merged)."""
        key = str(entity_id or "")
        return self.canonical.get(key, key)

    @classmethod
    def build(
        cls,
        topology: Any,
        role_uuid: str,
        scores: Optional[Dict[str, float]] = None,
    ) -> "FactCategoryGraph":
        data = getattr(topology, "data", {}) or {}
        nodes = data.get("nodes") or {}
        relations = data.get("relations") or {}
        whitelist = (data.get("whitelist") or {}).get(role_uuid) or {}
        allowed_nodes = {
            topology.resolve_node(item) for item in (whitelist.get("nodes") or []) if item
        }

        entity_names: Dict[str, str] = {}
        for node_id, record in nodes.items():
            kind = str(record.get("kind") or "node")
            if kind not in ENTITY_KINDS:
                continue
            if str(node_id) not in allowed_nodes:
                continue
            if str(record.get("status") or "active") == "deleted":
                continue
            entity_names[str(node_id)] = str(record.get("name") or node_id)

        # Collect the parents of facts first: a parent that is not itself an entity
        # (an older publication, a predicate-shaped name) still needs a name to be
        # merged and rendered.
        raw_facts: List[Dict[str, Any]] = []
        opinion_facts: List[Dict[str, Any]] = []
        for fact in topology.fact_records_for_role(role_uuid, include_invalidated=False):
            kind = node_memory.fact_kind(fact)
            if kind == "self_statement":
                # A self statement has no entity subject, so it used to be dropped
                # here entirely: published, never recallable. Anchor it to the
                # speaker (the ai_role node) so it rides the same fact channel.
                if not self_statement_view_enabled():
                    continue
                parent_id = str(fact.get("speaker_id") or "")
                if not parent_id:
                    continue
                raw_facts.append(fact)
                if parent_id not in entity_names:
                    speaker = nodes.get(parent_id) or {}
                    entity_names[parent_id] = str(speaker.get("name") or parent_id)
                continue
            if kind == "opinion":
                parent_id = str(fact.get("target_id") or "")
                if not parent_id:
                    continue
                raw_facts.append(fact)
                opinion_facts.append(fact)
                if parent_id not in entity_names:
                    target = nodes.get(parent_id) or {}
                    entity_names[parent_id] = str(target.get("name") or parent_id)
                continue
            parent_id = str(fact.get("parent_id") or "")
            if not parent_id:
                continue
            raw_facts.append(fact)
            if parent_id not in entity_names:
                parent = nodes.get(parent_id) or {}
                entity_names[parent_id] = str(parent.get("name") or parent_id)

        canonical = _entity_canonical_map(entity_names)

        facts_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        for fact in raw_facts:
            parent_key = str(fact.get("target_id") or fact.get("parent_id") or "")
            parent_id = canonical.get(parent_key, parent_key)
            facts_by_entity.setdefault(parent_id, []).append(fact)

        adjacency: Dict[str, List[Tuple[str, str, str]]] = {
            hub for hub in set(canonical.values()) | set(facts_by_entity)
        }
        adjacency = {key: [] for key in adjacency}
        seen_edges: set[Tuple[str, str, str]] = set()

        def add_edge(left: str, right: str, label: str, relation_id: str = "") -> None:
            left = canonical.get(str(left or ""), str(left or ""))
            right = canonical.get(str(right or ""), str(right or ""))
            if not left or not right or left == right:
                return
            if left not in entity_names or right not in entity_names:
                return
            key = (left, right, label)
            if key in seen_edges:
                return
            seen_edges.add(key)
            adjacency.setdefault(left, []).append((right, label, str(relation_id or "")))
            adjacency.setdefault(right, []).append((left, label, str(relation_id or "")))

        # An opinion is a relay: the AI and the described entity become reachable
        # through the opinion object, without pretending the opinion is a fact of
        # the target entity. The edge label is deliberately stable and subject-free;
        # the opinion's own text stays on the fact node.
        for fact in opinion_facts:
            speaker_id = str(fact.get("speaker_id") or "")
            target_id = str(fact.get("target_id") or "")
            if speaker_id and target_id:
                add_edge(speaker_id, target_id, "经看法关联")

        predicate_entities: Dict[str, List[str]] = {}
        for relation in relations.values():
            if str(relation.get("status") or "active") == "deleted":
                continue
            kind = str(relation.get("kind") or "")
            endpoints = [str(item) for item in (relation.get("endpoints") or []) if str(item)]
            if kind == "relation":
                label = "经关系「%s」" % (relation.get("name") or "相关")
                for index, left in enumerate(endpoints):
                    for right in endpoints[index + 1 :]:
                        add_edge(left, right, label, str(relation.get("id") or ""))
            elif kind == "predicate_link" and endpoints:
                predicate_id = next(
                    (item for item in endpoints if item not in entity_names),
                    "",
                )
                if not predicate_id:
                    continue
                predicate_entities.setdefault(predicate_id, [])
                for item in endpoints:
                    if item in entity_names:
                        predicate_entities[predicate_id].append(item)

        for predicate_id, entities in predicate_entities.items():
            predicate = nodes.get(predicate_id) or {}
            label = "经「%s」关联" % (predicate.get("name") or predicate_id)
            unique = list(dict.fromkeys(entities))
            for index, left in enumerate(unique):
                for right in unique[index + 1 :]:
                    add_edge(left, right, label, "")

        degree = {
            entity_id: len(adjacency.get(entity_id) or []) + len(facts_by_entity.get(entity_id) or [])
            for entity_id in set(adjacency) | set(facts_by_entity)
        }
        return cls(
            role_uuid=role_uuid,
            facts_by_entity=facts_by_entity,
            entity_names=entity_names,
            adjacency=adjacency,
            degree=degree,
            canonical=canonical,
        )

    def hub_factor(self, entity_id: str, strength: float = 0.15) -> float:
        """Degree-based damping for traversal strength; 1.0 when disabled."""
        if strength <= 0:
            return 1.0
        degree = max(0, int(self.degree.get(self.hub_of(entity_id), 0)) - 1)
        return 1.0 / (1.0 + float(strength) * math.log1p(degree))

    def facts_for(
        self,
        entity_id: str,
        scores: Optional[Dict[str, float]] = None,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        scores = scores or {}
        facts = list(self.facts_by_entity.get(self.hub_of(entity_id), []))
        facts.sort(
            key=lambda record: (
                -float(scores.get(node_memory.object_id_of(record), 0.0)),
                str(record.get("updated_at") or ""),
                node_memory.object_id_of(record),
            ),
            reverse=False,
        )
        # The sort above is ascending on updated_at and id only because the primary
        # key is negated; make recency deterministic without letting it outrank score.
        return facts[: max(0, int(limit))]

    def walk(
        self,
        seeds: Dict[str, Tuple[float, str]],
        *,
        depth: int = 2,
        strategy: str = "bfs",
        max_candidates: int = 60,
        neighbors_per_seed: int = 5,
        facts_per_entity: int = 4,
        decay: float = 0.70,
        scores: Optional[Dict[str, float]] = None,
        hub_strength: float = 0.15,
        depth_bonus: float = 0.0,
        edge_weight: Optional[Callable[[str], float]] = None,
        traversed: Optional[set[str]] = None,
    ) -> List[WalkedFact]:
        """Walk the category graph from seed entities and collect facts.

        Seeds map an entity id to (strength, source fact id). The source fact is not
        re-emitted; every other fact is a candidate with the path and hop count that
        reached it. BFS and DFS share the same visited set and hard caps; only the
        frontier order differs.

        ``depth_bonus`` adds ``depth_bonus * decay ** hop`` to every reached fact
        *after* the hub damping, and never lowers a fact's own cosine. The seed
        (depth 0) gains most and the third hop gains least: content reached by a
        short association chain can clear a cosine gate it would miss on its own,
        without a deep chain flooding the candidate pool.
        """
        max_candidates = max(0, int(max_candidates))
        if max_candidates == 0 or not seeds:
            return []
        scores = scores or {}
        depth = max(0, int(depth))
        strategy = "dfs" if str(strategy).lower() == "dfs" else "bfs"
        per_entity = max(0, int(facts_per_entity))
        per_seed = max(0, int(neighbors_per_seed))

        frontier: deque | list = deque() if strategy == "bfs" else []
        queued: set[str] = set()
        for entity_id, (strength, source_id) in seeds.items():
            entity_id = self.hub_of(str(entity_id))
            if entity_id not in self.entity_names and entity_id not in self.facts_by_entity:
                continue
            item = (entity_id, 0, "", float(strength), str(source_id))
            if strategy == "bfs":
                frontier.append(item)
            else:
                frontier.append(item)
            queued.add(entity_id)

        visited: set[str] = set()
        emitted: set[str] = set()
        out: List[WalkedFact] = []

        def pop_next():
            if strategy == "bfs":
                return frontier.popleft()
            return frontier.pop()

        while frontier and len(out) < max_candidates:
            entity_id, hop, path, strength, source_id = pop_next()
            if entity_id in visited:
                continue
            visited.add(entity_id)
            for fact in self.facts_for(entity_id, scores=scores, limit=per_entity):
                fact_id = node_memory.object_id_of(fact)
                if fact_id == source_id or fact_id in emitted:
                    continue
                # A fact's own semantic score can be stronger than the path-decayed
                # seed; keep the stronger signal, then damp hubs for ordering only.
                semantic = max(
                    float(scores.get(fact_id, 0.0)),
                    float(strength) * (float(decay) ** hop),
                )
                bonus = float(depth_bonus) * (float(decay) ** hop) if depth_bonus > 0 else 0.0
                factor = self.hub_factor(entity_id, hub_strength)
                path_score = float(strength) * (float(decay) ** hop) * factor
                # Hub damping applies to the *navigation* strength (the seed's
                # decayed score and the fact's own cosine), while the depth weight
                # is added afterwards. Damping the bonus too cancelled it for
                # exactly the deep facts it exists to admit: a depth-3 fact with
                # cosine 0.446 gained 0.126 then lost it again to a 0.74 hub
                # factor and still fell below 0.45 (measured).
                out.append(
                    WalkedFact(
                        record=fact,
                        parent_id=entity_id,
                        seed_id=source_id,
                        depth=hop,
                        path=path,
                        strength=(semantic * factor) + bonus,
                        hub_factor=factor,
                        depth_weight=bonus,
                        path_score=path_score,
                    )
                )
                emitted.add(fact_id)
                if len(out) >= max_candidates:
                    break
            if len(out) >= max_candidates or hop >= depth:
                continue
            neighbors = list(self.adjacency.get(entity_id) or [])[:per_seed]
            if strategy == "dfs":
                neighbors = list(reversed(neighbors))
            for neighbor_id, label, relation_id in neighbors:
                if neighbor_id in visited or neighbor_id in queued:
                    continue
                next_path = path + ("；" if path else "") + label
                hop_strength = float(strength)
                if relation_id:
                    if traversed is not None:
                        traversed.add(str(relation_id))
                    if edge_weight is not None:
                        try:
                            hop_strength *= max(0.0, min(1.0, float(edge_weight(str(relation_id)))))
                        except Exception:  # noqa: BLE001 - weight is a preference only
                            hop_strength = float(strength)
                item = (neighbor_id, hop + 1, next_path, hop_strength, source_id)
                if strategy == "bfs":
                    frontier.append(item)
                else:
                    frontier.append(item)
                queued.add(neighbor_id)
        return out
