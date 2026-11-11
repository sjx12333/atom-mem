"""Deterministic graph query operators for the LLM-facing memory tool.

The planner is the model; this module is the executor.  It never invents an
answer and never calls an LLM.  Inputs are a query, optional anchors, an
operator and an optional predicate; outputs are graph evidence with stable IDs
so the model can cite what it used.

Operators (method C):

* facts        -- facts owned by one or more anchors;
* list         -- facts of one anchor filtered by a predicate/target;
* union        -- every anchor's facts, grouped by anchor (the "分别/各自" case);
* intersection -- what the anchors share: common entities and common predicates;
* path         -- bounded shortest paths between two anchors;
* count        -- distinct mentioned objects/entities in the filtered facts;
* ppr          -- multi-hop Personalized PageRank, only when this tool operator
                  is explicitly called; the automatic recall path never uses it.

All access goes through topology.fact_records_for_role and the role's
whitelist; an anchor outside the whitelist simply does not resolve.
"""

from __future__ import annotations

import math
import time
import re
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import entity_linking
from .domain import normalized_text, score_text

_OPERATORS = {"facts", "list", "union", "intersection", "path", "count", "ppr"}


def _fact_parent(fact: Mapping[str, Any]) -> str:
    kind = str(fact.get("fact_kind") or "fact")
    if kind == "self_statement":
        return str(fact.get("speaker_id") or "")
    if kind == "opinion":
        return str(fact.get("target_id") or "")
    return str(fact.get("parent_id") or "")


def _text(fact: Mapping[str, Any]) -> str:
    return str(fact.get("text") or fact.get("name") or "").strip()


def _sort_facts(facts: Sequence[Mapping[str, Any]], query: str) -> List[Mapping[str, Any]]:
    return sorted(
        facts,
        key=lambda fact: (
            -float(score_text(query, _text(fact))) if str(query or "").strip() else 0.0,
            str(fact.get("updated_at") or ""),
            str(fact.get("id") or ""),
        ),
    )


class GraphQueryEngine:
    """One role-scoped, read-only executor over the shared topology."""

    def __init__(self, topology: Any, role_uuid: str):
        self.topology = topology
        self.role_uuid = str(role_uuid)
        self.topology.reload()
        self.entities = entity_linking.entity_records(topology, self.role_uuid)
        self.entity_by_id = {item["id"]: item for item in self.entities}
        self.entity_names = {item["id"]: item["name"] for item in self.entities}
        self.facts = list(
            topology.fact_records_for_role(self.role_uuid, include_invalidated=False)
        )
        self.facts_by_parent: Dict[str, List[Mapping[str, Any]]] = {}
        for fact in self.facts:
            parent = _fact_parent(fact)
            if parent:
                self.facts_by_parent.setdefault(parent, []).append(fact)
        self.adjacency: Dict[str, List[Tuple[str, str, str]]] = {
            entity_id: [] for entity_id in self.entity_names
        }
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.predicate_entities: Dict[str, set[str]] = {}
        self.predicate_names: Dict[str, str] = {}
        self._build_adjacency()

    # ------------------------------------------------------------------ index

    def _build_adjacency(self) -> None:
        relations = (self.topology.data or {}).get("relations") or {}
        nodes = (self.topology.data or {}).get("nodes") or {}
        for relation in relations.values():
            if str(relation.get("status") or "active") == "deleted":
                continue
            kind = str(relation.get("kind") or "")
            endpoints = [str(item) for item in (relation.get("endpoints") or []) if str(item)]
            if kind == "relation" and len(endpoints) == 2:
                label = "经关系「%s」" % (relation.get("name") or "相关")
                self._add_edge(endpoints[0], endpoints[1], label, str(relation.get("id") or ""))
            elif kind == "predicate_link" and endpoints:
                predicate_id = next((item for item in endpoints if item not in self.entity_names), "")
                if not predicate_id:
                    continue
                predicate = nodes.get(predicate_id) or {}
                label_name = str(predicate.get("name") or predicate.get("text") or predicate_id)
                self.predicate_names[predicate_id] = label_name
                self.predicate_entities.setdefault(predicate_id, set())
                for item in endpoints:
                    if item in self.entity_names:
                        self.predicate_entities[predicate_id].add(item)
        for predicate_id, entity_ids in self.predicate_entities.items():
            label = "经「%s」关联" % self.predicate_names.get(predicate_id, predicate_id)
            ordered = sorted(entity_ids)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1:]:
                    self._add_edge(left, right, label, "")

    def _add_edge(self, left: str, right: str, label: str, relation_id: str) -> None:
        if not left or not right or left == right:
            return
        if left not in self.adjacency or right not in self.adjacency:
            return
        self.adjacency[left].append((right, label, relation_id))
        self.adjacency[right].append((left, label, relation_id))
        # Shared edges carry no semantic direction or calibrated weight; the
        # default is uniform.  A future per-role overlay can set ``weight`` on
        # the relation without changing this query surface.
        self.edge_weights[(left, right)] = 1.0
        self.edge_weights[(right, left)] = 1.0

    # ---------------------------------------------------------------- helpers

    def _resolve_anchors(
        self, query: str, anchors: Optional[Sequence[str]]
    ) -> List[Dict[str, Any]]:
        if anchors:
            resolved: List[Dict[str, Any]] = []
            for value in anchors:
                entity = entity_linking.resolve_anchor_value(
                    self.topology, self.role_uuid, str(value)
                )
                if entity is not None and all(item["id"] != entity["id"] for item in resolved):
                    resolved.append({
                        "id": entity["id"],
                        "name": entity["name"],
                        "matched": str(value),
                        "source": "tool_argument",
                    })
            return resolved
        return entity_linking.resolve_query_anchors(
            self.topology, self.role_uuid, query, facts=self.facts, max_anchors=4
        )

    def _anchor_facts(self, anchor_id: str, query: str) -> List[Mapping[str, Any]]:
        return _sort_facts(self.facts_by_parent.get(str(anchor_id), []), query)

    def _matches(self, fact: Mapping[str, Any], predicate: str, target: str) -> bool:
        text = normalized_text(_text(fact))
        if predicate:
            needle = normalized_text(predicate)
            if needle and needle not in text:
                parent = self.entity_names.get(_fact_parent(fact), "")
                if needle not in normalized_text(parent):
                    return False
        if target:
            needle = normalized_text(target)
            if needle and needle not in text:
                return False
        return True

    def _fact_row(self, fact: Mapping[str, Any]) -> Dict[str, Any]:
        parent_id = _fact_parent(fact)
        return {
            "id": str(fact.get("id") or ""),
            "text": _text(fact),
            "parent_id": parent_id,
            "parent_name": self.entity_names.get(parent_id, parent_id),
            "fact_kind": str(fact.get("fact_kind") or "fact"),
        }

    def _facts_for(self, anchor_id: str, query: str, predicate: str, target: str) -> List[Mapping[str, Any]]:
        facts = self._anchor_facts(anchor_id, query)
        return [fact for fact in facts if self._matches(fact, predicate, target)]

    def _mentioned_objects(
        self, facts: Sequence[Mapping[str, Any]], anchor_id: str
    ) -> Dict[str, Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        for fact in facts:
            for entity in entity_linking.entities_mentioned(
                _text(fact), self.entities, exclude=[anchor_id]
            ):
                item = found.setdefault(entity["id"], {
                    "id": entity["id"],
                    "name": entity["name"],
                    "facts": [],
                })
                item["facts"].append(self._fact_row(fact))
        return found

    def _shared_predicates(self, anchor_ids: Sequence[str], predicate: str) -> List[Dict[str, Any]]:
        if len(anchor_ids) < 2:
            return []
        needle = normalized_text(predicate)
        shared: List[Dict[str, Any]] = []
        wanted = set(anchor_ids)
        for left in anchor_ids:
            for right, label, relation_id in self.adjacency.get(left, []):
                if right in wanted and left < right:
                    if needle and needle not in normalized_text(label):
                        continue
                    shared.append({"label": label, "entities": [left, right], "relation_id": relation_id})
        for predicate_id, entity_ids in self.predicate_entities.items():
            if not set(anchor_ids).issubset(entity_ids):
                continue
            label = self.predicate_names.get(predicate_id, predicate_id)
            if needle and needle not in normalized_text(label):
                continue
            shared.append({
                "label": "经「%s」关联" % label,
                "entities": list(anchor_ids),
                "relation_id": "",
                "predicate_id": predicate_id,
            })
        unique: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
        for item in shared:
            key = (item["label"], tuple(sorted(item["entities"])))
            unique.setdefault(key, item)
        return list(unique.values())

    def _shortest_paths(
        self, source: str, target: str, *, max_depth: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if source == target or source not in self.adjacency or target not in self.adjacency:
            return []
        max_depth = max(1, min(4, int(max_depth)))
        queue: deque = deque([(source, [source], [])])
        found: List[Dict[str, Any]] = []
        best_depth: Optional[int] = None
        while queue and len(found) < limit:
            node, path, edges = queue.popleft()
            if best_depth is not None and len(edges) >= best_depth:
                continue
            if len(edges) >= max_depth:
                continue
            for neighbor, label, relation_id in self.adjacency.get(node, []):
                if neighbor in path:
                    continue
                next_edges = edges + [{
                    "from": node,
                    "to": neighbor,
                    "label": label,
                    "relation_id": relation_id,
                }]
                if neighbor == target:
                    best_depth = len(next_edges) if best_depth is None else best_depth
                    found.append({
                        "nodes": path + [neighbor],
                        "names": [self.entity_names.get(item, item) for item in path + [neighbor]],
                        "edges": next_edges,
                    })
                    continue
                queue.append((neighbor, path + [neighbor], next_edges))
        return found[:limit]

    def _ppr(
        self,
        seeds: Sequence[str],
        *,
        damping: float = 0.85,
        max_iter: int = 30,
        tolerance: float = 1e-9,
    ) -> Dict[str, float]:
        """Personalized PageRank over the role-visible entity graph.

        Facts are deliberately **not** waypoints.  They are scored after the
        propagation by their parent's mass, so a long fact sentence can never
        become a bridge between two otherwise unrelated entities.  This mirrors
        the cold-layer rule that only entities/relations participate in
        multi-hop paths (ADR-0173).
        """
        nodes = [node for node in self.adjacency]
        if not nodes:
            return {}
        seed_mass: Dict[str, float] = {}
        for seed in seeds:
            if seed in self.adjacency:
                seed_mass[seed] = seed_mass.get(seed, 0.0) + 1.0
        if not seed_mass:
            return {}
        total = sum(seed_mass.values())
        personal = {node: seed_mass.get(node, 0.0) / total for node in nodes}
        rank = dict(personal)
        neighbor_weights: Dict[str, float] = {}
        for node in nodes:
            neighbor_weights[node] = sum(
                float(self.edge_weights.get((node, neighbor), 1.0))
                for neighbor, _, _ in self.adjacency.get(node, [])
            )
        for _ in range(max(1, int(max_iter))):
            dangling = sum(rank[node] for node in nodes if neighbor_weights.get(node, 0.0) <= 0)
            updated = {
                node: (1.0 - damping) * personal[node] + damping * dangling * personal[node]
                for node in nodes
            }
            for node in nodes:
                weight_sum = neighbor_weights.get(node, 0.0)
                if weight_sum <= 0:
                    continue
                share = damping * rank[node] / weight_sum
                for neighbor, _, _ in self.adjacency.get(node, []):
                    updated[neighbor] += share * float(
                        self.edge_weights.get((node, neighbor), 1.0)
                    )
            delta = sum(abs(updated[node] - rank[node]) for node in nodes)
            rank = updated
            if delta <= tolerance:
                break
        return rank

    def _ppr_evidence(
        self, seeds: Sequence[str], query: str, *, top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        started = time.perf_counter()
        rank = self._ppr(seeds)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if not rank:
            return [], [], {"iterations": 0, "elapsed_ms": elapsed_ms}
        nodes = sorted(
            rank.items(), key=lambda item: (-item[1], self.entity_names.get(item[0], item[0]))
        )[: max(1, min(20, int(top_k)))]
        rows: List[Dict[str, Any]] = []
        for parent, facts in self.facts_by_parent.items():
            mass = float(rank.get(parent, 0.0))
            if mass <= 0:
                continue
            for fact in facts:
                lexical = float(score_text(query, _text(fact)) or 0.0) if query else 0.0
                rows.append(
                    {
                        **self._fact_row(fact),
                        "anchor_id": parent,
                        "anchor_name": self.entity_names.get(parent, parent),
                        "ppr_score": round(mass, 9),
                        "lexical_score": round(lexical, 6),
                        "rank_score": round(mass * (0.7 + 0.3 * lexical), 9),
                    }
                )
        rows.sort(key=lambda item: (-float(item["rank_score"]), str(item.get("id") or "")))
        node_rows = [
            {
                "id": node,
                "name": self.entity_names.get(node, node),
                "ppr_score": round(score, 9),
            }
            for node, score in nodes
        ]
        diagnostics = {
            "iterations": 30,
            "elapsed_ms": elapsed_ms,
            "nodes": len(rank),
            "seeds": list(seeds),
            "fact_candidates": len(rows),
        }
        return node_rows, rows[: max(1, int(top_k))], diagnostics

    def _infer_predicate(self, query: str) -> str:
        text = normalized_text(query)
        patterns = (
            ("visited", ("visited", "visit", "去过", "到过", "访问")),
            ("wrote", ("written", "wrote", "write", "写了", "写信", "便条")),
            ("met", ("met ", "meet", "见过", "遇到")),
            ("helped", ("helped", "help", "帮助", "帮过")),
            ("lives", ("lives", "live in", "住在", "居住")),
            ("works", ("works", "worked", "工作")),
            ("owns", ("owns", "own ", "拥有")),
        )
        for label, needles in patterns:
            if any(needle in text for needle in needles):
                return label
        return ""

    def _infer_operator(self, query: str, anchors: Sequence[Dict[str, Any]]) -> str:
        text = normalized_text(query)
        if any(word in text for word in ("多跳", "multi-hop", "multihop", "ppr", "间接关联", "传播")):
            return "ppr"
        if re.search(r"\bhow\s+(?:many|much)\b|\bcount\b|多少|几个|几次", text):
            return "count"
        if any(word in text for word in ("both", "common", "shared", "共同", "都", "相同", "一样", "共同点")):
            return "intersection"
        if any(word in text for word in ("分别", "各自", "各", "respectively", "separate", "every")):
            return "union"
        if any(word in text for word in ("关系", "怎么认识", "路径", "relation", "path", "connect")):
            return "path"
        return "union" if len(anchors) >= 2 else "facts"

    # --------------------------------------------------------------- execute

    def execute(
        self,
        *,
        query: str = "",
        operator: Optional[str] = None,
        anchors: Optional[Sequence[str]] = None,
        predicate: str = "",
        target: str = "",
        top_k: int = 12,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        top_k = max(1, min(40, int(top_k)))
        query = str(query or "").strip()
        resolved = self._resolve_anchors(query, anchors)
        op = str(operator or "").strip().lower()
        if op not in _OPERATORS:
            op = self._infer_operator(query, resolved)
        inferred_predicate = ""
        if not predicate and op in {"intersection", "path", "count", "list"}:
            inferred_predicate = self._infer_predicate(query)
            predicate = inferred_predicate
        diagnostics: Dict[str, Any] = {
            "operator": op,
            "predicate": predicate,
            "predicate_inferred": inferred_predicate,
            "target": target,
            "anchor_count": len(resolved),
            "fact_count": len(self.facts),
            "entity_count": len(self.entities),
        }
        if not resolved:
            suggestions = [
                {"id": item["id"], "name": item["name"]}
                for item in self.entities[:8]
            ]
            return {
                "ok": False,
                "operator": op,
                "query": query,
                "anchors": [],
                "error": "no_anchor",
                "message": "query did not name a whitelisted entity; call memos_search_memory instead",
                "suggestions": suggestions,
                "diagnostics": diagnostics,
            }

        anchor_ids = [item["id"] for item in resolved]
        facts_by_anchor: List[Dict[str, Any]] = []
        for anchor in resolved:
            filtered = self._facts_for(anchor["id"], query, predicate, target)
            facts_by_anchor.append({
                "anchor": {"id": anchor["id"], "name": anchor["name"]},
                "matched": anchor.get("matched"),
                "facts": [self._fact_row(fact) for fact in filtered[:top_k]],
            })
        if predicate and not any(item["facts"] for item in facts_by_anchor):
            # The inferred predicate was too narrow for this graph; retry without
            # it rather than returning an authoritative-looking empty answer.
            diagnostics["predicate_fallback"] = predicate
            predicate = ""
            facts_by_anchor = []
            for anchor in resolved:
                filtered = self._facts_for(anchor["id"], query, predicate, target)
                facts_by_anchor.append({
                    "anchor": {"id": anchor["id"], "name": anchor["name"]},
                    "matched": anchor.get("matched"),
                    "facts": [self._fact_row(fact) for fact in filtered[:top_k]],
                })
        diagnostics["predicate_applied"] = bool(
            predicate and any(item["facts"] for item in facts_by_anchor)
        ) or not bool(predicate)

        evidence: Dict[str, Any] = {
            "facts_by_anchor": facts_by_anchor,
            "shared_entities": [],
            "shared_predicates": [],
            "paths": [],
            "count": None,
            "items": [],
            "ppr_nodes": [],
            "ppr_facts": [],
        }

        if op == "intersection":
            per_anchor_objects: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for anchor in resolved:
                facts = self._facts_for(anchor["id"], query, predicate, target)
                per_anchor_objects[anchor["id"]] = self._mentioned_objects(facts, anchor["id"])
            common_ids = set(per_anchor_objects.get(anchor_ids[0], {}))
            for anchor_id in anchor_ids[1:]:
                common_ids &= set(per_anchor_objects.get(anchor_id, {}))
            for object_id in sorted(common_ids):
                support = {
                    self.entity_names.get(anchor_id, anchor_id): per_anchor_objects[anchor_id][object_id]["facts"]
                    for anchor_id in anchor_ids
                    if object_id in per_anchor_objects.get(anchor_id, {})
                }
                evidence["shared_entities"].append({
                    "id": object_id,
                    "name": self.entity_names.get(object_id, object_id),
                    "support": support,
                })
            evidence["shared_predicates"] = self._shared_predicates(anchor_ids, predicate)
        elif op == "union":
            seen: set[str] = set()
            items: List[Dict[str, Any]] = []
            for anchor in resolved:
                for fact in self._facts_for(anchor["id"], query, predicate, target)[:top_k]:
                    row = self._fact_row(fact)
                    key = normalized_text(row["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(row)
            evidence["items"] = items[:top_k * max(1, len(resolved))]
        elif op == "count":
            values: Dict[str, Dict[str, Any]] = {}
            for anchor in resolved:
                facts = self._facts_for(anchor["id"], query, predicate, target)
                for object_id, payload in self._mentioned_objects(facts, anchor["id"]).items():
                    values.setdefault(object_id, payload)
                if not values:
                    for fact in facts:
                        for token in normalized_text(_text(fact)).split():
                            if any(char.isdigit() for char in token) and len(token) <= 8:
                                values.setdefault("num:" + token, {"id": "num:" + token, "name": token, "facts": []})
            evidence["items"] = list(values.values())
            evidence["count"] = len(values)
        elif op == "path":
            if len(anchor_ids) >= 2:
                evidence["paths"] = self._shortest_paths(
                    anchor_ids[0], anchor_ids[1], max_depth=max_depth
                )
            elif target:
                target_entity = entity_linking.resolve_anchor_value(
                    self.topology, self.role_uuid, target
                )
                if target_entity:
                    evidence["paths"] = self._shortest_paths(
                        anchor_ids[0], target_entity["id"], max_depth=max_depth
                    )
        elif op == "ppr":
            nodes, facts, ppr_diagnostics = self._ppr_evidence(
                anchor_ids, query, top_k=top_k
            )
            evidence["ppr_nodes"] = nodes
            evidence["ppr_facts"] = facts
            diagnostics["ppr"] = ppr_diagnostics
            diagnostics["ppr_only_on_tool_call"] = True

        return {
            "ok": True,
            "operator": op,
            "query": query,
            "anchors": [
                {"id": item["id"], "name": item["name"], "matched": item.get("matched"), "source": item.get("source")}
                for item in resolved
            ],
            "evidence": evidence,
            "diagnostics": diagnostics,
        }
