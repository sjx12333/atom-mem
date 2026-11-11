"""Recall over role-scoped node-attribute memory.

Two contracts live here:

* **entity-first** (legacy, fact_first_enabled=False): one retrieval object is
  one node/relation, and a hit returns that object's complete memory. Candidates
  come from the role whitelist, scores from the vector index, and the calibrated
  thresholds (ADR-0151) apply to the semantic and reranker scores.
* **fact-first** (ADR-0175, production default): facts are the candidates; entities
  are a category/navigation layer. Direct facts are reranked as their own evidence,
  a bounded BFS/DFS walks relations and shared predicates to related facts before
  reranking, near-duplicates are collapsed, and entities are only attached as
  category metadata.

Both paths use the same whitelist, vector index, reranker and gates. The legacy
path stays available for roles with no facts and for callers that have not opted in.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import entity_linking, node_memory
from .domain import normalized_text, score_text, text_hash
from .fact_graph import FactCategoryGraph
from .retrieval import RetrievalPolicy, escape_node_description, prompt_text
from .vector_store import VectorIndexError

#: Minimum query coverage accepted on the embedding-less lexical fallback path.
_LEXICAL_COVERAGE_FLOOR = 0.34

def _nodes_snapshot(topology) -> Dict[str, Any]:
    """Copy the topology nodes dict without crashing on a concurrent write.

    Extraction mutates topology.data["nodes"] in place while a query may be
    iterating it, which raised "dictionary changed size during iteration" and
    turned the whole search into an empty result. Retry a few times, then fall
    back to an empty mapping.
    """
    for _ in range(5):
        try:
            return dict(topology.data.get("nodes") or {})
        except RuntimeError:
            time.sleep(0.02)
    return {}




#: Object types the index may hold for node-attribute memory. A memory event and a
#: knowledge fact are retrieval objects of their own, exactly like an entity or a
#: relation — that is what lets recall gate them individually. ``predicate`` is a
#: relation's shared subject and competes as a candidate in its own right, because
#: hitting it is how two entities that share it become reachable. ``hook`` is scored
#: but never a candidate — see :meth:`NodeMemoryRecall._hook_resemblance`.
OBJECT_TYPES = ("node", "relation", "event", "fact", "predicate", "hook")


class NodeMemoryRecall:
    """Vector + rerank recall over topology records and their memory elements."""

    def __init__(
        self,
        topology: Any,
        *,
        memory_layer: Any = None,
        policy: Optional[RetrievalPolicy | Dict[str, Any]] = None,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        reranker: Optional[Callable[..., float | Dict[str, Any]]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        activation: Any = None,
    ):
        self.topology = topology
        self.memory_layer = memory_layer
        self.policy = RetrievalPolicy.from_value(policy)
        self.embedder = embedder
        self.reranker = reranker
        self.token_counter = token_counter or len
        #: `ActivationStore`-like provider. Only `weights_for`/`weight_for` and
        #: `edge_weight_for` are used; recall never writes activation itself.
        self.activation = activation

    # ------------------------------------------------------------------- helpers

    @property
    def vectors(self):
        layer = self.memory_layer
        return getattr(layer, "vectors", None) if layer is not None else None

    def _embeddings_configured(self) -> bool:
        return self.embedder is not None

    def _query_vector(self, query: str) -> Optional[List[float]]:
        if self.embedder is None:
            return None
        vector = self.embedder(query)
        if vector is None:
            return None
        return [float(item) for item in vector]

    def _index_failure(self, error: Exception) -> Dict[str, Any]:
        return {
            "memories": [],
            "count": 0,
            "token_estimate": 0,
            "policy_version": self.policy.version,
            "diagnostics": {
                "source": "NodeMemory",
                "status": "index_failure",
                "error": str(error),
            },
            "index_failure": True,
        }

    # ------------------------------------------------------------------- search

    def search(
        self,
        role_uuid: str,
        query: str,
        top_k: int = 20,
        token_budget: Optional[int] = None,
        similarity_threshold: float = 0.0,
        query_vector: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Recall the role's memories.

        ``fact_first_enabled`` selects the ADR-0175 path: facts are the candidates,
        entities are the category/navigation layer. The legacy entity-first path is
        kept as the fallback for roles with no facts and for policies that have not
        enabled the new contract yet.
        """
        if not bool(getattr(self.policy, "fact_first_enabled", False)):
            return self._search_entity_first(
                role_uuid,
                query,
                top_k=top_k,
                token_budget=token_budget,
                similarity_threshold=similarity_threshold,
                query_vector=query_vector,
            )
        return self._search_fact_first(
            role_uuid,
            query,
            top_k=top_k,
            token_budget=token_budget,
            similarity_threshold=similarity_threshold,
            query_vector=query_vector,
        )

    def _search_entity_first(
        self,
        role_uuid: str,
        query: str,
        top_k: int = 20,
        token_budget: Optional[int] = None,
        similarity_threshold: float = 0.0,
        query_vector: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        policy = self.policy
        effective_top_k = max(0, min(int(top_k), policy.direct_top_k))
        effective_budget = policy.token_budget if token_budget is None else max(0, int(token_budget))
        diagnostics: Dict[str, Any] = {
            "policy_version": policy.version,
            "source": "NodeMemory",
            "query_embeddings": 0,
            "candidate_counts": {"node": 0, "relation": 0},
            "thresholds": {
                "semantic": max(float(similarity_threshold), policy.semantic_threshold),
                "direct_rerank": policy.direct_rerank_threshold,
                "relative_floor_ratio": policy.relative_floor_ratio,
            },
            "rerank_enabled": bool(getattr(policy, "rerank_enabled", True)),
            "rerank_degraded": False,
            "evicted_by_rerank_gate": [],
            "evicted_by_relative_floor": [],
            "evicted_duplicate_content": [],
            "missing_index": [],
        }
        if not str(query or "").strip() or effective_top_k == 0 or effective_budget == 0:
            return self._empty(policy, diagnostics)

        self.topology.reload()
        endpoint_names = {
            str(node_id): str((node or {}).get("name") or node_id)
            for node_id, node in _nodes_snapshot(self.topology).items()
        }
        if not self._embeddings_configured():
            records = [
                record
                for record in self.topology.memory_records_for_role(role_uuid)
                if node_memory.index_text(record, role_uuid, endpoint_names=endpoint_names)
            ]
            diagnostics["whitelisted_objects"] = len(records)
            return self._lexical_fallback(
                records, query, "embedding_model_not_configured", policy, diagnostics, role_uuid
            )
        if self.vectors is None or not self.vectors.is_available():
            return self._index_failure(
                VectorIndexError("semantic recall is configured but the vector index is unavailable")
            )
        timings: Dict[str, float] = {}
        stage_started = time.perf_counter()

        def mark(stage: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            timings[stage] = round((now - stage_started) * 1000.0, 1)
            stage_started = now

        try:
            # The caller may already have embedded this query for another recall
            # source (the Warm Layer shares the same gateway request). Reusing that
            # vector keeps one query to one embedding call.
            if query_vector is None:
                query_vector = self._query_vector(query)
            else:
                query_vector = [float(item) for item in query_vector]
            if query_vector is None:
                raise VectorIndexError("embedding model returned no query vector")
            diagnostics["query_embeddings"] = 1
            scores = self._vector_scores(role_uuid, query_vector, policy, similarity_threshold)
            mark("vector")
            diagnostics["index_hits"] = len(scores)
            # The whitelist is the second boundary: a vector that exists for this
            # role but whose object is no longer granted to it is not a candidate.
            seeds, items = self._split_seeds_and_items(role_uuid, scores)
            diagnostics["item_candidates"] = len(items)
            diagnostics["nominated_entities"] = self._nominate_parents(
                role_uuid, seeds, items, policy, diagnostics
            )
            # 精准联想：hook 的完整短语逐字出现在查询里，才允许它**提名**母实体。
            # 这不是放行：母实体仍要和其它候选一起过重排门槛；它只是让“只在 hook 里
            # 出现过的关联”有机会进入候选，而不是永远停在分数表上。
            hook_admissions = self._nominate_hook_parents(
                role_uuid, query, scores, seeds, policy, diagnostics
            )
            candidates = [
                self._candidate(record, semantic, role_uuid, endpoint_names)
                for record, semantic in seeds
            ]
            # 把命中的 hook 作为**带预测标记的联想证据**挂到母实体上，让重排看到
            # “这次为什么把它找出来”。它不会变成 Knowledge of，也不会进入共享拓扑。
            self._attach_hook_evidence(candidates, hook_admissions)
            diagnostics["whitelisted_objects"] = len(candidates)
            diagnostics["candidate_counts"] = {
                key: sum(1 for item in candidates if item["object_type"] == key)
                for key in OBJECT_TYPES
            }
            # 重排必须看到证据：先在每个实体上挂少量事实（预览），再重排。
            # 旧顺序（先重排、后挂事实）让实体只拿 `Node: 名字 [id]` 参与打分，
            # 强事实提名上来的实体反而因为名字太短被 0.50 砍掉（见 policy 注释）。
            self._attach_items(
                candidates,
                items,
                policy,
                role_uuid,
                endpoint_names,
                diagnostics,
                preview_limit=int(getattr(policy, "preview_items_per_entity", 3)),
            )
            mark("preview")
            scored = self._rerank(query, candidates, diagnostics)
            mark("rerank")
            gated = self._gate(
                scored, policy, diagnostics, threshold=policy.direct_rerank_threshold
            )
            # 有界一跳扩展：谓词/关系命中后沿边取邻居（事件不展开）；扩展候选自己再过一次
            # 重排门槛（indirect_rerank_threshold，此前是死配置）。
            expanded = self._expand_gated(
                role_uuid,
                gated,
                items,
                policy,
                endpoint_names,
                diagnostics,
                effective_top_k,
            )
            mark("expansion")
            if expanded:
                self._attach_items(
                    expanded,
                    items,
                    policy,
                    role_uuid,
                    endpoint_names,
                    diagnostics,
                    preview_limit=int(getattr(policy, "preview_items_per_entity", 3)),
                )
                scored_expanded = self._rerank(query, expanded, diagnostics)
                mark("expansion_rerank")
                gated.extend(
                    self._gate(
                        scored_expanded,
                        policy,
                        diagnostics,
                        threshold=policy.indirect_rerank_threshold,
                        reason="indirect",
                    )
                )
            # 联想加成（hook）：这里只**记录**每个已过门槛实体最强的 hook 相似度，不加分。
            # 真正的加分由 gateway 在所有准入判定之后施加（``gateway._apply_hook_boost``），
            # 因此加成本身仍只是排序偏好。精准命中的**提名**发生在更早的
            # ``_nominate_hook_parents``，且被提名的母实体仍要过下面同一道门槛。
            self._attach_hook_resemblance(gated, role_uuid, scores, diagnostics)
            # 准入后补齐全部事实：预览只服务重排，注入内容仍是"该实体的完整记忆"。
            self._attach_items(gated, items, policy, role_uuid, endpoint_names, diagnostics)
            deduped = self._deduplicate_content(gated, diagnostics)
            deduped = self._apply_activation(deduped, role_uuid, diagnostics)
            selected = self._select(deduped, effective_top_k, effective_budget, diagnostics)
            mark("finalize")
            diagnostics["timings"] = timings
            diagnostics["selected_ids"] = [item.get("id") for item in selected]
            diagnostics["zero_recall"] = not selected
            return {
                "memories": selected,
                "count": len(selected),
                "token_estimate": sum(int(item.get("token_estimate", 0)) for item in selected),
                "policy_version": policy.version,
                "diagnostics": diagnostics,
            }
        except VectorIndexError as exc:
            return self._index_failure(exc)

    def _search_fact_first(
        self,
        role_uuid: str,
        query: str,
        top_k: int = 20,
        token_budget: Optional[int] = None,
        similarity_threshold: float = 0.0,
        query_vector: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """ADR-0175: facts are candidates; entities are the category graph.

        The old entity-first path asked "which entity dossier matches this query?",
        then let the entity carry up to three preview facts into the reranker. That
        made a generic hub (for example the user node) win broadly while the exact
        fact that answered the query was never scored. Here:

        1. every fact with a semantic score above the item floor is a candidate;
        2. entities with a direct vector hit seed a bounded BFS/DFS over the category
           graph; the walk reaches related entities and their facts;
        3. near-duplicate facts are collapsed before reranking;
        4. all candidates are reranked once, then gated as direct vs indirect;
        5. entities are never injected as answers; they are only attached as
           category/context metadata.

        A role with no fact candidates at all falls back to the legacy dossier path
        so that "no facts" does not become "no memory". Facts that exist but all
        miss the gate are a legitimate zero recall; the entity hub is not a backdoor.
        """
        policy = self.policy
        effective_top_k = max(0, min(int(top_k), policy.direct_top_k))
        effective_budget = (
            policy.token_budget if token_budget is None else max(0, int(token_budget))
        )
        diagnostics: Dict[str, Any] = {
            "policy_version": policy.version,
            "source": "FactFirst",
            "query_embeddings": 0,
            "candidate_counts": {"fact": 0, "direct_fact": 0, "expanded_fact": 0},
            "thresholds": {
                "semantic": max(float(similarity_threshold), policy.semantic_threshold),
                "item_semantic": policy.item_semantic_threshold,
                "direct_rerank": policy.direct_rerank_threshold,
                "indirect_rerank": policy.indirect_rerank_threshold,
                "relative_floor_ratio": policy.relative_floor_ratio,
            },
            "rerank_enabled": bool(getattr(policy, "rerank_enabled", True)),
            "rerank_degraded": False,
            "evicted_by_rerank_gate": [],
            "evicted_by_indirect_gate": [],
            "evicted_by_relative_floor": [],
            "evicted_by_indirect_floor": [],
            "evicted_duplicate_content": [],
            "evicted_by_chain_floor": [],
            "near_duplicate_facts": [],
        }
        if not str(query or "").strip() or effective_top_k == 0 or effective_budget == 0:
            return self._empty(policy, diagnostics)

        self.topology.reload()
        endpoint_names = {
            str(node_id): str((node or {}).get("name") or node_id)
            for node_id, node in _nodes_snapshot(self.topology).items()
        }
        if not self._embeddings_configured():
            fallback = self._search_entity_first(
                role_uuid,
                query,
                top_k=top_k,
                token_budget=token_budget,
                similarity_threshold=similarity_threshold,
                query_vector=query_vector,
            )
            fallback.setdefault("diagnostics", {})["fact_first_fallback"] = {
                "reason": "embedding_model_not_configured"
            }
            return fallback
        if self.vectors is None or not self.vectors.is_available():
            failure = self._index_failure(
                VectorIndexError("semantic recall is configured but the vector index is unavailable")
            )
            failure["diagnostics"]["source"] = "FactFirst"
            return failure

        timings: Dict[str, float] = {}
        stage_started = time.perf_counter()

        def mark(stage: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            timings[stage] = round((now - stage_started) * 1000.0, 1)
            stage_started = now

        try:
            if query_vector is None:
                query_vector = self._query_vector(query)
            else:
                query_vector = [float(item) for item in query_vector]
            if query_vector is None:
                raise VectorIndexError("embedding model returned no query vector")
            diagnostics["query_embeddings"] = 1
            scores = self._vector_scores(
                role_uuid, query_vector, policy, similarity_threshold
            )
            mark("vector")
            diagnostics["index_hits"] = len(scores)
            facts = self.topology.fact_records_for_role(
                role_uuid, include_invalidated=False
            )
            if not facts:
                fallback = self._search_entity_first(
                    role_uuid,
                    query,
                    top_k=top_k,
                    token_budget=token_budget,
                    similarity_threshold=similarity_threshold,
                    query_vector=query_vector,
                )
                fallback.setdefault("diagnostics", {})["fact_first_fallback"] = {
                    "reason": "role_has_no_fact_records"
                }
                return fallback

            graph = FactCategoryGraph.build(self.topology, role_uuid, scores)
            diagnostics["fact_graph"] = {
                "entities": len(graph.entity_names),
                "facts": sum(len(items) for items in graph.facts_by_entity.values()),
                "edges": sum(len(items) for items in graph.adjacency.values()) // 2,
            }

            def decorate(candidate: Dict[str, Any]) -> Dict[str, Any]:
                parent_id = str(candidate.get("entity_id") or "")
                parent = (self.topology.data.get("nodes") or {}).get(parent_id) or {}
                classes = parent.get("classes") or parent.get("class") or []
                if isinstance(classes, str):
                    classes = [classes]
                candidate["category"] = str(parent.get("name") or parent_id or "")
                candidate["categories"] = [str(item) for item in classes][:8]
                refs = list(candidate.get("refs") or [])
                if parent_id and parent_id not in refs:
                    refs.append(parent_id)
                candidate["refs"] = refs
                return candidate

            direct: List[Dict[str, Any]] = []
            item_floor = float(policy.item_semantic_threshold)
            for fact in facts:
                fact_id = node_memory.object_id_of(fact)
                semantic = float(scores.get(fact_id, 0.0))
                if semantic <= 0.0 or semantic < item_floor:
                    continue
                candidate = self._candidate(
                    fact, semantic, role_uuid, endpoint_names
                )
                candidate["match_type"] = "direct"
                decorate(candidate)
                direct.append(candidate)

            # Method A: entity-anchored recall. A two-entity question must see
            # facts from *both* entities; letting the vector top-k decide makes the
            # first entity's hub fill the whole direct list. The anchors are
            # candidates only: they still pass the same rerank gate.
            anchors: List[Dict[str, Any]] = []
            if bool(getattr(policy, "entity_anchoring_enabled", False)):
                anchors = entity_linking.resolve_query_anchors(
                    self.topology, role_uuid, query, facts=facts, max_anchors=4
                )
                diagnostics["query_anchors"] = [
                    {"id": item["id"], "name": item["name"], "matched": item.get("matched"), "source": item.get("source")}
                    for item in anchors
                ]
                known = {str(item.get("id") or "") for item in direct}
                per_anchor = max(0, int(getattr(policy, "anchor_facts_per_entity", 6)))
                for anchor in anchors:
                    owned = sorted(
                        entity_linking.facts_for_anchor(facts, anchor["id"]),
                        key=lambda fact: (
                            -float(scores.get(node_memory.object_id_of(fact), 0.0)),
                            str(fact.get("updated_at") or ""),
                            str(fact.get("id") or ""),
                        ),
                    )
                    for fact in owned[:per_anchor]:
                        fact_id = node_memory.object_id_of(fact)
                        if fact_id in known:
                            continue
                        candidate = self._candidate(
                            fact,
                            max(float(scores.get(fact_id, 0.0)), 0.0),
                            role_uuid,
                            endpoint_names,
                        )
                        candidate["match_type"] = "direct"
                        candidate["anchor_recall"] = True
                        candidate["anchor_name"] = anchor["name"]
                        decorate(candidate)
                        direct.append(candidate)
                        known.add(fact_id)

            direct = self._collapse_near_duplicate_facts(direct, diagnostics)
            direct_limit = max(1, int(policy.direct_top_k))
            direct = self._limit_direct_with_anchor_reserve(
                direct, anchors, direct_limit, policy, diagnostics
            )
            diagnostics["candidate_counts"]["direct_fact"] = len(direct)
            mark("direct_facts")

            # Only the **first** direct hit entity seeds the walk (configurable to a
            # larger number for diagnostics). A single anchor is what makes this a
            # test of the graph's associations rather than of "many seeds fill the
            # candidate pool": everything after the first hop is reachable only
            # through the category graph.
            seed_limit = max(1, int(getattr(policy, "expansion_seed_count", 1)))
            if anchors:
                # One seed per named entity, capped so a query that names four
                # entities cannot turn the walk into a four-root flood.
                seed_limit = min(
                    max(seed_limit, len(anchors)),
                    max(1, int(getattr(policy, "anchor_seed_limit", 3))),
                )
            # "首个命中实体" means the first **admitted** hit, not merely the
            # first candidate above the rough semantic pre-filter. In cosine-only
            # mode the admission threshold is the direct gate, so a weak 0.26
            # candidate (e.g. an unrelated query's best match) must not start a
            # depth-3 walk that then manufactures a 0.46 false positive.
            seed_floor = (
                float(policy.direct_rerank_threshold)
                if not bool(getattr(policy, "rerank_enabled", True))
                else float(policy.item_semantic_threshold)
            )
            seeds: Dict[str, float] = {}
            for candidate in direct:
                parent = str(candidate.get("entity_id") or "")
                if not parent or parent in seeds:
                    continue
                semantic = float(candidate.get("semantic_score") or 0.0)
                if semantic < seed_floor:
                    continue
                seeds[parent] = semantic
                if len(seeds) >= seed_limit:
                    break
            if len(seeds) < seed_limit:
                # A query that names an entity (not a fact) can only seed from the
                # entity vector hits; still stop at the configured count so the
                # first hit remains the anchor.
                entity_hits = []
                for record in self.topology.memory_records_for_role(role_uuid):
                    if node_memory.object_type_of(record) != "node":
                        continue
                    object_id = node_memory.object_id_of(record)
                    if object_id not in graph.entity_names or object_id in seeds:
                        continue
                    semantic = float(scores.get(object_id, 0.0))
                    if semantic > 0.0 and semantic >= max(item_floor, seed_floor):
                        entity_hits.append((object_id, semantic))
                entity_hits.sort(key=lambda pair: (-pair[1], pair[0]))
                for object_id, semantic in entity_hits[: seed_limit - len(seeds)]:
                    seeds[object_id] = semantic
            diagnostics["expansion_seeds"] = [
                {"id": key, "strength": round(value, 6)} for key, value in seeds.items()
            ]
            diagnostics["expansion_seed_limit"] = seed_limit

            candidates = list(direct)
            traversed_relations: set[str] = set()
            if seeds and bool(getattr(policy, "expansion_enabled", True)):
                walked = graph.walk(
                    {key: (value, "") for key, value in seeds.items()},
                    depth=int(getattr(policy, "expansion_max_depth", 2)),
                    strategy=str(getattr(policy, "expansion_strategy", "bfs")),
                    max_candidates=int(getattr(policy, "expansion_max_candidates", 60)),
                    neighbors_per_seed=int(
                        getattr(policy, "expansion_neighbors_per_seed", 5)
                    ),
                    facts_per_entity=int(
                        getattr(policy, "expansion_facts_per_entity", 4)
                    ),
                    decay=float(getattr(policy, "expansion_hop_decay", 0.70)),
                    scores=scores,
                    hub_strength=float(getattr(policy, "hub_penalty_strength", 0.15)),
                    depth_bonus=float(getattr(policy, "expansion_depth_bonus", 0.0)),
                    edge_weight=self._edge_weight_for(role_uuid),
                    traversed=traversed_relations,
                )
                diagnostics["traversed_relations"] = sorted(traversed_relations)
                known = {str(item["id"]): item for item in candidates}
                expanded: List[Dict[str, Any]] = []
                boosted: List[Dict[str, Any]] = []
                for walked_fact in walked:
                    fact_id = node_memory.object_id_of(walked_fact.record)
                    if fact_id in known:
                        # A fact can be both a direct cosine hit and reachable
                        # through the association walk. It must not be emitted
                        # twice, but it also must not lose the depth weight just
                        # because it happened to clear the semantic pre-filter:
                        # the depth-3 fact with cosine 0.44 is exactly the one the
                        # gate rejects without the association boost.
                        existing = known[fact_id]
                        if int(walked_fact.depth) <= 0:
                            continue
                        lifted = max(
                            float(existing.get("semantic_score") or 0.0),
                            float(walked_fact.strength),
                        )
                        existing["path_score"] = max(
                            float(existing.get("path_score") or 0.0),
                            float(walked_fact.path_score),
                        )
                        if lifted > float(existing.get("semantic_score") or 0.0) + 1e-9:
                            existing["semantic_score"] = lifted
                            existing["similarity"] = lifted
                            existing["bfs_association"] = True
                            existing["hop"] = int(walked_fact.depth)
                            existing["path"] = str(walked_fact.path or "")
                            existing["expanded_from"] = walked_fact.seed_id
                            existing["hub_factor"] = round(float(walked_fact.hub_factor), 6)
                            existing["depth_weight"] = round(float(walked_fact.depth_weight), 6)
                            path = str(walked_fact.path or "")
                            if path:
                                base = str(
                                    existing.get("base_content")
                                    or existing.get("content")
                                    or ""
                                )
                                existing["content"] = base + "\n（" + path + "）"
                                existing["content_hash"] = text_hash(str(existing["content"]))
                            boosted.append(
                                {
                                    "id": fact_id,
                                    "path": existing.get("path"),
                                    "from": existing.get("expanded_from"),
                                    "hop": existing.get("hop"),
                                    "depth_weight": existing.get("depth_weight"),
                                    "semantic_score": round(lifted, 6),
                                }
                            )
                        continue
                    own_score = float(scores.get(fact_id, 0.0))
                    if 0.0 < own_score < float(
                        getattr(policy, "chain_semantic_floor", 0.0)
                    ):
                        # Reachable is not the same as relevant: a hop that lands on a
                        # sentence with no textual relation to the query at all is
                        # graph noise, not a chain. Keep the path for the record.
                        diagnostics["evicted_by_chain_floor"].append(fact_id)
                        continue
                    candidate = self._candidate(
                        walked_fact.record,
                        walked_fact.strength,
                        role_uuid,
                        endpoint_names,
                    )
                    candidate["match_type"] = "indirect"
                    candidate["path_score"] = float(walked_fact.path_score)
                    candidate["hop"] = int(walked_fact.depth)
                    candidate["path"] = str(walked_fact.path or "")
                    candidate["expanded_from"] = walked_fact.seed_id
                    candidate["hub_factor"] = round(float(walked_fact.hub_factor), 6)
                    candidate["depth_weight"] = round(float(walked_fact.depth_weight), 6)
                    decorate(candidate)
                    if walked_fact.path:
                        candidate["content"] = (
                            str(candidate.get("base_content") or candidate.get("content") or "")
                            + "\n（"
                            + str(walked_fact.path)
                            + "）"
                        )
                        candidate["content_hash"] = text_hash(str(candidate["content"]))
                    expanded.append(candidate)
                diagnostics["expanded_candidates"] = [
                    {
                        "id": item["id"],
                        "path": item.get("path"),
                        "from": item.get("expanded_from"),
                        "hop": item.get("hop"),
                        "depth_weight": item.get("depth_weight"),
                        "hub_factor": item.get("hub_factor"),
                    }
                    for item in expanded
                ]
                diagnostics["associative_boosts"] = boosted
                candidates.extend(expanded)
                diagnostics["candidate_counts"]["expanded_fact"] = len(expanded)
                diagnostics["candidate_counts"]["associative_boost"] = len(boosted)
            mark("expansion")

            candidates = self._collapse_near_duplicate_facts(candidates, diagnostics)
            max_total = max(int(policy.direct_top_k), int(policy.expansion_max_candidates))
            candidates.sort(
                key=lambda item: (-float(item.get("semantic_score") or 0.0), item["id"])
            )
            candidates = candidates[:max_total]
            diagnostics["candidate_counts"]["fact"] = len(candidates)
            mark("dedup")

            scored = self._rerank(query, candidates, diagnostics)
            direct_scored = [
                item for item in scored if str(item.get("match_type")) == "direct"
            ]
            indirect_scored = [
                item for item in scored if str(item.get("match_type")) != "direct"
            ]
            passed = self._gate(
                direct_scored,
                policy,
                diagnostics,
                threshold=policy.direct_rerank_threshold,
            )
            passed.extend(
                self._gate(
                    indirect_scored,
                    policy,
                    diagnostics,
                    threshold=policy.indirect_rerank_threshold,
                    reason="indirect",
                )
            )
            passed.sort(
                key=lambda item: (-float(item.get("final_score") or 0.0), item["id"])
            )
            deduped = self._deduplicate_content(passed, diagnostics)
            deduped = self._apply_activation(deduped, role_uuid, diagnostics)
            selected = self._select(
                deduped, effective_top_k, effective_budget, diagnostics
            )
            mark("finalize")
            diagnostics["timings"] = timings
            diagnostics["selected_ids"] = [item.get("id") for item in selected]
            diagnostics["zero_recall"] = not selected

            if not selected and not candidates and bool(
                getattr(policy, "entity_fallback_enabled", True)
            ):
                fallback = self._search_entity_first(
                    role_uuid,
                    query,
                    top_k=top_k,
                    token_budget=token_budget,
                    similarity_threshold=similarity_threshold,
                    query_vector=query_vector,
                )
                fallback.setdefault("diagnostics", {})["fact_first_fallback"] = {
                    "reason": "no_fact_candidates",
                    "fact_candidates": len(candidates),
                }
                fallback["diagnostics"]["fact_first_attempt"] = {
                    "candidate_counts": diagnostics.get("candidate_counts"),
                    "near_duplicate_facts": diagnostics.get("near_duplicate_facts"),
                    "evicted_by_rerank_gate": (diagnostics.get("evicted_by_rerank_gate") or [])[:20],
                    "evicted_by_indirect_gate": (diagnostics.get("evicted_by_indirect_gate") or [])[:20],
                    "expanded_candidates": (diagnostics.get("expanded_candidates") or [])[:20],
                    "expansion_seeds": diagnostics.get("expansion_seeds"),
                }
                return fallback

            return {
                "memories": selected,
                "count": len(selected),
                "token_estimate": sum(
                    int(item.get("token_estimate", 0)) for item in selected
                ),
                "policy_version": policy.version,
                "diagnostics": diagnostics,
            }
        except VectorIndexError as exc:
            failure = self._index_failure(exc)
            failure["diagnostics"]["source"] = "FactFirst"
            return failure

    def _limit_direct_with_anchor_reserve(
        self,
        candidates: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
        limit: int,
        policy: RetrievalPolicy,
        diagnostics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Cut the direct list to length while keeping room for every anchor.

        A global sort by semantic score is exactly what loses the second entity:
        the hub's facts are individually stronger than a low-cosine fact about the
        other person. When at least two anchors are named, reserve a small slot
        per anchor before filling the rest by score. Reserved facts are still
        ordinary candidates; the reranker and gates decide whether they are
        admitted.
        """
        if not candidates:
            return []
        if len(anchors) < 2:
            candidates.sort(
                key=lambda item: (-float(item.get("semantic_score") or 0.0), item["id"])
            )
            return candidates[:limit]
        per_anchor = max(0, int(getattr(policy, "anchor_facts_per_entity", 6)))
        if per_anchor <= 0 or limit <= 0:
            candidates.sort(
                key=lambda item: (-float(item.get("semantic_score") or 0.0), item["id"])
            )
            return candidates[:limit]
        reserved: List[Dict[str, Any]] = []
        reserved_ids: set[str] = set()
        for anchor in anchors:
            if len(reserved) >= limit:
                break
            owned = [
                item for item in candidates
                if bool(item.get("anchor_recall"))
                and str(item.get("entity_id") or "") == str(anchor["id"])
            ]
            owned.sort(
                key=lambda item: (-float(item.get("semantic_score") or 0.0), item["id"])
            )
            for item in owned[:per_anchor]:
                if len(reserved) >= limit:
                    break
                identifier = str(item.get("id") or "")
                if identifier in reserved_ids:
                    continue
                reserved.append(item)
                reserved_ids.add(identifier)
        result = list(reserved)
        for item in sorted(
            candidates, key=lambda row: (-float(row.get("semantic_score") or 0.0), row["id"])
        ):
            if len(result) >= limit:
                break
            if str(item.get("id") or "") in reserved_ids:
                continue
            result.append(item)
        diagnostics["anchor_reserved"] = [
            str(item.get("id")) for item in reserved
        ]
        return result

    @staticmethod
    def _fact_text_for_dedup(candidate: Dict[str, Any]) -> str:
        record = candidate.get("record") or {}
        if node_memory.is_fact_record(record):
            return node_memory.fact_body(record)
        return str(candidate.get("content") or "")

    @staticmethod
    def _dedup_bigrams(text: str) -> set[str]:
        normalized = normalized_text(text)
        if len(normalized) < 2:
            return {normalized} if normalized else set()
        return {normalized[index : index + 2] for index in range(len(normalized) - 1)}

    def _collapse_near_duplicate_facts(
        self,
        candidates: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Collapse paraphrased facts before they consume rerank/budget slots.

        Repeated publication turns produce statements such as "主人喜欢阳台听雨" and
        "主人下雨时坐在阳台听雨很治愈". Exact content hashes do not catch those; a
        character-bigram Jaccard does. The highest-scoring copy wins, and the
        collapsed ids are recorded so the behaviour is auditable.
        """
        threshold = float(getattr(self.policy, "fact_near_duplicate_ratio", 0.62))
        if threshold <= 0 or threshold > 1 or len(candidates) < 2:
            return candidates
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item.get("semantic_score") or 0.0), item["id"]),
        )
        kept: List[tuple[Dict[str, Any], set[str]]] = []
        result: List[Dict[str, Any]] = []
        reported = {
            str(row.get("id"))
            for row in (diagnostics.get("near_duplicate_facts") or [])
            if isinstance(row, dict)
        }
        for candidate in ordered:
            text = self._fact_text_for_dedup(candidate)
            if len(normalized_text(text)) < 12:
                # Short statements such as "喜欢猫" / "喜欢狗" share most bigrams
                # while asserting different things; length is the cheap guard.
                result.append(candidate)
                continue
            grams = self._dedup_bigrams(text)
            duplicate = False
            for existing, existing_grams in kept:
                if not grams or not existing_grams:
                    continue
                union = grams | existing_grams
                if not union:
                    continue
                similarity = len(grams & existing_grams) / len(union)
                if similarity >= threshold:
                    candidate_id = str(candidate.get("id"))
                    if candidate_id not in reported:
                        diagnostics.setdefault("near_duplicate_facts", []).append(
                            {
                                "id": candidate.get("id"),
                                "kept": existing.get("id"),
                                "similarity": round(similarity, 4),
                            }
                        )
                        reported.add(candidate_id)
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append((candidate, grams))
            result.append(candidate)
        return result

    def _empty(self, policy: RetrievalPolicy, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memories": [],
            "count": 0,
            "token_estimate": 0,
            "policy_version": policy.version,
            "diagnostics": diagnostics,
        }

    def _vector_scores(
        self,
        role_uuid: str,
        query_vector: Sequence[float],
        policy: RetrievalPolicy,
        similarity_threshold: float,
    ) -> Dict[str, float]:
        """One vector query returning {object_id: score} for the role."""
        gate = max(float(similarity_threshold), policy.semantic_threshold)
        # Over-fetch so the reranker has something to reorder: the semantic gate is a
        # coarse floor (0.25), and the cut has to be deeper than direct_top_k or
        # reranking could only shuffle the top few. The floor and the cut both run
        # inside Neo4j (see vector_store), so over-fetching costs no row transfer
        # beyond the cut itself.
        over_fetch = max(policy.direct_top_k * max(1, int(policy.candidate_multiplier)), 50)
        hits = self.vectors.search(
            query_vector,
            role_uuid=role_uuid,
            object_types=OBJECT_TYPES,
            top_k=over_fetch,
            score_threshold=gate,
            status="active",
        )
        scores: Dict[str, float] = {}
        for hit in hits:
            object_id = str(hit.get("object_id") or "")
            if not object_id:
                continue
            scores[object_id] = max(scores.get(object_id, 0.0), float(hit.get("score") or 0.0))
        return scores

    def _candidate(
        self,
        record: Dict[str, Any],
        semantic: float,
        role_uuid: str,
        endpoint_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        object_id = node_memory.object_id_of(record)
        elements = node_memory.active_elements(record, role_uuid)
        object_type = node_memory.object_type_of(record)
        entity_id, entity_name = self._owner_entity(record, object_type, endpoint_names)
        return {
            "id": object_id,
            "object_id": object_id,
            "role_uuid": role_uuid,
            "object_type": object_type,
            "source_type": object_type,
            "name": str(record.get("name") or object_id),
            # 母实体节点：这条记忆挂在哪个实体上。召回逐条把它交给客户端，客户端
            # 才能标出"这是谁的事"——同一个角色图谱里既有本机主人的记忆，也有 QQ
            # 联系人的记忆，不标注两边就会串（需求：记忆串扰）。
            "entity_id": entity_id,
            "entity_name": entity_name,
            "record": record,
            "elements": elements,
            # ``base_content`` is the object's identity text before any facts are
            # mounted; every ``_attach_items`` pass rebuilds from it, so a preview pass
            # (before rerank) and the final pass (after admission) cannot stack the
            # same facts twice.
            "base_content": self.render_record(record, role_uuid),
            "content": self.render_record(record, role_uuid),
            "content_hash": node_memory.index_hash(record, role_uuid, endpoint_names=endpoint_names),
            "semantic_score": float(semantic),
            "similarity": float(semantic),
            "rerank_score": 0.0,
            "final_score": 0.0,
            "refs": [object_id],
            "match_type": "direct",
        }

    def _owner_entity(
        self,
        record: Dict[str, Any],
        object_type: str,
        endpoint_names: Optional[Dict[str, str]] = None,
    ) -> tuple[str, str]:
        """这条记忆的母实体节点 ``(id, name)``。

        - 一条 **fact** 挂在它的父实体上（节点细化），母实体就是 ``parent_id``；
        - 一条 **relation** 属于它的两端实体，母实体写成 ``A <-> B``；
        - 其余对象（实体自身、记忆事件）就是它自己。

        名字解析优先用调用方给的 ``endpoint_names``（词法回退时已经建好的一张表），
        没有就回拓扑里查。查不到时回退到 id，绝不返回空串：标注宁可难看，也不能
        悄悄消失——那正是"分不清是谁的记忆"的成因。
        """
        names = endpoint_names or {}

        def resolve(object_id: str) -> str:
            key = str(object_id or "")
            if not key:
                return ""
            known = names.get(key)
            if known:
                return str(known)
            found = (self.topology.data.get("nodes") or {}).get(key)
            return str((found or {}).get("name") or "")

        if object_type == "fact":
            kind = node_memory.fact_kind(record)
            if kind == "self_statement":
                speaker_id = str(record.get("speaker_id") or "")
                if speaker_id:
                    return speaker_id, resolve(speaker_id) or speaker_id
            if kind == "opinion":
                target_id = str(record.get("target_id") or "")
                if target_id:
                    return target_id, resolve(target_id) or target_id
            parent_id = str(record.get("parent_id") or "")
            if parent_id:
                return parent_id, resolve(parent_id) or parent_id
        if object_type == "relation":
            endpoints = [str(item) for item in (record.get("endpoints") or []) if str(item)]
            if endpoints:
                return " ".join(endpoints), " <-> ".join(resolve(item) or item for item in endpoints)
        object_id = node_memory.object_id_of(record)
        return object_id, resolve(object_id) or str(record.get("name") or object_id)

    # --------------------------------------------------------------- rendering

    def touches_focus(self, item: Dict[str, Any], focus_ids: Any) -> bool:
        """该条命中是否"与某个焦点对象有关"（召回加分的判据）。

        结构判据，不做语义猜测：
        - 命中对象本身就是焦点节点（该联系人自己的描述/事实/认知都在这个节点上）；
        - 命中对象是**连到**焦点节点的那条边（"这个联系人与 AI 的关系"属于关于他的记忆）。

        只在"网关已按门槛筛完"之后调用：它是排序偏好，不参与准入门槛，也不放宽
        白名单（候选本来就已经是该角色白名单内的对象）。
        """
        focus = {str(value) for value in (focus_ids or ()) if str(value)}
        if not focus:
            return False
        refs = {str(ref) for ref in (item.get("refs") or [])}
        if refs & focus:
            return True
        if not refs:
            return False
        object_id = next(iter(refs))
        relations = self.topology.data.get("relations") or {}
        links = self.topology.data.get("event_links") or {}
        record = relations.get(object_id) or links.get(object_id)
        if not isinstance(record, dict):
            return False
        endpoints = {str(endpoint) for endpoint in (record.get("endpoints") or [])}
        return bool(endpoints & focus)

    def render_record(self, record: Dict[str, Any], role_uuid: str) -> str:
        """The complete memory of one object, as injected.

        A node hit returns everything the role holds about that node; a relation
        hit returns its endpoints and every element about the edge; a memory event
        returns its own body plus the role's elements about it. This is the "pure
        node query" contract: one object, one complete answer.
        """
        is_relation = "endpoints" in record
        name = str(record.get("name") or node_memory.object_id_of(record))
        object_id = node_memory.object_id_of(record)
        if is_relation:
            endpoints = record.get("endpoints", [])
            names = " <-> ".join(self._endpoint_name(item) for item in endpoints)
            lines = [f"Relation: {names} [{object_id}]"]
        elif node_memory.is_fact_record(record):
            # One knowledge item of one entity (节点细化). Rendered as its own object
            # so an individual fact can be injected, gated or dropped on its own.
            kind = node_memory.fact_kind(record)
            body = escape_node_description(node_memory.fact_body(record))
            if kind == "self_statement":
                speaker = self._endpoint_name(str(record.get("speaker_id") or ""))
                lines = [f"Self-statement of {speaker}: {body} [{object_id}]"]
            elif kind == "opinion":
                speaker = self._endpoint_name(str(record.get("speaker_id") or ""))
                target = self._endpoint_name(str(record.get("target_id") or ""))
                lines = [f"Opinion of {speaker} about {target}: {body} [{object_id}]"]
            else:
                parent = self._endpoint_name(str(record.get("parent_id") or ""))
                lines = [f"Knowledge of {parent}: {body} [{object_id}]"]
        elif node_memory.is_predicate_record(record):
            # A shared predicate is an object two entities meet at. It is rendered as
            # its own subject rather than as one entity's knowledge, because it belongs
            # to no single entity — that sharing is what makes the link.
            lines = [f"Shared: {name} [{object_id}]"]
        elif node_memory.is_hook_record(record):
            # ⚠️ 联想节点是**模型预测**，不是对话陈述过的事实。标签必须让模型一眼看出
            # 这是推测：渲染成 "Knowledge of X" 或 "Node: X" 都会被读成已确认的记忆，
            # 那正是 recall-guard 那类事故的成因。
            parent = self._endpoint_name(str(record.get("parent_id") or ""))
            lines = [
                f"Association (predicted, not stated by anyone): "
                f"{parent} ~ {node_memory.hook_body(record)} [{object_id}]"
            ]
        elif node_memory.is_event_record(record):
            occurred = str(record.get("occurred_at") or record.get("formed_at") or "").strip()
            suffix = f" ({occurred})" if occurred else ""
            lines = [f"Memory: {name}{suffix} [{object_id}]"]
        else:
            lines = [f"Node: {name} [{object_id}]"]
        body = str(record.get("summary") or "").strip() if node_memory.is_event_record(record) else ""
        if body and body != name:
            lines.append(escape_node_description(body))
        description = node_memory.description_for_role(record, role_uuid).strip()
        if description:
            lines.append(escape_node_description(description))
        for element in node_memory.active_elements(record, role_uuid):
            text = str(element.get("text") or "").strip()
            if not text or text == description or text == body:
                continue
            lines.append(f"- {text}")
        return "\n".join(lines)

    def _endpoint_name(self, object_id: str) -> str:
        record = self.topology.data.get("nodes", {}).get(str(object_id))
        return str((record or {}).get("name") or object_id)

    # ------------------------------------------------- stage 2 (节点细化)

    def _split_seeds_and_items(
        self, role_uuid: str, scores: Dict[str, float]
    ) -> tuple[List[tuple[Dict[str, Any], float]], List[tuple[Dict[str, Any], float]]]:
        """Split this role's scored objects into seed objects and fact items.

        A fact is never a seed: recall is "entity first, then the entity's own
        knowledge items" (节点细化), so a fact cannot be injected on its own — it
        arrives as part of the entity it belongs to. ``_nominate_parents`` is what
        keeps a fact-first query from recalling nothing.

        A **hook** is neither a seed nor an item: it is scored (``_attach_hook_
        resemblance`` reads its cosine out of ``scores``) and normally only reorders
        entities that are already candidates. The one narrow exception is handled by
        ``_nominate_hook_parents``: a hook phrase that occurs **verbatim** in the
        query, and resolves to exactly one parent, may nominate that parent. Even then
        the parent still passes the rerank gate; a hook never injects on its own. One
        entity typically carries several hooks, so letting each of them compete as a
        seed would multiply the candidate set for no gain. ``predicate`` is
        deliberately **not** excluded: it is a real shared object, and it becoming a
        candidate is the whole point of the predicate node (命中谓词 → 它两侧的实体可达).
        """
        seeds: List[tuple[Dict[str, Any], float]] = []
        items: List[tuple[Dict[str, Any], float]] = []
        for record in self.topology.memory_records_for_role(role_uuid):
            object_id = node_memory.object_id_of(record)
            semantic = float(scores.get(object_id, 0.0))
            if semantic <= 0:
                continue
            if node_memory.is_hook_record(record):
                # Scored, but not a candidate and not a nominator. See docstring.
                continue
            if node_memory.is_fact_record(record):
                items.append((record, semantic))
            else:
                seeds.append((record, semantic))
        return seeds, items

    def _nominate_parents(
        self,
        role_uuid: str,
        seeds: List[tuple[Dict[str, Any], float]],
        items: List[tuple[Dict[str, Any], float]],
        policy: RetrievalPolicy,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, str]:
        """Let a strongly matching fact nominate its entity into the seed set.

        Without this, "问黑咖啡" would match the fact 主人喜欢喝黑咖啡 but not the
        entity's own identity text, and the entity-first order would recall nothing —
        the failure mode that makes a strict two-stage design unusable. The nominating
        fact only *adds* the entity as a candidate: the entity still has to pass the
        same semantic floor, rerank gate and relative floor as any other seed, and the
        nomination is recorded so the decision stays auditable.
        """
        by_id: Dict[str, tuple[Dict[str, Any], float]] = {
            node_memory.object_id_of(record): (record, semantic) for record, semantic in seeds
        }
        nominated: Dict[str, str] = {}
        for record, semantic in sorted(items, key=lambda pair: -pair[1]):
            if semantic < policy.item_semantic_threshold:
                continue
            parent_id = str(record.get("parent_id") or "")
            if not parent_id:
                continue
            existing = by_id.get(parent_id)
            if existing is not None:
                # An existing seed keeps its own score unless the fact scored higher:
                # the entity is what the reranker judges, so its own signal is the
                # honest one, and the fact's score only fills in when there is none.
                if semantic > float(existing[1]):
                    by_id[parent_id] = (existing[0], semantic)
                continue
            parent = self.topology.find_record(parent_id)
            if parent is None or node_memory.is_fact_record(parent):
                continue
            if not self.topology.is_allowed(role_uuid, parent_id):
                continue
            by_id[parent_id] = (parent, semantic)
            nominated[parent_id] = node_memory.object_id_of(record)
        seeds[:] = list(by_id.values())
        return nominated

    def _nominate_hook_parents(
        self,
        role_uuid: str,
        query: str,
        scores: Dict[str, float],
        seeds: List[tuple[Dict[str, Any], float]],
        policy: RetrievalPolicy,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Let a **verbatim** hook phrase nominate its one parent entity.

        This is the deliberately narrow opening in the old "hooks never admit" rule:

        * only an exact substring match counts — a cosine above the boost floor is not
          enough, because that is what already only reorders candidates;
        * the matched hook must resolve to exactly **one** parent in this role. A hook
          shared by two entities (汤圆/年糕 both carry 追激光笔) does not nominate
          either: the query does not say which one, and guessing would be worse than a
          zero recall;
        * the parent is only added to the candidate set. It still has to pass the same
          rerank gate and relative floor as every other candidate — this method never
          injects anything directly;
        * the matched hook is rendered as a prediction on the parent (see
          _attach_hook_evidence), never as a statement the conversation made.

        Returns {parent_id: {"hooks": [...], ...}} so evidence and diagnostics
        describe the same decision.
        """
        min_chars = max(1, int(getattr(policy, "hook_exact_min_chars", 2)))
        enabled = bool(getattr(policy, "hook_exact_admission_enabled", True))
        if not enabled or not str(query or "").strip():
            return {}
        max_candidates = max(0, int(getattr(policy, "hook_exact_max_candidates", 3)))
        if max_candidates == 0:
            return {}
        matches = self._exact_hook_hits(role_uuid, query, min_chars)
        if not matches:
            return {}

        # One hook text may belong to several parents. Only a unique parent is
        # actionable; shared text is recorded and left out.
        by_parent: Dict[str, Dict[str, Any]] = {}
        ambiguous: List[Dict[str, Any]] = []
        matched_rows: List[Dict[str, Any]] = []
        for text, records in sorted(matches.items()):
            parent_ids = sorted(
                {
                    str(record.get("parent_id") or "")
                    for record in records
                    if str(record.get("parent_id") or "")
                }
            )
            row = {"hook": text, "parent_ids": parent_ids}
            matched_rows.append(row)
            if len(parent_ids) != 1:
                ambiguous.append({**row, "reason": "shared_by_multiple_entities"})
                continue
            parent_id = parent_ids[0]
            parent = self.topology.find_record(parent_id)
            if (
                parent is None
                or node_memory.object_type_of(parent) != "node"
                or str(parent.get("status") or "active") != "active"
            ):
                ambiguous.append({**row, "reason": "parent_not_an_entity"})
                continue
            if not self.topology.is_allowed(role_uuid, parent_id):
                ambiguous.append({**row, "reason": "parent_not_whitelisted"})
                continue
            entry = by_parent.setdefault(
                parent_id,
                {"parent_id": parent_id, "parent_name": str(parent.get("name") or parent_id), "hooks": []},
            )
            for record in records:
                entry["hooks"].append(
                    {
                        "id": node_memory.object_id_of(record),
                        "text": node_memory.hook_body(record),
                        "semantic": round(
                            float(scores.get(node_memory.object_id_of(record), 0.0)), 6
                        ),
                    }
                )

        # Strongest first, bounded: a query full of shared associative words must not
        # turn into an unbounded second candidate source.
        ranked = sorted(
            by_parent.values(),
            key=lambda entry: (
                -max((float(hook.get("semantic") or 0.0) for hook in entry["hooks"]), default=0.0),
                entry["parent_id"],
            ),
        )[:max_candidates]
        admitted = {entry["parent_id"]: entry for entry in ranked}

        by_id: Dict[str, tuple[Dict[str, Any], float]] = {
            node_memory.object_id_of(record): (record, semantic) for record, semantic in seeds
        }
        added = 0
        for entry in admitted.values():
            parent_id = entry["parent_id"]
            if parent_id in by_id:
                continue
            parent = self.topology.find_record(parent_id)
            if parent is None:
                admitted.pop(parent_id, None)
                continue
            semantic = max(
                [float(hook.get("semantic") or 0.0) for hook in entry["hooks"]] or [0.0]
            )
            by_id[parent_id] = (parent, semantic)
            added += 1
        seeds[:] = list(by_id.values())

        diagnostics["hook_exact"] = {
            "enabled": True,
            "min_chars": min_chars,
            "matched": matched_rows,
            "ambiguous": ambiguous,
            "nominated_entities": [
                {
                    "parent_id": entry["parent_id"],
                    "parent_name": entry["parent_name"],
                    "hook_ids": [hook["id"] for hook in entry["hooks"]],
                }
                for entry in admitted.values()
            ],
            "candidates_added": added,
        }
        return admitted

    @staticmethod
    def _hook_phrase_in_query(phrase: str, query: str) -> bool:
        """True only for a real exact hit, with word boundaries for Latin phrases.

        Chinese has no spaces, so substring is the correct notion of 逐字出现. For
        ASCII hooks a raw substring would match cat inside category; the look-around
        keeps that from becoming a recall path. The phrase is already normalized
        (case-folded, whitespace-collapsed).
        """
        phrase = normalized_text(phrase)
        query = normalized_text(query)
        if not phrase or phrase not in query:
            return False
        if re.fullmatch(r"[a-z0-9][a-z0-9 ._'-]*", phrase):
            return re.search(
                rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", query
            ) is not None
        return True

    def _exact_hook_hits(
        self, role_uuid: str, query: str, min_chars: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group this role's active hooks by the ones that occur verbatim in query."""
        hits: Dict[str, List[Dict[str, Any]]] = {}
        for record in self.topology.hook_records_for_role(role_uuid):
            if str(record.get("status") or "active") != "active":
                continue
            text = normalized_text(node_memory.hook_body(record))
            if len(text) < max(1, int(min_chars)):
                continue
            if not self._hook_phrase_in_query(text, query):
                continue
            hits.setdefault(text, []).append(record)
        return hits

    def _attach_hook_evidence(
        self,
        candidates: List[Dict[str, Any]],
        admissions: Dict[str, Dict[str, Any]],
    ) -> None:
        """Mount the exact-hit hook on its parent, clearly marked as predicted.

        The line deliberately repeats the hook renderer's wording: a model reading the
        injected block must be able to tell this is an association, not something the
        conversation stated. The hook remains a separate topology object; nothing is
        written onto the parent.
        """
        if not admissions:
            return
        for candidate in candidates:
            entry = admissions.get(str(candidate.get("id") or ""))
            if entry is None:
                continue
            lines: List[str] = []
            seen: set[str] = set()
            for hook in entry.get("hooks") or []:
                text = str(hook.get("text") or "").strip()
                hook_id = str(hook.get("id") or "")
                if not text or hook_id in seen:
                    continue
                seen.add(hook_id)
                lines.append(
                    "Association (predicted, not stated by anyone): "
                    f"{candidate.get('name') or entry.get('parent_name') or '?'} ~ {text} [{hook_id}]"
                )
            if not lines:
                continue
            base = str(candidate.get("base_content") or candidate.get("content") or "").strip()
            candidate["base_content"] = "\n".join([base] + lines).strip()
            candidate["content"] = candidate["base_content"]
            candidate["content_hash"] = text_hash(str(candidate["content"]))
            candidate["hook_exact"] = True
            candidate["hook_exact_ids"] = [
                str(hook.get("id"))
                for hook in entry.get("hooks") or []
                if str(hook.get("id")) in seen
            ]
            candidate["hook_exact_texts"] = [
                str(hook.get("text"))
                for hook in entry.get("hooks") or []
                if str(hook.get("id")) in seen
            ]
            if str(candidate.get("match_type") or "direct") == "direct":
                candidate["match_type"] = "hook"

    def _attach_items(
        self,
        candidates: List[Dict[str, Any]],
        items: List[tuple[Dict[str, Any], float]],
        policy: RetrievalPolicy,
        role_uuid: str,
        endpoint_names: Optional[Dict[str, str]],
        diagnostics: Dict[str, Any],
        *,
        preview_limit: Optional[int] = None,
    ) -> None:
        """Mount each candidate entity's admitted facts onto its rendered answer.

        An item is admitted on **its own** semantic score, then bounded three ways:
        a floor, a relative floor inside the entity, and a per-entity cap. Items whose
        entity is not a candidate simply do not appear — that is the whole point of
        entity-first recall.

        ``preview_limit`` mounts only the strongest N items and is used **before**
        reranking: the reranker must judge the evidence that will actually be injected,
        not a bare entity name. It never changes the admission rules; the final pass
        (``preview_limit=None``) rebuilds the same answer from ``base_content`` and
        mounts every admitted item, so preview and final cannot stack duplicates.
        """
        from . import node_memory

        by_parent: Dict[str, List[tuple[Dict[str, Any], float]]] = {}
        for record, semantic in items:
            by_parent.setdefault(str(record.get("parent_id") or ""), []).append((record, semantic))
        floor = float(policy.item_semantic_threshold)
        cap = max(0, int(policy.max_items_per_entity))
        limit = cap if preview_limit is None else min(cap, max(0, int(preview_limit)))
        relative = float(policy.item_relative_floor_ratio)
        previewing = preview_limit is not None
        for candidate in candidates:
            if candidate["object_type"] != "node":
                continue
            pool = by_parent.get(str(candidate["id"])) or []
            admitted: List[Dict[str, Any]] = []
            for record, semantic in sorted(pool, key=lambda pair: -pair[1]):
                item_id = node_memory.object_id_of(record)
                if semantic < floor:
                    if not previewing:
                        diagnostics.setdefault("evicted_by_item_floor", []).append(
                            {"id": item_id, "parent_id": candidate["id"], "semantic_score": round(semantic, 6)}
                        )
                    continue
                if admitted and semantic < relative * float(admitted[0]["semantic_score"]):
                    if not previewing:
                        diagnostics.setdefault("evicted_by_item_floor", []).append(
                            {"id": item_id, "parent_id": candidate["id"], "semantic_score": round(semantic, 6)}
                        )
                    continue
                if len(admitted) >= limit:
                    if not previewing:
                        diagnostics.setdefault("evicted_by_item_cap", []).append(
                            {"id": item_id, "parent_id": candidate["id"]}
                        )
                    continue
                admitted.append(
                    {
                        "id": item_id,
                        "text": node_memory.fact_body(record),
                        "kind": str(record.get("fact_kind") or "fact"),
                        "semantic_score": round(semantic, 6),
                    }
                )
            if not admitted:
                continue
            # Rebuild from the identity text, never from the previous pass: the
            # preview call and the final call would otherwise append the same facts
            # twice (and the content hash would drift between rerank and injection).
            base = str(candidate.get("base_content") or candidate.get("content") or "")
            candidate["content"] = "\n".join(
                [base] + [f"- {item['text']}" for item in admitted]
            ).strip()
            # The hash must cover the assembled answer: it is what duplicate-content
            # dedup compares, and two entities that end up carrying the same sentences
            # should collapse no matter how the text was stored.
            candidate["content_hash"] = text_hash(str(candidate["content"]))
            candidate["items"] = admitted
            if previewing:
                diagnostics["items_previewed"] = int(diagnostics.get("items_previewed", 0)) + len(admitted)
            else:
                diagnostics["items_attached"] = int(diagnostics.get("items_attached", 0)) + len(admitted)

    # ------------------------------------------------- 有界图扩展（一跳）

    def _neighbors(self, seed: Dict[str, Any], role_uuid: str) -> List[tuple[str, str]]:
        """Objects one hop away from an admitted seed, as ``(id, path_label)``.

        Only the dead-end object types are walked:

        * **predicate** — its two linked entities (that is the whole reason the predicate
          node exists: "命中谓词 → 它两侧的实体可达");
        * **relation** — its two endpoints.

        Events are deliberately not walked (ADR-0143): a memory is recalled on its own,
        never as a path to the entities it mentions.

        An **entity** seed additionally walks predicate intersections when the direct
        result is sparse (``entity -> predicate -> other entity``), so a query that
        names one side can still reach "the other thing connected the same way".
        hooks and facts are never waypoints.
        """
        data = self.topology.data
        relations = data.get("relations") or {}
        nodes = data.get("nodes") or {}
        seed_id = str(seed.get("id") or "")
        object_type = str(seed.get("object_type") or "")
        out: List[tuple[str, str]] = []
        seen: set[str] = set()

        def add(other: str, label: str) -> None:
            other = str(other or "")
            if not other or other == seed_id or other in seen:
                return
            seen.add(other)
            out.append((other, label))

        def links(kind: str, node_id: str) -> List[str]:
            others: List[str] = []
            for relation in relations.values():
                if str(relation.get("kind") or "") != kind:
                    continue
                endpoints = [str(item) for item in (relation.get("endpoints") or [])]
                if node_id not in endpoints:
                    continue
                other = endpoints[1] if endpoints[0] == node_id else endpoints[0]
                if other and other not in others:
                    others.append(other)
            return others

        if object_type == "predicate":
            label = f"经「{seed.get('name') or '关联'}」关联"
            for other in links("predicate_link", seed_id):
                add(other, label)
        elif object_type == "relation":
            record = seed.get("record") or {}
            label = f"经关系「{record.get('name') or '相关'}」"
            for other in record.get("endpoints") or []:
                add(str(other), label)

        elif object_type == "node":
            for predicate_id in links("predicate_link", seed_id):
                predicate = nodes.get(predicate_id) or {}
                if str(predicate.get("kind") or "") != "predicate":
                    continue
                label = f"经「{predicate.get('name') or predicate_id}」关联"
                for other in links("predicate_link", predicate_id):
                    add(other, label)
        return out

    def _expand_gated(
        self,
        role_uuid: str,
        gated: List[Dict[str, Any]],
        items: List[tuple[Dict[str, Any], float]],
        policy: RetrievalPolicy,
        endpoint_names: Optional[Dict[str, str]],
        diagnostics: Dict[str, Any],
        effective_top_k: int,
    ) -> List[Dict[str, Any]]:
        """One bounded hop from admitted seeds, as fresh candidates.

        Expansion may only start from objects that already passed the direct gate — a
        rejected candidate is never allowed to nominate a neighbour. Each neighbour is
        a candidate in its own right and must pass its own (indirect) rerank gate, so
        "it lies on a path" is never sufficient reason to inject it.

        Budget: entity seeds expand only when the direct result is sparse; predicate /
        relation / event seeds always expand because they are dead ends on their own.
        """
        if not gated or not bool(getattr(policy, "expansion_enabled", True)):
            return []
        max_total = max(0, int(getattr(policy, "expansion_max_candidates", 60)))
        per_seed = max(0, int(getattr(policy, "expansion_neighbors_per_seed", 5)))
        decay = float(getattr(policy, "expansion_hop_decay", 0.7))
        if max_total == 0 or per_seed == 0:
            return []
        sparse = len(gated) < max(1, int(effective_top_k))
        known = {str(candidate.get("id")) for candidate in gated}
        expanded: Dict[str, Dict[str, Any]] = {}
        for seed in gated:
            object_type = str(seed.get("object_type") or "")
            # Events are independent memory units and deliberately stay out of
            # entity expansion (ADR-0143 / CONTEXT 事件独立召回): a memory is
            # recalled on its own, never as a path to the entities it mentions.
            if object_type in {"fact", "hook", "event"}:
                continue
            if object_type == "node" and not sparse:
                continue
            for neighbor_id, label in self._neighbors(seed, role_uuid)[:per_seed]:
                if neighbor_id in known or neighbor_id in expanded:
                    continue
                record = (self.topology.data.get("nodes") or {}).get(neighbor_id)
                if not record or str(record.get("kind") or "") != "node":
                    # Only knowledge entities are injection targets; facts, hooks,
                    # events, predicates and identity shells are not.
                    continue
                if not self.topology.is_allowed(role_uuid, neighbor_id):
                    continue
                semantic = decay * float(seed.get("semantic_score") or 0.0)
                candidate = self._candidate(record, semantic, role_uuid, endpoint_names)
                candidate["match_type"] = "indirect"
                candidate["hop"] = 1
                candidate["path"] = label
                candidate["expanded_from"] = seed.get("id")
                candidate["base_content"] = f"{candidate['base_content']}\n（{label}）"
                candidate["content"] = candidate["base_content"]
                expanded[neighbor_id] = candidate
                if len(expanded) >= max_total:
                    break
            if len(expanded) >= max_total:
                break
        if expanded:
            diagnostics["expanded_candidates"] = [
                {"id": item_id, "path": candidate.get("path"), "from": candidate.get("expanded_from")}
                for item_id, candidate in expanded.items()
            ]
        return list(expanded.values())
    # ------------------------------------------------- hook 联想加成（改分在 gateway）

    def _attach_hook_resemblance(
        self,
        candidates: List[Dict[str, Any]],
        role_uuid: str,
        scores: Dict[str, float],
        diagnostics: Dict[str, Any],
    ) -> None:
        """给每个已过门槛的候选记下"它的 hook 与本次查询最像有多像"。

        只**记录**，不加分——加分是 ``gateway._apply_hook_boost`` 在全部准入判定之后做的，
        所以这里的相似度永远不会把一条低于门槛的记忆放进来。

        三条刻意的边界：

        1. **hook 自己不做候选**（``_split_seeds_and_items`` 里钩子被排除在 seeds 之外）：
           否则一个实体的每组 hook 都会各自争一个 seed 名额，子图和 token 预算会成倍膨胀；
        2. **相似度不给实体提名**：向量相似度只用于排序。唯一的提名路径是
           ``_nominate_hook_parents`` 的**逐字命中 + 唯一母实体**，而且被提名者仍要过
           重排门槛；拿模糊相似度直接放行才是把联想变成事实；
        3. 命中 hook 与"这个实体已经因为别的原因进了候选"是两件事，这里只负责后者之上的
           排序偏好。

        ``hook_hits`` 进 diagnostics，所以"这次召回到底是不是 hook 起了作用"是可审计的。
        """
        hooks: Dict[str, float] = {}
        for record in self.topology.memory_records_for_role(role_uuid):
            if not node_memory.is_hook_record(record):
                continue
            parent_id = str(record.get("parent_id") or "")
            if not parent_id:
                continue
            semantic = float(scores.get(node_memory.object_id_of(record), 0.0))
            if semantic <= 0.0:
                continue
            if semantic > hooks.get(parent_id, 0.0):
                hooks[parent_id] = semantic
        if not hooks:
            return
        boosted: List[Dict[str, Any]] = []
        for candidate in candidates:
            semantic = hooks.get(str(candidate.get("id") or ""))
            if semantic is None:
                continue
            candidate["hook_semantic"] = round(float(semantic), 6)
            boosted.append(
                {"id": candidate.get("id"), "hook_semantic": round(float(semantic), 6)}
            )
        if boosted:
            diagnostics["hook_hits"] = boosted

    # --------------------------------------------------------------- reranking

    def _rerank(
        self, query: str, candidates: List[Dict[str, Any]], diagnostics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score every candidate, then combine semantic + rerank into ``final_score``.

        The reranker is called **once per batch**, not once per candidate. Its
        per-request overhead dominates (measured: 162 ms for one document, 250 ms for
        a hundred), so batching is what decouples recall latency from the number of
        candidates. An injected plain scoring callable still works — the batched path
        is used only when the adapter offers ``rerank_many``.
        """
        policy = self.policy
        if not bool(getattr(policy, "rerank_enabled", True)):
            # Comparison mode: admission is the candidate's own cosine. Keeping
            # ``rerank_score`` equal to the cosine lets the unchanged gate and
            # relative floor operate on the same number the vector store returned.
            diagnostics["rerank_disabled"] = True
            diagnostics["gate_score_kind"] = "cosine"
            for candidate in candidates:
                candidate["rerank_score"] = float(candidate.get("semantic_score") or 0.0)
                candidate["final_score"] = self._blend_path(policy, candidate)
            return sorted(candidates, key=lambda item: (-item["final_score"], item["id"]))
        if self.reranker is None:
            for candidate in candidates:
                candidate["rerank_score"] = float(candidate.get("semantic_score") or 0.0)
        elif callable(getattr(self.reranker, "rerank_many", None)):
            self._rerank_batched(query, candidates, diagnostics)
        else:
            diagnostics["rerank_calls"] = len(candidates)
            for candidate in candidates:
                candidate["rerank_score"] = self._rerank_score(query, candidate, diagnostics)
        for candidate in candidates:
            base = (
                policy.semantic_weight * candidate["semantic_score"]
                + policy.rerank_weight * candidate["rerank_score"]
            )
            candidate["text_score"] = base
            candidate["final_score"] = self._blend_path(policy, candidate, base)
        return sorted(candidates, key=lambda item: (-item["final_score"], item["id"]))

    @staticmethod
    def _blend_path(
        policy: RetrievalPolicy, candidate: Dict[str, Any], base: Optional[float] = None
    ) -> float:
        """Rank on text evidence, with the path as a bounded second opinion.

        The path score never replaces the text score: a fact reached through a strong
        seed is lifted, but a fact that literally matches the query still wins. Without
        the cap, one seed's neighbourhood (four hop-1 facts at 0.705 in the measured
        case) outranked the direct hit that answered the question.
        """
        text = float(
            base
            if base is not None
            else candidate.get("semantic_score") or 0.0
        )
        weight = float(getattr(policy, "chain_ranking_weight", 0.0) or 0.0)
        path = float(candidate.get("path_score") or 0.0)
        if weight <= 0.0 or path <= 0.0:
            return text
        weight = min(max(weight, 0.0), 0.9)
        return (1.0 - weight) * text + weight * path

    def _rerank_batched(
        self, query: str, candidates: List[Dict[str, Any]], diagnostics: Dict[str, Any]
    ) -> None:
        size = max(1, int(self.policy.rerank_batch_size))
        calls = 0
        for start in range(0, len(candidates), size):
            chunk = candidates[start : start + size]
            contents = [
                str(candidate.get("rerank_content") or candidate.get("content") or "")
                for candidate in chunk
            ]
            try:
                scores = self.reranker.rerank_many(query, contents)
            except Exception as exc:  # noqa: BLE001 - must not be silent (ADR-0151)
                self._rerank_failed(diagnostics, exc)
                for candidate in chunk:
                    candidate["rerank_score"] = float(candidate.get("semantic_score") or 0.0)
                continue
            calls += 1
            if len(scores) != len(chunk):
                # A short/long answer cannot be zipped without shifting every score
                # onto the wrong candidate, which would be worse than not reranking.
                self._rerank_failed(
                    diagnostics,
                    ValueError(f"reranker returned {len(scores)} scores for {len(chunk)} candidates"),
                )
                for candidate in chunk:
                    candidate["rerank_score"] = float(candidate.get("semantic_score") or 0.0)
                continue
            for candidate, score in zip(chunk, scores):
                try:
                    candidate["rerank_score"] = max(0.0, float(score))
                except (TypeError, ValueError):
                    candidate["rerank_score"] = float(candidate.get("semantic_score") or 0.0)
        diagnostics["rerank_calls"] = calls
        diagnostics["rerank_batch_size"] = size

    @staticmethod
    def _rerank_failed(diagnostics: Dict[str, Any], exc: Exception) -> None:
        diagnostics["rerank_degraded"] = True
        diagnostics.setdefault("rerank_errors", [])
        message = f"{type(exc).__name__}: {exc}"
        if message not in diagnostics["rerank_errors"]:
            diagnostics["rerank_errors"].append(message)

    def _rerank_score(
        self, query: str, candidate: Dict[str, Any], diagnostics: Dict[str, Any]
    ) -> float:
        content = str(candidate.get("rerank_content") or candidate.get("content") or "")
        if self.reranker is None:
            return float(candidate.get("semantic_score") or 0.0)
        try:
            value = self.reranker(query, content)
        except TypeError:
            value = self.reranker({"query": query, "content": content, "candidate": candidate})
        except Exception as exc:  # noqa: BLE001 - must not be silent (ADR-0151)
            diagnostics["rerank_degraded"] = True
            diagnostics.setdefault("rerank_errors", [])
            message = f"{type(exc).__name__}: {exc}"
            if message not in diagnostics["rerank_errors"]:
                diagnostics["rerank_errors"].append(message)
            return float(candidate.get("semantic_score") or 0.0)
        score: Any = value
        if isinstance(score, dict):
            score = score.get("score", score.get("rerank_score", 0.0))
        try:
            return max(0.0, float(score))
        except (TypeError, ValueError):
            diagnostics["rerank_degraded"] = True
            return 0.0

    # ----------------------------------------------------------------- gating

    def _gate(
        self,
        scored: List[Dict[str, Any]],
        policy: RetrievalPolicy,
        diagnostics: Dict[str, Any],
        *,
        threshold: Optional[float] = None,
        reason: str = "direct",
    ) -> List[Dict[str, Any]]:
        """Apply the rerank gate, then the relative floor inside this bucket.

        ``threshold`` lets the caller use a different floor for expanded (indirect)
        candidates: a neighbour reached through a path is not a direct hit, so it has
        to clear its **own** (higher) bar instead of riding on the seed's score.
        """
        gate = float(policy.direct_rerank_threshold if threshold is None else threshold)
        gate_key = "evicted_by_rerank_gate" if reason == "direct" else "evicted_by_indirect_gate"
        floor_key = "evicted_by_relative_floor" if reason == "direct" else "evicted_by_indirect_floor"
        passed = []
        for candidate in scored:
            rerank = float(candidate.get("rerank_score") or 0.0)
            if reason != "direct" and bool(getattr(policy, "chain_gate_enabled", False)):
                # A fact reached from a strong anchor is admitted on the path's
                # evidence. Its own text score is what made chain recall impossible:
                # "545 怎么带" cannot be textually similar to "4 发哈姆 + 3 发巡航",
                # which is exactly why the link, not the sentence, is the evidence.
                rerank = max(rerank, float(candidate.get("path_score") or 0.0))
            if rerank < gate:
                diagnostics.setdefault(gate_key, []).append(
                    {"id": candidate.get("id"), "rerank_score": round(rerank, 6)}
                )
                continue
            passed.append(candidate)
        if not passed:
            return []
        top_score = max(float(item.get("final_score") or 0.0) for item in passed)
        floor = policy.relative_floor_ratio * top_score
        kept = []
        for candidate in passed:
            final = float(candidate.get("final_score") or 0.0)
            if final < floor:
                diagnostics.setdefault(floor_key, []).append(
                    {"id": candidate.get("id"), "final_score": round(final, 6), "floor": round(floor, 6)}
                )
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _deduplicate_content(
        candidates: List[Dict[str, Any]], diagnostics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for candidate in candidates:
            key = str(candidate.get("content_hash") or f"id:{candidate.get('id')}")
            existing = best.get(key)
            if existing is None:
                best[key] = candidate
                order.append(key)
                continue
            if float(candidate.get("final_score") or 0.0) > float(existing.get("final_score") or 0.0):
                diagnostics["evicted_duplicate_content"].append(existing.get("id"))
                best[key] = candidate
            else:
                diagnostics["evicted_duplicate_content"].append(candidate.get("id"))
        return [best[key] for key in order]

    # --------------------------------------------------------------- selection

    def _edge_weight_for(self, role_uuid: str):
        store = self.activation
        if store is None or not bool(getattr(self.policy, "activation_edge_weighting", False)):
            return None
        if not hasattr(store, "edge_weight_for"):
            return None

        def weigh(relation_id: str) -> float:
            return float(store.edge_weight_for(role_uuid, relation_id))

        return weigh

    def _apply_activation(
        self,
        items: List[Dict[str, Any]],
        role_uuid: str,
        diagnostics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Add a bounded ranking bonus for previously recalled objects.

        Applied **after** `_gate` and before `_select`, so it can only change the
        order of memories that already passed every admission gate. It never
        touches `semantic_score`, `rerank_score` or `gate_score`, and it cannot
        raise the relative floor for the other candidates.
        """

        policy = self.policy
        store = self.activation
        if store is None or not bool(getattr(policy, "activation_enabled", False)):
            return items
        object_ids = [str(item.get("id") or "") for item in items if item.get("id")]
        try:
            if hasattr(store, "weights_for"):
                weights = store.weights_for(role_uuid, object_ids)
            else:
                weights = {
                    object_id: store.weight_for(role_uuid, object_id)
                    for object_id in object_ids
                }
        except Exception as exc:  # noqa: BLE001 - an authoritative activation file is fail-closed
            diagnostics["activation"] = {"status": "failed", "error": str(exc)}
            raise
        if not weights:
            diagnostics["activation"] = {"status": "ok", "boosted": 0}
            return items
        floor = float(getattr(policy, "activation_weight_floor", 0.25))
        maximum = float(getattr(policy, "activation_weight_max", 1.0))
        span = max(1e-9, maximum - floor)
        bonus_max = float(getattr(policy, "activation_rank_bonus", 0.0))
        boosted: List[Dict[str, Any]] = []
        marked: List[Dict[str, Any]] = []
        for item in items:
            weight = weights.get(str(item.get("id") or ""))
            if weight is None:
                boosted.append(item)
                continue
            normalized = max(0.0, min(1.0, (float(weight) - floor) / span))
            bonus = max(0.0, bonus_max) * normalized
            if bonus <= 0:
                boosted.append(item)
                continue
            base = float(item.get("final_score") or 0.0)
            updated = {
                **item,
                "activation_weight": round(float(weight), 6),
                "activation_bonus": round(bonus, 6),
            }
            updated["final_score"] = round(min(1.0, base + bonus), 6)
            boosted.append(updated)
            marked.append(
                {
                    "id": item.get("id"),
                    "weight": round(float(weight), 6),
                    "bonus": round(bonus, 6),
                    "final_score_before": round(base, 6),
                    "final_score_after": updated["final_score"],
                }
            )
        diagnostics["activation"] = {
            "status": "ok",
            "boosted": len(marked),
            "items": marked,
            "weight_floor": floor,
            "weight_max": maximum,
        }
        return boosted

    def _select(
        self,
        scored: List[Dict[str, Any]],
        top_k: int,
        token_budget: int,
        diagnostics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Greedy pack: keep the best candidates that still fit the budget.

        The overflow branch used to ``break``, so a single large object discarded
        every candidate behind it — including small ones that would have fit. It now
        skips just that object (``continue``), which is what a budget is for: the
        selection depends on size as well as score, not on position.
        """
        selected: List[Dict[str, Any]] = []
        used = 0
        for candidate in scored:
            if len(selected) >= max(0, top_k):
                break
            item = self._public(candidate)
            cost = max(0, int(item["token_estimate"]))
            if selected and used + cost > token_budget:
                diagnostics.setdefault("evicted_ids", []).append(item.get("id"))
                diagnostics.setdefault("budget_evicted", []).append(
                    {"id": item.get("id"), "token_estimate": cost, "used": used}
                )
                continue
            selected.append(item)
            used += cost
        return selected

    def _public(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            key: value
            for key, value in candidate.items()
            if key not in {"record", "elements", "base_content", "rerank_content", "role_uuid"}
        }
        item["similarity"] = round(float(item.get("similarity", 0.0)), 6)
        item["semantic_score"] = round(float(item.get("semantic_score", 0.0)), 6)
        item["rerank_score"] = round(float(item.get("rerank_score", 0.0)), 6)
        item["final_score"] = round(float(item.get("final_score", 0.0)), 6)
        item["path_score"] = round(float(item.get("path_score") or 0.0), 6)
        item["prompt_content"] = self._prompt_content(candidate)
        # The budget must be spent on what is actually injected. The audit copy
        # keeps ids and the path annotation; charging the prompt for them
        # overstates every hit and evicts memories that would have fit.
        item["token_estimate"] = max(0, int(self.token_counter(item["prompt_content"])))
        return item

    def _prompt_content(self, candidate: Dict[str, Any]) -> str:
        """The copy injected into the model prompt (compact when enabled)."""
        content = str(candidate.get("content") or "")
        if not bool(getattr(self.policy, "compact_prompt_content", False)):
            return prompt_text(content)
        record = candidate.get("record") or {}
        if not isinstance(record, dict) or not record:
            return prompt_text(content)
        return prompt_text(
            self.compact_render(record, str(candidate.get("role_uuid") or ""))
        )

    def compact_render(self, record: Dict[str, Any], role_uuid: str) -> str:
        """Render an object as the sentence itself, without labels or ids.

        The injected line already carries its source label and owning entity from
        the client, so repeating "Knowledge of X: description:" inside the text
        names the entity twice and costs roughly a quarter of a short fact. The
        audit view keeps content unchanged.

        Perspective facts are the exception: the client label names the target
        entity, not the speaker, so dropping "B 认为" would make the model read the
        AI's opinion as the target's own property. Keep a two-character speaker
        prefix for exactly these two fact kinds.
        """
        if node_memory.is_fact_record(record):
            body = node_memory.fact_body(record)
            kind = node_memory.fact_kind(record)
            if kind == "self_statement":
                speaker = self._endpoint_name(str(record.get("speaker_id") or ""))
                return f"{speaker}自述：{body}"
            if kind == "opinion":
                speaker = self._endpoint_name(str(record.get("speaker_id") or ""))
                return f"{speaker}认为：{body}"
            return body
        if node_memory.is_predicate_record(record):
            lines = [node_memory.predicate_body(record)]
        elif node_memory.is_hook_record(record):
            lines = [f"{node_memory.hook_body(record)}（推测）"]
        elif node_memory.is_event_record(record):
            lines = [str(record.get("summary") or record.get("name") or "").strip()]
        elif "endpoints" in record:
            lines = [
                " <-> ".join(
                    self._endpoint_name(item) for item in record.get("endpoints") or []
                )
            ]
        else:
            lines = []
        name = str(record.get("name") or "").strip()
        description = node_memory.description_for_role(record, role_uuid).strip()
        base = "\n".join(lines).strip()
        if not base and description:
            return description
        if description and description not in (base, name):
            lines.append(description)
        for element in node_memory.active_elements(record, role_uuid):
            text = str(element.get("text") or "").strip()
            if not text or text in (description, name) or text == base:
                continue
            if element.get("kind") == "description" and description:
                continue
            lines.append(text)
        return "\n".join(line for line in lines if line).strip() or name

    # --------------------------------------------------------------- freshness

    def vector_freshness(self, role_uuid: str) -> Dict[str, Any]:
        """Diagnostics: how many active objects still lack a vector."""
        layer = self.memory_layer
        store = getattr(layer, "store", None) if layer is not None else None
        if store is None:
            return {"role_uuid": role_uuid, "missing_embeddings": 0, "rebuild_queue": []}
        return {
            "role_uuid": role_uuid,
            "missing_embeddings": store.count_missing_embeddings(role_uuid),
            "rebuild_queue": store.list_embedding_rebuild_queue(limit=50),
            "pending_vector_operations": store.list_vector_operations(status="pending")[:50],
            "failed_vector_operations": store.list_vector_operations(status="failed")[:50],
        }

    # --------------------------------------------------------------- fallback

    def _lexical_fallback(
        self,
        records: List[Dict[str, Any]],
        query: str,
        reason: str,
        policy: RetrievalPolicy,
        diagnostics: Dict[str, Any],
        role_uuid: str = "",
    ) -> Dict[str, Any]:
        """Coverage-based keyword recall, explicitly degraded (no embedding model).

        The lexical score is a query-coverage ratio, not a cosine similarity, so
        the semantic threshold cannot be applied to it directly; the honest gate is
        coverage.
        """
        coverage_gate = max(policy.semantic_threshold, _LEXICAL_COVERAGE_FLOOR)
        endpoint_names = {
            str(node_id): str((node or {}).get("name") or node_id)
            for node_id, node in _nodes_snapshot(self.topology).items()
        }
        scored = []
        for record in records:
            text = node_memory.index_text(record, role_uuid, endpoint_names=endpoint_names)
            score = score_text(query, text)
            if score < coverage_gate:
                continue
            candidate = self._candidate(record, float(score), role_uuid, endpoint_names)
            candidate["rerank_score"] = float(score)
            candidate["final_score"] = float(score)
            scored.append(candidate)
        scored.sort(key=lambda item: (-item["final_score"], item["id"]))
        deduped = self._deduplicate_content(scored, diagnostics)
        selected = self._select(deduped, policy.direct_top_k, policy.token_budget, diagnostics)
        diagnostics.update(
            {
                "status": "lexical_degraded",
                "degraded_reason": reason,
                "thresholds": {"lexical_coverage": coverage_gate},
                "selected_ids": [item.get("id") for item in selected],
                "zero_recall": not selected,
            }
        )
        return {
            "memories": selected,
            "count": len(selected),
            "token_estimate": sum(int(item.get("token_estimate", 0)) for item in selected),
            "policy_version": policy.version,
            "diagnostics": diagnostics,
        }
