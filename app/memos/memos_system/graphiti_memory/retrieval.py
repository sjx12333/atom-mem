"""Versioned global retrieval policy and the shared result budget.

This module holds the retrieval settings that used to live inside the retired
``RecallService``: they are about *how much* may be admitted and with which gates,
not about where memory is stored, so they outlived the atom store unchanged
(ADR-0034: one versioned global policy, captured per session; no role config may
override it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalPolicy:
    """Immutable global retrieval settings captured for one recall request.

    The calibrated defaults are deliberately non-zero (ADR-0151). A zero gate
    never rejects anything, which is exactly how unrelated memories reached the
    prompt: the semantic gate passed every candidate and the selector filled
    ``top_k`` without ever looking at a score.
    """

    version: str = "graphiti-recall-v4"
    semantic_threshold: float = 0.25
    direct_rerank_threshold: float = 0.50
    indirect_rerank_threshold: float = 0.55
    #: When False the calibrated reranker is bypassed and a candidate's own
    #: query/object cosine becomes its admission score. This is the comparison
    #: mode against a plain vector store (native MemOS uses query/item cosine as
    #: its ``relativity`` score): it deliberately removes the cross-encoder's
    #: semantic calibration, so thresholds must be read as cosine thresholds.
    rerank_enabled: bool = True
    relative_floor_ratio: float = 0.40
    semantic_weight: float = 0.5
    rerank_weight: float = 0.5
    direct_top_k: int = 20
    token_budget: int = 8000
    #: Over-fetch multiplier for the semantic pre-filter: candidates are
    #: ``direct_top_k * candidate_multiplier`` objects taken best-first, then reranked
    #: in one batch. The reranker can only reorder what it is given, so this is the
    #: recall/precision knob — shrinking it makes reranking cheaper and blinder.
    candidate_multiplier: int = 4
    #: Documents per rerank request. Measured on qwen3-rerank: 1 document 162 ms,
    #: 100 documents 250 ms, and a document's score is independent of its position in
    #: the batch — but at 100 *long* documents the score drifted by 0.0185, so the
    #: batch stays bounded rather than "as large as possible".
    rerank_batch_size: int = 50
    #: Stage 2 (节点细化): an entity's facts are admitted by **their own** semantic
    #: score, so one weak sentence no longer rides into the prompt just because its
    #: entity was relevant — the thing that was impossible when a statement only
    #: existed as text glued to the entity's single vector.
    item_semantic_threshold: float = 0.25
    #: Per-entity cap on admitted facts: a well-known entity must not eat the budget.
    max_items_per_entity: int = 4
    #: Within one entity, the best item sets the bar and the rest must clear this
    #: fraction of it (same semantics as ``relative_floor_ratio``, one level down).
    item_relative_floor_ratio: float = 0.40

    #: --- Rerank sees the evidence (召回重构修正) --------------------------------
    #: 重排必须看到"最终要注入的内容"，否则一个只有名字的实体会被 0.50 门槛误杀，
    #: 而它携带的事实根本没有参与打分（实测：6 条事实命中并提名了 2 个实体，两个实体
    #: 都因为名字太短被砍，最终零召回）。重排前先挂少量事实作为"预览"，准入后
    #: 再把剩余事实按同一套规则补齐——预览不是最终注入内容的子集承诺，只是重排输入。
    preview_items_per_entity: int = 3

    #: --- 联想节点的精准命中准入 ------------------------------------------------
    #: hook 仍然不是候选；但当一个 hook 的完整短语**逐字出现**在查询里、且该文本在
    #: 本角色内只指向唯一母实体时，允许它把母实体提名为候选（不是直接放行）。母实体
    #: 仍要走正常重排门槛，`hook_exact` 只是把“为什么它进了候选”写进 diagnostics。
    #: 共享 hook（例如汤圆和年糕都有“追激光笔”）不提名：无法判断用户说的是谁。
    hook_exact_admission_enabled: bool = True
    #: 太短的 hook（单字）几乎必然误命中；两字是中文联想短语的最低可用长度。
    hook_exact_min_chars: int = 2
    #: 一次召回最多由 hook 精准提名几个母实体，防止查询里堆满联想词时放大候选集。
    hook_exact_max_candidates: int = 3

    #: --- 有界图扩展（一跳） ----------------------------------------------------
    #: 谓词/关系命中后，它们本身几乎是空的（谓词只渲染 ``Shared: X``，关系只渲染
    #: 两端名字）。不沿边走一步，25 个谓词节点和 51 条 predicate_link 在召回时永远用不上。
    #: 扩展只用**已过门槛的种子**、总候选有硬上限、扩展候选自己再过一次重排门槛。
    expansion_enabled: bool = True
    #: 一次召回最多收集多少个扩展候选（送重排前截断）。
    expansion_max_candidates: int = 60
    #: 每个种子最多沿几条边展开。
    expansion_neighbors_per_seed: int = 5
    #: 扩展候选的语义分衰减：它不是查询直接命中的对象，只是路径可达。
    expansion_hop_decay: float = 0.70
    #: 从多少个直接命中实体出发做 BFS。1 表示只沿**首个**命中实体走；这是联想
    #: 测试的默认形状——首跳锚点唯一，后续可达全由图的连通性决定，而不是用多个
    #: 种子把候选池灌满。
    expansion_seed_count: int = 1
    #: 每跳给内容节点（事实）的**加性**深度权值基数：第 d 跳加
    #: ``expansion_depth_bonus * expansion_hop_decay ** d``。深度越远权重越小，
    #: 但仍在候选自己的余弦之上，使“沿关系找到的内容”更容易越过召回门槛。
    #: 0 表示关闭（旧行为：只取 max(自身余弦, 路径衰减分)）。
    expansion_depth_bonus: float = 0.0

    #: --- Fact-first / category graph (ADR-0175) ---------------------------------
    #: 实体不再是答案候选，而是类别库/导航层：事实是召回单元，实体只负责
    #: 给事实分类、并作为 BFS/DFS 的跳点。默认关闭以保持既有库契约；生产配置
    #: （runtime.DEFAULT_CONFIG）开启。
    fact_first_enabled: bool = False
    #: Method A: resolve every entity named by the query and reserve a small
    #: number of direct facts per anchor. A two-entity question otherwise sees
    #: only the first entity's hub, because the vector top-k is a global ranking.
    #: Library default is off (existing callers keep their contract); production
    #: enables it in runtime.DEFAULT_CONFIG and graphiti_config.json.
    entity_anchoring_enabled: bool = False
    #: Direct facts reserved per resolved anchor (before the normal rerank gate).
    anchor_facts_per_entity: int = 6
    #: Maximum number of anchor entities that may seed the bounded graph walk.
    anchor_seed_limit: int = 3
    #: 角色只有实体、没有事实时，退回旧的实体档案召回，避免降级变成失忆。
    entity_fallback_enabled: bool = True
    #: 类别图上的扩展策略：bfs 先宽后深，dfs 先深后宽；两者都受硬上限约束。
    expansion_strategy: str = "bfs"
    #: 从种子实体出发的最大跳数。2 跳足够覆盖事实 -> 实体 -> 同类/同关系实体。
    expansion_max_depth: int = 2
    #: 每个实体最多带出多少条事实，防止主人这类 hub 一次灌满候选池。
    expansion_facts_per_entity: int = 4
    #: 近重复事实的字符二元组 Jaccard 阈值：达到即折叠，保留分数最高的一条。
    #: 0.62 覆盖“特别/非常”“闻到香菜/闻到”这类同句改写；短事实（少于 12 字）
    #: 不折叠，避免把“喜欢猫/喜欢狗”这种短句误判成重复。
    fact_near_duplicate_ratio: float = 0.62
    #: hub 惩罚强度：扩展语义分乘以 1/(1+strength*log1p(degree))；度越高越弱。
    hub_penalty_strength: float = 0.15
    #: --- 链式召回（v6）：路径证据与文本证据分开 --------------------------------
    #: 旧行为把深度加成直接加进 candidate 的 semantic_score，于是“沿边走到的事实”
    #: 并不按自己的文本相似度排序：实测一次召回里 4 条 hop-1 噪声（机炮、红方兵力、
    #: 雷达告警）都拿到 0.705，压过了真正回答问题的 0.651 直接命中。链式召回要的
    #: 是“沿路径可达”这件事本身作为准入证据，而不是让路径分冒充语义分。
    #: 排序分 = (1-w)*文本分 + w*路径分；路径分 = 种子分 * decay^hop * hub 衰减。
    chain_ranking_weight: float = 0.35
    #: 间接候选的准入用 max(自身重排分, 路径分)：无直接指代的链式事实文本分天然低，
    #: 只看文本分等于永远拒绝它。
    chain_gate_enabled: bool = True
    #: 间接候选自己的文本分下限：路径可达不等于可以塞进一条与查询毫无关系的句子。
    chain_semantic_floor: float = 0.18
    #: 注入副本（prompt_content）是否使用紧凑渲染：事实只留正文，实体/来源由
    #: 客户端字段标注。content 保持完整（含 id 与路径）供 WebUI/审计。
    compact_prompt_content: bool = True
    #: 查询路由：查询未点名任何实体时，RRF 给温层/正文更高权重；点名时给图谱更高。
    query_routing_enabled: bool = False
    #: 两路独立准入后的 RRF 融合。只改排序，不放行低于门槛的记忆。
    rrf_fusion_enabled: bool = False
    #: RRF 的平滑常数；60 是常用值，避免第一名吃掉全部权重。
    rrf_k: int = 60

    #: --- Warm Layer recall (三层记忆架构) --------------------------------------
    #: 温层正文自带一条**独立**的相似度地板，用温层自己算出的余弦（向量缺失时走
    #: 词法回退），所以既不要求实时精排，也不会因为"没有 rerank_score"而绕过任何
    #: 门槛。它同时是"近期对话"这一类来源（温层正文 + 本场工作记忆）的准入门槛。
    #:
    #: 0.15 是曾经的默认值，实测太松：整轮对话正文按句切分后，无关句子与 query 的
    #: 余弦普遍落在 0.2~0.3（云端 text-embedding-v4 这类模型对同语种短句的基线本就
    #: 偏高），于是"最近聊过的每一句"都可能被注入。0.35 是收紧后的默认下界——它只
    #: 排除明显无关，不替代精排。它仍可配置，但配置成 0/负数会被 ``validated()``
    #: 抬回默认值（0 等于"全部放行"，那正是无关召回事故的成因）。
    warm_similarity_threshold: float = 0.35
    #: 同一层内部的相对地板（相对于本层最高分，默认 0.55）。绝对地板负责"排除了
    #: 明显无关"，相对地板负责砍掉"勉强过线但远不如最强一条"的尾巴。最高分那条
    #: **永远不会**被它砍掉，所以"过得了绝对地板就至少召回一条"——收紧召回不会
    #: 退化成零召回。
    warm_relative_floor_ratio: float = 0.55
    #: 这一层的取数上限：温层装的是整批正文，给多了会把图谱知识挤出预算。
    warm_top_k: int = 8
    #: 温层召回改用 RAW-RAG 的“原始窗口 + dense top-k + rerank”机制。
    #: 自动召回仍以向量为主；这里只在温层内部替换取数形状，不引入图检索。
    warm_rag_enabled: bool = True
    warm_rag_chunk_tokens: int = 256
    warm_rag_chunk_overlap: int = 32
    warm_rag_dense_top_k: int = 128
    warm_rag_rerank_top_k: int = 15

    #: --- Chat-log recall (ADR-0163, retired in 阶段 6.1) ------------------------
    #: 聊天记录继续保留 WebUI 展示与回合审计的**写入**路径，但召回来源已经退役到
    #: 温层（温层装的是同一批完整回合，且属于角色而不是 session）。这些字段标记
    #: deprecated、暂不删除，以免破坏既有配置读取；它们不再影响 ``gateway.search()``。
    chatlog_enabled: bool = False
    chatlog_semantic_threshold: float = 0.15
    #: Within the source, relevance and recency are combined for *ordering only*
    #: (the floor above is the admission gate). Recency is measured as distance
    #: from the newest turn, so ``decay ** distance`` never depends on wall time.
    chatlog_semantic_weight: float = 0.6
    chatlog_recency_weight: float = 0.4
    chatlog_recency_decay: float = 0.97
    #: A turn is one injectable unit; the cap keeps the transcript from crowding
    #: out the graph knowledge it is meant to complement.
    chatlog_top_k: int = 4
    chatlog_token_budget: int = 2000
    #: Rolling window of the per-role recall corpus.
    chatlog_max_entries: int = 2000

    #: --- Calendar view (ADR-0178) ---------------------------------------------
    #: 时间表达句子的**按天视图**。它是纯规则侧表，不是抽取来源，也不参与事实图
    #: 的去重；召回只在查询出现时间表达（按天取回）或查询是时间问句（按内容匹配，
    #: 命中的句子带日期）时读它。条目占用调用方的 ``top_k`` 名额，不额外加塞，
    #: 否则「同一预算下比较」就不成立。
    calendar_enabled: bool = True
    #: 一次召回最多注入几条日历句；默认 3，避免一天里的所有句子挤掉图谱事实。
    calendar_top_k: int = 3
    #: 内容模式（时间问句）的词汇覆盖率地板，按 `domain.score_text` 的口径。
    calendar_lexical_floor: float = 0.30

    #: --- Temp-graph turn hit -> one extra entity recall (ADR-0164) -------------
    #: When the session-local temporary graph contributes turns to a result set,
    #: those turns name entities the graph pass did not see (they are archived
    #: before the 2-minute checkpoint). One *batched* entity recall over their
    #: text recovers those objects instead of running once per hit.
    temp_hit_entity_recall: bool = True
    temp_hit_entity_top_k: int = 5

    #: --- Activation weights (强化/遗忘) ---------------------------------------
    #: A recalled object's weight is a *ranking* preference applied after every
    #: admission gate. It can reorder candidates that already passed the gates but
    #: can never admit one below them. The math lives in `activation.py`.
    activation_enabled: bool = True
    #: Maximum additive bonus at `weight == activation_weight_max`. Keep it small:
    #: the calibrated gates and relative floor are deliberately untouched.
    activation_rank_bonus: float = 0.10
    #: Edge weight damps a multi-hop path (multiplier <= 1.0); unactivated edges
    #: stay at 1.0 so existing expansion behaviour is unchanged.
    activation_edge_weighting: bool = True
    activation_weight_floor: float = 0.25
    activation_weight_max: float = 1.0
    activation_initial_weight: float = 0.5
    activation_gain_alpha: float = 0.40
    activation_gain_decay: float = 0.80
    activation_gain_min: float = 0.04
    activation_half_life_hours: float = 168.0
    activation_half_life_growth: float = 1.2
    activation_edge_gain_ratio: float = 0.5

    @classmethod
    def from_value(cls, value: Optional["RetrievalPolicy | Dict[str, Any]"]) -> "RetrievalPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("retrieval policy must be a RetrievalPolicy or mapping")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def validated(self) -> "RetrievalPolicy":
        """Clamp a *configuration-derived* policy back to usable values.

        Called by the config loader only. A gate of ``0`` (or a negative one)
        re-opens the "everything passes" behaviour whenever a client omits its own
        threshold, so the floor is raised to the built-in default instead of being
        honoured. Zero recall is a legal outcome (ADR-0135); silently admitting
        everything is not.

        A policy constructed explicitly in code is used verbatim: callers that
        name their gates are not second-guessed, which is what keeps the gates
        testable.
        """
        defaults = type(self)()
        return type(self)(
            version=str(self.version or defaults.version),
            semantic_threshold=_positive_gate(self.semantic_threshold, defaults.semantic_threshold),
            rerank_enabled=bool(self.rerank_enabled),
            direct_rerank_threshold=_positive_gate(
                self.direct_rerank_threshold, defaults.direct_rerank_threshold
            ),
            indirect_rerank_threshold=_positive_gate(
                self.indirect_rerank_threshold, defaults.indirect_rerank_threshold
            ),
            relative_floor_ratio=_ratio(self.relative_floor_ratio, defaults.relative_floor_ratio),
            semantic_weight=_ratio(self.semantic_weight, defaults.semantic_weight, allow_zero=True),
            rerank_weight=_ratio(self.rerank_weight, defaults.rerank_weight, allow_zero=True),
            direct_top_k=max(0, int(self.direct_top_k)),
            token_budget=max(0, int(self.token_budget)),
            # A multiplier of 0 (or a negative one) would leave the reranker nothing
            # to rank, and a batch size of 0 nothing to send: both fall back to the
            # calibrated defaults instead of silently disabling reranking.
            candidate_multiplier=_positive_int(self.candidate_multiplier, defaults.candidate_multiplier),
            rerank_batch_size=_positive_int(self.rerank_batch_size, defaults.rerank_batch_size),
            item_semantic_threshold=_positive_gate(
                self.item_semantic_threshold, defaults.item_semantic_threshold
            ),
            max_items_per_entity=max(0, int(self.max_items_per_entity)),
            item_relative_floor_ratio=_ratio(
                self.item_relative_floor_ratio, defaults.item_relative_floor_ratio
            ),
            preview_items_per_entity=max(0, int(self.preview_items_per_entity)),
            hook_exact_admission_enabled=bool(self.hook_exact_admission_enabled),
            hook_exact_min_chars=_positive_int(
                self.hook_exact_min_chars, defaults.hook_exact_min_chars
            ),
            hook_exact_max_candidates=max(0, int(self.hook_exact_max_candidates)),
            expansion_enabled=bool(self.expansion_enabled),
            expansion_max_candidates=max(0, int(self.expansion_max_candidates)),
            expansion_neighbors_per_seed=max(0, int(self.expansion_neighbors_per_seed)),
            expansion_hop_decay=_unit_interval(
                self.expansion_hop_decay, defaults.expansion_hop_decay
            ),
            expansion_seed_count=max(1, int(self.expansion_seed_count)),
            expansion_depth_bonus=_ratio(
                self.expansion_depth_bonus, defaults.expansion_depth_bonus, allow_zero=True
            ),
            fact_first_enabled=bool(self.fact_first_enabled),
            entity_anchoring_enabled=bool(self.entity_anchoring_enabled),
            anchor_facts_per_entity=max(0, min(20, int(self.anchor_facts_per_entity))),
            anchor_seed_limit=max(1, min(4, int(self.anchor_seed_limit))),
            entity_fallback_enabled=bool(self.entity_fallback_enabled),
            expansion_strategy=_strategy(self.expansion_strategy, defaults.expansion_strategy),
            expansion_max_depth=max(1, min(3, int(self.expansion_max_depth))),
            expansion_facts_per_entity=max(0, min(20, int(self.expansion_facts_per_entity))),
            fact_near_duplicate_ratio=_ratio(
                self.fact_near_duplicate_ratio, defaults.fact_near_duplicate_ratio
            ),
            hub_penalty_strength=_ratio(
                self.hub_penalty_strength, defaults.hub_penalty_strength, allow_zero=True
            ),
            # A weight of 0 disables the path blend (pure text ranking); 1 would rank
            # by the path alone and let one seed drag in its whole neighbourhood, so
            # the blend is clamped into (0, 1) instead of trusting the config.
            chain_ranking_weight=_ratio(
                self.chain_ranking_weight, defaults.chain_ranking_weight
            ),
            chain_gate_enabled=bool(self.chain_gate_enabled),
            chain_semantic_floor=_ratio(
                self.chain_semantic_floor, defaults.chain_semantic_floor, allow_zero=True
            ),
            compact_prompt_content=bool(self.compact_prompt_content),
            query_routing_enabled=bool(self.query_routing_enabled),
            rrf_fusion_enabled=bool(self.rrf_fusion_enabled),
            rrf_k=_positive_int(self.rrf_k, defaults.rrf_k),
            chatlog_enabled=bool(self.chatlog_enabled),
            # The Warm Layer floor is the mirror image of the graph gate: it must
            # exist (a zero floor would inject every retained turn) but it is not
            # the graph's calibrated distribution, so it has its own value.
            warm_similarity_threshold=_positive_gate(
                self.warm_similarity_threshold, defaults.warm_similarity_threshold
            ),
            # Clamped into (0, 1]: a ratio above 1 would drop even the best item
            # (the zero-recall failure this field must not cause), and 0 falls back
            # to the default like every other gate, so a config edit cannot turn
            # the tightening off by accident. Code that constructs the policy
            # explicitly may still pass 0.0 to disable the relative floor.
            warm_relative_floor_ratio=_unit_interval(
                self.warm_relative_floor_ratio, defaults.warm_relative_floor_ratio
            ),
            warm_top_k=max(0, int(self.warm_top_k)),
            warm_rag_enabled=bool(self.warm_rag_enabled),
            warm_rag_chunk_tokens=max(32, int(self.warm_rag_chunk_tokens)),
            warm_rag_chunk_overlap=max(0, min(128, int(self.warm_rag_chunk_overlap))),
            warm_rag_dense_top_k=max(1, min(512, int(self.warm_rag_dense_top_k))),
            warm_rag_rerank_top_k=max(1, min(64, int(self.warm_rag_rerank_top_k))),
            # The chat-log floor is deliberately *lower* than the graph floor but
            # still non-zero: ADR-0135/0151 allow an empty recall, never "everything
            # the transcript contains".
            chatlog_semantic_threshold=_positive_gate(
                self.chatlog_semantic_threshold, defaults.chatlog_semantic_threshold
            ),
            chatlog_semantic_weight=_ratio(
                self.chatlog_semantic_weight, defaults.chatlog_semantic_weight, allow_zero=True
            ),
            chatlog_recency_weight=_ratio(
                self.chatlog_recency_weight, defaults.chatlog_recency_weight, allow_zero=True
            ),
            # A decay of 0 would make every turn older than the newest score 0
            # (only the last turn could ever win) and a decay above 1 would make
            # the oldest turn the strongest; both are clamped back to the default.
            chatlog_recency_decay=_unit_interval(
                self.chatlog_recency_decay, defaults.chatlog_recency_decay
            ),
            chatlog_top_k=max(0, int(self.chatlog_top_k)),
            chatlog_token_budget=max(0, int(self.chatlog_token_budget)),
            chatlog_max_entries=max(0, int(self.chatlog_max_entries)),
            calendar_enabled=bool(self.calendar_enabled),
            calendar_top_k=max(0, int(self.calendar_top_k)),
            # A floor of 0 would inject every calendar row into every time
            # question; the calibrated default is restored instead.
            calendar_lexical_floor=_positive_gate(
                self.calendar_lexical_floor, defaults.calendar_lexical_floor
            ),
            temp_hit_entity_recall=bool(self.temp_hit_entity_recall),
            temp_hit_entity_top_k=max(0, int(self.temp_hit_entity_top_k)),
            activation_enabled=bool(self.activation_enabled),
            activation_rank_bonus=max(0.0, min(0.5, _ratio(
                self.activation_rank_bonus, defaults.activation_rank_bonus, allow_zero=True
            ))),
            activation_edge_weighting=bool(self.activation_edge_weighting),
            activation_weight_floor=max(0.01, min(0.99, _ratio(
                self.activation_weight_floor, defaults.activation_weight_floor
            ))),
            activation_weight_max=max(
                max(0.01, min(0.99, _ratio(
                    self.activation_weight_floor, defaults.activation_weight_floor
                ))) + 1e-6,
                min(1.0, _ratio(self.activation_weight_max, defaults.activation_weight_max)),
            ),
            activation_initial_weight=max(
                max(0.01, min(0.99, _ratio(
                    self.activation_weight_floor, defaults.activation_weight_floor
                ))),
                min(1.0, _ratio(
                    self.activation_initial_weight, defaults.activation_initial_weight
                )),
            ),
            activation_gain_alpha=_unit_interval(
                self.activation_gain_alpha, defaults.activation_gain_alpha
            ),
            activation_gain_decay=_unit_interval(
                self.activation_gain_decay, defaults.activation_gain_decay
            ),
            activation_gain_min=_unit_interval(
                self.activation_gain_min, defaults.activation_gain_min
            ),
            activation_half_life_hours=_positive_gate(
                self.activation_half_life_hours, defaults.activation_half_life_hours
            ),
            activation_half_life_growth=_ratio(
                self.activation_half_life_growth, defaults.activation_half_life_growth, allow_zero=True
            ),
            activation_edge_gain_ratio=min(1.0, _ratio(
                self.activation_edge_gain_ratio, defaults.activation_edge_gain_ratio, allow_zero=True
            )),
        )


def _strategy(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"bfs", "dfs"} else default


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _positive_gate(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _ratio(value: Any, default: float, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    if number == 0 and not allow_zero:
        return default
    return number


def _unit_interval(value: Any, default: float) -> float:
    """Clamp a decay factor into ``(0, 1]``; anything else falls back to default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not 0 < number <= 1:
        return default
    return number


def escape_node_description(description: str) -> str:
    """Render a memory body as one escaped ``description:`` line.

    Escaping keeps the rendering unambiguous: a description containing newlines or
    the ``Node:`` marker cannot forge a second node entry in the injected block.
    """
    text = str(description or "").replace("\r", " ").replace("\n", " ").strip()
    return f"description: {text}"


#: Internal object ids are useful in the WebUI/audit trail but are noise in the
#: model's context: they invite the model to quote ``f_...`` back at the user and
#: spend tokens on a handle it cannot use.  The path suffix has the same problem:
#: it explains *why* a fact was reached, which belongs in diagnostics, not in the
#: sentence the model reads as a memory.
_AUDIT_ID_RE = re.compile(r"\s*\[(?:f|n|e|r|p|h)_[^\]]+\]")
_AUDIT_PATH_RE = re.compile(r"\s*（经[^）]*）")


def prompt_text(content: str) -> str:
    """Return the text that may be injected into an LLM prompt.

    ``content`` keeps ids and path annotations for audit/WebUI display.  The
    injected copy must not: those are bookkeeping, not memory.  Stripping them at
    the presentation boundary keeps the audit view intact while making the prompt
    shorter and preventing the model from echoing ids.
    """
    text = _AUDIT_PATH_RE.sub("", str(content or ""))
    text = _AUDIT_ID_RE.sub("", text)
    return text.strip()


def apply_budget(
    results: List[Dict[str, Any]],
    top_k: int,
    token_budget: int,
    diagnostics: Optional[Dict[str, Any]] = None,
    *,
    token_counter: Optional[Callable[[str], int]] = None,
) -> List[Dict[str, Any]]:
    """Apply the shared whole-result dual budget to any memory source.

    ``top_k`` is a **hard cap**: it is the number of fragments the caller asked to
    inject, so exceeding it is never correct. The token budget can only make the
    result smaller. The previous version treated the two as one condition ("drop
    only while both are exceeded"), so a generous token budget let the count run
    away — measured at 25 injected fragments for a ``top_k`` of 5.
    """
    counter = token_counter or len
    limit = max(0, int(top_k))
    budget = max(0, int(token_budget))

    selected = list(results)
    # 1. The count cap: drop the weakest first so the best ones stay.
    overflow = len(selected) - limit
    if overflow > 0:
        weakest = sorted(
            selected,
            key=lambda item: (
                float(item.get("final_score") or 0.0),
                int(item.get("token_estimate", counter(str(item.get("content", ""))))),
                str(item.get("id", "")),
            ),
        )[:overflow]
        for item in weakest:
            selected.remove(item)
            if diagnostics is not None:
                diagnostics.setdefault("evicted_ids", []).append(item.get("id"))
    # 2. The token budget: drop the weakest remaining until the batch fits.
    while selected and sum(
        int(item.get("token_estimate", counter(str(item.get("content", "")))))
        for item in selected
    ) > budget:
        removed = min(
            selected,
            key=lambda item: (
                item.get("final_score", 0.0),
                item.get("match_type") == "direct",
                item.get("id", ""),
            ),
        )
        selected.remove(removed)
        if diagnostics is not None:
            diagnostics.setdefault("evicted_ids", []).append(removed.get("id"))
    return selected
