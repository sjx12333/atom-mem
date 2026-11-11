"""Graphiti extraction engine reused from the reference story pipeline.

The reference project (``E:/demo_app``) never calls graphiti's high-level
``add_episode``. Instead it drives the low-level building blocks directly:
``extract_nodes`` -> ``resolve_extracted_nodes`` -> edge extraction/resolution
-> ``add_nodes_and_edges_bulk``, building the ``EpisodicNode`` in memory first.

That distinction matters: ``add_episode`` treats a provided ``uuid`` as an
already-existing episode and calls ``EpisodicNode.get_by_uuid``, which raises
``NodeNotFoundError`` (``node <uuid> not found``) when the episode is new. This
project passed a deterministic ``uuid5(role, conversation)`` to ``add_episode``,
so the *first* extraction of a conversation always failed with
``node 78cef462-... not found``. Building the episode directly and driving the
low-level pipeline avoids that path entirely and matches the intended 9-stage
memory pipeline (extract nodes -> dedup -> extract/resolve edges -> save).

抽取产物只有**知识节点**（实体/关系）：记忆节点（``kind="event"``）由日记通路
创建，本模块不再抽取任何事件，``events`` 只作为兼容字段保留并恒为空数组。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# The vendored graphiti_core lives one level up (plugins-dlc/memos). Make it
# importable regardless of how this module is loaded (pytest collection, the
# app server, or a direct script), mirroring neo4j_bridge/graphiti_factory.
_CORE_ROOT = Path(__file__).resolve().parents[2]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graphiti_core.nodes import EpisodeType, EpisodicNode  # noqa: E402
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT  # noqa: E402
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk  # noqa: E402
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges  # noqa: E402
from graphiti_core.utils.maintenance.node_operations import (  # noqa: E402
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from pydantic import BaseModel, Field  # noqa: E402

from .domain import normalized_text  # noqa: E402
from .runtime import load_config  # noqa: E402


class EmptyExtractionError(RuntimeError):
    """The extractor ran and reported nothing at all.

    Zero entities and zero relations is not the same claim as "this conversation
    contained nothing extractable". The observed cause (see
    ``scripts/graphiti_extraction_resistance_probe.py``) is the extraction model
    spending its whole token budget on reasoning and returning empty content with
    ``finish_reason=length``: the pipeline then publishes an empty batch and the
    conversation silently loses its knowledge.

    Raising makes that visible and retryable instead of silent. A conversation with
    truly nothing extractable is rare enough that an operator should see it once.
    """


class EntityStatement(BaseModel):
    """One statement the conversation makes about one extracted entity."""

    entity: str = Field(description="实体名，必须与给定清单中的名字完全一致")
    text: str = Field(description="一条关于该实体、脱离本次对话仍然成立的陈述")


class AISelfStatement(BaseModel):
    """One durable statement the AI makes about itself."""

    text: str = Field(description="一条 AI 对自己的、脱离本次对话仍然成立的自述")


class AIOpinion(BaseModel):
    """One durable evaluation the AI makes about a named entity."""

    target: str = Field(description="被评价的实体名，必须与给定清单中的名字完全一致")
    text: str = Field(description="一条 AI 对该实体、脱离本次对话仍然成立的看法")


class EntityStatements(BaseModel):
    statements: List[EntityStatement] = Field(default_factory=list)
    self_statements: List[AISelfStatement] = Field(default_factory=list)
    opinions: List[AIOpinion] = Field(default_factory=list)


#: 实体陈述抽取的提示词。
#:
#: 存在的理由：graphiti 的抽取提示词（以及本项目的 ``EXTRACTION_RESTRICTIONS``）明确把
#: **属性、偏好、身份、规则**排除在 ``relations`` 之外 —— "主人喜欢黑咖啡" 是偏好而不是
#: 关系。而实体本身在 graphiti 的 schema 里没有 description 字段（``ExtractedEntity`` 只有
#: 名字/类型/episode 索引），属性通道又因为 ``entity_types=None`` 被关掉（
#: ``_extract_entity_attributes`` 直接返回 ``{}``）。两边合起来的结果是：**偏好和属性
#: 抽出来了实体，却永远挂不到任何东西上**（端到端实测确认：整批只有实体、零条边、零条
#: 陈述）。这里补的就是这条缺失的通道 —— 一次批量调用，输出仍然是陈述，仍要过
#: ``knowledge.is_knowledge_statement``。
_ENTITY_STATEMENT_SYSTEM = (
    "You extract durable facts about named entities. "
    "You ONLY emit statements that the MESSAGES explicitly state about an entity in the "
    "given list. You output strictly the JSON of the response schema — no reasoning, no "
    "explanation, no commentary."
)

_ENTITY_STATEMENT_RULES = """\
You are given CONVERSATION and a list of ENTITIES that were extracted from it. For each
entity, output the statements the conversation makes about it.

HARD RULES — violating any of these is a failure:

1. `entity` MUST be copied exactly from the ENTITIES list. Never invent an entity, never
   use a pronoun, and never use the name of a person who is only being addressed.
2. `text` MUST still be true after this conversation ends. Write the statement, not the
   event that carried it:
   GOOD -> "主人喜欢喝黑咖啡"
   BAD  -> "主人说他喜欢喝黑咖啡" / "先发送你好两次，随后称自己是主人" / "小E回答了自己是谁"
3. NEVER describe what happened in this conversation, who said what, in which order, or
   how the AI felt about it. That belongs to the AI's diary, not here.
4. Static attributes, preferences, habits, rules, roles and identity statements ARE what
   you are here for (the relation extractor deliberately skips them):
   "主人每天早上喝一杯黑咖啡" / "烤糊の酸奶自称是主人的朋友" / "小E是被塞进电脑里的AI宠物".
5. Only what the CONVERSATION states. Never infer from the entity's name, from general
   world knowledge, or from what would be plausible.
6. NEVER copy these instructions, the schema, or any meta text into `text`.
7. If the conversation states nothing durable about an entity, output nothing for it. An
   empty list is a perfectly good answer — do not pad it.

ENTITIES: {entities}

CONVERSATION:
{conversation}
"""


#: AI 视角抽取：同一批对话、同一个 LLM 调用，额外要求模型把"助手自己"从
#: "被谈论的实体"里分出来。实体陈述通道会把"温兆伦不火"挂到温兆伦名下；这
#: 里补的是"这句话是 AI 说的"以及"AI 对自己的自述"。
_AI_PERSPECTIVE_RULES = """You also receive the ASSISTANT's display name and must separate the assistant's own
voice from statements about other entities. Return two extra lists.

SELF_STATEMENTS: durable statements the ASSISTANT makes about itself (its identity,
preferences, habits, history, abilities, feelings). Write each as a standalone
third-person sentence using the ASSISTANT name, not "I".
GOOD -> "{assistant}年轻的时候很喜欢温兆伦"
BAD  -> "我喜欢温兆伦" / "助手说它喜欢温兆伦"

OPINIONS: durable evaluations the ASSISTANT makes about one named entity from the
ENTITIES list (judgments, preferences, recommendations, dislike). The target MUST be
copied exactly from ENTITIES. Write a standalone third-person sentence; do not prefix
it with "assistant said".
GOOD -> "温兆伦不火，但给人非常干净干练的感觉" (target: 温兆伦)
BAD  -> "用户觉得温兆伦不火" / "助手发表了看法"

Do NOT put the user's statements, questions, or pure dialogue narration in either list.
Do NOT invent an opinion: a neutral factual sentence about an entity is not an opinion.
If the assistant says nothing durable about itself or no evaluative opinion, return an
empty list. An empty list is a good answer.

ASSISTANT: {assistant}
"""


#: 二次抽取的追加指令。
#:
#: 它必须单独放在提示词**末尾**：前一次陈述抽取的提示词前缀完全不变，平台的前缀缓存
#: 才能命中（实测同一段对话第二次发送时 3,072/3,223 tokens 命中）。把指令插在中间会
#: 让整个前缀失效，省下的钱又赔回去。
_SECOND_PASS_SUFFIX = """
SECOND PASS — a first extraction over this conversation already ran. Read it again with a
different focus: the concrete details that a summary usually drops — times, dates, places,
prices, quantities, model numbers, distances, physical attributes, and plans.
Copy every number and date exactly as written in the conversation.
Return the same JSON schema with ONLY these additional statements. If nothing is missing,
return empty lists.
"""


def _merge_statements(
    base: List[Dict[str, str]], extra: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Union two statement lists by normalized text, keeping the first wording."""

    seen = {normalized_text(row.get("text")) for row in base}
    merged = list(base)
    for row in extra:
        key = normalized_text(row.get("text"))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _merge_perspective(target: Dict[str, Any], extra: Dict[str, Any]) -> None:
    """Append the second pass's self statements / opinions without duplicating."""

    for key in ("self_statements", "opinions"):
        items = list(target.get(key) or [])
        if key == "self_statements":
            seen = {normalized_text(item) for item in items}
            for item in extra.get(key) or []:
                if normalized_text(item) not in seen:
                    seen.add(normalized_text(item))
                    items.append(item)
        else:
            seen = {(row.get("target"), normalized_text(row.get("text"))) for row in items}
            for row in extra.get(key) or []:
                ident = (row.get("target"), normalized_text(row.get("text")))
                if ident not in seen:
                    seen.add(ident)
                    items.append(row)
        if items:
            target[key] = items


def _apply_dedup_gate(opts: Dict[str, Any]) -> None:
    """Hand the lexical dedup gate its threshold (0 disables it)."""

    from graphiti_core.utils.maintenance import edge_operations as _edge_ops

    threshold = 0.0
    if opts.get("enabled", True) and opts.get("dedup_gate", True):
        try:
            threshold = float(opts.get("dedup_gate_min_similarity") or 0.0)
        except (TypeError, ValueError):
            threshold = 0.0
    if hasattr(_edge_ops, "set_dedup_gate"):
        _edge_ops.set_dedup_gate(threshold)


def extraction_options() -> Dict[str, Any]:
    """The extraction_opt block of graphiti_config.json, with defaults."""

    try:
        return dict((load_config() or {}).get("extraction_opt") or {})
    except Exception:  # noqa: BLE001 - options must never break extraction
        return {}


def _option_enabled(name: str, default: bool = True) -> bool:
    opts = extraction_options()
    if not opts.get("enabled", True):
        return False
    value = opts.get(name, default)
    return bool(value)


class HookProfile(BaseModel):
    """One entity: what it is *in this conversation*, and the hooks of that identity."""

    entity: str = Field(description="实体名，必须与给定清单中的名字完全一致")
    sense: str = Field(
        description=(
            "该实体在本次对话中是什么（身份/情景定位），例如「主人养的猫」；"
            "对话线索不足以确定时输出空字符串"
        )
    )
    hooks: List[str] = Field(
        default_factory=list,
        description="该身份下具体、可检索的联想短语，例如「开罐头的声音」；最多 5 条",
    )


class EntityHooks(BaseModel):
    entities: List[HookProfile] = Field(default_factory=list)


#: 联想节点（hook）抽取的提示词。
#:
#: 与实体陈述**刻意相反**：陈述只能说对话里明确说过的，hook 只能来自模型对实体的一般
#: 认识。所以它不是知识抽取，而是**预测**，产物带 ``basis="model_predicted"``，绝不
#: 能被渲染成"对话陈述过的事实"。
#:
#: 但"一般认识"必须建立在**这次对话里这个实体是什么**之上，而不是名字的字面意思：
#: 「年糕」是猫的名字时，它的联想是"开罐头的声音"，不是"软糯/红枣"。提示词因此把
#: 抽取决断拆成两步——先按对话线索确定 ``sense``（身份/情景定位），再只为这个身份预测
#: hook；`sense` 为空（对话没有交代它是什么）时整条实体不产出 hook。解析层同样把
#: 空 sense 当作"没有可依据的情景"丢弃，避免退回按名字自由联想。
#:
#: 为什么值得单独做：谓词节点只能连**语料里出现过**的关联；hook 能连从未被一起提到过的
#: 东西。猫和开罐器可能永远不在同一句话里出现，但只要"开罐头的声音"同时是猫的 hook 和
#: 开罐器的 hook，两者就在图里交会了。
_HOOK_SYSTEM = (
    "You first determine what an entity is in the given conversation, then predict the "
    "associations of that identity. You output ONLY high-confidence, broadly-known "
    "associations that a look-up would confirm. You never state that anyone said or did "
    "anything. You output strictly the JSON of the response schema — no reasoning, no "
    "explanation."
)

_HOOK_RULES = """\
You are given CONVERSATION, per-entity CONTEXT CLUES and a list of ENTITIES. For each
entity, do these two steps IN ORDER and record both results.

STEP 1 — SENSE: what is this entity IN THIS CONVERSATION?
  A name is only a label. The same label can denote completely different things (a pet,
  a food, a person, a song). The CONTEXT CLUES, not the surface meaning of the name,
  decide the sense.
  WORKED EXAMPLE — entity "年糕", clue "年糕一听到开罐头的声音就跑到厨房":
    sense = "主人养的猫"
    hooks = 开罐头的声音 / 怕打雷 / 晒太阳
    These hooks are WRONG for that sense: 软糯 / 红枣 / 糯米 / 蒸着吃 (they belong to
    the food sense of the label "年糕").
  If the clues do not establish what the entity is, output sense = "" and hooks = [].

STEP 2 — HOOKS FOR THAT SENSE: the concrete associations of the entity as established
  in STEP 1 — what it is drawn to, afraid of, used for, or recognised by.
  A hook must be plausible for the SENSE, not for the literal name. When the two
  disagree, the conversation wins.

WHAT A HOOK IS — a short, searchable association phrase:
  GOOD -> "开罐头的声音" / "晒太阳" / "怕打雷" / "会掉毛"
  BAD  -> "猫" (that is the entity's own classification, not an association)
  BAD  -> "是一种哺乳动物" (a classification, not an association)
  BAD  -> "很可爱" (an opinion, not a searchable association)

HARD RULES — violating any of these is a failure:

1. `entity` MUST be copied exactly from the ENTITIES list. Never invent an entity.
2. `sense` MUST be a short phrase grounded in the CONTEXT CLUES, and MUST NOT be the
   entity's own name. The clues may be quoted, paraphrased, or inferred from what the
   entity does in the conversation — but never from its name alone.
3. `hook` MUST be a short concrete phrase of at most 12 characters: a sound, an object,
   an action, a situation or a sensation that is characteristically tied to the entity.
   Prefer what the entity is *drawn to*, *afraid of*, *used for*, or *recognised by*.
4. NEVER output a hook that belongs to a different sense of the same name. If the sense
   is a pet cat, food associations are a failure even though the name is a food word.
5. Predict ONLY what is broadly and uncontroversially true of this kind of thing. You are
   NOT asserting that this particular instance does it. If you are not confident, output
   nothing for that entity.
6. NEVER output a hook that merely repeats a statement already made about the entity in
   the conversation — that is a fact, not an association, and it is stored elsewhere.
7. NEVER output a person's private details, a guess about what someone feels, or anything
   only true of this one individual. Hooks must be about the entity **type**.
8. 2 to 5 hooks per entity is the right amount. An empty list is a legal answer — never
   pad it. Fewer, better hooks are strictly better than many.

ENTITIES AND THEIR CONTEXT CLUES:
{entities}

CONVERSATION:
{conversation}
"""


#: 每个实体的 hook 抽取上下文上限。线索是"这次对话里它是什么"的唯一依据，但整段对话
#: 已经单独给出，所以每个实体只需少量点名句，避免 token 随实体数线性膨胀。
_HOOK_CONTEXT_LINES = 4
_HOOK_CONTEXT_LINE_CHARS = 240
_HOOK_CONTEXT_STATEMENTS = 4


def hook_entity_contexts(
    body: str,
    entity_names: List[str],
    statements: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Per-entity evidence for the hook pass: what the conversation says about it.

    A hook predictor that sees only names is forced to read the *name* instead of the
    *situation*: that is exactly how 年糕 (the user's cat) acquires food hooks. This
    helper collects the two grounded signals the pipeline already has — the conversation
    lines that mention the entity, and the statements the previous pass anchored to it
    (``{"entity", "text"}``) — so the prompt can ask "what is this entity *here*?"
    before asking for associations. Pure and bounded, so it is testable without a client.
    """
    statements_by_entity: Dict[str, List[str]] = {}
    for statement in statements or []:
        if not isinstance(statement, dict):
            continue
        entity = str(statement.get("entity") or "").strip()
        text = " ".join(str(statement.get("text") or "").split())
        if entity and text:
            statements_by_entity.setdefault(entity.casefold(), []).append(text)

    lines = [" ".join(line.split()) for line in str(body or "").splitlines()]
    contexts: Dict[str, Dict[str, List[str]]] = {}
    for raw_name in entity_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        seen: set[str] = set()
        conversation: List[str] = []
        for line in lines:
            if name not in line or line in seen:
                continue
            seen.add(line)
            conversation.append(line[:_HOOK_CONTEXT_LINE_CHARS])
            if len(conversation) >= _HOOK_CONTEXT_LINES:
                break
        contexts[name] = {
            "conversation": conversation,
            "statements": list(statements_by_entity.get(name.casefold(), []))[
                :_HOOK_CONTEXT_STATEMENTS
            ],
        }
    return contexts


def build_hook_prompt(
    body: str,
    contexts: Dict[str, Dict[str, List[str]]],
) -> str:
    """Render the hook prompt with one evidence block per entity (pure).

    The evidence block is what keeps the model from answering about the *word* rather
    than the entity: "年糕" appears next to "一听到开罐头的声音就跑到厨房", so the sense
    is a cat and food associations are visibly off-sense.
    """
    blocks: List[str] = []
    for name, context in contexts.items():
        block = [f"- {name}"]
        conversation = context.get("conversation") or []
        if conversation:
            block.append("  CONVERSATION CLUES:")
            block.extend(f"    · {line}" for line in conversation)
        else:
            block.append(
                "  CONVERSATION CLUES: (none — do not infer a sense from the name alone)"
            )
        statements = context.get("statements") or []
        if statements:
            block.append("  STATEMENTS ALREADY ANCHORED TO IT:")
            block.extend(f"    · {item}" for item in statements)
        blocks.append("\n".join(block))
    return "\n".join(
        [
            _HOOK_SYSTEM,
            _HOOK_RULES.format(
                entities="\n".join(blocks) if blocks else "(none)",
                conversation=str(body or "")[:12000],
            ),
        ]
    )


def _parsed_hooks(
    response: Any,
    names: List[str],
) -> List[Dict[str, str]]:
    """Validate a hook response into ``[{"entity": ..., "hook": ..., "sense": ...}]``.

    Applies the same "do not trust the model" rule the statement path does: an unknown
    entity, an empty hook, a self-referential hook (the entity's own name), a hook that
    merely restates the entity's sense, and a duplicate are all dropped here rather than
    published.

    ``sense`` is the conversation-grounded identity the model committed to before it
    predicted anything. A missing sense means the model had no situational evidence and
    fell back to the surface meaning of the name — the 年糕-as-food failure — so that
    entity's hooks are dropped here even if the prompt was ignored.
    """
    from .topology import MAX_PREDICATE_CHARS

    if isinstance(response, EntityHooks):
        profiles = response.entities
    elif isinstance(response, dict):
        try:
            profiles = EntityHooks(**response).entities
        except Exception:  # noqa: BLE001 - a malformed hook payload is never fatal
            return []
    else:
        return []

    by_name = {name.casefold(): name for name in names}
    seen: set[str] = set()
    hooks: List[Dict[str, str]] = []
    for profile in profiles:
        entity = by_name.get(str(getattr(profile, "entity", "") or "").strip().casefold())
        if not entity:
            continue
        sense = " ".join(str(getattr(profile, "sense", "") or "").split())
        if not sense or sense.casefold() == entity.casefold():
            # No identity grounded in this conversation: predicting from the name alone
            # is the failure this gate exists to stop, not a weaker answer.
            continue
        for raw_hook in getattr(profile, "hooks", []) or []:
            hook = " ".join(str(raw_hook or "").split())
            if not hook or len(hook) > MAX_PREDICATE_CHARS:
                continue
            if hook.casefold() == entity.casefold():
                # The entity's own name is not an association with itself.
                continue
            if hook.casefold() in sense.casefold():
                # "猫" restates the sense "主人养的猫"; a hook must add an association.
                continue
            key = f"{entity.casefold()}\x1f{hook.casefold()}"
            if key in seen:
                continue
            seen.add(key)
            hooks.append({"entity": entity, "hook": hook, "sense": sense})
    return hooks


async def predict_entity_hooks(
    llm_client: Any,
    body: str,
    entity_names: List[str],
    *,
    statements: Optional[List[Dict[str, str]]] = None,
    group_id: str = "",
) -> List[Dict[str, str]]:
    """One batched call: hooks predicted for each entity's identity *in this conversation*.

    ``statements`` are the already-validated statements about the entities; together with
    the conversation lines that mention each name they form the per-entity evidence the
    model must use to establish ``sense`` before predicting hooks. Returns
    ``[{"entity", "hook", "sense"}]``. A model/parse failure returns ``[]``: hooks are an
    *additive* ranking aid, and losing them must never fail a publication the way losing
    the knowledge itself would.
    """
    names = [str(name).strip() for name in entity_names if str(name or "").strip()]
    if not names or not str(body or "").strip() or llm_client is None:
        return []
    contexts = hook_entity_contexts(body, names, statements)
    prompt = build_hook_prompt(body, contexts)
    try:
        if hasattr(llm_client, "generate_response"):
            from graphiti_core.prompts.models import Message

            response = await llm_client.generate_response(
                [Message(role="user", content=prompt)],
                response_model=EntityHooks,
                group_id=group_id or None,
                prompt_name="memos.predict_entity_hooks",
            )
        elif callable(llm_client):
            response = await llm_client(prompt)
        else:
            return []
    except Exception:  # noqa: BLE001 - hooks are additive, never fatal
        return []
    hooks = _parsed_hooks(response, names)
    # 连一句点名对话、一条已锚定陈述都没有的实体，模型只能按名字猜 identity；
    # 即使它硬给了一个 sense，也不发布这个实体的 hook。这是提示词之外的最后一道闸门。
    return [
        hook
        for hook in hooks
        if (contexts.get(hook["entity"]) or {}).get("conversation")
        or (contexts.get(hook["entity"]) or {}).get("statements")
    ]


async def extract_entity_statements(
    llm_client: Any,
    body: str,
    entity_names: List[str],
    *,
    group_id: str = "",
    custom_extraction_instructions: Optional[str] = None,
    assistant_label: str = "",
    perspective_out: Optional[Dict[str, Any]] = None,
    extra_instruction: str = "",
) -> List[Dict[str, str]]:
    """One batched call: the durable statements this conversation makes about entities.

    Returns ``[{"entity": ..., "text": ...}]``, validated here rather than trusted:
    an unknown entity name and an inadmissible statement are both dropped (the caller
    records what was dropped). A model/parse failure returns ``[]`` — the knowledge
    pass must still publish the entities it already has.
    """
    from .knowledge import is_knowledge_statement

    names = [str(name).strip() for name in entity_names if str(name or "").strip()]
    if not names or not str(body or "").strip() or llm_client is None:
        return []
    prompt_parts = [
        _ENTITY_STATEMENT_SYSTEM,
        (custom_extraction_instructions or "").strip(),
        _ENTITY_STATEMENT_RULES.format(
            entities="\n".join(f"- {name}" for name in names),
            conversation=str(body)[:12000],
        ),
    ]
    assistant = str(assistant_label or "").strip()
    if assistant:
        prompt_parts.append(_AI_PERSPECTIVE_RULES.format(assistant=assistant))
    # 追加指令必须留在最末：前面整段与第一次调用逐字相同，缓存才能命中。
    if str(extra_instruction or "").strip():
        prompt_parts.append(str(extra_instruction))
    prompt = "\n".join(prompt_parts)
    if hasattr(llm_client, "generate_response"):
        from graphiti_core.prompts.models import Message

        try:
            response = await llm_client.generate_response(
                [Message(role="user", content=prompt)],
                response_model=EntityStatements,
                group_id=group_id or None,
                prompt_name="memos.extract_entity_statements",
            )
        except Exception:  # noqa: BLE001 - statements are additive, never fatal
            return []
    elif callable(llm_client):
        response = await llm_client(prompt)
    else:
        return []

    if isinstance(response, EntityStatements):
        parsed = response
    elif isinstance(response, dict):
        parsed = EntityStatements(**response)
    else:
        return []

    by_name = {name.casefold(): name for name in names}
    seen: set[str] = set()
    statements: List[Dict[str, str]] = []
    for item in parsed.statements:
        entity = by_name.get(str(item.entity or "").strip().casefold())
        text = " ".join(str(item.text or "").split())
        if not entity or not text:
            continue
        if not is_knowledge_statement(text):
            continue
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        statements.append({"entity": entity, "text": text})

    if perspective_out is not None:
        self_statements: List[str] = []
        opinions: List[Dict[str, str]] = []
        self_seen: set[str] = set()
        for item in parsed.self_statements:
            text = " ".join(str(item.text or "").split())
            if not text or not is_knowledge_statement(text):
                continue
            if assistant and text.startswith("我"):
                text = assistant + text[1:]
            if text.casefold() in self_seen:
                continue
            self_seen.add(text.casefold())
            self_statements.append(text)
        for item in parsed.opinions:
            target = by_name.get(str(item.target or "").strip().casefold())
            text = " ".join(str(item.text or "").split())
            if not target or not text or target == assistant:
                continue
            if not is_knowledge_statement(text):
                continue
            key = (target.casefold(), text.casefold())
            if key in {(row["target"].casefold(), row["text"].casefold()) for row in opinions}:
                continue
            opinions.append({"target": target, "text": text})
        perspective_out["self_statements"] = self_statements
        perspective_out["opinions"] = opinions
    return statements


async def _never_summarize(_node: Any) -> bool:
    """Tell graphiti's attribute pass to generate **no** entity summary.

    ``extract_attributes_from_nodes`` fills ``EntityNode.summary`` with the node's
    incident edge facts (no LLM call), or — once that glue exceeds
    ``MAX_SUMMARY_CHARS * 2`` — with an LLM retelling of the episode. Neither is a
    description of the entity: the first is the relation's text copied onto both
    endpoints, the second is "what happened in this conversation" ("先发送"你好"两次，
    随后称…"). Both used to be published as the entity's description.

    Returning False for every node makes graphiti skip the node entirely (it neither
    appends facts nor calls the summary model), so ``summary`` stays empty and the
    published description comes only from statements this pipeline vouches for.
    """
    return False


async def run_graphiti_extraction(
    client: Any,
    episode_uuid: str,
    body: str,
    role_uuid: str,
    conversation_id: str,
    progress: Optional[Callable[..., None]] = None,
    custom_extraction_instructions: Optional[str] = None,
    assistant_label: str = "",
) -> Dict[str, Any]:
    """Run graphiti's low-level extract/resolve/save pipeline.

    Returns a dict with the ``episode``, ``nodes`` and ``edges`` so the caller can
    map them into the MemOS-compatible ``extracted`` payload consumed by the
    publication saga. Extraction produces exactly two node kinds — knowledge nodes
    (here) and memory nodes (the diary pass) — so this pipeline extracts no memory
    node; the ``events`` key stays in the payload as an empty array.

    ``client`` is a graphiti_core ``Graphiti`` instance (primary or backup
    extraction key). All driver/LLM/embedder I/O runs on that client's single
    event loop (see ``AsyncLoopRunner``).
    """

    now = datetime.now(timezone.utc)

    episode = EpisodicNode(
        uuid=episode_uuid,
        name=f"conversation:{conversation_id}",
        group_id=role_uuid,
        labels=[],
        source=EpisodeType.message,
        content=body,
        source_description="MemOS conversation",
        created_at=now,
        valid_at=now,
    )

    if progress:
        progress("extract_nodes", "调用 Graphiti 抽取实体、关系和属性", 0)

    opts = extraction_options()
    schema_limit = int(opts.get("schema_limit") or RELEVANT_SCHEMA_LIMIT)
    _apply_dedup_gate(opts)
    previous_episodes = await client.retrieve_episodes(
        now,
        last_n=schema_limit,
        group_ids=[role_uuid],
        source=EpisodeType.message,
    )

    extracted_nodes, node_episode_index_map = await extract_nodes(
        client.clients,
        episode,
        previous_episodes,
        None,
        None,
        custom_extraction_instructions,
    )

    if progress:
        progress("resolve_nodes", "实体去重与关系解析", 1)

    resolved_nodes, uuid_map, _ = await resolve_extracted_nodes(
        client.clients,
        extracted_nodes,
        episode,
        previous_episodes,
        None,
    )

    if progress:
        progress("extract_edges", "提取关系", 2)

    edge_type_map = {("Entity", "Entity"): []}
    resolved_edges, invalidated_edges, new_edges = await client._extract_and_resolve_edges(
        episode,
        extracted_nodes,
        previous_episodes,
        edge_type_map,
        role_uuid,
        None,  # edge_types
        resolved_nodes,
        uuid_map,
        custom_extraction_instructions,
    )

    if progress:
        progress("resolve_edges", "解析关系", 3)

    entity_edges = resolved_edges + invalidated_edges

    hydrated_nodes = await extract_attributes_from_nodes(
        client.clients,
        resolved_nodes,
        episode,
        previous_episodes,
        None,
        should_summarize_node=_never_summarize,
        edges=new_edges,
    )

    episodic_edges = build_episodic_edges(
        hydrated_nodes, [episode.uuid], now, node_episode_index_map
    )
    episode.entity_edges = [edge.uuid for edge in entity_edges]
    await add_nodes_and_edges_bulk(
        client.driver,
        [episode],
        episodic_edges,
        hydrated_nodes,
        entity_edges,
        client.embedder,
    )

    if progress:
        progress("extract_statements", "抽取关于实体的陈述", 4)

    # 实体陈述：graphiti 的 relations 明确排除属性/偏好/身份，而实体本身在它的 schema 里
    # 没有 description 字段，于是这类内容必须由这一步补上（见 _ENTITY_STATEMENT_RULES）。
    perspective: Dict[str, Any] = {}
    statements = await extract_entity_statements(
        getattr(getattr(client, "clients", None), "llm_client", None),
        body,
        [str(node.name) for node in hydrated_nodes],
        group_id=role_uuid,
        custom_extraction_instructions=custom_extraction_instructions,
        assistant_label=assistant_label,
        perspective_out=perspective,
    )

    if _option_enabled("second_pass"):
        # 二次抽取：同一段对话、同一份实体清单、提示词前缀逐字不变，只在末尾追加
        # _SECOND_PASS_SUFFIX。平台按前缀缓存，第二次调用的输入约有 95% 走缓存价，
        # 换回来的是「总结时容易被丢掉的具体细节」——时间、地点、数字、计划。
        if progress:
            progress("second_pass", "补抽第一遍漏掉的具体细节", 4)
        extra_perspective: Dict[str, Any] = {}
        extra_statements = await extract_entity_statements(
            getattr(getattr(client, "clients", None), "llm_client", None),
            body,
            [str(node.name) for node in hydrated_nodes],
            group_id=role_uuid,
            custom_extraction_instructions=custom_extraction_instructions,
            assistant_label=assistant_label,
            perspective_out=extra_perspective,
            extra_instruction=_SECOND_PASS_SUFFIX,
        )
        statements = _merge_statements(statements, extra_statements)
        _merge_perspective(perspective, extra_perspective)

    if progress:
        progress("predict_hooks", "预测实体的联想节点", 5)

    # 联想节点（hook）：与上一步**取向相反**的一步——陈述只能说对话说过的，hook 只能来自
    # 模型对这类东西的一般认识。它不产生事实，只产生"可能相关"的交会点，供召回排序加成。
    # 但"这类东西"由**本次对话**界定，而不是名字的字面意思：抽取决断先按对话线索确定
    # 实体在这里是什么（sense），再只为这个身份预测 hook（见 _HOOK_RULES）。上一步已经
    # 产出的陈述正好是"它是什么"的现成证据，所以一并交给 hook 这一遍。
    # 放在同一个循环里串行执行：这两步的输入、实体清单完全相同，合起来仍只是每个会话
    # 一次批量调用，不会为每个实体各打一次模型。
    hooks = await predict_entity_hooks(
        getattr(getattr(client, "clients", None), "llm_client", None),
        body,
        [str(node.name) for node in hydrated_nodes],
        statements=statements,
        group_id=role_uuid,
    )

    return {
        "episode": episode,
        "nodes": hydrated_nodes,
        "edges": entity_edges,
        # Graphiti decided these older edges are contradicted by this episode. They
        # stay in ``edges`` for audit/history, but must not be republished as active
        # project facts; the publication layer uses this list to invalidate the
        # matching fact nodes instead.
        "invalidated_edges": [
            {
                "edge_uuid": str(edge.uuid),
                "fact": str(edge.fact),
                "source_id": str(edge.source_node_uuid),
                "target_id": str(edge.target_node_uuid),
            }
            for edge in invalidated_edges
        ],
        # The knowledge pass extracts no memory node: memory nodes come from the
        # diary pass, so this pipeline's event slot is always empty.
        "events": [],
        "statements": statements,
        # AI-spoken self statements and opinions. They are *not* extra statements
        # about an entity: the entity statement channel would attach "温兆伦不火" to
        # 温兆伦 and lose that the AI said it. The publication layer stores these as
        # fact_kind=self_statement / opinion.
        "ai_perspective": perspective,
        "assistant_label": assistant_label,
        # Model-predicted associations. Each carries ``basis="model_predicted"`` by the
        # time it is published, so nothing downstream can mistake it for knowledge.
        "hooks": hooks,
    }


def _claim(
    text: str,
    *,
    anchor_kind: str,
    anchor_id: str,
    anchor_name: str = "",
    order: int = 0,
    source_id: str = "",
    source_kind: str = "",
) -> Dict[str, Any]:
    """One extracted sentence, the object it is anchored to, and where it came from.

    ``anchor_id`` is where the sentence will be stored; ``source_id``/``source_kind``
    are what the extractor handed it in on (a node, or the edge whose fact it was).
    They differ exactly when a fact was re-anchored onto the entity it is about — in
    which case the audit trail should name the edge, and the publication saga still
    needs to know that edge (an edge whose endpoints collapse onto one identity
    carries no publishable statement, wherever the sentence was anchored).
    """
    return {
        "text": str(text).strip(),
        "normalized": normalized_text(text),
        "anchor_kind": anchor_kind,
        "anchor_id": str(anchor_id or ""),
        "anchor_name": str(anchor_name or ""),
        "source_id": str(source_id or anchor_id or ""),
        "source_kind": str(source_kind or anchor_kind or ""),
        "order": int(order),
    }


def _name_position(claim: Dict[str, Any]) -> Optional[int]:
    """Where the anchor's own name appears in the sentence (``None`` = not named).

    This is the "aboutness" signal the rule leans on: the sentence
    "小E 是被塞进电脑里的 AI 宠物，由主人的电脑养着" may be handed in as the
    description of *both* 小E and 主人 and as the fact of the 主人—小E edge, but only
    小E is its subject, and a subject is named at the front of its own sentence.
    """
    name = normalized_text(claim.get("anchor_name"))
    if not name:
        return None
    position = str(claim.get("normalized") or "").find(name)
    return position if position >= 0 else None


def choose_memory_anchors(
    claims: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collapse claims into one memory per distinct sentence, choosing its anchor.

    The rule, applied per distinct normalized sentence:

    1. prefer a **node** claim whose node name appears in the sentence, the
       earliest-mentioned name winning — the sentence is that entity's own
       description;
    2. otherwise prefer a **relation** claim — the sentence asserts a relation
       rather than describing a named entity;
    3. otherwise keep the first claim in extraction order.

    Both inputs and outputs are plain dicts so the rule can be exercised without a
    graphiti client. Returns ``(memories, dropped)``: the winners in memory-payload
    shape (``node_refs``/``relation_refs`` populated, ``content`` set) and the
    losing claims, recorded as diagnostics instead of being silently discarded.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for claim in claims:
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        grouped.setdefault(str(claim.get("normalized") or normalized_text(text)), []).append(claim)

    winners: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for _normalized, group in grouped.items():
        named_nodes = [
            (position, claim)
            for claim in group
            if claim.get("anchor_kind") == "node"
            for position in [_name_position(claim)]
            if position is not None
        ]
        if named_nodes:
            winner = min(named_nodes, key=lambda item: (item[0], item[1]["order"]))[1]
        else:
            relations = [claim for claim in group if claim.get("anchor_kind") == "relation"]
            winner = min(relations, key=lambda claim: claim["order"]) if relations else min(
                group, key=lambda claim: claim["order"]
            )
        winners.append(winner)
        for claim in group:
            if claim is winner:
                continue
            dropped.append(
                {
                    "content": claim["text"],
                    "anchor_kind": claim.get("anchor_kind"),
                    # The object the losing copy came in on, so an operator can tell
                    # which source duplicated the sentence.
                    "anchor_id": claim.get("source_id") or claim.get("anchor_id"),
                    "reason": "same_sentence_already_anchored",
                    "anchored_to": winner.get("anchor_id"),
                }
            )

    winners.sort(key=lambda claim: claim["order"])
    dropped.sort(key=lambda item: (str(item.get("content") or ""), str(item.get("anchor_id") or "")))
    memories: List[Dict[str, Any]] = []
    for claim in winners:
        is_node = claim.get("anchor_kind") == "node"
        memory = {
            "content": claim["text"],
            "node_refs": [claim["anchor_id"]] if is_node else [],
            "relation_refs": [] if is_node else [claim["anchor_id"]],
            "anchor_kind": claim.get("anchor_kind"),
            # Where the sentence came in on: "node" means the extractor stated it as
            # that entity's description (identity); "relation" means it was an edge
            # fact that has been re-anchored onto the entity it is about (a fact).
            "source_kind": claim.get("source_kind"),
        }
        if str(claim.get("source_kind") or "") == "relation":
            # Provenance only: the saga uses it to keep a statement of a collapsed
            # identity edge (an alias statement, whose endpoints are the same node)
            # out of the graph even when the sentence was re-anchored onto a node.
            memory["source_relation_refs"] = [str(claim.get("source_id") or "")]
        memories.append(memory)
    return memories, dropped


def statement_anchor(text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The object a statement is knowledge about (pure).

    ``candidates`` is an ordered list of ``{"kind", "id", "name"}`` whose **first**
    entry is the fallback — in practice the relation the fact came from, with an
    empty ``name`` so it can never win by name. The winner is the entity whose name
    appears **earliest** in the sentence: the subject of a sentence is named at its
    front, so "小E 是被塞进电脑里的 AI 宠物，由主人的电脑养着" is knowledge about
    小E even though 主人 is mentioned later in it.

    Why entities win over the relation they came from: in this topology a shared edge
    is semantic-free (kind plus endpoints only), and CONTEXT says the meaning of an
    edge belongs to the roles' private understanding of the objects involved. A fact
    that names an entity is therefore that entity's knowledge; the relation keeps the
    facts that name no entity at all ("两人互相调侃了几句。").
    """
    if not candidates:
        raise ValueError("a statement needs at least one candidate anchor")
    normalized = normalized_text(text)
    best: Optional[tuple[tuple[int, int], Dict[str, Any]]] = None
    for index, candidate in enumerate(candidates):
        name = normalized_text(candidate.get("name"))
        if not name:
            continue
        position = normalized.find(name)
        if position < 0:
            continue
        key = (position, index)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1] if best is not None else candidates[0]


def build_extracted_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Map graphiti nodes/edges into the ``extracted`` payload.

    The publication saga upserts nodes/relations into the shared topology and
    stores one memory (private cognition) per node. ``events`` is part of the
    payload shape and is always empty here: extraction produces exactly two node
    kinds, and this pass only produces **knowledge nodes** — memory nodes
    (``kind="event"``) come from the diary pass.

    Three rules are enforced here rather than trusted to the model (ADR-0150):

    * a node must be a **knowledge entity** — a figure of speech such as the
      target of "脸红的像苹果" is dropped, together with every relation that
      depends on it;
    * a **statement** must be about the entity and must still hold outside this
      conversation (``knowledge.is_knowledge_statement``). The description is taken
      from the extractor's own statements, never from graphiti's ``summary``: that
      field is the node's incident edge facts glued together (or, past the length
      cap, an LLM retelling of the episode), which is how "先发送"你好"两次，随后称…"
      became an entity's description. No description is a legal outcome, so the
      ``or name`` fallback is gone too — a node whose only content was its own name
      has nothing to say and must not become a recall candidate;
    * **one sentence is published once** (see :func:`choose_memory_anchors`), and it
      is anchored to the entity it is about (see :func:`statement_anchor`), so the
      same sentence no longer appears on both endpoints of an edge and on the edge
      itself.
    """
    from .knowledge import (
        filter_knowledge_nodes,
        filter_relations,
        filter_statements,
        statement_rejection,
    )

    nodes = list(result.get("nodes", []))
    edges = list(result.get("edges", []))
    raw_text = str(result.get("raw_text") or "")
    invalidated_edges = [
        item for item in (result.get("invalidated_edges") or []) if isinstance(item, dict)
    ]

    def node_attr(node: Any, *names: str) -> str:
        """Read a field from either a graphiti node object or a plain mapping."""
        for name in names:
            if isinstance(node, dict):
                value = node.get(name)
            else:
                value = getattr(node, name, None)
            if value:
                return str(value)
        return ""

    # An invalidated edge is history, not a new claim. Dropping it here keeps the
    # old sentence from being republished as an active fact; the publication saga
    # invalidates the matching fact node using ``invalidated_edges`` below.
    invalidated_ids = {str(item.get("edge_uuid") or "") for item in invalidated_edges}
    edges = [
        edge for edge in edges
        if str(node_attr(edge, "uuid")) not in invalidated_ids
    ]

    node_payloads = []
    dropped_statements: List[Dict[str, Any]] = []
    for node in nodes:
        node_id = node_attr(node, "uuid", "id")
        name = node_attr(node, "name")
        # NOT "summary": that field is graphiti's incident-edge-fact glue (see the
        # docstring). Only a real description field counts, and it is filtered.
        filtered = filter_statements(node_attr(node, "description"))
        for item in filtered["dropped"]:
            dropped_statements.append({"anchor": "node", "anchor_id": node_id, **item})
        node_payloads.append(
            {
                "id": node_id,
                "name": name,
                "description": filtered["text"],
            }
        )
    kept_nodes, dropped_nodes = filter_knowledge_nodes(node_payloads, raw_text)
    kept_ids = {item["id"] for item in kept_nodes}

    # A relation whose endpoint is not a knowledge entity is a relation built on a
    # figure of speech; it never reaches the topology.
    def edge_endpoints(edge: Any) -> tuple[str, str]:
        return (
            node_attr(edge, "source_node_uuid", "source_id"),
            node_attr(edge, "target_node_uuid", "target_id"),
        )

    usable_edges = []
    dropped_relations = []
    for edge in edges:
        source_id, target_id = edge_endpoints(edge)
        if source_id in kept_ids and target_id in kept_ids:
            usable_edges.append(edge)
        else:
            dropped_relations.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "reason": "endpoint_not_a_knowledge_entity",
                }
            )

    # One text -> one anchor, and that anchor is the entity the sentence is about.
    # An edge fact is the ordinary source of knowledge in this pipeline (graphiti
    # extracts names and edge facts; there is no entity-attribute channel), so a fact
    # that names an entity becomes that entity's knowledge and only a fact naming no
    # entity stays on the relation. Nodes come first in the claim order, which is
    # also the tie-break order.
    entity_choices = [
        {"kind": "node", "id": str(node["id"]), "name": str(node.get("name") or "")}
        for node in kept_nodes
    ]
    claims: List[Dict[str, Any]] = []
    order = 0
    # Statements about an entity become **facts of that entity** (节点细化): the
    # attribute/preference/identity channel graphiti's relation extractor deliberately
    # skips, anchored to the entity it is about.
    by_name = {normalized_text(node.get("name")): node for node in kept_nodes}
    for statement in result.get("statements") or []:
        if not isinstance(statement, dict):
            continue
        text = " ".join(str(statement.get("text") or "").split())
        node = by_name.get(normalized_text(statement.get("entity")))
        if not text:
            continue
        if node is None:
            dropped_statements.append(
                {"anchor": "statement", "anchor_id": str(statement.get("entity") or ""), "text": text,
                 "reason": "unknown_entity"}
            )
            continue
        reason = statement_rejection(text)
        if reason is not None:
            dropped_statements.append(
                {"anchor": "statement", "anchor_id": str(node["id"]), "text": text, "reason": reason}
            )
            continue
        anchor = statement_anchor(
            text,
            [{"kind": "node", "id": str(node["id"]), "name": str(node.get("name") or "")}] + entity_choices,
        )
        claims.append(
            _claim(
                text,
                anchor_kind="node",
                anchor_id=str(anchor["id"]),
                anchor_name=str(anchor.get("name") or ""),
                order=order,
                source_id=str(node["id"]),
                # NOT "node": a statement is not the entity's identity, it is one of its
                # knowledge items, so it is published as its own fact node.
                source_kind="statement",
            )
        )
        order += 1
    for node in kept_nodes:
        description = str(node.get("description") or "").strip()
        if not description:
            continue
        anchor = statement_anchor(
            description,
            [{"kind": "node", "id": str(node["id"]), "name": str(node.get("name") or "")}],
        )
        claims.append(
            _claim(
                description,
                anchor_kind=str(anchor["kind"]),
                anchor_id=str(anchor["id"]),
                anchor_name=str(anchor.get("name") or ""),
                order=order,
                source_id=str(node["id"]),
            )
        )
        order += 1
    for edge in usable_edges:
        content = node_attr(edge, "fact", "name")
        if not content:
            continue
        reason = statement_rejection(content)
        if reason is not None:
            dropped_statements.append(
                {"anchor": "relation", "anchor_id": node_attr(edge, "uuid"), "text": content, "reason": reason}
            )
            continue
        # The relation is the fallback (empty name => it can never win by name).
        anchor = statement_anchor(
            content,
            [{"kind": "relation", "id": node_attr(edge, "uuid"), "name": ""}] + entity_choices,
        )
        claims.append(
            _claim(
                content,
                anchor_kind=str(anchor["kind"]),
                anchor_id=str(anchor["id"]),
                anchor_name=str(anchor.get("name") or ""),
                order=order,
                source_id=node_attr(edge, "uuid"),
                source_kind="relation",
            )
        )
        order += 1
    memories, duplicate_memories = choose_memory_anchors(claims)
    for memory in memories:
        # Only a statement the extractor gave as the entity's own **description**
        # becomes the entity's identity (one element, always injected with the
        # entity). A statement that merely names an entity — which is where every
        # edge fact lands — is a **fact of that entity**: its own node, its own
        # vector, individually selectable (节点细化). Publishing it as a description
        # instead is exactly how one sentence used to become the entity's whole
        # content and lose per-item gating.
        source_kind = str(memory.pop("source_kind", "") or "")
        memory["node_description"] = (
            str(memory.get("anchor_kind")) == "node" and source_kind == "node"
        )
    relations: List[Dict[str, Any]] = []
    for edge in usable_edges:
        source_id, target_id = edge_endpoints(edge)
        relations.append({
            "source_id": source_id,
            "target_id": target_id,
            "name": node_attr(edge, "name"),
            # 保留 graphiti 边的原始 UUID，供发布 saga 把记忆里的 relation_refs
            # （edge.uuid）重映射到共享拓扑的 relation_id。
            "edge_uuid": node_attr(edge, "uuid"),
        })
    # 关系标签 → 谓词。这是"相关节点天然连通"的前置条件：只有关系有名字，发布层才能
    # 把它变成一个**共享的交会节点**，让"年糕"和"开罐器"通过"开罐头的声音"真的连上。
    # 开放谓词（"相关"）在这里被降级为 None（仍然 publish 成普通无名边，只是不建节点），
    # 因为拿"相关"建交会点会把所有东西连到所有东西。
    relations, rejected_predicates = filter_relations(relations)
    node_id_by_name = {
        normalized_text(node.get("name")): str(node["id"]) for node in kept_nodes
    }
    hooks: List[Dict[str, Any]] = []
    dropped_hooks: List[Dict[str, Any]] = []
    for item in result.get("hooks") or []:
        if not isinstance(item, dict):
            continue
        entity = normalized_text(item.get("entity"))
        hook = " ".join(str(item.get("hook") or "").split())
        parent_id = node_id_by_name.get(entity)
        if not hook:
            continue
        if parent_id is None:
            # The model predicted an association for an entity that did not survive the
            # knowledge filter (a simile target, say). Nothing to hang it on.
            dropped_hooks.append(
                {"entity": str(item.get("entity") or ""), "hook": hook,
                 "reason": "unknown_entity"}
            )
            continue
        hooks.append({"parent_id": parent_id, "text": hook})
    perspective: List[Dict[str, Any]] = []
    assistant_label = str(result.get("assistant_label") or "").strip()
    raw_perspective = result.get("ai_perspective") or {}
    if assistant_label and isinstance(raw_perspective, dict):
        for text in raw_perspective.get("self_statements") or []:
            statement = " ".join(str(text or "").split())
            if statement:
                perspective.append(
                    {
                        "kind": "self_statement",
                        "text": statement,
                        "speaker_name": assistant_label,
                    }
                )
        for row in raw_perspective.get("opinions") or []:
            if not isinstance(row, dict):
                continue
            target = " ".join(str(row.get("target") or "").split())
            statement = " ".join(str(row.get("text") or "").split())
            if target and statement:
                perspective.append(
                    {
                        "kind": "opinion",
                        "text": statement,
                        "speaker_name": assistant_label,
                        "target_name": target,
                    }
                )
    return {
        "nodes": [
            {"id": str(node["id"]), "name": node["name"], "description": node.get("description", "")}
            for node in kept_nodes
        ],
        "relations": relations,
        "memories": memories,
        # Older facts contradicted by this episode. The saga invalidates matching
        # active fact nodes (exact text, same role) instead of deleting them, so the
        # old assertion stays readable as history.
        "invalidated_edges": invalidated_edges,
        # AI-spoken self statements / opinions. Names are resolved against the
        # published node ids (and fixed identities) by the publication saga.
        "perspective": perspective,
        # 联想节点：模型对一个实体**可能**的关联所做的预测，已重映射到保留下来的实体 id。
        # 与 ``memories`` 分开一条通道，是为了让发布层能对它们施加完全不同的规则
        # （basis 标记、向量化、绝不渲染成知识），而不是混进事实里靠约定区分。
        "hooks": hooks,
        "dropped_hooks": dropped_hooks,
        # Kept in the payload shape for the saga/adapters, always empty: this pass
        # produces knowledge nodes only.
        "events": [],
        "dropped_non_knowledge_nodes": dropped_nodes,
        "dropped_non_knowledge_relations": dropped_relations,
        # Recorded, not silent: a regression that starts publishing the same
        # sentence on several objects again, or that lets episode narration back in,
        # is visible on the manifest.
        "duplicate_memories": duplicate_memories,
        "dropped_non_knowledge_statements": dropped_statements,
        # Recorded, not silent: a relation whose label was too open to become a shared
        # predicate node is visible on the manifest instead of vanishing into an
        # anonymous edge.
        "dropped_open_predicates": rejected_predicates,
    }

