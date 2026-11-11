"""Shared extraction protocol for the **knowledge** pass.

The knowledge pass extracts exactly two classes and nothing else:

- ``entities``: a node name must be something the dialogue talks about as a real
  object and can describe; what the dialogue states about it is its description;
- ``relations``: an edge that the dialogue explicitly states or unambiguously
  implies.

**Memory is a different pass.** A memory node comes from the AI's own diary entry
(see :mod:`graphiti_memory.diary`) and is extracted by
:mod:`graphiti_memory.memory_extraction`; it is deliberately subjective and may
connect to the knowledge entities this pass produces. The old "events" array of
this prompt — a concrete happened-in-the-conversation event — is retired: it
produced a second, competing notion of "an event" next to the memory node.

The restriction wording is validated by the parser as well: the LLM is not
trusted to obey the prompt (plan 阶段 3: "提示词必须同时提供中文和英文示例，
并在解析层再次校验事件字段，不能只依赖 LLM 遵守提示词").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# ``source_kind`` is a closed set: a memory either happened inside the
# conversation, was reported by the user as history, or comes from the source
# document a role was created out of (document import). Nothing else is allowed.
# The third member exists because 文档创建角色 publishes its own memory batch per
# source block: those memories did not happen in any conversation this service
# witnessed, and calling them ``conversation_happened`` would be a false claim on
# the memory node's provenance.
EVENT_SOURCE_KINDS = ("conversation_happened", "user_reported", "source_document")

MAX_EVENT_SUMMARY_CHARS = 600
MAX_EVENT_SOURCE_IDS = 32


class EventValidationError(ValueError):
    """Raised when a memory node candidate cannot be represented as a node."""


# --------------------------------------------------------------------------- prompt

EXTRACTION_RESTRICTIONS = """【抽取限制 / Extraction restrictions】
0. 只有两类可抽取内容：知识类实体与它们之间被明确陈述的关系。除此之外一律不抽取。
   Only two classes may be extracted: knowledge entities, and the relations between
   them that the dialogue explicitly states.
1. 知识类实体（"entities"）：节点名必须是对话中被当作真实对象谈论、并且可以被描述的
   知识实体；该实体在对话中被陈述到的特征就是它的描述。
   正例：「主人养了一只叫大橘的猫，大橘很黏人」→ 实体「大橘」，描述「主人养的猫，很黏人」。
   反例：比喻、类比、明喻的比较对象不是实体，不得抽取。例如「脸红的像苹果」中的
   "苹果"、「温柔得像天使」中的"天使"、「壮得像头牛」中的"牛"、「歌声像夜莺」中的
   "夜莺"都不抽取，也不得据此建立任何关系。
   Knowledge entities: the node name must be a real thing the dialogue talks about and
   can describe, and what the dialogue states about it is its description.
   NEVER extract the target of a metaphor, simile, analogy or comparison, and never
   build a relation from one: in "she is as gentle as an angel" do NOT extract
   "angel"; in "he is strong as an ox" do NOT extract "ox".
2. 关系（"relations"）：只抽取对话明确陈述或明确蕴含的关系；静态属性、偏好、概念、
   身份描述和常识不是关系，不得塞进 "relations"。
   Only relations the dialogue explicitly states or unambiguously implies may appear
   in "relations". Static attributes, preferences, concepts and identity statements
   are not relations. Example: "主人喜欢黑咖啡" is a preference, not a relation
   between two entities; "主人 5 月 1 日买了一只猫" states a relation between 主人
   and the cat and may be extracted.
3. 共现排除：仅因共同提及、文本邻近或出现在同一场景，不得创建任何关系。
   Never create a relation merely because two entities were mentioned together,
   appear near each other in the text, or share a scene.
4. 这条对话里"发生过什么、对 AI 有什么意义"不属于本次抽取：那是 AI 日记与记忆节点
   的职责。不要输出事件、心情或主观感受。
   What happened and what it meant to the AI is NOT this pass's job: that belongs to
   the AI diary and the memory nodes. Do not output events, moods or feelings.
5. 来源标注不是实体：消息开头的「本机消息」（本机主人说的）与「QQ消息」（QQ 联系人
   说的）只是渠道标注，紧跟其后的名字才是发信者。绝不要把「本机消息」「QQ消息」这类
   渠道标注抽成实体、为它们建节点或关系。
   The leading channel labels "本机消息" (said by the local owner) and "QQ消息" (said by
   a QQ contact) are labels, not entities: the name right after them is the speaker.
   NEVER extract a channel label as an entity, and never build a node or a relation
   for one."""


EXTRACTION_JSON_SHAPE = """{"entities": [{"name": "实体名"}],
 "relations": [{"source": "实体A", "relation": "关系", "target": "实体B"}]}

输出必须包含这两个键；没有内容时给出空数组。实体的描述由节点属性承载，不要为了写
描述而额外输出字段。Always emit both keys; use empty arrays when there is nothing."""


def build_extraction_prompt(*, conversation: str, history: str = "") -> str:
    """Full knowledge-extraction prompt shared by the temporary and formal paths."""
    history_block = f"\n\n【历史上下文（仅参考，不必输出）】：\n{history}" if history else ""
    return (
        EXTRACTION_RESTRICTIONS
        + "\n\n【输出格式 / Output format】\n"
        + EXTRACTION_JSON_SHAPE
        + history_block
        + "\n\n【本次对话条目 / Conversation entries】：\n"
        + conversation
        + "\n\n只输出 JSON，不要输出解释。Output JSON only."
    )


# ------------------------------------------------------------------------- parsing


def _norm_name(value: Any) -> str:
    cleaned = str(value or "").strip().strip("。，！？、；：\"'《》「」()（）-—_").strip()
    return cleaned[:64]


def _norm_summary(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_EVENT_SUMMARY_CHARS].strip()


def normalize_source_kind(value: Any) -> str:
    """Map an LLM-provided source_kind onto the closed set, else reject."""
    candidate = str(value or "").strip().lower()
    aliases = {
        "conversation_happened": "conversation_happened",
        "conversation": "conversation_happened",
        "happened": "conversation_happened",
        "in_conversation": "conversation_happened",
        "会话发生": "conversation_happened",
        "对话发生": "conversation_happened",
        "user_reported": "user_reported",
        "reported": "user_reported",
        "history": "user_reported",
        "historical": "user_reported",
        "user_reported_event": "user_reported",
        "用户讲述": "user_reported",
        "历史事件": "user_reported",
    }
    normalized = aliases.get(candidate)
    if normalized is None:
        raise EventValidationError(
            f"event source_kind must be one of {EVENT_SOURCE_KINDS}, got {value!r}"
        )
    return normalized


def normalize_occurred_at(value: Any) -> Optional[str]:
    """Accept only an explicit ISO-8601-ish timestamp; never invent one.

    A bare date, an "unknown"/relative phrase or a free-form sentence is dropped
    so the caller can fall back to the conversation message time.
    """
    candidate = str(value or "").strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered in {"unknown", "none", "null", "n/a", "未知", "不详", "不确定"}:
        return None
    if any(token in candidate for token in ("刚才", "昨天", "前天", "上周", "上个", "最近", "不久", "很多年", "小时候")):
        # Relative phrases are not a locatable precise time.
        return None
    digits = sum(character.isdigit() for character in candidate)
    if digits < 6:
        return None
    if not (candidate[:4].isdigit() and candidate[4] in {"-", "/", "."}):
        return None
    if "T" in candidate or " " in candidate:
        return candidate
    if len(candidate) >= 10:
        return candidate.replace("/", "-").replace(".", "-")
    return None


def normalize_events(raw_events: Any) -> List[Dict[str, Any]]:
    """Validate and normalise the ``events`` array of one extraction payload.

    Returns ``[]`` when the model reported no concrete event (a legal empty
    result). Each returned item carries only the fields the event store accepts.
    """
    if raw_events is None:
        return []
    if isinstance(raw_events, dict):
        raw_events = [raw_events]
    if not isinstance(raw_events, (list, tuple)):
        raise EventValidationError("events must be an array")
    events: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_events:
        if isinstance(item, str):
            item = {"summary": item}
        if not isinstance(item, dict):
            continue
        summary = _norm_summary(item.get("summary") or item.get("content") or item.get("text"))
        if not summary:
            continue
        key = summary.casefold()
        if key in seen:
            continue
        source_ids: List[str] = []
        raw_ids = item.get("source_event_ids") or item.get("source_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if isinstance(raw_ids, (list, tuple)):
            for value in raw_ids:
                text = str(value or "").strip()
                if text and text not in source_ids:
                    source_ids.append(text)
        events.append(
            {
                "summary": summary,
                "source_kind": normalize_source_kind(item.get("source_kind")),
                "occurred_at": normalize_occurred_at(item.get("occurred_at") or item.get("event_time")),
                "source_event_ids": source_ids[:MAX_EVENT_SOURCE_IDS],
            }
        )
        seen.add(key)
    return events


def parse_extraction_payload(content: str) -> Dict[str, Any]:
    """Parse a knowledge-extraction JSON payload leniently; empty is allowed.

    A model that still answers with the retired ``events`` array has it ignored:
    memory is not produced by this pass any more, and accepting the key would let
    an old prompt silently keep writing memory nodes.
    """
    data = _load_json_object(content)
    entities: List[Dict[str, str]] = []
    for item in data.get("entities") or []:
        # Models answer with a bare name, or with the object form
        # {"name": ..., "description": ...}. Reading the object form matters: the
        # previous version stringified the whole dict, so every entity that came
        # with a description was silently dropped (or turned into a junk name).
        if isinstance(item, dict):
            raw_name = item.get("name") or item.get("entity") or item.get("entity_name")
        else:
            raw_name = item
        name = _norm_name(raw_name)
        if name and not any(entry["name"] == name for entry in entities):
            entities.append({"name": name})
    relations: List[Dict[str, str]] = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source, target = _norm_name(item.get("source")), _norm_name(item.get("target"))
        relation = str(item.get("relation") or item.get("name") or "").strip()[:32]
        if source and target and relation and source != target:
            relations.append({"source": source, "relation": relation, "target": target})
    return {"entities": entities, "relations": relations, "events": []}


def _load_json_object(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    import json

    data: Any = None
    try:
        data = json.loads(content)
    except Exception:  # noqa: BLE001 - lenient parse of model output
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start : end + 1])
            except Exception:  # noqa: BLE001
                data = None
    return data if isinstance(data, dict) else {}


def events_for_persistence(
    events: Sequence[Dict[str, Any]],
    *,
    default_occurred_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resolve each event's time, falling back to the conversation message time.

    Plan 不变式 18: when no more precise time can be located, the event uses the
    corresponding conversation message time; a precise-looking time is never
    fabricated.

    当前没有生产调用方：知识通路不再抽取事件，记忆节点由日记通路直接创建（它
    自己决定时间）。保留它是为了给「记忆节点」这一类对象的校验/时间语义留一个
    唯一出处，供显式恢复路径与测试复用，而不是让每个写入点各写一份。
    """
    resolved: List[Dict[str, Any]] = []
    for event in events:
        item = dict(event)
        if not item.get("occurred_at") and default_occurred_at:
            item["occurred_at"] = default_occurred_at
            item["occurred_at_source"] = "conversation_message_time"
        else:
            item.setdefault("occurred_at_source", "extracted" if item.get("occurred_at") else None)
        resolved.append(item)
    return resolved


# ------------------------------------------------------------- LLM extraction


def build_event_extractor(config: Dict[str, Any], *, timeout: float = 60.0):
    """Deprecated: the knowledge pass no longer extracts events.

    Memory now comes from the AI's diary (see :mod:`graphiti_memory.diary` and
    :mod:`graphiti_memory.memory_extraction`), so the knowledge extractor has
    nothing to run here. Kept as an explicit ``None`` so a stale call site fails
    visibly at the "publish no memory" branch instead of silently publishing an
    extraction the protocol no longer defines.
    """
    return None

