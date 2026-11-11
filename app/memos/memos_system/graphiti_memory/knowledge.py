"""Knowledge-entity rules for the extraction pipeline (ADR-0150).

The extraction classes are exactly two:

* **knowledge entities** — a named thing the conversation describes. The node
  name is the entity name, the description is what the conversation stated about
  it, and the description is stored on the node;
* **events** — a concrete thing that happened, kept as their own independent
  memory unit (``events_protocol``).

Anything else is not extractable. Two rules are enforced here, and both are
re-checks of what the prompt already asks for, because the prompt is not trusted:

* a noun that only appears as the *target of a comparison* ("脸红的像苹果" -> 苹果 /
  "温柔得像天使" -> 天使) is a figure of speech, not an entity;
* a **statement** must be about the entity and must still hold outside this
  conversation (``is_knowledge_statement``). "先发送"你好"两次，随后称…" describes
  what happened in this episode; it is the diary's job to remember that, and it must
  not become an entity's description. See ``docs/memory-refactor/DESIGN.md`` §2.1.

The functions here are deliberately pure so both the LLM-payload path
(``extraction_engine``) and the publication path (``saga``) can apply the same
rule without touching a store.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .domain import normalized_text

#: Comparison markers that introduce a figure of speech. Grouped so the rest of
#: the pattern binds to the whole alternation, not just to its last branch.
_COMPARISON_MARKER = (
    r"(?:像|好像|就像|如同|宛如|宛若|仿佛|好似|似的|一般|一样|犹如|恰似|"
    r"好比|堪比|无异于|不亚于|比得上|像极了|活像|俨然)"
)

#: The comparison target: a short run of characters right after the marker.
#: Captured lazily and terminated by a clause particle, a degree word or
#: punctuation, then trimmed of the measure-word prefix. Being lazy matters:
#: "像机关枪一样快" must yield 机关枪, not 机关枪一样快.
_COMPARISON_TARGET = (
    r"[ \t]*(?P<target>[\u4e00-\u9fffA-Za-z0-9]{1,8}?)"
    r"(?=[\s，,。.！!？?；;：:、…\-—]|的|地|得|了|着|过|很|太|真|都|也|还|就"
    r"|而|却|一|样|般|似|是|在|把|被|跟|和|与|$)"
)

_MEASURE_PREFIXES = ("一只", "一头", "一匹", "一条", "一个", "一把", "一朵", "一片", "一座", "一位", "一名", "只", "头", "匹", "条", "个", "把", "朵", "片", "座", "位")
_COMPARISON_SUFFIXES = ("一样", "一般", "似的", "那般", "那样", "这么", "那么")

#: Generic words that are a classification rather than a knowledge entity name.
_GENERIC_NAMES = {
    "东西", "事情", "这个", "那个", "他们", "我们", "你们", "大家", "人们",
    "别人", "某人", "什么", "时候", "地方", "问题", "情况", "等等",
    "之类", "一切", "所有",
}

# ------------------------------------------------------------------ statements
#
# 过程叙述与结构冗余的信号词。判定规则见 ``statement_rejection``：单靠一个词不放行也不
# 拒绝，只有信号组合出现时才认定"这句话讲的是本次对话"而不是"这个实体是什么"。
# 每一组都刻意保持短小：宁可漏掉一句脏话，也不要把"主人要求公开前检查 key"这种
# 长期规则误杀。

#: 说话/对谈动作：出现即说明句子在转述一次交流。
_SPEECH_ACTIONS = (
    "发送", "发出", "回复", "回了", "询问", "追问", "反问", "回答", "答道",
    "说", "称", "表示", "强调", "告诉", "提到", "联系", "私聊", "加了我", "加我",
    "要求", "请求", "确认", "否认", "质疑", "道歉", "解释", "催促", "打招呼",
)

#: 时序词：把陈述绑在本次对话的时间线上。
_SEQUENCE_MARKERS = ("先", "随后", "之后", "然后", "接着", "最后", "又", "再", "一开始", "起初", "后来", "稍后")

#: 会话内指代：脱离本次对话无法解析。
_SESSION_REFERENCES = (
    "本次对话", "这次对话", "这条消息", "那条消息", "上一句", "上面那句",
    "刚才", "本次", "这次", "本轮", "前后消息",
)

#: 时间锚点：把陈述绑在"某一天/刚刚"上的句子通常不是长期知识。
#:
#: 来源：2026-09-26 记忆重构阶段 5 端到端验收第 3 轮漏过一条
#: "小E是被塞进电脑里的AI宠物，**而且刚刚**QQ有人自称是主人" —— 后半句只在那一刻
#: 成立，整句不该入图（见 ``docs/memory-refactor/e2e/RESULTS.md``）。
#:
#: ⚠️ 但当时实现成了"命中任一时间词即拒"，违背了本文件自己的"信号组合"原则，并把
#: **已经发生的事实**一起杀掉：实测 "主人的伞在昨天（2026年10月7日）丢了。" 被整句
#: 丢弃 → 伞节点下一条 fact 都没有 → "伞丢了"这件事永远召不回来（ADR-0171）。
#: 现在按语义分两组，两组都只贡献**信号**：
#:
#: * 当下/未来锚点：句子讲的是"此刻/即将"的状态或安排，它明天就可能不成立，因此单独
#:   成立（"今天主人心情不错"、"明天要下雨"、"目前主人还没有决定"）；
#: * 已发生锚点：句子记的是**过去发生的一件事** —— 那正是记忆该收的东西；只有它同时在
#:   **转述一次交流**（``_SPEECH_ACTIONS``）时，才说明它记的是"本次会话里发生的事"
#:   （"…而且刚刚QQ有人自称是主人"）。
_PRESENT_TIME_ANCHORS = ("今天", "今晚", "明天", "现在", "目前", "眼下", "这次", "上一次")

#: 已发生锚点：见上。它单独不判负 —— 只有配上"转述一次交流"才说明记的是本次会话内的事。
_PAST_TIME_ANCHORS = ("刚刚", "刚才", "昨天", "昨晚", "今早", "前天")

#: 明确的“状态已改变”信号。带“现在/目前”的句子通常是易逝状态，但“现在只喝豆浆”
#: 这类**更正后的新常态**必须留下；只有同时出现变更词时才放行，避免把“现在心情不错”
#: 这类临时状态写入长期知识。
_CHANGE_MARKERS = (
    "改成", "改为", "改到", "改回", "改喝", "改去", "改叫", "换车", "换了",
    "换成", "换回", "搬到", "转到", "调到", "变成", "戒了", "不再", "不过敏",
    "更正", "更新为", "调整为", "取消", "作废", "其实是", "误诊", "现在只",
    "只喝", "只吃", "已经改",
)

#: 两组之和。保留这个名字是因为诊断输出与回退文档按它检索。
_TIME_ANCHORS = _PRESENT_TIME_ANCHORS + _PAST_TIME_ANCHORS

#: 句子自带的绝对日期（"2026年10月7日"、"2026-10-07"）。它出现时，相对时间词只是同一件
#: 事的口语重复，不构成丢弃理由：抽取时常把两者都写出来（"在昨天（2026年10月7日）"）。
#: 这条短路对两组时间词都生效（见 ``_time_anchor_rejection``）。
_ABSOLUTE_DATE = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?|\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}"
)

#: 联系方式 + 号码：QQ 硬加入已经用节点属性与 qq_contact 边结构化记录了同一件事，
#: 再用散文复述一遍是零增量。
_CONTACT_VERBS = ("联系", "私聊", "加我", "加了我", "发消息", "发来消息")
_NUMBER_RUN = re.compile(r"\d{5,}")

#: 单条陈述的字数上限。脏度与长度强相关；合法的实体属性远短于这个值。
MAX_STATEMENT_CHARS = 160

#: 一个实体一次抽取最多留下几条陈述。超出的不再进入图谱（不是截断句子）。
MAX_STATEMENTS_PER_DESCRIPTION = 3

#: 关系谓词的字数上限。一个谓词短语远短于这个值。
MAX_PREDICATE_CHARS = 24

#: 开放谓词：它们不描述"两个实体**因为什么**有关"，因此拿它们建交会点只会造出一张
#: 所有东西都连到所有东西的毛球图，而不是联想。命中即拒收，绝不落图。
#:
#: 这份黑名单是**兜底**，不是主力：主力是抽取提示词要求模型给出具体谓词。留着它是因为
#: 模型总会在少数情况下退化成"相关"——那种边一旦进图，拓扑就永久被污染了。
_OPEN_PREDICATES = {
    "相关", "有关", "关联", "有关系", "关系", "涉及", "属于", "是",
    "有", "互动", "互动过", "认识", "连接", "联系", "存在",
    "relationship", "related", "relates", "related_to", "associated",
    "association", "involves", "is", "has",
}


def is_open_predicate(value: Any) -> bool:
    """True when a relation's label is an open placeholder rather than a predicate."""
    candidate = " ".join(str(value or "").split()).casefold()
    return candidate in _OPEN_PREDICATES


def normalize_predicate(value: Any) -> str:
    """Normalize a relation label into the text that will name a predicate node.

    Returns ``""`` when the label is unusable — an open predicate ("相关") or one long
    enough to be a sentence rather than a predicate. An empty result means the relation
    is published as a plain unnamed edge and **no** predicate node is created for it:
    inventing a交会点 from "相关" would connect everything to everything, which is worse
    than leaving the edge unnamed.
    """
    candidate = " ".join(str(value or "").split())[:MAX_PREDICATE_CHARS].strip()
    if not candidate:
        return ""
    if is_open_predicate(candidate):
        return ""
    return candidate


def filter_relations(
    relations: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach a usable ``predicate`` to each relation, recording the rejected ones.

    The relation itself is never dropped here: a relation whose label is an open
    predicate is still a real edge the conversation stated, it just does not get to
    become a shared node. Rejections are returned so the caller can record them on the
    publication manifest instead of losing the information silently.
    """
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for relation in relations:
        item = dict(relation)
        raw = item.get("name")
        predicate = normalize_predicate(raw)
        item["predicate"] = predicate or None
        if not predicate and str(raw or "").strip():
            rejected.append(
                {
                    "source_id": item.get("source_id"),
                    "target_id": item.get("target_id"),
                    "name": str(raw or ""),
                    "reason": (
                        "open_predicate" if is_open_predicate(raw) else "unusable_predicate"
                    ),
                }
            )
        kept.append(item)
    return kept, rejected



def comparison_matches(text: str) -> List[re.Match[str]]:
    """Every comparison construct in ``text`` (marker plus bounded target)."""
    return list(
        re.finditer(
            _COMPARISON_MARKER + r"[^，,。.！!？?；;：:\n]{0,12}?" + _COMPARISON_TARGET,
            text or "",
        )
    )


def normalize_comparison_target(raw: str) -> str:
    """Strip the measure-word prefix and comparison particles from a target.

    Peeling happens repeatedly so "一样快" (produced when the marker "一样" is
    itself the comparison word) collapses to none and the adverb 快 is not taken
    for an entity.
    """
    target = re.sub(r"\s+", "", str(raw or ""))
    changed = True
    while changed and target:
        changed = False
        for suffix in _COMPARISON_SUFFIXES:
            if target.startswith(suffix) and len(target) > len(suffix):
                target = target[len(suffix):]
                changed = True
                break
    changed = True
    while changed and target:
        changed = False
        for prefix in _MEASURE_PREFIXES:
            if target.startswith(prefix) and len(target) > len(prefix):
                target = target[len(prefix):]
                changed = True
                break
    changed = True
    while changed and target:
        changed = False
        for suffix in _COMPARISON_SUFFIXES:
            if target.endswith(suffix) and len(target) > len(suffix):
                target = target[: -len(suffix)]
                changed = True
                break
    return normalized_text(target)


def comparison_spans(text: str) -> List[tuple[int, int]]:
    """Character spans occupied by comparison constructs (marker + target)."""
    return [(match.start(), match.end()) for match in comparison_matches(text)]


def comparison_targets(text: str) -> set[str]:
    """Names that appear as the target of a comparison in ``text``."""
    targets: set[str] = set()
    for match in comparison_matches(text):
        target = normalize_comparison_target(match.group("target"))
        if target:
            targets.add(target)
    return targets


def is_knowledge_name(name: Any) -> bool:
    """A usable knowledge-entity name: non-empty, not a bare generic word."""
    candidate = normalized_text(str(name or ""))
    if not candidate:
        return False
    if candidate in _GENERIC_NAMES:
        return False
    return bool(re.search(r"[\w\u4e00-\u9fff]", candidate))


def is_knowledge_node(
    name: Any,
    description: Any = "",
    *,
    raw_text: str = "",
) -> bool:
    """True when an extracted entity may become a knowledge node.

    The name must be a usable noun and must not be a figure of speech. A
    description is not required: the conversation may name an entity before it
    says anything about it, and the node is still a legitimate knowledge node.
    """
    if not is_knowledge_name(name):
        return False
    if _is_comparison_only(name, raw_text):
        return False
    return True


def filter_knowledge_nodes(
    nodes: Sequence[Dict[str, Any]],
    raw_text: str = "",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split node payloads into (kept, dropped) using the knowledge rule."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for node in nodes:
        name = node.get("name") or node.get("id")
        if is_knowledge_node(name, node.get("description"), raw_text=raw_text):
            kept.append(node)
        else:
            dropped.append({"id": node.get("id"), "name": node.get("name"), "reason": "not_a_knowledge_entity"})
    return kept, dropped

def merge_descriptions(existing: str, incoming: str) -> Dict[str, Any]:
    """Accumulate one node's description without duplicating sentences.

    The node's description is the union of what the conversation has stated about
    it, in first-seen order. Repeated statements (the same sentence re-extracted
    from a later batch) collapse instead of growing the node forever.
    """
    current = _sentences(existing)
    added: List[str] = []
    for sentence in _sentences(incoming):
        if sentence not in current and sentence not in added:
            added.append(sentence)
    combined = current + added
    return {
        "description": "".join(combined)[:4000],
        "added": added,
        "changed": bool(added),
    }


def description_is_empty(value: Any) -> bool:
    return not str(value or "").strip()


# ------------------------------------------------------------- statements

def _time_anchor_rejection(sentence: str) -> Optional[str]:
    """时间锚点是否使这句话不能成为长期知识（``None`` = 可以）。

    判据分组见 ``_PRESENT_TIME_ANCHORS`` / ``_PAST_TIME_ANCHORS``。这里刻意维持
    "信号组合"：**已经发生过的事**不因为带一个"昨天"就被丢掉，只有它同时在转述一次
    交流（``_SPEECH_ACTIONS``）时才说明它记的是本次会话内的流水账。
    """
    if not any(marker in sentence for marker in _TIME_ANCHORS):
        return None
    if _ABSOLUTE_DATE.search(sentence):
        # "在昨天（2026年10月7日）" —— 句子自带确定日期，这件事是可归档的历史，
        # 相对时间词只是同一件事的口语重复。两组时间词都按这一条短路。
        return None
    if any(marker in sentence for marker in _PRESENT_TIME_ANCHORS):
        # A correction often names the new steady state with "现在": "现在只喝豆浆"
        # is not a passing mood, it is the post-change assertion. Keep it only when
        # the sentence also carries an explicit change marker, so "现在心情不错"
        # remains a transient state and stays out of long-term knowledge.
        if any(marker in sentence for marker in _CHANGE_MARKERS):
            return None
        return "not_durable_time_reference"
    if any(action in sentence for action in _SPEECH_ACTIONS):
        return "not_durable_time_reference"
    return None


def statement_rejection(text: Any) -> Optional[str]:
    """Why this sentence may not become an entity's knowledge, or ``None``.

    The question every rule answers is the same: *does this sentence say something
    about the entity that still holds after this conversation ends?* A sentence
    that narrates the exchange ("先发送…随后称…"), refers to this session
    ("这条消息"), or restates the contact identity the hard join already recorded
    ("QQ号1000000002，通过QQ私聊联系小E。") does not.

    Signals combine rather than act alone, so a durable rule that happens to contain
    one narration word is not lost: "主人要求公开提交前检查 key" contains 要求 and
    still passes; it takes a second signal (a sequence word or a session reference)
    to look like narration.

    时间锚点遵守同一条原则（见 ``_time_anchor_rejection``）：当下/未来锚点单独成立，
    已发生锚点要配上"转述一次交流"才判非持久，句子自带绝对日期时连那个信号也不需要。
    """
    sentence = str(text or "").strip()
    if not sentence:
        return "empty"
    if any(marker in sentence for marker in _SESSION_REFERENCES):
        return "session_reference"
    time_reason = _time_anchor_rejection(sentence)
    if time_reason is not None:
        return time_reason
    if len(sentence) > MAX_STATEMENT_CHARS:
        return "too_long"
    speech = [word for word in _SPEECH_ACTIONS if word in sentence]
    sequential = any(marker in sentence for marker in _SEQUENCE_MARKERS)
    if len(speech) >= 2 and any(marker in sentence for marker in _SEQUENCE_MARKERS + _SESSION_REFERENCES):
        return "process_narration"
    if sequential and speech and sentence.startswith(tuple(_SEQUENCE_MARKERS)):
        # "先发送"你好"两次" / "之后又发送"现在呢"" — the sentence opens on the
        # timeline of this conversation.
        return "process_narration"
    if _NUMBER_RUN.search(sentence) and any(verb in sentence for verb in _CONTACT_VERBS):
        # The number-keyed contact node plus its qq_contact edge already record this.
        return "structural_identity_restated"
    return None


def is_knowledge_statement(text: Any) -> bool:
    """True when a sentence may become knowledge on the object it names."""
    return statement_rejection(text) is None


def filter_statements(value: Any) -> Dict[str, Any]:
    """Split a description into sentences, keep the admissible ones.

    Returns ``{"text", "kept", "dropped"}``. ``dropped`` carries the sentence and the
    reason so a caller can record what was refused instead of silently losing it
    (the same "recorded, not silent" rule the rest of the pipeline follows), and
    ``text`` is at most ``MAX_STATEMENTS_PER_DESCRIPTION`` sentences long because a
    description that keeps growing is how an entity becomes a diary.
    """
    kept: List[str] = []
    dropped: List[Dict[str, str]] = []
    for sentence in _sentences(value):
        reason = statement_rejection(sentence)
        if reason is not None:
            dropped.append({"text": sentence, "reason": reason})
            continue
        if len(kept) >= MAX_STATEMENTS_PER_DESCRIPTION:
            dropped.append({"text": sentence, "reason": "over_statement_limit"})
            continue
        kept.append(sentence)
    return {"text": "".join(kept), "kept": kept, "dropped": dropped}


# ------------------------------------------------------------------ internals


def _is_comparison_only(name: Any, raw_text: str) -> bool:
    """True when every mention of ``name`` sits inside a comparison construct.

    A name that is also used as a real subject elsewhere is kept, so "苹果" in a
    conversation that actually discusses apples is not banned by one simile.
    """
    target = normalized_text(str(name or ""))
    if not target or not raw_text:
        return False
    lowered = normalized_text(raw_text)
    spans = comparison_spans(raw_text)
    if not spans:
        return False
    inside = 0
    total = 0
    for match in re.finditer(re.escape(target), lowered):
        total += 1
        if any(start <= match.start() and match.end() <= end for start, end in spans):
            inside += 1
    return total > 0 and inside == total


def _sentences(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [part.strip() for part in parts if part.strip()]
