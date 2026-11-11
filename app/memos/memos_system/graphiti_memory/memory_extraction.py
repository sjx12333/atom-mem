"""Memory extraction: turning one diary entry into memory nodes.

A **memory node** (graph kind ``event``) is one thing the AI remembers, written from
its own point of view. It is deliberately different from both other object kinds:

- unlike a **knowledge node**, it may be subjective ("我觉得他今天有点累"),
  because it records the AI's own reading of what happened rather than a fact about
  the world;
- unlike a **memory element on an object**, it is a memory in its own right and can
  be the entry point of a recall: it is not a property of some entity.

What keeps it honest is the source: memory nodes are extracted **only from the
diary**, and the diary is written only from a conversation that actually happened.
An empty extraction is a legal, common outcome — most conversations do not deserve
a memory node, and inventing one is worse than remembering nothing.

A memory node may be linked to the knowledge entities it mentions and to the role's
own identity nodes. The links are candidates: only an entity that exists in this
role's partition is linked, so a name the model invented cannot create a node.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .llm_chat import ChatCallError, ChatClient, load_json_object

MAX_MEMORY_SUMMARY_CHARS = 600
MAX_MEMORIES_PER_DIARY = 8
#: A source block of an imported document is longer than one diary entry, so its
#: memory batch may hold a few more items; the shape and the honesty rules are the
#: same ones (see ``SOURCE_MEMORY_RESTRICTIONS``).
MAX_MEMORIES_PER_SOURCE_BLOCK = 12
MAX_ENTITY_LINKS = 8

MEMORY_SYSTEM_PROMPT = "你只输出 JSON。"

MEMORY_RESTRICTIONS = """【记忆抽取 / Memory extraction】
1. 输入是这段对话结束后，AI 角色自己写的日记。请把日记拆成若干条"记忆"。
   The input is the diary the AI wrote after the conversation. Split it into memories.
2. 一条记忆 = 一件 AI 自己想记住的事，用 AI 的第一人称写清"发生了什么、我怎么想"。
   允许主观（感受、判断、决定），但不得加入日记里没有的信息，也不得编造事实。
   One memory is one thing the AI wants to keep, written in its own voice, stating
   what happened and how it read it. Subjective wording is allowed; inventing
   anything the diary does not contain is not.
3. 只有当日记里确实有值得记住的内容时才输出；日常寒暄、没有实际内容的一轮对话应
   返回空数组。宁可为空，也不要硬凑。
   Output a memory only when the diary actually holds something worth keeping. A
   round of small talk yields an empty array. Prefer empty over padding.
4. 每条记忆给出 "related_entities"：日记里提到的、真实存在的实体名（人名、宠物、地点、
   物品、概念等）。只写名字，不要写句子；不确定就留空。
   For each memory, list "related_entities": the real entity names the diary mentions
   (a person, pet, place, object, idea). Names only, never sentences; leave it empty
   when unsure.
5. 只有日记给出明确时间线索时才填写 "occurred_at"（ISO-8601）；否则省略，不要编造时间。
   Only fill "occurred_at" (ISO-8601) when the diary gives an explicit time clue;
   otherwise omit it and never invent a time.
6. summary 必须是一句自洽的话，最多 200 字，不要分点、不要小标题、不要情绪标签 <>。
   Each summary must be one self-contained sentence of at most 200 characters, with
   no bullets, no headings and no emotion tags."""

MEMORY_JSON_SHAPE = """{"memories": [{"summary": "我记住的事",
                           "related_entities": ["实体名"],
                           "occurred_at": "2026-05-01T10:00:00Z"}]}

没有值得记住的内容时输出 {"memories": []}。只输出 JSON，不要输出解释。"""


def build_memory_prompt(*, diary: str, appellation: str = "", display_name: str = "") -> str:
    """The memory-extraction prompt for one diary entry."""
    who = display_name or "AI 角色"
    human = appellation or "用户"
    return (
        MEMORY_RESTRICTIONS
        + "\n\n【输出格式 / Output format】\n"
        + MEMORY_JSON_SHAPE
        + f"\n\n【身份 / Identity】\n你是「{who}」，对话中的另一方是「{human}」。"
        + "\n\n【日记 / Diary】\n"
        + diary
        + "\n\n只输出 JSON，不要输出解释。Output JSON only."
    )


#: ``source_kind`` of every memory node published by 文档创建角色.
SOURCE_MEMORY_KIND = "source_document"

#: Restrictions for the document-import memory pass.
#:
#: The diary path (``MEMORY_RESTRICTIONS``) forbids taking memories straight from raw
#: text, because there the raw text is a *conversation* and the point of the diary is
#: the AI's own reading of it (ADR-0154). 文档创建角色 has no conversation: the source
#: block **is** the role's backstory, so this pass states the same honesty rules
#: against a different input — write what this character lived through, in its own
#: first person, and invent nothing the block does not say.
SOURCE_MEMORY_RESTRICTIONS = """【原文记忆抽取 / Source memory extraction】
1. 输入是角色设定原文的一个片段（小说/剧本/设定稿）。请抽出这个角色**本人**记得的事。
   The input is one block of the source text a role is created from. Extract the things
   **this character itself** remembers from it.
2. 一条记忆 = 这个角色经历过、做过、感受过或决定过的一件事，用该角色的第一人称写清
   "发生了什么、我怎么想"。允许主观（感受、判断、决定），但不得加入原文没有的信息。
   One memory is one thing the character lived through, did, felt or decided, written in
   its own first person. Subjective wording is allowed; inventing anything the source
   block does not contain is not.
3. 只写这个角色自己的经历。与它无关的旁白、别人的私事、纯环境描写不要写成它的记忆；
   原文只提到别人在做某件事时，不要把它写成"我"的经历。
   Only the character's own experience. Narration that does not involve it, other
   people's private business and pure scenery are not its memories.
4. 原文里没有值得记住的内容时输出空数组。宁可为空，也不要硬凑。
   When the block holds nothing worth remembering, output an empty array. Prefer empty
   over padding.
5. 每条记忆给出 "related_entities"：这一段里真实出现的实体名（人名、宠物、地点、物品、
   概念等）。只写名字，不要写句子；不确定就留空。
   For each memory, list "related_entities": the real entity names this block mentions.
   Names only, never sentences; leave it empty when unsure.
6. 只有原文给出明确时间线索时才填写 "occurred_at"（ISO-8601）；否则省略，不要编造时间。
   Only fill "occurred_at" (ISO-8601) when the source gives an explicit time clue;
   otherwise omit it and never invent a time.
7. summary 必须是一句自洽的话，最多 200 字，不要分点、不要小标题、不要情绪标签 <>。
   Each summary must be one self-contained sentence of at most 200 characters, with no
   bullets, no headings and no emotion tags."""


def build_source_memory_prompt(
    *,
    source: str,
    appellation: str = "",
    display_name: str = "",
    source_role_name: str = "",
) -> str:
    """The memory-extraction prompt for one block of an imported source document.

    ``source_role_name`` is the name the character carries *inside that text* (the
    wizard's 原文角色名): it is what the model has to recognise as "我", so it belongs
    in the prompt even when it differs from the display name.
    """
    who = display_name or "AI 角色"
    human = appellation or "用户"
    alias = f"（原文中称作「{source_role_name}」）" if str(source_role_name or "").strip() else ""
    return (
        SOURCE_MEMORY_RESTRICTIONS
        + "\n\n【输出格式 / Output format】\n"
        + MEMORY_JSON_SHAPE
        + f"\n\n【身份 / Identity】\n你要抽取的角色是「{who}」{alias}，它称呼「{human}」。"
        + "\n\n【原文片段 / Source block】\n"
        + str(source or "")
        + "\n\n只输出 JSON，不要输出解释。Output JSON only."
    )


def _norm_name(value: Any) -> str:
    return str(value or "").strip().strip("。，！？、；：\"'《》「」()（）-—_").strip()[:64]

def _norm_summary(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_MEMORY_SUMMARY_CHARS].strip()


def normalize_memories(raw: Any, *, limit: int = MAX_MEMORIES_PER_DIARY) -> List[Dict[str, Any]]:
    """Validate the model's ``memories`` array into storable candidates.

    Duplicate summaries inside one answer are collapsed (the model often restates a
    memory in two ways), malformed entries are dropped, and the result is capped:
    a diary is a short text and dozens of memories from it would be padding. A
    document-import block is longer, so it passes its own ``limit``.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    memories: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
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
        seen.add(key)
        entities: List[str] = []
        raw_entities = item.get("related_entities") or item.get("entities") or []
        if isinstance(raw_entities, str):
            raw_entities = [raw_entities]
        if isinstance(raw_entities, (list, tuple)):
            for value in raw_entities:
                name = _norm_name(value)
                if name and name not in entities:
                    entities.append(name)
        occurred = str(item.get("occurred_at") or "").strip() or None
        memories.append(
            {
                "summary": summary,
                "related_entities": entities[:MAX_ENTITY_LINKS],
                "occurred_at": occurred,
            }
        )
        if len(memories) >= max(1, int(limit)):
            break
    return memories


def parse_memory_payload(content: str, *, limit: int = MAX_MEMORIES_PER_DIARY) -> List[Dict[str, Any]]:
    """Parse one memory-extraction answer; anything unusable yields ``[]``."""
    data = load_json_object(content)
    return normalize_memories(data.get("memories") or data.get("memory"), limit=limit)


class MemoryExtractor:
    """Extracts memory-node candidates from one diary entry."""

    def __init__(self, *, client: Optional[ChatClient] = None, enabled: bool = True):
        self.client = client
        self.enabled = bool(enabled)

    def extract(
        self,
        *,
        diary: str,
        appellation: str = "",
        display_name: str = "",
    ) -> Dict[str, Any]:
        """Return ``{"status": ..., "memories": [...]}``.

        ``status`` distinguishes the three honest outcomes: ``skipped`` (no model
        configured), ``empty`` (the model found nothing worth remembering — a legal
        answer), ``ok`` (candidates produced), ``failed`` (the call itself failed).
        """
        if not self.enabled or self.client is None or not self.client.configured:
            return {"status": "skipped", "reason": "memory extractor is not configured", "memories": []}
        if not str(diary or "").strip():
            return {"status": "empty", "memories": []}
        try:
            raw = self.client.complete(
                build_memory_prompt(
                    diary=diary, appellation=appellation, display_name=display_name
                ),
                system=MEMORY_SYSTEM_PROMPT,
                json_object=True,
                temperature=0.2,
            )
        except ChatCallError as exc:
            return {"status": "failed", "error": str(exc), "memories": []}
        memories = parse_memory_payload(raw)
        if not memories:
            return {"status": "empty", "memories": []}
        return {"status": "ok", "memories": memories}


def resolve_entity_links(
    names: Iterable[str],
    lookup: Dict[str, str],
) -> List[str]:
    """Map model-supplied entity names onto this partition's node ids.

    ``lookup`` maps a normalized name to a node id and only contains nodes of the
    asking role, so a name that does not exist there resolves to nothing and no
    node is created from model output.
    """
    resolved: List[str] = []
    for name in names:
        key = _norm_name(name)
        if not key:
            continue
        node_id = lookup.get(key.casefold())
        if node_id and node_id not in resolved:
            resolved.append(node_id)
    return resolved


def entity_name_lookup(records: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """``{casefolded name: node id}`` for the knowledge nodes of one partition.

    The caller owns the candidate list, and it must contain objects of **one role
    only**: identity names are not unique across partitions (every role's user node
    is named after the same appellation), so a list built from all roles resolves a
    shared name to an arbitrary one of them.
    """
    lookup: Dict[str, str] = {}
    for record in records:
        if str(record.get("kind") or "") == "event":
            continue
        node_id = str(record.get("id") or "")
        name = _norm_name(record.get("name"))
        if node_id and name:
            lookup.setdefault(name.casefold(), node_id)
    return lookup
