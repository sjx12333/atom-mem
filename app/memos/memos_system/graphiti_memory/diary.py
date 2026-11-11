"""The AI's own diary: a short first-person entry written at session close.

**Why a diary at all.** The knowledge pass records what the conversation says about
the *world* — entities and the relations between them — and it deliberately records
nothing about what the conversation meant to the AI. A companion needs that second
kind of memory: what happened, how it felt, what it decided. That is what the diary
is for, and the memory nodes are extracted **from the diary**, not from the raw
conversation, so the AI's own viewpoint is the thing that survives.

**Why it is written from the AI's side.** The entry is produced by asking the AI to
write it, in its own persona and voice, through the normal chat channel (one system
line that pins the format plus the conversation as the user turn). It is allowed to
be subjective; that subjectivity is the point, which is exactly why it must never
be mixed into the knowledge pass.

**It is allowed to be empty.** Not every conversation is worth remembering. The
model may answer with a single ``-`` (or ``none``), and then no diary entry is
stored and no memory node is extracted. A forced "today I talked to the user"
entry would put a fabricated memory into the graph.

**It is a source, not a recall target.** The diary text is never injected into a
prompt and never scored by recall: its only consumer is the memory extractor. That
is why it can live as a per-role file on disk instead of a graph object — it is
kept as the provenance of the memory nodes derived from it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain import utc_timestamp
from .llm_chat import ChatCallError, ChatClient

#: An answer that means "nothing worth remembering".
_EMPTY_MARKERS = {"-", "—", "none", "null", "无", "没有", "无。", "（无）", "(无)"}

DEFAULT_DIARY_SYSTEM = (
    "你是这段对话里的 AI 角色本人。请只输出日记正文，不要输出标题、日期、编号或解释。"
)

DIARY_MIN_CHARS = 8
DIARY_MAX_CHARS = 1200


def _normalize_entry(text: str) -> str:
    """Clean a model answer into a diary body, or ``""`` when it is empty."""
    body = str(text or "").strip()
    if body.startswith("```"):
        # Some models wrap prose in a code fence even when asked not to. Only a
        # *leading* fence is stripped and only a trailing fence is removed from the
        # end, so prose that legitimately ends with punctuation is left intact.
        body = body[3:]
        if "\n" in body:
            first, rest = body.split("\n", 1)
            if first.strip().casefold() in {"text", "markdown", "md", "json", ""}:
                body = rest
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        body = body.strip()
    if body.casefold() in _EMPTY_MARKERS:
        return ""
    if len(body) < DIARY_MIN_CHARS:
        return ""
    return body[:DIARY_MAX_CHARS].strip()


class DiaryEntry(dict):
    """One stored diary entry (a dict, so it serialises without ceremony)."""


class DiaryWriter:
    """Writes one role's diary entries and keeps them as extraction provenance."""

    def __init__(
        self,
        root: Path | str,
        *,
        client: Optional[ChatClient] = None,
        enabled: bool = True,
        max_chars: int = DIARY_MAX_CHARS,
    ):
        self.root = Path(root)
        self.client = client
        self.enabled = bool(enabled)
        self.max_chars = int(max_chars)

    # ------------------------------------------------------------------ storage

    def path_for(self, role_uuid: str) -> Path:
        return self.root / "diary" / f"{role_uuid}.jsonl"

    def entries_for(self, role_uuid: str) -> List[Dict[str, Any]]:
        path = self.path_for(role_uuid)
        if not path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    entries.append(item)
        return entries

    def _append(self, role_uuid: str, entry: Dict[str, Any]) -> None:
        path = self.path_for(role_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------- writing

    def build_prompt(
        self,
        *,
        conversation: str,
        persona: str = "",
        appellation: str = "",
        display_name: str = "",
        date_label: str = "",
    ) -> str:
        """The diary instruction: write as yourself, about this conversation."""
        who = display_name or "你"
        human = appellation or "用户"
        lines = [
            f"你是「{who}」。请以第一人称，为下面这段你和「{human}」刚刚结束的对话写一篇简短的日记。",
        ]
        if persona:
            lines.append(
                "写作时保持你的人设与说话风格（下方是你的设定摘录）：\n"
                + str(persona)[:2000]
            )
        lines += [
            "要求：",
            "1. 只写你自己的感受、想法、判断和决定，允许主观；不要写成客观的事件报告。",
            "2. 只依据下面的对话内容，不要编造没有发生的事。",
            "3. 3-5 句话，不要小标题、不要日期、不要分点。",
            "4. 如果这段对话对你没有任何值得记住的东西，只输出一个减号 `-`，不要写别的。",
            f"今天的日期是 {date_label}。" if date_label else "",
            "\n【刚刚结束的对话】\n" + conversation,
        ]
        return "\n".join(line for line in lines if line)

    def write(
        self,
        *,
        role_uuid: str,
        conversation_id: str,
        conversation: str,
        batch_id: str = "",
        persona: str = "",
        appellation: str = "",
        display_name: str = "",
        date_label: str = "",
        source_message_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Produce and store one diary entry.

        Returns ``{"status": "empty"|"written"|"skipped", ...}``. ``skipped`` means
        the pass could not run (no client); ``empty`` means it ran and the AI had
        nothing to record, which is a legal outcome and publishes no memory.

        ``batch_id`` scopes the entry id: several chat batches share one
        ``conversation_id``, and a per-conversation id would make the second
        batch's diary look like the first one's (same id, same derived memory
        node ids). An empty ``batch_id`` keeps the legacy id shape.
        """
        if not self.enabled or self.client is None or not self.client.configured:
            return {"status": "skipped", "reason": "diary writer is not configured"}
        if not str(conversation or "").strip():
            return {"status": "skipped", "reason": "no conversation body"}
        prompt = self.build_prompt(
            conversation=conversation,
            persona=persona,
            appellation=appellation,
            display_name=display_name,
            date_label=date_label,
        )
        try:
            raw = self.client.complete(
                prompt,
                system=DEFAULT_DIARY_SYSTEM,
                json_object=False,
                temperature=0.8,
            )
        except ChatCallError as exc:
            return {"status": "failed", "error": str(exc)}
        body = _normalize_entry(raw)
        if not body:
            return {"status": "empty", "conversation_id": conversation_id}
        entry_id = f"diary:{role_uuid}:{conversation_id}"
        if str(batch_id or "").strip():
            entry_id = f"{entry_id}:{batch_id}"
        entry = {
            "id": entry_id,
            "role_uuid": role_uuid,
            "conversation_id": conversation_id,
            "batch_id": str(batch_id or ""),
            "date": date_label,
            "text": body,
            "source_message_ids": list(source_message_ids or []),
            "created_at": utc_timestamp(),
        }
        self._append(role_uuid, entry)
        return {"status": "written", "entry": entry}
