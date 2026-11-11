"""Versioned compile rule for the role prompt's 用语示例 (example-utterance) section.

ADR-0118 keeps exactly one compiled final body per ``prompt_version_id``: editors may
offer several edit areas, but submitting a version compiles them with an explicit,
versioned rule.  This module *is* that rule for the one structured edit area the role
wizard exposes today — a list of short lines the role itself would say.

Compiled shape (rule version 1)::

    <persona body>

    【用语示例】
    （示例与当前对话无关，只示范语气与长度，不要照搬内容）
    - 示例一
    - 示例二

``split_prompt`` is the inverse used to prefill the editor when an existing role is
edited.  The marker is always appended as the final section, so the *last* occurrence
wins; a persona body that happens to contain the marker earlier is unaffected.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple

PROMPT_COMPILE_RULE_VERSION = 1

EXAMPLES_MARKER = "【用语示例】"
EXAMPLES_INTRO = "（示例与当前对话无关，只示范语气与长度，不要照搬内容）"

#: Hard limit used by :func:`compile_prompt` so a pasted transcript cannot silently
#: turn the persona into a few thousand example lines.
EXAMPLES_MAX_ITEMS = 20
EXAMPLE_MAX_CHARS = 200
PROMPT_MAX_CHARS = 20000

_BULLET_PREFIX = re.compile(r"^(?:[-*·•—]|\d+[.、)）])\s*")
_PLACEHOLDER = re.compile(r"\{\{|\}\}")


def _lenient_lines(examples: Any) -> List[str]:
    """Split raw editor input into stripped, non-empty lines (no limits enforced)."""
    if examples is None:
        return []
    if isinstance(examples, str):
        raw: Iterable[Any] = examples.splitlines()
    elif isinstance(examples, (list, tuple)):
        raw = []
        for item in examples:
            raw.extend(str(item).splitlines())
    else:
        raise ValueError("用语示例必须是字符串或字符串列表")
    lines: List[str] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        text = _BULLET_PREFIX.sub("", text).strip()
        if text:
            lines.append(text)
    return lines


def normalize_examples(examples: Any) -> List[str]:
    """Normalise editor input into the canonical example list (strict).

    Accepts a list of lines or one blob of newline-separated text.  Strips a leading
    bullet/number marker, drops blank lines, removes duplicates while keeping order and
    rejects individual lines or the whole list that exceed the rule's limits.
    """
    lines = _lenient_lines(examples)

    deduped: List[str] = []
    seen = set()
    for text in lines:
        if len(text) > EXAMPLE_MAX_CHARS:
            raise ValueError(
                f"用语示例单条不能超过 {EXAMPLE_MAX_CHARS} 字：{text[:24]}…"
            )
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)

    if len(deduped) > EXAMPLES_MAX_ITEMS:
        raise ValueError(
            f"用语示例最多 {EXAMPLES_MAX_ITEMS} 条（当前 {len(deduped)} 条）"
        )
    return deduped


def validate_prompt_body(content: str) -> str:
    """ADR-0118 integrity check for a compiled prompt body."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("提示词正文不能为空")
    if len(content) > PROMPT_MAX_CHARS:
        raise ValueError(f"提示词正文不能超过 {PROMPT_MAX_CHARS} 字符")
    if _PLACEHOLDER.search(content):
        raise ValueError("提示词正文不能包含未解析的模板占位符 {{ }}")
    return content


def compile_prompt(persona: str, examples: Any = None) -> str:
    """Compile the persona edit area plus the example edit area into one final body."""
    body = (persona or "").strip()
    if not body:
        raise ValueError("Persona/Prompt 不能为空")
    items = normalize_examples(examples)
    if not items:
        return validate_prompt_body(body)
    block = "\n".join(
        [EXAMPLES_MARKER, EXAMPLES_INTRO] + [f"- {item}" for item in items]
    )
    return validate_prompt_body(f"{body}\n\n{block}")


def split_prompt(content: str) -> Tuple[str, List[str]]:
    """Recover ``(persona, examples)`` from a compiled body (lenient, never raises)."""
    text = content or ""
    index = text.rfind(f"\n\n{EXAMPLES_MARKER}")
    if index == -1:
        stripped = text.strip()
        if stripped == EXAMPLES_MARKER or stripped.startswith(f"{EXAMPLES_MARKER}\n"):
            return "", _lenient_lines(stripped[len(EXAMPLES_MARKER):])
        return stripped, []

    persona = text[:index].strip()
    tail = text[index + 2 + len(EXAMPLES_MARKER):]
    lines = _lenient_lines(tail)
    if lines and lines[0] == EXAMPLES_INTRO:
        lines = lines[1:]
    return persona, lines


def examples_block(examples: Sequence[str]) -> str:
    """Render just the example section (used by previews and tests)."""
    items = normalize_examples(list(examples))
    if not items:
        return ""
    return "\n".join([EXAMPLES_MARKER, EXAMPLES_INTRO] + [f"- {item}" for item in items])
