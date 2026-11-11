"""Small, dependency-free document chunker for role imports.

Targets are non-overlapping so the source can be audited exactly.  A bounded
context tail is returned separately and is never used as episode evidence.
"""
from __future__ import annotations

import re
from typing import Any


def estimate_tokens(text: str) -> int:
    # Chinese characters are close to one token; Latin words need a little less.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(0, len(text) - cjk)
    return max(1, cjk + (other + 3) // 4)


def _sentence_cut(text: str, limit: int) -> int:
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    matches = list(re.finditer(r"[。！？!?；;\n]", window))
    return (matches[-1].end() if matches else limit) or limit


def bisect_document_chunk(text: str) -> tuple[str, str]:
    """Keep exact source coverage and prefer a sentence boundary near halfway."""
    if len(text) < 2:
        raise ValueError("chunk cannot be split further")
    middle = len(text) // 2
    boundaries = [match.end() for match in re.finditer(r"[。！？!?；;\n]", text)
                  if len(text) // 4 <= match.end() <= len(text) * 3 // 4]
    cut = min(boundaries, key=lambda offset: abs(offset - middle)) if boundaries else middle
    return text[:cut], text[cut:]


def chunk_chapter(text: str, *, max_tokens: int = 4000, context_tokens: int = 400) -> list[dict[str, Any]]:
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    chunks: list[dict[str, Any]] = []
    start = 0
    # Convert the token budget into a conservative character window, then cut
    # backwards at punctuation.  A final hard cut guarantees progress.
    char_budget = max(1, max_tokens * 2)
    while start < len(text):
        end = min(len(text), start + char_budget)
        while end > start + 1 and estimate_tokens(text[start:end]) > max_tokens:
            end -= max(1, (end - start) // 8)
        end = max(start + 1, _sentence_cut(text[start:end], end - start) + start)
        target = text[start:end]
        context = text[max(0, start - context_tokens * 2):start]
        chunks.append({"index": len(chunks), "target_text": target, "context_text": context})
        start = end
    return chunks


def split_document(text: str, *, max_tokens: int = 4000, context_tokens: int = 400) -> list[dict[str, Any]]:
    """Split chapters while preserving exact source order."""
    if not text or not text.strip():
        return []
    chapters = re.split(r"(?=^\s*(?:第[^\n]{1,40}[章节回]|Chapter\s+\d+).*$)", text, flags=re.MULTILINE | re.IGNORECASE)
    result: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(chapters):
        for chunk in chunk_chapter(chapter, max_tokens=max_tokens, context_tokens=context_tokens):
            chunk["chapter_index"] = chapter_index
            chunk["index"] = len(result)
            result.append(chunk)
    return result
