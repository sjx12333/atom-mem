"""Raw-RAG style recall for the Warm Layer.

The warm layer keeps the complete recent transcript, so it can be searched with
the same shape as the RAW-RAG baseline: build raw token windows, score them
densely, rerank the candidate windows, then admit only the best few.  Nothing
here extracts facts, builds a graph, or mutates memory.

Design notes
------------
* Windows are built from **whole turn bodies** and only split between turns.
  A turn therefore never appears as a half sentence in the prompt, which is the
  main failure mode of a fixed-size character split on Chinese chat text.
* A window's dense vector is the token-weighted mean of its turns' existing
  vectors.  This keeps the warm layer at one embedding call per query (the
  query itself) instead of re-embedding the whole transcript on every search.
* Rerank is optional and batched: when the configured reranker is available the
  dense candidates are reordered by relevance, but admission still uses the
  window's own cosine/lexical score.  A reranker outage can never admit a
  window that the layer's floor would have rejected.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # cl100k_base is what RAW-RAG uses; a plain approximation is the fallback.
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency
    _ENCODER = None


def token_count(text: str) -> int:
    value = str(text or "")
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(value, disallowed_special=()))
        except Exception:
            pass
    # Cheap conservative approximation: CJK characters are roughly one token,
    # Latin words about 1.3 tokens.  Only used when tiktoken is unavailable.
    latin = sum(1 for char in value if char.isascii() and char.isalnum())
    other = sum(1 for char in value if not char.isascii())
    return max(1, int(other + latin / 4))


@dataclass
class WarmWindow:
    id: str
    text: str
    entries: List[Dict[str, Any]]
    vector: Optional[List[float]]
    token_estimate: int
    latest_at: str


def _normalize(vector: Sequence[float]) -> List[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        return []
    return [item / norm for item in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _entry_text(entry: Dict[str, Any]) -> str:
    body = str(entry.get("content") or "").strip()
    if not body:
        return ""
    stamp = str(entry.get("created_at") or "")[:19].replace("T", " ")
    role = str(entry.get("role") or "user")
    return f"[{stamp}] {role}: {body}" if stamp else f"{role}: {body}"


def _window_vector(entries: Sequence[Dict[str, Any]], vectors: Dict[str, List[float]]) -> Optional[List[float]]:
    total = 0
    accumulator: Optional[List[float]] = None
    for entry in entries:
        vector = vectors.get(str(entry.get("id") or ""))
        if not vector:
            continue
        weight = max(1, token_count(str(entry.get("content") or "")))
        if accumulator is None:
            accumulator = [0.0] * len(vector)
        if len(vector) != len(accumulator):
            continue
        for index, value in enumerate(vector):
            accumulator[index] += float(value) * weight
        total += weight
    if accumulator is None or total <= 0:
        return None
    return _normalize([value / total for value in accumulator])


def build_windows(
    entries: Sequence[Dict[str, Any]],
    vectors: Dict[str, List[float]],
    *,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
) -> List[WarmWindow]:
    """Group consecutive turns into RAW-RAG-like token windows."""
    size = max(32, int(chunk_tokens))
    overlap = max(0, min(int(overlap_tokens), size // 2))
    windows: List[WarmWindow] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n".join(filter(None, (_entry_text(entry) for entry in current)))
        if text.strip():
            key = "|".join(str(entry.get("id") or "") for entry in current)
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
            windows.append(
                WarmWindow(
                    id="warmwin:" + digest,
                    text=text,
                    entries=list(current),
                    vector=_window_vector(current, vectors),
                    token_estimate=token_count(text),
                    latest_at=str(current[-1].get("created_at") or ""),
                )
            )
        # Keep a tail of roughly ``overlap`` tokens so the next window repeats
        # the boundary turn(s).  Whole turns are kept, never a sliced message.
        keep: List[Dict[str, Any]] = []
        kept = 0
        for entry in reversed(current):
            weight = max(1, token_count(str(entry.get("content") or "")))
            if keep and kept + weight > overlap:
                break
            keep.append(entry)
            kept += weight
        current = list(reversed(keep))
        current_tokens = kept

    for entry in entries:
        if not str(entry.get("content") or "").strip():
            continue
        weight = max(1, token_count(_entry_text(entry)))
        if current and current_tokens + weight > size:
            flush()
        current.append(entry)
        current_tokens += weight
    flush()
    return windows


def search_windows(
    query: str,
    *,
    entries: Sequence[Dict[str, Any]],
    vectors: Dict[str, List[float]],
    query_vector: Optional[Sequence[float]],
    gate: float,
    top_k: int,
    relative_floor_ratio: float = 0.0,
    reranker: Optional[Callable[..., Any]] = None,
    dense_top_k: int = 128,
    rerank_top_k: int = 15,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Dense recall + optional rerank over raw warm windows."""
    diagnostics = diagnostics if diagnostics is not None else {}
    windows = build_windows(
        entries,
        vectors,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    diagnostics.update(
        {
            "mode": "raw_rag_windows",
            "windows": len(windows),
            "chunk_tokens": chunk_tokens,
            "chunk_overlap": overlap_tokens,
            "dense_top_k": int(dense_top_k),
            "rerank_top_k": int(rerank_top_k),
        }
    )
    if not windows or not str(query or "").strip():
        return []

    normalized_query = _normalize(query_vector or [])
    scored: List[Tuple[float, WarmWindow]] = []
    for window in windows:
        if normalized_query and window.vector:
            score = _cosine(normalized_query, window.vector)
        else:
            # No query vector: the caller's lexical fallback happens below.
            score = 0.0
        scored.append((score, window))
    scored.sort(key=lambda item: (-item[0], item[1].latest_at), reverse=False)

    dense = scored[: max(1, min(int(dense_top_k), len(scored)))]
    selected = dense[: max(1, min(int(rerank_top_k), len(dense)))]
    rerank_status = "disabled"
    if reranker is not None and len(dense) > 1:
        try:
            method = getattr(reranker, "rerank_many", None)
            if callable(method):
                scores = method(str(query), [window.text for _, window in dense])
                if scores and len(scores) == len(dense):
                    ordered = sorted(
                        zip(dense, scores), key=lambda item: -float(item[1] or 0.0)
                    )
                    selected = [item[0] for item in ordered[: max(1, min(int(rerank_top_k), len(ordered)))]]
                    selected_scores = {id(item[0][1]): float(item[1] or 0.0) for item in zip(dense, scores)}
                    rerank_status = "ok"
                else:
                    selected_scores = {}
                    rerank_status = "empty"
            else:
                selected_scores = {}
                rerank_status = "unsupported"
        except Exception as exc:  # a rerank outage must not kill warm recall
            selected_scores = {}
            rerank_status = f"failed: {str(exc)[:160]}"
    else:
        selected_scores = {}

    ranked: List[Dict[str, Any]] = []
    for dense_score, window in selected:
        lexical = 0.0
        if not normalized_query or not window.vector:
            try:
                from .domain import score_text

                lexical = float(score_text(str(query), window.text) or 0.0)
            except Exception:
                lexical = 0.0
        similarity = max(float(dense_score), lexical)
        if similarity < max(0.0, float(gate)):
            continue
        latest = window.entries[-1] if window.entries else {}
        ranked.append(
            {
                "id": window.id,
                "layer": "WarmLayer",
                "content": window.text,
                "source": "warm_layer",
                "source_type": "warm_layer",
                "match_type": "direct",
                "similarity": round(similarity, 6),
                "final_score": round(similarity, 6),
                "score_source": "warm_raw_embedding" if window.vector else "warm_lexical_fallback",
                "semantic_score": round(float(dense_score), 6),
                "lexical_score": round(lexical, 6),
                "rerank_score": selected_scores.get(id(window)) if selected_scores else None,
                "content_hash": latest.get("content_hash"),
                "batch_id": latest.get("batch_id"),
                "conversation_id": latest.get("conversation_id"),
                "round_no": latest.get("round_no"),
                "role": latest.get("role"),
                "created_at": window.latest_at,
                "created_epoch": 0,
                "token_estimate": window.token_estimate,
                "window_entries": [str(entry.get("id") or "") for entry in window.entries],
                "window_turns": len(window.entries),
            }
        )
    ranked.sort(
        key=lambda item: (
            -float(item.get("rerank_score") if item.get("rerank_score") is not None else item.get("similarity") or 0.0),
            -float(item.get("similarity") or 0.0),
            str(item.get("created_at") or ""),
        )
    )
    if relative_floor_ratio and ranked:
        best = max(float(item.get("similarity") or 0.0) for item in ranked)
        cutoff = best * min(1.0, float(relative_floor_ratio))
        ranked = [item for item in ranked if float(item.get("similarity") or 0.0) >= cutoff]
        diagnostics["relative_floor_cutoff"] = round(cutoff, 6)
    diagnostics["rerank"] = rerank_status
    diagnostics["dense_candidates"] = len(dense)
    diagnostics["selected"] = len(ranked[: max(0, int(top_k))])
    return ranked[: max(0, int(top_k))]
