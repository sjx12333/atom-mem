"""Shared pure helpers for the role-level Warm Layer.

Extracted from ``temporary_graph.py`` so the temporary graph and the Warm Layer
cannot drift apart: one implementation of cosine similarity, entry identity,
name normalization, embedding and vector packing, imported by both.

The functions here are deliberately free of state and of any store knowledge.
``TemporaryGraph`` keeps importing them (behaviour unchanged); ``WarmStore``
builds on them. The names are private-looking because they were private in their
original module — the cross-module import is the point, not a new public API.
"""

from __future__ import annotations

import base64
import math
import struct
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 on unusable input."""
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _entry_key(entry: Dict[str, Any]) -> Optional[str]:
    """The identity of one archived entry (``event_id`` wins over ``id``)."""
    return str(entry.get("event_id") or entry.get("id") or "")


def _norm_name(name: Any) -> str:
    """Normalize an extracted entity name; empty when nothing survives."""
    cleaned = str(name or "").strip().strip("。，！？、；：\"'《》「」()（）-—_").strip()
    return cleaned[:64]


def _embed(embedder: Optional[Callable[[str], Sequence[float]]], text: str) -> Optional[List[float]]:
    """Best-effort embedding: any failure yields ``None`` (caller degrades).

    A missing vector must never be fatal: the Warm Layer falls back to lexical
    scoring, and the temporary graph simply skips the entry.
    """
    if embedder is None:
        return None
    try:
        vector = embedder(text)
    except Exception:  # noqa: BLE001 - a broken embedder degrades, never raises
        return None
    if vector is None:
        return None
    try:
        values = [float(item) for item in vector]
    except (TypeError, ValueError):
        return None
    return values or None


def _entry_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("content") or "")


def pack_vector(vector: Sequence[float]) -> str:
    """Encode a float vector as base64 float32 (the sidecar file's format)."""
    values = [float(item) for item in vector]
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def unpack_vector(blob: str) -> List[float]:
    """Decode a base64 float32 vector blob; raises ``ValueError`` when torn."""
    raw = base64.b64decode(blob)
    if len(raw) % 4:
        raise ValueError("vector blob is not a whole number of float32 values")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(item) / norm for item in vector]


def batch_cosine(query_vector: Optional[Sequence[float]], vectors: Iterable[Sequence[float]]) -> List[float]:
    """Cosine of one query against many vectors, numpy-accelerated when present.

    The Warm Layer scans one role's whole retained transcript at most, so this
    stays a single pass; numpy is an optimization, never a requirement.
    """
    rows = [list(item) for item in vectors]
    if query_vector is None or not rows:
        return [0.0 for _ in rows]
    width = len(query_vector)
    usable = [row for row in rows if len(row) == width]
    if len(usable) != len(rows):
        return [
            _cosine(query_vector, row) if len(row) == width else 0.0 for row in rows
        ]
    try:
        import numpy as np  # type: ignore

        query = np.asarray(query_vector, dtype="float32")
        matrix = np.asarray(usable, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        query_norm = float(np.linalg.norm(query)) or 1.0
        scores = (matrix @ query) / (norms * query_norm)
        return [float(item) for item in scores]
    except Exception:  # noqa: BLE001 - numpy is optional
        return [_cosine(query_vector, row) for row in rows]
