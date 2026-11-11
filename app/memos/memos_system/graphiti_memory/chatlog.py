"""Per-role chat log: the memory-invocation view **and** a recall source.

Two representations of the same rounds live here, on purpose:

* ``<role_uuid>.json`` — the WebUI view. Each entry records the user input, the
  memories recalled for it and the assistant reply, so the panel can show what
  memory was used for a round (记忆调用情况窗口). This file is a bounded rolling
  window (``max_entries``) and is rewritten on every turn, which is why it is
  *not* what recall reads.
* ``<role_uuid>.recall.jsonl`` — the **recall corpus** (ADR-0163). Append-only,
  one line per complete turn, text only. Recall reads this file, so it never pays
  for the view's ``memories`` payload and never depends on the view's rolling
  truncation of *other* conversations.

Neither representation is a Graphiti extraction source: entries here still never
enter ``ConversationSnapshot`` events or the publication saga. What changed with
ADR-0163 is that the corpus is now readable by recall — a turn is an injectable
fragment with its own, lower relevance floor, ordered by distance from the
newest turn. The session-local temporary graph stays a **separate** source; the
corpus deliberately contains only persisted, completed turns.

Vectors are a cache, never authority: ``<role_uuid>.recall.vec.jsonl`` stores one
``{"id", "d", "v": <base64 float32>}`` line per embedded turn, produced by a
single background worker so a slow embedding endpoint can never delay a turn.
A turn without a vector still recalls, via lexical overlap.
"""

from __future__ import annotations

import base64
import json
import math
import os
import queue
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import replace_file, score_text, utc_timestamp

#: Pure-Python cosine fallback scans at most this many of the newest entries.
#: Without numpy the cost is O(entries x dimension); capping the window keeps a
#: query in the tens of milliseconds instead of seconds.
_PYTHON_SCAN_LIMIT = 400
#: One turn is clipped before it becomes a recall unit. The view keeps the full
#: text; the corpus only has to be enough to recognize and cite the round.
_RECALL_TEXT_LIMIT = 1200
#: How long a recall result may wait for its turn before it is dropped. A search
#: runs before the LLM call and the turn is recorded after the reply, so the gap
#: is one model response. An unconsumed result must expire: a turn whose search
#: never produced a record (extraction-busy interception, pet killed mid-turn)
#: must not be attachable to some unrelated round much later (ADR-0166).
_PENDING_TTL_SECONDS = 900.0
#: Hard cap on unconsumed recall results, so a long-lived process whose turns are
#: never recorded cannot grow the dict without bound.
_PENDING_MAX_ENTRIES = 32


def _pack_vector(vector: Sequence[float]) -> str:
    values = [float(item) for item in vector]
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def _unpack_vector(blob: str) -> List[float]:
    raw = base64.b64decode(blob)
    if len(raw) % 4:
        raise ValueError("vector blob is not a whole number of float32 values")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(item) / norm for item in vector]


class _VectorWorker:
    """One background thread turning turn text into sidecar vector lines.

    Best effort by construction: every failure (no embedder, network down, wrong
    dimension) is dropped. A missing vector costs recall *precision*, never a
    turn, and never the user's reply.
    """

    def __init__(self, store: "ChatLogStore"):
        self._store = store
        self._queue: "queue.Queue[Optional[Tuple[str, str, str]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.submitted = 0
        self.stored = 0
        self.failed = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="chatlog-vectors", daemon=True)
            self._thread.start()

    def submit(self, role_uuid: str, entry_id: str, text: str) -> None:
        if not role_uuid or not entry_id or not str(text or "").strip():
            return
        self.start()
        self.submitted += 1
        self._queue.put((role_uuid, entry_id, str(text)))

    def stop(self) -> None:
        """Drain-and-exit. Only used by tests and orderly shutdown."""
        self._queue.put(None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            role_uuid, entry_id, text = item
            try:
                vector = self._store.embed(text)
            except Exception:
                self.failed += 1
                continue
            if not vector:
                self.failed += 1
                continue
            try:
                self._store.write_vector(role_uuid, entry_id, vector)
                self.stored += 1
            except Exception:
                self.failed += 1


class ChatLogStore:
    """Append-only per-role round log with a token-paired search->turn link."""

    MAX_ENTRIES_PER_ROLE = 500
    #: The WebUI window stays at its historical size: that file is rewritten in
    #: full on every turn *including* each round's recalled-memories payload, so
    #: raising it multiplies per-turn I/O. The corpus is append-only and cheap,
    #: which is where the longer window belongs (ADR-0163).
    DEFAULT_MAX_ENTRIES = 500
    DEFAULT_RECALL_MAX_ENTRIES = 2000

    def __init__(
        self,
        root: Path | str,
        max_entries: Optional[int] = None,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        recall_max_entries: Optional[int] = None,
    ):
        self.root = Path(root) / "chatlog"
        self.max_entries = max(1, int(max_entries or self.DEFAULT_MAX_ENTRIES))
        # The corpus is allowed to be larger than the WebUI window: what the panel
        # can show and what recall may read are different questions, and coupling
        # them is how "the view rolled over" silently became "the memory is gone".
        self.recall_max_entries = max(
            1, int(recall_max_entries or self.DEFAULT_RECALL_MAX_ENTRIES)
        )
        # Recall results waiting for the turn they belong to, keyed by the recall
        # token the caller echoes back (ADR-0166). Never keyed by the searched
        # text: the search sees the raw utterance while the recorded turn carries
        # the channel/sender prefix, so the two are different strings for the same
        # input and an equality test on them silently dropped every recall.
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._embedder = embedder
        self._vectors = _VectorWorker(self)
        # role_uuid -> (corpus_size, entries); invalidated by our own appends.
        self._corpus_cache: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
        # role_uuid -> (vector_file_size, {entry_id: List[float]})
        self._vector_cache: Dict[str, Tuple[int, Dict[str, List[float]]]] = {}

    # ------------------------------------------------------------------ paths

    def _check_role(self, role_uuid: str) -> str:
        if not role_uuid or "/" in role_uuid or "\\" in role_uuid or role_uuid in {".", ".."}:
            raise ValueError("invalid role UUID")
        return role_uuid

    def _path(self, role_uuid: str) -> Path:
        return self.root / f"{self._check_role(role_uuid)}.json"

    def corpus_path(self, role_uuid: str) -> Path:
        return self.root / f"{self._check_role(role_uuid)}.recall.jsonl"

    def vector_path(self, role_uuid: str) -> Path:
        return self.root / f"{self._check_role(role_uuid)}.recall.vec.jsonl"

    # ------------------------------------------------------------------- view

    def _read(self, role_uuid: str) -> List[Dict[str, Any]]:
        path = self._path(role_uuid)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    def _write(self, role_uuid: str, records: List[Dict[str, Any]]) -> None:
        path = self._path(role_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def record_search(
        self,
        role_uuid: str,
        conversation_id: str,
        query: str,
        memories: List[Dict[str, Any]],
    ) -> str:
        """Hold one recall result until its turn arrives; returns its token.

        The token is the **only** thing that links a search to a turn. The
        previous implementation matched on the user text, which cannot work: the
        search runs on the hook's raw utterance (``本机消息：主人：…`` is added
        afterwards, see ADR-0161) while the recorded turn is the message that
        actually went to the model. Once the pet started prefixing local input
        the two strings never matched again, so every round recorded an empty
        ``memories`` list while recall itself kept working — the WebUI
        (记忆调用窗口) showed "no recall" for rounds whose memories had in fact
        been injected (ADR-0166).

        A search is not required to know its conversation: the backend may
        inherit or reassign the conversation id mid-turn, so the token must not
        depend on one. Each search gets its own token; a turn consumes exactly
        one, and the newest search of a turn is the one whose block was injected.
        """
        token = f"rc_{uuid.uuid4().hex}"
        self._pending[token] = {
            "role_uuid": str(role_uuid or ""),
            "conversation_id": str(conversation_id or ""),
            "query": str(query or ""),
            "memories": list(memories or []),
            "at": time.monotonic(),
        }
        # Bound **after** inserting, so the cap is what the dict actually holds.
        self._prune_pending()
        return token

    def _prune_pending(self, now: Optional[float] = None) -> None:
        """Drop recall results whose turn never came, newest kept.

        The comparison is ``>=``, not ``>``: a zero TTL must prune immediately
        instead of surviving on a floating-point tie (the search and the claim can
        land on the same clock reading).
        """
        moment = time.monotonic() if now is None else now
        for token in [
            key
            for key, value in self._pending.items()
            if moment - float(value.get("at") or 0.0) >= _PENDING_TTL_SECONDS
        ]:
            self._pending.pop(token, None)
        while len(self._pending) > _PENDING_MAX_ENTRIES:
            oldest = min(self._pending, key=lambda key: float(self._pending[key].get("at") or 0.0))
            self._pending.pop(oldest, None)

    def record_turn(
        self,
        role_uuid: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        recall_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Finalize one round: user input + recalled memories + AI reply.

        The memories come from the search the caller identifies with
        ``recall_token``. Without a token the round is still recorded, with an
        empty list and ``recall_linked: false`` — a turn that reported no search
        must not inherit an earlier round's recall (ADR-0166).
        """
        user_content = str(user_content or "").strip()
        assistant_content = str(assistant_content or "").strip()
        self._prune_pending()
        # Prune for the moment the token is claimed, **not** for "now": a token
        # whose turn arrives later than the TTL must not be claimable. The search
        # result describes a prompt that has already been sent, so attaching it to
        # a much later round prints one turn's recall on another turn. Opening the
        # window by one tick makes the expiry deterministic instead of depending on
        # whether two `time.monotonic()` reads happen to be distinguishable.
        pending: Optional[Dict[str, Any]] = None
        if recall_token:
            claim_at = time.monotonic() + 1.0
            candidate = self._pending.pop(str(recall_token), None)
            self._prune_pending(claim_at)
            # A token is only good for the role that was searched: one pet
            # process serves one role, but a mismatched pair here would print one
            # role's recall on another role's round.
            if candidate is not None and candidate.get("role_uuid") == str(role_uuid or ""):
                pending = candidate
        memories: List[Dict[str, Any]] = list(pending.get("memories") or []) if pending else []

        records = self._read(role_uuid)
        conv_rounds = [r for r in records if r.get("conversation_id") == conversation_id]
        round_no = max([int(r.get("round_no", 0)) for r in conv_rounds], default=0) + 1
        entry_id = f"chatlog:{role_uuid}:{conversation_id}:{round_no}"
        recorded_at = utc_timestamp()
        entry = {
            "id": entry_id,
            "role_uuid": role_uuid,
            "conversation_id": conversation_id,
            "round_no": round_no,
            "user_content": user_content,
            "assistant_content": assistant_content,
            "memories": memories,
            "recall_linked": pending is not None,
            "recorded_at": recorded_at,
            "complete": True,
        }
        records.append(entry)
        records = records[-self.max_entries :]
        self._write(role_uuid, records)
        # The recall corpus is a separate, append-only projection: recall must not
        # depend on this file's rolling truncation or on its (large) memories field.
        self.append_recall(role_uuid, entry)
        return entry

    def list(
        self,
        role_uuid: str,
        conversation_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return rounds in chronological order (oldest first), newest kept."""
        records = self._read(role_uuid)
        if conversation_id:
            records = [r for r in records if r.get("conversation_id") == conversation_id]
        records = records[-max(0, int(limit)) :]
        return records

    def clear(self, role_uuid: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Remove rounds (used by test cleanup and role reset)."""
        records = self._read(role_uuid)
        if conversation_id is None:
            self._write(role_uuid, [])
            self._pending = {
                token: value
                for token, value in self._pending.items()
                if value.get("role_uuid") != str(role_uuid or "")
            }
            for path in (self.corpus_path(role_uuid), self.vector_path(role_uuid)):
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            self._corpus_cache.pop(role_uuid, None)
            self._vector_cache.pop(role_uuid, None)
            return {"cleared": len(records)}
        kept = [r for r in records if r.get("conversation_id") != conversation_id]
        self._write(role_uuid, kept)
        self._pending = {
            token: value
            for token, value in self._pending.items()
            if not (
                value.get("role_uuid") == str(role_uuid or "")
                and value.get("conversation_id") == str(conversation_id or "")
            )
        }
        self._rewrite_corpus(role_uuid, conversation_id)
        return {"cleared": len(records) - len(kept)}

    # ---------------------------------------------------------- recall corpus

    def _rewrite_corpus(self, role_uuid: str, conversation_id: str) -> None:
        """Drop one conversation from the corpus (only used by ``clear``)."""
        entries = [
            e
            for e in self._read_corpus(role_uuid, use_cache=False)
            if e.get("conversation_id") != conversation_id
        ]
        self._write_jsonl(self.corpus_path(role_uuid), entries)
        wanted = {str(e.get("id") or "") for e in entries}
        vectors = {
            key: value
            for key, value in self._read_vectors(role_uuid, use_cache=False).items()
            if key in wanted
        }
        self._write_jsonl(
            self.vector_path(role_uuid),
            [{"id": key, "d": len(value), "v": _pack_vector(value)} for key, value in vectors.items()],
        )
        self._corpus_cache.pop(role_uuid, None)
        self._vector_cache.pop(role_uuid, None)

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _append_line(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        # One append per record keeps the corpus crash-safe and cheap: the view
        # file is rewritten in full, this one never is.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def append_recall(self, role_uuid: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Append a completed turn to the recall corpus and queue its vector."""
        text = self._recall_text(entry)
        if not text:
            return {}
        row = {
            "id": str(entry.get("id") or ""),
            "conversation_id": str(entry.get("conversation_id") or ""),
            "round_no": int(entry.get("round_no") or 0),
            "recorded_at": str(entry.get("recorded_at") or utc_timestamp()),
            "text": text,
        }
        self._append_line(self.corpus_path(role_uuid), row)
        self._corpus_cache.pop(role_uuid, None)
        # Without an embedder there is nothing to compute, so do not wake the
        # worker at all: a deployment that recalls lexically must not pay a
        # thread per turn for a vector it will never get.
        if self._embedder is not None:
            self._vectors.submit(role_uuid, row["id"], text)
        return row

    @staticmethod
    def _recall_text(entry: Dict[str, Any]) -> str:
        user = " ".join(str(entry.get("user_content") or "").split())
        assistant = " ".join(str(entry.get("assistant_content") or "").split())
        lines = []
        if user:
            lines.append(f"用户：{user}")
        if assistant:
            lines.append(f"AI：{assistant}")
        text = "\n".join(lines)
        return text[:_RECALL_TEXT_LIMIT]

    def _read_corpus(self, role_uuid: str, *, use_cache: bool = True) -> List[Dict[str, Any]]:
        path = self.corpus_path(role_uuid)
        if not path.exists():
            self._corpus_cache.pop(role_uuid, None)
            return []
        try:
            size = path.stat().st_size
        except OSError:
            return []
        if use_cache:
            cached = self._corpus_cache.get(role_uuid)
            if cached is not None and cached[0] == size:
                return cached[1]
        entries: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn trailing line is the only expected damage (crash
                        # during append); skip it rather than losing the corpus.
                        continue
                    if isinstance(row, dict) and row.get("text"):
                        entries.append(row)
        except OSError:
            return []
        entries = entries[-self.recall_max_entries :]
        if use_cache:
            self._corpus_cache[role_uuid] = (size, entries)
        return entries

    # ----------------------------------------------------------------- vectors

    def set_embedder(self, embedder: Optional[Callable[[str], Sequence[float]]]) -> None:
        self._embedder = embedder

    def embed(self, text: str) -> Optional[List[float]]:
        embedder = self._embedder
        if embedder is None:
            return None
        vector = embedder(text)
        if not vector:
            return None
        return [float(item) for item in vector]

    def write_vector(self, role_uuid: str, entry_id: str, vector: Sequence[float]) -> None:
        row = {"id": str(entry_id), "d": len(vector), "v": _pack_vector(vector)}
        self._append_line(self.vector_path(role_uuid), row)
        self._vector_cache.pop(role_uuid, None)

    def _read_vectors(self, role_uuid: str, *, use_cache: bool = True) -> Dict[str, List[float]]:
        path = self.vector_path(role_uuid)
        if not path.exists():
            self._vector_cache.pop(role_uuid, None)
            return {}
        try:
            size = path.stat().st_size
        except OSError:
            return {}
        if use_cache:
            cached = self._vector_cache.get(role_uuid)
            if cached is not None and cached[0] == size:
                return cached[1]
        vectors: Dict[str, List[float]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not row.get("id"):
                            continue
                        vectors[str(row["id"])] = _unpack_vector(str(row.get("v") or ""))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
        except OSError:
            return {}
        if use_cache:
            self._vector_cache[role_uuid] = (size, vectors)
        return vectors

    # ------------------------------------------------------------------ recall

    def search(
        self,
        role_uuid: str,
        query: str,
        *,
        policy: Any,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Recall completed turns, ordered by relevance blended with recency.

        Admission is the source's **own** floor (``chatlog_semantic_threshold``);
        recency (``decay ** distance_from_newest``) only decides the order, so a
        recent but off-topic turn can never buy its way into the prompt. Vectors
        are optional: without them every turn is still scored lexically, which is
        weaker but never empty-by-accident.
        """
        query = str(query or "").strip()
        entries = self._read_corpus(role_uuid)
        diagnostics: Dict[str, Any] = {
            "source": "ChatLog",
            "entries": len(entries),
            "query_embeddings": 0,
            "threshold": float(getattr(policy, "chatlog_semantic_threshold", 0.15)),
            "admitted": 0,
            "lexical_only": True,
            "vector_hits": 0,
        }
        if not query or not entries:
            return {"memories": [], "diagnostics": diagnostics}

        vectors = self._read_vectors(role_uuid) if embedder is not None else {}
        query_vector: Optional[List[float]] = None
        if embedder is not None and vectors:
            try:
                raw = embedder(query)
                if raw:
                    query_vector = _normalize(raw)
                    diagnostics["query_embeddings"] = 1
                    diagnostics["lexical_only"] = False
            except Exception as exc:
                diagnostics["embedding_error"] = str(exc)

        threshold = float(getattr(policy, "chatlog_semantic_threshold", 0.15))
        decay = float(getattr(policy, "chatlog_recency_decay", 0.97))
        semantic_weight = float(getattr(policy, "chatlog_semantic_weight", 0.6))
        recency_weight = float(getattr(policy, "chatlog_recency_weight", 0.4))
        limit = int(top_k if top_k is not None else getattr(policy, "chatlog_top_k", 4))

        sims = self._cosine_scores(query_vector, entries, vectors, diagnostics) if query_vector else {}
        newest = len(entries) - 1
        scored: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            lexical = float(score_text(query, entry.get("text", "")) or 0.0)
            semantic = float(sims.get(index, 0.0))
            relevance = max(semantic, lexical)
            if relevance < threshold:
                continue
            recency = decay ** max(0, newest - index)
            final = semantic_weight * relevance + recency_weight * recency
            scored.append(
                {
                    "id": entry.get("id"),
                    "content": entry.get("text", ""),
                    "source": "chatlog",
                    "source_type": "chatlog",
                    "layer": "ChatLog",
                    "match_type": "direct",
                    "score_source": "chatlog_recall",
                    # The floor above is this source's gate; the graph gates are
                    # applied to graph atoms and must not re-filter a transcript.
                    "rerank_score": None,
                    "similarity": round(relevance, 6),
                    "semantic_score": round(semantic, 6),
                    "lexical_score": round(lexical, 6),
                    "recency_weight": round(recency, 6),
                    "final_score": round(final, 6),
                    "conversation_id": entry.get("conversation_id"),
                    "round_no": entry.get("round_no"),
                    "created_at": entry.get("recorded_at"),
                    "token_estimate": len(str(entry.get("text", ""))),
                }
            )
        scored.sort(key=lambda item: (-item["final_score"], str(item.get("id") or "")))
        selected = scored[: max(0, limit)]
        diagnostics["admitted"] = len(selected)
        diagnostics["candidates"] = len(scored)
        return {"memories": selected, "diagnostics": diagnostics}

    def _cosine_scores(
        self,
        query_vector: Optional[List[float]],
        entries: List[Dict[str, Any]],
        vectors: Dict[str, List[float]],
        diagnostics: Dict[str, Any],
    ) -> Dict[int, float]:
        usable: List[Tuple[int, List[float]]] = []
        for index, entry in enumerate(entries):
            vector = vectors.get(str(entry.get("id") or ""))
            if not vector or len(vector) != len(query_vector):
                continue
            usable.append((index, vector))
        if not usable:
            return {}
        diagnostics["vector_hits"] = len(usable)
        try:
            import numpy as np  # type: ignore

            query_array = np.asarray(query_vector, dtype="float32")
            matrix = np.asarray([item[1] for item in usable], dtype="float32")
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            scores = (matrix @ query_array) / (norms * float(np.linalg.norm(query_array) or 1.0))
            return {
                usable[position][0]: float(scores[position])
                for position in range(len(usable))
            }
        except Exception:
            diagnostics["numpy"] = False
        window = usable[-_PYTHON_SCAN_LIMIT:]
        query_norm = math.sqrt(sum(item * item for item in query_vector)) or 1.0
        result: Dict[int, float] = {}
        for index, vector in window:
            norm = math.sqrt(sum(item * item for item in vector)) or 1.0
            dot = sum(a * b for a, b in zip(query_vector, vector))
            result[index] = dot / (norm * query_norm)
        return result
