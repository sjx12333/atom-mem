"""Small domain values shared by the Graphiti memory services."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


PREFIXES = {
    "node": "n_",
    "relation": "r_",
    "cognition": "k_",
    "preference": "p_",
    "memory": "m_",
    "event": "e_",
    "fact": "f_",
    "relation_group": "g_",
    "summary": "s_",
    "role": "role_",
    "prompt": "pv_",
    "conversation": "c_",
    "operation": "o_",
}


def typed_id(kind: str) -> str:
    """Return a collision-resistant typed identifier.

    The UUID portion is deliberately opaque to callers; ``kind`` remains an
    explicit field on every persisted object and is never inferred from this
    prefix alone.
    """

    if kind not in PREFIXES:
        raise ValueError(f"unsupported object kind: {kind}")
    return f"{PREFIXES[kind]}{uuid.uuid4().hex}"


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def replace_file(temporary_name: os.PathLike | str, path: os.PathLike | str) -> None:
    """Atomically replace ``path`` with ``temporary_name``, retrying transient
    Windows lock/antivirus errors.

    ``os.replace`` (MoveFileEx) is not reliably atomic when the destination is
    momentarily held open by another handle, an active scan, or a concurrently
    writing thread. On Windows this surfaces as WinError 5 (access denied), 32
    (sharing violation), 33 (lock violation) or 145 (directory not empty). We
    retry those transient codes briefly before giving up.
    """
    last_error: Optional[OSError] = None
    for attempt in range(6):
        try:
            os.replace(temporary_name, path)
            return
        except OSError as exc:
            last_error = exc
            winerror = getattr(exc, "winerror", None)
            if attempt >= 5 or winerror not in (5, 32, 33, 145):
                raise
            time.sleep(0.05 * (attempt + 1))
    raise last_error


def stable_pair(left: str, right: str) -> Tuple[str, str]:
    return tuple(sorted((left, right)))


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def publication_key(snapshot: Mapping[str, Any]) -> str:
    """The identity that tells two frozen publications apart.

    A chat batch and *its own* conversation share a ``conversation_id``: the pet
    keeps one conversation across the 1-hour batch boundary, so a second batch of
    the same conversation carries the same ``conversation_id`` as the first. Using
    that field as the replay key made the second batch look "already published",
    which silently skipped its extraction (measured: two batches, one extractor
    call) and then deleted its transcript as if it had been published.

    The batch is the extraction unit (``memoryreworkplan.md`` 阶段 3), so it wins.
    ``snapshot_id`` identifies a legacy session-scoped freeze, and a snapshot with
    neither falls back to the conversation — exactly the pre-batch behaviour.
    """
    for field in ("batch_id", "snapshot_id"):
        value = str(snapshot.get(field) or "").strip()
        if value:
            return value
    return str(snapshot.get("conversation_id") or "").strip()


def same_publication(manifest: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
    """True when ``manifest`` is the publication of *this* frozen snapshot.

    Manifests written before the Warm Layer carry no ``publication_key``; for them
    the old conversation-scoped rule still applies — but only to a snapshot that
    has no batch identity of its own, so a warm batch can never be mistaken for a
    legacy publication of the conversation it happens to share.
    """
    if not isinstance(manifest, Mapping) or not manifest:
        return False
    stored = str(manifest.get("publication_key") or "").strip()
    if stored:
        return stored == publication_key(snapshot)
    if str(snapshot.get("batch_id") or "").strip():
        return False
    return str(manifest.get("conversation_id") or "").strip() == str(
        snapshot.get("conversation_id") or ""
    ).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    role: str
    content: str
    timestamp: str
    complete: bool = True
    observable: bool = True
    tool_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: 这一条消息的来源（外部渠道，如 QQ 私聊/群聊）。逐条保存而不是整批一个：
    #: 一段会话里可能来回换人（同一角色同时接着几个联系人），整批只留"最近一次
    #: 外部来源"会把 A 说的话记成 B 的事实。``None`` = 本机主人的输入。
    source: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationSnapshot:
    role_uuid: str
    conversation_id: str
    reason: str
    events: List[Dict[str, Any]]
    created_at: str
    snapshot_id: str = field(default_factory=lambda: typed_id("conversation"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    role_uuid: str
    content: str
    kind: str = "memory"
    status: str = "active"
    source: str = "conversation"
    source_id: Optional[str] = None
    node_refs: List[str] = field(default_factory=list)
    relation_refs: List[str] = field(default_factory=list)
    index_brief: Optional[str] = None
    promoted_summary: Optional[str] = None
    manual_protected: bool = False
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def as_complete_events(events: Iterable[ConversationEvent]) -> List[Dict[str, Any]]:
    return [
        event.to_dict()
        for event in events
        if event.complete and event.observable and event.content.strip()
    ]


#: Chinese text has no whitespace token boundaries. Splitting a whole sentence
#: with ``[\w\u4e00-\u9fff]+`` turns it into one token, so a paraphrase can only
#: match by literal substring. Recall therefore uses overlapping 2-grams for CJK
#: runs (single characters are too noisy) and word-level tokens for Latin text.
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LEXICAL_WORD = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")

#: A multi-feature query must overlap at least this many features. One shared
#: 2-gram (e.g. "第一" in "第一轮" vs "第一答") is not enough: common function
#: words and shared prefixes would otherwise admit unrelated sentences.
_MIN_MULTI_FEATURE_HITS = 2


def lexical_features(text: str) -> set[str]:
    """Return the lexical match features for one text.

    CJK runs contribute overlapping 2-grams; single-character runs contribute
    nothing (they carry too little meaning and match almost every sentence).
    Latin words/numbers stay word-level.
    """

    normalized = normalized_text(text)
    features: set[str] = set()
    for run in _CJK_RUN.findall(normalized):
        if len(run) < 2:
            continue
        features.update(run[index:index + 2] for index in range(len(run) - 1))
    features.update(_LEXICAL_WORD.findall(normalized))
    return features


def score_text(query: str, content: str) -> float:
    """Deterministic lexical fallback used when no embedding adapter is configured.

    The score is **query coverage** (how many of the query's features occur in
    the content), not Jaccard: a long memory sentence must not be penalised for
    containing more words than the query. A one-feature query (e.g. a two-
    character Chinese word) passes on that feature; a multi-feature query needs
    at least two hits, so a shared function word or prefix is not enough to
    recall an unrelated sentence.
    """

    normalized_query = normalized_text(query)
    normalized_content = normalized_text(content)
    if not normalized_query or not normalized_content:
        return 0.0
    if normalized_query in normalized_content:
        return 1.0
    query_features = lexical_features(normalized_query)
    content_features = lexical_features(normalized_content)
    if not query_features or not content_features:
        return 0.0
    hits = len(query_features & content_features)
    if hits == 0:
        return 0.0
    if len(query_features) > 1 and hits < _MIN_MULTI_FEATURE_HITS:
        return 0.0
    return hits / len(query_features)
