"""SQLite bookkeeping store for role-scoped Graphiti memory.

This module is **not** the authority for memory text any more: a role's memory is
stored in the attributes of the node (or relation) it is about, inside the shared
topology, and a memory event is a node of its own. What SQLite keeps is the
rebuildable bookkeeping that graph attributes cannot express:

- working-memory turns, the chat/audit trail and the conversation ledger;
- event bodies with their occurrence/formation time and provenance;
- role whitelist grants;
- the ``(role_uuid, object_id)`` embedding fingerprint cache and the retryable
  publication/vector operation log.

Invariants implemented here (see the plan's "最终不变式" section):

- every active event owns exactly one non-semantic logical edge to its AI role
  node; the edge is created by a dedicated transaction so the node and its
  ownership cannot diverge;
- events are never merged and never connected to a user, entity, concept or other
  event;
- an event keeps its own summary body, occurrence/formation time, provenance and
  retrieval vector metadata;
- ``content_hash`` plus ``embedding_model``/``embedding_version`` allow vector
  reuse without merging two events that share the same text;
- the whole schema is created idempotently and versioned by
  ``schema_migrations``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .domain import normalized_text, text_hash, typed_id, utc_timestamp

SCHEMA_VERSION = 3

#: Mirrors ``events_protocol.EVENT_SOURCE_KINDS`` (the closed set is checked here
#: on the way into the ``events`` table): ``source_document`` is the provenance of
#: a memory node published by 文档创建角色 from a source block.
EVENT_SOURCE_KINDS = ("conversation_happened", "user_reported", "source_document")
EVENT_LINK_KIND = "event_link"

#: Tables that carried a retired storage scheme and are dropped when a database
#: created before their removal is opened. `cognitions`/`memories` held the private
#: cognition atoms and the unanchored plain memories — both are node attributes of
#: the shared topology now. `embedding_records`/`embedding_rebuild_queue` are
#: recreated with a ``(role_uuid, object_id)`` key, because one vector per object
#: let two roles describing the same node overwrite each other.
_RETIRED_TABLES: tuple[str, ...] = (
    "cognitions",
    "memories",
    "embedding_records",
    "embedding_rebuild_queue",
)

#: For each schema-managed table, a column that exists **only** in the current
#: shape. A missing one means the table predates the current scheme and must be
#: rebuilt; the recorded version alone is not trusted because a database can be
#: stamped with a version whose tables were never recreated.
_TABLE_SHAPE_PROBES: dict[str, str] = {
    "embedding_records": "role_uuid",
    "embedding_rebuild_queue": "role_uuid",
}


def _table_has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    """True when ``table`` exists and already carries ``column``."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if row is None:
        return True  # absent: CREATE will build the current shape
    columns = {item["name"] for item in connection.execute(f"PRAGMA table_info({table})")}
    return column in columns

#: Logical object types that can carry a vector.
#:
#: ⚠️ 这是**第三个**必须同步维护的对象类型白名单（另两个在 ``vector_store`` 与
#: ``node_recall``）。``record_embedding`` 用它做校验，漏加一种类型的后果是：
#: 节点已经写进拓扑、向量也已经写进 Neo4j，但发布在这里抛
#: ``unsupported embedding object type`` 并**回滚整批发布**——静默失败的反面，
#: 一个会让整次发布失败的口子。新增节点类型时三处一起改。
OBJECT_TYPES = ("node", "relation", "event", "fact", "relation_group", "predicate", "hook")

# Retryable vector writes/deletes and publication operations share one状态机.
OPERATION_PENDING = "pending"
OPERATION_DONE = "done"
OPERATION_FAILED = "failed"

DEFAULT_BUSY_TIMEOUT_MS = 10_000

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL DEFAULT 'node',
        name TEXT NOT NULL DEFAULT '',
        role_uuid TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS relations (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'relation',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        role_uuid TEXT NOT NULL,
        summary TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        occurred_at TEXT,
        formed_at TEXT NOT NULL,
        source_conversation_id TEXT,
        source_event_ids_json TEXT NOT NULL DEFAULT '[]',
        content_hash TEXT NOT NULL,
        embedding_model TEXT,
        embedding_version TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS working_memory (
        record_key TEXT PRIMARY KEY,
        id TEXT NOT NULL,
        role_uuid TEXT NOT NULL,
        user_id TEXT NOT NULL,
        conversation_id TEXT,
        role TEXT NOT NULL DEFAULT 'user',
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'memos_working_memory',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS whitelist (
        role_uuid TEXT NOT NULL,
        object_id TEXT NOT NULL,
        object_type TEXT NOT NULL DEFAULT 'node',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (role_uuid, object_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_records (
        role_uuid TEXT NOT NULL,
        object_id TEXT NOT NULL,
        object_type TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        model TEXT,
        version TEXT,
        dimension INTEGER,
        vector_point_id TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (role_uuid, object_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_rebuild_queue (
        role_uuid TEXT NOT NULL DEFAULT '',
        object_id TEXT NOT NULL,
        object_type TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT 'missing_embedding',
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (role_uuid, object_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS publication_operations (
        operation_id TEXT PRIMARY KEY,
        role_uuid TEXT NOT NULL,
        conversation_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        manifest_json TEXT NOT NULL DEFAULT '{}',
        published_ids_json TEXT NOT NULL DEFAULT '[]',
        source_message_ids_json TEXT NOT NULL DEFAULT '[]',
        embedding_model TEXT,
        embedding_version TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vector_operations (
        operation_id TEXT PRIMARY KEY,
        object_id TEXT NOT NULL,
        object_type TEXT NOT NULL,
        action TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)

_SCHEMA_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON nodes(kind, name)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_role_status ON nodes(role_uuid, status)",
    "CREATE INDEX IF NOT EXISTS idx_relations_endpoints ON relations(source_id, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_relations_kind_status ON relations(kind, status)",
    "CREATE INDEX IF NOT EXISTS idx_events_role_status ON events(role_uuid, status)",
    "CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_events_conversation ON events(source_conversation_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_formed_at ON events(role_uuid, formed_at)",
    "CREATE INDEX IF NOT EXISTS idx_working_memory_scope ON working_memory(role_uuid, user_id, conversation_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_working_memory_content_hash ON working_memory(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_whitelist_role_status ON whitelist(role_uuid, status)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_records_hash ON embedding_records(content_hash, model, version)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_records_object ON embedding_records(object_type)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_records_role ON embedding_records(role_uuid, object_type)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_queue_status ON embedding_rebuild_queue(status, object_type)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_queue_role ON embedding_rebuild_queue(role_uuid, status)",
    "CREATE INDEX IF NOT EXISTS idx_publication_role_conversation ON publication_operations(role_uuid, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_publication_status ON publication_operations(status)",
    "CREATE INDEX IF NOT EXISTS idx_vector_operations_status ON vector_operations(status, object_type)",
)

# Each active event owns exactly one non-semantic edge to its AI role node.
_EVENT_LINK_UNIQUE_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_event_link_unique "
    "ON relations(source_id) WHERE kind = 'event_link' AND status = 'active'"
)


class MemoryStoreError(RuntimeError):
    """Raised when a write would violate a memory-store invariant."""


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _stable_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def working_memory_key(
    role_uuid: str, user_id: str, conversation_id: Optional[str], record_id: str
) -> str:
    """Identity of one working-memory turn, matching the legacy JSON key.

    The JSON store de-duplicated on
    ``(role_uuid, user_id, conversation_id, id)``, i.e. the same message id may
    legitimately appear once per conversation. Storing that tuple as a hashed
    primary key keeps the same semantics — a plain ``id`` primary key would
    collide across conversations — and avoids SQLite's ``NULL``-is-distinct
    behaviour in unique indexes by mapping a missing conversation to ``""``.
    """
    return text_hash(
        "\x1f".join((str(role_uuid), str(user_id), str(conversation_id or ""), str(record_id)))
    )


class SqliteMemoryStore:
    """Thread-safe SQLite authority for Graphiti memory objects.

    Every connection is created per thread (``check_same_thread`` stays at its
    default) because the service drives recall from request threads and
    extraction/publication from extraction-queue worker threads. All state that
    must be shared lives in the database file, never in the connection.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        initialize: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        if initialize:
            self.initialize()

    # ------------------------------------------------------------------ engine

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, opening and configuring it lazily."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self.path), timeout=self.busy_timeout_ms / 1000.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one immediate transaction, rolling back on error.

        ``BEGIN IMMEDIATE`` takes the write lock up front so a concurrent writer
        fails fast with ``database is locked`` (retried by ``busy_timeout``)
        instead of surfacing a half-applied ``SQLITE_BUSY`` at COMMIT time.
        """
        connection = self.connect()
        with self._write_lock:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            connection.commit()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        yield self.connect()

    def initialize(self) -> Dict[str, Any]:
        """Create the schema and retire tables from superseded storage schemes.

        Order matters, and so does the shape check:

        * `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a table
          left over from an older scheme keeps its old columns. Creating an index
          over `role_uuid` then fails with "no such column: role_uuid" and the whole
          service refuses to start (observed on a real database that an earlier
          cleanup had already stamped with the new version).
        * The recorded schema version alone is therefore not enough. Every
          schema-managed table is checked for a column that only the current shape
          has, and a mismatch means "old shape" — the table is dropped and recreated
          once, before any index statement runs.

        Memory rows in a retired table have no reader by definition, so rebuilding
        them costs nothing that was not already unreachable.
        """
        with self.transaction() as transaction:
            transaction.execute(_SCHEMA_STATEMENTS[0])  # schema_migrations
            applied = {
                row["version"]
                for row in transaction.execute("SELECT version FROM schema_migrations").fetchall()
            }
            # Tables that no longer exist in the schema at all: they are not
            # recreated, so they must be dropped explicitly or they live on.
            for table in _RETIRED_TABLES:
                transaction.execute(f"DROP TABLE IF EXISTS {table}")
            # Schema-managed tables that are present but in the old shape.
            for table, probe in _TABLE_SHAPE_PROBES.items():
                if _table_has_column(transaction, table, probe):
                    continue
                transaction.execute(f"DROP TABLE IF EXISTS {table}")
            for statement in _SCHEMA_STATEMENTS:
                transaction.execute(statement)
            for statement in _SCHEMA_INDEXES:
                transaction.execute(statement)
            if SCHEMA_VERSION not in applied:
                transaction.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, utc_timestamp()),
                )
            transaction.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
            transaction.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('revision', '0') "
                "ON CONFLICT(key) DO NOTHING"
            )
            # The event-link uniqueness index is created after the tables so an
            # existing database with legacy rows fails loudly instead of
            # silently dropping the invariant.
            try:
                transaction.execute(_EVENT_LINK_UNIQUE_INDEX)
            except sqlite3.IntegrityError as exc:  # pragma: no cover - corruption guard
                raise MemoryStoreError(
                    "event link uniqueness cannot be established; repair duplicate "
                    f"active event links before migrating: {exc}"
                ) from exc
        return {"schema_version": SCHEMA_VERSION, "revision": self.revision()}

    # --------------------------------------------------------------- utilities

    def revision(self) -> int:
        row = self.connect().execute(
            "SELECT value FROM schema_meta WHERE key = 'revision'"
        ).fetchone()
        try:
            return int(row["value"]) if row is not None else 0
        except (TypeError, ValueError):
            return 0

    def bump_revision(self) -> int:
        with self.transaction() as transaction:
            transaction.execute(
                "UPDATE schema_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
                "WHERE key = 'revision'"
            )
        return self.revision()

    @staticmethod
    def migrate(path: os.PathLike[str] | str) -> Dict[str, Any]:
        """Apply schema migrations to an existing database file."""
        store = SqliteMemoryStore(path)
        try:
            return store.initialize()
        finally:
            store.close()

    # ------------------------------------------------------------ node/relation

    def upsert_node(
        self,
        node_id: str,
        name: str,
        *,
        kind: str = "node",
        role_uuid: Optional[str] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not node_id:
            raise ValueError("node id is required")
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO nodes (id, kind, name, role_uuid, status, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    name = excluded.name,
                    role_uuid = COALESCE(excluded.role_uuid, nodes.role_uuid),
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (node_id, kind, name, role_uuid, status, now, now, _as_json(metadata or {})),
            )
        return self.get_node(node_id) or {}

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._node_row(row) if row else None

    def list_nodes(
        self,
        *,
        role_uuid: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM nodes WHERE 1 = 1"
        params: List[Any] = []
        if role_uuid is not None:
            query += " AND role_uuid = ?"
            params.append(role_uuid)
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id"
        rows = self.connect().execute(query, params).fetchall()
        return [self._node_row(row) for row in rows]

    def list_nodes_bulk(self, node_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        ids = [str(item) for item in node_ids if item]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connect().execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {row["id"]: self._node_row(row) for row in rows}

    @staticmethod
    def _node_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "name": row["name"],
            "role_uuid": row["role_uuid"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _from_json(row["metadata_json"], {}),
            "object_type": "event" if row["kind"] == "event" else "node",
        }

    def upsert_relation(
        self,
        source_id: str,
        target_id: str,
        *,
        relation_id: Optional[str] = None,
        kind: str = "relation",
        status: str = "active",
    ) -> Dict[str, Any]:
        """Create or reuse the one logical edge for an unordered node pair.

        Rejects events: an event may only own the single non-semantic
        ``event_link`` edge created by :meth:`link_event_to_role`.
        """
        if not source_id or not target_id:
            raise ValueError("relation endpoints are required")
        if source_id == target_id:
            raise ValueError("self relations are not supported")
        normal_kind = kind != EVENT_LINK_KIND
        for endpoint in (source_id, target_id):
            if normal_kind and self._is_event(endpoint):
                raise MemoryStoreError(
                    f"event {endpoint} cannot participate in a normal relation"
                )
        pair = _stable_pair(source_id, target_id)
        now = utc_timestamp()
        existing = self.find_relation(pair[0], pair[1], kind=kind)
        if existing is not None:
            return existing
        new_id = relation_id or typed_id("relation")
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO relations (id, source_id, target_id, kind, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id = excluded.source_id,
                    target_id = excluded.target_id,
                    kind = excluded.kind,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (new_id, pair[0], pair[1], kind, status, now, now),
            )
        return self.get_relation(new_id) or {}

    def _is_event(self, node_id: str) -> bool:
        """An event is either an ``events`` row or a ``kind='event'`` node row."""
        row = self.connect().execute(
            "SELECT 1 FROM events WHERE id = ? "
            "UNION ALL SELECT 1 FROM nodes WHERE id = ? AND kind = 'event' LIMIT 1",
            (node_id, node_id),
        ).fetchone()
        return row is not None

    def find_relation(
        self, source_id: str, target_id: str, *, kind: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        pair = _stable_pair(source_id, target_id)
        query = "SELECT * FROM relations WHERE source_id = ? AND target_id = ? AND status = 'active'"
        params: List[Any] = [pair[0], pair[1]]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY id LIMIT 1"
        row = self.connect().execute(query, params).fetchone()
        return self._relation_row(row) if row else None

    def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute(
            "SELECT * FROM relations WHERE id = ?", (relation_id,)
        ).fetchone()
        return self._relation_row(row) if row else None

    def list_relations(
        self,
        *,
        node_id: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM relations WHERE 1 = 1"
        params: List[Any] = []
        if node_id is not None:
            query += " AND (source_id = ? OR target_id = ?)"
            params.extend([node_id, node_id])
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id"
        rows = self.connect().execute(query, params).fetchall()
        return [self._relation_row(row) for row in rows]

    @staticmethod
    def _relation_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "endpoints": [row["source_id"], row["target_id"]],
            "kind": row["kind"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------ events

    def create_event(
        self,
        role_uuid: str,
        summary: str,
        *,
        source_kind: str,
        occurred_at: Optional[str] = None,
        formed_at: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        source_event_ids: Optional[Iterable[str]] = None,
        event_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        """Create an event node and its single logical link in one transaction.

        The link is non-semantic: it carries no name/fact/description.
        ``occurred_at`` falls back to ``formed_at`` only when the caller could not
        locate a more precise time, which is the plan's rule ("无法定位更精确时间时，事件时间使用对应会话消息时间"); the
        caller passes the message time explicitly, we never invent one.
        """
        if not role_uuid:
            raise ValueError("event role_uuid is required")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("event summary cannot be empty")
        if source_kind not in EVENT_SOURCE_KINDS:
            raise ValueError(
                f"unsupported event source_kind: {source_kind!r}; "
                f"expected one of {EVENT_SOURCE_KINDS}"
            )
        formed = formed_at or utc_timestamp()
        new_id = event_id or typed_id("event")
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO events (
                    id, role_uuid, summary, source_kind, occurred_at, formed_at,
                    source_conversation_id, source_event_ids_json, content_hash,
                    embedding_model, embedding_version, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary = excluded.summary,
                    source_kind = excluded.source_kind,
                    occurred_at = excluded.occurred_at,
                    formed_at = excluded.formed_at,
                    source_conversation_id = excluded.source_conversation_id,
                    source_event_ids_json = excluded.source_event_ids_json,
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    new_id,
                    role_uuid,
                    summary.strip(),
                    source_kind,
                    occurred_at,
                    formed,
                    source_conversation_id,
                    _as_json([str(item) for item in (source_event_ids or [])]),
                    text_hash(summary),
                    embedding_model,
                    embedding_version,
                    status,
                    now,
                    now,
                ),
            )
            # The role node itself is owned by the shared topology; the link only
            # requires the id to exist, so a partially-migrated database cannot
            # end up with an unlinked event.
            transaction.execute(
                """
                INSERT INTO relations (id, source_id, target_id, kind, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET status = 'active', updated_at = excluded.updated_at
                """,
                (f"{EVENT_LINK_KIND}:{new_id}", new_id, role_uuid, EVENT_LINK_KIND, now, now),
            )
        return self.get_event(new_id) or {}

    def link_event_to_role(
        self, event_id: str, role_uuid: str, *, operation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Idempotently ensure the event's single logical link to its role.

        Raises :class:`MemoryStoreError` when the event already belongs to a
        different role, so a retried publication can never re-parent an event.
        """
        event = self.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        if event["role_uuid"] != role_uuid:
            raise MemoryStoreError(
                f"event {event_id} already belongs to role {event['role_uuid']}"
            )
        now = utc_timestamp()
        link_id = f"{EVENT_LINK_KIND}:{event_id}"
        link = self.get_relation(link_id)
        existing = self.list_relations(node_id=event_id, kind=EVENT_LINK_KIND)
        if link is None and existing:
            # Repair a legacy/generated link id instead of adding a second edge.
            link = existing[0]
        if link is not None:
            return link
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO relations (id, source_id, target_id, kind, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (link_id, event_id, role_uuid, EVENT_LINK_KIND, now, now),
            )
        return self.get_relation(link_id) or {}

    def event_role_link(self, event_id: str) -> Optional[Dict[str, Any]]:
        links = self.list_relations(node_id=event_id, kind=EVENT_LINK_KIND)
        return links[0] if links else None

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._event_row(row) if row else None

    def list_events(
        self,
        role_uuid: Optional[str] = None,
        *,
        status: Optional[str] = "active",
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE 1 = 1"
        params: List[Any] = []
        if role_uuid is not None:
            query += " AND role_uuid = ?"
            params.append(role_uuid)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if since is not None:
            query += " AND COALESCE(occurred_at, formed_at) >= ?"
            params.append(since)
        if until is not None:
            query += " AND COALESCE(occurred_at, formed_at) <= ?"
            params.append(until)
        query += " ORDER BY COALESCE(occurred_at, formed_at), formed_at, id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = self.connect().execute(query, params).fetchall()
        return [self._event_row(row) for row in rows]

    def list_events_bulk(self, event_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Batch-read event bodies so recall never scans one row at a time."""
        ids = [str(item) for item in event_ids if item]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connect().execute(
            f"SELECT * FROM events WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {row["id"]: self._event_row(row) for row in rows}

    @staticmethod
    def _event_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "role_uuid": row["role_uuid"],
            "summary": row["summary"],
            "content": row["summary"],
            "source_kind": row["source_kind"],
            "occurred_at": row["occurred_at"],
            "formed_at": row["formed_at"],
            "event_time": row["occurred_at"] or row["formed_at"],
            "source_conversation_id": row["source_conversation_id"],
            "source_event_ids": _from_json(row["source_event_ids_json"], []),
            "content_hash": row["content_hash"],
            "embedding_model": row["embedding_model"],
            "embedding_version": row["embedding_version"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "object_type": "event",
        }

    def events_by_content_hash(self, content_hash: str, role_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE content_hash = ?"
        params: List[Any] = [content_hash]
        if role_uuid is not None:
            query += " AND role_uuid = ?"
            params.append(role_uuid)
        rows = self.connect().execute(query, params).fetchall()
        return [self._event_row(row) for row in rows]

    def count_orphan_events(self) -> int:
        """Events without exactly one active role link (diagnostics invariant)."""
        row = self.connect().execute(
            """
            SELECT COUNT(*) AS total FROM events e
            WHERE e.status = 'active' AND (
                SELECT COUNT(*) FROM relations r
                WHERE r.kind = ? AND r.status = 'active'
                  AND (r.source_id = e.id OR r.target_id = e.id)
            ) <> 1
            """,
            (EVENT_LINK_KIND,),
        ).fetchone()
        return int(row["total"]) if row else 0

    # --------------------------------------------------------- working memory

    def save_working_memory(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert working-memory turns, skipping identity duplicates.

        Returns only the rows that were actually inserted, which is what callers
        report as ``added``. One transaction for the whole batch: a turn's user
        and assistant message land together or not at all.
        """
        prepared: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            content = str(record.get("content") or "").strip()
            role_uuid = str(record.get("role_uuid") or "")
            user_id = str(record.get("user_id") or "")
            record_id = str(record.get("id") or "")
            if not content or not role_uuid or not record_id:
                continue
            conversation_id = record.get("conversation_id")
            conversation_id = None if conversation_id is None else str(conversation_id)
            now = utc_timestamp()
            prepared.append(
                {
                    "record_key": working_memory_key(role_uuid, user_id, conversation_id, record_id),
                    "id": record_id,
                    "role_uuid": role_uuid,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "role": str(record.get("role") or "user"),
                    "content": content,
                    "content_hash": text_hash(content),
                    "source": str(record.get("source") or "memos_working_memory"),
                    "status": str(record.get("status") or "active"),
                    "created_at": str(record.get("created_at") or now),
                    "updated_at": str(record.get("updated_at") or now),
                }
            )
        if not prepared:
            return []
        added: List[Dict[str, Any]] = []
        with self.transaction() as transaction:
            for item in prepared:
                cursor = transaction.execute(
                    """
                    INSERT INTO working_memory (
                        record_key, id, role_uuid, user_id, conversation_id, role,
                        content, content_hash, source, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_key) DO NOTHING
                    """,
                    (
                        item["record_key"],
                        item["id"],
                        item["role_uuid"],
                        item["user_id"],
                        item["conversation_id"],
                        item["role"],
                        item["content"],
                        item["content_hash"],
                        item["source"],
                        item["status"],
                        item["created_at"],
                        item["updated_at"],
                    ),
                )
                if cursor.rowcount:
                    added.append(self._working_memory_row(item))
        return added

    def list_working_memory(
        self,
        role_uuid: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        *,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        """Turns for one role/user, newest-last in insertion order.

        Ordering is by ``rowid`` (insertion order) rather than ``created_at``:
        a turn's user and assistant messages share a timestamp, so ordering by
        time would shuffle them, and ``id`` order would put the assistant first.
        """
        query = "SELECT * FROM working_memory WHERE role_uuid = ? AND user_id = ?"
        params: List[Any] = [role_uuid, user_id]
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY rowid"
        rows = self.connect().execute(query, params).fetchall()
        return [self._working_memory_row(row) for row in rows]

    def count_working_memory(self, *, status: Optional[str] = "active") -> int:
        query = "SELECT COUNT(*) AS total FROM working_memory"
        params: List[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        row = self.connect().execute(query, params).fetchone()
        return int(row["total"]) if row else 0

    @staticmethod
    def _working_memory_row(row: Any) -> Dict[str, Any]:
        if isinstance(row, dict):
            return {key: row.get(key) for key in (
                "record_key", "id", "role_uuid", "user_id", "conversation_id", "role",
                "content", "content_hash", "source", "status", "created_at", "updated_at",
            )}
        return {
            "record_key": row["record_key"],
            "id": row["id"],
            "role_uuid": row["role_uuid"],
            "user_id": row["user_id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "source": row["source"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --------------------------------------------------------------- whitelist

    def grant(self, role_uuid: str, object_id: str, *, object_type: str = "node") -> Dict[str, Any]:
        if not role_uuid or not object_id:
            raise ValueError("whitelist grants require a role and an object")
        if object_type not in OBJECT_TYPES:
            raise ValueError(f"unsupported whitelist object type: {object_type}")
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO whitelist (role_uuid, object_id, object_type, status, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                ON CONFLICT(role_uuid, object_id) DO UPDATE SET
                    object_type = excluded.object_type,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (role_uuid, object_id, object_type, now, now),
            )
        return {"role_uuid": role_uuid, "object_id": object_id, "object_type": object_type}

    def is_allowed(self, role_uuid: str, object_id: str) -> bool:
        row = self.connect().execute(
            "SELECT 1 FROM whitelist WHERE role_uuid = ? AND object_id = ? AND status = 'active'",
            (role_uuid, object_id),
        ).fetchone()
        return row is not None

    def allowed_ids(self, role_uuid: str, *, object_type: Optional[str] = None) -> List[str]:
        query = "SELECT object_id FROM whitelist WHERE role_uuid = ? AND status = 'active'"
        params: List[Any] = [role_uuid]
        if object_type is not None:
            query += " AND object_type = ?"
            params.append(object_type)
        query += " ORDER BY object_id"
        return [row["object_id"] for row in self.connect().execute(query, params).fetchall()]

    def revoke(self, role_uuid: str, object_id: str) -> bool:
        with self.transaction() as transaction:
            cursor = transaction.execute(
                "UPDATE whitelist SET status = 'revoked', updated_at = ? "
                "WHERE role_uuid = ? AND object_id = ? AND status = 'active'",
                (utc_timestamp(), role_uuid, object_id),
            )
        return cursor.rowcount > 0

    # ---------------------------------------------------- embedding bookkeeping

    def record_embedding(
        self,
        object_id: str,
        object_type: str,
        *,
        role_uuid: str,
        content_hash: str,
        model: Optional[str],
        version: Optional[str],
        dimension: int,
        vector_point_id: str,
    ) -> Dict[str, Any]:
        """Remember that one role's vector for one object is up to date.

        Keyed by ``(role_uuid, object_id)``: the vector now lives on the role's own
        copy inside Neo4j, so the bookkeeping has to be role-scoped too. With a
        bare ``object_id`` key the second role to describe a shared node silently
        overwrote the first role's fingerprint.
        """
        if object_type not in OBJECT_TYPES:
            raise ValueError(f"unsupported embedding object type: {object_type}")
        if not str(role_uuid or "").strip():
            raise ValueError("role_uuid is required to record an embedding")
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO embedding_records (
                    role_uuid, object_id, object_type, content_hash, model, version,
                    dimension, vector_point_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(role_uuid, object_id) DO UPDATE SET
                    object_type = excluded.object_type,
                    content_hash = excluded.content_hash,
                    model = excluded.model,
                    version = excluded.version,
                    dimension = excluded.dimension,
                    vector_point_id = excluded.vector_point_id,
                    updated_at = excluded.updated_at
                """,
                (
                    str(role_uuid), object_id, object_type, content_hash, model, version,
                    int(dimension), vector_point_id, now,
                ),
            )
        return self.get_embedding(object_id, role_uuid=role_uuid) or {}

    def get_embedding(self, object_id: str, *, role_uuid: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute(
            "SELECT * FROM embedding_records WHERE role_uuid = ? AND object_id = ?",
            (str(role_uuid), object_id),
        ).fetchone()
        if row is None:
            return None
        return self._embedding_row(row)

    @staticmethod
    def _embedding_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "role_uuid": row["role_uuid"],
            "object_id": row["object_id"],
            "object_type": row["object_type"],
            "content_hash": row["content_hash"],
            "model": row["model"],
            "version": row["version"],
            "dimension": row["dimension"],
            "vector_point_id": row["vector_point_id"],
            "updated_at": row["updated_at"],
        }

    def reusable_point(
        self, content_hash: str, model: Optional[str], version: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Find an existing vector whose text hash and model fingerprint match.

        Reusing a vector is not event merging: the caller still writes its own
        object with its own id. The result is role-scoped, so a caller must only
        reuse it inside the role that owns it.
        """
        row = self.connect().execute(
            """
            SELECT * FROM embedding_records
            WHERE content_hash = ? AND COALESCE(model, '') = COALESCE(?, '')
              AND COALESCE(version, '') = COALESCE(?, '')
            ORDER BY updated_at LIMIT 1
            """,
            (content_hash, model, version),
        ).fetchone()
        if row is None:
            return None
        return self._embedding_row(row)

    def delete_embedding_record(self, object_id: str, *, role_uuid: Optional[str] = None) -> None:
        with self.transaction() as transaction:
            if role_uuid is None:
                transaction.execute("DELETE FROM embedding_records WHERE object_id = ?", (object_id,))
            else:
                transaction.execute(
                    "DELETE FROM embedding_records WHERE role_uuid = ? AND object_id = ?",
                    (str(role_uuid), object_id),
                )

    def enqueue_embedding_rebuild(
        self, object_id: str, object_type: str, *, reason: str, role_uuid: str = ""
    ) -> Dict[str, Any]:
        """Queue a missing/renamed embedding instead of recomputing every recall."""
        now = utc_timestamp()
        role = str(role_uuid or "")
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO embedding_rebuild_queue (
                    role_uuid, object_id, object_type, reason, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(role_uuid, object_id) DO UPDATE SET
                    object_type = excluded.object_type,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (role, object_id, object_type, reason, now, now),
            )
        return {
            "role_uuid": role,
            "object_id": object_id,
            "object_type": object_type,
            "reason": reason,
            "status": "pending",
        }

    def list_embedding_rebuild_queue(
        self, *, status: Optional[str] = "pending", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM embedding_rebuild_queue"
        params: List[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at, object_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        return [dict(row) for row in self.connect().execute(query, params).fetchall()]

    def complete_embedding_rebuild(self, object_id: str, *, role_uuid: Optional[str] = None) -> None:
        with self.transaction() as transaction:
            if role_uuid is None:
                transaction.execute(
                    "UPDATE embedding_rebuild_queue SET status = 'done', updated_at = ? "
                    "WHERE object_id = ?",
                    (utc_timestamp(), object_id),
                )
            else:
                transaction.execute(
                    "UPDATE embedding_rebuild_queue SET status = 'done', updated_at = ? "
                    "WHERE role_uuid = ? AND object_id = ?",
                    (utc_timestamp(), str(role_uuid), object_id),
                )

    def count_missing_embeddings(self, role_uuid: Optional[str] = None) -> int:
        """Active events that still have no embedding record (diagnostics).

        Events are the only object whose body SQLite still owns; a node's or a
        relation's vector freshness is derived from its attribute text and checked
        by the recall path itself, so it is not counted here.
        """
        connection = self.connect()
        role_clause = " AND t.role_uuid = ?" if role_uuid is not None else ""
        params: List[Any] = [role_uuid] if role_uuid is not None else []
        role_match = " AND r.role_uuid = t.role_uuid"
        query = (
            "SELECT COUNT(*) AS total FROM events t WHERE t.status = 'active'"
            f"{role_clause}"
            " AND NOT EXISTS (SELECT 1 FROM embedding_records r"
            " WHERE r.object_id = t.id AND r.object_type = 'event'"
            f"{role_match})"
        )
        row = connection.execute(query, params).fetchone()
        return int(row["total"]) if row else 0

    # --------------------------------------------------- publications/vectors

    def begin_publication(
        self,
        operation_id: str,
        role_uuid: str,
        *,
        conversation_id: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO publication_operations (
                    operation_id, role_uuid, conversation_id, status, manifest_json,
                    published_ids_json, source_message_ids_json, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, '[]', '[]', 0, ?, ?)
                ON CONFLICT(operation_id) DO NOTHING
                """,
                (operation_id, role_uuid, conversation_id, _as_json(manifest or {}), now, now),
            )
        return self.get_publication(operation_id) or {}

    def get_publication(self, operation_id: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute(
            "SELECT * FROM publication_operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return self._publication_row(row) if row else None

    def complete_publication(
        self,
        operation_id: str,
        *,
        published_ids: Sequence[str],
        source_message_ids: Sequence[str] = (),
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Idempotently record a finished publication (retries must not duplicate)."""
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                UPDATE publication_operations SET
                    status = 'done',
                    published_ids_json = ?,
                    source_message_ids_json = ?,
                    embedding_model = COALESCE(?, embedding_model),
                    embedding_version = COALESCE(?, embedding_version),
                    manifest_json = CASE WHEN ? = '{}' THEN manifest_json ELSE ? END,
                    updated_at = ?
                WHERE operation_id = ?
                """,
                (
                    _as_json([str(item) for item in published_ids]),
                    _as_json([str(item) for item in source_message_ids]),
                    embedding_model,
                    embedding_version,
                    _as_json(manifest or {}),
                    _as_json(manifest or {}),
                    now,
                    operation_id,
                ),
            )
        return self.get_publication(operation_id) or {}

    def fail_publication(self, operation_id: str, error: str) -> Dict[str, Any]:
        with self.transaction() as transaction:
            transaction.execute(
                """
                UPDATE publication_operations SET
                    status = 'retryable',
                    attempts = attempts + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE operation_id = ?
                """,
                (str(error)[:2000], utc_timestamp(), operation_id),
            )
        return self.get_publication(operation_id) or {}

    def list_publications(
        self, *, status: Optional[str] = None, role_uuid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM publication_operations WHERE 1 = 1"
        params: List[Any] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if role_uuid is not None:
            query += " AND role_uuid = ?"
            params.append(role_uuid)
        query += " ORDER BY created_at, operation_id"
        return [
            self._publication_row(row) for row in self.connect().execute(query, params).fetchall()
        ]

    @staticmethod
    def _publication_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "operation_id": row["operation_id"],
            "role_uuid": row["role_uuid"],
            "conversation_id": row["conversation_id"],
            "status": row["status"],
            "manifest": _from_json(row["manifest_json"], {}),
            "published_ids": _from_json(row["published_ids_json"], []),
            "source_message_ids": _from_json(row["source_message_ids_json"], []),
            "embedding_model": row["embedding_model"],
            "embedding_version": row["embedding_version"],
            "attempts": row["attempts"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def log_vector_operation(
        self,
        object_id: str,
        object_type: str,
        action: str,
        *,
        operation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        status: str = OPERATION_PENDING,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a retryable log entry for every vector write/delete."""
        if action not in {"upsert", "delete"}:
            raise ValueError(f"unsupported vector action: {action}")
        new_id = operation_id or typed_id("operation")
        now = utc_timestamp()
        with self.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO vector_operations (
                    operation_id, object_id, object_type, action, payload_json,
                    status, attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    status = excluded.status,
                    last_error = excluded.last_error,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (new_id, object_id, object_type, action, _as_json(payload or {}), status, error, now, now),
            )
        return self.get_vector_operation(new_id) or {}

    def get_vector_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        row = self.connect().execute(
            "SELECT * FROM vector_operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return self._vector_operation_row(row) if row else None

    def mark_vector_operation(
        self,
        operation_id: str,
        *,
        status: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if status not in {OPERATION_PENDING, OPERATION_DONE, OPERATION_FAILED}:
            raise ValueError(f"unsupported vector operation status: {status}")
        with self.transaction() as transaction:
            transaction.execute(
                """
                UPDATE vector_operations SET status = ?, attempts = attempts + 1,
                    last_error = ?, updated_at = ?
                WHERE operation_id = ?
                """,
                (status, (str(error)[:2000] if error else None), utc_timestamp(), operation_id),
            )
        return self.get_vector_operation(operation_id) or {}

    def list_vector_operations(
        self, *, status: Optional[str] = None, object_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM vector_operations WHERE 1 = 1"
        params: List[Any] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if object_type is not None:
            query += " AND object_type = ?"
            params.append(object_type)
        query += " ORDER BY created_at, operation_id"
        return [
            self._vector_operation_row(row)
            for row in self.connect().execute(query, params).fetchall()
        ]

    @staticmethod
    def _vector_operation_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "operation_id": row["operation_id"],
            "object_id": row["object_id"],
            "object_type": row["object_type"],
            "action": row["action"],
            "payload": _from_json(row["payload_json"], {}),
            "status": row["status"],
            "attempts": row["attempts"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------- diagnostics

    def delete_role(self, role_uuid: str, *, object_ids: Sequence[str] = ()) -> Dict[str, int]:
        """Delete every row that belongs to one role's partition.

        Removing a role (a cancelled or failed document import, or an explicit
        cleanup) dropped its directory and its registry entry but left its SQLite
        bookkeeping behind: stale whitelist rows, event rows and embedding fingerprints
        for objects that no longer exist anywhere. ``nodes.role_uuid`` is nullable and
        ``relations`` is shared, so relations are deleted by the endpoint ids the caller
        removed from the topology rather than by role.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required to delete a role's rows")
        ids = [str(item) for item in object_ids if str(item)]
        removed: Dict[str, int] = {}
        with self.transaction() as connection:
            for table in (
                "events", "working_memory", "whitelist",
                "embedding_records", "embedding_rebuild_queue", "publication_operations",
            ):
                cursor = connection.execute(f"DELETE FROM {table} WHERE role_uuid = ?", [role])
                removed[table] = max(0, int(cursor.rowcount or 0))
            for start in range(0, len(ids), 400):
                part = ids[start:start + 400]
                marks = ",".join("?" for _ in part)
                for table, column in (
                    ("nodes", "id"), ("vector_operations", "object_id"),
                    ("relations", "source_id"), ("relations", "target_id"),
                ):
                    cursor = connection.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({marks})", part
                    )
                    removed[table] = removed.get(table, 0) + max(0, int(cursor.rowcount or 0))
        self.bump_revision()
        return removed

    def counts(self, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Active-object and pending-operation counts for the diagnostics command.

        Memory text is no longer a SQLite table: everything a role knows lives in
        the shared topology, so ``events``/``working_memory``/``whitelist`` are the
        only role-scoped tables left here.
        """
        connection = self.connect()

        def count(query: str, params: Sequence[Any]) -> int:
            row = connection.execute(query, list(params)).fetchone()
            return int(row["total"]) if row else 0

        role_clause = " AND role_uuid = ?" if role_uuid is not None else ""
        role_params: List[Any] = [role_uuid] if role_uuid is not None else []

        def role_count(table: str) -> int:
            return count(
                f"SELECT COUNT(*) AS total FROM {table} WHERE status = 'active'{role_clause}",
                role_params,
            )

        return {
            "events": role_count("events"),
            "working_memory": role_count("working_memory"),
            "relations": count(
                "SELECT COUNT(*) AS total FROM relations WHERE status = 'active'"
                " AND kind != 'event_link'",
                [],
            ),
            "event_links": count(
                "SELECT COUNT(*) AS total FROM relations WHERE status = 'active'"
                " AND kind = 'event_link'",
                [],
            ),
            "whitelist": role_count("whitelist"),
            "embedding_records": count("SELECT COUNT(*) AS total FROM embedding_records", []),
            "pending_vector_operations": count(
                "SELECT COUNT(*) AS total FROM vector_operations WHERE status != 'done'", []
            ),
            "pending_publications": count(
                "SELECT COUNT(*) AS total FROM publication_operations WHERE status != 'done'", []
            ),
            "orphan_events": self.count_orphan_events(),
            "missing_embeddings": self.count_missing_embeddings(role_uuid),
            "revision": self.revision(),
            "schema_version": SCHEMA_VERSION,
        }
