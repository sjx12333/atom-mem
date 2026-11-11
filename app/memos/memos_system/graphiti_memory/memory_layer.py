"""Factory for the SQLite authority plus the Neo4j-native vector index.

The service composes the online memory layer through this module so the storage
boundaries are declared in exactly one place:

- SQLite (:class:`graphiti_memory.sqlite_store.SqliteMemoryStore`) keeps the
  rebuildable bookkeeping: the event bodies, the conversation audit trail and the
  ``embedding_records`` cache that says which ``(role, object)`` pair needs its
  vector recomputed;
- Neo4j holds the memory itself — every retrieval object's attributes *and* its
  vector, one vector per role (see :mod:`graphiti_memory.vector_store`). There is
  no separate vector database any more: a second store would duplicate the object
  identity and force the two to be kept in sync;
- an unavailable Neo4j is reported as an index failure, so recall can surface it
  instead of pretending a zero recall.

Construction is defensive: an unusable Neo4j configuration yields a store with
``available == False`` and an explicit ``error`` instead of raising during service
startup, because entity/relation publication must keep working without vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .sqlite_store import SqliteMemoryStore
from .vector_store import GraphitiVectorStore


def sqlite_path_for(root: Path | str, config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the SQLite authority file for one deployment root.

    A configured ``memory.sqlite_path`` is resolved **relative to the given
    root**, so an operations run against ``--data-root X`` can never touch the
    default deployment database.
    """
    memory = (config or {}).get("memory", {}) or {}
    configured = str(memory.get("sqlite_path") or "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        # Strip the conventional data-root prefix so the configured value stays
        # meaningful for both the default root and an explicit --data-root.
        parts = candidate.parts
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "graphiti":
            candidate = Path(*parts[2:])
        if not candidate.parts:
            candidate = Path("memory.sqlite3")
        return Path(root) / candidate
    return Path(root) / "memory.sqlite3"


def vector_index_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Vector settings that still matter now that the index lives in Neo4j."""
    vector = (config or {}).get("vector", {}) or {}
    embedding = (config or {}).get("embedding", {}) or {}
    dimension = vector.get("dimension") or embedding.get("dimension") or 1024
    return {
        "collection": str(vector.get("collection") or "graphiti_memory"),
        "dimension": int(dimension),
        "distance": str(vector.get("distance") or "COSINE"),
    }


class MemoryLayer:
    """Bundle of the SQLite bookkeeping store and the Neo4j vector index."""

    def __init__(
        self,
        store: SqliteMemoryStore,
        vectors: Optional[GraphitiVectorStore],
        *,
        vector_error: Optional[str] = None,
        client: Any = None,
        owns_bridge: bool = False,
    ):
        self.store = store
        self.vectors = vectors
        self.vector_error = vector_error
        #: The Neo4j bridge backing the vector index, when one was built here.
        self.client = client
        #: True when this layer created the bridge and must close it.
        self.owns_bridge = owns_bridge

    @property
    def vectors_available(self) -> bool:
        return self.vectors is not None and self.vectors.is_available()

    def require_vectors(self) -> GraphitiVectorStore:
        """Return the vector store or raise an explicit index failure."""
        from .vector_store import VectorIndexError

        if not self.vectors_available:
            raise VectorIndexError(
                "Neo4j vector index is unavailable"
                + (f": {self.vector_error}" if self.vector_error else "")
            )
        assert self.vectors is not None
        return self.vectors

    def status(self) -> Dict[str, Any]:
        return {
            "sqlite_path": str(self.store.path),
            "sqlite_schema_version": self.store.counts().get("schema_version"),
            "revision": self.store.revision(),
            "vectors_available": self.vectors_available,
            "vector_backend": "neo4j",
            "vector_collection": getattr(self.vectors, "collection_name", None),
            "vector_error": self.vector_error,
        }

    def drop_vector_collection(self) -> Dict[str, Any]:
        """Delete every role's memory vectors from Neo4j."""
        vectors = self.require_vectors()
        return vectors.drop_collection()

    def close(self) -> None:
        """Release the SQLite connection and this layer's Neo4j bridge."""
        self.store.close()
        if self.client is not None and getattr(self, "owns_bridge", False):
            close = getattr(self.client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - shutdown is best effort
                    pass
        self.client = None


def build_memory_layer(
    root: Path | str,
    config: Optional[Dict[str, Any]] = None,
    *,
    vector_store: Any = None,
    neo4j_bridge: Any = None,
    sqlite_path: Optional[Path | str] = None,
) -> MemoryLayer:
    """Build the online memory layer for one deployment root.

    ``vector_store`` lets tests/diagnostics inject a fake index; by default a
    :class:`GraphitiVectorStore` over a :class:`GraphitiNeo4jBridge` for the
    configured database is created.
    """
    resolved_sqlite = Path(sqlite_path) if sqlite_path else sqlite_path_for(root, config)
    store = SqliteMemoryStore(
        resolved_sqlite,
        busy_timeout_ms=int(((config or {}).get("memory", {}) or {}).get("busy_timeout_ms", 10000)),
    )
    settings = vector_index_settings(config)
    bridge = neo4j_bridge
    error: Optional[str] = None
    owns_bridge = False
    if vector_store is None and bridge is None:
        try:
            from .neo4j_bridge import GraphitiNeo4jBridge

            bridge = GraphitiNeo4jBridge(config or {})
            owns_bridge = True
        except Exception as exc:  # noqa: BLE001 - vector index stays optional
            bridge = None
            error = f"{type(exc).__name__}: {exc}"
    vectors: Optional[GraphitiVectorStore] = None
    if vector_store is not None:
        vectors = vector_store
    elif bridge is not None:
        vectors = GraphitiVectorStore(
            bridge,
            collection_name=settings["collection"],
            vector_size=settings["dimension"],
            distance=settings["distance"],
        )
        try:
            vectors.ensure_collection()
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    return MemoryLayer(
        store, vectors, vector_error=error, client=bridge, owns_bridge=owns_bridge
    )
