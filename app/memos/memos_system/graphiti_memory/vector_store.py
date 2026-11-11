"""Neo4j-native vector index for the Graphiti memory layer.

**Why this replaced Qdrant.** Every retrieval object is a node or a relation of the
shared graph, and a role's memory about it lives in that record's attributes. A
separate vector database therefore duplicated the object identity in a second
store that had to be kept in sync, and its point id was derived from the object id
alone — so two roles describing one shared node overwrote each other. Vectors now
live **inside Neo4j**, next to the object they describe:

* one ``:MemoryVector`` node per ``(role_uuid, object_id)`` pair, so the same
  object carries a different vector for every role and nothing collides;
* the vector itself is written with ``db.create.setNodeVectorProperty`` — the same
  procedure ``graphiti_core`` uses for ``Entity.name_embedding``
  (``graphiti_core/models/nodes/node_db_queries.py``), available in Neo4j
  Community, so no enterprise-only ``CREATE VECTOR INDEX`` is needed;
* similarity is computed **inside Neo4j** by ``vector.similarity.cosine``, so a
  recall moves object ids and scores instead of every stored vector. The function
  answers in ``[0, 1]`` — measured on this deployment: identical vectors ``1.0``,
  orthogonal ``0.5``, opposite ``0.0``, i.e. ``(cosine + 1) / 2`` — and is mapped
  back to the ``[-1, 1]`` convention the recall policy's thresholds are stated in;
* a bridge that cannot score in Cypher (the in-process stand-in the tests use, or a
  server that rejects the function) falls back to scoring the fetched vectors in
  Python. ``last_scoring_backend`` records which of the two answered;
* no ANN index is used. On Neo4j 5.26 ``db.index.vector.queryNodes`` accepts only
  ``(index, k, query)`` — measured: the filtered overload does not exist — so an
  index query cannot be scoped to one role's partition, and post-filtering a global
  top-k would silently drop that role's own nearest neighbours. A role-scoped scan
  is both correct and small;
* ``memory_embedding`` (a single shared property name) is kept so the vector can
  be read back with a plain ``RETURN``.

``content_hash`` and the embedding fingerprint stay the caller's concern (they are
derived, rebuildable bookkeeping in SQLite), exactly as they were before.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Object types the index may hold. ``node``/``relation``/``memory`` are the
#: node-attribute memory objects; ``event`` is the independent memory unit
#: (a diary-derived memory node); ``predicate`` is a relation's shared subject and
#: ``hook`` a model-predicted association, both of which must be indexable or they
#: exist in the topology while staying invisible to recall.
#:
#: This tuple is a whitelist: ``upsert()`` rejects a type outside it and the search
#: filters on it, so a new node kind that is not added here is silently unsearchable.
OBJECT_TYPES = ("node", "relation", "memory", "event", "fact", "predicate", "hook")

#: Neo4j label of the vector carrier node.
VECTOR_LABEL = "MemoryVector"
#: Neo4j property holding the embedding.
VECTOR_PROPERTY = "memory_embedding"
#: Object id prefixes that must be addressable in Cypher without a lookup table.
_POINT_PREFIX = "mv"


class VectorIndexError(RuntimeError):
    """Raised when the vector index cannot serve a request.

    Recall must surface this as an index failure instead of degrading into an
    empty, "successful-looking" zero recall.
    """


def role_key(role_uuid: str) -> str:
    """A Cypher-safe per-role discriminator (role UUIDs may hold ``-``/``_``)."""
    candidate = str(role_uuid or "").strip()
    if not candidate:
        raise ValueError("role_uuid is required for a vector point")
    return hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:20]


def point_id_for(role_uuid: str, object_id: str) -> str:
    """Stable vector-node id for one role's copy of one object.

    Role-scoped on purpose: two roles describing the same shared node must hold
    two independent vectors, and the old ``point_id_for(object_id)`` made that
    impossible (the second writer silently replaced the first).
    """
    role = str(role_uuid or "").strip()
    obj = str(object_id or "").strip()
    if not role or not obj:
        raise ValueError("role_uuid and object_id are both required for a vector point")
    digest = hashlib.sha1(f"{role}\x1f{obj}".encode("utf-8")).hexdigest()
    return f"{_POINT_PREFIX}_{digest}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, tolerating zero vectors instead of dividing by zero."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for index in range(size):
        a = float(left[index])
        b = float(right[index])
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / ((norm_left ** 0.5) * (norm_right ** 0.5))


class GraphitiVectorStore:
    """Thin, defensive wrapper over Neo4j for one role's memory vectors."""

    def __init__(
        self,
        bridge: Any,
        *,
        collection_name: str = "graphiti_memory",
        vector_size: int = 1024,
        distance: str = "COSINE",
    ):
        self.bridge = bridge
        self.collection_name = collection_name
        self.vector_size = int(vector_size)
        self.distance = distance
        self._ready = False
        #: Which path answered the last ``search``: ``"neo4j"`` when the server
        #: scored the query, ``"caller"`` when the fetched vectors were scored here.
        #: Read by the probes and by diagnose; nothing branches on it.
        self.last_scoring_backend: Optional[str] = None

    # ------------------------------------------------------------------- setup

    def is_available(self) -> bool:
        return self.bridge is not None

    def _execute(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if self.bridge is None:
            raise VectorIndexError("Neo4j vector bridge is not configured")
        try:
            return self.bridge.execute(cypher, params or {})
        except VectorIndexError:
            raise
        except Exception as exc:  # noqa: BLE001 - becomes an explicit index failure
            raise VectorIndexError(f"neo4j vector query failed: {exc}") from exc

    @staticmethod
    def _records(result: Any) -> List[Dict[str, Any]]:
        return [dict(record) for record in (getattr(result, "records", None) or [])]

    def ensure_collection(self) -> Dict[str, Any]:
        """Create the supporting index if missing (idempotent)."""
        self._execute(
            f"CREATE INDEX memory_vector_object IF NOT EXISTS "
            f"FOR (n:{VECTOR_LABEL}) ON (n.object_id)"
        )
        self._ready = True
        return {
            "collection": self.collection_name,
            "created": False,
            "backend": "neo4j",
            "vector_size": self.vector_size,
            "label": VECTOR_LABEL,
        }

    def drop_collection(self, **_kwargs: Any) -> Dict[str, Any]:
        """Delete every vector node (used by reset and by the rebuild command)."""
        result = self._execute(f"MATCH (n:{VECTOR_LABEL}) DETACH DELETE n RETURN count(*) AS removed")
        removed = self._records(result)
        return {
            "mode": "neo4j_delete",
            "storage_reset": False,
            "count": int(removed[0].get("removed", 0)) if removed else 0,
        }

    def payload_index_status(self) -> Dict[str, Any]:
        required = ["role_uuid", "object_id"]
        result = self._execute(
            f"MATCH (n:{VECTOR_LABEL}) RETURN count(n) AS total"
        )
        records = self._records(result)
        return {
            "collection": self.collection_name,
            "backend": "neo4j",
            "required": required,
            "present": required,
            "missing": [],
            "ready": self.is_available(),
            "count": int(records[0].get("total", 0)) if records else 0,
        }

    def count(self, role_uuid: Optional[str] = None, **payload_filters: Any) -> int:
        if role_uuid:
            result = self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{role_uuid: $role}}) RETURN count(n) AS total",
                {"role": str(role_uuid)},
            )
        else:
            result = self._execute(f"MATCH (n:{VECTOR_LABEL}) RETURN count(n) AS total")
        records = self._records(result)
        return int(records[0].get("total", 0)) if records else 0

    # -------------------------------------------------------------- write paths

    def upsert(
        self,
        object_id: str,
        vector: Sequence[float],
        *,
        object_type: str,
        role_uuid: str,
        content_hash: Optional[str] = None,
        status: str = "active",
        source_conversation_id: Optional[str] = None,
        content: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Idempotently write one role's vector for one object."""
        if object_type not in OBJECT_TYPES:
            raise ValueError(f"unsupported object_type: {object_type}")
        if not vector:
            raise VectorIndexError(f"refusing to index an empty vector for {object_id}")
        point_id = point_id_for(role_uuid, object_id)
        payload: Dict[str, Any] = {
            "object_id": str(object_id),
            "object_type": object_type,
            "role_uuid": str(role_uuid),
            "status": status,
            "content_hash": content_hash,
            "source_conversation_id": source_conversation_id,
            "role_key": role_key(role_uuid),
        }
        if content is not None:
            payload["content"] = content
        if extra:
            payload.update(extra)
        self._execute(
            f"MERGE (n:{VECTOR_LABEL} {{id: $id}}) "
            "SET n += $payload "
            "WITH n CALL db.create.setNodeVectorProperty(n, $property, $vector) "
            "RETURN n.id AS id",
            {
                "id": point_id,
                "payload": payload,
                "property": VECTOR_PROPERTY,
                "vector": [float(item) for item in vector],
            },
        )
        return {"point_id": point_id, "object_id": str(object_id), "object_type": object_type}

    def delete_role(self, role_uuid: str) -> Dict[str, Any]:
        """Delete every vector point belonging to one role's partition.

        One vector per ``(role_uuid, object_id)`` means dropping a role has to drop its
        points by role: the objects they describe only exist inside that partition
        (topology objects are shared, but a vector is always role-scoped), and leaving
        them behind would keep a deleted role's text in the recall index.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required to delete a role's vectors")
        before = self.count(role_uuid=role)
        self._execute(
            f"MATCH (n:{VECTOR_LABEL} {{role_uuid: $role}}) DETACH DELETE n",
            {"role": role},
        )
        return {"role_uuid": role, "deleted": before}

    def delete(self, object_id: str, *, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Delete one object's vector — for one role, or for every role."""
        if role_uuid:
            self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id, role_uuid: $role}}) "
                "DETACH DELETE n",
                {"object_id": str(object_id), "role": str(role_uuid)},
            )
        else:
            self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id}}) DETACH DELETE n",
                {"object_id": str(object_id)},
            )
        return {"object_id": str(object_id), "role_uuid": role_uuid, "status": "deleted"}

    def set_status(
        self, object_id: str, status: str, *, role_uuid: Optional[str] = None
    ) -> Dict[str, Any]:
        if role_uuid:
            self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id, role_uuid: $role}}) "
                "SET n.status = $status",
                {"object_id": str(object_id), "role": str(role_uuid), "status": status},
            )
        else:
            self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id}}) SET n.status = $status",
                {"object_id": str(object_id), "status": status},
            )
        return {"object_id": str(object_id), "role_uuid": role_uuid, "status": status}

    def retrieve_payload(self, object_id: str, *, role_uuid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if role_uuid:
            result = self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id, role_uuid: $role}}) "
                "RETURN properties(n) AS payload LIMIT 1",
                {"object_id": str(object_id), "role": str(role_uuid)},
            )
        else:
            result = self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id}}) "
                "RETURN properties(n) AS payload LIMIT 1",
                {"object_id": str(object_id)},
            )
        records = self._records(result)
        if not records:
            return None
        payload = records[0].get("payload")
        if isinstance(payload, dict):
            payload.pop(VECTOR_PROPERTY, None)
            return payload
        return None

    def retrieve_vector(self, object_id: str, *, role_uuid: Optional[str] = None) -> Optional[List[float]]:
        if role_uuid:
            result = self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id, role_uuid: $role}}) "
                f"RETURN n.{VECTOR_PROPERTY} AS vector LIMIT 1",
                {"object_id": str(object_id), "role": str(role_uuid)},
            )
        else:
            result = self._execute(
                f"MATCH (n:{VECTOR_LABEL} {{object_id: $object_id}}) "
                f"RETURN n.{VECTOR_PROPERTY} AS vector LIMIT 1",
                {"object_id": str(object_id)},
            )
        records = self._records(result)
        if not records:
            return None
        vector = records[0].get("vector")
        if vector is None:
            return None
        return [float(item) for item in vector]

    # --------------------------------------------------------------- read path

    def list_vectors(
        self,
        *,
        role_uuid: str,
        object_types: Optional[Iterable[str]] = None,
        status: str = "active",
    ) -> List[Dict[str, Any]]:
        """Every vector this role holds, with the vector itself.

        This is the candidate source for node-attribute recall: the scope is one
        role, so nothing outside the role can be scored, and the caller already
        holds the role's whitelist as a second boundary.
        """
        types = [item for item in (object_types or OBJECT_TYPES) if item in OBJECT_TYPES]
        result = self._execute(
            f"MATCH (n:{VECTOR_LABEL}) "
            "WHERE n.role_uuid = $role AND n.status = $status AND n.object_type IN $types "
            "RETURN n.object_id AS object_id, n.object_type AS object_type, "
            "n.content_hash AS content_hash, n.source_conversation_id AS source_conversation_id, "
            f"n.{VECTOR_PROPERTY} AS vector",
            {"role": str(role_uuid), "status": status, "types": types},
        )
        hits: List[Dict[str, Any]] = []
        for record in self._records(result):
            vector = record.get("vector")
            hits.append(
                {
                    "object_id": str(record.get("object_id") or ""),
                    "object_type": record.get("object_type"),
                    "role_uuid": str(role_uuid),
                    "status": status,
                    "content_hash": record.get("content_hash"),
                    "source_conversation_id": record.get("source_conversation_id"),
                    "vector": [float(item) for item in vector] if vector else [],
                }
            )
        return hits

    def search(
        self,
        query_vector: Sequence[float],
        *,
        role_uuid: str,
        object_types: Optional[Iterable[str]] = None,
        top_k: int = 50,
        score_threshold: float = 0.0,
        status: str = "active",
    ) -> List[Dict[str, Any]]:
        """Score one query against the role's own vectors, best first.

        Scoring runs **inside Neo4j** (``vector.similarity.cosine``), so only an
        object's identity and its score cross the wire instead of every stored
        vector. The scope is still exactly one role: the query filters on
        ``role_uuid`` before scoring, so nothing outside the partition can be a
        candidate, and the caller holds the role's whitelist as a second boundary.

        A server that cannot evaluate the function — or a bridge that answers the
        Cypher without computing scores, which is what the in-process stand-in does
        — falls back to scoring the fetched vectors in Python. Both paths return the
        same convention, so a caller cannot tell them apart except through
        ``last_scoring_backend``.

        ``score_threshold`` (a **cosine**, the convention the recall policy is stated
        in) and ``top_k`` are pushed into Cypher as well, so rows below the floor and
        rows past the cut never cross the wire. The conversion into Neo4j's ``[0, 1]``
        similarity scale is in ``_score_in_neo4j`` — see the warning there.
        """
        vector = [float(item) for item in (query_vector or [])]
        if not vector:
            return []
        types = [item for item in (object_types or OBJECT_TYPES) if item in OBJECT_TYPES]
        try:
            hits = self._score_in_neo4j(
                vector,
                role_uuid=role_uuid,
                types=types,
                status=status,
                score_threshold=score_threshold,
                limit=top_k,
            )
            self.last_scoring_backend = "neo4j"
        except VectorIndexError:
            # Either the deployment cannot score vectors in Cypher or the bridge is
            # not a real server. Scoring the role's own vectors here is the same
            # answer with more traffic, never a different one.
            hits = self._score_in_python(
                vector,
                role_uuid=role_uuid,
                types=types,
                status=status,
                score_threshold=score_threshold,
            )
            self.last_scoring_backend = "caller"
        hits.sort(key=lambda item: (-float(item["score"]), str(item.get("object_id") or "")))
        return hits[: max(0, int(top_k))]

    @staticmethod
    def _score_from_similarity(similarity: Any) -> float:
        """Map Neo4j's ``[0, 1]`` similarity back to a ``[-1, 1]`` cosine.

        ``vector.similarity.cosine`` answers in ``[0, 1]`` (measured: identical
        vectors ``1.0``, orthogonal ``0.5``, opposite ``0.0``). The recall policy's
        thresholds — ``semantic_threshold``, the relative floor — were calibrated
        against a plain cosine, so the value is converted here and a threshold
        keeps meaning exactly what it meant when the caller did the arithmetic.
        """
        return 2.0 * float(similarity) - 1.0

    @staticmethod
    def _similarity_floor(score_threshold: float) -> float:
        """Convert a **cosine** floor into Neo4j's ``[0, 1]`` similarity scale.

        ⚠️ The unit trap: ``_score_from_similarity`` maps Neo4j's ``[0, 1]`` back to
        ``[-1, 1]`` (``2s - 1``), and the retrieval policy is stated in the mapped
        scale (``semantic_threshold = 0.25`` means *cosine* 0.25). Writing
        ``WHERE similarity >= 0.25`` would therefore mean cosine >= -0.5, i.e. admit
        almost everything — silently, and with no error anywhere. The floor is
        converted here, once, so callers keep thinking in cosine.
        """
        cosine = float(score_threshold)
        if cosine <= -1.0:
            return 0.0
        if cosine >= 1.0:
            return 1.0
        return (cosine + 1.0) / 2.0

    def _score_in_neo4j(
        self,
        query_vector: Sequence[float],
        *,
        role_uuid: str,
        types: Sequence[str],
        status: str,
        score_threshold: float,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        # The floor and the cut are applied inside the database: this store has no
        # ANN index (see the module docstring), so the scan is O(role's vectors) and
        # the only way to keep that cost off the wire is to drop rows in Cypher.
        statement = (
            f"MATCH (n:{VECTOR_LABEL}) "
            "WHERE n.role_uuid = $role AND n.status = $status AND n.object_type IN $types "
            f"AND n.{VECTOR_PROPERTY} IS NOT NULL "
            f"WITH n, vector.similarity.cosine(n.{VECTOR_PROPERTY}, $query) AS similarity "
            "WHERE similarity >= $similarity_floor "
            "RETURN n.object_id AS object_id, n.object_type AS object_type, "
            "n.content_hash AS content_hash, "
            "n.source_conversation_id AS source_conversation_id, similarity "
            "ORDER BY similarity DESC"
        )
        bound = max(0, int(limit))
        if bound:
            statement += " LIMIT $limit"
        result = self._execute(
            statement,
            {
                "role": str(role_uuid),
                "status": status,
                "types": list(types),
                "query": [float(item) for item in query_vector],
                "similarity_floor": self._similarity_floor(score_threshold),
                "limit": bound,
            },
        )
        hits: List[Dict[str, Any]] = []
        for record in self._records(result):
            if "similarity" not in record:
                # The bridge answered the shape without the score: this is not a
                # Neo4j answer, and treating it as one would silently return an
                # empty (or arbitrarily ordered) candidate set.
                raise VectorIndexError("vector query returned no similarity column")
            score = self._score_from_similarity(record.get("similarity") or 0.0)
            if score_threshold and score < float(score_threshold):
                continue
            object_id = str(record.get("object_id") or "")
            if not object_id:
                continue
            hits.append(
                {
                    "object_id": object_id,
                    "object_type": record.get("object_type"),
                    "role_uuid": str(role_uuid),
                    "status": status,
                    "content_hash": record.get("content_hash"),
                    "source_conversation_id": record.get("source_conversation_id"),
                    "score": score,
                    "point_id": point_id_for(role_uuid, object_id),
                }
            )
        return hits

    def _score_in_python(
        self,
        query_vector: Sequence[float],
        *,
        role_uuid: str,
        types: Sequence[str],
        status: str,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for row in self.list_vectors(role_uuid=role_uuid, object_types=types, status=status):
            vector = row.pop("vector", None)
            if not vector:
                continue
            score = _cosine(query_vector, vector)
            if score_threshold and score < float(score_threshold):
                continue
            hits.append({**row, "score": score, "point_id": point_id_for(role_uuid, row["object_id"])})
        return hits

    def rebuild_from(
        self,
        records: Iterable[Dict[str, Any]],
        embedder,
        *,
        object_type: str,
        role_uuid_field: str = "role_uuid",
    ) -> Dict[str, Any]:
        """Recreate vectors for records that carry text, using fresh embeddings."""
        written = 0
        failed: List[Dict[str, Any]] = []
        for record in records:
            object_id = str(record.get("id") or "")
            text = str(record.get("content") or record.get("summary") or record.get("text") or "")
            if not object_id or not text:
                continue
            role_uuid = str(record.get(role_uuid_field) or "")
            if not role_uuid:
                failed.append({"object_id": object_id, "error": "record has no role_uuid"})
                continue
            try:
                vector = embedder(text)
                if vector is None:
                    failed.append({"object_id": object_id, "error": "embedder returned no vector"})
                    continue
                self.upsert(
                    object_id,
                    vector,
                    object_type=object_type,
                    role_uuid=role_uuid,
                    content_hash=record.get("content_hash"),
                    status=record.get("status", "active"),
                    source_conversation_id=record.get("source_conversation_id"),
                    content=text,
                )
                written += 1
            except Exception as exc:  # noqa: BLE001 - collect per-record failures
                failed.append({"object_id": object_id, "error": str(exc)})
        return {"written": written, "failed": failed}
