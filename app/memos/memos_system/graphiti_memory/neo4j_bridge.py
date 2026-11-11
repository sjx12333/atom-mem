"""Neo4j identity and lifecycle bridge built on vendored Graphiti Core."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from graphiti_core.driver.neo4j_driver import Neo4jDriver

from .async_loop import AsyncLoopRunner


class GraphitiNeo4jBridge:
    """Small synchronous facade over the reference async Neo4j driver.

    Existing MemOS routes are synchronous internally but run inside async
    FastAPI handlers. Every operation runs on ONE persistent background event
    loop (shared by all bridges in the process).

    Why not ``asyncio.run()`` per call: this Windows Python build lacks
    ``_socket.socketpair``, so every new event loop goes through the pure
    Python ``socket.socketpair`` fallback (listening socket + blocking
    ``accept``). Publication calls this bridge once per indexed object, i.e.
    thousands of short-lived loops per conversation; eventually one of those
    fallback ``accept`` calls never returns and the single Neo4j executor
    thread blocks forever, which stalls the whole extraction queue (observed
    2026-11-07: two batches stuck at "resolve_temporal" for 20+ minutes).
    """

    _runner: Optional[AsyncLoopRunner] = None

    def __init__(self, config: Dict[str, Any]):
        database = config.get("database", {})
        self.uri = str(database.get("neo4j_uri") or "")
        self.user = str(database.get("neo4j_user") or "")
        self.password = str(database.get("neo4j_password") or "")
        self.database = str(database.get("neo4j_database") or "neo4j")
        self.embedding_dimension = int(config.get("embedding", {}).get("dimension") or 1024)
        if GraphitiNeo4jBridge._runner is None:
            GraphitiNeo4jBridge._runner = AsyncLoopRunner()
        self._runner = GraphitiNeo4jBridge._runner

    def _run(self, operation):
        return self._runner.run(operation())

    async def _query(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        driver = Neo4jDriver(self.uri, self.user, self.password, database=self.database)
        try:
            return await driver.execute_query(cypher, params=params or {})
        finally:
            await driver.close()

    def execute(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        """Synchronous parameterized Cypher for projection adapters.

        Projection writes (event nodes, identity validation, residue reporting)
        go through this single method so every statement shares the same driver
        and event loop as the rest of the bridge.
        """
        return self._run(lambda: self._query(cypher, params))


    def health(self) -> Dict[str, Any]:
        return self._run(self._health_async)

    async def _health_async(self) -> Dict[str, Any]:
        result = await self._query("RETURN 1 AS ok")
        return {"status": "connected", "database": self.database, "rows": len(result.records)}

    def ensure_schema(self) -> Dict[str, Any]:
        return self._run(self._ensure_schema_async)

    async def _ensure_schema_async(self) -> Dict[str, Any]:
        driver = Neo4jDriver(self.uri, self.user, self.password, database=self.database)
        try:
            await driver.build_indices_and_constraints()
        finally:
            await driver.close()
        return {"status": "ready", "database": self.database, "source": "graphiti_core"}

    def ensure_user_node(self, role_uuid: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Create/refresh one role's own user identity node (ADR-0153).

        The identity is ``user:<role_uuid>`` in the same ``group_id`` as the role,
        so the role's extraction results and its identity node live in one
        partition and no other role's sweep can reach them.
        """
        role = str(role_uuid).strip()
        if not role:
            raise ValueError("role_uuid cannot be empty")
        return self._run(
            lambda: self._ensure_identity_async("User", role, display_name or role, role, prefix="user:")
        )

    def ensure_ai_node(self, role_uuid: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        role_uuid = str(role_uuid).strip()
        if not role_uuid:
            raise ValueError("role_uuid cannot be empty")
        return self._run(lambda: self._ensure_identity_async("AI", role_uuid, display_name or role_uuid, role_uuid))

    async def _ensure_identity_async(
        self,
        label: str,
        identity: str,
        name: str,
        group_id: str,
        *,
        prefix: str = "ai:",
    ) -> Dict[str, Any]:
        await self._ensure_schema_async()
        uuid = f"{prefix}{identity}"
        driver = Neo4jDriver(self.uri, self.user, self.password, database=self.database)
        try:
            # 身份节点没有语义向量（name_embedding=None），不能走
            # entity_node_ops.save —— 其 Neo4j 实现无条件调用
            # setNodeVectorProperty，null 向量会抛 "vector must not be null"。
            # 直接 MERGE 写入节点（保留 Entity + 类型标签），幂等且不建向量属性。
            await driver.execute_query(
                f"MERGE (n:Entity {{uuid: $uuid}}) SET n:{label}:Entity, "
                "n.name = $name, n.group_id = $group_id, n.kind = $kind "
                "RETURN n.uuid AS uuid",
                params={
                    "uuid": uuid,
                    "name": name,
                    "group_id": group_id,
                    "kind": "user" if label == "User" else "ai_role",
                },
            )
            result = await driver.execute_query(
                "MATCH (n:Entity {uuid: $uuid}) "
                "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.group_id AS group_id",
                params={"uuid": uuid},
            )
            row = result.records[0].data() if result.records else {"uuid": uuid, "name": name, "labels": ["Entity", label], "group_id": group_id}
        finally:
            await driver.close()
        return {"status": "created_or_existing", **row}

    def close(self) -> None:
        # The event loop is process-wide and shared by every bridge instance;
        # shutting it down here would break any other live bridge.
        return None
