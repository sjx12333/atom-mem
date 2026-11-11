"""Neo4j projection adapter for shared topology and event nodes.

Neo4j is a graph query/display projection, never the authority for event bodies
or cognition (plan 最终不变式 2). This adapter therefore:

- writes an event with its **stable event id** (``event:<uuid>``), never relying
  on a Neo4j node merge to decide event uniqueness;
- stores summary/time/vector attributes on the event node;
- creates the event's single non-semantic ownership edge to its AI role node and
  refuses to attach an event to any other node;
- keeps the default user/AI identity nodes and companion edge intact and can
  report non-default residue for the cleanup command.

All Cypher uses parameters; no user input is concatenated into a statement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Identity constants mirror roles.py / saga.py / neo4j_bridge.py.
AI_NODE_KIND = "ai_role"
EVENT_NODE_KIND = "event"
COMPANION_EDGE_KIND = "companion"


def event_node_uuid(event_id: str) -> str:
    """Stable projection id for one event; one event is always one Neo4j node."""
    return f"event:{event_id}"


def ai_node_uuid(role_uuid: str) -> str:
    return f"ai:{role_uuid}"


def user_node_uuid(role_uuid: str) -> str:
    """The Neo4j uuid of one role's own user node (ADR-0153).

    ``user:<role_uuid>`` — the same identity the topology uses, so the graph
    projection and the topology never disagree about who "the user" is in a role's
    partition.
    """
    return f"user:{role_uuid}"


class Neo4jEventProjection:
    """Synchronous facade over the shared Graphiti Neo4j bridge."""

    def __init__(self, bridge: Any):
        self.bridge = bridge

    # ------------------------------------------------------------------ events

    def project_event(
        self,
        *,
        event_id: str,
        role_uuid: str,
        summary: str,
        occurred_at: Optional[str] = None,
        formed_at: Optional[str] = None,
        source_kind: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
        source_event_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Write/refresh one event node plus its ownership edge. Idempotent."""
        if not event_id or not role_uuid:
            raise ValueError("event_id and role_uuid are required")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("event summary cannot be empty")
        node_uuid = event_node_uuid(event_id)
        role_node = ai_node_uuid(role_uuid)
        statement = (
            "MERGE (n:Event {uuid: $uuid}) "
            "SET n.kind = $kind, n.name = $name, n.group_id = $group_id, "
            "n.role_uuid = $role_uuid, n.summary = $summary, "
            "n.occurred_at = $occurred_at, n.formed_at = $formed_at, "
            "n.source_kind = $source_kind, n.source_conversation_id = $conversation_id, "
            "n.content_hash = $content_hash, n.embedding_model = $embedding_model, "
            "n.embedding_version = $embedding_version, n.source_event_ids = $source_event_ids "
            "WITH n "
            "MERGE (r:Entity:AI {uuid: $role_node}) "
            "SET r.kind = 'ai_role', r.group_id = $group_id "
            "MERGE (n)-[e:EVENT_OF]->(r) "
            "SET e.kind = 'event_link' "
            "RETURN n.uuid AS uuid, r.uuid AS role_uuid, type(e) AS link"
        )
        result = self.bridge.execute(
            statement,
            {
                "uuid": node_uuid,
                "kind": EVENT_NODE_KIND,
                "name": summary.strip(),
                "group_id": role_uuid,
                "role_uuid": role_node,
                "role_node": role_node,
                "summary": summary.strip(),
                "occurred_at": occurred_at,
                "formed_at": formed_at,
                "source_kind": source_kind,
                "conversation_id": source_conversation_id,
                "content_hash": content_hash,
                "embedding_model": embedding_model,
                "embedding_version": embedding_version,
                "source_event_ids": list(source_event_ids or []),
            },
        )
        return {
            "status": "projected",
            "event_id": event_id,
            "node_uuid": node_uuid,
            "role_uuid": role_uuid,
            "rows": len(getattr(result, "records", []) or []),
        }

    def delete_event(self, event_id: str) -> Dict[str, Any]:
        node_uuid = event_node_uuid(event_id)
        result = self.bridge.execute(
            "MATCH (n:Event {uuid: $uuid}) DETACH DELETE n RETURN count(*) AS removed",
            {"uuid": node_uuid},
        )
        removed = 0
        records = getattr(result, "records", None) or []
        if records:
            try:
                removed = int(records[0].data().get("removed", 0))
            except Exception:  # noqa: BLE001 - projection diagnostics only
                removed = len(records)
        return {"status": "deleted", "event_id": event_id, "removed": removed}

    def event_projection(self, event_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Read back projected events, optionally restricted to stable ids."""
        ids = [event_node_uuid(str(item)) for item in (event_ids or [])]
        if ids:
            statement = (
                "MATCH (n:Event) WHERE n.uuid IN $uuids "
                "OPTIONAL MATCH (n)-[:EVENT_OF]->(r) "
                "RETURN n.uuid AS uuid, n.role_uuid AS role_uuid, n.summary AS summary, "
                "n.occurred_at AS occurred_at, n.formed_at AS formed_at, "
                "n.source_kind AS source_kind, r.uuid AS linked_role "
                "ORDER BY n.uuid"
            )
            result = self.bridge.execute(statement, {"uuids": ids})
        else:
            statement = (
                "MATCH (n:Event) "
                "OPTIONAL MATCH (n)-[:EVENT_OF]->(r) "
                "RETURN n.uuid AS uuid, n.role_uuid AS role_uuid, n.summary AS summary, "
                "n.occurred_at AS occurred_at, n.formed_at AS formed_at, "
                "n.source_kind AS source_kind, r.uuid AS linked_role "
                "ORDER BY n.uuid"
            )
            result = self.bridge.execute(statement, {})
        rows = [record.data() for record in (getattr(result, "records", None) or [])]
        valid_role_nodes = {ai_node_uuid(str(row.get("linked_role", "")).removeprefix("ai:")) for row in rows}
        return [
            {**row, "object_type": EVENT_NODE_KIND, "linked_role_valid": row.get("linked_role") in valid_role_nodes}
            for row in rows
        ]

    def delete_events_without_role(self) -> Dict[str, Any]:
        """Remove projected events that lost their ownership edge (repair)."""
        result = self.bridge.execute(
            "MATCH (n:Event) WHERE NOT (n)-[:EVENT_OF]->(:Entity) "
            "DETACH DELETE n RETURN count(*) AS removed",
            {},
        )
        records = getattr(result, "records", None) or []
        removed = 0
        if records:
            try:
                removed = int(records[0].data().get("removed", 0))
            except Exception:  # noqa: BLE001
                removed = len(records)
        return {"status": "repaired", "removed": removed}

    # --------------------------------------------------------------- identity

    def validate_defaults(self) -> Dict[str, Any]:
        """Report the per-role identity graph: one user node and one AI node each.

        The AI role nodes are found by their ``AI`` label rather than by a
        ``kind='ai_role'`` property: the label is what both writers set (the identity
        bridge and the event projection), while the property is only written by the
        projection, so a role that has never published an event would be missed.
        """
        result = self.bridge.execute(
            "MATCH (r:Entity:AI) RETURN collect(r.uuid) AS roles",
            {},
        )
        records = getattr(result, "records", None) or []
        ai_roles = records[0].data().get("roles", []) if records else []
        users = self.bridge.execute(
            "MATCH (u:Entity) WHERE u.uuid STARTS WITH 'user:' "
            "RETURN collect(u.uuid) AS users",
            {},
        )
        user_records = getattr(users, "records", None) or []
        user_nodes = user_records[0].data().get("users", []) if user_records else []
        links = self.bridge.execute(
            "MATCH (u:Entity)-[c:COMPANION]->(r:Entity) WHERE u.uuid STARTS WITH 'user:' "
            "RETURN count(DISTINCT c) AS edges",
            {},
        )
        link_records = getattr(links, "records", None) or []
        companion_edges = int(link_records[0].data().get("edges", 0)) if link_records else 0
        # Every AI role must own exactly one user node, and every user node must
        # belong to exactly one role: an unpaired identity means a role's partition
        # was never created (or was pruned by an old single-user cleanup).
        role_ids = [str(item).removeprefix("ai:") for item in ai_roles]
        expected = {user_node_uuid(role) for role in role_ids}
        return {
            "user_nodes": sorted(str(item) for item in user_nodes),
            "ai_roles": sorted(role_ids),
            "companion_edges": companion_edges,
            "missing_user_nodes": sorted(expected - {str(item) for item in user_nodes}),
            "orphan_user_nodes": sorted(
                str(item) for item in user_nodes if str(item) not in expected
            ),
            "valid": bool(role_ids) and expected.issubset({str(item) for item in user_nodes}),
        }

    def non_default_residue(self) -> Dict[str, Any]:
        """Count projection objects outside the per-role identity graph."""
        result = self.bridge.execute(
            "MATCH (n:Entity) WHERE NOT n:Event AND NOT n:AI "
            "AND NOT n.uuid STARTS WITH 'user:' "
            "RETURN count(n) AS nodes",
            {},
        )
        records = getattr(result, "records", None) or []
        nodes = int(records[0].data().get("nodes", 0)) if records else 0
        events = self.bridge.execute("MATCH (n:Event) RETURN count(n) AS events", {})
        event_records = getattr(events, "records", None) or []
        event_count = int(event_records[0].data().get("events", 0)) if event_records else 0
        return {"non_default_nodes": nodes, "event_nodes": event_count}
