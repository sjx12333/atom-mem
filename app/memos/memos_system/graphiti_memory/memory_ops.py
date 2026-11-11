"""Operations commands: init, rebuild, validate, diagnose and reset.

Every command returns an auditable report and never performs an unvalidated
broad ``DELETE`` (plan 阶段 0/7). The commands are also the implementation behind
``scripts/graphiti_memory_ops.py`` and ``scripts/reset_memory_v2.py``.

Reset classification (plan 阶段 0.5 + ADR-0140 §5):

**preserved** — role configuration, prompt versions, identity configuration, each
role's own user node and AI role node, and their per-role companion edges.
**removed** — relation groups, tombstones, hot zone, publication manifest,
session archives, chat log, the SQLite bookkeeping rows, every memory vector and
every non-identity graph node/relation.

Old memory data is deleted, never migrated: nothing has a reader for it and a
half-migrated store is worse than an empty one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain import utc_timestamp
from .identity import is_user_node_id

# Role-directory JSON files that hold legacy memory content.
MEMORY_ROLE_FILES: tuple[str, ...] = (
    "relation_groups.json",
    "tombstones.json",
    "hot_zone.json",
    "publication_manifest.json",
    "explicit_submissions.json",
    "events.json",
)
# Role-directory subtrees that hold memory content and must be dropped whole.
MEMORY_ROLE_DIRS: tuple[str, ...] = ("access", "cognition", "memory")
# Root-level subtrees under the authority root that hold memory content.
# `diary` is here because the diary is the provenance of the memory nodes: a reset
# that dropped the nodes but kept the diaries would leave text that no longer
# explains anything, and a re-run would look like it had already happened.
MEMORY_ROOT_DIRS: tuple[str, ...] = ("conversations", "sessions", "chatlog", "diary")
# Preserved role configuration: identity/behaviour, never memory.
PRESERVED_ROLE_FILES: tuple[str, ...] = ("config.json", "identity.json", "prompt", "prompts", "state.json")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _count_items(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 1 if value else 0


class MemoryOperations:
    """Operational commands over the SQLite authority and its projections."""

    def __init__(
        self,
        root: Path | str,
        *,
        memory_layer: Any = None,
        topology: Any = None,
        neo4j_projection: Any = None,
        authority: Any = None,
        config: Optional[Dict[str, Any]] = None,
        embedder: Any = None,
    ):
        self.root = Path(root)
        self.memory_layer = memory_layer
        self.topology = topology
        self.neo4j_projection = neo4j_projection
        self.authority = authority
        self.config = config or {}
        self.embedder = embedder

    # ------------------------------------------------------------------ helpers

    @property
    def store(self):
        if self.memory_layer is None:
            raise RuntimeError("memory layer is not configured")
        return self.memory_layer.store

    def _authority_root(self) -> Path:
        if self.authority is not None:
            return Path(self.authority.root)
        return self.root

    def _role_dirs(self) -> List[Path]:
        characters = self._authority_root() / "characters"
        if not characters.exists():
            return []
        return sorted(path for path in characters.iterdir() if path.is_dir())

    # --------------------------------------------------------------------- init

    def init_sqlite(self) -> Dict[str, Any]:
        """Create/upgrade the SQLite schema idempotently."""
        store = self.store
        result = store.initialize()
        return {
            "command": "init",
            "sqlite_path": str(store.path),
            "schema_version": result.get("schema_version"),
            "revision": store.revision(),
            "counts": store.counts(),
            "created_at": utc_timestamp(),
        }

    # ------------------------------------------------------------------ rebuild

    def rebuild_vectors(self, *, drop: bool = True, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Re-embed every retrieval object's memory text into the Neo4j index.

        The text re-embedded is exactly what recall will score: the object's own
        name plus the role's active memory elements about it
        (``node_memory.index_text``). Rebuilding from anything else — a SQLite row,
        a JSON mirror — would index text recall never reads.
        """
        from . import node_memory

        layer = self.memory_layer
        if layer is None:
            raise RuntimeError("memory layer is not configured")
        if self.embedder is None:
            raise RuntimeError("rebuild requires a configured embedding model")
        vectors = layer.require_vectors()
        report: Dict[str, Any] = {
            "command": "rebuild_vectors",
            "backend": "neo4j",
            "role_uuid": role_uuid,
            "dropped": False,
            "written": {},
            "failed": {},
            "created_at": utc_timestamp(),
        }
        if drop:
            try:
                report["drop"] = layer.drop_vector_collection()
                report["dropped"] = True
            except Exception as exc:  # noqa: BLE001 - report, never silently skip
                report["drop_error"] = f"{type(exc).__name__}: {exc}"
            report["ensure"] = vectors.ensure_collection()
        roles = [role_uuid] if role_uuid else sorted(self.topology.data.get("whitelist", {}))
        for role in roles:
            written = 0
            failures: List[Dict[str, Any]] = []
            self.topology.reload()
            endpoint_names = {
                str(node_id): str((node or {}).get("name") or node_id)
                for node_id, node in (self.topology.data.get("nodes") or {}).items()
            }
            records = self.topology.memory_records_for_role(role)
            for record in records:
                object_id = node_memory.object_id_of(record)
                text = node_memory.index_text(record, role, endpoint_names=endpoint_names)
                if not text:
                    continue
                object_type = node_memory.object_type_of(record)
                try:
                    vector = self.embedder(text)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"object_id": object_id, "error": str(exc)})
                    continue
                if not vector:
                    failures.append({"object_id": object_id, "error": "embedder returned no vector"})
                    continue
                content_hash = node_memory.index_hash(record, role, endpoint_names=endpoint_names)
                result = vectors.upsert(
                    object_id,
                    vector,
                    object_type=object_type,
                    role_uuid=role,
                    content_hash=content_hash,
                    status="active",
                    source_conversation_id=None,
                    content=text,
                )
                self.store.record_embedding(
                    object_id,
                    object_type,
                    role_uuid=role,
                    content_hash=content_hash,
                    model=str((self.config.get("embedding", {}) or {}).get("model") or ""),
                    version=str((self.config.get("embedding", {}) or {}).get("dimension") or ""),
                    dimension=len(list(vector)),
                    vector_point_id=result["point_id"],
                )
                self.store.complete_embedding_rebuild(object_id, role_uuid=role)
                written += 1
            self.store.bump_revision()
            report["written"][role] = written
            if failures:
                report["failed"][role] = failures
        report["count"] = {role: vectors.count(role_uuid=role) for role in roles}
        return report

    # ---------------------------------------------------------------- validation

    def validate_neo4j_defaults(self) -> Dict[str, Any]:
        if self.neo4j_projection is None:
            return {"command": "validate_neo4j", "status": "skipped", "reason": "no neo4j bridge"}
        defaults = self.neo4j_projection.validate_defaults()
        residue = self.neo4j_projection.non_default_residue()
        return {
            "command": "validate_neo4j",
            "defaults": defaults,
            "residue": residue,
            "status": "ok" if defaults.get("valid") else "invalid",
            "created_at": utc_timestamp(),
        }

    def clean_neo4j_non_default(self, *, confirm: bool = False) -> Dict[str, Any]:
        """Delete non-default projection data (events + orphan entities).

        Requires ``confirm=True``; every deleted count is reported.
        """
        if self.neo4j_projection is None:
            return {"command": "clean_neo4j", "status": "skipped", "reason": "no neo4j bridge"}
        if not confirm:
            return {
                "command": "clean_neo4j",
                "status": "dry_run",
                "residue": self.neo4j_projection.non_default_residue(),
            }
        removed_events = self.neo4j_projection.delete_events_without_role()
        result = self.neo4j_projection.bridge.execute(
            "MATCH (n:Entity) WHERE NOT n:Event "
            "AND NOT n.uuid STARTS WITH 'user:' "
            "AND coalesce(n.kind, '') <> 'ai_role' "
            "DETACH DELETE n RETURN count(*) AS removed",
        )
        records = getattr(result, "records", None) or []
        removed_nodes = int(records[0].data().get("removed", 0)) if records else 0
        return {
            "command": "clean_neo4j",
            "status": "cleaned",
            "removed_entities": removed_nodes,
            "removed_orphan_events": removed_events.get("removed", 0),
            "residue": self.neo4j_projection.non_default_residue(),
            "created_at": utc_timestamp(),
        }

    # ---------------------------------------------------------------- diagnostics

    def diagnose(self, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Counts, revisions, missing vectors and orphan events per store."""
        store = self.store
        report: Dict[str, Any] = {
            "command": "diagnose",
            "sqlite": {
                "path": str(store.path),
                "revision": store.revision(),
                "counts": store.counts(role_uuid),
            },
            "vector": {
                "available": getattr(self.memory_layer, "vectors_available", False),
                "collection": getattr(getattr(self.memory_layer, "vectors", None), "collection_name", None),
                "error": getattr(self.memory_layer, "vector_error", None),
                "index_status": None,
                "counts": None,
            },
            "neo4j": None,
            "roles": [path.name for path in self._role_dirs()],
            "created_at": utc_timestamp(),
        }
        if getattr(self.memory_layer, "vectors_available", False):
            vectors = self.memory_layer.vectors
            try:
                report["vector"]["index_status"] = vectors.payload_index_status()
                roles = [role_uuid] if role_uuid else report["roles"]
                report["vector"]["counts"] = {role: vectors.count(role_uuid=role) for role in roles}
            except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
                report["vector"]["error"] = f"{type(exc).__name__}: {exc}"
        if self.neo4j_projection is not None:
            try:
                report["neo4j"] = {
                    "defaults": self.neo4j_projection.validate_defaults(),
                    "residue": self.neo4j_projection.non_default_residue(),
                }
            except Exception as exc:  # noqa: BLE001
                report["neo4j"] = {"error": f"{type(exc).__name__}: {exc}"}
        report["orphan_events"] = store.count_orphan_events()
        report["missing_embeddings"] = store.count_missing_embeddings(role_uuid)
        report["rebuild_queue"] = store.list_embedding_rebuild_queue(limit=100)
        report["pending_publications"] = store.list_publications(status="retryable")
        report["pending_vector_operations"] = store.list_vector_operations(status="failed")
        return report

    # --------------------------------------------------------------------- reset

    def reset_plan(self) -> Dict[str, Any]:
        """Classify every path this reset would delete or preserve."""
        authority_root = self._authority_root()
        remove_files: List[str] = []
        remove_dirs: List[str] = []
        preserve_files: List[str] = []
        for role_dir in self._role_dirs():
            for name in MEMORY_ROLE_FILES:
                candidate = role_dir / name
                if candidate.exists():
                    # Role-scoped entries are nested exactly one level so a root
                    # subtree with the same leading name can never be confused
                    # with a role file.
                    remove_files.append(f"{role_dir.name}/{name}")
            for name in MEMORY_ROLE_DIRS:
                candidate = role_dir / name
                if candidate.exists():
                    remove_dirs.append(f"{role_dir.name}/{name}")
            for name in PRESERVED_ROLE_FILES:
                candidate = role_dir / name
                if candidate.exists():
                    preserve_files.append(f"{role_dir.name}/{name}")
        for name in MEMORY_ROOT_DIRS:
            candidate = authority_root / name
            if candidate.exists():
                remove_dirs.append(f"@root/{name}")
        sqlite_path = None
        if self.memory_layer is not None:
            sqlite_path = str(self.store.path)
        shared = authority_root / "shared"
        topology_nodes = 0
        if self.topology is not None:
            topology_nodes = len(self.topology.data.get("nodes", {}))
        return {
            "authority_root": str(authority_root),
            "roles": [path.name for path in self._role_dirs()],
            "remove_files": sorted(remove_files),
            "remove_dirs": sorted(remove_dirs),
            "preserve_files": sorted(preserve_files),
            "sqlite_path": sqlite_path,
            "shared_dir": self._relative(shared) if shared.exists() else None,
            "topology_nodes_before": topology_nodes,
            "preserved_objects": {
                "user_nodes": "every role's own user node (id user:<role_uuid>)",
                "ai_role_nodes": "all nodes with kind=ai_role",
                "companion_edges": "every per-role user-AI edge (kind=companion)",
            },
        }

    def _relative(self, path: Path) -> str:
        """Path relative to the authority root, role-scoped paths made readable.

        ``characters/<role>/<file>`` is reported as ``<role>/<file>`` so audit
        reports match the documented cleanup list instead of leaking the internal
        directory layout.
        """
        try:
            relative = path.relative_to(self._authority_root())
        except ValueError:
            return str(path)
        parts = relative.parts
        if len(parts) >= 2 and parts[0] == "characters":
            parts = parts[1:]
        return str(Path(*parts)).replace("\\", "/")

    def reset_memory(
        self,
        *,
        confirm: bool = False,
        phrase: str = "",
        expected_phrase: str = "DELETE LEGACY MEMORY",
        include_neo4j: bool = True,
    ) -> Dict[str, Any]:
        """One-time irreversible cleanup of legacy memory (plan 阶段 0).

        Preserves role configuration, prompts, identity configuration, the
        default user node, every ``kind=ai_role`` node and their default
        ``companion`` edges. Every deletion is counted and verified; the command
        refuses to run without an explicit confirmation phrase.
        """
        plan = self.reset_plan()
        report: Dict[str, Any] = {
            "command": "reset_memory_v2",
            "confirm": confirm,
            "plan": plan,
            "deleted": {"files": [], "dirs": [], "sqlite_rows": {}, "vectors": 0, "neo4j": {}},
            "kept": plan["preserved_objects"],
            "created_at": utc_timestamp(),
        }
        if not confirm:
            report["status"] = "dry_run"
            return report
        if phrase != expected_phrase:
            raise ValueError(
                f"confirmation phrase mismatch; pass --phrase \"{expected_phrase}\""
            )

        authority_root = self._authority_root()

        def resolve(relative: str) -> Path:
            """Resolve a plan entry: role-scoped, ``@root/`` subtree or root file."""
            if relative.startswith("@root/"):
                return authority_root / relative[len("@root/") :]
            return authority_root / "characters" / relative

        for relative in plan["remove_files"]:
            target = resolve(relative)
            if target.exists():
                target.unlink()
                report["deleted"]["files"].append(relative)
        for relative in plan["remove_dirs"]:
            target = resolve(relative)
            if target.exists():
                shutil.rmtree(target)
                report["deleted"]["dirs"].append(relative)

        if self.memory_layer is not None:
            store = self.store
            before = store.counts()
            with store.transaction() as transaction:
                # Validated, table-scoped deletes: everything SQLite still holds is
                # memory bookkeeping, and identity lives in the topology, not here.
                for table in (
                    "events",
                    "relations",
                    "whitelist",
                    "nodes",
                    "embedding_records",
                    "embedding_rebuild_queue",
                    "vector_operations",
                    "publication_operations",
                ):
                    transaction.execute(f"DELETE FROM {table}")
            report["deleted"]["sqlite_rows"] = before
            report["deleted"]["sqlite_counts_after"] = store.counts()
            store.bump_revision()

        if getattr(self.memory_layer, "vectors_available", False):
            try:
                report["deleted"]["vectors"] = self.memory_layer.drop_vector_collection()
            except Exception as exc:  # noqa: BLE001
                report["deleted"]["vector_error"] = f"{type(exc).__name__}: {exc}"

        if include_neo4j and self.neo4j_projection is not None:
            try:
                events = self.neo4j_projection.delete_events_without_role()
                result = self.neo4j_projection.bridge.execute(
                    "MATCH (n:Entity) WHERE NOT n:Event "
                    "AND NOT n.uuid STARTS WITH 'user:' "
                    "AND coalesce(n.kind, '') <> 'ai_role' "
                    "DETACH DELETE n RETURN count(*) AS removed",
                )
                records = getattr(result, "records", None) or []
                report["deleted"]["neo4j"] = {
                    "removed_entities": int(records[0].data().get("removed", 0)) if records else 0,
                    "removed_orphan_events": events.get("removed", 0),
                }
            except Exception as exc:  # noqa: BLE001
                report["deleted"]["neo4j"] = {"error": f"{type(exc).__name__}: {exc}"}

        # Shared topology: keep identity nodes and default companion edges only.
        if self.topology is not None:
            report["deleted"]["topology"] = self._prune_topology()
        report["verification"] = self.verify_reset()
        report["status"] = "reset" if report["verification"].get("clean") else "reset_with_residue"
        return report

    def _prune_topology(self) -> Dict[str, Any]:
        """Keep every role's two identity nodes and their companion edges."""
        topology = self.topology
        topology.reload()
        data = topology.data
        nodes = data.get("nodes", {})
        keep_nodes = {
            node_id
            for node_id, node in nodes.items()
            if node.get("kind") == "ai_role" or is_user_node_id(node_id)
        }
        removed_nodes = [node_id for node_id in nodes if node_id not in keep_nodes]
        removed_relations = []
        for relation_id, relation in list(data.get("relations", {}).items()):
            endpoints = set(relation.get("endpoints", []))
            if not endpoints or not endpoints.issubset(keep_nodes):
                removed_relations.append(relation_id)
            elif relation.get("kind") != "companion":
                # A non-companion edge between two identity nodes is still user
                # data created by extraction; only the default companion edge is
                # preserved.
                removed_relations.append(relation_id)
        removed_event_links = list(data.get("event_links", {}).keys())
        data["nodes"] = {node_id: nodes[node_id] for node_id in keep_nodes}
        data["relations"] = {
            relation_id: relation
            for relation_id, relation in data.get("relations", {}).items()
            if relation_id not in removed_relations
        }
        data["event_links"] = {}
        data["merges"] = {"nodes": {}, "relations": {}, "operations": {}}
        data.pop("tombstones", None)
        data["tombstones"] = []
        whitelist = data.get("whitelist", {})
        for role_uuid, entry in list(whitelist.items()):
            entry["nodes"] = [node_id for node_id in entry.get("nodes", []) if node_id in keep_nodes]
            entry["relations"] = [
                relation_id for relation_id in entry.get("relations", [])
                if relation_id in data["relations"]
            ]
        # A wipe is authoritative: it is the one write that must not adopt objects
        # another writer added, or the prune would silently keep the legacy residue
        # it just removed (see ``SharedTopology._save`` and ``_merge_disk_view``).
        topology._save(merge=False)
        return {
            "removed_nodes": len(removed_nodes),
            "removed_relations": len(removed_relations),
            "removed_event_links": len(removed_event_links),
            "kept_nodes": sorted(keep_nodes),
        }

    def verify_reset(self, *, role_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Post-reset verification: only defaults remain, legacy residue is zero."""
        non_default_nodes: List[str] = []
        non_default_relations: List[str] = []
        if self.topology is not None:
            self.topology.reload()
            nodes = self.topology.data.get("nodes", {})
            relations = self.topology.data.get("relations", {})
            non_default_nodes = [
                node_id for node_id, node in nodes.items()
                if node.get("kind") != "ai_role" and not is_user_node_id(node_id)
            ]
            non_default_relations = [
                relation_id for relation_id, relation in relations.items()
                if relation.get("kind") != "companion"
                or not set(relation.get("endpoints", [])).issubset(set(nodes))
            ]
        legacy_files = []
        for role_dir in self._role_dirs():
            for name in MEMORY_ROLE_FILES:
                if (role_dir / name).exists():
                    legacy_files.append(f"{role_dir.name}/{name}")
            for name in MEMORY_ROLE_DIRS:
                if (role_dir / name).exists():
                    legacy_files.append(f"{role_dir.name}/{name}/")
        sqlite_counts = self.store.counts(role_uuid) if self.memory_layer is not None else {}
        vector_points = None
        if getattr(self.memory_layer, "vectors_available", False):
            try:
                vector_points = self.memory_layer.vectors.count()
            except Exception:  # noqa: BLE001
                vector_points = None
        clean = (
            not non_default_nodes
            and not non_default_relations
            and not legacy_files
            and all(
                int(sqlite_counts.get(key, 0)) == 0
                for key in ("events", "relations", "working_memory")
            )
            and (vector_points in (0, None))
        )
        return {
            "clean": clean,
            "non_default_nodes": non_default_nodes,
            "non_default_relations": non_default_relations,
            "legacy_files": legacy_files,
            "sqlite_counts": sqlite_counts,
            "vector_points": vector_points,
            "user_nodes": sorted(
                node_id
                for node_id in (self.topology.data.get("nodes", {}) if self.topology is not None else {})
                if is_user_node_id(node_id)
            ),
            "kept_nodes": sorted(
                (self.topology.data.get("nodes", {}) if self.topology is not None else {}).keys()
            ),
            "kept_relations": sorted(
                (self.topology.data.get("relations", {}) if self.topology is not None else {}).keys()
            ),
        }

    # -------------------------------------------------------------- export/backup

    def export_json(self, role_uuid: str, *, target: Optional[Path | str] = None) -> Dict[str, Any]:
        """Export one role's graph memory and its event bodies as a JSON snapshot.

        Memory is exported from the graph (objects plus this role's elements), not
        from a SQLite mirror: an export that read a second store could disagree
        with what recall actually returns.
        """
        from . import node_memory

        store = self.store
        objects: List[Dict[str, Any]] = []
        if self.topology is not None:
            self.topology.reload()
            objects = self.projection_objects(role_uuid)
        payload = {
            "schema_version": store.counts().get("schema_version"),
            "role_uuid": role_uuid,
            "revision": store.revision(),
            "exported_at": utc_timestamp(),
            "objects": objects,
            "events": store.list_events(role_uuid, status=None),
            "whitelist": store.allowed_ids(role_uuid),
        }
        destination = Path(target) if target else (
            self._authority_root() / "exports" / f"{role_uuid}-{utc_timestamp().replace(':', '')}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        # The per-role JSON files are what `restore_role_memory` reads back, so the
        # export refreshes them too; otherwise a restore would silently find nothing.
        restore_source = {}
        if self.authority is not None:
            try:
                restore_source["events.json"] = self.authority.save_role_file(
                    role_uuid, "events.json", payload["events"]
                )
            except Exception as exc:  # noqa: BLE001 - the portable archive is authoritative
                restore_source["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "command": "export_json",
            "role_uuid": role_uuid,
            "path": str(destination),
            "restore_source": sorted(key for key in restore_source if key.endswith(".json")),
            "counts": {
                "objects": len(objects),
                "memory_elements": sum(len(item.get("memory") or []) for item in objects),
                "events": len(payload["events"]),
            },
        }

    def projection_objects(self, role_uuid: str) -> List[Dict[str, Any]]:
        """Every object this role holds memory about, with that memory."""
        from . import node_memory

        if self.topology is None:
            return []
        self.topology.reload()
        objects: List[Dict[str, Any]] = []
        for record in self.topology.memory_records_for_role(role_uuid):
            elements = node_memory.active_elements(record, role_uuid)
            if not elements:
                continue
            objects.append({
                "id": str(record.get("id") or ""),
                "object_type": node_memory.object_type_of(record),
                "name": str(record.get("name") or ""),
                "memory": elements,
            })
        return objects
