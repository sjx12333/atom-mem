"""Operational services: backups, rebuildable projections and audit logs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .authority import AuthorityStore
from .domain import utc_timestamp


class BackupManager:
    """Create validated, registered authority checkpoints.

    Projections, sessions and WorkingMemory are deliberately excluded.  They
    can be rebuilt or are live conversation state rather than authority.
    """

    RETENTION = {"startup": 3, "pre_migration": 1}
    SPECIAL_KINDS = {"manual", "restore_protection"}

    def __init__(self, authority: AuthorityStore):
        self.authority = authority
        # Serialise the read-modify-write of the shared backup registry so
        # concurrent startup backup threads (one per role) never race on
        # ``backups/registry.json`` (which produced Windows os.replace errors).
        self._lock = threading.RLock()

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def _checksum(cls, value: Any) -> str:
        return "sha256:" + hashlib.sha256(cls._canonical(value)).hexdigest()

    def _registry(self) -> Dict[str, Any]:
        return self.authority.load_root_file("backups/registry.json", {"schema_version": 1, "backups": []})

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        self.authority.save_root_file("backups/registry.json", registry)

    def create(self, role_uuid: str, kind: str = "startup", *, reason: Optional[str] = None) -> Path:
        if kind not in self.RETENTION and kind not in self.SPECIAL_KINDS:
            raise ValueError("backup kind must be startup, pre_migration, manual, or restore_protection")
        backup_id = f"b_{uuid.uuid4().hex}"
        # Keep the public layout role-scoped; the backup kind lives in the
        # immutable manifest and registry so retention can separate classes.
        path = self.authority.root / "backups" / role_uuid / backup_id
        path.mkdir(parents=True, exist_ok=False)
        files = self.authority.authority_files(role_uuid)
        # Include the shared topology and role registry in every checkpoint.
        files.update(self.authority.shared_authority_files())
        checksums: Dict[str, str] = {}
        for name, value in sorted(files.items()):
            target = path / name
            self.authority._write(target, value)
            checksums[name] = self._checksum(value)
        manifest = {
            "schema_version": 1,
            "backup_id": backup_id,
            "kind": kind,
            "role_uuid": role_uuid,
            "memory_generation": self.authority.role_state(role_uuid).get("memory_generation", 0),
            "created_at": utc_timestamp(),
            "reason": reason,
            "files": checksums,
            "status": "valid",
        }
        self.authority._write(path / "backup-manifest.json", manifest)
        with self._lock:
            registry = self._registry()
            next_sequence = max((int(item.get("sequence", 0)) for item in registry.get("backups", [])), default=0) + 1
            manifest["sequence"] = next_sequence
            self.authority._write(path / "backup-manifest.json", manifest)
            registry.setdefault("backups", []).append({**manifest, "path": str(path.relative_to(self.authority.root))})
            self._save_registry(registry)
            self.validate(path)
            self.prune(kind, role_uuid)
        return path

    def validate(self, path: Path | str) -> Dict[str, Any]:
        path = Path(path)
        manifest = self.authority._read(path / "backup-manifest.json", {})
        if manifest.get("status") != "valid" or not manifest.get("files"):
            raise ValueError("backup is not a valid registered checkpoint")
        for name, expected in manifest["files"].items():
            value = self.authority._read(path / name, None)
            if self._checksum(value) != expected:
                raise ValueError(f"backup checksum mismatch: {name}")
        return manifest

    def resolve_registered(self, path: Path | str) -> Path:
        root = self.authority.root.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("backup path is outside the memory authority root") from exc
        manifest = self.validate(candidate)
        registered = {
            str((self.authority.root / item.get("path", "")).resolve())
            for item in self._registry().get("backups", [])
            if item.get("status") == "valid"
        }
        if str(candidate) not in registered:
            raise ValueError("backup is not registered as a valid recovery point")
        return candidate

    def list(self, role_uuid: Optional[str] = None, kind: Optional[str] = None) -> list[Dict[str, Any]]:
        entries = self._registry().get("backups", [])
        return [item for item in entries if (role_uuid is None or item.get("role_uuid") == role_uuid) and (kind is None or item.get("kind") == kind) and item.get("status") == "valid"]

    def prune(self, kind: str, role_uuid: str) -> None:
        limit = self.RETENTION.get(kind)
        if limit is None:
            return
        with self._lock:
            registry = self._registry()
            candidates = sorted(
                [item for item in registry.get("backups", []) if item.get("kind") == kind and item.get("role_uuid") == role_uuid and item.get("status") == "valid"],
                key=lambda item: (int(item.get("sequence", 0)), item.get("backup_id", "")),
                reverse=True,
            )
            keep = {item.get("backup_id") for item in candidates[:limit]}
            for item in candidates[limit:]:
                backup_path = self.authority.root / item["path"]
                if backup_path.exists():
                    for child in sorted(backup_path.rglob("*"), reverse=True):
                        if child.is_file():
                            child.unlink()
                        elif child.is_dir():
                            child.rmdir()
                    backup_path.rmdir()
                item["status"] = "pruned"
            self._save_registry(registry)

    def restore(self, path: Path | str, role_uuid: str) -> Dict[str, Any]:
        path = Path(path)
        manifest = self.validate(path)
        if manifest.get("role_uuid") != role_uuid:
            raise ValueError("backup belongs to another role")
        role_files = {
            name: self.authority._read(path / name, None)
            for name in manifest["files"]
            if name.startswith("characters/") and name.split("/", 2)[1] == role_uuid
        }
        shared_files = {
            name: self.authority._read(path / name, None)
            for name in manifest["files"]
            if name.startswith("shared/")
        }
        self.authority.restore_authority_files(role_uuid, role_files, shared_files)
        state = self.authority.save_role_state(role_uuid, memory_generation=int(manifest.get("memory_generation", 0)) + 1, lifecycle="active")
        return {"status": "restored", "backup_id": manifest["backup_id"], "role_uuid": role_uuid, "state": state}

    def create_async(self, role_uuid: str) -> threading.Thread:
        thread = threading.Thread(target=self.create, args=(role_uuid,), kwargs={"kind": "startup"}, daemon=True)
        thread.start()
        return thread

    def create_async_checked(self, role_uuid: str, kind: str = "startup") -> threading.Thread:
        thread = threading.Thread(target=self.create, args=(role_uuid,), kwargs={"kind": kind}, daemon=True)
        thread.start()
        return thread


class ProjectionStore:
    """A per-role, rebuildable snapshot of what a role's graph looks like.

    The projection is a *view*, never a source of truth: the memory itself lives in
    the shared topology's object attributes, so this file only has to record which
    objects the role holds memory about, for a rebuild to be verifiable.
    """

    def __init__(self, root, topology=None):
        self.root = Path(root) / "projections"
        self.topology = topology

    def _memory_objects(self, role_uuid: str) -> List[Dict[str, Any]]:
        from . import node_memory

        if self.topology is None:
            return []
        self.topology.reload()
        objects: List[Dict[str, Any]] = []
        for record in self.topology.memory_records_for_role(role_uuid):
            elements = node_memory.active_elements(record, role_uuid)
            is_fact = node_memory.is_fact_record(record)
            if not elements and not is_fact:
                # An entity with no elements carries no memory; a fact *is* its own
                # memory and must be part of the rebuildable projection (节点细化),
                # otherwise a rebuild would silently drop every fact.
                continue
            objects.append({
                "id": str(record.get("id") or ""),
                "object_type": node_memory.object_type_of(record),
                "name": str(record.get("name") or ""),
                "parent_id": str(record.get("parent_id") or "") or None,
                "fact_kind": node_memory.fact_kind(record) if is_fact else "",
                "speaker_id": str(record.get("speaker_id") or "") or None,
                "target_id": str(record.get("target_id") or "") or None,
                "text": node_memory.fact_body(record) if is_fact else "",
                "memory": [
                    {"id": item.get("id"), "kind": item.get("kind"), "text": item.get("text", "")}
                    for item in elements
                ],
            })
        return objects

    def rebuild(self, role_uuid: str, authority: AuthorityStore) -> Dict[str, Any]:
        records = authority.list_memories(role_uuid, include_inactive=False)
        groups = authority.load_role_file(role_uuid, "relation_groups.json", [])
        whitelist = authority.load_role_file(role_uuid, "access/whitelist.json", {"nodes": [], "relations": []})
        projection = {
            "schema_version": 2,
            "role_uuid": role_uuid,
            "version": utc_timestamp(),
            "memory_generation": authority.role_state(role_uuid).get("memory_generation", 0),
            "records": [{"id": item["id"], "kind": item.get("kind", "memory"), "content": item.get("content", "")} for item in records],
            "objects": self._memory_objects(role_uuid),
            "relation_groups": [{"id": item["id"], "kind": "relation_group", "content": item.get("content", "")} for item in groups if item.get("status", "active") == "active"],
            "whitelist": whitelist,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{role_uuid}.json"
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{role_uuid}.", suffix=".tmp", dir=str(self.root))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(projection, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return projection

    def validate(self, role_uuid: str) -> Dict[str, Any]:
        path = self.root / f"{role_uuid}.json"
        if not path.exists():
            raise ValueError("projection does not exist")
        with path.open("r", encoding="utf-8") as handle:
            projection = json.load(handle)
        if (
            projection.get("role_uuid") != role_uuid
            or not isinstance(projection.get("records"), list)
            or not isinstance(projection.get("objects"), list)
        ):
            raise ValueError("projection is invalid")
        return projection


class AuditLogger:
    SENSITIVE_KEYS = {"api_key", "authorization", "prompt", "provider_payload"}

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: "[REDACTED]" if key in self.SENSITIVE_KEYS else self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def log(self, event: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._redact(event), ensure_ascii=False) + "\n")

    def read(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    result.append(value)
        return result


class MaintenanceBarrier:
    """Persistent global barrier for irreversible cutover/restore operations."""

    def __init__(self, authority: AuthorityStore):
        self.authority = authority

    def status(self) -> Dict[str, Any]:
        return self.authority.load_root_file("maintenance.json", {"status": "clear", "generation": 0})

    def enter(self, mode: str) -> Dict[str, Any]:
        current = self.status()
        if current.get("status") not in {"clear", mode}:
            raise RuntimeError(f"memory maintenance barrier is active: {current.get('status')}")
        value = {**current, "status": mode, "generation": int(current.get("generation", 0)) + 1, "started_at": utc_timestamp()}
        self.authority.save_root_file("maintenance.json", value)
        return value

    def complete(self, mode: str, *, error: Optional[str] = None) -> Dict[str, Any]:
        current = self.status()
        if current.get("status") != mode:
            raise RuntimeError("maintenance barrier mode does not match")
        if error:
            current.update({"status": mode, "error": error, "updated_at": utc_timestamp()})
        else:
            current.update({"status": "clear", "completed_at": utc_timestamp(), "error": None})
        self.authority.save_root_file("maintenance.json", current)
        return current


class CutoverManager:
    """Make the publication manifest the only visibility switch for legacy data."""

    def __init__(self, authority: AuthorityStore):
        self.authority = authority

    def _visibility(self) -> Dict[str, Any]:
        return self.authority.load_root_file("legacy/visibility.json", {"schema_version": 1, "sources": {}, "operations": {}})

    def _save_visibility(self, value: Dict[str, Any]) -> None:
        self.authority.save_root_file("legacy/visibility.json", value)

    def _cutover(self) -> Dict[str, Any]:
        return self.authority.load_root_file("legacy/cutover.json", {"schema_version": 1, "status": "not_started"})

    def _save_cutover(self, value: Dict[str, Any]) -> None:
        self.authority.save_root_file("legacy/cutover.json", value)

    def register_sources(self, role_uuid: str, source_ids: Iterable[str], *, layer: str = "LongTermMemory") -> Dict[str, Any]:
        if layer not in {"LongTermMemory", "UserMemory"}:
            raise ValueError("only LongTermMemory and UserMemory can be cut over")
        visibility = self._visibility()
        for source_id in source_ids:
            source_id = str(source_id)
            current = visibility["sources"].get(source_id)
            if current is None:
                visibility["sources"][source_id] = {"source_id": source_id, "role_uuid": role_uuid, "layer": layer, "status": "visible"}
            elif current.get("role_uuid") != role_uuid:
                raise ValueError("legacy source belongs to another role")
        self._save_visibility(visibility)
        return visibility

    def visible_sources(self, role_uuid: str, source_ids: Optional[Iterable[str]] = None) -> list[str]:
        visibility = self._visibility()["sources"]
        allowed = set(str(item) for item in source_ids) if source_ids is not None else None
        return [source_id for source_id, item in visibility.items() if item.get("role_uuid") == role_uuid and item.get("status") == "visible" and (allowed is None or source_id in allowed)]

    def hide_sources(self, role_uuid: str, operation_id: str, source_ids: Iterable[str]) -> Dict[str, Any]:
        source_ids = list(dict.fromkeys(str(item) for item in source_ids))
        visibility = self._visibility()
        operation = visibility["operations"].get(operation_id)
        if operation and operation.get("status") == "hidden":
            return operation
        for source_id in source_ids:
            item = visibility["sources"].get(source_id)
            if item is not None and item.get("role_uuid") != role_uuid:
                raise ValueError("legacy source belongs to another role")
        for source_id in source_ids:
            visibility["sources"].setdefault(source_id, {"source_id": source_id, "role_uuid": role_uuid, "layer": "LongTermMemory"})["status"] = "hidden"
        operation = {"operation_id": operation_id, "role_uuid": role_uuid, "source_ids": source_ids, "status": "hidden", "hidden_at": utc_timestamp()}
        visibility["operations"][operation_id] = operation
        self._save_visibility(visibility)
        return operation

    def claim_batch(self, role_uuid: str, batch_id: str, source_ids: Iterable[str], *, layer: str = "UserMemory") -> Dict[str, Any]:
        """Idempotently claim a legacy MemOS batch for the Graphiti publisher."""

        if not batch_id or not str(batch_id).strip():
            raise ValueError("batch_id is required")
        visibility = self.register_sources(role_uuid, source_ids, layer=layer)
        cutover = self._cutover()
        claimed = cutover.setdefault("claimed_batches", {})
        existing = claimed.get(batch_id)
        if existing is not None:
            if existing.get("role_uuid") != role_uuid:
                raise ValueError("batch belongs to another role")
            return existing
        record = {
            "batch_id": str(batch_id),
            "role_uuid": role_uuid,
            "source_ids": list(dict.fromkeys(str(item) for item in source_ids)),
            "layer": layer,
            "status": "claimed",
            "claimed_at": utc_timestamp(),
        }
        claimed[str(batch_id)] = record
        cutover["schema_version"] = 1
        self._save_cutover(cutover)
        return record

    def clean_cutover(
        self,
        role_uuid: str,
        operation_id: str,
        source_ids: Iterable[str],
        *,
        confirm: bool = False,
        delete_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Commit the legacy visibility boundary and optionally clean old storage.

        The default operation is ledger-only: the actual MemOS store is owned
        by its service and must be supplied as an explicit callback.  This
        prevents a migration command from guessing a database or deleting an
        unrelated installation.
        """

        if not confirm:
            raise PermissionError("clean cutover requires explicit confirmation")
        source_ids = list(dict.fromkeys(str(item) for item in source_ids))
        cutover = self._cutover()
        if cutover.get("status") == "completed" and cutover.get("operation_id") == operation_id:
            return cutover
        hidden = self.hide_sources(role_uuid, operation_id, source_ids)
        cleanup_status = "cleanup_pending"
        deleted: list[str] = []
        if delete_source is not None:
            for source_id in source_ids:
                delete_source(source_id)
                deleted.append(source_id)
            cleanup_status = "completed"
        result = {
            "schema_version": 1,
            "status": cleanup_status,
            "operation_id": operation_id,
            "role_uuid": role_uuid,
            "source_ids": source_ids,
            "hidden_at": hidden.get("hidden_at"),
            "deleted_source_ids": deleted,
            "updated_at": utc_timestamp(),
        }
        self._save_cutover(result)
        return result

    def mark_source_hidden(self, role_uuid: str, operation_id: str, source_ids: list[str]) -> Dict[str, Any]:
        manifest = self.authority.load_role_file(role_uuid, "publication_manifest.json", {})
        if manifest.get("operation_id") != operation_id:
            raise ValueError("publication operation does not match the current manifest")
        self.hide_sources(role_uuid, operation_id, source_ids)
        manifest["source_hidden"] = True
        manifest["hidden_source_ids"] = list(dict.fromkeys(source_ids))
        self.authority.save_role_file(role_uuid, "publication_manifest.json", manifest)
        return manifest
