"""Filesystem authority for role-private Graphiti data."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain import MemoryRecord, normalized_text, replace_file, typed_id, utc_timestamp


class AuthorityStore:
    """Persist role-private records without treating projections as authority."""

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root)
        self.characters_root = self.root / "characters"
        self.shared_root = self.root / "shared"
        self.characters_root.mkdir(parents=True, exist_ok=True)
        self.shared_root.mkdir(parents=True, exist_ok=True)

    def role_dir(self, role_uuid: str) -> Path:
        if not role_uuid or "/" in role_uuid or "\\" in role_uuid or role_uuid in {".", ".."}:
            raise ValueError("invalid role UUID")
        directory = self.characters_root / role_uuid
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _read(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read authority file {path}: {exc}") from exc

    def _write(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def load_role_file(self, role_uuid: str, name: str, default: Any) -> Any:
        return self._read(self._role_file_path(role_uuid, name), default)

    def save_role_file(self, role_uuid: str, name: str, value: Any) -> None:
        self._write(self._role_file_path(role_uuid, name), value)

    def _role_file_path(self, role_uuid: str, name: str) -> Path:
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid role file name")
        return self.role_dir(role_uuid) / relative

    def load_root_file(self, name: str, default: Any) -> Any:
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid root file name")
        return self._read(self.root / relative, default)

    def save_root_file(self, name: str, value: Any) -> None:
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid root file name")
        self._write(self.root / relative, value)

    def authority_files(self, role_uuid: str) -> Dict[str, Any]:
        """Return every role-authority JSON file using root-relative names."""

        directory = self.role_dir(role_uuid)
        result: Dict[str, Any] = {}
        for path in directory.rglob("*.json"):
            result[str(path.relative_to(self.root)).replace("\\", "/")] = self._read(path, {})
        return result

    def shared_authority_files(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not self.shared_root.exists():
            return result
        for path in self.shared_root.rglob("*.json"):
            result[str(path.relative_to(self.root)).replace("\\", "/")] = self._read(path, {})
        return result

    def restore_authority_files(
        self,
        role_uuid: str,
        role_files: Dict[str, Any],
        shared_files: Optional[Dict[str, Any]] = None,
    ) -> None:
        directory = self.role_dir(role_uuid)
        for path in directory.rglob("*.json"):
            path.unlink()
        for name, value in role_files.items():
            if not name.startswith(f"characters/{role_uuid}/"):
                raise ValueError("backup contains an invalid role file")
            self._write(self.root / name, value)
        for name, value in (shared_files or {}).items():
            if not name.startswith("shared/"):
                raise ValueError("backup contains an invalid shared file")
            self._write(self.root / name, value)

    def role_state(self, role_uuid: str) -> Dict[str, Any]:
        state = self.load_role_file(role_uuid, "state.json", {})
        if not isinstance(state, dict):
            raise RuntimeError("state.json must contain an object")
        return {
            **state,
            "role_uuid": role_uuid,
            "lifecycle": state.get("lifecycle", "active"),
            "memory_generation": state.get("memory_generation", 0),
            "updated_at": state.get("updated_at"),
        }

    def save_role_state(self, role_uuid: str, **updates: Any) -> Dict[str, Any]:
        state = self.role_state(role_uuid)
        state.update(updates)
        state["updated_at"] = utc_timestamp()
        self.save_role_file(role_uuid, "state.json", state)
        return state

    def role_registry(self) -> Dict[str, Any]:
        registry = self._read(self.shared_root / "roles.json", {"roles": {}})
        if not isinstance(registry, dict) or not isinstance(registry.get("roles", {}), dict):
            raise RuntimeError("roles registry must contain an object named roles")
        return registry

    def save_role_registry(self, registry: Dict[str, Any]) -> None:
        self._write(self.shared_root / "roles.json", registry)

    def role_ids(self) -> List[str]:
        """Every registered role UUID, sorted, excluding deleted roles.

        Deleted roles keep a registry entry so their UUID is never reused, but they
        must not appear in a sweep that maintains live roles' identities.
        """
        registry = self.role_registry()
        return sorted(
            role_uuid
            for role_uuid, entry in registry.get("roles", {}).items()
            if str((entry or {}).get("status") or "") != "deleted"
        )

    #: Object types a tombstone may be written for. A tombstone records that a role
    #: deliberately forgot something, so a later re-extraction cannot silently
    #: resurrect it as a fresh belief.
    TOMBSTONE_OBJECT_TYPES = ("node", "relation", "event", "event_link", "memory")

    def add_tombstone(
        self, role_uuid: str, object_type: str, object_id: str, reason: str
    ) -> Dict[str, Any]:
        """Record one explicit forgetting event on the role."""
        if object_type not in self.TOMBSTONE_OBJECT_TYPES:
            raise ValueError(f"unsupported tombstone object type: {object_type}")
        tombstones = self.load_role_file(role_uuid, "tombstones.json", [])
        if not isinstance(tombstones, list):
            raise RuntimeError("tombstones.json must contain an array")
        entry = {
            "object_type": str(object_type),
            "object_id": str(object_id),
            "reason": str(reason or "explicit"),
            "at": utc_timestamp(),
        }
        tombstones = [
            item
            for item in tombstones
            if not (
                str(item.get("object_type")) == entry["object_type"]
                and str(item.get("object_id")) == entry["object_id"]
            )
        ]
        tombstones.append(entry)
        self.save_role_file(role_uuid, "tombstones.json", tombstones)
        return entry

    def list_tombstones(self, role_uuid: str) -> List[Dict[str, Any]]:
        records = self.load_role_file(role_uuid, "tombstones.json", [])
        return list(records) if isinstance(records, list) else []

    def list_memories(self, role_uuid: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        records = self.load_role_file(role_uuid, "memories.json", [])
        if not isinstance(records, list):
            raise RuntimeError("memories.json must contain an array")
        result = []
        for record in records:
            normalized = dict(record)
            normalized.setdefault("kind", "memory")
            normalized.setdefault("status", "active")
            normalized.setdefault("node_refs", [])
            normalized.setdefault("relation_refs", [])
            normalized.setdefault("manual_protected", False)
            if include_inactive or normalized["status"] == "active":
                result.append(normalized)
        return result

    def save_memory(
        self,
        role_uuid: str,
        content: str,
        memory_id: Optional[str] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("memory content cannot be empty")
        records = self.list_memories(role_uuid, include_inactive=True)
        record = MemoryRecord(
            id=memory_id or typed_id("memory"),
            role_uuid=role_uuid,
            content=content.strip(),
            kind=metadata.pop("kind", "memory"),
            status=metadata.pop("status", "active"),
            source=metadata.pop("source", "conversation"),
            source_id=metadata.pop("source_id", None),
            node_refs=list(metadata.pop("node_refs", [])),
            relation_refs=list(metadata.pop("relation_refs", [])),
            index_brief=metadata.pop("index_brief", None),
            promoted_summary=metadata.pop("promoted_summary", None),
            manual_protected=bool(metadata.pop("manual_protected", False)),
            metadata=metadata,
        ).to_dict()
        replaced = False
        for index, existing in enumerate(records):
            if existing.get("id") == record["id"]:
                records[index] = record
                replaced = True
                break
        if not replaced:
            records.append(record)
        self.save_role_file(role_uuid, "memories.json", records)
        return record

    def invalidate_memory(self, role_uuid: str, memory_id: str, reason: str) -> Dict[str, Any]:
        records = self.list_memories(role_uuid, include_inactive=True)
        for record in records:
            if record.get("id") == memory_id:
                record["status"] = "invalidated"
                record["updated_at"] = utc_timestamp()
                history = self.load_role_file(role_uuid, "tombstones.json", [])
                history.append({"object_type": "memory", "object_id": memory_id, "reason": reason, "at": utc_timestamp()})
                self.save_role_file(role_uuid, "memories.json", records)
                self.save_role_file(role_uuid, "tombstones.json", history)
                return record
        raise KeyError(memory_id)

    def restore_memory_from_json(self, role_uuid: str, memory_store: Any) -> Dict[str, Any]:
        """Load a JSON export back into SQLite (the restore path).

        Only the two things JSON is still the authority for are restored: event
        bodies and explicit "remember this" submissions. A role's memory about an
        object is the graph itself, so re-importing it from a file is what made the
        old mirror drift — it is deliberately not supported.
        """
        restored = {"events": 0, "memories": 0}
        for row in self.load_role_file(role_uuid, "events.json", []) or []:
            if not row.get("id") or not row.get("summary"):
                continue
            memory_store.create_event(
                role_uuid,
                str(row["summary"]),
                source_kind=str(row.get("source_kind") or "user_reported"),
                occurred_at=row.get("occurred_at"),
                formed_at=row.get("formed_at"),
                source_conversation_id=row.get("source_conversation_id"),
                source_event_ids=row.get("source_event_ids") or [],
                event_id=str(row["id"]),
                status=str(row.get("status") or "active"),
            )
            restored["events"] += 1
        for row in self.load_role_file(role_uuid, "memories.json", []) or []:
            if not row.get("id") or not row.get("content"):
                continue
            self.save_memory(
                role_uuid,
                str(row["content"]),
                memory_id=str(row["id"]),
                status=str(row.get("status") or "active"),
            )
            restored["memories"] += 1
        return restored

    def save_archive(self, snapshot: Dict[str, Any]) -> Path:
        role_uuid = snapshot["role_uuid"]
        conversation_id = snapshot["conversation_id"]
        path = self.root / "conversations" / role_uuid / f"{conversation_id}.json"
        self._write(path, snapshot)
        return path

    def session_path(self, role_uuid: str, conversation_id: str) -> Path:
        if not conversation_id or "/" in conversation_id or "\\" in conversation_id or conversation_id in {".", ".."}:
            raise ValueError("invalid conversation ID")
        self.role_dir(role_uuid)
        return self.root / "sessions" / role_uuid / f"{conversation_id}.json"

    def save_session(self, session: Dict[str, Any]) -> Path:
        path = self.session_path(session["role_uuid"], session["conversation_id"])
        self._write(path, session)
        return path

    def load_session(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        return self._read(self.session_path(role_uuid, conversation_id), {})

    def list_sessions(self) -> List[Dict[str, Any]]:
        root = self.root / "sessions"
        if not root.exists():
            return []
        sessions = []
        for path in sorted(root.glob("*/*.json")):
            session = self._read(path, {})
            if session:
                sessions.append(session)
        return sessions

    def delete_session(self, role_uuid: str, conversation_id: str) -> None:
        path = self.session_path(role_uuid, conversation_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def cleanup_archive(self, role_uuid: str, conversation_id: str) -> Dict[str, Any]:
        path = self.root / "conversations" / role_uuid / f"{conversation_id}.json"
        snapshot = self._read(path, {})
        raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
        stub = {
            "role_uuid": role_uuid,
            "conversation_id": conversation_id,
            "status": "content_deleted_after_publication",
            "event_count": len(snapshot.get("events", [])),
            "checksum": hashlib.sha256(raw).hexdigest(),
            "published_at": utc_timestamp(),
        }
        self._write(path, stub)
        return stub

    def snapshot_role(self, role_uuid: str) -> Dict[str, Any]:
        directory = self.role_dir(role_uuid)
        snapshot: Dict[str, Any] = {}
        for path in directory.glob("*.json"):
            snapshot[path.name] = self._read(path, {})
        return snapshot

    def restore_role(self, role_uuid: str, snapshot: Dict[str, Any]) -> None:
        directory = self.role_dir(role_uuid)
        for path in directory.glob("*.json"):
            path.unlink()
        for name, value in snapshot.items():
            self.save_role_file(role_uuid, name, value)
