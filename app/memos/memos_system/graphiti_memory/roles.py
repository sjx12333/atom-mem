"""Role configuration authority and browser-safe projections."""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .authority import AuthorityStore
from .domain import typed_id, utc_timestamp
from .identity import PENDING_ROLE_NODE_NAME, PENDING_USER_NODE_NAME
from .identity import user_node_id as role_user_node_id
from .topology import SharedTopology


class RoleLifecycleError(RuntimeError):
    pass


class ConnectionTestError(RoleLifecycleError):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


class RoleAuthority:
    CONFIG_FIELDS = {
        "role_uuid", "display_name", "provider", "base_url", "model", "api_key",
        "prompt_version_id", "runtime_options", "avatar", "config_version", "connection_test",
    }
    SAFE_FIELDS = {"role_uuid", "display_name", "provider", "model", "prompt_version_id", "avatar"}
    # Identity nodes are auto-created beside every role: the AI role itself and
    # **its own** user node. The two form one role partition: the AI role node is
    # keyed by the role UUID, the user node by ``user:<role_uuid>`` (ADR-0153).
    # 用户保留主体默认名称与默认称呼一致（"主人"），抽取时也按此称呼记录用户实体。
    USER_NODE_NAME = "主人"
    # The default "companion" edge auto-created inside the partition, so the graph
    # starts with the human-AI bond present instead of a pair of isolated identity
    # nodes.
    COMPANION_EDGE_KIND = "companion"
    COMPANION_EDGE_NAME = "陪伴"
    STARTABLE_LIFECYCLES = {"active"}
    #: Lifecycle written by :meth:`assert_startable` when the role's files do not
    #: add up. It is a cached verdict about the configuration rather than a stage of
    #: the role, which is why it is re-checked instead of being trusted forever.
    BLOCKED_LIFECYCLE = "blocked"
    #: Lifecycles an edit may run from. ``blocked`` is included deliberately: the
    #: edit wizard is the only place a user can repair the configuration that
    #: blocked the role, so refusing the edit made the state permanent.
    EDITABLE_LIFECYCLES = {"active", BLOCKED_LIFECYCLE}
    RUNTIME_OPTIONS = {
        "voice_enabled": bool,
        "live2d_enabled": bool,
        "input_mode": {"text", "voice"},
        "output_mode": {"text", "voice"},
    }
    #: Role-scoped sampling temperature (ADR-0145). ``None`` means the parameter is
    #: never sent to the provider, which is required by models that reject it.
    TEMPERATURE_OPTION = "temperature"
    TEMPERATURE_RANGE = (0.0, 2.0)

    def __init__(self, root, graph_identity_callback=None):
        self.authority = AuthorityStore(root)
        self.topology = SharedTopology(self.authority.shared_root / "topology.json")
        self.graph_identity_callback = graph_identity_callback
        # Serializes role-configuration publication against the startability check
        # that every request runs. Without it a check landing between "the new prompt
        # version became current" and "config.json points at it" sees an inconsistent
        # pair and marks a healthy role ``blocked``.
        self._lock = threading.RLock()

    @classmethod
    def _normalize_runtime_options(cls, runtime_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate runtime options and coerce the role-scoped temperature (ADR-0145)."""
        if runtime_options is None:
            return {}
        if not isinstance(runtime_options, dict):
            raise ValueError("runtime options must be an object")
        normalized: Dict[str, Any] = {}
        for key, value in runtime_options.items():
            if key == cls.TEMPERATURE_OPTION:
                normalized[key] = cls._normalize_temperature(value)
                continue
            expected = cls.RUNTIME_OPTIONS.get(key)
            if expected is None:
                raise ValueError(f"unsupported runtime option: {key}")
            if isinstance(expected, set):
                if value not in expected:
                    raise ValueError(f"invalid runtime option: {key}")
            elif not isinstance(value, expected):
                raise ValueError(f"invalid runtime option: {key}")
            normalized[key] = value
        return normalized

    @classmethod
    def _normalize_temperature(cls, value: Any) -> Optional[float]:
        """Return a float in range, or ``None`` when temperature must not be sent."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid runtime option: temperature")
        temperature = float(value)
        low, high = cls.TEMPERATURE_RANGE
        if not (low <= temperature <= high):
            raise ValueError(f"temperature must be between {low} and {high}, or null to disable")
        return temperature

    def _validate_config(
        self,
        display_name: str,
        provider: str,
        model: str,
        api_key: str,
        runtime_options: Optional[Dict[str, Any]],
        base_url: Optional[str],
        avatar: Optional[str],
    ) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (display_name, provider, model, api_key)):
            raise ValueError("display name, provider, model, and API key are required")
        if len(display_name.strip()) > 120:
            raise ValueError("display name is too long")
        if avatar is not None:
            avatar_path = Path(avatar)
            if not isinstance(avatar, str) or avatar_path.is_absolute() or avatar_path.name != avatar or avatar in {".", ".."}:
                raise ValueError("avatar must be a file name relative to the role directory")
        if base_url is not None and (not isinstance(base_url, str) or not re.fullmatch(r"https?://[^\s/]+(?:/[^\s]*)?", base_url.strip())):
            raise ValueError("base URL must be an http or https URL")
        self._normalize_runtime_options(runtime_options)

    def _role_uuid(self, role_uuid: Optional[str]) -> str:
        candidate = role_uuid or typed_id("role")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,160}", candidate):
            raise ValueError("invalid role UUID")
        return candidate

    def _display_name_owner(self, display_name: str, exclude: Optional[str] = None) -> Optional[str]:
        """Return the role UUID already using ``display_name``, or ``None``.

        The display name is the user's only handle on a role: it is what the desktop
        selector, the role list and the WebUI show. Two roles sharing one name are
        indistinguishable everywhere, so the name is unique across the registry —
        including a role whose document import is still running (it is hidden from
        :meth:`list_safe` but already owns its directory), and ignoring case and
        surrounding whitespace because a difference the user cannot see is not a
        difference. ``exclude`` is the role being edited, which may keep its own name.
        """
        wanted = str(display_name or "").strip().casefold()
        if not wanted:
            return None
        for role_uuid, entry in self.authority.role_registry().get("roles", {}).items():
            if role_uuid == exclude or (entry or {}).get("status") == "deleted":
                continue
            # Read the same file the name is stored in, and only when it exists:
            # ``load_role_file`` creates the role directory as a side effect, which a
            # name check has no business doing for an unrelated role.
            if not (self.authority.characters_root / role_uuid / "config.json").exists():
                continue
            config = self.authority.load_role_file(role_uuid, "config.json", {})
            existing = str((config or {}).get("display_name") or "").strip()
            if existing and existing.casefold() == wanted:
                return role_uuid
        return None

    def _assert_display_name_is_free(self, display_name: str, exclude: Optional[str] = None) -> None:
        owner = self._display_name_owner(display_name, exclude=exclude)
        if owner:
            raise ValueError(self._name_conflict_message(display_name, owner))

    @staticmethod
    def _name_conflict_message(display_name: str, owner: str) -> str:
        # Surfaced verbatim by both the desktop wizard and the WebUI, so it names the
        # role that holds the name instead of only saying the name is taken.
        return f"角色名称「{str(display_name or '').strip()}」已被角色 {owner} 使用，请换一个名称"

    @staticmethod
    def user_node_id(role_uuid: str) -> str:
        """The user identity node belonging to one role's partition (ADR-0153)."""
        return role_user_node_id(role_uuid)

    @staticmethod
    def _connection_fingerprint(config: Dict[str, Any]) -> str:
        fields = tuple(str(config.get(key) or "") for key in ("provider", "base_url", "model", "api_key"))
        return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()

    @staticmethod
    def _connection_error_category(error: Exception) -> str:
        message = str(error).casefold()
        if any(token in message for token in ("401", "403", "auth", "credential", "permission", "unauthor")):
            return "invalid_credentials_or_permission"
        if any(token in message for token in ("model", "not found", "unsupported")):
            return "invalid_model_configuration"
        if any(token in message for token in ("429", "rate", "quota", "balance", "credit")):
            return "provider_quota_or_rate_limit"
        if any(token in message for token in ("timeout", "dns", "proxy", "connection refused", "network")):
            return "retryable_network_failure"
        return "provider_unavailable"

    def test_connection(self, config: Dict[str, Any], probe: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """Run a side-effect-free local probe and return a redacted result."""
        try:
            probe(dict(config))
        except Exception as exc:
            raise ConnectionTestError(self._connection_error_category(exc)) from exc
        return {
            "result_id": typed_id("operation"),
            "status": "success",
            "tested_at": utc_timestamp(),
            "connection_fingerprint": self._connection_fingerprint(config),
        }

    def _write_prompt_version(self, role_uuid: str, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        prompt_version_id = typed_id("prompt")
        prompt_record = {
            "prompt_version_id": prompt_version_id,
            "role_uuid": role_uuid,
            "content": prompt.strip(),
            "parent_version_id": None,
            "created_at": utc_timestamp(),
        }
        self.authority.save_role_file(role_uuid, f"prompt/versions/{prompt_version_id}.json", prompt_record)
        self.authority.save_role_file(role_uuid, "prompt/current.json", {
            "current_version_id": prompt_version_id,
            "pending_version_id": None,
            "effective_boundary": "next_session",
        })
        return prompt_version_id

    def _assert_config_is_startable(self, role_uuid: str, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise RoleLifecycleError("role configuration is invalid")
        if set(config) - self.CONFIG_FIELDS:
            raise RoleLifecycleError("role configuration contains unsupported fields")
        required = {"role_uuid", "display_name", "provider", "model", "api_key", "prompt_version_id", "runtime_options", "connection_test"}
        if not required.issubset(config) or config.get("role_uuid") != role_uuid:
            raise RoleLifecycleError("role configuration is incomplete")
        current = self.authority.load_role_file(role_uuid, "prompt/current.json", {})
        prompt_id = config["prompt_version_id"]
        prompt = self.authority.load_role_file(role_uuid, f"prompt/versions/{prompt_id}.json", {})
        if current.get("current_version_id") != prompt_id or prompt.get("role_uuid") != role_uuid or not prompt.get("content"):
            raise RoleLifecycleError("role prompt reference is invalid")
        connection = config["connection_test"]
        if not isinstance(connection, dict) or connection.get("status") != "success" or connection.get("connection_fingerprint") != self._connection_fingerprint(config):
            raise RoleLifecycleError("role connection test is invalid")

    def create(
        self,
        role_uuid: Optional[str],
        display_name: str,
        provider: str,
        model: str,
        api_key: str,
        prompt: str,
        runtime_options: Optional[Dict[str, Any]] = None,
        avatar: Optional[str] = None,
        base_url: Optional[str] = None,
        connection_test: Optional[Callable[[Dict[str, Any]], None]] = None,
        activate: bool = True,
        name_pending: bool = False,
    ) -> Dict[str, Any]:
        """Create one role partition.

        ``name_pending`` is the document-import flow's first phase
        (``RoleImportCoordinator``): configuration and both identity nodes exist so
        extraction has a partition to publish onto, but the two identity nodes keep
        placeholder names until the user confirms the generated prompt and
        :meth:`assign_identity_names` gives them the real ones. The wizard's direct
        creation path leaves it ``False`` and names them immediately.
        """
        self._validate_config(display_name, provider, model, api_key, runtime_options, base_url, avatar)
        if connection_test is None:
            raise ValueError("a successful connection test is required before creating a role")
        role_uuid = self._role_uuid(role_uuid)
        registry = self.authority.role_registry()
        if role_uuid in registry["roles"]:
            raise ValueError("role UUID is already registered")
        self._assert_display_name_is_free(display_name)
        config = {
            "role_uuid": role_uuid,
            "display_name": display_name.strip(),
            "provider": provider.strip(),
            "model": model.strip(),
            "api_key": api_key,
            "base_url": base_url.strip() if base_url else None,
            "runtime_options": self._normalize_runtime_options(runtime_options),
            "avatar": avatar,
            "config_version": 1,
        }
        config["connection_test"] = self.test_connection(config, connection_test)
        directory = self.authority.characters_root / role_uuid
        directory_created = False
        with self._lock:
            topology_snapshot = self.topology.snapshot()
            try:
                if directory.exists():
                    raise ValueError("role directory already exists")
                directory.mkdir(parents=True)
                directory_created = True
                prompt_version_id = self._write_prompt_version(role_uuid, prompt)
                config["prompt_version_id"] = prompt_version_id
                self.authority.save_role_file(role_uuid, "config.json", config)
                self.authority.save_role_state(
                    role_uuid,
                    lifecycle="initializing",
                    memory_generation=0,
                    content_version=prompt_version_id,
                    cognition_manifest_revision=0,
                    whitelist_revision=0,
                    last_published_operation_id=None,
                    fifo_high_watermark=None,
                    blocked_reason=None,
                )
                for path in ("cognition/entities", "cognition/relations", "cognition/history", "access"):
                    (directory / path).mkdir(parents=True, exist_ok=True)
                for name, value in {
                    "cognition.json": [],
                    "cognition_history.json": [],
                    "memories.json": [],
                    "relation_groups.json": [],
                    "tombstones.json": [],
                    "hot_zone.json": {"objects": [], "status": "empty"},
                    "emotion_state.json": {},
                    "task_state.json": {},
                    "whitelist.json": {"nodes": [], "relations": []},
                    "cognition/manifest.json": {"revision": 0, "entities": {}, "relations": {}},
                    "access/whitelist.json": {"revision": 0, "nodes": [], "relations": []},
                    "access/tombstones.json": [],
                }.items():
                    self.authority.save_role_file(role_uuid, name, value)
                self.topology.upsert_node(
                    role_uuid,
                    PENDING_ROLE_NODE_NAME if name_pending else config["display_name"],
                    kind="ai_role",
                )
                # Auto-create this role's own user node beside it, and grant both
                # identity nodes to the role's whitelist so they appear in the
                # role-perspective (cognition) view of the main graph.
                user_id = self.user_node_id(role_uuid)
                self.topology.upsert_node(
                    user_id,
                    PENDING_USER_NODE_NAME if name_pending else self.USER_NODE_NAME,
                    kind="user",
                )
                self.topology.grant_node(role_uuid, role_uuid)
                self.topology.grant_node(role_uuid, user_id)
                # Auto-create the default user <-> AI companion edge so the graph is
                # not left with a pair of isolated identity nodes.
                self.ensure_identity_edge(role_uuid)
                if name_pending:
                    # ``ensure_identity_edge`` names the user node after the standard
                    # appellation; a pending import keeps both nodes unnamed instead.
                    self.topology.upsert_node(role_uuid, PENDING_ROLE_NODE_NAME, kind="ai_role")
                    self.topology.upsert_node(user_id, PENDING_USER_NODE_NAME, kind="user")
                if self.graph_identity_callback:
                    self.graph_identity_callback(role_uuid, config["display_name"])
                if activate:
                    self._assert_config_is_startable(role_uuid, config)
                    lifecycle = "active"
                else:
                    lifecycle = "initializing"
                self.authority.save_role_state(role_uuid, lifecycle=lifecycle, memory_generation=0)
                registry["roles"][role_uuid] = {
                    "role_uuid": role_uuid,
                    "created_at": utc_timestamp(),
                    "status": "active" if activate else "initializing",
                }
                self.authority.save_role_registry(registry)
            except Exception:
                self.topology.restore(topology_snapshot)
                if directory_created and directory.exists():
                    shutil.rmtree(directory)
                raise
        return config

    def assign_identity_names(self, role_uuid: str) -> Dict[str, Any]:
        """Give a pending-name role's two identity nodes their real names.

        The document-import flow extracts the graph **before** the user confirms the
        role (ADR-0087's wizard), so the AI node and its user node spend that phase
        on the placeholders from :mod:`identity`. Confirmation is where naming
        happens: this renames both nodes, keeps every grant and every edge untouched
        (the node ids are the identity, the name is only a label), and reports what
        changed so the caller can refresh the vectors derived from the names.
        """
        role_uuid = self._role_uuid(role_uuid)
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        if not config:
            raise ValueError("role configuration is missing")
        display_name = str(config.get("display_name") or "").strip() or role_uuid
        appellation = str(
            (self.authority.load_role_file(role_uuid, "identity.json", {}) or {}).get("user_appellation")
            or self.USER_NODE_NAME
        ).strip() or self.USER_NODE_NAME
        self.topology.reload()
        renamed: List[Dict[str, str]] = []
        for node_id, desired in ((role_uuid, display_name), (self.user_node_id(role_uuid), appellation)):
            node = self.topology.data.get("nodes", {}).get(node_id)
            if node is None or not desired or str(node.get("name") or "") == desired:
                continue
            previous = str(node.get("name") or "")
            self.topology.upsert_node(node_id, desired, kind=str(node.get("kind") or "node"))
            renamed.append({"node_id": node_id, "from": previous, "to": desired})
        self.topology.reload()
        return {"role_uuid": role_uuid, "renamed": renamed, "appellation": appellation, "display_name": display_name}

    def activate_imported(self, role_uuid: str, prompt: str) -> Dict[str, Any]:
        """Finalize a role created for a file import after prompt review."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        role_uuid = self._role_uuid(role_uuid)
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        if not config:
            raise ValueError("role configuration is missing")
        with self._lock:
            prompt_version_id = self._write_prompt_version(role_uuid, prompt)
            config["prompt_version_id"] = prompt_version_id
            config["config_version"] = int(config.get("config_version", 1)) + 1
            self.authority.save_role_file(role_uuid, "config.json", config)
            self._assert_config_is_startable(role_uuid, config)
            self.authority.save_role_state(role_uuid, lifecycle="active", blocked_reason=None, content_version=prompt_version_id)
            registry = self.authority.role_registry()
            if role_uuid in registry.get("roles", {}):
                registry["roles"][role_uuid]["status"] = "active"
                self.authority.save_role_registry(registry)
        return config

    def ensure_identity_edge(self, role_uuid: str) -> Dict[str, Any]:
        """Create (idempotently) this role's user node, companion edge and grants.

        The edge carries a stable companion kind/name so the graph shows the
        human-AI bond from the moment the role exists. It is safe to call for
        roles created before this behaviour was added (backfill), and it is the
        only place that has to know the two identity nodes of a partition.
        """
        role_uuid = self._role_uuid(role_uuid)
        user_id = self.user_node_id(role_uuid)
        self.topology.upsert_node(user_id, self.USER_NODE_NAME, kind="user")
        self.topology.grant_node(role_uuid, user_id)
        relation = self.topology.upsert_relation(
            user_id,
            role_uuid,
            kind=self.COMPANION_EDGE_KIND,
            name=self.COMPANION_EDGE_NAME,
        )
        self.topology.grant_relation(role_uuid, relation["id"])
        return relation

    def ensure_identity_edges(self) -> List[Dict[str, Any]]:
        """Backfill the default user <-> AI companion edge for every role."""
        edges = []
        for role_uuid, entry in self.authority.role_registry()["roles"].items():
            if entry.get("status") == "deleted":
                continue
            if not (self.authority.characters_root / role_uuid / "config.json").exists():
                continue
            edges.append(self.ensure_identity_edge(role_uuid))
        return edges

    def create_new(
        self,
        display_name: str,
        provider: str,
        model: str,
        api_key: str,
        **options: Any,
    ) -> Dict[str, Any]:
        return self.create(None, display_name, provider, model, api_key, **options)

    def update(
        self,
        role_uuid: str,
        display_name: str,
        provider: str,
        model: str,
        api_key: str,
        prompt: str,
        runtime_options: Optional[Dict[str, Any]] = None,
        avatar: Optional[str] = None,
        base_url: Optional[str] = None,
        connection_test: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Edit an existing role (see ADR ``角色配置编辑生效``).

        - Display name and optional avatar update immediately on the same UUID /
          AI node (no identity migration).
        - Persona/Prompt and provider/model/key produce a new prompt version and
          re-run the connection test; they become effective from the next session
          (existing sessions keep their frozen prompt/config).
        - The UUID is never changed; the graph AI node is re-labelled but not
          re-created.
        - A ``blocked`` role stays editable. That state is written by
          :meth:`assert_startable` when the files did not add up — possibly only for
          the moment an earlier edit was being published — and this wizard is the
          only place that can repair the configuration, so refusing the edit turned
          one transient mismatch into a role the user could neither start nor fix.
        """
        if connection_test is None:
            raise ValueError("a successful connection test is required before updating a role")
        self._validate_config(display_name, provider, model, api_key, runtime_options, base_url, avatar)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        role_uuid = self._role_uuid(role_uuid)
        registry = self.authority.role_registry()
        if role_uuid not in registry["roles"] or registry["roles"][role_uuid].get("status") == "deleted":
            raise ValueError("role is not registered")
        state = self.authority.role_state(role_uuid)
        if state["lifecycle"] not in self.EDITABLE_LIFECYCLES:
            raise RoleLifecycleError(f"role cannot be edited while {state['lifecycle']}")
        self._assert_display_name_is_free(display_name, exclude=role_uuid)
        existing = self.authority.load_role_file(role_uuid, "config.json", {})
        if not existing:
            raise ValueError("role configuration is missing")
        config = {
            "role_uuid": role_uuid,
            "display_name": display_name.strip(),
            "provider": provider.strip(),
            "model": model.strip(),
            "api_key": api_key,
            "base_url": base_url.strip() if base_url else None,
            "runtime_options": self._normalize_runtime_options(
                runtime_options or existing.get("runtime_options")
            ),
            "avatar": avatar,
            "config_version": int(existing.get("config_version", 1)) + 1,
        }
        # The probe is a network round trip: it must not hold the lock, because the
        # pet validates the same role from other threads on every turn and would
        # queue behind it.
        config["connection_test"] = self.test_connection(config, connection_test)
        with self._lock:
            topology_snapshot = self.topology.snapshot()
            try:
                # A new prompt version becomes the current version; it is frozen at
                # session start, so it only takes effect from the next session. It is
                # published together with the config that points at it, under the lock:
                # a startability check landing in between (every turn of the pet runs
                # one) used to see the new prompt beside the old config, mark the role
                # ``blocked``, and leave the edit itself failing.
                prompt_version_id = self._write_prompt_version(role_uuid, prompt)
                config["prompt_version_id"] = prompt_version_id
                self.authority.save_role_file(role_uuid, "config.json", config)
                # Re-label the existing AI identity node (same UUID, no re-create).
                self.topology.upsert_node(role_uuid, config["display_name"], kind="ai_role")
                # Ensure the default user <-> AI edge is present (backfill for roles
                # created before auto-edges existed).
                self.ensure_identity_edge(role_uuid)
                if self.graph_identity_callback:
                    self.graph_identity_callback(role_uuid, config["display_name"])
                self._assert_config_is_startable(role_uuid, config)
                # Saving a valid configuration is also the repair path for a role that
                # was blocked: keeping the stale verdict would keep it unstartable.
                self.authority.save_role_state(
                    role_uuid, lifecycle="active", blocked_reason=None,
                )
            except Exception:
                self.topology.restore(topology_snapshot)
                raise
        return config

    def safe_config(self, role_uuid: str) -> Dict[str, Any]:
        config = self.authority.load_role_file(role_uuid, "config.json", {})
        safe = {"role_uuid": role_uuid}
        safe.update({key: config[key] for key in self.SAFE_FIELDS if key != "role_uuid" and isinstance(config, dict) and key in config})
        state = self.authority.role_state(role_uuid)
        configuration_valid = True
        try:
            self._assert_config_is_startable(role_uuid, config)
        except RoleLifecycleError:
            configuration_valid = False
        safe.update({
            "lifecycle": state["lifecycle"],
            "memory_generation": state["memory_generation"],
            # ``blocked`` is this module's cached verdict about the files, so a role
            # carrying it is only unstartable while its configuration is actually
            # broken: a valid one is startable — the next check clears the verdict —
            # and the alternative (refusing on the label alone) is what made a
            # transient mismatch look permanent in the role list.
            "startable": configuration_valid and state["lifecycle"] in (
                self.STARTABLE_LIFECYCLES | {self.BLOCKED_LIFECYCLE}
            ),
        })
        if state.get("blocked_reason"):
            safe["blocked_reason"] = state["blocked_reason"]
        return safe

    def list_safe(self) -> List[Dict[str, Any]]:
        result = []
        for role_uuid, entry in self.authority.role_registry()["roles"].items():
            if entry.get("status") not in {"deleted", "initializing"} and (self.authority.characters_root / role_uuid / "config.json").exists():
                result.append(self.safe_config(role_uuid))
        return sorted(result, key=lambda item: item.get("role_uuid", ""))

    def assert_startable(self, role_uuid: str) -> Dict[str, Any]:
        """Refuse a role whose files do not add up, and record why it is refused.

        ``blocked`` is not a lifecycle of its own but this check's cached verdict, so
        a role already carrying it is re-checked instead of being rejected on sight.
        Trusting the stale verdict is what made one transient mismatch permanent: the
        role could no longer be selected *and* — before :meth:`update` accepted
        ``blocked`` — no longer edited, so there was no way back at all. A role whose
        configuration is still broken stays blocked.
        """
        registry = self.authority.role_registry()
        if role_uuid not in registry["roles"]:
            raise RoleLifecycleError("role is not registered")
        state = self.authority.role_state(role_uuid)
        lifecycle = state["lifecycle"]
        if lifecycle not in self.STARTABLE_LIFECYCLES and lifecycle != self.BLOCKED_LIFECYCLE:
            # ``initializing`` / ``deleting``: a real stage, not a verdict. It must not
            # be recorded as blocked.
            raise RoleLifecycleError(f"role cannot start while {lifecycle}")
        # Validating and recording the verdict share one lock with the publishers of
        # the configuration (create/update/activate_imported), so a check can no longer
        # land between a new prompt version and the config that points at it.
        with self._lock:
            config = self.authority.load_role_file(role_uuid, "config.json", {})
            try:
                self._assert_config_is_startable(role_uuid, config)
            except RoleLifecycleError as exc:
                self.authority.save_role_state(role_uuid, lifecycle=self.BLOCKED_LIFECYCLE, blocked_reason=str(exc))
                raise
            if lifecycle == self.BLOCKED_LIFECYCLE:
                self.authority.save_role_state(role_uuid, lifecycle="active", blocked_reason=None)
        return self.safe_config(role_uuid)

    def select(self, role_uuid: str) -> Dict[str, Any]:
        """Validate an explicit role selection without persisting an active role."""
        return self.assert_startable(role_uuid)

    def mark_deleting(self, role_uuid: str) -> Dict[str, Any]:
        self.assert_startable(role_uuid)
        state = self.authority.save_role_state(role_uuid, lifecycle="deleting")
        registry = self.authority.role_registry()
        registry["roles"][role_uuid]["status"] = "deleting"
        self.authority.save_role_registry(registry)
        return state
