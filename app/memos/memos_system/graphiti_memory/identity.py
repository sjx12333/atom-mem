"""Explicit source aliases for the fixed user and role identities."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, Set


#: Namespace for deterministic external-contact identities. A QQ number (or a
#: group number) always resolves to the same node label, so a rename changes only
#: the display name and never creates a second node.
_QQ_NAMESPACE = uuid.UUID("6f2c1f7e-9b4a-4c1d-9d3e-2a5b7c8e1f40")


#: Prefix of the per-role user identity node. Each role's partition owns exactly
#: one user node, keyed by that role's UUID, so two roles can never share a user
#: identity (ADR-0153).
USER_NODE_PREFIX = "user:"


#: Placeholder names the two identity nodes of a role carry while a role created
#: from a document is still being extracted. Naming is the **last** step of that
#: flow (the user confirms the generated prompt first), so the extracted graph is
#: published onto an unnamed partition and
#: ``RoleAuthority.assign_identity_names`` gives both nodes their real names at
#: confirmation. The alias→identity mapping does not depend on the node names, so
#: extraction resolves "原文角色名" onto them either way.
PENDING_ROLE_NODE_NAME = "未命名角色"
PENDING_USER_NODE_NAME = "用户"


def user_node_id(role_uuid: str) -> str:
    """The user identity node of one role's partition.

    Every role owns one AI role node and one user node. They are distinct nodes
    for distinct roles even when the same human talks to all of them: a human's
    relationship with role A is not role B's memory, so the identity is scoped by
    the role UUID rather than being one global "the user" node.
    """
    role = str(role_uuid or "").strip()
    if not role:
        raise ValueError("role_uuid is required to resolve the role user node")
    return f"{USER_NODE_PREFIX}{role}"


def is_user_node_id(node_id: str) -> bool:
    """True when a node id is a per-role user identity node."""
    return str(node_id or "").startswith(USER_NODE_PREFIX)


def user_role_uuid(node_id: str) -> str:
    """The role a ``user:<role_uuid>`` identity node belongs to (``""`` if none)."""
    text = str(node_id or "")
    if not text.startswith(USER_NODE_PREFIX):
        return ""
    return text[len(USER_NODE_PREFIX):]


def is_foreign_user_node_id(role_uuid: str, node_id: str) -> bool:
    """True when ``node_id`` is another role's *user* identity node.

    The user identity is role-scoped (ADR-0153), so ``user:<other role>`` is not
    this role's knowledge even when a lookup resolved it by a shared appellation
    such as "主人".
    """
    role = str(role_uuid or "")
    node = str(node_id or "")
    if not role or not node or not is_user_node_id(node):
        return False
    owner = user_role_uuid(node)
    return bool(owner) and owner != role


def foreign_identity_scope(
    role_uuid: str,
    nodes: Dict[str, Dict[str, Any]],
    name_sets: Dict[str, Set[str]],
) -> "tuple[Set[str], Set[str]]":
    """``(foreign node ids, foreign names)`` that a role's view must not show.

    A foreign node id is another partition's identity: an ``ai_role`` node keyed
    by a different role UUID, or a ``user:<other role>`` node. Foreign names are
    the names those roles answer to, so a reference re-created as a plain entity
    by extraction is recognised in a *view* as well (its memory is untouched and
    stays recallable — only the display is scoped).
    """
    role = str(role_uuid or "")
    ids: Set[str] = set()
    for node_id, node in (nodes or {}).items():
        candidate = str(node_id)
        if candidate == role:
            continue
        if is_foreign_user_node_id(role, candidate):
            ids.add(candidate)
        elif str((node or {}).get("kind") or "") == "ai_role":
            ids.add(candidate)
    names: Set[str] = set()
    for other_role, other_names in (name_sets or {}).items():
        if str(other_role) == role:
            continue
        names |= {str(name) for name in other_names if str(name or "").strip()}
    return ids, names


def role_name_sets(configs: Iterable[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """``role_uuid -> {casefolded names this role answers to}``.

    Covers the display name and any source alias the role imports, so the view of
    one role can recognise another AI both by identity node and by name.
    """
    names: Dict[str, Set[str]] = {}
    for config in configs:
        if not isinstance(config, dict):
            continue
        role = str(config.get("role_uuid") or "").strip()
        if not role:
            continue
        found = names.setdefault(role, set())
        for candidate in (config.get("display_name"), *source_names(str(config.get("source_role_name") or ""))):
            text = str(candidate or "").strip().casefold()
            if text:
                found.add(text)
    return names


def qq_node_id(kind: str, number: str) -> str:
    """Deterministic node label for a QQ contact or group (ADR-0152).

    ``kind`` is ``"user"`` or ``"group"``; ``number`` is the QQ number / group
    number. The result is a plain UUID string because that is the only node
    identity the topology accepts.
    """
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind not in {"user", "group"}:
        raise ValueError("qq node kind must be user or group")
    number_text = str(number or "").strip()
    if not number_text:
        raise ValueError("qq number is required")
    return str(uuid.uuid5(_QQ_NAMESPACE, f"qq:{normalized_kind}:{number_text}"))


def source_names(value: str) -> list[str]:
    names = {}
    for part in str(value or "").split("/"):
        name = part.strip()
        if name:
            names.setdefault(name.casefold(), name)
    return list(names.values())


def normalize_source_identity(spec: dict) -> dict:
    result = dict(spec)
    role_names = source_names(spec.get("source_role_name", ""))
    user_names = source_names(spec.get("source_user_name", ""))
    if not role_names or not user_names:
        raise ValueError("原文角色名和原文用户名都必须包含至少一个名称，多个名称用 / 分隔")
    role_set = {name.casefold() for name in role_names + [str(spec.get("display_name") or ""), "assistant", "ai"]}
    user_set = {name.casefold() for name in user_names + [str(spec.get("user_appellation") or ""), "user", "用户"]}
    if role_set & user_set:
        raise ValueError("角色和用户的名称或称呼不能重叠，请分别填写各自主体的别名")
    result["source_role_name"] = "/".join(role_names)
    result["source_user_name"] = "/".join(user_names)
    return result
