"""Hard registration of external chat sources (ADR-0152).

Any message the desktop pet accepts from an external channel must leave the sender
(and, for a group message, the group) present in the graph. The rule is:

* **no edge to the "owner" node.** That would be inventing a real-world
  relationship the system was never told about (CONTEXT: 认知抽取边界 forbids
  relation-by-cooccurrence);
* **one edge to the AI role node**, because "this AI has actually been talking to
  this contact" is an observed fact, and it is what makes the contact reachable by
  multi-hop recall instead of sitting isolated in the graph. The edge is
  semantic-free like every shared edge: its meaning belongs to each role's private
  understanding;
* **no description, no cognition text**: the node is an empty identity that the
  ordinary extraction pipeline can later hang knowledge on;
* the identity is keyed by the QQ number, never by the display name, so a rename
  updates the name of the existing node instead of creating a second one;
* the grant is role-scoped: another role does not learn this contact.

The node is where the AI's knowledge about that contact belongs. Publication
resolves an extracted contact entity onto this number-keyed identity
(:func:`resolve_contact_identity`), and a duplicate an older batch created under the
extractor's own uuid is absorbed into it once its provenance confirms the same
contact (``GraphitiMemoryService._absorb_duplicate_contact_entities``).

Registration is idempotent and never blocks the conversation turn that triggered
it: a failure is reported to the caller, which retries on its own schedule.

**Self-healing.** Idempotent is not the same as complete: registration writes three
things (the node, the edge, the role's grants) through three separate saves, and it
is driven by a request the desktop pet can lose entirely (gateway not up yet, the
bounded retry budget exhausted, a restart mid-flight). A lost or half-written
registration used to stay broken forever, leaving a contact node with no edge to
its AI role node — visible in the WebUI as an isolated node. :meth:`reconcile` is
the repair path: it re-derives what should exist for a role and writes the missing
piece, so the state converges on the next QQ message or the next extraction of a
batch that came from that contact.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .domain import normalized_text, stable_pair, utc_timestamp
from .identity import qq_node_id

#: Edge kind for "this AI has been talking to this contact". Shared edges carry no
#: semantics, so this string only records provenance for diagnostics.
QQ_CONTACT_EDGE_KIND = "qq_contact"

#: Node metadata marking a node as created by the QQ hard join. It is what makes a
#: contact node recognisable to :meth:`ExternalSourceRegistrar.reconcile` without
#: guessing from its display name.
QQ_SOURCE_MARKER = "qq"

#: Knowledge-entity node kind. A node extracted from a QQ conversation is an
#: ordinary entity, and reconcile only ever links nodes of this kind.
KNOWLEDGE_NODE_KIND = "node"

#: The flat-transcript label a declared source reaches the extractor under
#: (``service.event_source_label``): ``昵称(QQ 号码)`` /
#: ``昵称(QQ 号码，群 群名)``. The extraction prompt tells the model to name the
#: contact after its source label, so the same contact can come back either bare or
#: label-shaped; both spellings name one identity.
_CONTACT_LABEL_SUFFIX = re.compile(
    r"[（(]\s*qq\s*[:：]?\s*\d+[^）)]*[）)]\s*$", re.IGNORECASE
)


def declared_external_sources(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The external sources one frozen batch declares, first occurrence per identity.

    Per-event sources are the authority, because a single batch can switch contacts
    mid-way (ADR-0155). A snapshot that only carries the batch-level
    ``external_source`` (the older shape) falls back to it. Entries without a number
    are dropped: an external identity is keyed by its number, never by a name.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}

    def remember(source: Any) -> None:
        if not isinstance(source, dict):
            return
        number = str(source.get("number") or "").strip()
        if not number:
            return
        key = (
            str(source.get("kind") or "private"),
            number,
            str(source.get("group_number") or ""),
        )
        seen.setdefault(key, source)

    for item in (snapshot or {}).get("events") or []:
        if isinstance(item, dict):
            remember(item.get("source"))
    if not seen:
        remember((snapshot or {}).get("external_source"))
    return list(seen.values())


def contact_identity_index(sources: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """``normalized name -> deterministic node id`` for one batch's own contacts.

    This is the *explicit* coreference the extraction pass is told to respect: the
    instructions tell the model to name an external contact exactly as its source
    label does, so a contact entity under that name is that contact — not a new
    same-named entity. The index is built only from the batch's declared sources (and
    only for this role), so it is never a global name lookup: a name that the batch
    did not declare stays an ordinary entity (CONTEXT: 同名不代表同一实体).
    """
    index: Dict[str, str] = {}

    def add(candidate: Any, node_id: str) -> None:
        key = normalized_text(str(candidate or ""))
        if key:
            index.setdefault(key, node_id)

    for source in sources or []:
        if not isinstance(source, dict):
            continue
        number = str(source.get("number") or "").strip()
        if not number:
            continue
        kind = "group" if str(source.get("kind") or "") == "group" else "private"
        display = str(source.get("display_name") or "").strip()
        try:
            sender_id = qq_node_id("user", number)
        except ValueError:
            continue
        # The display name, and the label the registrar falls back to when the pet
        # did not send one.
        for candidate in (display, number, f"QQ {number}"):
            add(candidate, sender_id)
        if kind != "group":
            continue
        group_number = str(source.get("group_number") or "").strip()
        if not group_number:
            continue
        group_name = str(source.get("group_name") or "").strip()
        try:
            group_id = qq_node_id("group", group_number)
        except ValueError:
            continue
        for candidate in (group_name, group_number, f"QQ群 {group_number}"):
            add(candidate, group_id)
    return index


def resolve_contact_identity(name: Any, index: Dict[str, str]) -> Optional[str]:
    """The contact one extracted entity name refers to, or ``None``.

    ``None`` is the safe answer and the common one: only a name this batch declared
    as one of its own contacts resolves, and a label-shaped spelling of it
    (``昵称(QQ 号码)``) resolves to the same identity.
    """
    if not index:
        return None
    candidate = normalized_text(str(name or ""))
    if not candidate:
        return None
    found = index.get(candidate)
    if found:
        return found
    stripped = _CONTACT_LABEL_SUFFIX.sub("", candidate).strip()
    if stripped and stripped != candidate:
        return index.get(stripped)
    return None


def same_named_plain_entities(
    topology: Any,
    role_uuid: str,
    contact_node_id: str,
) -> List[str]:
    """Plain knowledge entities this role holds under the contact's display name.

    These are the nodes a defect left behind: extraction published an entity named
    like the contact under its own uuid because nothing mapped that name onto the
    number-keyed identity (see :meth:`ExternalSourceRegistrar.register`). The scan is
    deliberately narrow — ``kind="node"`` only, no external-source marker (another
    contact's identity is never absorbed), and only nodes this role already holds.
    Whether a candidate really is that contact is decided by its provenance, not by
    this name match.
    """
    topology.reload()
    resolved_contact = str(topology.resolve_node(str(contact_node_id or "")) or "")
    contact = topology.data.get("nodes", {}).get(resolved_contact)
    if contact is None:
        return []
    target = normalized_text(contact.get("name"))
    if not target:
        return []
    found: List[str] = []
    for node_id, node in list(topology.data.get("nodes", {}).items()):
        canonical = str(topology.resolve_node(node_id) or node_id)
        if canonical == resolved_contact:
            continue
        if str(node.get("kind") or "") != KNOWLEDGE_NODE_KIND:
            continue
        if str(node.get("external_source") or ""):
            continue
        if normalized_text(node.get("name")) != target:
            continue
        if not topology.is_allowed(role_uuid, canonical):
            continue
        if canonical not in found:
            found.append(canonical)
    return found


class ExternalSourceRegistrar:
    """Register QQ contacts and groups into the shared topology, hard."""

    def __init__(self, topology, authority=None):
        self.topology = topology
        self.authority = authority

    def register(
        self,
        role_uuid: str,
        *,
        kind: str,
        number: str,
        display_name: Optional[str] = None,
        group_number: Optional[str] = None,
        group_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register one inbound source and grant it to ``role_uuid``.

        ``kind`` is ``"private"`` or ``"group"``. A group message registers both
        the group and its sender; a private message registers only the sender.
        Both the sender and the group get their own link to the AI role node.

        Every ``nodes[]`` entry reports what the call had to write: ``node_created``
        and ``edge_created`` are true only when that piece was actually missing.
        ``repaired`` is true only when an object that **already existed** was
        missing its edge and this call wrote it — i.e. the call healed a previously
        incomplete state rather than registering a new contact.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required")
        self.topology.reload()
        registered = []

        if kind == "group":
            group_id = self._register_one(
                kind="group",
                number=group_number,
                display_name=group_name,
                role_uuid=role,
            )
            link = self._link_to_role(role, group_id["id"])
            registered.append(
                {
                    "object_type": "group",
                    "id": group_id["id"],
                    "name": group_id["name"],
                    "relation_id": link["relation_id"],
                    "node_created": group_id["node_created"],
                    "edge_created": link["edge_created"],
                }
            )
        elif kind != "private":
            raise ValueError("kind must be private or group")

        user = self._register_one(
            kind="user",
            number=number,
            display_name=display_name,
            role_uuid=role,
        )
        link = self._link_to_role(role, user["id"])
        registered.append(
            {
                "object_type": "user",
                "id": user["id"],
                "name": user["name"],
                "relation_id": link["relation_id"],
                "node_created": user["node_created"],
                "edge_created": link["edge_created"],
            }
        )

        return {
            "status": "registered",
            "role_uuid": role,
            "kind": kind,
            "nodes": registered,
            "granted_to": role,
            # A repair, not a registration: the object was already there and only its
            # edge was missing.
            "repaired": any(
                item["edge_created"] and not item["node_created"] for item in registered
            ),
        }

    def reconcile(
        self,
        role_uuid: str,
        sources: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Verify and repair the QQ objects of one role; never invent a contact.

        Two things are repaired, both derived from state the caller already has:

        * every QQ node this role already holds a grant for must also have its
          default edge to the AI role node and the edge grant. A node whose edge
          write was lost (or that was granted before the edge rule existed) is
          exactly the "QQ 节点没有与 AI 的关系边" this exists to fix.
        * every source the caller names is registered through :meth:`register`
          (node + edge + grants), which covers a contact whose *node* was never
          created at all; and the knowledge entity the extractor created under that
          same display name is linked to the role node too, because that entity is
          the contact as the AI understands it.

        Returns a report of what was written; an all-empty report means everything
        already existed.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required")
        self.topology.reload()
        if role not in self.topology.data.get("nodes", {}):
            raise ValueError("role is not available")

        repaired: Dict[str, List[Any]] = {
            "nodes_created": [],
            "edges_created": [],
            "grants_added": [],
            "entities_linked": [],
        }
        checked = 0
        for node_id, node in list(self.topology.data.get("nodes", {}).items()):
            if str(node.get("external_source") or "") != QQ_SOURCE_MARKER:
                continue
            if not self._role_holds_node(role, node_id):
                # Another role's contact: this role was never told about it, and
                # reconcile must not hand it over as a side effect of a repair.
                continue
            checked += 1
            self._record_repairs(repaired, self.ensure_role_edge(role, node_id))

        for source in self._normalize_sources(sources):
            registered = self.register(role, **source)
            for item in registered["nodes"]:
                checked += 1
                if item["node_created"]:
                    repaired["nodes_created"].append(item["id"])
                if item["edge_created"]:
                    repaired["edges_created"].append(item["relation_id"])
            if source.get("display_name"):
                repaired["entities_linked"].extend(
                    self.link_named_entities(role, str(source["display_name"]))
                )

        return {
            "status": "reconciled",
            "role_uuid": role,
            "checked": checked,
            "repaired": repaired,
            "changed": any(bool(value) for value in repaired.values()),
        }

    def ensure_role_edge(self, role_uuid: str, node_id: str) -> Dict[str, Any]:
        """Ensure one object's role edge and the role's grants; idempotent.

        Returns ``{"relation_id", "edge_created", "node_granted",
        "relation_granted"}``. ``node_granted``/``relation_granted`` are true only
        when the grant was missing, so a caller can report a real repair.
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required")
        node_id = str(self.topology.resolve_node(node_id) or "")
        if node_id not in self.topology.data.get("nodes", {}):
            raise KeyError(node_id)
        edge_created = self._existing_relation_id(node_id, role) is None
        link = self._link_to_role(role, node_id)
        node_granted = self._grant_node_reporting(role, node_id)
        relation_granted = self._grant_relation_reporting(role, link["relation_id"])
        return {
            "node_id": node_id,
            "relation_id": link["relation_id"],
            "edge_created": edge_created,
            "node_granted": node_granted,
            "relation_granted": relation_granted,
        }

    def link_named_entities(self, role_uuid: str, display_name: str) -> List[Dict[str, Any]]:
        """Link knowledge entities named exactly like a contact to the role node.

        The extractor names a QQ contact by its display name (the transcript labels
        it "昵称(QQ 号码)"), so the entity it produces and the number-keyed hard-join
        node are two nodes for one person. This ensures the entity carries the same
        default edge as the hard-join node, so the contact the AI actually talks
        about is reachable by multi-hop recall instead of floating detached.

        Deliberately narrow: only ``kind="node"`` knowledge entities, only exact
        (case-folded) display-name matches, and only nodes this role was already
        granted — a same-named node that this role cannot see is never touched, and
        nothing is merged (CONTEXT: 同名不代表同一实体).
        """
        role = str(role_uuid or "").strip()
        target = normalized_text(display_name)
        if not role or not target:
            return []
        linked: List[Dict[str, Any]] = []
        for node_id, node in list(self.topology.data.get("nodes", {}).items()):
            if str(node.get("kind") or "") != KNOWLEDGE_NODE_KIND:
                continue
            if normalized_text(node.get("name")) != target:
                continue
            if not self._role_holds_node(role, node_id):
                continue
            if self._existing_relation_id(node_id, role) is not None:
                continue
            link = self._link_to_role(role, node_id)
            linked.append({"id": node_id, "name": node.get("name"), "relation_id": link["relation_id"]})
        return linked

    # ------------------------------------------------------------- internals

    def _record_repairs(self, repaired: Dict[str, List[Any]], outcome: Dict[str, Any]) -> None:
        if outcome.get("edge_created"):
            repaired["edges_created"].append(outcome["relation_id"])
        if outcome.get("node_granted"):
            repaired["grants_added"].append(outcome["node_id"])
        if outcome.get("relation_granted"):
            repaired["grants_added"].append(outcome["relation_id"])

    def _role_holds_node(self, role_uuid: str, node_id: str) -> bool:
        """True when the role's whitelist already contains this node."""
        whitelist = self.topology.data.get("whitelist", {}).get(str(role_uuid), {})
        return str(node_id) in list(whitelist.get("nodes", []))

    def _existing_relation_id(self, source_id: str, target_id: str) -> Optional[str]:
        """The id of the role edge between a contact and its role, if it exists."""
        if source_id == target_id:
            return None
        pair = stable_pair(str(source_id), str(target_id))
        for relation in self.topology.data.get("relations", {}).values():
            if tuple(relation.get("endpoints") or []) == pair:
                return str(relation.get("id"))
        return None

    def _grant_node_reporting(self, role_uuid: str, node_id: str) -> bool:
        before = self._role_holds_node(role_uuid, node_id)
        self.topology.grant_node(role_uuid, node_id)
        return not before and self._role_holds_node(role_uuid, node_id)

    def _grant_relation_reporting(self, role_uuid: str, relation_id: str) -> bool:
        whitelist = self.topology.data.get("whitelist", {}).get(str(role_uuid), {})
        before = str(relation_id) in list(whitelist.get("relations", []))
        self.topology.grant_relation(role_uuid, relation_id)
        whitelist = self.topology.data.get("whitelist", {}).get(str(role_uuid), {})
        return not before and str(relation_id) in list(whitelist.get("relations", []))

    @staticmethod
    def _normalize_sources(sources: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Accept only well-formed source specs; a bad one is skipped, not fatal."""
        normalized: List[Dict[str, Any]] = []
        for source in sources or []:
            if not isinstance(source, dict):
                continue
            number = str(source.get("number") or "").strip()
            if not number:
                continue
            kind = str(source.get("kind") or "private").strip() or "private"
            if kind not in {"private", "group"}:
                continue
            normalized.append(
                {
                    "kind": kind,
                    "number": number,
                    "display_name": source.get("display_name"),
                    "group_number": source.get("group_number"),
                    "group_name": source.get("group_name"),
                }
            )
        return normalized

    def _register_one(
        self,
        *,
        kind: str,
        number: Optional[str],
        display_name: Optional[str],
        role_uuid: str,
    ) -> Dict[str, Any]:
        node_id = qq_node_id(kind, number or "")
        label = str(display_name or "").strip() or f"{'QQ群' if kind == 'group' else 'QQ'} {number}"
        existing = self.topology.data.get("nodes", {}).get(node_id)
        node_created = existing is None
        if node_created:
            node = self.topology.upsert_node(
                node_id,
                label,
                kind="node",
                external_source=QQ_SOURCE_MARKER,
                external_kind=kind,
                external_number=str(number),
                registered_at=utc_timestamp(),
            )
        else:
            node = existing
            # A rename updates the display name on the same node; the number-keyed
            # identity never changes.
            if node.get("name") != label:
                self.topology.upsert_node(
                    node_id,
                    label,
                    kind=node.get("kind", "node"),
                    external_source=QQ_SOURCE_MARKER,
                    external_kind=kind,
                    external_number=str(number),
                )
            elif str(node.get("external_source") or "") != QQ_SOURCE_MARKER:
                # A node that predates the marker (or was written by an older
                # version) is adopted rather than duplicated.
                self.topology.upsert_node(
                    node_id,
                    label,
                    kind=node.get("kind", "node"),
                    external_source=QQ_SOURCE_MARKER,
                    external_kind=kind,
                    external_number=str(number),
                )
        self.topology.grant_node(role_uuid, node_id)
        return {"id": node_id, "name": label, "node_created": node_created}

    def _link_to_role(self, role_uuid: str, node_id: str) -> Dict[str, Any]:
        """Ensure the semantic-free edge between the contact and the AI role.

        Idempotent: ``upsert_relation`` reuses the single edge for the unordered
        pair. The edge is granted to the role, which makes the contact traversable
        by multi-hop recall. No cognition text is attached to it.
        """
        edge_created = self._existing_relation_id(node_id, role_uuid) is None
        relation = self.topology.upsert_relation(
            node_id, role_uuid, kind=QQ_CONTACT_EDGE_KIND
        )
        self.topology.grant_relation(role_uuid, relation["id"])
        return {"relation_id": str(relation["id"]), "edge_created": edge_created}
