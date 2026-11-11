"""Node-attribute memory: every memory element lives on its own node/relation.

This module replaces the private-cognition-atom store. There is no separate atom
object, no separate SQLite table and no separate vector object type: a memory
element is an entry inside the topology record of the node (or relation) it is
about, and recall is a node query followed by a vector lookup keyed by node id.

Schema written onto a topology record (``node["memory"]`` / ``relation["memory"]``):

    {
      "<role_uuid>": [                # MEMORY IS ROLE-KEYED, ALWAYS
        {
          "id": "m_...",              # stable id of this element
          "kind": "fact|preference|description|relation|correction|history",
          "text": "主人养的猫，很黏人。",
          "content_hash": "<sha256>",  # normalized-text hash, the dedup key
          "status": "active|invalidated",
          "source_conversation_id": "c_..",
          "created_at": "...Z",
          "updated_at": "...Z"
        }
      ]
    }

**Why role-keyed.** A node is shared: two roles can both know the same entity or the
same QQ contact (deterministic uuid5), and their understanding of it must differ
(ADR-0047/ADR-0153). Storing "the memory" as one flat list would make one role's
private prose readable by every other role whitelisted on that node —
`topology_for_role` spreads the record's fields. Keying by role keeps one node
identity while giving each role its own slice, and `topology_for_role` projects only
the calling role's slice.

Every element the old scheme carried is present. Sub-role view: two roles never
share an element id space in practice, but reads always go through a role argument,
so a caller cannot accidentally read "all roles' memory" — the only function that
returns more than one role's slice is the explicit migration/admin helper.

The vector entry and the embedding fingerprint are derived: an object's index text
for a role is that role's active element texts, and its ``content_hash`` hashes the
same text, so any change is detected as an index miss and re-embedded.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .domain import text_hash, typed_id, utc_timestamp

#: Element kinds. ``description`` is the entity's own description (ADR-0150),
#: ``correction`` is an explicit user correction.
MEMORY_KINDS = ("description", "fact", "preference", "relation", "correction", "history")
DEFAULT_MEMORY_KIND = "fact"


def object_id_of(record: Dict[str, Any]) -> str:
    """The retrieval object id of a topology record."""
    return str(record.get("id") or "")


def object_type_of(record: Dict[str, Any], *, is_relation: bool = False) -> str:
    """The retrieval object type of a topology record.

    ``predicate`` and ``hook`` are their own retrieval object types rather than plain
    ``node``: recall must be able to score a shared predicate as a candidate in its own
    right (that is what turns "命中谓词" into "召回它两侧的实体"), and it must be able to
    tell a model-predicted hook apart from a real knowledge entity.
    """
    if is_relation or record.get("kind") == "relation" or "endpoints" in record:
        return "relation"
    kind = str(record.get("kind") or "")
    if kind == "event":
        return "event"
    if kind == "fact":
        return "fact"
    if kind == "predicate":
        return "predicate"
    if kind == "hook":
        return "hook"
    return "node"


def _role_memory(record: Dict[str, Any], role_uuid: str, *, create: bool = False) -> List[Dict[str, Any]]:
    """The element list of one role on this record.

    Two shapes are accepted, and both are load-bearing:

    * the **stored** shape, ``{"<role_uuid>": [...]}`` — one shared record carries
      every role's slice;
    * the **projected** shape, ``[...]`` — the flat list produced by
      ``project_for_role``, which is what role-facing reads and recall operate on.

    Accepting both means a caller cannot accidentally reinterpret one as the other;
    a stored map without the requested role yields ``[]`` rather than leaking
    another role's elements.
    """
    key = str(role_uuid or "")
    if not key:
        raise ValueError("role_uuid is required to read or write memory")
    memory = record.get("memory")
    if isinstance(memory, list):
        return memory
    if not isinstance(memory, dict):
        if not create:
            return []
        memory = {}
        record["memory"] = memory
    elements = memory.get(key)
    if not isinstance(elements, list):
        if not create:
            return []
        elements = []
        memory[key] = elements
    return elements


def stored_elements(record: Dict[str, Any], role_uuid: str) -> List[Dict[str, Any]]:
    """One role's slice of a **stored** record, tolerating the projected shape.

    Used by tests and diagnostics that hold a record straight from the topology
    (role-keyed map) or a record returned by a save (flat projected list).
    """
    memory = record.get("memory")
    if isinstance(memory, list):
        return [dict(item) for item in memory if isinstance(item, dict)]
    return elements_for_role(record, role_uuid)


def elements_for_role(record: Dict[str, Any], role_uuid: str) -> List[Dict[str, Any]]:
    """All elements one role holds about an object.

    Tolerates both shapes: a stored role-keyed map (returns that role's slice) and
    a record already projected for a role (returns the flat list). A stored map
    without this role yields ``[]`` so another role's prose can never be returned.
    """
    return [dict(item) for item in _role_memory(record, role_uuid) if isinstance(item, dict)]


def roles_with_memory(record: Dict[str, Any]) -> List[str]:
    """Roles that hold at least one element about an object (admin/diagnostics)."""
    memory = record.get("memory")
    if not isinstance(memory, dict):
        return []
    return sorted(key for key, value in memory.items() if isinstance(value, list) and value)


def memory_elements(record: Dict[str, Any], role_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
    """All elements on a record.

    With ``role_uuid`` this is the role's slice; without it the union across roles
    is returned **for migration and admin views only** — recall must always pass a
    role so one role's prose can never reach another.
    """
    if role_uuid is not None:
        return elements_for_role(record, role_uuid)
    memory = record.get("memory")
    if not isinstance(memory, dict):
        return []
    union: List[Dict[str, Any]] = []
    for key in sorted(memory):
        for item in memory[key]:
            if isinstance(item, dict):
                union.append({**item, "role_uuid": key})
    return union


def active_elements(record: Dict[str, Any], role_uuid: str) -> List[Dict[str, Any]]:
    return [item for item in elements_for_role(record, role_uuid) if item.get("status", "active") == "active"]


def history_elements(record: Dict[str, Any], role_uuid: str) -> List[Dict[str, Any]]:
    return [item for item in elements_for_role(record, role_uuid) if item.get("status") == "invalidated"]


def is_event_record(record: Dict[str, Any]) -> bool:
    """True when the record is a memory event rather than an entity/relation."""
    return str(record.get("kind") or "") == "event" or "summary" in record


def is_fact_record(record: Dict[str, Any]) -> bool:
    """True when the record is one knowledge/memory item of an entity.

    A fact is its own object: it carries its own text and its own vector, so it can
    be gated and invalidated individually instead of riding along with its parent
    entity. It is deliberately *not* an entity — see ``topology`` for how it is kept
    out of entity rendering and multi-hop adjacency.
    """
    return str(record.get("kind") or "") == "fact"


def fact_body(record: Dict[str, Any]) -> str:
    """A fact's own statement (its text, or its name as a fallback)."""
    return str(record.get("text") or record.get("name") or "").strip()


def fact_kind(record: Dict[str, Any]) -> str:
    """The statement kind inside the fact object (fact / self_statement / opinion)."""
    return str(record.get("fact_kind") or "fact")


def is_self_statement(record: Dict[str, Any]) -> bool:
    """True when this fact is the AI's own self-description, not entity knowledge."""
    return is_fact_record(record) and fact_kind(record) == "self_statement"


def is_opinion(record: Dict[str, Any]) -> bool:
    """True when this fact is an AI opinion relayed to a described entity."""
    return is_fact_record(record) and fact_kind(record) == "opinion"


def is_predicate_record(record: Dict[str, Any]) -> bool:
    """True when the record is a shared predicate node (a relation's common subject).

    A predicate is a *real object* in the graph, not an item of one entity: two
    entities point at it and therefore become reachable from each other. It is
    retrievable on its own so that hitting "开罐头的声音" can surface both sides.
    """
    return str(record.get("kind") or "") == "predicate"


def is_hook_record(record: Dict[str, Any]) -> bool:
    """True when the record is a model-predicted associative hook.

    Kept a distinct type from a fact **on purpose**: a hook is a prediction
    (``basis="model_predicted"``), and callers must be able to exclude it from
    anything that renders as "the conversation told us this".
    """
    return str(record.get("kind") or "") == "hook"


def predicate_body(record: Dict[str, Any]) -> str:
    """A predicate's own phrase (its ``text``, or its name as a fallback)."""
    return str(record.get("text") or record.get("name") or "").strip()


def hook_body(record: Dict[str, Any]) -> str:
    """A hook's own predicted phrase (its ``text``, or its name as a fallback)."""
    return str(record.get("text") or record.get("name") or "").strip()


def index_text(
    record: Dict[str, Any],
    role_uuid: str,
    *,
    separator: str = "\n",
    endpoint_names: Optional[Dict[str, str]] = None,
) -> str:
    """One role's retrievable text for an object: name + body + active elements.

    The object's **own name comes first**, because searching for the entity's name
    must find the entity: with only the prose indexed, the query "大橘" could not
    match the node called 大橘. For a relation the endpoints' names are included
    through ``endpoint_names`` so "谁和谁有关系" is searchable too.

    A **memory event** contributes its summary body: the body is the memory, and a
    role's elements about it are an additional view of the same memory rather than
    the whole of it.

    A **fact** contributes its parent entity's name plus its own statement: that is
    what makes "问大橘" retrieve the facts about 大橘, while each fact still scores as
    its own object (so one weak fact can be dropped without dropping the entity).

    An object with **no retrievable text at all** yields the empty string. A bare
    identity node that only carries a name and a whitelist grant — which is exactly
    what a QQ hard-join creates — must not become a recall candidate, or recall
    would inject empty shells whose only content is the name.
    """
    parts: List[str] = []
    name = str(record.get("name") or "").strip()
    if name:
        parts.append(name)
    for endpoint in record.get("endpoints") or []:
        endpoint_id = str(endpoint)
        resolved = str((endpoint_names or {}).get(endpoint_id) or endpoint_id)
        if resolved and resolved not in parts:
            parts.append(resolved)
    body = ""
    if is_event_record(record):
        body = str(record.get("summary") or "").strip()
    elif is_fact_record(record):
        body = fact_body(record)
    elif is_predicate_record(record):
        body = predicate_body(record)
    elif is_hook_record(record):
        body = hook_body(record)
    if is_fact_record(record) or is_hook_record(record):
        # A fact and a hook are both **one item of one entity**, so both carry the
        # parent entity's name into their indexed text: that is what makes "问年糕"
        # reach 年糕's items, while each item still scores as its own object and can be
        # dropped on its own.
        parent_id = str(record.get("parent_id") or "")
        parent_name = str((endpoint_names or {}).get(parent_id) or "").strip()
        if parent_name and parent_name not in parts:
            # Parent first: the entity is the subject the query is usually about.
            parts.insert(0, parent_name)
    if is_fact_record(record) and fact_kind(record) in ("self_statement", "opinion"):
        # A perspective fact is indexed by who said it and, for an opinion, who it
        # is about. Without these labels "AI 对温兆伦的看法" can only match the
        # sentence text, never the speaker/target pair.
        labels: List[str] = []
        speaker_id = str(record.get("speaker_id") or "")
        speaker_name = str((endpoint_names or {}).get(speaker_id) or "").strip()
        if speaker_name:
            labels.append(speaker_name)
        target_id = str(record.get("target_id") or "")
        target_name = str((endpoint_names or {}).get(target_id) or "").strip()
        if target_name:
            labels.append(target_name)
        for label in reversed(labels):
            if label not in parts:
                parts.insert(0, label)
    if body and body not in parts:
        parts.append(body)
    elements = active_elements(record, role_uuid)
    if not elements:
        if body:
            return separator.join(parts)
        # A **knowledge entity** is recallable by its name alone: its knowledge lives
        # in its own fact nodes (节点细化), which stage 2 mounts onto it when it is
        # hit, and a fact-first query nominates it through those facts. The old
        # "a bare node yields no text" rule existed to keep empty identity shells (a
        # QQ hard join, a role identity) out of recall; an entity with facts is not a
        # shell. Identity kinds stay out either way — they are never a recall target
        # on their own.
        #
        # A **predicate** is the same case and must be included: its whole purpose is
        # to be the shared object two entities meet at, so it has to be indexable on
        # its own text even when no role has written an element about it yet. Leaving
        # it out here would make every predicate unindexable — present in the topology,
        # invisible to recall, which is the exact failure this change exists to fix.
        # A **relation** belongs in this list for the same reason as an entity:
        # "年糕 跑去 厨房" is its whole retrievable text (name + endpoint names),
        # and it has no elements of its own until somebody writes an understanding
        # of the edge. Leaving it out made every relation unindexable — the publish
        # path could call `_index_object` and still get `""`, so the relation object
        # type existed in the whitelist and never in the index.
        # Only a **semantic relation** (kind="relation") gets this fallback.
        # Structural edges (has_fact / predicate_link / hook_link / memory_link /
        # companion / qq_contact) also carry `endpoints`, but their identity is
        # bookkeeping, not meaning: indexing them would inject bare "A <-> B" pairs.
        if str(record.get("kind") or "") in ("node", "predicate", "relation"):
            return separator.join(parts)
        return ""
    for element in elements:
        text = str(element.get("text") or "").strip()
        if text and text not in parts:
            parts.append(text)
    return separator.join(parts)


def description_for_role(record: Dict[str, Any], role_uuid: str) -> str:
    """One role's description view of a record (entity description, merged)."""
    texts = [
        str(element.get("text"))
        for element in active_elements(record, role_uuid)
        if element.get("kind") == "description"
    ]
    return "".join(dict.fromkeys(texts))


def index_hash(record: Dict[str, Any], role_uuid: str, *, endpoint_names: Optional[Dict[str, str]] = None) -> str:
    """Content hash of one role's retrievable text; the re-embed trigger."""
    return text_hash(index_text(record, role_uuid, endpoint_names=endpoint_names))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def new_element(
    text: Any,
    *,
    kind: str = DEFAULT_MEMORY_KIND,
    element_id: Optional[str] = None,
    source_conversation_id: Optional[str] = None,
    status: str = "active",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one memory element (does not attach it to a record)."""
    content = normalize_text(text)
    if not content:
        raise ValueError("memory element text is required")
    normalized_kind = str(kind or DEFAULT_MEMORY_KIND)
    if normalized_kind not in MEMORY_KINDS:
        raise ValueError(f"unsupported memory kind: {kind}")
    now = created_at or utc_timestamp()
    return {
        "id": element_id or typed_id("memory"),
        "kind": normalized_kind,
        "text": content,
        "content_hash": text_hash(content),
        "status": status,
        "source_conversation_id": source_conversation_id,
        "created_at": now,
        "updated_at": now,
    }


def contains_text(record: Dict[str, Any], role_uuid: str, text: Any) -> Optional[Dict[str, Any]]:
    """The role's active element whose normalized text equals ``text``, if any."""
    content = normalize_text(text)
    if not content:
        return None
    for element in elements_for_role(record, role_uuid):
        if element.get("status", "active") != "active":
            continue
        if normalize_text(element.get("text")) == content:
            return element
    return None


def upsert_elements(
    record: Dict[str, Any],
    role_uuid: str,
    texts: Iterable[Any],
    *,
    kind: str = DEFAULT_MEMORY_KIND,
    source_conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach ``texts`` to one role's slice of a record, skipping duplicates.

    Returns ``{"record": ..., "added": [...], "skipped": [...]}``. The record is
    mutated in place *and* returned so the caller can hand it to the topology
    writer.
    """
    elements = _role_memory(record, role_uuid, create=True)
    added: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for text in texts:
        content = normalize_text(text)
        if not content:
            continue
        if contains_text(record, role_uuid, content) is not None:
            skipped.append(content)
            continue
        element = new_element(
            content, kind=kind, source_conversation_id=source_conversation_id
        )
        elements.append(element)
        added.append(element)
    record.setdefault("memory", {})[str(role_uuid)] = elements
    return {"record": record, "added": added, "skipped": skipped}


def invalidate_element(
    record: Dict[str, Any],
    role_uuid: str,
    element_id: str,
    *,
    reason: str = "",
    replacement_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark one of a role's elements invalidated; it stays as read-only history."""
    elements = _role_memory(record, role_uuid, create=True)
    target: Optional[Dict[str, Any]] = None
    for element in elements:
        if str(element.get("id")) == str(element_id):
            target = element
            break
    if target is None:
        raise KeyError(element_id)
    target["status"] = "invalidated"
    target["invalidated_at"] = utc_timestamp()
    if reason:
        target["invalidated_reason"] = str(reason)
    target["updated_at"] = target["invalidated_at"]
    record.setdefault("memory", {})[str(role_uuid)] = elements
    if replacement_text:
        upsert_elements(record, role_uuid, [replacement_text], kind="correction")
    return target


def set_description(record: Dict[str, Any], role_uuid: str, description: str) -> Dict[str, Any]:
    """Attach an entity description to one role's slice, accumulating.

    The description is an element of kind ``description``. The record's
    ``description`` field is a **role-keyed map** of merged views so the graph
    display does not have to reassemble the elements, and so one role's
    description is never shown to another.
    """
    result = upsert_elements(record, role_uuid, [description], kind="description")
    merged = description_for_role(record, role_uuid)
    views = record.get("description")
    if not isinstance(views, dict):
        views = {}
        record["description"] = views
    views[str(role_uuid)] = merged
    hashes = record.get("description_hash")
    if not isinstance(hashes, dict):
        hashes = {}
        record["description_hash"] = hashes
    hashes[str(role_uuid)] = text_hash(merged)
    stamps = record.get("description_updated_at")
    if not isinstance(stamps, dict):
        stamps = {}
        record["description_updated_at"] = stamps
    stamps[str(role_uuid)] = utc_timestamp()
    return result


def description_view(record: Dict[str, Any], role_uuid: str) -> str:
    """One role's merged description view, recomputed from its elements."""
    return description_for_role(record, role_uuid)


def project_for_role(record: Dict[str, Any], role_uuid: str) -> Dict[str, Any]:
    """A copy of a record carrying only ``role_uuid``'s memory.

    This is the projection every role-facing read must go through. Without it a
    node's role-keyed memory map would be spread into the response and one role
    would read another's private prose.
    """
    projected = {
        key: value
        for key, value in record.items()
        if key not in {"memory", "description", "description_hash", "description_updated_at"}
    }
    elements = elements_for_role(record, role_uuid)
    if elements:
        projected["memory"] = list(elements)
    description = description_for_role(record, role_uuid)
    if not description and isinstance(record.get("description"), dict):
        description = str(record["description"].get(str(role_uuid)) or "")
    if description:
        projected["description"] = description
        projected["description_hash"] = text_hash(description)
    projected["memory_projected"] = True
    # The owning role travels with the projection so a save can put the flat list
    # back under the right key without guessing.
    projected["memory_role"] = str(role_uuid)
    return projected


def migrate_elements(
    record: Dict[str, Any],
    role_uuid: str,
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Import rows from the retired cognition/memory stores into one role's slice.

    ``rows`` carry the old shape (``id``/``cognition_id``, ``kind``, ``text`` or
    ``content``, ``content_hash``, ``status``, ``source_conversation_id``,
    ``created_at``). Old kinds are mapped onto the new closed set so a migrated
    element stays recognisable.
    """
    kind_map = {
        "entity_attribute": "fact",
        "relation_description": "relation",
        "preference": "preference",
        "node_description": "description",
        "memory": "fact",
        "relation_group": "relation",
    }
    converted: List[Dict[str, Any]] = []
    for row in rows:
        text = row.get("text") or row.get("content")
        if not normalize_text(text):
            continue
        element = new_element(
            text,
            kind=kind_map.get(str(row.get("kind") or ""), DEFAULT_MEMORY_KIND),
            element_id=str(row.get("cognition_id") or row.get("id") or "") or None,
            source_conversation_id=row.get("source_conversation_id"),
            status="invalidated" if str(row.get("status") or "active") != "active" else "active",
            created_at=row.get("created_at"),
        )
        if row.get("content_hash"):
            element["content_hash"] = str(row["content_hash"])
        converted.append(element)
    result = upsert_elements(record, role_uuid, [item["text"] for item in converted])
    # Re-apply the migrated metadata over the freshly created elements so
    # provenance and per-statement timestamps survive the import verbatim.
    by_text = {normalize_text(item["text"]): item for item in _role_memory(record, role_uuid, create=True)}
    for original in converted:
        target = by_text.get(normalize_text(original["text"]))
        if target is not None:
            target.update({key: value for key, value in original.items() if value is not None})
    if description_for_role(record, role_uuid):
        set_description(record, role_uuid, description_for_role(record, role_uuid))
    return {"record": record, "imported": len(converted), "added": len(result["added"])}

