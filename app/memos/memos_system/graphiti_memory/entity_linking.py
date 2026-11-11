"""Resolve query text to graph entities, without embeddings.

Method A (entity-anchored recall) and the graph query tool both need the same
answer to one question: *which whitelisted entities does this query name?*  The
old gateway only answered a boolean (the private _query_mentions_entity helper)
for RRF weights; that cannot retrieve a second entity's facts, so a two-entity
question was answered from whichever entity the vector top-k happened to favour.

This module is deliberately deterministic and cheap:

* exact node names / aliases, longest match first;
* an English-capitalised-token fallback for names that live in an entity's facts
  but not in the node label (the LoCoMo "Jon" facts hang off the user node
  named "主人");
* no LLM call, no embedding, no write.

It is a *candidate* resolver.  The caller still applies the whitelist and the
normal admission gates; a wrong anchor can add a candidate, never bypass a gate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .domain import normalized_text

ENTITY_KINDS = {"node", "user", "ai_role"}

#: Capitalised English words that are question grammar, not names.
_STOP_CAPITALISED = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "the", "a", "an", "and", "or", "but", "both", "all", "each", "every",
    "common", "same", "different", "tell", "name", "list", "give", "does",
    "did", "do", "is", "are", "was", "were", "has", "have", "had", "can",
    "could", "would", "should", "please", "answer", "according", "memory",
    "memories", "remember", "mentioned", "user", "assistant", "ai",
}

_CAPITALISED = re.compile(r"[A-Z][A-Za-z0-9_'\-]{1,}")


def _aliases(record: Mapping[str, Any]) -> List[str]:
    raw = record.get("aliases") or record.get("alias") or record.get("other_names")
    if isinstance(raw, str):
        raw = [part for part in re.split(r"[、,，;；/|]", raw) if part.strip()]
    if not isinstance(raw, (list, tuple, set)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def entity_records(topology: Any, role_uuid: str) -> List[Dict[str, Any]]:
    """Every whitelisted entity the role may see, with names and aliases."""
    data = getattr(topology, "data", {}) or {}
    whitelist = (data.get("whitelist") or {}).get(role_uuid) or {}
    nodes = data.get("nodes") or {}
    out: List[Dict[str, Any]] = []
    for raw_id in whitelist.get("nodes") or []:
        node_id = str(topology.resolve_node(raw_id) if hasattr(topology, "resolve_node") else raw_id)
        record = nodes.get(node_id) or {}
        if not record or str(record.get("status") or "active") != "active":
            continue
        if str(record.get("kind") or "node") not in ENTITY_KINDS:
            continue
        name = str(record.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "id": node_id,
            "name": name,
            "aliases": _aliases(record),
            "kind": str(record.get("kind") or "node"),
            "record": record,
        })
    return out


def _fact_parent(fact: Mapping[str, Any]) -> str:
    kind = str(fact.get("fact_kind") or "fact")
    if kind == "self_statement":
        return str(fact.get("speaker_id") or "")
    if kind == "opinion":
        return str(fact.get("target_id") or "")
    return str(fact.get("parent_id") or "")


def facts_for_anchor(
    facts: Iterable[Mapping[str, Any]], anchor_id: str
) -> List[Mapping[str, Any]]:
    """Facts owned by one anchor (ordinary, self-statement, opinion)."""
    key = str(anchor_id or "")
    if not key:
        return []
    return [fact for fact in facts if _fact_parent(fact) == key]


def _name_terms(entity: Mapping[str, Any]) -> List[str]:
    terms = [str(entity.get("name") or "")] + list(entity.get("aliases") or [])
    seen: List[str] = []
    for term in terms:
        normalized = normalized_text(term)
        if len(normalized) >= 2 and normalized not in seen:
            seen.append(normalized)
    return seen


def _name_fragment(name: str, query: str) -> str:
    """The longest part of a node name that occurs verbatim in the query.

    Entity labels are often full names (托马斯·爱迪生, 理查德·尼克松) while the
    question uses the surname (爱迪生, 尼克松).  Exact whole-name matching lost
    both anchors of m01/m03 in the CCC graph.  A 2..4 character fragment is enough
    to nominate the entity; the rerank/gate still decides admission.
    """
    if not name or not query:
        return ""
    if name in query:
        return name
    if name.isascii():
        # Latin names are matched whole.  A 2–4 character fragment lets
        # "Zephyria" match "Aria" through the shared "ria", which is worse than
        # no anchor: it invents an entity the query never named.
        return ""
    for size in range(min(4, len(name) - 1), 1, -1):
        for start in range(0, len(name) - size + 1):
            fragment = name[start:start + size]
            if fragment in query:
                return fragment
    return ""


def _longest_common_span(left: str, right: str, *, minimum: int = 4) -> int:
    """Length of the longest common substring, capped at the query length.

    Deterministic O(len(left) * len(right)) dynamic programming; the fact pool is
    small (hundreds per role) and this only runs when name matching found fewer
    anchors than requested.
    """
    if not left or not right or len(left) < minimum or len(right) < minimum:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0] * (len(right) + 1)
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current[index] = previous[index - 1] + 1
                if current[index] > best:
                    best = current[index]
        previous = current
    return best if best >= minimum else 0


def _capitalised_terms(query: str) -> List[str]:
    terms: List[str] = []
    for match in _CAPITALISED.finditer(str(query or "")):
        term = match.group(0)
        if term.casefold() in _STOP_CAPITALISED:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def resolve_query_anchors(
    topology: Any,
    role_uuid: str,
    query: str,
    *,
    facts: Optional[Sequence[Mapping[str, Any]]] = None,
    max_anchors: int = 4,
) -> List[Dict[str, Any]]:
    """Return up to max_anchors entities named by the query.

    Exact name/alias hits are preferred.  If the query uses an English name that
    only appears inside an entity's facts ("Jon" under the user node "主人"),
    a capitalised-token fallback nominates that entity's parent.  The fallback is
    intentionally conservative: it needs a token of at least three characters and
    at least one supporting fact.
    """
    if max_anchors <= 0 or not normalized_text(query):
        return []
    entities = entity_records(topology, role_uuid)
    normalized_query = normalized_text(query)

    matched: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        best = ""
        for term in _name_terms(entity):
            found = _name_fragment(term, normalized_query)
            if found and len(found) > len(best):
                best = found
        if best:
            matched[entity["id"]] = {
                "id": entity["id"],
                "name": entity["name"],
                "matched": best,
                "source": "name",
                "score": len(best),
                "position": normalized_query.find(best),
            }

    if len(matched) < max_anchors:
        fact_pool = list(facts) if facts is not None else _role_facts(topology, role_uuid)
        if fact_pool:
            normalised_facts = [
                (fact, normalized_text(str(fact.get("text") or fact.get("name") or "")))
                for fact in fact_pool
            ]
            for term in _capitalised_terms(str(query or "")):
                if len(term) < 3:
                    continue
                # If the token is itself a node name/alias, exact matching already
                # handled it.  Do not also nominate every parent whose fact merely
                # mentions that name ("Aria trusts Borin" would otherwise make
                # Aria a second anchor for a question only about Borin).
                if any(term.casefold() in _name_terms(entity) for entity in entities):
                    continue
                needle = term.casefold()
                counts: Dict[str, int] = {}
                for fact, text in normalised_facts:
                    if needle in text:
                        parent = _fact_parent(fact)
                        if parent:
                            counts[parent] = counts.get(parent, 0) + 1
                ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:2]
                for entity_id, count in ranked:
                    if entity_id in matched:
                        continue
                    entity = next((item for item in entities if item["id"] == entity_id), None)
                    if entity is None:
                        continue
                    matched[entity_id] = {
                        "id": entity_id,
                        "name": entity["name"],
                        "matched": term,
                        "source": "fact_token",
                        "score": count,
                        "position": str(query).casefold().find(term.casefold()),
                    }
                    if len(matched) >= max_anchors:
                        break
                if len(matched) >= max_anchors:
                    break

    if not matched and fact_pool and re.search(r"[\u4e00-\u9fff]", str(query or "")):
        # Attribute-described anchors: "美国第37任总统..." names nobody, but the
        # answer fact says 尼克松是美国第37任总统. Use the longest verbatim overlap
        # between query and fact text to nominate that fact's parent. Four
        # characters is the floor; shorter overlaps are grammatical noise.
        overlaps: Dict[str, int] = {}
        for fact, text in normalised_facts:
            span = _longest_common_span(normalized_query, text, minimum=4)
            if span <= 0:
                continue
            parent = _fact_parent(fact)
            if parent and parent not in matched:
                overlaps[parent] = max(overlaps.get(parent, 0), span)
        for entity_id, span in sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))[:2]:
            entity = next((item for item in entities if item["id"] == entity_id), None)
            if entity is None:
                continue
            matched[entity_id] = {
                "id": entity_id,
                "name": entity["name"],
                "matched": "attribute_overlap:%d" % span,
                "source": "fact_overlap",
                "score": span,
                "position": 10**6,
            }

    ordered = sorted(
        matched.values(),
        key=lambda item: (
            0 if item["source"] == "name" else 1,
            item.get("position", 10**6),
            -item["score"],
            item["id"],
        ),
    )
    return ordered[:max_anchors]


def resolve_anchor_value(
    topology: Any, role_uuid: str, value: str
) -> Optional[Dict[str, Any]]:
    """Resolve one tool-supplied anchor name or id to a whitelisted entity."""
    raw = str(value or "").strip()
    if not raw:
        return None
    entities = entity_records(topology, role_uuid)
    for entity in entities:
        if entity["id"] == raw:
            return entity
    normalized = normalized_text(raw)
    exact = [
        entity for entity in entities
        if normalized in _name_terms(entity)
    ]
    if exact:
        return sorted(exact, key=lambda item: len(item["name"]), reverse=True)[0]
    contains = [
        entity for entity in entities
        if any(normalized in term or term in normalized for term in _name_terms(entity))
    ]
    if contains:
        return sorted(contains, key=lambda item: len(item["name"]), reverse=True)[0]
    return None


def entities_mentioned(
    text: str, entities: Sequence[Mapping[str, Any]], *, exclude: Iterable[str] = ()
) -> List[Dict[str, Any]]:
    """Known entities named in a fact text, longest name first.

    Used by the graph query tool to compare what two anchors have in common.
    Only known whitelisted entities are returned, so a generic token is never
    mistaken for a graph object.
    """
    normalized = normalized_text(text)
    excluded = {str(item) for item in exclude}
    found: List[Dict[str, Any]] = []
    for entity in entities:
        if entity["id"] in excluded:
            continue
        terms = [term for term in _name_terms(entity) if term in normalized]
        if terms:
            found.append({**entity, "matched": max(terms, key=len)})
    found.sort(key=lambda item: (-len(item.get("matched") or ""), item["id"]))
    return found


def _role_facts(topology: Any, role_uuid: str) -> List[Mapping[str, Any]]:
    try:
        return list(topology.fact_records_for_role(role_uuid, include_invalidated=False))
    except Exception:  # noqa: BLE001 - an old topology without facts is legal
        return []
