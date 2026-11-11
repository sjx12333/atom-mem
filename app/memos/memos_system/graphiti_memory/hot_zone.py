"""Role-scoped persistent hot-zone placement and session snapshots.

The hot zone is a **rebuildable projection**: it stores stable object ids and the
statistics that justified their placement, never a copy of the memory body. A
session freezes the current placement once and keeps it for the whole session;
a new placement only takes effect in the next session (ADR-0015/0084).

`HotZoneBuilder` is the producer that was missing. It turns `ActivationStore`
counters into a placement using the policy in `runtime.DEFAULT_CONFIG["hot_zone"]`:
a **fact** enters only when the last 7 days contain at least 5 direct hits and at
least 2 batches whose own hit rate is above 30%.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .authority import AuthorityStore
from .domain import utc_timestamp
from . import node_memory


class HotZoneStore:
    """Keep hot-zone placement separate from the memories it references."""

    QUIET_SECONDS = 3600.0

    def __init__(
        self,
        authority: AuthorityStore,
        root: Optional[Path | str] = None,
        quiet_seconds: float = QUIET_SECONDS,
    ):
        self.authority = authority
        self.root = Path(root) if root is not None else None
        self.quiet_seconds = float(quiet_seconds)

    def path_for(self, role_uuid: str) -> Path:
        if self.root is not None:
            return self.root / "projections" / "hot_zone" / f"{role_uuid}.json"
        return self.authority.role_dir(role_uuid) / "hot_zone.json"

    def _legacy_path(self, role_uuid: str) -> Path:
        return self.authority.role_dir(role_uuid) / "hot_zone.json"

    def load(self, role_uuid: str) -> Dict[str, Any]:
        raw: Any = {}
        path = self.path_for(role_uuid)
        if path.exists():
            raw = self.authority._read(path, {})
        elif self.root is not None:
            legacy = self._legacy_path(role_uuid)
            if legacy.exists():
                raw = self.authority._read(legacy, {})
        if not isinstance(raw, dict):
            raise RuntimeError("hot_zone.json must contain an object")
        objects = raw.get("objects", [])
        if not isinstance(objects, list):
            raise RuntimeError("hot_zone.json objects must contain an array")
        normalized = {
            "role_uuid": role_uuid,
            "version": int(raw.get("version", 0)),
            "status": raw.get("status", "empty"),
            "objects": [copy.deepcopy(item) for item in objects if isinstance(item, dict)],
            "last_output_at": raw.get("last_output_at"),
            "recalculated_at": raw.get("recalculated_at"),
            "policy_version": raw.get("policy_version"),
        }
        return copy.deepcopy(normalized)

    def save(self, role_uuid: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        path = self.path_for(role_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.authority._write(path, snapshot)
        return copy.deepcopy(snapshot)

    def freeze(
        self,
        role_uuid: str,
        *,
        renderer: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        """Return a detached snapshot that callers may safely hold for a session."""

        snapshot = self.load(role_uuid)
        if renderer is not None:
            rendered: List[str] = []
            kept: List[Dict[str, Any]] = []
            for item in snapshot.get("objects", []):
                if not isinstance(item, dict):
                    continue
                text = str(renderer(role_uuid, item) or "").strip()
                # A deleted/invalidated object must disappear from the frozen
                # snapshot rather than inject its last rendered copy. Legacy
                # placements (no object_type, stored by the old caller) keep their
                # own text so existing data/tests stay readable.
                if not text:
                    if not item.get("object_type"):
                        text = str(item.get("text") or item.get("content") or "").strip()
                    if not text:
                        continue
                item["text"] = text
                kept.append(item)
                rendered.append(text)
            snapshot["objects"] = kept
            snapshot["rendered_texts"] = rendered
        else:
            snapshot["rendered_texts"] = [
                str(item.get("text") or item.get("content") or "")
                for item in snapshot.get("objects", [])
                if isinstance(item, dict)
            ]
        return snapshot

    def recalculate(
        self,
        role_uuid: str,
        objects: Iterable[Dict[str, Any]],
        *,
        last_output_at: float,
        now: float,
        session_active: bool = False,
        policy_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        current = self.load(role_uuid)
        if session_active:
            return {**current, "status": "unchanged", "reason": "session_active", "snapshot": current}
        if float(now) - float(last_output_at) <= self.quiet_seconds:
            return {
                **current,
                "status": "unchanged",
                "reason": "output_quiet_period",
                "snapshot": current,
            }

        next_objects: List[Dict[str, Any]] = [
            copy.deepcopy(item) for item in objects if isinstance(item, dict)
        ]
        updated = {
            **current,
            "version": current["version"] + 1,
            "status": "active" if next_objects else "empty",
            "objects": next_objects,
            "last_output_at": float(last_output_at),
            "recalculated_at": utc_timestamp(),
            "policy_version": policy_version or current.get("policy_version"),
        }
        self.save(role_uuid, updated)
        return {**copy.deepcopy(updated), "status": "updated", "snapshot": copy.deepcopy(updated)}

    def replace(
        self,
        role_uuid: str,
        objects: Iterable[Dict[str, Any]],
        *,
        last_output_at: float,
        now: float,
    ) -> Dict[str, Any]:
        """Recalculate placement without requiring a running session."""

        return self.recalculate(
            role_uuid,
            objects,
            last_output_at=last_output_at,
            now=now,
        )

    def remove(self, role_uuid: str, object_id: str) -> Dict[str, Any]:
        """Remove placement only; the referenced authoritative memory remains."""

        current = self.load(role_uuid)
        remaining = [item for item in current["objects"] if item.get("id") != object_id]
        if len(remaining) == len(current["objects"]):
            return current
        updated = {
            **current,
            "version": current["version"] + 1,
            "status": "active" if remaining else "empty",
            "objects": remaining,
            "recalculated_at": utc_timestamp(),
        }
        self.save(role_uuid, updated)
        return copy.deepcopy(updated)


class HotZoneBuilder:
    """Build the next frozen hot-zone placement from activation statistics."""

    def __init__(
        self,
        authority: AuthorityStore,
        topology: Any,
        activation: Any,
        *,
        store: Optional[HotZoneStore] = None,
        config: Optional[Dict[str, Any]] = None,
        policy: Any = None,
        token_counter: Optional[Callable[[str], int]] = None,
        renderer: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ):
        self.authority = authority
        self.topology = topology
        self.activation = activation
        self.store = store or HotZoneStore(authority)
        self.config = dict(config or {})
        self.policy = policy
        self.token_counter = token_counter or len
        self.renderer = renderer

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    def _is_active_record(self, role_uuid: str, object_id: str) -> Optional[Dict[str, Any]]:
        if not self.topology.is_allowed(role_uuid, object_id):
            return None
        record = self.topology.find_record(object_id)
        if record is None or str(record.get("status") or "active") == "deleted":
            return None
        return record

    def _text_for(self, role_uuid: str, item: Dict[str, Any], record: Dict[str, Any]) -> str:
        if self.renderer is not None:
            try:
                return str(self.renderer(role_uuid, item) or "").strip()
            except Exception:  # noqa: BLE001 - fall back to the raw body
                pass
        if node_memory.is_fact_record(record):
            return node_memory.fact_body(record)
        if node_memory.is_event_record(record):
            return str(record.get("summary") or "").strip()
        return str(record.get("name") or "").strip()

    def _qualifies(self, metrics: Dict[str, Any]) -> tuple[bool, int]:
        min_hits = max(1, int(self._cfg("min_hit_rounds", 5)))
        min_batches = max(1, int(self._cfg("min_hit_batches", 2)))
        min_rate = float(self._cfg("min_batch_hit_rate", 0.30) or 0.30)
        metric = str(self._cfg("batch_rate_metric", "round") or "round").lower()
        batches = metrics.get("batches") or {}
        if metric == "coverage":
            total = len(batches)
            hit_batches = sum(1 for row in batches.values() if int(row.get("hit_rounds") or 0) > 0)
            rate = (hit_batches / total) if total else 0.0
            return (
                int(metrics.get("hit_rounds_7d") or 0) >= min_hits
                and hit_batches >= min_batches
                and rate > min_rate,
                hit_batches,
            )
        if metric == "pooled":
            hit_batches = sum(1 for row in batches.values() if int(row.get("hit_rounds") or 0) > 0)
            return (
                int(metrics.get("hit_rounds_7d") or 0) >= min_hits
                and hit_batches >= min_batches
                and float(metrics.get("pooled_rate") or 0.0) > min_rate,
                hit_batches,
            )
        qualifying = [
            row
            for row in batches.values()
            if int(row.get("hit_rounds") or 0) > 0 and float(row.get("rate") or 0.0) > min_rate
        ]
        return (
            int(metrics.get("hit_rounds_7d") or 0) >= min_hits
            and len(qualifying) >= min_batches,
            len(qualifying),
        )

    def _score(self, metrics: Dict[str, Any], weight: Optional[float], qualifying: int) -> float:
        floor = float(getattr(self.policy, "activation_weight_floor", 0.25))
        maximum = float(getattr(self.policy, "activation_weight_max", 1.0))
        span = max(1e-9, maximum - floor)
        weight_norm = max(0.0, min(1.0, ((weight if weight is not None else floor) - floor) / span))
        recency = math.exp(-max(0.0, float(metrics.get("_days_since_hit") or 0.0)) / 7.0)
        return (
            0.40 * float(metrics.get("pooled_rate") or 0.0)
            + 0.20 * min(1.0, qualifying / 4.0)
            + 0.25 * weight_norm
            + 0.15 * recency
        )

    def build(
        self,
        role_uuid: str,
        *,
        now: Optional[float] = None,
        last_output_at: Optional[float] = None,
        session_active: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        import time

        role_uuid = str(role_uuid)
        moment = time.time() if now is None else float(now)
        current = self.store.load(role_uuid)
        quiet = float(self._cfg("settle_quiet_seconds", self.store.quiet_seconds) or self.store.quiet_seconds)
        last_output = current.get("last_output_at") if last_output_at is None else last_output_at
        if session_active and not force:
            return {**current, "status": "unchanged", "reason": "session_active", "snapshot": current}
        if not force and (last_output is None or moment - float(last_output) <= quiet):
            return {**current, "status": "unchanged", "reason": "output_quiet_period", "snapshot": current}
        if (
            not force
            and current.get("recalculated_at")
            and last_output is not None
            and current.get("last_output_at") is not None
            and abs(float(current.get("last_output_at")) - float(last_output)) < 1e-6
        ):
            return {**current, "status": "unchanged", "reason": "already_settled", "snapshot": current}

        self.topology.reload()
        include_events = bool(self._cfg("include_events", False))
        max_age_days = float(self._cfg("max_age_days", 0) or 0)
        table = self.activation.load(role_uuid) if hasattr(self.activation, "load") else {"data": {}}
        objects = (table.get("data", {}) or {}).get("objects") or {}
        candidates: List[Dict[str, Any]] = []
        for object_id in sorted(objects):
            row = objects.get(object_id)
            if not isinstance(row, dict):
                continue
            object_type = str(row.get("object_type") or "")
            if object_type != "fact" and not (include_events and object_type == "event"):
                continue
            record = self._is_active_record(role_uuid, object_id)
            if record is None:
                continue
            metrics = self.activation.hot_zone_metrics(role_uuid, object_id, now=moment)
            if max_age_days > 0:
                last_hit = row.get("last_hit_at")
                if last_hit is not None and (moment - float(last_hit)) > max_age_days * 86400.0:
                    continue
            ok, qualifying = self._qualifies(metrics)
            if not ok:
                continue
            weight = metrics.get("weight")
            if weight is not None:
                last_hit = row.get("last_hit_at")
                metrics["_days_since_hit"] = (
                    max(0.0, (moment - float(last_hit)) / 86400.0) if last_hit is not None else 0.0
                )
            item = {
                "id": object_id,
                "object_type": object_type,
                "anchor_id": str(row.get("anchor_id") or record.get("parent_id") or ""),
                "hit_rounds_7d": int(metrics.get("hit_rounds_7d") or 0),
                "qualifying_batches": int(qualifying),
                "pooled_rate": round(float(metrics.get("pooled_rate") or 0.0), 6),
                "weight": round(float(weight), 6) if weight is not None else None,
                "last_hit_at": row.get("last_hit_at"),
                "score": round(self._score(metrics, weight, qualifying), 6),
                "_text": self._text_for(role_uuid, {"id": object_id}, record),
            }
            if item["_text"]:
                candidates.append(item)
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["id"])))

        token_budget = int(
            max(0.0, float(getattr(self.policy, "token_budget", 8000))) * float(self._cfg("token_ratio", 0.25))
        )
        per_anchor = max(0, int(self._cfg("max_items_per_anchor", 2) or 0))
        selected: List[Dict[str, Any]] = []
        used = 0
        anchor_counts: Dict[str, int] = {}
        for item in candidates:
            anchor = str(item.get("anchor_id") or "")
            if per_anchor > 0 and anchor and anchor_counts.get(anchor, 0) >= per_anchor:
                continue
            text = str(item.pop("_text", "") or "")
            cost = max(0, int(self.token_counter(text)))
            if selected and used + cost > token_budget:
                continue
            if not selected and cost > token_budget:
                return {
                    **current,
                    "status": "error",
                    "reason": "object_exceeds_hot_zone_budget",
                    "object_id": item.get("id"),
                    "token_estimate": cost,
                    "token_budget": token_budget,
                    "snapshot": current,
                }
            item["text"] = text
            item["token_estimate"] = cost
            selected.append(item)
            used += cost
            if anchor:
                anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        updated = {
            **current,
            "version": int(current.get("version") or 0) + 1,
            "status": "active" if selected else "empty",
            "objects": selected,
            "last_output_at": float(last_output) if last_output is not None else current.get("last_output_at"),
            "recalculated_at": utc_timestamp(),
            "policy_version": str(getattr(self.policy, "version", "") or current.get("policy_version") or ""),
        }
        self.store.save(role_uuid, updated)
        return {**copy.deepcopy(updated), "status": "updated", "snapshot": copy.deepcopy(updated)}
