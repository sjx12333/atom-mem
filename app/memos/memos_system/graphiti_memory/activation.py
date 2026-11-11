"""Role-scoped activation weights for nodes, relations and facts.

Every memory object a role can recall may carry an **activation weight**: a
number in `[floor, max]` that grows when the object is directly recalled and
decays with an Ebbinghaus-style curve while it is not. The weight is a *ranking
preference* only; admission gates never read it (see `node_recall._apply_activation`).

The file is an authoritative per-role sidecar (`characters/<role>/activation.json`)
because hit history cannot be derived from the memory it describes. It records:

* `batches` — a role-level ledger of completed recall rounds per chat batch, used
  as the denominator of the hot-zone "batch hit rate";
* `objects` — one entry per recalled object with its weight, cumulative direct
  hits, and per-batch hit counts.

Hot-zone admission is intentionally *not* computed here: this module only keeps
the facts. `hot_zone.HotZoneBuilder` applies the policy (7-day window, >=5 hits,
>=2 qualifying batches, facts only).
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .authority import AuthorityStore
from .domain import utc_timestamp

DEFAULT_WINDOW_DAYS = 7.0
DEFAULT_FLOOR = 0.25
DEFAULT_MAX = 1.0
DEFAULT_INITIAL = 0.5
DEFAULT_ALPHA0 = 0.40
DEFAULT_RHO = 0.80
DEFAULT_ALPHA_MIN = 0.04
DEFAULT_HALF_LIFE_HOURS = 168.0
DEFAULT_HALF_LIFE_GROWTH = 1.2
DEFAULT_EDGE_GAIN = 0.5

#: Object types that may carry activation. `hook` is deliberately absent: it is a
#: model prediction, not a recalled memory object.
ACTIVATION_OBJECT_TYPES = ("node", "relation", "fact", "event", "predicate")
ACTIVATION_LAYER = "GraphitiMemory"

_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class ActivationError(RuntimeError):
    """The authoritative activation file is unreadable or invalid."""


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve()).casefold()
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def _finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def decay_weight(
    weight: float,
    hits: int,
    elapsed_seconds: float,
    *,
    floor: float = DEFAULT_FLOOR,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    growth: float = DEFAULT_HALF_LIFE_GROWTH,
) -> float:
    """Apply the Ebbinghaus decay, never going below `floor`.

    `half_life_hours` is the *base* half-life (at zero hits); each hit stretches
    it by `1 + growth * ln(1 + hits)`. The stored value is a snapshot, so this is
    always evaluated lazily against the wall clock.
    """

    if weight <= floor:
        return floor
    hits = max(0, int(hits))
    base = max(0.0, _finite_float(half_life_hours, DEFAULT_HALF_LIFE_HOURS))
    growth = max(0.0, _finite_float(growth, DEFAULT_HALF_LIFE_GROWTH))
    half_life_seconds = max(1.0, base * 3600.0 * (1.0 + growth * math.log1p(hits)))
    # exp(-t/S) with S = half_life/ln2 is exactly 0.5 at t=half_life.
    time_constant = half_life_seconds / math.log(2.0)
    elapsed = max(0.0, _finite_float(elapsed_seconds, 0.0))
    value = floor + (weight - floor) * math.exp(-elapsed / time_constant)
    return max(floor, min(weight, value))


def reinforce_weight(
    weight: float,
    hits: int,
    *,
    floor: float = DEFAULT_FLOOR,
    max_weight: float = DEFAULT_MAX,
    alpha0: float = DEFAULT_ALPHA0,
    rho: float = DEFAULT_RHO,
    alpha_min: float = DEFAULT_ALPHA_MIN,
) -> float:
    """One direct hit: a geometrically decreasing gain towards `max_weight`."""

    floor = _finite_float(floor, DEFAULT_FLOOR)
    max_weight = max(floor, _finite_float(max_weight, DEFAULT_MAX))
    hits = max(1, int(hits))
    alpha0 = min(1.0, max(0.0, _finite_float(alpha0, DEFAULT_ALPHA0)))
    rho = min(1.0, max(0.0, _finite_float(rho, DEFAULT_RHO)))
    alpha_min = min(1.0, max(0.0, _finite_float(alpha_min, DEFAULT_ALPHA_MIN)))
    alpha = max(alpha_min, alpha0 * (rho ** (hits - 1)))
    current = max(floor, min(max_weight, _finite_float(weight, floor)))
    return min(max_weight, current + (max_weight - current) * alpha)


def weight_norm(weight: Optional[float], *, floor: float = DEFAULT_FLOOR, max_weight: float = DEFAULT_MAX) -> float:
    if weight is None:
        return 0.0
    span = max(1e-9, max_weight - floor)
    return max(0.0, min(1.0, (float(weight) - floor) / span))


def _empty_table(role_uuid: str, memory_generation: int = 0) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "revision": 0,
        "role_uuid": str(role_uuid),
        "memory_generation": int(memory_generation or 0),
        "updated_at": utc_timestamp(),
        "data": {"batches": {}, "objects": {}},
    }


class ActivationStore:
    """Read and write one role's authoritative activation sidecar."""

    def __init__(
        self,
        authority: AuthorityStore,
        *,
        policy: Any = None,
        window_days: float = DEFAULT_WINDOW_DAYS,
        clock=time.time,
    ):
        self.authority = authority
        self.policy = policy
        self.window_days = max(1.0, _finite_float(window_days, DEFAULT_WINDOW_DAYS))
        self.clock = clock

    # ------------------------------------------------------------------ settings

    def _setting(self, name: str, default: float) -> float:
        if self.policy is None:
            return default
        return _finite_float(getattr(self.policy, name, default), default)

    @property
    def floor(self) -> float:
        return self._setting("activation_weight_floor", DEFAULT_FLOOR)

    @property
    def max_weight(self) -> float:
        return max(self.floor, self._setting("activation_weight_max", DEFAULT_MAX))

    @property
    def initial(self) -> float:
        return max(self.floor, min(self.max_weight, self._setting("activation_initial_weight", DEFAULT_INITIAL)))

    @property
    def alpha0(self) -> float:
        return self._setting("activation_gain_alpha", DEFAULT_ALPHA0)

    @property
    def rho(self) -> float:
        return self._setting("activation_gain_decay", DEFAULT_RHO)

    @property
    def alpha_min(self) -> float:
        return self._setting("activation_gain_min", DEFAULT_ALPHA_MIN)

    @property
    def half_life_hours(self) -> float:
        return self._setting("activation_half_life_hours", DEFAULT_HALF_LIFE_HOURS)

    @property
    def half_life_growth(self) -> float:
        return self._setting("activation_half_life_growth", DEFAULT_HALF_LIFE_GROWTH)

    @property
    def edge_gain_ratio(self) -> float:
        return self._setting("activation_edge_gain_ratio", DEFAULT_EDGE_GAIN)

    # ------------------------------------------------------------------ storage

    def path_for(self, role_uuid: str) -> Path:
        return self.authority.role_dir(role_uuid) / "activation.json"

    def _load(self, role_uuid: str) -> Dict[str, Any]:
        raw = self.authority.load_role_file(role_uuid, "activation.json", None)
        if raw is None:
            return _empty_table(role_uuid)
        if not isinstance(raw, dict):
            raise ActivationError("activation.json must contain an object")
        if int(raw.get("schema_version") or 0) != 1:
            raise ActivationError(f"unsupported activation schema: {raw.get('schema_version')}")
        if str(raw.get("role_uuid") or role_uuid) != str(role_uuid):
            raise ActivationError("activation.json belongs to another role")
        data = raw.get("data")
        if not isinstance(data, dict):
            raise ActivationError("activation.json data must contain an object")
        batches = data.get("batches")
        objects = data.get("objects")
        if not isinstance(batches, dict) or not isinstance(objects, dict):
            raise ActivationError("activation.json requires batches and objects objects")
        return raw

    def _save(self, role_uuid: str, table: Dict[str, Any]) -> None:
        table["revision"] = int(table.get("revision") or 0) + 1
        table["updated_at"] = utc_timestamp()
        self.authority.save_role_file(role_uuid, "activation.json", table)

    def load(self, role_uuid: str) -> Dict[str, Any]:
        """Return a detached copy of the role's table (empty when absent)."""

        with _lock_for(self.path_for(role_uuid)):
            table = self._load(role_uuid)
        return table

    def _prune(self, table: Dict[str, Any], now: float) -> None:
        cutoff = float(now) - self.window_days * 86400.0
        data = table.setdefault("data", {})
        batches = data.setdefault("batches", {})
        for batch_id in list(batches):
            record = batches.get(batch_id)
            if not isinstance(record, dict) or _finite_float(record.get("last_at"), 0.0) < cutoff:
                batches.pop(batch_id, None)
        objects = data.setdefault("objects", {})
        for object_id in list(objects):
            row = objects.get(object_id)
            if not isinstance(row, dict):
                objects.pop(object_id, None)
                continue
            hit_batches = row.get("hit_batches")
            if not isinstance(hit_batches, dict):
                hit_batches = {}
                row["hit_batches"] = hit_batches
            for batch_id in list(hit_batches):
                if batch_id not in batches:
                    hit_batches.pop(batch_id, None)

    def _current_weight(self, row: Mapping[str, Any], now: float) -> float:
        saved = _finite_float(row.get("weight"), self.initial)
        saved_at = _finite_float(row.get("weight_evaluated_at"), float(now))
        hits = int(row.get("direct_hit_rounds") or 0)
        return decay_weight(
            saved,
            hits,
            max(0.0, float(now) - saved_at),
            floor=self.floor,
            half_life_hours=self.half_life_hours,
            growth=self.half_life_growth,
        )

    # ------------------------------------------------------------------- reads

    def weight_for(self, role_uuid: str, object_id: str, *, now: Optional[float] = None) -> Optional[float]:
        """Current (lazily decayed) weight, or `None` when never activated."""

        object_id = str(object_id or "")
        if not object_id:
            return None
        moment = self.clock() if now is None else float(now)
        table = self.load(role_uuid)
        row = (table.get("data", {}).get("objects") or {}).get(object_id)
        if not isinstance(row, dict):
            return None
        return self._current_weight(row, moment)

    def weights_for(
        self, role_uuid: str, object_ids: Iterable[str], *, now: Optional[float] = None
    ) -> Dict[str, float]:
        moment = self.clock() if now is None else float(now)
        table = self.load(role_uuid)
        objects = table.get("data", {}).get("objects") or {}
        result: Dict[str, float] = {}
        for object_id in object_ids:
            row = objects.get(str(object_id))
            if isinstance(row, dict):
                result[str(object_id)] = self._current_weight(row, moment)
        return result

    def edge_weight_for(self, role_uuid: str, relation_id: str, *, now: Optional[float] = None) -> float:
        """Navigation multiplier for one edge; 1.0 when never activated."""

        weight = self.weight_for(role_uuid, relation_id, now=now)
        return 1.0 if weight is None else max(0.0, min(1.0, weight))

    def hot_zone_metrics(self, role_uuid: str, object_id: str, *, now: Optional[float] = None) -> Dict[str, Any]:
        """Windowed hot-zone metrics for one object.

        The default metric is per-batch: a batch qualifies when the object was
        directly hit at least once and `hits / recall_rounds > 30%`. Callers may
        override the rate/threshold through the policy in `hot_zone`.
        """

        moment = self.clock() if now is None else float(now)
        table = self.load(role_uuid)
        data = table.get("data", {})
        batches = data.get("batches") or {}
        row = (data.get("objects") or {}).get(str(object_id)) or {}
        hit_batches = row.get("hit_batches") if isinstance(row.get("hit_batches"), dict) else {}
        cutoff = moment - self.window_days * 86400.0
        windows: Dict[str, Dict[str, Any]] = {}
        for batch_id, batch in batches.items():
            if not isinstance(batch, dict):
                continue
            if _finite_float(batch.get("last_at"), 0.0) < cutoff:
                continue
            rounds = max(0, int(batch.get("recall_rounds") or 0))
            if rounds <= 0:
                continue
            hits = max(0, int(hit_batches.get(batch_id) or 0))
            windows[str(batch_id)] = {
                "recall_rounds": rounds,
                "hit_rounds": hits,
                "rate": hits / rounds,
            }
        total_hits = sum(item["hit_rounds"] for item in windows.values())
        total_rounds = sum(item["recall_rounds"] for item in windows.values())
        return {
            "object_id": str(object_id),
            "weight": self._current_weight(row, moment) if row else None,
            "hit_rounds_7d": total_hits,
            "recall_rounds_7d": total_rounds,
            "hit_batches": sum(1 for item in windows.values() if item["hit_rounds"] > 0),
            "batches": windows,
            "pooled_rate": (total_hits / total_rounds) if total_rounds else 0.0,
        }

    # ------------------------------------------------------------------ writes

    def record_recall(
        self,
        role_uuid: str,
        memories: Sequence[Mapping[str, Any]],
        *,
        batch_id: str,
        traversed_relations: Sequence[str] = (),
        now: Optional[float] = None,
        count: bool = True,
    ) -> Dict[str, Any]:
        """Record one completed recall request.

        A memory object counts only when it came from the graph layer, was a
        **direct** hit (not a path/expansion result), and survived dedupe. The
        batch ledger grows by one recall round even when nothing was hit, so a
        zero-recall round still contributes to the denominator.
        """

        if not count:
            return {"status": "disabled", "batch_id": str(batch_id), "hit_objects": 0, "traversed_relations": 0}
        role_uuid = str(role_uuid or "")
        if not role_uuid:
            raise ValueError("role_uuid is required")
        batch = str(batch_id or "").strip() or "unbatched"
        moment = self.clock() if now is None else float(now)
        path = self.path_for(role_uuid)
        with _lock_for(path):
            table = self._load(role_uuid)
            self._prune(table, moment)
            data = table.setdefault("data", {})
            batches = data.setdefault("batches", {})
            batch_row = batches.setdefault(
                batch,
                {"recall_rounds": 0, "first_at": moment, "last_at": moment},
            )
            batch_row["recall_rounds"] = int(batch_row.get("recall_rounds") or 0) + 1
            batch_row["last_at"] = moment
            batch_row.setdefault("first_at", moment)
            objects = data.setdefault("objects", {})
            seen: set[str] = set()
            hit_objects: list[dict[str, Any]] = []
            for item in memories:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("layer") or "") != ACTIVATION_LAYER:
                    continue
                if str(item.get("match_type") or "direct") != "direct":
                    continue
                object_id = str(item.get("id") or "")
                object_type = str(item.get("object_type") or item.get("source_type") or "")
                if not object_id or object_type not in ACTIVATION_OBJECT_TYPES or object_id in seen:
                    continue
                seen.add(object_id)
                row = objects.get(object_id)
                if not isinstance(row, dict):
                    row = {
                        "object_type": object_type,
                        "anchor_id": str(item.get("entity_id") or item.get("parent_id") or ""),
                        "weight": self.initial,
                        "weight_evaluated_at": moment,
                        "direct_hit_rounds": 0,
                        "hit_batches": {},
                        "traversal_rounds": 0,
                        "first_hit_at": None,
                        "last_hit_at": None,
                        "last_settled_at": None,
                    }
                    objects[object_id] = row
                current = self._current_weight(row, moment)
                row["direct_hit_rounds"] = int(row.get("direct_hit_rounds") or 0) + 1
                row["weight"] = reinforce_weight(
                    current,
                    int(row["direct_hit_rounds"]),
                    floor=self.floor,
                    max_weight=self.max_weight,
                    alpha0=self.alpha0,
                    rho=self.rho,
                    alpha_min=self.alpha_min,
                )
                row["weight_evaluated_at"] = moment
                row["last_hit_at"] = moment
                if not row.get("first_hit_at"):
                    row["first_hit_at"] = moment
                hit_batches = row.setdefault("hit_batches", {})
                hit_batches[batch] = int(hit_batches.get(batch) or 0) + 1
                hit_objects.append(
                    {"id": object_id, "object_type": object_type, "weight": row["weight"]}
                )
            touched_edges: list[str] = []
            for relation_id in traversed_relations:
                relation_id = str(relation_id or "")
                if not relation_id or relation_id in seen:
                    continue
                row = objects.get(relation_id)
                if not isinstance(row, dict):
                    row = {
                        "object_type": "relation",
                        "anchor_id": "",
                        "weight": self.initial,
                        "weight_evaluated_at": moment,
                        "direct_hit_rounds": 0,
                        "hit_batches": {},
                        "traversal_rounds": 0,
                        "first_hit_at": None,
                        "last_hit_at": None,
                        "last_settled_at": None,
                    }
                    objects[relation_id] = row
                current = self._current_weight(row, moment)
                row["traversal_rounds"] = int(row.get("traversal_rounds") or 0) + 1
                row["weight"] = reinforce_weight(
                    current,
                    int(row["traversal_rounds"]) + int(row.get("direct_hit_rounds") or 0),
                    floor=self.floor,
                    max_weight=self.max_weight,
                    alpha0=self.alpha0 * max(0.0, min(1.0, self.edge_gain_ratio)),
                    rho=self.rho,
                    alpha_min=self.alpha_min,
                )
                row["weight_evaluated_at"] = moment
                touched_edges.append(relation_id)
            self._save(role_uuid, table)
        return {
            "status": "ok",
            "batch_id": batch,
            "hit_objects": len(hit_objects),
            "traversed_relations": len(set(touched_edges)),
            "boosted": hit_objects,
        }

    def reset(self, role_uuid: str, *, actor: str = "local_management") -> Dict[str, Any]:
        role_uuid = str(role_uuid or "")
        if not role_uuid:
            raise ValueError("role_uuid is required")
        table = _empty_table(role_uuid, int(self.authority.role_state(role_uuid).get("memory_generation") or 0))
        path = self.path_for(role_uuid)
        with _lock_for(path):
            self._save(role_uuid, table)
        return {"role_uuid": role_uuid, "actor": actor, "status": "reset"}

    def prune(self, role_uuid: str, existing_ids: Iterable[str]) -> int:
        keep = {str(item) for item in existing_ids if item}
        path = self.path_for(role_uuid)
        removed = 0
        with _lock_for(path):
            table = self._load(role_uuid)
            objects = table.setdefault("data", {}).setdefault("objects", {})
            for object_id in list(objects):
                if object_id not in keep:
                    objects.pop(object_id, None)
                    removed += 1
            if removed:
                self._save(role_uuid, table)
        return removed

    def status(self, role_uuid: str, *, now: Optional[float] = None) -> Dict[str, Any]:
        moment = self.clock() if now is None else float(now)
        table = self.load(role_uuid)
        data = table.get("data", {})
        objects = data.get("objects") or {}
        return {
            "role_uuid": str(role_uuid),
            "schema_version": table.get("schema_version"),
            "revision": table.get("revision"),
            "window_days": self.window_days,
            "batches": len(data.get("batches") or {}),
            "objects": len(objects),
            "weights": {
                object_id: round(self._current_weight(row, moment), 6)
                for object_id, row in objects.items()
                if isinstance(row, dict)
            },
        }
