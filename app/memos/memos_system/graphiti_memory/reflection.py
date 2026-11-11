"""Idle reflection and the persona overlay it produces.

The AI writes a short self-observation during an idle window (never during a
conversation), using its own diary, the memories it keeps recalling, and the
current persona. The result is **not** a new Prompt version: it is a separate,
immutable `persona overlay` that is compiled next to the base prompt at session
start. That keeps user-authored `prompt/versions/*` untouched and makes the
overlay easy to pause, edit, pin, or roll back.

Scheduling lives in the background sweep (one role per tick). The gate is:
enabled + active role + no live session + quiet for the configured idle window
+ interval elapsed + new material since the last reflection.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .domain import typed_id, utc_timestamp


KINDS = ("style", "relationship", "preference", "self_observation")
MAX_ITEM_CHARS = 60
MAX_TOTAL_CHARS = 240
MAX_OVERLAY_ITEMS = 12
MAX_OVERLAY_TOKENS = 400

_BANNED = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"根据(?:记忆|系统|数据库)|记忆系统|数据库"),
    re.compile(r"你之前说过|你曾经说过"),
    re.compile(r"你应该|你必须"),
)


def _now() -> float:
    return time.time()


def _iso_to_epoch(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _grams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", str(text or "")).casefold()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[index:index + 3] for index in range(len(normalized) - 2)}


def _too_similar(text: str, base: str) -> bool:
    left = _grams(text)
    right = _grams(base)
    if not left or not right:
        return False
    union = left | right
    return bool(union) and (len(left & right) / len(union)) > 0.8


class OverlayStore:
    """Versioned, immutable persona overlays under the role's prompt directory."""

    def __init__(self, authority: Any):
        self.authority = authority

    def _pointer_name(self) -> str:
        return "prompt/overlay_current.json"

    def _version_name(self, overlay_id: str) -> str:
        return f"prompt/overlays/{overlay_id}.json"

    def pointer(self, role_uuid: str) -> Dict[str, Any]:
        raw = self.authority.load_role_file(role_uuid, self._pointer_name(), {})
        if not isinstance(raw, dict):
            return {}
        return dict(raw)

    def _save_pointer(self, role_uuid: str, pointer: Dict[str, Any]) -> Dict[str, Any]:
        pointer = {**pointer, "updated_at": utc_timestamp()}
        self.authority.save_role_file(role_uuid, self._pointer_name(), pointer)
        return pointer

    def versions(self, role_uuid: str) -> List[Dict[str, Any]]:
        directory = self.authority.role_dir(role_uuid) / "prompt" / "overlays"
        if not directory.exists():
            return []
        items: List[Dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            try:
                value = self.authority._read(path, {})
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, dict):
                items.append(value)
        return items

    def active(self, role_uuid: str) -> Optional[Dict[str, Any]]:
        pointer = self.pointer(role_uuid)
        overlay_id = str(pointer.get("active_overlay_id") or "")
        if not overlay_id:
            return None
        raw = self.authority.load_role_file(role_uuid, self._version_name(overlay_id), {})
        if not isinstance(raw, dict) or not raw.get("items"):
            return None
        return raw

    def active_text(self, role_uuid: str) -> str:
        pointer = self.pointer(role_uuid)
        if bool(pointer.get("paused")):
            return ""
        overlay = self.active(role_uuid)
        return str((overlay or {}).get("content") or "").strip()

    @staticmethod
    def compile(items: Sequence[Dict[str, Any]]) -> str:
        return "\n".join(
            f"- {str(item.get('text') or '').strip()}"
            for item in items
            if str(item.get("text") or "").strip()
        )

    def save(
        self,
        role_uuid: str,
        items: Sequence[Dict[str, Any]],
        *,
        base_prompt_version_id: str,
        created_by: str = "reflection",
        auto_activate: bool = True,
        token_counter=None,
    ) -> Dict[str, Any]:
        cleaned = [dict(item) for item in items if str(item.get("text") or "").strip()]
        content = self.compile(cleaned)
        token_count = int((token_counter or len)(content))
        overlay_id = typed_id("prompt")
        overlay = {
            "schema_version": 1,
            "overlay_version_id": overlay_id,
            "role_uuid": role_uuid,
            "base_prompt_version_id": str(base_prompt_version_id or ""),
            "items": cleaned,
            "content": content,
            "token_count": token_count,
            "tokenizer_id": "default",
            "created_by": str(created_by or "reflection"),
            "created_at": utc_timestamp(),
            "supersedes": str(self.pointer(role_uuid).get("active_overlay_id") or "") or None,
        }
        self.authority.save_role_file(role_uuid, self._version_name(overlay_id), overlay)
        pointer = self.pointer(role_uuid)
        pointer.update(
            {
                "base_prompt_version_id": str(base_prompt_version_id or ""),
                "needs_review": False,
            }
        )
        # A user pause is an explicit override: a later automatic reflection
        # may record a new version but must not silently un-pause it.
        if auto_activate:
            paused = bool(pointer.get("paused"))
            if created_by != "reflection" or not paused:
                pointer["active_overlay_id"] = overlay_id
                pointer["paused"] = False
        self._save_pointer(role_uuid, pointer)
        return overlay

    def set_paused(self, role_uuid: str, paused: bool) -> Dict[str, Any]:
        pointer = self.pointer(role_uuid)
        pointer["paused"] = bool(paused)
        return self._save_pointer(role_uuid, pointer)

    def activate(self, role_uuid: str, overlay_id: str) -> Dict[str, Any]:
        raw = self.authority.load_role_file(role_uuid, self._version_name(overlay_id), {})
        if not isinstance(raw, dict) or not raw.get("items"):
            raise KeyError(overlay_id)
        pointer = self.pointer(role_uuid)
        pointer["active_overlay_id"] = overlay_id
        pointer["paused"] = False
        pointer["needs_review"] = False
        return self._save_pointer(role_uuid, pointer)

    def edit(self, role_uuid: str, items: Sequence[Dict[str, Any]], *, base_prompt_version_id: str = "") -> Dict[str, Any]:
        return self.save(
            role_uuid,
            items,
            base_prompt_version_id=base_prompt_version_id,
            created_by="user",
            auto_activate=True,
        )

    def set_locked(self, role_uuid: str, item_id: str, locked: bool) -> Dict[str, Any]:
        pointer = self.pointer(role_uuid)
        overlay_id = str(pointer.get("active_overlay_id") or "")
        if not overlay_id:
            raise KeyError(item_id)
        raw = self.authority.load_role_file(role_uuid, self._version_name(overlay_id), {})
        if not isinstance(raw, dict):
            raise KeyError(item_id)
        items = [dict(item) for item in (raw.get("items") or []) if isinstance(item, dict)]
        found = False
        for item in items:
            if str(item.get("id") or "") == str(item_id):
                item["locked"] = bool(locked)
                found = True
                break
        if not found:
            raise KeyError(item_id)
        return self.save(
            role_uuid,
            items,
            base_prompt_version_id=str(raw.get("base_prompt_version_id") or ""),
            created_by="user",
            auto_activate=True,
        )


class ReflectionWriter:
    """Call the role's model and validate one reflection answer."""

    def __init__(self, client: Any, *, max_items: int = 3, token_counter=None):
        self.client = client
        self.max_items = max(1, int(max_items))
        self.token_counter = token_counter or len

    def build_prompt(
        self,
        *,
        display_name: str,
        persona: str,
        previous: str,
        diaries: Sequence[Dict[str, Any]],
        hot_zone: Sequence[Dict[str, Any]],
        memories: Sequence[Dict[str, Any]],
    ) -> str:
        human = display_name or "你"
        lines = [
            f"你是「{human}」。现在没有对话，请做一次安静的自我反思。",
            "只根据下面的日记、最近常想起的记忆和当前人设，总结 1-3 条：",
            "- 我说话的方式/节奏/用词倾向；",
            "- 我对人与关系的态度变化；",
            "- 我最近反复在意/回避的主题（只写倾向，不写具体事件）。",
            "不要新增事实、事件、约定、人物、日期；不要复述记忆原文；不要提“记忆/系统/数据库”；",
            "不要给用户下指令。每条不超过 60 字，用第一人称。",
            '输出 JSON：{"items":[{"text":"...","kind":"style|relationship|preference|self_observation","evidence":["diary:...","f_..."]}]}',
        ]
        if persona:
            lines.append("\n【当前人设】\n" + str(persona)[:2000])
        if previous:
            lines.append("\n【上一版补注（仅保持连续性，不是证据）】\n" + str(previous)[:400])
        if diaries:
            lines.append("\n【最近的日记】")
            for entry in diaries[-5:]:
                lines.append(f"[{entry.get('id')}] {str(entry.get('text') or '')[:200]}")
        if hot_zone:
            lines.append("\n【热区里常出现的事实】")
            for item in hot_zone[:8]:
                lines.append(f"[{item.get('id')}] {str(item.get('text') or '')[:80]}")
        if memories:
            lines.append("\n【最近被反复想起的记忆】")
            for item in memories[:10]:
                lines.append(f"[{item.get('id')}] {str(item.get('text') or '')[:80]}")
        return "\n".join(lines)

    def _validate(
        self,
        payload: Any,
        *,
        allowed_evidence: set[str],
        corpus: str,
        persona: str,
    ) -> tuple[bool, str, List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return False, "not_an_object", []
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return False, "empty_items", []
        if len(raw_items) > self.max_items:
            return False, "too_many_items", []
        items: List[Dict[str, Any]] = []
        total = 0
        for raw in raw_items:
            if not isinstance(raw, dict):
                return False, "item_not_an_object", []
            text = " ".join(str(raw.get("text") or "").split())
            kind = str(raw.get("kind") or "")
            evidence = raw.get("evidence")
            if not text or len(text) > MAX_ITEM_CHARS:
                return False, "item_length", []
            if kind not in KINDS:
                return False, "bad_kind", []
            if not isinstance(evidence, list) or not evidence:
                return False, "missing_evidence", []
            refs = [str(item) for item in evidence if str(item)]
            if len(refs) != len(evidence) or any(ref not in allowed_evidence for ref in refs):
                return False, "unknown_evidence", []
            for pattern in _BANNED:
                if pattern.search(text):
                    return False, "banned_phrase", []
            for quoted in re.findall(r"[「“\"]([^」”\"]{1,30})[」”\"]", text):
                # A reflection may name something that was actually in its evidence.
                if quoted and quoted not in corpus:
                    return False, "unknown_proper_noun", []
            if _too_similar(text, persona):
                return False, "duplicates_persona", []
            total += len(text)
            items.append(
                {
                    "id": typed_id("operation"),
                    "text": text,
                    "kind": kind,
                    "evidence": refs,
                    "locked": False,
                    "created_at": utc_timestamp(),
                }
            )
        if total > MAX_TOTAL_CHARS:
            return False, "total_length", []
        return True, "ok", items

    def reflect(
        self,
        *,
        display_name: str,
        persona: str,
        previous: str,
        diaries: Sequence[Dict[str, Any]],
        hot_zone: Sequence[Dict[str, Any]],
        memories: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.client is None or not getattr(self.client, "configured", False):
            return {"status": "skipped", "reason": "reflection writer is not configured"}
        prompt = self.build_prompt(
            display_name=display_name,
            persona=persona,
            previous=previous,
            diaries=diaries,
            hot_zone=hot_zone,
            memories=memories,
        )
        allowed = {str(item.get("id")) for item in diaries if item.get("id")}
        allowed.update(str(item.get("id")) for item in hot_zone if item.get("id"))
        allowed.update(str(item.get("id")) for item in memories if item.get("id"))
        corpus = "\n".join(
            str(item.get("text") or "")
            for item in list(diaries) + list(hot_zone) + list(memories)
        )
        raw = None
        last_error = ""
        for _ in range(2):
            try:
                raw = self.client.complete(
                    prompt,
                    system="你是一个只输出 JSON 的角色自我反思器。",
                    json_object=True,
                    temperature=0.7,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError) as exc:
                last_error = f"invalid_json: {exc}"
                continue
            ok, reason, items = self._validate(
                payload, allowed_evidence=allowed, corpus=corpus, persona=persona
            )
            if ok:
                return {"status": "written", "items": items}
            return {"status": "rejected", "reason": reason}
        return {"status": "failed", "error": last_error or "no_response"}


class ReflectionScheduler:
    """One role per background sweep, with a persistent cursor and backoff."""

    def __init__(self, service: Any, *, writer: Optional[ReflectionWriter] = None, overlays: Optional[OverlayStore] = None):
        self.service = service
        self.writer = writer or ReflectionWriter(
            getattr(service, "diary", None).client if getattr(service, "diary", None) else None,
            max_items=int((service.runtime_config.get("reflection", {}) or {}).get("max_items", 3) or 3),
            token_counter=getattr(service.gateway.node_recall, "token_counter", len),
        )
        self.overlays = overlays or OverlayStore(service.authority)

    # ------------------------------------------------------------------ state

    def _config(self) -> Dict[str, Any]:
        return dict(self.service.runtime_config.get("reflection", {}) or {})

    def _load_state(self, role_uuid: str) -> Dict[str, Any]:
        raw = self.service.authority.load_role_file(role_uuid, "task_state.json", {})
        if not isinstance(raw, dict):
            return {}
        state = raw.get("reflection")
        return dict(state) if isinstance(state, dict) else {}

    def _save_state(self, role_uuid: str, value: Dict[str, Any]) -> None:
        raw = self.service.authority.load_role_file(role_uuid, "task_state.json", {})
        if not isinstance(raw, dict):
            raw = {}
        raw["reflection"] = dict(value, updated_at=utc_timestamp())
        self.service.authority.save_role_file(role_uuid, "task_state.json", raw)

    # ------------------------------------------------------------------ gating

    def _role_last_output(self, role_uuid: str) -> Optional[float]:
        helper = getattr(self.service, "_role_last_output_epoch", None)
        return helper(role_uuid) if callable(helper) else None

    def _has_new_material(self, role_uuid: str, since: Optional[float]) -> bool:
        diary = getattr(self.service, "diary", None)
        if diary is not None:
            for entry in diary.entries_for(role_uuid):
                moment = _iso_to_epoch(entry.get("created_at"))
                if since is None or (moment is not None and moment > since):
                    return True
        try:
            table = self.service.activations.load(role_uuid)
        except Exception:  # noqa: BLE001
            return False
        for row in (table.get("data", {}).get("objects") or {}).values():
            if not isinstance(row, dict):
                continue
            moment = row.get("last_hit_at")
            if since is None or (isinstance(moment, (int, float)) and float(moment) > since):
                return True
        return False

    def _eligible(self, role_uuid: str, now: float, state: Dict[str, Any], config: Dict[str, Any]) -> tuple[bool, str]:
        if not bool(config.get("enabled", True)):
            return False, "disabled"
        runtime_state = self.service.authority.role_state(role_uuid)
        if str(runtime_state.get("lifecycle") or "active") != "active":
            return False, "role_not_active"
        if getattr(self.service, "maintenance", None) is not None:
            if self.service.maintenance.status().get("status") != "clear":
                return False, "maintenance"
        if self.service._role_has_active_session(role_uuid):
            return False, "session_active"
        interval = max(60.0, float(config.get("interval_seconds", 86400) or 86400))
        next_due = float(state.get("next_due_at") or 0.0)
        if now < next_due:
            return False, "not_due"
        quiet = max(
            float(config.get("min_idle_seconds", 3600) or 3600),
            float((self.service.runtime_config.get("hot_zone", {}) or {}).get("settle_quiet_seconds", 3600) or 3600),
        )
        last_output = self._role_last_output(role_uuid)
        if last_output is None or now - last_output < quiet:
            return False, "not_idle"
        if not self._has_new_material(role_uuid, state.get("last_run_at")):
            return False, "no_new_material"
        return True, "ok"

    # ------------------------------------------------------------------ inputs

    def _persona(self, role_uuid: str) -> tuple[str, str, str]:
        prompt = self.service._management_prompt(role_uuid)
        content = str(prompt.get("content") or "")
        persona = str(prompt.get("persona") or content)
        version_id = str(prompt.get("current_version_id") or prompt.get("prompt_version_id") or "")
        return content, persona, version_id

    def _diaries(self, role_uuid: str, since: Optional[float]) -> List[Dict[str, Any]]:
        entries = getattr(self.service.diary, "entries_for", lambda _role: [])(role_uuid)
        if since is None:
            return list(entries)[-5:]
        return [
            entry
            for entry in entries
            if (_iso_to_epoch(entry.get("created_at")) or 0.0) > since
        ][-5:]

    def _hot_zone(self, role_uuid: str) -> List[Dict[str, Any]]:
        snapshot = self.service.hot_zones.freeze(
            role_uuid, renderer=getattr(self.service, "_render_hot_zone_item", None)
        )
        return [dict(item) for item in (snapshot.get("objects") or []) if isinstance(item, dict)][:8]

    def _memories(self, role_uuid: str) -> List[Dict[str, Any]]:
        try:
            table = self.service.activations.load(role_uuid)
        except Exception:  # noqa: BLE001
            return []
        objects = (table.get("data", {}).get("objects") or {})
        ranked: List[tuple[float, str]] = []
        for object_id, row in objects.items():
            if not isinstance(row, dict):
                continue
            try:
                weight = self.service.activations.weight_for(role_uuid, object_id)
            except Exception:  # noqa: BLE001
                continue
            if weight is not None and weight > self.service.activations.floor:
                ranked.append((float(weight), object_id))
        ranked.sort(reverse=True)
        result: List[Dict[str, Any]] = []
        for _, object_id in ranked[:10]:
            try:
                self.service.topology.reload()
                record = self.service.topology.find_record(object_id)
            except Exception:  # noqa: BLE001
                record = None
            if not record:
                continue
            text = self.service._render_hot_zone_item(role_uuid, {"id": object_id})
            if text:
                result.append({"id": object_id, "text": text})
        return result

    # ------------------------------------------------------------------ action

    def run(self, role_uuid: str, *, now: Optional[float] = None, force: bool = False) -> Dict[str, Any]:
        role_uuid = str(role_uuid)
        moment = _now() if now is None else float(now)
        config = self._config()
        state = self._load_state(role_uuid)
        if not force:
            ok, reason = self._eligible(role_uuid, moment, state, config)
            if not ok:
                return {"role_uuid": role_uuid, "status": "skipped", "reason": reason}
        content, persona, version_id = self._persona(role_uuid)
        previous = self.overlays.active_text(role_uuid)
        diaries = self._diaries(role_uuid, state.get("last_run_at"))
        hot_zone = self._hot_zone(role_uuid)
        memories = self._memories(role_uuid)
        result = self.writer.reflect(
            display_name=str(
                self.service.authority.load_role_file(role_uuid, "config.json", {}).get("display_name")
                or role_uuid
            ),
            persona=persona,
            previous=previous,
            diaries=diaries,
            hot_zone=hot_zone,
            memories=memories,
        )
        if result.get("status") != "written":
            attempts = int(state.get("attempts") or 0) + 1
            backoff = float(config.get("backoff_seconds", 3600) or 3600)
            interval = float(config.get("interval_seconds", 86400) or 86400)
            next_due = moment + (backoff if attempts < 3 else interval)
            self._save_state(
                role_uuid,
                {
                    **state,
                    "last_status": result.get("status"),
                    "last_error": result.get("reason") or result.get("error"),
                    "attempts": attempts,
                    "next_due_at": next_due,
                },
            )
            self.service._log_event(
                "reflection",
                status=str(result.get("status") or "failed"),
                details={"role_uuid": role_uuid, "reason": result.get("reason") or result.get("error")},
            )
            return {"role_uuid": role_uuid, **result}
        # Preserve user-pinned lines; the model may rewrite only the rest.
        existing = self.overlays.active(role_uuid) or {}
        locked = [
            dict(item)
            for item in (existing.get("items") or [])
            if isinstance(item, dict) and item.get("locked")
        ]
        items = locked + [item for item in result.get("items", []) if not _too_similar(item.get("text", ""), self.overlays.compile(locked))]
        unique: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            key = " ".join(str(item.get("text") or "").split())
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        max_items = max(1, int(config.get("max_overlay_items", MAX_OVERLAY_ITEMS) or MAX_OVERLAY_ITEMS))
        max_tokens = int(config.get("max_overlay_tokens", MAX_OVERLAY_TOKENS) or MAX_OVERLAY_TOKENS)
        selected: List[Dict[str, Any]] = []
        for item in unique[:max_items]:
            candidate = selected + [item]
            if len(self.overlays.compile(candidate)) > max_tokens and selected:
                continue
            selected.append(item)
        if not selected:
            self._save_state(
                role_uuid,
                {**state, "last_status": "empty", "attempts": 0, "next_due_at": moment + float(config.get("interval_seconds", 86400) or 86400)},
            )
            return {"role_uuid": role_uuid, "status": "empty"}
        overlay = self.overlays.save(
            role_uuid,
            selected,
            base_prompt_version_id=version_id,
            created_by="reflection",
            auto_activate=bool(config.get("auto_activate", True)),
            token_counter=getattr(self.service.gateway.node_recall, "token_counter", len),
        )
        self._save_state(
            role_uuid,
            {
                **state,
                "last_run_at": moment,
                "last_status": "written",
                "last_error": None,
                "attempts": 0,
                "next_due_at": moment + float(config.get("interval_seconds", 86400) or 86400),
            },
        )
        self.service._log_event(
            "reflection",
            status="written",
            details={"role_uuid": role_uuid, "overlay_id": overlay.get("overlay_version_id"), "items": len(selected)},
        )
        return {"role_uuid": role_uuid, "status": "written", "overlay": overlay}

    def sweep(self) -> Dict[str, Any]:
        config = self._config()
        if not bool(config.get("enabled", True)):
            return {"status": "disabled", "generated": []}
        roles = [role for role in self.service._warm_roles() if role]
        if not roles:
            return {"status": "ok", "generated": []}
        cursor = 0
        try:
            root = self.service.authority.load_root_file("reflection_cursor.json", {})
            cursor = int(root.get("cursor") or 0)
        except Exception:  # noqa: BLE001
            cursor = 0
        generated: List[Dict[str, Any]] = []
        for offset in range(len(roles)):
            role = roles[(cursor + offset) % len(roles)]
            state = self._load_state(role)
            ok, _reason = self._eligible(role, _now(), state, config)
            if not ok:
                continue
            result = self.run(role)
            if result.get("status") in {"written", "empty", "rejected", "failed"}:
                generated.append(result)
                self.service.authority.save_root_file(
                    "reflection_cursor.json",
                    {"cursor": (cursor + offset + 1) % len(roles)},
                )
            break
        return {"status": "ok", "generated": generated}
