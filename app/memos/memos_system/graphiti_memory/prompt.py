"""Stable prompt ordering and per-request transient recall snapshots."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping


#: Recall boundary handed to every client that assembles a prompt.
#:
#: It states *only* that the recalled fragments were retrieved, plus the rules that
#: keep the model from narrating them as facts. It deliberately does not claim the
#: fragments are "known memories about the user": most relationships in a role graph
#: have the AI itself as their subject, and a heading that claims otherwise turns the
#: AI's own past utterances into the user's history — the model then invents events
#: that never happened (issue: 虚假回忆 / fabricated recall).
#:
#: Kept in sync with ``live-2d/plugins/built-in/memos/recall-guard.js``, which owns
#: the same boundary for the desktop pet's automatic injection and memory tool.
RECALL_BOUNDARY = (
    "以下是记忆系统检索到的片段，不是完整历史，可能与本轮无关。"
    "只使用字面写出的内容，不要补充、推测或编造任何细节（人物、时间、地点、数量、原因、感受、约定都算编造）。"
    "条目的主语是 AI 自己时，那是 AI 说过的话或设定，不是用户的经历。"
    "每条片段标注的实体名说明这条记忆属于谁（它的母实体节点）；实体不是当前说话人时，"
    "那是关于别人的记忆，不要说成当前说话人的经历，也不要主动拿它来接话。"
    "片段里没有写到的用户经历一律当作不记得，直接说不记得比编造更正确；"
    "只有片段里明写的事才可以用“我记得”“你之前说过”“上次”这类说法。"
)


def build_prompt_context(
    role_prompt: str,
    hot_zone: List[str],
    history: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    corrections: List[str] | None = None,
    persona_overlay: List[str] | None = None,
) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = [{"region": "role_prompt", "content": role_prompt}]
    if persona_overlay:
        context.append({"region": "persona_overlay", "content": list(persona_overlay)})
    context.append({"region": "hot_zone", "content": list(hot_zone)})
    context.append({"region": "chat_history", "content": list(history)})
    context.append({"region": "post_chat_memory", "content": list(memories)})
    if corrections:
        context.append({"region": "corrections", "content": list(corrections)})
    return context


class RequestSnapshotFactory:
    def __init__(self, gateway: Any):
        self.gateway = gateway
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        request_id: str,
        role_uuid: str,
        user_id: str,
        conversation_id: str,
        query: str,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        if request_id not in self._snapshots:
            result = self.gateway.search(role_uuid, user_id, conversation_id, query, top_k=top_k)
            self._snapshots[request_id] = {
                "request_id": request_id,
                "kind": "transient_memory_snapshot",
                "memories": result.get("memories", []),
                "boundary": RECALL_BOUNDARY,
            }
        return self._snapshots[request_id]

    def assemble(
        self,
        request_id: str,
        role_prompt: str,
        hot_zone: List[str],
        history: List[Dict[str, Any]],
        user_message: str,
        tool_events: Iterable[Dict[str, Any]] = (),
        post_chat_memory: List[Dict[str, Any]] | None = None,
        persona_overlay: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        snapshot = self._snapshots.get(request_id)
        if snapshot is None:
            raise KeyError(f"no transient snapshot for request: {request_id}")
        return build_request_context(
            role_prompt=role_prompt,
            hot_zone=hot_zone,
            history=history,
            user_message=user_message,
            transient_memory_snapshot=snapshot,
            tool_events=tool_events,
            post_chat_memory=post_chat_memory,
            persona_overlay=persona_overlay,
        )

    def discard(self, request_id: str) -> None:
        self._snapshots.pop(request_id, None)


def build_request_context(
    role_prompt: str,
    hot_zone: List[str],
    history: List[Dict[str, Any]],
    user_message: str,
    transient_memory_snapshot: Mapping[str, Any],
    tool_events: Iterable[Dict[str, Any]] = (),
    post_chat_memory: List[Dict[str, Any]] | None = None,
    persona_overlay: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build a request copy while keeping transient recall out of history."""

    context: List[Dict[str, Any]] = [
        {"region": "role_prompt", "content": role_prompt},
    ]
    if persona_overlay:
        context.append({"region": "persona_overlay", "content": copy.deepcopy(persona_overlay)})
    context.extend([
        {"region": "hot_zone", "content": copy.deepcopy(hot_zone)},
        {"region": "chat_history", "content": copy.deepcopy(history)},
        {"region": "post_chat_memory", "content": copy.deepcopy(post_chat_memory or [])},
        {
            "region": "current_user_request",
            "content": {
                "message": user_message,
                "boundary": RECALL_BOUNDARY,
                "transient_memory_snapshot": transient_memory_snapshot,
            },
        },
    ])
    context.extend(
        {"region": "tool_event", "content": copy.deepcopy(event)}
        for event in tool_events
    )
    return context
