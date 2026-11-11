"""One synchronous chat-completion call, shared by the post-conversation passes.

Both the diary writer and the memory extractor talk to the same chat endpoint with
the same shape: a system line that pins the output format, one user prompt, a JSON
(or plain-text) answer. Keeping the transport here means the "which credential,
which timeout, how many tokens" questions are answered in exactly one place.

The client is deliberately synchronous and built on ``urllib`` from the standard
library: these passes run at session close on the extraction worker, where an
async client would have to be driven through the service's single event loop for
no benefit.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class ChatCallError(RuntimeError):
    """The chat endpoint could not be reached or answered with an error."""


class ChatClient:
    """A chat-completions client bound to one credential/provider section."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def complete(
        self,
        prompt: str,
        *,
        system: str = "你只输出 JSON。",
        json_object: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Return the assistant message content, or raise :class:`ChatCallError`."""
        if not self.configured:
            raise ChatCallError("chat endpoint is not configured")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(prompt or "")},
            ],
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": int(max_tokens if max_tokens is not None else self.max_tokens),
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001 - the status is the important part
                detail = ""
            raise ChatCallError(f"chat HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - network failure is one category
            raise ChatCallError(f"chat call failed: {type(exc).__name__}: {exc}") from exc
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise ChatCallError("chat endpoint did not return JSON") from exc
        choices = body.get("choices") or []
        if not choices:
            raise ChatCallError("chat endpoint returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            # A reasoning model that ran out of budget answers with empty content
            # and finish_reason=length; surfacing the reason beats treating it as an
            # empty-but-valid answer.
            reason = choices[0].get("finish_reason")
            raise ChatCallError(f"chat endpoint returned empty content (finish_reason={reason})")
        return str(content)


def chat_client_from_config(config: Dict[str, Any], *, section: str = "chat") -> ChatClient:
    """Build a :class:`ChatClient` from a Graphiti config section.

    ``section`` is a *preference*: a section that is not usable (disabled, or
    missing its key/model/endpoint) falls back to the main ``chat`` section, so a
    deployment without a second credential simply uses the one it has instead of
    silently skipping the pass.
    """
    chat = (config or {}).get(section) or {}
    client = ChatClient(
        base_url=str(chat.get("base_url") or ""),
        api_key=str(chat.get("api_key") or ""),
        model=str(chat.get("model") or ""),
        timeout=float(chat.get("timeout") or 60.0),
        max_tokens=int(chat.get("max_tokens") or 8192),
    )
    if client.configured and section == "chat":
        return client
    if client.configured and bool(chat.get("enabled", True)):
        return client
    if section != "chat":
        return chat_client_from_config(config, section="chat")
    return client


def load_json_object(content: str) -> Dict[str, Any]:
    """Parse a model answer into an object, tolerating prose around the JSON."""
    if not content:
        return {}
    data: Any = None
    try:
        data = json.loads(content)
    except Exception:  # noqa: BLE001 - lenient parse of model output
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start : end + 1])
            except Exception:  # noqa: BLE001
                data = None
    return data if isinstance(data, dict) else {}
