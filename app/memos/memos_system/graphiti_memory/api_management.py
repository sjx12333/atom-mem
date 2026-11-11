"""Authentication primitives for the local Graphiti management surface."""

from __future__ import annotations

import hmac
import time
from hashlib import sha256
from urllib.parse import urlsplit


class ManagementAccessError(RuntimeError):
    """Raised when a management request is not a local authenticated request."""

    LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

    @classmethod
    def issue(cls, secret: str, *, now: int | None = None, ttl_seconds: int = 300) -> str:
        if not secret or ttl_seconds <= 0:
            raise cls("management session cannot be issued")
        expires_at = int(time.time() if now is None else now) + ttl_seconds
        signature = hmac.new(secret.encode("utf-8"), str(expires_at).encode("ascii"), sha256).hexdigest()
        return f"{expires_at}.{signature}"

    @classmethod
    def validate(
        cls,
        host: str | None,
        presented_token: str | None,
        expected_secret: str | None,
        *,
        origin: str | None = None,
        now: int | None = None,
    ) -> None:
        if host not in cls.LOOPBACK_HOSTS:
            raise cls("management endpoints are loopback-only")
        if origin and urlsplit(origin).hostname not in cls.LOOPBACK_HOSTS:
            raise cls("management request origin is not local")
        if not expected_secret or not presented_token or "." not in presented_token:
            raise cls("invalid management session")
        expires_at, signature = presented_token.split(".", 1)
        try:
            expiry = int(expires_at)
        except ValueError as exc:
            raise cls("invalid management session") from exc
        current_time = int(time.time() if now is None else now)
        expected_signature = hmac.new(
            expected_secret.encode("utf-8"), expires_at.encode("ascii"), sha256
        ).hexdigest()
        if expiry < current_time or not hmac.compare_digest(signature, expected_signature):
            raise cls("invalid management session")
