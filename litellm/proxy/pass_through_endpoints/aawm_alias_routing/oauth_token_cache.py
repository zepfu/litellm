"""Provider-neutral process-local OAuth access-token cache primitive."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OAuthAccessTokenCache:
    """Map of cache_key -> (access_token, expiry_epoch_ms_or_seconds)."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tokens: dict[str, tuple[str, int]] = field(default_factory=dict)

    def get_if_valid(
        self,
        cache_key: str,
        *,
        now: Optional[float] = None,
        skew_seconds: float = 30.0,
        expiry_is_millis: bool = True,
    ) -> Optional[str]:
        cached = self.tokens.get(cache_key)
        if not cached:
            return None
        token, expiry = cached
        clock = time.time() if now is None else now
        expiry_seconds = (expiry / 1000.0) if expiry_is_millis else float(expiry)
        if expiry_seconds - skew_seconds > clock and token:
            return token
        return None

    def set(self, cache_key: str, access_token: str, expiry: int) -> None:
        self.tokens[cache_key] = (access_token, expiry)

    def clear(self, cache_key: Optional[str] = None) -> None:
        if cache_key is None:
            self.tokens.clear()
            return
        self.tokens.pop(cache_key, None)
