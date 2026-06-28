"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio
from typing import Any, List

from .in_memory_cache import InMemoryCache

# Upper bound on evicted client references held until explicit shutdown cleanup.
MAX_EVICTED_LLM_CLIENT_RETENTION = 256


class LLMClientCache(InMemoryCache):
    """Cache for LLM HTTP clients (OpenAI, Azure, httpx, etc.).

    IMPORTANT: This cache intentionally does NOT close clients on eviction.
    Evicted clients may still be in use by in-flight requests. Closing them
    eagerly causes ``RuntimeError: Cannot send a request, as the client has
    been closed.`` errors in production after the TTL (1 hour) expires.

    Evicted client-like values are retained in ``_evicted_clients_retained`` so
    they are not garbage-collected unclosed before shutdown. For explicit
    shutdown cleanup, use ``close_litellm_async_clients()``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evicted_clients_retained: List[Any] = []

    @staticmethod
    def _is_client_like(value: Any) -> bool:
        if value is None or isinstance(
            value, (str, bytes, int, float, bool, dict, list)
        ):
            return False
        if hasattr(value, "client") or hasattr(value, "aclose"):
            return True
        close_fn = getattr(value, "close", None)
        return callable(close_fn)

    def _retain_evicted_client_if_needed(self, value: Any) -> None:
        if not self._is_client_like(value):
            return
        self._evicted_clients_retained.append(value)
        overflow = (
            len(self._evicted_clients_retained) - MAX_EVICTED_LLM_CLIENT_RETENTION
        )
        if overflow > 0:
            del self._evicted_clients_retained[:overflow]

    def _remove_key(self, key: str) -> None:
        value = self.cache_dict.get(key)
        super()._remove_key(key)
        if value is not None:
            self._retain_evicted_client_if_needed(value)

    def flush_cache(self):
        for value in self.cache_dict.values():
            self._retain_evicted_client_if_needed(value)
        super().flush_cache()

    def update_cache_key_with_event_loop(self, key):
        """
        Add the event loop to the cache key, to prevent event loop closed errors.
        If none, use the key as is.
        """
        try:
            event_loop = asyncio.get_running_loop()
            stringified_event_loop = str(id(event_loop))
            return f"{key}-{stringified_event_loop}"
        except RuntimeError:  # handle no current running event loop
            return key

    def set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return super().set_cache(key, value, **kwargs)

    async def async_set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return await super().async_set_cache(key, value, **kwargs)

    def get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return super().get_cache(key, **kwargs)

    async def async_get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return await super().async_get_cache(key, **kwargs)
