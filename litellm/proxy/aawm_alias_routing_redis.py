"""Dedicated Redis manager for AAWM alias-routing cooldown/state persistence.

This module owns only alias-routing Redis cache wiring and does not touch shared
LiteLLM cache/Router internals.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Union

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache


logger = logging.getLogger(__name__)


def resolve_alias_routing_state_namespace() -> str:
    """Resolve an explicit or environment-isolated routing-state namespace."""
    explicit_namespace = (os.getenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE") or "").strip()
    if explicit_namespace:
        return explicit_namespace

    runtime_environment = (
        (
            os.getenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT")
            or os.getenv("LITELLM_AAWM_ERROR_LOG_ENV")
            or ""
        )
        .strip()
        .lower()
    )
    if runtime_environment in {"dev", "development"}:
        return "aawm-routing-dev-v1"
    if runtime_environment in {"prod", "production"}:
        return "aawm-routing-prod-v1"
    return AAWMAliasRoutingRedisManager.DEFAULT_NAMESPACE


class AAWMAliasRoutingRedisManager:
    """Standalone alias-routing Redis cache manager."""

    DEFAULT_NAMESPACE = "default"
    # Bounded startup retries for transient sidecar readiness races. Kept small so
    # permanent failures still fall back to memory quickly and visibly.
    STARTUP_CONNECT_ATTEMPTS = 3
    STARTUP_RETRY_DELAY_SECONDS = 0.05

    def __init__(self) -> None:
        self._cache: Optional[RedisCache] = None
        self._managed_dual_cache: Optional[DualCache] = None
        self._configured: bool = False
        self._config_mode: str = "unconfigured"
        self._initialized: bool = False
        self._reachable: Union[bool, str] = "unknown"
        self._error_type: Optional[str] = None
        self._namespace: str = self.DEFAULT_NAMESPACE

    @staticmethod
    def _parse_bool(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            logger.warning(
                "Invalid integer config for AAWM alias routing Redis settings; using fallback."
            )
            return None

    def _load_env_config(self) -> Dict[str, Any]:
        namespace = resolve_alias_routing_state_namespace()

        redis_url = os.getenv("AAWM_ALIAS_ROUTING_REDIS_URL")
        if redis_url:
            if (os.getenv("AAWM_ALIAS_ROUTING_REDIS_HOST") or "").strip():
                logger.warning(
                    "AAWM alias routing Redis config has both URL and host; URL mode is enabled."
                )
            return {
                "mode": "url",
                "config_mode": "url",
                "namespace": namespace,
                "url": redis_url.strip(),
                "configured": True,
            }

        redis_host = (os.getenv("AAWM_ALIAS_ROUTING_REDIS_HOST") or "").strip()
        if not redis_host:
            return {
                "mode": "memory",
                "config_mode": "unconfigured",
                "namespace": namespace,
                "configured": False,
            }

        redis_port = self._parse_int(os.getenv("AAWM_ALIAS_ROUTING_REDIS_PORT"))
        if redis_port is None:
            redis_port = 6379

        return {
            "mode": "host",
            "config_mode": "host",
            "namespace": namespace,
            "host": redis_host,
            "port": redis_port,
            "password": os.getenv("AAWM_ALIAS_ROUTING_REDIS_PASSWORD"),
            "username": os.getenv("AAWM_ALIAS_ROUTING_REDIS_USERNAME"),
            "db": self._parse_int(os.getenv("AAWM_ALIAS_ROUTING_REDIS_DB")),
            "ssl": self._parse_bool(os.getenv("AAWM_ALIAS_ROUTING_REDIS_SSL")),
            "configured": True,
        }

    @staticmethod
    def _build_key_prefix(namespace: str) -> str:
        return f"aawm:alias-routing:{namespace}"

    @classmethod
    def _build_redis_cache(cls, config: Dict[str, Any]) -> RedisCache:
        if config["mode"] == "url":
            return RedisCache(url=config["url"], socket_timeout=5.0)

        redis_kwargs = {
            "host": config["host"],
            "port": config["port"],
            "password": config["password"],
            "username": config["username"],
            "db": config["db"],
            "socket_timeout": 5.0,
        }
        # LiteLLM's shared async pool currently selects SSLConnection whenever
        # the ssl key is present, including ssl=False. Omit the key for plaintext.
        if config["ssl"]:
            redis_kwargs["ssl"] = True
        return RedisCache(**redis_kwargs)

    @classmethod
    async def _build_redis_cache_off_loop(cls, config: Dict[str, Any]) -> RedisCache:
        """Construct RedisCache off the running event-loop thread.

        RedisCache schedules an unowned async health-ping task when constructed
        on a live loop. Building off-loop keeps that constructor-side effect off
        the proxy event loop and avoids unretrieved task exceptions during
        alias-routing startup.
        """
        return await asyncio.to_thread(cls._build_redis_cache, config)

    def _ensure_dual_cache(self, cache: RedisCache) -> DualCache:
        if self._managed_dual_cache is None:
            return DualCache(default_in_memory_ttl=1, redis_cache=cache)
        self._managed_dual_cache.redis_cache = cache
        return self._managed_dual_cache

    def get_dual_cache(self) -> Optional[Any]:
        if self._managed_dual_cache is None:
            return None
        if self._cache is None:
            return None
        if self._managed_dual_cache.redis_cache is not self._cache:
            return None
        return self._managed_dual_cache

    def _is_cache_attached(self) -> bool:
        if self._cache is None or self._managed_dual_cache is None:
            return False
        try:
            return getattr(self._managed_dual_cache, "redis_cache", None) is self._cache
        except Exception:
            return False

    def _resolve_state_source(self) -> str:
        """Classify durable vs local fallback for manager/readiness telemetry.

        - durable_cache: manager-owned Redis client is attached to DualCache
        - local_fallback: Redis is configured but unavailable / not attached
        - memory: Redis is not configured
        """
        if self._is_cache_attached():
            return "durable_cache"
        if self._configured:
            return "local_fallback"
        return "memory"

    def _detach_cached_client(self) -> None:
        try:
            if (
                self._managed_dual_cache is not None
                and self._managed_dual_cache.redis_cache is self._cache
            ):
                self._managed_dual_cache.redis_cache = None
        except Exception:
            logger.debug(
                "AAWM alias routing Redis dual cache detachment failed", exc_info=True
            )

        self._managed_dual_cache = None

    async def _disconnect_cached_client(self) -> None:
        if self._cache is None:
            return

        try:
            await self._cache.disconnect()
        except Exception:
            logger.debug("AAWM alias routing Redis disconnect failed", exc_info=True)

    async def _disconnect_attempt_cache(self, cache: Optional[RedisCache]) -> None:
        if cache is None:
            return
        try:
            await cache.disconnect()
        except Exception:
            logger.debug(
                "AAWM alias routing Redis close during initialize failed",
                exc_info=True,
            )

    async def _connect_with_startup_retries(
        self, config: Dict[str, Any]
    ) -> tuple[Optional[RedisCache], Optional[BaseException]]:
        attempts = max(1, int(self.STARTUP_CONNECT_ATTEMPTS))
        delay_seconds = max(0.0, float(self.STARTUP_RETRY_DELAY_SECONDS))
        last_error: Optional[BaseException] = None

        for attempt in range(attempts):
            cache: Optional[RedisCache] = None
            try:
                cache = await self._build_redis_cache_off_loop(config)
                if not await cache.ping():
                    raise RuntimeError("redis_ping_unreachable")
                return cache, None
            except Exception as exc:
                last_error = exc
                await self._disconnect_attempt_cache(cache)
                if attempt + 1 >= attempts:
                    break
                logger.warning(
                    "AAWM alias routing Redis connect attempt %s/%s failed (%s); retrying.",
                    attempt + 1,
                    attempts,
                    type(exc).__name__,
                )
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

        return None, last_error

    async def _disconnect_previous_cache(
        self, previous_cache: Optional[RedisCache], *, message: str
    ) -> None:
        if previous_cache is None:
            return
        try:
            await previous_cache.disconnect()
        except Exception:
            logger.debug(message, exc_info=True)

    async def initialize(self) -> None:
        """Initialize the alias-routing cache and wire it when reachable."""
        config = self._load_env_config()
        self._namespace = config["namespace"]
        self._configured = bool(config["configured"])
        self._config_mode = config["config_mode"]

        if not self._configured:
            previous_cache = self._cache
            self._initialized = True
            self._reachable = "unknown"
            self._error_type = None
            self._detach_cached_client()
            await self._disconnect_previous_cache(
                previous_cache,
                message="AAWM alias routing Redis close during disable failed",
            )
            self._cache = None
            return

        if self._initialized and self._is_cache_attached() and self._reachable is True:
            return

        # Remove a prior attachment if this manager installed it before attempting
        # to reconfigure/reattach.
        self._detach_cached_client()
        previous_cache = self._cache

        cache, last_error = await self._connect_with_startup_retries(config)
        if cache is not None:
            dual_cache = self._ensure_dual_cache(cache)
            if previous_cache is not cache:
                await self._disconnect_previous_cache(
                    previous_cache,
                    message=(
                        "AAWM alias routing Redis close during reconfigure failed"
                    ),
                )
            self._cache = cache
            self._managed_dual_cache = dual_cache
            self._reachable = True
            self._error_type = None
            self._initialized = True
            return

        # Failed reconfiguration must not leave a detached previous client
        # retained as a silent open connection; drop it and fall back visibly.
        await self._disconnect_previous_cache(
            previous_cache,
            message=("AAWM alias routing Redis close during failed reconfigure failed"),
        )
        self._cache = None
        self._managed_dual_cache = None
        self._reachable = False
        self._error_type = type(last_error).__name__ if last_error is not None else None
        self._initialized = False
        logger.warning(
            "AAWM alias routing Redis unavailable; using memory cache fallback."
        )

    async def shutdown(self) -> None:
        """Detach and disconnect the manager-owned Redis client."""
        self._detach_cached_client()
        await self._disconnect_cached_client()
        self._cache = None
        self._initialized = False
        self._reachable = "unknown"
        self._error_type = None
        self._configured = False
        self._config_mode = "unconfigured"
        self._namespace = self.DEFAULT_NAMESPACE

    def get_status(self) -> Dict[str, Any]:
        """Return sanitized runtime status for alias-routing state cache."""
        namespace = self._namespace or self.DEFAULT_NAMESPACE
        mode = "redis" if self._is_cache_attached() else "memory"
        return {
            "configured": self._configured,
            "config_mode": self._config_mode,
            "mode": mode,
            "state_source": self._resolve_state_source(),
            "reachable": self._reachable,
            "namespace": namespace,
            "key_prefix": self._build_key_prefix(namespace),
            "error_type": self._error_type,
        }

    def reset(self) -> None:
        """Reset local manager state for tests without disconnecting live clients."""
        self._detach_cached_client()
        self._initialized = False
        self._reachable = "unknown"
        self._error_type = None
        self._configured = False
        self._config_mode = "unconfigured"
        self._namespace = self.DEFAULT_NAMESPACE


aawm_alias_routing_redis_manager = AAWMAliasRoutingRedisManager()


def get_dual_cache() -> Optional[Any]:
    return aawm_alias_routing_redis_manager.get_dual_cache()


async def initialize(**_: Any) -> None:
    await aawm_alias_routing_redis_manager.initialize()


async def shutdown() -> None:
    await aawm_alias_routing_redis_manager.shutdown()


def get_status() -> Dict[str, Any]:
    return aawm_alias_routing_redis_manager.get_status()


def reset() -> None:
    aawm_alias_routing_redis_manager.reset()
