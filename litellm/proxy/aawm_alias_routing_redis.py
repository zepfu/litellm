"""Dedicated Redis manager for AAWM alias-routing cooldown/state persistence.

This module owns only alias-routing Redis cache wiring and does not touch shared
LiteLLM cache/Router internals.
"""

from __future__ import annotations

import asyncio
import math
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache


logger = logging.getLogger(__name__)

RedisClientOwnership = tuple[RedisCache, int]
InitializeTransition = tuple[
    Dict[str, Any],
    int,
    Optional[RedisClientOwnership],
    Optional[asyncio.Task[Any]],
    str,
]

# Durable-write retry policy lives here so connection/timeout and write resilience
# share one subsystem owner. redis.exceptions is imported at module top with a
# guarded fallback so hot write paths never pay an inline import.
try:
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    _REDIS_RETRYABLE_EXCEPTIONS: Tuple[type[BaseException], ...] = (
        RedisConnectionError,
        RedisTimeoutError,
    )
except Exception:  # pragma: no cover - redis may be unavailable in some installs
    _REDIS_RETRYABLE_EXCEPTIONS = ()

DURABLE_WRITE_RETRY_ATTEMPTS = 1
DURABLE_WRITE_RETRY_BACKOFF_SECONDS_DEFAULT = 0.25
DURABLE_WRITE_RETRY_BACKOFF_SECONDS_MIN = 0.05
DURABLE_WRITE_RETRY_BACKOFF_SECONDS_MAX = 2.0
DURABLE_WRITE_RETRY_BACKOFF_SECONDS_ENV = (
    "AAWM_ALIAS_ROUTING_REDIS_DURABLE_WRITE_RETRY_BACKOFF_SECONDS"
)

# Host-style settings that RedisCache(url=...) does not consume.
_URL_MODE_IGNORED_ENV_VARS = (
    "AAWM_ALIAS_ROUTING_REDIS_HOST",
    "AAWM_ALIAS_ROUTING_REDIS_PASSWORD",
    "AAWM_ALIAS_ROUTING_REDIS_USERNAME",
    "AAWM_ALIAS_ROUTING_REDIS_SSL",
    "AAWM_ALIAS_ROUTING_REDIS_DB",
    "AAWM_ALIAS_ROUTING_REDIS_PORT",
)


def _env_is_set(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(str(value).strip())


class AliasRoutingStateNamespace:
    """String-comparable namespace that also reports how it was resolved.

    Existing callers compare this value to a plain string. D1-552 requires the
    source to be distinguishable: ``explicit``, ``observability_derived``, or
    ``default``. This is intentionally not a ``str`` subclass so callers can
    inspect ``namespace_source`` instead of treating the result as a bare label.
    """

    __slots__ = ("namespace", "namespace_source")

    def __init__(
        self, namespace: str, namespace_source: str = "default"
    ) -> None:
        self.namespace = str(namespace)
        self.namespace_source = namespace_source

    @property
    def source(self) -> str:
        return self.namespace_source

    def __str__(self) -> str:
        return self.namespace

    def __repr__(self) -> str:
        return (
            f"AliasRoutingStateNamespace({self.namespace!r}, "
            f"{self.namespace_source!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AliasRoutingStateNamespace):
            return self.namespace == other.namespace
        if isinstance(other, str):
            return self.namespace == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.namespace)


def resolve_alias_routing_state_namespace() -> AliasRoutingStateNamespace:
    """Resolve an explicit or environment-isolated routing-state namespace."""
    explicit_namespace = (os.getenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE") or "").strip()
    if explicit_namespace:
        return AliasRoutingStateNamespace(explicit_namespace, "explicit")

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
        return AliasRoutingStateNamespace(
            "aawm-routing-dev-v1", "observability_derived"
        )
    if runtime_environment in {"prod", "production"}:
        return AliasRoutingStateNamespace(
            "aawm-routing-prod-v1", "observability_derived"
        )
    return AliasRoutingStateNamespace(
        AAWMAliasRoutingRedisManager.DEFAULT_NAMESPACE, "default"
    )


class AAWMAliasRoutingRedisManager:
    """Standalone alias-routing Redis cache manager."""

    DEFAULT_NAMESPACE = "default"
    DEFAULT_SOCKET_TIMEOUT_SECONDS = 10.0
    MIN_SOCKET_TIMEOUT_SECONDS = 1.0
    MAX_SOCKET_TIMEOUT_SECONDS = 60.0
    SOCKET_TIMEOUT_ENV_VAR = "AAWM_ALIAS_ROUTING_REDIS_TIMEOUT_SECONDS"
    # Bounded startup retries for transient sidecar readiness races. Kept small so
    # permanent failures still fall back to memory quickly and visibly.
    STARTUP_CONNECT_ATTEMPTS = 3
    STARTUP_RETRY_DELAY_SECONDS = 0.05
    # Background self-heal after startup fallback. Runs off the critical path so a
    # slow/unavailable Redis does not block proxy readiness, but can reattach once
    # the sidecar becomes reachable. Interval is intentionally longer than the
    # startup retry delay.
    DEFAULT_SELF_HEAL_INTERVAL_SECONDS = 30.0
    MIN_SELF_HEAL_INTERVAL_SECONDS = 5.0
    MAX_SELF_HEAL_INTERVAL_SECONDS = 300.0
    SELF_HEAL_INTERVAL_ENV_VAR = "AAWM_ALIAS_ROUTING_REDIS_SELF_HEAL_INTERVAL_SECONDS"

    def __init__(self) -> None:
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_generation = 0
        self._cache: Optional[RedisCache] = None
        self._cache_generation: Optional[int] = None
        self._closing_clients: list[RedisClientOwnership] = []
        self._managed_dual_cache: Optional[DualCache] = None
        self._configured: bool = False
        self._config_mode: str = "unconfigured"
        self._initialized: bool = False
        self._reachable: Union[bool, str] = "unknown"
        self._error_type: Optional[str] = None
        self._namespace: str = self.DEFAULT_NAMESPACE
        self._config_signature: Optional[tuple[tuple[str, Any], ...]] = None
        self._active_attempts: list[RedisClientOwnership] = []
        self._self_heal_task: Optional[asyncio.Task[Any]] = None
        self._self_heal_stop: Optional[asyncio.Event] = None
        # Set when initialize() has observed a configured Redis that fell back to
        # memory so status/tests can tell self-heal is armed without inspecting tasks.
        self._self_heal_armed: bool = False
        self._local_state_reconciled: bool = False

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

    @staticmethod
    def _parse_float(
        value: Optional[str], *, default: float, minimum: float, maximum: float
    ) -> float:
        if not value:
            return default
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            logger.warning(
                "Invalid float config for AAWM alias routing Redis settings; using fallback."
            )
            return default
        if not math.isfinite(parsed):
            logger.warning(
                "Non-finite float config for AAWM alias routing Redis settings; using fallback."
            )
            return default
        return max(minimum, min(maximum, parsed))

    @classmethod
    def _resolve_alias_routing_redis_socket_timeout(cls) -> float:
        return cls._parse_float(
            os.getenv(cls.SOCKET_TIMEOUT_ENV_VAR),
            default=cls.DEFAULT_SOCKET_TIMEOUT_SECONDS,
            minimum=cls.MIN_SOCKET_TIMEOUT_SECONDS,
            maximum=cls.MAX_SOCKET_TIMEOUT_SECONDS,
        )

    def _load_env_config(self) -> Dict[str, Any]:
        namespace = str(resolve_alias_routing_state_namespace())
        socket_timeout = self._resolve_alias_routing_redis_socket_timeout()

        redis_url = os.getenv("AAWM_ALIAS_ROUTING_REDIS_URL")
        if redis_url:
            ignored = [
                env_var
                for env_var in _URL_MODE_IGNORED_ENV_VARS
                if _env_is_set(env_var)
            ]
            if ignored:
                logger.warning(
                    "AAWM alias routing Redis config uses URL mode; "
                    "ignoring host-style settings that RedisCache(url=...) does not consume: %s",
                    ", ".join(ignored),
                )
            return {
                "mode": "url",
                "config_mode": "url",
                "namespace": namespace,
                "url": redis_url.strip(),
                "socket_timeout": socket_timeout,
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
            "socket_timeout": socket_timeout,
            "configured": True,
        }

    @staticmethod
    def _build_config_signature(config: Dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(config.items()))

    @staticmethod
    def _build_key_prefix(namespace: object) -> str:
        return f"aawm:alias-routing:{namespace}"

    @classmethod
    def _build_redis_cache(cls, config: Dict[str, Any]) -> RedisCache:
        if config["mode"] == "url":
            return RedisCache(url=config["url"], socket_timeout=config["socket_timeout"])

        redis_kwargs = {
            "host": config["host"],
            "port": config["port"],
            "password": config["password"],
            "username": config["username"],
            "db": config["db"],
            "socket_timeout": config["socket_timeout"],
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

    @classmethod
    def _resolve_self_heal_interval_seconds(cls) -> float:
        return cls._parse_float(
            os.getenv(cls.SELF_HEAL_INTERVAL_ENV_VAR),
            default=cls.DEFAULT_SELF_HEAL_INTERVAL_SECONDS,
            minimum=cls.MIN_SELF_HEAL_INTERVAL_SECONDS,
            maximum=cls.MAX_SELF_HEAL_INTERVAL_SECONDS,
        )

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

    def _begin_lifecycle_transition_locked(self) -> int:
        self._lifecycle_generation += 1
        return self._lifecycle_generation

    def _take_cached_client_locked(self) -> Optional[RedisClientOwnership]:
        if self._cache is None:
            self._detach_cached_client()
            self._cache = None
            self._cache_generation = None
            return None

        ownership = (
            self._cache,
            self._cache_generation or self._lifecycle_generation,
        )
        self._detach_cached_client()
        self._cache = None
        self._cache_generation = None
        return ownership

    def _claim_self_heal_task_locked(self) -> Optional[asyncio.Task[Any]]:
        task = self._self_heal_task
        if self._self_heal_stop is not None:
            self._self_heal_stop.set()
        self._self_heal_task = None
        self._self_heal_stop = None
        return task

    def _remove_active_attempt_locked(
        self, cache: RedisCache, generation: int
    ) -> None:
        self._active_attempts = [
            (owned_cache, attempt_generation)
            for owned_cache, attempt_generation in self._active_attempts
            if not (owned_cache is cache and attempt_generation == generation)
        ]

    def _is_client_closing_locked(self, cache: RedisCache) -> bool:
        return any(owned_cache is cache for owned_cache, _ in self._closing_clients)

    async def _disconnect_previous_cache(
        self, previous_cache: Optional[RedisCache], *, message: str
    ) -> None:
        if previous_cache is None:
            return
        try:
            await previous_cache.disconnect()
        except Exception:
            logger.debug(message, exc_info=True)

    async def _disconnect_owned_client(
        self, ownership: Optional[RedisClientOwnership], *, message: str
    ) -> None:
        """Close a client only after claiming its ownership under the lock."""
        if ownership is None:
            return

        cache, generation = ownership
        async with self._lifecycle_lock:
            self._remove_active_attempt_locked(cache, generation)
            if (
                self._cache is cache
                or self._is_client_closing_locked(cache)
                or any(
                    owned_cache is cache and attempt_generation > generation
                    for owned_cache, attempt_generation in self._active_attempts
                )
            ):
                return

            self._closing_clients.append(ownership)
        try:
            await self._disconnect_previous_cache(cache, message=message)
        finally:
            async with self._lifecycle_lock:
                self._closing_clients = [
                    (owned_cache, owned_generation)
                    for owned_cache, owned_generation in self._closing_clients
                    if owned_cache is not cache
                ]

    async def _disconnect_attempt_cache(
        self, cache: Optional[RedisCache], generation: int
    ) -> None:
        if cache is None:
            return
        await self._disconnect_owned_client(
            (cache, generation),
            message="AAWM alias routing Redis close during initialize failed",
        )

    async def _connect_with_startup_retries(
        self, config: Dict[str, Any], generation: int
    ) -> tuple[Optional[RedisCache], Optional[BaseException]]:
        attempts = max(1, int(self.STARTUP_CONNECT_ATTEMPTS))
        delay_seconds = max(0.0, float(self.STARTUP_RETRY_DELAY_SECONDS))
        last_error: Optional[BaseException] = None

        for attempt in range(attempts):
            cache: Optional[RedisCache] = None
            try:
                cache = await self._build_redis_cache_off_loop(config)
                async with self._lifecycle_lock:
                    self._active_attempts.append((cache, generation))
                if not await cache.ping():
                    raise RuntimeError("redis_ping_unreachable")
                return cache, None
            except asyncio.CancelledError:
                await self._disconnect_attempt_cache(cache, generation)
                raise
            except Exception as exc:
                last_error = exc
                await self._disconnect_attempt_cache(cache, generation)
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

    def _prepare_initialize_transition_locked(self) -> InitializeTransition:
        config = self._load_env_config()
        generation = self._begin_lifecycle_transition_locked()
        config_signature = self._build_config_signature(config)
        self._namespace = config["namespace"]
        self._configured = bool(config["configured"])
        self._config_mode = config["config_mode"]
        configuration_changed = self._config_signature != config_signature
        self._config_signature = config_signature

        if not self._configured:
            previous_cache = self._take_cached_client_locked()
            self._initialized = True
            self._reachable = "unknown"
            self._error_type = None
            self._self_heal_armed = False
            task_to_stop = self._claim_self_heal_task_locked()
            return config, generation, previous_cache, task_to_stop, "disabled"

        if (
            self._initialized
            and self._is_cache_attached()
            and self._reachable is True
            and not configuration_changed
        ):
            self._self_heal_armed = False
            task_to_stop = self._claim_self_heal_task_locked()
            return config, generation, None, task_to_stop, "already_initialized"

        return config, generation, self._take_cached_client_locked(), None, "connect"

    def _commit_initialize_result_locked(
        self,
        generation: int,
        cache: Optional[RedisCache],
        last_error: Optional[BaseException],
    ) -> tuple[
        bool,
        Optional[asyncio.Task[Any]],
        Optional[RedisClientOwnership],
    ]:
        if generation != self._lifecycle_generation:
            return True, None, None

        if cache is not None and not self._is_client_closing_locked(cache):
            was_degraded = self._self_heal_armed or self._reachable is False
            replaced_cache = None
            if self._cache is not None and self._cache is not cache:
                replaced_cache = self._take_cached_client_locked()
            self._remove_active_attempt_locked(cache, generation)
            self._cache = cache
            self._cache_generation = generation
            self._managed_dual_cache = self._ensure_dual_cache(cache)
            self._reachable = True
            self._error_type = None
            self._initialized = True
            self._self_heal_armed = False
            if was_degraded:
                self.reconcile_local_routing_state()
            return False, self._claim_self_heal_task_locked(), replaced_cache

        if cache is not None:
            return True, None, None

        replaced_cache = self._take_cached_client_locked()
        self._reachable = False
        self._error_type = type(last_error).__name__ if last_error is not None else None
        self._initialized = False
        self._self_heal_armed = True
        logger.warning(
            "AAWM alias routing Redis unavailable; using memory cache fallback."
        )
        self._ensure_self_heal_task()
        return False, None, replaced_cache

    async def initialize(self) -> None:
        """Initialize the alias-routing cache and wire it when reachable."""
        async with self._lifecycle_lock:
            config, generation, previous_cache, task_to_stop, transition = (
                self._prepare_initialize_transition_locked()
            )

        if transition == "disabled":
            await self._finish_self_heal_task(task_to_stop)
            await self._disconnect_owned_client(
                previous_cache,
                message="AAWM alias routing Redis close during disable failed",
            )
            return

        if transition == "already_initialized":
            await self._finish_self_heal_task(task_to_stop)
            return

        cache, last_error = await self._connect_with_startup_retries(
            config, generation
        )
        async with self._lifecycle_lock:
            stale_transition, task_to_stop, replaced_cache = (
                self._commit_initialize_result_locked(generation, cache, last_error)
            )

        if stale_transition:
            await self._disconnect_owned_client(
                (cache, generation) if cache is not None else None,
                message=(
                    "AAWM alias routing Redis close for stale lifecycle attempt failed"
                ),
            )

        await self._finish_self_heal_task(task_to_stop)
        await self._disconnect_owned_client(
            replaced_cache,
            message="AAWM alias routing Redis close during reconfigure failed",
        )
        await self._disconnect_owned_client(
            previous_cache,
            message="AAWM alias routing Redis close during reconfigure failed",
        )

    def _ensure_self_heal_task(self) -> None:
        """Schedule at most one background reconnect task after startup fallback."""
        if self._self_heal_task is not None and not self._self_heal_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (sync test path); self-heal arms only under asyncio.
            return
        if self._self_heal_stop is None:
            self._self_heal_stop = asyncio.Event()
        else:
            self._self_heal_stop.clear()
        self._self_heal_task = loop.create_task(self._self_heal_loop())

    async def _finish_self_heal_task(
        self, task: Optional[asyncio.Task[Any]]
    ) -> None:
        if task is None:
            return
        current = asyncio.current_task()
        if task is current:
            return
        if task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "AAWM alias routing Redis self-heal task stop failed",
                exc_info=True,
            )

    async def _stop_self_heal_task(self) -> None:
        async with self._lifecycle_lock:
            task = self._claim_self_heal_task_locked()
        await self._finish_self_heal_task(task)

    async def _self_heal_loop(self) -> None:
        """Periodically reattempt Redis connectivity after a configured fallback.

        Does not run during the initial startup critical path. Uses the same
        initialize() entrypoint so reconnect shares wiring/status semantics.

        Manual re-init and shutdown share the lifecycle lock and generation
        checks, so stale self-heal attempts cannot replace newer state.
        """
        stop_event = self._self_heal_stop
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                interval = max(0.0, float(self._resolve_self_heal_interval_seconds()))
                # One interruptible wait for the full interval: set stop_event
                # wakes shutdown promptly without polling every 50ms.
                if stop_event is None:
                    if interval > 0:
                        await asyncio.sleep(interval)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=interval)
                        return
                    except asyncio.TimeoutError:
                        pass

                if stop_event is not None and stop_event.is_set():
                    return
                if not self._configured:
                    return
                if self._is_cache_attached() and self._reachable is True:
                    self._self_heal_armed = False
                    return

                logger.info(
                    "AAWM alias routing Redis self-heal reattempting connectivity."
                )
                try:
                    await self.initialize()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "AAWM alias routing Redis self-heal attempt failed",
                        exc_info=True,
                    )
                    continue

                if self._is_cache_attached() and self._reachable is True:
                    logger.info(
                        "AAWM alias routing Redis self-heal restored durable cache."
                    )
                    self._self_heal_armed = False
                    if not self._local_state_reconciled:
                        self.reconcile_local_routing_state()
                    return
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if self._self_heal_task is current:
                self._self_heal_task = None

    async def shutdown(self) -> None:
        """Detach and disconnect the manager-owned Redis client."""
        async with self._lifecycle_lock:
            self._begin_lifecycle_transition_locked()
            task_to_stop = self._claim_self_heal_task_locked()
            previous_cache = self._take_cached_client_locked()
            self._initialized = False
            self._reachable = "unknown"
            self._error_type = None
            self._configured = False
            self._config_mode = "unconfigured"
            self._namespace = self.DEFAULT_NAMESPACE
            self._config_signature = None
            self._self_heal_armed = False
            self._local_state_reconciled = False

        await self._finish_self_heal_task(task_to_stop)
        await self._disconnect_owned_client(
            previous_cache,
            message="AAWM alias routing Redis disconnect failed",
        )

    def get_status(self) -> Dict[str, Any]:
        """Return sanitized runtime status for alias-routing state cache."""
        # Report the same live namespace/key prefix durable key construction uses.
        # Do not reuse the initialize-time snapshot; runtime env can drift.
        resolution = resolve_alias_routing_state_namespace()
        namespace = str(resolution) if resolution else self.DEFAULT_NAMESPACE
        namespace_source = (
            getattr(resolution, "namespace_source", None)
            or getattr(resolution, "source", None)
        )
        mode = "redis" if self._is_cache_attached() else "memory"
        self_heal_active = bool(
            self._self_heal_armed
            and self._configured
            and not (self._is_cache_attached() and self._reachable is True)
        )
        return {
            "configured": self._configured,
            "config_mode": self._config_mode,
            "mode": mode,
            "state_source": self._resolve_state_source(),
            "reachable": self._reachable,
            "namespace": namespace,
            "namespace_source": namespace_source,
            "key_prefix": self._build_key_prefix(namespace),
            "error_type": self._error_type,
            "self_heal_active": self_heal_active,
            "local_state_reconciled": self._local_state_reconciled,
        }

    def reset(self) -> None:
        """Reset local manager state for tests without disconnecting live clients."""
        self._lifecycle_generation += 1
        task = self._self_heal_task
        if task is not None and not task.done():
            task.cancel()
        self._self_heal_task = None
        if self._self_heal_stop is not None:
            self._self_heal_stop.set()
        self._self_heal_stop = None
        self._detach_cached_client()
        self._active_attempts.clear()
        self._initialized = False
        self._reachable = "unknown"
        self._error_type = None
        self._configured = False
        self._config_mode = "unconfigured"
        self._namespace = self.DEFAULT_NAMESPACE
        self._config_signature = None
        self._self_heal_armed = False
        self._local_state_reconciled = False

    def reconcile_local_routing_state(self) -> None:
        """Invalidate process-local affinity/cooldown maps after Redis reattach.

        Outage-window local pins must not remain authoritative once durable
        cache is reachable again. Callers then rehydrate from Redis.
        """
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
                alias_routing_state,
            )
        except Exception:
            alias_routing_state = None
        managers = []
        if alias_routing_state is not None:
            managers.append(alias_routing_state)
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
                cooldown_state as cooldown_state_mod,
            )

            bound = getattr(cooldown_state_mod, "_manager", None)
            if bound is not None and bound not in managers:
                managers.append(bound)
        except Exception:
            pass
        for manager in managers:
            for family_name in ("codex", "anthropic"):
                family = getattr(manager, family_name, None)
                if family is None:
                    continue
                session_map = getattr(family, "session_affinity_by_key", None)
                if session_map is not None:
                    session_map.clear()
                cooldown_map = getattr(family, "cooldown_until_monotonic_by_key", None)
                if cooldown_map is not None:
                    cooldown_map.clear()
                negative_map = getattr(
                    family, "cooldown_negative_until_monotonic_by_key", None
                )
                if negative_map is not None:
                    negative_map.clear()
        self._local_state_reconciled = True


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


def _parse_durable_write_retry_backoff_seconds(raw: Optional[str]) -> float:
    return AAWMAliasRoutingRedisManager._parse_float(
        raw,
        default=DURABLE_WRITE_RETRY_BACKOFF_SECONDS_DEFAULT,
        minimum=DURABLE_WRITE_RETRY_BACKOFF_SECONDS_MIN,
        maximum=DURABLE_WRITE_RETRY_BACKOFF_SECONDS_MAX,
    )


def get_durable_write_retry_attempts() -> int:
    """Return how many additional SET attempts follow the first durable write."""
    return max(0, int(DURABLE_WRITE_RETRY_ATTEMPTS))


def get_durable_write_retry_backoff_seconds() -> float:
    """Resolve env-tunable backoff between durable write retries."""
    return _parse_durable_write_retry_backoff_seconds(
        os.getenv(DURABLE_WRITE_RETRY_BACKOFF_SECONDS_ENV)
    )


def is_retryable_redis_error(exc: BaseException) -> bool:
    """Classify whether a durable-write failure should receive one bounded retry."""
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    if _REDIS_RETRYABLE_EXCEPTIONS and isinstance(exc, _REDIS_RETRYABLE_EXCEPTIONS):
        return True
    return False
