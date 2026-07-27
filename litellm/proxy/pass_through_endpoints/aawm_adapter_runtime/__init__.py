"""AAWM adapter runtime package (Wave 4)."""

from __future__ import annotations

from functools import lru_cache
from types import FunctionType, ModuleType
from typing import Any

from . import (
    anthropic_adapter_calls,
    anthropic_dispatch,
    codex_candidate_calls,
    codex_dispatch,
    payload_validation,
    request_build,
    sse,
    stream_collect,
    tool_call_restore,
)

__all__ = [
    "install",
    "request_build",
    "sse",
    "tool_call_restore",
    "stream_collect",
    "payload_validation",
    "anthropic_adapter_calls",
    "codex_candidate_calls",
    "codex_dispatch",
    "anthropic_dispatch",
    "install_wave6f",
]


def _rebind_cached_function(
    module: ModuleType,
    name: str,
    host_globals: dict[str, Any],
) -> None:
    cached = getattr(module, name)
    wrapped = cached.__wrapped__
    if wrapped.__globals__ is host_globals:
        host_globals[name] = cached
        return

    rebound = FunctionType(
        wrapped.__code__,
        host_globals,
        wrapped.__name__,
        wrapped.__defaults__,
        wrapped.__closure__,
    )
    rebound.__kwdefaults__ = wrapped.__kwdefaults__
    rebound.__annotations__ = wrapped.__annotations__
    rebound.__doc__ = wrapped.__doc__
    rebound.__module__ = wrapped.__module__
    rebound.__qualname__ = wrapped.__qualname__
    if wrapped.__dict__:
        rebound.__dict__.update(wrapped.__dict__)

    cache_parameters = cached.cache_parameters()
    rebound_cached = lru_cache(
        maxsize=cache_parameters["maxsize"],
        typed=cache_parameters["typed"],
    )(rebound)
    setattr(module, name, rebound_cached)
    host_globals[name] = rebound_cached


def install(host_globals: dict[str, Any]) -> None:
    """Install Wave 6A facades with live host-global lookup."""
    request_build.install(host_globals)
    _rebind_cached_function(
        request_build,
        "_get_anthropic_grok_composer_repair_runtime",
        host_globals,
    )
    sse.install(host_globals)
    tool_call_restore.install(host_globals)
    stream_collect.install(host_globals)
    payload_validation.install(host_globals)


def install_wave6f(host_globals: dict[str, Any]) -> None:
    """Install Wave 6F facades after the Wave 6D/6E host bindings exist."""
    canonical_route_metadata = host_globals.get(
        "_add_route_family_logging_metadata"
    )
    anthropic_adapter_calls.install(host_globals)
    if canonical_route_metadata is not None:
        anthropic_adapter_calls._add_route_family_logging_metadata = (
            canonical_route_metadata
        )
        host_globals["_add_route_family_logging_metadata"] = (
            canonical_route_metadata
        )
    codex_candidate_calls.install(
        host_globals,
        publish_to_module=True,
    )
    codex_dispatch.install(
        host_globals,
        publish_to_module=True,
    )
