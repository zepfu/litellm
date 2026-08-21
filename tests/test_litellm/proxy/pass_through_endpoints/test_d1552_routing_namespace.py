"""D1-552: routing namespace source must be explicit vs observability-derived.

Explicit ``AAWM_ALIAS_ROUTING_STATE_NAMESPACE`` wins. Missing explicit may still
derive from Langfuse env (compatibility) but the source must be distinguishable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from litellm.proxy import aawm_alias_routing_redis
from litellm.proxy.aawm_alias_routing_redis import AAWMAliasRoutingRedisManager


def _clear_namespace_env(monkeypatch) -> None:
    for key in (
        "AAWM_ALIAS_ROUTING_STATE_NAMESPACE",
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LITELLM_AAWM_ERROR_LOG_ENV",
        "AAWM_ALIAS_ROUTING_REDIS_HOST",
        "AAWM_ALIAS_ROUTING_REDIS_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _resolution_source(resolution) -> str:
    if isinstance(resolution, str):
        pytest.fail(
            "resolve_alias_routing_state_namespace must report source "
            "(explicit | observability_derived | default), not a bare string"
        )
    if isinstance(resolution, dict):
        source = resolution.get("namespace_source") or resolution.get("source")
        namespace = resolution.get("namespace")
    else:
        source = getattr(resolution, "namespace_source", None) or getattr(
            resolution, "source", None
        )
        namespace = getattr(resolution, "namespace", None)
    assert source in {"explicit", "observability_derived", "default"}
    assert isinstance(namespace, str) and namespace
    return source


def _resolution_namespace(resolution) -> str:
    if isinstance(resolution, dict):
        return str(resolution.get("namespace"))
    return str(getattr(resolution, "namespace"))


def test_explicit_routing_namespace_wins_over_langfuse_env(monkeypatch) -> None:
    _clear_namespace_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "aawm-routing-dev-v1")
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "prod")

    resolution = aawm_alias_routing_redis.resolve_alias_routing_state_namespace()
    assert _resolution_source(resolution) == "explicit"
    assert _resolution_namespace(resolution) == "aawm-routing-dev-v1"


def test_missing_explicit_derives_from_langfuse_but_is_distinguishable(
    monkeypatch,
) -> None:
    _clear_namespace_env(monkeypatch)
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")

    resolution = aawm_alias_routing_redis.resolve_alias_routing_state_namespace()
    assert _resolution_source(resolution) == "observability_derived"
    assert _resolution_namespace(resolution) == "aawm-routing-dev-v1"


def test_missing_explicit_and_observability_env_is_default_source(monkeypatch) -> None:
    _clear_namespace_env(monkeypatch)
    resolution = aawm_alias_routing_redis.resolve_alias_routing_state_namespace()
    assert _resolution_source(resolution) == "default"
    assert _resolution_namespace(resolution) == AAWMAliasRoutingRedisManager.DEFAULT_NAMESPACE


def test_status_reports_namespace_source_when_redis_configured(monkeypatch) -> None:
    _clear_namespace_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "prod")
    manager = AAWMAliasRoutingRedisManager()
    manager._configured = True
    manager._config_mode = "host"
    status = manager.get_status()
    assert status["namespace"] == "aawm-routing-prod-v1"
    assert status.get("namespace_source") == "observability_derived"


def test_explicit_namespaces_do_not_join_when_langfuse_env_differs(monkeypatch) -> None:
    _clear_namespace_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "plane-a")
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")
    first = aawm_alias_routing_redis.resolve_alias_routing_state_namespace()

    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "plane-b")
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "prod")
    second = aawm_alias_routing_redis.resolve_alias_routing_state_namespace()

    assert _resolution_source(first) == "explicit"
    assert _resolution_source(second) == "explicit"
    assert _resolution_namespace(first) != _resolution_namespace(second)
