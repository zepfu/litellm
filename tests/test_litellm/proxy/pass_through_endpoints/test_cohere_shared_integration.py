"""COHERE-001 shared behavior for the Codex direct-Cohere lane."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    policy,
    rollup,
    selection,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.cohere import (
    classify_cohere_failure,
    is_cohere_api_url,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.registry import (
    _run_passthrough_provider_failure_classifiers,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)

_COHERE_CHAT_URL = httpx.URL("https://api.cohere.com/v2/chat")
_COHERE_MODEL = "cohere/north-mini-code-1-0"
_COHERE_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 43123),
            "server": ("testserver", 80),
        }
    )


def _cohere_candidate(*, route_family: str) -> dict[str, Any]:
    return {
        "provider": policy.CODEX_AUTO_AGENT_COHERE_PROVIDER,
        "model": _COHERE_MODEL,
        "route_family": route_family,
    }


@pytest.fixture()
def _selection_runtime():
    runtime_names = {
        "_get_codex_active_cooldown_state",
        "_get_anthropic_active_cooldown_state",
        "_get_anthropic_merged_codex_openai_cooldown_state",
        "_set_codex_cooldown",
        "_set_anthropic_cooldown",
        "_get_codex_session_affinity",
        "_get_anthropic_session_affinity",
        "_get_openrouter_adapter_active_cooldown_seconds",
        "_extract_client_product_label",
        "_resolve_codex_session_key",
        "_resolve_anthropic_session_key",
        "_has_continuation_state",
        "_is_grok_account_quota_candidate",
        "_get_grok_account_quota_lane_cooldown_key",
        "_is_kimi_code_candidate",
        "_get_kimi_managed_account_cooldown_key",
        "_get_codex_quota_observation_pool",
        "_get_codex_quota_observation_environment",
    }
    previous_runtime = {name: getattr(selection, name) for name in runtime_names}

    async def _zero_cooldown_state(key: str) -> tuple[float, str]:
        return (0.0, "local_fallback")

    async def _noop_cooldown(key: str, seconds: float) -> None:
        pass

    async def _zero_adapter(model: Optional[str]) -> float:
        return 0.0

    selection.configure_selection_runtime(
        get_codex_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_merged_codex_openai_cooldown_state=_zero_cooldown_state,
        set_codex_cooldown=_noop_cooldown,
        set_anthropic_cooldown=_noop_cooldown,
        get_codex_session_affinity=AsyncMock(return_value=None),
        get_anthropic_session_affinity=AsyncMock(return_value=None),
        get_openrouter_adapter_active_cooldown_seconds=_zero_adapter,
        extract_client_product_label=lambda r, b: None,
        resolve_codex_session_key=lambda r, b, *, alias_model: None,
        resolve_anthropic_session_key=lambda r, b, *, alias_model: None,
        has_continuation_state=lambda v: False,
        is_grok_account_quota_candidate=lambda c: False,
        get_grok_account_quota_lane_cooldown_key=lambda c, lk: None,
        is_kimi_code_candidate=lambda c: (
            isinstance(c, dict) and c.get("provider") == "kimi_code"
        ),
        get_kimi_managed_account_cooldown_key=(
            lambda: "kimi_code:__managed_account__:kimi_code_managed_account"
        ),
    )
    runtime_globals = selection._build_codex_auto_agent_candidate_state.__globals__
    overrides = {
        "_resolve_codex_auto_agent_openai_cooldown_lane_key": (
            lambda request: "openai:primary"
        ),
        "_resolve_anthropic_auto_agent_native_cooldown_lane_key": (
            lambda request: "anthropic:primary"
        ),
    }
    try:
        with patch.dict(runtime_globals, overrides):
            yield
    finally:
        for name, value in previous_runtime.items():
            setattr(selection, name, value)


@pytest.mark.parametrize(
    ("status_code", "exc", "name", "failure_class"),
    [
        (401, Exception("invalid credentials"), "cohere_authentication", "auth"),
        (
            429,
            Exception("requests per minute limit reached"),
            "cohere_rpm_rate_limit",
            "rate_limit",
        ),
        (
            429,
            Exception("monthly trial quota exhausted"),
            "cohere_monthly_trial_exhausted",
            "quota_exhausted",
        ),
        (
            404,
            Exception("model retired"),
            "cohere_model_unavailable",
            "model_unavailable",
        ),
        (
            422,
            Exception("invalid request body"),
            "cohere_validation",
            "provider_4xx_other",
        ),
        (
            None,
            httpx.ReadTimeout("timed out"),
            "cohere_timeout_connectivity",
            "transient",
        ),
        (503, Exception("provider unavailable"), "cohere_provider_failure", "provider_5xx"),
    ],
)
def test_cohere_failure_classifier_covers_shared_failure_vocabulary(
    status_code: int | None,
    exc: Exception,
    name: str,
    failure_class: str,
) -> None:
    classification = classify_cohere_failure(
        url=_COHERE_CHAT_URL,
        custom_llm_provider="cohere",
        status_code=status_code,
        exc=exc,
    )

    assert classification is not None
    assert classification.name == name
    assert classification.failure_kind == name
    assert classification.failure_class == failure_class
    assert classification.cooldown_scope == "candidate"
    assert classification.advance_fresh_candidate is True
    assert classification.suppress_traceback is True


def test_cohere_failure_classifier_redacts_upstream_secret_details() -> None:
    secret = "cohere-secret-value"
    classification = classify_cohere_failure(
        url=_COHERE_CHAT_URL,
        custom_llm_provider="cohere",
        status_code=503,
        exc=Exception(f"provider rejected authorization token {secret}"),
    )

    assert classification is not None
    assert secret not in classification.log_error_summary
    assert secret not in repr(classification)


def test_cohere_classifier_requires_an_exact_cohere_host() -> None:
    assert is_cohere_api_url("https://api.cohere.com/v2/chat") is True
    assert is_cohere_api_url("https://api.cohere.ai/v2/chat") is True
    assert is_cohere_api_url("https://api.cohere.com.example/v2/chat") is False
    assert is_cohere_api_url("https://openrouter.ai/api/v1/chat/completions") is False

    assert (
        classify_cohere_failure(
            url=httpx.URL("https://openrouter.ai/api/v1/chat/completions"),
            custom_llm_provider="openrouter",
            status_code=429,
            exc=Exception("rate limit"),
        )
        is None
    )
    assert (
        classify_cohere_failure(
            url=_COHERE_CHAT_URL,
            custom_llm_provider="openrouter",
            status_code=429,
            exc=Exception("rate limit"),
        )
        is None
    )


def test_registry_retains_cohere_failure_classification_metadata() -> None:
    results = _run_passthrough_provider_failure_classifiers(
        request=MagicMock(),
        url=_COHERE_CHAT_URL,
        custom_llm_provider="cohere",
        status_code=429,
        exc=Exception("monthly trial exhausted"),
    )

    assert len(results) == 1
    classification = results[0]
    assert classification.failure_kind == "cohere_monthly_trial_exhausted"
    assert classification.failure_class == "quota_exhausted"
    assert classification.cooldown_scope == "candidate"
    assert classification.advance_fresh_candidate is True


@pytest.mark.asyncio
async def test_cohere_native_lane_stays_distinct_from_openrouter_fallback(
    _selection_runtime,
) -> None:
    request = _request("/v1/responses")
    cohere_state = await selection._build_codex_auto_agent_candidate_state(
        request,
        candidate_template=_cohere_candidate(route_family=_COHERE_ROUTE_FAMILY),
    )
    openrouter_state = await selection._build_codex_auto_agent_candidate_state(
        request,
        candidate_template={
            "provider": "openrouter",
            "model": "openrouter/some/paid-model",
            "route_family": "codex_openrouter_completion_adapter",
        },
    )

    assert cohere_state["lane_key"] == "cohere_native"
    assert (
        openrouter_state["lane_key"]
        == policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    )
    assert cohere_state["lane_key"] != openrouter_state["lane_key"]
    assert cohere_state["cooldown_key"] != openrouter_state["cooldown_key"]


def test_cohere_adapter_preserves_rollup_and_session_provenance_labels() -> None:
    prepared_request_body = {
        "model": "cohere-alias",
        "input": "hello",
        "stream": True,
        "litellm_metadata": {
            "alias_model": "cohere-alias",
            "canonical_session_identity": "session-cohere-1",
            "codex_auto_agent_lane_key": "cohere_native",
            "cost_classification": "trial_reference",
        },
    }

    adapted = codex_candidate_calls._build_codex_cohere_adapter_request_body(
        prepared_request_body=prepared_request_body,
        adapter_model=_COHERE_MODEL,
        upstream_model="north-mini-code-1-0",
        config=adapter_config.CODEX_COHERE_CHAT_COMPLETIONS,
    )
    metadata = adapted["litellm_metadata"]

    assert metadata["passthrough_route_family"] == _COHERE_ROUTE_FAMILY
    assert metadata["route_family"] == _COHERE_ROUTE_FAMILY
    assert metadata["codex_cohere_adapter_model"] == _COHERE_MODEL
    assert metadata["codex_cohere_upstream_model"] == "north-mini-code-1-0"
    assert metadata["codex_adapter_target_endpoint"] == "cohere:/v2/chat"
    assert metadata["canonical_session_identity"] == "session-cohere-1"
    assert metadata["codex_auto_agent_lane_key"] == "cohere_native"
    assert metadata["cost_classification"] == "trial_reference"
    assert f"route:{_COHERE_ROUTE_FAMILY}" in metadata["tags"]
    assert metadata["langfuse_spans"][-1]["name"] == (
        "codex.cohere_chat_completions_adapter"
    )


def test_cohere_rollup_labels_native_target_and_cooling_status(monkeypatch) -> None:
    status_events: list[dict] = []
    rollup_records: list[dict] = []
    rollup_globals = rollup._record_auto_agent_alias_route_status_rollup.__globals__
    monkeypatch.setitem(
        rollup_globals,
        "emit_aawm_route_status_event",
        lambda **kwargs: status_events.append(kwargs),
    )
    monkeypatch.setitem(
        rollup_globals,
        "record_aawm_route_rollup",
        lambda **kwargs: rollup_records.append(kwargs),
    )

    rollup._record_auto_agent_alias_route_status_rollup(
        {
            "event_type": "candidate_retryable_failure",
            "candidate_status": "cooldown_set",
            "failure_class": "quota_exhausted",
            "cooldown_scope": "candidate",
            "alias_model": "cohere-alias",
            "model": _COHERE_MODEL,
            "route_family": _COHERE_ROUTE_FAMILY,
            "incoming_endpoint": "/v1/responses",
            "rollup_group_header_label": "litellm#Codex",
            "host_name": "thoth",
            "source_error": "monthly trial exhausted",
        }
    )

    assert status_events[0]["model_label"] == _COHERE_MODEL
    assert status_events[0]["status"] == "Cooling Down"
    assert rollup_records == [
        {
            "group_header_label": "litellm#Codex@thoth",
            "incoming_endpoint": "/v1/responses",
            "outgoing_target": "api.cohere.com/v2/chat",
            "model_label": f"{_COHERE_MODEL}(cohere-alias)",
            "effort": "none",
            "turns": 0,
            "status": "Cooling Down",
            "message": "monthly trial exhausted",
        }
    ]


def test_success_handler_cohere_gate_requires_native_host_or_provider() -> None:
    handler = PassThroughEndpointLogging()

    assert handler.is_cohere_route("https://api.cohere.com/v2/chat") is True
    assert handler.is_cohere_route("https://api.cohere.ai/v1/embed") is True
    assert handler.is_cohere_route("/v2/chat", custom_llm_provider="cohere") is True
    assert handler.is_cohere_route("/v2/chat", custom_llm_provider="openrouter") is False
    assert handler.is_cohere_route("https://api.cohere.com.example/v2/chat") is False
    assert handler.is_cohere_route("https://openrouter.ai/v2/chat") is False
