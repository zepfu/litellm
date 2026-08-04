from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_apply,
    cooldown_state,
    attempt_records,
    durable,
    error_signals,
    policy,
    selection,
    snapshot_select,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)


class _FakeDurableAliasCache:
    def __init__(self) -> None:
        self.redis_cache = self
        self.payloads: dict[str, dict[str, Any]] = {}

    async def async_get_cache(self, *, key: str, **_: Any) -> Any:
        return self.payloads.get(key)

    async def async_set_cache(
        self,
        *,
        key: str,
        value: dict[str, Any],
        **_: Any,
    ) -> None:
        self.payloads[key] = dict(value)


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


def _codex_body(alias: str, *, continuation: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": alias,
        "input": "implement the requested change",
        "litellm_metadata": {"session_id": "moonshot-codex-session"},
    }
    if continuation:
        body["previous_response_id"] = "resp_moonshot_continuation"
    return body


def _anthropic_body(alias: str, *, continuation: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": alias,
        "messages": [{"role": "user", "content": "implement the requested change"}],
        "litellm_metadata": {"session_id": "moonshot-anthropic-session"},
    }
    if continuation:
        body["messages"].append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_moonshot_continuation",
                        "name": "Read",
                        "input": {"path": "README.md"},
                    }
                ],
            }
        )
    return body


@pytest.fixture(autouse=True)
def _reset_moonshot_alias_state() -> None:
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(compile_directory(DEFAULT_CONFIG_DIR))
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    alias_routing_state.anthropic.session_affinity_by_key.clear()
    yield
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    alias_routing_state.anthropic.session_affinity_by_key.clear()
    snapshot_select.set_active_routing_snapshot(previous_snapshot)


def test_should_register_the_canonical_moonshot_alias_for_both_ingresses() -> None:
    codex_candidates = snapshot_select._get_codex_auto_agent_candidates_for_alias(
        "sota-moonshot"
    )
    anthropic_candidates = (
        selection._get_anthropic_candidates_for_alias_snapshot_aware(
            "sota-moonshot"
        )
    )

    assert [candidate["model"] for candidate in codex_candidates] == [
        "kimi_code/k3"
    ]
    assert [candidate["model"] for candidate in anthropic_candidates] == [
        "kimi_code/k3"
    ]
    assert codex_candidates[0]["route_family"] == (
        "codex_kimi_chat_completions_adapter"
    )
    assert anthropic_candidates[0]["route_family"] == (
        "anthropic_kimi_chat_completions_adapter"
    )


@pytest.mark.asyncio
async def test_should_select_an_available_work_other_branch_after_spark_cooldown() -> None:
    spark_key = "openai:gpt-5.3-codex-spark:__default__"
    await cooldown_state._set_codex_auto_agent_cooldown(spark_key, 60.0)

    codex_selection = await selection._select_codex_auto_agent_candidate(
        request=_request("/v1/responses"),
        request_body=_codex_body("work"),
    )
    anthropic_selection = await selection._select_anthropic_auto_agent_candidate(
        request=_request("/v1/messages"),
        request_body=_anthropic_body("work"),
    )

    expected_codex_routes = {
        "oa_xai/grok-4.5": "codex_xai_oauth_responses_adapter",
        "kimi_code/k3": "codex_kimi_chat_completions_adapter",
        "alibaba_token_plan/qwen3.8-max": (
            "codex_alibaba_token_plan_chat_completions_adapter"
        ),
    }
    expected_anthropic_routes = {
        "oa_xai/grok-4.5": "anthropic_xai_oauth_responses_adapter",
        "kimi_code/k3": "anthropic_kimi_chat_completions_adapter",
        "alibaba_token_plan/qwen3.8-max": (
            "anthropic_alibaba_token_plan_chat_completions_adapter"
        ),
    }
    assert codex_selection["candidate"]["route_family"] == expected_codex_routes[
        codex_selection["candidate"]["model"]
    ]
    assert (
        anthropic_selection["candidate"]["route_family"]
        == expected_anthropic_routes[anthropic_selection["candidate"]["model"]]
    )


@pytest.mark.asyncio
async def test_should_preserve_sota_moonshot_continuation_affinity_per_ingress() -> None:
    codex_request = _request("/v1/responses")
    codex_initial = await selection._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("sota-moonshot"),
    )
    await cooldown_state._set_codex_auto_agent_session_affinity(
        codex_initial["session_key"],
        codex_initial["candidate"],
    )
    codex_continuation = await selection._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("sota-moonshot", continuation=True),
    )

    anthropic_request = _request("/v1/messages")
    anthropic_initial = await selection._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("sota-moonshot"),
    )
    await cooldown_state._set_anthropic_auto_agent_session_affinity(
        anthropic_initial["session_key"],
        anthropic_initial["candidate"],
    )
    anthropic_continuation = await selection._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("sota-moonshot", continuation=True),
    )

    assert codex_continuation["candidate"]["model"] == "kimi_code/k3"
    assert codex_continuation["selection_reason"] == "session_affinity"
    assert anthropic_continuation["candidate"]["model"] == "kimi_code/k3"
    assert anthropic_continuation["selection_reason"] == "session_affinity"


@pytest.mark.asyncio
async def test_should_not_retry_bounded_kimi_invalid_request_for_continuation() -> None:
    request = _request("/v1/responses")
    body = _codex_body("sota-moonshot", continuation=True)
    terminal_error = HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": "Managed Kimi Code rejected the request shape.",
                "type": "invalid_request_error",
                "code": "kimi_code_invalid_request",
            }
        },
    )

    with patch.object(
        lpe,
        "_handle_codex_kimi_chat_completions_adapter_route",
        new=AsyncMock(side_effect=terminal_error),
    ) as kimi_handler:
        with pytest.raises(HTTPException) as caught:
            await lpe._handle_codex_auto_agent_alias_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                target_url="https://chatgpt.com/backend-api/codex/responses",
                api_key=None,
                forward_headers=True,
            )

    assert caught.value is terminal_error
    assert kimi_handler.await_count == 1


@pytest.mark.asyncio
async def test_should_persist_one_kimi_managed_account_lane_and_continue_to_grok() -> None:
    cache = _FakeDurableAliasCache()
    kimi_candidate = {
        "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "reasoning_effort": "max",
    }
    safe_quota_metadata = {
        "kind": "quota",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 429,
        "trace_id": "kimi-trace_016",
        "reset_reason": "quota_exhausted",
    }
    exact_reset_seconds = 17.0

    with patch.object(
        durable,
        "get_aawm_alias_routing_dual_cache",
        return_value=cache,
    ), patch.object(
        lpe,
        "_get_aawm_alias_routing_dual_cache",
        return_value=cache,
    ):
        scope = await cooldown_apply._set_codex_auto_agent_candidate_cooldowns(
            request=_request("/v1/responses"),
            candidate=kimi_candidate,
            lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
            selected_cooldown_key=(
                "kimi_code:kimi_code/k3:kimi_code_managed_account"
            ),
            cooldown_seconds=exact_reset_seconds,
            error_class="kimi_code_managed_account",
            kimi_failure_metadata=safe_quota_metadata,
        )
        assert scope == "managed_account"

        managed_key = error_signals._get_kimi_code_managed_account_cooldown_key()
        assert await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(managed_key) > 0
        alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
        alias_routing_state.anthropic.cooldown_until_monotonic_by_key.clear()

        highspeed_state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template={
                "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
                "model": "kimi_code/kimi-for-coding-highspeed",
                "route_family": "codex_kimi_chat_completions_adapter",
                "last_resort": False,
            },
        )
        standard_state = await selection._build_anthropic_auto_agent_candidate_state(
            _request("/v1/messages"),
            candidate_template={
                "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
                "model": "kimi_code/kimi-for-coding",
                "route_family": "anthropic_kimi_chat_completions_adapter",
                "last_resort": False,
            },
        )

        assert highspeed_state["cooldown_seconds"] > 0
        assert standard_state["cooldown_seconds"] > 0
        assert highspeed_state["cooldown_scope"] == "managed_account"
        assert standard_state["cooldown_scope"] == "managed_account"

        await cooldown_state._set_codex_auto_agent_cooldown(
            "openai:gpt-5.3-codex-spark:__default__",
            exact_reset_seconds,
        )
        selection_result = await selection._select_codex_auto_agent_candidate(
            request=_request("/v1/responses"),
            request_body=_codex_body("work"),
        )

    assert selection_result["candidate"]["model"] in {
        "oa_xai/grok-4.5",
        "alibaba_token_plan/qwen3.8-max",
    }
    skipped_kimi = next(
        item
        for item in selection_result["skipped"]
        if item["model"] == "kimi_code/k3"
    )
    assert skipped_kimi["cooldown_scope"] == "managed_account"
    assert skipped_kimi["cooldown_seconds"] <= exact_reset_seconds
    assert skipped_kimi["cooldown_seconds"] > exact_reset_seconds - 2.0
    assert time.time() < next(iter(cache.payloads.values()))["expires_at_epoch"]


@pytest.mark.asyncio
async def test_should_keep_kimi_capability_failures_candidate_scoped_and_malformed_telemetry_non_cooling() -> None:
    candidate = {
        "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "reasoning_effort": "max",
    }
    candidate_key = "kimi_code:kimi_code/k3:kimi_code_managed_account"
    capability_metadata = {
        "kind": "unsupported_effort",
        "scope": "candidate",
        "upstream_id": "k3",
        "metadata_gate": "think_effort",
        "status_code": 400,
        "trace_id": "kimi-trace_019",
        "reset_reason": "unsupported_effort",
    }
    scope = await cooldown_apply._set_codex_auto_agent_candidate_cooldowns(
        request=_request("/v1/responses"),
        candidate=candidate,
        lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
        selected_cooldown_key=candidate_key,
        cooldown_seconds=60.0,
        error_class="kimi_code_candidate_failure",
        kimi_failure_metadata=capability_metadata,
    )

    assert scope == "candidate"
    assert await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(candidate_key) > 0
    assert (
        await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(error_signals._get_kimi_code_managed_account_cooldown_key()) == 0
    )

    malformed_metadata = {
        "kind": "malformed",
        "scope": "telemetry",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 422,
        "trace_id": "kimi-trace_020",
        "reset_reason": "malformed_provider_response",
    }
    malformed_scope = await cooldown_apply._set_codex_auto_agent_candidate_cooldowns(
        request=_request("/v1/responses"),
        candidate={
            "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
            "model": "kimi_code/kimi-for-coding",
            "route_family": "codex_kimi_chat_completions_adapter",
            "last_resort": False,
        },
        lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
        selected_cooldown_key=("kimi_code:kimi_code/kimi-for-coding:kimi_code_managed_account"),
        cooldown_seconds=3 * 60 * 60.0,
        error_class="kimi_code_no_cooldown",
        kimi_failure_metadata=malformed_metadata,
    )

    assert malformed_scope == "none"
    assert (
        await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(
            "kimi_code:kimi_code/kimi-for-coding:kimi_code_managed_account"
        )
        == 0
    )


def test_should_record_allowlisted_kimi_selection_telemetry_without_secrets() -> None:
    secret = "Bearer moonshot-secret-token"
    exc = RuntimeError(secret)
    candidate = {
        "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "reasoning_effort": "max",
    }
    metadata = {
        "kind": "quota",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 429,
        "trace_id": "kimi-trace_021",
        "reset_reason": "quota_exhausted",
    }
    setattr(exc, "kimi_code_probe_failure_metadata", metadata)
    attempt: dict[str, Any] = {}

    safe_metadata = error_signals._get_safe_kimi_code_probe_failure_metadata(
        exc,
        candidate=candidate,
    )
    attempt_records._update_codex_auto_agent_retryable_attempt_record(
        attempt_record=attempt,
        exc=exc,
        error_class="kimi_code_managed_account",
        cooldown_seconds=12.0,
        cooldown_scope="managed_account",
        alias_model="work",
        candidate=candidate,
        kimi_failure_metadata=safe_metadata,
    )

    assert attempt["kimi_code_failure"] == {
        "alias": "work",
        "candidate": "kimi_code/k3",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "scope": "managed_account",
        "reset_reason": "quota_exhausted",
        "kind": "quota",
        "status_code": 429,
        "trace_id": "kimi-trace_021",
    }
    assert secret not in repr(attempt)
