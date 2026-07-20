from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy


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
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()
    yield
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()


def test_should_keep_the_complete_moonshot_alias_order_and_only_one_cross_ingress_alias() -> None:
    assert [candidate["model"] for candidate in policy.CODEX_AAWM_CODE_CANDIDATES] == [
        "gpt-5.3-codex-spark",
        "kimi_code/k3-high",
        "xai/grok-4.5",
        "grok-composer-2.5-fast",
        "oa_xai/grok-build",
        "gpt-5.6-terra",
        "gpt-5.5",
    ]
    assert [candidate["model"] for candidate in policy.ANTHROPIC_AAWM_CODE_CANDIDATES] == [
        "gpt-5.3-codex-spark",
        "kimi_code/k3-high",
        "xai/grok-4.5",
        "grok-composer-2.5-fast",
        "oa_xai/grok-build",
        "claude-sonnet-5[1m]",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ]
    assert [candidate["model"] for candidate in policy.CODEX_AAWM_LOW_CANDIDATES[-2:]] == [
        "kimi_code/kimi-for-coding",
        "gpt-5.4-mini",
    ]
    assert [candidate["model"] for candidate in policy.ANTHROPIC_AAWM_LOW_CANDIDATES[-2:]] == [
        "kimi_code/kimi-for-coding",
        "claude-haiku-4-5-20251001",
    ]

    for candidates in (
        policy.CODEX_AAWM_LOW_CANDIDATES,
        policy.ANTHROPIC_AAWM_LOW_CANDIDATES,
    ):
        assert all(candidate["model"] != "kimi_code/kimi-for-coding-highspeed" for candidate in candidates)

    assert [candidate["model"] for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS["aawm-sota-moonshot"]] == [
        "kimi_code/k3-max",
        "kimi_code/k3-high",
    ]
    assert [
        candidate["route_family"] for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS["aawm-sota-moonshot"]
    ] == [
        "codex_kimi_chat_completions_adapter",
        "codex_kimi_chat_completions_adapter",
    ]
    assert [
        candidate["route_family"] for candidate in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS["aawm-sota-moonshot"]
    ] == [
        "anthropic_kimi_chat_completions_adapter",
        "anthropic_kimi_chat_completions_adapter",
    ]
    assert all(
        candidate["metadata_gate"] == "think_effort"
        for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS["aawm-sota-moonshot"]
    )
    assert "sota-moonshot" not in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "aawm-sota-moonshot-anthropic" not in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS


@pytest.mark.asyncio
async def test_should_select_k3_high_after_forced_spark_cooldown_for_both_code_ingresses() -> None:
    spark_key = "openai:gpt-5.3-codex-spark:__default__"
    await lpe._set_codex_auto_agent_cooldown(spark_key, 60.0)

    codex_selection = await lpe._select_codex_auto_agent_candidate(
        request=_request("/v1/responses"),
        request_body=_codex_body("aawm-code"),
    )
    anthropic_selection = await lpe._select_anthropic_auto_agent_candidate(
        request=_request("/v1/messages"),
        request_body=_anthropic_body("aawm-code-anthropic"),
    )

    assert codex_selection["candidate"]["model"] == "kimi_code/k3-high"
    assert codex_selection["candidate"]["route_family"] == ("codex_kimi_chat_completions_adapter")
    assert anthropic_selection["candidate"]["model"] == "kimi_code/k3-high"
    assert anthropic_selection["candidate"]["route_family"] == ("anthropic_kimi_chat_completions_adapter")


@pytest.mark.asyncio
async def test_should_preserve_sota_moonshot_continuation_affinity_per_ingress() -> None:
    codex_request = _request("/v1/responses")
    codex_initial = await lpe._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("aawm-sota-moonshot"),
    )
    await lpe._set_codex_auto_agent_session_affinity(
        codex_initial["session_key"],
        codex_initial["candidate"],
    )
    codex_continuation = await lpe._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("aawm-sota-moonshot", continuation=True),
    )

    anthropic_request = _request("/v1/messages")
    anthropic_initial = await lpe._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("aawm-sota-moonshot"),
    )
    await lpe._set_anthropic_auto_agent_session_affinity(
        anthropic_initial["session_key"],
        anthropic_initial["candidate"],
    )
    anthropic_continuation = await lpe._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("aawm-sota-moonshot", continuation=True),
    )

    assert codex_continuation["candidate"]["model"] == "kimi_code/k3-max"
    assert codex_continuation["selection_reason"] == "session_affinity"
    assert anthropic_continuation["candidate"]["model"] == "kimi_code/k3-max"
    assert anthropic_continuation["selection_reason"] == "session_affinity"


@pytest.mark.asyncio
async def test_should_persist_one_kimi_managed_account_lane_and_continue_to_grok() -> None:
    cache = _FakeDurableAliasCache()
    kimi_candidate = dict(policy.CODEX_AAWM_CODE_CANDIDATES[1])
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
        lpe,
        "_get_aawm_alias_routing_dual_cache",
        return_value=cache,
    ):
        scope = await lpe._set_codex_auto_agent_candidate_cooldowns(
            request=_request("/v1/responses"),
            candidate=kimi_candidate,
            lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
            selected_cooldown_key=("kimi_code:kimi_code/k3-high:kimi_code_managed_account"),
            cooldown_seconds=exact_reset_seconds,
            error_class="kimi_code_managed_account",
            kimi_failure_metadata=safe_quota_metadata,
        )
        assert scope == "managed_account"

        managed_key = lpe._get_kimi_code_managed_account_cooldown_key()
        assert await lpe._get_codex_auto_agent_active_cooldown_seconds(managed_key) > 0
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
        lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()

        highspeed_state = await lpe._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template={
                "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
                "model": "kimi_code/kimi-for-coding-highspeed",
                "route_family": "codex_kimi_chat_completions_adapter",
                "last_resort": False,
            },
        )
        standard_state = await lpe._build_anthropic_auto_agent_candidate_state(
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

        await lpe._set_codex_auto_agent_cooldown(
            "openai:gpt-5.3-codex-spark:__default__",
            exact_reset_seconds,
        )
        selection = await lpe._select_codex_auto_agent_candidate(
            request=_request("/v1/responses"),
            request_body=_codex_body("aawm-code"),
        )

    assert selection["candidate"]["model"] == "xai/grok-4.5"
    skipped_kimi = next(item for item in selection["skipped"] if item["model"] == "kimi_code/k3-high")
    assert skipped_kimi["cooldown_scope"] == "managed_account"
    assert skipped_kimi["cooldown_seconds"] <= exact_reset_seconds
    assert skipped_kimi["cooldown_seconds"] > exact_reset_seconds - 2.0
    assert time.time() < next(iter(cache.payloads.values()))["expires_at_epoch"]


@pytest.mark.asyncio
async def test_should_keep_kimi_capability_failures_candidate_scoped_and_malformed_telemetry_non_cooling() -> None:
    candidate = dict(policy.CODEX_AAWM_CODE_CANDIDATES[1])
    candidate_key = "kimi_code:kimi_code/k3-high:kimi_code_managed_account"
    capability_metadata = {
        "kind": "unsupported_effort",
        "scope": "candidate",
        "upstream_id": "k3",
        "metadata_gate": "think_effort",
        "status_code": 400,
        "trace_id": "kimi-trace_019",
        "reset_reason": "unsupported_effort",
    }
    scope = await lpe._set_codex_auto_agent_candidate_cooldowns(
        request=_request("/v1/responses"),
        candidate=candidate,
        lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
        selected_cooldown_key=candidate_key,
        cooldown_seconds=60.0,
        error_class="kimi_code_candidate_failure",
        kimi_failure_metadata=capability_metadata,
    )

    assert scope == "candidate"
    assert await lpe._get_codex_auto_agent_active_cooldown_seconds(candidate_key) > 0
    assert (
        await lpe._get_codex_auto_agent_active_cooldown_seconds(lpe._get_kimi_code_managed_account_cooldown_key()) == 0
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
    malformed_scope = await lpe._set_codex_auto_agent_candidate_cooldowns(
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
        await lpe._get_codex_auto_agent_active_cooldown_seconds(
            "kimi_code:kimi_code/kimi-for-coding:kimi_code_managed_account"
        )
        == 0
    )


def test_should_record_allowlisted_kimi_selection_telemetry_without_secrets() -> None:
    secret = "Bearer moonshot-secret-token"
    exc = RuntimeError(secret)
    candidate = dict(policy.CODEX_AAWM_CODE_CANDIDATES[1])
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

    safe_metadata = lpe._get_safe_kimi_code_probe_failure_metadata(
        exc,
        candidate=candidate,
    )
    lpe._update_codex_auto_agent_retryable_attempt_record(
        attempt_record=attempt,
        exc=exc,
        error_class="kimi_code_managed_account",
        cooldown_seconds=12.0,
        cooldown_scope="managed_account",
        alias_model="aawm-code",
        candidate=candidate,
        kimi_failure_metadata=safe_metadata,
    )

    assert attempt["kimi_code_failure"] == {
        "alias": "aawm-code",
        "candidate": "kimi_code/k3-high",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "scope": "managed_account",
        "reset_reason": "quota_exhausted",
        "kind": "quota",
        "status_code": 429,
        "trace_id": "kimi-trace_021",
    }
    assert secret not in repr(attempt)
