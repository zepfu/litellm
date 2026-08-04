from __future__ import annotations

from typing import Any

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state,
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
        "litellm_metadata": {"session_id": "alibaba-codex-session"},
    }
    if continuation:
        body["previous_response_id"] = "resp_alibaba_continuation"
    return body


def _anthropic_body(alias: str, *, continuation: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": alias,
        "messages": [{"role": "user", "content": "implement the requested change"}],
        "litellm_metadata": {"session_id": "alibaba-anthropic-session"},
    }
    if continuation:
        body["messages"].append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_alibaba_continuation",
                        "name": "Bash",
                        "input": {"command": "date --iso-8601=seconds"},
                    }
                ],
            }
        )
    return body


@pytest.fixture(autouse=True)
def _reset_alibaba_alias_state() -> None:
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


def test_should_register_all_alibaba_aliases_for_both_ingresses() -> None:
    expected_models = {
        "sota-alibaba": [
            "alibaba_token_plan/qwen3.8-max",
            "alibaba_token_plan/qwen3.7-max",
        ],
        "sota-deepseek": ["alibaba_token_plan/deepseek-v4-pro"],
    }

    for alias, models in expected_models.items():
        codex_candidates = snapshot_select._get_codex_auto_agent_candidates_for_alias(
            alias
        )
        anthropic_candidates = (
            selection._get_anthropic_candidates_for_alias_snapshot_aware(alias)
        )
        assert [candidate["model"] for candidate in codex_candidates] == models
        assert [candidate["model"] for candidate in anthropic_candidates] == models
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in codex_candidates
        )
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in anthropic_candidates
        )


def test_should_place_qwen_flash_immediately_before_kimi_and_terminal_fallback() -> None:
    assert [candidate["model"] for candidate in policy.CODEX_AAWM_LOW_CANDIDATES[-3:]] == [
        "alibaba_token_plan/qwen3.6-flash",
        "kimi_code/kimi-for-coding",
        "gpt-5.4-mini",
    ]
    assert [candidate["model"] for candidate in policy.ANTHROPIC_AAWM_LOW_CANDIDATES[-3:]] == [
        "alibaba_token_plan/qwen3.6-flash",
        "kimi_code/kimi-for-coding",
        "claude-haiku-4-5-20251001",
    ]
@pytest.mark.asyncio
async def test_should_preserve_alibaba_continuation_affinity_per_ingress() -> None:
    codex_request = _request("/v1/responses")
    codex_initial = await selection._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("sota-alibaba"),
    )
    await cooldown_state._set_codex_auto_agent_session_affinity(
        codex_initial["session_key"],
        codex_initial["candidate"],
    )
    codex_continuation = await selection._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("sota-alibaba", continuation=True),
    )

    anthropic_request = _request("/v1/messages")
    anthropic_initial = await selection._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("sota-alibaba"),
    )
    await cooldown_state._set_anthropic_auto_agent_session_affinity(
        anthropic_initial["session_key"],
        anthropic_initial["candidate"],
    )
    anthropic_continuation = await selection._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("sota-alibaba", continuation=True),
    )

    assert codex_continuation["candidate"]["model"] == "alibaba_token_plan/qwen3.8-max"
    assert codex_continuation["selection_reason"] == "session_affinity"
    assert (
        anthropic_continuation["candidate"]["model"]
        == "alibaba_token_plan/qwen3.8-max"
    )
    assert anthropic_continuation["selection_reason"] == "session_affinity"


@pytest.mark.asyncio
async def test_should_share_one_alibaba_credential_lane_across_models_and_ingresses() -> None:
    codex_state = await selection._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=policy.CODEX_AAWM_SOTA_ALIBABA_CANDIDATES[0],
    )
    codex_fallback_state = await selection._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=policy.CODEX_AAWM_SOTA_ALIBABA_CANDIDATES[1],
    )
    anthropic_state = await selection._build_anthropic_auto_agent_candidate_state(
        _request("/v1/messages"),
        candidate_template=policy.ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES[0],
    )

    assert codex_state["lane_key"] == policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    assert codex_fallback_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
    assert anthropic_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
