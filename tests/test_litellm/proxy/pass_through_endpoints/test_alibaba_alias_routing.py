from __future__ import annotations

from typing import Any

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy


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


def test_should_register_all_alibaba_aliases_for_both_ingresses() -> None:
    expected_models = {
        "aawm-sota-alibaba": [
            "alibaba_token_plan/qwen3.8-max-preview",
            "alibaba_token_plan/qwen3.7-max",
        ],
        "aawm-sota-deepseek": ["alibaba_token_plan/deepseek-v4-pro"],
        "aawm-sota-glm": ["alibaba_token_plan/glm-5.2"],
    }

    for alias, models in expected_models.items():
        assert [candidate["model"] for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS[alias]] == models
        assert [candidate["model"] for candidate in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS[alias]] == models
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS[alias]
        )
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS[alias]
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
    assert [candidate["model"] for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS["aawm-sota-moonshot"]] == [
        "kimi_code/k3-max",
        "kimi_code/k3-high",
    ]


@pytest.mark.asyncio
async def test_should_preserve_alibaba_continuation_affinity_per_ingress() -> None:
    codex_request = _request("/v1/responses")
    codex_initial = await lpe._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("aawm-sota-alibaba"),
    )
    await lpe._set_codex_auto_agent_session_affinity(
        codex_initial["session_key"],
        codex_initial["candidate"],
    )
    codex_continuation = await lpe._select_codex_auto_agent_candidate(
        request=codex_request,
        request_body=_codex_body("aawm-sota-alibaba", continuation=True),
    )

    anthropic_request = _request("/v1/messages")
    anthropic_initial = await lpe._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("aawm-sota-alibaba"),
    )
    await lpe._set_anthropic_auto_agent_session_affinity(
        anthropic_initial["session_key"],
        anthropic_initial["candidate"],
    )
    anthropic_continuation = await lpe._select_anthropic_auto_agent_candidate(
        request=anthropic_request,
        request_body=_anthropic_body("aawm-sota-alibaba", continuation=True),
    )

    assert codex_continuation["candidate"]["model"] == ("alibaba_token_plan/qwen3.8-max-preview")
    assert codex_continuation["selection_reason"] == "session_affinity"
    assert anthropic_continuation["candidate"]["model"] == ("alibaba_token_plan/qwen3.8-max-preview")
    assert anthropic_continuation["selection_reason"] == "session_affinity"


@pytest.mark.asyncio
async def test_should_share_one_alibaba_credential_lane_across_models_and_ingresses() -> None:
    codex_state = await lpe._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=policy.CODEX_AAWM_SOTA_ALIBABA_CANDIDATES[0],
    )
    codex_fallback_state = await lpe._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=policy.CODEX_AAWM_SOTA_ALIBABA_CANDIDATES[1],
    )
    anthropic_state = await lpe._build_anthropic_auto_agent_candidate_state(
        _request("/v1/messages"),
        candidate_template=policy.ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES[0],
    )

    assert codex_state["lane_key"] == policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    assert codex_fallback_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
    assert anthropic_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
