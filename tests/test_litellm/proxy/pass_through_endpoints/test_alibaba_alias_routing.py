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
    for alias in ("sota-alibaba", "sota-deepseek"):
        assert (
            snapshot_select._lookup_active_snapshot_canonical_alias(alias)
            == alias
        )
        codex_candidates = snapshot_select._select_snapshot_candidates(
            alias,
            ingress="codex",
        )
        anthropic_candidates = snapshot_select._select_snapshot_candidates(
            alias,
            ingress="anthropic",
        )
        assert codex_candidates
        assert {
            (candidate["provider"], candidate["model"])
            for candidate in codex_candidates
        } == {
            (candidate["provider"], candidate["model"])
            for candidate in anthropic_candidates
        }
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in codex_candidates
        )
        assert all(
            candidate["provider"] == "alibaba_token_plan"
            for candidate in anthropic_candidates
        )


def test_should_prefer_coding_plan_then_alibaba_on_public_sota_zai() -> None:
    assert (
        snapshot_select._lookup_active_snapshot_canonical_alias("sota-zai")
        == "sota-zai"
    )
    codex_candidates = snapshot_select._select_snapshot_candidates(
        "sota-zai",
        ingress="codex",
    )
    anthropic_candidates = snapshot_select._select_snapshot_candidates(
        "sota-zai",
        ingress="anthropic",
    )
    assert [
        (candidate["provider"], candidate["model"], candidate["route_family"])
        for candidate in codex_candidates
    ] == [
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3",
            "codex_zai_coding_plan_chat_completions_adapter",
        ),
        (
            "alibaba_token_plan",
            "alibaba_token_plan/glm-5.2",
            "codex_alibaba_token_plan_chat_completions_adapter",
        ),
    ]
    assert [candidate["model"] for candidate in anthropic_candidates] == [
        "alibaba_token_plan/glm-5.2"
    ]
    assert all(
        candidate["provider"] == "alibaba_token_plan"
        for candidate in anthropic_candidates
    )


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
    codex_candidates = snapshot_select._select_snapshot_candidates(
        "sota-alibaba",
        ingress="codex",
    )
    anthropic_candidates = snapshot_select._select_snapshot_candidates(
        "sota-alibaba",
        ingress="anthropic",
    )
    assert len(codex_candidates) >= 2
    assert anthropic_candidates

    codex_state = await selection._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=codex_candidates[0],
    )
    codex_fallback_state = await selection._build_codex_auto_agent_candidate_state(
        _request("/v1/responses"),
        candidate_template=codex_candidates[1],
    )
    anthropic_state = await selection._build_anthropic_auto_agent_candidate_state(
        _request("/v1/messages"),
        candidate_template=anthropic_candidates[0],
    )

    assert codex_state["lane_key"] == policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    assert codex_fallback_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
    assert anthropic_state["lane_key"] == (policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY)
