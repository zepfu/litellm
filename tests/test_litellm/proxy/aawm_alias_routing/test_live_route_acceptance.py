"""Live-route acceptance harness (Wave 1 of the god-module decomposition +
R3 remediation plan, ``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``).

This file grows across Waves 1-3 into the full 5-scenario harness. Wave 1
covers scenario (e) -- round-robin rotation semantics -- plus the supporting
per-request selection-context / commit-tracking behaviors.

Every scenario drives the REAL wrapper ``_handle_codex_auto_agent_alias_route``
(the wrapper at ``llm_passthrough_endpoints.py:25381``, so the counting call
inside it -- ``max_candidate_attempts=len(_get_codex_auto_agent_candidates_for_alias(...))``
-- actually executes), the REAL selector (``_select_codex_auto_agent_candidate``),
and REAL process-local state (cooldown maps, round-robin cursor, affinity map).
Only ``perform_candidate_request`` (via the OpenRouter completion performer)
and the durable writer are stubbed, following the ``AsyncMock`` pattern used
by ``test_rr054_candidate_singleflight.py:786``.

Pre-fix (before the Wave-1 R3-2 fix lands), the round-robin cursor is mutated
by EVERY call to ``_get_codex_auto_agent_candidates_for_alias`` -- including
the wrapper's own attempt-count getter call and the selector's/affinity
resolver's internal getter calls -- so a single live request advances the
cursor multiple times. With two tied candidates this means the cursor advances
an even number of times per request and rotation never reaches live traffic
(one candidate is selected on every request). These tests pin the correct
per-request-single-commit behavior and MUST fail against pre-fix develop.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _minimal_request(session_id: str, *, continuation: bool = False) -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {
        "session_id": session_id,
        "user-agent": "codex-cli/1.0",
        "originator": "codex_cli_rs",
    }
    request.query_params = {}
    request.url = MagicMock()
    request.scope = {
        "path": "/openai_passthrough/v1/responses",
        "query_string": b"",
        "parsed_body": None,
    }
    request.state = MagicMock()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


def _reset_all_alias_routing_state() -> None:
    """Full process-local reset via the Wave-0 guardrail helper.

    ``reset_alias_routing_state_for_tests`` clears both the manager-owned
    state (cooldown/affinity/probe-locks) and the god-module-owned
    singletons (read-pilot gate, round-robin cursor, active snapshot).
    """
    lpe.reset_alias_routing_state_for_tests()


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    previous_snapshot = lpe.get_active_routing_snapshot()
    _reset_all_alias_routing_state()
    yield
    _reset_all_alias_routing_state()
    lpe.set_active_routing_snapshot(previous_snapshot)


def _two_candidate_round_robin_yaml(
    *,
    name_a: str = "rr-a",
    name_b: str = "rr-b",
    priority: int = 50,
) -> str:
    return f"""
defaults: {{}}
aliases:
  - name: read
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: {name_a}
        route_family: codex_openrouter_completion_adapter
        priority: {priority}
      - provider: openrouter
        model: {name_b}
        route_family: codex_openrouter_completion_adapter
        priority: {priority}
"""


def _three_candidate_round_robin_yaml(priority: int = 50) -> str:
    return f"""
defaults: {{}}
aliases:
  - name: read
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: rr-a
        route_family: codex_openrouter_completion_adapter
        priority: {priority}
      - provider: openrouter
        model: rr-b
        route_family: codex_openrouter_completion_adapter
        priority: {priority}
      - provider: openrouter
        model: rr-c
        route_family: codex_openrouter_completion_adapter
        priority: {priority}
"""


def _lower_priority_fallback_yaml() -> str:
    return """
defaults: {}
aliases:
  - name: read
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: rr-a
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: rr-b
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: rr-fallback
        route_family: codex_openrouter_completion_adapter
        priority: 10
"""


def _lane_key_for_model(model: str) -> str:
    candidate = {
        "provider": "openrouter",
        "model": model,
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    }
    lane_key = lpe._CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    return lpe._codex_auto_agent_candidate_key(candidate, lane_key)


class _StructuredCapacityError(RuntimeError):
    """A retryable capacity failure classified for cooldown."""

    def __init__(self) -> None:
        super().__init__("Selected model is at capacity. Please try a different model.")


def _install_openrouter_performer(
    handler: Callable[..., Any],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Patch the OpenRouter completion performer + durable writer; return restorers."""
    original_perform = lpe._perform_codex_auto_agent_openrouter_completion_request
    original_write = lpe._write_aawm_alias_routing_durable_payload
    lpe._perform_codex_auto_agent_openrouter_completion_request = handler  # type: ignore[assignment]
    lpe._write_aawm_alias_routing_durable_payload = AsyncMock(return_value=True)  # type: ignore[assignment]

    def _restore() -> None:
        lpe._perform_codex_auto_agent_openrouter_completion_request = original_perform  # type: ignore[assignment]
        lpe._write_aawm_alias_routing_durable_payload = original_write  # type: ignore[assignment]

    return handler, _restore


async def _drive_wrapper(
    *,
    session_id: str,
    body_extra: dict[str, Any] | None = None,
) -> Response:
    request = _minimal_request(session_id)
    body: dict[str, Any] = {
        "model": "read",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "litellm_metadata": {"session_id": session_id},
    }
    if body_extra:
        body.update(body_extra)
    return await lpe._handle_codex_auto_agent_alias_route(
        endpoint="/v1/responses",
        request=request,
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body=body,
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=True,
    )


_SUCCESS_RESPONSE = Response(content='{"ok":true}', media_type="application/json")


# ---------------------------------------------------------------------------
# Scenario e1: round-robin rotation reaches live traffic
# ---------------------------------------------------------------------------


async def test_scenario_e1_round_robin_rotation_reaches_live_traffic() -> None:
    snapshot = compiler.compile_yaml(_two_candidate_round_robin_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    leaders: list[str] = []

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        for i in range(4):
            result = await _drive_wrapper(session_id=f"e1-session-{i}")
            assert isinstance(result, Response)
    finally:
        restore()

    assert len(leaders) == 4
    assert leaders[0] != leaders[1], f"leaders did not alternate: {leaders!r}"
    assert leaders[1] != leaders[2], f"leaders did not alternate: {leaders!r}"
    assert leaders[2] != leaders[3], f"leaders did not alternate: {leaders!r}"
    assert set(leaders) == {"rr-a", "rr-b"}
    assert leaders.count("rr-a") == 2, f"leaders={leaders!r}"
    assert leaders.count("rr-b") == 2, f"leaders={leaders!r}"


# ---------------------------------------------------------------------------
# Scenario e2: affinity re-selection does not consume rotation
# ---------------------------------------------------------------------------


async def test_scenario_e2_affinity_does_not_consume_rotation() -> None:
    snapshot = compiler.compile_yaml(_two_candidate_round_robin_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    leaders: list[str] = []

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        # Request 1 (cold S1) selects leader L1 and establishes affinity.
        await _drive_wrapper(session_id="e2-session-1")
        assert len(leaders) == 1
        l1 = leaders[0]

        # Request 2: same session, WITH continuation state, so affinity
        # re-selects L1 (not a fresh rotation pick).
        await _drive_wrapper(
            session_id="e2-session-1",
            body_extra={"previous_response_id": "resp_e2_continuation_1"},
        )
        assert len(leaders) == 2
        assert leaders[1] == l1, f"affinity path did not re-select L1: {leaders!r}"

        # Request 3: cold session S2 must select the NEXT rotation slot --
        # i.e. the other candidate, not a repeat of L1 and not a skipped slot.
        await _drive_wrapper(session_id="e2-session-2")
        assert len(leaders) == 3
        l3 = leaders[2]
        assert l3 != l1, (
            "cold request after affinity re-selection did not advance to the "
            f"next rotation slot: leaders={leaders!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Selection-context enumeration memoization (unit-level proof)
# ---------------------------------------------------------------------------


async def test_selection_context_enumeration_called_once_per_request() -> None:
    snapshot = compiler.compile_yaml(_two_candidate_round_robin_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    call_count = 0
    original_getter = lpe._get_codex_auto_agent_candidates_for_alias

    def _counting_getter(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_getter(*args, **kwargs)

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        return _SUCCESS_RESPONSE

    lpe._get_codex_auto_agent_candidates_for_alias = _counting_getter  # type: ignore[assignment]
    _, restore = _install_openrouter_performer(_performer)
    try:
        result = await _drive_wrapper(session_id="enum-once-session")
        assert isinstance(result, Response)
    finally:
        restore()
        lpe._get_codex_auto_agent_candidates_for_alias = original_getter  # type: ignore[assignment]

    assert call_count == 1, (
        "alias enumeration must resolve exactly once per request (memoized "
        f"selection context); observed {call_count} calls"
    )


# ---------------------------------------------------------------------------
# Round-robin commit tracks the ACTUAL selected tied-tier member
# ---------------------------------------------------------------------------


async def test_round_robin_commit_tracks_selected_tier_member() -> None:
    snapshot = compiler.compile_yaml(_three_candidate_round_robin_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    # Pre-cool A so the first cold request must fall through to B.
    a_key = _lane_key_for_model("rr-a")
    await lpe._set_codex_auto_agent_cooldown(a_key, 30.0)

    leaders: list[str] = []

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        await _drive_wrapper(session_id="commit-tracks-1")
        assert leaders == ["rr-b"], f"expected B selected while A cools: {leaders!r}"

        # Clear A's cooldown before the next cold request.
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(a_key, None)

        await _drive_wrapper(session_id="commit-tracks-2")
        assert leaders[-1] == "rr-c", (
            "next rotated leader after a fallback selection must be the slot "
            f"AFTER the actual selected candidate (B), i.e. C, not a blind "
            f"cursor+1 repeat of B: leaders={leaders!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Lower-priority fallback does not consume top-tier rotation
# ---------------------------------------------------------------------------


async def test_lower_priority_fallback_does_not_consume_top_tier_rotation() -> None:
    snapshot = compiler.compile_yaml(_lower_priority_fallback_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    top_tier_a_key = _lane_key_for_model("rr-a")
    top_tier_b_key = _lane_key_for_model("rr-b")
    # Cool BOTH top-tier candidates so selection falls through to rr-fallback.
    await lpe._set_codex_auto_agent_cooldown(top_tier_a_key, 30.0)
    await lpe._set_codex_auto_agent_cooldown(top_tier_b_key, 30.0)

    leaders: list[str] = []

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        await _drive_wrapper(session_id="lowtier-1")
        assert leaders == ["rr-fallback"], f"expected fallback selected: {leaders!r}"

        # Recover top tier.
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(top_tier_a_key, None)
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(top_tier_b_key, None)

        await _drive_wrapper(session_id="lowtier-2")
        first_recovered_leader = leaders[-1]
        assert first_recovered_leader in {"rr-a", "rr-b"}, f"leaders={leaders!r}"

        # Repeat the cold-recovery request to confirm the top-tier cursor
        # position was NOT advanced by the earlier lower-tier selection --
        # i.e. the top-tier rotation still starts from its original position
        # (rr-a) rather than having silently advanced during the fallback.
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(top_tier_a_key, None)
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(top_tier_b_key, None)
        # Reset rotation to a known baseline and re-derive expected leader.
        lpe.reset_module_singletons()
        lpe.set_active_routing_snapshot(snapshot)

        await _drive_wrapper(session_id="lowtier-3")
        baseline_leader = leaders[-1]
        assert baseline_leader == "rr-a", f"expected baseline top-tier leader rr-a: leaders={leaders!r}"
        assert baseline_leader == first_recovered_leader, (
            "a selection outside the rotated top tier (the lower-priority "
            "fallback) must not have advanced the top-tier rotation cursor: "
            f"leaders={leaders!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Retry failover commits each actual rotated selection
# ---------------------------------------------------------------------------


async def test_retry_failover_commits_each_actual_rotated_selection() -> None:
    snapshot = compiler.compile_yaml(_three_candidate_round_robin_yaml())
    lpe.set_active_routing_snapshot(snapshot)

    leaders: list[str] = []

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        if adapter_model == "rr-a":
            raise _StructuredCapacityError()
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        # A is selected first (cold rotation start) and fails; B is selected
        # on the SAME request (retry) and succeeds.
        result = await _drive_wrapper(session_id="failover-1")
        assert isinstance(result, Response)
        assert leaders == ["rr-a", "rr-b"], f"expected A-fails then B-succeeds: {leaders!r}"

        # The next cold request must start at C (the slot after the actual
        # selected-and-succeeded candidate B), not repeat A or B.
        await _drive_wrapper(session_id="failover-2")
        assert leaders[-1] == "rr-c", (
            "next request after an A-fails/B-succeeds retry must rotate to C: " f"leaders={leaders!r}"
        )
    finally:
        restore()
