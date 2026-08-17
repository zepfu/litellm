"""Live-route acceptance harness (Waves 1-2 of the god-module decomposition +
R3 remediation plan, ``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``).

This file grows across Waves 1-3 into the full 5-scenario harness. Wave 1
covers scenario (e) -- round-robin rotation semantics -- plus the supporting
per-request selection-context / commit-tracking behaviors. Wave 2 adds
scenarios (b) and (c) plus the (e-scope) R3-3 model-scope test: the R3-1
exact-key in-lock cooldown publish (single-flight under contention, Kimi
managed-account / no-cooldown scope targets, Grok account-quota lane
publish), the R3-3 bounded model-scope classification, and regression pins
for the round-2 evidence-gate live-path behaviors re-driven through the
WRAPPER.

Every scenario drives the REAL wrapper ``_handle_codex_auto_agent_alias_route``
(including its canonical snapshot enumeration used to derive
``max_candidate_attempts``), the REAL selector
(``_select_codex_auto_agent_candidate``),
and REAL process-local state (cooldown maps, round-robin cursor, affinity map).
Only ``perform_candidate_request`` (via the OpenRouter completion performer)
and the durable writer are stubbed, following the ``AsyncMock`` pattern used
by ``test_rr054_candidate_singleflight.py:786``.

The round-robin tests pin one request-local snapshot enumeration and one
selection commit per request, including affinity re-selection paths.

Pre-fix (before the Wave-2 R3-1/R3-3 changes land), the candidate-loop's
cooldown apply happens OUTSIDE the probe lock -- a suspension point between
the failed probe and the applied cooldown lets a concurrent sibling also
enter upstream before the cooldown/evidence becomes visible -- and structured
429/rate-limit classifies ``scope="provider"`` instead of the DECIDED
``scope="model"``. See individual test docstrings for the concrete pre-fix
values.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_apply,
    cooldown_state,
    error_signals,
    lane_keys,
    policy,
    selection,
    snapshot_select,
    state as alias_state,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    CandidateSelection,
    CooldownPublicationPlan,
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
    request.state = SimpleNamespace()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


def _reset_all_alias_routing_state() -> None:
    """Full process-local reset via the Wave-0 guardrail helper.

    ``reset_alias_routing_state_for_tests`` clears both the manager-owned
    state (cooldown/affinity/probe-locks) and the god-module-owned
    singletons (failure evidence, round-robin cursor, active snapshot).
    """
    lpe.reset_alias_routing_state_for_tests()


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    _reset_all_alias_routing_state()
    yield
    _reset_all_alias_routing_state()
    snapshot_select.set_active_routing_snapshot(previous_snapshot)


def _two_candidate_round_robin_yaml(
    *,
    name_a: str = "rr-a",
    name_b: str = "rr-b",
    priority: int = 50,
) -> str:
    return f"""
defaults: {{}}
aliases:
  - name: basic
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
  - name: basic
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
  - name: basic
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
    canonical_alias = snapshot_select._lookup_active_snapshot_canonical_alias(
        "basic"
    )
    assert canonical_alias is not None
    candidate = next(
        candidate
        for candidate in snapshot_select._select_snapshot_candidates(
            canonical_alias,
            ingress="codex",
        )
        if candidate["model"] == model
    )
    lane_key = policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    return lane_keys._codex_auto_agent_candidate_key(
        candidate,
        lane_key,
        cooldown_identity_tag=candidate.get("cooldown_identity_tag"),
    )


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


def _install_candidate_request_performer(
    handler: Callable[..., Any],
) -> Callable[[], None]:
    """Patch the wrapper-level candidate performer and durable writer."""
    original_perform = lpe._perform_codex_auto_agent_alias_candidate_request
    original_write = lpe._write_aawm_alias_routing_durable_payload
    lpe._perform_codex_auto_agent_alias_candidate_request = handler  # type: ignore[assignment]
    lpe._write_aawm_alias_routing_durable_payload = AsyncMock(return_value=True)  # type: ignore[assignment]

    def _restore() -> None:
        lpe._perform_codex_auto_agent_alias_candidate_request = original_perform  # type: ignore[assignment]
        lpe._write_aawm_alias_routing_durable_payload = original_write  # type: ignore[assignment]

    return _restore


async def _drive_wrapper(
    *,
    session_id: str,
    body_extra: dict[str, Any] | None = None,
) -> Response:
    request = _minimal_request(session_id)
    body: dict[str, Any] = {
        "model": "basic",
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
        canonical_alias="basic",
    )


_SUCCESS_RESPONSE = Response(content='{"ok":true}', media_type="application/json")


# ---------------------------------------------------------------------------
# Scenario e1: round-robin rotation reaches live traffic
# ---------------------------------------------------------------------------


async def test_scenario_e1_round_robin_rotation_reaches_live_traffic() -> None:
    snapshot = compiler.compile_yaml(_two_candidate_round_robin_yaml())
    snapshot_select.set_active_routing_snapshot(snapshot)

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
    snapshot_select.set_active_routing_snapshot(snapshot)

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
    snapshot_select.set_active_routing_snapshot(snapshot)

    call_count = 0
    original_selector = snapshot_select._select_snapshot_candidates

    def _counting_selector(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_selector(*args, **kwargs)

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        return _SUCCESS_RESPONSE

    snapshot_select._select_snapshot_candidates = _counting_selector  # type: ignore[assignment]
    _, restore = _install_openrouter_performer(_performer)
    try:
        result = await _drive_wrapper(session_id="enum-once-session")
        assert isinstance(result, Response)
    finally:
        restore()
        snapshot_select._select_snapshot_candidates = original_selector  # type: ignore[assignment]

    assert call_count == 1, (
        "alias enumeration must resolve exactly once per request (memoized "
        f"selection context); observed {call_count} calls"
    )


# ---------------------------------------------------------------------------
# Round-robin commit tracks the ACTUAL selected tied-tier member
# ---------------------------------------------------------------------------


async def test_round_robin_commit_tracks_selected_tier_member() -> None:
    snapshot = compiler.compile_yaml(_three_candidate_round_robin_yaml())
    snapshot_select.set_active_routing_snapshot(snapshot)

    # Pre-cool A so the first cold request must fall through to B.
    a_key = _lane_key_for_model("rr-a")
    await cooldown_state._set_codex_auto_agent_cooldown(a_key, 30.0)

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
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(a_key, None)

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
    snapshot_select.set_active_routing_snapshot(snapshot)

    top_tier_a_key = _lane_key_for_model("rr-a")
    top_tier_b_key = _lane_key_for_model("rr-b")
    # Cool BOTH top-tier candidates so selection falls through to rr-fallback.
    await cooldown_state._set_codex_auto_agent_cooldown(top_tier_a_key, 30.0)
    await cooldown_state._set_codex_auto_agent_cooldown(top_tier_b_key, 30.0)

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
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(top_tier_a_key, None)
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(top_tier_b_key, None)

        await _drive_wrapper(session_id="lowtier-2")
        first_recovered_leader = leaders[-1]
        assert first_recovered_leader in {"rr-a", "rr-b"}, f"leaders={leaders!r}"

        # Repeat the cold-recovery request to confirm the top-tier cursor
        # position was NOT advanced by the earlier lower-tier selection --
        # i.e. the top-tier rotation still starts from its original position
        # (rr-a) rather than having silently advanced during the fallback.
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(top_tier_a_key, None)
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(top_tier_b_key, None)
        # Reset rotation to a known baseline and re-derive expected leader.
        lpe.reset_module_singletons()
        snapshot_select.set_active_routing_snapshot(snapshot)

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
    snapshot_select.set_active_routing_snapshot(snapshot)

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


# ---------------------------------------------------------------------------
# Wave 2 helpers: structured-failure exceptions + a local probe counter
# ---------------------------------------------------------------------------


class _StructuredUpstream429(Exception):
    """Upstream failure that surfaces a structured HTTP 429 + Retry-After."""

    def __init__(self, retry_after_seconds: int = 12) -> None:
        super().__init__("rate limited by upstream")
        self.status_code = 429
        self.upstream_headers = {"Retry-After": str(retry_after_seconds)}


def _marker_only_capacity_error() -> RuntimeError:
    """A retryable capacity failure with NO structured status code (marker tier)."""
    return RuntimeError("Selected model is at capacity. Please try a different model.")


class _ProbeCounter:
    """Tracks total and peak concurrent probe entries for one upstream candidate.

    Local copy of the pattern in
    ``test_rr054_candidate_singleflight.py:_ProbeCounter`` -- kept file-local
    so this harness has no cross-test-module coupling.
    """

    def __init__(self, *, hold_seconds: float = 0.15) -> None:
        self.total = 0
        self.current = 0
        self.max_current = 0
        self._guard = asyncio.Lock()
        self.hold_seconds = hold_seconds
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def run(
        self,
        *,
        outcome: str = "fail",
        success_response: Optional[Response] = None,
        structured_failure: bool = False,
    ) -> Response:
        async with self._guard:
            self.total += 1
            self.current += 1
            self.max_current = max(self.max_current, self.current)
            self.entered.set()
        try:
            if not self.release.is_set():
                try:
                    await asyncio.wait_for(self.release.wait(), timeout=self.hold_seconds)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0)
            if outcome == "success":
                assert success_response is not None
                return success_response
            if structured_failure:
                raise _StructuredUpstream429()
            raise _StructuredCapacityError()
        finally:
            async with self._guard:
                self.current -= 1


def _install_openrouter_and_opencode_performers(
    *,
    openrouter_handler: Callable[..., Any],
) -> Callable[[], None]:
    """Patch the OpenRouter performer + durable writer; return a restorer."""
    original_perform = lpe._perform_codex_auto_agent_openrouter_completion_request
    original_write = lpe._write_aawm_alias_routing_durable_payload
    lpe._perform_codex_auto_agent_openrouter_completion_request = openrouter_handler  # type: ignore[assignment]
    lpe._write_aawm_alias_routing_durable_payload = AsyncMock(return_value=True)  # type: ignore[assignment]

    def _restore() -> None:
        lpe._perform_codex_auto_agent_openrouter_completion_request = original_perform  # type: ignore[assignment]
        lpe._write_aawm_alias_routing_durable_payload = original_write  # type: ignore[assignment]

    return _restore


# ---------------------------------------------------------------------------
# (R3-1, scenario c1) concurrent cold probes single-flight under contention
# ---------------------------------------------------------------------------


async def test_scenario_c1_concurrent_cold_probes_single_flight_under_contention() -> None:
    """Four concurrent wrapper requests for one cold candidate must single-flight.

    The memory cooldown WRITE (``_set_codex_auto_agent_cooldown``, which
    acquires ``_codex_auto_agent_lock``) is monkeypatched to inject a real
    ``await`` suspension AFTER the underlying probe fails but BEFORE the
    cooldown becomes visible in ``_codex_auto_agent_cooldown_until_monotonic_by_key``.
    Pre-fix, the loop releases ``probe_lock`` in its inner ``finally`` clause
    -- BEFORE ``apply_cooldown_fn`` (and therefore this delayed write) even
    runs -- so a follower queued on the SAME probe lock acquires it, reads
    the still-zero cooldown state, and re-probes: ``probe_total == 2``.
    Post-fix (R3-1: widen the locked region so the cooldown publish happens
    INSIDE the probe lock), a follower cannot acquire the probe lock until
    the cooldown write has completed, so it never re-probes:
    ``probe_total == 1``.
    """
    primary_model = "openrouter/cohere/north-mini-code:free"
    secondary_model = "openrouter/owl-alpha"
    snapshot_select.set_active_routing_snapshot(
        compiler.compile_yaml(
            f"""
defaults: {{}}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: {primary_model}
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: {secondary_model}
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""
        )
    )
    probe = _ProbeCounter(hold_seconds=0.05)

    original_set_cooldown = cooldown_state._set_codex_auto_agent_cooldown

    async def _delayed_set_cooldown(cooldown_key: str, cooldown_seconds: float) -> None:
        # Suspend AFTER the probe has already failed (and, pre-fix, after the
        # probe lock has already been released) but BEFORE the cooldown
        # becomes visible -- the exact window a follower can race into.
        await asyncio.sleep(0.1)
        await original_set_cooldown(cooldown_key, cooldown_seconds)

    async def _openrouter_performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        if adapter_model == primary_model:
            return await probe.run(outcome="fail", structured_failure=True)
        return _SUCCESS_RESPONSE

    restore = _install_openrouter_and_opencode_performers(openrouter_handler=_openrouter_performer)
    cooldown_state._set_codex_auto_agent_cooldown = _delayed_set_cooldown  # type: ignore[assignment]
    try:
        results = await asyncio.gather(
            _drive_wrapper(session_id="c1-session-1"),
            _drive_wrapper(session_id="c1-session-2"),
            _drive_wrapper(session_id="c1-session-3"),
            _drive_wrapper(session_id="c1-session-4"),
            return_exceptions=True,
        )
    finally:
        cooldown_state._set_codex_auto_agent_cooldown = original_set_cooldown  # type: ignore[assignment]
        restore()

    assert probe.max_current == 1, (
        "concurrent cold probes for the same candidate entered upstream "
        f"together (max_current={probe.max_current}, total={probe.total})"
    )
    assert probe.total == 1, (
        "R3-1 gap: more than one upstream probe ran for the same cold "
        f"candidate before the failure/cooldown became visible "
        f"(total={probe.total}, max_current={probe.max_current}). Expected "
        "the cooldown publish to happen inside the probe lock so followers "
        "never re-probe."
    )

    successes = [r for r in results if isinstance(r, Response)]
    assert len(successes) >= 1, (
        "expected at least one request to recover onto the alternate "
        f"candidate ({secondary_model!r}) once the cold probe's failure was "
        f"visible; results={results!r}"
    )


# ---------------------------------------------------------------------------
# (R3-1 negative control, scenario c2) non-cooling failure does not false-singleflight
# ---------------------------------------------------------------------------


async def test_scenario_c2_non_cooling_failure_does_not_false_singleflight() -> None:
    """A basic-lane single marker-only failure (gate says don't cool yet) must
    NOT trigger single-flight suppression for concurrent waiters -- each
    waiter's own attempt is independently serialized through the probe lock
    (total == N, max_concurrent == 1), and the legacy (non-read) success path
    is unaffected by the R3-1 restructure."""
    raw_yaml = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/c2-marker-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    snapshot_select.set_active_routing_snapshot(snapshot)

    probe = _ProbeCounter(hold_seconds=0.05)

    async def _openrouter_performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        return await probe.run(outcome="fail")

    async def _drive_read_once(session_id: str) -> Any:
        request = _minimal_request(session_id)
        body: dict[str, Any] = {
            "model": "basic",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "litellm_metadata": {"session_id": session_id},
        }
        return await lpe._handle_codex_auto_agent_alias_route(
            endpoint="/v1/responses",
            request=request,
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body=body,
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            canonical_alias="basic",
        )

    restore = _install_openrouter_and_opencode_performers(openrouter_handler=_openrouter_performer)
    try:
        results = await asyncio.gather(
            _drive_read_once("c2-session-1"),
            _drive_read_once("c2-session-2"),
            _drive_read_once("c2-session-3"),
            return_exceptions=True,
        )
    finally:
        restore()

    # A single-candidate lane with only one marker-tier failure per request
    # never meets the N-of-M evidence threshold, so the gate does not cool --
    # every waiter must independently probe (no false single-flight
    # suppression across unrelated, non-cooling failures).
    assert probe.total == 3, f"expected every waiter to probe independently: total={probe.total}"
    assert probe.max_current == 1, (
        "waiters must still be serialized through the per-candidate probe "
        f"lock even though no cooldown is published: max_current={probe.max_current}"
    )
    for result in results:
        assert isinstance(result, Exception) or hasattr(result, "status_code")


# ---------------------------------------------------------------------------
# (R3-1 scope targets) Kimi managed-account / no-cooldown publish
# ---------------------------------------------------------------------------


async def _drive_scope_target_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: dict[str, Any],
    lane_key: str,
    selected_cooldown_key: str,
    error_class: str,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    grok_account_quota_exhausted: bool = False,
    cooldown_seconds: float = 30.0,
) -> tuple[
    CooldownPublicationPlan,
    list[tuple[str, tuple[str, ...]]],
]:
    """Drive one failure through the real Codex wrapper and candidate loop."""
    selection = {
        "candidate": candidate,
        "alias_model": "scope-target-fixture",
        "lane_key": lane_key,
        "cooldown_key": selected_cooldown_key,
        "session_key": "wave2-scope-target",
        "selection_reason": "first_available",
        "skipped": [],
        "in_flight_session": False,
        "cooldown_seconds": 0.0,
        "cooldown_state_source": "local_fallback",
    }
    publication_events: list[tuple[str, tuple[str, ...]]] = []
    plans: list[CooldownPublicationPlan] = []
    original_resolver = cooldown_apply._resolve_auto_agent_cooldown_publication_plan
    original_memory_publisher = cooldown_state._publish_codex_cooldown_memory

    async def _select_candidate(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return selection

    def _resolve_publication_plan(
        *,
        request: Optional[Request],
        candidate: dict[str, Any],
        lane_key: Optional[str],
        selected_cooldown_key: str,
        cooldown_seconds: float,
        error_class: Optional[str],
        grok_account_quota_exhausted: bool = False,
        kimi_failure_metadata: Optional[dict[str, Any]] = None,
        codex_failure_evidence_alias: Optional[str] = None,
    ) -> CooldownPublicationPlan:
        plan = original_resolver(
            request=request,
            candidate=candidate,
            lane_key=lane_key,
            selected_cooldown_key=selected_cooldown_key,
            cooldown_seconds=cooldown_seconds,
            error_class=error_class,
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
            codex_failure_evidence_alias=codex_failure_evidence_alias,
        )
        plans.append(plan)
        return plan

    def _publish_memory(*, keys: Sequence[str], seconds: float) -> None:
        publication_events.append(("memory", tuple(keys)))
        original_memory_publisher(keys=keys, seconds=seconds)

    async def _persist_durable(*, keys: Sequence[str], seconds: float) -> None:
        publication_events.append(("durable", tuple(keys)))

    snapshot_select.set_active_routing_snapshot(
        compiler.compile_yaml(
            f"""
defaults: {{}}
aliases:
  - name: scope-target-fixture
    candidates:
      - provider: {candidate["provider"]}
        model: {candidate["model"]}
        route_family: {candidate["route_family"]}
        priority: 100
"""
        )
    )
    monkeypatch.setattr(lpe, "_select_codex_auto_agent_candidate", _select_candidate)
    monkeypatch.setattr(
        lpe,
        "_perform_codex_auto_agent_alias_candidate_request",
        AsyncMock(side_effect=_StructuredUpstream429()),
    )
    monkeypatch.setattr(
        lpe,
        "_get_safe_kimi_code_probe_failure_metadata",
        lambda exc, *, candidate: kimi_failure_metadata,
    )
    monkeypatch.setattr(
        lpe,
        "_classify_kimi_code_auto_agent_probe_failure",
        lambda metadata: (
            error_class if kimi_failure_metadata is not None else None
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_classify_codex_auto_agent_retryable_exhaustion",
        lambda exc: error_class,
    )
    monkeypatch.setattr(
        lpe,
        "_is_codex_auto_agent_grok_account_quota_exhaustion",
        lambda exc, *, candidate: grok_account_quota_exhausted,
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_auto_agent_cooldown_seconds",
        lambda exc, *, candidate: cooldown_seconds,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_auto_agent_cooldown_publication_plan",
        _resolve_publication_plan,
    )
    monkeypatch.setattr(lpe, "_publish_codex_cooldown_memory", _publish_memory)
    monkeypatch.setattr(lpe, "_persist_codex_cooldown_durable", _persist_durable)

    request = _minimal_request("wave2-scope-target")
    with pytest.raises(Exception):
        await lpe._handle_codex_auto_agent_alias_route(
            endpoint="/v1/responses",
            request=request,
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "scope-target-fixture",
                "input": "hello",
                "stream": False,
                "litellm_metadata": {"session_id": "wave2-scope-target"},
            },
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            canonical_alias="scope-target-fixture",
        )

    assert len(plans) == 1
    return plans[0], publication_events


async def test_kimi_managed_account_publishes_only_managed_account_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Kimi managed-account failure publishes ONLY the managed-account
    sentinel key under the probe lock -- not the selected candidate's own
    cooldown key."""
    candidate = {
        "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-high",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
    }
    candidate_key = "kimi_code:kimi_code/k3-high:kimi_code_managed_account"
    metadata = {
        "kind": "quota",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 429,
        "trace_id": "wave2-c1-managed",
        "reset_reason": "quota_exhausted",
    }

    plan, publication_events = await _drive_scope_target_failure(
        monkeypatch,
        candidate=candidate,
        lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
        selected_cooldown_key=candidate_key,
        cooldown_seconds=30.0,
        error_class="kimi_code_managed_account",
        kimi_failure_metadata=metadata,
    )

    managed_key = error_signals._get_kimi_code_managed_account_cooldown_key()
    assert plan.applied_scope == "managed_account"
    assert plan.memory_keys == (managed_key,)
    assert plan.durable_keys == (managed_key,)
    assert publication_events == [
        ("memory", plan.memory_keys),
        ("durable", plan.durable_keys),
    ]
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(managed_key) > 0
    assert candidate_key not in alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key


async def test_kimi_no_cooldown_publishes_no_shared_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Kimi no-cooldown / request-local failure must publish an EMPTY
    shared-memory key set -- neither the candidate key nor the managed-account
    sentinel is cooled."""
    candidate = {
        "provider": policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/kimi-for-coding",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
    }
    candidate_key = "kimi_code:kimi_code/kimi-for-coding:kimi_code_managed_account"
    metadata = {
        "kind": "malformed",
        "scope": "telemetry",
        "upstream_id": "kimi-for-coding",
        "metadata_gate": "none",
        "status_code": 422,
        "trace_id": "wave2-c1-no-cooldown",
        "reset_reason": "malformed_provider_response",
    }

    plan, publication_events = await _drive_scope_target_failure(
        monkeypatch,
        candidate=candidate,
        lane_key=policy.CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
        selected_cooldown_key=candidate_key,
        cooldown_seconds=3 * 60 * 60.0,
        error_class="kimi_code_no_cooldown",
        kimi_failure_metadata=metadata,
    )

    managed_key = error_signals._get_kimi_code_managed_account_cooldown_key()
    assert plan.applied_scope == "none"
    assert plan.memory_keys == ()
    assert plan.durable_keys == ()
    assert publication_events == []
    assert candidate_key not in alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key
    assert managed_key not in alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key


# ---------------------------------------------------------------------------
# (R3-1 scope targets) Grok account-quota candidate + lane publish
# ---------------------------------------------------------------------------


async def test_grok_account_quota_publishes_candidate_and_lane_keys_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Grok account-quota exhaustion failure publishes BOTH the selected
    candidate key and the ``provider:__account_quota__:lane`` key -- a
    concurrent sibling candidate on the same lane must observe the lane key
    and skip probing the exhausted account."""
    grok_candidate = {
        "provider": policy.CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-composer-2.5-fast",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    }
    lane_key = policy.CODEX_AUTO_AGENT_XAI_LANE_KEY
    selected_key = lane_keys._codex_auto_agent_candidate_key(grok_candidate, lane_key)
    lane_cooldown_key = f"{grok_candidate['provider']}:__account_quota__:{lane_key}"

    plan, publication_events = await _drive_scope_target_failure(
        monkeypatch,
        candidate=grok_candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_key,
        cooldown_seconds=3 * 60 * 60.0,
        error_class="capacity_exhausted",
        grok_account_quota_exhausted=True,
    )

    expected_keys = (selected_key, lane_cooldown_key)
    assert plan.applied_scope == "candidate"
    assert plan.memory_keys == expected_keys
    assert plan.durable_keys == expected_keys
    assert publication_events == [
        ("memory", plan.memory_keys),
        ("durable", plan.durable_keys),
    ]
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(selected_key) > 0
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(lane_cooldown_key) > 0

    # A concurrent sibling xAI candidate on the SAME lane (native Grok) must
    # see the lane key's cooldown when it builds its own candidate state.
    sibling_candidate_template = {
        "provider": policy.CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "xai/grok-4.5",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    }
    sibling_state = await selection._build_codex_auto_agent_candidate_state(
        _minimal_request("grok-quota-sibling-session"),
        candidate_template=sibling_candidate_template,
    )
    assert sibling_state["cooldown_seconds"] > 0, (
        "a sibling candidate resolving on the exhausted account's lane must "
        "observe the account-quota lane cooldown, not probe the exhausted "
        f"account again: sibling_state={sibling_state!r}"
    )


# ---------------------------------------------------------------------------
# (R3-3, scenario e3) structured 429 cools only the failing model
# ---------------------------------------------------------------------------


async def test_scenario_e3_structured_429_cools_only_the_failing_model() -> None:
    """Two same-provider OpenRouter candidates + one opencode_zen candidate.

    A structured 429 (Retry-After: 12) on the leader must cool ONLY the
    failing candidate's ``provider:model:lane`` key -- the sibling OpenRouter
    candidate's key stays uncooled, the next attempt selects the sibling, and
    the applied attempt record reports ``cooldown_scope == "model"``.

    Pre-fix: ``cooldown_scope == "provider"`` in telemetry (the classifier
    still emits ``scope="provider"`` for structured 429) while only one key
    is actually cooled -- assertion (4) below is the named pre-fix failure.
    """
    raw_yaml = """
defaults: {}
aliases:
  - name: basic
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: e3-leader
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: e3-sibling
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: opencode_zen
        model: e3-opencode
        route_family: codex_opencode_zen_adapter
        priority: 10
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    snapshot_select.set_active_routing_snapshot(snapshot)

    leaders: list[str] = []
    request_bodies: list[dict[str, Any]] = []

    async def _openrouter_performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        request_bodies.append(request_body)
        if adapter_model == "e3-leader":
            raise _StructuredUpstream429(retry_after_seconds=12)
        return _SUCCESS_RESPONSE

    async def _drive_read_once(session_id: str) -> Any:
        request = _minimal_request(session_id)
        body: dict[str, Any] = {
            "model": "basic",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "litellm_metadata": {"session_id": session_id},
        }
        return await lpe._handle_codex_auto_agent_alias_route(
            endpoint="/v1/responses",
            request=request,
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body=body,
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            canonical_alias="basic",
        )

    restore = _install_openrouter_and_opencode_performers(openrouter_handler=_openrouter_performer)
    try:
        result = await _drive_read_once("e3-session-1")
    finally:
        restore()

    assert isinstance(result, Response)
    assert leaders[0] == "e3-leader"
    assert "e3-sibling" in leaders, f"expected the sibling candidate to be attempted: leaders={leaders!r}"

    leader_key = _lane_key_for_model("e3-leader")
    sibling_key = _lane_key_for_model("e3-sibling")

    leader_remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(leader_key)
    sibling_remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(sibling_key)

    assert leader_remaining == pytest.approx(
        12.0, abs=2.0
    ), f"expected the failing candidate's key to cool ~12s, got {leader_remaining!r}"
    assert sibling_remaining == 0.0, (
        "the sibling OpenRouter candidate's key must NOT be cooled by the "
        f"leader's structured 429: sibling_remaining={sibling_remaining!r}"
    )

    leader_request_body = next(
        (body for model, body in zip(leaders, request_bodies) if model == "e3-sibling"),
        None,
    )
    assert leader_request_body is not None, f"expected a request body for the retried e3-sibling attempt: {leaders!r}"
    litellm_metadata = leader_request_body.get("litellm_metadata", {})
    attempts = litellm_metadata.get("codex_auto_agent_attempts", [])
    leader_attempt = next((a for a in attempts if a.get("model") == "e3-leader"), None)
    assert leader_attempt is not None, f"expected an attempt record for e3-leader: attempts={attempts!r}"
    assert leader_attempt.get("cooldown_scope") == "candidate", (
        "a structured 429 must apply/report generic candidate scope, not "
        f"provider scope; got {leader_attempt.get('cooldown_scope')!r}"
    )


# ---------------------------------------------------------------------------
# (scenario b, port of round-2 live-path tests through the WRAPPER)
# ---------------------------------------------------------------------------


async def test_scenario_b1_structured_429_cools_with_gate_duration() -> None:
    """A structured 429 on the LIVE basic-lane path (through the WRAPPER) must
    cool the live cooldown key with the gate's retry-after-derived duration.

    Regression pin: passes both pre-fix AND post-fix.
    """
    raw_yaml = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/b1-live-model
        route_family: codex_openrouter_completion_adapter
        priority: 500
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    snapshot_select.set_active_routing_snapshot(snapshot)

    async def _openrouter_performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        raise _StructuredUpstream429(retry_after_seconds=12)

    async def _drive_read_once(session_id: str) -> Any:
        request = _minimal_request(session_id)
        body: dict[str, Any] = {
            "model": "basic",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "litellm_metadata": {"session_id": session_id},
        }
        return await lpe._handle_codex_auto_agent_alias_route(
            endpoint="/v1/responses",
            request=request,
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body=body,
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            canonical_alias="basic",
        )

    restore = _install_openrouter_and_opencode_performers(openrouter_handler=_openrouter_performer)
    try:
        with pytest.raises(Exception):
            await _drive_read_once("b1-live-session")
    finally:
        restore()

    live_key = _lane_key_for_model("openrouter/b1-live-model")
    applied_remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
    assert applied_remaining == pytest.approx(12.0, abs=1.5)


async def test_scenario_b2_single_marker_failure_does_not_cool() -> None:
    """A single marker-only (non-structured) failure on the LIVE basic-lane
    path must NOT cool the candidate -- the N-of-M gate needs multiple marker
    events within its window before a key advances toward cooling.

    Regression pin: passes both pre-fix AND post-fix.
    """
    raw_yaml = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/b2-live-model
        route_family: codex_openrouter_completion_adapter
        priority: 500
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    snapshot_select.set_active_routing_snapshot(snapshot)

    async def _openrouter_performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        raise _marker_only_capacity_error()

    async def _drive_read_once(session_id: str) -> Any:
        request = _minimal_request(session_id)
        body: dict[str, Any] = {
            "model": "basic",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "litellm_metadata": {"session_id": session_id},
        }
        return await lpe._handle_codex_auto_agent_alias_route(
            endpoint="/v1/responses",
            request=request,
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body=body,
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            canonical_alias="basic",
        )

    restore = _install_openrouter_and_opencode_performers(openrouter_handler=_openrouter_performer)
    try:
        with pytest.raises(Exception):
            await _drive_read_once("b2-live-session")
    finally:
        restore()

    live_key = _lane_key_for_model("openrouter/b2-live-model")
    applied_remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
    assert applied_remaining == 0.0
    assert (
        alias_state.alias_routing_state.codex_failure_evidence_gate.is_cooled(
            canonical_alias="basic",
            cooldown_key=live_key,
        )
        is False
    )


# ===========================================================================
# Wave 3: R3-4 -- stable cooldown identity + continuation-safe affinity
# ===========================================================================

_REFRESH_PATH = "/aawm/alias-config/refresh"


async def test_snapshot_config_hash_and_cooldown_identity_are_scoped_separately() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: other
    candidates:
      - provider: openrouter
        model: shared-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
  - name: basic
    candidates:
      - provider: opencode_zen
        model: shared-model
        route_family: codex_opencode_zen_adapter
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot)

    basic = snapshot_select._select_snapshot_candidates(
        "basic",
        ingress="codex",
    )
    other = snapshot_select._select_snapshot_candidates(
        "other",
        ingress="codex",
    )
    assert [(candidate["provider"], candidate["model"]) for candidate in basic] == [
        ("opencode_zen", "shared-model")
    ]
    assert [(candidate["provider"], candidate["model"]) for candidate in other] == [
        ("openrouter", "shared-model")
    ]
    assert basic[0]["config_epoch_tag"] == snapshot.config_hash
    assert other[0]["config_epoch_tag"] == snapshot.config_hash
    assert basic[0]["cooldown_identity_tag"] == (
        "alias:basic:opencode_zen:shared-model:codex_opencode_zen_adapter"
    )
    assert other[0]["cooldown_identity_tag"] == (
        "alias:other:openrouter:shared-model:"
        "codex_openrouter_completion_adapter"
    )


async def test_snapshot_selection_exposes_top_level_config_epoch_tag() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: tagged-selection
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot)
    request = _minimal_request("tagged-selection-session")
    request_body = {
        "model": "basic",
        "input": [{"role": "user", "content": "hello"}],
        "litellm_metadata": {"session_id": "tagged-selection-session"},
    }

    selection_result = await selection._select_codex_auto_agent_candidate(
        request=request,
        request_body=request_body,
    )
    typed_selection = CandidateSelection.from_legacy_dict(selection_result)

    assert selection_result["config_epoch_tag"] == snapshot.config_hash
    assert typed_selection.config_epoch_tag == snapshot.config_hash
    assert selection_result["candidate"]["cooldown_identity_tag"] == (
        "alias:basic:openrouter:tagged-selection:"
        "codex_openrouter_completion_adapter"
    )


class _ASGIClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    async def post(self, path: str, *, json: object) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(path, json=json)


def _refresh_client() -> _ASGIClient:
    """Build an async client wrapping the real passthrough refresh router."""
    app = FastAPI()
    app.include_router(lpe.router)
    return _ASGIClient(app)


async def _post_refresh(client: _ASGIClient, yaml_str: str) -> httpx.Response:
    return await client.post(_REFRESH_PATH, json={"yaml": yaml_str})


def _bypass_session_owner_for_cooldown_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep this cooldown test independent of the durable owner subsystem."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity,
    )

    async def _get_unowned_session_owner_record(**_kwargs: Any) -> tuple[None, None, None]:
        return None, None, None

    async def _allow_unowned_session_owner(
        *,
        session_identity: Optional[str] = None,
        **_kwargs: Any,
    ) -> Any:
        return session_affinity.SessionOwnerGuardResult(
            decision=session_affinity.SessionOwnerGuardDecision.NO_SESSION,
            session_identity=session_identity,
        )

    monkeypatch.setattr(
        session_affinity,
        "get_session_owner_record",
        _get_unowned_session_owner_record,
    )
    monkeypatch.setattr(
        session_affinity,
        "ensure_session_owner_guard_for_request",
        _allow_unowned_session_owner,
    )


def _snapshot_candidate_key(
    snapshot: Any,
    model: str,
    *,
    alias_name: str = "basic",
    provider: Optional[str] = "openrouter",
    lane_key: str = policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY,
) -> str:
    assert snapshot_select.get_active_routing_snapshot() is snapshot
    candidate = next(
        candidate
        for candidate in snapshot_select._select_snapshot_candidates(
            alias_name,
            ingress="codex",
        )
        if candidate["model"] == model
        and (provider is None or candidate["provider"] == provider)
    )
    assert candidate["config_epoch_tag"] == snapshot.config_hash
    cooldown_identity_tag = candidate["cooldown_identity_tag"]
    expected = (
        f"h{cooldown_identity_tag}:{candidate['provider']}:"
        f"{candidate['model']}:{lane_key}"
    )
    actual = lane_keys._codex_auto_agent_candidate_key(
        candidate,
        lane_key,
        cooldown_identity_tag=cooldown_identity_tag,
    )
    assert actual == expected
    return actual


# ---------------------------------------------------------------------------
# (scenario a1) refresh activates snapshot and selection uses it
# ---------------------------------------------------------------------------


async def test_scenario_a1_refresh_activates_snapshot_and_selection_uses_it() -> None:
    """Regression pin (passes pre-fix): POST inline YAML to the REAL refresh
    HTTP route; assert 200 + changed=True; drive one wrapper request with
    model='basic'; assert the selected candidate is from the active snapshot."""
    yaml_str = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: a1-snapshot-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    client = _refresh_client()
    resp = await _post_refresh(client, yaml_str)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["changed"] is True

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
        result = await _drive_wrapper(session_id="a1-session")
        assert isinstance(result, Response)
    finally:
        restore()

    assert leaders == ["a1-snapshot-model"], (
        f"expected the snapshot candidate to be selected, got {leaders!r}"
    )


# ---------------------------------------------------------------------------
# (scenario a2) refresh rejects bad YAML with 400
# ---------------------------------------------------------------------------


async def test_scenario_a2_refresh_rejects_bad_yaml_with_400() -> None:
    """Regression pin (passes pre-fix): non-string yaml field -> 400;
    malformed YAML -> 400 with last-known-good retained."""
    client = _refresh_client()

    # Non-string yaml field
    resp_non_string = await client.post(_REFRESH_PATH, json={"yaml": 12345})
    assert resp_non_string.status_code == 400

    # First activate a known-good snapshot
    good_yaml = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: a2-good-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    good_resp = await _post_refresh(client, good_yaml)
    assert good_resp.status_code == 200
    good_hash = good_resp.json()["active_config_hash"]

    # Syntactically malformed YAML -> 400, last-known-good retained
    bad_yaml = "defaults: {}\naliases: [\n"
    bad_resp = await _post_refresh(client, bad_yaml)
    assert bad_resp.status_code == 400

    active = snapshot_select.get_active_routing_snapshot()
    assert active is not None
    assert active.config_hash == good_hash, (
        "last-known-good snapshot must remain active after a failed refresh"
    )


# ---------------------------------------------------------------------------
# (CFG-019, scenario d1) priority refresh preserves cooldown for reused identity
# ---------------------------------------------------------------------------

_D1_SNAPSHOT_A = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d1-leader
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d1-fallback
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""

_D1_SNAPSHOT_B = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d1-leader
        route_family: codex_openrouter_completion_adapter
        priority: 200
      - provider: openrouter
        model: d1-fallback
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


async def test_scenario_d1_priority_refresh_preserves_cooldown_for_reused_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A priority-only refresh changes ``config_hash`` but not cooldown identity.

    After the leader cools under snapshot A, snapshot B reuses its alias,
    provider, model, resolved route semantics, and lane. The third request
    must still select the fallback rather than re-probe the cooled leader.
    """
    _bypass_session_owner_for_cooldown_contract(monkeypatch)
    client = _refresh_client()

    # Activate snapshot A
    resp_a = await _post_refresh(client, _D1_SNAPSHOT_A)
    assert resp_a.status_code == 200
    assert resp_a.json()["changed"] is True
    snapshot_a = snapshot_select.get_active_routing_snapshot()
    assert snapshot_a is not None
    leader_key_a = _snapshot_candidate_key(snapshot_a, "d1-leader")

    calls: list[str] = []
    fail_leader = True

    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        calls.append(adapter_model)
        if adapter_model == "d1-leader" and fail_leader:
            raise _StructuredUpstream429(retry_after_seconds=60)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        # Request 1: leader 429s, fallback succeeds
        result1 = await _drive_wrapper(session_id="d1-session-1")
        assert isinstance(result1, Response)
        assert calls[0] == "d1-leader"
        assert "d1-fallback" in calls

        # Verify leader's cooldown key is cooled
        remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(
            leader_key_a
        )
        assert remaining > 0, f"expected leader key to be cooled, got remaining={remaining}"

        # Request 2: leader cooled -> fallback selected
        calls.clear()
        result2 = await _drive_wrapper(session_id="d1-session-2")
        assert isinstance(result2, Response)
        assert calls == ["d1-fallback"], (
            f"expected fallback while leader cools, got {calls!r}"
        )

        # Activate snapshot B (changed priority -> different config_hash).
        resp_b = await _post_refresh(client, _D1_SNAPSHOT_B)
        assert resp_b.status_code == 200
        assert resp_b.json()["changed"] is True
        assert resp_b.json()["active_config_hash"] != resp_a.json()["active_config_hash"]
        snapshot_b = snapshot_select.get_active_routing_snapshot()
        assert snapshot_b is not None
        leader_key_b = _snapshot_candidate_key(snapshot_b, "d1-leader")

        # If the stable cooldown identity were wrong, a fresh probe would now
        # succeed and expose the regression.
        fail_leader = False

        # Request 3: priority did not change candidate semantics, so the
        # existing leader cooldown survives and fallback remains selected.
        calls.clear()
        result3 = await _drive_wrapper(session_id="d1-session-3")
        assert isinstance(result3, Response)
        assert calls == ["d1-fallback"], (
            "CFG-019: a priority-only refresh must preserve the leader "
            f"cooldown. Got calls={calls!r}."
        )
        assert leader_key_b == leader_key_a
        assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(
            leader_key_b
        ) > 0
    finally:
        restore()


@pytest.mark.parametrize(
    (
        "change",
        "provider_a",
        "model_a",
        "route_family_a",
        "provider_b",
        "model_b",
        "route_family_b",
    ),
    [
        (
            "provider",
            "openrouter",
            "semantic-model",
            "codex_openrouter_completion_adapter",
            "opencode_zen",
            "semantic-model",
            "codex_opencode_zen_adapter",
        ),
        (
            "model",
            "openrouter",
            "semantic-model-a",
            "codex_openrouter_completion_adapter",
            "openrouter",
            "semantic-model-b",
            "codex_openrouter_completion_adapter",
        ),
        (
            "resolved route semantics",
            "openrouter",
            "semantic-model",
            "codex_openrouter_completion_adapter",
            "openrouter",
            "semantic-model",
            "codex_responses",
        ),
    ],
)
async def test_candidate_semantic_change_invalidates_only_its_cooldown_identity(
    change: str,
    provider_a: str,
    model_a: str,
    route_family_a: str,
    provider_b: str,
    model_b: str,
    route_family_b: str,
) -> None:
    snapshot_a = compiler.compile_yaml(
        f"""
defaults: {{}}
aliases:
  - name: basic
    candidates:
      - provider: {provider_a}
        model: {model_a}
        route_family: {route_family_a}
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot_a)
    key_a = _snapshot_candidate_key(
        snapshot_a,
        model_a,
        provider=provider_a,
        lane_key="semantic-lane",
    )
    await cooldown_state._set_codex_auto_agent_cooldown(key_a, 60.0)
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(key_a) > 0

    snapshot_b = compiler.compile_yaml(
        f"""
defaults: {{}}
aliases:
  - name: basic
    candidates:
      - provider: {provider_b}
        model: {model_b}
        route_family: {route_family_b}
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot_b)
    key_b = _snapshot_candidate_key(
        snapshot_b,
        model_b,
        provider=provider_b,
        lane_key="semantic-lane",
    )

    assert snapshot_b.config_hash != snapshot_a.config_hash
    assert key_b != key_a, f"{change} change must rotate cooldown identity"
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(key_b) == 0.0


async def test_same_candidate_in_distinct_aliases_has_isolated_cooldown_identity() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: shared-alias-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
  - name: other
    candidates:
      - provider: openrouter
        model: shared-alias-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot)
    basic_key = _snapshot_candidate_key(
        snapshot,
        "shared-alias-model",
        alias_name="basic",
        lane_key="shared-lane",
    )
    other_key = _snapshot_candidate_key(
        snapshot,
        "shared-alias-model",
        alias_name="other",
        lane_key="shared-lane",
    )

    assert basic_key != other_key
    await cooldown_state._set_codex_auto_agent_cooldown(basic_key, 60.0)
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(basic_key) > 0
    assert alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(other_key) == 0.0


# ---------------------------------------------------------------------------
# (R3-4, scenario d2) no-op refresh retains cooldown
# ---------------------------------------------------------------------------


async def test_scenario_d2_noop_refresh_retains_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin: after a failure under snapshot A, re-POST IDENTICAL
    YAML (changed=False, same snapshot object); assert the cooldown remains
    because the candidate's stable identity and key are unchanged."""
    _bypass_session_owner_for_cooldown_contract(monkeypatch)
    client = _refresh_client()

    yaml_str = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d2-leader
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d2-fallback
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""
    resp1 = await _post_refresh(client, yaml_str)
    assert resp1.status_code == 200
    assert resp1.json()["changed"] is True
    snapshot_before = snapshot_select.get_active_routing_snapshot()
    assert snapshot_before is not None

    # Structured-429 the leader
    async def _performer(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        if adapter_model == "d2-leader":
            raise _StructuredUpstream429(retry_after_seconds=60)
        return _SUCCESS_RESPONSE

    _, restore = _install_openrouter_performer(_performer)
    try:
        result = await _drive_wrapper(session_id="d2-session-1")
        assert isinstance(result, Response)
    finally:
        restore()

    leader_key = _snapshot_candidate_key(snapshot_before, "d2-leader")
    remaining_before = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(leader_key)
    assert remaining_before > 0

    # Re-post IDENTICAL YAML -> no-op
    resp2 = await _post_refresh(client, yaml_str)
    assert resp2.status_code == 200
    assert resp2.json()["changed"] is False

    # Same snapshot object retained
    snapshot_after = snapshot_select.get_active_routing_snapshot()
    assert snapshot_after is snapshot_before, (
        "no-op refresh must retain the exact same snapshot object"
    )
    assert _snapshot_candidate_key(snapshot_after, "d2-leader") == leader_key

    # Cooldown must REMAIN
    remaining_after = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(leader_key)
    assert remaining_after > 0, (
        "no-op refresh must not invalidate the cooldown; "
        f"remaining={remaining_after}"
    )


# ---------------------------------------------------------------------------
# (R3-4, scenario d3) semantically identical refresh retains state
# ---------------------------------------------------------------------------

_D3_YAML_ORIGINAL = """
defaults: {}
aliases:
  - name: basic
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: d3-model-a
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d3-model-b
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""

# Semantically identical: comment added, mapping-key order changed, extra whitespace
_D3_YAML_REFORMATTED = """
# Reformatted: comment added, key order changed, extra whitespace
defaults: {}
aliases:
  - name: basic
    distribution_strategy: round_robin
    candidates:
      - route_family: codex_openrouter_completion_adapter
        provider: openrouter
        priority: 100
        model: d3-model-a
      - route_family: codex_openrouter_completion_adapter
        provider:   openrouter
        priority:   100
        model:   d3-model-b
"""


async def test_scenario_d3_semantically_identical_refresh_retains_state() -> None:
    """R3-4 RED pre-fix: re-post YAML with comment, whitespace, and
    mapping-key-order changes only. Assert changed=False, same active
    snapshot object, same cooldown key, retained cooldown, retained RR
    cursor, and retained continuation affinity.

    Pre-fix failure: config_hash = sha256(raw_yaml), so formatting changes
    produce a different hash -> changed=True."""
    client = _refresh_client()

    # Activate original
    resp1 = await _post_refresh(client, _D3_YAML_ORIGINAL)
    assert resp1.status_code == 200
    assert resp1.json()["changed"] is True
    snapshot_before = snapshot_select.get_active_routing_snapshot()
    assert snapshot_before is not None

    # Drive a request to advance the RR cursor and establish affinity
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
        await _drive_wrapper(session_id="d3-session-1")
    finally:
        restore()
    affinity_before = {
        key: dict(value)
        for key, value in alias_state.alias_routing_state.codex.session_affinity_by_key.items()
    }
    assert len(affinity_before) == 1
    assert next(iter(affinity_before.values()))["config_hash"] == (
        snapshot_before.config_hash
    )

    # Cool one candidate to have a cooldown to retain
    model_a_key = _snapshot_candidate_key(snapshot_before, "d3-model-a")
    await cooldown_state._set_codex_auto_agent_cooldown(model_a_key, 60.0)

    # Capture RR cursor state
    cursor_before = dict(alias_state.alias_routing_state.round_robin_cursor)

    # Re-post reformatted (semantically identical) YAML
    resp2 = await _post_refresh(client, _D3_YAML_REFORMATTED)
    assert resp2.status_code == 200
    assert resp2.json()["changed"] is False, (
        "R3-4: semantically identical YAML (comment/whitespace/key-order "
        "changes only) must report changed=False. Pre-fix: config_hash = "
        "sha256(raw_yaml) so formatting changes produce a different hash."
    )

    # Same active snapshot object
    snapshot_after = snapshot_select.get_active_routing_snapshot()
    assert snapshot_after is snapshot_before, (
        "semantically identical refresh must retain the same snapshot object"
    )
    assert _snapshot_candidate_key(snapshot_after, "d3-model-a") == model_a_key

    # Retained cooldown
    remaining = alias_state.alias_routing_state.codex.get_memory_cooldown_remaining(model_a_key)
    assert remaining > 0, (
        f"cooldown must be retained across a semantic no-op refresh; remaining={remaining}"
    )

    # Retained RR cursor
    cursor_after = dict(alias_state.alias_routing_state.round_robin_cursor)
    assert cursor_after == cursor_before, (
        f"RR cursor must be retained; before={cursor_before!r}, after={cursor_after!r}"
    )
    affinity_after = {
        key: dict(value)
        for key, value in alias_state.alias_routing_state.codex.session_affinity_by_key.items()
    }
    assert affinity_after == affinity_before

    # Retained continuation affinity: drive a continuation and verify same candidate
    async def _performer2(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        leaders.append(adapter_model)
        return _SUCCESS_RESPONSE

    _, restore2 = _install_openrouter_performer(_performer2)
    try:
        # Clear the cooldown so the affinity candidate is available
        alias_state.alias_routing_state.codex.cooldown_until_monotonic_by_key.pop(model_a_key, None)
        first_leader = leaders[0]
        await _drive_wrapper(
            session_id="d3-session-1",
            body_extra={"previous_response_id": "resp_d3_continuation"},
        )
        assert leaders[-1] == first_leader, (
            "continuation affinity must be retained across a semantic no-op "
            f"refresh; expected {first_leader!r}, got {leaders[-1]!r}"
        )
    finally:
        restore2()


# ---------------------------------------------------------------------------
# (R3-4, scenario d4) changed refresh preserves compatible continuation affinity
# ---------------------------------------------------------------------------

_D4_SNAPSHOT_A = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d4-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d4-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""

_D4_SNAPSHOT_B = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d4-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 200
      - provider: openrouter
        model: d4-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


async def test_scenario_d4_changed_refresh_preserves_compatible_continuation_affinity() -> None:
    """Regression pin (passes pre-fix): establish affinity, then submit a
    continuation, refresh with changed priority while retaining the pinned
    provider/model/route_family, and assert the continuation remains on the
    pinned candidate."""
    client = _refresh_client()

    # Activate snapshot A
    resp_a = await _post_refresh(client, _D4_SNAPSHOT_A)
    assert resp_a.status_code == 200

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
        # Cold request: establishes affinity for the leader (d4-pinned)
        await _drive_wrapper(session_id="d4-session")
        assert len(leaders) == 1
        pinned = leaders[0]
        assert pinned == "d4-pinned"
        affinity_before = {
            key: dict(value)
            for key, value in alias_state.alias_routing_state.codex.session_affinity_by_key.items()
        }
        assert len(affinity_before) == 1
        assert next(iter(affinity_before.values()))["config_hash"] == (
            resp_a.json()["active_config_hash"]
        )

        # Refresh to snapshot B (changed priority, same provider/model/route_family)
        resp_b = await _post_refresh(client, _D4_SNAPSHOT_B)
        assert resp_b.status_code == 200
        assert resp_b.json()["changed"] is True

        # Continuation after refresh: compatible candidate remains pinned
        await _drive_wrapper(
            session_id="d4-session",
            body_extra={"previous_response_id": "resp_d4_cont_2"},
        )
        assert leaders[-1] == pinned, (
            "R3-4: a compatible continuation (same provider/model/route_family) "
            "must remain pinned after a changed refresh; expected "
            f"{pinned!r}, got {leaders[-1]!r}"
        )
        assert set(alias_state.alias_routing_state.codex.session_affinity_by_key) == set(
            affinity_before
        )
    finally:
        restore()


_D4_SCHEDULE_SNAPSHOT_A = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d4-scheduled-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d4-scheduled-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""

_D4_SCHEDULE_SNAPSHOT_B = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d4-scheduled-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 100
        schedule:
          start: "2020-01-01T00:00:00Z"
          end: "2020-01-02T00:00:00Z"
      - provider: openrouter
        model: d4-scheduled-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


async def test_scenario_d4_schedule_only_refresh_preserves_continuation_affinity() -> None:
    client = _refresh_client()
    response_a = await _post_refresh(client, _D4_SCHEDULE_SNAPSHOT_A)
    assert response_a.status_code == 200

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
        await _drive_wrapper(session_id="d4-schedule-session")
        assert leaders == ["d4-scheduled-pinned"]

        response_b = await _post_refresh(client, _D4_SCHEDULE_SNAPSHOT_B)
        assert response_b.status_code == 200
        assert response_b.json()["changed"] is True

        await _drive_wrapper(
            session_id="d4-schedule-session",
            body_extra={"previous_response_id": "resp_d4_schedule_continuation"},
        )
        assert leaders[-1] == "d4-scheduled-pinned"

        await _drive_wrapper(session_id="d4-schedule-cold-session")
        assert leaders[-1] == "d4-scheduled-other"
    finally:
        restore()


# ---------------------------------------------------------------------------
# (R3-4, scenario d5) refresh removing affinity candidate requires redispatch
# ---------------------------------------------------------------------------

_D5_SNAPSHOT_A = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d5-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d5-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""

_D5_SNAPSHOT_B = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d5-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


_D5_SNAPSHOT_ROUTE_INCOMPATIBLE = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d5-pinned
        route_family: codex_responses
        priority: 100
      - provider: openrouter
        model: d5-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


@pytest.mark.parametrize(
    "replacement_yaml",
    [_D5_SNAPSHOT_B, _D5_SNAPSHOT_ROUTE_INCOMPATIBLE],
    ids=["removed", "route-incompatible"],
)
async def test_scenario_d5_refresh_removing_affinity_candidate_requires_redispatch(
    replacement_yaml: str,
) -> None:
    """R3-4 RED pre-fix: establish affinity, refresh to remove the pinned
    candidate, then submit a continuation. Assert the explicit
    redispatch-required/fail-closed error and zero upstream calls to another
    candidate.

    Pre-fix failure: the removed affinity candidate falls through to normal
    selection, which selects d5-other instead of failing closed."""
    client = _refresh_client()

    # Activate snapshot A
    resp_a = await _post_refresh(client, _D5_SNAPSHOT_A)
    assert resp_a.status_code == 200

    leaders: list[str] = []

    async def _performer(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Any,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
        target_url: str,
        api_key: Optional[str],
        forward_headers: bool,
    ) -> Response:
        leaders.append(candidate["model"])
        return _SUCCESS_RESPONSE

    restore = _install_candidate_request_performer(_performer)
    try:
        # Cold request: establishes affinity for d5-pinned
        await _drive_wrapper(session_id="d5-session")
        assert leaders[0] == "d5-pinned"

        # Refresh to snapshot B (d5-pinned removed)
        resp_b = await _post_refresh(client, replacement_yaml)
        assert resp_b.status_code == 200
        assert resp_b.json()["changed"] is True

        # Continuation after removal: must fail closed with redispatch-required
        leaders.clear()
        with pytest.raises(HTTPException) as exc_info:
            await _drive_wrapper(
                session_id="d5-session",
                body_extra={"previous_response_id": "resp_d5_cont"},
            )
        exc = exc_info.value
        assert exc.status_code == 429
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
        assert detail.get("redispatch_required") is True or "redispatch" in str(
            detail.get("code", "")
        ), (
            "R3-4: a continuation whose pinned candidate was removed must "
            f"fail closed with redispatch-required; got detail={detail!r}"
        )
        assert len(leaders) == 0, (
            "R3-4: zero upstream calls to another candidate when the pinned "
            f"candidate is removed; got calls={leaders!r}"
        )
    finally:
        restore()


async def test_scenario_d5_none_to_concrete_route_family_requires_redispatch() -> None:
    client = _refresh_client()
    response = await _post_refresh(
        client,
        """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: d5-none-route-pinned
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openrouter
        model: d5-none-route-other
        route_family: codex_openrouter_completion_adapter
        priority: 50
""",
    )
    assert response.status_code == 200
    snapshot = snapshot_select.get_active_routing_snapshot()
    assert snapshot is not None

    request = _minimal_request("d5-none-route-session")
    request_body = {
        "model": "basic",
        "input": [{"role": "user", "content": "hello"}],
        "litellm_metadata": {"session_id": "d5-none-route-session"},
    }
    session_key = lpe._resolve_codex_auto_agent_session_key(
        request,
        request_body,
        alias_model="basic",
    )
    await cooldown_state._set_codex_auto_agent_session_affinity(
        session_key,
        {
            "provider": "openrouter",
            "model": "d5-none-route-pinned",
            "route_family": None,
            "last_resort": False,
            "config_epoch_tag": "prior-semantic-config",
        },
    )

    leaders: list[str] = []

    async def _performer(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Any,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
        target_url: str,
        api_key: Optional[str],
        forward_headers: bool,
    ) -> Response:
        leaders.append(candidate["model"])
        return _SUCCESS_RESPONSE

    restore = _install_candidate_request_performer(_performer)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await _drive_wrapper(
                session_id="d5-none-route-session",
                body_extra={"previous_response_id": "resp_d5_none_route"},
            )
        detail = exc_info.value.detail
        assert exc_info.value.status_code == 429
        assert isinstance(detail, dict)
        assert detail.get("redispatch_required") is True
        assert leaders == []
    finally:
        restore()
