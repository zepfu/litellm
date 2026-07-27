"""Live-path reproduction tests for the D1-583/D1-584 ROUND 2 findings.

See ``.analysis/remediation-d1-583-584-2026-07-22.md`` (section "ROUND 2").
The prior ``#2`` fix passed its tests only because they hand-built
``read_pilot:`` cooldown keys and recorded evidence into the gate before
applying -- neither of which the LIVE request path did. These tests instead
drive the real Codex auto-agent retry handler (``_handle_auto_agent_alias_route``
with the real selector / metadata / cooldown applicators), stubbing ONLY the
upstream request, so the live sequence generates the ``provider:model:lane``
cooldown key and records evidence on its own. The applied cooldown is then
looked up by that live key.

Findings covered:
- R2-1 / R2-2: the evidence gate must be authoritative on the live path
  (structured 429 cools with the gate's retry-after duration; a single
  marker-only failure does not cool), and evidence must be recorded before the
  cooldown is applied for the same attempt.
- R2-3: ``distribution_strategy: round_robin`` must rotate across the
  equal-top-priority candidates through the live selector.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request, Response

# Retained only for _handle_auto_agent_alias_route (route orchestrator).
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    attempt_records,
    cooldown_apply,
    cooldown_state,
    lane_keys,
    policy,
    selection,
    snapshot_select,
    state,
)


def _minimal_request(session_id: str = "round2-live-session") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": session_id}
    request.query_params = {}
    request.state = MagicMock()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


def _reset_alias_routing_ambient_state_now() -> None:
    """Reset all process-local alias-routing state via the extracted state manager."""
    state.alias_routing_state.reset_for_tests()
    snapshot_select.set_active_routing_snapshot(None)


@pytest.fixture(autouse=True)
def _reset_alias_routing_ambient_state() -> Any:
    """Neutralize shared cooldown / affinity / snapshot / gate / round-robin state."""
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    _reset_alias_routing_ambient_state_now()
    yield
    _reset_alias_routing_ambient_state_now()
    snapshot_select.set_active_routing_snapshot(previous_snapshot)


_SINGLE_CANDIDATE_YAML = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/round2-live-model
        route_family: codex_openrouter_completion_adapter
        priority: 500
"""


class _StructuredUpstream429(Exception):
    """Upstream failure that surfaces a structured HTTP 429 + Retry-After."""

    def __init__(self) -> None:
        super().__init__("rate limited by upstream")
        self.status_code = 429
        self.upstream_headers = {"Retry-After": "12"}


def _marker_only_capacity_error() -> RuntimeError:
    """A retryable capacity failure with NO structured status code (marker tier)."""
    return RuntimeError("Selected model is at capacity. Please try a different model.")


async def _run_read_lane_once(
    *,
    session_id: str,
    raise_exc: Exception,
) -> Any:
    """Drive the REAL codex retry handler for ``model="read"`` with a stubbed upstream.

    Only ``perform_candidate_request_fn`` is stubbed; selection, metadata,
    cooldown-state reads, and cooldown application all use the production
    functions, so the live path builds the cooldown key and records evidence.
    """
    request = _minimal_request(session_id)
    body = {
        "model": "read",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "litellm_metadata": {"session_id": session_id},
    }

    async def _perform_candidate_request(**_kwargs: Any) -> Response:
        raise raise_exc

    max_attempts = len(snapshot_select._get_codex_auto_agent_candidates_for_alias("read"))
    return await lpe._handle_auto_agent_alias_route(
        alias_family="codex_auto_agent",
        alias_model="read",
        request=request,
        prepared_request_body=body,
        max_candidate_attempts=max_attempts,
        select_candidate_fn=selection._select_codex_auto_agent_candidate,
        add_alias_metadata_fn=attempt_records._add_codex_auto_agent_alias_metadata,
        perform_candidate_request_fn=_perform_candidate_request,
        get_active_cooldown_state_fn=cooldown_state._get_codex_auto_agent_active_cooldown_state,
        set_session_affinity_fn=cooldown_state._set_codex_auto_agent_session_affinity,
        apply_cooldown_fn=cooldown_apply._set_codex_auto_agent_candidate_cooldowns,
        raise_redispatch_required_fn=selection._raise_codex_auto_agent_redispatch_required,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="No read-lane candidates were available.",
        log_label="Round2-Live",
    )


def _live_cooldown_key() -> str:
    """Return the exact state key for the resolved ``read`` snapshot candidate."""
    candidate: dict[str, Any] = {
        "provider": "openrouter",
        "model": "openrouter/round2-live-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    }
    lane_key = policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    snapshot = snapshot_select.get_active_routing_snapshot()
    epoch_tag: str | None = None
    if snapshot is not None:
        alias = snapshot.aliases.get("read")
        identity = (
            candidate["provider"],
            candidate["model"],
            candidate["route_family"],
        )
        if alias is not None and any(
            (compiled.provider, compiled.model, compiled.route_family) == identity
            for compiled in alias.candidates
        ):
            epoch_tag = snapshot.config_hash
    return lane_keys._codex_auto_agent_candidate_key(candidate, lane_key, epoch_tag=epoch_tag)


def test_live_cooldown_key_requires_exact_read_alias_ownership() -> None:
    assert not _live_cooldown_key().startswith("h")

    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: other
    candidates:
      - provider: openrouter
        model: openrouter/round2-live-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
  - name: read
    candidates:
      - provider: opencode_zen
        model: openrouter/round2-live-model
        route_family: codex_opencode_zen_adapter
        priority: 100
"""
    )
    snapshot_select.set_active_routing_snapshot(snapshot)

    assert not _live_cooldown_key().startswith("h")


@pytest.mark.asyncio
async def test_live_read_lane_structured_429_cools_with_gate_duration() -> None:
    """A structured 429 on the LIVE read-lane path must cool the live cooldown key
    with the gate's retry-after-derived duration -- proving the gate is
    authoritative and evidence is recorded before the cooldown is applied."""
    snapshot = compiler.compile_yaml(_SINGLE_CANDIDATE_YAML)
    snapshot_select.set_active_routing_snapshot(snapshot)

    with pytest.raises(Exception):
        await _run_read_lane_once(
            session_id="structured-live",
            raise_exc=_StructuredUpstream429(),
        )

    live_key = _live_cooldown_key()
    applied_remaining = state.alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
    # The gate resolved a 12s retry-after-derived duration; the APPLIED cooldown
    # -- looked up by the live provider:model:lane key the selector produced --
    # must reflect that gate duration.
    assert applied_remaining == pytest.approx(12.0, abs=1.5)


@pytest.mark.asyncio
async def test_live_read_lane_single_marker_failure_does_not_cool() -> None:
    """A single marker-only (non-structured) failure on the LIVE read-lane path
    must NOT cool the candidate -- the N-of-M gate needs multiple marker events
    within its window before a key advances toward cooling."""
    snapshot = compiler.compile_yaml(_SINGLE_CANDIDATE_YAML)
    snapshot_select.set_active_routing_snapshot(snapshot)

    with pytest.raises(Exception):
        await _run_read_lane_once(
            session_id="marker-live",
            raise_exc=_marker_only_capacity_error(),
        )

    live_key = _live_cooldown_key()
    applied_remaining = state.alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
    assert applied_remaining == 0.0
    # And the gate itself must agree the key is not cooled after one marker event.
    assert state.alias_routing_state.read_pilot_gate.is_cooled(cooldown_key=live_key) is False


_ROUND_ROBIN_YAML = """
defaults: {}
aliases:
  - name: read
    distribution_strategy: round_robin
    candidates:
      - provider: openrouter
        model: rr-a
        route_family: codex_openrouter_completion_adapter
        priority: 50
      - provider: openrouter
        model: rr-b
        route_family: codex_openrouter_completion_adapter
        priority: 50
"""


def test_live_round_robin_rotates_across_equal_priority_candidates() -> None:
    """``distribution_strategy: round_robin`` must rotate the leading candidate
    across the equal-top-priority pair on successive LIVE selections, rather than
    always returning declaration order (the pre-fix behavior) or a random pick.

    Wave-1 R3-2 purified the enumeration: ``_select_read_pilot_snapshot_candidates``
    now READS the rotation cursor (it no longer self-advances), so the getter
    cannot double-count within a single request. The cursor advances exactly once
    per ACTUAL selection via ``_commit_round_robin_selection`` -- the same commit
    the live selector performs on its ``first_available`` return. This test drives
    that read+commit pair per selection to prove deterministic rotation."""
    snapshot = compiler.compile_yaml(_ROUND_ROBIN_YAML)
    snapshot_select.set_active_routing_snapshot(snapshot)

    leaders: list[str] = []
    for _ in range(4):
        token = snapshot_select._derive_round_robin_commit_token("read", client_product_label=None)
        leader = snapshot_select._select_read_pilot_snapshot_candidates()[0]
        leaders.append(leader["model"])
        snapshot_select._commit_round_robin_selection(token, selected_candidate=leader)
    # Deterministic rotation: consecutive leaders must alternate and both
    # candidates must lead within any two consecutive selections.
    assert leaders[0] != leaders[1]
    assert leaders[1] != leaders[2]
    assert leaders[2] != leaders[3]
    assert set(leaders) == {"rr-a", "rr-b"}
    # Exactly two of four selections lead with each model (a-b-a-b or b-a-b-a).
    assert leaders.count("rr-a") == 2
    assert leaders.count("rr-b") == 2
