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

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
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


@pytest.fixture(autouse=True)
def _reset_alias_routing_ambient_state() -> Any:
    """Neutralize shared cooldown / affinity / snapshot / gate / round-robin state."""
    previous_snapshot = lpe.get_active_routing_snapshot()
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._read_pilot_cooldown_gate._key_state.clear()
    lpe._read_pilot_cooldown_gate._family_state.evidence_events_by_key.clear()
    # ``getattr`` guard so the fixture also runs against pre-fix develop (where
    # ``_round_robin_cursor_by_alias`` does not exist), letting each test fail
    # on its behavioral assertion rather than a fixture AttributeError.
    getattr(lpe, "_round_robin_cursor_by_alias", {}).clear()
    yield
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._read_pilot_cooldown_gate._key_state.clear()
    lpe._read_pilot_cooldown_gate._family_state.evidence_events_by_key.clear()
    # ``getattr`` guard so the fixture also runs against pre-fix develop (where
    # ``_round_robin_cursor_by_alias`` does not exist), letting each test fail
    # on its behavioral assertion rather than a fixture AttributeError.
    getattr(lpe, "_round_robin_cursor_by_alias", {}).clear()
    lpe.set_active_routing_snapshot(previous_snapshot)


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

    max_attempts = len(lpe._get_codex_auto_agent_candidates_for_alias("read"))
    return await lpe._handle_auto_agent_alias_route(
        alias_family="codex_auto_agent",
        alias_model="read",
        request=request,
        prepared_request_body=body,
        max_candidate_attempts=max_attempts,
        select_candidate_fn=lpe._select_codex_auto_agent_candidate,
        add_alias_metadata_fn=lpe._add_codex_auto_agent_alias_metadata,
        perform_candidate_request_fn=_perform_candidate_request,
        get_active_cooldown_state_fn=lpe._get_codex_auto_agent_active_cooldown_state,
        set_session_affinity_fn=lpe._set_codex_auto_agent_session_affinity,
        apply_cooldown_fn=lpe._set_codex_auto_agent_candidate_cooldowns,
        raise_redispatch_required_fn=lpe._raise_codex_auto_agent_redispatch_required,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="No read-lane candidates were available.",
        log_label="Round2-Live",
    )


def _live_cooldown_key() -> str:
    """The ``provider:model:lane`` key the live selector builds for the snapshot candidate."""
    candidate = {
        "provider": "openrouter",
        "model": "openrouter/round2-live-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    }
    lane_key = lpe._CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    return lpe._codex_auto_agent_candidate_key(candidate, lane_key)


@pytest.mark.asyncio
async def test_live_read_lane_structured_429_cools_with_gate_duration() -> None:
    """A structured 429 on the LIVE read-lane path must cool the live cooldown key
    with the gate's retry-after-derived duration -- proving the gate is
    authoritative and evidence is recorded before the cooldown is applied."""
    snapshot = compiler.compile_yaml(_SINGLE_CANDIDATE_YAML)
    lpe.set_active_routing_snapshot(snapshot)

    with pytest.raises(Exception):
        await _run_read_lane_once(
            session_id="structured-live",
            raise_exc=_StructuredUpstream429(),
        )

    live_key = _live_cooldown_key()
    applied_remaining = lpe._alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
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
    lpe.set_active_routing_snapshot(snapshot)

    with pytest.raises(Exception):
        await _run_read_lane_once(
            session_id="marker-live",
            raise_exc=_marker_only_capacity_error(),
        )

    live_key = _live_cooldown_key()
    applied_remaining = lpe._alias_routing_state.codex.get_memory_cooldown_remaining(live_key)
    assert applied_remaining == 0.0
    # And the gate itself must agree the key is not cooled after one marker event.
    assert lpe._read_pilot_cooldown_gate.is_cooled(cooldown_key=live_key) is False


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
    always returning declaration order (the pre-fix behavior) or a random pick."""
    snapshot = compiler.compile_yaml(_ROUND_ROBIN_YAML)
    lpe.set_active_routing_snapshot(snapshot)

    leaders = [lpe._select_read_pilot_snapshot_candidates()[0]["model"] for _ in range(4)]
    # Deterministic rotation: consecutive leaders must alternate and both
    # candidates must lead within any two consecutive selections.
    assert leaders[0] != leaders[1]
    assert leaders[1] != leaders[2]
    assert leaders[2] != leaders[3]
    assert set(leaders) == {"rr-a", "rr-b"}
    # Exactly two of four selections lead with each model (a-b-a-b or b-a-b-a).
    assert leaders.count("rr-a") == 2
    assert leaders.count("rr-b") == 2
