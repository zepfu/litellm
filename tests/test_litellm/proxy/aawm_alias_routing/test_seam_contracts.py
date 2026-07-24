"""Wave 0 guardrail: seam-contract tests for the alias-routing retry loop.

These tests turn the ``b9c97f9540``-class silent stub-rot failure mode (a
production applicator gaining a new required/optional kwarg that a stub
mirror silently falls behind on) into named, collection-time-stable
assertions, per
``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``
Wave 0.

``test_reset_alias_routing_state_for_tests_clears_everything`` is RED until
the Wave-0 engineer adds ``reset_alias_routing_state_for_tests()`` -- that is
expected and intentional; do not add the helper here.
"""

from __future__ import annotations

import inspect

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)

# The exact kwarg set passed at the loop's ``apply_cooldown_fn`` call site
# (``llm_passthrough_endpoints.py:22223-22233``). Encoded explicitly here so
# a future signature drift on either applicator is a named, readable failure
# rather than a silent ``**_kwargs`` swallow.
_APPLY_COOLDOWN_CALL_SITE_KWARGS = [
    "request",
    "candidate",
    "lane_key",
    "selected_cooldown_key",
    "cooldown_seconds",
    "error_class",
    "grok_account_quota_exhausted",
    "kimi_failure_metadata",
    "is_read_pilot_lane",
]

# The exact key set the loop consumes off the ``select_candidate_fn`` return
# value (``llm_passthrough_endpoints.py:22143-22187``).
_SELECT_CANDIDATE_REQUIRED_KEYS = {
    "candidate",
    "lane_key",
    "cooldown_key",
    "session_key",
    "selection_reason",
    "skipped",
    "in_flight_session",
}


def _accepts_all_kwargs(fn: object, kwarg_names: list[str]) -> bool:
    signature = inspect.signature(fn)  # type: ignore[arg-type]
    parameters = signature.parameters
    has_var_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if has_var_keyword:
        return True
    return all(name in parameters for name in kwarg_names)


def test_apply_cooldown_fn_call_site_kwargs_match_applicators() -> None:
    """Both production ``apply_cooldown_fn`` applicators accept the full call-site kwarg set."""
    assert _accepts_all_kwargs(
        lpe._set_codex_auto_agent_candidate_cooldowns,
        _APPLY_COOLDOWN_CALL_SITE_KWARGS,
    ), (
        "_set_codex_auto_agent_candidate_cooldowns no longer accepts every "
        f"kwarg the loop passes at its call site: {_APPLY_COOLDOWN_CALL_SITE_KWARGS}"
    )
    assert _accepts_all_kwargs(
        lpe._apply_anthropic_auto_agent_alias_cooldown,
        _APPLY_COOLDOWN_CALL_SITE_KWARGS,
    ), (
        "_apply_anthropic_auto_agent_alias_cooldown no longer accepts every "
        f"kwarg the loop passes at its call site: {_APPLY_COOLDOWN_CALL_SITE_KWARGS}"
    )


@pytest.mark.asyncio
async def test_select_candidate_fn_returns_required_selection_keys() -> None:
    """``_select_codex_auto_agent_candidate`` returns exactly the keys the loop consumes.

    Drives the session-affinity branch (continuation state + a matching
    session-affinity record) because that is the branch that returns
    ``in_flight_session`` -- the plain first-available branch omits it.
    """
    raw_yaml = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/seam-contract-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    previous_snapshot = lpe.get_active_routing_snapshot()
    lpe.set_active_routing_snapshot(snapshot)
    session_key = "read:seam-contract-session:session:seam-contract-session"
    previous_affinity = lpe._codex_auto_agent_session_affinity_by_key.get(session_key)
    lpe._codex_auto_agent_session_affinity_by_key[session_key] = {
        "provider": "openrouter",
        "model": "openrouter/seam-contract-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
        "expires_at_monotonic": __import__("time").monotonic() + 3600.0,
    }
    try:
        from unittest.mock import MagicMock

        from fastapi import Request

        request = MagicMock(spec=Request)
        request.method = "POST"
        request.headers = {"session_id": "seam-contract-session"}
        request.query_params = {}
        request.state = MagicMock()
        request.state.aawm_alias_request_local_cooldown_until = {}
        request.state.aawm_alias_request_local_excluded_keys = set()

        selection = await lpe._select_codex_auto_agent_candidate(
            request=request,
            request_body={"model": "read", "previous_response_id": "resp_seam_contract"},
        )
    finally:
        lpe.set_active_routing_snapshot(previous_snapshot)
        if previous_affinity is None:
            lpe._codex_auto_agent_session_affinity_by_key.pop(session_key, None)
        else:
            lpe._codex_auto_agent_session_affinity_by_key[session_key] = previous_affinity

    assert selection.get("selection_reason") == "session_affinity"
    assert _SELECT_CANDIDATE_REQUIRED_KEYS <= set(selection.keys()), (
        "_select_codex_auto_agent_candidate no longer returns every key the "
        f"retry loop consumes: missing {_SELECT_CANDIDATE_REQUIRED_KEYS - set(selection.keys())}"
    )


def test_reset_alias_routing_state_for_tests_clears_everything() -> None:
    """RED until Wave-0 engineer adds ``reset_alias_routing_state_for_tests()``.

    Once added, the helper must clear: both family (codex/anthropic)
    cooldown/negative/affinity/evidence maps, ``candidate_probe_locks``, the
    read-pilot gate's ``_key_state`` + ``_family_state.evidence_events_by_key``,
    ``_round_robin_cursor_by_alias``, and the active routing snapshot (set to
    ``None``).
    """
    reset_fn = getattr(lpe, "reset_alias_routing_state_for_tests", None)
    assert reset_fn is not None, (
        "reset_alias_routing_state_for_tests() does not exist yet -- expected "
        "RED until the Wave-0 engineer lands it."
    )

    # Seed every piece of state the helper is required to clear.
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key["seed"] = 1.0
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    lpe._codex_auto_agent_session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.codex.evidence_events_by_key["seed"] = [1.0]
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key["seed"] = 1.0
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    lpe._anthropic_auto_agent_session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.anthropic.evidence_events_by_key["seed"] = [1.0]
    alias_routing_state.candidate_probe_locks["seed"] = object()
    lpe._read_pilot_cooldown_gate._key_state["seed"] = object()
    lpe._read_pilot_cooldown_gate._family_state.evidence_events_by_key["seed"] = [1.0]
    lpe._round_robin_cursor_by_alias["seed"] = 1

    raw_yaml = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/reset-helper-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    lpe.set_active_routing_snapshot(compiler.compile_yaml(raw_yaml))

    reset_fn()

    assert lpe._codex_auto_agent_cooldown_until_monotonic_by_key == {}
    assert lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key == {}
    assert lpe._codex_auto_agent_session_affinity_by_key == {}
    assert alias_routing_state.codex.evidence_events_by_key == {}
    assert lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key == {}
    assert lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key == {}
    assert lpe._anthropic_auto_agent_session_affinity_by_key == {}
    assert alias_routing_state.anthropic.evidence_events_by_key == {}
    assert alias_routing_state.candidate_probe_locks == {}
    assert lpe._read_pilot_cooldown_gate._key_state == {}
    assert lpe._read_pilot_cooldown_gate._family_state.evidence_events_by_key == {}
    assert lpe._round_robin_cursor_by_alias == {}
    assert lpe.get_active_routing_snapshot() is None
