"""RED-phase reproduction tests for the D1-583/D1-584 review findings.

See ``.analysis/remediation-d1-583-584-2026-07-22.md`` for the full finding
list. These tests exercise the LIVE request path (module-level helpers as
actually wired into the request/response flow, the real FastAPI route, and
the real compiler), not isolated helper internals -- the original Wave
tests passed while the runtime stayed broken precisely because they only
covered helper internals and compiled-config properties.

Each test below is expected to FAIL against the current (broken) code and
is expected to PASS once the corresponding fix lands. One concern per test
so each finding's fix can be validated independently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    classification as lpe_classification,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_snapshot,
)

REFRESH_PATH = "/aawm/alias-config/refresh"


def _minimal_request(session_id: str = "review-findings-session") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": session_id}
    request.query_params = {}
    request.state = MagicMock()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(lpe.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_alias_routing_ambient_state():
    """Neutralize shared/process-global cooldown, affinity, and snapshot state.

    Mirrors ``clear_codex_auto_agent_alias_state`` /
    ``test_read_pilot_shadow_parity``'s reset fixture so these tests cannot
    flap on state left over from other tests in the same process.
    """
    previous_snapshot = lpe.get_active_routing_snapshot()
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    yield
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe.set_active_routing_snapshot(previous_snapshot)


_SNAPSHOT_YAML = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/snapshot-only-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""


# ---------------------------------------------------------------------------
# Finding #1: ``read`` is unreachable through the live model-normalization
# path -- ``_normalize_codex_auto_agent_alias_model`` only recognizes keys of
# ``_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS``, which has no ``read`` key, so
# the getter's snapshot-driven ``read`` branch is dead code from any live
# request.
# ---------------------------------------------------------------------------


def test_read_is_recognized_as_a_codex_auto_agent_alias_model() -> None:
    """``read`` must be recognized by the live alias-model normalizer."""
    assert lpe._is_codex_auto_agent_alias_model("read") is True


def test_read_resolves_through_the_live_request_body_resolver() -> None:
    """A real inbound request body with ``model: "read"`` must resolve to "read"."""
    request_body = {"model": "read"}
    resolved = lpe._resolve_codex_auto_agent_alias_model(request_body, "/v1/responses")
    assert resolved == "read"


def test_recognized_read_routes_to_snapshot_derived_candidates() -> None:
    """Once reachable, a recognized ``read`` alias must resolve from the active snapshot."""
    snapshot = compiler.compile_yaml(_SNAPSHOT_YAML)
    lpe.set_active_routing_snapshot(snapshot)

    alias_model = lpe._resolve_codex_auto_agent_alias_model({"model": "read"}, "/v1/responses")
    assert alias_model is not None
    candidates = lpe._get_codex_auto_agent_candidates_for_alias(alias_model)
    models = [c["model"] for c in candidates]
    assert "openrouter/snapshot-only-model" in models


# ---------------------------------------------------------------------------
# Finding #2: the N-of-M cooldown-evidence gate decision is discarded -- the
# legacy ``apply_cooldown_fn`` path stays authoritative for what cooldown is
# actually applied, regardless of what the gate decided.
#
# ROUND 2 update: the original reproductions here hand-built ``read_pilot:``
# keys and called ``_record_read_pilot_cooldown_evidence`` directly -- exactly
# the shell the live path never exercised. The authoritative live-path
# reproductions now live in ``test_round2_live_path_remediation.py``
# (``test_live_read_lane_structured_429_cools_with_gate_duration`` and
# ``test_live_read_lane_single_marker_failure_does_not_cool``), which drive the
# real Codex retry handler so the live sequence builds the
# ``provider:model:lane`` key and records evidence on its own. The two
# gate-unit checks below are retained only as direct-gate assertions and no
# longer hand-build a ``read_pilot:`` key or assert applied-cooldown behavior.
# ---------------------------------------------------------------------------


def test_gate_marker_only_single_failure_is_not_cooled() -> None:
    """A single marker-tier event must not advance a key to cooled in the gate."""
    gate = lpe_classification.CooldownEvidenceGate()
    event = lpe_classification.classify_failure(
        status_code=None,
        provider=None,
        message="Selected model is at capacity. Please try a different model.",
    )
    assert event.confidence == "marker"
    gate.record(cooldown_key="unit-marker-key", event=event)
    assert gate.is_cooled(cooldown_key="unit-marker-key") is False


def test_gate_structured_429_cools_with_retry_after_duration() -> None:
    """A single structured 429 event cools immediately, using the retry-after duration."""
    gate = lpe_classification.CooldownEvidenceGate()
    event = lpe_classification.classify_failure(
        status_code=429,
        provider=None,
        message="rate limited",
        retry_after_seconds=12.0,
    )
    assert event.confidence == "structured"
    decision = gate.record(cooldown_key="unit-structured-key", event=event)
    assert decision.should_cool is True
    assert decision.duration_seconds == pytest.approx(12.0, abs=0.001)
    assert gate.is_cooled(cooldown_key="unit-structured-key") is True


# ---------------------------------------------------------------------------
# Finding #3: proportional weighting is defined (``_select_proportional_snapshot_candidate``)
# but never invoked from the live selector -- ``_select_read_pilot_snapshot_candidates``
# always returns priority-ordered candidates regardless of ``distribution_strategy``.
# ---------------------------------------------------------------------------


def test_live_selection_distribution_matches_declared_weights() -> None:
    """Repeated live selection of an equal-priority, weighted pair must realize the
    declared proportional distribution within tolerance -- not always return
    declaration/priority order."""
    raw = """
defaults: {}
aliases:
  - name: read
    distribution_strategy: proportional
    candidates:
      - provider: openrouter
        model: a
        route_family: codex_openrouter_completion_adapter
        priority: 50
        weight: 1
      - provider: openrouter
        model: b
        route_family: codex_openrouter_completion_adapter
        priority: 50
        weight: 3
"""
    snapshot = compiler.compile_yaml(raw)
    lpe.set_active_routing_snapshot(snapshot)

    counts: dict[str, int] = {"a": 0, "b": 0}
    n_trials = 2000
    for _ in range(n_trials):
        selected = lpe._select_read_pilot_snapshot_candidates()
        top_model = selected[0]["model"]
        counts[top_model] = counts.get(top_model, 0) + 1

    ratio_a = counts["a"] / n_trials
    ratio_b = counts["b"] / n_trials
    assert abs(ratio_a - 0.25) < 0.1
    assert abs(ratio_b - 0.75) < 0.1


# ---------------------------------------------------------------------------
# Finding #4: the live getter never threads a request's ``client_product_label``
# into the TUI-eligibility filter, and when all candidates are ineligible the
# selector fails OPEN (returns the unfiltered list) instead of failing closed.
# ---------------------------------------------------------------------------


def test_live_getter_threads_client_product_label_into_tui_filter() -> None:
    """A ``tui_attached: Claude`` candidate must be excluded from the LIVE getter's
    result when the request presents no/other TUI identity."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: claude-only-model
        route_family: codex_responses
        priority: 100
        tui_attached: Claude
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    lpe.set_active_routing_snapshot(snapshot)

    # No client_product_label supplied -- must default to "no known TUI",
    # excluding the tui_attached candidate.
    candidates = lpe._get_codex_auto_agent_candidates_for_alias("read")
    models = [c["model"] for c in candidates]
    assert "claude-only-model" not in models

    # INCLUSION: once the live getter actually threads a request's
    # client_product_label through, a tui_attached: Claude candidate IS
    # eligible when the request presents a matching Claude/x.y label. This
    # proves the threading is real (not just an exclusion default that would
    # pass vacuously even if client_product_label were silently dropped).
    candidates_with_claude_label = lpe._get_codex_auto_agent_candidates_for_alias(
        "read",
        client_product_label="Claude/1.2",
    )
    models_with_claude_label = [c["model"] for c in candidates_with_claude_label]
    assert "claude-only-model" in models_with_claude_label


def test_all_ineligible_candidates_fail_closed_not_unfiltered() -> None:
    """When every candidate in the alias is TUI-gated and none match, the selector
    must fail closed (return an empty/last-resort result, or raise) -- not
    silently fall back to the full unfiltered candidate list."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: claude-only-model
        route_family: codex_responses
        priority: 100
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    lpe.set_active_routing_snapshot(snapshot)

    selected = lpe._select_read_pilot_snapshot_candidates(client_product_label=None)
    models = [c["model"] for c in selected]
    # Fail-closed: the ineligible tui_attached-only candidate must not be
    # returned once all candidates are ineligible.
    assert "claude-only-model" not in models


# ---------------------------------------------------------------------------
# Finding #5: non-string ``yaml`` payloads on the refresh endpoint crash with
# an unhandled AttributeError (HTTP 500) instead of a validated HTTP 400.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_yaml_payload",
    [5, [1, 2], {"k": "v"}],
    ids=["int", "list", "dict"],
)
def test_non_string_yaml_payload_returns_400(bad_yaml_payload: object) -> None:
    client = _client()
    response = client.post(REFRESH_PATH, json={"yaml": bad_yaml_payload})
    assert response.status_code == 400


def test_valid_string_yaml_payload_still_returns_200() -> None:
    client = _client()
    valid_yaml = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/refresh-review-findings-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    response = client.post(REFRESH_PATH, json={"yaml": valid_yaml})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Finding #6: the schema accepts duplicate alias names, duplicate models
# within an alias, negative weights, empty candidate lists, and inverted
# schedule windows; the compiled snapshot's ``aliases`` mapping is mutable.
# ---------------------------------------------------------------------------


def test_schema_rejects_duplicate_alias_names() -> None:
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    with pytest.raises(Exception):
        compiler.compile_yaml(raw)


def test_schema_rejects_duplicate_models_within_an_alias() -> None:
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    with pytest.raises(Exception):
        compiler.compile_yaml(raw)


def test_schema_rejects_negative_weights() -> None:
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
        weight: -1
"""
    with pytest.raises(Exception):
        compiler.compile_yaml(raw)


def test_schema_rejects_empty_candidate_lists() -> None:
    raw = """
defaults: {}
aliases:
  - name: read
    candidates: []
"""
    with pytest.raises(Exception):
        compiler.compile_yaml(raw)


def test_schema_rejects_inverted_schedule_windows() -> None:
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
        schedule:
          start: "2026-07-15T00:00:00Z"
          end: "2026-07-01T00:00:00Z"
"""
    with pytest.raises(Exception):
        compiler.compile_yaml(raw)


def test_compiled_snapshot_aliases_mapping_is_immutable() -> None:
    """The compiled snapshot's ``aliases`` mapping must reject mutation --
    assignment into it must raise, not silently succeed."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    with pytest.raises((TypeError, AttributeError)):
        snapshot.aliases["read"] = None  # type: ignore[index]


def test_compiled_snapshot_aliases_attribute_reassignment_raises() -> None:
    """Reassigning the ``aliases`` attribute itself on a frozen snapshot must raise."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    with pytest.raises((TypeError, AttributeError)):
        snapshot.aliases = {}  # type: ignore[misc]


def test_active_snapshot_reference_unaffected_by_dict_holder_leak() -> None:
    """Sanity: ``config_snapshot.get_active_snapshot()`` reflects the module-level
    holder used by the refresh endpoint, confirming the immutability checks
    above are exercised against the same snapshot type the live path serves."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    config_snapshot.active_routing_snapshot_holder.swap(snapshot)
    active = config_snapshot.get_active_snapshot()
    assert active is not None
    with pytest.raises((TypeError, AttributeError)):
        active.aliases["read"] = None  # type: ignore[index]
