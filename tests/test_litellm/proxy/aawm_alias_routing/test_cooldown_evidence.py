"""RED-phase tests for Wave 2: N-of-M cooldown evidence + bounded per-key counters.

Exercises the evidence-counter side of the seam (``state.py`` additions) plus
integration with the ``classification.CooldownEvidenceGate`` from a state-map
perspective. Modules under test do not exist / lack these members yet:
``classification.py`` (new) and ``state.py`` additions (evidence counters).
"""

from __future__ import annotations


from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    classification as clsf,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    failure_vocabulary as fv,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import state as state_mod


def test_alias_family_state_has_bounded_evidence_counters() -> None:
    """AliasFamilyState exposes a bounded sliding-window evidence-counter map."""
    family_state = state_mod.AliasFamilyState()
    assert hasattr(family_state, "record_failure_evidence")
    assert hasattr(family_state, "evidence_count_within_window")


def test_evidence_counters_are_bounded() -> None:
    """The evidence-counter structure enforces a max_size bound like other memory maps."""
    family_state = state_mod.AliasFamilyState()
    for i in range(5000):
        family_state.record_failure_evidence(
            cooldown_key=f"key-{i}",
            confidence="marker",
            window_seconds=60.0,
            max_size=100,
        )
    # Internal evidence structure must be bounded, mirroring bound_memory_map elsewhere.
    size = family_state.evidence_map_size()
    assert size <= 100


def test_only_upstream_origin_events_advance_evidence_toward_cooling() -> None:
    """Client/unknown-origin events must never advance a key toward cooling."""
    gate = clsf.CooldownEvidenceGate(marker_n=2, marker_window_seconds=60.0)

    client_event = clsf.classify_exception(__import__("asyncio").CancelledError())
    assert fv.is_coolable(client_event) is False
    decision = gate.record(cooldown_key="never-cools", event=client_event)
    assert decision.should_cool is False

    unknown_event = clsf.classify_failure(status_code=None, provider="x", message="")
    assert fv.is_coolable(unknown_event) is False
    decision2 = gate.record(cooldown_key="never-cools", event=unknown_event)
    assert decision2.should_cool is False


def test_structured_event_bypasses_marker_n_of_m_threshold() -> None:
    """A single structured event cools immediately, unaffected by the marker N-of-M gate."""
    gate = clsf.CooldownEvidenceGate(marker_n=5, marker_window_seconds=60.0)
    structured_event = clsf.classify_failure(status_code=429, provider="openai", message="rate limited")
    decision = gate.record(cooldown_key="structured-key", event=structured_event)
    assert decision.should_cool is True


def test_marker_window_expiry_resets_count() -> None:
    """Marker events outside the sliding window do not accumulate toward N."""
    gate = clsf.CooldownEvidenceGate(marker_n=2, marker_window_seconds=0.01)
    event = clsf.classify_failure(status_code=None, provider="kimi_code", message="capacity exceeded, try later")
    first = gate.record(cooldown_key="expiring-key", event=event, now_monotonic=0.0)
    assert first.should_cool is False
    # Second marker event arrives long after the window has elapsed — count resets.
    second = gate.record(cooldown_key="expiring-key", event=event, now_monotonic=100.0)
    assert second.should_cool is False
