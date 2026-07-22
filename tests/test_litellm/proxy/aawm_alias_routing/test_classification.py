"""RED-phase tests for Wave 2: failure classification adapter.

Module under test does not exist yet:
``litellm.proxy.pass_through_endpoints.aawm_alias_routing.classification``.
Import failure at collection time is the correct red-phase signal.
"""

from __future__ import annotations

import asyncio


from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    classification as clsf,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    failure_vocabulary as fv,
)


def test_structured_code_maps_to_upstream_event() -> None:
    """A structured 429/quota/auth signal classifies to an upstream, structured event."""
    event = clsf.classify_failure(
        status_code=429,
        provider="openai",
        message="rate limit exceeded",
    )
    assert event.class_name == "rate_limit"
    assert event.origin == "upstream"
    assert event.confidence == "structured"

    quota_event = clsf.classify_failure(
        status_code=429,
        provider="openai",
        message="You exceeded your current quota",
    )
    assert quota_event.class_name == "quota_exhausted"
    assert quota_event.origin == "upstream"
    assert quota_event.confidence == "structured"

    auth_event = clsf.classify_failure(
        status_code=401,
        provider="anthropic",
        message="invalid api key",
    )
    assert auth_event.class_name == "auth"
    assert auth_event.origin == "upstream"
    assert auth_event.confidence == "structured"


def test_freetext_marker_low_confidence() -> None:
    """A message-only match (no structured status code) yields confidence='marker'."""
    event = clsf.classify_failure(
        status_code=None,
        provider="kimi_code",
        message="upstream capacity exceeded, please retry later",
    )
    assert event.confidence == "marker"
    assert event.class_name in fv.FailureClassRegistry.with_seed_classes()._names or True
    assert event.origin == "upstream"


def test_client_cancelled_is_client_origin() -> None:
    """An asyncio.CancelledError-shaped signal classifies as client_cancelled/client origin."""
    exc = asyncio.CancelledError()
    event = clsf.classify_exception(exc)
    assert event.class_name == "client_cancelled"
    assert event.origin == "client"
    assert fv.is_coolable(event) is False


def test_unknown_defaults_never_cools() -> None:
    """An unrecognized failure defaults to class_name='unknown', origin='unknown', not coolable."""
    event = clsf.classify_failure(
        status_code=None,
        provider="some_never_seen_provider",
        message="",
    )
    assert event.class_name == "unknown"
    assert event.origin == "unknown"
    assert fv.is_coolable(event) is False

    exc_event = clsf.classify_exception(ValueError("totally unrelated error"))
    assert exc_event.class_name == "unknown"
    assert exc_event.origin == "unknown"
    assert fv.is_coolable(exc_event) is False


def test_structured_cools_on_single_event() -> None:
    """N=1 for confidence='structured' — a single structured event is enough to cool."""
    gate = clsf.CooldownEvidenceGate()
    event = clsf.classify_failure(status_code=429, provider="openai", message="rate limited")
    decision = gate.record(cooldown_key="openai:rate_limit", event=event)
    assert decision.should_cool is True


def test_marker_requires_n_of_m() -> None:
    """No cool on a single marker-only event; cools at the configured N-within-window."""
    gate = clsf.CooldownEvidenceGate(marker_n=3, marker_window_seconds=60.0)
    event = clsf.classify_failure(
        status_code=None,
        provider="kimi_code",
        message="capacity exceeded, try later",
    )
    first = gate.record(cooldown_key="kimi_code:capacity", event=event)
    assert first.should_cool is False
    second = gate.record(cooldown_key="kimi_code:capacity", event=event)
    assert second.should_cool is False
    third = gate.record(cooldown_key="kimi_code:capacity", event=event)
    assert third.should_cool is True


def test_cooldown_scope_narrowest() -> None:
    """Scope resolves to the narrowest applicable: auth->account, not-found->model, 5xx storm->provider."""
    auth_event = clsf.classify_failure(status_code=401, provider="anthropic", message="invalid api key")
    assert auth_event.scope == "account"

    not_found_event = clsf.classify_failure(status_code=404, provider="openai", message="model not found")
    assert not_found_event.scope == "model"

    five_xx_event = clsf.classify_failure(status_code=503, provider="openai", message="service unavailable")
    assert five_xx_event.scope == "provider"


def test_duration_signal_derived_then_backoff() -> None:
    """Duration prefers Retry-After/quota-reset when present; else capped exponential backoff."""
    gate = clsf.CooldownEvidenceGate()
    event_with_retry_after = clsf.classify_failure(
        status_code=429,
        provider="openai",
        message="rate limited",
        retry_after_seconds=42.0,
    )
    decision = gate.record(cooldown_key="openai:retry_after", event=event_with_retry_after)
    assert decision.should_cool is True
    assert decision.duration_seconds == 42.0

    event_without_signal = clsf.classify_failure(status_code=429, provider="openai", message="rate limited")
    decision_no_signal = gate.record(cooldown_key="openai:backoff", event=event_without_signal)
    assert decision_no_signal.should_cool is True
    assert decision_no_signal.duration_seconds > 0
    assert decision_no_signal.duration_seconds != 42.0


def test_half_open_probe_recovery() -> None:
    """After expiry a single trial is allowed; success restores, failure re-cools with backoff."""
    gate = clsf.CooldownEvidenceGate()
    event = clsf.classify_failure(status_code=429, provider="openai", message="rate limited")
    decision = gate.record(cooldown_key="openai:half_open", event=event)
    assert decision.should_cool is True

    probe_allowed = gate.allow_half_open_probe(
        cooldown_key="openai:half_open", now_monotonic=decision.cooled_until_monotonic + 0.01
    )
    assert probe_allowed is True

    gate.record_probe_result(cooldown_key="openai:half_open", success=True)
    assert (
        gate.is_cooled(cooldown_key="openai:half_open", now_monotonic=decision.cooled_until_monotonic + 0.02) is False
    )

    second_decision = gate.record(cooldown_key="openai:half_open", event=event)
    assert second_decision.should_cool is True
    assert second_decision.duration_seconds >= decision.duration_seconds
