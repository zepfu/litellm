"""D1-586 body 1: failure-action vocabulary + shadow decisions (enforcement off)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    failure_actions as fa,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    failure_vocabulary as fv,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import error_signals
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    build_shadow_failure_action_decision,
    build_shadow_failure_action_decision_from_exc,
)


def test_action_vocabulary_is_typed_and_closed() -> None:
    assert "observe" in fa.FAILURE_ACTIONS
    assert "retry_same" in fa.FAILURE_ACTIONS
    assert "failover" in fa.FAILURE_ACTIONS
    assert "cooldown" in fa.FAILURE_ACTIONS
    assert "terminal" in fa.FAILURE_ACTIONS
    assert "redispatch" in fa.FAILURE_ACTIONS
    assert fa.FAILURE_ACTION_ENFORCEMENT_ENABLED is False
    assert fa.default_policy_covers_seed_classes() is True


def test_default_policy_maps_seed_classes_without_provider_hardcoding() -> None:
    policy = fa.DEFAULT_FAILURE_ACTION_POLICY
    assert policy.action_for("rate_limit") == "cooldown"
    assert policy.action_for("transient") == "retry_same"
    assert policy.action_for("auth") == "terminal"
    assert policy.action_for("model_unavailable") == "failover"
    assert policy.action_for("unknown") == "observe"
    # Open registry growth: unmapped class falls back without raising.
    assert policy.action_for("brand_new_never_seen_class") == "observe"


def test_configurable_policy_override() -> None:
    policy = fa.FailureActionPolicy(
        action_by_class={"rate_limit": "failover"},
        default_action="observe",
        source="test-override",
    )
    event = fv.FailureEvent(
        class_name="rate_limit",
        origin="upstream",
        confidence="structured",
        provider=None,
        scope="model",
        retryable=True,
        evidence={},
    )
    decision = fa.decide_shadow_failure_action(event, policy=policy)
    assert decision.mapped_action == "failover"
    assert decision.effective_action == "failover"
    assert decision.policy_source == "test-override"
    assert decision.enforcement_enabled is False
    assert decision.mode == "shadow"


@pytest.mark.parametrize(
    "origin,mapped_class,expected_effective,expected_retry,expected_coolable",
    [
        ("upstream", "rate_limit", "cooldown", True, True),
        ("upstream", "transient", "retry_same", True, True),
        ("upstream", "auth", "terminal", False, True),
        ("client", "rate_limit", "observe", False, False),
        ("client", "client_cancelled", "terminal", False, False),
        ("unknown", "unknown", "observe", False, False),
        ("unknown", "capacity", "observe", False, False),
    ],
)
def test_origin_gates_preserve_non_cooling_non_retryable(
    origin: str,
    mapped_class: str,
    expected_effective: str,
    expected_retry: bool,
    expected_coolable: bool,
) -> None:
    event = fv.FailureEvent(
        class_name=mapped_class,
        origin=origin,  # type: ignore[arg-type]
        confidence="structured",
        provider=None,
        scope="lane",
        retryable=True if origin == "upstream" else False,
        evidence={},
    )
    decision = fa.decide_shadow_failure_action(event)
    assert decision.coolable_by_origin is expected_coolable
    assert decision.effective_action == expected_effective
    assert decision.retry_eligible is expected_retry
    assert decision.enforcement_enabled is False
    # Mapping alone must not invent coolability for client/unknown.
    if origin != "upstream":
        assert decision.effective_action not in {"cooldown", "retry_same", "failover", "redispatch"}
        assert decision.retry_eligible is False


def test_event_not_retryable_clamps_retry_mapped_action() -> None:
    event = fv.FailureEvent(
        class_name="transient",
        origin="upstream",
        confidence="structured",
        provider=None,
        scope="provider",
        retryable=False,
        evidence={},
    )
    decision = fa.decide_shadow_failure_action(event)
    assert decision.mapped_action == "retry_same"
    assert decision.effective_action == "observe"
    assert decision.retry_eligible is False
    assert decision.gate_reason == "event_not_retryable"


def test_shadow_decision_is_frozen_and_secret_safe() -> None:
    event = fv.FailureEvent(
        class_name="rate_limit",
        origin="upstream",
        confidence="structured",
        provider="openai",
        scope="model",
        retryable=True,
        evidence={"status_code": "429"},
    )
    decision = fa.decide_shadow_failure_action(
        event,
        current_error_class="rate_limited",
        current_cooldown_scope="model",
        current_status="cooldown_set",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.effective_action = "terminal"  # type: ignore[misc]
    payload = decision.to_observability_dict()
    assert payload["mode"] == "shadow"
    assert payload["enforcement_enabled"] is False
    assert payload["current_error_class"] == "rate_limited"
    assert payload["current_cooldown_scope"] == "model"
    assert payload["current_status"] == "cooldown_set"
    # No secret-bearing keys.
    joined = " ".join(f"{k}={v}" for k, v in payload.items()).lower()
    for banned in ("authorization", "api_key", "token", "password", "secret", "bearer"):
        assert banned not in joined


def test_error_signals_build_shadow_decision_separates_classification() -> None:
    decision = build_shadow_failure_action_decision(
        status_code=429,
        message="rate limit exceeded",
        current_error_class="rate_limited",
        current_cooldown_scope="model",
        current_status="cooldown_set",
    )
    assert decision.class_name == "rate_limit"
    assert decision.origin == "upstream"
    assert decision.mapped_action == "cooldown"
    assert decision.effective_action == "cooldown"
    assert decision.current_error_class == "rate_limited"
    assert decision.enforcement_enabled is False

    client_decision = build_shadow_failure_action_decision(
        status_code=None,
        message="request cancelled by client",
    )
    assert client_decision.origin == "client"
    assert client_decision.coolable_by_origin is False
    assert client_decision.retry_eligible is False
    # Even if class maps to terminal, never cooling/retry via mapping alone.
    assert client_decision.effective_action in {"terminal", "observe"}
    assert client_decision.effective_action not in {
        "cooldown",
        "retry_same",
        "failover",
        "redispatch",
    }


def test_error_signals_shadow_from_exc_uses_extractors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        error_signals,
        "_extract_adapter_exception_status_code",
        lambda exc: 503,
    )
    monkeypatch.setattr(
        error_signals,
        "_get_codex_auto_agent_source_error_summary",
        lambda exc, status_code=None: "upstream capacity exhausted",
    )
    monkeypatch.setattr(
        error_signals,
        "_parse_codex_auto_agent_header_wait_seconds",
        lambda exc: 12.0,
    )
    decision = build_shadow_failure_action_decision_from_exc(
        SimpleNamespace(),
        current_error_class="capacity_exhausted",
        current_cooldown_scope="provider",
        current_status="cooldown_set",
    )
    assert decision.class_name in {"capacity", "provider_5xx"}
    assert decision.origin == "upstream"
    assert decision.current_error_class == "capacity_exhausted"
    assert decision.enforcement_enabled is False
    assert fa.FAILURE_ACTION_ENFORCEMENT_ENABLED is False


def test_candidate_loop_stamps_shadow_field_without_enforcement() -> None:
    """Structural pin: loop emits shadow field; never branches on enforcement."""
    from pathlib import Path

    source = Path(
        "litellm/proxy/pass_through_endpoints/aawm_alias_routing/candidate_loop.py"
    ).read_text(encoding="utf-8")
    assert 'attempt_record["shadow_failure_action"]' in source
    assert "build_shadow_failure_action_decision_from_exc" in source
    # Enforcement must remain a non-branching observational stamp.
    assert "FAILURE_ACTION_ENFORCEMENT_ENABLED" not in source
    assert "effective_action ==" not in source
    assert "shadow_failure_action" in source
