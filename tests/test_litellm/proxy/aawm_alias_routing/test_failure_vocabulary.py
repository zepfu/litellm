"""RED-phase tests for Wave 1: open error-class vocabulary + FailureEvent.

D1-583 / D1-584 shared seam. Module under test does not exist yet:
``litellm.proxy.pass_through_endpoints.aawm_alias_routing.failure_vocabulary``.
Import failure at collection time is the correct red-phase signal.
"""

from __future__ import annotations

import dataclasses

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    failure_vocabulary as fv,
)

SEED_CLASSES = {
    "rate_limit",
    "capacity",
    "usage_limit",
    "transient",
    "auth",
    "quota_exhausted",
    "model_unavailable",
    "provider_5xx",
    "provider_4xx_other",
    "serialization",
    "client_cancelled",
    "unknown",
}


def test_vocabulary_is_open_registry_not_enum() -> None:
    """An unknown class name registers/looks up without raising; not a frozen Enum."""
    registry = fv.FailureClassRegistry()
    assert not isinstance(registry, type) or not issubclass(registry, object)
    # Registering a brand-new name must not raise.
    registry.register("brand_new_never_seen_class")
    assert registry.contains("brand_new_never_seen_class") is True
    # Looking up an unknown name (without pre-registering) must not raise either.
    assert registry.contains("another_unseen_class") is False
    registry.register("another_unseen_class")
    assert registry.contains("another_unseen_class") is True


def test_seed_classes_present() -> None:
    """The ~12 structured seed classes exist in the default registry."""
    registry = fv.FailureClassRegistry.with_seed_classes()
    for class_name in SEED_CLASSES:
        assert registry.contains(class_name), f"missing seed class: {class_name}"


def test_failure_event_fields() -> None:
    """FailureEvent carries the exact field set with exact types/literals."""
    event = fv.FailureEvent(
        class_name="rate_limit",
        origin="upstream",
        confidence="structured",
        provider="openai",
        scope="provider",
        retryable=True,
        evidence={"status_code": "429"},
    )
    assert event.class_name == "rate_limit"
    assert event.origin == "upstream"
    assert event.confidence == "structured"
    assert event.provider == "openai"
    assert event.scope == "provider"
    assert event.retryable is True
    assert event.evidence == {"status_code": "429"}

    # Frozen dataclass — mutation raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.class_name = "capacity"  # type: ignore[misc]

    # origin must be constrained to the three-valued literal.
    with pytest.raises((ValueError, TypeError)):
        fv.FailureEvent(
            class_name="rate_limit",
            origin="not_a_valid_origin",  # type: ignore[arg-type]
            confidence="structured",
            provider=None,
            scope="provider",
            retryable=None,
            evidence={},
        )

    # confidence must be constrained to the three-valued literal.
    with pytest.raises((ValueError, TypeError)):
        fv.FailureEvent(
            class_name="rate_limit",
            origin="upstream",
            confidence="not_a_valid_confidence",  # type: ignore[arg-type]
            provider=None,
            scope="provider",
            retryable=None,
            evidence={},
        )

    # scope must be constrained to the five-valued literal.
    with pytest.raises((ValueError, TypeError)):
        fv.FailureEvent(
            class_name="rate_limit",
            origin="upstream",
            confidence="structured",
            provider=None,
            scope="not_a_valid_scope",  # type: ignore[arg-type]
            retryable=None,
            evidence={},
        )


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("upstream", True),
        ("client", False),
        ("unknown", False),
    ],
)
def test_only_upstream_is_coolable(origin: str, expected: bool) -> None:
    """is_coolable(event) truth table: only origin == 'upstream' cools."""
    event = fv.FailureEvent(
        class_name="rate_limit",
        origin=origin,  # type: ignore[arg-type]
        confidence="structured",
        provider="openai",
        scope="provider",
        retryable=True,
        evidence={},
    )
    assert fv.is_coolable(event) is expected
