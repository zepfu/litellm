"""Configurable failure-action vocabulary + shadow decisions (D1-586 body 1).

Classification (:mod:`failure_vocabulary` / :mod:`classification`) stays
separate from policy. This module owns only:

- the typed action vocabulary
- a configurable class_name -> action map
- deterministic **shadow** decisions that never enforce retry/failover/cooldown

Enforcement remains disabled. Origin and coolability gates from
:func:`failure_vocabulary.is_coolable` are preserved: ``client`` /
``unknown`` origin events never become cooling or retryable merely because a
class maps to a cooling/retry action.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Optional

from .failure_vocabulary import (
    FailureEvent,
    Origin,
    SEED_FAILURE_CLASSES,
    is_coolable,
)

# Hard gate for this body: shadow decisions are observational only.
FAILURE_ACTION_ENFORCEMENT_ENABLED: bool = False

FailureAction = Literal[
    "observe",
    "retry_same",
    "failover",
    "cooldown",
    "terminal",
    "redispatch",
]

FAILURE_ACTIONS: frozenset[str] = frozenset(
    {
        "observe",
        "retry_same",
        "failover",
        "cooldown",
        "terminal",
        "redispatch",
    }
)

# Seed defaults keyed by the open FailureEvent class_name vocabulary.
# Unknown/unmapped classes fall back to ``observe`` (no implied enforcement).
_DEFAULT_ACTION_BY_CLASS: dict[str, FailureAction] = {
    "rate_limit": "cooldown",
    "capacity": "cooldown",
    "usage_limit": "cooldown",
    "transient": "retry_same",
    "auth": "terminal",
    "quota_exhausted": "cooldown",
    "model_unavailable": "failover",
    "provider_5xx": "failover",
    "provider_4xx_other": "terminal",
    "serialization": "terminal",
    "client_cancelled": "terminal",
    "unknown": "observe",
}

DEFAULT_FAILURE_ACTION_BY_CLASS: Mapping[str, FailureAction] = MappingProxyType(
    dict(_DEFAULT_ACTION_BY_CLASS)
)

_COOLING_ACTIONS: frozenset[str] = frozenset({"cooldown"})
_RETRY_ACTIONS: frozenset[str] = frozenset({"retry_same", "failover", "redispatch"})


@dataclass(frozen=True)
class FailureActionPolicy:
    """Configurable mapping from failure class_name -> FailureAction.

    Unknown class names resolve to ``default_action`` without raising so the
    open FailureClassRegistry can grow without a schema change here.
    """

    action_by_class: Mapping[str, FailureAction] = DEFAULT_FAILURE_ACTION_BY_CLASS
    default_action: FailureAction = "observe"
    source: str = "default"

    def action_for(self, class_name: str) -> FailureAction:
        mapped = self.action_by_class.get(class_name)
        if mapped is None:
            return self.default_action
        if mapped not in FAILURE_ACTIONS:
            return self.default_action
        return mapped


DEFAULT_FAILURE_ACTION_POLICY = FailureActionPolicy()


@dataclass(frozen=True)
class ShadowFailureActionDecision:
    """Deterministic shadow action decision for one FailureEvent.

    Never authorizes runtime enforcement. Carries enough sanitized fields to
    compare legacy loop behavior with the would-be policy action.
    """

    class_name: str
    origin: Origin
    mapped_action: FailureAction
    effective_action: FailureAction
    coolable_by_origin: bool
    retry_eligible: bool
    gate_reason: str
    enforcement_enabled: bool = False
    mode: Literal["shadow"] = "shadow"
    policy_source: str = "default"
    current_error_class: Optional[str] = None
    current_cooldown_scope: Optional[str] = None
    current_status: Optional[str] = None
    event_retryable: Optional[bool] = None
    event_confidence: Optional[str] = None
    event_scope: Optional[str] = None

    def to_observability_dict(self) -> dict[str, object]:
        """Secret-safe, JSON-friendly shadow fields for attempt/audit records."""
        payload: dict[str, object] = {
            "mode": self.mode,
            "enforcement_enabled": bool(self.enforcement_enabled)
            and bool(FAILURE_ACTION_ENFORCEMENT_ENABLED),
            "class_name": self.class_name,
            "origin": self.origin,
            "mapped_action": self.mapped_action,
            "effective_action": self.effective_action,
            "coolable_by_origin": self.coolable_by_origin,
            "retry_eligible": self.retry_eligible,
            "gate_reason": self.gate_reason,
            "policy_source": self.policy_source,
        }
        if self.current_error_class is not None:
            payload["current_error_class"] = self.current_error_class
        if self.current_cooldown_scope is not None:
            payload["current_cooldown_scope"] = self.current_cooldown_scope
        if self.current_status is not None:
            payload["current_status"] = self.current_status
        if self.event_retryable is not None:
            payload["event_retryable"] = self.event_retryable
        if self.event_confidence is not None:
            payload["event_confidence"] = self.event_confidence
        if self.event_scope is not None:
            payload["event_scope"] = self.event_scope
        return payload


def _clamp_action_for_origin(
    *,
    mapped_action: FailureAction,
    coolable: bool,
    event_retryable: Optional[bool],
) -> tuple[FailureAction, bool, str]:
    """Apply origin/retryable gates without mutating the configured mapping.

    Returns ``(effective_action, retry_eligible, gate_reason)``.
    """
    if not coolable:
        # client/unknown must never become cooling or retryable via mapping alone
        if mapped_action in _COOLING_ACTIONS or mapped_action in _RETRY_ACTIONS:
            return "observe", False, "origin_not_upstream"
        return mapped_action, False, "origin_not_upstream"

    # Upstream: cooling actions stay cooling; retry eligibility follows event.
    if mapped_action in _COOLING_ACTIONS:
        return mapped_action, True, "policy_mapped"
    if mapped_action in _RETRY_ACTIONS:
        if event_retryable is False:
            # Mapped retry/failover but classifier marked non-retryable.
            return "observe", False, "event_not_retryable"
        return mapped_action, True, "policy_mapped"
    if mapped_action == "terminal":
        return mapped_action, False, "policy_mapped"
    return mapped_action, bool(event_retryable) if event_retryable is not None else False, "policy_mapped"


def decide_shadow_failure_action(
    event: FailureEvent,
    *,
    policy: Optional[FailureActionPolicy] = None,
    current_error_class: Optional[str] = None,
    current_cooldown_scope: Optional[str] = None,
    current_status: Optional[str] = None,
) -> ShadowFailureActionDecision:
    """Return a deterministic shadow decision for ``event``.

    Does not consult or mutate cooldown evidence gates. Coolability is purely
    the three-valued origin rule from :func:`is_coolable`.
    """
    active_policy = policy if policy is not None else DEFAULT_FAILURE_ACTION_POLICY
    mapped = active_policy.action_for(event.class_name)
    coolable = is_coolable(event)
    effective, retry_eligible, gate_reason = _clamp_action_for_origin(
        mapped_action=mapped,
        coolable=coolable,
        event_retryable=event.retryable,
    )
    return ShadowFailureActionDecision(
        class_name=event.class_name,
        origin=event.origin,
        mapped_action=mapped,
        effective_action=effective,
        coolable_by_origin=coolable,
        retry_eligible=retry_eligible,
        gate_reason=gate_reason,
        enforcement_enabled=False,
        mode="shadow",
        policy_source=active_policy.source,
        current_error_class=current_error_class,
        current_cooldown_scope=current_cooldown_scope,
        current_status=current_status,
        event_retryable=event.retryable,
        event_confidence=event.confidence,
        event_scope=event.scope,
    )


def default_policy_covers_seed_classes() -> bool:
    """Return whether every seed failure class has an explicit default action."""
    return all(name in DEFAULT_FAILURE_ACTION_BY_CLASS for name in SEED_FAILURE_CLASSES)


__all__ = [
    "DEFAULT_FAILURE_ACTION_BY_CLASS",
    "DEFAULT_FAILURE_ACTION_POLICY",
    "FAILURE_ACTIONS",
    "FAILURE_ACTION_ENFORCEMENT_ENABLED",
    "FailureAction",
    "FailureActionPolicy",
    "ShadowFailureActionDecision",
    "decide_shadow_failure_action",
    "default_policy_covers_seed_classes",
]
