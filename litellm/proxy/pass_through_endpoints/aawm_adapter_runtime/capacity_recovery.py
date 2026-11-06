"""Request-scoped capacity recovery decisions shared by pass-through routes.

Provider adapters, stream parsing, and alias routing used to each decide when a
429/overload was terminal.  This small owner keeps the safety decision in one
place while leaving provider-specific selection and cooldown code intact.
"""

from __future__ import annotations

import time
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class CapacityRecoveryAction(str, Enum):
    RETRY_SAME_ROUTE = "retry_same_route"
    ROUTE_TRANSITION = "route_transition"
    REDISPATCH = "redispatch"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPLAY_VETO = "replay_veto"
    TERMINAL = "terminal"


_CAPACITY_CLASSES = frozenset(
    {
        "rate_limited",
        "server_overloaded",
        "upstream_overloaded",
        "capacity_exhausted",
        "upstream_transient_internal",
    }
)
_CAPACITY_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
_HARD_REPLAY_REASONS = frozenset(
    {
        "assistant_output",
        "client_tool_payload",
        "provider_execution_possible",
        "unknown_semantics",
        "successful_terminal",
        "terminal_marker",
        "event_limit",
        "prefix_limit",
    }
)


@dataclass(frozen=True)
class CapacityRecoveryObservation:
    """Facts about one failed provider attempt."""

    provider_returned: Optional[bool]
    status_code: Optional[int]
    error_class: Optional[str]
    retryable: bool
    commitment: Optional[Mapping[str, Any]]
    continuation: bool = False
    cancelled: bool = False
    ledger_valid: bool = True
    alternate_route_available: bool = False
    redispatch_available: bool = False
    error_payload: Optional[Mapping[str, Any]] = None
    event_type: Optional[str] = None
    ledger_snapshot: Optional[Mapping[str, Any]] = None
    # The exception (or other request-local failure token) that represents
    # this provider attempt.  Stream/failover layers can see the same object;
    # observing it twice must not consume another owner ordinal.
    observation_key: Any = None


@dataclass(frozen=True)
class CapacityRecoveryDecision:
    action: CapacityRecoveryAction
    reason: str
    retry_ordinal: int
    provider_returned: Optional[bool]
    capacity: bool
    root_recovery_id: str
    ledger_snapshot: Optional[Mapping[str, Any]] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "retry_ordinal": self.retry_ordinal,
            "provider_returned": self.provider_returned,
            "capacity": self.capacity,
            "root_recovery_id": self.root_recovery_id,
            "ledger_snapshot": dict(self.ledger_snapshot)
            if isinstance(self.ledger_snapshot, Mapping)
            else None,
        }


class CapacityRecoveryOwner:
    """One request-scoped authority for capacity retry/transition decisions."""

    def __init__(
        self,
        *,
        deadline_seconds: float = 7200.0,
        max_same_route_retries: int = 1,
    ) -> None:
        self.started_at = time.monotonic()
        self.deadline_seconds = max(0.0, float(deadline_seconds))
        self.max_same_route_retries = max(0, int(max_same_route_retries))
        self.retry_ordinal = 0
        self.last_decision: Optional[CapacityRecoveryDecision] = None
        self.root_recovery_id = uuid4().hex

    @staticmethod
    def _prior_observation(
        observation: CapacityRecoveryObservation,
    ) -> Optional[CapacityRecoveryDecision]:
        key = observation.observation_key
        if key is None:
            return None
        decision = getattr(key, "_aawm_capacity_recovery_decision", None)
        if isinstance(decision, CapacityRecoveryDecision):
            return decision
        return None

    @property
    def remaining_seconds(self) -> float:
        if self.deadline_seconds <= 0:
            return 0.0
        return max(0.0, self.deadline_seconds - (time.monotonic() - self.started_at))

    @staticmethod
    def _is_capacity(observation: CapacityRecoveryObservation) -> bool:
        normalized = str(observation.error_class or "").strip().lower()
        if normalized in _CAPACITY_CLASSES:
            return True
        if observation.status_code in {429, 529}:
            return True
        return observation.retryable and observation.status_code in {
            500,
            502,
            503,
            504,
        }

    @staticmethod
    def _wire_veto(observation: CapacityRecoveryObservation) -> Optional[str]:
        commitment = observation.commitment
        if not isinstance(commitment, Mapping):
            return None
        for flag in ("response_start_sent", "first_body_sent", "terminal_wire_committed", "done_wire_committed"):
            if commitment.get(flag) is True:
                return flag
        reason = str(commitment.get("precommit_commitment_reason") or "").strip().lower()
        if reason in _HARD_REPLAY_REASONS:
            return reason
        return None

    @staticmethod
    def _typed_no_effect_capacity(observation: CapacityRecoveryObservation) -> bool:
        payload = observation.error_payload
        if not isinstance(payload, Mapping):
            return False
        response = payload.get("response")
        if not isinstance(response, Mapping):
            return False
        if response.get("status") != "failed" or not response.get("id"):
            return False
        output = response.get("output")
        return "output" not in response or (isinstance(output, list) and not output)

    def observe(self, observation: CapacityRecoveryObservation) -> CapacityRecoveryDecision:
        """Return the only allowed next action for the observed attempt."""
        prior_decision = self._prior_observation(observation)
        if prior_decision is not None and prior_decision.root_recovery_id == self.root_recovery_id:
            self.last_decision = prior_decision
            return prior_decision
        capacity = self._is_capacity(observation)
        veto = self._wire_veto(observation)
        if (
            veto == "provider_execution_possible"
            and observation.provider_returned is True
            and capacity
        ):
            # Dispatch capability is provisional until the provider either
            # rejects the request or emits an effect-bearing event.
            veto = None
        if (
            veto == "unknown_semantics"
            and observation.provider_returned is True
            and capacity
            and self._typed_no_effect_capacity(observation)
        ):
            veto = None
        if observation.cancelled:
            action, reason = CapacityRecoveryAction.REPLAY_VETO, "cancelled"
        elif observation.provider_returned is not True:
            action, reason = CapacityRecoveryAction.REPLAY_VETO, "provider_outcome_unknown"
        elif not observation.ledger_valid:
            action, reason = CapacityRecoveryAction.REPLAY_VETO, "invalid_ledger"
        elif observation.continuation:
            action, reason = CapacityRecoveryAction.REPLAY_VETO, "continuation_replay_forbidden"
        elif veto is not None:
            action, reason = CapacityRecoveryAction.REPLAY_VETO, f"wire_{veto}"
        elif not capacity:
            action, reason = (
                (CapacityRecoveryAction.RETRY_SAME_ROUTE, "retryable_precommit")
                if observation.retryable
                else (CapacityRecoveryAction.TERMINAL, "non_capacity_failure")
            )
        elif self.remaining_seconds <= 0.0:
            action, reason = CapacityRecoveryAction.BUDGET_EXHAUSTED, "deadline_exhausted"
        elif self._ledger_has_no_ordinary_call(
            observation,
            capacity=capacity,
        ):
            # The request-wide ledger is authoritative.  A transition that
            # still needs another send cannot be selected after its ordinary
            # budget is gone.  A real caller-owned redispatch is still safe:
            # it hands the child to a fresh request with a new ledger.
            if observation.redispatch_available and not observation.continuation:
                action, reason = CapacityRecoveryAction.REDISPATCH, "ledger_exhausted_redispatch"
            else:
                action, reason = CapacityRecoveryAction.BUDGET_EXHAUSTED, "logical_call_budget_exhausted"
        elif self.retry_ordinal < self.max_same_route_retries:
            self.retry_ordinal += 1
            action, reason = CapacityRecoveryAction.RETRY_SAME_ROUTE, "provider_capacity_precommit"
        elif observation.alternate_route_available:
            self.retry_ordinal += 1
            action, reason = CapacityRecoveryAction.ROUTE_TRANSITION, "same_route_retry_exhausted"
        elif observation.redispatch_available:
            action, reason = CapacityRecoveryAction.REDISPATCH, "no_safe_route"
        else:
            action, reason = CapacityRecoveryAction.BUDGET_EXHAUSTED, "capacity_retry_budget_exhausted"
        decision = CapacityRecoveryDecision(
            action=action,
            reason=reason,
            retry_ordinal=self.retry_ordinal,
            provider_returned=observation.provider_returned,
            capacity=capacity,
            root_recovery_id=self.root_recovery_id,
            ledger_snapshot=observation.ledger_snapshot,
        )
        self.last_decision = decision
        key = observation.observation_key
        if key is not None:
            try:
                setattr(key, "_aawm_capacity_recovery_decision", decision)
            except Exception:
                pass
        return decision

    @staticmethod
    def _ledger_has_no_ordinary_call(
        observation: CapacityRecoveryObservation,
        *,
        capacity: bool,
    ) -> bool:
        if not capacity or not isinstance(observation.ledger_snapshot, Mapping):
            return False
        remaining = observation.ledger_snapshot.get("remaining_logical_provider_calls")
        if not isinstance(remaining, int) or isinstance(remaining, bool):
            return False
        # A capacity permit is a separate, tightly-bound one-use admission and
        # may legally consume a send even when ordinary calls are exhausted.
        if observation.ledger_snapshot.get("capacity_retry_authorized") is True:
            return False
        return remaining <= 0


def get_or_create_capacity_recovery_owner(
    request: Any,
    *,
    deadline_seconds: float = 7200.0,
    max_same_route_retries: int = 1,
) -> CapacityRecoveryOwner:
    state = getattr(request, "state", None)
    if state is None:
        return CapacityRecoveryOwner(
            deadline_seconds=deadline_seconds,
            max_same_route_retries=max_same_route_retries,
        )
    owner = getattr(state, "aawm_capacity_recovery_owner", None)
    if not isinstance(owner, CapacityRecoveryOwner):
        owner = CapacityRecoveryOwner(
            deadline_seconds=deadline_seconds,
            max_same_route_retries=max_same_route_retries,
        )
        setattr(state, "aawm_capacity_recovery_owner", owner)
    elif max_same_route_retries > owner.max_same_route_retries:
        owner.max_same_route_retries = int(max_same_route_retries)
        owner.deadline_seconds = max(
            owner.deadline_seconds,
            max(0.0, float(deadline_seconds)),
        )
    return owner


def annotate_capacity_recovery(request: Any, decision: CapacityRecoveryDecision) -> None:
    state = getattr(request, "state", None)
    if state is not None:
        setattr(state, "aawm_capacity_recovery_decision", decision.to_metadata())


__all__ = [
    "CapacityRecoveryAction",
    "CapacityRecoveryDecision",
    "CapacityRecoveryObservation",
    "CapacityRecoveryOwner",
    "annotate_capacity_recovery",
    "get_or_create_capacity_recovery_owner",
]
