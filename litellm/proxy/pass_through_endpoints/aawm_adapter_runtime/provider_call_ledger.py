"""Provider-neutral request-wide logical-call accounting.

The ledger is intentionally small and transport-agnostic.  Adapters decide
which requests belong to their provider lane; the final HTTP request builder
reserves the next logical-call ordinal immediately before sending it.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.requests import Request


DEFAULT_OPENAI_MAX_LOGICAL_PROVIDER_CALLS = 3
OPENAI_MAX_LOGICAL_PROVIDER_CALLS_ENV = (
    "AAWM_OPENAI_MAX_LOGICAL_PROVIDER_CALLS"
)
_MAX_LOGICAL_PROVIDER_CALLS_LIMIT = 32
_ACTIVE_RESPONSE_STATE_KEY = "aawm_openai_active_upstream_response"
_CANDIDATE_CONTEXT_STATE_KEY = "aawm_openai_candidate_context"
_LEDGER_STATE_KEY = "aawm_openai_provider_call_ledger"
_WIRE_COMMITMENT_STATE_KEY = "_aawm_openai_responses_wire_commitment"
_WIRE_REPLAY_BLOCKING_FLAGS = (
    "response_start_sent",
    "first_body_sent",
    "terminal_wire_committed",
    "done_wire_committed",
    "precommit_replay_committed",
)
_WIRE_REPLAY_BLOCKING_DISPOSITIONS = frozenset({"cancelled", "disconnected"})

# Ordered from lowest to highest commitment priority.  When the same read
# produces multiple commitment reasons the highest-priority reason wins.
_RESPONSES_REPLAY_COMMITMENT_REASONS = frozenset({
    "assistant_output",
    "client_tool_payload",
    "provider_execution_possible",
    "unknown_semantics",
    "successful_terminal",
    "terminal_marker",
    "prefix_limit",
    "event_limit",
    "delivery_started",
})

_RESPONSES_REPLAY_COMMITMENT_PRIORITY = {
    "delivery_started": 1,
    "unknown_semantics": 2,
    "prefix_limit": 3,
    "event_limit": 3,
    "terminal_marker": 4,
    "client_tool_payload": 5,
    "provider_execution_possible": 6,
    "assistant_output": 7,
    "successful_terminal": 8,
}

_RESPONSES_REPLAY_LATCH_REASON_KEY = "precommit_commitment_reason"
_RESPONSES_REPLAY_LATCH_EVENT_KEY = "precommit_commitment_event_type"
_RESPONSES_REPLAY_PUBLISHED_LATCH_REASON_KEY = (
    "precommit_published_replay_reason"
)
_RESPONSES_REPLAY_EVENT_LABELS = frozenset(
    {
        "[DONE]",
        "error",
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_text.annotation.added",
        "response.refusal.delta",
        "response.refusal.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call.completed",
        "response.computer_call.in_progress",
        "response.computer_call.completed",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.completed",
        "response.shell_call.in_progress",
        "response.shell_call.completed",
        "response.shell_call_output.done",
        "response.apply_patch_call.in_progress",
        "response.apply_patch_call.completed",
    }
)


def _responses_replay_event_label(
    event_type: Optional[str],
    *,
    reason: Optional[str] = None,
) -> str:
    if event_type is None:
        return (
            "provider_dispatch"
            if reason == "provider_execution_possible"
            else "unknown_event"
        )
    if event_type == "[DONE]":
        return event_type
    if event_type in _RESPONSES_REPLAY_EVENT_LABELS:
        return event_type
    return "unknown_event"


def _safe_context_value(value: Any, *, maximum: int = 128) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    if not normalized:
        return None
    return normalized[:maximum]


def _fingerprint(value: Any) -> Optional[str]:
    normalized = _safe_context_value(value)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _resolve_max_logical_provider_calls() -> int:
    raw_value = os.getenv(OPENAI_MAX_LOGICAL_PROVIDER_CALLS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_OPENAI_MAX_LOGICAL_PROVIDER_CALLS
    try:
        parsed = int(raw_value.strip())
    except ValueError:
        return DEFAULT_OPENAI_MAX_LOGICAL_PROVIDER_CALLS
    return max(1, min(_MAX_LOGICAL_PROVIDER_CALLS_LIMIT, parsed))


def is_openai_logical_send_target(
    *,
    custom_llm_provider: Optional[str],
    target: Any = None,
) -> bool:
    """Return whether a pass-through request belongs to the OpenAI ledger."""

    provider = _safe_context_value(custom_llm_provider)
    if provider is None or provider.lower() != "openai":
        return False
    if target is None:
        return True
    parsed = urlsplit(str(target))
    return parsed.scheme.lower() in {"http", "https"}


def _request_state(request: Request) -> Any:
    return getattr(request, "state", None)


def _selected_account_context(request: Request) -> dict[str, Any]:
    state = _request_state(request)
    selected = (
        getattr(state, "aawm_codex_oauth_selected_account", None)
        if state is not None
        else None
    )
    if not isinstance(selected, dict):
        return {}
    return {
        "account_hash": selected.get("account_hash"),
        "lane_key": selected.get("lane_key"),
        "model": selected.get("model"),
    }


def current_candidate_context(request: Request) -> dict[str, Any]:
    """Return the current server-selected, credential-free candidate context."""

    state = _request_state(request)
    context = (
        getattr(state, _CANDIDATE_CONTEXT_STATE_KEY, None)
        if state is not None
        else None
    )
    result = dict(context) if isinstance(context, dict) else {}
    selected_account = _selected_account_context(request)
    for key, value in selected_account.items():
        if result.get(key) in (None, ""):
            result[key] = value
    return result


def bind_openai_candidate_context(
    request: Request,
    candidate: Optional[dict[str, Any]],
) -> None:
    """Publish only safe candidate identity fields for final-send telemetry."""

    state = _request_state(request)
    if state is None:
        return
    if not isinstance(candidate, dict):
        setattr(state, _CANDIDATE_CONTEXT_STATE_KEY, {})
        return
    setattr(
        state,
        _CANDIDATE_CONTEXT_STATE_KEY,
        {
            "provider": _safe_context_value(candidate.get("provider")),
            "model": _safe_context_value(candidate.get("model")),
            "route_family": _safe_context_value(candidate.get("route_family")),
            "config_epoch_tag": _safe_context_value(
                candidate.get("config_epoch_tag")
            ),
            "account_hash": _safe_context_value(
                candidate.get("codex_oauth_account_hash")
            ),
            "lane_key": _safe_context_value(candidate.get("codex_oauth_lane_key")),
        },
    )


@dataclass(frozen=True)
class ProviderCallReservation:
    """One reserved logical provider request."""

    ordinal: int
    provider: str
    reason: str
    target_fingerprint: Optional[str]
    candidate_fingerprint: Optional[str]
    model_fingerprint: Optional[str]
    account_fingerprint: Optional[str]
    reserved_at_monotonic: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "provider": self.provider,
            "reason": self.reason,
            "target_fingerprint": self.target_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "account_fingerprint": self.account_fingerprint,
        }


@dataclass(frozen=True)
class _CapacityRetryAuthorization:
    """One-use admission granted by a classified precommit capacity failure."""

    prior_ordinal: int
    target_fingerprint: str
    model_fingerprint: str
    account_fingerprint: str
    deadline_at_monotonic: float
    authorized_at_monotonic: float

    def to_metadata(self) -> dict[str, Any]:
        remaining_seconds = max(
            0.0,
            self.deadline_at_monotonic - time.monotonic(),
        )
        return {
            "prior_ordinal": self.prior_ordinal,
            "target_fingerprint": self.target_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "account_fingerprint": self.account_fingerprint,
            "remaining_seconds": round(remaining_seconds, 3),
        }


class ProviderCallLedgerExhausted(RuntimeError):
    """Raised before a send when the immutable request ledger is exhausted."""

    aawm_call_ledger_exhausted = True
    attempted_provider_call = False
    status_code = 503

    def __init__(
        self,
        *,
        ledger: "ProviderCallLedger",
        reason: str,
    ) -> None:
        self.reason = reason
        self.ledger_snapshot = ledger.snapshot()
        self.detail = {
            "error": {
                "message": (
                    "The request-wide provider-call budget was exhausted "
                    "before another upstream request."
                ),
                "type": "provider_call_ledger_exhausted",
                "code": "aawm_request_call_ledger_exhausted",
                "retryable": False,
                "reason": reason,
                "logical_provider_calls": self.ledger_snapshot[
                    "logical_provider_calls"
                ],
                "max_logical_provider_calls": self.ledger_snapshot[
                    "max_logical_provider_calls"
                ],
                "remaining_logical_provider_calls": self.ledger_snapshot[
                    "remaining_logical_provider_calls"
                ],
                "ordinary_calls_available": self.ledger_snapshot[
                    "ordinary_calls_available"
                ],
                "capacity_retry_authorized": self.ledger_snapshot[
                    "capacity_retry_authorized"
                ],
            }
        }
        super().__init__(
            "aawm_request_call_ledger_exhausted:"
            f"{reason}:"
            f"{self.ledger_snapshot['logical_provider_calls']}/"
            f"{self.ledger_snapshot['max_logical_provider_calls']}"
        )


class ProviderCallLedger:
    """Immutable request-wide authority for logical provider sends."""

    def __init__(
        self,
        *,
        provider: str,
        max_logical_provider_calls: int,
        deadline_seconds: Optional[float] = None,
    ) -> None:
        self.provider = _safe_context_value(provider) or "unknown"
        self.max_logical_provider_calls = max(
            1,
            min(_MAX_LOGICAL_PROVIDER_CALLS_LIMIT, int(max_logical_provider_calls)),
        )
        self.deadline_seconds = (
            max(0.0, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        )
        self._started_at_monotonic = time.monotonic()
        self._next_ordinal = 1
        self._logical_provider_calls = 0
        self._transport_connection_failures = 0
        self._records: list[ProviderCallReservation] = []
        self._capacity_retry_authorization: Optional[
            _CapacityRetryAuthorization
        ] = None
        self._request_fingerprint = uuid4().hex[:16]

    @property
    def logical_provider_calls(self) -> int:
        return self._logical_provider_calls

    @property
    def transport_connection_failures(self) -> int:
        return self._transport_connection_failures

    @property
    def transport_connection_attempts(self) -> int:
        """Backward-compatible alias for the failure-only counter."""

        return self.transport_connection_failures

    @property
    def reservations(self) -> tuple[ProviderCallReservation, ...]:
        return tuple(self._records)

    @staticmethod
    def _model_fingerprint(context: Mapping[str, Any]) -> Optional[str]:
        return _fingerprint(context.get("model"))

    @staticmethod
    def _account_fingerprint(context: Mapping[str, Any]) -> Optional[str]:
        account_hash = _safe_context_value(context.get("account_hash"))
        lane_key = _safe_context_value(context.get("lane_key"))
        if account_hash is None and lane_key is None:
            return None
        return _fingerprint(
            "|".join(
                value or ""
                for value in (
                    account_hash,
                    lane_key,
                )
            )
        )

    def authorize_capacity_retry(
        self,
        *,
        target: Any,
        candidate_context: Optional[dict[str, Any]],
        deadline_at_monotonic: float,
    ) -> bool:
        """Grant one narrowly-bound send after a classified precommit failure.

        The grant is tied to the latest reservation and cannot be used for a
        changed target, model, account, or lane.  The caller supplies the
        coordinator's original absolute deadline; this method never starts a
        new retry timer.
        """

        if self._capacity_retry_authorization is not None:
            return False
        if not self._records:
            return False
        prior = self._records[-1]
        context = candidate_context or {}
        target_fingerprint = _fingerprint(target)
        model_fingerprint = self._model_fingerprint(context)
        account_fingerprint = self._account_fingerprint(context)
        if (
            target_fingerprint is None
            or model_fingerprint is None
            or account_fingerprint is None
            or prior.target_fingerprint != target_fingerprint
            or prior.model_fingerprint != model_fingerprint
            or prior.account_fingerprint != account_fingerprint
        ):
            return False
        deadline_at_monotonic = float(deadline_at_monotonic)
        if deadline_at_monotonic <= time.monotonic():
            return False
        self._capacity_retry_authorization = _CapacityRetryAuthorization(
            prior_ordinal=prior.ordinal,
            target_fingerprint=target_fingerprint,
            model_fingerprint=model_fingerprint,
            account_fingerprint=account_fingerprint,
            deadline_at_monotonic=deadline_at_monotonic,
            authorized_at_monotonic=time.monotonic(),
        )
        return True

    def _capacity_retry_authorization_matches(
        self,
        authorization: _CapacityRetryAuthorization,
        *,
        target_fingerprint: Optional[str],
        model_fingerprint: Optional[str],
        account_fingerprint: Optional[str],
    ) -> bool:
        if not self._records:
            return False
        prior = self._records[-1]
        return (
            prior.ordinal == authorization.prior_ordinal
            and target_fingerprint == authorization.target_fingerprint
            and model_fingerprint == authorization.model_fingerprint
            and account_fingerprint == authorization.account_fingerprint
            and prior.target_fingerprint == authorization.target_fingerprint
            and prior.model_fingerprint == authorization.model_fingerprint
            and prior.account_fingerprint == authorization.account_fingerprint
        )

    def reserve(
        self,
        *,
        target: Any = None,
        reason: str = "provider_request",
        candidate_context: Optional[dict[str, Any]] = None,
        prior_response_closed: bool,
        allow_capacity_retry: bool = False,
    ) -> ProviderCallReservation:
        if not prior_response_closed:
            raise RuntimeError(
                "provider call reservation requires prior response closure"
            )
        elapsed_seconds = time.monotonic() - self._started_at_monotonic
        if (
            self.deadline_seconds is not None
            and elapsed_seconds >= self.deadline_seconds
        ):
            raise ProviderCallLedgerExhausted(
                ledger=self,
                reason="deadline_exhausted",
            )
        context = candidate_context or {}
        target_fingerprint = _fingerprint(target)
        model_fingerprint = self._model_fingerprint(context)
        account_fingerprint = self._account_fingerprint(context)
        capacity_retry_authorized = False
        authorization = self._capacity_retry_authorization
        if authorization is not None and allow_capacity_retry:
            if time.monotonic() >= authorization.deadline_at_monotonic:
                self._capacity_retry_authorization = None
                raise ProviderCallLedgerExhausted(
                    ledger=self,
                    reason="capacity_retry_deadline_exhausted",
                )
            if not self._capacity_retry_authorization_matches(
                authorization,
                target_fingerprint=target_fingerprint,
                model_fingerprint=model_fingerprint,
                account_fingerprint=account_fingerprint,
            ):
                self._capacity_retry_authorization = None
                raise ProviderCallLedgerExhausted(
                    ledger=self,
                    reason="capacity_retry_authorization_mismatch",
                )
            capacity_retry_authorized = True
        if (
            self._logical_provider_calls >= self.max_logical_provider_calls
            and not capacity_retry_authorized
        ):
            raise ProviderCallLedgerExhausted(
                ledger=self,
                reason="logical_call_cap_exhausted",
            )

        if capacity_retry_authorized:
            self._capacity_retry_authorization = None
        reservation = ProviderCallReservation(
            ordinal=self._next_ordinal,
            provider=self.provider,
            reason=_safe_context_value(reason) or "provider_request",
            target_fingerprint=target_fingerprint,
            candidate_fingerprint=_fingerprint(
                "|".join(
                    str(context.get(key) or "")
                    for key in ("provider", "model", "route_family", "config_epoch_tag")
                )
            ),
            model_fingerprint=model_fingerprint,
            account_fingerprint=account_fingerprint,
            reserved_at_monotonic=time.monotonic(),
        )
        self._records.append(reservation)
        self._logical_provider_calls += 1
        self._next_ordinal += 1
        return reservation

    def record_transport_connection_failure(self) -> None:
        """Record an observable connection failure separately from sends."""

        self._transport_connection_failures += 1

    def record_transport_connection_attempt(self) -> None:
        """Backward-compatible alias for the failure-only counter."""

        self.record_transport_connection_failure()

    def snapshot(self) -> dict[str, Any]:
        remaining = max(
            0,
            self.max_logical_provider_calls - self._logical_provider_calls,
        )
        elapsed_seconds = time.monotonic() - self._started_at_monotonic
        capacity_authorization = self._capacity_retry_authorization
        capacity_remaining_seconds = (
            max(
                0.0,
                capacity_authorization.deadline_at_monotonic
                - time.monotonic(),
            )
            if capacity_authorization is not None
            else 0.0
        )
        return {
            "request_fingerprint": self._request_fingerprint,
            "provider": self.provider,
            "max_logical_provider_calls": self.max_logical_provider_calls,
            "logical_provider_calls": self._logical_provider_calls,
            "remaining_logical_provider_calls": remaining,
            "ordinary_calls_available": remaining > 0,
            "capacity_retry_authorized": capacity_authorization is not None,
            "capacity_retry_remaining_seconds": round(
                capacity_remaining_seconds,
                3,
            ),
            "next_attempt_ordinal": self._next_ordinal,
            "transport_connection_failures": self._transport_connection_failures,
            "deadline_seconds": self.deadline_seconds,
            "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
            "reservations": [record.to_metadata() for record in self._records],
            "capacity_retry_authorization": (
                capacity_authorization.to_metadata()
                if capacity_authorization is not None
                else None
            ),
        }


class ProviderCallReplayBlocked(RuntimeError):
    """Raised before a send when the delivered wire forbids request replay."""

    aawm_openai_wire_replay_blocked = True
    attempted_provider_call = False
    status_code = 409

    def __init__(
        self,
        *,
        commitment: Mapping[str, Any],
        ledger: Optional[ProviderCallLedger] = None,
    ) -> None:
        self.wire_commitment = _safe_wire_commitment(commitment)
        self.ledger_snapshot = ledger.snapshot() if ledger is not None else None
        self.detail = {
            "error": {
                "message": (
                    "The request cannot be replayed after the OpenAI "
                    "Responses wire was committed."
                ),
                "type": "openai_wire_replay_blocked",
                "code": "aawm_openai_wire_replay_blocked",
                "retryable": False,
                "wire_commitment": self.wire_commitment,
            }
        }
        super().__init__("aawm_openai_wire_replay_blocked")


def _safe_wire_commitment(
    commitment: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only bounded, non-secret wire lifecycle fields."""

    published_reason = commitment.get(
        _RESPONSES_REPLAY_PUBLISHED_LATCH_REASON_KEY
    )
    safe_commitment = {
        key: commitment.get(key)
        for key in (
            "state",
            "response_start_sent",
            "first_body_sent",
            "precommit_replay_committed",
            "commitment",
            "terminal_event_type",
            "disposition",
            "terminal_selected",
            "terminal_sent",
            "done_sent",
            "terminal_wire_committed",
            "done_wire_committed",
            "finalization_started",
            "finalized",
            _RESPONSES_REPLAY_LATCH_REASON_KEY,
            _RESPONSES_REPLAY_LATCH_EVENT_KEY,
        )
        if key in commitment
    }
    if isinstance(published_reason, str):
        safe_commitment[_RESPONSES_REPLAY_LATCH_REASON_KEY] = (
            published_reason
            if published_reason in _RESPONSES_REPLAY_COMMITMENT_REASONS
            else "unknown_semantics"
        )
    if _RESPONSES_REPLAY_LATCH_REASON_KEY in safe_commitment:
        reason = safe_commitment[_RESPONSES_REPLAY_LATCH_REASON_KEY]
        if not isinstance(reason, str) or reason not in _RESPONSES_REPLAY_COMMITMENT_REASONS:
            safe_commitment[_RESPONSES_REPLAY_LATCH_REASON_KEY] = (
                "unknown_semantics"
            )
    if _RESPONSES_REPLAY_LATCH_EVENT_KEY in safe_commitment:
        event_type = safe_commitment[_RESPONSES_REPLAY_LATCH_EVENT_KEY]
        reason = safe_commitment.get(_RESPONSES_REPLAY_LATCH_REASON_KEY)
        safe_commitment[_RESPONSES_REPLAY_LATCH_EVENT_KEY] = (
            _responses_replay_event_label(
                event_type if isinstance(event_type, str) else None,
                reason=reason if isinstance(reason, str) else None,
            )
        )
    return safe_commitment


def get_request_openai_wire_commitment(
    request: Request,
) -> Optional[dict[str, Any]]:
    """Return an immutable-by-convention copy of the wire lifecycle snapshot."""

    state = _request_state(request)
    commitment = (
        getattr(state, _WIRE_COMMITMENT_STATE_KEY, None)
        if state is not None
        else None
    )
    if not isinstance(commitment, Mapping):
        return None
    return _safe_wire_commitment(commitment)


def is_openai_wire_replay_prohibited(
    commitment: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether a request-scoped wire state forbids another provider send."""

    if not isinstance(commitment, Mapping):
        return False
    if any(
        commitment.get(flag) is True for flag in _WIRE_REPLAY_BLOCKING_FLAGS
    ):
        return True
    return commitment.get("disposition") in _WIRE_REPLAY_BLOCKING_DISPOSITIONS


def ensure_openai_wire_replay_allowed(
    request: Request,
    *,
    ledger: Optional[ProviderCallLedger] = None,
) -> None:
    """Fail closed before reservation when the delivered wire forbids replay."""

    commitment = get_request_openai_wire_commitment(request)
    if is_openai_wire_replay_prohibited(commitment):
        raise ProviderCallReplayBlocked(
            commitment=commitment or {},
            ledger=ledger,
        )


def publish_wire_commitment_snapshot(
    request: Request,
    *,
    commitment: Optional[Mapping[str, Any]] = None,
) -> None:
    """Attach wire state to ledger telemetry without becoming its authority."""

    state = _request_state(request)
    if state is None:
        return
    snapshot = (
        _safe_wire_commitment(commitment)
        if isinstance(commitment, Mapping)
        else get_request_openai_wire_commitment(request)
    )
    if snapshot is None:
        return
    ledger = get_request_provider_call_ledger(request)
    if ledger is None:
        return
    ledger_snapshot = ledger.snapshot()
    ledger_snapshot["wire_commitment"] = snapshot
    setattr(state, "aawm_openai_send_ledger_snapshot", ledger_snapshot)


def get_or_create_openai_provider_call_ledger(
    request: Request,
    *,
    custom_llm_provider: Optional[str],
    target: Any = None,
) -> Optional[ProviderCallLedger]:
    if not is_openai_logical_send_target(
        custom_llm_provider=custom_llm_provider,
        target=target,
    ):
        return None
    state = _request_state(request)
    if state is None:
        return None
    ledger = getattr(state, _LEDGER_STATE_KEY, None)
    if isinstance(ledger, ProviderCallLedger):
        return ledger
    ledger = ProviderCallLedger(
        provider="openai",
        max_logical_provider_calls=_resolve_max_logical_provider_calls(),
    )
    setattr(state, _LEDGER_STATE_KEY, ledger)
    return ledger


def get_request_provider_call_ledger(
    request: Request,
) -> Optional[ProviderCallLedger]:
    state = _request_state(request)
    ledger = getattr(state, _LEDGER_STATE_KEY, None) if state is not None else None
    return ledger if isinstance(ledger, ProviderCallLedger) else None


def get_request_provider_call_ledger_snapshot(request: Request) -> Optional[dict[str, Any]]:
    ledger = get_request_provider_call_ledger(request)
    if ledger is None:
        return None
    snapshot = ledger.snapshot()
    commitment = get_request_openai_wire_commitment(request)
    if commitment is not None:
        snapshot["wire_commitment"] = commitment
    return snapshot


def publish_reservation_metadata(
    request: Request,
    *,
    reservation: ProviderCallReservation,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    state = _request_state(request)
    ledger = get_request_provider_call_ledger(request)
    if state is None or ledger is None:
        return
    setattr(state, "aawm_openai_send_ledger_ordinal", reservation.ordinal)
    ledger_snapshot = get_request_provider_call_ledger_snapshot(request)
    setattr(state, "aawm_openai_send_ledger_snapshot", ledger_snapshot)
    if isinstance(metadata, dict):
        metadata["aawm_openai_send_ledger_ordinal"] = reservation.ordinal
        metadata["aawm_openai_send_ledger"] = ledger_snapshot
        metadata["aawm_openai_send_ledger_reservation"] = (
            reservation.to_metadata()
        )


def record_transport_connection_failure(request: Request) -> None:
    ledger = get_request_provider_call_ledger(request)
    if ledger is None:
        return
    ledger.record_transport_connection_failure()
    state = _request_state(request)
    if state is not None:
        setattr(
            state,
            "aawm_openai_send_ledger_snapshot",
            get_request_provider_call_ledger_snapshot(request),
        )


def record_transport_connection_attempt(request: Request) -> None:
    """Backward-compatible alias for the failure-only telemetry."""

    record_transport_connection_failure(request)


def register_active_upstream_response(
    request: Request,
    response: Any,
) -> None:
    state = _request_state(request)
    if state is not None:
        setattr(state, _ACTIVE_RESPONSE_STATE_KEY, response)


def clear_active_upstream_response(
    request: Request,
    *,
    response: Any = None,
) -> None:
    state = _request_state(request)
    if state is None:
        return
    active = getattr(state, _ACTIVE_RESPONSE_STATE_KEY, None)
    if response is None or active is response:
        setattr(state, _ACTIVE_RESPONSE_STATE_KEY, None)


async def close_active_upstream_response(request: Request) -> None:
    state = _request_state(request)
    if state is None:
        return
    response = getattr(state, _ACTIVE_RESPONSE_STATE_KEY, None)
    if response is None:
        return
    try:
        close_fn = getattr(response, "aclose", None)
        if callable(close_fn):
            await close_fn()
        else:
            close_fn = getattr(response, "close", None)
            if callable(close_fn):
                close_fn()
    finally:
        setattr(state, _ACTIVE_RESPONSE_STATE_KEY, None)


def publish_responses_replay_commitment(
    request: Request,
    *,
    reason: str,
    event_type: Optional[str] = None,
) -> None:
    """Set the monotonic replay-commit latch before any yield that could expose
    committing data.

    Once set the latch forbids every internal resend path through the existing
    ``ensure_openai_wire_replay_allowed`` guard.  Callers must set it *before*
    an ``await`` or ``yield`` that hands committed output to the client.
    """

    if reason not in _RESPONSES_REPLAY_COMMITMENT_REASONS:
        raise ValueError(
            f"Unknown precommit replay-commitment reason: {reason!r}"
        )
    state = _request_state(request)
    if state is None:
        return
    commitment = getattr(state, _WIRE_COMMITMENT_STATE_KEY, None)
    if not isinstance(commitment, dict):
        commitment = {}
        setattr(state, _WIRE_COMMITMENT_STATE_KEY, commitment)

    existing_reason = commitment.get(_RESPONSES_REPLAY_LATCH_REASON_KEY)
    if existing_reason is not None:
        existing_priority = _RESPONSES_REPLAY_COMMITMENT_PRIORITY.get(
            existing_reason, 0
        )
        new_priority = _RESPONSES_REPLAY_COMMITMENT_PRIORITY.get(reason, 0)
        if new_priority <= existing_priority:
            return

    commitment["precommit_replay_committed"] = True
    commitment[_RESPONSES_REPLAY_LATCH_REASON_KEY] = reason
    commitment[_RESPONSES_REPLAY_PUBLISHED_LATCH_REASON_KEY] = reason
    commitment[_RESPONSES_REPLAY_LATCH_EVENT_KEY] = (
        _responses_replay_event_label(event_type, reason=reason)
    )


def get_responses_replay_commitment_reason(
    request: Request,
) -> Optional[str]:
    """Return the replay-commit reason without asserting replay prohibition."""

    state = _request_state(request)
    commitment = (
        getattr(state, _WIRE_COMMITMENT_STATE_KEY, None)
        if state is not None
        else None
    )
    if not isinstance(commitment, Mapping):
        return None
    reason = commitment.get(_RESPONSES_REPLAY_LATCH_REASON_KEY)
    return reason if isinstance(reason, str) else None


_PROVIDER_EXECUTABLE_TOOL_TYPES = frozenset({
    "mcp",
    "web_search",
    "web_search_preview",
    "code_interpreter",
    "computer",
    "image_generation",
    "file_search",
})


def responses_effective_provider_tools_allowed(
    provider_bound_body: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether a body is unsafe to replay because tool capability is unknown."""
    if not isinstance(provider_bound_body, Mapping):
        return True
    if "tools" not in provider_bound_body:
        return False
    tools = provider_bound_body.get("tools")
    if not isinstance(tools, list):
        return True
    for tool in tools:
        if not isinstance(tool, Mapping):
            return True
        tool_type = tool.get("type")
        if not isinstance(tool_type, str):
            return True
        if tool_type in _PROVIDER_EXECUTABLE_TOOL_TYPES:
            return True
        if tool_type == "function":
            if not isinstance(tool.get("name"), str) or not tool.get("name"):
                return True
            if "description" in tool and not isinstance(
                tool.get("description"), str
            ):
                return True
            if "parameters" in tool and not isinstance(
                tool.get("parameters"), Mapping
            ):
                return True
            if "strict" in tool and not isinstance(tool.get("strict"), bool):
                return True
            if not set(tool).issubset(
                {"type", "name", "description", "parameters", "strict"}
            ):
                return True
            continue
        if tool_type == "custom":
            if not isinstance(tool.get("name"), str) or not tool.get("name"):
                return True
            if "description" in tool and not isinstance(
                tool.get("description"), str
            ):
                return True
            if "format" in tool and not isinstance(tool.get("format"), Mapping):
                return True
            if not set(tool).issubset(
                {"type", "name", "description", "format"}
            ):
                return True
            continue
        # Any capability not explicitly known to be client-executed commits
        # before dispatch; this includes provider-executable and future tools.
        return True
    return False


__all__ = [
    "DEFAULT_OPENAI_MAX_LOGICAL_PROVIDER_CALLS",
    "ProviderCallLedger",
    "ProviderCallLedgerExhausted",
    "ProviderCallReplayBlocked",
    "ProviderCallReservation",
    "bind_openai_candidate_context",
    "clear_active_upstream_response",
    "close_active_upstream_response",
    "current_candidate_context",
    "ensure_openai_wire_replay_allowed",
    "get_or_create_openai_provider_call_ledger",
    "get_request_openai_wire_commitment",
    "get_request_provider_call_ledger",
    "get_request_provider_call_ledger_snapshot",
    "is_openai_wire_replay_prohibited",
    "is_openai_logical_send_target",
    "publish_reservation_metadata",
    "publish_wire_commitment_snapshot",
    "record_transport_connection_failure",
    "record_transport_connection_attempt",
    "register_active_upstream_response",
]
