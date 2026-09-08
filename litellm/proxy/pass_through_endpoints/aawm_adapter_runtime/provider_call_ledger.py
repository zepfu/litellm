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
)
_WIRE_REPLAY_BLOCKING_DISPOSITIONS = frozenset({"cancelled", "disconnected"})


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
    account_fingerprint: Optional[str]
    reserved_at_monotonic: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "provider": self.provider,
            "reason": self.reason,
            "target_fingerprint": self.target_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "account_fingerprint": self.account_fingerprint,
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

    def ensure_reservation_allowed(self, *, wire_commitment: Any = None) -> None:
        """Reject sends after the final OpenAI Responses wire is committed."""

        reason = _wire_commitment_denial_reason(wire_commitment)
        if reason is not None:
            raise ProviderCallReplayBlocked(
                commitment=wire_commitment,
                ledger=self,
                reason=reason,
            )

    def reserve(
        self,
        *,
        target: Any = None,
        reason: str = "provider_request",
        candidate_context: Optional[dict[str, Any]] = None,
        prior_response_closed: bool,
        wire_commitment: Any = None,
    ) -> ProviderCallReservation:
        if not prior_response_closed:
            raise RuntimeError(
                "provider call reservation requires prior response closure"
            )
        self.ensure_reservation_allowed(wire_commitment=wire_commitment)
        elapsed_seconds = time.monotonic() - self._started_at_monotonic
        if (
            self.deadline_seconds is not None
            and elapsed_seconds >= self.deadline_seconds
        ):
            raise ProviderCallLedgerExhausted(
                ledger=self,
                reason="deadline_exhausted",
            )
        if self._logical_provider_calls >= self.max_logical_provider_calls:
            raise ProviderCallLedgerExhausted(
                ledger=self,
                reason="logical_call_cap_exhausted",
            )

        context = candidate_context or {}
        reservation = ProviderCallReservation(
            ordinal=self._next_ordinal,
            provider=self.provider,
            reason=_safe_context_value(reason) or "provider_request",
            target_fingerprint=_fingerprint(target),
            candidate_fingerprint=_fingerprint(
                "|".join(
                    str(context.get(key) or "")
                    for key in ("provider", "model", "route_family", "config_epoch_tag")
                )
            ),
            account_fingerprint=_fingerprint(
                context.get("account_hash") or context.get("lane_key")
            ),
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
        return {
            "request_fingerprint": self._request_fingerprint,
            "provider": self.provider,
            "max_logical_provider_calls": self.max_logical_provider_calls,
            "logical_provider_calls": self._logical_provider_calls,
            "remaining_logical_provider_calls": remaining,
            "next_attempt_ordinal": self._next_ordinal,
            "transport_connection_failures": self._transport_connection_failures,
            "deadline_seconds": self.deadline_seconds,
            "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
            "reservations": [record.to_metadata() for record in self._records],
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
        reason: str = "wire_replay_prohibited",
    ) -> None:
        self.reason = _safe_context_value(reason) or "wire_replay_prohibited"
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
                "reason": self.reason,
                "wire_commitment": self.wire_commitment,
            }
        }
        super().__init__("aawm_openai_wire_replay_blocked")


def _safe_wire_commitment(
    commitment: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only bounded, non-secret wire lifecycle fields."""

    return {
        key: commitment.get(key)
        for key in (
            "state",
            "response_start_sent",
            "first_body_sent",
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
        )
        if key in commitment
    }


def get_request_openai_wire_commitment(
    request: Request,
) -> Optional[dict[str, Any]]:
    """Return an immutable-by-convention copy of the wire lifecycle snapshot."""

    state = _request_state(request)
    if state is None:
        return None
    commitment = getattr(state, _WIRE_COMMITMENT_STATE_KEY, None)
    if not isinstance(commitment, Mapping):
        trace = getattr(state, "_aawm_openai_responses_wire_trace", None)
        snapshot = getattr(trace, "snapshot", None)
        if not callable(snapshot):
            return None
        try:
            commitment = snapshot()
        except Exception:
            return None
    if not isinstance(commitment, Mapping):
        return None
    return _safe_wire_commitment(commitment)


def get_openai_wire_commitment(request: Request) -> Optional[dict[str, Any]]:
    """Backward-compatible alias for the request-scoped wire snapshot."""

    return get_request_openai_wire_commitment(request)


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
    return str(commitment.get("disposition") or "").strip().lower() in (
        _WIRE_REPLAY_BLOCKING_DISPOSITIONS
    )


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


def _wire_commitment_denial_reason(
    wire_commitment: Any,
) -> Optional[str]:
    if not isinstance(wire_commitment, Mapping):
        return None

    disposition = str(wire_commitment.get("disposition") or "").strip().lower()
    if disposition in {"cancelled", "disconnected"}:
        return f"wire_{disposition}"

    commitment = str(wire_commitment.get("commitment") or "").strip().lower()
    if commitment in {"headers", "body", "terminal", "done", "finalized"}:
        return f"wire_commitment_{commitment}"

    state = str(wire_commitment.get("state") or "").strip().lower()
    if state in {
        "headers_started",
        "body_started",
        "terminal_selected",
        "terminal_sent",
        "done_sent",
    }:
        return f"wire_state_{state}"

    for field in (
        "response_start_sent",
        "first_body_sent",
        "terminal_selected",
        "terminal_sent",
        "done_sent",
        "terminal_wire_committed",
        "done_wire_committed",
        "finalization_started",
        "finalized",
    ):
        if wire_commitment.get(field) is True:
            return f"wire_{field}"
    return None


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
    "get_openai_wire_commitment",
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
