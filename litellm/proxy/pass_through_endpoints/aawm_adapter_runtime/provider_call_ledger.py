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
from typing import Any, Optional
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
        self._transport_connection_attempts = 0
        self._records: list[ProviderCallReservation] = []
        self._request_fingerprint = uuid4().hex[:16]

    @property
    def logical_provider_calls(self) -> int:
        return self._logical_provider_calls

    @property
    def transport_connection_attempts(self) -> int:
        return self._transport_connection_attempts

    @property
    def reservations(self) -> tuple[ProviderCallReservation, ...]:
        return tuple(self._records)

    def reserve(
        self,
        *,
        target: Any = None,
        reason: str = "provider_request",
        candidate_context: Optional[dict[str, Any]] = None,
        prior_response_closed: bool,
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

    def record_transport_connection_attempt(self) -> None:
        """Record an observable connection failure separately from sends."""

        self._transport_connection_attempts += 1

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
            "transport_connection_attempts": self._transport_connection_attempts,
            "deadline_seconds": self.deadline_seconds,
            "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
            "reservations": [record.to_metadata() for record in self._records],
        }


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
    return ledger.snapshot() if ledger is not None else None


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
    setattr(state, "aawm_openai_send_ledger_snapshot", ledger.snapshot())
    if isinstance(metadata, dict):
        metadata["aawm_openai_send_ledger_ordinal"] = reservation.ordinal
        metadata["aawm_openai_send_ledger"] = ledger.snapshot()
        metadata["aawm_openai_send_ledger_reservation"] = (
            reservation.to_metadata()
        )


def record_transport_connection_attempt(request: Request) -> None:
    ledger = get_request_provider_call_ledger(request)
    if ledger is None:
        return
    ledger.record_transport_connection_attempt()
    state = _request_state(request)
    if state is not None:
        setattr(state, "aawm_openai_send_ledger_snapshot", ledger.snapshot())


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
    "ProviderCallReservation",
    "bind_openai_candidate_context",
    "clear_active_upstream_response",
    "close_active_upstream_response",
    "current_candidate_context",
    "get_or_create_openai_provider_call_ledger",
    "get_request_provider_call_ledger",
    "get_request_provider_call_ledger_snapshot",
    "is_openai_logical_send_target",
    "publish_reservation_metadata",
    "record_transport_connection_attempt",
    "register_active_upstream_response",
]
