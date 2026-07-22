"""Failure classification adapter + N-of-M cooldown evidence gate (Wave 2).

Wraps existing provider-classifier outputs and exception shapes into the
open ``FailureEvent`` vocabulary (Wave 1) without changing the return
contracts of the wrapped classifiers, and separately owns the
confidence-tiered N-of-M cooldown-evidence policy:

- ``structured`` confidence cools on a single event (N=1).
- ``marker`` (free-text-only) confidence requires N-within-a-sliding-window
  (default 3-in-60s) before cooling.
- ``unknown``/``client`` origin events are never coolable and never advance
  evidence toward cooling (see :func:`litellm...failure_vocabulary.is_coolable`).
- Cooldown duration prefers a signal-derived value (e.g. Retry-After) when
  present, else falls back to a capped exponential backoff.
- After a cooldown expires, a single half-open probe is allowed; success
  restores the candidate, failure re-cools with continued backoff.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from . import failure_vocabulary as fv
from .retry import exponential_backoff_seconds
from .state import AliasFamilyState

_DEFAULT_MARKER_N = 3
_DEFAULT_MARKER_WINDOW_SECONDS = 60.0
_DEFAULT_STRUCTURED_N = 1
_DEFAULT_BASE_SECONDS = 30.0
_DEFAULT_MAX_SECONDS = 1800.0

_QUOTA_MARKERS = ("quota",)
_CAPACITY_MARKERS = ("capacity",)
_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "too many requests")
_AUTH_MARKERS = ("api key", "unauthorized", "invalid auth", "auth")
_CLIENT_CANCELLED_MARKERS = ("cancelled", "canceled", "cancel")


def classify_failure(
    *,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    message: str = "",
    retry_after_seconds: Optional[float] = None,
) -> fv.FailureEvent:
    """Classify a status-code/message failure signal into a ``FailureEvent``.

    Structured classification (exact status code known) always yields
    ``confidence="structured"``. When no structured status code is present,
    falls back to free-text marker matching with ``confidence="marker"``.
    Unrecognized signals default to ``class_name="unknown"``,
    ``origin="unknown"`` (never coolable).
    """
    text = (message or "").lower()
    evidence: dict[str, str] = {}
    if status_code is not None:
        evidence["status_code"] = str(status_code)
    if retry_after_seconds is not None:
        evidence["retry_after_seconds"] = str(float(retry_after_seconds))

    if status_code == 429:
        if any(marker in text for marker in _QUOTA_MARKERS):
            return fv.FailureEvent(
                class_name="quota_exhausted",
                origin="upstream",
                confidence="structured",
                provider=provider,
                scope="provider",
                retryable=True,
                evidence=evidence,
            )
        return fv.FailureEvent(
            class_name="rate_limit",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if status_code in (401, 403):
        return fv.FailureEvent(
            class_name="auth",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="account",
            retryable=False,
            evidence=evidence,
        )
    if status_code == 404:
        return fv.FailureEvent(
            class_name="model_unavailable",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="model",
            retryable=False,
            evidence=evidence,
        )
    if status_code is not None and 500 <= status_code <= 599:
        return fv.FailureEvent(
            class_name="provider_5xx",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if status_code is not None and 400 <= status_code <= 499:
        return fv.FailureEvent(
            class_name="provider_4xx_other",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )

    # No structured status code: free-text marker matching (low confidence).
    if any(marker in text for marker in _CAPACITY_MARKERS):
        return fv.FailureEvent(
            class_name="capacity",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _QUOTA_MARKERS):
        return fv.FailureEvent(
            class_name="quota_exhausted",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return fv.FailureEvent(
            class_name="rate_limit",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _AUTH_MARKERS):
        return fv.FailureEvent(
            class_name="auth",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="account",
            retryable=False,
            evidence=evidence,
        )
    if any(marker in text for marker in _CLIENT_CANCELLED_MARKERS):
        return fv.FailureEvent(
            class_name="client_cancelled",
            origin="client",
            confidence="marker",
            provider=provider,
            scope="lane",
            retryable=False,
            evidence=evidence,
        )

    return fv.FailureEvent(
        class_name="unknown",
        origin="unknown",
        confidence="unknown",
        provider=provider,
        scope="lane",
        retryable=None,
        evidence=evidence,
    )


def classify_exception(exc: BaseException) -> fv.FailureEvent:
    """Classify an exception instance into a ``FailureEvent``.

    ``asyncio.CancelledError`` is a ``BaseException`` (not ``Exception``) and
    represents a caller-initiated abort, not an upstream failure -- it must
    classify as ``client_cancelled``/``origin="client"`` (never coolable).
    Everything else falls back to free-text classification of ``str(exc)``.
    """
    if isinstance(exc, asyncio.CancelledError):
        return fv.FailureEvent(
            class_name="client_cancelled",
            origin="client",
            confidence="structured",
            provider=None,
            scope="lane",
            retryable=False,
            evidence={"exception_type": type(exc).__name__},
        )
    return classify_failure(status_code=None, provider=None, message=str(exc))


@dataclass(frozen=True)
class CooldownDecision:
    """Result of feeding one ``FailureEvent`` into the evidence gate."""

    should_cool: bool
    duration_seconds: float = 0.0
    cooled_until_monotonic: Optional[float] = None
    scope: Optional[str] = None
    class_name: Optional[str] = None


@dataclass
class _KeyCooldownState:
    attempt: int = 0
    cooled_until_monotonic: float = 0.0
    probe_in_flight: bool = False


class CooldownEvidenceGate:
    """Confidence-tiered N-of-M cooldown-evidence policy.

    - ``structured`` confidence: cools on a single event.
    - ``marker`` confidence: requires ``marker_n`` events within
      ``marker_window_seconds`` (sliding window), backed by
      ``AliasFamilyState.record_failure_evidence``.
    - Duration prefers a signal-derived value (e.g. ``retry_after_seconds``
      carried in ``FailureEvent.evidence``) when present, else a capped
      exponential backoff keyed off a per-cooldown-key attempt counter.
    - After expiry, a single half-open probe is allowed; success clears the
      key's state, failure leaves the attempt counter to continue escalating.
    """

    def __init__(
        self,
        *,
        marker_n: int = _DEFAULT_MARKER_N,
        marker_window_seconds: float = _DEFAULT_MARKER_WINDOW_SECONDS,
        structured_n: int = _DEFAULT_STRUCTURED_N,
        base_seconds: float = _DEFAULT_BASE_SECONDS,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
        family_state: Optional[AliasFamilyState] = None,
    ) -> None:
        self._marker_n = max(1, int(marker_n))
        self._marker_window_seconds = max(0.0, float(marker_window_seconds))
        self._structured_n = max(1, int(structured_n))
        self._base_seconds = base_seconds
        self._max_seconds = max_seconds
        self._family_state = family_state if family_state is not None else AliasFamilyState()
        self._key_state: dict[str, _KeyCooldownState] = {}

    def _state_for(self, cooldown_key: str) -> _KeyCooldownState:
        state = self._key_state.get(cooldown_key)
        if state is None:
            state = _KeyCooldownState()
            self._key_state[cooldown_key] = state
        return state

    def record(
        self,
        *,
        cooldown_key: str,
        event: fv.FailureEvent,
        now_monotonic: Optional[float] = None,
    ) -> CooldownDecision:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if not fv.is_coolable(event):
            return CooldownDecision(should_cool=False)

        if event.confidence == "structured":
            evidence_met = True
        else:
            count = self._family_state.record_failure_evidence(
                cooldown_key=cooldown_key,
                confidence=event.confidence,
                window_seconds=self._marker_window_seconds,
                now_monotonic=now,
            )
            evidence_met = count >= self._marker_n

        if not evidence_met:
            return CooldownDecision(should_cool=False)

        key_state = self._state_for(cooldown_key)
        key_state.attempt += 1
        key_state.probe_in_flight = False

        duration = self._resolve_duration(event, attempt=key_state.attempt)
        cooled_until = now + duration
        key_state.cooled_until_monotonic = cooled_until
        return CooldownDecision(
            should_cool=True,
            duration_seconds=duration,
            cooled_until_monotonic=cooled_until,
            scope=event.scope,
            class_name=event.class_name,
        )

    def _resolve_duration(self, event: fv.FailureEvent, *, attempt: int) -> float:
        raw_retry_after = (event.evidence or {}).get("retry_after_seconds")
        if raw_retry_after is not None:
            try:
                return float(raw_retry_after)
            except (TypeError, ValueError):
                pass
        return exponential_backoff_seconds(
            attempt,
            base_seconds=self._base_seconds,
            max_seconds=self._max_seconds,
        )

    def is_cooled(
        self,
        *,
        cooldown_key: str,
        now_monotonic: Optional[float] = None,
    ) -> bool:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        state = self._key_state.get(cooldown_key)
        if state is None:
            return False
        return state.cooled_until_monotonic > now

    def allow_half_open_probe(
        self,
        *,
        cooldown_key: str,
        now_monotonic: Optional[float] = None,
    ) -> bool:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if self.is_cooled(cooldown_key=cooldown_key, now_monotonic=now):
            return False
        state = self._state_for(cooldown_key)
        if state.probe_in_flight:
            return False
        state.probe_in_flight = True
        return True

    def record_probe_result(self, *, cooldown_key: str, success: bool) -> None:
        state = self._state_for(cooldown_key)
        state.probe_in_flight = False
        if success:
            state.attempt = 0
            state.cooled_until_monotonic = 0.0
