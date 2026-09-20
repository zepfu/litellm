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

Scope policy (R3-3, operator-DECIDED, bounded):
- Structured 429 (plain rate-limit AND quota-text) and marker-only
  ``rate_limit`` classify ``scope="model"`` so a throttled model cools only its
  own ``provider:model:lane`` key, not the whole provider. Marker-only
  ``rate_limit`` still only cools AFTER its existing N-of-M evidence threshold
  is met (the per-event ``scope`` is model; the gate's decision to cool is a
  separate concern).
- Marker-only ``capacity`` stays ``scope="provider"`` and marker-only
  ``quota_exhausted`` is ``scope="account"``: these lower-confidence signals are
  deliberately NOT widened by the OpenRouter model-scope decision.
- Unchanged: auth ``"account"``, 404 ``"model"``, 5xx ``"provider"``,
  client_cancelled ``"lane"``, unknown ``"lane"``.

Non-goal: this wave makes a bounded operator scope decision only. NO global
provider/account fan-out framework is built here -- the capacity/quota scopes
above are preserved precisely so the model-scope decision does not become a
framework-wide under-cooling policy.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

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
_OVERLOAD_MARKERS = ("provider_overloaded", "overloaded_error")
_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "too many requests")
_AUTH_MARKERS = ("api key", "unauthorized", "invalid auth", "auth")
_CLIENT_CANCELLED_MARKERS = ("cancelled", "canceled", "cancel")

# D1-587: fixture-backed image-content sub-errors (open-registry growth).
# Only markers with direct catalog CSV evidence. No zero-hit phrase variants
# such as "invalid media", hyphenated "content-filter", or compact
# "finish_reason=error".
_INVALID_MEDIA_MARKERS = (
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "invalid_image_mode",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_file_too_large",
    "unsupported_image_media_type",
    "unsupported_image_format",
    "empty_image_file",
    "image_not_found",
    "image_file_not_found",
)
_IMAGE_DOWNLOAD_MARKERS = (
    "failed_to_download_image",
    "image_download_failed",
)
_CONTENT_POLICY_MARKERS = (
    "image_content_policy_violation",
    "content_policy_violation",
    "content_policy",
    "content_filter",
    "data_inspection_failed",
    "datainspectionfailed",
    "ipinfringementsuspect",
)

# D1-587: HTTP-200-body / SSE stream failure markers from catalog fixtures.
_STREAM_FAILURE_MARKERS = (
    "finish_reason = error",
    "response.failed",
    "response.error",
    "top-level error",
    "http remains 200",
)

# D1-587: fixture-backed usage/limit/quota machine codes (open-registry growth).
# Exact catalog tokens only. Deliberately excludes successful non-error
# truncation prose such as "finish_reason = length".
_USAGE_LIMIT_MARKERS = (
    "context_length_exceeded",
    "max_tokens_exceeded",
    "max_output_tokens",
    "max_steps_reached",
    "token_limit_exceeded",
)
_QUOTA_MACHINE_MARKERS = (
    "payment_required",
)
_PAYLOAD_LIMIT_MARKERS = (
    "payload_too_large",
)

# D1-587: fixture-backed invalid-request machine codes (open-registry growth).
# Exact catalog tokens only. Matched via hyphen-aware boundaries (see
# _invalid_request_marker_match). Deliberately excludes hyphen/spaced variants
# and successful non-error prose.
_INVALID_REQUEST_MARKERS = (
    "invalid_request_error",
    "invalid_request",
    "invalid_prompt",
    "string_too_long",
    "unprocessable",
)

# D1-587: unambiguous residual provider machine codes. Exact catalog tokens
# only; hyphen-aware boundaries and negation/docs guards prevent prose or
# glued variants from cooling candidates.
_EXACT_MACHINE_CODE_MAPPINGS = (
    ("previous_response_not_found", "provider_4xx_other", False),
    ("websocket_connection_limit_reached", "transient", True),
    ("server_error", "provider_5xx", True),
    ("vector_store_timeout", "transient", True),
    ("precondition_failed", "provider_4xx_other", False),
)

# JSON-RPC wire codes observed in Wire mode catalog fixtures.
# Exact token form only: reject glued/prefixed junk such as x-32600y / --32600.
_JSON_RPC_CODE_RE = re.compile(
    r"(?<![a-z0-9_-])(-32(?:700|600|601|602|603|000|001|002|003))(?![a-z0-9_])"
)
_JSON_RPC_STRUCTURED: dict[int, tuple[str, str, bool]] = {
    # class_name, scope, retryable
    -32700: ("serialization", "provider", False),
    -32600: ("provider_4xx_other", "provider", False),
    -32601: ("model_unavailable", "model", False),
    -32602: ("provider_4xx_other", "provider", False),
    -32603: ("transient", "provider", True),
    -32000: ("provider_4xx_other", "provider", False),
    -32001: ("provider_4xx_other", "provider", False),
    -32002: ("model_unavailable", "model", False),
    -32003: ("transient", "provider", True),
}

# Documentation-only / fixture prose must not cool from marker hits alone.
_DOC_ONLY_RE = re.compile(
    r"\b(?:documentation only|docs only|for documentation|example only)\b"
)
# Negation immediately before a marker token (e.g. "not an invalid_image").
_NEGATION_BEFORE_MARKER_RE = re.compile(
    r"(?:\bnot\s+(?:an?\s+)?|\bno\s+|\bwithout\s+|\bnon-?)\s*$"
)
# Token boundary for machine-code / short phrase markers.
_MARKER_BOUNDARY_LEFT = r"(?<![a-z0-9_])"
_MARKER_BOUNDARY_RIGHT = r"(?![a-z0-9_])"

# Open-registry class names introduced for fixture-backed D1-587 themes.
D1_587_FAILURE_CLASSES: tuple[str, ...] = (
    "invalid_media",
    "content_policy",
    "stream_failure",
)


def register_d1_587_failure_classes(
    registry: Optional[fv.FailureClassRegistry] = None,
) -> fv.FailureClassRegistry:
    """Register fixture-backed D1-587 class names on an open registry."""
    target = registry if registry is not None else fv.FailureClassRegistry.with_seed_classes()
    for class_name in D1_587_FAILURE_CLASSES:
        target.register(class_name)
    # Ensure previously unused seed classes remain explicitly available.
    for class_name in ("serialization", "transient", "usage_limit"):
        target.register(class_name)
    return target


def _event(
    *,
    class_name: str,
    origin: fv.Origin,
    confidence: fv.Confidence,
    provider: Optional[str],
    scope: fv.Scope,
    retryable: Optional[bool],
    evidence: dict[str, str],
) -> fv.FailureEvent:
    return fv.FailureEvent(
        class_name=class_name,
        origin=origin,
        confidence=confidence,
        provider=provider,
        scope=scope,
        retryable=retryable,
        evidence=evidence,
    )


def _json_rpc_code_from_message(message_lower: str) -> Optional[int]:
    match = _JSON_RPC_CODE_RE.search(message_lower)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _is_documentation_only_text(text: str) -> bool:
    """True when the text is explicitly documentation/example-only prose."""
    return _DOC_ONLY_RE.search(text) is not None


def _marker_match(text: str, marker: str) -> bool:
    """Return True when ``marker`` appears as a non-negated token in ``text``.

    Uses left/right token boundaries so longer machine codes do not collapse
    into shorter prefixes (``invalid_image`` inside ``invalid_image_format``)
    and rejects negated prose such as ``not an invalid_image fixture``.
    """
    if not marker:
        return False
    pattern = re.compile(
        _MARKER_BOUNDARY_LEFT + re.escape(marker) + _MARKER_BOUNDARY_RIGHT
    )
    for match in pattern.finditer(text):
        prefix = text[: match.start()]
        if _NEGATION_BEFORE_MARKER_RE.search(prefix):
            continue
        return True
    return False


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_marker_match(text, marker) for marker in markers)


def classify_failure(
    *,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    message: str = "",
    retry_after_seconds: Optional[float] = None,
    provider_returned: bool = False,
    error_payload: Optional[dict] = None,
) -> fv.FailureEvent:
    """Classify a status-code/message failure signal into a ``FailureEvent``.

    Structured classification (exact status code known) always yields
    ``confidence="structured"``. When no structured status code is present,
    falls back to free-text marker matching with ``confidence="marker"``.
    Unrecognized signals default to ``class_name="unknown"``,
    ``origin="unknown"`` (never coolable).

    D1-587 grows mappings for fixture-backed image-content sub-errors,
    JSON-RPC negative wire codes, HTTP-200-body stream failures,
    usage/limit/quota machine codes, invalid-request machine codes,
    provider-overload markers, and exact residual machine codes without
    changing the open ``FailureEvent`` schema.
    """
    text = (message or "").lower()
    evidence: dict[str, str] = {}
    if status_code is not None:
        evidence["status_code"] = str(status_code)
    if retry_after_seconds is not None:
        evidence["retry_after_seconds"] = str(float(retry_after_seconds))

    # JSON-RPC wire codes are structured provider signals (negative ints).
    if status_code is not None and status_code in _JSON_RPC_STRUCTURED:
        class_name, scope, retryable = _JSON_RPC_STRUCTURED[status_code]
        evidence["json_rpc_code"] = str(status_code)
        return _event(
            class_name=class_name,
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope=scope,  # type: ignore[arg-type]
            retryable=retryable,
            evidence=evidence,
        )

    # Only an attributed, structured OpenRouter billing rejection can cool
    # a credential. HTTP 402 alone (including local budget guards) is not evidence.
    if status_code == 402 and provider == "openrouter" and provider_returned is True:
        error = error_payload.get("error") if isinstance(error_payload, dict) else None
        if isinstance(error, dict) and error.get("code") in (402, "402"):
            credit_message = " ".join(str(error.get("message") or "").casefold().split())
            if credit_message.startswith((
                "insufficient credits", "credit limit exceeded", "key limit exceeded",
                "key limit reached", "budget exceeded", "budget limit exceeded",
                "this request requires more credits", "you have exceeded your credit limit",
                "you've exceeded your credit limit",
            )):
                return _event(
                    class_name=fv.OPENROUTER_CREDIT_EXHAUSTED,
                    origin="upstream", confidence="structured", provider=provider,
                    scope="account", retryable=True, evidence=evidence,
                )

    if status_code == 429:
        if any(marker in text for marker in _QUOTA_MARKERS):
            return _event(
                class_name="quota_exhausted",
                origin="upstream",
                confidence="structured",
                provider=provider,
                scope="model",
                retryable=True,
                evidence=evidence,
            )
        return _event(
            class_name="rate_limit",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="model",
            retryable=True,
            evidence=evidence,
        )
    if status_code in (401, 403):
        return _event(
            class_name="auth",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="account",
            retryable=False,
            evidence=evidence,
        )
    if status_code == 404:
        return _event(
            class_name="model_unavailable",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="model",
            retryable=False,
            evidence=evidence,
        )
    if status_code is not None and 500 <= status_code <= 599:
        return _event(
            class_name="provider_5xx",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if status_code is not None and 400 <= status_code <= 499:
        # Prefer fixture-backed content/media/limit subclasses inside 4xx bodies.
        # Skip documentation-only prose so it does not subclass as coolable.
        if not _is_documentation_only_text(text):
            media_event = _classify_media_or_policy_markers(
                text=text,
                provider=provider,
                confidence="structured",
                evidence=evidence,
            )
            if media_event is not None:
                return media_event
            limit_event = _classify_limit_or_quota_markers(
                text=text,
                provider=provider,
                confidence="structured",
                evidence=evidence,
            )
            if limit_event is not None:
                return limit_event
            invalid_event = _classify_invalid_request_markers(
                text=text,
                provider=provider,
                confidence="structured",
                evidence=evidence,
            )
            if invalid_event is not None:
                return invalid_event
        return _event(
            class_name="provider_4xx_other",
            origin="upstream",
            confidence="structured",
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )

    # Structured HTTP 2xx streaming connections still carry body/event failures.
    if status_code == 200 and not _is_documentation_only_text(text):
        stream_event = _classify_stream_failure_markers(
            text=text,
            provider=provider,
            confidence="structured",
            evidence=evidence,
        )
        if stream_event is not None:
            return stream_event

    # Message-embedded JSON-RPC codes (catalog rows without a parsed status).
    # Exact token only; documentation-only prose stays unknown.
    if not _is_documentation_only_text(text):
        rpc_code = _json_rpc_code_from_message(text)
        if rpc_code is not None and rpc_code in _JSON_RPC_STRUCTURED:
            class_name, scope, retryable = _JSON_RPC_STRUCTURED[rpc_code]
            evidence["json_rpc_code"] = str(rpc_code)
            return _event(
                class_name=class_name,
                origin="upstream",
                confidence="marker",
                provider=provider,
                scope=scope,  # type: ignore[arg-type]
                retryable=retryable,
                evidence=evidence,
            )

    # No structured HTTP status (or unresolved 2xx): free-text markers.
    # Documentation-only prose must stay unknown / never coolable.
    if not _is_documentation_only_text(text):
        media_event = _classify_media_or_policy_markers(
            text=text,
            provider=provider,
            confidence="marker",
            evidence=evidence,
        )
        if media_event is not None:
            return media_event

        stream_event = _classify_stream_failure_markers(
            text=text,
            provider=provider,
            confidence="marker",
            evidence=evidence,
        )
        if stream_event is not None:
            return stream_event

        limit_event = _classify_limit_or_quota_markers(
            text=text,
            provider=provider,
            confidence="marker",
            evidence=evidence,
        )
        if limit_event is not None:
            return limit_event

        invalid_event = _classify_invalid_request_markers(
            text=text,
            provider=provider,
            confidence="marker",
            evidence=evidence,
        )
        if invalid_event is not None:
            return invalid_event

        if status_code is None:
            machine_code_event = _classify_exact_machine_code_markers(
                text=text,
                provider=provider,
                confidence="marker",
                evidence=evidence,
            )
            if machine_code_event is not None:
                return machine_code_event

        if _any_strict_machine_code_marker(text, _OVERLOAD_MARKERS):
            return _event(
                class_name="capacity",
                origin="upstream",
                confidence="marker",
                provider=provider,
                scope="provider",
                retryable=True,
                evidence=evidence,
            )
    if any(marker in text for marker in _CAPACITY_MARKERS):
        return _event(
            class_name="capacity",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _QUOTA_MARKERS):
        return _event(
            class_name="quota_exhausted",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="account",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return _event(
            class_name="rate_limit",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="model",
            retryable=True,
            evidence=evidence,
        )
    if any(marker in text for marker in _AUTH_MARKERS):
        return _event(
            class_name="auth",
            origin="upstream",
            confidence="marker",
            provider=provider,
            scope="account",
            retryable=False,
            evidence=evidence,
        )
    if any(marker in text for marker in _CLIENT_CANCELLED_MARKERS):
        return _event(
            class_name="client_cancelled",
            origin="client",
            confidence="marker",
            provider=provider,
            scope="lane",
            retryable=False,
            evidence=evidence,
        )

    return _event(
        class_name="unknown",
        origin="unknown",
        confidence="unknown",
        provider=provider,
        scope="lane",
        retryable=None,
        evidence=evidence,
    )


def _classify_media_or_policy_markers(
    *,
    text: str,
    provider: Optional[str],
    confidence: fv.Confidence,
    evidence: dict[str, str],
) -> Optional[fv.FailureEvent]:
    if _any_marker(text, _CONTENT_POLICY_MARKERS):
        return _event(
            class_name="content_policy",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )
    if _any_marker(text, _IMAGE_DOWNLOAD_MARKERS):
        return _event(
            class_name="transient",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    if _any_marker(text, _INVALID_MEDIA_MARKERS):
        return _event(
            class_name="invalid_media",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )
    return None


def _classify_limit_or_quota_markers(
    *,
    text: str,
    provider: Optional[str],
    confidence: fv.Confidence,
    evidence: dict[str, str],
) -> Optional[fv.FailureEvent]:
    """Map fixture-backed usage/limit/quota machine codes to seed classes."""
    if _any_marker(text, _QUOTA_MACHINE_MARKERS):
        return _event(
            class_name="quota_exhausted",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="account",
            # Catalog Billing/quota payment_required is non-retryable.
            retryable=False,
            evidence=evidence,
        )
    if _any_marker(text, _USAGE_LIMIT_MARKERS):
        return _event(
            class_name="usage_limit",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="model",
            retryable=False,
            evidence=evidence,
        )
    if _any_marker(text, _PAYLOAD_LIMIT_MARKERS):
        return _event(
            class_name="provider_4xx_other",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )
    return None


def _strict_machine_code_marker_match(text: str, marker: str) -> bool:
    """Match a machine-code token with hyphen treated as identifier-adjacent.

    Narrower than :func:`_marker_match`: left/right boundaries reject both
    alphanumerics/underscore *and* hyphen so catalog tokens do not match
    hyphen-glued variants. Global marker matching is unchanged.
    """
    if not marker:
        return False
    pattern = re.compile(
        r"(?<![a-z0-9_-])" + re.escape(marker) + r"(?![a-z0-9_-])"
    )
    for match in pattern.finditer(text):
        prefix = text[: match.start()]
        if _NEGATION_BEFORE_MARKER_RE.search(prefix):
            continue
        return True
    return False


def _invalid_request_marker_match(text: str, marker: str) -> bool:
    """Match an invalid-request machine-code token exactly."""
    return _strict_machine_code_marker_match(text, marker)


def _any_invalid_request_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_invalid_request_marker_match(text, marker) for marker in markers)


def _any_strict_machine_code_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_strict_machine_code_marker_match(text, marker) for marker in markers)


def _classify_invalid_request_markers(
    *,
    text: str,
    provider: Optional[str],
    confidence: fv.Confidence,
    evidence: dict[str, str],
) -> Optional[fv.FailureEvent]:
    """Map fixture-backed invalid-request machine codes to provider_4xx_other."""
    if _any_invalid_request_marker(text, _INVALID_REQUEST_MARKERS):
        return _event(
            class_name="provider_4xx_other",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=False,
            evidence=evidence,
        )
    return None


def _classify_exact_machine_code_markers(
    *,
    text: str,
    provider: Optional[str],
    confidence: fv.Confidence,
    evidence: dict[str, str],
) -> Optional[fv.FailureEvent]:
    """Map only the unambiguous residual provider machine-code markers."""
    for marker, class_name, retryable in _EXACT_MACHINE_CODE_MAPPINGS:
        if _strict_machine_code_marker_match(text, marker):
            return _event(
                class_name=class_name,
                origin="upstream",
                confidence=confidence,
                provider=provider,
                scope="provider",
                retryable=retryable,
                evidence=evidence,
            )
    return None


def _classify_stream_failure_markers(
    *,
    text: str,
    provider: Optional[str],
    confidence: fv.Confidence,
    evidence: dict[str, str],
) -> Optional[fv.FailureEvent]:
    if _any_marker(text, _STREAM_FAILURE_MARKERS):
        return _event(
            class_name="stream_failure",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    # Responses-compatible bare stream error event type (catalog fixture):
    # machine code "error" plus stream-context wording in Meaning/Protocol.
    # Require a non-negated standalone "error" token near stream context.
    if _marker_match(text, "stream") and _marker_match(text, "error"):
        return _event(
            class_name="stream_failure",
            origin="upstream",
            confidence=confidence,
            provider=provider,
            scope="provider",
            retryable=True,
            evidence=evidence,
        )
    return None


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
    last_scope: Optional[str] = None
    last_class_name: Optional[str] = None


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
        key_state.last_scope = event.scope
        key_state.last_class_name = event.class_name
        return CooldownDecision(
            should_cool=True,
            duration_seconds=duration,
            cooled_until_monotonic=cooled_until,
            scope=event.scope,
            class_name=event.class_name,
        )

    def current_decision(
        self,
        *,
        cooldown_key: str,
        now_monotonic: Optional[float] = None,
    ) -> CooldownDecision:
        """Reconstruct the gate's current authoritative decision for ``cooldown_key``.

        Unlike :meth:`record`, this does not consume new evidence -- it
        reports the outcome of the most recent ``record()`` call for this
        key (should-cool + remaining duration + scope/class), so callers
        that already fed evidence via ``record()`` can later apply the
        gate's decision without re-classifying a failure.
        """
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        state = self._key_state.get(cooldown_key)
        if state is None or state.cooled_until_monotonic <= now:
            return CooldownDecision(should_cool=False)
        return CooldownDecision(
            should_cool=True,
            duration_seconds=max(0.0, state.cooled_until_monotonic - now),
            cooled_until_monotonic=state.cooled_until_monotonic,
            scope=state.last_scope,
            class_name=state.last_class_name,
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


_ZEN_MESSAGES = {
    "auth": "OpenCode Zen authentication failed.",
    "billing": "OpenCode Zen billing requires attention.",
    "quota": "OpenCode Zen usage quota is exhausted. Retry later.",
    "rate": "OpenCode Zen rate limit reached. Retry later.",
    "capacity": "OpenCode Zen upstream capacity is temporarily exhausted. Retry later.",
    "model": "OpenCode Zen selected model is unavailable.",
    "format": "OpenCode Zen does not support the requested wire format.",
    "invalid_request": "OpenCode Zen rejected the request.",
    "timeout": "OpenCode Zen upstream request timed out.",
    "transient": "OpenCode Zen upstream is temporarily unavailable.",
    "terminal": "OpenCode Zen request failed.",
}


def classify_zen_failure(
    *, status_code: Optional[int], message: str = "",
    origin: fv.Origin = "unknown", route: Literal["direct", "alias"] = "direct",
    model_unavailable: bool = False,
    retry_after_seconds: Optional[float] = None,
    reset_after_seconds: Optional[float] = None,
) -> fv.ZenFailure:
    """Exact Zen status policy; marker refinement never overrides auth status.

    Only provider-attributed errors authorize retries/fallback/cooldowns. Bare
    404s are endpoint errors, not evidence that a configured model is disabled.
    Public details come exclusively from a fixed vocabulary, never provider text.
    """
    from .policy import (
        CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS,
        CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS,
    )

    safe_status = status_code if type(status_code) is int and 400 <= status_code <= 599 else 502
    text = " ".join(message.casefold().split())
    kind: fv.ZenFailureClass = "terminal"
    if origin == "upstream":
        if status_code in {401, 403}:
            kind = "auth"
        elif status_code == 402:
            kind = "billing"
        elif status_code in {400, 404, 429, 500, 502, 503, 529}:
            if status_code in {400, 404} and _any_marker(text, ("authenticationerror", "authentication_error", "invalid_api_key", "authorization_error")):
                kind = "auth"
            elif _any_marker(text, ("freeusagelimiterror", "free usage limit", "usage_limit_reached", "quota_exceeded", "insufficient_quota")):
                kind = "quota"
            elif _any_marker(text, ("creditserror", "no payment method", "add a payment method", "billing", "payment required", "payment_required")):
                kind = "billing"
            elif status_code in {400, 404} and "not supported for format openai" in text:
                kind = "format"
            elif status_code in {400, 404} and model_unavailable:
                kind = "model"
            elif status_code in {429, 500, 502, 503, 529} and _any_marker(text, ("model_at_capacity", "capacity_exhausted", "server_overloaded", "server_is_overloaded", "high_demand", "model_overloaded", "upstream_busy")):
                kind = "capacity"
            elif status_code == 429:
                kind = "rate"
            elif status_code in {500, 502, 503, 529}:
                kind = "transient"
            elif status_code == 400:
                kind = "invalid_request"
        elif status_code in {408, 504}:
            kind = "timeout"

    retryable = kind in {"quota", "rate", "capacity", "timeout", "transient"}
    fallback = origin == "upstream" and route == "alias" and kind not in {"invalid_request", "terminal"}
    scope: Literal["account", "candidate", "none"] = "account" if kind in {"auth", "billing", "quota"} else "candidate" if kind in {"rate", "capacity", "model", "format", "timeout", "transient"} else "none"
    legacy = {
        "auth": "candidate_unavailable", "billing": "candidate_unavailable",
        "quota": "usage_limit_reached", "rate": "rate_limited",
        "capacity": "capacity_exhausted", "model": "candidate_unavailable",
        "format": "provider_format_rejected", "timeout": "upstream_timeout",
        "transient": "upstream_transient_internal",
    }.get(kind)

    def bounded(value: Optional[float]) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) and 0 <= value <= 86400 else None

    retry_after = bounded(retry_after_seconds) if retryable else None
    reset_after = bounded(reset_after_seconds) if retryable else None
    wait = retry_after if retry_after is not None else reset_after
    cooldown = 0.0
    if fallback:
        cooldown = CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS if kind in {"timeout", "transient"} else CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
        if wait is not None:
            cooldown = max(1.0, wait)
    public_status = 429 if kind in {"quota", "rate", "capacity"} else safe_status
    return fv.ZenFailure(
        class_name=kind, origin=origin, route=route, status_code=safe_status,
        public_status_code=public_status, scope=scope, retryable=retryable,
        fallback=fallback, error_class=legacy, cooldown_seconds=cooldown,
        retry_after_seconds=retry_after, reset_after_seconds=reset_after,
        public_detail=_ZEN_MESSAGES[kind],
    )


def zen_failure_event(failure: fv.ZenFailure) -> fv.FailureEvent:
    """Project the authoritative Zen result into the open shadow vocabulary."""
    names = {"quota": "quota_exhausted", "rate": "rate_limit", "model": "model_unavailable", "format": "serialization", "invalid_request": "provider_4xx_other", "timeout": "transient", "terminal": "provider_4xx_other"}
    evidence = {"status_code": str(failure.status_code)}
    if failure.retry_after_seconds is not None:
        evidence["retry_after_seconds"] = str(failure.retry_after_seconds)
    return fv.FailureEvent(
        class_name=names.get(failure.class_name, failure.class_name),
        origin=failure.origin, confidence="structured", provider="opencode_zen",
        scope="account" if failure.scope == "account" else "model",
        retryable=failure.retryable or failure.fallback, evidence=evidence,
    )
