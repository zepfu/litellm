"""Safe failure classification for managed Kimi Code provider calls.

This module only classifies provider failures for later consumers. It does not
perform retries, candidate routing, cooldowns, logging, or credential refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping, Optional

import httpx

from .model_metadata import is_managed_kimi_code_model_id


class KimiCodeFailureKind(str, Enum):
    """Normalized managed-provider failure categories."""

    REFRESH_REQUIRED_AUTH = "refresh_required_auth"
    QUOTA = "quota"
    PROVIDER_CAPACITY = "provider_capacity"
    TRANSIENT = "transient"
    MALFORMED = "malformed"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_EFFORT = "unsupported_effort"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNKNOWN = "unknown"


class KimiCodeFailureScope(str, Enum):
    """The narrowest future recovery scope supported by the evidence."""

    MANAGED_ACCOUNT = "managed_account"
    CANDIDATE = "candidate"
    TELEMETRY = "telemetry"
    NONE = "none"


class KimiCodeMetadataGate(str, Enum):
    """The `/models` metadata gate relevant to a candidate-specific failure."""

    NONE = "none"
    MODEL_ID = "model_id"
    THINK_EFFORT = "think_effort"
    CAPABILITY = "capability"


class KimiCodeManagedEndpoint(str, Enum):
    """Managed Kimi Code endpoint families with distinct failure semantics."""

    CHAT_COMPLETIONS = "chat_completions"
    USAGES = "usages"


@dataclass(frozen=True)
class KimiCodeFailureMetadata:
    """Safe, normalized metadata for a managed Kimi Code failure.

    `upstream_id` is limited to known managed model IDs. `trace_id` is copied
    only from a valid `X-Trace-Id`; no authorization, `X-Msh-*`, or payload
    fields are retained.
    """

    kind: KimiCodeFailureKind
    scope: KimiCodeFailureScope
    upstream_id: Optional[str]
    metadata_gate: KimiCodeMetadataGate
    status_code: Optional[int]
    trace_id: Optional[str]
    reset_reason: str

    @property
    def is_account_scoped(self) -> bool:
        """Return whether this result concerns the shared managed account."""

        return self.scope == KimiCodeFailureScope.MANAGED_ACCOUNT

    @property
    def is_candidate_scoped(self) -> bool:
        """Return whether this result concerns only the selected candidate."""

        return self.scope == KimiCodeFailureScope.CANDIDATE

    def to_safe_metadata(self) -> dict[str, object]:
        """Return the allowlisted representation suitable for later telemetry."""

        return {
            "kind": self.kind.value,
            "scope": self.scope.value,
            "upstream_id": self.upstream_id,
            "metadata_gate": self.metadata_gate.value,
            "status_code": self.status_code,
            "trace_id": self.trace_id,
            "reset_reason": self.reset_reason,
        }


_AUTH_MARKERS: Final[tuple[str, ...]] = (
    "authentication",
    "authorization",
    "unauthorized",
    "forbidden",
    "invalid token",
    "invalid credential",
    "credential expired",
    "token expired",
    "token is expired",
    "refresh token",
)
_QUOTA_MARKERS: Final[tuple[str, ...]] = (
    "insufficient_quota",
    "quota exceeded",
    "quota exhausted",
    "usage limit",
    "credit exhausted",
)
_CAPACITY_MARKERS: Final[tuple[str, ...]] = (
    "provider_capacity",
    "capacity exceeded",
    "server busy",
    "high demand",
    "overloaded",
    "overload",
)
_TRANSIENT_MARKERS: Final[tuple[str, ...]] = (
    "timeout",
    "temporarily unavailable",
    "temporary failure",
    "internal server error",
    "upstream error",
    "try again",
)
_MALFORMED_MARKERS: Final[tuple[str, ...]] = (
    "malformed",
    "invalid json",
    "parse error",
    "invalid_request",
    "bad request",
)
_UNSUPPORTED_MODEL_MARKERS: Final[tuple[str, ...]] = (
    "model_not_found",
    "unsupported_model",
    "unsupported model",
    "model not found",
    "invalid_model",
    "invalid model",
)
_UNSUPPORTED_EFFORT_MARKERS: Final[tuple[str, ...]] = (
    "unsupported_reasoning_effort",
    "unsupported_effort",
    "unsupported reasoning effort",
    "reasoning effort is not supported",
    "thinking effort is not supported",
)
_UNSUPPORTED_CAPABILITY_MARKERS: Final[tuple[str, ...]] = (
    "unsupported_capability",
    "unsupported capability",
    "capability is not supported",
    "does not support capability",
)
_SAFE_TRACE_CHARACTERS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "._-"
)


def classify_kimi_code_failure(
    *,
    status_code: object,
    error_code: object = None,
    message: object = None,
    upstream_id: object = None,
    endpoint: KimiCodeManagedEndpoint = KimiCodeManagedEndpoint.CHAT_COMPLETIONS,
    headers: Optional[Mapping[str, object]] = None,
) -> KimiCodeFailureMetadata:
    """Classify an OpenAI-compatible Kimi error without retaining raw details."""

    normalized_status_code = _safe_status_code(status_code)
    detail = _normalized_detail(error_code, message)
    kind, scope, metadata_gate, reset_reason = _classify_failure(
        status_code=normalized_status_code,
        detail=detail,
        endpoint=endpoint,
    )
    return KimiCodeFailureMetadata(
        kind=kind,
        scope=scope,
        upstream_id=_safe_upstream_id(upstream_id),
        metadata_gate=metadata_gate,
        status_code=normalized_status_code,
        trace_id=_safe_trace_id(headers),
        reset_reason=reset_reason,
    )


def classify_kimi_code_http_failure(
    response: httpx.Response,
    *,
    upstream_id: object = None,
    endpoint: KimiCodeManagedEndpoint = KimiCodeManagedEndpoint.CHAT_COMPLETIONS,
) -> KimiCodeFailureMetadata:
    """Classify one HTTP response while discarding its untrusted payload."""

    error_code, message = _extract_openai_compatible_error(response)
    return classify_kimi_code_failure(
        status_code=response.status_code,
        error_code=error_code,
        message=message,
        upstream_id=upstream_id,
        endpoint=endpoint,
        headers=response.headers,
    )


def _classify_failure(
    *,
    status_code: Optional[int],
    detail: str,
    endpoint: KimiCodeManagedEndpoint,
) -> tuple[
    KimiCodeFailureKind,
    KimiCodeFailureScope,
    KimiCodeMetadataGate,
    str,
]:
    if status_code == 401:
        return _result(
            KimiCodeFailureKind.REFRESH_REQUIRED_AUTH,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            reset_reason="refresh_required",
        )
    if _contains_any(detail, _UNSUPPORTED_MODEL_MARKERS) or status_code == 404:
        return _result(
            KimiCodeFailureKind.UNSUPPORTED_MODEL,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.MODEL_ID,
            "unsupported_model",
        )
    if _contains_any(detail, _UNSUPPORTED_EFFORT_MARKERS):
        return _result(
            KimiCodeFailureKind.UNSUPPORTED_EFFORT,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.THINK_EFFORT,
            "unsupported_effort",
        )
    if _contains_any(detail, _UNSUPPORTED_CAPABILITY_MARKERS):
        return _result(
            KimiCodeFailureKind.UNSUPPORTED_CAPABILITY,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.CAPABILITY,
            "unsupported_capability",
        )
    if _contains_any(detail, _QUOTA_MARKERS):
        return _result(
            KimiCodeFailureKind.QUOTA,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            reset_reason="quota_exhausted",
        )
    if status_code == 529 or _contains_any(detail, _CAPACITY_MARKERS):
        return _result(
            KimiCodeFailureKind.PROVIDER_CAPACITY,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            reset_reason="provider_capacity",
        )
    if status_code == 403 or _contains_any(detail, _AUTH_MARKERS):
        return _result(
            KimiCodeFailureKind.REFRESH_REQUIRED_AUTH,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            reset_reason="refresh_required",
        )
    if _contains_any(detail, _MALFORMED_MARKERS) or status_code in {400, 422}:
        malformed_scope = (
            KimiCodeFailureScope.TELEMETRY if endpoint == KimiCodeManagedEndpoint.USAGES else KimiCodeFailureScope.NONE
        )
        return _result(
            KimiCodeFailureKind.MALFORMED,
            malformed_scope,
            reset_reason="malformed_provider_response",
        )
    if _contains_any(detail, _TRANSIENT_MARKERS) or status_code in {408, 425, 429, 500, 502, 503, 504}:
        return _result(
            KimiCodeFailureKind.TRANSIENT,
            KimiCodeFailureScope.CANDIDATE,
            reset_reason="transient_upstream_failure",
        )
    return _result(
        KimiCodeFailureKind.UNKNOWN,
        KimiCodeFailureScope.NONE,
        reset_reason="unclassified_failure",
    )


def _result(
    kind: KimiCodeFailureKind,
    scope: KimiCodeFailureScope,
    metadata_gate: KimiCodeMetadataGate = KimiCodeMetadataGate.NONE,
    reset_reason: str = "",
) -> tuple[
    KimiCodeFailureKind,
    KimiCodeFailureScope,
    KimiCodeMetadataGate,
    str,
]:
    return kind, scope, metadata_gate, reset_reason


def _safe_status_code(value: object) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _safe_upstream_id(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    model_id = value.removeprefix("kimi_code/")
    return model_id if is_managed_kimi_code_model_id(model_id) else None


def _safe_trace_id(headers: Optional[Mapping[str, object]]) -> Optional[str]:
    if headers is None:
        return None
    for name, value in headers.items():
        if str(name).lower() != "x-trace-id" or not isinstance(value, str):
            continue
        trace_id = value.strip()
        if 1 <= len(trace_id) <= 128 and all(character in _SAFE_TRACE_CHARACTERS for character in trace_id):
            return trace_id
    return None


def _normalized_detail(error_code: object, message: object) -> str:
    return " ".join(value.lower() for value in (error_code, message) if isinstance(value, str))


def _contains_any(detail: str, markers: tuple[str, ...]) -> bool:
    return any(marker in detail for marker in markers)


def _extract_openai_compatible_error(response: httpx.Response) -> tuple[object, object]:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return "malformed_response", None
    if not isinstance(payload, Mapping):
        return "malformed_response", None

    error = payload.get("error", payload)
    if not isinstance(error, Mapping):
        return "malformed_response", None

    error_code = error.get("code", error.get("type"))
    message = error.get("message")
    if error_code is None and message is None:
        return "malformed_response", None
    return error_code, message
