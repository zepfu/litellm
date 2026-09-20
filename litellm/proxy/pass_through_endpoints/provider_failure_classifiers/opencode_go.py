"""Typed OpenCode Go failure classification.

Replaces exact-message matching and synthetic HTTP-status rewrites. The
classifier preserves origin, the provider's safe status, and whether the
provider returned. Unsupported-model and account scope require known
structured ``error.code`` / ``error.type`` tokens; broad message text is
never used to infer those scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from urllib.parse import urlparse

import httpx

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)

Origin = Literal["client", "upstream"]
Scope = Literal["candidate", "account", "request_local"]
Kind = Literal[
    "local_timeout",
    "provider_timeout",
    "unsupported_model",
    "unsupported_contract",
    "auth",
    "account",
    "rate_limit",
    "transient",
    "provider_4xx_other",
    "provider_5xx",
]

OPENCODE_GO_PROVIDER = "opencode_go"
OPENCODE_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
OPENCODE_GO_API_HOSTS: frozenset[str] = frozenset({"opencode.ai"})
_OPENCODE_GO_PATH_MARKERS: tuple[str, ...] = (
    "/zen/go/",
    "/zen/go",
)

# Exact structured tokens only. Do not match these as message substrings.
_UNSUPPORTED_MODEL_CODES: frozenset[str] = frozenset(
    {
        "model_not_found",
        "unsupported_model",
        "invalid_model",
    }
)
_UNSUPPORTED_CONTRACT_CODES: frozenset[str] = frozenset(
    {
        "model_opt_in_required",
        "opt_in_required",
        "explicit_opt_in_required",
        "terms_not_accepted",
    }
)
_ACCOUNT_CODES: frozenset[str] = frozenset(
    {
        "account_deactivated",
        "account_suspended",
        "account_not_authorized",
        "insufficient_permissions",
        "model_not_supported_for_account",
    }
)
_UNSUPPORTED_MODEL_STATUSES: frozenset[int] = frozenset({400, 404})
_AUTH_STATUSES: frozenset[int] = frozenset({401, 403})
_PUBLIC_DETAIL: dict[str, str] = {
    "local_timeout": "OpenCode Go local timeout",
    "provider_timeout": "OpenCode Go provider timeout",
    "unsupported_model": "OpenCode Go model is unsupported",
    "unsupported_contract": "OpenCode Go contract is unsupported",
    "auth": "OpenCode Go authentication failed",
    "account": "OpenCode Go account restriction",
    "rate_limit": "OpenCode Go rate limit",
    "transient": "OpenCode Go transient failure",
    "provider_4xx_other": "OpenCode Go provider request failed",
    "provider_5xx": "OpenCode Go provider error",
}
_KIND_TO_ALIAS_ERROR_CLASS: dict[str, str] = {
    "local_timeout": "local_timeout",
    "provider_timeout": "upstream_timeout",
    "unsupported_model": "candidate_unavailable",
    "unsupported_contract": "candidate_unavailable",
    "auth": "provider_terminal_error",
    "account": "provider_terminal_error",
    "rate_limit": "rate_limited",
    "transient": "upstream_transient_internal",
    "provider_4xx_other": "provider_terminal_error",
    "provider_5xx": "upstream_transient_internal",
}
_KIND_TO_SCOPE: dict[str, Scope] = {
    "local_timeout": "request_local",
    "provider_timeout": "candidate",
    "unsupported_model": "candidate",
    "unsupported_contract": "candidate",
    "auth": "account",
    "account": "account",
    "rate_limit": "candidate",
    "transient": "candidate",
    "provider_4xx_other": "candidate",
    "provider_5xx": "candidate",
}
_RETRYABLE_KINDS: frozenset[str] = frozenset(
    {
        "local_timeout",
        "provider_timeout",
        "rate_limit",
        "transient",
        "provider_5xx",
    }
)


@dataclass(frozen=True)
class OpenCodeGoFailureClassification:
    """Sanitized OpenCode Go failure decision for logs and alias adapters."""

    kind: Kind
    failure_class: str
    origin: Origin
    provider_returned: bool
    status_code: Optional[int]
    scope: Scope
    retryable: bool
    public_detail: str
    structured_code: Optional[str] = None

    def to_safe_metadata(self) -> dict[str, Any]:
        """Return allowlisted fields with no provider payload, headers, or secrets."""

        return {
            "kind": self.kind,
            "failure_class": self.failure_class,
            "origin": self.origin,
            "provider_returned": self.provider_returned,
            "status_code": self.status_code,
            "scope": self.scope,
            "retryable": self.retryable,
            "public_detail": self.public_detail,
            "structured_code": self.structured_code,
        }


def is_opencode_go_url(url: Optional[Union[str, httpx.URL]]) -> bool:
    """Return whether ``url`` targets an OpenCode Go path on a known host."""

    parsed = urlparse(str(url or ""))
    hostname = str(parsed.hostname or "").lower()
    path = str(parsed.path or "").lower()
    if hostname not in OPENCODE_GO_API_HOSTS:
        return False
    return any(marker in path for marker in _OPENCODE_GO_PATH_MARKERS)


def apply_opencode_go_failure_classification(
    exc: Exception,
    classification: OpenCodeGoFailureClassification,
) -> None:
    """Stamp origin, provider-return, and safe public fields without rewriting status."""

    setattr(exc, "_aawm_provider_returned", classification.provider_returned)
    setattr(exc, "provider_returned", classification.provider_returned)
    setattr(exc, "_aawm_failure_origin", classification.origin)
    setattr(exc, "failure_origin", classification.origin)
    setattr(exc, "_aawm_opencode_go_failure", classification.to_safe_metadata())


def extract_opencode_go_status_code(exc: Exception) -> Optional[int]:
    """Return a numeric HTTP status from exception attributes, never from message text."""

    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        value = getattr(source, "status_code", None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
            if 100 <= parsed <= 599:
                return parsed
    return None


def classify_opencode_go_failure(
    *,
    exc: Exception,
    url: Optional[Union[str, httpx.URL]] = None,
    custom_llm_provider: Optional[str] = None,
    status_code: Optional[int] = None,
    attempted_provider_call: bool = False,
    provider_returned: Optional[bool] = None,
    local_timeout: bool = False,
) -> Optional[OpenCodeGoFailureClassification]:
    """Classify an OpenCode Go failure from status and known structured codes.

    ``attempted_provider_call`` is retained for callers that already resolved
    transport commitment; this classifier does not relabel provider returns as
    local preflight.
    """

    _ = attempted_provider_call
    provider = str(custom_llm_provider or "").strip().lower()
    if provider and provider != OPENCODE_GO_PROVIDER:
        return None
    if not provider and (url is None or not is_opencode_go_url(url)):
        return None

    resolved_status = status_code if isinstance(status_code, int) else None
    if resolved_status is None:
        resolved_status = extract_opencode_go_status_code(exc)
    structured_codes = _structured_error_codes(exc)
    stamped_origin = getattr(exc, "_aawm_failure_origin", None)
    stamped_provider_returned = getattr(exc, "_aawm_provider_returned", None)
    if provider_returned is None:
        if isinstance(stamped_provider_returned, bool):
            provider_returned = stamped_provider_returned
        else:
            provider_returned = resolved_status is not None
    if local_timeout or stamped_origin == "client":
        is_local_timeout = resolved_status != 408
    else:
        is_local_timeout = resolved_status != 408 and _is_local_timeout_exception(exc)

    if resolved_status == 408:
        return _classification(
            kind="provider_timeout",
            origin="upstream",
            provider_returned=True,
            status_code=408,
            structured_code=_first_code(structured_codes),
        )
    if is_local_timeout:
        return _classification(
            kind="local_timeout",
            origin="client",
            provider_returned=False,
            status_code=None,
            structured_code=None,
        )

    matched_model = _first_matching_code(structured_codes, _UNSUPPORTED_MODEL_CODES)
    if (
        matched_model is not None
        and provider_returned is True
        and resolved_status in _UNSUPPORTED_MODEL_STATUSES
    ):
        return _classification(
            kind="unsupported_model",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=matched_model,
        )

    matched_contract = _first_matching_code(
        structured_codes, _UNSUPPORTED_CONTRACT_CODES
    )
    if matched_contract is not None and provider_returned is True:
        return _classification(
            kind="unsupported_contract",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=matched_contract,
        )

    matched_account = _first_matching_code(structured_codes, _ACCOUNT_CODES)
    if matched_account is not None and provider_returned is True:
        return _classification(
            kind="account",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=matched_account,
        )

    if resolved_status in _AUTH_STATUSES:
        return _classification(
            kind="auth",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=_first_code(structured_codes),
        )
    if resolved_status == 429:
        return _classification(
            kind="rate_limit",
            origin="upstream",
            provider_returned=True,
            status_code=429,
            structured_code=_first_code(structured_codes),
        )
    if resolved_status is not None and 500 <= resolved_status <= 599:
        return _classification(
            kind="provider_5xx",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=_first_code(structured_codes),
        )
    if resolved_status is not None and 400 <= resolved_status <= 499:
        return _classification(
            kind="provider_4xx_other",
            origin="upstream",
            provider_returned=True,
            status_code=resolved_status,
            structured_code=_first_code(structured_codes),
        )
    return _classification(
        kind="transient",
        origin="upstream" if provider_returned else "client",
        provider_returned=bool(provider_returned),
        status_code=resolved_status,
        structured_code=_first_code(structured_codes),
    )


def _classification(
    *,
    kind: Kind,
    origin: Origin,
    provider_returned: bool,
    status_code: Optional[int],
    structured_code: Optional[str],
) -> OpenCodeGoFailureClassification:
    return OpenCodeGoFailureClassification(
        kind=kind,
        failure_class=_KIND_TO_ALIAS_ERROR_CLASS[kind],
        origin=origin,
        provider_returned=provider_returned,
        status_code=status_code,
        scope=_KIND_TO_SCOPE[kind],
        retryable=kind in _RETRYABLE_KINDS,
        public_detail=_PUBLIC_DETAIL[kind],
        structured_code=structured_code,
    )


def _is_local_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, httpx.TimeoutException)


def _normalize_structured_token(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value).strip().lower() or None
    if isinstance(value, str):
        token = value.strip().lower()
        return token or None
    return None


def _iter_structured_error_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    values = [value]
    nested = value.get("error")
    if isinstance(nested, dict):
        values.extend(_iter_structured_error_objects(nested))
    nested_detail = value.get("detail")
    if isinstance(nested_detail, dict):
        values.extend(_iter_structured_error_objects(nested_detail))
    return values


def _structured_error_codes(exc: Exception) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        token = _normalize_structured_token(value)
        if token is None or token in seen:
            return
        seen.add(token)
        tokens.append(token)

    for attr_name in ("code", "type"):
        _add(getattr(exc, attr_name, None))
    payload = getattr(exc, "error_payload", None)
    if not isinstance(payload, dict):
        detail = _extract_passthrough_exception_detail(exc)
        payload = _coerce_upstream_error_payload(detail)
    if isinstance(payload, dict):
        for error_object in _iter_structured_error_objects(payload):
            _add(error_object.get("code"))
            _add(error_object.get("type"))
    return tokens


def _first_code(tokens: list[str]) -> Optional[str]:
    return tokens[0] if tokens else None


def _first_matching_code(
    tokens: list[str],
    known: frozenset[str],
) -> Optional[str]:
    for token in tokens:
        if token in known:
            return token
    return None


__all__ = [
    "OPENCODE_GO_API_HOSTS",
    "OPENCODE_GO_PROVIDER",
    "OPENCODE_GO_ROUTE_FAMILY",
    "OpenCodeGoFailureClassification",
    "apply_opencode_go_failure_classification",
    "classify_opencode_go_failure",
    "extract_opencode_go_status_code",
    "is_opencode_go_url",
]
