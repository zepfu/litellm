"""Direct Cohere failure classification for pass-through and alias adapters."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

from litellm.llms.cohere.cancellation import COHERE_CANCELLATION_FAILURE_CLASS
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)

COHERE_API_HOSTS: frozenset[str] = frozenset(
    {
        "api.cohere.com",
        "api.cohere.ai",
    }
)
_COHERE_CHAT_V2_PATH = "/v2/chat"
_COHERE_CODEX_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"

# The exhausted noun must be the monthly allowance. The reverse clause may
# include only a determiner, so an RPM exhaustion verb cannot reach a separate
# monthly quota, capacity, or limit across an arbitrary word span.
_MONTHLY_ALLOWANCE_EXHAUSTION_RE = re.compile(
    r"monthly(?:\s+trial)?\s+(?:quota|capacity|limit)"
    r"(?:\s+(?:is|was|has|been|the|your)){0,4}"
    r"\s+(?:exhausted|exhaustion|exceeded|reached|depleted)"
    r"|"
    r"(?:exhausted|exhaustion|exceeded|reached|depleted)"
    r"(?:\s+(?:your|the|our|their|its|my|his|her|a|an|this|that))?"
    r"\s+monthly(?:\s+trial)?\s+(?:quota|capacity|limit)"
)
_COHERE_MODEL_TOKEN = r"""['"]?(?P<model>[^\s,'";]+)['"]?"""
_COHERE_MODEL_UNAVAILABLE_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"""\bmodel\s+{_COHERE_MODEL_TOKEN}\s+"""
        r"(?:not found|does not exist|is not available|is unsupported|"
        r"is not supported by the generate api|"
        r"is not supported(?!\s+by\b)|"
        r"retired|is retired|has been retired|deprecated|is deprecated|"
        r"has been deprecated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"""\bfinetuned model(?:\s+with name)?\s+{_COHERE_MODEL_TOKEN}\s+"""
        r"(?:not found|is not ready for serving)\b",
        re.IGNORECASE,
    ),
)
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "too many requests",
    "requests per minute",
    "rpm",
)
# Declared retryable upstream statuses. 408/504 are timeout statuses. The
# remaining codes are the shared transient set plus declared 520. Other 5xx,
# including 501, stay terminal. HTTP 499 is intentionally absent.
_COHERE_TIMEOUT_STATUS_CODES: frozenset[int] = frozenset({408, 504})
_COHERE_TRANSIENT_RETRY_STATUS_CODES: frozenset[int] = frozenset(
    {500, 502, 503, 520, 529}
)
_COHERE_UNSUPPORTED_OPERATION_MARKERS: tuple[str, ...] = (
    "unsupported operation",
    "unsupported_operation",
    "not implemented",
    "not_implemented",
    "operation not supported",
    "operation is not supported",
)
_COHERE_COOLDOWN_SCOPE_DECISIONS = frozenset({"credential", "candidate", "none"})
_COHERE_CREDENTIAL_SCOPE_FAILURES = frozenset(
    {
        "cohere_authentication",
        "cohere_billing_exhausted",
        "cohere_monthly_trial_exhausted",
    }
)
_COHERE_UNSCOPED_FAILURES = frozenset(
    {
        "cohere_validation",
        "cohere_cancellation",
    }
)


@dataclass(frozen=True)
class CohereFailureClassification:
    """Sanitized Cohere failure decision shared by logs and alias adapters."""

    name: str
    failure_kind: str
    failure_class: str
    cooldown_scope: str = "candidate"
    advance_fresh_candidate: bool = True
    suppress_traceback: bool = True
    log_error_summary: Optional[str] = None


def is_cohere_api_url(url: Optional[Union[str, httpx.URL]]) -> bool:
    """Return whether ``url`` targets an exact Cohere-owned API hostname."""

    hostname = str(urlparse(str(url or "")).hostname or "").lower()
    return hostname in COHERE_API_HOSTS


def _is_exact_cohere_chat_v2_url(url: Optional[Union[str, httpx.URL]]) -> bool:
    try:
        parsed = urlparse(str(url or ""))
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme.lower() == "https"
        and str(parsed.hostname or "").lower() in COHERE_API_HOSTS
        and parsed.path == _COHERE_CHAT_V2_PATH
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
    )


def _iter_error_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ("message", "detail", "error", "type", "code"):
            if key in value:
                values.extend(_iter_error_text_values(value[key]))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_iter_error_text_values(item))
        return values
    return []


def _normalized_error_text(exc: Exception) -> str:
    detail = _extract_passthrough_exception_detail(exc)
    payload = _coerce_upstream_error_payload(detail)
    values = _iter_error_text_values(payload)
    if not values and detail is not None:
        values = [str(detail)]
    if not values:
        values = [str(exc)]
    return " ".join(values).strip().lower()


def _iter_structured_error_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []

    values = [value]
    for key in ("error", "detail"):
        nested = value.get(key)
        if isinstance(nested, dict):
            values.extend(_iter_structured_error_objects(nested))
    return values


def _normalize_cohere_model_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip("\"'`").rstrip(".,;").casefold()


def _cohere_model_identity_variants(
    selected_upstream_model: Optional[str],
) -> frozenset[str]:
    normalized = _normalize_cohere_model_name(selected_upstream_model)
    if not normalized:
        return frozenset()
    if normalized.startswith("cohere/"):
        return frozenset({normalized, normalized[len("cohere/") :]})
    return frozenset({normalized, f"cohere/{normalized}"})


def _is_model_bound_cohere_error_message(
    value: Any,
    *,
    selected_model_variants: frozenset[str],
) -> bool:
    if not isinstance(value, str):
        return False

    normalized = " ".join(
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .split()
    ).lower()
    for pattern in _COHERE_MODEL_UNAVAILABLE_MESSAGE_PATTERNS:
        match = pattern.search(normalized)
        if (
            match is not None
            and _normalize_cohere_model_name(match.group("model"))
            in selected_model_variants
        ):
            return True
    return False


def _has_structured_model_unavailable_evidence(
    *,
    url: Optional[Union[str, httpx.URL]],
    custom_llm_provider: Optional[str],
    route_family: Optional[str],
    selected_upstream_model: Optional[str],
    status_code: Optional[int],
    attempted_provider_call: bool,
    provider_returned: bool,
    exc: Exception,
) -> bool:
    if (
        not attempted_provider_call
        or provider_returned is not True
        or getattr(exc, "_aawm_provider_returned", False) is not True
        or str(custom_llm_provider or "").strip().lower() != "cohere"
        or route_family != _COHERE_CODEX_ROUTE_FAMILY
        or not _is_exact_cohere_chat_v2_url(url)
        or status_code not in (400, 404)
    ):
        return False

    selected_model_variants = _cohere_model_identity_variants(
        selected_upstream_model
    )
    if not selected_model_variants:
        return False

    detail = _extract_passthrough_exception_detail(exc)
    payload = _coerce_upstream_error_payload(detail)
    if not isinstance(payload, dict):
        return False

    for error_object in _iter_structured_error_objects(payload):
        for key in ("message", "detail", "error"):
            if _is_model_bound_cohere_error_message(
                error_object.get(key),
                selected_model_variants=selected_model_variants,
            ):
                return True
    return False


def _is_cohere_unsupported_operation(text: str) -> bool:
    return any(marker in text for marker in _COHERE_UNSUPPORTED_OPERATION_MARKERS)


def _classify_cohere_retryable_status(
    *,
    status_code: Optional[int],
    text: str,
) -> Optional[CohereFailureClassification]:
    """Map declared timeout and transient statuses onto the shared retry class.

    Unsupported-operation responses stay unclassified here so the caller can
    retain them as terminal. Transport exceptions are classified earlier.
    """

    if status_code in _COHERE_TIMEOUT_STATUS_CODES:
        if _is_cohere_unsupported_operation(text):
            return None
        return CohereFailureClassification(
            name="cohere_timeout_status",
            failure_kind="cohere_timeout_status",
            failure_class="transient",
            log_error_summary="Cohere timeout status",
        )
    if status_code in _COHERE_TRANSIENT_RETRY_STATUS_CODES:
        if _is_cohere_unsupported_operation(text):
            return None
        return CohereFailureClassification(
            name="cohere_transient_upstream",
            failure_kind="cohere_transient_upstream",
            failure_class="transient",
            log_error_summary="Cohere transient upstream failure",
        )
    return None


def cohere_cooldown_scope_decision(
    failure_name: str,
    *,
    explicit_scope: Optional[str] = None,
) -> str:
    """Resolve one explicit Cohere cooldown scope.

    ``explicit_scope`` may be ``credential``, ``candidate``, or ``none``.
    Any other explicit value is ignored and the failure name decides:
    authentication, billing, and monthly quota are credential-wide; validation
    and cancellation have no cooldown scope; remaining Cohere failures stay
    on the single candidate.
    """

    if (
        isinstance(explicit_scope, str)
        and explicit_scope in _COHERE_COOLDOWN_SCOPE_DECISIONS
    ):
        return explicit_scope
    if failure_name in _COHERE_CREDENTIAL_SCOPE_FAILURES:
        return "credential"
    if failure_name in _COHERE_UNSCOPED_FAILURES:
        return "none"
    return "candidate"


def _is_cohere_cancellation(exc: Exception, status_code: Optional[int]) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return True
    if type(exc).__name__ == "CancelledError":
        return True
    return status_code == 499


def _cohere_classification(
    *,
    name: str,
    failure_class: str,
    log_error_summary: str,
    explicit_cooldown_scope: Optional[str] = None,
) -> CohereFailureClassification:
    return CohereFailureClassification(
        name=name,
        failure_kind=name,
        failure_class=failure_class,
        cooldown_scope=cohere_cooldown_scope_decision(
            name,
            explicit_scope=explicit_cooldown_scope,
        ),
        log_error_summary=log_error_summary,
    )


def _cohere_text_has_monthly_quota_exhaustion(text: str) -> bool:
    """Credential scope requires exhaustion of the monthly quota or capacity.

    A monthly-trial plan name is not that evidence. Exhausting an explicitly
    per-minute quota, including on the monthly trial plan or beside a separate
    monthly-usage figure, stays candidate-scoped. "monthly trial quota
    exhausted" and other monthly-quota or monthly-capacity exhaustion stay
    credential-scoped.
    """

    if "monthly" not in text:
        return False
    return _MONTHLY_ALLOWANCE_EXHAUSTION_RE.search(text) is not None


def classify_cohere_failure(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
    attempted_provider_call: bool = False,
    provider_returned: bool = False,
    route_family: Optional[str] = None,
    selected_upstream_model: Optional[str] = None,
    explicit_cooldown_scope: Optional[str] = None,
) -> Optional[CohereFailureClassification]:
    """Classify only direct Cohere failures, never OpenRouter-hosted Cohere.

    Scope is one explicit decision: ``credential`` for authentication, billing,
    and monthly quota exhaustion; ``candidate`` for model-scoped failures such
    as RPM; ``none`` for validation and cancellation. A trial mention is not
    monthly evidence. ``explicit_cooldown_scope`` may select one of those three
    decisions directly.
    """

    provider = str(custom_llm_provider or "").strip().lower()
    if provider and provider != "cohere":
        return None
    if not is_cohere_api_url(url):
        return None

    if (
        _is_cohere_cancellation(exc, status_code)
        or isinstance(exc, GeneratorExit)
        or getattr(exc, "_aawm_cohere_cancellation", False) is True
    ):
        return CohereFailureClassification(
            name="cohere_cancellation",
            failure_kind="cohere_cancellation",
            failure_class=COHERE_CANCELLATION_FAILURE_CLASS,
            cooldown_scope=cohere_cooldown_scope_decision(
                "cohere_cancellation",
                explicit_scope=explicit_cooldown_scope,
            ),
            advance_fresh_candidate=False,
            log_error_summary="Cohere request cancelled",
        )
    text = _normalized_error_text(exc)
    # 498 is Cohere Invalid Token: credential-wide, same as 401 and 403.
    if status_code in (401, 403, 498):
        return _cohere_classification(
            name="cohere_authentication",
            failure_class="auth",
            log_error_summary="Cohere authentication failed",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if status_code == 402:
        return _cohere_classification(
            name="cohere_billing_exhausted",
            failure_class="quota_exhausted",
            log_error_summary="Cohere billing capacity is exhausted",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if status_code == 429 and _cohere_text_has_monthly_quota_exhaustion(text):
        return _cohere_classification(
            name="cohere_monthly_trial_exhausted",
            failure_class="quota_exhausted",
            log_error_summary="Cohere monthly trial capacity is exhausted",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if status_code == 429:
        return _cohere_classification(
            name="cohere_rpm_rate_limit",
            failure_class="rate_limit",
            log_error_summary="Cohere request rate limit reached",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return _cohere_classification(
            name="cohere_timeout_connectivity",
            failure_class="transient",
            log_error_summary="Cohere timeout or connectivity failure",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if _has_structured_model_unavailable_evidence(
        url=url,
        custom_llm_provider=provider,
        route_family=route_family,
        selected_upstream_model=selected_upstream_model,
        status_code=status_code,
        attempted_provider_call=attempted_provider_call,
        provider_returned=provider_returned,
        exc=exc,
    ):
        return _cohere_classification(
            name="cohere_model_unavailable",
            failure_class="model_unavailable",
            log_error_summary="Cohere model is unsupported or unavailable",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if status_code in (400, 422):
        return _cohere_classification(
            name="cohere_validation",
            failure_class="provider_4xx_other",
            log_error_summary="Cohere request validation failed",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    if status_code == 404:
        return _cohere_classification(
            name="cohere_provider_failure",
            failure_class="provider_4xx_other",
            log_error_summary="Cohere provider request failed",
            explicit_cooldown_scope=explicit_cooldown_scope,
        )
    retryable_status = _classify_cohere_retryable_status(
        status_code=status_code,
        text=text,
    )
    if retryable_status is not None:
        return retryable_status
    if status_code is not None and 500 <= status_code <= 599:
        failure_class = "provider_5xx"
    elif any(marker in text for marker in _RATE_LIMIT_MARKERS):
        failure_class = "rate_limit"
    else:
        failure_class = "transient"
    return _cohere_classification(
        name="cohere_provider_failure",
        failure_class=failure_class,
        log_error_summary="Cohere provider request failed",
        explicit_cooldown_scope=explicit_cooldown_scope,
    )


__all__ = [
    "COHERE_API_HOSTS",
    "CohereFailureClassification",
    "classify_cohere_failure",
    "cohere_cooldown_scope_decision",
    "is_cohere_api_url",
]
