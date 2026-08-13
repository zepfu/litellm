"""Direct Cohere failure classification for pass-through and alias adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

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

_MONTHLY_TRIAL_MARKERS: tuple[str, ...] = (
    "monthly trial",
    "trial monthly",
    "monthly limit",
    "trial limit",
    "trial quota",
    "monthly quota",
    "trial usage",
    "free trial",
)
_MODEL_UNAVAILABLE_MARKERS: tuple[str, ...] = (
    "model not found",
    "model does not exist",
    "unsupported model",
    "model is not supported",
    "model retired",
    "retired model",
    "model deprecated",
    "model is not available",
)
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "too many requests",
    "requests per minute",
    "rpm",
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


def classify_cohere_failure(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[CohereFailureClassification]:
    """Classify only direct Cohere failures, never OpenRouter-hosted Cohere."""

    provider = str(custom_llm_provider or "").strip().lower()
    if provider and provider != "cohere":
        return None
    if not is_cohere_api_url(url):
        return None

    text = _normalized_error_text(exc)
    if status_code in (401, 403):
        return CohereFailureClassification(
            name="cohere_authentication",
            failure_kind="cohere_authentication",
            failure_class="auth",
            log_error_summary="Cohere authentication failed",
        )
    if status_code == 429 and any(marker in text for marker in _MONTHLY_TRIAL_MARKERS):
        return CohereFailureClassification(
            name="cohere_monthly_trial_exhausted",
            failure_kind="cohere_monthly_trial_exhausted",
            failure_class="quota_exhausted",
            log_error_summary="Cohere monthly trial capacity is exhausted",
        )
    if status_code == 429:
        return CohereFailureClassification(
            name="cohere_rpm_rate_limit",
            failure_kind="cohere_rpm_rate_limit",
            failure_class="rate_limit",
            log_error_summary="Cohere request rate limit reached",
        )
    if status_code == 404 or any(marker in text for marker in _MODEL_UNAVAILABLE_MARKERS):
        return CohereFailureClassification(
            name="cohere_model_unavailable",
            failure_kind="cohere_model_unavailable",
            failure_class="model_unavailable",
            log_error_summary="Cohere model is unsupported or unavailable",
        )
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return CohereFailureClassification(
            name="cohere_timeout_connectivity",
            failure_kind="cohere_timeout_connectivity",
            failure_class="transient",
            log_error_summary="Cohere timeout or connectivity failure",
        )
    if status_code in (400, 422):
        return CohereFailureClassification(
            name="cohere_validation",
            failure_kind="cohere_validation",
            failure_class="provider_4xx_other",
            log_error_summary="Cohere request validation failed",
        )
    if status_code is not None and 500 <= status_code <= 599:
        failure_class = "provider_5xx"
    elif any(marker in text for marker in _RATE_LIMIT_MARKERS):
        failure_class = "rate_limit"
    else:
        failure_class = "transient"
    return CohereFailureClassification(
        name="cohere_provider_failure",
        failure_kind="cohere_provider_failure",
        failure_class=failure_class,
        log_error_summary="Cohere provider request failed",
    )


__all__ = [
    "COHERE_API_HOSTS",
    "CohereFailureClassification",
    "classify_cohere_failure",
    "is_cohere_api_url",
]
