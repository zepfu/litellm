"""Grok / xAI known failure classifiers for pass-through (RR-056 #3)."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import Request, status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_CODE,
    _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_ERROR,
    _GROK_BUILD_USAGE_BALANCE_EXHAUSTED_ERROR_MARKER,
    _GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER,
    _GROK_REPLICAS_UPDATE_NOT_OWNED_ERROR_MARKERS,
    _GROK_SIGNALS_AUTH_CONTEXT_ERROR_MARKERS,
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
    _extract_passthrough_grok_billing_timeout_cancel_hint,
    _get_passthrough_request_url_path,
    _is_xai_passthrough_target,
)


def _is_grok_signals_path(path: str) -> bool:
    normalized_path = path.rstrip("/")
    return (
        normalized_path.startswith("/grok/v1/sessions/")
        and normalized_path.endswith("/signals")
    ) or (
        normalized_path.startswith("/v1/sessions/")
        and normalized_path.endswith("/signals")
    )


def _is_grok_replicas_update_path(path: str) -> bool:
    normalized_path = path.rstrip("/")
    return (
        normalized_path.startswith("/grok/v1/sessions/")
        and normalized_path.endswith("/replicas/update")
    ) or (
        normalized_path.startswith("/v1/sessions/")
        and normalized_path.endswith("/replicas/update")
    )


def _is_known_grok_billing_passthrough_timeout_cancel_response(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_400_BAD_REQUEST:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    request_path = _get_passthrough_request_url_path(request)
    upstream_path = urlparse(str(url or "")).path
    if not (
        request_path.rstrip("/").endswith("/grok/v1/billing")
        or upstream_path.rstrip("/").endswith("/v1/billing")
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    error_hint = _extract_passthrough_grok_billing_timeout_cancel_hint(detail)
    if not error_hint:
        return False

    normalized_hint = error_hint.strip().lower()
    return (
        _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_CODE in normalized_hint
        or _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_ERROR in normalized_hint
    )


def _is_known_grok_replicas_update_not_owned_response(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_404_NOT_FOUND:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    request_path = _get_passthrough_request_url_path(request)
    upstream_path = urlparse(str(url or "")).path
    if not (
        _is_grok_replicas_update_path(request_path)
        or _is_grok_replicas_update_path(upstream_path)
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).strip().lower()
    return all(
        marker in normalized_detail
        for marker in _GROK_REPLICAS_UPDATE_NOT_OWNED_ERROR_MARKERS
    )


def _get_passthrough_grok_replicas_update_not_owned_failure_kind() -> str:
    return "degraded_grok_replicas_update_not_owned"


def _is_known_grok_personal_team_spending_limit_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_403_FORBIDDEN:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).strip().lower()
    if _GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER not in normalized_detail:
        return False

    payload = _coerce_upstream_error_payload(detail)
    if isinstance(payload, dict):
        error = payload.get("error")
        if (
            isinstance(error, str)
            and _GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER in error.lower()
        ):
            return True
        code = payload.get("code")
        if (
            isinstance(code, str)
            and _GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER in code.lower()
        ):
            return True

    return _GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER in normalized_detail


def _get_passthrough_grok_personal_team_spending_limit_failure_kind() -> str:
    return "upstream_grok_account_quota_exhaustion"


def _is_known_grok_build_usage_balance_exhausted_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_402_PAYMENT_REQUIRED:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).strip().lower()
    payload = _coerce_upstream_error_payload(detail)
    if isinstance(payload, dict):
        error = payload.get("error")
        if (
            isinstance(error, str)
            and error.strip().lower()
            == _GROK_BUILD_USAGE_BALANCE_EXHAUSTED_ERROR_MARKER
        ):
            return True

    return normalized_detail == _GROK_BUILD_USAGE_BALANCE_EXHAUSTED_ERROR_MARKER


def _get_passthrough_grok_build_usage_balance_exhausted_failure_kind() -> str:
    return "upstream_grok_account_quota_exhaustion"


def _is_known_grok_signals_auth_context_response(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_401_UNAUTHORIZED:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    request_path = _get_passthrough_request_url_path(request)
    upstream_path = urlparse(str(url or "")).path
    if not (
        _is_grok_signals_path(request_path) or _is_grok_signals_path(upstream_path)
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).strip().lower()
    return all(
        marker in normalized_detail
        for marker in _GROK_SIGNALS_AUTH_CONTEXT_ERROR_MARKERS
    )


def _get_passthrough_grok_billing_timeout_failure_kind() -> str:
    return "degraded_grok_billing_timeout"


def _get_passthrough_grok_signals_auth_context_failure_kind() -> str:
    return "degraded_grok_signals_auth_context"
