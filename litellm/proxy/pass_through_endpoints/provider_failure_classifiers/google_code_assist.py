"""Google Code Assist known failure classifiers for pass-through (RR-056 #3)."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _GOOGLE_CODE_ASSIST_HOST_SUFFIX,
    _GOOGLE_CODE_ASSIST_PERMISSION_DENIED_STATUS,
    _GOOGLE_CODE_ASSIST_TOS_REASON,
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
    _is_google_code_assist_passthrough_target,
)


def _is_google_code_assist_tos_violation_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    code = error.get("code")
    status_text = str(error.get("status") or "").strip().upper()
    if code != status.HTTP_403_FORBIDDEN and status_text != (
        _GOOGLE_CODE_ASSIST_PERMISSION_DENIED_STATUS
    ):
        return False

    details = error.get("details")
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        reason = str(detail.get("reason") or "").strip().upper()
        domain = str(detail.get("domain") or "").strip().lower()
        if reason == _GOOGLE_CODE_ASSIST_TOS_REASON and domain.endswith(
            _GOOGLE_CODE_ASSIST_HOST_SUFFIX
        ):
            return True
    return False


def _is_known_google_code_assist_tos_violation_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_403_FORBIDDEN:
        return False

    if not _is_google_code_assist_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return False

    payload = _coerce_upstream_error_payload(detail)
    return _is_google_code_assist_tos_violation_payload(payload)
