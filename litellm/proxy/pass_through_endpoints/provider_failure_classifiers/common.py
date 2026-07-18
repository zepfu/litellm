"""Shared helpers for pass-through provider failure classification (RR-056 #3)."""

from __future__ import annotations

import ast
import json
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import Request

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

_ANTHROPIC_INVALID_AUTHENTICATION_MARKER = "invalid authentication credentials"
_ANTHROPIC_MODEL_NOT_FOUND_PREFIX = "model:"
_ANTHROPIC_CONTEXT_OVERFLOW_MARKERS = (
    "prompt is too long",
    "context length exceeded",
    "maximum context length",
    "too many tokens",
)
_GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_CODE = "the operation was cancelled"
_GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_ERROR = "timeout expired"
_GROK_SIGNALS_AUTH_CONTEXT_ERROR_MARKERS = (
    "invalid or expired credentials",
    "x_xai_token_auth=xai-grok-cli",
    "no auth context",
)
_GROK_REPLICAS_UPDATE_NOT_OWNED_ERROR_MARKERS = ("session not found or not owned",)
_GROK_PERSONAL_TEAM_SPENDING_LIMIT_MARKER = "personal-team-blocked:spending-limit"
_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_ERROR_MARKER = "grok build usage balance exhausted"
_CHATGPT_CODEX_BLOCK_PAGE_MARKERS = (
    "unable to load site",
    "cdn-cgi/challenge-platform",
)
_CHATGPT_CODEX_INVALID_ENCRYPTED_CONTENT_ERROR_CODE = "invalid_encrypted_content"
_CHATGPT_CODEX_MODEL_NOT_SUPPORTED_FOR_ACCOUNT_MARKERS = (
    "not supported when using codex with a chatgpt account",
    "model is not supported",
)
_GOOGLE_CODE_ASSIST_HOST_SUFFIX = "cloudcode-pa.googleapis.com"
_GOOGLE_CODE_ASSIST_TOS_REASON = "TOS_VIOLATION"
_GOOGLE_CODE_ASSIST_PERMISSION_DENIED_STATUS = "PERMISSION_DENIED"


def _get_passthrough_request_url_path(request: Request) -> str:
    """Extract URL path from a FastAPI/Starlette request without circular imports."""
    request_url = getattr(request, "url", "")
    return urlparse(str(request_url)).path


def _coerce_upstream_error_payload(detail: Any) -> Optional[dict[str, Any]]:
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    elif isinstance(detail, str):
        detail_text = detail
        stripped_detail = detail_text.strip()
        if stripped_detail.startswith(("b'", 'b"')):
            try:
                literal_detail = ast.literal_eval(stripped_detail)
            except Exception:
                literal_detail = None
            if isinstance(literal_detail, bytes):
                detail_text = literal_detail.decode("utf-8", errors="replace")
            elif isinstance(literal_detail, str):
                detail_text = literal_detail
    elif isinstance(detail, dict):
        return detail
    else:
        return None

    try:
        parsed = json.loads(detail_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_passthrough_exception_detail(exc: Exception) -> Optional[str]:
    for attr_name in ("detail", "message"):
        value = getattr(exc, attr_name, None)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            try:
                return safe_dumps(value)
            except Exception:
                continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def _extract_passthrough_grok_billing_timeout_cancel_hint(
    detail: Optional[Any],
) -> Optional[str]:
    if detail is None:
        return None

    if isinstance(detail, dict):
        code = detail.get("code")
        error = detail.get("error")
        if isinstance(code, str) and code.strip():
            return code.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
        return None

    detail_text = str(detail).strip()
    if not detail_text:
        return None

    try:
        parsed_detail = json.loads(detail_text)
    except json.JSONDecodeError:
        return detail_text

    if isinstance(parsed_detail, dict):
        return _extract_passthrough_grok_billing_timeout_cancel_hint(parsed_detail)
    return detail_text


def _is_xai_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return (
        provider == "xai"
        or hostname
        in {
            "api.x.ai",
            "cli-chat-proxy.grok.com",
        }
        or hostname.endswith(".x.ai")
        or hostname.endswith(".grok.com")
    )


def _is_anthropic_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return provider == "anthropic" or hostname == "api.anthropic.com"


def _is_google_code_assist_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return provider in {"google_code_assist", "antigravity"} or hostname.endswith(
        _GOOGLE_CODE_ASSIST_HOST_SUFFIX
    )
