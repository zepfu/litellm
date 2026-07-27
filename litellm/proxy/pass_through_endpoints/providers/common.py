"""Provider-shaped candidate-unavailable error vocabulary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Never, Optional

from fastapi import HTTPException

from litellm.proxy._types import ProxyException


@dataclass(frozen=True)
class Runtime:
    """Injected exception-detail extraction callbacks."""

    extract_status_code: Callable[[Any], Optional[int]]
    extract_detail: Callable[[Any], Any]
    http_exception_type: type[Exception] = HTTPException


def _raise_candidate_unavailable(
    exc: Exception,
    *,
    message: str,
    error_type: str,
    status_code: int,
) -> Never:
    proxy_exc = ProxyException(
        message=message,
        type=error_type,
        param="model",
        code=status_code,
    )
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise proxy_exc from exc


def _raise_opencode_zen_auto_agent_candidate_unavailable(
    exc: Exception,
) -> Never:
    _raise_candidate_unavailable(
        exc,
        message=(
            "OpenCode Zen auto-agent candidate requires a valid OpenCode "
            f"API-key credential: {exc}"
        ),
        error_type="rate_limit_error",
        status_code=429,
    )


def _opencode_zen_candidate_unavailable_detail(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Optional[str]:
    status_code = runtime.extract_status_code(exc)
    detail = runtime.extract_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail or exc)
    detail_text = " ".join(
        str(part)
        for part in (
            getattr(exc, "message", None),
            getattr(exc, "code", None),
            detail_text,
            str(exc),
        )
        if part is not None
    )
    normalized = detail_text.lower()
    if any(
        marker in normalized
        for marker in (
            "freeusagelimiterror",
            "free usage limit",
            "creditserror",
            "no payment method",
            "add a payment method",
            "billing",
            "payment required",
        )
    ):
        return detail_text
    if "not supported for format openai" in normalized:
        return detail_text
    if status_code in {401, 402, 403} and any(
        marker in normalized
        for marker in (
            "authentication",
            "authorization",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "api-key",
            "api key",
            "credential",
            "opencode",
        )
    ):
        return detail_text
    return None


def _antigravity_candidate_unavailable_detail(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Optional[str]:
    if not isinstance(exc, runtime.http_exception_type):
        return None
    detail = getattr(exc, "detail", None)
    if isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    else:
        detail_text = str(detail or exc)
    normalized = detail_text.lower()
    if "agy cli" in normalized and "auth refresh" in normalized:
        return detail_text
    if "antigravity oauth" in normalized or "antigravity cli" in normalized:
        return detail_text
    if "antigravity" not in normalized:
        return None
    if not any(
        marker in normalized
        for marker in (
            "auth provider",
            "authentication",
            "authorization",
            "credential",
            "credentials",
            "log in",
            "login",
            "not logged in",
            "not logged into",
            "oauth",
            "token source",
        )
    ):
        return None
    return detail_text


def _raise_antigravity_auto_agent_candidate_unavailable(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Never:
    detail = (
        _antigravity_candidate_unavailable_detail(exc, runtime=runtime)
        or str(exc)
    )
    _raise_candidate_unavailable(
        exc,
        message=(
            "Antigravity auto-agent candidate requires a valid Antigravity "
            f"OAuth credential: {detail}"
        ),
        error_type="invalid_request_error",
        status_code=502,
    )


def _is_grok_unsupported_reasoning_parameter_detail(
    normalized_detail: str,
) -> bool:
    if "grok" not in normalized_detail:
        return False
    if not any(
        marker in normalized_detail
        for marker in (
            "reasoningeffort",
            "reasoning_effort",
            "output_config.effort",
            "reasoning",
        )
    ):
        return False
    return any(
        marker in normalized_detail
        for marker in (
            "does not support parameter",
            "unsupported parameter",
            "invalid-argument",
            "invalid argument",
        )
    )


def _codex_native_openai_candidate_unavailable_detail(
    exc: Any,
    *,
    runtime: Runtime,
) -> Optional[str]:
    status_code = runtime.extract_status_code(exc)
    if status_code != 400:
        return None
    detail = runtime.extract_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    elif isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    normalized = detail_text.lower()
    if (
        "not supported when using codex with a chatgpt account"
        not in normalized
    ):
        return None
    if (
        "model is not supported" not in normalized
        and "is not supported" not in normalized
    ):
        return None
    return detail_text


def _raise_codex_native_openai_auto_agent_candidate_unavailable(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Never:
    detail = (
        _codex_native_openai_candidate_unavailable_detail(
            exc,
            runtime=runtime,
        )
        or str(exc)
    )
    _raise_candidate_unavailable(
        exc,
        message=(
            "ChatGPT/Codex native OpenAI auto-agent candidate is unavailable "
            f"for this account: {detail}"
        ),
        error_type="rate_limit_error",
        status_code=429,
    )


def _grok_native_candidate_unavailable_detail(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Optional[str]:
    status_code = runtime.extract_status_code(exc)
    detail = runtime.extract_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    elif isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    normalized = detail_text.lower()
    if _is_grok_unsupported_reasoning_parameter_detail(normalized):
        return detail_text
    if "could not decode the compaction blob" in normalized:
        return detail_text
    if (
        status_code == 403
        and "permission-denied" in normalized
        and "access to the chat endpoint is denied" in normalized
        and (
            "correct credentials" in normalized
            or "update the permissions" in normalized
        )
    ):
        return detail_text
    if (
        "xai oauth credential" not in normalized
        and "grok oidc credential" not in normalized
        and "grok native" not in normalized
    ):
        return None
    return detail_text


def _xai_oauth_candidate_unavailable_detail(
    exc: Exception,
) -> Optional[str]:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    normalized = detail_text.lower()
    if _is_grok_unsupported_reasoning_parameter_detail(normalized):
        return detail_text
    if "could not decode the compaction blob" in normalized:
        return detail_text
    if not any(
        marker in normalized
        for marker in (
            "xai oauth credential",
            "xai oauth-managed",
            "managed xai oauth",
            "litellm_xai_oauth_auth_file",
        )
    ):
        return None
    return detail_text


def _raise_xai_oauth_auto_agent_candidate_unavailable(
    exc: Exception,
) -> Never:
    detail = _xai_oauth_candidate_unavailable_detail(exc) or str(exc)
    _raise_candidate_unavailable(
        exc,
        message=(
            "xAI OAuth auto-agent candidate requires a valid managed xAI "
            f"OAuth credential: {detail}"
        ),
        error_type="rate_limit_error",
        status_code=429,
    )


def _raise_grok_native_auto_agent_candidate_unavailable(
    exc: Exception,
    *,
    runtime: Runtime,
) -> Never:
    detail = (
        _grok_native_candidate_unavailable_detail(exc, runtime=runtime)
        or str(exc)
    )
    _raise_candidate_unavailable(
        exc,
        message=(
            "Grok native auto-agent candidate requires a valid managed "
            f"xAI/Grok credential: {detail}"
        ),
        error_type="rate_limit_error",
        status_code=429,
    )


__all__ = [
    "Runtime",
    "_antigravity_candidate_unavailable_detail",
    "_codex_native_openai_candidate_unavailable_detail",
    "_grok_native_candidate_unavailable_detail",
    "_is_grok_unsupported_reasoning_parameter_detail",
    "_opencode_zen_candidate_unavailable_detail",
    "_raise_antigravity_auto_agent_candidate_unavailable",
    "_raise_codex_native_openai_auto_agent_candidate_unavailable",
    "_raise_grok_native_auto_agent_candidate_unavailable",
    "_raise_opencode_zen_auto_agent_candidate_unavailable",
    "_raise_xai_oauth_auto_agent_candidate_unavailable",
    "_xai_oauth_candidate_unavailable_detail",
]
