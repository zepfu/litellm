"""Provider-shaped candidate-unavailable error vocabulary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Never, Optional

import httpx
from fastapi import HTTPException

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.openai import (
    _is_known_openai_model_not_found_response,
)


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


def _opencode_go_candidate_unavailable_detail(
    exc: Exception,
) -> Optional[str]:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        normalized_status_code = int(status_code)
    except (TypeError, ValueError):
        return None
    if normalized_status_code != 401:
        return None

    detail = getattr(exc, "detail", None)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    elif isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    exception_text = " ".join(
        str(part)
        for part in (
            getattr(exc, "message", None),
            detail_text,
            str(exc),
        )
        if part is not None
    )
    normalized = " ".join(exception_text.lower().split())
    if "model ox-alpha-free is not supported" not in normalized:
        return None
    return exception_text


def _raise_opencode_go_auto_agent_candidate_unavailable(
    exc: Exception,
) -> Never:
    detail = _opencode_go_candidate_unavailable_detail(exc) or str(exc)
    _raise_candidate_unavailable(
        exc,
        message=(
            "OpenCode Go auto-agent candidate does not support ox-alpha-free: "
            f"{detail}"
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
    target_url: Any = None,
    custom_llm_provider: Optional[str] = None,
    provider_returned: bool = False,
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

    if provider_returned and target_url is not None:
        try:
            classifier_url = (
                target_url
                if isinstance(target_url, httpx.URL)
                else httpx.URL(str(target_url))
            )
        except (TypeError, ValueError):
            classifier_url = None
        if (
            classifier_url is not None
            and classifier_url.path.rstrip("/").lower().endswith("/responses")
            and _is_known_openai_model_not_found_response(
                url=classifier_url,
                custom_llm_provider=custom_llm_provider,
                status_code=status_code,
                exc=exc,
            )
        ):
            return detail_text

    if status_code != 400:
        return None
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
    target_url: Any = None,
    custom_llm_provider: Optional[str] = None,
    provider_returned: bool = False,
) -> Never:
    detail = (
        _codex_native_openai_candidate_unavailable_detail(
            exc,
            runtime=runtime,
            target_url=target_url,
            custom_llm_provider=custom_llm_provider,
            provider_returned=provider_returned,
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
    "_codex_native_openai_candidate_unavailable_detail",
    "_grok_native_candidate_unavailable_detail",
    "_is_grok_unsupported_reasoning_parameter_detail",
    "_opencode_go_candidate_unavailable_detail",
    "_opencode_zen_candidate_unavailable_detail",
    "_raise_codex_native_openai_auto_agent_candidate_unavailable",
    "_raise_grok_native_auto_agent_candidate_unavailable",
    "_raise_opencode_go_auto_agent_candidate_unavailable",
    "_raise_opencode_zen_auto_agent_candidate_unavailable",
    "_raise_xai_oauth_auto_agent_candidate_unavailable",
    "_xai_oauth_candidate_unavailable_detail",
]
