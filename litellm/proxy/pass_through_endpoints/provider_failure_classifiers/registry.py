"""Registry dispatch for pass-through provider failure classifiers (RR-056 #3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import httpx
from fastapi import Request

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.anthropic import (
    _get_known_anthropic_passthrough_failure_kind,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.chatgpt_codex import (
    _get_passthrough_chatgpt_codex_model_not_supported_failure_kind,
    _is_known_chatgpt_codex_block_page_response,
    _is_known_chatgpt_codex_invalid_encrypted_content_response,
    _is_known_chatgpt_codex_model_not_supported_for_account_response,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.google_code_assist import (
    _is_known_google_code_assist_tos_violation_response,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.grok import (
    _get_passthrough_grok_billing_timeout_failure_kind,
    _get_passthrough_grok_build_usage_balance_exhausted_failure_kind,
    _get_passthrough_grok_personal_team_spending_limit_failure_kind,
    _get_passthrough_grok_replicas_update_not_owned_failure_kind,
    _get_passthrough_grok_signals_auth_context_failure_kind,
    _is_known_grok_billing_passthrough_timeout_cancel_response,
    _is_known_grok_build_usage_balance_exhausted_response,
    _is_known_grok_personal_team_spending_limit_response,
    _is_known_grok_replicas_update_not_owned_response,
    _is_known_grok_signals_auth_context_response,
)

ProviderFailureClassifier = Callable[..., Optional["PassthroughProviderFailureClassification"]]


@dataclass
class PassthroughProviderFailureClassification:
    """Result of a provider-specific failure classifier (RR-056 #3)."""

    name: str
    suppress_traceback: bool = True
    log_level: str = "warning"
    failure_kind: Optional[str] = None
    log_message: Optional[str] = None
    # Historical contract: known Grok account/auth failures skip the generic
    # post_call_failure_hook path (already classified / noisy).
    skip_post_call_failure_hook: bool = False


def _classify_grok_billing_timeout_cancel(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_grok_billing_passthrough_timeout_cancel_response(
        request=request,
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="grok_billing_timeout_cancel",
        failure_kind=_get_passthrough_grok_billing_timeout_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced known Grok billing timeout/cancel "
            "status=%s error=%s"
        ),
        skip_post_call_failure_hook=True,
    )


def _classify_grok_signals_auth_context(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_grok_signals_auth_context_response(
        request=request,
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="grok_signals_auth_context",
        failure_kind=_get_passthrough_grok_signals_auth_context_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced known Grok signals auth-context "
            "status=%s error=%s"
        ),
        skip_post_call_failure_hook=True,
    )


def _classify_grok_personal_team_spending_limit(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_grok_personal_team_spending_limit_response(
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="grok_personal_team_spending_limit",
        failure_kind=_get_passthrough_grok_personal_team_spending_limit_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced known Grok personal-team spending-limit "
            "status=%s error=%s"
        ),
        skip_post_call_failure_hook=True,
    )


def _classify_grok_build_usage_balance_exhausted(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_grok_build_usage_balance_exhausted_response(
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="grok_build_usage_balance_exhausted",
        failure_kind=_get_passthrough_grok_build_usage_balance_exhausted_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced known Grok Build usage balance "
            "exhaustion status=%s error=%s"
        ),
        skip_post_call_failure_hook=True,
    )


def _classify_grok_replicas_update_not_owned(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_grok_replicas_update_not_owned_response(
        request=request,
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="grok_replicas_update_not_owned",
        failure_kind=_get_passthrough_grok_replicas_update_not_owned_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced known Grok replicas/update not-owned "
            "status=%s error=%s"
        ),
        skip_post_call_failure_hook=True,
    )


def _classify_chatgpt_codex_block_page(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_chatgpt_codex_block_page_response(
        url=url,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="chatgpt_codex_block_page",
        failure_kind="openai_chatgpt_codex_block_page",
        log_message=(
            "Pass through endpoint surfaced ChatGPT Codex block page "
            "status=%s error=%s"
        ),
    )


def _classify_chatgpt_codex_invalid_encrypted_content(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_chatgpt_codex_invalid_encrypted_content_response(
        url=url,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="chatgpt_codex_invalid_encrypted_content",
        failure_kind="openai_chatgpt_codex_invalid_encrypted_content",
        log_message=(
            "Pass through endpoint surfaced ChatGPT Codex invalid encrypted "
            "content status=%s error=%s"
        ),
    )


def _classify_chatgpt_codex_model_not_supported(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_chatgpt_codex_model_not_supported_for_account_response(
        url=url,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="chatgpt_codex_model_not_supported",
        failure_kind=_get_passthrough_chatgpt_codex_model_not_supported_failure_kind(),
        log_message=(
            "Pass through endpoint surfaced ChatGPT Codex unsupported model for "
            "account status=%s error=%s"
        ),
    )


def _classify_google_code_assist_tos(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    if not _is_known_google_code_assist_tos_violation_response(
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    ):
        return None
    return PassthroughProviderFailureClassification(
        name="google_code_assist_tos",
        failure_kind="google_code_assist_tos_violation",
        log_message=(
            "Pass through endpoint surfaced Google Code Assist account TOS "
            "violation status=%s error=%s"
        ),
    )


def _classify_anthropic_known_failure(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[PassthroughProviderFailureClassification]:
    anthropic_kind = _get_known_anthropic_passthrough_failure_kind(
        url=url,
        custom_llm_provider=custom_llm_provider,
        status_code=status_code,
        exc=exc,
    )
    if not anthropic_kind:
        return None
    return PassthroughProviderFailureClassification(
        name="anthropic_known_failure",
        failure_kind=str(anthropic_kind),
        log_message=(
            "Pass through endpoint surfaced Anthropic provider/client "
            "failure status=%s error=%s"
        ),
    )


# Ordered data-driven registry. First matching classifier wins (short-circuit).
PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS: Sequence[ProviderFailureClassifier] = (
    _classify_grok_billing_timeout_cancel,
    _classify_grok_signals_auth_context,
    _classify_grok_personal_team_spending_limit,
    _classify_grok_build_usage_balance_exhausted,
    _classify_grok_replicas_update_not_owned,
    _classify_chatgpt_codex_block_page,
    _classify_chatgpt_codex_invalid_encrypted_content,
    _classify_chatgpt_codex_model_not_supported,
    _classify_google_code_assist_tos,
    _classify_anthropic_known_failure,
)


def _run_passthrough_provider_failure_classifiers(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> list[PassthroughProviderFailureClassification]:
    """
    Data-driven first-match dispatch over the ordered classifier registry.

    Returns a 0-or-1 element list so callers can keep a simple list API while
    still short-circuiting after the first match.
    """
    for classifier in PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS:
        result = classifier(
            request=request,
            url=url,
            custom_llm_provider=custom_llm_provider,
            status_code=status_code,
            exc=exc,
        )
        if result is not None:
            return [result]
    return []
