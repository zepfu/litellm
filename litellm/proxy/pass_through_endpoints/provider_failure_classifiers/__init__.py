"""Provider-specific pass-through failure classifiers (RR-056 #3).

Classifiers implement known vendor failure kinds and are invoked via
``run_passthrough_provider_failure_classifiers`` / the re-exported
``_run_passthrough_provider_failure_classifiers`` entry point from the shared
request engine. New vendor quirks should land in a provider module under this
package and be registered in ``registry.py``.
"""

from __future__ import annotations

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.anthropic import (
    _get_known_anthropic_passthrough_failure_kind,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.chatgpt_codex import (
    _get_passthrough_chatgpt_codex_model_not_supported_failure_kind,
    _is_known_chatgpt_codex_block_page_response,
    _is_known_chatgpt_codex_invalid_encrypted_content_response,
    _is_known_chatgpt_codex_model_not_supported_for_account_response,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
    _is_anthropic_passthrough_target,
    _is_google_code_assist_passthrough_target,
    _is_xai_passthrough_target,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.google_code_assist import (
    _is_known_google_code_assist_tos_violation_response,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.openai import (
    _get_openai_model_not_found_error_summary,
    _get_openai_model_not_found_failure_kind,
    _is_known_openai_model_not_found_response,
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
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.registry import (
    PassthroughProviderFailureClassification,
    _run_passthrough_provider_failure_classifiers,
)

# Public alias matching the finding's registry interface name.
run_passthrough_provider_failure_classifiers = (
    _run_passthrough_provider_failure_classifiers
)

__all__ = [
    "PassthroughProviderFailureClassification",
    "run_passthrough_provider_failure_classifiers",
    "_run_passthrough_provider_failure_classifiers",
    "_coerce_upstream_error_payload",
    "_extract_passthrough_exception_detail",
    "_is_xai_passthrough_target",
    "_is_anthropic_passthrough_target",
    "_is_google_code_assist_passthrough_target",
    "_is_known_grok_billing_passthrough_timeout_cancel_response",
    "_is_known_grok_signals_auth_context_response",
    "_is_known_grok_personal_team_spending_limit_response",
    "_is_known_grok_build_usage_balance_exhausted_response",
    "_is_known_grok_replicas_update_not_owned_response",
    "_get_passthrough_grok_billing_timeout_failure_kind",
    "_get_passthrough_grok_signals_auth_context_failure_kind",
    "_get_passthrough_grok_personal_team_spending_limit_failure_kind",
    "_get_passthrough_grok_build_usage_balance_exhausted_failure_kind",
    "_get_passthrough_grok_replicas_update_not_owned_failure_kind",
    "_is_known_chatgpt_codex_block_page_response",
    "_is_known_chatgpt_codex_invalid_encrypted_content_response",
    "_is_known_chatgpt_codex_model_not_supported_for_account_response",
    "_get_passthrough_chatgpt_codex_model_not_supported_failure_kind",
    "_is_known_google_code_assist_tos_violation_response",
    "_is_known_openai_model_not_found_response",
    "_get_openai_model_not_found_error_summary",
    "_get_openai_model_not_found_failure_kind",
    "_get_known_anthropic_passthrough_failure_kind",
]
