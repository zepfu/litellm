"""Config-driven Anthropic adapter route descriptors (RR-054 #9).

Provider-specific credential/URL resolution stays in callables supplied by
``llm_passthrough_endpoints``; this module owns the shared config shapes and
driver contracts so nine near-duplicate handlers can shrink to thin wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from typing_extensions import NotRequired, TypedDict

from .types import Payload

# Callable type aliases keep the god-file free of re-declaring the same shapes.
PrepareResponsesBody = Callable[..., Payload]
PassThroughCall = Callable[..., Awaitable[object]]
FinalizeResponses = Callable[..., Awaitable[object]]
PrepareCompletionBody = Callable[..., Payload]
CompletionCall = Callable[..., Awaitable[object]]


class ResponsesFinalizeKwargs(TypedDict):
    """Typed keyword bundle consumed by the shared Responses finalizer."""

    adapter: str
    adapter_label: str
    provider: str
    unexpected_detail: str
    skip_stream_probe_validation: bool
    use_codex_native_tools: bool
    response_builder_kwargs: NotRequired[Payload]
    stream_builder_kwargs: NotRequired[Payload]
    malformed_upstream_url: NotRequired[object]


@dataclass(frozen=True)
class AnthropicResponsesAdapterConfig:
    """Descriptor for one Anthropic→Responses pass-through adapter route."""

    adapter: str
    adapter_label: str
    provider: str
    unexpected_detail: str
    parallel_policy_log_label: str = ""
    forced_tool_choice_log_label: str = ""
    use_openai_parallel_policy: bool = True
    reject_empty_success: bool = False
    diagnostic_adapter_name: Optional[str] = None
    skip_stream_probe_validation: bool = False
    default_use_codex_native_tools: bool = False


@dataclass(frozen=True)
class AnthropicCompletionAdapterConfig:
    """Descriptor for one Anthropic→chat.completions adapter route."""

    adapter: str
    adapter_label: str
    route_family: str
    tag_prefix: str
    span_name: str
    target_endpoint_label: str
    credential_family: str
    expected_target_family: str
    custom_llm_provider: str


# Canonical configs for the Anthropic Responses family (RR-054 #9).
OPENAI_RESPONSES = AnthropicResponsesAdapterConfig(
    adapter="anthropic_openai_responses_adapter",
    adapter_label="OpenAI",
    provider="openai",
    unexpected_detail="Unexpected upstream response type from OpenAI Responses passthrough.",
    parallel_policy_log_label="OpenAI adapter",
    forced_tool_choice_log_label="OpenAI adapter",
    default_use_codex_native_tools=True,
)

XAI_OAUTH_RESPONSES = AnthropicResponsesAdapterConfig(
    adapter="anthropic_xai_oauth_responses_adapter",
    adapter_label="xAI OAuth",
    provider="xai",
    unexpected_detail="Unexpected upstream response type from xAI Responses passthrough.",
    parallel_policy_log_label="xAI OAuth responses adapter",
    forced_tool_choice_log_label="xAI OAuth adapter",
)

GROK_NATIVE_RESPONSES = AnthropicResponsesAdapterConfig(
    adapter="anthropic_grok_native_responses_adapter",
    adapter_label="Grok native",
    provider="xai",
    unexpected_detail="Unexpected upstream response type from Grok native Responses passthrough.",
    parallel_policy_log_label="Grok native responses adapter",
    forced_tool_choice_log_label="Grok native adapter",
)

OPENROUTER_RESPONSES = AnthropicResponsesAdapterConfig(
    adapter="anthropic_openrouter_responses_adapter",
    adapter_label="OpenRouter",
    provider="openrouter",
    unexpected_detail="Unexpected upstream response type from OpenRouter Responses passthrough.",
    use_openai_parallel_policy=False,
    reject_empty_success=True,
    diagnostic_adapter_name="openrouter_responses",
)

OPENCODE_ZEN_RESPONSES = AnthropicResponsesAdapterConfig(
    adapter="anthropic_opencode_zen_responses_adapter",
    adapter_label="OpenCode Zen",
    provider="opencode",
    unexpected_detail="Unexpected upstream response type from OpenCode Zen Responses passthrough.",
    use_openai_parallel_policy=False,
    reject_empty_success=True,
    diagnostic_adapter_name="opencode_zen_responses",
    skip_stream_probe_validation=True,
)

XAI_OAUTH_COMPLETION = AnthropicCompletionAdapterConfig(
    adapter="anthropic_xai_oauth_completion_adapter",
    adapter_label="xAI OAuth",
    route_family="anthropic_xai_oauth_completion_adapter",
    tag_prefix="anthropic-xai-oauth-completion-adapter",
    span_name="anthropic.xai_oauth_completion_adapter",
    target_endpoint_label="xai:/v1/chat/completions",
    credential_family="xai",
    expected_target_family="xai",
    custom_llm_provider="xai",
)

NVIDIA_COMPLETION = AnthropicCompletionAdapterConfig(
    adapter="anthropic_nvidia_completion_adapter",
    adapter_label="NVIDIA",
    route_family="anthropic_nvidia_completion_adapter",
    tag_prefix="anthropic-nvidia-completion-adapter",
    span_name="anthropic.nvidia_completion_adapter",
    target_endpoint_label="nvidia:/v1/chat/completions",
    credential_family="nvidia",
    expected_target_family="nvidia",
    custom_llm_provider="nvidia_nim",
)

OPENROUTER_COMPLETION = AnthropicCompletionAdapterConfig(
    adapter="anthropic_openrouter_completion_adapter",
    adapter_label="OpenRouter",
    route_family="anthropic_openrouter_completion_adapter",
    tag_prefix="anthropic-openrouter-completion-adapter",
    span_name="anthropic.openrouter_completion_adapter",
    target_endpoint_label="openrouter:/v1/chat/completions",
    credential_family="openrouter",
    expected_target_family="openrouter",
    custom_llm_provider="openrouter",
)

OPENCODE_ZEN_COMPLETION = AnthropicCompletionAdapterConfig(
    adapter="anthropic_opencode_zen_completion_adapter",
    adapter_label="OpenCode Zen",
    route_family="anthropic_opencode_zen_completion_adapter",
    tag_prefix="anthropic-opencode-zen-completion-adapter",
    span_name="anthropic.opencode_zen_completion_adapter",
    target_endpoint_label="opencode_zen:/v1/chat/completions",
    credential_family="opencode",
    expected_target_family="opencode",
    custom_llm_provider="openai",  # OpenCode uses OpenAI-compatible base
)

CODEX_KIMI_CHAT_COMPLETIONS = AnthropicCompletionAdapterConfig(
    adapter="codex_kimi_chat_completions_adapter",
    adapter_label="Kimi Code",
    route_family="codex_kimi_chat_completions_adapter",
    tag_prefix="codex-kimi-chat-completions-adapter",
    span_name="codex.kimi_chat_completions_adapter",
    target_endpoint_label="kimi_code:/coding/v1/chat/completions",
    credential_family="kimi_code",
    expected_target_family="kimi_code",
    custom_llm_provider="kimi_code",
)

ANTHROPIC_KIMI_CHAT_COMPLETIONS = AnthropicCompletionAdapterConfig(
    adapter="anthropic_kimi_chat_completions_adapter",
    adapter_label="Kimi Code",
    route_family="anthropic_kimi_chat_completions_adapter",
    tag_prefix="anthropic-kimi-chat-completions-adapter",
    span_name="anthropic.kimi_chat_completions_adapter",
    target_endpoint_label="kimi_code:/coding/v1/chat/completions",
    credential_family="kimi_code",
    expected_target_family="kimi_code",
    custom_llm_provider="kimi_code",
)

CODEX_ALIBABA_TOKEN_PLAN = AnthropicCompletionAdapterConfig(
    adapter="codex_alibaba_token_plan_chat_completions_adapter",
    adapter_label="Alibaba Token Plan",
    route_family="codex_alibaba_token_plan_chat_completions_adapter",
    tag_prefix="codex-alibaba-token-plan-chat-completions-adapter",
    span_name="codex.alibaba_token_plan_chat_completions_adapter",
    target_endpoint_label="alibaba_token_plan:/compatible-mode/v1/chat/completions",
    credential_family="alibaba_token_plan",
    expected_target_family="alibaba_token_plan",
    custom_llm_provider="alibaba_token_plan",
)

ANTHROPIC_ALIBABA_TOKEN_PLAN = AnthropicCompletionAdapterConfig(
    adapter="anthropic_alibaba_token_plan_chat_completions_adapter",
    adapter_label="Alibaba Token Plan",
    route_family="anthropic_alibaba_token_plan_chat_completions_adapter",
    tag_prefix="anthropic-alibaba-token-plan-chat-completions-adapter",
    span_name="anthropic.alibaba_token_plan_chat_completions_adapter",
    target_endpoint_label="alibaba_token_plan:/compatible-mode/v1/chat/completions",
    credential_family="alibaba_token_plan",
    expected_target_family="alibaba_token_plan",
    custom_llm_provider="alibaba_token_plan",
)


def responses_finalize_kwargs(
    config: AnthropicResponsesAdapterConfig,
    *,
    adapter_model: str,
    translated_request_body: Payload,
) -> ResponsesFinalizeKwargs:
    """Build finalize() kwargs from a Responses adapter config."""
    kwargs: ResponsesFinalizeKwargs = {
        "adapter": config.adapter,
        "adapter_label": config.adapter_label,
        "provider": config.provider,
        "unexpected_detail": config.unexpected_detail,
        "skip_stream_probe_validation": config.skip_stream_probe_validation,
        "use_codex_native_tools": config.default_use_codex_native_tools,
    }
    if config.reject_empty_success:
        diagnostic_name = config.diagnostic_adapter_name or config.adapter
        kwargs["response_builder_kwargs"] = {
            "reject_empty_success": True,
            "diagnostic_context": {
                "adapter": diagnostic_name,
                "adapter_model": adapter_model,
                "request_model": translated_request_body.get("model"),
                "request_stream": translated_request_body.get("stream"),
            },
        }
        kwargs["stream_builder_kwargs"] = {"reject_empty_success": True}
    return kwargs
