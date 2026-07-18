"""Shared contracts for provider-owned Anthropic adapter preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from typing_extensions import TypeGuard

import litellm
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import adapter_config
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
)


@dataclass(frozen=True)
class ShapingRuntime:
    """Injected low-level transforms used by shared provider shaping."""

    normalize_function_tool_schemas: Callable[..., Any]
    add_native_tool_metadata: Callable[..., Any]
    apply_tool_description_patches: Callable[..., Any]
    merge_metadata: Callable[..., Any]
    add_route_family_metadata: Callable[..., Any]
    build_span: Callable[..., Any]
    apply_openai_parallel_policy: Callable[..., Any]
    apply_forced_responses_tool_choice: Callable[..., Any]
    apply_forced_completion_tool_choice: Callable[..., Any]
    log_debug: Callable[..., Any]


def _build_anthropic_messages_request(
    request_body: Payload,
    *,
    adapter_model: str,
) -> AnthropicMessagesRequest:
    raw_messages = request_body.get("messages")
    messages = (
        [item for item in raw_messages if isinstance(item, dict)]
        if isinstance(raw_messages, list)
        else []
    )

    anthropic_request: AnthropicMessagesRequest = {
        "model": adapter_model,
        "messages": messages,
    }
    max_tokens = request_body.get("max_tokens")
    if isinstance(max_tokens, int):
        anthropic_request["max_tokens"] = max_tokens
    context_management = request_body.get("context_management")
    if isinstance(context_management, dict):
        anthropic_request["context_management"] = context_management
    mcp_servers = request_body.get("mcp_servers")
    if isinstance(mcp_servers, list):
        anthropic_request["mcp_servers"] = mcp_servers
    metadata = request_body.get("metadata")
    if isinstance(metadata, dict):
        anthropic_request["metadata"] = metadata
    output_config = request_body.get("output_config")
    if _is_anthropic_output_config(output_config):
        anthropic_request["output_config"] = output_config
    output_format = request_body.get("output_format")
    if _is_anthropic_output_schema(output_format):
        anthropic_request["output_format"] = output_format
    stop_sequences = request_body.get("stop_sequences")
    if isinstance(stop_sequences, list):
        anthropic_request["stop_sequences"] = stop_sequences
    system = request_body.get("system")
    if isinstance(system, (str, list)):
        anthropic_request["system"] = system
    temperature = request_body.get("temperature")
    if isinstance(temperature, (int, float)):
        anthropic_request["temperature"] = float(temperature)
    thinking = request_body.get("thinking")
    if isinstance(thinking, dict):
        anthropic_request["thinking"] = thinking
    tool_choice = request_body.get("tool_choice")
    if isinstance(tool_choice, dict):
        anthropic_request["tool_choice"] = tool_choice
    tools = request_body.get("tools")
    if isinstance(tools, list):
        anthropic_request["tools"] = tools
    top_p = request_body.get("top_p")
    if isinstance(top_p, (int, float)):
        anthropic_request["top_p"] = float(top_p)
    return anthropic_request


def _is_anthropic_output_config(value: object) -> TypeGuard[AnthropicOutputConfig]:
    return isinstance(value, dict)


def _is_anthropic_output_schema(value: object) -> TypeGuard[AnthropicOutputSchema]:
    return isinstance(value, dict)


def build_responses_request_body(
    runtime: ShapingRuntime,
    request_body: Payload,
    *,
    adapter_model: str,
    route_family: str = "anthropic_openai_responses_adapter",
    tag_prefix: str = "anthropic-openai-responses-adapter",
    span_name: str = "anthropic.openai_responses_adapter",
    target_endpoint: str = "/v1/responses",
    use_chatgpt_codex_defaults: bool = False,
) -> Payload:
    """Translate Anthropic Messages input into the shared Responses shape."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
        derive_prompt_cache_key,
        normalize_reasoning_effort_for_provider,
        request_contains_cache_control,
    )
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    anthropic_request = _build_anthropic_messages_request(
        request_body,
        adapter_model=adapter_model,
    )
    translated_body = LiteLLMAnthropicToResponsesAPIAdapter().translate_request(
        anthropic_request,
        custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        use_codex_native_tools=use_chatgpt_codex_defaults,
    )
    runtime.normalize_function_tool_schemas(translated_body)
    if request_body.get("stream") is True:
        translated_body["stream"] = True
    if use_chatgpt_codex_defaults:
        translated_body.pop("user", None)
        instructions = translated_body.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            translated_body["instructions"] = "You are a helpful assistant."
        translated_body.setdefault("store", False)
        translated_body["stream"] = True
        include = list(translated_body.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        translated_body["include"] = include
        for unsupported_field in ("max_output_tokens", "temperature", "top_p"):
            translated_body.pop(unsupported_field, None)

    normalized_effort = normalize_reasoning_effort_for_provider(
        thinking=request_body.get("thinking"),
        output_config=request_body.get("output_config"),
        reasoning_effort=request_body.get("reasoning_effort"),
        model=adapter_model,
        custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        native_provider="openai",
        native_field="reasoning.effort",
    )
    adapter_tags: list[str] = []
    adapter_extra_fields: Payload = {}
    runtime.add_native_tool_metadata(
        adapter_tags,
        adapter_extra_fields,
        enabled=use_chatgpt_codex_defaults,
    )
    if normalized_effort is not None:
        adapter_tags.extend(normalized_effort.tags())
        adapter_extra_fields.update(normalized_effort.metadata())

    cache_requested = request_contains_cache_control(request_body)
    if not cache_requested:
        translated_body.pop("prompt_cache_key", None)
    else:
        prompt_cache_key = request_body.get("prompt_cache_key")
        if not isinstance(prompt_cache_key, str) or not prompt_cache_key.strip():
            prompt_cache_key = derive_prompt_cache_key(request_body)
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            if len(prompt_cache_key) > 64:
                prompt_cache_key = derive_prompt_cache_key(
                    {"prompt_cache_key": prompt_cache_key},
                    prefix="anthropic-cache-key",
                )
            translated_body["prompt_cache_key"] = prompt_cache_key
            adapter_extra_fields["openai_prompt_cache_key_present"] = True
            adapter_extra_fields["anthropic_adapter_cache_control_present"] = True

    existing_metadata = request_body.get("litellm_metadata")
    if isinstance(existing_metadata, dict):
        translated_body["litellm_metadata"] = {
            **existing_metadata,
            **dict(translated_body.get("litellm_metadata") or {}),
        }
    translated_body, _patch_events = runtime.apply_tool_description_patches(
        translated_body
    )
    return runtime.merge_metadata(
        runtime.add_route_family_metadata(translated_body, route_family),
        tags_to_add=[
            tag_prefix,
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint}",
            *adapter_tags,
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": request_body.get("model"),
            "anthropic_adapter_target_endpoint": target_endpoint,
            **adapter_extra_fields,
            "langfuse_spans": [
                runtime.build_span(
                    name=span_name,
                    metadata={
                        "requested_model": request_body.get("model"),
                        "adapter_model": adapter_model,
                        "stream": bool(request_body.get("stream")),
                    },
                )
            ],
        },
    )


def prepare_completion_request_body(
    runtime: ShapingRuntime,
    prepared_request_body: Payload,
    *,
    adapter_model: str,
    route_family: str,
    tag_prefix: str,
    span_name: str,
    target_endpoint_label: str,
    span_metadata_extra: Optional[Payload] = None,
) -> Payload:
    """Apply shared completion metadata and forced-tool policy."""
    requested_model = prepared_request_body.get("model")
    span_metadata = {
        "requested_model": requested_model,
        "adapter_model": adapter_model,
        "stream": bool(prepared_request_body.get("stream")),
    }
    if span_metadata_extra:
        span_metadata.update(span_metadata_extra)
    prepared_request_body = runtime.merge_metadata(
        runtime.add_route_family_metadata(prepared_request_body, route_family),
        tags_to_add=[
            tag_prefix,
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint_label}",
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_target_endpoint": target_endpoint_label,
            "langfuse_spans": [
                runtime.build_span(name=span_name, metadata=span_metadata)
            ],
        },
    )
    forced_changes = runtime.apply_forced_completion_tool_choice(
        prepared_request_body
    )
    if forced_changes:
        prepared_request_body = runtime.merge_metadata(
            prepared_request_body,
            extra_fields=forced_changes,
        )
    return prepared_request_body


def apply_responses_policies(
    runtime: ShapingRuntime,
    prepared_request_body: Payload,
    translated_request_body: Payload,
    *,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> Payload:
    """Apply the config-selected Responses policy path."""
    if not config.use_openai_parallel_policy:
        translated_request_body, _changes = (
            runtime.apply_forced_responses_tool_choice(
                prepared_request_body,
                translated_request_body,
            )
        )
        return translated_request_body
    translated_request_body, parallel_changes = runtime.apply_openai_parallel_policy(
        translated_request_body
    )
    if parallel_changes:
        runtime.log_debug(
            "Applied %s parallel instruction policy; tools=%s original_chars=%s "
            "rewritten_chars=%s",
            config.parallel_policy_log_label or config.adapter_label,
            parallel_changes.get(
                "openai_adapter_parallel_instruction_tool_names"
            ),
            parallel_changes.get(
                "openai_adapter_parallel_instruction_original_chars"
            ),
            parallel_changes.get(
                "openai_adapter_parallel_instruction_rewritten_chars"
            ),
        )
    translated_request_body, forced_changes = (
        runtime.apply_forced_responses_tool_choice(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_changes:
        runtime.log_debug(
            "Applied %s explicit Bash tool choice: %s",
            config.forced_tool_choice_log_label or config.adapter_label,
            forced_changes.get("forced_explicit_bash_tool_choice"),
        )
    return translated_request_body


def reject_raw_mcp_tools(
    translated_request_body: Payload,
    *,
    contains_mcp_tools: Callable[[Payload], bool],
) -> None:
    """Reject raw Responses MCP declarations unsupported by the adapters."""
    if not contains_mcp_tools(translated_request_body):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Anthropic adapter does not currently support raw MCP server/toolset "
            "requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools "
            "such as `mcp__...` or call the native OpenAI Responses API directly."
        ),
    )
