"""OpenAI Responses and Anthropic Messages ingress for Alibaba Token Plan."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterable, Optional, cast

from litellm.llms.alibaba_token_plan.chat.transformation import (
    ALIBABA_TOKEN_PLAN_API_BASE,
    ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
    ALIBABA_TOKEN_PLAN_RAW_CHOICES_HIDDEN_PARAM,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
    policy,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL = "canonical-alibaba-token-plan-credential"
_CODEX_AUTO_REVIEW_ALIASES = frozenset(
    {
        "codex-auto-review",
        "auto-review",
        "chatgpt/codex-auto-review",
    }
)


def normalize_alibaba_token_plan_adapter_model_name(
    model: Any,
    *,
    allowed_models: Iterable[str] = (),
) -> Optional[str]:
    """Normalize canonical `alibaba_token_plan/<model-id>` direct routes.

    Any structurally valid explicit `alibaba_token_plan/<nonempty-model-id>`
    route is admitted without a Python model enumeration and the exact suffix
    is forwarded upstream; `allowed_models` is retained for source
    compatibility and is not consulted.
    """

    _ = allowed_models
    return policy.normalize_alibaba_token_plan_adapter_model_name(model)


def _resolve_upstream_model(adapter_model: str) -> str:
    normalized = policy.normalize_alibaba_token_plan_adapter_model_name(adapter_model)
    if normalized is None:
        raise ValueError(f"Unsupported Alibaba Token Plan adapter model {adapter_model!r}.")
    return normalized.removeprefix("alibaba_token_plan/")


def _is_codex_auto_review_request(request_body: Mapping[str, Any]) -> bool:
    """Identify the canonical review alias after candidate model replacement."""

    values: list[Any] = [request_body.get("model")]
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, Mapping):
        values.extend(
            metadata.get(key)
            for key in (
                "codex_auto_agent_alias",
                "model_alias",
                "requested_model_alias",
                "inbound_model_alias",
            )
        )
    return any(
        isinstance(value, str)
        and value.strip().casefold() in _CODEX_AUTO_REVIEW_ALIASES
        for value in values
    )


def _codex_auto_review_response_schema(
    request_body: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Return the original Responses JSON schema, when one was requested."""

    if not _is_codex_auto_review_request(request_body):
        return None
    text_param = request_body.get("text")
    if not isinstance(text_param, Mapping):
        return None
    format_param = text_param.get("format")
    if not isinstance(format_param, Mapping):
        return None
    if format_param.get("type") != "json_schema":
        return None
    schema = format_param.get("schema")
    return dict(schema) if isinstance(schema, dict) else None


def _append_codex_auto_review_schema_instruction(
    completion_kwargs: dict[str, Any],
    *,
    schema: dict[str, Any],
) -> None:
    """Give prompt-only Alibaba review generation the complete source schema."""

    schema_instruction = (
        "Return exactly one complete JSON object matching this JSON Schema. "
        "Do not use Markdown fences or add commentary:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        completion_kwargs["messages"] = [
            {"role": "system", "content": schema_instruction}
        ]
        return

    updated_messages = list(messages)
    for index, message in enumerate(updated_messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            updated_message = dict(message)
            updated_message["content"] = f"{content}\n\n{schema_instruction}"
            updated_messages[index] = updated_message
            completion_kwargs["messages"] = updated_messages
            return
        break

    completion_kwargs["messages"] = [
        {"role": "system", "content": schema_instruction},
        *updated_messages,
    ]


_MISSING = object()


def _response_value(response: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(response, Mapping):
        return response.get(key, default)
    return getattr(response, key, default)


def _has_nonempty_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, (str, bytes, Mapping, list, tuple, set)):
        return bool(value)
    return True


def _raw_alibaba_completion_choices(
    completion_response: Any,
) -> tuple[Any, bool]:
    if isinstance(completion_response, Mapping):
        return completion_response.get("choices", _MISSING), True

    hidden_params = getattr(completion_response, "_hidden_params", None)
    if isinstance(hidden_params, Mapping):
        raw_choices = hidden_params.get(
            ALIBABA_TOKEN_PLAN_RAW_CHOICES_HIDDEN_PARAM,
            _MISSING,
        )
        if raw_choices is not _MISSING:
            return raw_choices, True

    return getattr(completion_response, "choices", _MISSING), False


def validate_codex_auto_review_completion(
    completion_response: Any,
    *,
    schema: Optional[dict[str, Any]],
) -> None:
    """Reject non-terminal or mixed Alibaba review choices before conversion."""

    if schema is None:
        return

    choices, has_provider_native_choices = _raw_alibaba_completion_choices(
        completion_response
    )
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(
            "Alibaba Token Plan auto-review requires exactly one completion choice."
        )

    choice = choices[0]
    finish_reason = _response_value(choice, "finish_reason")
    if not has_provider_native_choices:
        provider_specific_fields = _response_value(
            choice,
            "provider_specific_fields",
            {},
        )
        native_finish_reason = _response_value(
            provider_specific_fields,
            "native_finish_reason",
        )
        if native_finish_reason is _MISSING:
            raise ValueError(
                "Alibaba Token Plan auto-review requires provider-native "
                "finish-reason evidence."
            )
        finish_reason = native_finish_reason
    if finish_reason != "stop":
        raise ValueError(
            "Alibaba Token Plan auto-review requires a native stop finish reason."
        )

    message = _response_value(choice, "message")
    if message is _MISSING or _response_value(message, "role") != "assistant":
        raise ValueError(
            "Alibaba Token Plan auto-review requires one assistant decision message."
        )

    if _has_nonempty_value(_response_value(choice, "refusal")) or _has_nonempty_value(
        _response_value(message, "refusal")
    ):
        raise ValueError(
            "Alibaba Token Plan auto-review does not allow refusal output."
        )
    if _has_nonempty_value(_response_value(choice, "tool_calls")) or _has_nonempty_value(
        _response_value(message, "tool_calls")
    ):
        raise ValueError(
            "Alibaba Token Plan auto-review does not allow tool calls."
        )
    if _has_nonempty_value(
        _response_value(choice, "function_call")
    ) or _has_nonempty_value(_response_value(message, "function_call")):
        raise ValueError(
            "Alibaba Token Plan auto-review does not allow function calls."
        )
    if _has_nonempty_value(_response_value(choice, "messages")):
        raise ValueError(
            "Alibaba Token Plan auto-review does not allow additional messages."
        )

    content = _response_value(message, "content")
    if not isinstance(content, str):
        raise ValueError(
            "Alibaba Token Plan auto-review requires one text decision message."
        )


def validate_codex_auto_review_response_body(
    response_body: Mapping[str, Any],
    *,
    schema: Optional[dict[str, Any]],
) -> None:
    """Fail closed when an Alibaba review response violates its source schema."""

    if schema is None:
        return

    output_texts: list[str] = []
    output = response_body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if (
                    isinstance(content_item, Mapping)
                    and content_item.get("type") == "output_text"
                    and isinstance(content_item.get("text"), str)
                ):
                    output_texts.append(content_item["text"])

    from litellm.litellm_core_utils.json_validation_rule import validate_schema

    validate_schema(
        schema=schema,
        response=output_texts[0] if len(output_texts) == 1 else "",
    )


def _add_adapter_metadata(
    *,
    request_body: dict[str, Any],
    config: adapter_config.AnthropicCompletionAdapterConfig,
    adapter_model: str,
    upstream_model: str,
    ingress: str,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    metadata = dict(updated_body.get("litellm_metadata") or {})
    tags = list(metadata.get("tags") or [])
    for tag in (
        f"route:{config.route_family}",
        config.tag_prefix,
        f"{config.tag_prefix}-model:{adapter_model}",
        f"{config.tag_prefix}-target:{config.target_endpoint_label}",
    ):
        if tag not in tags:
            tags.append(tag)

    spans = list(metadata.get("langfuse_spans") or [])
    spans.append(
        {
            "name": config.span_name,
            "metadata": {
                "requested_model": request_body.get("model"),
                "adapter_model": adapter_model,
                "upstream_model": upstream_model,
                "stream": bool(request_body.get("stream")),
            },
        }
    )
    metadata.update(
        {
            "tags": tags,
            "langfuse_spans": spans,
            "passthrough_route_family": config.route_family,
            "route_family": config.route_family,
            "alibaba_token_plan_adapter_model": adapter_model,
            "alibaba_token_plan_upstream_model": upstream_model,
            "alibaba_token_plan_api_base": ALIBABA_TOKEN_PLAN_API_BASE,
            "billing_mode": "alibaba_token_plan_subscription",
            "actual_invoice_cost_known": False,
            "reference_cost_kind": "provider_token_plan_no_public_per_token_rate",
            f"{ingress}_adapter_model": adapter_model,
            f"{ingress}_adapter_original_model": request_body.get("model"),
            f"{ingress}_adapter_target_endpoint": config.target_endpoint_label,
        }
    )
    updated_body["litellm_metadata"] = metadata
    return updated_body


def normalize_alibaba_token_plan_custom_tool_outputs(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Convert Codex custom-tool results to the function-tool wire shape."""

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body

    changed = False
    updated_items: list[Any] = []
    for item in input_items:
        if (
            isinstance(item, dict)
            and item.get("type") == "custom_tool_call_output"
            and isinstance(item.get("call_id"), str)
            and item["call_id"].strip()
        ):
            updated_item = dict(item)
            updated_item["type"] = "function_call_output"
            updated_items.append(updated_item)
            changed = True
        else:
            updated_items.append(item)
    if not changed:
        return request_body
    updated_body = dict(request_body)
    updated_body["input"] = updated_items
    return updated_body


def _restore_codex_agent_message_payloads(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize Codex collaboration assignments through the shared owner."""
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_collaboration_dispatch import (
        restore_codex_agent_message_payloads_for_openai_egress,
    )

    updated_body = restore_codex_agent_message_payloads_for_openai_egress(
        request_body,
    )
    if updated_body is request_body:
        return request_body, {}
    return updated_body, {
        "alibaba_token_plan_codex_agent_task_payload_normalized": True,
    }


async def prepare_codex_alibaba_token_plan_adapter_route(
    *,
    request: object,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Translate OpenAI Responses ingress to Token Plan chat completions."""

    _ = request, use_alias_candidate_probe
    upstream_model = _resolve_upstream_model(adapter_model)
    config = adapter_config.CODEX_ALIBABA_TOKEN_PLAN
    prepared_request_body, task_payload_changes = _restore_codex_agent_message_payloads(prepared_request_body)
    request_body = _add_adapter_metadata(
        request_body=prepared_request_body,
        config=config,
        adapter_model=adapter_model,
        upstream_model=upstream_model,
        ingress="codex",
    )
    if task_payload_changes:
        metadata = dict(request_body.get("litellm_metadata") or {})
        metadata.update(task_payload_changes)
        request_body["litellm_metadata"] = metadata

    request_input = request_body.get("input", "")
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {key: value for key, value in request_body.items() if key not in {"input", "model", "litellm_metadata"}},
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider="alibaba_token_plan",
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    is_auto_review = _is_codex_auto_review_request(request_body)
    auto_review_schema = _codex_auto_review_response_schema(request_body)
    if is_auto_review:
        # Alibaba Token Plan accepts the translated chat request but not
        # OpenAI's response_format field. Keep the source Responses schema for
        # the post-egress validation gate below.
        completion_kwargs.pop("response_format", None)
        if auto_review_schema is not None:
            _append_codex_auto_review_schema_instruction(
                completion_kwargs,
                schema=auto_review_schema,
            )
    completion_kwargs.update(
        {
            "metadata": litellm_metadata,
            "custom_llm_provider": "alibaba_token_plan",
            "num_retries": 0,
        }
    )
    previous_response_id = responses_api_request.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=previous_response_id,
            litellm_completion_request=completion_kwargs,
        )

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
        api_key=ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL,
        api_base=ALIBABA_TOKEN_PLAN_API_BASE,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "completion_kwargs": completion_kwargs,
            "request_input": request_input,
            "responses_api_request": responses_api_request,
            "litellm_metadata": litellm_metadata,
            "upstream_model": upstream_model,
            "auto_review_schema": auto_review_schema,
        },
    )


async def prepare_anthropic_alibaba_token_plan_adapter_route(
    *,
    request: object,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Prepare Anthropic Messages ingress for Token Plan chat completions."""

    _ = request, use_alias_candidate_probe
    upstream_model = _resolve_upstream_model(adapter_model)
    config = adapter_config.ANTHROPIC_ALIBABA_TOKEN_PLAN
    request_body = _add_adapter_metadata(
        request_body=prepared_request_body,
        config=config,
        adapter_model=adapter_model,
        upstream_model=upstream_model,
        ingress="anthropic",
    )
    extra_handler_kwargs: dict[str, Any] = {"num_retries": 0}
    parallel_tool_calls = request_body.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        extra_handler_kwargs["parallel_tool_calls"] = parallel_tool_calls

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
        api_key=ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL,
        api_base=ALIBABA_TOKEN_PLAN_API_BASE,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "custom_llm_provider": "alibaba_token_plan",
            "model_for_upstream": upstream_model,
            "extra_handler_kwargs": extra_handler_kwargs,
        },
    )
