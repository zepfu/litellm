"""OpenAI Responses and Anthropic Messages ingress for Alibaba Token Plan."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, cast

from litellm.llms.alibaba_token_plan.chat.transformation import (
    ALIBABA_TOKEN_PLAN_API_BASE,
    ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
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
_CODEX_AGENT_MESSAGE_EMPTY_PAYLOAD_PATTERN = re.compile(
    r"\AMessage Type: (?:NEW_TASK|MESSAGE)\n" r"Task name: [^\n]+\n" r"Sender: [^\n]+\n" r"Payload:\n?\Z"
)


def normalize_alibaba_token_plan_adapter_model_name(
    model: Any,
    *,
    allowed_models: Iterable[str],
) -> Optional[str]:
    """Normalize only canonical `alibaba_token_plan/<model-id>` routes."""

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if not separator or provider_prefix != "alibaba_token_plan" or not model_id.strip():
        return None
    return candidate if candidate in allowed_models else None


def _resolve_upstream_model(adapter_model: str) -> str:
    if adapter_model not in policy.ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS:
        raise ValueError(f"Unsupported Alibaba Token Plan adapter model {adapter_model!r}.")
    return adapter_model.removeprefix("alibaba_token_plan/")


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
    """Restore only explicit Codex collaboration task payloads."""

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, {}

    restored_count = 0
    restored_chars = 0
    updated_items: list[Any] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            updated_items.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list) or len(content) != 2:
            updated_items.append(item)
            continue
        visible_part, payload_part = content
        if not isinstance(visible_part, dict) or not isinstance(payload_part, dict):
            updated_items.append(item)
            continue
        visible_text = visible_part.get("text")
        payload = payload_part.get("encrypted_content")
        if (
            visible_part.get("type") not in {"input_text", "text"}
            or not isinstance(visible_text, str)
            or _CODEX_AGENT_MESSAGE_EMPTY_PAYLOAD_PATTERN.fullmatch(visible_text) is None
            or payload_part.get("type") != "encrypted_content"
            or not isinstance(payload, str)
            or not payload
        ):
            updated_items.append(item)
            continue

        separator = "" if visible_text.endswith("\n") else "\n"
        updated_item = dict(item)
        updated_item["content"] = [
            {
                "type": visible_part["type"],
                "text": f"{visible_text}{separator}{payload}",
            }
        ]
        updated_items.append(updated_item)
        restored_count += 1
        restored_chars += len(payload)

    if restored_count == 0:
        return request_body, {}
    updated_body = dict(request_body)
    updated_body["input"] = updated_items
    return updated_body, {
        "alibaba_token_plan_codex_agent_task_payload_restored": True,
        "alibaba_token_plan_codex_agent_task_payload_restored_count": restored_count,
        "alibaba_token_plan_codex_agent_task_payload_restored_chars": restored_chars,
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
