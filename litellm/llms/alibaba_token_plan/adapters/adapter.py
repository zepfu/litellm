"""OpenAI Responses and Anthropic Messages ingress for Alibaba Token Plan."""

from __future__ import annotations

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
    if updated_body is request_body or updated_body.get("input") is request_body.get(
        "input",
    ):
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
