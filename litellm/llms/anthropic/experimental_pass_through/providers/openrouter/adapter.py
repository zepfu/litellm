"""OpenRouter-owned Anthropic request preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Optional, Protocol

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

from ..common import reject_raw_mcp_tools


class RaiseCandidateUnavailable(Protocol):
    def __call__(self, detail: object) -> NoReturn: ...


@dataclass(frozen=True)
class Runtime:
    """Injected route-layer services used by OpenRouter request preparation."""

    compact_context: Callable[..., Any]
    log_debug: Callable[..., Any]
    build_responses_body: Callable[..., Any]
    apply_parallel_policy: Callable[..., Any]
    apply_forced_tool_choice: Callable[..., Any]
    contains_mcp_tools: Callable[..., Any]
    get_api_key: Callable[[], Optional[str]]
    raise_candidate_unavailable: RaiseCandidateUnavailable
    get_target_base: Callable[..., Any]
    normalize_endpoint: Callable[..., Any]
    join_url: Callable[..., Any]
    url_factory: Callable[..., Any]
    assemble_headers: Callable[..., Any]
    build_default_headers: Callable[..., Any]
    perform_responses_request: Callable[..., Any]
    get_completion_model: Callable[..., Any]
    prepare_completion_body: Callable[..., Any]
    validate_egress: Callable[..., Any]
    perform_completion_operation: Callable[..., Any]
    provider: str
    provider_target: Any


async def prepare_responses_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.ResponsesAdapterRoutePlan:
    """Build the complete OpenRouter Responses route plan."""
    client_requested_stream = bool(prepared_request_body.get("stream"))
    (
        prepared_request_body,
        compacted_count,
        compacted_markers,
        _compaction_metadata,
    ) = runtime.compact_context(
        prepared_request_body,
        tag_prefix="openrouter-adapter",
        metadata_prefix="openrouter_adapter",
        span_name="openrouter_adapter.claude_context_compaction",
    )
    if compacted_count > 0:
        runtime.log_debug(
            "Compacted Claude Code context for OpenRouter Responses adapter; "
            "count=%s markers=%s",
            compacted_count,
            sorted(compacted_markers),
        )
    translated_request_body = runtime.build_responses_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_openrouter_responses_adapter",
        tag_prefix="anthropic-openrouter-responses-adapter",
        span_name="anthropic.openrouter_responses_adapter",
        target_endpoint="openrouter:/v1/responses",
    )
    translated_request_body, parallel_changes = runtime.apply_parallel_policy(
        translated_request_body
    )
    if parallel_changes:
        runtime.log_debug(
            "Applied OpenRouter adapter parallel instruction policy; tools=%s",
            parallel_changes.get(
                "openrouter_adapter_parallel_instruction_tool_names"
            ),
        )
    translated_request_body, _forced_changes = runtime.apply_forced_tool_choice(
        prepared_request_body,
        translated_request_body,
    )
    reject_raw_mcp_tools(
        translated_request_body,
        contains_mcp_tools=runtime.contains_mcp_tools,
    )
    api_key = runtime.get_api_key()
    if api_key is None:
        runtime.raise_candidate_unavailable(
            "Anthropic adapter requests for OpenRouter models require "
            "'AAWM_OPENROUTER_API_KEY' or 'OPENROUTER_API_KEY' in environment."
        )
    target_base_url = runtime.get_target_base()
    normalized_endpoint = runtime.normalize_endpoint(
        endpoint="/v1/responses",
        base_target_url=target_base_url,
    )
    target_url = runtime.join_url(
        runtime.url_factory(target_base_url),
        normalized_endpoint,
        runtime.provider_target,
    )
    custom_headers = runtime.assemble_headers(
        api_key=api_key,
        request=request,
    )
    custom_headers.update(runtime.build_default_headers())

    async def pass_through_fn(**kwargs: Any) -> Any:
        return await runtime.perform_responses_request(
            adapter_model=adapter_model,
            log_warnings=not use_alias_candidate_probe,
            use_alias_candidate_probe=use_alias_candidate_probe,
            **kwargs,
        )

    return adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.OPENROUTER_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=target_url,
        custom_headers=custom_headers,
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": False,
            "allowed_forward_headers": [],
            "allowed_pass_through_prefixed_headers": [],
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": "openrouter",
            "expected_target_family": "openrouter",
            "pass_through_fn": pass_through_fn,
        },
    )


async def prepare_completion_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Build the complete OpenRouter completion route plan."""
    _ = request
    config = adapter_config.OPENROUTER_COMPLETION
    client_requested_stream = bool(prepared_request_body.get("stream"))
    upstream_adapter_model = runtime.get_completion_model(adapter_model) or adapter_model
    prepared_request_body = runtime.prepare_completion_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
    )
    api_key = runtime.get_api_key()
    if api_key is None:
        runtime.raise_candidate_unavailable(
            "Anthropic adapter requests for OpenRouter models require "
            "'AAWM_OPENROUTER_API_KEY' or 'OPENROUTER_API_KEY' in environment."
        )
    target_base_url = runtime.get_target_base()
    normalized_endpoint = runtime.normalize_endpoint(
        endpoint="/v1/chat/completions",
        base_target_url=target_base_url,
    )
    target_url = runtime.join_url(
        runtime.url_factory(target_base_url),
        normalized_endpoint,
        runtime.provider_target,
    )
    validation_headers = {
        **runtime.build_default_headers(),
        "Authorization": f"Bearer {api_key}",
    }
    runtime.validate_egress(
        url=str(target_url),
        headers=validation_headers,
        credential_family=config.credential_family,
        expected_target_family=config.expected_target_family,
    )

    async def operation_wrapper(operation: Any) -> Any:
        return await runtime.perform_completion_operation(
            adapter_model=upstream_adapter_model,
            operation=operation,
            log_warnings=not use_alias_candidate_probe,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=prepared_request_body,
        target_url=target_url,
        api_key=api_key,
        api_base=f"{target_base_url.rstrip('/')}/v1",
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "custom_llm_provider": runtime.provider,
            "model_for_upstream": upstream_adapter_model,
            "operation_wrapper": operation_wrapper,
            "extra_handler_kwargs": {"headers": runtime.build_default_headers()},
        },
    )
