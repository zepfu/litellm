"""OpenCode Zen-owned Anthropic request preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

from ..common import reject_raw_mcp_tools


@dataclass(frozen=True)
class Runtime:
    """Injected route-layer services used by OpenCode Zen preparation."""

    build_responses_body: Callable[..., Any]
    add_logging_metadata: Callable[..., Any]
    apply_parallel_policy: Callable[..., Any]
    apply_forced_tool_choice: Callable[..., Any]
    log_debug: Callable[..., Any]
    contains_mcp_tools: Callable[..., Any]
    get_target_base: Callable[..., Any]
    join_url: Callable[..., Any]
    build_headers: Callable[..., Any]
    unavailable_detail: Callable[..., Any]
    raise_candidate_unavailable: Callable[..., Any]
    url_factory: Callable[..., Any]
    prepare_completion_body: Callable[..., Any]
    load_api_key: Callable[..., Any]
    assemble_headers: Callable[..., Any]
    validate_egress: Callable[..., Any]
    provider: str
    completion_provider: str


async def prepare_responses_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.ResponsesAdapterRoutePlan:
    """Build the complete OpenCode Zen Responses route plan."""
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = runtime.build_responses_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_opencode_zen_responses_adapter",
        tag_prefix="anthropic-opencode-zen-responses-adapter",
        span_name="anthropic.opencode_zen_responses_adapter",
        target_endpoint="opencode_zen:/v1/responses",
    )
    translated_request_body = runtime.add_logging_metadata(
        translated_request_body,
        route_family="anthropic_opencode_zen_responses_adapter",
        tag_prefix="anthropic-opencode-zen-responses-adapter",
        requested_model=prepared_request_body.get("model"),
        adapter_model=adapter_model,
        input_shape="anthropic_messages",
        output_shape="anthropic_messages",
    )
    translated_request_body, parallel_changes = runtime.apply_parallel_policy(
        translated_request_body
    )
    if parallel_changes:
        runtime.log_debug(
            "Applied OpenCode Zen adapter parallel instruction policy; tools=%s",
            parallel_changes.get(
                "openrouter_adapter_parallel_instruction_tool_names"
            ),
        )
    translated_request_body, forced_changes = runtime.apply_forced_tool_choice(
        prepared_request_body,
        translated_request_body,
    )
    if forced_changes:
        runtime.log_debug(
            "Applied OpenCode Zen adapter explicit Bash tool choice: %s",
            forced_changes.get("forced_explicit_bash_tool_choice"),
        )
    reject_raw_mcp_tools(
        translated_request_body,
        contains_mcp_tools=runtime.contains_mcp_tools,
    )
    target_base_url = runtime.get_target_base()
    target_url = runtime.join_url(
        base_target_url=target_base_url,
        endpoint="/v1/responses",
    )
    custom_headers = await runtime.build_headers(
        request,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )

    def handle_exception(exc: Exception) -> None:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)

    return adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.OPENCODE_ZEN_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=runtime.url_factory(target_url),
        custom_headers=custom_headers,
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": False,
            "allowed_forward_headers": [],
            "allowed_pass_through_prefixed_headers": [],
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": "opencode",
            "expected_target_family": "opencode",
            "malformed_upstream_url": target_url,
        },
        handle_exception=handle_exception,
    )


async def prepare_completion_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Build the complete OpenCode Zen completion route plan."""
    config = adapter_config.OPENCODE_ZEN_COMPLETION
    client_requested_stream = bool(prepared_request_body.get("stream"))
    requested_model = prepared_request_body.get("model")
    prepared_request_body = runtime.add_logging_metadata(
        prepared_request_body,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        requested_model=requested_model,
        adapter_model=adapter_model,
        input_shape="anthropic_messages",
        output_shape="anthropic_messages",
    )
    prepared_request_body = runtime.prepare_completion_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
    )
    target_base_url = runtime.get_target_base()
    target_url = runtime.join_url(
        base_target_url=target_base_url,
        endpoint="/v1/chat/completions",
    )
    api_key = await runtime.load_api_key(
        use_alias_candidate_probe=use_alias_candidate_probe
    )
    custom_headers = runtime.assemble_headers(api_key=api_key, request=request)
    runtime.validate_egress(
        url=target_url,
        headers=custom_headers,
        credential_family=config.credential_family,
        expected_target_family=config.expected_target_family,
    )

    def handle_exception(exc: Exception) -> None:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=prepared_request_body,
        target_url=runtime.url_factory(target_url),
        api_key=api_key,
        api_base=f"{target_base_url.rstrip('/')}/v1",
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "custom_llm_provider": runtime.completion_provider,
            "model_for_upstream": adapter_model,
        },
        handle_exception=handle_exception,
    )
