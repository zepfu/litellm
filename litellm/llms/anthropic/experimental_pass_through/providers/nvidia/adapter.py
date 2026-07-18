"""NVIDIA-owned Anthropic-to-completion request preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


@dataclass(frozen=True)
class Runtime:
    """Injected route-layer services used by NVIDIA request preparation."""

    should_force_fake_stream: Callable[..., Any]
    prepare_request_body: Callable[..., Any]
    get_api_key: Callable[..., Any]
    get_target_base: Callable[..., Any]
    normalize_endpoint: Callable[..., Any]
    join_url: Callable[..., Any]
    url_factory: Callable[..., Any]
    validate_egress: Callable[..., Any]
    perform_operation: Callable[..., Any]
    get_timeout_seconds: Callable[..., Any]
    get_inner_max_retries: Callable[..., Any]
    provider: str
    provider_target: Any


async def prepare_completion_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Build the complete NVIDIA completion route plan."""
    _ = request, use_alias_candidate_probe
    client_requested_stream = bool(prepared_request_body.get("stream"))
    use_fake_stream = client_requested_stream and runtime.should_force_fake_stream(
        adapter_model
    )
    upstream_stream = client_requested_stream and not use_fake_stream
    config = adapter_config.NVIDIA_COMPLETION
    prepared_request_body = runtime.prepare_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
        span_metadata_extra={
            "upstream_stream": upstream_stream,
            "fake_stream": use_fake_stream,
        },
    )
    api_key = runtime.get_api_key()
    if api_key is None:
        raise Exception(
            "Anthropic adapter requests for NVIDIA models require "
            "'AAWM_NVIDIA_API_KEY', 'NVIDIA_NIM_API_KEY', or 'NVIDIA_API_KEY' "
            "in environment."
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
    runtime.validate_egress(
        url=str(target_url),
        headers={"Authorization": f"Bearer {api_key}"},
        credential_family=config.credential_family,
        expected_target_family=config.expected_target_family,
    )

    async def operation_wrapper(operation: Any) -> Any:
        return await runtime.perform_operation(
            adapter_model=adapter_model,
            operation=operation,
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
            "model_for_upstream": adapter_model,
            "stream_override": upstream_stream,
            "timeout": runtime.get_timeout_seconds(adapter_model),
            "max_retries": runtime.get_inner_max_retries(),
            "operation_wrapper": operation_wrapper,
            "fake_stream": use_fake_stream,
        },
    )
