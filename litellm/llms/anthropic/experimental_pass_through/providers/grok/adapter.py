"""Grok-owned Anthropic-to-Responses request preparation."""

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
    """Injected route-layer services used by Grok request preparation."""

    build_request_body: Callable[..., Any]
    apply_policies: Callable[..., Any]
    drop_unsupported_params: Callable[..., Any]
    drop_prior_replay: Callable[..., Any]
    prepare_passthrough_request: Callable[..., Any]
    unavailable_detail: Callable[..., Any]
    raise_candidate_unavailable: Callable[..., Any]
    join_url: Callable[..., Any]
    provider: str


async def prepare_responses_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.ResponsesAdapterRoutePlan:
    """Build the complete Grok native OAuth Responses route plan."""
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = runtime.build_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_grok_native_responses_adapter",
        tag_prefix="anthropic-grok-native-responses-adapter",
        span_name="anthropic.grok_native_responses_adapter",
        target_endpoint="xai:/v1/responses",
    )
    translated_request_body = runtime.apply_policies(
        prepared_request_body,
        translated_request_body,
        config=adapter_config.GROK_NATIVE_RESPONSES,
    )
    translated_request_body, _unsupported = runtime.drop_unsupported_params(
        translated_request_body
    )
    translated_request_body, _dropped_replay = runtime.drop_prior_replay(
        translated_request_body
    )
    try:
        (
            prepared,
            target_base_url,
            headers,
            translated_request_body,
        ) = await runtime.prepare_passthrough_request(
            translated_request_body,
            request=request,
            tags_to_add=["anthropic-grok-native-responses-adapter-entrypoint"],
            extra_fields={
                "anthropic_grok_native_requested_model": prepared_request_body.get(
                    "model"
                ),
                "anthropic_grok_native_adapter_model": adapter_model,
                "anthropic_adapter_target_endpoint": "xai:/v1/responses",
                "grok_native_entrypoint": "anthropic_messages",
                "passthrough_route_family": "anthropic_grok_native_responses_adapter",
                "route_family": "anthropic_grok_native_responses_adapter",
            },
        )
    except Exception as exc:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)
        raise
    if not prepared or target_base_url is None:
        missing_credential_error = Exception(
            "Anthropic adapter requests for Grok native OAuth models require a "
            "Grok OIDC credential."
        )
        if use_alias_candidate_probe:
            runtime.raise_candidate_unavailable(missing_credential_error)
        raise missing_credential_error

    target_url = runtime.join_url(
        base_target_url=target_base_url,
        endpoint="/v1/responses",
    )

    def handle_exception(exc: Exception) -> None:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)

    return adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.GROK_NATIVE_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=target_url,
        custom_headers=headers or {},
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": False,
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": "xai",
            "expected_target_family": "xai",
        },
        handle_exception=handle_exception,
    )
