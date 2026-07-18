"""xAI-owned Anthropic request preparation for Responses and completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NoReturn, Optional, Protocol

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


class PreparePassthroughRequest(Protocol):
    def __call__(
        self,
        request_body: Payload,
        *,
        sanitize_responses_request: bool = False,
    ) -> Awaitable[tuple[bool, Optional[str], Optional[str]]]: ...


class RaiseCandidateUnavailable(Protocol):
    def __call__(self, detail: object) -> NoReturn: ...


@dataclass(frozen=True)
class Runtime:
    """Injected route-layer services used by xAI request preparation."""

    build_responses_body: Callable[..., Any]
    apply_responses_policies: Callable[..., Any]
    drop_unsupported_params: Callable[..., Any]
    prepare_passthrough_request: PreparePassthroughRequest
    unavailable_detail: Callable[..., Any]
    raise_candidate_unavailable: RaiseCandidateUnavailable
    to_native_model: Callable[..., Any]
    normalize_endpoint: Callable[..., Any]
    join_url: Callable[..., Any]
    url_factory: Callable[..., Any]
    assemble_headers: Callable[..., Any]
    prepare_completion_body: Callable[..., Any]
    validate_egress: Callable[..., Any]
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
    """Build the complete xAI OAuth Responses route plan."""
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = runtime.build_responses_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_xai_oauth_responses_adapter",
        tag_prefix="anthropic-xai-oauth-responses-adapter",
        span_name="anthropic.xai_oauth_responses_adapter",
        target_endpoint="xai:/v1/responses",
    )
    translated_request_body = runtime.apply_responses_policies(
        prepared_request_body,
        translated_request_body,
        config=adapter_config.XAI_OAUTH_RESPONSES,
    )
    translated_request_body, _unsupported = runtime.drop_unsupported_params(
        translated_request_body
    )
    try:
        prepared, target_base_url, api_key = await runtime.prepare_passthrough_request(
            translated_request_body,
            sanitize_responses_request=True,
        )
    except Exception as exc:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)
        raise
    if not prepared or target_base_url is None or api_key is None:
        missing_credential_error = Exception(
            "Anthropic adapter requests for xAI OAuth models require a managed "
            "xAI OAuth credential."
        )
        if use_alias_candidate_probe:
            runtime.raise_candidate_unavailable(missing_credential_error)
        raise missing_credential_error

    translated_request_body["model"] = runtime.to_native_model(
        translated_request_body.get("model")
    )
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

    def handle_exception(exc: Exception) -> None:
        if use_alias_candidate_probe and runtime.unavailable_detail(exc) is not None:
            runtime.raise_candidate_unavailable(exc)

    return adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.XAI_OAUTH_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=target_url,
        custom_headers=custom_headers,
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": False,
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": "xai",
            "expected_target_family": "xai",
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
    """Build the complete xAI OAuth completion route plan."""
    _ = request, use_alias_candidate_probe
    config = adapter_config.XAI_OAUTH_COMPLETION
    client_requested_stream = bool(prepared_request_body.get("stream"))
    prepared_request_body = runtime.prepare_completion_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
    )
    prepared, target_base_url, api_key = await runtime.prepare_passthrough_request(
        prepared_request_body
    )
    if not prepared or target_base_url is None or api_key is None:
        raise Exception(
            "Anthropic adapter requests for xAI OAuth models require a managed "
            "xAI OAuth credential."
        )
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
    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=prepared_request_body,
        target_url=target_url,
        api_key=api_key,
        api_base=target_base_url,
        client_requested_stream=client_requested_stream,
        perform_kwargs={"custom_llm_provider": runtime.provider},
    )
