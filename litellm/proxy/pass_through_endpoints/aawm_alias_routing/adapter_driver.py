"""Shared route execution plans for Anthropic adapter providers (RR-054 #1/#9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, TypeVar

from .adapter_config import (
    AnthropicCompletionAdapterConfig,
    AnthropicResponsesAdapterConfig,
)
from .types import Payload


@dataclass(frozen=True)
class ResponsesAdapterRoutePlan:
    """Fully prepared provider request consumed by the shared Responses driver."""

    config: AnthropicResponsesAdapterConfig
    translated_request_body: Payload
    target_url: object
    custom_headers: Payload
    client_requested_stream: bool
    perform_kwargs: Payload = field(default_factory=dict)
    handle_exception: Optional[Callable[[Exception], None]] = None
    retry_after_exception: Optional[
        Callable[[Exception], Awaitable[Optional["ResponsesAdapterRoutePlan"]]]
    ] = None
    max_retry_attempts: int = 1


@dataclass(frozen=True)
class CompletionAdapterRoutePlan:
    """Fully prepared provider request consumed by the shared completion driver."""

    config: AnthropicCompletionAdapterConfig
    prepared_request_body: Payload
    target_url: object
    api_key: str
    api_base: str
    client_requested_stream: bool
    perform_kwargs: Payload = field(default_factory=dict)
    handle_exception: Optional[Callable[[Exception], None]] = None
    retry_after_exception: Optional[
        Callable[[Exception], Awaitable[Optional["CompletionAdapterRoutePlan"]]]
    ] = None
    max_retry_attempts: int = 1


ResponsesPrepare = Callable[..., Awaitable[ResponsesAdapterRoutePlan]]
CompletionPrepare = Callable[..., Awaitable[CompletionAdapterRoutePlan]]
RouteResult = TypeVar("RouteResult")


async def run_responses_adapter_route(
    *,
    prepare: ResponsesPrepare,
    perform: Callable[..., Awaitable[RouteResult]],
    request: object,
    user_api_key_dict: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool,
) -> RouteResult:
    """Prepare and execute one config-selected Responses adapter route."""
    if isinstance(prepared_request_body, dict):
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            strip_route_identity_from_request_body,
        )

        prepared_request_body = strip_route_identity_from_request_body(
            prepared_request_body
        )
    plan = await prepare(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )

    async def perform_plan(route_plan: ResponsesAdapterRoutePlan) -> RouteResult:
        return await perform(
            config=route_plan.config,
            request=request,
            user_api_key_dict=user_api_key_dict,
            translated_request_body=route_plan.translated_request_body,
            adapter_model=adapter_model,
            target_url=route_plan.target_url,
            custom_headers=route_plan.custom_headers,
            client_requested_stream=route_plan.client_requested_stream,
            use_alias_candidate_probe=use_alias_candidate_probe,
            **route_plan.perform_kwargs,
        )

    current_plan = plan
    retries_remaining = max(0, int(plan.max_retry_attempts))
    while True:
        try:
            return await perform_plan(current_plan)
        except Exception as exc:
            retry_plan = (
                await current_plan.retry_after_exception(exc)
                if retries_remaining > 0
                and current_plan.retry_after_exception is not None
                else None
            )
            if retry_plan is None:
                if current_plan.handle_exception is not None:
                    current_plan.handle_exception(exc)
                raise
            retries_remaining -= 1
            current_plan = retry_plan


async def run_completion_adapter_route(
    *,
    prepare: CompletionPrepare,
    perform: Callable[..., Awaitable[RouteResult]],
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool,
) -> RouteResult:
    """Prepare and execute one config-selected completion adapter route."""
    if isinstance(prepared_request_body, dict):
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            strip_route_identity_from_request_body,
        )

        prepared_request_body = strip_route_identity_from_request_body(
            prepared_request_body
        )
    plan = await prepare(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )

    async def perform_plan(route_plan: CompletionAdapterRoutePlan) -> RouteResult:
        return await perform(
            config=route_plan.config,
            request=request,
            prepared_request_body=route_plan.prepared_request_body,
            adapter_model=adapter_model,
            target_url=route_plan.target_url,
            api_key=route_plan.api_key,
            api_base=route_plan.api_base,
            client_requested_stream=route_plan.client_requested_stream,
            **route_plan.perform_kwargs,
        )

    current_plan = plan
    retries_remaining = max(0, int(plan.max_retry_attempts))
    while True:
        try:
            return await perform_plan(current_plan)
        except Exception as exc:
            retry_plan = (
                await current_plan.retry_after_exception(exc)
                if retries_remaining > 0
                and current_plan.retry_after_exception is not None
                else None
            )
            if retry_plan is None:
                if current_plan.handle_exception is not None:
                    current_plan.handle_exception(exc)
                raise
            retries_remaining -= 1
            current_plan = retry_plan
