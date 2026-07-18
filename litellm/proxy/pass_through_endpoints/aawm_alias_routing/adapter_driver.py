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
    plan = await prepare(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    try:
        return await perform(
            config=plan.config,
            request=request,
            user_api_key_dict=user_api_key_dict,
            translated_request_body=plan.translated_request_body,
            adapter_model=adapter_model,
            target_url=plan.target_url,
            custom_headers=plan.custom_headers,
            client_requested_stream=plan.client_requested_stream,
            use_alias_candidate_probe=use_alias_candidate_probe,
            **plan.perform_kwargs,
        )
    except Exception as exc:
        if plan.handle_exception is not None:
            plan.handle_exception(exc)
        raise


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
    plan = await prepare(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    try:
        return await perform(
            config=plan.config,
            request=request,
            prepared_request_body=plan.prepared_request_body,
            adapter_model=adapter_model,
            target_url=plan.target_url,
            api_key=plan.api_key,
            api_base=plan.api_base,
            client_requested_stream=plan.client_requested_stream,
            **plan.perform_kwargs,
        )
    except Exception as exc:
        if plan.handle_exception is not None:
            plan.handle_exception(exc)
        raise
