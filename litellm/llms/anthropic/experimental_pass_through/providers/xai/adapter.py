"""xAI-owned Anthropic request preparation for Responses and completions."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, NoReturn, Optional, Protocol

from litellm.llms.xai.route_descriptors import XAI_OAUTH_CREDENTIAL_FAMILY
from litellm.llms.xai.oauth import (
    XaiOAuthCredentialSnapshot,
    bind_xai_oauth_snapshot_to_request,
    clear_xai_oauth_snapshot_from_request,
    get_xai_oauth_snapshot_from_request,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.xai_oauth import (
    XaiOAuthDirectAccountTraversal,
    XaiOAuthSelectedAccount,
    bind_xai_oauth_selected_account_to_request,
    build_xai_oauth_direct_account_traversal,
    recover_xai_oauth_direct_request,
)


class PreparePassthroughRequest(Protocol):
    def __call__(
        self,
        request_body: Payload,
        *,
        request: Optional[object] = None,
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


def _prepare_passthrough_request_accepts_request(
    callback: Callable[..., Any],
) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    return "request" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def _prepare_passthrough_request(
    runtime: Runtime,
    request_body: Payload,
    *,
    request: object,
    sanitize_responses_request: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    kwargs: dict[str, Any] = {
        "sanitize_responses_request": sanitize_responses_request,
    }
    if _prepare_passthrough_request_accepts_request(
        runtime.prepare_passthrough_request
    ):
        kwargs["request"] = request
    return await runtime.prepare_passthrough_request(request_body, **kwargs)


async def _recover_xai_oauth_direct_retry(
    *,
    request: object,
    ingress_request_body: Payload,
    traversal: Optional[XaiOAuthDirectAccountTraversal],
    exc: Exception,
    api_base: str,
    use_alias_candidate_probe: bool,
) -> tuple[Optional[XaiOAuthCredentialSnapshot], Optional[XaiOAuthSelectedAccount]]:
    if use_alias_candidate_probe or traversal is None:
        return None, None
    snapshot = get_xai_oauth_snapshot_from_request(request)
    recovery = await recover_xai_oauth_direct_request(
        traversal=traversal,
        request_body=ingress_request_body,
        exc=exc,
        snapshot=snapshot,
        api_base=api_base,
        ingress_format="anthropic",
    )
    if recovery is None:
        return None, None
    if recovery.refreshed_snapshot is not None:
        bind_xai_oauth_snapshot_to_request(request, recovery.refreshed_snapshot)
        return recovery.refreshed_snapshot, None
    return None, recovery.selected_account


async def prepare_responses_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.ResponsesAdapterRoutePlan:
    """Build the complete xAI OAuth Responses route plan."""
    ingress_request_body = copy.deepcopy(prepared_request_body)
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
    rollover_request_body = copy.deepcopy(translated_request_body)
    direct_traversal = (
        None
        if use_alias_candidate_probe
        else await build_xai_oauth_direct_account_traversal(
            cooldown_family="anthropic",
            request=request,
        )
    )
    try:
        prepared, target_base_url, api_key = await _prepare_passthrough_request(
            runtime,
            translated_request_body,
            request=request,
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

    plan = adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.XAI_OAUTH_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=target_url,
        custom_headers=custom_headers,
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": False,
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": XAI_OAUTH_CREDENTIAL_FAMILY,
            "expected_target_family": "xai",
        },
        handle_exception=handle_exception,
        max_retry_attempts=(
            direct_traversal.max_retry_attempts
            if direct_traversal is not None
            else 1
        ),
    )

    def _with_direct_retry(
        route_plan: adapter_driver.ResponsesAdapterRoutePlan,
    ) -> adapter_driver.ResponsesAdapterRoutePlan:
        async def retry_after_exception(
            exc: Exception,
        ) -> Optional[adapter_driver.ResponsesAdapterRoutePlan]:
            refreshed_snapshot, selected_account = (
                await _recover_xai_oauth_direct_retry(
                    request=request,
                    ingress_request_body=ingress_request_body,
                    traversal=direct_traversal,
                    exc=exc,
                    api_base=target_base_url,
                    use_alias_candidate_probe=use_alias_candidate_probe,
                )
            )
            if refreshed_snapshot is not None:
                return _with_direct_retry(
                    replace(
                        route_plan,
                        custom_headers=runtime.assemble_headers(
                            api_key=refreshed_snapshot.access_token,
                            request=request,
                        ),
                    )
                )
            if selected_account is None:
                return None
            bind_xai_oauth_selected_account_to_request(
                request,
                selected_account,
            )
            clear_xai_oauth_snapshot_from_request(request)
            rollover_body = copy.deepcopy(rollover_request_body)
            prepared, rollover_base_url, rollover_api_key = (
                await _prepare_passthrough_request(
                    runtime,
                    rollover_body,
                    request=request,
                    sanitize_responses_request=True,
                )
            )
            if (
                not prepared
                or rollover_base_url is None
                or rollover_api_key is None
            ):
                raise Exception(
                    "Anthropic adapter requests for xAI OAuth models require "
                    "a managed xAI OAuth credential."
                )
            rollover_body["model"] = runtime.to_native_model(
                rollover_body.get("model")
            )
            rollover_target_url = runtime.join_url(
                runtime.url_factory(rollover_base_url),
                runtime.normalize_endpoint(
                    endpoint="/v1/responses",
                    base_target_url=rollover_base_url,
                ),
                runtime.provider_target,
            )
            return _with_direct_retry(
                replace(
                    route_plan,
                    translated_request_body=rollover_body,
                    target_url=rollover_target_url,
                    custom_headers=runtime.assemble_headers(
                        api_key=rollover_api_key,
                        request=request,
                    ),
                )
            )

        return replace(route_plan, retry_after_exception=retry_after_exception)

    return _with_direct_retry(plan)


async def prepare_completion_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Build the complete xAI OAuth completion route plan."""
    config = adapter_config.XAI_OAUTH_COMPLETION
    ingress_request_body = copy.deepcopy(prepared_request_body)
    client_requested_stream = bool(prepared_request_body.get("stream"))
    prepared_request_body = runtime.prepare_completion_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
    )
    rollover_request_body = copy.deepcopy(prepared_request_body)
    direct_traversal = (
        None
        if use_alias_candidate_probe
        else await build_xai_oauth_direct_account_traversal(
            cooldown_family="anthropic",
            request=request,
        )
    )
    prepared, target_base_url, api_key = await _prepare_passthrough_request(
        runtime,
        prepared_request_body,
        request=request,
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
    plan = adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=prepared_request_body,
        target_url=target_url,
        api_key=api_key,
        api_base=target_base_url,
        client_requested_stream=client_requested_stream,
        perform_kwargs={"custom_llm_provider": runtime.provider},
        max_retry_attempts=(
            direct_traversal.max_retry_attempts
            if direct_traversal is not None
            else 1
        ),
    )

    def _with_direct_retry(
        route_plan: adapter_driver.CompletionAdapterRoutePlan,
    ) -> adapter_driver.CompletionAdapterRoutePlan:
        async def retry_after_exception(
            exc: Exception,
        ) -> Optional[adapter_driver.CompletionAdapterRoutePlan]:
            refreshed_snapshot, selected_account = (
                await _recover_xai_oauth_direct_retry(
                    request=request,
                    ingress_request_body=ingress_request_body,
                    traversal=direct_traversal,
                    exc=exc,
                    api_base=target_base_url,
                    use_alias_candidate_probe=use_alias_candidate_probe,
                )
            )
            if refreshed_snapshot is not None:
                return _with_direct_retry(
                    replace(
                        route_plan,
                        api_key=refreshed_snapshot.access_token,
                    )
                )
            if selected_account is None:
                return None
            bind_xai_oauth_selected_account_to_request(
                request,
                selected_account,
            )
            clear_xai_oauth_snapshot_from_request(request)
            rollover_body = copy.deepcopy(rollover_request_body)
            prepared, rollover_base_url, rollover_api_key = (
                await _prepare_passthrough_request(
                    runtime,
                    rollover_body,
                    request=request,
                )
            )
            if (
                not prepared
                or rollover_base_url is None
                or rollover_api_key is None
            ):
                raise Exception(
                    "Anthropic adapter requests for xAI OAuth models require "
                    "a managed xAI OAuth credential."
                )
            rollover_target_url = runtime.join_url(
                runtime.url_factory(rollover_base_url),
                runtime.normalize_endpoint(
                    endpoint="/v1/chat/completions",
                    base_target_url=rollover_base_url,
                ),
                runtime.provider_target,
            )
            runtime.validate_egress(
                url=str(rollover_target_url),
                headers={"Authorization": f"Bearer {rollover_api_key}"},
                credential_family=config.credential_family,
                expected_target_family=config.expected_target_family,
            )
            return _with_direct_retry(
                replace(
                    route_plan,
                    prepared_request_body=rollover_body,
                    target_url=rollover_target_url,
                    api_key=rollover_api_key,
                    api_base=rollover_base_url,
                )
            )

        return replace(route_plan, retry_after_exception=retry_after_exception)

    return _with_direct_retry(plan)
