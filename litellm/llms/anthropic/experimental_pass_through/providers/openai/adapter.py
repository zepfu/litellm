"""OpenAI-owned Anthropic-to-Responses request preparation."""

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
    """Injected route-layer services used by OpenAI request preparation."""

    resolve_auth_context: Callable[..., Any]
    compact_context: Callable[..., Any]
    log_debug: Callable[..., Any]
    build_request_body: Callable[..., Any]
    apply_policies: Callable[..., Any]
    add_breakout_metadata: Callable[..., Any]
    contains_mcp_tools: Callable[..., Any]
    get_target_base: Callable[..., Any]
    normalize_endpoint: Callable[..., Any]
    join_url: Callable[..., Any]
    url_factory: Callable[..., Any]
    provider: str
    forward_header_allowlist: tuple[str, ...]
    xpass_header_allowlist: tuple[str, ...]


async def prepare_responses_route(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.ResponsesAdapterRoutePlan:
    """Build the complete OpenAI Responses route plan."""
    _ = use_alias_candidate_probe
    client_requested_stream = bool(prepared_request_body.get("stream"))
    (
        custom_headers,
        forward_headers,
        use_chatgpt_codex_defaults,
        egress_credential_family,
    ) = await runtime.resolve_auth_context(request)
    (
        prepared_request_body,
        compacted_count,
        compacted_markers,
        _compaction_metadata,
    ) = runtime.compact_context(prepared_request_body)
    if compacted_count > 0:
        runtime.log_debug(
            "Compacted Claude Code context for OpenAI Responses adapter; "
            "count=%s markers=%s",
            compacted_count,
            sorted(compacted_markers),
        )
    translated_request_body = runtime.build_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        use_chatgpt_codex_defaults=use_chatgpt_codex_defaults,
    )
    translated_request_body = runtime.apply_policies(
        prepared_request_body,
        translated_request_body,
        config=adapter_config.OPENAI_RESPONSES,
    )
    if use_chatgpt_codex_defaults:
        translated_request_body = runtime.add_breakout_metadata(
            translated_request_body
        )
    reject_raw_mcp_tools(
        translated_request_body,
        contains_mcp_tools=runtime.contains_mcp_tools,
    )

    target_base_url = runtime.get_target_base(
        request,
        prefer_chatgpt_codex_backend=use_chatgpt_codex_defaults,
    )
    normalized_endpoint = runtime.normalize_endpoint(
        endpoint="/v1/responses",
        base_target_url=target_base_url,
    )
    target_url = runtime.join_url(
        runtime.url_factory(target_base_url),
        normalized_endpoint,
        runtime.provider,
    )
    return adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.OPENAI_RESPONSES,
        translated_request_body=translated_request_body,
        target_url=target_url,
        custom_headers=custom_headers,
        client_requested_stream=client_requested_stream,
        perform_kwargs={
            "forward_headers": forward_headers,
            "allowed_forward_headers": list(runtime.forward_header_allowlist),
            "allowed_pass_through_prefixed_headers": list(
                runtime.xpass_header_allowlist
            ),
            "custom_llm_provider": runtime.provider,
            "egress_credential_family": egress_credential_family,
            "expected_target_family": "openai",
            "use_codex_native_tools": use_chatgpt_codex_defaults,
        },
    )
