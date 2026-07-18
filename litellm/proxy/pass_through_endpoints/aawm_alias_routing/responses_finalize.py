"""Package-owned Anthropic Responses adapter finalization (RR-054 #1/#9)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from .types import Payload


@dataclass(frozen=True)
class ResponsesFinalizeRuntime:
    annotate_request: Callable[[object, object], None]
    validate_stream: Callable[..., Awaitable[StreamingResponse]]
    collect_stream: Callable[[StreamingResponse], Awaitable[Payload]]
    build_response: Callable[..., Response]
    copy_headers: Callable[..., None]
    build_streaming_response: Callable[..., StreamingResponse]
    decode_response_body: Callable[[object], str]
    build_malformed_context: Callable[..., Payload]


_runtime: Optional[ResponsesFinalizeRuntime] = None


def configure_responses_finalize_runtime(
    runtime: ResponsesFinalizeRuntime,
) -> None:
    global _runtime
    _runtime = runtime


def _get_runtime() -> ResponsesFinalizeRuntime:
    if _runtime is None:
        raise RuntimeError("Responses finalize runtime has not been configured")
    return _runtime


async def finalize_anthropic_responses_adapter_upstream_response(
    *,
    upstream_response: object,
    request: object,
    translated_request_body: Payload,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    provider: str,
    target_url: object,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool,
    use_codex_native_tools: bool = False,
    unexpected_detail: str,
    response_builder_kwargs: Optional[Payload] = None,
    stream_builder_kwargs: Optional[Payload] = None,
    malformed_upstream_url: Optional[object] = None,
    skip_stream_probe_validation: bool = False,
) -> Response:
    """Translate one provider Responses result into the Anthropic wire shape."""
    runtime = _get_runtime()
    runtime.annotate_request(request, target_url)
    builder_kwargs = dict(response_builder_kwargs or {})
    stream_kwargs = dict(stream_builder_kwargs or {})
    malformed_url = (
        malformed_upstream_url if malformed_upstream_url is not None else str(target_url)
    )

    malformed_context = runtime.build_malformed_context(
        request=request,
        request_body=translated_request_body,
        adapter=adapter,
        adapter_model=adapter_model,
        upstream_url=malformed_url,
        provider=provider,
    )

    if isinstance(upstream_response, StreamingResponse):
        if not skip_stream_probe_validation:
            upstream_response = await runtime.validate_stream(
                upstream_response,
                enabled=use_alias_candidate_probe,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                request=request,
                request_body=translated_request_body,
                upstream_url=str(target_url),
                provider=provider,
            )
        if not client_requested_stream:
            response_body = await runtime.collect_stream(upstream_response)
            translated_response = runtime.build_response(
                response_body,
                use_codex_native_tools=use_codex_native_tools,
                retryable_failed_response=use_alias_candidate_probe,
                failed_response_adapter_model=adapter_model,
                failed_response_adapter=adapter,
                failed_response_adapter_label=adapter_label,
                malformed_intake_context=malformed_context,
                **builder_kwargs,
            )
            runtime.copy_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return runtime.build_streaming_response(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
            use_codex_native_tools=use_codex_native_tools,
            **stream_kwargs,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(status_code=502, detail=unexpected_detail)

    response_body = json.loads(runtime.decode_response_body(upstream_response.body))
    translated_response = runtime.build_response(
        response_body,
        use_codex_native_tools=use_codex_native_tools,
        retryable_failed_response=use_alias_candidate_probe,
        failed_response_adapter_model=adapter_model,
        failed_response_adapter=adapter,
        failed_response_adapter_label=adapter_label,
        malformed_intake_context=malformed_context,
        **builder_kwargs,
    )
    runtime.copy_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response
