"""CFG-028 response-path watermark hooks (non-stream + live SSE audit)."""

from __future__ import annotations

import copy
import json
import inspect
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Optional

from fastapi import HTTPException

# CFG-028 import contract. Do not import aawm_adapter_runtime.sse at
# module top: aawm_adapter_runtime/__init__.py pulls anthropic_adapter_calls
# -> pass_through_endpoints and deadlocks this module. Lazy-import the
# splitter inside maybe_wrap_passthrough_watermark_responses_stream.

from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
    OpenAIPassthroughTextWatermarkSettings,
    load_text_watermark_config,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
    _policy_applies,
    apply_watermark_policy,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.unicode_detector import (
    detect_unicode_carriers,
    sanitize_unicode_carriers,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_events import (
    ParsedResponsesEvent,
    ResponsesEventPipeline,
    attach_responses_emissions,
    attach_responses_events,
    responses_emissions_from_chunk,
    responses_events_from_chunk,
)

_OUTPUT_AUDIT_KEY = "watermark_output_audit"
_VISIBLE_DELTA_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.output_text.done",
        "response.completed",
    }
)
_MUTATING_STREAM_POLICIES = frozenset({"buffer_response", "buffer_text_item"})
_UNIMPLEMENTED_STREAM_POLICIES = frozenset({"safe_subset", "buffer_text_item"})


def _coerce_config(config: Any) -> OpenAIPassthroughTextWatermarkSettings:
    if isinstance(config, OpenAIPassthroughTextWatermarkSettings):
        return config
    return load_text_watermark_config(config)


def _attach_output_audit(success_handler_kwargs: Any, audit: Any) -> None:
    if audit is None or not isinstance(success_handler_kwargs, dict):
        return
    params = success_handler_kwargs.setdefault("litellm_params", {})
    if not isinstance(params, dict):
        return
    metadata = params.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata[_OUTPUT_AUDIT_KEY] = audit


def _minimal_output_audit(*, mode: str, signal_detected: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "direction": "response",
        "mode": mode,
        "status": "detected" if signal_detected else "clean",
        "signal_detected": signal_detected,
        "confirmed_watermark_detected": False,
        "vendor_attribution": "unknown",
    }


def _policy_mutates(config: OpenAIPassthroughTextWatermarkSettings) -> bool:
    return config.mode in {"sanitize", "enforce"} and bool(config.removal.enabled)


def _reserialize_content(body: dict[str, Any], content: Any) -> Any:
    if isinstance(content, (bytes, bytearray)):
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    if isinstance(content, str):
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return content


def _visible_texts_from_event(payload: Mapping[str, Any], event_type: str) -> list[str]:
    texts: list[str] = []
    if event_type == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str):
            texts.append(delta)
        return texts
    if event_type == "response.output_text.done":
        done_text = payload.get("text")
        if isinstance(done_text, str):
            texts.append(done_text)
        return texts
    if event_type != "response.completed":
        return texts
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return texts
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        texts.append(output_text)
    output = response.get("output")
    if not isinstance(output, list):
        return texts
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return texts


def _event_payload_and_type(event: Any) -> tuple[Optional[Mapping[str, Any]], str]:
    if isinstance(event, ParsedResponsesEvent):
        return event.payload, event.event_type or ""
    if isinstance(event, Mapping):
        event_type = event.get("type")
        return event, event_type if isinstance(event_type, str) else ""
    payload = getattr(event, "payload", None)
    event_type = getattr(event, "event_type", None) or getattr(event, "type", None)
    if isinstance(payload, Mapping):
        return payload, event_type if isinstance(event_type, str) else ""
    return None, ""


def _visible_texts_from_sse_block(
    block: Any,
    *,
    parse_sse_block: Optional[Callable[[str], list[Any]]] = None,
) -> list[str]:
    texts: list[str] = []
    associated_emissions = responses_emissions_from_chunk(block)
    if associated_emissions is not None:
        for emission in associated_emissions:
            payload = emission.effective_payload
            event_type = emission.source_event.event_type or ""
            if (
                not isinstance(payload, Mapping)
                or not event_type
                or event_type not in _VISIBLE_DELTA_TYPES
            ):
                continue
            texts.extend(_visible_texts_from_event(payload, event_type))
        return texts
    parser = parse_sse_block or _parse_responses_sse_block
    for parsed_event in parser(block):
        payload, event_type = _event_payload_and_type(parsed_event)
        if (
            not isinstance(payload, Mapping)
            or event_type not in _VISIBLE_DELTA_TYPES
            or (
                isinstance(parsed_event, ParsedResponsesEvent)
                and (parsed_event.is_malformed or parsed_event.is_done_marker)
            )
        ):
            continue
        texts.extend(_visible_texts_from_event(payload, event_type))
    return texts


def _sanitize_visible_payload_text(
    text: str, config: OpenAIPassthroughTextWatermarkSettings
) -> str:
    if not config.unicode.enabled:
        return text
    return sanitize_unicode_carriers(
        text,
        policy=config.unicode.policy,
        normalize_spaces=config.unicode.normalize_spaces,
        nfkc=config.unicode.nfkc,
    ).text


def _rewrite_visible_sse_payload(
    payload: dict[str, Any],
    event_type: str,
    config: OpenAIPassthroughTextWatermarkSettings,
) -> dict[str, Any]:
    if event_type == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str):
            payload["delta"] = _sanitize_visible_payload_text(delta, config)
        return payload
    if event_type == "response.output_text.done":
        done_text = payload.get("text")
        if isinstance(done_text, str):
            payload["text"] = _sanitize_visible_payload_text(done_text, config)
        return payload
    if event_type != "response.completed":
        return payload
    response = payload.get("response")
    if not isinstance(response, dict):
        return payload
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        response["output_text"] = _sanitize_visible_payload_text(
            output_text, config
        )
    output = response.get("output")
    if not isinstance(output, list):
        return payload
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = _sanitize_visible_payload_text(part["text"], config)
    return payload


def _rewrite_visible_sse_emission(
    emission: Any,
    config: OpenAIPassthroughTextWatermarkSettings,
) -> Any:
    payload = emission.effective_payload
    event_type = emission.source_event.event_type or ""
    if not isinstance(payload, dict) or event_type not in _VISIBLE_DELTA_TYPES:
        return emission
    rewritten_payload = _rewrite_visible_sse_payload(
        copy.deepcopy(payload),
        event_type,
        config,
    )
    if rewritten_payload == payload:
        return emission
    return emission.transformed(rewritten_payload)


def _rewrite_visible_sse_block(
    event_block: Any, config: OpenAIPassthroughTextWatermarkSettings
) -> Any:
    associated_emissions = responses_emissions_from_chunk(event_block)
    if associated_emissions is None:
        return event_block
    rewritten = [
        _rewrite_visible_sse_emission(emission, config)
        for emission in associated_emissions
    ]
    if rewritten == associated_emissions:
        return event_block
    return attach_responses_emissions(b"", rewritten)


def _parse_responses_sse_block(block: str) -> list[Any]:
    """Parse one complete SSE block with configured event bounds."""
    parsed_events = responses_events_from_chunk(block)
    if parsed_events is None:
        parsed_events = getattr(block, "aawm_parsed_responses_events", None)
    if isinstance(parsed_events, list):
        return parsed_events

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_events import (
        ResponsesEventPipeline,
        resolve_responses_event_limits,
    )

    pipeline = ResponsesEventPipeline(
        max_event_bytes=resolve_responses_event_limits()[0],
        max_raw_events=0,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _sse_block_wire_bytes,
    )

    encoded = _sse_block_wire_bytes(block)
    if not encoded.endswith((b"\r\n\r\n", b"\n\n", b"\r\r")):
        encoded += b"\n\n"
    return pipeline.feed(encoded)


async def _associate_unassociated_watermark_iterator(
    iterator: Any,
    *,
    event_pipeline: Optional[ResponsesEventPipeline],
) -> Any:
    """Require one parsing/association boundary before buffered mutation."""
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_events import (
        responses_events_from_chunk,
    )

    pipeline = event_pipeline

    def _associated_frames(events: list[ParsedResponsesEvent]) -> bytes:
        return attach_responses_events(
            b"".join(
                event.forward_event or event.raw_event or b""
                for event in events
            ),
            events,
        )

    async def _wrapped():
        nonlocal pipeline
        async for chunk in iterator:
            if responses_events_from_chunk(chunk) is not None:
                yield chunk
                continue
            if pipeline is None:
                pipeline = ResponsesEventPipeline(max_raw_events=0)
            data = (
                chunk
                if isinstance(chunk, (bytes, bytearray))
                else str(chunk).encode("utf-8")
            )
            events = pipeline.feed(bytes(data))
            if events:
                yield _associated_frames(events)
        if pipeline is not None:
            leftover = pipeline.finish()
            if leftover:
                yield _associated_frames(leftover)

    return _wrapped()


def _encode_sse_block(event_block: Any, had_separator: bool) -> bytes:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _sse_block_wire_bytes,
    )

    wire = _sse_block_wire_bytes(event_block, had_separator)
    associated_emissions = responses_emissions_from_chunk(event_block)
    if associated_emissions is None:
        return wire
    return attach_responses_emissions(wire, associated_emissions)


def _append_bounded_visible_text(
    retained: str, visible: str, max_bytes: int
) -> str:
    if max_bytes <= 0:
        return ""
    combined = retained + visible
    encoded = combined.encode("utf-8")
    if len(encoded) <= max_bytes:
        return combined
    truncated = encoded[-max_bytes:]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[1:]
    return ""


def _append_required_visible_text(
    retained: str,
    visible: str,
    max_bytes: int,
) -> str:
    """Append required validation text or fail closed at the configured bound."""
    if not visible:
        return retained
    combined = retained + visible
    if max_bytes <= 0 or len(combined.encode("utf-8")) > max_bytes:
        raise ValueError(
            "Responses watermark visible-text validation exceeded configured bound"
        )
    return combined


def _synthetic_visible_body(text: str) -> dict[str, Any]:
    return {
        "output_text": text,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def _audit_retained_visible_text(
    retained: str,
    config: OpenAIPassthroughTextWatermarkSettings,
    endpoint: str,
) -> Optional[dict[str, Any]]:
    if not retained:
        return None
    result = apply_watermark_policy(
        _synthetic_visible_body(retained),
        config,
        direction="response",
        endpoint=endpoint,
    )
    return result.audit or _minimal_output_audit(
        mode=config.mode, signal_detected=True
    )


def maybe_apply_passthrough_watermark_response(
    body: Any,
    content: Any = None,
    config: Any = None,
    success_handler_kwargs: Any = None,
    endpoint: str = "responses",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Detect/sanitize visible Responses JSON. ``mode=off`` is a no-op."""

    del kwargs
    loaded = _coerce_config(config)
    if not _policy_applies(loaded, "response", endpoint):
        return (body, content)

    working_body = body
    if not isinstance(working_body, dict) and isinstance(
        content, (bytes, bytearray, str)
    ):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            working_body = parsed
    if not isinstance(working_body, dict):
        return (body, content)

    result = apply_watermark_policy(
        working_body,
        loaded,
        direction="response",
        endpoint=endpoint,
    )
    if result.audit is not None:
        _attach_output_audit(success_handler_kwargs, result.audit)

    if not _policy_mutates(loaded):
        return (body, content)
    mutated = result.body if isinstance(result.body, dict) else working_body
    return (mutated, _reserialize_content(mutated, content))


def _raise_unimplemented_stream_policy(
    config: OpenAIPassthroughTextWatermarkSettings,
) -> None:
    raise ValueError(
        "openai_passthrough_text_watermark removal.stream_policy="
        f"{config.removal.stream_policy!r} is not implemented for "
        f"mode={config.mode!r}; use audit_only or buffer_response"
    )


def _raise_unremovable_stream(audit: Optional[dict[str, Any]]) -> None:
    exception = HTTPException(
        status_code=409,
        detail={"watermark_output_audit": audit or {"mode": "enforce"}},
    )
    setattr(
        exception,
        "_aawm_policy_failure",
        {
            "policy_failure_kind": "watermark_output_rejected",
            "policy_failure_code": "aawm_watermark_output_rejected",
            "policy_failure_class": "output_policy",
        },
    )
    raise exception


async def _wrap_passthrough_watermark_responses_stream_audit_only(
    iterator: Any,
    *,
    config: OpenAIPassthroughTextWatermarkSettings,
    success_handler_kwargs: Any,
    endpoint: str,
    stream_policy: str,
    sse_iter: Any,
    parse_sse_block: Optional[Callable[[str], list[Any]]] = None,
) -> AsyncIterator[Any]:
    """Yield original SSE blocks; attach bounded output audit during iteration."""

    del stream_policy
    retained = ""
    attached = False
    max_bytes = config.limits.max_text_bytes_per_direction
    async for event_block, had_separator in sse_iter(iterator):
        if not attached and config.unicode.enabled:
            for visible in _visible_texts_from_sse_block(
                event_block,
                parse_sse_block=parse_sse_block,
            ):
                retained = _append_bounded_visible_text(
                    retained, visible, max_bytes
                )
                detection = detect_unicode_carriers(
                    visible,
                    policy=config.unicode.policy,
                    normalize_spaces=config.unicode.normalize_spaces,
                )
                if not detection.signal_detected:
                    continue
                audit = _audit_retained_visible_text(retained, config, endpoint)
                if audit is not None:
                    _attach_output_audit(success_handler_kwargs, audit)
                    attached = True
                    break
        yield _encode_sse_block(event_block, had_separator)


async def _wrap_passthrough_watermark_responses_stream_buffer_response(
    iterator: Any,
    *,
    config: OpenAIPassthroughTextWatermarkSettings,
    success_handler_kwargs: Any,
    endpoint: str,
    sse_iter: Any,
    parse_sse_block: Optional[Callable[[str], list[Any]]] = None,
) -> AsyncIterator[Any]:
    """Buffer the complete SSE stream, sanitize/verify, then emit."""

    buffered: list[tuple[Any, bool]] = []
    retained = ""
    max_bytes = config.limits.max_text_bytes_per_direction
    buffered_bytes = 0
    max_stream_bytes = config.limits.max_sse_stream_bytes
    async for event_block, had_separator in sse_iter(iterator):
        encoded_block = _encode_sse_block(event_block, had_separator)
        buffered_bytes += len(encoded_block)
        if buffered_bytes > max_stream_bytes:
            raise ValueError("Responses watermark stream exceeded configured bound")
        buffered.append((event_block, had_separator))
        for visible in _visible_texts_from_sse_block(
            event_block,
            parse_sse_block=parse_sse_block,
        ):
            retained = _append_required_visible_text(retained, visible, max_bytes)

    audit = _audit_retained_visible_text(retained, config, endpoint)
    if audit is not None:
        _attach_output_audit(success_handler_kwargs, audit)

    rewrite = _policy_mutates(config)
    emitted_blocks: list[tuple[Any, bool]] = []
    emitted_visible = ""
    for event_block, had_separator in buffered:
        emitted = (
            _rewrite_visible_sse_block(event_block, config)
            if rewrite
            else event_block
        )
        for visible in _visible_texts_from_sse_block(
            emitted,
            parse_sse_block=parse_sse_block,
        ):
            emitted_visible = _append_required_visible_text(
                emitted_visible, visible, max_bytes
            )
        emitted_blocks.append((emitted, had_separator))

    post_hits = 0
    verification_text = emitted_visible if rewrite else retained
    if verification_text and config.unicode.enabled:
        cleaned = (
            verification_text
            if rewrite
            else _sanitize_visible_payload_text(verification_text, config)
        )
        post_hits = detect_unicode_carriers(
            cleaned,
            policy=config.unicode.policy,
            normalize_spaces=config.unicode.normalize_spaces,
        ).hit_count

    if config.mode == "enforce" and post_hits > 0:
        _raise_unremovable_stream(audit)

    for emitted, had_separator in emitted_blocks:
        yield _encode_sse_block(emitted, had_separator)


async def maybe_wrap_passthrough_watermark_responses_stream(
    iterator: Any,
    config: Any = None,
    success_handler_kwargs: Any = None,
    endpoint: str = "responses",
    event_pipeline: Optional[ResponsesEventPipeline] = None,
    **kwargs: Any,
) -> Any:
    """Wrap a Responses SSE iterator for detect/audit and buffered sanitize.

    ``mode=off`` and excluded endpoints return the original iterator.
    ``stream_policy=audit_only`` yields original chunks after incremental
    SSE reassembly. Named sanitize/enforce plus ``buffer_response`` buffer
    the complete stream, then sanitize or reject. Unimplemented stream
    policies fail closed instead of silently becoming audit-only.
    """

    del kwargs
    loaded = _coerce_config(config)
    if not _policy_applies(loaded, "response", endpoint):
        return iterator

    stream_policy = loaded.removal.stream_policy
    mutating = loaded.mode in {"sanitize", "enforce"} and loaded.removal.enabled
    if mutating and stream_policy in _UNIMPLEMENTED_STREAM_POLICIES:
        _raise_unimplemented_stream_policy(loaded)
    if mutating and stream_policy not in _MUTATING_STREAM_POLICIES:
        if stream_policy != "audit_only":
            _raise_unimplemented_stream_policy(loaded)

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _iter_sse_event_blocks_with_separator,
    )
    if inspect.iscoroutine(iterator):
        iterator = await iterator()
    if mutating and stream_policy == "buffer_response":
        iterator = _associate_unassociated_watermark_iterator(
            iterator,
            event_pipeline=event_pipeline,
        )
        return _wrap_passthrough_watermark_responses_stream_buffer_response(
            iterator,
            config=loaded,
            success_handler_kwargs=success_handler_kwargs,
            endpoint=endpoint,
            sse_iter=_iter_sse_event_blocks_with_separator,
            parse_sse_block=_parse_responses_sse_block,
        )

    return _wrap_passthrough_watermark_responses_stream_audit_only(
        iterator,
        config=loaded,
        success_handler_kwargs=success_handler_kwargs,
        endpoint=endpoint,
        stream_policy=stream_policy,
        sse_iter=_iter_sse_event_blocks_with_separator,
        parse_sse_block=_parse_responses_sse_block,
    )
