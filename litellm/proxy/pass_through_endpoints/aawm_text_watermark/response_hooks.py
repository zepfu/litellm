"""CFG-028 response-path watermark hooks (non-stream + live SSE audit)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

# CFG-028 import contract. Do not import aawm_adapter_runtime.sse at
# module top: aawm_adapter_runtime/__init__.py pulls anthropic_adapter_calls
# -> pass_through_endpoints and deadlocks this module. Lazy-import the
# splitter inside maybe_wrap_passthrough_watermark_responses_stream.

from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
    OpenAIPassthroughTextWatermarkSettings,
    load_text_watermark_config,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
    apply_watermark_policy,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.unicode_detector import (
    detect_unicode_carriers,
)

_OUTPUT_AUDIT_KEY = "watermark_output_audit"
_VISIBLE_DELTA_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.output_text.done",
        "response.completed",
    }
)


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


def _visible_texts_from_responses_sse_chunk(chunk: Any) -> list[str]:
    if isinstance(chunk, (bytes, bytearray)):
        try:
            text = bytes(chunk).decode("utf-8")
        except UnicodeDecodeError:
            return []
    else:
        text = str(chunk)
    texts: list[str] = []
    event_type = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        parsed_type = payload.get("type")
        if isinstance(parsed_type, str) and parsed_type:
            event_type = parsed_type
        if event_type not in _VISIBLE_DELTA_TYPES:
            continue
        texts.extend(_visible_texts_from_event(payload, event_type))
    return texts


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
    if loaded.mode == "off" or not loaded.directions.response:
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


async def _wrap_passthrough_watermark_responses_stream_audit_only(
    iterator: Any,
    *,
    config: OpenAIPassthroughTextWatermarkSettings,
    success_handler_kwargs: Any,
    endpoint: str,
    stream_policy: str,
) -> AsyncIterator[Any]:
    """Yield original SSE chunks; attach output audit during iteration."""

    del stream_policy
    accumulated = ""
    attached = False
    async for chunk in iterator:
        if not attached and config.unicode.enabled:
            for visible in _visible_texts_from_responses_sse_chunk(chunk):
                accumulated += visible
                detection = detect_unicode_carriers(
                    visible,
                    policy=config.unicode.policy,
                    normalize_spaces=config.unicode.normalize_spaces,
                )
                if not detection.signal_detected:
                    continue
                synthetic = {
                    "output_text": accumulated,
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": accumulated}
                            ],
                        }
                    ],
                }
                result = apply_watermark_policy(
                    synthetic,
                    config,
                    direction="response",
                    endpoint=endpoint,
                )
                audit = result.audit or _minimal_output_audit(
                    mode=config.mode, signal_detected=True
                )
                _attach_output_audit(success_handler_kwargs, audit)
                attached = True
                break
        yield chunk


def maybe_wrap_passthrough_watermark_responses_stream(
    iterator: Any,
    config: Any = None,
    success_handler_kwargs: Any = None,
    endpoint: str = "responses",
    **kwargs: Any,
) -> Any:
    """Wrap a Responses SSE iterator for detect/audit_only output audits.

    Live SSE bytes are never rewritten. ``stream_policy=audit_only`` (and
    detect / mode-off) yield the original chunks unchanged. Enforce without
    ``buffer_response`` is a CFG-026 config-load error, not a rewrite here.
    ``_iter_sse_event_blocks_with_separator`` is imported for the existing
    SSE splitter contract; this wrapper still yields the caller's chunks.
    """

    del kwargs
    loaded = _coerce_config(config)
    if loaded.mode == "off" or not loaded.directions.response:
        return iterator

    stream_policy = loaded.removal.stream_policy
    if stream_policy != "audit_only" and loaded.mode in {"sanitize", "enforce"}:
        # Do not reconstruct or sanitize live SSE; audit-only observation only.
        stream_policy = "audit_only"

    try:
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
            _iter_sse_event_blocks_with_separator,
        )

        _sse_iter = _iter_sse_event_blocks_with_separator
    except Exception:
        _sse_iter = None
    del _sse_iter  # referenced for CFG-028; live SSE is not reconstructed

    return _wrap_passthrough_watermark_responses_stream_audit_only(
        iterator,
        config=loaded,
        success_handler_kwargs=success_handler_kwargs,
        endpoint=endpoint,
        stream_policy=stream_policy,
    )
