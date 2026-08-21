"""RR-105 / RR-106 / RR-107: watermark stream policy, incremental SSE, endpoints.

Focused failing tests. Do not implement production watermark code here.
Shipped default remains ``mode=off`` except in tests that opt in.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError


ZWSP = "\u200b"
ZWSP_UTF8 = ZWSP.encode("utf-8")


def _load_text_watermark_config(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )

    return load_text_watermark_config(*args, **kwargs)


def _apply_watermark_policy(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_watermark_policy,
    )

    return apply_watermark_policy(*args, **kwargs)


def _apply_request_watermark_intake(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_request_watermark_intake,
    )

    return apply_request_watermark_intake(*args, **kwargs)


def _apply_request_watermark_egress(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_request_watermark_egress,
    )

    return apply_request_watermark_egress(*args, **kwargs)


def _apply_nonstream_hook(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.response_hooks import (
        maybe_apply_passthrough_watermark_response,
    )

    return maybe_apply_passthrough_watermark_response(*args, **kwargs)


def _wrap_stream_hook(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.response_hooks import (
        maybe_wrap_passthrough_watermark_responses_stream,
    )

    return maybe_wrap_passthrough_watermark_responses_stream(*args, **kwargs)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _optional_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _empty_success_handler_kwargs() -> dict[str, Any]:
    return {"litellm_params": {"metadata": {}}}


def _metadata_from_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = kwargs.get("litellm_params")
    if not isinstance(params, dict):
        return {}
    metadata = params.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _detect_config(**overrides: Any) -> Any:
    payload: dict[str, Any] = {
        "mode": "detect",
        "unicode": {
            "enabled": True,
            "policy": "conservative",
            "normalize_spaces": True,
            "nfkc": False,
        },
        "removal": {
            "enabled": False,
            "stream_policy": "audit_only",
            "on_unremovable": "allow",
        },
        "statistical_detectors": [],
    }
    payload.update(overrides)
    return _load_text_watermark_config(payload)


def _sanitize_config(**overrides: Any) -> Any:
    payload: dict[str, Any] = {
        "mode": "sanitize",
        "unicode": {
            "enabled": True,
            "policy": "conservative",
            "normalize_spaces": True,
            "nfkc": False,
        },
        "removal": {
            "enabled": True,
            "stream_policy": "audit_only",
            "on_unremovable": "allow",
        },
        "statistical_detectors": [],
    }
    payload.update(overrides)
    return _load_text_watermark_config(payload)


def _enforce_buffer_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "enforce",
        "unicode": {
            "enabled": True,
            "policy": "conservative",
            "normalize_spaces": True,
            "nfkc": False,
        },
        "removal": {
            "enabled": True,
            "stream_policy": "buffer_response",
            "on_unremovable": "allow",
        },
        "statistical_detectors": [],
    }
    payload.update(overrides)
    return payload


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    body = {"type": event_type, **payload}
    return (
        f"event: {event_type}\ndata: "
        + json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        + "\n\n"
    ).encode("utf-8")


def _delta_event(visible: str) -> bytes:
    return _sse_event(
        "response.output_text.delta",
        {
            "item_id": "msg_rr105",
            "output_index": 0,
            "content_index": 0,
            "delta": visible,
        },
    )


def _done_event(visible: str) -> bytes:
    return _sse_event(
        "response.output_text.done",
        {
            "item_id": "msg_rr105",
            "output_index": 0,
            "content_index": 0,
            "text": visible,
        },
    )


def _completed_event(visible: str) -> bytes:
    return _sse_event(
        "response.completed",
        {
            "response": {
                "id": "resp_rr105",
                "status": "completed",
                "output_text": visible,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": visible}],
                    }
                ],
            }
        },
    )


def _watermarked_sse_chunks(*, visible: str | None = None) -> list[bytes]:
    text = f"hello{ZWSP}world" if visible is None else visible
    return [
        _delta_event(text),
        _done_event(text),
        _completed_event(text),
        b"data: [DONE]\n\n",
    ]


def _split_bytes(raw: bytes, *cuts: int) -> list[bytes]:
    parts: list[bytes] = []
    start = 0
    for cut in cuts:
        parts.append(raw[start:cut])
        start = cut
    parts.append(raw[start:])
    return [part for part in parts if part]


async def _collect_wrapped(wrapped: Any) -> list[bytes]:
    yielded: list[bytes] = []
    async for chunk in wrapped:
        if isinstance(chunk, (bytes, bytearray)):
            yielded.append(bytes(chunk))
        else:
            yielded.append(str(chunk).encode("utf-8"))
    return yielded


def _responses_output_body(*, visible: str) -> dict[str, Any]:
    return {
        "id": "resp_rr107",
        "object": "response",
        "status": "completed",
        "output_text": visible,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": visible}],
            }
        ],
    }


def _chat_request_body(*, visible: str) -> dict[str, Any]:
    return {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "System visible text."},
            {"role": "user", "content": visible},
        ],
    }


def _chat_response_body(*, visible: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_rr107",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": visible},
            }
        ],
    }


def _extract_nonstream_body(returned: Any, original: dict[str, Any]) -> dict[str, Any]:
    if returned is None:
        return original
    if isinstance(returned, tuple) and returned:
        first = returned[0]
        if isinstance(first, dict):
            return first
    body = _optional_field(returned, "body", None)
    if isinstance(body, dict):
        return body
    return original


# ---------------------------------------------------------------------------
# Shared default: mode remains off unless a test opts in
# ---------------------------------------------------------------------------


def test_rr105_107_shipped_default_mode_is_off() -> None:
    cfg = _load_text_watermark_config(None)
    assert _field(cfg, "mode") == "off"
    assert _field(_field(cfg, "removal"), "enabled") is False
    assert _field(_field(cfg, "removal"), "stream_policy") == "audit_only"


# ---------------------------------------------------------------------------
# RR-105: accepted sanitize/enforce + buffer_response must not silently
# downgrade live SSE to audit_only while still claiming the named mode.
# ---------------------------------------------------------------------------


def test_rr105_sanitize_with_buffer_response_is_accepted_or_rejected_explicitly() -> None:
    payload = {
        "mode": "sanitize",
        "removal": {
            "enabled": True,
            "stream_policy": "buffer_response",
        },
    }
    try:
        cfg = _load_text_watermark_config(payload)
    except (ValueError, TypeError, ValidationError) as exc:
        message = str(exc).lower()
        assert "buffer_response" in message or "stream" in message or "sanitize" in message
        return
    assert _field(cfg, "mode") == "sanitize"
    assert _field(_field(cfg, "removal"), "stream_policy") == "buffer_response"


@pytest.mark.asyncio
async def test_rr105_streaming_sanitize_with_buffer_response_sanitizes_or_config_rejects() -> None:
    payload = {
        "mode": "sanitize",
        "unicode": {"enabled": True, "policy": "conservative"},
        "removal": {
            "enabled": True,
            "stream_policy": "buffer_response",
            "on_unremovable": "allow",
        },
        "statistical_detectors": [],
    }
    try:
        config = _load_text_watermark_config(payload)
    except (ValueError, TypeError, ValidationError):
        return

    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=config,
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded = await _collect_wrapped(wrapped)
    joined = b"".join(yielded)
    assert ZWSP_UTF8 not in joined, (
        "sanitize + buffer_response must buffer/sanitize streamed visible text; "
        "silently forwarding carriers while claiming sanitize is forbidden"
    )
    assert b"hello" in joined
    assert b"world" in joined
    audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
    if audit is not None:
        assert _field(audit, "mode") == "sanitize"
        assert _field(audit, "mode") != "audit_only"


@pytest.mark.asyncio
async def test_rr105_streaming_enforce_with_buffer_response_does_not_deliver_carriers() -> None:
    config = _load_text_watermark_config(_enforce_buffer_payload())
    assert _field(config, "mode") == "enforce"
    assert _field(_field(config, "removal"), "stream_policy") == "buffer_response"

    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=config,
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded: list[bytes] = []
    blocked: BaseException | None = None
    try:
        yielded = await _collect_wrapped(wrapped)
    except BaseException as exc:  # HTTPException / stream abort after buffering
        blocked = exc

    joined = b"".join(yielded)
    assert ZWSP_UTF8 not in joined, (
        "enforce + buffer_response must buffer then sanitize or reject; "
        "live SSE must not silently keep the named mode while emitting carriers"
    )
    if blocked is None:
        assert b"hello" in joined
        assert b"world" in joined
        audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
        if audit is not None:
            assert _field(audit, "mode") == "enforce"
    else:
        status = getattr(blocked, "status_code", None)
        assert status in {400, 403, 409, 422, None} or "watermark" in str(blocked).lower()


@pytest.mark.asyncio
async def test_rr105_detect_audit_only_still_forwards_live_sse_unchanged() -> None:
    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_detect_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded = await _collect_wrapped(wrapped)
    assert yielded == chunks
    assert ZWSP_UTF8 in b"".join(yielded)


def test_rr105_stream_wrapper_must_not_force_named_sanitize_enforce_to_audit_only() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark import response_hooks

    wrap_source = inspect.getsource(
        response_hooks.maybe_wrap_passthrough_watermark_responses_stream
    )
    helper_source = inspect.getsource(
        response_hooks._wrap_passthrough_watermark_responses_stream_audit_only
    )
    joined = wrap_source + "\n" + helper_source
    assert "stream_policy = \"audit_only\"" not in wrap_source.replace("'", '"')
    assert "stream_policy = 'audit_only'" not in wrap_source
    silent_downgrade = (
        'stream_policy != "audit_only"' in wrap_source
        or "stream_policy != 'audit_only'" in wrap_source
    ) and (
        'stream_policy = "audit_only"' in wrap_source
        or "stream_policy = 'audit_only'" in wrap_source
        or "stream_policy = \"audit_only\"" in joined
    )
    assert silent_downgrade is False, (
        "maybe_wrap_passthrough_watermark_responses_stream must not silently "
        "force sanitize/enforce buffer_response down to audit_only"
    )


# ---------------------------------------------------------------------------
# RR-106: split UTF-8 / SSE / JSON still detect; retained text is bounded;
# reuse _iter_sse_event_blocks_with_separator.
# ---------------------------------------------------------------------------


def _split_carrier_sse_http_chunks() -> list[bytes]:
    """One SSE event whose UTF-8, field boundary, and JSON are split across HTTP chunks."""

    visible = f"hello{ZWSP}world"
    event = _delta_event(visible)
    zwsp_at = event.index(ZWSP_UTF8)
    # Split inside the 3-byte ZWSP, then again across the SSE field/JSON remainder.
    data_at = event.index(b"data:")
    return _split_bytes(event, data_at, data_at + 5, zwsp_at + 1, zwsp_at + 2)


@pytest.mark.asyncio
async def test_rr106_split_utf8_sse_fields_and_json_still_detect() -> None:
    chunks = _split_carrier_sse_http_chunks()
    assert len(chunks) >= 4
    assert all(ZWSP_UTF8 not in chunk for chunk in chunks), (
        "fixture must not leave a complete U+200B inside any single HTTP chunk"
    )
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_detect_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded = await _collect_wrapped(wrapped)
    assert b"".join(yielded)  # stream still completes
    audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
    assert audit is not None, (
        "carrier split across UTF-8 code points, SSE fields, and JSON must still "
        "be detected; per-HTTP-chunk decode is not enough"
    )
    assert _field(audit, "signal_detected") is True


@pytest.mark.asyncio
async def test_rr106_reuses_incremental_sse_splitter_contract() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _iter_sse_event_blocks_with_separator as real_splitter,
    )

    calls: list[int] = []

    async def _tracking_splitter(body_iterator: Any):
        calls.append(1)
        async for item in real_splitter(body_iterator):
            yield item

    chunks = _split_carrier_sse_http_chunks()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse._iter_sse_event_blocks_with_separator",
        _tracking_splitter,
    ):
        wrapped = _wrap_stream_hook(
            _gen(),
            config=_detect_config(),
            success_handler_kwargs=_empty_success_handler_kwargs(),
            endpoint="responses",
        )
        await _collect_wrapped(wrapped)

    assert calls, (
        "stream wrapper must iterate via _iter_sse_event_blocks_with_separator "
        "instead of decoding each HTTP chunk independently"
    )


@pytest.mark.asyncio
async def test_rr106_retained_clean_text_honors_max_text_bytes_per_direction() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark import (
        response_hooks,
    )

    limit = 64
    config = _detect_config(
        limits={
            "max_text_bytes_per_direction": limit,
            "max_text_nodes_per_direction": 256,
            "max_reported_paths": 32,
            "max_reported_hits_per_path": 16,
        }
    )
    seen_bodies: list[str] = []
    real_apply = response_hooks.apply_watermark_policy

    def _tracking_apply(body: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(body, Mapping):
            text = body.get("output_text")
            if isinstance(text, str):
                seen_bodies.append(text)
            output = body.get("output")
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, Mapping):
                        continue
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                            seen_bodies.append(part["text"])
        return real_apply(body, *args, **kwargs)

    prefix = "clean-visible-text-" * 20  # well over 64 bytes
    chunks = [
        _delta_event(prefix),
        _delta_event(f"{ZWSP}tail"),
        b"data: [DONE]\n\n",
    ]

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    with patch.object(response_hooks, "apply_watermark_policy", _tracking_apply):
        wrapped = _wrap_stream_hook(
            _gen(),
            config=config,
            success_handler_kwargs=_empty_success_handler_kwargs(),
            endpoint="responses",
        )
        await _collect_wrapped(wrapped)

    assert seen_bodies, (
        "detect mode should still attach a policy audit when a later chunk carries "
        "a signal; retained clean text must be passed through a bounded buffer"
    )
    for text in seen_bodies:
        assert len(text.encode("utf-8")) <= limit, (
            "retained clean/visible stream text must honor "
            "limits.max_text_bytes_per_direction; unbounded accumulated += is forbidden"
        )


@pytest.mark.asyncio
async def test_rr106_mode_off_does_not_scan_split_stream() -> None:
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in _split_carrier_sse_http_chunks():
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_load_text_watermark_config(None),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded = await _collect_wrapped(wrapped)
    assert _metadata_from_kwargs(kwargs).get("watermark_output_audit") is None
    assert yielded  # original iterator still yields


# ---------------------------------------------------------------------------
# RR-107: endpoints membership gates request and response policy.
# ---------------------------------------------------------------------------


def _responses_only_detect_config() -> Any:
    return _detect_config(endpoints=["responses"])


def _responses_only_sanitize_config() -> Any:
    return _sanitize_config(endpoints=["responses"])


def test_rr107_included_responses_request_receives_configured_mode() -> None:
    original = {
        "model": "gpt-5.4",
        "instructions": f"Follow{ZWSP}the user.",
        "input": "visible user text",
    }
    harness = copy.deepcopy(original)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {}
    config = _responses_only_sanitize_config()

    intake = _apply_request_watermark_intake(
        body=harness,
        config=config,
        endpoint="responses",
        direction="request",
    )
    egress = _apply_request_watermark_egress(
        body=provider_bound,
        intake=intake,
        config=config,
        endpoint="responses",
        direction="request",
        metadata=metadata,
        litellm_metadata={},
    )
    out_body = _optional_field(egress, "body", provider_bound)
    assert isinstance(out_body, dict)
    assert ZWSP not in out_body["instructions"]
    assert metadata.get("watermark_input_audit") is not None
    assert _field(metadata["watermark_input_audit"], "mode") == "sanitize"


def test_rr107_excluded_chat_completions_request_is_unchanged() -> None:
    original = _chat_request_body(visible=f"user{ZWSP}question")
    harness = copy.deepcopy(original)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {}
    config = _responses_only_sanitize_config()

    intake = _apply_request_watermark_intake(
        body=harness,
        config=config,
        endpoint="chat_completions",
        direction="request",
    )
    egress = _apply_request_watermark_egress(
        body=provider_bound,
        intake=intake,
        config=config,
        endpoint="/openai/v1/chat/completions",
        direction="request",
        metadata=metadata,
        litellm_metadata={},
    )
    out_body = _optional_field(egress, "body", provider_bound)
    assert harness == original
    assert out_body == original
    assert ZWSP in out_body["messages"][1]["content"]
    assert _optional_field(intake, "audit") is None
    assert _optional_field(egress, "audit") is None
    assert metadata.get("watermark_input_audit") is None


def test_rr107_included_responses_nonstream_response_receives_configured_mode() -> None:
    visible = f"hello{ZWSP}world"
    body = _responses_output_body(visible=visible)
    original = copy.deepcopy(body)
    kwargs = _empty_success_handler_kwargs()
    returned = _apply_nonstream_hook(
        copy.deepcopy(body),
        content=json.dumps(original, ensure_ascii=False).encode("utf-8"),
        config=_responses_only_sanitize_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    out_body = _extract_nonstream_body(returned, original)
    assert ZWSP not in out_body["output_text"]
    audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
    assert audit is not None
    assert _field(audit, "mode") == "sanitize"


def test_rr107_excluded_chat_completions_nonstream_response_is_unchanged() -> None:
    visible = f"hello{ZWSP}world"
    body = _chat_response_body(visible=visible)
    original = copy.deepcopy(body)
    original_content = json.dumps(original, ensure_ascii=False).encode("utf-8")
    kwargs = _empty_success_handler_kwargs()
    returned = _apply_nonstream_hook(
        copy.deepcopy(body),
        content=original_content,
        config=_responses_only_sanitize_config(),
        success_handler_kwargs=kwargs,
        endpoint="chat_completions",
    )
    out_body = _extract_nonstream_body(returned, original)
    assert out_body == original
    assert ZWSP in out_body["choices"][0]["message"]["content"]
    assert _metadata_from_kwargs(kwargs).get("watermark_output_audit") is None


def test_rr107_policy_helper_skips_excluded_endpoint() -> None:
    body = _chat_response_body(visible=f"hello{ZWSP}world")
    original = copy.deepcopy(body)
    result = _apply_watermark_policy(
        body=body,
        config=_responses_only_detect_config(),
        direction="response",
        endpoint="chat_completions",
    )
    assert _optional_field(result, "audit") is None
    assert _optional_field(result, "body", body) == original
    assert ZWSP in body["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_rr107_excluded_endpoint_stream_is_unchanged() -> None:
    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_responses_only_detect_config(),
        success_handler_kwargs=kwargs,
        endpoint="chat_completions",
    )
    yielded = await _collect_wrapped(wrapped)
    assert yielded == chunks
    assert ZWSP_UTF8 in b"".join(yielded)
    assert _metadata_from_kwargs(kwargs).get("watermark_output_audit") is None


def test_rr107_mode_off_ignores_endpoints_membership() -> None:
    body = {
        "instructions": f"hello{ZWSP}world",
        "input": "visible user text",
    }
    original = copy.deepcopy(body)
    result = _apply_watermark_policy(
        body=body,
        config=_load_text_watermark_config(
            {"mode": "off", "endpoints": ["responses", "chat_completions"]}
        ),
        direction="request",
        endpoint="responses",
    )
    assert _optional_field(result, "audit") is None
    assert _optional_field(result, "body", body) == original
