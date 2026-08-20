"""CFG-028: failing response-path watermark hook suite.

HTTP/stream install only. Detector/config units belong to CFG-026.
Request-path hooks belong to CFG-027. Do not implement production here.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError


ZWSP = "\u200b"
TOOL_ARG_LOCK = '{"cmd":"pwd\\u200b"}'
ENCRYPTED_LOCK = "CFG028_ENCRYPTED_REASONING_LOCK"

REPO_ROOT = Path(__file__).resolve().parents[4]
PASS_THROUGH_REQUEST_PATH = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/pass_through_endpoints.py"
)
STREAMING_HANDLER_PATH = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/streaming_handler.py"
)
REPETITIVE_OUTPUT_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/repetitive_output.py"
)
RESPONSE_UTILS_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/response_utils.py"
)
CODEX_CANDIDATE_CALLS_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/codex_candidate_calls.py"
)
WATERMARK_PACKAGE_DIR = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/aawm_text_watermark"
)
CFG025_WRAP_NAME = "maybe_wrap_passthrough_responses_stream"
CFG025_REJECT_NAME = "maybe_reject_passthrough_responses_body"
CFG028_NONSTREAM_HOOK = "maybe_apply_passthrough_watermark_response"
CFG028_STREAM_HOOK = "maybe_wrap_passthrough_watermark_responses_stream"
SSE_ITER_NAME = "_iter_sse_event_blocks_with_separator"
LOGGING_ROUTE_NAME = "_route_streaming_logging_to_handler"


def _load_text_watermark_config(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )

    return load_text_watermark_config(*args, **kwargs)


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


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _call_names_in_source_order(fn_source: str) -> list[str]:
    module = ast.parse(fn_source)
    names: list[tuple[int, int, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append((node.lineno, node.col_offset, func.id))
        elif isinstance(func, ast.Attribute):
            names.append((node.lineno, node.col_offset, func.attr))
    names.sort()
    return [item[2] for item in names]


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


def _off_config() -> Any:
    return _load_text_watermark_config(None)


def _empty_success_handler_kwargs() -> dict[str, Any]:
    return {"litellm_params": {"metadata": {}}}


def _metadata_from_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = kwargs.get("litellm_params")
    if not isinstance(params, dict):
        return {}
    metadata = params.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _extract_hook_body(returned: Any, original: dict[str, Any]) -> dict[str, Any]:
    if returned is None:
        return original
    if isinstance(returned, tuple) and returned:
        first = returned[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, (bytes, bytearray)):
            parsed = json.loads(bytes(first))
            assert isinstance(parsed, dict)
            return parsed
    if isinstance(returned, Mapping) and "output" in returned:
        return dict(returned)
    body = _optional_field(returned, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(returned, Mapping) and "body" in returned:
        nested = returned["body"]
        assert isinstance(nested, dict)
        return nested
    if isinstance(returned, (bytes, bytearray)):
        parsed = json.loads(bytes(returned))
        assert isinstance(parsed, dict)
        return parsed
    raise AssertionError(f"CFG-028 non-stream hook returned unsupported value {returned!r}")


def _extract_hook_content(returned: Any, original_content: bytes) -> bytes:
    if isinstance(returned, tuple) and len(returned) >= 2:
        content = returned[1]
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        if isinstance(content, str):
            return content.encode("utf-8")
    content = _optional_field(returned, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    return original_content


def _responses_output_body(*, visible: str) -> dict[str, Any]:
    return {
        "id": "resp_cfg028",
        "object": "response",
        "status": "completed",
        "output_text": visible,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": visible}],
            },
            {
                "type": "function_call",
                "id": "fc_keep",
                "call_id": "call_keep",
                "name": "bash",
                "arguments": TOOL_ARG_LOCK,
            },
            {
                "type": "reasoning",
                "id": "rs_keep",
                "encrypted_content": ENCRYPTED_LOCK,
            },
        ],
    }


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    body = {"type": event_type, **payload}
    return (
        f"event: {event_type}\ndata: "
        + json.dumps(body, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def _watermarked_sse_chunks() -> list[bytes]:
    visible = f"hello{ZWSP}world"
    return [
        _sse_event(
            "response.output_text.delta",
            {
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": visible,
            },
        ),
        _sse_event(
            "response.output_text.done",
            {
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": visible,
            },
        ),
        _sse_event(
            "response.completed",
            {
                "response": {
                    "id": "resp_cfg028",
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
        ),
        b"data: [DONE]\n\n",
    ]


def _watermark_python_sources() -> list[Path]:
    assert WATERMARK_PACKAGE_DIR.is_dir(), (
        "CFG-028 requires aawm_text_watermark.response_hooks plus the CFG-026 package"
    )
    return sorted(WATERMARK_PACKAGE_DIR.glob("*.py"))


# ---------------------------------------------------------------------------
# 1. Non-stream install order
# ---------------------------------------------------------------------------


def test_pass_through_request_installs_nonstream_watermark_after_cfg025_before_capture() -> None:
    source = _function_source(PASS_THROUGH_REQUEST_PATH, "pass_through_request")
    assert CFG025_REJECT_NAME in source
    assert CFG028_NONSTREAM_HOOK in source
    reject_at = source.find(CFG025_REJECT_NAME)
    success_at = source.find("pass_through_async_success_handler", reject_at)
    assert success_at > reject_at
    window = source[reject_at:success_at]
    assert CFG028_NONSTREAM_HOOK in window
    assert "capture_passthrough_shape" in window
    assert window.find(CFG028_NONSTREAM_HOOK) < window.find("capture_passthrough_shape")
    names = _call_names_in_source_order(source)
    assert names.index(CFG025_REJECT_NAME) < names.index(CFG028_NONSTREAM_HOOK)
    assert names.index(CFG028_NONSTREAM_HOOK) < names.index("capture_passthrough_shape")
    imported = PASS_THROUGH_REQUEST_PATH.read_text(encoding="utf-8")
    assert CFG028_NONSTREAM_HOOK in imported
    assert "aawm_text_watermark" in imported


# ---------------------------------------------------------------------------
# 2. Stream wrap after CFG-025, reuse existing SSE iterator
# ---------------------------------------------------------------------------


def test_pass_through_request_wraps_stream_after_cfg025_using_existing_sse_iterator() -> None:
    source = _function_source(PASS_THROUGH_REQUEST_PATH, "pass_through_request")
    assert CFG025_WRAP_NAME in source
    assert CFG028_STREAM_HOOK in source
    wrap_sites = 0
    cursor = 0
    while True:
        cfg025_at = source.find(CFG025_WRAP_NAME, cursor)
        if cfg025_at < 0:
            break
        cfg028_at = source.find(CFG028_STREAM_HOOK, cfg025_at)
        streaming_at = source.find("StreamingResponse", cfg025_at)
        assert cfg028_at > cfg025_at
        assert streaming_at > cfg028_at
        wrap_sites += 1
        cursor = cfg025_at + len(CFG025_WRAP_NAME)
    assert wrap_sites >= 2

    watermark_sources = _watermark_python_sources()
    joined = "\n".join(path.read_text(encoding="utf-8") for path in watermark_sources)
    assert SSE_ITER_NAME in joined
    assert "from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import" in joined or (
        "aawm_adapter_runtime.sse" in joined
    )
    hook_path = WATERMARK_PACKAGE_DIR / "response_hooks.py"
    assert hook_path.is_file()
    hook_source = hook_path.read_text(encoding="utf-8")
    assert CFG028_STREAM_HOOK in hook_source
    assert SSE_ITER_NAME in hook_source or "responses_stream" in hook_source


# ---------------------------------------------------------------------------
# 3. Non-stream detect vs sanitize
# ---------------------------------------------------------------------------


def test_nonstream_output_text_zwsp_detect_keeps_bytes_sanitize_mutates_visible_text_only() -> None:
    visible = f"hello{ZWSP}world"
    body = _responses_output_body(visible=visible)
    original = copy.deepcopy(body)
    original_content = json.dumps(original, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    kwargs = _empty_success_handler_kwargs()

    detect_returned = _apply_nonstream_hook(
        copy.deepcopy(body),
        content=original_content,
        config=_detect_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    detect_body = _extract_hook_body(detect_returned, original)
    detect_content = _extract_hook_content(detect_returned, original_content)
    detect_audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
    if detect_audit is None:
        detect_audit = _optional_field(detect_returned, "audit")
    assert detect_audit is not None
    assert _field(detect_audit, "signal_detected") is True
    assert detect_body == original
    assert ZWSP in detect_body["output_text"]
    assert detect_content == original_content
    assert detect_body["output"][1]["arguments"] == TOOL_ARG_LOCK
    assert detect_body["output"][2]["encrypted_content"] == ENCRYPTED_LOCK

    sanitize_kwargs = _empty_success_handler_kwargs()
    sanitize_returned = _apply_nonstream_hook(
        copy.deepcopy(body),
        content=original_content,
        config=_sanitize_config(),
        success_handler_kwargs=sanitize_kwargs,
        endpoint="responses",
    )
    sanitize_body = _extract_hook_body(sanitize_returned, original)
    sanitize_audit = _metadata_from_kwargs(sanitize_kwargs).get("watermark_output_audit")
    if sanitize_audit is None:
        sanitize_audit = _optional_field(sanitize_returned, "audit")
    assert sanitize_audit is not None
    assert _field(sanitize_audit, "signal_detected") is True
    assert ZWSP not in sanitize_body["output_text"]
    assert ZWSP not in sanitize_body["output"][0]["content"][0]["text"]
    assert "hello" in sanitize_body["output_text"]
    assert "world" in sanitize_body["output_text"]
    assert sanitize_body["output"][1]["arguments"] == TOOL_ARG_LOCK
    assert sanitize_body["output"][2]["encrypted_content"] == ENCRYPTED_LOCK
    assert sanitize_body["output"][1]["id"] == "fc_keep"
    assert sanitize_body["output"][2]["id"] == "rs_keep"


# ---------------------------------------------------------------------------
# 4. Streaming audit_only does not rewrite SSE; incremental metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_audit_only_does_not_rewrite_sse_and_attaches_audit_before_logging_route() -> None:
    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()
    scheduled_metadata: list[dict[str, Any]] = []

    async def _chunk_processor_like() -> Any:
        for chunk in chunks:
            yield chunk
        scheduled_metadata.append(copy.deepcopy(_metadata_from_kwargs(kwargs)))

    wrapped = _wrap_stream_hook(
        _chunk_processor_like(),
        config=_detect_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded: list[bytes] = []
    async for chunk in wrapped:
        yielded.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))

    assert yielded == chunks
    assert ZWSP.encode("utf-8") in b"".join(yielded)
    assert scheduled_metadata, (
        "inner chunk_processor must complete and snapshot metadata before "
        f"{LOGGING_ROUTE_NAME} is scheduled"
    )
    scheduled_audit = scheduled_metadata[0].get("watermark_output_audit")
    assert scheduled_audit is not None
    assert _field(scheduled_audit, "signal_detected") is True
    live_audit = _metadata_from_kwargs(kwargs).get("watermark_output_audit")
    assert live_audit is not None
    assert _field(live_audit, "signal_detected") is True

    handler_source = STREAMING_HANDLER_PATH.read_text(encoding="utf-8")
    processor_source = _function_source(STREAMING_HANDLER_PATH, "chunk_processor")
    assert LOGGING_ROUTE_NAME in processor_source
    assert "asyncio.create_task" in processor_source
    create_at = processor_source.find("asyncio.create_task")
    route_at = processor_source.find(LOGGING_ROUTE_NAME, create_at)
    assert route_at > create_at
    assert CFG028_STREAM_HOOK not in processor_source
    assert CFG028_STREAM_HOOK in _function_source(
        PASS_THROUGH_REQUEST_PATH, "pass_through_request"
    )
    assert "success_handler_kwargs" in handler_source


# ---------------------------------------------------------------------------
# 5. CFG-025 still owns degeneration abort
# ---------------------------------------------------------------------------


def test_cfg025_still_owns_degeneration_abort_and_existing_wrap_remains() -> None:
    request_source = _function_source(PASS_THROUGH_REQUEST_PATH, "pass_through_request")
    assert CFG025_WRAP_NAME in request_source
    assert CFG025_REJECT_NAME in request_source

    cfg025_source = REPETITIVE_OUTPUT_PATH.read_text(encoding="utf-8")
    assert "wrap_responses_sse_with_repetitive_output_guard" in cfg025_source
    assert SSE_ITER_NAME in cfg025_source
    assert "CFG-025 aborted repetitive Responses stream" in cfg025_source

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.repetitive_output import (
        maybe_wrap_passthrough_responses_stream,
        wrap_responses_sse_with_repetitive_output_guard,
    )

    wrap_source = inspect.getsource(wrap_responses_sse_with_repetitive_output_guard)
    maybe_source = inspect.getsource(maybe_wrap_passthrough_responses_stream)
    assert SSE_ITER_NAME in wrap_source
    assert "VisibleTextRepetitionDetector" in wrap_source
    assert "wrap_responses_sse_with_repetitive_output_guard" in maybe_source

    assert CFG028_STREAM_HOOK in request_source
    assert CFG028_NONSTREAM_HOOK in request_source
    assert request_source.find(CFG025_WRAP_NAME) < request_source.find(CFG028_STREAM_HOOK)
    assert request_source.find(CFG025_REJECT_NAME) < request_source.find(
        CFG028_NONSTREAM_HOOK
    )


# ---------------------------------------------------------------------------
# 6. enforce without buffer_response stays a config-load error
# ---------------------------------------------------------------------------


def test_enforce_streamed_output_without_buffer_response_is_config_error_not_live_sse_rewrite() -> None:
    payload = {
        "mode": "enforce",
        "removal": {
            "enabled": True,
            "stream_policy": "audit_only",
        },
    }
    with pytest.raises((ValueError, TypeError, ValidationError)) as info:
        _load_text_watermark_config(payload)
    message = str(info.value).lower()
    assert "buffer_response" in message or "stream" in message

    hook_path = WATERMARK_PACKAGE_DIR / "response_hooks.py"
    assert hook_path.is_file()
    hook_source = hook_path.read_text(encoding="utf-8")
    wrap_source = ""
    parsed = ast.parse(hook_source)
    for node in parsed.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == CFG028_STREAM_HOOK
        ):
            wrap_source = ast.get_source_segment(hook_source, node) or ""
    assert wrap_source
    assert "audit_only" in wrap_source or "stream_policy" in wrap_source
    assert "sanitize_unicode_carriers" not in wrap_source or "audit_only" in wrap_source


@pytest.mark.asyncio
async def test_stream_hook_does_not_silently_enforce_on_live_sse() -> None:
    chunks = _watermarked_sse_chunks()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_detect_config(),
        success_handler_kwargs=_empty_success_handler_kwargs(),
        endpoint="responses",
    )
    yielded: list[bytes] = []
    async for chunk in wrapped:
        yielded.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
    assert yielded == chunks
    assert ZWSP.encode("utf-8") in b"".join(yielded)


# ---------------------------------------------------------------------------
# 7. mode off is a no-op
# ---------------------------------------------------------------------------


def test_mode_off_is_noop_on_nonstream_response() -> None:
    visible = f"hello{ZWSP}world"
    body = _responses_output_body(visible=visible)
    original = copy.deepcopy(body)
    original_content = json.dumps(original, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    kwargs = _empty_success_handler_kwargs()
    returned = _apply_nonstream_hook(
        body,
        content=original_content,
        config=_off_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    out_body = _extract_hook_body(returned, original)
    out_content = _extract_hook_content(returned, original_content)
    assert out_body == original
    assert out_content == original_content
    assert _metadata_from_kwargs(kwargs).get("watermark_output_audit") is None
    assert _optional_field(returned, "audit") is None
    assert ZWSP in out_body["output_text"]


@pytest.mark.asyncio
async def test_mode_off_is_noop_on_stream_response() -> None:
    chunks = _watermarked_sse_chunks()
    kwargs = _empty_success_handler_kwargs()

    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    wrapped = _wrap_stream_hook(
        _gen(),
        config=_off_config(),
        success_handler_kwargs=kwargs,
        endpoint="responses",
    )
    yielded: list[bytes] = []
    async for chunk in wrapped:
        yielded.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
    assert yielded == chunks
    assert _metadata_from_kwargs(kwargs).get("watermark_output_audit") is None


# ---------------------------------------------------------------------------
# 8. Managed Codex JSON helper currently bypasses generic finalizer
# ---------------------------------------------------------------------------


def test_managed_codex_json_helper_must_install_cfg028_response_hook() -> None:
    helper_source = _function_source(
        RESPONSE_UTILS_PATH, "_build_responses_response_from_adapter_response"
    )
    callers_source = CODEX_CANDIDATE_CALLS_PATH.read_text(encoding="utf-8")
    assert "_build_responses_response_from_adapter_response" in callers_source
    assert "pass_through_request" not in helper_source
    assert CFG025_REJECT_NAME not in helper_source
    assert CFG028_NONSTREAM_HOOK in helper_source, (
        "CFG-028 required install site: _build_responses_response_from_adapter_response "
        "bypasses the generic pass_through_request finalizer, so managed Codex JSON "
        "owners must call maybe_apply_passthrough_watermark_response from that helper "
        "(or from every caller before returning the FastAPI Response)."
    )
    imported = RESPONSE_UTILS_PATH.read_text(encoding="utf-8")
    assert "aawm_text_watermark" in imported or CFG028_NONSTREAM_HOOK in imported
