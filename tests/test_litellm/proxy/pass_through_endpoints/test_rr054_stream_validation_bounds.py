"""RR-054 #14 stream-validation bounds / replay / TTFB regression tests.

Tests-only coverage for:
- chunk and byte overflow bounds on responses stream validation
- non-truncating replay when under bounds
- terminal event preservation on replay
- OpenRouter validator parity with the main bounds contract
- first-byte ordering / TTFB contract for validated streams

No production edits. Failures here document remaining RR-054 #14 gaps.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, AsyncIterator, Optional
from unittest.mock import patch

import pytest
from starlette.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed_response_body(
    *,
    response_id: str = "resp_rr054_bounds",
    model: str = "test-model",
    text: str = "hello",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        },
    }


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


def _minimal_valid_stream_chunks(
    *,
    response_id: str = "resp_rr054_bounds",
    model: str = "test-model",
    text: str = "hello",
) -> list[bytes]:
    body = _completed_response_body(
        response_id=response_id,
        model=model,
        text=text,
    )
    return [
        _sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
        ),
        _sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "delta": text,
            },
        ),
        _sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": body,
            },
        ),
        b"data: [DONE]\n\n",
    ]


def _decode_chunk(chunk: Any) -> str:
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk).decode("utf-8", errors="replace")
    return str(chunk)


async def _drain_stream(response: StreamingResponse) -> list[Any]:
    chunks: list[Any] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


def _rendered(chunks: list[Any]) -> str:
    return "".join(_decode_chunk(c) for c in chunks)


def _chunk_size(raw_chunk: Any) -> int:
    if isinstance(raw_chunk, (bytes, bytearray)):
        return len(raw_chunk)
    return len(str(raw_chunk).encode("utf-8", errors="replace"))


async def _upstream_from_chunks(
    chunks: list[Any],
    *,
    delay_s: float = 0.0,
    first_chunk_ready: Optional[asyncio.Event] = None,
) -> StreamingResponse:
    async def _gen() -> AsyncIterator[Any]:
        for index, chunk in enumerate(chunks):
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if index == 0 and first_chunk_ready is not None:
                first_chunk_ready.set()
            yield chunk

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Constants / ownership of bounds
# ---------------------------------------------------------------------------


def test_rr054_stream_validation_bounds_constants_are_positive() -> None:
    assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS >= 1
    assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES >= 1
    # Document the reviewed operational defaults so silent regressions fail loudly.
    assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS == 5000
    assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES == 8 * 1024 * 1024


def test_rr054_stream_validation_openrouter_source_declares_bound_helpers() -> None:
    """OpenRouter stream validator should share the same bound contract as main path.

    RR-054 #14 / OpenRouter parity: unbounded double-buffer is the residual gap.
    """
    source = inspect.getsource(lpe._validate_codex_auto_agent_openrouter_responses_stream)
    assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS" in source
    assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES" in source
    assert "peek.exhausted" in source


def test_rr054_stream_validation_main_path_source_enforces_bounds() -> None:
    source = inspect.getsource(lpe._validate_codex_auto_agent_responses_payload)
    assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS" in source
    assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES" in source
    assert "peek.exhausted" in source
    assert '"peek overflow (chunks=%s bytes=%s adapter=%s); preserving the "' in source
    assert '"complete upstream stream"' in source


# ---------------------------------------------------------------------------
# Non-truncating replay + terminal preservation (under bounds)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_stream_validation_under_bounds_replays_all_chunks_in_order() -> None:
    original = _minimal_valid_stream_chunks(text="under-bounds")
    upstream = await _upstream_from_chunks(original)

    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )

    replayed = await _drain_stream(response)
    assert len(replayed) == len(original)
    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]
    rendered = _rendered(replayed)
    assert "response.completed" in rendered
    assert "under-bounds" in rendered
    assert "data: [DONE]" in rendered


@pytest.mark.asyncio
async def test_rr054_stream_validation_preserves_terminal_completed_event() -> None:
    original = _minimal_valid_stream_chunks(
        response_id="resp_terminal_keep",
        text="terminal-keep",
    )
    upstream = await _upstream_from_chunks(original)

    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )
    rendered = _rendered(await _drain_stream(response))

    assert "event: response.completed" in rendered
    completed_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith("data: ")
        and '"type": "response.completed"' in line.replace(" ", "")
        or (
            line.startswith("data: ")
            and '"type":"response.completed"' in line.replace(" ", "")
        )
    )
    # Tolerate both compact and spaced JSON encodings.
    payload = json.loads(completed_line.removeprefix("data: ").strip())
    assert payload["type"] == "response.completed"
    assert payload["response"]["id"] == "resp_terminal_keep"
    assert payload["response"]["status"] == "completed"
    assert "terminal-keep" in rendered


# ---------------------------------------------------------------------------
# Chunk / byte overflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_stream_validation_chunk_overflow_replays_losslessly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny cap so a valid multi-event stream overflows after the first few chunks.
    monkeypatch.setattr(lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 2)
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES", 8 * 1024 * 1024
    )

    original = _minimal_valid_stream_chunks(text="chunk-overflow")
    assert len(original) > 2
    upstream = await _upstream_from_chunks(original)

    with patch.object(lpe, "_should_log_aawm_alias_routing_event", return_value=True):
        response = await lpe._validate_codex_auto_agent_responses_payload(
            upstream,
            adapter_model="test-model",
            adapter="codex_auto_agent_openai_responses",
            adapter_label="OpenAI Responses",
        )

    replayed = await _drain_stream(response)
    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]
    assert "response.completed" in _rendered(replayed)


@pytest.mark.asyncio
async def test_rr054_stream_validation_byte_overflow_replays_losslessly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _minimal_valid_stream_chunks(text="byte-overflow")
    first_size = _chunk_size(original[0])
    # Allow only the first chunk by bytes; second chunk must trip overflow.
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES", first_size
    )
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 5000
    )

    upstream = await _upstream_from_chunks(original)
    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )

    replayed = await _drain_stream(response)
    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]
    assert "response.completed" in _rendered(replayed)


@pytest.mark.asyncio
async def test_rr054_stream_validation_overflow_bypasses_validation_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overflow stops eager validation and preserves the complete upstream stream."""
    monkeypatch.setattr(lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 1)

    failed_body = {
        "id": "resp_failed",
        "object": "response",
        "status": "failed",
        "model": "test-model",
        "output": [],
        "error": {"type": "upstream_error", "message": "boom"},
    }
    chunks = [
        _sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_failed",
                    "status": "in_progress",
                    "model": "test-model",
                    "output": [],
                },
            },
        ),
        _sse_event(
            "response.failed",
            {"type": "response.failed", "response": failed_body},
        ),
        b"data: [DONE]\n\n",
    ]
    upstream = await _upstream_from_chunks(chunks)

    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )
    replayed = await _drain_stream(response)
    assert [_decode_chunk(c) for c in replayed] == [_decode_chunk(c) for c in chunks]
    assert "response.failed" in _rendered(replayed)


@pytest.mark.asyncio
async def test_rr054_kimi_stream_overflow_restores_parallel_collaboration_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lpe,
        "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES",
        64 * 1024,
    )
    monkeypatch.setattr(
        lpe,
        "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS",
        5000,
    )

    function_calls = [
        {
            "type": "function_call",
            "id": f"fc_spawn_{index}",
            "call_id": f"call_spawn_{index}",
            "name": "spawn_agent",
            "arguments": json.dumps(
                {
                    "task_name": f"child_{index}",
                    "message": "inspect the assigned scope",
                }
            ),
        }
        for index in range(2)
    ]
    response_body = {
        "id": "resp_kimi_parallel_spawn",
        "object": "response",
        "status": "completed",
        "model": "kimi_code/k3-max",
        "output": function_calls,
    }
    chunks = [
        _sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": response_body["id"],
                    "object": "response",
                    "status": "in_progress",
                    "model": response_body["model"],
                    "output": [],
                },
            },
        ),
        _sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_large",
                "output_index": 0,
                "delta": "x" * (70 * 1024),
            },
        ),
        *[
            event
            for output_index, function_call in enumerate(function_calls, start=1)
            for event in (
                _sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": function_call,
                    },
                ),
                _sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": function_call,
                    },
                ),
            )
        ],
        _sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": response_body,
            },
        ),
        b"data: [DONE]\n\n",
    ]
    upstream = await _upstream_from_chunks(chunks)

    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="kimi_code/k3-max",
        adapter="codex_kimi_chat_completions_adapter",
        adapter_label="Kimi Code",
        request_body={
            "model": "kimi_code/k3-max",
            "tools": [
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        }
                    ],
                }
            ],
        },
    )

    rendered = _rendered(await _drain_stream(response))
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in rendered.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    item_payloads = [
        payload
        for payload in payloads
        if payload.get("type")
        in {"response.output_item.added", "response.output_item.done"}
    ]
    assert len(item_payloads) == 4
    assert {
        payload["item"]["call_id"] for payload in item_payloads
    } == {"call_spawn_0", "call_spawn_1"}
    assert all(
        payload["item"]["namespace"] == "collaboration"
        for payload in item_payloads
    )

    completed_payload = next(
        payload
        for payload in payloads
        if payload.get("type") == "response.completed"
    )
    assert [
        item["namespace"] for item in completed_payload["response"]["output"]
    ] == ["collaboration", "collaboration"]
    assert "x" * 1024 in rendered
    assert rendered.endswith("data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# OpenRouter parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_stream_validation_openrouter_under_bounds_non_truncating_replay() -> None:
    original = _minimal_valid_stream_chunks(
        response_id="resp_or_under",
        model="openrouter/test",
        text="openrouter-under",
    )
    upstream = await _upstream_from_chunks(original)

    response = await lpe._validate_codex_auto_agent_openrouter_responses_stream(
        upstream,
        adapter_model="openrouter/test",
    )
    replayed = await _drain_stream(response)
    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]
    assert "response.completed" in _rendered(replayed)
    assert "openrouter-under" in _rendered(replayed)


@pytest.mark.asyncio
async def test_rr054_stream_validation_openrouter_chunk_overflow_matches_main_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 2)
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES", 8 * 1024 * 1024
    )

    original = _minimal_valid_stream_chunks(
        response_id="resp_or_overflow",
        model="openrouter/test",
        text="openrouter-overflow",
    )
    assert len(original) > 2
    upstream = await _upstream_from_chunks(original)

    response = await lpe._validate_codex_auto_agent_openrouter_responses_stream(
        upstream,
        adapter_model="openrouter/test",
    )
    replayed = await _drain_stream(response)

    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]


@pytest.mark.asyncio
async def test_rr054_stream_validation_openrouter_byte_overflow_matches_main_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _minimal_valid_stream_chunks(
        response_id="resp_or_byte",
        model="openrouter/test",
        text="openrouter-byte",
    )
    first_size = _chunk_size(original[0])
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES", first_size
    )
    monkeypatch.setattr(
        lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 5000
    )

    upstream = await _upstream_from_chunks(original)
    response = await lpe._validate_codex_auto_agent_openrouter_responses_stream(
        upstream,
        adapter_model="openrouter/test",
    )
    replayed = await _drain_stream(response)
    assert [_decode_chunk(c) for c in replayed] == [
        _decode_chunk(c) for c in original
    ]


# ---------------------------------------------------------------------------
# First-byte ordering / TTFB contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_stream_validation_replay_preserves_first_chunk_identity() -> None:
    original = _minimal_valid_stream_chunks(text="first-byte-order")
    upstream = await _upstream_from_chunks(original)

    response = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )
    replayed = await _drain_stream(response)

    assert _decode_chunk(replayed[0]) == _decode_chunk(original[0])
    assert "response.created" in _decode_chunk(replayed[0])
    # Ordering contract: completed is never before the first delta/created chunks.
    created_idx = next(
        i for i, c in enumerate(replayed) if "response.created" in _decode_chunk(c)
    )
    completed_idx = next(
        i for i, c in enumerate(replayed) if "response.completed" in _decode_chunk(c)
    )
    assert created_idx < completed_idx


@pytest.mark.asyncio
async def test_rr054_stream_validation_ttfb_does_not_wait_for_full_upstream_completion() -> None:
    """Validated streams must not collapse TTFB to full upstream completion time.

    RR-054 #14: buffering the entire SSE before replaying destroys streaming
    TTFB. Client-visible first byte should become available before the upstream
    iterator finishes producing later chunks.
    """
    chunks = _minimal_valid_stream_chunks(text="ttfb-contract")
    # Slow tail after the first event so full-drain validation is distinguishable.
    slow_tail_delay_s = 0.15

    async def _slow_upstream() -> AsyncIterator[bytes]:
        yield chunks[0]
        for chunk in chunks[1:]:
            await asyncio.sleep(slow_tail_delay_s)
            yield chunk

    upstream = StreamingResponse(_slow_upstream(), media_type="text/event-stream")

    validated = await lpe._validate_codex_auto_agent_responses_payload(
        upstream,
        adapter_model="test-model",
        adapter="codex_auto_agent_openai_responses",
        adapter_label="OpenAI Responses",
    )

    # Desired contract: body_iterator is usable and first byte arrives without
    # waiting for the whole upstream completion budget.
    # For a full-buffer-then-replay implementation this fails because
    # `_validate_*` only returns after the upstream is fully drained.
    started = time.monotonic()
    # The await above already waited for full validation in the current design.
    # Measure residual latency to first client chunk from return of validate().
    first = await validated.body_iterator.__anext__()
    to_first_after_return = time.monotonic() - started
    await validated.body_iterator.aclose()

    assert "response.created" in _decode_chunk(first)
    # Residual replay of an already-buffered stream should be near-instant.
    assert to_first_after_return < 0.05

    # Stronger TTFB contract: validation itself must not serialize behind the
    # full slow tail. Approximate by comparing against (n-1)*delay budget.
    # We re-run with a probe that records when first upstream chunk was ready
    # vs when the client can first read.
    first_ready = asyncio.Event()
    client_first_at: dict[str, float] = {}
    upstream_first_at: dict[str, float] = {}

    async def _instrumented() -> AsyncIterator[bytes]:
        upstream_first_at["t"] = time.monotonic()
        first_ready.set()
        yield chunks[0]
        for chunk in chunks[1:]:
            await asyncio.sleep(slow_tail_delay_s)
            yield chunk

    upstream2 = StreamingResponse(_instrumented(), media_type="text/event-stream")

    async def _validate_and_read_first() -> None:
        validated2 = await lpe._validate_codex_auto_agent_responses_payload(
            upstream2,
            adapter_model="test-model",
            adapter="codex_auto_agent_openai_responses",
            adapter_label="OpenAI Responses",
        )
        first_chunk = await validated2.body_iterator.__anext__()
        client_first_at["t"] = time.monotonic()
        assert "response.created" in _decode_chunk(first_chunk)
        await validated2.body_iterator.aclose()

    validation_task = asyncio.create_task(_validate_and_read_first())
    # Validation must start before the upstream generator can publish first_ready.
    await asyncio.wait_for(first_ready.wait(), timeout=1.0)
    await asyncio.wait_for(validation_task, timeout=1.0)

    assert "t" in upstream_first_at and "t" in client_first_at
    lag = client_first_at["t"] - upstream_first_at["t"]
    full_tail_budget = slow_tail_delay_s * (len(chunks) - 1)
    # Client first-byte lag must be materially less than waiting for full drain.
    assert lag < full_tail_budget * 0.5, (
        f"TTFB collapsed to full upstream completion: lag={lag:.3f}s "
        f"full_tail_budget={full_tail_budget:.3f}s"
    )


@pytest.mark.asyncio
async def test_rr054_stream_validation_openrouter_ttfb_parity_with_main_contract() -> None:
    chunks = _minimal_valid_stream_chunks(
        response_id="resp_or_ttfb",
        model="openrouter/test",
        text="openrouter-ttfb",
    )
    slow_tail_delay_s = 0.15
    upstream_first_at: dict[str, float] = {}
    client_first_at: dict[str, float] = {}

    async def _instrumented() -> AsyncIterator[bytes]:
        upstream_first_at["t"] = time.monotonic()
        yield chunks[0]
        for chunk in chunks[1:]:
            await asyncio.sleep(slow_tail_delay_s)
            yield chunk

    upstream = StreamingResponse(_instrumented(), media_type="text/event-stream")
    validated = await lpe._validate_codex_auto_agent_openrouter_responses_stream(
        upstream,
        adapter_model="openrouter/test",
    )
    first = await validated.body_iterator.__anext__()
    client_first_at["t"] = time.monotonic()
    assert "response.created" in _decode_chunk(first)
    await validated.body_iterator.aclose()

    lag = client_first_at["t"] - upstream_first_at["t"]
    full_tail_budget = slow_tail_delay_s * (len(chunks) - 1)
    assert lag < full_tail_budget * 0.5, (
        f"OpenRouter TTFB collapsed to full upstream completion: lag={lag:.3f}s "
        f"full_tail_budget={full_tail_budget:.3f}s"
    )
