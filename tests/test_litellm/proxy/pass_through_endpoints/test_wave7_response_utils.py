"""Wave 7 response_utils unit tests.

Directly exercises the extracted response construction and streaming
release utilities without importing the god module.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator, List

import pytest
from fastapi.responses import Response, StreamingResponse
from starlette.requests import ClientDisconnect

_RUNTIME_PACKAGE_NAME = "_wave7_aawm_adapter_runtime"
_RUNTIME_PACKAGE_PATH = (
    Path(__file__).resolve().parents[4]
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime"
)
_runtime_package = ModuleType(_RUNTIME_PACKAGE_NAME)
_runtime_package.__path__ = [str(_RUNTIME_PACKAGE_PATH)]  # type: ignore[attr-defined]
sys.modules[_RUNTIME_PACKAGE_NAME] = _runtime_package
_response_utils = importlib.import_module(
    f"{_RUNTIME_PACKAGE_NAME}.response_utils"
)
_build_responses_response_from_adapter_response = (
    _response_utils._build_responses_response_from_adapter_response
)
_wrap_streaming_response_with_release_callback = (
    _response_utils._wrap_streaming_response_with_release_callback
)


# Helpers


class _FakePydanticV2:
    """Mimics a Pydantic v2 model with model_dump_json."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        if exclude_none:
            return json.dumps(
                {k: v for k, v in self._data.items() if v is not None}
            )
        return json.dumps(self._data)


class _FakePydanticV1:
    """Mimics a Pydantic v1 model with .json()."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def json(self, *, exclude_none: bool = False) -> str:
        if exclude_none:
            return json.dumps(
                {k: v for k, v in self._data.items() if v is not None}
            )
        return json.dumps(self._data)


async def _async_chunks(chunks: List[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


async def _drain(it: AsyncIterator[str]) -> List[str]:
    out: List[str] = []
    async for chunk in it:
        out.append(chunk)
    return out


# _build_responses_response_from_adapter_response


class TestBuildResponsesResponse:
    def test_pydantic_v2_serialization(self) -> None:
        obj = _FakePydanticV2({"id": "resp_1", "output": [], "extra": None})
        resp = _build_responses_response_from_adapter_response(obj)
        assert isinstance(resp, Response)
        assert resp.media_type == "application/json"
        assert resp.body == b'{"id": "resp_1", "output": []}'

    def test_pydantic_v1_serialization(self) -> None:
        obj = _FakePydanticV1({"id": "resp_2", "val": None})
        resp = _build_responses_response_from_adapter_response(obj)
        assert resp.body == b'{"id": "resp_2"}'

    def test_plain_dict_serialization(self) -> None:
        obj = {"id": "resp_3", "nested": {"a": 1}}
        resp = _build_responses_response_from_adapter_response(obj)
        assert resp.body == json.dumps(obj).encode()

    def test_custom_serializer_injection(self) -> None:
        sentinel = '{"custom":true}'
        resp = _build_responses_response_from_adapter_response(
            object(),
            serializer=lambda _obj: sentinel,
        )
        assert resp.body.decode() == sentinel
        assert resp.media_type == "application/json"


# _wrap_streaming_response_with_release_callback


class TestWrapStreamingRelease:
    def test_release_on_normal_completion(self) -> None:
        released: List[bool] = []
        sr = StreamingResponse(
            _async_chunks(["a", "b", "c"]),
            media_type="text/event-stream",
        )
        wrapped = _wrap_streaming_response_with_release_callback(
            sr, lambda: released.append(True)
        )
        chunks = asyncio.run(_drain(wrapped.body_iterator))
        assert chunks == ["a", "b", "c"]
        assert released == [True]

    def test_release_exactly_once(self) -> None:
        count = 0

        def release() -> None:
            nonlocal count
            count += 1

        sr = StreamingResponse(_async_chunks(["x"]))
        wrapped = _wrap_streaming_response_with_release_callback(sr, release)

        async def _consume_and_close() -> None:
            await _drain(wrapped.body_iterator)
            await wrapped.body_iterator.aclose()

        asyncio.run(_consume_and_close())
        assert count == 1

    def test_release_on_close_before_iteration(self) -> None:
        released: List[bool] = []

        class _ClosableAsyncIterator:
            def __init__(self) -> None:
                self.close_calls = 0

            def __aiter__(self) -> "_ClosableAsyncIterator":
                return self

            async def __anext__(self) -> str:
                return "never"

            async def aclose(self) -> None:
                self.close_calls += 1

        inner = _ClosableAsyncIterator()
        sr = StreamingResponse(inner)
        wrapped = _wrap_streaming_response_with_release_callback(
            sr, lambda: released.append(True)
        )
        asyncio.run(wrapped.body_iterator.aclose())
        assert released == [True]
        assert inner.close_calls == 1

    def test_preserves_sync_iterable_stream_shape(self) -> None:
        released: List[bool] = []

        class _SyncBodyResponse:
            def __init__(self) -> None:
                self.body_iterator = iter(["a", b"b"])

        sr = _SyncBodyResponse()
        wrapped = _wrap_streaming_response_with_release_callback(
            sr,  # type: ignore[arg-type]
            lambda: released.append(True),
        )
        chunks = asyncio.run(_drain(wrapped.body_iterator))
        assert chunks == ["a", b"b"]
        assert released == [True]

    def test_release_on_iteration_failure(self) -> None:
        released: List[bool] = []

        async def _failing_iter() -> AsyncIterator[str]:
            yield "ok"
            raise RuntimeError("upstream exploded")

        sr = StreamingResponse(_failing_iter())
        wrapped = _wrap_streaming_response_with_release_callback(
            sr, lambda: released.append(True)
        )
        with pytest.raises(RuntimeError, match="upstream exploded"):
            asyncio.run(_drain(wrapped.body_iterator))
        assert released == [True]

    def test_release_on_cancellation(self) -> None:
        released: List[bool] = []

        async def _exercise_cancellation() -> None:
            consumed_first_chunk = asyncio.Event()
            block_iteration = asyncio.Event()

            async def _slow_iter() -> AsyncIterator[str]:
                yield "first"
                await block_iteration.wait()
                yield "never"

            sr = StreamingResponse(_slow_iter())
            wrapped = _wrap_streaming_response_with_release_callback(
                sr, lambda: released.append(True)
            )

            async def _consume() -> None:
                async for chunk in wrapped.body_iterator:
                    if chunk == "first":
                        consumed_first_chunk.set()

            task = asyncio.create_task(_consume())
            await consumed_first_chunk.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_exercise_cancellation())
        assert released == [True]

    def test_release_callback_failure_is_logged_not_raised(self) -> None:
        log_calls: List[str] = []
        release_calls = 0

        def bad_release() -> None:
            nonlocal release_calls
            release_calls += 1
            raise ValueError("release boom")

        sr = StreamingResponse(_async_chunks(["z"]))
        wrapped = _wrap_streaming_response_with_release_callback(
            sr,
            bad_release,
            log_exception=lambda msg: log_calls.append(msg),
        )

        async def _consume_and_close() -> List[str]:
            chunks = await _drain(wrapped.body_iterator)
            await wrapped.body_iterator.aclose()
            return chunks

        assert asyncio.run(_consume_and_close()) == ["z"]
        assert release_calls == 1
        assert log_calls == [
            "Failed to release adapted streaming response guard callback"
        ]

    def test_no_body_iterator_releases_immediately(self) -> None:
        released: List[bool] = []

        class _Bare:
            pass

        bare = _Bare()
        result = _wrap_streaming_response_with_release_callback(
            bare,  # type: ignore[arg-type]
            lambda: released.append(True),
        )
        assert result is bare
        assert released == [True]

    def test_release_callback_failure_does_not_block_chunks(self) -> None:
        """Chunks yielded before release failure are still delivered."""

        def bad_release() -> None:
            raise RuntimeError("cleanup error")

        sr = StreamingResponse(_async_chunks(["d1", "d2"]))
        wrapped = _wrap_streaming_response_with_release_callback(
            sr,
            bad_release,
            log_exception=lambda _msg: None,
        )
        chunks = asyncio.run(_drain(wrapped.body_iterator))
        assert chunks == ["d1", "d2"]

    def test_release_on_consumer_break_with_body_iterator_held(self) -> None:
        released: List[bool] = []

        class _ConsumerBreak(Exception):
            pass

        async def _exercise_break() -> None:
            inner_closed = asyncio.Event()

            async def _source() -> AsyncIterator[str]:
                try:
                    yield "first"
                    yield "second"
                finally:
                    inner_closed.set()

            sr = StreamingResponse(_source())
            wrapped = _wrap_streaming_response_with_release_callback(
                sr, lambda: released.append(True)
            )
            held_iterator = wrapped.body_iterator

            async def _send(message: dict[str, Any]) -> None:
                if (
                    message["type"] == "http.response.body"
                    and message["more_body"]
                ):
                    raise _ConsumerBreak

            with pytest.raises(_ConsumerBreak):
                await wrapped.stream_response(_send)
            assert wrapped.body_iterator is held_iterator
            assert released == [True]
            assert inner_closed.is_set()

        asyncio.run(_exercise_break())
        assert released == [True]

    def test_release_on_send_oserror_with_body_iterator_held(self) -> None:
        released: List[bool] = []

        async def _exercise_send_failure() -> None:
            inner_closed = asyncio.Event()

            async def _source() -> AsyncIterator[str]:
                try:
                    yield "first"
                    yield "second"
                finally:
                    inner_closed.set()

            sr = StreamingResponse(_source())
            wrapped = _wrap_streaming_response_with_release_callback(
                sr, lambda: released.append(True)
            )
            held_iterator = wrapped.body_iterator

            async def _receive() -> dict[str, str]:
                return {"type": "http.disconnect"}

            async def _send(message: dict[str, Any]) -> None:
                if (
                    message["type"] == "http.response.body"
                    and message["more_body"]
                ):
                    raise OSError("client disconnected")

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            with pytest.raises(ClientDisconnect):
                await wrapped(scope, _receive, _send)
            assert wrapped.body_iterator is held_iterator
            assert released == [True]
            assert inner_closed.is_set()

        asyncio.run(_exercise_send_failure())
        assert released == [True]

    def test_release_on_between_chunk_cancellation_with_iterator_held(
        self,
    ) -> None:
        released: List[bool] = []

        async def _exercise_cancellation() -> None:
            waiting_for_next_chunk = asyncio.Event()
            inner_closed = asyncio.Event()
            unblock = asyncio.Event()

            async def _source() -> AsyncIterator[str]:
                try:
                    yield "first"
                    waiting_for_next_chunk.set()
                    await unblock.wait()
                    yield "second"
                finally:
                    inner_closed.set()

            sr = StreamingResponse(_source())
            wrapped = _wrap_streaming_response_with_release_callback(
                sr, lambda: released.append(True)
            )
            held_iterator = wrapped.body_iterator

            async def _send(_message: dict[str, Any]) -> None:
                return None

            task = asyncio.create_task(wrapped.stream_response(_send))
            await waiting_for_next_chunk.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert wrapped.body_iterator is held_iterator
            assert released == [True]
            assert inner_closed.is_set()

        asyncio.run(_exercise_cancellation())
        assert released == [True]


# install() smoke


class TestInstall:
    def test_install_publishes_to_host_globals(self) -> None:
        host: dict[str, Any] = {}
        _response_utils.install(host)
        assert "_build_responses_response_from_adapter_response" in host
        assert "_wrap_streaming_response_with_release_callback" in host
        # Rebound functions should resolve against host namespace.
        assert (
            host["_build_responses_response_from_adapter_response"].__globals__
            is host
        )
