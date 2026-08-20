"""Wave 7 extraction: response construction and streaming release utilities.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.

Owned symbols:
    _build_responses_response_from_adapter_response
    _wrap_streaming_response_with_release_callback

Integration seams (injectable for testability):
    serializer: callable matching _serialize_responses_adapter_response
        (defaults to sse._serialize_responses_adapter_response)
    log_exception: callable for release-failure logging
        (defaults to verbose_proxy_logger.exception)
"""

from __future__ import annotations

from types import FunctionType
from typing import Any, Callable, Optional

from fastapi.responses import Response, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.response_hooks import (
    maybe_apply_passthrough_watermark_response,
)

from .sse import _serialize_responses_adapter_response


# Public API


def _build_responses_response_from_adapter_response(
    response_obj: Any,
    *,
    serializer: Optional[Callable[[Any], str]] = None,
    success_handler_kwargs: Optional[dict[str, Any]] = None,
    config: Any = None,
) -> Response:
    """Build a JSON Response from an adapter response object.

    Serialization is delegated to *serializer* (default: the SSE module's
    ``_serialize_responses_adapter_response``) so that the exact wire format
    is preserved without importing the god module. CFG-028 applies
    ``maybe_apply_passthrough_watermark_response`` before building Response.
    """
    import json

    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.response_hooks import (
        maybe_apply_passthrough_watermark_response,
    )

    serialize = (
        serializer
        if serializer is not None
        else _serialize_responses_adapter_response
    )
    serialized = serialize(response_obj)
    parsed_body: Any = {}
    if isinstance(serialized, (bytes, bytearray, str)):
        try:
            loaded = json.loads(serialized)
        except (TypeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            parsed_body = loaded
    elif isinstance(response_obj, dict):
        parsed_body = response_obj

    hooked_kwargs = (
        success_handler_kwargs
        if isinstance(success_handler_kwargs, dict)
        else {"litellm_params": {"metadata": {}}}
    )
    watermark_config = config
    if watermark_config is None:
        try:
            from litellm.proxy.proxy_server import general_settings as _gs

            if isinstance(_gs, dict):
                watermark_config = _gs.get("openai_passthrough_text_watermark")
            else:
                watermark_config = getattr(
                    _gs, "openai_passthrough_text_watermark", None
                )
        except Exception:
            watermark_config = None

    _hooked_body, hooked_content = maybe_apply_passthrough_watermark_response(
        parsed_body,
        content=serialized,
        config=watermark_config,
        success_handler_kwargs=hooked_kwargs,
        endpoint="responses",
    )
    if hooked_content is None:
        hooked_content = serialized
    return Response(
        content=hooked_content,
        media_type="application/json",
    )


def _wrap_streaming_response_with_release_callback(
    response: StreamingResponse,
    release_callback: Any,
    *,
    log_exception: Optional[Callable[[str], None]] = None,
) -> StreamingResponse:
    """Wrap *response*'s body iterator so *release_callback* fires exactly once.

    The callback is guaranteed to run in a ``finally`` block covering:
    - normal iteration completion
    - mid-iteration exception propagation
    - generator cancellation (GeneratorExit / CancelledError)

    If *release_callback* itself raises, the exception is logged (not
    propagated) via *log_exception* (default: ``verbose_proxy_logger.exception``).

    If the response has no ``body_iterator``, the callback fires immediately.
    """
    _log: Callable[[str], None] = (
        log_exception
        if log_exception is not None
        else lambda msg: verbose_proxy_logger.exception(msg)
    )
    released = False

    def _release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            release_callback()
        except Exception:
            _log(
                "Failed to release adapted streaming response guard callback"
            )

    original_iterator = getattr(response, "body_iterator", None)
    if original_iterator is None:
        _release_once()
        return response

    original_async_iterator = (
        original_iterator.__aiter__()
        if hasattr(original_iterator, "__aiter__")
        else None
    )
    original_sync_iterator = (
        None if original_async_iterator is not None else iter(original_iterator)
    )
    iterator_closed = False

    async def _close_original_iterator_once() -> None:
        nonlocal iterator_closed
        if iterator_closed:
            return
        iterator_closed = True
        try:
            if original_async_iterator is not None:
                close = getattr(original_async_iterator, "aclose", None)
                if close is not None:
                    await close()
            else:
                close = getattr(original_sync_iterator, "close", None)
                if close is not None:
                    close()
        except Exception:
            _log("Failed to close adapted streaming response body iterator")

    class _ReleaseWrapper:
        __slots__ = ()

        def __aiter__(self) -> "_ReleaseWrapper":
            return self

        async def __anext__(self) -> Any:
            if original_async_iterator is None:
                try:
                    return next(original_sync_iterator)
                except StopIteration:
                    await _close_original_iterator_once()
                    _release_once()
                    raise StopAsyncIteration
                except BaseException:
                    try:
                        await _close_original_iterator_once()
                    finally:
                        _release_once()
                    raise

            try:
                return await original_async_iterator.__anext__()
            except StopAsyncIteration:
                await _close_original_iterator_once()
                _release_once()
                raise
            except BaseException:
                try:
                    await _close_original_iterator_once()
                finally:
                    _release_once()
                raise

        async def aclose(self) -> None:
            try:
                await _close_original_iterator_once()
            finally:
                _release_once()

    wrapped_iterator = _ReleaseWrapper()
    response.body_iterator = wrapped_iterator

    original_stream_response = getattr(response, "stream_response", None)
    if original_stream_response is not None:

        async def _stream_response_with_release(send: Any) -> None:
            try:
                await original_stream_response(send)
            finally:
                await wrapped_iterator.aclose()

        response.stream_response = _stream_response_with_release

    return response


# Host-global install

_HOST_FUNCTION_NAMES = (
    "_build_responses_response_from_adapter_response",
    "_wrap_streaming_response_with_release_callback",
)


def install(host_globals: dict[str, Any]) -> None:
    """Rebind owned functions into *host_globals* for live lookup.

    Follows the same FunctionType-rebinding pattern used by sibling modules
    so that the god module's namespace resolves to these implementations.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound
