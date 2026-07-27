"""Direct tests for the Wave 6C Google retry runtime extraction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from unittest.mock import AsyncMock
from typing import Any, Optional

import pytest
from fastapi import Response

from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    MonotonicCooldownMap,
)
from litellm.proxy.pass_through_endpoints.providers.google import retry_runtime


class _UpstreamError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        reason: Optional[str] = None,
        wait_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.wait_seconds = wait_seconds


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __call__(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.calls.append((message, args, kwargs))


@pytest.fixture(autouse=True)
def _clear_process_cache_semaphores() -> Any:
    previous_runtime = retry_runtime._runtime
    process_cache._google_adapter_semaphores.clear()
    yield
    process_cache._google_adapter_semaphores.clear()
    retry_runtime._runtime = previous_runtime


def _coerce_non_negative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return max(0, int(value))
    except Exception:
        return default


def _coerce_non_negative_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return max(0.0, float(value))
    except Exception:
        return default


def _record_hidden_retry_metadata(
    kwargs: dict[str, Any],
    *,
    attempt_number: int,
    max_attempts: int,
    status_code: Optional[int],
    failure_class: str,
    wait_seconds: float,
    final_outcome: Optional[str] = None,
    failure_classification: Optional[str] = None,
) -> None:
    metadata = kwargs["litellm_params"]["metadata"]
    metadata["aawm_passthrough_hidden_retry_count"] = max(
        int(metadata.get("aawm_passthrough_hidden_retry_count") or 0),
        attempt_number,
    )
    metadata["aawm_passthrough_hidden_retry_max_attempts"] = max_attempts
    metadata["aawm_passthrough_hidden_retry_status_code"] = status_code
    metadata["aawm_passthrough_hidden_retry_failure_class"] = failure_class
    metadata["aawm_passthrough_hidden_retry_wait_seconds"] = wait_seconds
    if final_outcome is not None:
        metadata["aawm_passthrough_hidden_retry_final_outcome"] = final_outcome
    if failure_classification is not None:
        metadata["aawm_passthrough_hidden_retry_failure_classification"] = (
            failure_classification
        )


async def _unused_post_json(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
) -> process_cache.HttpResponse:
    _ = (url, headers, body, timeout)
    raise AssertionError("process-cache network callback was not expected")


def _process_cache_runtime(
    *,
    max_concurrent: int = 2,
) -> process_cache.Runtime:
    return process_cache.Runtime(
        get_target_base=lambda _provider: "https://google.example",
        build_headers=lambda **_kwargs: {},
        validate_egress=lambda **_kwargs: None,
        post_json=_unused_post_json,
        capture_shape=lambda **_kwargs: None,
        clean_value=lambda value: (
            value.strip()
            if isinstance(value, str) and value.strip()
            else None
        ),
        raise_http_error=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected HTTP error")
        ),
        get_prime_ttl_seconds=lambda: 30.0,
        get_prime_cache_key=lambda token, project: f"{token}:{project}",
        sanitize_quota=lambda _value, _project: None,
        get_max_concurrent=lambda: max_concurrent,
        get_rate_limit_key=lambda model, **_kwargs: model or "__default__",
        monotonic=lambda: 0.0,
        debug_enabled=lambda: False,
        log_info=lambda _message, _value: None,
    )


def _configure_runtime(
    *,
    pass_through_request: Callable[..., Awaitable[Response]],
    max_retries: int = 1,
    capacity_max_retries: int = 1,
    hidden_retry_budget_seconds: float = 0.0,
    transient_max_attempts: int = 2,
    sleeps: Optional[list[float]] = None,
    bounded_caches: Optional[list[MutableMapping[Any, Any]]] = None,
    warning: Optional[_RecordingLogger] = None,
    error: Optional[_RecordingLogger] = None,
) -> MonotonicCooldownMap:
    recorded_sleeps = sleeps if sleeps is not None else []
    recorded_caches = bounded_caches if bounded_caches is not None else []

    async def sleep(seconds: float) -> None:
        recorded_sleeps.append(seconds)

    def record_error_for_logging(
        passthrough_kwargs: dict[str, Any],
        **details: Any,
    ) -> None:
        custom_body = passthrough_kwargs.get("custom_body")
        if not isinstance(custom_body, dict):
            return
        metadata = custom_body.setdefault("litellm_metadata", {})
        metadata["google_generate_content_error"] = dict(details)
        metadata["google_generate_content_error_count"] = (
            int(metadata.get("google_generate_content_error_count") or 0) + 1
        )

    rate_limit = MonotonicCooldownMap()
    retry_runtime.configure_google_retry_runtime(
        retry_runtime.Runtime(
            process_cache_runtime=_process_cache_runtime(),
            rate_limit=rate_limit,
            get_rate_limit_key_from_kwargs=lambda kwargs: str(
                kwargs.get("google_adapter_rate_limit_key") or "lane"
            ),
            get_max_retries=lambda: max_retries,
            coerce_non_negative_int=_coerce_non_negative_int,
            coerce_non_negative_float=_coerce_non_negative_float,
            get_model_capacity_max_retries=lambda: capacity_max_retries,
            get_capacity_backoff_seconds=lambda attempt: float(attempt * 2),
            get_hidden_retry_budget_seconds=lambda: (
                hidden_retry_budget_seconds
            ),
            get_transient_retry_max_attempts=lambda: transient_max_attempts,
            get_transient_backoff_seconds=lambda attempt: (
                float(attempt) / 4.0
            ),
            extract_exception_status_code=lambda exc: getattr(
                exc, "status_code", None
            ),
            extract_error_reason=lambda exc: getattr(exc, "reason", None),
            parse_rate_limit_reset_seconds=lambda exc: float(
                getattr(exc, "wait_seconds", 0.0)
            ),
            is_transient_retryable_failure=lambda _exc, **details: (
                details["status_code"] in {408, 500, 502, 503, 504}
            ),
            classify_hidden_retry_failure=lambda exc: (
                getattr(exc, "status_code", None),
                exc.__class__.__name__,
                "pre_first_byte_transient",
            ),
            record_error_for_logging=record_error_for_logging,
            record_hidden_retry_metadata=_record_hidden_retry_metadata,
            build_terminal_error_log_context=lambda *_args, **kwargs: kwargs,
            pass_through_request=pass_through_request,
            bound_token_cache=lambda cache: recorded_caches.append(cache),
            sleep=sleep,
            log_debug=lambda *_args, **_kwargs: None,
            log_warning=warning or _RecordingLogger(),
            log_error=error or _RecordingLogger(),
        )
    )
    return rate_limit


@pytest.mark.asyncio
async def test_success_returns_exact_response_and_cleans_runtime_kwargs() -> None:
    response = Response(content=b"ok", status_code=207)
    received_kwargs: list[dict[str, Any]] = []

    async def pass_through_request(**kwargs: Any) -> Response:
        received_kwargs.append(dict(kwargs))
        return response

    _configure_runtime(pass_through_request=pass_through_request)
    custom_body = {"model": "gemini-test"}

    result = await retry_runtime._perform_google_adapter_pass_through_request(
        target="https://google.example/generate",
        custom_body=custom_body,
        google_access_token="secret",
        google_adapter_rate_limit_key="shared-lane",
        google_adapter_max_retries=4,
        google_adapter_model_capacity_max_retries=5,
        google_adapter_hidden_retry_budget_seconds=6,
    )

    assert result is response
    assert received_kwargs == [
        {
            "target": "https://google.example/generate",
            "custom_body": custom_body,
            "retryable_upstream_status_codes": sorted(
                {
                    429,
                    *retry_runtime._GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES,
                }
            ),
            "caller_managed_hidden_retry": True,
        }
    ]
    assert "litellm_metadata" not in custom_body


@pytest.mark.asyncio
async def test_rate_limit_failure_sets_cooldown_and_retries_to_success() -> None:
    response = Response(content=b"complete")
    calls = 0
    sleeps: list[float] = []
    bounded_caches: list[MutableMapping[Any, Any]] = []
    custom_body: dict[str, Any] = {}

    async def pass_through_request(**_kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _UpstreamError(
                "rate limited",
                status_code=429,
                reason="RATE_LIMIT_EXCEEDED",
                wait_seconds=3.0,
            )
        return response

    rate_limit = _configure_runtime(
        pass_through_request=pass_through_request,
        sleeps=sleeps,
        bounded_caches=bounded_caches,
    )

    result = await retry_runtime._perform_google_adapter_pass_through_request(
        custom_body=custom_body,
        google_adapter_rate_limit_key="shared-lane",
    )

    assert result is response
    assert calls == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(4.0, abs=0.1)
    assert bounded_caches == [rate_limit.until_monotonic_by_key]
    assert "shared-lane" in rate_limit.until_monotonic_by_key
    assert custom_body["litellm_metadata"][
        "google_generate_content_error_count"
    ] == 1


@pytest.mark.asyncio
async def test_transient_failure_exhaustion_records_terminal_taxonomy() -> None:
    sleeps: list[float] = []
    warning = _RecordingLogger()
    error = _RecordingLogger()
    calls = 0
    custom_body: dict[str, Any] = {}
    terminal_error = _UpstreamError("unavailable", status_code=503)

    async def pass_through_request(**_kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        raise terminal_error

    _configure_runtime(
        pass_through_request=pass_through_request,
        transient_max_attempts=2,
        sleeps=sleeps,
        warning=warning,
        error=error,
    )

    with pytest.raises(_UpstreamError) as exc_info:
        await retry_runtime._perform_google_adapter_pass_through_request(
            custom_body=custom_body,
        )

    assert exc_info.value is terminal_error
    assert calls == 2
    assert sleeps == [0.25]
    metadata = custom_body["litellm_metadata"]
    assert metadata["aawm_passthrough_hidden_retry_final_outcome"] == (
        "failed_after_retry"
    )
    assert metadata[
        "aawm_passthrough_hidden_retry_failure_classification"
    ] == "pre_first_byte_transient"
    assert metadata["google_generate_content_error_count"] == 1
    assert "will not be retried" in warning.calls[-1][0]
    assert "exhausted hidden retries" in error.calls[-1][0]
    assert error.calls[-1][2]["exc_info"] is True


@pytest.mark.asyncio
async def test_non_retryable_failure_is_raised_without_wait_or_cooldown() -> None:
    sleeps: list[float] = []
    bounded_caches: list[MutableMapping[Any, Any]] = []
    terminal_error = _UpstreamError("bad request", status_code=400)

    async def pass_through_request(**_kwargs: Any) -> Response:
        raise terminal_error

    rate_limit = _configure_runtime(
        pass_through_request=pass_through_request,
        sleeps=sleeps,
        bounded_caches=bounded_caches,
    )

    with pytest.raises(_UpstreamError) as exc_info:
        await retry_runtime._perform_google_adapter_pass_through_request(
            custom_body={},
        )

    assert exc_info.value is terminal_error
    assert sleeps == []
    assert bounded_caches == []
    assert rate_limit.until_monotonic_by_key == {}


def test_semaphore_delegates_to_canonical_process_cache_state() -> None:
    async def pass_through_request(**_kwargs: Any) -> Response:
        return Response()

    _configure_runtime(pass_through_request=pass_through_request)

    first = retry_runtime._get_google_adapter_semaphore(
        model="gemini-test",
        rate_limit_key="shared-lane",
    )
    second = retry_runtime._get_google_adapter_semaphore(
        model="different-model",
        rate_limit_key="shared-lane",
    )
    other = retry_runtime._get_google_adapter_semaphore(
        model="gemini-test",
        rate_limit_key="other-lane",
    )

    assert first is second
    assert first is process_cache._google_adapter_semaphores[
        ("shared-lane", 2)
    ]
    assert first is not other
    assert first._value == 2


@pytest.mark.asyncio
async def test_late_installed_cooldown_callback_is_awaited_with_lane_key() -> None:
    """After configure_google_retry_runtime, installing a callback into
    host_globals must be picked up by _runtime_function on the next call
    (late-binding regression for _wait_for_google_adapter_cooldown_if_needed)."""

    response = Response(content=b"late-binding-ok", status_code=200)

    async def pass_through_request(**_kwargs: Any) -> Response:
        return response

    _configure_runtime(pass_through_request=pass_through_request)

    # After configuration, install an async spy into host_globals.
    runtime = retry_runtime._require_runtime()
    assert isinstance(runtime.host_globals, dict)
    cooldown_spy = AsyncMock(return_value=None)
    runtime.host_globals["_wait_for_google_adapter_cooldown_if_needed"] = (
        cooldown_spy
    )

    try:
        result = await retry_runtime._perform_google_adapter_pass_through_request(
            custom_body={"model": "gemini-late-bind"},
            google_adapter_rate_limit_key="late-lane-42",
        )

        # The response is returned successfully.
        assert result is response
        # The late-installed callback was awaited exactly once with the lane key.
        cooldown_spy.assert_awaited_once_with("late-lane-42")
    finally:
        # Restore host_globals to avoid leaking into other tests.
        del runtime.host_globals["_wait_for_google_adapter_cooldown_if_needed"]


def test_same_object_fallback_does_not_recurse() -> None:
    """When host_globals contains the exact same object as the fallback,
    _runtime_function must return the fallback without infinite recursion."""

    async def pass_through_request(**_kwargs: Any) -> Response:
        return Response()

    _configure_runtime(pass_through_request=pass_through_request)

    runtime = retry_runtime._require_runtime()
    assert isinstance(runtime.host_globals, dict)

    # Put the module's own function into host_globals (same object as fallback)
    runtime.host_globals["_google_adapter_hidden_retry_metadata"] = (
        retry_runtime._google_adapter_hidden_retry_metadata
    )

    # This must not recurse; _runtime_function should detect candidate is fallback
    resolved = retry_runtime._runtime_function(
        "_google_adapter_hidden_retry_metadata",
        retry_runtime._google_adapter_hidden_retry_metadata,
    )
    assert resolved is retry_runtime._google_adapter_hidden_retry_metadata

    # Also verify it works end-to-end without RecursionError
    result = retry_runtime._google_adapter_hidden_retry_metadata(
        {"custom_body": {"litellm_metadata": {"key": "val"}}}
    )
    assert result == {"key": "val"}

    # Clean up
    del runtime.host_globals["_google_adapter_hidden_retry_metadata"]
