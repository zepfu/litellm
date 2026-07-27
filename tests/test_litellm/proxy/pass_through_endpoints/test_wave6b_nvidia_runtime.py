"""Direct tests for the Wave 6B NVIDIA runtime extraction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.providers.nvidia import (
    runtime as nvidia_runtime,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __call__(self, message: str, *args: Any) -> None:
        self.calls.append((message, args))


class _UpstreamError(Exception):
    def __init__(self, message: str, status_code: Any) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _reset_runtime_dependencies() -> Any:
    nvidia_runtime.configure_nvidia_runtime(
        nvidia_runtime.DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES
    )
    yield
    nvidia_runtime.configure_nvidia_runtime(
        nvidia_runtime.DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES
    )


def _dependencies(
    **overrides: Any,
) -> nvidia_runtime.NvidiaRuntimeDependencies:
    return replace(
        nvidia_runtime.DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES,
        **overrides,
    )


def test_target_resolution_preserves_precedence_and_v1_normalization() -> None:
    env = {
        "NVIDIA_NIM_API_BASE": ' "https://nim.example.test/root/v1/" ',
        "AAWM_NVIDIA_API_BASE": "https://aawm.example.test/v1",
    }
    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(get_env=lambda name: env.get(name))
    )

    assert (
        nvidia_runtime._get_anthropic_adapter_nvidia_target_base()
        == "https://nim.example.test/root"
    )

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(
            get_env=lambda name: (
                "https://aawm.example.test/v1"
                if name == "AAWM_NVIDIA_API_BASE"
                else None
            )
        )
    )
    assert (
        nvidia_runtime._get_anthropic_adapter_nvidia_target_base()
        == "https://aawm.example.test"
    )

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(get_env=lambda _name: None)
    )
    assert (
        nvidia_runtime._get_anthropic_adapter_nvidia_target_base()
        == "https://integrate.api.nvidia.com"
    )


def test_configuration_callbacks_drive_keys_and_runtime_settings() -> None:
    secret_requests: list[tuple[str, ...]] = []
    cleaned_values: list[Any] = []
    env = {
        "AAWM_NVIDIA_ADAPTER_MAX_RETRIES": " 3 ",
        "AAWM_NVIDIA_ADAPTER_INNER_MAX_RETRIES": " 2 ",
        "AAWM_NVIDIA_ADAPTER_REQUEST_TIMEOUT_SECONDS": " 4 ",
        "AAWM_NVIDIA_ADAPTER_FORCE_FAKE_STREAM_MODELS": "model/a, model/b",
    }

    def get_first_secret_value(names: tuple[str, ...]) -> str:
        secret_requests.append(names)
        return "configured-key"

    def clean_auth_value(value: Any) -> Optional[str]:
        cleaned_values.append(value)
        if not isinstance(value, str):
            return None
        return value.strip() or None

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(
            get_first_secret_value=get_first_secret_value,
            clean_auth_value=clean_auth_value,
            get_env=lambda name: env.get(name),
        )
    )

    assert (
        nvidia_runtime._get_anthropic_adapter_nvidia_api_key()
        == "configured-key"
    )
    assert secret_requests == [
        (
            "AAWM_NVIDIA_API_KEY",
            "NVIDIA_NIM_API_KEY",
            "NVIDIA_API_KEY",
        )
    ]
    assert nvidia_runtime._get_nvidia_adapter_max_retries() == 3
    assert nvidia_runtime._get_nvidia_adapter_inner_max_retries() == 2
    assert (
        nvidia_runtime._get_nvidia_adapter_request_timeout_seconds("model/a")
        == 5.0
    )
    assert nvidia_runtime._should_force_fake_stream_for_nvidia_adapter_model(
        "model/b"
    )
    assert cleaned_values


@pytest.mark.asyncio
async def test_perform_operation_returns_success_without_retry() -> None:
    debug = _RecordingLogger()
    warning = _RecordingLogger()
    calls = 0

    async def operation() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(
            get_env=lambda _name: None,
            log_debug=debug,
            log_warning=warning,
        )
    )

    result = await nvidia_runtime._perform_nvidia_completion_adapter_operation(
        adapter_model="nvidia/test-model",
        operation=operation,
    )

    assert result == {"ok": True}
    assert calls == 1
    assert len(debug.calls) == 1
    assert warning.calls == []


@pytest.mark.asyncio
async def test_perform_operation_retries_and_waits_before_success() -> None:
    waits: list[float] = []
    warning = _RecordingLogger()
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _UpstreamError("temporary failure", "503")
        return "complete"

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(
            get_env=lambda name: (
                "2"
                if name == "AAWM_NVIDIA_ADAPTER_MAX_RETRIES"
                else None
            ),
            sleep=sleep,
            log_warning=warning,
        )
    )

    result = await nvidia_runtime._perform_nvidia_completion_adapter_operation(
        adapter_model="nvidia/test-model",
        operation=operation,
    )

    assert result == "complete"
    assert calls == 2
    assert waits == [1.0]
    assert len(warning.calls) == 1
    assert "backoff %.1fs" in warning.calls[0][0]


@pytest.mark.asyncio
async def test_perform_operation_raises_http_exception_on_terminal_failure() -> None:
    waits: list[float] = []
    warning = _RecordingLogger()
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise _UpstreamError("still unavailable", 503)

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    nvidia_runtime.configure_nvidia_runtime(
        _dependencies(
            get_env=lambda name: (
                "1"
                if name == "AAWM_NVIDIA_ADAPTER_MAX_RETRIES"
                else None
            ),
            sleep=sleep,
            log_warning=warning,
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await nvidia_runtime._perform_nvidia_completion_adapter_operation(
            adapter_model="nvidia/test-model",
            operation=operation,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "still unavailable"
    assert calls == 2
    assert waits == [1.0]
    assert len(warning.calls) == 2
    assert "will not be retried" in warning.calls[-1][0]


def test_status_and_wait_helpers_preserve_error_classification() -> None:
    timeout_error = type("Timeout", (Exception,), {})("request timed out")

    assert (
        nvidia_runtime._extract_nvidia_adapter_exception_status_code(
            _UpstreamError("rate limited", "429")
        )
        == 429
    )
    assert (
        nvidia_runtime._extract_nvidia_adapter_exception_status_code(
            timeout_error
        )
        == 504
    )
    assert (
        nvidia_runtime._extract_nvidia_adapter_exception_status_code(
            RuntimeError("upstream returned 502")
        )
        == 502
    )
    assert (
        nvidia_runtime._extract_nvidia_adapter_exception_status_code(
            RuntimeError("unclassified")
        )
        is None
    )
    assert [
        nvidia_runtime._get_nvidia_adapter_retry_wait_seconds(attempt)
        for attempt in (1, 2, 3, 4, 5, 8)
    ] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
