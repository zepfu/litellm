"""NVIDIA adapter target, configuration, and retry runtime.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``. Host
dependencies are supplied through ``configure_nvidia_runtime`` so this module
does not import the god module.
"""

from __future__ import annotations

import asyncio
import os
import re
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import get_secret_str


_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS = (
    "AAWM_NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_API_KEY",
)
_ANTHROPIC_ADAPTER_NVIDIA_RETRYABLE_STATUS_CODES = frozenset(
    {408, 429, 500, 502, 503, 504}
)
NVIDIA_API_BASE_VERSION_SEGMENT = "/v1"
"""The single API version segment appended when building NVIDIA transport URLs."""

NVIDIA_TARGET_BASE_DEFAULT = "https://integrate.api.nvidia.com"
"""Canonical default NVIDIA target root, stored without a version segment."""


@dataclass(frozen=True)
class NvidiaRuntimeDependencies:
    """Callbacks supplied by the passthrough host during integration."""

    get_first_secret_value: Callable[[tuple[str, ...]], Optional[str]]
    clean_secret_string: Callable[[Optional[str]], Optional[str]]
    clean_auth_value: Callable[[Any], Optional[str]]
    get_env: Callable[[str], Optional[str]]
    sleep: Callable[[float], Awaitable[Any]]
    log_debug: Callable[..., None]
    log_warning: Callable[..., None]


def _default_clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _default_get_first_secret_value(
    secret_names: tuple[str, ...],
) -> Optional[str]:
    for secret_name in secret_names:
        value = _default_clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None


def _default_clean_auth_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _default_get_env(name: str) -> Optional[str]:
    return os.getenv(name)


async def _default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES = NvidiaRuntimeDependencies(
    get_first_secret_value=_default_get_first_secret_value,
    clean_secret_string=_default_clean_secret_string,
    clean_auth_value=_default_clean_auth_value,
    get_env=_default_get_env,
    sleep=_default_sleep,
    log_debug=verbose_proxy_logger.debug,
    log_warning=verbose_proxy_logger.warning,
)

_runtime_dependencies = DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES


def configure_nvidia_runtime(
    dependencies: NvidiaRuntimeDependencies,
) -> None:
    """Install the callbacks used by the extracted NVIDIA runtime."""

    global _runtime_dependencies
    _runtime_dependencies = dependencies


def _get_anthropic_adapter_nvidia_api_key() -> Optional[str]:
    return _runtime_dependencies.get_first_secret_value(
        _ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS
    )


def _nvidia_target_base_api_version_violation(target_base: str) -> Optional[str]:
    """Return a deterministic rejection reason for ambiguous version paths.

    The canonical NVIDIA target root never carries a version segment: exactly
    one trailing ``/v1`` is normalized away, and any additional version
    segment nested inside the path is rejected instead of silently rewritten,
    so an operator's intentional gateway path cannot be doubled or mangled.
    """

    parsed = urlsplit(target_base)
    segments = [
        segment
        for segment in parsed.path.split("/")
        if segment
    ]
    version_indexes = [
        index
        for index, segment in enumerate(segments)
        if segment == NVIDIA_API_BASE_VERSION_SEGMENT.lstrip("/")
    ]
    if not version_indexes:
        return None
    if version_indexes == [len(segments) - 1]:
        return None
    return (
        "NVIDIA API base target "
        f"'{target_base}' must carry at most one trailing "
        f"'{NVIDIA_API_BASE_VERSION_SEGMENT}' version segment; nested version "
        "paths are ambiguous and are rejected instead of rewritten."
    )


def _canonical_nvidia_target_base(target_base: str) -> str:
    """Normalize one NVIDIA target root without any version segment."""

    cleaned = target_base.rstrip("/")
    if cleaned.endswith(NVIDIA_API_BASE_VERSION_SEGMENT):
        cleaned = cleaned[: -len(NVIDIA_API_BASE_VERSION_SEGMENT)]
    return cleaned.rstrip("/") or NVIDIA_TARGET_BASE_DEFAULT


def _get_anthropic_adapter_nvidia_target_base() -> str:
    raw_base = (
        _runtime_dependencies.clean_secret_string(
            _runtime_dependencies.get_env("NVIDIA_NIM_API_BASE")
        )
        or _runtime_dependencies.clean_secret_string(
            _runtime_dependencies.get_env("AAWM_NVIDIA_API_BASE")
        )
        or NVIDIA_TARGET_BASE_DEFAULT
    )
    if raw_base == NVIDIA_TARGET_BASE_DEFAULT:
        return NVIDIA_TARGET_BASE_DEFAULT
    violation = _nvidia_target_base_api_version_violation(raw_base)
    if violation is not None:
        raise ValueError(violation)
    return _canonical_nvidia_target_base(raw_base)


def _nvidia_api_base_from_target_base(target_base: str) -> str:
    """Build the NVIDIA API base by appending the version segment once."""
    violation = _nvidia_target_base_api_version_violation(target_base)
    if violation is not None:
        raise ValueError(violation)
    return f"{_canonical_nvidia_target_base(target_base)}{NVIDIA_API_BASE_VERSION_SEGMENT}"


def _get_nvidia_adapter_max_retries() -> int:
    raw_value = _runtime_dependencies.clean_auth_value(
        _runtime_dependencies.get_env("AAWM_NVIDIA_ADAPTER_MAX_RETRIES")
    )
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except Exception:
        return 1
    return max(0, parsed)


def _get_nvidia_adapter_request_timeout_seconds(
    adapter_model: Optional[str] = None,
) -> float:
    raw_value = _runtime_dependencies.clean_auth_value(
        _runtime_dependencies.get_env(
            "AAWM_NVIDIA_ADAPTER_REQUEST_TIMEOUT_SECONDS"
        )
    )
    if raw_value is None:
        if _should_force_fake_stream_for_nvidia_adapter_model(adapter_model):
            return 240.0
        return 120.0
    try:
        parsed = float(raw_value)
    except Exception:
        if _should_force_fake_stream_for_nvidia_adapter_model(adapter_model):
            return 240.0
        return 120.0
    return max(5.0, parsed)


def _get_nvidia_adapter_inner_max_retries() -> int:
    raw_value = _runtime_dependencies.clean_auth_value(
        _runtime_dependencies.get_env(
            "AAWM_NVIDIA_ADAPTER_INNER_MAX_RETRIES"
        )
    )
    if raw_value is None:
        return 0
    try:
        parsed = int(raw_value)
    except Exception:
        return 0
    return max(0, parsed)


def _should_force_fake_stream_for_nvidia_adapter_model(
    adapter_model: Optional[str],
) -> bool:
    configured_models = _runtime_dependencies.clean_auth_value(
        _runtime_dependencies.get_env(
            "AAWM_NVIDIA_ADAPTER_FORCE_FAKE_STREAM_MODELS"
        )
    )
    if configured_models is None:
        normalized_models = {"minimaxai/minimax-m2.7"}
    else:
        normalized_models = {
            item.strip() for item in configured_models.split(",") if item.strip()
        }
    return bool(adapter_model and adapter_model in normalized_models)


def _extract_nvidia_adapter_exception_status_code(
    exc: Any,
) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue

    text_value = str(exc)
    if (
        "Timeout Error" in text_value
        or exc.__class__.__name__.lower() == "timeout"
    ):
        return 504

    match = re.search(r"\b(408|429|500|502|503|504)\b", text_value)
    if match is not None:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _get_nvidia_adapter_retry_wait_seconds(attempt: int) -> float:
    return min(float(2 ** max(0, attempt - 1)), 8.0)


async def _perform_nvidia_completion_adapter_operation(
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    max_retries = _get_nvidia_adapter_max_retries()
    total_attempts = max_retries + 1
    attempt = 0
    while True:
        attempt += 1
        _runtime_dependencies.log_debug(
            "NVIDIA completion adapter upstream attempt %s/%s for model=%s",
            attempt,
            total_attempts,
            adapter_model,
        )
        try:
            return await operation()
        except Exception as exc:
            status_code = _extract_nvidia_adapter_exception_status_code(exc)
            raw_message = str(exc)
            if (
                status_code
                not in _ANTHROPIC_ADAPTER_NVIDIA_RETRYABLE_STATUS_CODES
                or attempt >= total_attempts
            ):
                _runtime_dependencies.log_warning(
                    "NVIDIA completion adapter upstream attempt %s failed with %s (%s, raw=%s) and will not be retried",
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    raw_message,
                )
                raise HTTPException(
                    status_code=status_code or 502,
                    detail=raw_message,
                )
            wait_seconds = _get_nvidia_adapter_retry_wait_seconds(attempt)
            _runtime_dependencies.log_warning(
                "NVIDIA completion adapter upstream attempt %s hit %s (%s, raw=%s); backoff %.1fs",
                attempt,
                status_code,
                exc.__class__.__name__,
                raw_message,
                wait_seconds,
            )
            await _runtime_dependencies.sleep(wait_seconds)
