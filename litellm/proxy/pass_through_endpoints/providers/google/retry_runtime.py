"""Google adapter retry, cooldown, and semaphore runtime.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``. Host
dependencies are supplied explicitly so this module does not import the god
module, while semaphore state remains owned by Google ``process_cache``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Response

from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache as _anthropic_google_process_cache,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    retry as _aawm_alias_retry,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    MonotonicCooldownMap,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES,
)

_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES = frozenset(
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES
)


@dataclass(frozen=True)
class Runtime:
    """Host-owned dependencies required by the Google retry runtime."""

    process_cache_runtime: _anthropic_google_process_cache.Runtime
    rate_limit: MonotonicCooldownMap
    get_rate_limit_key_from_kwargs: Callable[[dict[str, Any]], str]
    get_max_retries: Callable[[], int]
    coerce_non_negative_int: Callable[[Any, int], int]
    coerce_non_negative_float: Callable[[Any, float], float]
    get_model_capacity_max_retries: Callable[[], int]
    get_capacity_backoff_seconds: Callable[[int], float]
    get_hidden_retry_budget_seconds: Callable[[], float]
    get_transient_retry_max_attempts: Callable[[], int]
    get_transient_backoff_seconds: Callable[[int], float]
    extract_exception_status_code: Callable[[Any], Optional[int]]
    extract_error_reason: Callable[[Any], Optional[str]]
    parse_rate_limit_reset_seconds: Callable[[Any], float]
    is_transient_retryable_failure: Callable[..., bool]
    classify_hidden_retry_failure: Callable[
        [Exception], tuple[Optional[int], str, Optional[str]]
    ]
    record_error_for_logging: Callable[..., None]
    record_hidden_retry_metadata: Callable[..., None]
    build_terminal_error_log_context: Callable[..., dict[str, Any]]
    pass_through_request: Callable[..., Awaitable[Response]]
    bound_token_cache: Callable[[MutableMapping[Any, Any]], None]
    sleep: Callable[[float], Awaitable[None]]
    log_debug: Callable[..., None]
    log_warning: Callable[..., None]
    log_error: Callable[..., None]
    host_globals: Mapping[str, object] = field(default_factory=dict)


_runtime: Optional[Runtime] = None


def configure_google_retry_runtime(runtime: Runtime) -> None:
    """Install host callbacks and shared state for Google retry operations."""

    global _runtime
    _runtime = runtime


def _require_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Google retry runtime has not been configured")
    return _runtime


def _runtime_function(
    name: str,
    fallback: Callable[..., Any],
) -> Callable[..., Any]:
    candidate = _require_runtime().host_globals.get(name)
    if callable(candidate) and candidate is not fallback:
        return candidate
    return fallback


def _get_google_adapter_semaphore(
    model: Optional[str] = None,
    *,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> asyncio.Semaphore:
    runtime = _require_runtime()
    return _anthropic_google_process_cache._get_google_adapter_semaphore(
        model,
        runtime=runtime.process_cache_runtime,
        access_token=access_token,
        companion_project=companion_project,
        rate_limit_key=rate_limit_key,
    )


def _google_adapter_hidden_retry_kwargs_from_passthrough_kwargs(
    passthrough_kwargs: dict[str, Any],
) -> dict[str, Any]:
    custom_body = passthrough_kwargs.get("custom_body")
    if not isinstance(custom_body, dict):
        return {}
    metadata = custom_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        custom_body["litellm_metadata"] = metadata
    return {"litellm_params": {"metadata": metadata}}


def _record_google_adapter_hidden_retry_metadata(
    passthrough_kwargs: dict[str, Any],
    *,
    attempt_number: int,
    max_attempts: int,
    status_code: Optional[int],
    failure_class: str,
    wait_seconds: float,
    final_outcome: Optional[str] = None,
    failure_classification: Optional[str] = None,
) -> None:
    kwargs = _runtime_function(
        "_google_adapter_hidden_retry_kwargs_from_passthrough_kwargs",
        _google_adapter_hidden_retry_kwargs_from_passthrough_kwargs,
    )(passthrough_kwargs)
    if not kwargs:
        return
    _require_runtime().record_hidden_retry_metadata(
        kwargs,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=wait_seconds,
        final_outcome=final_outcome,
        failure_classification=failure_classification,
    )


def _record_google_adapter_terminal_transient_failure_metadata(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    attempt: int,
    max_attempts: int,
    status_code: Optional[int],
    error_reason: Optional[str],
    failure_class: str,
    failure_classification: Optional[str],
) -> None:
    runtime = _require_runtime()
    runtime.record_error_for_logging(
        passthrough_kwargs,
        exc=exc,
        status_code=status_code,
        error_reason=error_reason,
        attempt=attempt,
        wait_seconds=0.0,
    )
    _runtime_function(
        "_record_google_adapter_hidden_retry_metadata",
        _record_google_adapter_hidden_retry_metadata,
    )(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=0.0,
        final_outcome=(
            "failed_after_retry" if attempt > 1 else "failed_without_retry"
        ),
        failure_classification=failure_classification,
    )


def _google_adapter_hidden_retry_metadata(
    passthrough_kwargs: dict[str, Any],
) -> dict[str, Any]:
    custom_body = passthrough_kwargs.get("custom_body")
    if not isinstance(custom_body, dict):
        return {}
    metadata = custom_body.get("litellm_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _record_google_adapter_success_after_transient_retry(
    passthrough_kwargs: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int,
) -> None:
    metadata = _runtime_function(
        "_google_adapter_hidden_retry_metadata",
        _google_adapter_hidden_retry_metadata,
    )(passthrough_kwargs)
    if not metadata.get("aawm_passthrough_hidden_retry_count"):
        return
    if metadata.get("aawm_passthrough_hidden_retry_final_outcome"):
        return
    _runtime_function(
        "_record_google_adapter_hidden_retry_metadata",
        _record_google_adapter_hidden_retry_metadata,
    )(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=max_attempts,
        status_code=None,
        failure_class="success",
        wait_seconds=0.0,
        final_outcome="success_after_retry",
    )


def _log_google_adapter_terminal_transient_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    failure_classification: Optional[str],
) -> None:
    runtime = _require_runtime()
    metadata = _runtime_function(
        "_google_adapter_hidden_retry_metadata",
        _google_adapter_hidden_retry_metadata,
    )(passthrough_kwargs)
    runtime.log_error(
        (
            "Google adapter exhausted hidden retries for transient upstream "
            "failure status=%s error=%s final_outcome=%s retry_count=%s"
        ),
        status_code,
        str(exc),
        metadata.get("aawm_passthrough_hidden_retry_final_outcome"),
        metadata.get("aawm_passthrough_hidden_retry_count"),
        extra=runtime.build_terminal_error_log_context(
            passthrough_kwargs,
            status_code=status_code,
            failure_classification=failure_classification,
        ),
        exc_info=True,
    )


async def _wait_for_google_adapter_cooldown_if_needed(
    rate_limit_key: str,
) -> None:
    runtime = _require_runtime()
    await _aawm_alias_retry.wait_for_monotonic_cooldown_map(
        runtime.rate_limit,
        rate_limit_key,
        log_label="Google adapter",
        sleep=runtime.sleep,
    )


async def _set_google_adapter_cooldown(
    rate_limit_key: str,
    wait_seconds: float,
) -> None:
    runtime = _require_runtime()
    async with runtime.rate_limit.lock:
        runtime.rate_limit.extend(
            rate_limit_key,
            wait_seconds,
            max_size=None,
        )
        runtime.bound_token_cache(
            runtime.rate_limit.until_monotonic_by_key
        )


async def _handle_google_adapter_rate_limit_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    error_reason: Optional[str],
    attempt: int,
    retry_limit: int,
    wait_seconds: float,
    rate_limit_key: str,
    accumulated_hidden_wait_seconds: float,
    hidden_retry_budget_seconds: float,
    is_capacity_retry: bool,
) -> float:
    runtime = _require_runtime()
    runtime.record_error_for_logging(
        passthrough_kwargs,
        exc=exc,
        status_code=status_code,
        error_reason=error_reason,
        attempt=attempt,
        wait_seconds=wait_seconds,
    )
    projected_hidden_wait_seconds = (
        accumulated_hidden_wait_seconds + wait_seconds
    )
    within_hidden_budget = (
        hidden_retry_budget_seconds > 0
        and projected_hidden_wait_seconds <= hidden_retry_budget_seconds
    )
    if attempt >= retry_limit and not within_hidden_budget:
        runtime.log_warning(
            "Google adapter upstream attempt %s failed with %s (%s, reason=%s) and will not be retried",
            attempt,
            status_code,
            exc.__class__.__name__,
            error_reason,
        )
        raise exc
    if attempt >= retry_limit and within_hidden_budget:
        runtime.log_warning(
            "Google adapter keeping 429 hidden from client for %s; hidden retry wait %.1fs/%.1fs (reason=%s)",
            rate_limit_key,
            projected_hidden_wait_seconds,
            hidden_retry_budget_seconds,
            error_reason,
        )
    if is_capacity_retry:
        runtime.log_warning(
            "Google adapter upstream attempt %s hit 429 (%s, reason=%s); exponential backoff %.1fs",
            attempt,
            exc.__class__.__name__,
            error_reason,
            wait_seconds,
        )
    else:
        runtime.log_warning(
            "Google adapter upstream attempt %s hit 429 (%s, reason=%s); parsed reset %.1fs",
            attempt,
            exc.__class__.__name__,
            error_reason,
            wait_seconds,
        )
    await _runtime_function(
        "_set_google_adapter_cooldown",
        _set_google_adapter_cooldown,
    )(rate_limit_key, wait_seconds + 1.0)
    return projected_hidden_wait_seconds


async def _handle_google_adapter_transient_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    error_reason: Optional[str],
    attempt: int,
    transient_retry_max_attempts: int,
    failure_class: str,
    failure_classification: Optional[str],
) -> None:
    runtime = _require_runtime()
    transient_wait_seconds = runtime.get_transient_backoff_seconds(attempt)
    if attempt >= transient_retry_max_attempts:
        _runtime_function(
            "_record_google_adapter_terminal_transient_failure_metadata",
            _record_google_adapter_terminal_transient_failure_metadata,
        )(
            passthrough_kwargs,
            exc=exc,
            attempt=attempt,
            max_attempts=transient_retry_max_attempts,
            status_code=status_code,
            error_reason=error_reason,
            failure_class=failure_class,
            failure_classification=failure_classification,
        )
        runtime.log_warning(
            "Google adapter upstream attempt %s failed with transient %s (%s, reason=%s) and will not be retried",
            attempt,
            status_code,
            exc.__class__.__name__,
            error_reason,
        )
        _runtime_function(
            "_log_google_adapter_terminal_transient_failure",
            _log_google_adapter_terminal_transient_failure,
        )(
            passthrough_kwargs,
            exc=exc,
            status_code=status_code,
            failure_classification=failure_classification,
        )
        raise exc
    runtime.log_warning(
        "Google adapter upstream attempt %s hit transient %s (%s, reason=%s); hidden retry wait %.1fs",
        attempt,
        status_code,
        exc.__class__.__name__,
        error_reason,
        transient_wait_seconds,
    )
    _runtime_function(
        "_record_google_adapter_hidden_retry_metadata",
        _record_google_adapter_hidden_retry_metadata,
    )(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=transient_retry_max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=transient_wait_seconds,
        failure_classification=failure_classification,
    )
    await runtime.sleep(transient_wait_seconds)


async def _perform_google_adapter_pass_through_request(
    **kwargs: Any,
) -> Response:
    runtime = _require_runtime()
    passthrough_kwargs = dict(kwargs)
    max_retries = runtime.coerce_non_negative_int(
        passthrough_kwargs.pop("google_adapter_max_retries", None),
        runtime.get_max_retries(),
    )
    total_attempts = max_retries + 1
    capacity_total_attempts = (
        runtime.coerce_non_negative_int(
            passthrough_kwargs.pop(
                "google_adapter_model_capacity_max_retries",
                None,
            ),
            runtime.get_model_capacity_max_retries(),
        )
        + 1
    )
    hidden_retry_budget_seconds = runtime.coerce_non_negative_float(
        passthrough_kwargs.pop(
            "google_adapter_hidden_retry_budget_seconds",
            None,
        ),
        runtime.get_hidden_retry_budget_seconds(),
    )
    accumulated_hidden_wait_seconds = 0.0
    rate_limit_key = runtime.get_rate_limit_key_from_kwargs(kwargs)
    transient_retry_max_attempts = (
        runtime.get_transient_retry_max_attempts()
    )
    passthrough_kwargs.pop("google_access_token", None)
    passthrough_kwargs.pop("google_adapter_rate_limit_key", None)

    async def _before_attempt(attempt: int) -> None:
        runtime.log_debug(
            "Google adapter upstream attempt %s/%s",
            attempt,
            max(
                total_attempts,
                capacity_total_attempts,
                transient_retry_max_attempts,
            ),
        )
        await _runtime_function(
            "_wait_for_google_adapter_cooldown_if_needed",
            _wait_for_google_adapter_cooldown_if_needed,
        )(rate_limit_key)

    async def _operation() -> Response:
        passthrough_kwargs["retryable_upstream_status_codes"] = sorted(
            {429, *_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES}
        )
        passthrough_kwargs["caller_managed_hidden_retry"] = True
        return await runtime.pass_through_request(**passthrough_kwargs)

    async def _on_success(_response: Response, attempt: int) -> None:
        _runtime_function(
            "_record_google_adapter_success_after_transient_retry",
            _record_google_adapter_success_after_transient_retry,
        )(
            passthrough_kwargs,
            attempt=attempt,
            max_attempts=transient_retry_max_attempts,
        )

    async def _on_failure(exc: Exception, attempt: int) -> bool:
        nonlocal accumulated_hidden_wait_seconds
        status_code = runtime.extract_exception_status_code(exc)
        error_reason = runtime.extract_error_reason(exc)
        is_capacity_retry = error_reason == "MODEL_CAPACITY_EXHAUSTED"
        is_rate_limit_retry = status_code == 429 or error_reason in {
            "MODEL_CAPACITY_EXHAUSTED",
            "RATE_LIMIT_EXCEEDED",
        }
        is_transient_retry = runtime.is_transient_retryable_failure(
            exc,
            status_code=status_code,
            error_reason=error_reason,
        )
        failure_class = exc.__class__.__name__
        failure_classification: Optional[str] = None
        if is_transient_retry:
            (
                classified_status_code,
                failure_class,
                failure_classification,
            ) = runtime.classify_hidden_retry_failure(exc)
            if classified_status_code is not None and status_code is None:
                status_code = classified_status_code
        retry_limit = (
            capacity_total_attempts if is_capacity_retry else total_attempts
        )
        if is_capacity_retry:
            wait_seconds = runtime.get_capacity_backoff_seconds(attempt)
        else:
            wait_seconds = runtime.parse_rate_limit_reset_seconds(exc)
        if is_rate_limit_retry:
            accumulated_hidden_wait_seconds = (
                await _runtime_function(
                    "_handle_google_adapter_rate_limit_failure",
                    _handle_google_adapter_rate_limit_failure,
                )(
                    passthrough_kwargs,
                    exc=exc,
                    status_code=status_code,
                    error_reason=error_reason,
                    attempt=attempt,
                    retry_limit=retry_limit,
                    wait_seconds=wait_seconds,
                    rate_limit_key=rate_limit_key,
                    accumulated_hidden_wait_seconds=(
                        accumulated_hidden_wait_seconds
                    ),
                    hidden_retry_budget_seconds=(
                        hidden_retry_budget_seconds
                    ),
                    is_capacity_retry=is_capacity_retry,
                )
            )
            return True
        if is_transient_retry:
            await _runtime_function(
                "_handle_google_adapter_transient_failure",
                _handle_google_adapter_transient_failure,
            )(
                passthrough_kwargs,
                exc=exc,
                status_code=status_code,
                error_reason=error_reason,
                attempt=attempt,
                transient_retry_max_attempts=(
                    transient_retry_max_attempts
                ),
                failure_class=failure_class,
                failure_classification=failure_classification,
            )
            return True
        runtime.log_warning(
            "Google adapter upstream attempt %s failed with %s (%s, reason=%s) and will not be retried",
            attempt,
            status_code,
            exc.__class__.__name__,
            error_reason,
        )
        return False

    return await _aawm_alias_retry.run_adapter_retry_policy(
        _operation,
        policy=_aawm_alias_retry.AdapterRetryPolicy(
            before_attempt=_before_attempt,
            on_failure=_on_failure,
            on_success=_on_success,
        ),
    )
