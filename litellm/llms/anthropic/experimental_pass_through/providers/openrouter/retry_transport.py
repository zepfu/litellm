"""OpenRouter-owned retry, cooldown, error shaping, and transport execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from fastapi import HTTPException, Response

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import retry
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    MonotonicCooldownMap,
)
from .error_shape import (
    extract_error_headers,
    extract_error_payload,
    extract_exception_status_code,
    extract_provider_name,
    extract_raw_message,
    extract_reset_wait_seconds,
    extract_retry_after_seconds,
    get_header_value,
    is_long_window_rate_limit,
    is_no_endpoint_candidate_error,
    is_provider_raw_error,
)

RetryResultT = TypeVar("RetryResultT")


@dataclass(frozen=True)
class Runtime:
    """Route-layer state and callbacks required by OpenRouter retry/transport."""

    rate_limit: MonotonicCooldownMap
    failure_circuit_until_monotonic_by_key: dict[str, float]
    clean_secret_string: Callable[[Optional[str]], Optional[str]]
    extract_embedded_json_payload_candidates: Callable[
        [object], Iterable[str]
    ]
    parse_json_payloads_from_text_candidates: Callable[
        [Iterable[str]], Iterable[object]
    ]
    extract_upstream_headers: Callable[[object], Mapping[str, object]]
    parse_retry_after_seconds_from_headers: Callable[
        [Mapping[str, object]], Optional[float]
    ]
    get_header_value: Callable[[Mapping[str, object], str], Optional[str]]
    parse_reset_wait_seconds_from_headers: Callable[
        [Mapping[str, object]], Optional[float]
    ]
    raise_candidate_unavailable: Callable[[str], None]
    maybe_raise_alias_probe_cooldown: Callable[..., Awaitable[None]]
    get_completion_model: Callable[[Optional[str]], Optional[str]]
    pass_through_request: Callable[..., Awaitable[Response]]
    wait_for_cooldown: Callable[..., Awaitable[None]]
    set_cooldown_callback: Callable[..., Awaitable[None]]
    maybe_raise_failure_circuit_open_callback: Callable[..., Awaitable[None]]
    open_failure_circuit_callback: Callable[..., Awaitable[None]]
    clear_failure_circuit_callback: Callable[[Optional[str]], None]
    log_debug: Callable[..., object]
    log_warning: Callable[..., object]
    getenv: Callable[[str], Optional[str]]
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic


def get_rate_limit_key(runtime: Runtime, model: Optional[str]) -> str:
    cleaned_model = runtime.clean_secret_string(model)
    return cleaned_model or "__default__"


def is_free_model(runtime: Runtime, model: Optional[str]) -> bool:
    cleaned_model = runtime.clean_secret_string(model)
    if not cleaned_model:
        return False
    return (
        cleaned_model == "openrouter/elephant-alpha"
        or cleaned_model == "openrouter/free"
        or cleaned_model.endswith(":free")
    )


def get_wait_keys(runtime: Runtime, model: Optional[str]) -> str:
    return get_rate_limit_key(runtime, model)


def maybe_raise_alias_probe_no_endpoint_unavailable(
    runtime: Runtime,
    exc: object,
    *,
    adapter_model: Optional[str],
    use_alias_candidate_probe: bool,
    status_code: Optional[int] = None,
    raw_message: Optional[str] = None,
) -> None:
    if not use_alias_candidate_probe:
        return
    if not is_no_endpoint_candidate_error(
        runtime,
        exc,
        status_code=status_code,
        raw_message=raw_message,
    ):
        return
    model_label = get_rate_limit_key(runtime, adapter_model)
    detail_text = raw_message or str(exc)
    runtime.raise_candidate_unavailable(
        f"OpenRouter auto-agent candidate {model_label} has no available "
        f"endpoints: {detail_text}"
    )


def get_cooldown_keys(
    runtime: Runtime,
    *,
    model: Optional[str],
    exc: object,
) -> str:
    _ = exc
    return get_rate_limit_key(runtime, model)


def get_retry_wait_seconds(runtime: Runtime, exc: object, attempt: int) -> float:
    wait_seconds = get_backoff_seconds(runtime, attempt)
    retry_after_seconds = extract_retry_after_seconds(runtime, exc)
    if retry_after_seconds is not None:
        retry_after_backoff_seconds = min(
            max(retry_after_seconds + 1.0, 1.0),
            60.0,
        )
        return max(wait_seconds, retry_after_backoff_seconds)
    headers = extract_error_headers(runtime, exc)
    remaining_value = get_header_value(runtime, headers, "X-RateLimit-Remaining")
    reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
    if remaining_value in {"0", "0.0"} and reset_wait_seconds is not None:
        reset_backoff_seconds = min(max(reset_wait_seconds + 1.0, 1.0), 60.0)
        return max(wait_seconds, reset_backoff_seconds)
    return wait_seconds


def get_max_retries(runtime: Runtime) -> int:
    return retry.parse_non_negative_int_env(
        "AAWM_OPENROUTER_ADAPTER_MAX_RETRIES",
        default=3,
        getenv=runtime.getenv,
    )


def get_backoff_seconds(runtime: Runtime, attempt: int) -> float:
    raw_value = runtime.clean_secret_string(
        runtime.getenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS")
    )
    if raw_value:
        try:
            values = [
                max(1.0, float(item.strip()))
                for item in raw_value.split(",")
                if item.strip()
            ]
        except Exception:
            values = []
        if values:
            index = min(max(1, attempt) - 1, len(values) - 1)
            return values[index]
    schedule = (2.0, 10.0, 20.0, 30.0)
    index = min(max(1, attempt) - 1, len(schedule) - 1)
    return schedule[index]


def get_hidden_retry_budget_seconds(runtime: Runtime) -> float:
    return retry.parse_non_negative_float_env(
        "AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS",
        default=0.0,
        getenv=runtime.getenv,
    )


def get_post_failure_cooldown_seconds(runtime: Runtime) -> float:
    raw_value = runtime.clean_secret_string(
        runtime.getenv("AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS")
    )
    if raw_value is None:
        return 60.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 60.0
    return max(0.0, parsed)


async def maybe_raise_failure_circuit_open(
    runtime: Runtime,
    adapter_model: Optional[str],
) -> None:
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    async with runtime.rate_limit.lock:
        wait_seconds = (
            runtime.failure_circuit_until_monotonic_by_key.get(
                rate_limit_key,
                0.0,
            )
            - runtime.monotonic()
        )
    if wait_seconds > 0:
        rounded_wait = max(1, int(wait_seconds))
        runtime.log_warning(
            "OpenRouter adapter failure circuit open for %s; "
            "failing fast for %ss",
            rate_limit_key,
            rounded_wait,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"OpenRouter model {rate_limit_key} is temporarily cooling "
                "down after repeated provider 429s. "
                f"Retry after ~{rounded_wait}s."
            ),
        )


async def open_failure_circuit(
    runtime: Runtime,
    adapter_model: Optional[str],
    *,
    exc: object,
) -> None:
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    cooldown_seconds = get_post_failure_cooldown_seconds(runtime)
    retry_after_seconds = extract_retry_after_seconds(runtime, exc)
    reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
    for candidate in (retry_after_seconds, reset_wait_seconds):
        if candidate is not None:
            cooldown_seconds = max(cooldown_seconds, candidate)
    cooldown_seconds = min(max(cooldown_seconds, 0.0), 300.0)
    async with runtime.rate_limit.lock:
        until = runtime.monotonic() + cooldown_seconds
        current_until = runtime.failure_circuit_until_monotonic_by_key.get(
            rate_limit_key,
            0.0,
        )
        if until > current_until:
            runtime.failure_circuit_until_monotonic_by_key[rate_limit_key] = until


def clear_failure_circuit(
    runtime: Runtime,
    adapter_model: Optional[str],
) -> None:
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    runtime.failure_circuit_until_monotonic_by_key.pop(rate_limit_key, None)


async def get_active_cooldown_seconds(
    runtime: Runtime,
    adapter_model: Optional[str],
) -> float:
    candidate_keys = [get_rate_limit_key(runtime, adapter_model)]
    upstream_model = runtime.get_completion_model(adapter_model)
    upstream_key = get_rate_limit_key(runtime, upstream_model)
    if upstream_key not in candidate_keys:
        candidate_keys.append(upstream_key)
    async with runtime.rate_limit.lock:
        now = runtime.monotonic()
        rate_wait = max(
            (
                runtime.rate_limit.until_monotonic_by_key.get(key, 0.0) - now
                for key in candidate_keys
            ),
            default=0.0,
        )
        circuit_wait = max(
            (
                runtime.failure_circuit_until_monotonic_by_key.get(key, 0.0)
                - now
                for key in candidate_keys
            ),
            default=0.0,
        )
    return max(0.0, rate_wait, circuit_wait)


async def wait_for_cooldown_if_needed(
    runtime: Runtime,
    rate_limit_keys: str | Sequence[str],
    *,
    adapter_model: Optional[str] = None,
    use_alias_candidate_probe: bool = False,
) -> None:
    async def _on_active(_keys: list[str], _wait: float) -> None:
        if use_alias_candidate_probe:
            await runtime.maybe_raise_alias_probe_cooldown(
                adapter_model,
                use_alias_candidate_probe=True,
            )

    await retry.wait_for_monotonic_cooldown_map(
        runtime.rate_limit,
        rate_limit_keys,
        log_label="OpenRouter adapter",
        sleep=runtime.sleep,
        on_active=_on_active,
    )


async def set_cooldown(
    runtime: Runtime,
    rate_limit_keys: str | Sequence[str],
    wait_seconds: float,
) -> None:
    await retry.set_monotonic_cooldown_map(
        runtime.rate_limit,
        rate_limit_keys,
        wait_seconds,
        max_size=None,
    )


async def run_retry_loop(
    runtime: Runtime,
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[RetryResultT]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
    attempt_label: str,
    rate_limit_key_for_log: Optional[str] = None,
) -> RetryResultT:
    """Run the OpenRouter retry, cooldown, and failure-circuit policy."""
    max_retries = get_max_retries(runtime)
    total_attempts = max_retries + 1
    hidden_retry_budget_seconds = get_hidden_retry_budget_seconds(runtime)
    accumulated_hidden_wait_seconds = 0.0
    wait_keys = get_wait_keys(runtime, adapter_model)
    log_model_key = (
        rate_limit_key_for_log
        if rate_limit_key_for_log is not None
        else adapter_model
    )
    await runtime.maybe_raise_alias_probe_cooldown(
        adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    await runtime.maybe_raise_failure_circuit_open_callback(adapter_model)

    async def _before_attempt(attempt: int) -> None:
        runtime.log_debug(
            "%s upstream attempt %s/%s for model=%s",
            attempt_label,
            attempt,
            total_attempts,
            log_model_key,
        )
        await runtime.wait_for_cooldown(
            wait_keys,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    async def _on_success(_result: RetryResultT, _attempt: int) -> None:
        runtime.clear_failure_circuit_callback(adapter_model)

    async def _on_failure(exc: Exception, attempt: int) -> bool:
        nonlocal accumulated_hidden_wait_seconds
        status_code = extract_exception_status_code(runtime, exc)
        provider_name = extract_provider_name(runtime, exc)
        raw_message = extract_raw_message(runtime, exc)
        reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
        is_long_window = is_long_window_rate_limit(
            runtime,
            exc,
            hidden_retry_budget_seconds=hidden_retry_budget_seconds,
        )
        wait_seconds = get_retry_wait_seconds(runtime, exc, attempt)
        projected_hidden_wait_seconds, within_hidden_budget = (
            retry.projected_hidden_retry_within_budget(
                accumulated_hidden_wait_seconds=accumulated_hidden_wait_seconds,
                next_wait_seconds=wait_seconds,
                hidden_retry_budget_seconds=hidden_retry_budget_seconds,
            )
        )
        if status_code == 429 and is_long_window:
            cooldown_seconds = min(max(reset_wait_seconds or 0.0, 30.0), 300.0)
            if log_warnings:
                runtime.log_warning(
                    "%s upstream attempt %s hit long-window 429 "
                    "(%s, provider=%s, raw=%s, reset_wait=%.1fs) "
                    "and will not be hidden-retried",
                    attempt_label,
                    attempt,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                    reset_wait_seconds or 0.0,
                )
            await runtime.set_cooldown_callback(
                get_cooldown_keys(runtime, model=adapter_model, exc=exc),
                cooldown_seconds,
            )
            await runtime.open_failure_circuit_callback(adapter_model, exc=exc)
            return False
        maybe_raise_alias_probe_no_endpoint_unavailable(
            runtime,
            exc,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
            status_code=status_code,
            raw_message=raw_message,
        )
        if status_code != 429 or (
            attempt >= total_attempts and not within_hidden_budget
        ):
            if log_warnings:
                runtime.log_warning(
                    "%s upstream attempt %s failed with %s "
                    "(%s, provider=%s, raw=%s) and will not be retried",
                    attempt_label,
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                )
            if status_code == 429:
                await runtime.open_failure_circuit_callback(adapter_model, exc=exc)
            return False
        if attempt >= total_attempts and within_hidden_budget and log_warnings:
            runtime.log_warning(
                "%s keeping 429 hidden from client for model=%s; "
                "hidden retry wait %.1fs/%.1fs",
                attempt_label,
                adapter_model,
                projected_hidden_wait_seconds,
                hidden_retry_budget_seconds,
            )
        if log_warnings:
            runtime.log_warning(
                "%s upstream attempt %s hit 429 "
                "(%s, provider=%s, raw=%s); backoff %.1fs",
                attempt_label,
                attempt,
                exc.__class__.__name__,
                provider_name,
                raw_message,
                wait_seconds,
            )
        accumulated_hidden_wait_seconds = projected_hidden_wait_seconds
        await runtime.set_cooldown_callback(
            get_cooldown_keys(runtime, model=adapter_model, exc=exc),
            wait_seconds,
        )
        return True

    return await retry.run_adapter_retry_policy(
        operation,
        policy=retry.AdapterRetryPolicy(
            before_attempt=_before_attempt,
            on_failure=_on_failure,
            on_success=_on_success,
        ),
    )


async def perform_completion_operation(
    runtime: Runtime,
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[RetryResultT]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
) -> RetryResultT:
    return await run_retry_loop(
        runtime,
        adapter_model=adapter_model,
        operation=operation,
        log_warnings=log_warnings,
        use_alias_candidate_probe=use_alias_candidate_probe,
        attempt_label="OpenRouter completion adapter",
    )


async def perform_pass_through_request(
    runtime: Runtime,
    *,
    adapter_model: Optional[str],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
    request: object,
    target: str = "",
    custom_headers: Optional[Mapping[str, object]] = None,
    user_api_key_dict: object = None,
    custom_body: Optional[Mapping[str, object]] = None,
    forward_headers: bool = False,
    merge_query_params: bool = False,
    query_params: Optional[Mapping[str, object]] = None,
    default_query_params: Optional[Mapping[str, object]] = None,
    stream: Optional[bool] = None,
    cost_per_request: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    guardrails_config: Optional[Mapping[str, object]] = None,
    egress_credential_family: Optional[str] = None,
    expected_target_family: Optional[str] = None,
    allowed_forward_headers: Optional[list[str]] = None,
    allowed_pass_through_prefixed_headers: Optional[list[str]] = None,
    blocked_pass_through_prefixed_headers: Optional[list[str]] = None,
    retryable_upstream_status_codes: Optional[Sequence[int]] = None,
    caller_managed_hidden_retry: bool = False,
    raw_body_passthrough: bool = False,
    passthrough_logging_metadata: Optional[Mapping[str, object]] = None,
) -> Response:
    _ = caller_managed_hidden_retry
    effective_retryable_status_codes = list(
        retryable_upstream_status_codes or [429, 500, 502, 503, 504]
    )

    async def _operation() -> Response:
        return await runtime.pass_through_request(
            request=request,
            target=target,
            custom_headers=dict(custom_headers or {}),
            user_api_key_dict=user_api_key_dict,
            custom_body=dict(custom_body) if custom_body is not None else None,
            forward_headers=forward_headers,
            merge_query_params=merge_query_params,
            query_params=dict(query_params) if query_params is not None else None,
            default_query_params=(
                dict(default_query_params)
                if default_query_params is not None
                else None
            ),
            stream=stream,
            cost_per_request=cost_per_request,
            custom_llm_provider=custom_llm_provider,
            guardrails_config=(
                dict(guardrails_config) if guardrails_config is not None else None
            ),
            egress_credential_family=egress_credential_family,
            expected_target_family=expected_target_family,
            allowed_forward_headers=allowed_forward_headers,
            allowed_pass_through_prefixed_headers=(
                allowed_pass_through_prefixed_headers
            ),
            blocked_pass_through_prefixed_headers=(
                blocked_pass_through_prefixed_headers
            ),
            retryable_upstream_status_codes=effective_retryable_status_codes,
            caller_managed_hidden_retry=True,
            raw_body_passthrough=raw_body_passthrough,
            passthrough_logging_metadata=(
                dict(passthrough_logging_metadata)
                if passthrough_logging_metadata is not None
                else None
            ),
        )

    return await run_retry_loop(
        runtime,
        adapter_model=adapter_model,
        operation=_operation,
        log_warnings=log_warnings,
        use_alias_candidate_probe=use_alias_candidate_probe,
        attempt_label="OpenRouter adapter",
        rate_limit_key_for_log=get_rate_limit_key(runtime, adapter_model),
    )


__all__ = [
    "Runtime",
    "clear_failure_circuit",
    "extract_error_headers",
    "extract_error_payload",
    "extract_exception_status_code",
    "extract_provider_name",
    "extract_raw_message",
    "extract_reset_wait_seconds",
    "extract_retry_after_seconds",
    "get_active_cooldown_seconds",
    "get_backoff_seconds",
    "get_cooldown_keys",
    "get_header_value",
    "get_hidden_retry_budget_seconds",
    "get_max_retries",
    "get_post_failure_cooldown_seconds",
    "get_rate_limit_key",
    "get_retry_wait_seconds",
    "get_wait_keys",
    "is_free_model",
    "is_long_window_rate_limit",
    "is_no_endpoint_candidate_error",
    "is_provider_raw_error",
    "maybe_raise_alias_probe_no_endpoint_unavailable",
    "maybe_raise_failure_circuit_open",
    "open_failure_circuit",
    "perform_completion_operation",
    "perform_pass_through_request",
    "run_retry_loop",
    "set_cooldown",
    "wait_for_cooldown_if_needed",
]
