"""OpenRouter-owned retry, cooldown, error shaping, and transport execution."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import (
    Any,
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
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
    bound_memory_map,
    normalize_monotonic_cooldown_key,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    is_openrouter_free_model,
)
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
    is_retired_ox_alpha_candidate_error,
)

RetryResultT = TypeVar("RetryResultT")
InnerSendSink = Callable[..., None]

_INNER_SEND_SINK: ContextVar[Optional[InnerSendSink]] = ContextVar(
    "aawm_openrouter_inner_send_sink",
    default=None,
)


def bind_inner_send_sink(
    callback: Optional[InnerSendSink],
) -> Token[Optional[InnerSendSink]]:
    """Bind a per-task sink that receives every inner OpenRouter send."""
    return _INNER_SEND_SINK.set(callback)


def reset_inner_send_sink(token: Token[Optional[InnerSendSink]]) -> None:
    _INNER_SEND_SINK.reset(token)


@dataclass(frozen=True)
class Runtime:
    """Route-layer state and callbacks required by OpenRouter retry/transport."""

    rate_limit: MonotonicCooldownMap
    failure_circuit_until_monotonic_by_key: dict[str, float]
    clean_secret_string: Callable[[Optional[str]], Optional[str]]
    extract_embedded_json_payload_candidates: Callable[[object], Iterable[str]]
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
    open_failure_circuit_callback: Callable[..., Awaitable[Any]]
    clear_failure_circuit_callback: Callable[[Optional[str]], None]
    log_debug: Callable[..., object]
    log_warning: Callable[..., object]
    getenv: Callable[[str], Optional[str]]
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic


def _emit_inner_send(
    runtime: Runtime,
    **payload: Any,
) -> None:
    callback = _INNER_SEND_SINK.get()
    if callback is None:
        return
    try:
        callback(**payload)
    except Exception:
        runtime.log_debug(
            "OpenRouter inner-send sink failed for attempt=%s disposition=%s",
            payload.get("inner_attempt"),
            payload.get("disposition"),
            exc_info=True,
        )


@dataclass
class _StartedSendTracker:
    """Track one started inner send until its final row is emitted.

    Proposed cooldown values are ignored. Only ``acknowledge`` can attach an
    applied ``cooldown_seconds`` value that settlement may copy.
    """

    runtime: Runtime
    started_inner_attempt: int = 0
    started_row_finalized: bool = True
    known_final_payload: dict[str, Any] = field(default_factory=dict)
    published_cooldown_seconds: Optional[float] = None

    def note(self, **payload: Any) -> None:
        if self.started_row_finalized or self.started_inner_attempt <= 0:
            return
        self.known_final_payload.update(
            {
                key: value
                for key, value in payload.items()
                if value is not None and key != "cooldown_seconds"
            }
        )
        staged = {
            key: value
            for key, value in self.known_final_payload.items()
            if key not in {"status", "disposition", "inner_attempt"}
            and value is not None
        }
        _emit_inner_send(
            self.runtime,
            inner_attempt=self.started_inner_attempt,
            status="in_flight",
            disposition="started",
            attempted_provider_call=True,
            **staged,
        )

    def acknowledge(self, seconds: float) -> None:
        published = max(0.0, float(seconds))
        if self.published_cooldown_seconds is None:
            self.published_cooldown_seconds = published
        else:
            self.published_cooldown_seconds = max(
                self.published_cooldown_seconds,
                published,
            )
        if self.started_row_finalized or self.started_inner_attempt <= 0:
            return
        self.known_final_payload["cooldown_seconds"] = self.published_cooldown_seconds
        self.note()

    def emit_final(self, **payload: Any) -> None:
        self.known_final_payload.update(payload)
        _emit_inner_send(self.runtime, **payload)
        self.started_row_finalized = True

    def applied_cooldown_seconds(self) -> float:
        if self.published_cooldown_seconds is None:
            return 0.0
        return self.published_cooldown_seconds

    def settle_outstanding(self) -> None:
        if self.started_row_finalized or self.started_inner_attempt <= 0:
            return
        payload: dict[str, Any] = {
            "inner_attempt": self.started_inner_attempt,
            "status": "cancelled",
            "disposition": "cancelled",
            "delay_seconds": 0.0,
            "cooldown_seconds": self.applied_cooldown_seconds(),
            "attempted_provider_call": True,
        }
        payload.update(
            {
                key: value
                for key, value in self.known_final_payload.items()
                if key != "cooldown_seconds"
            }
        )
        payload["inner_attempt"] = self.started_inner_attempt
        payload["cooldown_seconds"] = self.applied_cooldown_seconds()
        self.emit_final(**payload)

    def mark_started(self, attempt: int) -> None:
        self.started_inner_attempt = attempt
        self.started_row_finalized = False
        self.known_final_payload.clear()
        self.published_cooldown_seconds = None
        _emit_inner_send(
            self.runtime,
            inner_attempt=attempt,
            status="in_flight",
            disposition="started",
            attempted_provider_call=True,
        )


def get_rate_limit_key(runtime: Runtime, model: Optional[str]) -> str:
    cleaned_model = runtime.clean_secret_string(model)
    if not cleaned_model:
        return "__default__"
    return normalize_monotonic_cooldown_key(cleaned_model)


def is_free_model(runtime: Runtime, model: Optional[str]) -> bool:
    return is_openrouter_free_model(runtime.clean_secret_string(model))


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


def maybe_raise_alias_probe_retired_ox_alpha_unavailable(
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
    upstream_model = runtime.get_completion_model(adapter_model)
    if not any(
        is_retired_ox_alpha_candidate_error(
            runtime,
            exc,
            model=model,
            status_code=status_code,
            raw_message=raw_message,
        )
        for model in (adapter_model, upstream_model)
    ):
        return
    model_label = get_rate_limit_key(runtime, adapter_model)
    detail_text = raw_message or str(exc)
    runtime.raise_candidate_unavailable(
        f"OpenRouter auto-agent candidate {model_label} is retired after its "
        f"testing period: {detail_text}"
    )


def get_cooldown_keys(
    runtime: Runtime,
    *,
    model: Optional[str],
    exc: object,
) -> str:
    _ = exc
    return get_rate_limit_key(runtime, model)


_PROVIDER_RETRY_WAIT_CAP_SECONDS = 60.0


def _provider_retry_wait_seconds(runtime: Runtime, exc: object) -> Optional[float]:
    retry_after_seconds = extract_retry_after_seconds(runtime, exc)
    if retry_after_seconds is not None:
        return min(
            max(retry_after_seconds + 1.0, 1.0),
            _PROVIDER_RETRY_WAIT_CAP_SECONDS,
        )
    headers = extract_error_headers(runtime, exc)
    remaining_value = get_header_value(runtime, headers, "X-RateLimit-Remaining")
    reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
    if remaining_value in {"0", "0.0"} and reset_wait_seconds is not None:
        return min(
            max(reset_wait_seconds + 1.0, 1.0),
            _PROVIDER_RETRY_WAIT_CAP_SECONDS,
        )
    return None


def get_retry_wait_seconds(runtime: Runtime, exc: object, attempt: int) -> float:
    wait_seconds = get_backoff_seconds(runtime, attempt)
    provider_wait_seconds = _provider_retry_wait_seconds(runtime, exc)
    if provider_wait_seconds is None:
        return wait_seconds
    return max(wait_seconds, provider_wait_seconds)


def get_max_retries(runtime: Runtime) -> int:
    """Configured inner retries after the first upstream send."""
    return retry.parse_non_negative_int_env(
        "AAWM_OPENROUTER_ADAPTER_MAX_RETRIES",
        default=3,
        getenv=runtime.getenv,
    )


def get_total_attempts(runtime: Runtime) -> int:
    """Hard cap on inner wire sends: one initial attempt plus configured retries."""
    return get_max_retries(runtime) + 1


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


def get_failure_circuit_cooldown_seconds(runtime: Runtime, exc: object) -> float:
    """Return the proposed circuit duration for *exc*.

    Policy owner: max(post-failure floor, Retry-After, reset-wait), then
    clamp to ``[0, 300]``. This is the proposal ``open_failure_circuit``
    applies; the committed remaining duration, including any longer expiry
    already stored, is the owner's return value. Recording sites must not
    treat this proposal as an applied cooldown.
    """
    cooldown_seconds = get_post_failure_cooldown_seconds(runtime)
    retry_after_seconds = extract_retry_after_seconds(runtime, exc)
    reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
    for candidate in (retry_after_seconds, reset_wait_seconds):
        if candidate is not None:
            cooldown_seconds = max(cooldown_seconds, candidate)
    return min(max(cooldown_seconds, 0.0), 300.0)


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
            "OpenRouter adapter failure circuit open for %s; " "failing fast for %ss",
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
) -> float:
    """Publish the failure circuit and return the committed remaining duration.

    The proposal comes from ``get_failure_circuit_cooldown_seconds``. A
    longer existing expiry is retained. The return is the remaining time
    until that committed expiry, not the proposal.
    """
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    proposed_seconds = get_failure_circuit_cooldown_seconds(runtime, exc)
    async with runtime.rate_limit.lock:
        now = runtime.monotonic()
        proposed_until = now + proposed_seconds
        current_until = runtime.failure_circuit_until_monotonic_by_key.get(
            rate_limit_key,
            0.0,
        )
        if proposed_until > current_until:
            runtime.failure_circuit_until_monotonic_by_key[
                rate_limit_key
            ] = proposed_until
            bound_memory_map(runtime.failure_circuit_until_monotonic_by_key)
            committed_until = proposed_until
        else:
            committed_until = current_until
        return max(committed_until - now, 0.0)


def _peek_failure_circuit_remaining_seconds(
    runtime: Runtime,
    adapter_model: Optional[str],
) -> float:
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    remaining = (
        runtime.failure_circuit_until_monotonic_by_key.get(rate_limit_key, 0.0)
        - runtime.monotonic()
    )
    return max(remaining, 0.0)


async def _await_published_failure_circuit(
    runtime: Runtime,
    adapter_model: Optional[str],
    *,
    exc: object,
) -> float:
    """Return the committed circuit remaining after publication completes.

    Prefer the owner's return (the committed duration, including a retained
    longer expiry). If a callback swallows that return, read the map written
    by the completed publication. Does not catch ``CancelledError``.
    """
    published = await runtime.open_failure_circuit_callback(adapter_model, exc=exc)
    if isinstance(published, (int, float)) and not isinstance(published, bool):
        return max(float(published), 0.0)
    return _peek_failure_circuit_remaining_seconds(runtime, adapter_model)


def clear_failure_circuit(
    runtime: Runtime,
    adapter_model: Optional[str],
) -> None:
    rate_limit_key = get_rate_limit_key(runtime, adapter_model)
    runtime.failure_circuit_until_monotonic_by_key.pop(rate_limit_key, None)


_ANTHROPIC_SSE_EVENT_PREFIX = "event:"
_ANTHROPIC_STREAM_TERMINAL_EVENT = "message_stop"


def _is_anthropic_sse_message_stop(payload: str) -> bool:
    """Return True when *payload* contains an SSE ``event: message_stop`` line.

    Only the event field is inspected.  ``data:`` JSON, tool arguments, and
    content text that happen to include the token ``message_stop`` are not
    terminals.  Lines are split on CRLF, CR, and LF only, so Unicode
    separators such as U+2028 inside JSON text are not line boundaries.
    The field name ``event:`` is literal; at most one ASCII space after the
    colon is removed, then the remaining value must equal ``message_stop``.
    """
    # SSE line breaks are CRLF / CR / LF only.  ``str.splitlines()`` also
    # splits on Unicode separators (U+0085 / U+2028 / U+2029).
    for lf_piece in payload.split("\n"):
        for line in lf_piece.split("\r"):
            if not line.startswith(_ANTHROPIC_SSE_EVENT_PREFIX):
                continue
            value = line[len(_ANTHROPIC_SSE_EVENT_PREFIX) :]
            if value.startswith(" "):
                value = value[1:]
            if value == _ANTHROPIC_STREAM_TERMINAL_EVENT:
                return True
    return False


def _chunk_has_openai_finish_reason(chunk: object) -> bool:
    """Return True when *chunk* is an OpenAI finish / validated-commit chunk."""
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not isinstance(choices, (list, tuple)):
        return False
    for choice in choices:
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
        else:
            finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            return True
    return False


def _is_accepted_stream_terminal(chunk: object) -> bool:
    """Return True at a provider-authored stream success boundary.

    ``perform_completion_operation`` wraps two stream kinds:

    - OpenAI / ``CustomStreamWrapper``: ``ModelResponseStream`` (or dict)
      chunks whose ``choices[].finish_reason`` is set.
    - Anthropic: a ``message_stop`` dict envelope, or an SSE frame whose
      event line is ``event: message_stop``.

    Content or tool JSON that merely contains the token ``message_stop``
    is not terminal.  Construction never reaches this helper.
    """
    if isinstance(chunk, dict):
        if chunk.get("type") == _ANTHROPIC_STREAM_TERMINAL_EVENT:
            return True
    if _chunk_has_openai_finish_reason(chunk):
        return True
    if isinstance(chunk, (bytes, bytearray)):
        try:
            payload = bytes(chunk).decode("utf-8")
        except UnicodeDecodeError:
            return False
        return _is_anthropic_sse_message_stop(payload)
    if isinstance(chunk, str):
        return _is_anthropic_sse_message_stop(chunk)
    return False


class _ValidatedStreamCircuitBoundary:
    """Lazy pass-through stream wrapper deferring the failure-circuit clear.

    Construction performs no provider I/O and NEVER clears the failure
    circuit (OR-034): stream creation alone is not provider success.  The
    circuit is cleared exactly once, when a provider-authored accepted
    terminal is observed (OpenAI finish / ``CustomStreamWrapper.sent_last_chunk``
    commit, or Anthropic ``message_stop``).  Upstream failures and
    cancellation propagate untouched, preserving the exact circuit state.
    """

    def __init__(
        self,
        stream: Any,
        clear_once: Callable[[], None],
    ) -> None:
        self.__dict__["_stream"] = stream
        self.__dict__["_clear_once"] = clear_once
        self.__dict__["_circuit_cleared"] = False

    def __aiter__(self) -> "_ValidatedStreamCircuitBoundary":
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_stream"], name)

    async def __anext__(self) -> Any:
        stream = self.__dict__["_stream"]
        chunk = await stream.__anext__()
        if not self.__dict__["_circuit_cleared"] and (
            _is_accepted_stream_terminal(chunk)
            or getattr(stream, "sent_last_chunk", False) is True
        ):
            self.__dict__["_circuit_cleared"] = True
            self.__dict__["_clear_once"]()
        return chunk


def _finalize_circuit_success_boundary(
    runtime: Runtime,
    result: Any,
    *,
    adapter_model: Optional[str],
) -> Any:
    """Apply the OR-034 success boundary for a completion-adapter result.

    Lazy streams are wrapped: the circuit clears only at the validated
    terminal boundary inside the stream (OpenAI finish / stream-commit
    seam, or Anthropic ``message_stop``).  Materialized (nonstream)
    results are validated commits and clear the circuit immediately,
    exactly once.
    """
    if callable(getattr(result, "__anext__", None)):
        return _ValidatedStreamCircuitBoundary(
            result,
            lambda: runtime.clear_failure_circuit_callback(adapter_model),
        )
    runtime.clear_failure_circuit_callback(adapter_model)
    return result


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
                runtime.failure_circuit_until_monotonic_by_key.get(key, 0.0) - now
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
    )


async def _publish_long_window_terminal(
    runtime: Runtime,
    exc: Exception,
    *,
    attempt: int,
    adapter_model: Optional[str],
    status_code: Optional[int],
    reset_wait_seconds: Optional[float],
    note: Callable[..., None],
    emit_final: Callable[..., None],
    acknowledge: Callable[[float], None],
) -> None:
    rate_limit_cooldown_seconds = min(max(reset_wait_seconds or 0.0, 30.0), 300.0)
    note(
        status="terminal",
        disposition="long_window_rate_limit",
        delay_seconds=0.0,
        error_status_code=status_code,
        failure_class="long_window_rate_limit",
    )
    await runtime.set_cooldown_callback(
        get_cooldown_keys(runtime, model=adapter_model, exc=exc),
        rate_limit_cooldown_seconds,
    )
    applied_cooldown = rate_limit_cooldown_seconds
    acknowledge(applied_cooldown)
    circuit_committed = await _await_published_failure_circuit(
        runtime,
        adapter_model,
        exc=exc,
    )
    applied_cooldown = max(applied_cooldown, circuit_committed)
    acknowledge(applied_cooldown)
    emit_final(
        inner_attempt=attempt,
        status="terminal",
        disposition="long_window_rate_limit",
        delay_seconds=0.0,
        cooldown_seconds=applied_cooldown,
        error_status_code=status_code,
        failure_class="long_window_rate_limit",
        attempted_provider_call=True,
    )


async def _publish_terminal_failure(
    runtime: Runtime,
    exc: Exception,
    *,
    attempt: int,
    adapter_model: Optional[str],
    status_code: Optional[int],
    note: Callable[..., None],
    emit_final: Callable[..., None],
    acknowledge: Callable[[float], None],
) -> None:
    applied_cooldown: Optional[float] = None
    note(
        status="terminal",
        disposition="terminal",
        delay_seconds=0.0,
        error_status_code=status_code,
        failure_class=exc.__class__.__name__,
    )
    if status_code == 429:
        applied_cooldown = await _await_published_failure_circuit(
            runtime,
            adapter_model,
            exc=exc,
        )
        acknowledge(applied_cooldown)
    emit_final(
        inner_attempt=attempt,
        status="terminal",
        disposition="terminal",
        delay_seconds=0.0,
        cooldown_seconds=applied_cooldown,
        error_status_code=status_code,
        failure_class=exc.__class__.__name__,
        attempted_provider_call=True,
    )


async def _retry_loop_on_failure(
    runtime: Runtime,
    exc: Exception,
    *,
    attempt: int,
    adapter_model: Optional[str],
    attempt_label: str,
    log_warnings: bool,
    use_alias_candidate_probe: bool,
    hidden_retry_budget_seconds: float,
    accumulated_hidden_wait_seconds: float,
    total_attempts: int,
    note_started_row: Optional[Callable[..., None]] = None,
    emit_final_row: Optional[Callable[..., None]] = None,
    acknowledge_published_cooldown: Optional[Callable[[float], None]] = None,
) -> tuple[bool, float]:
    """Return (should_retry, updated_hidden_wait_seconds)."""

    def _note(**payload: Any) -> None:
        if note_started_row is not None:
            note_started_row(**payload)
            return
        staged = {
            key: value
            for key, value in payload.items()
            if key not in {"status", "disposition", "inner_attempt"}
            and value is not None
        }
        _emit_inner_send(
            runtime,
            inner_attempt=attempt,
            status="in_flight",
            disposition="started",
            attempted_provider_call=True,
            **staged,
        )

    def _emit_final(**payload: Any) -> None:
        if emit_final_row is not None:
            emit_final_row(**payload)
            return
        _emit_inner_send(runtime, **payload)

    def _acknowledge(seconds: float) -> None:
        if acknowledge_published_cooldown is not None:
            acknowledge_published_cooldown(seconds)

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
    provider_wait_seconds = _provider_retry_wait_seconds(runtime, exc)
    (
        projected_hidden_wait_seconds,
        within_hidden_budget,
    ) = retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=accumulated_hidden_wait_seconds,
        next_wait_seconds=wait_seconds,
        hidden_retry_budget_seconds=hidden_retry_budget_seconds,
    )
    retries_exhausted = attempt >= total_attempts
    delay_budget_exhausted = (
        hidden_retry_budget_seconds > 0
        and provider_wait_seconds is not None
        and not within_hidden_budget
    )
    _note(
        error_status_code=status_code,
        failure_class=exc.__class__.__name__,
    )
    if status_code == 429 and is_long_window:
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
        await _publish_long_window_terminal(
            runtime,
            exc,
            attempt=attempt,
            adapter_model=adapter_model,
            status_code=status_code,
            reset_wait_seconds=reset_wait_seconds,
            note=_note,
            emit_final=_emit_final,
            acknowledge=_acknowledge,
        )
        return False, accumulated_hidden_wait_seconds
    try:
        maybe_raise_alias_probe_retired_ox_alpha_unavailable(
            runtime,
            exc,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
            status_code=status_code,
            raw_message=raw_message,
        )
        maybe_raise_alias_probe_no_endpoint_unavailable(
            runtime,
            exc,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
            status_code=status_code,
            raw_message=raw_message,
        )
    except Exception as probe_exc:
        _emit_final(
            inner_attempt=attempt,
            status="terminal",
            disposition="candidate_unavailable",
            delay_seconds=0.0,
            cooldown_seconds=0.0,
            error_status_code=status_code,
            failure_class=probe_exc.__class__.__name__,
            attempted_provider_call=True,
        )
        raise
    if status_code != 429 or retries_exhausted or delay_budget_exhausted:
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
        await _publish_terminal_failure(
            runtime,
            exc,
            attempt=attempt,
            adapter_model=adapter_model,
            status_code=status_code,
            note=_note,
            emit_final=_emit_final,
            acknowledge=_acknowledge,
        )
        return False, accumulated_hidden_wait_seconds
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
    _note(
        status="retrying",
        disposition="retry_backoff",
        delay_seconds=wait_seconds,
        error_status_code=status_code,
        failure_class="rate_limited",
    )
    await runtime.set_cooldown_callback(
        get_cooldown_keys(runtime, model=adapter_model, exc=exc),
        wait_seconds,
    )
    _acknowledge(wait_seconds)
    _emit_final(
        inner_attempt=attempt,
        status="retrying",
        disposition="retry_backoff",
        delay_seconds=wait_seconds,
        cooldown_seconds=wait_seconds,
        error_status_code=status_code,
        failure_class="rate_limited",
        attempted_provider_call=True,
    )
    return True, projected_hidden_wait_seconds


async def run_retry_loop(
    runtime: Runtime,
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[RetryResultT]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
    attempt_label: str,
    rate_limit_key_for_log: Optional[str] = None,
    clear_on_success: bool = True,
) -> RetryResultT:
    """Run the OpenRouter retry, cooldown, and failure-circuit policy.

    ``AAWM_OPENROUTER_ADAPTER_MAX_RETRIES`` is the retry count after the first
    send. ``total_attempts`` is that value plus one and is a hard cap on wire
    calls. A configured delay budget may refuse a Retry-After/reset wait; it
    cannot add sends past the cap.
    """
    total_attempts = get_total_attempts(runtime)
    hidden_retry_budget_seconds = get_hidden_retry_budget_seconds(runtime)
    accumulated_hidden_wait_seconds = 0.0
    wait_keys = get_wait_keys(runtime, adapter_model)
    log_model_key = (
        rate_limit_key_for_log if rate_limit_key_for_log is not None else adapter_model
    )
    await runtime.maybe_raise_alias_probe_cooldown(
        adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    await runtime.maybe_raise_failure_circuit_open_callback(adapter_model)
    started = _StartedSendTracker(runtime)

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
        started.mark_started(attempt)

    async def _on_success(_result: RetryResultT, attempt: int) -> None:
        # OR-034: lazy stream construction is not provider success.  Callers
        # that return lazy streams pass clear_on_success=False and apply the
        # validated boundary themselves (_finalize_circuit_success_boundary);
        # nonstream/transport callers keep the eager validated clear here.
        if clear_on_success:
            runtime.clear_failure_circuit_callback(adapter_model)
        started.emit_final(
            inner_attempt=attempt,
            status="succeeded",
            disposition="succeeded",
            delay_seconds=0.0,
            cooldown_seconds=0.0,
            attempted_provider_call=True,
        )

    async def _on_failure(exc: Exception, attempt: int) -> bool:
        nonlocal accumulated_hidden_wait_seconds
        started.note(
            error_status_code=extract_exception_status_code(runtime, exc),
            failure_class=exc.__class__.__name__,
        )
        should_retry, accumulated_hidden_wait_seconds = await _retry_loop_on_failure(
            runtime,
            exc,
            attempt=attempt,
            adapter_model=adapter_model,
            attempt_label=attempt_label,
            log_warnings=log_warnings,
            use_alias_candidate_probe=use_alias_candidate_probe,
            hidden_retry_budget_seconds=hidden_retry_budget_seconds,
            accumulated_hidden_wait_seconds=accumulated_hidden_wait_seconds,
            total_attempts=total_attempts,
            note_started_row=started.note,
            emit_final_row=started.emit_final,
            acknowledge_published_cooldown=started.acknowledge,
        )
        return should_retry

    try:
        return await retry.run_adapter_retry_policy(
            operation,
            policy=retry.AdapterRetryPolicy(
                before_attempt=_before_attempt,
                on_failure=_on_failure,
                on_success=_on_success,
            ),
        )
    finally:
        started.settle_outstanding()


_INVALID_TOOL_DETAIL_LIMIT = 200
_QUOTE_CHARS = ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019")


def _is_completion_invalid_tool_error(
    runtime: Runtime,
    exc: object,
) -> bool:
    if extract_exception_status_code(runtime, exc) != 400:
        return False
    raw_message = extract_raw_message(runtime, exc)
    if not isinstance(raw_message, str) or not raw_message:
        return False
    lowered = raw_message.casefold()
    tools_text = lowered.replace("`", "")
    function_text = lowered
    for quote in _QUOTE_CHARS:
        function_text = function_text.replace(quote, "")
    return "invalid tools" in tools_text and "expected function" in function_text


def _bounded_completion_invalid_tool_detail(detail: str) -> str:
    compact = " ".join(detail.split())
    if len(compact) <= _INVALID_TOOL_DETAIL_LIMIT:
        return compact
    return compact[: _INVALID_TOOL_DETAIL_LIMIT - 3] + "..."


def _provider_return_int_status_code(exc: object) -> Optional[int]:
    """Return a proven HTTP status from int .status_code/.code only.

    Checks the exception, then its .response. String codes and
    `"429" in str(exc)` are not treated as a provider return.
    """
    sources = (exc, getattr(exc, "response", None))
    for source in sources:
        if source is None:
            continue
        for attr in ("status_code", "code"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
    return None


async def perform_completion_operation(
    runtime: Runtime,
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[RetryResultT]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
) -> RetryResultT:
    async def _provider_return_stamp_operation() -> RetryResultT:
        # OR-029: stamp provenance on exceptions escaping the retried
        # provider operation itself. Preflight raises (alias cooldown,
        # failure circuit) happen outside this closure and stay unstamped.
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            setattr(exc, "attempted_provider_call", True)
            setattr(exc, "failure_phase", "provider_attempt")
            setattr(exc, "provider_name", "openrouter")
            already_returned = (
                getattr(exc, "_aawm_provider_returned", False) is True
                or getattr(exc, "provider_returned", False) is True
            )
            status_code = _provider_return_int_status_code(exc)
            if already_returned or status_code is not None:
                setattr(exc, "_aawm_provider_returned", True)
                setattr(exc, "provider_returned", True)
            if status_code is not None:
                setattr(exc, "upstream_status_code", status_code)
            raise

    try:
        result = await run_retry_loop(
            runtime,
            adapter_model=adapter_model,
            operation=_provider_return_stamp_operation,
            log_warnings=log_warnings,
            use_alias_candidate_probe=use_alias_candidate_probe,
            attempt_label="OpenRouter completion adapter",
            clear_on_success=False,
        )
        # OR-034: circuit success moves to the validated boundary.  A lazy
        # stream wrapper is created without touching circuit state; the
        # circuit clears exactly once at the accepted terminal (OpenAI
        # finish / stream-commit, or Anthropic message_stop) or immediately
        # for a materialized (nonstream) result.  Failures and cancellation
        # after wrapper creation propagate and preserve the exact circuit
        # state.
        return _finalize_circuit_success_boundary(
            runtime,
            result,
            adapter_model=adapter_model,
        )
    except Exception as exc:
        if use_alias_candidate_probe and _is_completion_invalid_tool_error(
            runtime, exc
        ):
            raw_message = extract_raw_message(runtime, exc)
            detail_text = (
                raw_message
                if isinstance(raw_message, str) and raw_message
                else str(exc)
            )
            runtime.raise_candidate_unavailable(
                "OpenRouter completion invalid-tool: "
                + _bounded_completion_invalid_tool_detail(detail_text)
            )
        raise


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
                dict(default_query_params) if default_query_params is not None else None
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
    "bind_inner_send_sink",
    "clear_failure_circuit",
    "_finalize_circuit_success_boundary",
    "_ValidatedStreamCircuitBoundary",
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
    "get_failure_circuit_cooldown_seconds",
    "get_post_failure_cooldown_seconds",
    "get_rate_limit_key",
    "get_retry_wait_seconds",
    "get_total_attempts",
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
    "reset_inner_send_sink",
    "run_retry_loop",
    "set_cooldown",
    "wait_for_cooldown_if_needed",
]
