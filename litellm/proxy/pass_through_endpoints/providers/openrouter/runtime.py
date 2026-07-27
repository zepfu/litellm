"""OpenRouter route runtime extracted from ``llm_passthrough_endpoints``.

Provider retry mechanics remain owned by the existing
``providers.openrouter.retry_transport`` module. Route-layer state and host
helpers are supplied explicitly so this module does not import the god module.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from fastapi import Response

from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    retry_transport as _anthropic_openrouter_retry_transport,
)

_RetryResultT = TypeVar("_RetryResultT")

_ANTHROPIC_ADAPTER_OPENROUTER_API_KEY_ENV_VARS = (
    "AAWM_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
)


@dataclass(frozen=True)
class Runtime:
    """Host-owned dependencies required by the OpenRouter route runtime."""

    retry_transport_runtime: _anthropic_openrouter_retry_transport.Runtime
    clean_secret_string: Callable[[Optional[str]], Optional[str]]
    get_first_secret_value: Callable[[tuple[str, ...]], Optional[str]]
    getenv: Callable[[str], Optional[str]]
    get_secret_str: Callable[[str], Optional[str]]
    sanitize_opencode_zen_completion_messages: Callable[
        [dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ]
    chat_message_role: Callable[[Any], Optional[str]]
    chat_message_tool_call_ids: Callable[[Any], list[str]]
    chat_message_tool_result_id: Callable[[Any], Optional[str]]
    is_empty_text_content: Callable[[Any], bool]
    merge_litellm_metadata: Callable[..., dict[str, Any]]
    build_langfuse_span_descriptor: Callable[..., dict[str, Any]]


_runtime: Optional[Runtime] = None


def configure_openrouter_runtime(runtime: Runtime) -> None:
    """Install route-layer callbacks and shared retry state."""

    global _runtime
    _runtime = runtime


def _require_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("OpenRouter runtime has not been configured")
    return _runtime


def _retry_runtime() -> _anthropic_openrouter_retry_transport.Runtime:
    return _require_runtime().retry_transport_runtime


def _get_openrouter_adapter_rate_limit_key(model: Optional[str]) -> str:
    return _anthropic_openrouter_retry_transport.get_rate_limit_key(
        _retry_runtime(),
        model,
    )


def _is_openrouter_adapter_free_model(model: Optional[str]) -> bool:
    return _anthropic_openrouter_retry_transport.is_free_model(
        _retry_runtime(),
        model,
    )


def _get_openrouter_adapter_wait_keys(model: Optional[str]) -> str:
    return _anthropic_openrouter_retry_transport.get_wait_keys(
        _retry_runtime(),
        model,
    )


def _extract_openrouter_adapter_exception_status_code(
    exc: Any,
) -> Optional[int]:
    return _anthropic_openrouter_retry_transport.extract_exception_status_code(
        _retry_runtime(),
        exc,
    )


def _extract_openrouter_adapter_error_payload(
    exc: Any,
) -> Optional[dict[str, Any]]:
    return _anthropic_openrouter_retry_transport.extract_error_payload(
        _retry_runtime(),
        exc,
    )


def _extract_openrouter_adapter_provider_name(exc: Any) -> Optional[str]:
    return _anthropic_openrouter_retry_transport.extract_provider_name(
        _retry_runtime(),
        exc,
    )


def _extract_openrouter_adapter_retry_after_seconds(
    exc: Any,
) -> Optional[float]:
    return _anthropic_openrouter_retry_transport.extract_retry_after_seconds(
        _retry_runtime(),
        exc,
    )


def _extract_openrouter_adapter_raw_message(exc: Any) -> Optional[str]:
    return _anthropic_openrouter_retry_transport.extract_raw_message(
        _retry_runtime(),
        exc,
    )


def _is_openrouter_adapter_no_endpoint_candidate_error(
    exc: Any,
    *,
    status_code: Optional[int] = None,
    raw_message: Optional[str] = None,
) -> bool:
    return _anthropic_openrouter_retry_transport.is_no_endpoint_candidate_error(
        _retry_runtime(),
        exc,
        status_code=status_code,
        raw_message=raw_message,
    )


def _maybe_raise_openrouter_adapter_alias_probe_no_endpoint_unavailable(
    exc: Any,
    *,
    adapter_model: Optional[str],
    use_alias_candidate_probe: bool,
    status_code: Optional[int] = None,
    raw_message: Optional[str] = None,
) -> None:
    return _anthropic_openrouter_retry_transport.maybe_raise_alias_probe_no_endpoint_unavailable(
        _retry_runtime(),
        exc,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
        status_code=status_code,
        raw_message=raw_message,
    )


def _is_openrouter_adapter_provider_raw_error(exc: Any) -> bool:
    return _anthropic_openrouter_retry_transport.is_provider_raw_error(
        _retry_runtime(),
        exc,
    )


def _extract_openrouter_adapter_error_headers(exc: Any) -> dict[str, Any]:
    return dict(
        _anthropic_openrouter_retry_transport.extract_error_headers(
            _retry_runtime(),
            exc,
        )
    )


def _get_openrouter_adapter_header_value(
    headers: dict[str, Any],
    header_name: str,
) -> Optional[str]:
    return _anthropic_openrouter_retry_transport.get_header_value(
        _retry_runtime(),
        headers,
        header_name,
    )


def _extract_openrouter_adapter_reset_wait_seconds(
    exc: Any,
) -> Optional[float]:
    return _anthropic_openrouter_retry_transport.extract_reset_wait_seconds(
        _retry_runtime(),
        exc,
    )


def _is_openrouter_adapter_long_window_rate_limit(
    exc: Any,
    *,
    hidden_retry_budget_seconds: float,
) -> bool:
    return _anthropic_openrouter_retry_transport.is_long_window_rate_limit(
        _retry_runtime(),
        exc,
        hidden_retry_budget_seconds=hidden_retry_budget_seconds,
    )


def _get_openrouter_adapter_cooldown_keys(
    *,
    model: Optional[str],
    exc: Any,
) -> str:
    return _anthropic_openrouter_retry_transport.get_cooldown_keys(
        _retry_runtime(),
        model=model,
        exc=exc,
    )


def _get_openrouter_adapter_retry_wait_seconds(
    exc: Any,
    attempt: int,
) -> float:
    return _anthropic_openrouter_retry_transport.get_retry_wait_seconds(
        _retry_runtime(),
        exc,
        attempt,
    )


def _get_openrouter_adapter_max_retries() -> int:
    return _anthropic_openrouter_retry_transport.get_max_retries(
        _retry_runtime()
    )


def _get_openrouter_adapter_backoff_seconds(attempt: int) -> float:
    return _anthropic_openrouter_retry_transport.get_backoff_seconds(
        _retry_runtime(),
        attempt,
    )


def _get_openrouter_adapter_hidden_retry_budget_seconds() -> float:
    return _anthropic_openrouter_retry_transport.get_hidden_retry_budget_seconds(
        _retry_runtime()
    )


def _get_openrouter_adapter_post_failure_cooldown_seconds() -> float:
    return _anthropic_openrouter_retry_transport.get_post_failure_cooldown_seconds(
        _retry_runtime()
    )


async def _maybe_raise_openrouter_adapter_failure_circuit_open(
    adapter_model: Optional[str],
) -> None:
    return await _anthropic_openrouter_retry_transport.maybe_raise_failure_circuit_open(
        _retry_runtime(),
        adapter_model,
    )


async def _openrouter_adapter_open_failure_circuit(
    adapter_model: Optional[str],
    *,
    exc: Any,
) -> None:
    return await _anthropic_openrouter_retry_transport.open_failure_circuit(
        _retry_runtime(),
        adapter_model,
        exc=exc,
    )


def _clear_openrouter_adapter_failure_circuit(
    adapter_model: Optional[str],
) -> None:
    return _anthropic_openrouter_retry_transport.clear_failure_circuit(
        _retry_runtime(),
        adapter_model,
    )


async def _get_openrouter_adapter_active_cooldown_seconds(
    adapter_model: Optional[str],
) -> float:
    return await _anthropic_openrouter_retry_transport.get_active_cooldown_seconds(
        _retry_runtime(),
        adapter_model,
    )


async def _wait_for_openrouter_adapter_cooldown_if_needed(
    rate_limit_keys: Union[str, list[str], tuple[str, ...]],
    *,
    adapter_model: Optional[str] = None,
    use_alias_candidate_probe: bool = False,
) -> None:
    return await _anthropic_openrouter_retry_transport.wait_for_cooldown_if_needed(
        _retry_runtime(),
        rate_limit_keys,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _set_openrouter_adapter_cooldown(
    rate_limit_keys: Union[str, list[str], tuple[str, ...]],
    wait_seconds: float,
) -> None:
    return await _anthropic_openrouter_retry_transport.set_cooldown(
        _retry_runtime(),
        rate_limit_keys,
        wait_seconds,
    )


async def _run_openrouter_adapter_retry_loop(
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[_RetryResultT]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
    attempt_label: str,
    rate_limit_key_for_log: Optional[str] = None,
) -> _RetryResultT:
    return await _anthropic_openrouter_retry_transport.run_retry_loop(
        _retry_runtime(),
        adapter_model=adapter_model,
        operation=operation,
        log_warnings=log_warnings,
        use_alias_candidate_probe=use_alias_candidate_probe,
        attempt_label=attempt_label,
        rate_limit_key_for_log=rate_limit_key_for_log,
    )


async def _perform_openrouter_completion_adapter_operation(
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[Any]],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
) -> Any:
    return await _anthropic_openrouter_retry_transport.perform_completion_operation(
        _retry_runtime(),
        adapter_model=adapter_model,
        operation=operation,
        log_warnings=log_warnings,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_openrouter_adapter_pass_through_request(
    *,
    adapter_model: Optional[str],
    log_warnings: bool = True,
    use_alias_candidate_probe: bool = False,
    **kwargs: Any,
) -> Response:
    return await _anthropic_openrouter_retry_transport.perform_pass_through_request(
        _retry_runtime(),
        adapter_model=adapter_model,
        log_warnings=log_warnings,
        use_alias_candidate_probe=use_alias_candidate_probe,
        **kwargs,
    )


def _get_openrouter_api_key() -> Optional[str]:
    return _require_runtime().get_first_secret_value(
        _ANTHROPIC_ADAPTER_OPENROUTER_API_KEY_ENV_VARS
    )


def _get_anthropic_adapter_openrouter_api_key() -> Optional[str]:
    return _get_openrouter_api_key()


def _get_openrouter_target_base() -> str:
    runtime = _require_runtime()
    cleaned = (
        runtime.clean_secret_string(runtime.getenv("OPENROUTER_API_BASE"))
        or "https://openrouter.ai/api"
    ).rstrip("/")
    if cleaned.endswith("/api/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_anthropic_adapter_openrouter_target_base() -> str:
    return _get_openrouter_target_base()


def _build_openrouter_default_headers() -> dict[str, str]:
    runtime = _require_runtime()
    return {
        "HTTP-Referer": (
            runtime.clean_secret_string(runtime.get_secret_str("OR_SITE_URL"))
            or "https://litellm.ai"
        ),
        "X-Title": (
            runtime.clean_secret_string(runtime.get_secret_str("OR_APP_NAME"))
            or "liteLLM"
        ),
    }


def _openrouter_chat_message_function_call(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("function_call")
    return getattr(message, "function_call", None)


def _openrouter_chat_message_has_valid_content_or_tool_calls(
    message: Any,
) -> bool:
    runtime = _require_runtime()
    role = runtime.chat_message_role(message)
    if role == "tool":
        return runtime.chat_message_tool_result_id(message) is not None

    if runtime.chat_message_tool_call_ids(message):
        return True
    if _openrouter_chat_message_function_call(message):
        return True

    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return not runtime.is_empty_text_content(content)


def _copy_openrouter_message_value(
    value: Any,
    *,
    field_name: str,
    field_value: Any,
) -> Any:
    if isinstance(value, dict):
        updated = dict(value)
        updated[field_name] = field_value
        return updated

    updated = copy.deepcopy(value)
    setattr(updated, field_name, field_value)
    return updated


def _serialize_openrouter_tool_call_arguments(
    arguments: Any,
) -> tuple[str, str]:
    if isinstance(arguments, dict):
        argument_kind = "object"
    elif isinstance(arguments, (list, tuple)):
        argument_kind = "array"
    else:
        argument_kind = "scalar"

    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError):
        serialized = json.dumps(
            str(arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return serialized, argument_kind


def _normalize_openrouter_chat_message_tool_call_arguments(
    message: Any,
) -> tuple[Any, dict[str, int]]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return message, {}

    updated_tool_calls: list[Any] = []
    normalized_counts = {
        "object": 0,
        "array": 0,
        "scalar": 0,
    }
    changed = False
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function = tool_call.get("function")
        else:
            function = getattr(tool_call, "function", None)
        if function is None:
            updated_tool_calls.append(tool_call)
            continue

        if isinstance(function, dict):
            if "arguments" not in function:
                updated_tool_calls.append(tool_call)
                continue
            arguments = function.get("arguments")
        else:
            if not hasattr(function, "arguments"):
                updated_tool_calls.append(tool_call)
                continue
            arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str):
            updated_tool_calls.append(tool_call)
            continue

        normalized_arguments, argument_kind = (
            _serialize_openrouter_tool_call_arguments(arguments)
        )
        updated_function = _copy_openrouter_message_value(
            function,
            field_name="arguments",
            field_value=normalized_arguments,
        )
        updated_tool_calls.append(
            _copy_openrouter_message_value(
                tool_call,
                field_name="function",
                field_value=updated_function,
            )
        )
        normalized_counts[argument_kind] += 1
        changed = True

    if not changed:
        return message, {}
    return (
        _copy_openrouter_message_value(
            message,
            field_name="tool_calls",
            field_value=updated_tool_calls,
        ),
        normalized_counts,
    )


def _sanitize_openrouter_completion_messages_for_chat_completion(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _require_runtime()
    completion_kwargs, adjacency_changes = (
        runtime.sanitize_opencode_zen_completion_messages(completion_kwargs)
    )

    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, adjacency_changes

    updated_messages: list[Any] = []
    removed_empty_message_count = 0
    normalized_tool_argument_message_count = 0
    normalized_tool_argument_counts = {
        "object": 0,
        "array": 0,
        "scalar": 0,
    }
    for message in messages:
        if not _openrouter_chat_message_has_valid_content_or_tool_calls(message):
            removed_empty_message_count += 1
            continue
        normalized_message, normalized_counts = (
            _normalize_openrouter_chat_message_tool_call_arguments(message)
        )
        updated_messages.append(normalized_message)
        if normalized_counts:
            normalized_tool_argument_message_count += 1
            for key, count in normalized_counts.items():
                normalized_tool_argument_counts[key] += count

    normalized_tool_argument_count = sum(
        normalized_tool_argument_counts.values()
    )
    if (
        removed_empty_message_count == 0
        and normalized_tool_argument_count == 0
        and not adjacency_changes
    ):
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: dict[str, Any] = {
        "openrouter_chat_message_shape_sanitized": True,
        "openrouter_chat_message_shape_messages_from_count": len(messages),
        "openrouter_chat_message_shape_messages_to_count": len(
            updated_messages
        ),
        "openrouter_chat_message_shape_removed_empty_message_count": (
            removed_empty_message_count
        ),
    }
    if normalized_tool_argument_count:
        changes.update(
            {
                "openrouter_chat_tool_arguments_sanitized": True,
                "openrouter_chat_tool_arguments_normalized_count": (
                    normalized_tool_argument_count
                ),
                "openrouter_chat_tool_arguments_message_count": (
                    normalized_tool_argument_message_count
                ),
                "openrouter_chat_tool_arguments_object_count": (
                    normalized_tool_argument_counts["object"]
                ),
                "openrouter_chat_tool_arguments_array_count": (
                    normalized_tool_argument_counts["array"]
                ),
                "openrouter_chat_tool_arguments_scalar_count": (
                    normalized_tool_argument_counts["scalar"]
                ),
            }
        )
    if adjacency_changes:
        changes.update(adjacency_changes)
        changes["openrouter_chat_tool_adjacency_sanitized"] = True
    return updated_kwargs, changes


def _apply_openrouter_completion_message_sanitization(
    *,
    request_body: dict[str, Any],
    completion_kwargs: dict[str, Any],
    litellm_metadata: dict[str, Any],
    span_name: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = _require_runtime()
    completion_kwargs, sanitization_changes = (
        _sanitize_openrouter_completion_messages_for_chat_completion(
            completion_kwargs
        )
    )
    if not sanitization_changes:
        return request_body, completion_kwargs, litellm_metadata

    metadata_body = runtime.merge_litellm_metadata(
        {"litellm_metadata": litellm_metadata},
        tags_to_add=[tag],
        extra_fields={
            **sanitization_changes,
            "langfuse_spans": [
                runtime.build_langfuse_span_descriptor(
                    name=span_name,
                    metadata=sanitization_changes,
                )
            ],
        },
    )
    litellm_metadata = dict(metadata_body.get("litellm_metadata") or {})
    request_body = dict(request_body)
    request_body["litellm_metadata"] = litellm_metadata
    completion_kwargs = dict(completion_kwargs)
    completion_kwargs["metadata"] = litellm_metadata
    return request_body, completion_kwargs, litellm_metadata
