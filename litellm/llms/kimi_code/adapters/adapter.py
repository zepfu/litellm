"""OpenAI Responses and Anthropic Messages ingress for managed Kimi Code.

The managed Kimi provider owns authentication and the exact chat-completions
egress URL. These adapters only translate ingress shapes and construct the
shared AAWM completion route plan; they never copy, refresh, or replace the
CLI credential.
"""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from inspect import isawaitable
import re
from typing import Any, AsyncIterator, Iterable, Optional, cast

from fastapi import HTTPException

from litellm.llms.kimi_code import (
    KimiCodeManagedEndpoint,
    classify_kimi_code_failure,
    is_k3_model_id,
)
from litellm.llms.kimi_code.chat.transformation import (
    KIMI_CODE_API_BASE,
    KIMI_CODE_CHAT_COMPLETIONS_URL,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
    policy,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL = "managed-kimi-code-credential"
_K3_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
_K3_VARIANT_EFFORTS = {
    "kimi_code/k3-low": "low",
    "kimi_code/k3-high": "high",
    "kimi_code/k3-max": "max",
}
_CODEX_AGENT_MESSAGE_EMPTY_PAYLOAD_PATTERN = re.compile(
    r"\AMessage Type: (?:NEW_TASK|MESSAGE)\n"
    r"Task name: [^\n]+\n"
    r"Sender: [^\n]+\n"
    r"Payload:\n?\Z"
)


def normalize_kimi_code_chat_completions_adapter_model_name(
    model: Any,
    *,
    allowed_models: Iterable[str],
) -> Optional[str]:
    """Normalize only canonical `kimi_code/<adapter-key>` direct routes."""

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if not separator or provider_prefix != "kimi_code" or not model_id.strip():
        return None
    return candidate if candidate in allowed_models else None


def _resolve_adapter_selection(adapter_key: str) -> tuple[str, Optional[str]]:
    if adapter_key not in policy.KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS:
        raise ValueError(f"Unsupported managed Kimi Code adapter key {adapter_key!r}.")
    forced_effort = _K3_VARIANT_EFFORTS.get(adapter_key)
    upstream_model = adapter_key.removeprefix("kimi_code/")
    if forced_effort is not None:
        upstream_model = "k3"
    return upstream_model, forced_effort


def _add_adapter_metadata(
    *,
    request_body: dict[str, Any],
    config: adapter_config.AnthropicCompletionAdapterConfig,
    adapter_key: str,
    upstream_model: str,
    ingress: str,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    metadata = dict(updated_body.get("litellm_metadata") or {})
    existing_tags = metadata.get("tags")
    tags = list(existing_tags) if isinstance(existing_tags, list) else []
    for tag in (
        f"route:{config.route_family}",
        config.tag_prefix,
        f"{config.tag_prefix}-model:{adapter_key}",
        f"{config.tag_prefix}-target:{config.target_endpoint_label}",
    ):
        if tag not in tags:
            tags.append(tag)

    existing_spans = metadata.get("langfuse_spans")
    spans = list(existing_spans) if isinstance(existing_spans, list) else []
    spans.append(
        {
            "name": config.span_name,
            "metadata": {
                "requested_model": request_body.get("model"),
                "adapter_key": adapter_key,
                "upstream_model": upstream_model,
                "stream": bool(request_body.get("stream")),
            },
        }
    )

    metadata.update(
        {
            "tags": tags,
            "langfuse_spans": spans,
            "passthrough_route_family": config.route_family,
            "route_family": config.route_family,
            "kimi_code_adapter_key": adapter_key,
            "kimi_code_adapter_candidate": adapter_key,
            "kimi_code_upstream_model": upstream_model,
            f"{ingress}_adapter_model": adapter_key,
            f"{ingress}_adapter_original_model": request_body.get("model"),
            f"{ingress}_adapter_target_endpoint": config.target_endpoint_label,
        }
    )
    updated_body["litellm_metadata"] = metadata
    return updated_body


def _extract_explicit_responses_reasoning_effort(
    request_body: dict[str, Any],
) -> Optional[str]:
    reasoning_effort = request_body.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort:
        return reasoning_effort
    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort
    return None


def _apply_kimi_reasoning_effort(
    *,
    request_body: dict[str, Any],
    upstream_model: str,
    forced_effort: Optional[str],
    completion_kwargs: dict[str, Any],
) -> None:
    """Map explicit K3 Responses effort without inventing effort for K2.7."""

    if not is_k3_model_id(upstream_model):
        completion_kwargs.pop("reasoning_effort", None)
        return
    if forced_effort is not None:
        completion_kwargs["reasoning_effort"] = forced_effort
        return
    explicit_effort = _extract_explicit_responses_reasoning_effort(request_body)
    if explicit_effort is None:
        return
    mapped_effort = _K3_REASONING_EFFORT_MAP.get(explicit_effort)
    if mapped_effort is not None:
        completion_kwargs["reasoning_effort"] = mapped_effort


def _get_kimi_message_field(message: Any, field: str) -> Any:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _is_kimi_empty_text_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if not isinstance(content, list):
        return False
    if not content:
        return True
    for part in content:
        if isinstance(part, str):
            if part.strip():
                return False
            continue
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type not in (None, "text", "output_text"):
            return False
        text = part.get("text")
        if not isinstance(text, str) or text.strip():
            return False
    return True


def _kimi_message_has_tool_call(message: Any) -> bool:
    tool_calls = _get_kimi_message_field(message, "tool_calls")
    if isinstance(tool_calls, list) and bool(tool_calls):
        return True
    return bool(_get_kimi_message_field(message, "function_call"))


def _copy_kimi_message_without_content(message: Any) -> Any:
    if isinstance(message, dict):
        updated_message = dict(message)
    else:
        model_dump = getattr(message, "model_dump", None)
        if not callable(model_dump):
            return message
        updated_message = model_dump()
    updated_message.pop("content", None)
    return updated_message


def _sanitize_kimi_chat_messages(
    messages: list[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Remove replay-only empty text without dropping valid tool history."""

    updated_messages: list[Any] = []
    removed_empty_message_count = 0
    stripped_tool_call_content_count = 0
    for message in messages:
        role = _get_kimi_message_field(message, "role")
        content = _get_kimi_message_field(message, "content")
        if role == "tool":
            updated_messages.append(message)
            continue
        if _kimi_message_has_tool_call(message):
            if _is_kimi_empty_text_content(content):
                message = _copy_kimi_message_without_content(message)
                stripped_tool_call_content_count += 1
            updated_messages.append(message)
            continue
        if _is_kimi_empty_text_content(content):
            removed_empty_message_count += 1
            continue
        updated_messages.append(message)

    if (
        removed_empty_message_count == 0
        and stripped_tool_call_content_count == 0
    ):
        return messages, {}
    return updated_messages, {
        "kimi_code_chat_message_shape_sanitized": True,
        "kimi_code_chat_message_shape_messages_from_count": len(messages),
        "kimi_code_chat_message_shape_messages_to_count": len(updated_messages),
        "kimi_code_chat_message_shape_removed_empty_message_count": (
            removed_empty_message_count
        ),
        "kimi_code_chat_message_shape_stripped_tool_call_content_count": (
            stripped_tool_call_content_count
        ),
    }


def normalize_kimi_code_custom_tool_outputs(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Preserve Codex custom-tool results for Kimi's function-tool wire shape."""

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body

    updated_input_items: list[Any] = []
    changed = False
    for item in input_items:
        if (
            isinstance(item, dict)
            and item.get("type") == "custom_tool_call_output"
            and isinstance(item.get("call_id"), str)
            and item["call_id"].strip()
        ):
            updated_item = dict(item)
            updated_item["type"] = "function_call_output"
            updated_input_items.append(updated_item)
            changed = True
            continue
        updated_input_items.append(item)

    if not changed:
        return request_body

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    return updated_body


def _restore_kimi_codex_agent_message_payloads(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose only Codex collaboration payloads that Kimi can consume."""

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, {}

    updated_input_items: list[Any] = []
    restored_count = 0
    restored_chars = 0
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            updated_input_items.append(item)
            continue

        content = item.get("content")
        if not isinstance(content, list) or len(content) != 2:
            updated_input_items.append(item)
            continue

        visible_part = content[0]
        payload_part = content[1]
        if not isinstance(visible_part, dict) or not isinstance(payload_part, dict):
            updated_input_items.append(item)
            continue
        if visible_part.get("type") not in {"input_text", "text"}:
            updated_input_items.append(item)
            continue
        visible_text = visible_part.get("text")
        if not isinstance(visible_text, str) or not (
            _CODEX_AGENT_MESSAGE_EMPTY_PAYLOAD_PATTERN.fullmatch(visible_text)
        ):
            updated_input_items.append(item)
            continue
        if payload_part.get("type") != "encrypted_content":
            updated_input_items.append(item)
            continue
        payload = payload_part.get("encrypted_content")
        if not isinstance(payload, str) or not payload:
            updated_input_items.append(item)
            continue

        separator = "" if visible_text.endswith("\n") else "\n"
        updated_item = dict(item)
        updated_item["content"] = [
            {
                "type": visible_part.get("type"),
                "text": f"{visible_text}{separator}{payload}",
            }
        ]
        updated_input_items.append(updated_item)
        restored_count += 1
        restored_chars += len(payload)

    if restored_count == 0:
        return request_body, {}

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    return updated_body, {
        "kimi_code_codex_agent_task_payload_restored": True,
        "kimi_code_codex_agent_task_payload_restored_count": restored_count,
        "kimi_code_codex_agent_task_payload_restored_chars": restored_chars,
    }


def _normalize_kimi_tool_result_adjacency(messages: list[Any]) -> list[Any]:
    """Place replayed tool results immediately after their assistant calls."""

    assistant_call_ids_by_index: dict[int, list[str]] = {}
    assistant_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if _get_kimi_message_field(message, "role") != "assistant":
            continue
        tool_calls = _get_kimi_message_field(message, "tool_calls")
        if not tool_calls:
            continue
        if not isinstance(tool_calls, list):
            raise ValueError("Kimi Code continuation history contains invalid assistant tool_calls.")
        call_ids: list[str] = []
        for tool_call in tool_calls:
            call_id = _get_kimi_message_field(tool_call, "id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(
                    "Kimi Code continuation history contains an assistant tool call " "without a valid id."
                )
            if call_id in assistant_call_ids:
                raise ValueError(
                    "Kimi Code continuation history contains duplicate assistant " f"tool_call_id {call_id!r}."
                )
            assistant_call_ids.add(call_id)
            call_ids.append(call_id)
        assistant_call_ids_by_index[index] = call_ids

    tool_results_by_id: dict[str, Any] = {}
    for message in messages:
        if _get_kimi_message_field(message, "role") != "tool":
            continue
        tool_call_id = _get_kimi_message_field(message, "tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("Kimi Code continuation history contains a tool result without a " "valid tool_call_id.")
        if tool_call_id not in assistant_call_ids:
            raise ValueError(
                "Kimi Code continuation history contains a tool result for unknown " f"tool_call_id {tool_call_id!r}."
            )
        if tool_call_id in tool_results_by_id:
            raise ValueError(
                "Kimi Code continuation history contains duplicate tool results for " f"tool_call_id {tool_call_id!r}."
            )
        tool_results_by_id[tool_call_id] = message

    missing_result_ids = assistant_call_ids.difference(tool_results_by_id)
    if missing_result_ids:
        formatted_ids = ", ".join(repr(call_id) for call_id in sorted(missing_result_ids))
        raise ValueError(
            "Kimi Code continuation history is missing tool results for " f"tool_call_id(s): {formatted_ids}."
        )

    normalized_messages: list[Any] = []
    for index, message in enumerate(messages):
        if _get_kimi_message_field(message, "role") == "tool":
            continue
        normalized_messages.append(message)
        for call_id in assistant_call_ids_by_index.get(index, []):
            normalized_messages.append(tool_results_by_id[call_id])
    return normalized_messages


def _clone_kimi_stream_chunk(chunk: Any) -> Any:
    model_copy = getattr(chunk, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    copy_method = getattr(chunk, "copy", None)
    if callable(copy_method):
        return copy_method(deep=True)
    return deepcopy(chunk)


def _split_kimi_anthropic_stream_chunk(chunk: Any) -> tuple[Any, ...]:
    """Separate Kimi reasoning/text/tool deltas for the shared Anthropic wrapper."""

    choices = getattr(chunk, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        return (chunk,)
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return (chunk,)

    reasoning_content = getattr(delta, "reasoning_content", None)
    text_content = getattr(delta, "content", None)
    tool_calls = getattr(delta, "tool_calls", None)
    has_reasoning = isinstance(reasoning_content, str) and bool(reasoning_content)
    has_text = isinstance(text_content, str) and bool(text_content)
    has_tools = isinstance(tool_calls, list) and bool(tool_calls)
    if sum((has_reasoning, has_text, has_tools)) <= 1:
        return (chunk,)

    split_chunks: list[Any] = []
    if has_reasoning:
        reasoning_chunk = _clone_kimi_stream_chunk(chunk)
        reasoning_delta = reasoning_chunk.choices[0].delta
        reasoning_delta.content = None
        reasoning_delta.tool_calls = None
        split_chunks.append(reasoning_chunk)
    if has_text:
        text_chunk = _clone_kimi_stream_chunk(chunk)
        text_delta = text_chunk.choices[0].delta
        text_delta.reasoning_content = None
        text_delta.thinking_blocks = None
        text_delta.tool_calls = None
        split_chunks.append(text_chunk)
    if has_tools:
        tool_chunk = _clone_kimi_stream_chunk(chunk)
        tool_delta = tool_chunk.choices[0].delta
        tool_delta.reasoning_content = None
        tool_delta.thinking_blocks = None
        tool_delta.content = None
        split_chunks.append(tool_chunk)

    for index, split_chunk in enumerate(split_chunks):
        if index:
            split_chunk.choices[0].delta.role = None
        if index < len(split_chunks) - 1:
            split_chunk.choices[0].finish_reason = None
            if hasattr(split_chunk, "usage"):
                split_chunk.usage = None
    return tuple(split_chunks)


def _kimi_anthropic_stream_block_type(chunk: Any) -> Optional[str]:
    choices = getattr(chunk, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None
    tool_calls = getattr(delta, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        return "tool_use"
    thinking_blocks = getattr(delta, "thinking_blocks", None)
    reasoning_content = getattr(delta, "reasoning_content", None)
    if (isinstance(thinking_blocks, list) and thinking_blocks) or (
        isinstance(reasoning_content, str) and reasoning_content
    ):
        return "thinking"
    content = getattr(delta, "content", None)
    if isinstance(content, str) and content:
        return "text"
    return None


def _build_kimi_anthropic_stream_transition_chunk(chunk: Any, *, block_type: str) -> Any:
    transition_chunk = _clone_kimi_stream_chunk(chunk)
    transition_choice = transition_chunk.choices[0]
    transition_delta = transition_choice.delta
    transition_delta.content = None
    transition_delta.reasoning_content = None
    transition_delta.tool_calls = None
    transition_delta.role = None
    transition_delta.thinking_blocks = (
        [{"type": "thinking", "thinking": "", "signature": ""}] if block_type == "thinking" else None
    )
    transition_choice.finish_reason = None
    if hasattr(transition_chunk, "usage"):
        transition_chunk.usage = None
    return transition_chunk


async def normalize_anthropic_kimi_completion_stream(
    completion_stream: Any,
) -> AsyncIterator[Any]:
    """Preserve all combined Kimi deltas before shared Anthropic SSE conversion."""

    current_block_type: Optional[str] = None
    try:
        async for chunk in completion_stream:
            for normalized_chunk in _split_kimi_anthropic_stream_chunk(chunk):
                block_type = _kimi_anthropic_stream_block_type(normalized_chunk)
                if (
                    block_type in {"text", "thinking"}
                    and current_block_type is not None
                    and block_type != current_block_type
                ):
                    yield _build_kimi_anthropic_stream_transition_chunk(
                        normalized_chunk,
                        block_type=block_type,
                    )
                yield normalized_chunk
                if block_type is not None:
                    current_block_type = block_type
    finally:
        close_fn = getattr(completion_stream, "aclose", None)
        if not callable(close_fn):
            close_fn = getattr(completion_stream, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_result = close_fn()
                if isawaitable(close_result):
                    await close_result


def _classify_kimi_adapter_failure(
    exc: Exception,
    *,
    adapter_model: str,
) -> dict[str, object]:
    """Return only the safe managed-Kimi classifier fields."""

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    error_code = getattr(exc, "error_code", None) or getattr(exc, "type", None)
    headers = getattr(exc, "headers", None)
    metadata = classify_kimi_code_failure(
        status_code=status_code,
        error_code=error_code,
        message=message,
        upstream_id=adapter_model,
        endpoint=KimiCodeManagedEndpoint.CHAT_COMPLETIONS,
        headers=headers if isinstance(headers, dict) else None,
    )
    return metadata.to_safe_metadata()


def _handle_kimi_adapter_exception(
    exc: Exception,
    *,
    adapter_model: str,
    use_alias_candidate_probe: bool,
) -> None:
    """Preserve alias metadata while bounding direct managed-auth failures."""

    metadata = _classify_kimi_adapter_failure(
        exc,
        adapter_model=adapter_model,
    )
    if (
        metadata.get("kind") == "malformed"
        and metadata.get("status_code") in {400, 422}
    ):
        raise HTTPException(
            status_code=int(cast(int, metadata["status_code"])),
            detail={
                "error": {
                    "message": "Managed Kimi Code rejected the request shape.",
                    "type": "invalid_request_error",
                    "code": "kimi_code_invalid_request",
                }
            },
        ) from exc
    if use_alias_candidate_probe:
        setattr(exc, "kimi_code_probe_failure_metadata", metadata)
        return
    if (
        metadata.get("kind") == "refresh_required_auth"
        and metadata.get("scope") == "managed_account"
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": (
                        "Managed Kimi Code authentication requires the shared "
                        "credential to be refreshed."
                    ),
                    "type": "authentication_error",
                    "code": "kimi_code_auth_refresh_required",
                }
            },
        ) from exc


async def prepare_codex_kimi_chat_completions_adapter_route(
    *,
    request: object,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Translate OpenAI Responses ingress to managed Kimi chat completions."""

    _ = request
    upstream_model, forced_effort = _resolve_adapter_selection(adapter_model)
    config = adapter_config.CODEX_KIMI_CHAT_COMPLETIONS
    prepared_request_body, task_payload_changes = (
        _restore_kimi_codex_agent_message_payloads(prepared_request_body)
    )
    request_body = _add_adapter_metadata(
        request_body=prepared_request_body,
        config=config,
        adapter_key=adapter_model,
        upstream_model=upstream_model,
        ingress="codex",
    )
    if task_payload_changes:
        task_payload_metadata = dict(request_body.get("litellm_metadata") or {})
        tags = list(task_payload_metadata.get("tags") or [])
        tag = "codex-kimi-agent-task-payload-restored"
        if tag not in tags:
            tags.append(tag)
        task_payload_metadata["tags"] = tags
        task_payload_metadata.update(task_payload_changes)
        request_body["litellm_metadata"] = task_payload_metadata
    request_input = request_body.get("input", "")
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {key: value for key, value in request_body.items() if key not in {"input", "model", "litellm_metadata"}},
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider="kimi_code",
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs["metadata"] = litellm_metadata
    completion_kwargs["custom_llm_provider"] = "kimi_code"
    completion_kwargs["num_retries"] = 0
    previous_response_id = responses_api_request.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=previous_response_id,
            litellm_completion_request=completion_kwargs,
        )
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Kimi Code request history must contain a messages list.")
    normalized_messages = _normalize_kimi_tool_result_adjacency(messages)
    normalized_messages, sanitization_changes = _sanitize_kimi_chat_messages(
        normalized_messages
    )
    completion_kwargs["messages"] = normalized_messages
    if sanitization_changes:
        litellm_metadata.update(sanitization_changes)
        request_body["litellm_metadata"] = litellm_metadata
        completion_kwargs["metadata"] = litellm_metadata
    _apply_kimi_reasoning_effort(
        request_body=request_body,
        upstream_model=upstream_model,
        forced_effort=forced_effort,
        completion_kwargs=completion_kwargs,
    )

    def handle_exception(exc: Exception) -> None:
        _handle_kimi_adapter_exception(
            exc,
            adapter_model=upstream_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=KIMI_CODE_CHAT_COMPLETIONS_URL,
        api_key=KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL,
        api_base=KIMI_CODE_API_BASE,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "completion_kwargs": completion_kwargs,
            "request_input": request_input,
            "responses_api_request": responses_api_request,
            "litellm_metadata": litellm_metadata,
            "upstream_model": upstream_model,
        },
        handle_exception=handle_exception,
    )


async def prepare_anthropic_kimi_chat_completions_adapter_route(
    *,
    request: object,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> adapter_driver.CompletionAdapterRoutePlan:
    """Prepare Anthropic Messages ingress for the shared completion adapter."""

    _ = request
    upstream_model, forced_effort = _resolve_adapter_selection(adapter_model)
    config = adapter_config.ANTHROPIC_KIMI_CHAT_COMPLETIONS
    request_body = _add_adapter_metadata(
        request_body=prepared_request_body,
        config=config,
        adapter_key=adapter_model,
        upstream_model=upstream_model,
        ingress="anthropic",
    )
    # K2.7 is always-thinking but has no explicit K3 effort control. Removing
    # the source thinking request prevents the shared transformer from
    # synthesizing an unsupported `reasoning_effort`.
    if not is_k3_model_id(upstream_model) or forced_effort is not None:
        request_body.pop("thinking", None)

    def handle_exception(exc: Exception) -> None:
        _handle_kimi_adapter_exception(
            exc,
            adapter_model=upstream_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    extra_handler_kwargs: dict[str, Any] = {}
    extra_handler_kwargs["num_retries"] = 0
    if forced_effort is not None:
        extra_handler_kwargs["reasoning_effort"] = forced_effort
    parallel_tool_calls = request_body.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        extra_handler_kwargs["parallel_tool_calls"] = parallel_tool_calls

    return adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=KIMI_CODE_CHAT_COMPLETIONS_URL,
        api_key=KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL,
        api_base=KIMI_CODE_API_BASE,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "custom_llm_provider": "kimi_code",
            "model_for_upstream": upstream_model,
            "extra_handler_kwargs": extra_handler_kwargs,
            "completion_stream_normalizer": normalize_anthropic_kimi_completion_stream,
        },
        handle_exception=handle_exception,
    )
