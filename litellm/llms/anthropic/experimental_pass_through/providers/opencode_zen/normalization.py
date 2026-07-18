"""OpenCode Zen-owned request and stream normalization."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Optional, Protocol, TypeGuard, Union
from uuid import uuid4

from openai.types.responses.response_create_params import ResponseInputParam
from starlette.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


class StreamingResponseLike(Protocol):
    """Streaming response surface consumed by OpenCode normalization."""

    @property
    def body_iterator(self) -> object: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def status_code(self) -> int: ...


@dataclass(frozen=True)
class Runtime:
    """Route-layer callbacks used by OpenCode Zen normalization."""

    clean_secret_string: Callable[[object], Optional[str]]
    merge_metadata: Callable[..., Payload]
    add_logging_metadata: Callable[..., Payload]
    build_span: Callable[..., Payload]
    transform_responses_api_request_to_chat_completion_request: Callable[
        ..., Payload
    ]
    async_responses_api_session_handler: Callable[..., Awaitable[Payload]]
    iterate_responses_sse_events: Callable[[object], AsyncIterator[object]]
    coerce_namespace_to_mapping: Callable[[object], object]
    responses_output_item_has_meaningful_content: Callable[[object], bool]
    streaming_response_factory: Callable[..., StreamingResponse]


@dataclass(frozen=True)
class CodexRequestNormalization:
    """Normalized request state consumed by the Codex OpenCode route."""

    request_body: Payload
    request_input: Union[str, ResponseInputParam]
    responses_api_request: Payload
    litellm_metadata: Payload
    completion_kwargs: Payload
    requested_model: object
    client_requested_stream: bool


def _is_response_input(value: object) -> TypeGuard[ResponseInputParam]:
    return isinstance(value, list)


def get_responses_tool_name(
    runtime: Runtime,
    tool: object,
) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    name = runtime.clean_secret_string(tool.get("name"))
    if name:
        return name
    function = tool.get("function")
    if isinstance(function, dict):
        return runtime.clean_secret_string(function.get("name"))
    return None


def _ordered_unique_str_values(values: list[Optional[str]]) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def strip_unsupported_responses_tools(
    runtime: Runtime,
    request_body: Payload,
) -> Payload:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body

    supported_tools: list[object] = []
    removed_tool_types: list[Optional[str]] = []
    removed_tool_names: list[Optional[str]] = []
    for tool in tools:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type == "function":
            supported_tools.append(tool)
            continue
        removed_tool_types.append(
            str(tool_type) if tool_type is not None else "unknown"
        )
        removed_tool_names.append(get_responses_tool_name(runtime, tool))

    removed_count = len(tools) - len(supported_tools)
    if removed_count <= 0:
        return request_body

    updated_body = dict(request_body)
    if supported_tools:
        updated_body["tools"] = supported_tools
    else:
        updated_body.pop("tools", None)

    return runtime.merge_metadata(
        updated_body,
        tags_to_add=["opencode-zen-unsupported-tools-stripped"],
        extra_fields={
            "opencode_zen_removed_unsupported_tool_count": removed_count,
            "opencode_zen_removed_unsupported_tool_types": (
                _ordered_unique_str_values(removed_tool_types)
            ),
            "opencode_zen_removed_unsupported_tool_names": (
                _ordered_unique_str_values(removed_tool_names)
            ),
        },
    )


def chat_message_role(message: object) -> Optional[str]:
    role = (
        message.get("role")
        if isinstance(message, dict)
        else getattr(message, "role", None)
    )
    return role if isinstance(role, str) else None


def chat_tool_call_id(tool_call: object) -> Optional[str]:
    tool_call_id = (
        tool_call.get("id")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "id", None)
    )
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def chat_message_tool_call_ids(message: object) -> list[str]:
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(tool_calls, list):
        return []

    tool_call_ids: list[str] = []
    for tool_call in tool_calls:
        tool_call_id = chat_tool_call_id(tool_call)
        if tool_call_id is not None:
            tool_call_ids.append(tool_call_id)
    return tool_call_ids


def chat_message_tool_result_id(message: object) -> Optional[str]:
    tool_call_id = (
        message.get("tool_call_id")
        if isinstance(message, dict)
        else getattr(message, "tool_call_id", None)
    )
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def collect_following_tool_block(
    messages: list[object],
    start_index: int,
) -> tuple[list[object], list[Optional[str]], int]:
    tool_block: list[object] = []
    tool_block_ids: list[Optional[str]] = []
    next_index = start_index
    while next_index < len(messages):
        next_message = messages[next_index]
        if chat_message_role(next_message) != "tool":
            break
        tool_block.append(next_message)
        tool_block_ids.append(chat_message_tool_result_id(next_message))
        next_index += 1
    return tool_block, tool_block_ids, next_index


def sanitize_completion_messages_for_chat_completion(
    completion_kwargs: Payload,
) -> tuple[Payload, Payload]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages: list[object] = []
    removed_assistant_count = 0
    removed_orphan_tool_count = 0
    removed_partial_tool_count = 0
    removed_extra_tool_count = 0

    index = 0
    while index < len(messages):
        message = messages[index]
        role = chat_message_role(message)

        if role == "tool":
            removed_orphan_tool_count += 1
            index += 1
            continue

        if role != "assistant":
            updated_messages.append(message)
            index += 1
            continue

        required_tool_call_ids = chat_message_tool_call_ids(message)
        if not required_tool_call_ids:
            updated_messages.append(message)
            index += 1
            continue

        required_tool_call_id_set = set(required_tool_call_ids)
        tool_block, tool_block_ids, next_index = collect_following_tool_block(
            messages,
            index + 1,
        )
        present_tool_call_ids = {
            tool_call_id for tool_call_id in tool_block_ids if tool_call_id is not None
        }
        if not required_tool_call_id_set.issubset(present_tool_call_ids):
            removed_assistant_count += 1
            removed_partial_tool_count += len(tool_block)
            index = next_index
            continue

        updated_messages.append(message)
        retained_tool_call_ids: set[str] = set()
        for tool_message, tool_call_id in zip(tool_block, tool_block_ids):
            if (
                tool_call_id is None
                or tool_call_id not in required_tool_call_id_set
                or tool_call_id in retained_tool_call_ids
            ):
                removed_extra_tool_count += 1
                continue
            updated_messages.append(tool_message)
            retained_tool_call_ids.add(tool_call_id)
        index = next_index

    if (
        removed_assistant_count == 0
        and removed_orphan_tool_count == 0
        and removed_partial_tool_count == 0
        and removed_extra_tool_count == 0
    ):
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: Payload = {
        "opencode_zen_chat_tool_adjacency_sanitized": True,
        "opencode_zen_chat_tool_adjacency_removed_assistant_count": (
            removed_assistant_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_orphan_tool_count": (
            removed_orphan_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_partial_tool_count": (
            removed_partial_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_extra_tool_count": (
            removed_extra_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_messages_from_count": len(messages),
        "opencode_zen_chat_tool_adjacency_messages_to_count": len(updated_messages),
    }
    return updated_kwargs, changes


async def normalize_codex_request(
    runtime: Runtime,
    prepared_request_body: Payload,
    *,
    adapter_model: str,
) -> CodexRequestNormalization:
    """Normalize OpenAI Responses input for OpenCode chat completions."""
    requested_model = prepared_request_body.get("model")
    request_body: Payload = copy.deepcopy(prepared_request_body)
    request_body["model"] = adapter_model
    removed_format = request_body.pop("format", None)
    request_body = strip_unsupported_responses_tools(runtime, request_body)
    request_body = runtime.add_logging_metadata(
        request_body,
        route_family="codex_opencode_zen_adapter",
        tag_prefix="codex-opencode-zen-adapter",
        requested_model=requested_model,
        adapter_model=adapter_model,
        input_shape="openai_responses",
        output_shape="openai_responses",
    )
    target_endpoint = "opencode_zen:/v1/chat/completions"
    tags_to_add = [
        f"codex-adapter-model:{adapter_model}",
        f"codex-adapter-target:{target_endpoint}",
    ]
    extra_fields: Payload = {
        "codex_adapter_model": adapter_model,
        "codex_adapter_original_model": requested_model,
        "codex_adapter_provider": "opencode_zen",
        "codex_adapter_target_endpoint": target_endpoint,
    }
    if removed_format is not None:
        tags_to_add.append("opencode-zen-unsupported-format-stripped")
        extra_fields["opencode_zen_removed_unsupported_format"] = removed_format
    request_body = runtime.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )

    raw_request_input = request_body.get("input")
    request_input: Union[str, ResponseInputParam]
    if isinstance(raw_request_input, str) or _is_response_input(raw_request_input):
        request_input = raw_request_input
    else:
        request_input = ""
    responses_api_request: Payload = {
        key: value
        for key, value in request_body.items()
        if key not in {"input", "model", "litellm_metadata"}
    }
    raw_metadata = request_body.get("litellm_metadata")
    litellm_metadata: Payload = (
        dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    )
    completion_kwargs = (
        runtime.transform_responses_api_request_to_chat_completion_request(
            model=adapter_model,
            input=request_input,
            responses_api_request=responses_api_request,
            custom_llm_provider="openai",
            stream=bool(request_body.get("stream")),
            metadata=litellm_metadata,
        )
    )
    completion_kwargs["metadata"] = litellm_metadata
    previous_response_id = responses_api_request.get("previous_response_id")
    if previous_response_id:
        completion_kwargs = await runtime.async_responses_api_session_handler(
            previous_response_id=str(previous_response_id),
            litellm_completion_request=completion_kwargs,
        )

    completion_kwargs, sanitization_changes = (
        sanitize_completion_messages_for_chat_completion(completion_kwargs)
    )
    if sanitization_changes:
        metadata_body = runtime.merge_metadata(
            {"litellm_metadata": litellm_metadata},
            tags_to_add=["opencode-zen-chat-tool-adjacency-sanitized"],
            extra_fields={
                **sanitization_changes,
                "langfuse_spans": [
                    runtime.build_span(
                        name="opencode_zen.chat_tool_adjacency_sanitized",
                        metadata=sanitization_changes,
                    )
                ],
            },
        )
        updated_metadata = metadata_body.get("litellm_metadata")
        litellm_metadata = (
            dict(updated_metadata) if isinstance(updated_metadata, dict) else {}
        )
        request_body["litellm_metadata"] = litellm_metadata
        completion_kwargs["metadata"] = litellm_metadata

    return CodexRequestNormalization(
        request_body=request_body,
        request_input=request_input,
        responses_api_request=responses_api_request,
        litellm_metadata=litellm_metadata,
        completion_kwargs=completion_kwargs,
        requested_model=requested_model,
        client_requested_stream=bool(request_body.get("stream")),
    )


def responses_sse_event(event_type: str, payload: Payload) -> str:
    return (
        f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


def response_payload_for_stream(
    *,
    response_id: str,
    model: str,
    status: str,
    output: Optional[list[Payload]] = None,
    usage: Optional[Payload] = None,
) -> Payload:
    payload: Payload = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
    }
    if output is not None:
        payload["output"] = output
    if usage is not None:
        payload["usage"] = usage
    return payload


def message_item_for_stream(
    *,
    message_id: str,
    status: str,
    output_text: str = "",
) -> Payload:
    content: list[Payload] = []
    if status == "completed":
        content.append(
            {
                "type": "output_text",
                "text": output_text,
                "annotations": [],
            }
        )
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def completed_response_for_stream(
    runtime: Runtime,
    *,
    response_event: Payload,
    response_id: str,
    model: str,
    message_id: Optional[str],
    output_text: str,
) -> Payload:
    response_payload = response_event.get("response")
    response_dict: Payload = (
        dict(response_payload) if isinstance(response_payload, dict) else {}
    )
    response_dict.setdefault("id", response_id)
    response_dict.setdefault("object", "response")
    response_dict.setdefault("created_at", int(time.time()))
    response_dict.setdefault("status", "completed")
    response_dict.setdefault("model", model)
    output = response_dict.get("output")
    if (
        message_id is not None
        and output_text
        and not (
            isinstance(output, list)
            and any(
                runtime.responses_output_item_has_meaningful_content(item)
                for item in output
            )
        )
    ):
        response_dict["output"] = [
            message_item_for_stream(
                message_id=message_id,
                status="completed",
                output_text=output_text,
            )
        ]
    return response_dict


async def normalize_responses_stream_for_codex(  # noqa: PLR0915
    runtime: Runtime,
    response: StreamingResponseLike,
    *,
    adapter_model: str,
) -> AsyncIterator[str]:
    response_id: Optional[str] = None
    message_id: Optional[str] = None
    response_created_sent = False
    message_started = False
    output_text_parts: list[str] = []

    async for event in runtime.iterate_responses_sse_events(response.body_iterator):
        mapped_event = runtime.coerce_namespace_to_mapping(event)
        if not isinstance(mapped_event, dict):
            continue
        event_dict: Payload = dict(mapped_event)
        event_type = event_dict.get("type")
        if not isinstance(event_type, str) or not event_type:
            continue

        if event_type == "response.output_text.delta":
            raw_response_payload = event_dict.get("response")
            response_payload = (
                raw_response_payload if isinstance(raw_response_payload, dict) else {}
            )
            response_id = (
                runtime.clean_secret_string(event_dict.get("response_id"))
                or runtime.clean_secret_string(event_dict.get("id"))
                or runtime.clean_secret_string(response_payload.get("id"))
                or response_id
                or f"resp_{uuid4().hex}"
            )
            response_model = (
                runtime.clean_secret_string(response_payload.get("model"))
                or runtime.clean_secret_string(event_dict.get("model"))
                or adapter_model
            )
            if not response_created_sent:
                yield responses_sse_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": response_payload_for_stream(
                            response_id=response_id,
                            model=response_model,
                            status="in_progress",
                            output=[],
                        ),
                    },
                )
                response_created_sent = True

            message_id = (
                runtime.clean_secret_string(event_dict.get("item_id"))
                or message_id
                or f"msg_{uuid4().hex[:24]}"
            )
            if not message_started:
                yield responses_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": message_item_for_stream(
                            message_id=message_id,
                            status="in_progress",
                        ),
                    },
                )
                yield responses_sse_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                )
                message_started = True

            delta = event_dict.get("delta")
            if isinstance(delta, str):
                output_text_parts.append(delta)
            event_dict["item_id"] = message_id
            event_dict.setdefault("output_index", 0)
            event_dict.setdefault("content_index", 0)
            yield responses_sse_event(event_type, event_dict)
            continue

        if event_type == "response.completed":
            raw_response_payload = event_dict.get("response")
            response_payload = (
                raw_response_payload if isinstance(raw_response_payload, dict) else {}
            )
            response_id = (
                runtime.clean_secret_string(response_payload.get("id"))
                or runtime.clean_secret_string(event_dict.get("id"))
                or response_id
                or f"resp_{uuid4().hex}"
            )
            response_model = (
                runtime.clean_secret_string(response_payload.get("model"))
                or runtime.clean_secret_string(event_dict.get("model"))
                or adapter_model
            )
            if not response_created_sent:
                yield responses_sse_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": response_payload_for_stream(
                            response_id=response_id,
                            model=response_model,
                            status="in_progress",
                            output=[],
                        ),
                    },
                )
                response_created_sent = True

            output_text = "".join(output_text_parts)
            if message_started and message_id is not None:
                yield responses_sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": output_text,
                    },
                )
                yield responses_sse_event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": output_text,
                            "annotations": [],
                        },
                    },
                )
                yield responses_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": message_item_for_stream(
                            message_id=message_id,
                            status="completed",
                            output_text=output_text,
                        ),
                    },
                )

            event_dict["response"] = completed_response_for_stream(
                runtime,
                response_event=event_dict,
                response_id=response_id,
                model=response_model,
                message_id=message_id,
                output_text=output_text,
            )
            yield responses_sse_event(event_type, event_dict)
            continue

        yield responses_sse_event(event_type, event_dict)

    yield "data: [DONE]\n\n"


def build_codex_streaming_response(
    runtime: Runtime,
    response: StreamingResponseLike,
    *,
    adapter_model: str,
) -> StreamingResponse:
    return runtime.streaming_response_factory(
        normalize_responses_stream_for_codex(
            runtime,
            response,
            adapter_model=adapter_model,
        ),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type="text/event-stream",
    )
