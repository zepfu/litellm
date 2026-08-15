"""
OpenAI Passthrough Logging Handler

Handles cost tracking and logging for OpenAI passthrough endpoints, specifically /chat/completions.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.openai import OpenAIConfig as OpenAIConfigType
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
    apply_passthrough_logging_contract,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.llms.openai import RESPONSES_API_TERMINAL_STREAM_EVENTS
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageResponse,
    LlmProviders,
    Message,
    PassthroughCallTypes,
    Usage,
)
from litellm.utils import ModelResponse, TextCompletionResponse


# Process-lifetime caches for on-disk model price fallback.
# litellm.model_cost stays a live first-source check; these only avoid re-reading
# and re-parsing the large JSON price maps (and re-scanning them for known misses).
_MODEL_PRICE_MAP_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_MODEL_PRICE_DISK_MISS_CACHE: set[str] = set()
_MODEL_PRICE_DISK_MISS_CACHE_MAXSIZE = 4096


class _ResponsesSSEStateTracker:
    """Track decoded Responses SSE events and emit synthetic terminal payloads.

    The tracker is intentionally private and bounded: it stores compact state,
    hashes tool arguments, and emits stable omission metadata whenever parsed
    streams exceed internal bounds.
    """

    _EVENT_TYPES_LIMIT = 64
    _IDENTITY_BYTES_LIMIT = 1024
    _ITEM_TEXT_BYTES_LIMIT = 6 * 1024
    _ITEM_ALIAS_LIMIT = 128
    _OPEN_STREAM_LIMIT = 128
    _TOOL_ITEM_LIMIT = 32
    _USAGE_BYTES_LIMIT = 16 * 1024

    _TERMINAL_EVENTS = set(RESPONSES_API_TERMINAL_STREAM_EVENTS)
    _BASE_RESPONSE_EVENTS = {"response.created", "response.in_progress"}
    _TOOL_ITEM_TYPES = {
        "code_interpreter_call",
        "computer_call",
        "file_search_call",
        "function_call",
        "image_generation_call",
        "mcp_call",
        "local_shell_call",
        "apply_patch_call",
        "custom_tool_call",
        "shell_call",
        "web_search_call",
    }
    _ARGUMENT_TOOL_ITEM_TYPES = {
        "apply_patch_call",
        "custom_tool_call",
        "function_call",
        "mcp_call",
    }
    _ITEM_STATUSES = {"completed", "failed", "in_progress", "incomplete"}
    _TOOL_LIFECYCLE_SUFFIXES = {
        "completed",
        "failed",
        "in_progress",
        "interpreting",
        "searching",
    }

    _OUTPUT_TEXT_EVENTS = {
        "response.output_text.delta",
        "response.output_text.done",
    }
    _REFUSAL_EVENTS = {"response.refusal.delta", "response.refusal.done"}
    _CONTENT_EVENTS = {"response.content_part.added", "response.content_part.done"}
    _REASONING_EVENTS = {
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
    }
    _TOOL_ARGUMENT_EVENTS = {
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
    }
    _OUTPUT_ITEM_EVENTS = {"response.output_item.added", "response.output_item.done"}

    def __init__(self, *, fallback_model: str) -> None:
        self._event_count = 0
        self._seen_event_types: set[str] = set()
        self._unsupported_event_types: set[str] = set()
        self._omitted_event_types = 0

        self._last_sequence_number: Optional[int] = None
        self._last_event_had_sequence_number = False
        self._sequence_numbers_seen = 0

        self._response_payload: Dict[str, Any] = {}
        self._has_response_envelope = False
        self._terminal_event_type: Optional[str] = None
        self._synthetic_terminal_event_type: Optional[str] = None

        self._usage_payload: Optional[Dict[str, Any]] = None
        self._usage_omitted = False

        self._item_states: Dict[str, Dict[str, Any]] = {}
        self._item_order: List[str] = []
        self._item_key_by_output_index: Dict[int, str] = {}
        self._item_key_aliases: Dict[str, str] = {}

        self._open_output_items: set[str] = set()
        self._open_text_items: set[str] = set()
        self._open_refusal_items: set[str] = set()
        self._open_reasoning_items: set[str] = set()
        self._open_content_items: set[str] = set()
        self._open_tool_argument_items: set[str] = set()
        self._open_tool_lifecycle_items: set[str] = set()

        self._parse_ambiguity = False
        self._sequence_ambiguity = False
        self._partial_frame = False
        self._state_omitted = False
        self._omitted_tool_items = 0
        self._omitted_output_text_bytes = 0
        self._fallback_model = self._bounded_string(fallback_model) or "unknown"

    @staticmethod
    def _coerce_non_empty_str(value: Any) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None

    def _bounded_string(self, value: Any) -> Optional[str]:
        normalized = self._coerce_non_empty_str(value)
        if normalized is None:
            return None
        encoded = normalized.encode("utf-8")
        if len(encoded) <= self._IDENTITY_BYTES_LIMIT:
            return normalized
        self._state_omitted = True
        return encoded[: self._IDENTITY_BYTES_LIMIT].decode("utf-8", "ignore")

    @staticmethod
    def _stream_item_key(
        *,
        item: Optional[dict] = None,
        item_id: Any = None,
        output_index: Any = None,
        fallback_key: str,
    ) -> str:
        if isinstance(item, dict):
            for key in ("id", "call_id"):
                value = _ResponsesSSEStateTracker._coerce_non_empty_str(item.get(key))
                if value:
                    return value
        value = _ResponsesSSEStateTracker._coerce_non_empty_str(item_id)
        if value:
            return value
        if isinstance(output_index, int):
            return f"output:{output_index}"
        return fallback_key

    def _mark_parse_ambiguity(self) -> None:
        self._parse_ambiguity = True

    def _mark_sequence_ambiguity(self) -> None:
        self._sequence_ambiguity = True

    def _record_sequence_number(self, event: dict) -> None:
        if "sequence_number" not in event:
            self._last_event_had_sequence_number = False
            return

        sequence_number = event.get("sequence_number")
        if (
            not isinstance(sequence_number, int)
            or isinstance(sequence_number, bool)
            or sequence_number < 0
        ):
            self._mark_parse_ambiguity()
            self._last_event_had_sequence_number = False
            return

        self._sequence_numbers_seen += 1
        if self._last_sequence_number is not None:
            if sequence_number <= self._last_sequence_number:
                self._mark_sequence_ambiguity()
            elif (
                self._last_event_had_sequence_number
                and sequence_number != self._last_sequence_number + 1
            ):
                self._mark_sequence_ambiguity()
        self._last_sequence_number = sequence_number
        self._last_event_had_sequence_number = True

    @staticmethod
    def _stream_key(item_key: str, parsed_chunk: dict, stream_type: str) -> str:
        content_index = parsed_chunk.get("content_index")
        summary_index = parsed_chunk.get("summary_index")
        index = content_index if isinstance(content_index, int) else summary_index
        suffix = str(index) if isinstance(index, int) else "default"
        return f"{item_key}:{stream_type}:{suffix}"

    def _open_stream(self, open_streams: set[str], stream_key: str) -> None:
        if stream_key in open_streams:
            return
        total_open_streams = sum(self._open_state_counts().values())
        if total_open_streams >= self._OPEN_STREAM_LIMIT:
            self._state_omitted = True
            return
        open_streams.add(stream_key)

    @staticmethod
    def _canonicalize_payload(event: Any) -> Optional[dict]:
        if not isinstance(event, dict):
            return None
        response_payload = event.get("response")
        if isinstance(response_payload, dict):
            return response_payload
        if {"id", "model", "status", "object"}.intersection(event.keys()):
            return event
        return None

    def _track_response_envelope(self, event: dict, response_payload: dict) -> None:
        self._has_response_envelope = self._has_response_envelope or isinstance(
            event.get("response"), dict
        )
        for key, value in (
            ("id", self._bounded_string(response_payload.get("id"))),
            ("model", self._bounded_string(response_payload.get("model"))),
            ("object", self._bounded_string(response_payload.get("object"))),
        ):
            if value is None:
                continue
            existing = self._response_payload.get(key)
            if existing is not None and existing != value:
                self._mark_sequence_ambiguity()
                continue
            self._response_payload[key] = value

        created_at = response_payload.get(
            "created_at", response_payload.get("created")
        )
        if isinstance(created_at, str):
            created_at = self._bounded_string(created_at)
        elif isinstance(created_at, bool) or not isinstance(
            created_at, (int, float, type(None))
        ):
            self._mark_parse_ambiguity()
            created_at = None
        if created_at is not None:
            existing_created_at = self._response_payload.get("created_at")
            if existing_created_at is not None and existing_created_at != created_at:
                self._mark_sequence_ambiguity()
            else:
                self._response_payload["created_at"] = created_at

        status = self._bounded_string(response_payload.get("status"))
        if status is not None:
            self._response_payload["status"] = status.lower()
        if response_payload.get("error") is not None:
            self._response_payload["error_seen"] = True
        if response_payload.get("incomplete_details") is not None:
            self._response_payload["incomplete_details_seen"] = True

    @staticmethod
    def _coerce_payload_item_type(item: Any) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        return _ResponsesSSEStateTracker._coerce_non_empty_str(item.get("type"))

    def _get_item_state(self, key: str) -> Dict[str, Any]:
        state = self._item_states.get(key)
        if state is not None:
            return state

        if len(self._item_states) >= self._TOOL_ITEM_LIMIT:
            self._omitted_tool_items += 1
            self._state_omitted = True
            return {}

        state = {
            "id": None,
            "call_id": None,
            "name": None,
            "role": None,
            "item_type": None,
            "output_added": False,
            "output_done": False,
            "status": None,
            "text_open": False,
            "text_done": False,
            "refusal_open": False,
            "refusal_done": False,
            "content_open": False,
            "content_done": False,
            "reasoning_open": False,
            "reasoning_done": False,
            "tool_arguments_open": False,
            "tool_arguments_done": False,
            "tool_lifecycle_done": False,
            "tool_argument_delta_hasher": hashlib.sha256(),
            "tool_argument_delta_size_bytes": 0,
            "tool_argument_final_hash": None,
            "tool_argument_final_size_bytes": 0,
            "tool_argument_size_bytes": 0,
            "text": "",
            "text_bytes": 0,
            "text_delta_hasher": hashlib.sha256(),
            "text_hash": None,
            "text_truncated": False,
            "refusal": "",
            "refusal_bytes": 0,
            "refusal_delta_hasher": hashlib.sha256(),
            "refusal_hash": None,
            "refusal_truncated": False,
        }
        self._item_states[key] = state
        self._item_order.append(key)
        return state

    def _resolve_item_key(self, parsed_chunk: dict) -> Optional[str]:
        item = parsed_chunk.get("item")
        item_id = parsed_chunk.get("item_id")
        output_index = parsed_chunk.get("output_index")
        if isinstance(output_index, bool):
            self._mark_parse_ambiguity()
            output_index = None
        aliases = [
            value
            for value in (
                self._bounded_string(item_id),
                self._bounded_string(item.get("id"))
                if isinstance(item, dict)
                else None,
                self._bounded_string(item.get("call_id"))
                if isinstance(item, dict)
                else None,
            )
            if value is not None
        ]
        existing_keys = {
            self._item_key_aliases[alias]
            for alias in aliases
            if alias in self._item_key_aliases
        }
        if isinstance(output_index, int) and output_index in self._item_key_by_output_index:
            existing_keys.add(self._item_key_by_output_index[output_index])
        if len(existing_keys) > 1:
            self._mark_sequence_ambiguity()
        if existing_keys:
            key = sorted(existing_keys)[0]
        else:
            fallback = f"output:{output_index}" if isinstance(output_index, int) else ""
            key = self._stream_item_key(
                item=item if isinstance(item, dict) else None,
                item_id=item_id,
                output_index=output_index,
                fallback_key=fallback,
            )
        if not key:
            self._mark_parse_ambiguity()
            return None

        if isinstance(output_index, int):
            existing = self._item_key_by_output_index.get(output_index)
            if existing is not None and existing != key:
                self._mark_sequence_ambiguity()
            if existing is not None or len(self._item_key_by_output_index) < self._ITEM_ALIAS_LIMIT:
                self._item_key_by_output_index[output_index] = key
            else:
                self._state_omitted = True
        for alias in aliases:
            existing = self._item_key_aliases.get(alias)
            if existing is not None and existing != key:
                self._mark_sequence_ambiguity()
            if existing is not None or len(self._item_key_aliases) < self._ITEM_ALIAS_LIMIT:
                self._item_key_aliases[alias] = key
            else:
                self._state_omitted = True
        return key

    def _consume_terminal_event(
        self,
        event_type: str,
        response_payload: dict,
        event: dict,
    ) -> None:
        if self._terminal_event_type is not None:
            self._mark_sequence_ambiguity()
        self._terminal_event_type = event_type
        self._track_response_envelope(response_payload=response_payload, event=event)

    def _set_item_value(
        self,
        state: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        normalized = self._bounded_string(value)
        if normalized is None:
            return
        existing = state.get(key)
        if existing is not None and existing != normalized:
            self._mark_sequence_ambiguity()
            return
        state[key] = normalized

    def _consume_output_item_event(  # noqa: PLR0915
        self, event_type: str, parsed_chunk: dict
    ) -> None:
        item = parsed_chunk.get("item")
        if not isinstance(item, dict):
            self._mark_parse_ambiguity()
            return

        key = self._resolve_item_key(parsed_chunk)
        if not isinstance(key, str):
            self._mark_parse_ambiguity()
            return
        state = self._get_item_state(key)
        if not state:
            return

        for field in ("id", "call_id", "name", "role"):
            self._set_item_value(state, field, item.get(field))
        item_type = self._coerce_payload_item_type(item)
        if item_type is not None:
            self._set_item_value(state, "item_type", item_type)
        else:
            self._mark_parse_ambiguity()

        status = self._coerce_non_empty_str(item.get("status"))
        if status:
            status = status.lower()
            if status not in self._ITEM_STATUSES:
                self._mark_parse_ambiguity()
                status = None
        if status:
            current_status = state.get("status")
            if current_status in {"completed", "failed", "incomplete"} and (
                current_status != status
            ):
                self._mark_sequence_ambiguity()
            else:
                state["status"] = status

        if event_type == "response.output_item.added":
            if state.get("output_done") or state.get("output_added"):
                self._mark_sequence_ambiguity()
            state["output_added"] = True
            self._open_stream(self._open_output_items, key)
            return

        if event_type == "response.output_item.done":
            if state.get("output_done"):
                self._mark_sequence_ambiguity()
                return
            state["output_done"] = True
            state["output_added"] = True
            self._open_output_items.discard(key)
            self._open_tool_lifecycle_items.discard(key)
            if state.get("status") in {None, "in_progress"}:
                state["status"] = "completed"

            if state.get("item_type") == "message":
                content = item.get("content")
                if content is not None and not isinstance(content, list):
                    self._mark_parse_ambiguity()
                elif isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            self._mark_parse_ambiguity()
                            continue
                        if part.get("type") == "output_text":
                            self._replace_final_text_value(
                                "text", state, part.get("text")
                            )
                        elif part.get("type") == "refusal":
                            self._replace_final_text_value(
                                "refusal", state, part.get("refusal")
                            )

            if state.get("item_type") in self._TOOL_ITEM_TYPES:
                arguments = item.get("arguments", item.get("input"))
                if arguments is not None:
                    self._finalize_tool_arguments(state, key, arguments)
            return

    def _append_text_stream_value(
        self,
        bucket: str,
        state: Dict[str, Any],
        value: str,
    ) -> None:
        if not isinstance(value, str):
            self._mark_parse_ambiguity()
            return
        if state.get(f"{bucket}_done"):
            self._mark_sequence_ambiguity()
            return

        encoded = value.encode("utf-8")
        state[f"{bucket}_delta_hasher"].update(encoded)
        state[f"{bucket}_bytes"] = state.get(f"{bucket}_bytes", 0) + len(encoded)
        retained = state.get(bucket, "").encode("utf-8")
        if len(retained) < self._ITEM_TEXT_BYTES_LIMIT:
            retained += encoded[: self._ITEM_TEXT_BYTES_LIMIT - len(retained)]
            state[bucket] = retained.decode("utf-8", "ignore")
        state[f"{bucket}_hash"] = state[f"{bucket}_delta_hasher"].hexdigest()
        state[f"{bucket}_truncated"] = (
            state[f"{bucket}_bytes"] > self._ITEM_TEXT_BYTES_LIMIT
        )

    def _replace_final_text_value(
        self,
        bucket: str,
        state: Dict[str, Any],
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            self._mark_parse_ambiguity()
            return

        encoded = value.encode("utf-8")
        final_hash = hashlib.sha256(encoded).hexdigest()
        if state.get(f"{bucket}_done"):
            if (
                state.get(f"{bucket}_bytes") != len(encoded)
                or state.get(f"{bucket}_hash") != final_hash
            ):
                self._mark_sequence_ambiguity()
            return
        if state.get(f"{bucket}_bytes", 0) and (
            state.get(f"{bucket}_bytes") != len(encoded)
            or state[f"{bucket}_delta_hasher"].hexdigest() != final_hash
        ):
            self._mark_sequence_ambiguity()

        state[f"{bucket}_bytes"] = len(encoded)
        state[f"{bucket}_hash"] = final_hash
        state[f"{bucket}_truncated"] = len(encoded) > self._ITEM_TEXT_BYTES_LIMIT
        state[bucket] = encoded[: self._ITEM_TEXT_BYTES_LIMIT].decode(
            "utf-8", "ignore"
        )
        state[f"{bucket}_done"] = True

    def _consume_text_stream(self, parsed_chunk: dict, stream_type: str) -> None:
        key = self._resolve_item_key(parsed_chunk)
        if not isinstance(key, str):
            self._mark_parse_ambiguity()
            return
        state = self._get_item_state(key)
        if not state:
            return

        event_type = parsed_chunk.get("type")
        is_done = str(event_type).endswith(".done")
        stream_key = self._stream_key(key, parsed_chunk, stream_type)
        if state.get("item_type") is None:
            state["item_type"] = "message"
        if state.get("role") is None:
            state["role"] = "assistant"

        if stream_type == "text":
            state["text_open"] = True
            if is_done:
                self._replace_final_text_value(
                    "text", state, parsed_chunk.get("text")
                )
                self._open_text_items.discard(stream_key)
            else:
                self._append_text_stream_value(
                    "text", state, parsed_chunk.get("delta")
                )
                self._open_stream(self._open_text_items, stream_key)
            return

        state["refusal_open"] = True
        if is_done:
            self._replace_final_text_value(
                "refusal", state, parsed_chunk.get("refusal")
            )
            self._open_refusal_items.discard(stream_key)
        else:
            self._append_text_stream_value(
                "refusal", state, parsed_chunk.get("delta")
            )
            self._open_stream(self._open_refusal_items, stream_key)

    def _consume_reasoning_or_content_stream(
        self,
        event_type: str,
        key: str,
        parsed_chunk: dict,
    ) -> None:
        if event_type in {"response.content_part.added", "response.content_part.done"}:
            state = self._get_item_state(key)
            if not state:
                return
            part = parsed_chunk.get("part")
            if not isinstance(part, dict):
                self._mark_parse_ambiguity()
                return
            stream_key = self._stream_key(key, parsed_chunk, "content")
            if event_type.endswith(".added"):
                state["content_open"] = True
                self._open_stream(self._open_content_items, stream_key)
            else:
                state["content_done"] = True
                self._open_content_items.discard(stream_key)
                if part.get("type") == "output_text":
                    self._replace_final_text_value("text", state, part.get("text"))
                elif part.get("type") == "refusal":
                    self._replace_final_text_value(
                        "refusal", state, part.get("refusal")
                    )
            return

        state = self._get_item_state(key)
        if not state:
            return
        if event_type.startswith("response.reasoning_summary_"):
            stream_type = "reasoning_part" if "_part." in event_type else "reasoning_text"
            stream_key = f"{key}:{stream_type}"
            if event_type.endswith(".added") or event_type.endswith(".delta"):
                value = (
                    parsed_chunk.get("part")
                    if event_type.endswith(".added")
                    else parsed_chunk.get("delta")
                )
                if (
                    event_type.endswith(".added")
                    and not isinstance(value, dict)
                ) or (event_type.endswith(".delta") and not isinstance(value, str)):
                    self._mark_parse_ambiguity()
                    return
                state["reasoning_open"] = True
                self._open_stream(self._open_reasoning_items, stream_key)
            else:
                value = (
                    parsed_chunk.get("part")
                    if "_part." in event_type
                    else parsed_chunk.get("text")
                )
                if ("_part." in event_type and not isinstance(value, dict)) or (
                    "_part." not in event_type and not isinstance(value, str)
                ):
                    self._mark_parse_ambiguity()
                    return
                state["reasoning_done"] = True
                self._open_reasoning_items.discard(stream_key)

    def _consume_tool_argument_event(self, event_type: str, parsed_chunk: dict) -> None:
        key = self._resolve_item_key(parsed_chunk)
        if not isinstance(key, str):
            self._mark_parse_ambiguity()
            return

        state = self._get_item_state(key)
        if not state:
            return
        if state.get("item_type") is None:
            if event_type.startswith("response.mcp_call_arguments"):
                state["item_type"] = "mcp_call"
            elif event_type.startswith("response.custom_tool_call_input"):
                state["item_type"] = "custom_tool_call"
            else:
                state["item_type"] = "function_call"

        if event_type.endswith(".done"):
            self._finalize_tool_arguments(
                state,
                key,
                parsed_chunk.get("arguments", parsed_chunk.get("input")),
            )
            return

        value = parsed_chunk.get("delta")
        if not isinstance(value, str):
            self._mark_parse_ambiguity()
            return
        if state.get("tool_arguments_done"):
            self._mark_sequence_ambiguity()
            return
        encoded = value.encode("utf-8")
        state["tool_argument_delta_hasher"].update(encoded)
        state["tool_argument_delta_size_bytes"] += len(encoded)
        state["tool_argument_size_bytes"] = state[
            "tool_argument_delta_size_bytes"
        ]
        state["tool_arguments_open"] = True
        self._open_stream(self._open_tool_argument_items, key)

    def _finalize_tool_arguments(
        self,
        state: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            self._mark_parse_ambiguity()
            return
        encoded = value.encode("utf-8")
        final_hash = hashlib.sha256(encoded).hexdigest()
        if state.get("tool_arguments_done"):
            if (
                state.get("tool_argument_final_hash") != final_hash
                or state.get("tool_argument_final_size_bytes") != len(encoded)
            ):
                self._mark_sequence_ambiguity()
            return
        if state.get("tool_argument_delta_size_bytes", 0) and (
            state.get("tool_argument_delta_size_bytes") != len(encoded)
            or state["tool_argument_delta_hasher"].hexdigest() != final_hash
        ):
            self._mark_sequence_ambiguity()

        state["tool_arguments_done"] = True
        state["tool_arguments_open"] = False
        state["tool_argument_final_hash"] = final_hash
        state["tool_argument_final_size_bytes"] = len(encoded)
        state["tool_argument_size_bytes"] = len(encoded)
        self._open_tool_argument_items.discard(key)

    def _consume_tool_lifecycle_event(
        self,
        event_type: str,
        parsed_chunk: dict,
    ) -> bool:
        if not event_type.startswith("response."):
            return False
        item_type, separator, suffix = event_type.removeprefix("response.").rpartition(
            "."
        )
        if (
            not separator
            or item_type not in self._TOOL_ITEM_TYPES
            or suffix not in self._TOOL_LIFECYCLE_SUFFIXES
        ):
            return False

        key = self._resolve_item_key(parsed_chunk)
        if key is None:
            return True
        state = self._get_item_state(key)
        if not state:
            return True
        self._set_item_value(state, "item_type", item_type)
        if suffix in {"in_progress", "interpreting", "searching"}:
            if state.get("tool_lifecycle_done") or state.get("status") in {
                "completed",
                "failed",
                "incomplete",
            }:
                self._mark_sequence_ambiguity()
            else:
                state["status"] = "in_progress"
            self._open_stream(self._open_tool_lifecycle_items, key)
            return True

        if state.get("status") in {"completed", "failed", "incomplete"}:
            expected_status = "failed" if suffix == "failed" else "completed"
            if state.get("status") != expected_status:
                self._mark_sequence_ambiguity()
        state["tool_lifecycle_done"] = True
        state["status"] = "failed" if suffix == "failed" else "completed"
        self._open_tool_lifecycle_items.discard(key)
        return True

    def _rebuild_tool_argument_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        arguments_hash = state.get("tool_argument_final_hash")
        arguments_size = state.get("tool_argument_final_size_bytes", 0)
        if arguments_hash is None and state.get("tool_argument_delta_size_bytes", 0):
            arguments_hash = state["tool_argument_delta_hasher"].hexdigest()
            arguments_size = state.get("tool_argument_delta_size_bytes", 0)
        return {
            "arguments_done": bool(state.get("tool_arguments_done", False)),
            "arguments_hash": arguments_hash,
            "arguments_hash_algorithm": "sha256",
            "arguments_raw_bytes_retained": 0,
            "arguments_size_bytes": arguments_size,
            "tool_arguments_open": bool(state.get("tool_arguments_open", False)),
        }

    def _consume_usage(self, response_payload: dict) -> None:
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            return
        try:
            serialized_usage = json.dumps(
                usage,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            self._usage_payload = None
            self._usage_omitted = True
            return
        if len(serialized_usage.encode("utf-8")) > self._USAGE_BYTES_LIMIT:
            self._usage_payload = None
            self._usage_omitted = True
            return
        self._usage_payload = json.loads(serialized_usage)
        self._usage_omitted = False

    def consume(self, decoded_event: Any) -> None:  # noqa: PLR0915
        if not isinstance(decoded_event, dict):
            self._mark_parse_ambiguity()
            return

        self._event_count += 1
        event_type = decoded_event.get("type")
        if not isinstance(event_type, str) or not event_type.strip():
            self._mark_parse_ambiguity()
            return
        event_type = event_type.strip()
        encoded_event_type = event_type.encode("utf-8")
        if len(encoded_event_type) > self._IDENTITY_BYTES_LIMIT:
            self._state_omitted = True
            event_type = encoded_event_type[: self._IDENTITY_BYTES_LIMIT].decode(
                "utf-8", "ignore"
            )

        if (
            event_type not in self._seen_event_types
            and len(self._seen_event_types) >= self._EVENT_TYPES_LIMIT
        ):
            self._omitted_event_types += 1
        else:
            self._seen_event_types.add(event_type)
        self._record_sequence_number(decoded_event)

        response_id = self._bounded_string(decoded_event.get("response_id"))
        if response_id is not None:
            existing_response_id = self._response_payload.get("id")
            if existing_response_id is not None and existing_response_id != response_id:
                self._mark_sequence_ambiguity()
            elif existing_response_id is None:
                self._response_payload["id"] = response_id

        response_payload = self._canonicalize_payload(decoded_event)
        if response_payload:
            self._track_response_envelope(
                response_payload=response_payload, event=decoded_event
            )
            self._consume_usage(response_payload)
        elif event_type in self._BASE_RESPONSE_EVENTS:
            self._mark_parse_ambiguity()

        if event_type in self._TERMINAL_EVENTS:
            if not isinstance(response_payload, dict):
                self._mark_parse_ambiguity()
                response_payload = {}
            self._consume_terminal_event(event_type, response_payload, decoded_event)
            return
        if self._terminal_event_type is not None:
            self._mark_sequence_ambiguity()
            return

        if event_type in self._BASE_RESPONSE_EVENTS:
            return

        if event_type in self._OUTPUT_ITEM_EVENTS:
            self._consume_output_item_event(event_type, decoded_event)
            return

        if event_type in self._OUTPUT_TEXT_EVENTS:
            self._consume_text_stream(decoded_event, stream_type="text")
            return

        if event_type in self._REFUSAL_EVENTS:
            self._consume_text_stream(decoded_event, stream_type="refusal")
            return

        if event_type in self._CONTENT_EVENTS or event_type in self._REASONING_EVENTS:
            key = self._resolve_item_key(decoded_event)
            if isinstance(key, str):
                self._consume_reasoning_or_content_stream(
                    event_type, key, decoded_event
                )
            else:
                self._mark_parse_ambiguity()
            return

        if event_type in self._TOOL_ARGUMENT_EVENTS:
            self._consume_tool_argument_event(event_type, decoded_event)
            return

        if self._consume_tool_lifecycle_event(event_type, decoded_event):
            return

        if event_type in {"response.output_text.annotation.added", "token_count"}:
            return

        if len(self._unsupported_event_types) < self._EVENT_TYPES_LIMIT:
            self._unsupported_event_types.add(event_type)
        else:
            self._omitted_event_types += 1

    def _has_open_streams(self) -> bool:
        return bool(
            self._open_output_items
            or self._open_text_items
            or self._open_refusal_items
            or self._open_reasoning_items
            or self._open_content_items
            or self._open_tool_argument_items
            or self._open_tool_lifecycle_items
        )

    def _has_failed_outputs(self) -> bool:
        for key in self._item_order:
            if key not in self._item_states:
                continue
            if self._item_states[key].get("status") in {
                "failed",
                "incomplete",
                "in_progress",
            }:
                return True
        return False

    def _has_final_assistant_output(self) -> bool:
        for state in self._item_states.values():
            if state.get("status") in {"failed", "incomplete", "in_progress"}:
                continue
            if state.get("role") not in {None, "assistant"}:
                continue
            if state.get("item_type") not in {None, "message"}:
                continue
            if state.get("text_done") and state.get("text_bytes", 0) > 0:
                return True
            if state.get("refusal_done") and state.get("refusal_bytes", 0) > 0:
                return True
        return False

    def _has_final_tool_call(self) -> bool:
        for state in self._item_states.values():
            item_type = state.get("item_type")
            if item_type not in self._TOOL_ITEM_TYPES:
                continue
            if state.get("status") in {"failed", "incomplete", "in_progress"}:
                continue
            if not (
                state.get("output_done") or state.get("tool_lifecycle_done")
            ):
                continue
            if item_type in self._ARGUMENT_TOOL_ITEM_TYPES and not state.get(
                "tool_arguments_done"
            ):
                continue
            return True
        return False

    def _open_state_counts(self) -> Dict[str, int]:
        return {
            "content_parts": len(self._open_content_items),
            "output_items": len(self._open_output_items),
            "output_text": len(self._open_text_items),
            "reasoning": len(self._open_reasoning_items),
            "refusals": len(self._open_refusal_items),
            "tool_arguments": len(self._open_tool_argument_items),
            "tool_lifecycle": len(self._open_tool_lifecycle_items),
        }

    @property
    def provider_terminal_observed(self) -> bool:
        return self._terminal_event_type is not None

    @property
    def provider_terminal_event_type(self) -> Optional[str]:
        return self._terminal_event_type

    @property
    def synthetic_terminal_event_type(self) -> Optional[str]:
        return self._synthetic_terminal_event_type

    @property
    def has_partial_frame(self) -> bool:
        return self._partial_frame

    @property
    def has_ambiguity(self) -> bool:
        return self._parse_ambiguity or self._sequence_ambiguity

    @property
    def has_open_items(self) -> bool:
        return self._has_open_streams()

    def mark_partial_frame(self) -> None:
        self._partial_frame = True

    def mark_parse_ambiguity(self) -> None:
        self._mark_parse_ambiguity()

    def _build_tool_output_item(self, key: str, state: Dict[str, Any]) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": state.get("id") or key,
            "type": state.get("item_type") or "unknown",
        }
        for field in ("call_id", "name", "role"):
            if state.get(field) is not None:
                item[field] = state[field]
        content: List[Dict[str, Any]] = []
        if state.get("text_done"):
            content.append(
                {
                    "annotations": [],
                    "text": state.get("text", ""),
                    "type": "output_text",
                }
            )
        if state.get("refusal_done"):
            content.append(
                {
                    "refusal": state.get("refusal", ""),
                    "type": "refusal",
                }
            )
        if content:
            item["content"] = content
        if state.get("item_type") in self._TOOL_ITEM_TYPES:
            item["aawm_argument_summary"] = self._rebuild_tool_argument_state(state)
        if state.get("status"):
            item["status"] = state.get("status")
        elif state.get("output_done"):
            item["status"] = "completed"

        return item

    def _build_omission_metadata(self) -> Dict[str, Any]:
        omitted_text_bytes = 0
        for state in self._item_states.values():
            omitted_text_bytes += max(
                0,
                int(state.get("text_bytes", 0))
                - len(str(state.get("text", "")).encode("utf-8")),
            )
            omitted_text_bytes += max(
                0,
                int(state.get("refusal_bytes", 0))
                - len(str(state.get("refusal", "")).encode("utf-8")),
            )
        self._omitted_output_text_bytes = omitted_text_bytes
        return {
            "responses_stream_events_seen": self._event_count,
            "responses_stream_event_types_omitted": self._omitted_event_types,
            "responses_stream_state_omitted": self._state_omitted,
            "responses_stream_tool_items_omitted": self._omitted_tool_items,
            "responses_stream_output_text_bytes_truncated": bool(
                self._omitted_output_text_bytes
            ),
            "responses_stream_output_text_bytes_omitted": self._omitted_output_text_bytes,
            "responses_stream_unsupported_event_types": sorted(
                self._unsupported_event_types
            ),
            "responses_stream_parse_ambiguity": self._parse_ambiguity,
            "responses_stream_sequence_ambiguity": self._sequence_ambiguity,
            "responses_stream_tool_argument_raw_bytes_retained": 0,
            "responses_stream_usage_omitted": self._usage_omitted,
        }

    def _tool_argument_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for key in self._item_order:
            state = self._item_states.get(key)
            if state is None or state.get("item_type") not in self._TOOL_ITEM_TYPES:
                continue
            summaries.append(
                {
                    "item_id": state.get("id") or key,
                    "item_type": state.get("item_type"),
                    **self._rebuild_tool_argument_state(state),
                }
            )
        return summaries

    def _item_status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for state in self._item_states.values():
            status = state.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items()))

    def metadata_summary(self) -> Dict[str, Any]:
        return {
            "event_count": self._event_count,
            "final_assistant_output": self._has_final_assistant_output(),
            "final_tool_call": self._has_final_tool_call(),
            "item_status_counts": self._item_status_counts(),
            "last_sequence_number": self._last_sequence_number,
            "omission_metadata": self._build_omission_metadata(),
            "open_state_counts": self._open_state_counts(),
            "parse_ambiguity": self._parse_ambiguity,
            "partial_frame": self._partial_frame,
            "provider_terminal_event_type": self._terminal_event_type,
            "provider_terminal_observed": self.provider_terminal_observed,
            "response_base_envelope_seen": self._has_response_envelope,
            "response_identity_seen": bool(self._response_payload.get("id")),
            "sequence_ambiguity": self._sequence_ambiguity,
            "sequence_numbers_seen": self._sequence_numbers_seen,
            "state_omitted": self._state_omitted,
            "synthetic_terminal_event_type": self._synthetic_terminal_event_type,
            "tool_argument_summaries": self._tool_argument_summaries(),
            "unsupported_event_types": sorted(self._unsupported_event_types),
        }

    def _build_incomplete_payload(self, reason: str) -> Dict[str, Any]:
        return self._build_synthetic_terminal_event(
            event_type="response.incomplete",
            incomplete_reason=reason,
        )

    def _build_completed_payload(self) -> Dict[str, Any]:
        return self._build_synthetic_terminal_event(
            event_type="response.completed",
        )

    def _build_synthetic_terminal_event(
        self,
        *,
        event_type: Literal["response.completed", "response.incomplete"],
        incomplete_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._synthetic_terminal_event_type = event_type
        response_payload: Dict[str, Any] = {
            "metadata": {
                "aawm_stream_omission_metadata": self._build_omission_metadata(),
                "aawm_stream_provider_terminal_omitted": True,
                "aawm_stream_synthetic_terminal": True,
                "aawm_stream_tracker_state": self.metadata_summary(),
            },
            "model": self._response_payload.get("model") or self._fallback_model,
            "object": self._response_payload.get("object") or "response",
            "output": [
                self._build_tool_output_item(key, self._item_states[key])
                for key in self._item_order
                if key in self._item_states
            ],
            "status": "completed"
            if event_type == "response.completed"
            else "incomplete",
        }
        if "id" in self._response_payload:
            response_payload["id"] = self._response_payload["id"]
        if "created_at" in self._response_payload:
            response_payload["created_at"] = self._response_payload["created_at"]
        if self._usage_payload is not None:
            response_payload["usage"] = self._usage_payload
        if incomplete_reason is not None:
            response_payload["incomplete_details"] = {"reason": incomplete_reason}
            response_payload["metadata"][
                "aawm_stream_incomplete_reason"
            ] = incomplete_reason

        synthetic_event: Dict[str, Any] = {
            "response": response_payload,
            "type": event_type,
        }
        if self._last_sequence_number is not None and not self._sequence_ambiguity:
            synthetic_event["sequence_number"] = self._last_sequence_number + 1
        return synthetic_event

    def _determine_incomplete_reason(self) -> Optional[str]:
        if self._partial_frame:
            return "upstream_stream_partial_frame"
        if not self._has_response_envelope or not self._response_payload.get("id"):
            return "upstream_stream_missing_base_envelope"
        if self._parse_ambiguity:
            return "upstream_stream_parse_ambiguity"
        if self._sequence_ambiguity:
            return "upstream_stream_sequence_ambiguity"
        if self._state_omitted:
            return "upstream_stream_state_omitted"
        if self._unsupported_event_types:
            return "upstream_stream_unsupported_payload_shape"
        if (
            self._response_payload.get("error_seen")
            or self._response_payload.get("incomplete_details_seen")
            or self._response_payload.get("status") in {"failed", "incomplete"}
            or self._has_failed_outputs()
        ):
            return "upstream_stream_incomplete_item_state"
        if self._has_open_streams():
            return "upstream_stream_unfinished_streams"
        if not (
            self._has_final_assistant_output() or self._has_final_tool_call()
        ):
            return "upstream_stream_missing_final_assistant_or_tool_output"
        return None

    def classify_clean_eof(self) -> Optional[Dict[str, Any]]:
        """Return one synthetic terminal event when the provider omitted its own."""
        if self._terminal_event_type is not None:
            return None
        incomplete_reason = self._determine_incomplete_reason()
        if incomplete_reason is not None:
            return self._build_incomplete_payload(incomplete_reason)
        return self._build_completed_payload()

    def synthetic_terminal_payload_at_eof(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for ``classify_clean_eof``."""
        return self.classify_clean_eof()


def _reset_model_price_lookup_caches() -> None:
    """Clear on-disk model-price map and negative-lookup caches.

    Intended for tests so process-lifetime memoization does not leak across cases.
    """
    _MODEL_PRICE_MAP_CACHE.clear()
    _MODEL_PRICE_DISK_MISS_CACHE.clear()


class OpenAIPassthroughLoggingHandler(BasePassthroughLoggingHandler):
    """
    OpenAI-specific passthrough logging handler that provides cost tracking for /chat/completions endpoints.
    """

    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.OPENAI

    @staticmethod
    def _create_responses_sse_state_tracker(
        *,
        fallback_model: str,
    ) -> _ResponsesSSEStateTracker:
        return _ResponsesSSEStateTracker(fallback_model=fallback_model)

    @staticmethod
    def _consume_responses_sse_event(
        tracker: _ResponsesSSEStateTracker,
        decoded_event: Any,
    ) -> None:
        tracker.consume(decoded_event)

    @staticmethod
    def _mark_responses_sse_partial_frame(
        tracker: _ResponsesSSEStateTracker,
    ) -> None:
        tracker.mark_partial_frame()

    @staticmethod
    def _mark_responses_sse_parse_ambiguity(
        tracker: _ResponsesSSEStateTracker,
    ) -> None:
        tracker.mark_parse_ambiguity()

    @staticmethod
    def _responses_sse_tracker_metadata(
        tracker: _ResponsesSSEStateTracker,
    ) -> Dict[str, Any]:
        return tracker.metadata_summary()

    @staticmethod
    def _classify_responses_sse_clean_eof(
        tracker: _ResponsesSSEStateTracker,
    ) -> Optional[Dict[str, Any]]:
        return tracker.classify_clean_eof()

    @staticmethod
    def _is_openai_compatible_hostname(hostname: Optional[str]) -> bool:
        if not hostname:
            return False
        return (
            "api.openai.com" in hostname
            or "openai.azure.com" in hostname
            or "chatgpt.com" in hostname
            or hostname == "integrate.api.nvidia.com"
            or hostname == "ai.api.nvidia.com"
            or hostname == "openrouter.ai"
            or hostname.endswith(".openrouter.ai")
            or hostname == "opencode.ai"
            or hostname.endswith(".opencode.ai")
            or hostname == "api.x.ai"
            or hostname == "cli-chat-proxy.grok.com"
        )

    def get_provider_config(self, model: str) -> OpenAIConfigType:
        """Get OpenAI provider configuration for the given model."""
        return OpenAIConfig()

    @staticmethod
    def _candidate_model_price_keys(
        model: str,
        custom_llm_provider: Optional[str],
    ) -> List[str]:
        candidates = [model]
        if custom_llm_provider == "openrouter":
            if not model.startswith("openrouter/"):
                candidates.append(f"openrouter/{model}")
            else:
                candidates.append(model.removeprefix("openrouter/"))
        elif custom_llm_provider == "xai":
            if not model.startswith("xai/"):
                candidates.append(f"xai/{model}")
            else:
                candidates.append(model.removeprefix("xai/"))
        elif custom_llm_provider == "opencode_zen":
            if not model.startswith("opencode/"):
                candidates.append(f"opencode/{model}")
            else:
                candidates.append(model.removeprefix("opencode/"))
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    @staticmethod
    def _model_price_file_paths() -> tuple[Path, ...]:
        """Return on-disk model price sources in lookup precedence order."""
        package_root = Path(getattr(litellm, "__file__", "")).resolve().parent
        return (
            package_root / "model_prices_and_context_window.json",
            package_root / "bundled_model_prices_and_context_window_fallback.json",
            package_root.parent / "model_prices_and_context_window.json",
        )

    @staticmethod
    def _get_cached_model_price_map(price_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load and cache a single on-disk model price map for process lifetime.

        Missing, unreadable, or non-dict files are cached as None so later
        fallbacks do not re-stat/re-read them on every request.
        """
        cache_key = str(price_file)
        if cache_key in _MODEL_PRICE_MAP_CACHE:
            return _MODEL_PRICE_MAP_CACHE[cache_key]

        price_map: Optional[Dict[str, Any]]
        if not price_file.is_file():
            price_map = None
        else:
            try:
                loaded = json.loads(price_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                price_map = None
            else:
                price_map = loaded if isinstance(loaded, dict) else None

        _MODEL_PRICE_MAP_CACHE[cache_key] = price_map
        return price_map

    @staticmethod
    def _record_model_price_disk_miss(candidates: List[str]) -> None:
        """Remember candidates confirmed missing from all on-disk price maps."""
        for candidate in candidates:
            if candidate in _MODEL_PRICE_DISK_MISS_CACHE:
                continue
            if len(_MODEL_PRICE_DISK_MISS_CACHE) >= _MODEL_PRICE_DISK_MISS_CACHE_MAXSIZE:
                # Drop an arbitrary entry; correctness only requires eventual re-scan.
                _MODEL_PRICE_DISK_MISS_CACHE.pop()
            _MODEL_PRICE_DISK_MISS_CACHE.add(candidate)

    @staticmethod
    def _lookup_model_price_info(
        model: str,
        custom_llm_provider: Optional[str],
    ) -> Optional[dict]:
        candidates = OpenAIPassthroughLoggingHandler._candidate_model_price_keys(
            model,
            custom_llm_provider,
        )

        model_cost = getattr(litellm, "model_cost", None)
        if isinstance(model_cost, dict):
            for candidate in candidates:
                model_info = model_cost.get(candidate)
                if isinstance(model_info, dict):
                    return model_info

        # Candidates already confirmed absent from every on-disk map can skip
        # the disk scan. Runtime litellm.model_cost was already checked above.
        disk_candidates = [
            candidate
            for candidate in candidates
            if candidate not in _MODEL_PRICE_DISK_MISS_CACHE
        ]
        if not disk_candidates:
            return None

        for price_file in OpenAIPassthroughLoggingHandler._model_price_file_paths():
            price_map = OpenAIPassthroughLoggingHandler._get_cached_model_price_map(
                price_file
            )
            if price_map is None:
                continue
            for candidate in disk_candidates:
                model_info = price_map.get(candidate)
                if isinstance(model_info, dict):
                    return model_info
        OpenAIPassthroughLoggingHandler._record_model_price_disk_miss(disk_candidates)
        return None

    @staticmethod
    def _completion_cost_with_model_price_fallback(
        *,
        completion_response: Any,
        model: str,
        custom_llm_provider: Optional[str],
        call_type: Optional[str] = None,
    ) -> float:
        try:
            return litellm.completion_cost(
                completion_response=completion_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                call_type=call_type,
            )
        except Exception as exc:
            model_info = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
                model,
                custom_llm_provider,
            )
            usage = getattr(completion_response, "usage", None)
            if not isinstance(model_info, dict) or usage is None:
                if custom_llm_provider == "openrouter" and usage is not None:
                    verbose_proxy_logger.debug(
                        "OpenAI passthrough cost unavailable for unmapped OpenRouter model=%s; recording zero cost and preserving usage.",
                        model,
                    )
                    return 0.0
                raise
            input_cost_per_token = model_info.get("input_cost_per_token")
            output_cost_per_token = model_info.get("output_cost_per_token")
            if not isinstance(input_cost_per_token, (int, float)) or not isinstance(
                output_cost_per_token,
                (int, float),
            ):
                raise
            prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
            completion_tokens = getattr(usage, "completion_tokens", None) or 0
            fallback_cost = (
                float(prompt_tokens) * float(input_cost_per_token)
                + float(completion_tokens) * float(output_cost_per_token)
            )
            verbose_proxy_logger.warning(
                "OpenAI passthrough cost fallback used for model=%s provider=%s after completion_cost error: %s",
                model,
                custom_llm_provider,
                str(exc),
            )
            return fallback_cost

    @staticmethod
    def is_openai_chat_completions_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI chat completions endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            OpenAIPassthroughLoggingHandler._is_openai_compatible_hostname(
                parsed_url.hostname
            )
            and "/v1/chat/completions" in parsed_url.path
        )

    @staticmethod
    def is_openai_image_generation_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image generation endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            OpenAIPassthroughLoggingHandler._is_openai_compatible_hostname(
                parsed_url.hostname
            )
            and "/v1/images/generations" in parsed_url.path
        )

    @staticmethod
    def is_openai_image_editing_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image editing endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            OpenAIPassthroughLoggingHandler._is_openai_compatible_hostname(
                parsed_url.hostname
            )
            and "/v1/images/edits" in parsed_url.path
        )

    @staticmethod
    def is_openai_responses_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI responses API endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            OpenAIPassthroughLoggingHandler._is_openai_compatible_hostname(
                parsed_url.hostname
            )
            and ("/v1/responses" in parsed_url.path or "/responses" in parsed_url.path)
        )

    @staticmethod
    def is_openai_embeddings_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI embeddings endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            OpenAIPassthroughLoggingHandler._is_openai_compatible_hostname(
                parsed_url.hostname
            )
            and ("/v1/embeddings" in parsed_url.path or "/embeddings" in parsed_url.path)
        )

    @staticmethod
    def _extract_passthrough_model_fallback(kwargs: dict) -> Optional[str]:
        litellm_params = kwargs.get("litellm_params")
        metadata = (
            litellm_params.get("metadata")
            if isinstance(litellm_params, dict)
            and isinstance(litellm_params.get("metadata"), dict)
            else {}
        )
        for candidate in (
            metadata.get("grok_model_override"),
            metadata.get("model_group"),
            metadata.get("model"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        header_sources = (
            kwargs.get("passthrough_logging_payload", {}).get("request_headers")
            if isinstance(kwargs.get("passthrough_logging_payload"), dict)
            else None,
            litellm_params.get("proxy_server_request", {}).get("headers")
            if isinstance(litellm_params, dict)
            and isinstance(litellm_params.get("proxy_server_request"), dict)
            else None,
        )
        for headers in header_sources:
            if not isinstance(headers, dict):
                continue
            for header_name, header_value in headers.items():
                if (
                    isinstance(header_name, str)
                    and header_name.lower() == "x-grok-model-override"
                    and isinstance(header_value, str)
                    and header_value.strip()
                ):
                    return header_value.strip()
        return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_embedding_usage(response_body: dict) -> Usage:
        usage = response_body.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = (
            OpenAIPassthroughLoggingHandler._safe_int(usage.get("prompt_tokens"))
            or OpenAIPassthroughLoggingHandler._safe_int(usage.get("input_tokens"))
            or OpenAIPassthroughLoggingHandler._safe_int(usage.get("total_tokens"))
            or 0
        )
        completion_tokens = (
            OpenAIPassthroughLoggingHandler._safe_int(usage.get("completion_tokens"))
            or OpenAIPassthroughLoggingHandler._safe_int(usage.get("output_tokens"))
            or 0
        )
        total_tokens = (
            OpenAIPassthroughLoggingHandler._safe_int(usage.get("total_tokens"))
            or prompt_tokens + completion_tokens
        )
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details"),
        )

    @staticmethod
    def _calculate_embedding_cost(
        *,
        embedding_response: EmbeddingResponse,
        model: str,
        custom_llm_provider: Optional[str],
    ) -> float:
        try:
            return litellm.completion_cost(
                completion_response=embedding_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                call_type="embedding",
            )
        except Exception as exc:
            model_info = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
                model,
                custom_llm_provider,
            )
            usage = getattr(embedding_response, "usage", None)
            if not isinstance(model_info, dict) or usage is None:
                raise
            input_cost_per_token = model_info.get("input_cost_per_token")
            if not isinstance(input_cost_per_token, (int, float)):
                raise

            prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = 0
            if isinstance(prompt_details, dict):
                cached_tokens = int(prompt_details.get("cached_tokens") or 0)
            else:
                cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
            cache_read_cost = model_info.get("cache_read_input_token_cost")
            if not isinstance(cache_read_cost, (int, float)):
                cache_read_cost = input_cost_per_token
            uncached_tokens = max(int(prompt_tokens) - cached_tokens, 0)
            fallback_cost = (
                uncached_tokens * float(input_cost_per_token)
                + cached_tokens * float(cache_read_cost)
            )
            verbose_proxy_logger.warning(
                "OpenAI passthrough embedding cost fallback used for model=%s provider=%s after completion_cost error: %s",
                model,
                custom_llm_provider,
                str(exc),
            )
            return fallback_cost

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        """Extract user information from passthrough logging payload."""
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return request_body.get("user")
        return None

    @staticmethod
    def _append_langfuse_span_to_kwargs(
        kwargs: dict,
        *,
        name: str,
        span_metadata: Optional[dict] = None,
    ) -> None:
        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        langfuse_spans = metadata.get("langfuse_spans") or []
        if not isinstance(langfuse_spans, list):
            langfuse_spans = []
        descriptor = {"name": name}
        if span_metadata:
            descriptor["metadata"] = span_metadata
        langfuse_spans.append(descriptor)
        metadata["langfuse_spans"] = langfuse_spans
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params

    @staticmethod
    def _backfill_responses_api_model_response(
        model_response: Optional[ModelResponse],
        response_body: Optional[dict],
        fallback_model: str,
    ) -> Optional[ModelResponse]:
        if model_response is None or not isinstance(response_body, dict):
            return model_response

        response_model = response_body.get("model")
        current_model = getattr(model_response, "model", None)
        if (
            not isinstance(current_model, str)
            or not current_model.strip()
            or current_model == "unknown"
        ):
            if isinstance(response_model, str) and response_model.strip():
                model_response.model = response_model
            elif isinstance(fallback_model, str) and fallback_model.strip():
                model_response.model = fallback_model

        response_usage = response_body.get("usage")
        if isinstance(response_usage, dict):
            transformed_usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                response_usage
            )
            transformed_total = getattr(transformed_usage, "total_tokens", None) or 0
            current_usage = getattr(model_response, "usage", None)
            current_total = getattr(current_usage, "total_tokens", None) or 0
            if current_usage is None or (current_total == 0 and transformed_total > 0):
                model_response.usage = transformed_usage

        response_output = response_body.get("output")
        if isinstance(response_output, list):
            hidden_params = getattr(model_response, "_hidden_params", None)
            if not isinstance(hidden_params, dict):
                hidden_params = {}
                model_response._hidden_params = hidden_params
            hidden_params["responses_output"] = response_output

        return model_response

    @staticmethod
    def _extract_responses_api_output_text(response_body: dict) -> str:
        output_text = response_body.get("output_text")
        if isinstance(output_text, str):
            return output_text

        text_parts: List[str] = []
        response_output = response_body.get("output")
        if not isinstance(response_output, list):
            return ""

        for item in response_output:
            if not isinstance(item, dict):
                continue
            item_text = item.get("text")
            if isinstance(item_text, str):
                text_parts.append(item_text)
                continue

            content = item.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                content_text = content_item.get("text")
                if isinstance(content_text, str):
                    text_parts.append(content_text)

        return "".join(text_parts)

    @staticmethod
    def _extract_responses_api_reasoning_summary_text(
        response_output: Any,
    ) -> Optional[str]:
        if not isinstance(response_output, list):
            return None

        reasoning_parts: List[str] = []
        for item in response_output:
            item_type = (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            if item_type != "reasoning":
                continue

            summary_items = (
                item.get("summary")
                if isinstance(item, dict)
                else getattr(item, "summary", None)
            )
            if not isinstance(summary_items, list):
                summary_items = []

            for summary_item in summary_items:
                summary_text = (
                    summary_item.get("text")
                    if isinstance(summary_item, dict)
                    else getattr(summary_item, "text", None)
                )
                if isinstance(summary_text, str) and summary_text:
                    reasoning_parts.append(summary_text)

            content_items = (
                item.get("content")
                if isinstance(item, dict)
                else getattr(item, "content", None)
            )
            if not isinstance(content_items, list):
                continue

            for content_item in content_items:
                content_text = (
                    content_item.get("text")
                    if isinstance(content_item, dict)
                    else getattr(content_item, "text", None)
                )
                if not isinstance(content_text, str) or not content_text:
                    content_text = (
                        content_item.get("reasoning")
                        if isinstance(content_item, dict)
                        else getattr(content_item, "reasoning", None)
                    )
                if isinstance(content_text, str) and content_text:
                    reasoning_parts.append(content_text)

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    @staticmethod
    def _build_responses_api_fallback_model_response(
        response_body: dict,
        fallback_model: str,
        assistant_content: str,
        reasoning_content: Optional[str] = None,
        responses_output: Optional[List[Any]] = None,
        raw_hidden_params: Optional[dict] = None,
    ) -> ModelResponse:
        model_response = litellm.ModelResponse()
        response_model = response_body.get("model")
        model_response.model = (
            response_model
            if isinstance(response_model, str) and response_model.strip()
            else fallback_model
        )

        model_response.choices = [
            Choices(
                message=Message(
                    role="assistant",
                    content=assistant_content,
                    reasoning_content=reasoning_content,
                ),
                finish_reason="stop",
                index=0,
            )
        ]

        response_usage = response_body.get("usage")
        if isinstance(response_usage, dict):
            model_response.usage = (
                ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                    response_usage
                )
            )

        response_id = response_body.get("id")
        if isinstance(response_id, str) and response_id:
            model_response.id = response_id
        response_created = response_body.get("created_at", response_body.get("created"))
        if isinstance(response_created, int):
            model_response.created = response_created

        hidden_params = getattr(model_response, "_hidden_params", None)
        if not isinstance(hidden_params, dict):
            hidden_params = {}
            model_response._hidden_params = hidden_params

        if isinstance(raw_hidden_params, dict) and raw_hidden_params:
            hidden_params.update(raw_hidden_params)
        if responses_output:
            hidden_params["responses_output"] = responses_output

        return model_response

    @staticmethod
    def _build_responses_api_model_response_from_body(
        response_body: dict,
        fallback_model: str,
    ) -> ModelResponse:
        response_output = response_body.get("output")
        return OpenAIPassthroughLoggingHandler._build_responses_api_fallback_model_response(
            response_body=response_body,
            fallback_model=fallback_model,
            assistant_content=OpenAIPassthroughLoggingHandler._extract_responses_api_output_text(
                response_body
            ),
            reasoning_content=OpenAIPassthroughLoggingHandler._extract_responses_api_reasoning_summary_text(
                response_output
            ),
            responses_output=response_output if isinstance(response_output, list) else None,
        )

    @staticmethod
    def _estimate_token_count_from_value(value: Any) -> int:
        if value is None:
            return 1
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                text = str(value)
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _extract_responses_api_stream_text(all_chunks: List[str]) -> str:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        text_parts: List[str] = []
        delta_keys_seen: set[str] = set()

        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue

            chunk_type = parsed_chunk.get("type")
            if chunk_type == "response.output_text.delta":
                delta = parsed_chunk.get("delta")
                if isinstance(delta, str) and delta:
                    text_parts.append(delta)
                    text_key = parsed_chunk.get("item_id")
                    if not isinstance(text_key, str) or not text_key:
                        output_index = parsed_chunk.get("output_index")
                        text_key = (
                            f"output:{output_index}"
                            if isinstance(output_index, int)
                            else "output:0"
                        )
                    delta_keys_seen.add(text_key)
                continue

            # Some OpenRouter Responses streams end with output_text.done plus
            # [DONE], but omit response.completed. Preserve that text once.
            if chunk_type == "response.output_text.done":
                text = parsed_chunk.get("text")
                if not isinstance(text, str) or not text:
                    continue
                text_key = parsed_chunk.get("item_id")
                if not isinstance(text_key, str) or not text_key:
                    output_index = parsed_chunk.get("output_index")
                    text_key = (
                        f"output:{output_index}"
                        if isinstance(output_index, int)
                        else "output:0"
                    )
                if text_key not in delta_keys_seen:
                    text_parts.append(text)

        return "".join(text_parts)

    @staticmethod
    def _extract_responses_api_incomplete_reason_from_payload(
        payload: Any,
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            reason = metadata.get("aawm_stream_incomplete_reason")
            if isinstance(reason, str):
                reason_value = reason.strip()
                if reason_value:
                    return reason_value

        reason = payload.get("aawm_stream_incomplete_reason")
        if isinstance(reason, str):
            reason_value = reason.strip()
            if reason_value:
                return reason_value

        incomplete_details = payload.get("incomplete_details")
        if not isinstance(incomplete_details, dict):
            return None

        reason = incomplete_details.get("reason")
        if isinstance(reason, str):
            reason_value = reason.strip()
            if reason_value:
                return reason_value

        return None

    @staticmethod
    def _extract_responses_api_incomplete_reason_from_stream(
        all_chunks: List[str],
    ) -> Optional[str]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue

            reason = OpenAIPassthroughLoggingHandler._extract_responses_api_incomplete_reason_from_payload(
                parsed_chunk
            )
            if reason is not None:
                return reason

            response_payload = parsed_chunk.get("response")
            if not isinstance(response_payload, dict):
                continue

            reason = OpenAIPassthroughLoggingHandler._extract_responses_api_incomplete_reason_from_payload(
                response_payload
            )
            if reason is not None:
                return reason

        return None

    @staticmethod
    def _extract_responses_api_usage_from_stream(all_chunks: List[str]) -> Optional[dict]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        observed_usage: Optional[dict] = None
        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue

            direct_usage = parsed_chunk.get("usage")
            if isinstance(direct_usage, dict):
                observed_usage = direct_usage

            response_payload = parsed_chunk.get("response")
            if not isinstance(response_payload, dict):
                continue

            response_usage = response_payload.get("usage")
            if isinstance(response_usage, dict):
                observed_usage = response_usage

        return observed_usage

    @staticmethod
    def _extract_responses_api_response_id_from_stream(all_chunks: List[str]) -> Optional[str]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue

            response_id = parsed_chunk.get("id")
            if isinstance(response_id, str) and response_id.strip():
                return response_id

            response_payload = parsed_chunk.get("response")
            if not isinstance(response_payload, dict):
                continue

            response_id = response_payload.get("id")
            if isinstance(response_id, str) and response_id.strip():
                return response_id

        return None

    @staticmethod
    def _sanitize_responses_terminal_error_for_logging(error: Any, *, limit: int = 500) -> Optional[str]:
        if error is None:
            return None
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                text_value = message.strip()
            else:
                try:
                    text_value = json.dumps(error, ensure_ascii=False, sort_keys=True)
                except (TypeError, ValueError):
                    text_value = str(error)
        elif isinstance(error, str):
            text_value = error.strip()
        else:
            text_value = str(error)
        if not text_value:
            return None
        text_value = _redact_string(text_value)
        if len(text_value) > limit:
            return text_value[: limit - 3] + "..."
        return text_value

    @staticmethod
    def _sanitize_responses_terminal_incomplete_details_for_logging(
        incomplete_details: Any, *, limit: int = 500
    ) -> Optional[str]:
        if not isinstance(incomplete_details, dict):
            return None
        reason = incomplete_details.get("reason")
        if not isinstance(reason, str):
            return None
        reason_value = reason.strip()
        if not reason_value:
            return None
        redacted = _redact_string(reason_value)
        if len(redacted) > limit:
            return redacted[: limit - 3] + "..."
        return redacted

    @staticmethod
    def _annotate_responses_terminal_hidden_params(
        model_response: ModelResponse,
        *,
        terminal_event_type: str,
        response_payload: dict,
    ) -> None:
        hidden_params = getattr(model_response, "_hidden_params", None)
        if not isinstance(hidden_params, dict):
            hidden_params = {}
            model_response._hidden_params = hidden_params
        hidden_params["responses_terminal_event_type"] = terminal_event_type
        hidden_params["responses_terminal_status"] = response_payload.get("status")
        hidden_params["responses_terminal_error"] = (
            OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                response_payload.get("error")
            )
        )
        hidden_params["responses_terminal_incomplete_details"] = (
            OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_incomplete_details_for_logging(
                response_payload.get("incomplete_details")
            )
        )

    @staticmethod
    def _responses_stream_has_terminal_event(all_chunks: List[str]) -> bool:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue
            if parsed_chunk.get("type") in RESPONSES_API_TERMINAL_STREAM_EVENTS:
                return True
        return False

    @staticmethod
    def _stream_qualifies_for_output_text_done_synthesis(all_chunks: List[str]) -> bool:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        if OpenAIPassthroughLoggingHandler._responses_stream_has_terminal_event(
            all_chunks
        ):
            return False

        saw_output_text_done = False
        saw_done_marker = False
        for chunk_str in all_chunks:
            if str(chunk_str).strip() == "data: [DONE]":
                saw_done_marker = True
                continue
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue
            if parsed_chunk.get("type") == "response.output_text.done":
                text = parsed_chunk.get("text")
                if isinstance(text, str) and text:
                    saw_output_text_done = True
        return saw_output_text_done and saw_done_marker

    @staticmethod
    def _extract_terminal_response_payload_from_stream(
        all_chunks: List[str],
    ) -> tuple[Optional[str], Optional[dict]]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        terminal_event_type: Optional[str] = None
        terminal_response_payload: Optional[dict] = None
        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue
            chunk_type = parsed_chunk.get("type")
            if chunk_type in RESPONSES_API_TERMINAL_STREAM_EVENTS:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    terminal_event_type = str(chunk_type)
                    terminal_response_payload = response_payload
        return terminal_event_type, terminal_response_payload

    @staticmethod
    def _responses_stream_standard_logging_status(
        *,
        terminal_event_type: Optional[str],
        response_payload: Optional[dict],
    ) -> Literal["success", "failure"]:
        if terminal_event_type in {
            "response.failed",
            "response.incomplete",
        }:
            return "failure"
        if isinstance(response_payload, dict):
            status = response_payload.get("status")
            if status in {"failed", "incomplete"}:
                return "failure"
        return "success"

    @staticmethod
    def _build_responses_api_terminal_model_response(
        *,
        response_body: dict,
        fallback_model: str,
        all_chunks: List[str],
        terminal_event_type: str,
        assistant_content: str = "",
    ) -> ModelResponse:
        reconstructed_output = (
            OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
                all_chunks
            )
        )
        responses_output = response_body.get("output")
        merged_output = OpenAIPassthroughLoggingHandler._merge_responses_output_lists(
            responses_output if isinstance(responses_output, list) else [],
            reconstructed_output,
        )
        model_response = (
            OpenAIPassthroughLoggingHandler._build_responses_api_fallback_model_response(
                response_body=response_body,
                fallback_model=fallback_model,
                assistant_content=assistant_content,
                reasoning_content=OpenAIPassthroughLoggingHandler._extract_responses_api_reasoning_summary_text(
                    merged_output
                ),
                responses_output=merged_output or None,
            )
        )
        OpenAIPassthroughLoggingHandler._annotate_responses_terminal_hidden_params(
            model_response,
            terminal_event_type=terminal_event_type,
            response_payload=response_body,
        )
        return model_response

    @staticmethod
    def _build_responses_api_no_terminal_incomplete_model_response(
        *,
        all_chunks: List[str],
        model: str,
    ) -> Optional[ModelResponse]:
        incomplete_reason = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_incomplete_reason_from_stream(
                all_chunks
            )
        )
        if incomplete_reason != "upstream_stream_ended_without_terminal_event":
            return None

        reconstructed_output = (
            OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
                all_chunks
            )
        )
        stream_output_text = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_stream_text(
                all_chunks
            )
        )
        synthetic_terminal_payload = {
            "object": "response",
            "status": "incomplete",
            "model": model,
            "output": reconstructed_output,
            "incomplete_details": {"reason": incomplete_reason},
        }
        terminal_response_id = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_response_id_from_stream(
                all_chunks
            )
        )
        if terminal_response_id is not None:
            synthetic_terminal_payload["id"] = terminal_response_id

        observed_usage = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_usage_from_stream(
                all_chunks
            )
        )
        if observed_usage is not None:
            synthetic_terminal_payload["usage"] = observed_usage

        return OpenAIPassthroughLoggingHandler._build_responses_api_terminal_model_response(
            response_body=synthetic_terminal_payload,
            fallback_model=model,
            all_chunks=all_chunks,
            terminal_event_type="response.incomplete",
            assistant_content=stream_output_text,
        )

    @staticmethod
    def _build_responses_api_fallback_model_response_from_stream(
        *,
        all_chunks: List[str],
        request_body: dict,
        fallback_model: str,
    ) -> ModelResponse:
        assistant_content = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_stream_text(
                all_chunks
            )
        )
        prompt_tokens = (
            OpenAIPassthroughLoggingHandler._estimate_token_count_from_value(
                request_body
            )
        )
        completion_tokens = (
            OpenAIPassthroughLoggingHandler._estimate_token_count_from_value(
                assistant_content
            )
        )
        response_body = {
            "model": fallback_model,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"text_tokens": completion_tokens},
            },
            "output": [
                {
                    "type": "message",
                    "id": "msg_openai_responses_stream_fallback",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": assistant_content,
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
        model_response = OpenAIPassthroughLoggingHandler._build_responses_api_model_response_from_body(
            response_body=response_body,
            fallback_model=fallback_model,
        )
        hidden_params = getattr(model_response, "_hidden_params", None)
        if not isinstance(hidden_params, dict):
            hidden_params = {}
            model_response._hidden_params = hidden_params
        hidden_params["openai_responses_stream_missing_completed"] = True
        hidden_params["openai_responses_stream_usage_estimated"] = True
        hidden_params["openai_responses_stream_synthesized_terminal"] = True
        hidden_params["openai_responses_stream_missing_formal_terminal_event"] = True
        return model_response

    @staticmethod
    def _response_output_stream_key(
        *,
        item: Optional[dict] = None,
        output_index: Any = None,
        item_id: Any = None,
        fallback_index: Optional[int] = None,
    ) -> str:
        if isinstance(item, dict):
            for key in ("call_id", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(item_id, str) and item_id.strip():
            return item_id.strip()
        if isinstance(output_index, int):
            return f"output:{output_index}"
        if fallback_index is not None:
            return f"fallback:{fallback_index}"
        return "fallback:0"

    @staticmethod
    def _merge_responses_output_lists(
        completed_output: Optional[List[dict]],
        streamed_output: Optional[List[dict]],
    ) -> List[dict]:
        merged_by_key: Dict[str, dict] = {}
        ordered_keys: List[str] = []

        for output_list in (streamed_output or [], completed_output or []):
            for item in output_list:
                if not isinstance(item, dict):
                    continue
                key = OpenAIPassthroughLoggingHandler._response_output_stream_key(
                    item=item,
                    fallback_index=len(ordered_keys),
                )
                if key not in ordered_keys:
                    ordered_keys.append(key)
                existing = merged_by_key.get(key, {})
                merged_item = {**existing, **item}
                if "arguments" in existing and "arguments" not in item:
                    merged_item["arguments"] = existing["arguments"]
                merged_by_key[key] = merged_item

        return [merged_by_key[key] for key in ordered_keys if key in merged_by_key]

    @staticmethod
    def _record_responses_output_item_stream_event(
        *,
        parsed_chunk: dict,
        output_items: Dict[str, dict],
        ordered_keys: List[str],
        key_aliases: Dict[str, str],
        key_by_output_index: Dict[int, str],
    ) -> None:
        item = parsed_chunk.get("item")
        if not isinstance(item, dict):
            return
        raw_key = OpenAIPassthroughLoggingHandler._response_output_stream_key(
            item=item,
            output_index=parsed_chunk.get("output_index"),
            fallback_index=len(ordered_keys),
        )
        output_index = parsed_chunk.get("output_index")
        if isinstance(output_index, int) and output_index in key_by_output_index:
            key = key_by_output_index[output_index]
        else:
            key = key_aliases.get(raw_key, raw_key)
        if key not in ordered_keys:
            ordered_keys.append(key)
        existing = output_items.get(key, {})
        merged_item = {**existing, **item}
        if "arguments" in existing and "arguments" not in item:
            merged_item["arguments"] = existing["arguments"]
        output_items[key] = merged_item
        if isinstance(output_index, int):
            key_by_output_index[output_index] = key
        for alias in (raw_key, item.get("id"), item.get("call_id")):
            if isinstance(alias, str) and alias.strip():
                key_aliases[alias.strip()] = key

    @staticmethod
    def _record_responses_arguments_stream_event(
        *,
        parsed_chunk: dict,
        output_items: Dict[str, dict],
        ordered_keys: List[str],
        key_aliases: Dict[str, str],
        key_by_output_index: Dict[int, str],
    ) -> None:
        event_type = parsed_chunk.get("type")
        item_id = parsed_chunk.get("item_id")
        output_index = parsed_chunk.get("output_index")
        raw_key = OpenAIPassthroughLoggingHandler._response_output_stream_key(
            output_index=parsed_chunk.get("output_index"),
            item_id=item_id,
            fallback_index=len(ordered_keys),
        )
        if isinstance(output_index, int) and output_index in key_by_output_index:
            key = key_by_output_index[output_index]
        else:
            key = key_aliases.get(raw_key, raw_key)
        if key not in ordered_keys:
            ordered_keys.append(key)
        existing = output_items.get(key, {})
        if not existing:
            item_type = "mcp_call" if "mcp_call" in str(event_type) else "function_call"
            existing = {
                "type": item_type,
                "id": item_id,
            }
            if item_type == "function_call" and isinstance(item_id, str) and item_id:
                existing["call_id"] = item_id
        value = parsed_chunk.get("arguments")
        if not isinstance(value, str):
            value = parsed_chunk.get("delta")
        if isinstance(value, str):
            if str(event_type).endswith(".delta"):
                existing["arguments"] = f"{existing.get('arguments', '')}{value}"
            else:
                existing["arguments"] = value
        output_items[key] = existing
        if isinstance(output_index, int):
            key_by_output_index[output_index] = key
        if isinstance(item_id, str) and item_id.strip():
            key_aliases[item_id.strip()] = key

    @staticmethod
    def _reconstruct_responses_output_items_from_stream(
        all_chunks: List[str],
    ) -> List[dict]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        output_items: Dict[str, dict] = {}
        ordered_keys: List[str] = []
        key_aliases: Dict[str, str] = {}
        key_by_output_index: Dict[int, str] = {}

        for chunk_str in all_chunks:
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue

            event_type = parsed_chunk.get("type")
            if event_type in {"response.output_item.added", "response.output_item.done"}:
                OpenAIPassthroughLoggingHandler._record_responses_output_item_stream_event(
                    parsed_chunk=parsed_chunk,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )
                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
            }:
                OpenAIPassthroughLoggingHandler._record_responses_arguments_stream_event(
                    parsed_chunk=parsed_chunk,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )

        return [output_items[key] for key in ordered_keys if key in output_items]

    @staticmethod
    def _responses_stream_tool_argument_canonical_bytes(item: dict) -> bytes:
        for key in ("arguments", "input", "action", "patch"):
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                return value.encode("utf-8")
            try:
                return json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=str,
                ).encode("utf-8")
            except (TypeError, ValueError):
                return str(value).encode("utf-8")
        return b""

    @staticmethod
    def _responses_stream_tool_argument_summary(item: dict) -> Dict[str, Any]:
        payload = (
            OpenAIPassthroughLoggingHandler._responses_stream_tool_argument_canonical_bytes(
                item
            )
        )
        if not payload:
            return {"arguments_hash": None, "arguments_size_bytes": 0}
        return {
            "arguments_hash": hashlib.sha256(payload).hexdigest(),
            "arguments_size_bytes": len(payload),
        }

    @staticmethod
    def _summarize_responses_stream_tool_state(
        all_chunks: List[str],
    ) -> Dict[str, Any]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        event_types: List[str] = []
        event_counts: Dict[str, int] = {}
        def record_event_type(event_type: Any) -> None:
            if not isinstance(event_type, str) or not event_type:
                return
            if event_type not in event_types:
                event_types.append(event_type)
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        current_sse_event_type: Optional[str] = None
        for chunk_str in all_chunks:
            stripped_line = chunk_str.strip()
            if stripped_line.startswith("event:"):
                record_event_type(stripped_line.split(":", 1)[1].strip())
                current_sse_event_type = stripped_line.split(":", 1)[1].strip()
                continue
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue
            parsed_event_type = parsed_chunk.get("type")
            if parsed_event_type != current_sse_event_type:
                record_event_type(parsed_event_type)
            current_sse_event_type = None

        tool_state: List[Dict[str, Any]] = []
        for item in OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
            all_chunks
        ):
            item_type = item.get("type")
            if item_type not in {
                "function_call",
                "local_shell_call",
                "apply_patch_call",
                "custom_tool_call",
                "mcp_call",
            }:
                continue
            tool_name = item.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                tool_name = item_type
            argument_summary = (
                OpenAIPassthroughLoggingHandler._responses_stream_tool_argument_summary(
                    item
                )
            )
            tool_state.append(
                {
                    "type": item_type,
                    "name": tool_name,
                    "call_id": item.get("call_id") or item.get("id"),
                    **argument_summary,
                }
            )

        return {
            "event_types": event_types,
            "event_counts": event_counts,
            "tool_call_count": len(tool_state),
            "tool_names": [
                item["name"]
                for item in tool_state
                if isinstance(item.get("name"), str) and item.get("name")
            ],
            "tool_state": tool_state,
        }

    @staticmethod
    def _record_responses_stream_tool_state_metadata(
        kwargs: dict,
        all_chunks: List[str],
    ) -> None:
        summary = OpenAIPassthroughLoggingHandler._summarize_responses_stream_tool_state(
            all_chunks
        )
        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        metadata["responses_stream_event_types"] = summary["event_types"]
        metadata["responses_stream_event_counts"] = summary["event_counts"]
        metadata["responses_stream_tool_call_count"] = summary["tool_call_count"]
        metadata["responses_stream_tool_names"] = summary["tool_names"]
        metadata["responses_stream_tool_state"] = summary["tool_state"]
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params

    @staticmethod
    def _extract_responses_stream_codex_token_count_event(
        all_chunks: List[str],
    ) -> Optional[Dict[str, Any]]:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        latest_token_count: Optional[Dict[str, Any]] = None
        current_sse_event_type: Optional[str] = None
        for chunk_str in all_chunks:
            stripped_line = chunk_str.strip()
            if stripped_line.startswith("event:"):
                current_sse_event_type = stripped_line.split(":", 1)[1].strip()
                continue
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                str_line=chunk_str
            )
            if not isinstance(parsed_chunk, dict):
                continue
            candidate_events = [parsed_chunk]
            payload = parsed_chunk.get("payload")
            if isinstance(payload, dict):
                candidate_events.insert(0, payload)
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            for event in candidate_events:
                rate_limits = event.get("rate_limits")
                if not isinstance(rate_limits, dict):
                    continue
                if not (
                    isinstance(rate_limits.get("primary"), dict)
                    or isinstance(rate_limits.get("secondary"), dict)
                ):
                    continue
                event_type = event.get("type")
                parent_type = parsed_chunk.get("type")
                if (
                    event_type != "token_count"
                    and parent_type != "token_count"
                    and payload_type != "token_count"
                    and current_sse_event_type != "token_count"
                ):
                    continue
                token_count_source = (
                    payload
                    if event is parsed_chunk and isinstance(payload, dict)
                    else event
                )
                latest_token_count = {
                    key: token_count_source.get(key, event.get(key))
                    for key in {
                        "type",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cache_read_input_tokens",
                        "reasoning_output_tokens",
                        "rate_limits",
                    }
                    if token_count_source.get(key, event.get(key)) is not None
                }
                latest_token_count.update(
                    {
                        key: value
                        for key, value in event.items()
                        if key == "rate_limits"
                    }
                )
                latest_token_count = {
                    key: value
                    for key, value in latest_token_count.items()
                    if key
                    in {
                        "type",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cache_read_input_tokens",
                        "reasoning_output_tokens",
                        "rate_limits",
                    }
                }
        return latest_token_count

    @staticmethod
    def _record_responses_stream_rate_limit_metadata(
        kwargs: dict,
        all_chunks: List[str],
    ) -> None:
        token_count_event = (
            OpenAIPassthroughLoggingHandler._extract_responses_stream_codex_token_count_event(
                all_chunks
            )
        )
        if not token_count_event:
            return

        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        metadata["codex_token_count"] = token_count_event
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params

    @staticmethod
    def _calculate_image_generation_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image generation."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")
            quality = request_body.get("quality", None)

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=quality,
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image generation cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def _calculate_image_editing_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image editing."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            # Image edit typically uses multipart/form-data (because of files), so all fields arrive as strings (e.g., n = "1").
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=None,  # Image editing doesn't have quality parameter
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image editing cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def openai_passthrough_handler(  # noqa: PLR0915
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle OpenAI passthrough logging with cost tracking for chat completions, image generation, image editing, and responses API.
        """
        # Check if this is a supported endpoint for cost tracking
        is_chat_completions = (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
        )
        is_image_generation = (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
        )
        is_image_editing = (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
        )
        is_responses = OpenAIPassthroughLoggingHandler.is_openai_responses_route(
            url_route
        )
        is_embeddings = OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
            url_route
        )

        if not (
            is_chat_completions
            or is_image_generation
            or is_image_editing
            or is_responses
            or is_embeddings
        ):
            # For unsupported endpoints, return None to let the system fall back to generic behavior
            return {
                "result": None,
                "kwargs": kwargs,
            }

        custom_llm_provider = kwargs.get("custom_llm_provider") or "openai"
        passthrough_model_fallback = (
            OpenAIPassthroughLoggingHandler._extract_passthrough_model_fallback(kwargs)
        )

        # Extract model from request or response. Native Grok Build embeddings
        # send an upstream-only embedding model, but the reporting/cost contract
        # is tied to the Grok Build model override.
        if (
            is_embeddings
            and custom_llm_provider == "xai"
            and passthrough_model_fallback
        ):
            model = passthrough_model_fallback
        else:
            model = (
                request_body.get("model")
                or response_body.get("model")
                or passthrough_model_fallback
                or ""
            )
        if not model:
            verbose_proxy_logger.warning(
                "No model found in request or response for OpenAI passthrough cost tracking"
            )
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

        try:
            response_cost = 0.0
            litellm_model_response: Optional[
                Union[
                    ModelResponse,
                    TextCompletionResponse,
                    ImageResponse,
                    EmbeddingResponse,
                ]
            ] = None
            handler_instance = OpenAIPassthroughLoggingHandler()

            if is_chat_completions:
                # Handle chat completions with existing logic
                provider_config = handler_instance.get_provider_config(model=model)
                # Preserve existing litellm_params to maintain metadata tags
                existing_litellm_params = kwargs.get("litellm_params", {}) or {}
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=litellm.encoding,
                    json_mode=request_body.get("response_format", {}).get("type")
                    == "json_object",
                    litellm_params=existing_litellm_params,
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = OpenAIPassthroughLoggingHandler._completion_cost_with_model_price_fallback(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            elif is_image_generation:
                # Handle image generation cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_image_editing:
                # Handle image editing cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_responses:
                # Handle responses API cost calculation
                existing_litellm_params = kwargs.get("litellm_params", {}) or {}
                if (
                    response_body.get("object") == "response"
                    or isinstance(response_body.get("output"), list)
                ):
                    litellm_model_response = (
                        handler_instance._build_responses_api_model_response_from_body(
                            response_body=response_body,
                            fallback_model=model,
                        )
                    )
                else:
                    provider_config = handler_instance.get_provider_config(model=model)
                    litellm_model_response = provider_config.transform_response(
                        raw_response=httpx_response,
                        model_response=litellm.ModelResponse(),
                        model=model,
                        messages=request_body.get("messages", []),
                        logging_obj=logging_obj,
                        optional_params=request_body.get("optional_params", {}),
                        api_key="",
                        request_data=request_body,
                        encoding=litellm.encoding,
                        json_mode=False,
                        litellm_params=existing_litellm_params,
                    )
                litellm_model_response = handler_instance._backfill_responses_api_model_response(
                    litellm_model_response,
                    response_body,
                    model,
                )

                # Calculate cost using LiteLLM's cost calculator with responses call type
                response_cost = OpenAIPassthroughLoggingHandler._completion_cost_with_model_price_fallback(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    call_type="responses",
                )
            elif is_embeddings:
                try:
                    logging_obj.call_type = "embedding"
                except Exception:
                    pass
                kwargs["call_type"] = "embedding"
                usage = OpenAIPassthroughLoggingHandler._build_embedding_usage(
                    response_body
                )
                litellm_model_response = EmbeddingResponse(
                    data=response_body.get("data", []),
                    model=model,
                    usage=usage,
                    _response_headers=dict(httpx_response.headers),
                )
                response_id = response_body.get("id")
                if isinstance(response_id, str) and response_id:
                    litellm_model_response.id = response_id
                response_cost = OpenAIPassthroughLoggingHandler._calculate_embedding_cost(
                    embedding_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )

            apply_passthrough_logging_contract(
                litellm_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                logging_obj=logging_obj,
                response_cost=response_cost,
                custom_llm_provider=custom_llm_provider,
            )

            # Create standard logging object
            if litellm_model_response is not None:
                if (
                    is_responses
                    and (
                        (kwargs.get("litellm_params", {}) or {}).get("metadata", {})
                        or {}
                    ).get("passthrough_route_family")
                    == "codex_responses"
                ):
                    usage = getattr(litellm_model_response, "usage", None)
                    handler_instance._append_langfuse_span_to_kwargs(
                        kwargs,
                        name="codex.usage_normalize",
                        span_metadata={
                            "streaming": False,
                            "call_type": "responses",
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "response_cost": response_cost,
                        },
                    )
                kwargs["standard_logging_object"] = get_standard_logging_object_payload(
                    kwargs=kwargs,
                    init_response_obj=litellm_model_response,
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    status="success",
                )

            endpoint_type = (
                "chat_completions"
                if is_chat_completions
                else "image_generation"
                if is_image_generation
                else "image_editing"
                if is_image_editing
                else "responses"
                if is_responses
                else "embeddings"
            )
            verbose_proxy_logger.debug(
                f"OpenAI passthrough cost tracking - Endpoint: {endpoint_type}, Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI passthrough cost tracking: {str(e)}"
            )
            # Fall back to base handler without cost tracking
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

    def _build_complete_streaming_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
        request_body: Optional[dict] = None,
        litellm_params: Optional[dict] = None,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw chunks for OpenAI streaming responses.

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        try:
            if self.is_openai_responses_route(url_route):
                return self._build_complete_streaming_responses_api_response(
                    all_chunks=all_chunks,
                    litellm_logging_obj=litellm_logging_obj,
                    model=model,
                    request_body=request_body or {},
                    litellm_params=litellm_params or {},
                )
            # OpenAI's response iterator to parse chunks
            from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator

            openai_iterator = OpenAIChatCompletionResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )

            all_openai_chunks = []
            for chunk_str in all_chunks:
                try:
                    # Parse the string chunk using the base iterator's string parser
                    from litellm.llms.base_llm.base_model_iterator import (
                        BaseModelResponseIterator,
                    )

                    # Convert string chunk to dict
                    stripped_json_chunk = (
                        BaseModelResponseIterator._string_to_dict_parser(
                            str_line=chunk_str
                        )
                    )

                    if stripped_json_chunk:
                        # Parse the chunk using OpenAI's chunk parser
                        transformed_chunk = openai_iterator.chunk_parser(
                            chunk=stripped_json_chunk
                        )
                        if transformed_chunk is not None:
                            all_openai_chunks.append(transformed_chunk)

                except (StopIteration, StopAsyncIteration, Exception) as e:
                    verbose_proxy_logger.debug(f"Error parsing streaming chunk: {e}")
                    continue

            if not all_openai_chunks:
                verbose_proxy_logger.warning(
                    "No valid chunks found in streaming response"
                )
                return None

            # Build complete response from chunks
            complete_streaming_response = litellm.stream_chunk_builder(
                chunks=all_openai_chunks
            )

            return complete_streaming_response

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error building complete streaming response: {str(e)}"
            )
            return None

    def _build_complete_streaming_responses_api_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        request_body: dict,
        litellm_params: dict,
    ) -> Optional[ModelResponse]:
        """
        Rebuild a complete response from Responses API streaming events.

        For native Codex passthrough we only need the final `response.completed.response`
        payload plus any `response.output_text.delta` text that was streamed before it.
        Parsing the raw completed payload directly is more reliable than replaying the
        event stream through the generic chunk transformer, which can drop usage for
        some ChatGPT/Codex stream shapes.
        """
        try:
            from litellm.types.llms.openai import ResponsesAPIResponse

            responses_transformer = LiteLLMResponsesTransformationHandler()
            streamed_output_text = (
                OpenAIPassthroughLoggingHandler._extract_responses_api_stream_text(
                    all_chunks
                )
            )
            terminal_event_type, terminal_response_payload = (
                OpenAIPassthroughLoggingHandler._extract_terminal_response_payload_from_stream(
                    all_chunks
                )
            )

            if terminal_response_payload is None:
                if OpenAIPassthroughLoggingHandler._stream_qualifies_for_output_text_done_synthesis(
                    all_chunks
                ):
                    return OpenAIPassthroughLoggingHandler._build_responses_api_fallback_model_response_from_stream(
                        all_chunks=all_chunks,
                        request_body=request_body,
                        fallback_model=model,
                    )
                no_terminal_incomplete_response = (
                    OpenAIPassthroughLoggingHandler._build_responses_api_no_terminal_incomplete_model_response(
                        all_chunks=all_chunks,
                        model=model,
                    )
                )
                if no_terminal_incomplete_response is not None:
                    return no_terminal_incomplete_response
                verbose_proxy_logger.warning(
                    "No recognized Responses terminal event found in OpenAI responses stream"
                )
                return None

            if terminal_event_type in {
                "response.failed",
                "response.incomplete",
            } or terminal_response_payload.get("status") in {"failed", "incomplete"}:
                terminal_model_response = (
                    OpenAIPassthroughLoggingHandler._build_responses_api_terminal_model_response(
                        response_body=terminal_response_payload,
                        fallback_model=model,
                        all_chunks=all_chunks,
                        terminal_event_type=terminal_event_type
                        or str(terminal_response_payload.get("status")),
                    )
                )
                return terminal_model_response

            completed_response_payload = terminal_response_payload
            responses_output = completed_response_payload.get("output")
            completed_response_model_payload = completed_response_payload
            if "output" not in completed_response_model_payload:
                completed_response_model_payload = {
                    **completed_response_payload,
                    "output": [],
                }

            completed_response = ResponsesAPIResponse(
                **completed_response_model_payload
            )
            reconstructed_output = (
                OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
                    all_chunks
                )
            )
            merged_output = OpenAIPassthroughLoggingHandler._merge_responses_output_lists(
                responses_output if isinstance(responses_output, list) else [],
                reconstructed_output,
            )
            reasoning_summary_text = OpenAIPassthroughLoggingHandler._extract_responses_api_reasoning_summary_text(
                merged_output
            )
            raw_response_hidden_params = getattr(completed_response, "_hidden_params", {})
            if len(getattr(completed_response, "output", []) or []) == 0:
                empty_fallback_response = OpenAIPassthroughLoggingHandler._build_responses_api_fallback_model_response(
                    response_body=completed_response_payload,
                    fallback_model=model,
                    assistant_content=streamed_output_text,
                    reasoning_content=reasoning_summary_text,
                    responses_output=merged_output,
                    raw_hidden_params=raw_response_hidden_params,
                )
                if terminal_event_type:
                    OpenAIPassthroughLoggingHandler._annotate_responses_terminal_hidden_params(
                        empty_fallback_response,
                        terminal_event_type=terminal_event_type,
                        response_payload=completed_response_payload,
                    )
                return empty_fallback_response

            try:
                model_response = responses_transformer.transform_response(
                    model=model,
                    raw_response=completed_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=litellm_logging_obj,
                    request_data=request_body,
                    messages=request_body.get("messages", []),
                    optional_params=request_body.get("optional_params", {}),
                    litellm_params=litellm_params,
                    encoding=litellm.encoding,
                    json_mode=False,
                )
            except ValueError as e:
                if "Unknown items in responses API response" in str(e):
                    ve_fallback_response = OpenAIPassthroughLoggingHandler._build_responses_api_fallback_model_response(
                        response_body=completed_response_payload,
                        fallback_model=model,
                        assistant_content=streamed_output_text,
                        reasoning_content=reasoning_summary_text,
                        responses_output=merged_output,
                        raw_hidden_params=raw_response_hidden_params,
                    )
                    if terminal_event_type:
                        OpenAIPassthroughLoggingHandler._annotate_responses_terminal_hidden_params(
                            ve_fallback_response,
                            terminal_event_type=terminal_event_type,
                            response_payload=completed_response_payload,
                        )
                    return ve_fallback_response
                raise
            backfilled_response = self._backfill_responses_api_model_response(
                model_response,
                completed_response_payload,
                model,
            )
            if backfilled_response is None:
                return None
            if not hasattr(backfilled_response, "_hidden_params") or not isinstance(
                getattr(backfilled_response, "_hidden_params", None), dict
            ):
                backfilled_response._hidden_params = {}
            if merged_output:
                backfilled_response._hidden_params["responses_output"] = merged_output
            if terminal_event_type:
                OpenAIPassthroughLoggingHandler._annotate_responses_terminal_hidden_params(
                    backfilled_response,
                    terminal_event_type=terminal_event_type,
                    response_payload=completed_response_payload,
                )
            return backfilled_response
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error rebuilding complete responses API stream: {str(e)}"
            )
            return None

    @staticmethod
    def _handle_logging_openai_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
        kwargs: Optional[dict] = None,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle logging for collected OpenAI streaming chunks with cost tracking.
        """
        try:
            # Extract model from request body
            model = request_body.get("model", "gpt-4o")

            # Build complete response from chunks using our streaming handler
            handler = OpenAIPassthroughLoggingHandler()
            is_responses = handler.is_openai_responses_route(url_route)
            logging_kwargs = kwargs if isinstance(kwargs, dict) else {}
            existing_litellm_params = logging_kwargs.get("litellm_params")
            if not isinstance(existing_litellm_params, dict):
                existing_litellm_params = (
                    litellm_logging_obj.model_call_details.get("litellm_params", {})
                    or {}
                )
            if not isinstance(existing_litellm_params, dict):
                existing_litellm_params = {}
            logging_kwargs["litellm_params"] = existing_litellm_params
            if is_responses:
                handler._record_responses_stream_tool_state_metadata(
                    logging_kwargs,
                    all_chunks,
                )
                handler._record_responses_stream_rate_limit_metadata(
                    logging_kwargs,
                    all_chunks,
                )
            complete_response = handler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
                url_route=url_route,
                request_body=request_body,
                litellm_params=existing_litellm_params,
            )

            if complete_response is None:
                verbose_proxy_logger.warning(
                    "Failed to build complete response from OpenAI streaming chunks"
                )
                return {
                    "result": None,
                    "kwargs": logging_kwargs,
                }

            custom_llm_provider = litellm_logging_obj.model_call_details.get(
                "custom_llm_provider", "openai"
            )
            # Calculate cost using LiteLLM's cost calculator
            response_cost = OpenAIPassthroughLoggingHandler._completion_cost_with_model_price_fallback(
                completion_response=complete_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                call_type="responses" if is_responses else None,
            )

            # Preserve existing litellm_params to maintain metadata tags
            # Prepare kwargs for logging
            logging_kwargs.update(
                {
                    "response_cost": response_cost,
                    "model": model,
                    "custom_llm_provider": custom_llm_provider,
                    "litellm_params": existing_litellm_params,
                }
            )
            passthrough_logging_payload = litellm_logging_obj.model_call_details.get(
                "passthrough_logging_payload"
            )
            if passthrough_logging_payload:
                logging_kwargs["passthrough_logging_payload"] = passthrough_logging_payload

            apply_passthrough_logging_contract(
                litellm_response=complete_response,
                model=model,
                kwargs=logging_kwargs,
                logging_obj=litellm_logging_obj,
                response_cost=response_cost,
                custom_llm_provider=custom_llm_provider,
            )

            # Create standard logging object
            if (
                is_responses
                and (
                    (logging_kwargs.get("litellm_params", {}) or {}).get(
                        "metadata", {}
                    )
                    or {}
                ).get("passthrough_route_family")
                == "codex_responses"
            ):
                usage = getattr(complete_response, "usage", None)
                handler._append_langfuse_span_to_kwargs(
                    logging_kwargs,
                    name="codex.usage_normalize",
                    span_metadata={
                        "streaming": True,
                        "call_type": "responses",
                        "total_tokens": getattr(usage, "total_tokens", None),
                        "response_cost": response_cost,
                    },
                )
            hidden_params = getattr(complete_response, "_hidden_params", None)
            terminal_event_type = None
            terminal_response_payload = None
            terminal_event_type, terminal_response_payload = (
                OpenAIPassthroughLoggingHandler._extract_terminal_response_payload_from_stream(
                    all_chunks
                )
            )
            if terminal_event_type is None and isinstance(hidden_params, dict):
                terminal_event_type = hidden_params.get(
                    "responses_terminal_event_type"
                )
            standard_logging_status = (
                OpenAIPassthroughLoggingHandler._responses_stream_standard_logging_status(
                    terminal_event_type=terminal_event_type,
                    response_payload=terminal_response_payload,
                )
            )
            # Compute sanitized error for failure terminals so it flows through
            # the standard logging payload constructor (error_str param) and
            # appears in the canonical schema.
            error_str_for_logging = None
            if standard_logging_status == "failure":
                if isinstance(terminal_response_payload, dict):
                    error_str_for_logging = (
                        OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                            terminal_response_payload.get("error")
                        )
                    )
                if not error_str_for_logging and isinstance(hidden_params, dict):
                    error_str_for_logging = hidden_params.get("responses_terminal_error")

            logging_kwargs["standard_logging_object"] = get_standard_logging_object_payload(
                kwargs=logging_kwargs,
                init_response_obj=complete_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
                status=standard_logging_status,
                error_str=error_str_for_logging,
            )

            verbose_proxy_logger.debug(
                f"OpenAI streaming passthrough cost tracking - Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": complete_response,
                "kwargs": logging_kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI streaming passthrough cost tracking: {str(e)}"
            )
            return {
                "result": None,
                "kwargs": {},
            }
