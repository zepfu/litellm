#### What this does ####
#    On success, logs events to Langfuse
import hashlib
import json
import os
import traceback
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from packaging.version import Version

import litellm
from litellm._logging import record_langfuse_enqueue_size_audit, verbose_logger
from litellm.constants import MAX_LANGFUSE_INITIALIZED_CLIENTS
from litellm.litellm_core_utils.core_helpers import (
    safe_deep_copy,
    reconstruct_model_name,
    filter_exceptions_from_params,
)
from litellm.litellm_core_utils.redact_messages import redact_user_api_key_info
from litellm.integrations.langfuse.langfuse_mock_client import (
    create_mock_langfuse_client,
    should_use_langfuse_mock,
)
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.langfuse import *
from litellm.types.llms.openai import HttpxBinaryResponseContent, ResponsesAPIResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    RerankResponse,
    StandardLoggingPayload,
    StandardLoggingPromptManagementMetadata,
    TextCompletionResponse,
    TranscriptionResponse,
)

if TYPE_CHECKING:
    from langfuse.client import Langfuse, StatefulTraceClient

    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
else:
    DynamicLoggingCache = Any
    StatefulTraceClient = Any
    Langfuse = Any


_LANGFUSE_DEFAULT_MAX_EVENT_SIZE_BYTES = 1_000_000
_LANGFUSE_SIZE_AUDIT_THRESHOLD_RATIO = 0.9
_LANGFUSE_SIZE_AUDIT_METADATA_KEY_LIMIT = 10
_LANGFUSE_EVENT_FIT_TARGET_RATIO = 0.95
_LANGFUSE_INPUT_TRUNCATION_MARKER_TYPE = "litellm_langfuse_input_truncated"
_LANGFUSE_INPUT_SUMMARY_TYPE = "litellm_langfuse_input_summary"
_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV = "AAWM_LANGFUSE_INPUT_SHAPE_HASH_ONLY"
_LANGFUSE_FIELD_TRUNCATION_MARKER_TYPE = "litellm_langfuse_field_truncated"
_LANGFUSE_FIELD_OMISSION_MARKER_TYPE = "litellm_langfuse_field_omitted"
_LANGFUSE_FIELD_FIT_PRIORITY = (
    "input",
    "output",
    "metadata",
    "model_parameters",
    "status_message",
    "prompt",
)
_LANGFUSE_FIELD_FIT_EXCLUDED_KEYS = frozenset(
    {
        "id",
        "name",
        "start_time",
        "end_time",
        "model",
        "usage",
        "usage_details",
        "cost_details",
        "level",
        "version",
        "completion_start_time",
        "parent_observation_id",
    }
)
_AAWM_TOOL_DEFINITION_METADATA_SNAPSHOT_KEY = "aawm_tool_definition_snapshot"
_AAWM_LANGFUSE_METADATA_COMPACTED_TYPE = "litellm_langfuse_metadata_compacted"
_AAWM_LANGFUSE_METADATA_PATH_SAMPLE_LIMIT = 5
_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT = 20
_AAWM_LANGFUSE_METADATA_STRING_LIMIT = 200
_AAWM_LANGFUSE_INPUT_SHAPE_SAMPLE_LIMIT = 2
_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT = 1
_AAWM_LANGFUSE_INPUT_SHAPE_STRING_LIMIT = 120
_SENSITIVE_METADATA_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "password",
    "secret",
    "token",
)
_LANGFUSE_INPUT_SENSITIVE_VALUE_KEYS = frozenset(
    {
        "content",
        "text",
        "instructions",
        "prompt",
        "arguments",
        "input",
        "source",
        "snippet",
        "file",
        "file_content",
        "path",
        "local_path",
        "headers",
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "token",
        "secret",
    }
)

_LANGFUSE_INPUT_IDENTIFIER_VALUE_KEYS = frozenset(
    {
        "id",
        "call_id",
        "tool_call_id",
        "name",
    }
)
_LANGFUSE_INPUT_LOW_CARDINALITY_VALUE_KEYS = frozenset(
    {
        "role",
        "type",
    }
)


def _get_langfuse_max_event_size_bytes() -> int:
    raw_limit = os.environ.get("LANGFUSE_MAX_EVENT_SIZE_BYTES")
    if raw_limit is None:
        return _LANGFUSE_DEFAULT_MAX_EVENT_SIZE_BYTES
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        return _LANGFUSE_DEFAULT_MAX_EVENT_SIZE_BYTES
    return parsed_limit if parsed_limit > 0 else _LANGFUSE_DEFAULT_MAX_EVENT_SIZE_BYTES


def _langfuse_input_shape_hash_only_enabled() -> bool:
    raw_value = os.environ.get(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV)
    if raw_value is None:
        return False
    parsed = str_to_bool(raw_value)
    if parsed is not None:
        return parsed
    return raw_value.strip().lower() in {"1", "yes", "on"}


def _langfuse_input_shape_string(value: Any) -> str:
    value_text = str(value)
    if len(value_text) <= _AAWM_LANGFUSE_INPUT_SHAPE_STRING_LIMIT:
        return value_text
    return value_text[: _AAWM_LANGFUSE_INPUT_SHAPE_STRING_LIMIT - 3] + "..."


def _langfuse_input_shape_short_hash(value: Any) -> str:
    return _stable_langfuse_metadata_hash(value)[:16]


def _langfuse_input_shape_key_descriptor(key: Any, *, key_index: int) -> Dict[str, Any]:
    key_text = str(key)
    lowered_key = key_text.lower()
    category = "sensitive" if any(
        fragment in lowered_key for fragment in _SENSITIVE_METADATA_KEY_FRAGMENTS
    ) else "standard"
    return {
        "key_index": key_index,
        "key_hash": _langfuse_input_shape_short_hash(key_text),
        "key_length": len(key_text),
        "category": category,
    }


def _langfuse_input_shape_identifier_value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool"}
    if isinstance(value, int):
        return {"type": "int", "magnitude_bytes": _json_size_bytes(value)}
    if isinstance(value, float):
        return {"type": "float", "magnitude_bytes": _json_size_bytes(value)}
    if isinstance(value, str):
        return {
            "type": "string",
            "length": len(value),
            "hash": _langfuse_input_shape_short_hash(value),
        }
    return {
        "type": type(value).__name__,
        "hash": _langfuse_input_shape_short_hash(value),
    }


def _langfuse_input_shape_non_primitive_scalar(value: Any) -> Dict[str, Any]:
    return {
        "type": type(value).__name__,
        "hash": _langfuse_input_shape_short_hash(value),
    }


def _langfuse_input_shape_scalar(
    value: Any,
    *,
    redact_preview: bool = False,
    identifier_like: bool = False,
) -> Any:
    if identifier_like:
        return _langfuse_input_shape_identifier_value(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        shaped = {
            "type": "string",
            "length": len(value),
        }
        if not redact_preview:
            shaped["hash"] = _langfuse_input_shape_short_hash(value)
            shaped["preview"] = _langfuse_input_shape_string(value)
        else:
            shaped["hash"] = _langfuse_input_shape_short_hash(value)
        return shaped
    if isinstance(value, list):
        return {
            "type": "list",
            "item_count": len(value),
            "sample_item_shapes": [
                _langfuse_input_shape_value(item, depth=1)
                for item in value[:_AAWM_LANGFUSE_INPUT_SHAPE_SAMPLE_LIMIT]
            ],
        }
    if isinstance(value, dict):
        return _langfuse_input_shape_value(value)
    return _langfuse_input_shape_non_primitive_scalar(value)


def _langfuse_input_shape_entry_value(key: Any, item_value: Any) -> Any:
    key_text = str(key)
    lowered_key = key_text.lower()
    if any(fragment in lowered_key for fragment in _SENSITIVE_METADATA_KEY_FRAGMENTS):
        return {"type": "redacted"}
    if lowered_key in _LANGFUSE_INPUT_IDENTIFIER_VALUE_KEYS:
        return _langfuse_input_shape_identifier_value(item_value)
    if lowered_key in _LANGFUSE_INPUT_SENSITIVE_VALUE_KEYS:
        return {
            "type": "redacted",
            "container_type": _langfuse_input_container_type(item_value),
            "item_count": _langfuse_input_item_count(item_value),
            "hash": _langfuse_input_shape_short_hash(item_value),
        }
    redact_preview = lowered_key not in _LANGFUSE_INPUT_LOW_CARDINALITY_VALUE_KEYS
    if isinstance(item_value, dict):
        return {
            "type": "dict",
            "key_count": len(item_value),
            "hash": _langfuse_input_shape_short_hash(item_value),
        }
    if isinstance(item_value, list):
        return {
            "type": "list",
            "item_count": len(item_value),
            "hash": _langfuse_input_shape_short_hash(item_value),
        }
    return _langfuse_input_shape_scalar(
        item_value,
        redact_preview=redact_preview,
    )


def _langfuse_input_shape_compact_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return {"type": type(value).__name__ if value is not None else "null"}
    if isinstance(value, str):
        return {
            "type": "string",
            "length": len(value),
            "hash": _langfuse_input_shape_short_hash(value),
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "item_count": len(value),
            "hash": _langfuse_input_shape_short_hash(value),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "key_count": len(value),
            "hash": _langfuse_input_shape_short_hash(value),
        }
    return _langfuse_input_shape_non_primitive_scalar(value)


def _langfuse_input_shape_conversation_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return _langfuse_input_shape_compact_value(item)

    entries: List[Dict[str, Any]] = []
    for key_index, (key, item_value) in enumerate(item.items()):
        key_text = str(key)
        lowered_key = key_text.lower()
        if lowered_key in _LANGFUSE_INPUT_LOW_CARDINALITY_VALUE_KEYS:
            value_shape: Any = _langfuse_input_shape_compact_value(item_value)
        elif lowered_key in _LANGFUSE_INPUT_IDENTIFIER_VALUE_KEYS:
            value_shape = _langfuse_input_shape_identifier_value(item_value)
        elif lowered_key in _LANGFUSE_INPUT_SENSITIVE_VALUE_KEYS or lowered_key == "content":
            value_shape = {
                "type": "redacted",
                "container_type": _langfuse_input_container_type(item_value),
                "item_count": _langfuse_input_item_count(item_value),
                "content_block_type_counts": _langfuse_input_content_block_type_counts(
                    item_value
                ),
            }
        else:
            value_shape = _langfuse_input_shape_compact_value(item_value)
        entries.append(
            {
                "key_descriptor": _langfuse_input_shape_key_descriptor(
                    key,
                    key_index=key_index,
                ),
                "value_shape": value_shape,
            }
        )
    return {
        "type": "dict",
        "key_count": len(item),
        "sample_entries": entries[:_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT],
    }


def _langfuse_input_shape_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 1:
        if isinstance(value, dict):
            return _langfuse_input_shape_conversation_item(value)
        return _langfuse_input_shape_compact_value(value)
    if isinstance(value, dict):
        entries: List[Dict[str, Any]] = []
        for key_index, (key, item_value) in enumerate(value.items()):
            entries.append(
                {
                    "key_descriptor": _langfuse_input_shape_key_descriptor(
                        key,
                        key_index=key_index,
                    ),
                    "value_shape": _langfuse_input_shape_entry_value(key, item_value),
                }
            )
        return {
            "type": "dict",
            "key_count": len(value),
            "sample_entries": entries[:_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT],
        }
    return _langfuse_input_shape_scalar(value, redact_preview=True)


def _langfuse_input_container_type(value: Any) -> str:
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _langfuse_input_item_count(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if value is None:
        return 0
    return 1


def _langfuse_input_role_counts(value: Any) -> Dict[str, int]:
    role_counts: Dict[str, int] = {}
    if not isinstance(value, list):
        return role_counts
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role is None:
            continue
        role_name = str(role)
        role_counts[role_name] = role_counts.get(role_name, 0) + 1
    return role_counts


def _langfuse_input_content_block_type_counts(value: Any) -> Dict[str, int]:
    block_counts: Dict[str, int] = {}

    def _record_block_type(block_type: Any) -> None:
        if block_type is None:
            return
        block_name = str(block_type)
        block_counts[block_name] = block_counts.get(block_name, 0) + 1

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            if "type" in item:
                _record_block_type(item.get("type"))
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _record_block_type(block.get("type"))
            elif isinstance(content, dict):
                _record_block_type(content.get("type"))
    elif isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    _record_block_type(block.get("type"))
        elif isinstance(content, dict):
            _record_block_type(content.get("type"))
    return block_counts


def _langfuse_input_shape_samples(value: Any) -> Tuple[List[Any], List[Any]]:
    if isinstance(value, str):
        return [], []
    if isinstance(value, list):
        if not value:
            return [], []
        head = [
            _langfuse_input_shape_conversation_item(item)
            for item in value[:_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT]
        ]
        tail = [
            _langfuse_input_shape_conversation_item(item)
            for item in value[-_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT :]
        ]
        return head, tail
    if isinstance(value, dict):
        entries = list(value.items())
        head_entries = entries[:_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT]
        tail_entries = entries[-_AAWM_LANGFUSE_INPUT_SHAPE_ROOT_SAMPLE_LIMIT :]
        head = [
            {
                "key_descriptor": _langfuse_input_shape_key_descriptor(
                    key,
                    key_index=index,
                ),
                "value_shape": _langfuse_input_shape_entry_value(key, item_value),
            }
            for index, (key, item_value) in enumerate(head_entries)
        ]
        tail = [
            {
                "key_descriptor": _langfuse_input_shape_key_descriptor(
                    key,
                    key_index=len(entries) - len(tail_entries) + offset,
                ),
                "value_shape": _langfuse_input_shape_entry_value(key, item_value),
            }
            for offset, (key, item_value) in enumerate(tail_entries)
        ]
        return head, tail
    return [_langfuse_input_shape_scalar(value, redact_preview=True)], []


def _langfuse_input_reconstruction_status(
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cold_storage_object_key_present = False
    if isinstance(metadata, dict):
        cold_storage_object_key_present = bool(
            metadata.get("cold_storage_object_key")
        )

    full_payload_capture_required = False
    try:
        from litellm.integrations.aawm_passthrough_shape_capture import (
            passthrough_full_payload_capture_enabled,
        )

        full_payload_capture_required = passthrough_full_payload_capture_enabled()
    except Exception:
        full_payload_capture_required = False

    if cold_storage_object_key_present:
        source = "cold_storage_object_key"
    elif full_payload_capture_required:
        source = "full_payload_capture_required"
    else:
        source = "not_available_by_default"

    return {
        "source": source,
        "cold_storage_object_key_present": cold_storage_object_key_present,
        "full_payload_capture_required": full_payload_capture_required,
    }


def _build_langfuse_input_shape_hash_summary(
    input_value: Any,
    *,
    original_input_size_bytes: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item_count = _langfuse_input_item_count(input_value)
    head, tail = _langfuse_input_shape_samples(input_value)
    omitted_items = max(0, item_count - len(head) - len(tail))
    summary = {
        "type": _LANGFUSE_INPUT_SUMMARY_TYPE,
        "hash": _stable_langfuse_metadata_hash(input_value),
        "original_size_bytes": original_input_size_bytes,
        "container_type": _langfuse_input_container_type(input_value),
        "item_count": item_count,
        "role_counts": _langfuse_input_role_counts(input_value),
        "content_block_type_counts": _langfuse_input_content_block_type_counts(
            input_value
        ),
        "head": head,
        "tail": tail,
        "omitted_items": omitted_items,
        "omitted_bytes_estimate": max(0, original_input_size_bytes),
        "raw_reconstruction": _langfuse_input_reconstruction_status(metadata),
    }
    summary["final_size_bytes"] = _json_size_bytes(summary)
    return summary


def _json_size_bytes(value: Any) -> int:
    try:
        serialized_value = json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        serialized_value = f"<unserializable:{type(value).__name__}>"
    return len(serialized_value.encode("utf-8"))


def _stable_langfuse_metadata_hash(value: Any) -> str:
    try:
        serialized_value = json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        serialized_value = f"<unserializable:{type(value).__name__}>"
    return hashlib.sha256(serialized_value.encode("utf-8")).hexdigest()


def _bounded_langfuse_metadata_string(value: Any) -> str:
    value_text = str(value)
    if len(value_text) <= _AAWM_LANGFUSE_METADATA_STRING_LIMIT:
        return value_text
    return value_text[: _AAWM_LANGFUSE_METADATA_STRING_LIMIT - 3] + "..."


def _base_langfuse_metadata_compaction_summary(
    *, field_name: str, value: Any
) -> Dict[str, Any]:
    return {
        "type": _AAWM_LANGFUSE_METADATA_COMPACTED_TYPE,
        "field": field_name,
        "hash": _stable_langfuse_metadata_hash(value),
        "original_size_bytes": _json_size_bytes(value),
    }


def _compact_langfuse_path_list_metadata(
    *,
    field_name: str,
    value: Any,
) -> Dict[str, Any]:
    summary = _base_langfuse_metadata_compaction_summary(
        field_name=field_name,
        value=value,
    )
    if isinstance(value, dict):
        bucket_counts: Dict[str, int] = {}
        sample_paths_by_bucket: Dict[str, List[str]] = {}
        total_count = 0
        for bucket, bucket_value in value.items():
            bucket_name = _bounded_langfuse_metadata_string(bucket)
            path_values = bucket_value if isinstance(bucket_value, list) else [bucket_value]
            paths = [
                _bounded_langfuse_metadata_string(path)
                for path in path_values
                if path is not None
            ]
            bucket_counts[bucket_name] = len(paths)
            total_count += len(paths)
            if paths:
                sample_paths_by_bucket[bucket_name] = paths[
                    :_AAWM_LANGFUSE_METADATA_PATH_SAMPLE_LIMIT
                ]
        summary.update(
            {
                "count": total_count,
                "bucket_counts": bucket_counts,
                "sample_paths": sample_paths_by_bucket,
            }
        )
        return summary

    path_values = value if isinstance(value, list) else [value]
    paths = [
        _bounded_langfuse_metadata_string(path)
        for path in path_values
        if path is not None
    ]
    summary.update(
        {
            "count": len(paths),
            "sample_paths": paths[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
        }
    )
    return summary


def _compact_langfuse_codex_response_headers(value: Any) -> Dict[str, Any]:
    summary = _base_langfuse_metadata_compaction_summary(
        field_name="codex_response_headers",
        value=value,
    )
    if not isinstance(value, dict):
        summary.update({"header_count": 0, "source": None})
        return summary

    header_names = sorted(str(key).lower() for key in value.keys() if key != "source")
    rate_limit_header_names = [
        name
        for name in header_names
        if name.startswith("x-codex-") or name.startswith("x-ratelimit-")
    ]
    request_id_present = any("request-id" in name for name in header_names)
    summary.update(
        {
            "source": value.get("source"),
            "header_count": len(header_names),
            "header_names": header_names[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
            "rate_limit_header_names": rate_limit_header_names[
                :_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT
            ],
            "request_id_present": request_id_present,
        }
    )
    return summary


def _compact_langfuse_responses_stream_tool_state(value: Any) -> Dict[str, Any]:
    summary = _base_langfuse_metadata_compaction_summary(
        field_name="responses_stream_tool_state",
        value=value,
    )
    tool_entries = value if isinstance(value, list) else []
    tool_names: List[str] = []
    type_counts: Dict[str, int] = {}
    sample_tool_calls: List[Dict[str, Any]] = []
    for item in tool_entries:
        if not isinstance(item, dict):
            continue
        item_type = _bounded_langfuse_metadata_string(item.get("type") or "unknown")
        item_name = _bounded_langfuse_metadata_string(item.get("name") or item_type)
        if item_name not in tool_names:
            tool_names.append(item_name)
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
        if len(sample_tool_calls) >= _AAWM_LANGFUSE_METADATA_PATH_SAMPLE_LIMIT:
            continue
        argument_value = item.get("arguments")
        argument_hash = item.get("arguments_hash")
        if argument_hash is None and argument_value is not None:
            argument_hash = _stable_langfuse_metadata_hash(argument_value)
        argument_size_bytes = item.get("arguments_size_bytes")
        if argument_size_bytes is None and argument_value is not None:
            argument_size_bytes = _json_size_bytes(argument_value)
        sample_entry = {
            "type": item_type,
            "name": item_name,
            "call_id": _bounded_langfuse_metadata_string(
                item.get("call_id") or item.get("id") or ""
            ),
        }
        if argument_hash is not None:
            sample_entry["arguments_hash"] = argument_hash
        if argument_size_bytes is not None:
            sample_entry["arguments_size_bytes"] = argument_size_bytes
        sample_tool_calls.append(sample_entry)

    summary.update(
        {
            "tool_call_count": len(tool_entries),
            "tool_names": tool_names[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
            "tool_type_counts": type_counts,
            "sample_tool_calls": sample_tool_calls,
        }
    )
    return summary


def _compact_langfuse_claude_tool_compaction_events(value: Any) -> Dict[str, Any]:
    summary = _base_langfuse_metadata_compaction_summary(
        field_name="claude_tool_advertisement_compaction_events",
        value=value,
    )
    events = value if isinstance(value, list) else []
    tool_names: List[str] = []
    statuses: List[str] = []
    cc_versions: List[str] = []
    total_original_chars = 0
    total_compacted_chars = 0
    total_saved_chars = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        tool_name = event.get("tool_name") or event.get("name")
        if tool_name is not None:
            bounded_tool_name = _bounded_langfuse_metadata_string(tool_name)
            if bounded_tool_name not in tool_names:
                tool_names.append(bounded_tool_name)
        status = event.get("status")
        if status is not None:
            bounded_status = _bounded_langfuse_metadata_string(status)
            if bounded_status not in statuses:
                statuses.append(bounded_status)
        cc_version = event.get("cc_version") or event.get("claude_code_version")
        if cc_version is not None:
            bounded_cc_version = _bounded_langfuse_metadata_string(cc_version)
            if bounded_cc_version not in cc_versions:
                cc_versions.append(bounded_cc_version)
        for source_key, total_key in (
            ("original_chars", "total_original_chars"),
            ("compacted_chars", "total_compacted_chars"),
            ("saved_chars", "total_saved_chars"),
        ):
            try:
                value_int = int(event.get(source_key) or 0)
            except (TypeError, ValueError):
                value_int = 0
            if total_key == "total_original_chars":
                total_original_chars += value_int
            elif total_key == "total_compacted_chars":
                total_compacted_chars += value_int
            else:
                total_saved_chars += value_int

    summary.update(
        {
            "count": len(events),
            "tool_names": tool_names[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
            "statuses": statuses[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
            "cc_versions": cc_versions[:_AAWM_LANGFUSE_METADATA_VALUE_SAMPLE_LIMIT],
            "total_original_chars": total_original_chars,
            "total_compacted_chars": total_compacted_chars,
            "total_saved_chars": total_saved_chars,
        }
    )
    return summary


_LANGFUSE_METADATA_COMPACTOR_FIELD_NAMES = (
    "prompt_overhead_component_paths",
    "prompt_overhead_excluded_component_paths",
    "codex_response_headers",
    "responses_stream_tool_state",
    "claude_tool_advertisement_compaction_events",
)
_LANGFUSE_COMPACTION_SAVINGS_AUDIT_ENTRY_LIMIT = 32


def _langfuse_compaction_saved_ratio(
    *, original_size_bytes: int, final_size_bytes: int
) -> Optional[float]:
    if original_size_bytes <= 0:
        return None
    saved_bytes = max(0, original_size_bytes - final_size_bytes)
    if saved_bytes <= 0:
        return 0.0
    return round(saved_bytes / original_size_bytes, 4)


def _langfuse_compaction_savings_entry(
    *,
    family: str,
    field: str,
    original_size_bytes: int,
    final_size_bytes: int,
    classification: str,
    mode: str,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    saved_bytes = max(0, original_size_bytes - final_size_bytes)
    entry: Dict[str, Any] = {
        "family": family,
        "field": field,
        "original_size_bytes": original_size_bytes,
        "final_size_bytes": final_size_bytes,
        "saved_bytes": saved_bytes,
        "saved_ratio": _langfuse_compaction_saved_ratio(
            original_size_bytes=original_size_bytes,
            final_size_bytes=final_size_bytes,
        ),
        "mode": mode,
        "classification": classification,
    }
    if strategy is not None:
        entry["strategy"] = strategy
    return entry


def _langfuse_metadata_compactors() -> Dict[str, Callable[[Any], Any]]:
    return {
        "prompt_overhead_component_paths": lambda value: _compact_langfuse_path_list_metadata(
            field_name="prompt_overhead_component_paths",
            value=value,
        ),
        "prompt_overhead_excluded_component_paths": lambda value: _compact_langfuse_path_list_metadata(
            field_name="prompt_overhead_excluded_component_paths",
            value=value,
        ),
        "codex_response_headers": _compact_langfuse_codex_response_headers,
        "responses_stream_tool_state": _compact_langfuse_responses_stream_tool_state,
        "claude_tool_advertisement_compaction_events": _compact_langfuse_claude_tool_compaction_events,
    }


def _is_langfuse_metadata_compacted_summary(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == _AAWM_LANGFUSE_METADATA_COMPACTED_TYPE
    )


def _compact_langfuse_generation_metadata_for_enqueue(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    compacted_metadata = dict(metadata)
    compacted_metadata.pop(_AAWM_TOOL_DEFINITION_METADATA_SNAPSHOT_KEY, None)

    compactors = _langfuse_metadata_compactors()
    for metadata_key, compactor in compactors.items():
        if metadata_key in compacted_metadata:
            if _is_langfuse_metadata_compacted_summary(compacted_metadata[metadata_key]):
                continue
            compacted_metadata[metadata_key] = compactor(
                compacted_metadata[metadata_key]
            )

    return compacted_metadata


def _langfuse_metadata_compaction_field_entry(
    *,
    field_name: str,
    original_value: Any,
    compactor: Callable[[Any], Any],
) -> Dict[str, Any]:
    if _is_langfuse_metadata_compacted_summary(original_value):
        raw_original_size = original_value.get("original_size_bytes")
        original_size_bytes = (
            int(raw_original_size)
            if isinstance(raw_original_size, (int, float))
            else _json_size_bytes(original_value)
        )
        final_size_bytes = _json_size_bytes(original_value)
        classification = (
            "already_handled"
            if final_size_bytes < original_size_bytes
            else "no_op"
        )
        return _langfuse_compaction_savings_entry(
            family=f"metadata.{field_name}",
            field=field_name,
            original_size_bytes=original_size_bytes,
            final_size_bytes=final_size_bytes,
            classification=classification,
            mode="metadata_compaction",
            strategy=str(original_value.get("type")),
        )

    compacted_value = compactor(original_value)
    original_size_bytes = _json_size_bytes(original_value)
    final_size_bytes = _json_size_bytes(compacted_value)
    if final_size_bytes >= original_size_bytes:
        classification = "no_op"
        strategy = "unchanged"
    else:
        classification = "already_handled"
        strategy = (
            compacted_value.get("type")
            if isinstance(compacted_value, dict)
            else _AAWM_LANGFUSE_METADATA_COMPACTED_TYPE
        )
    return _langfuse_compaction_savings_entry(
        family=f"metadata.{field_name}",
        field=field_name,
        original_size_bytes=original_size_bytes,
        final_size_bytes=final_size_bytes,
        classification=classification,
        mode="metadata_compaction",
        strategy=strategy,
    )


def _langfuse_compaction_audit_identifiers(
    generation_params: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = generation_params.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    identifiers: Dict[str, Any] = {
        "generation_id": _langfuse_summary_identifier(generation_params.get("id")),
        "generation_name": _langfuse_summary_identifier(generation_params.get("name")),
        "model": _langfuse_summary_identifier(generation_params.get("model")),
    }
    for metadata_key in (
        "custom_llm_provider",
        "user_api_key_alias",
        "user_api_key_team_alias",
        "route_family",
        "model_alias",
        "litellm_model_name",
    ):
        value = metadata_dict.get(metadata_key)
        if value is not None and value != "":
            identifiers[metadata_key] = _langfuse_summary_identifier(value)
    return identifiers


def _build_langfuse_compaction_savings_audit(  # noqa: PLR0915
    generation_params: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    call_type: Optional[str] = None,
    fit_summary: Optional[Dict[str, Any]] = None,
    max_event_size_bytes: Optional[int] = None,
    input_shape_hash_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return bounded compaction savings evidence without raw payload values."""

    original_metadata = generation_params.get("metadata")
    original_metadata_dict = (
        original_metadata if isinstance(original_metadata, dict) else {}
    )
    compacted_metadata = _compact_langfuse_generation_metadata_for_enqueue(
        original_metadata_dict
    )
    compacted_params = {
        **generation_params,
        "metadata": compacted_metadata,
    }

    fitted_params = compacted_params
    resolved_fit_summary = fit_summary
    if resolved_fit_summary is None:
        fitted_params, resolved_fit_summary = _fit_langfuse_generation_params_to_event_size(
            compacted_params,
            max_event_size_bytes=max_event_size_bytes,
            input_shape_hash_only=input_shape_hash_only,
        )

    entries: List[Dict[str, Any]] = []
    classification_counts = {
        "already_handled": 0,
        "remaining_candidate": 0,
        "no_op": 0,
    }
    totals = {
        "already_handled_saved_bytes": 0,
        "remaining_candidate_saved_bytes": 0,
    }
    entry_count = 0

    def append_entry(entry: Dict[str, Any]) -> None:
        nonlocal entry_count
        entry_count += 1
        classification = str(entry.get("classification"))
        if classification in classification_counts:
            classification_counts[classification] += 1
        saved_bytes = int(entry.get("saved_bytes") or 0)
        if classification == "already_handled":
            totals["already_handled_saved_bytes"] += saved_bytes
        elif classification == "remaining_candidate":
            totals["remaining_candidate_saved_bytes"] += saved_bytes
        if len(entries) >= _LANGFUSE_COMPACTION_SAVINGS_AUDIT_ENTRY_LIMIT:
            return
        entries.append(entry)

    original_metadata_size = _json_size_bytes(original_metadata_dict)
    compacted_metadata_size = _json_size_bytes(compacted_metadata)
    metadata_classification = (
        "already_handled"
        if compacted_metadata_size < original_metadata_size
        else "no_op"
    )
    append_entry(
        _langfuse_compaction_savings_entry(
            family="metadata",
            field="metadata",
            original_size_bytes=original_metadata_size,
            final_size_bytes=compacted_metadata_size,
            classification=metadata_classification,
            mode="metadata_compaction",
            strategy=(
                _AAWM_LANGFUSE_METADATA_COMPACTED_TYPE
                if metadata_classification == "already_handled"
                else "unchanged"
            ),
        )
    )

    compactors = _langfuse_metadata_compactors()
    for field_name in _LANGFUSE_METADATA_COMPACTOR_FIELD_NAMES:
        if field_name not in original_metadata_dict:
            continue
        append_entry(
            _langfuse_metadata_compaction_field_entry(
                field_name=field_name,
                original_value=original_metadata_dict[field_name],
                compactor=compactors[field_name],
            )
        )

    handled_metadata_fields = set(_LANGFUSE_METADATA_COMPACTOR_FIELD_NAMES)
    for field_name, field_value in original_metadata_dict.items():
        if field_name in handled_metadata_fields:
            continue
        original_size_bytes = _json_size_bytes(field_value)
        final_size_bytes = _json_size_bytes(compacted_metadata.get(field_name))
        if final_size_bytes != original_size_bytes:
            continue
        append_entry(
            _langfuse_compaction_savings_entry(
                family=f"metadata.{field_name}",
                field=field_name,
                original_size_bytes=original_size_bytes,
                final_size_bytes=final_size_bytes,
                classification="no_op",
                mode="metadata_passthrough",
                strategy="unchanged",
            )
        )

    field_truncations = (
        resolved_fit_summary.get("field_truncations", [])
        if isinstance(resolved_fit_summary, dict)
        else []
    )
    for field_summary in field_truncations:
        if not isinstance(field_summary, dict):
            continue
        field_name = str(field_summary.get("field") or "")
        if not field_name:
            continue
        original_size_bytes = int(field_summary.get("original_size_bytes") or 0)
        final_size_bytes = int(field_summary.get("final_size_bytes") or 0)
        strategy = str(field_summary.get("strategy") or "truncated")
        if field_name == "input":
            classification = "remaining_candidate"
            mode = "event_size_fit"
        elif field_name in {"metadata", "model_parameters"}:
            classification = "remaining_candidate"
            mode = "event_size_fit"
        else:
            classification = (
                "remaining_candidate"
                if final_size_bytes < original_size_bytes
                else "no_op"
            )
            mode = "event_size_fit"
        append_entry(
            _langfuse_compaction_savings_entry(
                family=field_name,
                field=field_name,
                original_size_bytes=original_size_bytes,
                final_size_bytes=final_size_bytes,
                classification=classification,
                mode=mode,
                strategy=strategy,
            )
        )

    if not field_truncations:
        original_input_size = _json_size_bytes(generation_params.get("input"))
        final_input_size = _json_size_bytes(fitted_params.get("input"))
        if original_input_size > 0:
            classification = (
                "remaining_candidate"
                if final_input_size < original_input_size
                else "no_op"
            )
            append_entry(
                _langfuse_compaction_savings_entry(
                    family="input",
                    field="input",
                    original_size_bytes=original_input_size,
                    final_size_bytes=final_input_size,
                    classification=classification,
                    mode="event_size_fit",
                    strategy="unchanged" if classification == "no_op" else None,
                )
            )

    audit = {
        "trace_id": trace_id,
        "call_type": call_type,
        "identifiers": _langfuse_compaction_audit_identifiers(generation_params),
        "classification_counts": classification_counts,
        "totals": totals,
        "entries": entries,
        "entry_count": entry_count,
        "retained_entry_count": len(entries),
        "dropped_entry_count": max(0, entry_count - len(entries)),
        "entries_truncated": entry_count > len(entries),
        "event_fit_failed": bool(
            isinstance(resolved_fit_summary, dict)
            and resolved_fit_summary.get("event_fit_failed")
        ),
    }
    if isinstance(resolved_fit_summary, dict):
        audit["event_fit_target_bytes"] = resolved_fit_summary.get(
            "event_fit_target_bytes"
        )
        audit["final_total_size_bytes"] = resolved_fit_summary.get(
            "final_total_size_bytes"
        )
    return audit


def _sanitize_metadata_key_for_size_audit(key: Any) -> str:
    key_text = str(key)
    key_lower = key_text.lower()
    if any(fragment in key_lower for fragment in _SENSITIVE_METADATA_KEY_FRAGMENTS):
        return "<redacted-key>"
    return key_text


def _largest_metadata_keys_by_size(metadata: Any) -> List[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []

    key_summaries: List[Dict[str, Any]] = []
    for key, value in metadata.items():
        key_summaries.append(
            {
                "key": _sanitize_metadata_key_for_size_audit(key),
                "size_bytes": _json_size_bytes(value),
                "value_type": type(value).__name__,
            }
        )

    return sorted(
        key_summaries,
        key=lambda item: int(item.get("size_bytes", 0)),
        reverse=True,
    )[:_LANGFUSE_SIZE_AUDIT_METADATA_KEY_LIMIT]


def _langfuse_summary_identifier(value: Any) -> Any:
    if value is None:
        return None
    value_text = str(value)
    value_size_bytes = _json_size_bytes(value_text)
    if value_size_bytes <= 200:
        return value
    return {
        "type": "litellm_langfuse_identifier_omitted",
        "size_bytes": value_size_bytes,
    }


def _langfuse_event_fit_target_bytes(max_event_size_bytes: int) -> int:
    return max(0, int(max_event_size_bytes * _LANGFUSE_EVENT_FIT_TARGET_RATIO))


def _langfuse_string_truncation_marker(
    *, omitted_bytes_estimate: int, omitted_count: int, field_name: str = "input"
) -> str:
    if field_name != "input":
        return (
            "\n...[litellm_langfuse_field_truncated "
            f"field={field_name} "
            f"omitted_chars={omitted_count} "
            f"omitted_bytes_estimate={max(0, omitted_bytes_estimate)}]...\n"
        )
    return (
        "\n...[litellm_langfuse_input_truncated "
        f"omitted_chars={omitted_count} "
        f"omitted_bytes_estimate={max(0, omitted_bytes_estimate)}]...\n"
    )


def _langfuse_structured_truncation_marker(
    *, omitted_bytes_estimate: int, omitted_count: int, field_name: str = "input"
) -> Dict[str, Any]:
    if field_name != "input":
        return {
            "type": _LANGFUSE_FIELD_TRUNCATION_MARKER_TYPE,
            "field": field_name,
            "omitted_items": max(0, omitted_count),
            "omitted_bytes_estimate": max(0, omitted_bytes_estimate),
        }
    return {
        "type": _LANGFUSE_INPUT_TRUNCATION_MARKER_TYPE,
        "omitted_items": max(0, omitted_count),
        "omitted_bytes_estimate": max(0, omitted_bytes_estimate),
    }


def _langfuse_field_omission_marker(
    *,
    field_name: str,
    omitted_bytes_estimate: int,
    omitted_count: int,
) -> Dict[str, Any]:
    return {
        "type": _LANGFUSE_FIELD_OMISSION_MARKER_TYPE,
        "field": field_name,
        "omitted_items": max(0, omitted_count),
        "omitted_bytes_estimate": max(0, omitted_bytes_estimate),
        "reason": "langfuse_event_size_limit",
    }


def _truncate_langfuse_string_input(
    value: str,
    *,
    original_input_size_bytes: int,
    fits_event: Callable[[Any], bool],
    field_name: str = "input",
) -> Tuple[str, int]:
    low = 0
    high = len(value)
    best_candidate: Optional[str] = None
    best_omitted_count = len(value)

    while low <= high:
        kept_count = (low + high) // 2
        omitted_count = len(value) - kept_count
        omitted_bytes_estimate = max(0, original_input_size_bytes)
        if kept_count > 0:
            head_count = (kept_count + 1) // 2
            tail_count = kept_count // 2
            omitted_tail_index = -tail_count if tail_count > 0 else len(value)
            omitted_bytes_estimate = _json_size_bytes(
                value[head_count:omitted_tail_index]
            )
        candidate = (
            value
            if omitted_count == 0
            else (
                value[: (kept_count + 1) // 2]
                + _langfuse_string_truncation_marker(
                    omitted_bytes_estimate=omitted_bytes_estimate,
                    omitted_count=omitted_count,
                    field_name=field_name,
                )
                + (value[-(kept_count // 2) :] if kept_count // 2 > 0 else "")
            )
        )

        if fits_event(candidate):
            best_candidate = candidate
            best_omitted_count = omitted_count
            low = kept_count + 1
        else:
            high = kept_count - 1

    if best_candidate is not None:
        return best_candidate, best_omitted_count

    fallback_marker = _langfuse_string_truncation_marker(
        omitted_bytes_estimate=original_input_size_bytes,
        omitted_count=len(value),
        field_name=field_name,
    )
    return fallback_marker, len(value)


def _list_with_langfuse_truncation_marker(
    value: List[Any],
    *,
    kept_count: int,
    original_input_size_bytes: int,
    field_name: str = "input",
) -> Tuple[List[Any], int]:
    if kept_count >= len(value):
        return safe_deep_copy(value), 0

    head_count = (kept_count + 1) // 2
    tail_count = kept_count // 2
    head = safe_deep_copy(value[:head_count])
    tail = safe_deep_copy(value[-tail_count:]) if tail_count > 0 else []
    omitted_items = len(value) - kept_count
    omitted_slice = value[head_count : len(value) - tail_count]
    omitted_bytes_estimate = _json_size_bytes(omitted_slice)
    if omitted_bytes_estimate <= 0:
        omitted_bytes_estimate = original_input_size_bytes
    marker = _langfuse_structured_truncation_marker(
        omitted_bytes_estimate=omitted_bytes_estimate,
        omitted_count=omitted_items,
        field_name=field_name,
    )
    return head + [marker] + tail, omitted_items


def _truncate_langfuse_list_input(
    value: List[Any],
    *,
    original_input_size_bytes: int,
    fits_event: Callable[[Any], bool],
    field_name: str = "input",
) -> Tuple[List[Any], int]:
    low = 0
    high = len(value)
    best_candidate: Optional[List[Any]] = None
    best_omitted_count = len(value)

    while low <= high:
        kept_count = (low + high) // 2
        candidate, omitted_count = _list_with_langfuse_truncation_marker(
            value,
            kept_count=kept_count,
            original_input_size_bytes=original_input_size_bytes,
            field_name=field_name,
        )
        if fits_event(candidate):
            best_candidate = candidate
            best_omitted_count = omitted_count
            low = kept_count + 1
        else:
            high = kept_count - 1

    if best_candidate is not None:
        return best_candidate, best_omitted_count

    return [
        _langfuse_structured_truncation_marker(
            omitted_bytes_estimate=original_input_size_bytes,
            omitted_count=len(value),
            field_name=field_name,
        )
    ], len(value)


def _truncate_langfuse_dict_input(
    value: Dict[Any, Any],
    *,
    original_input_size_bytes: int,
    fits_event: Callable[[Any], bool],
    field_name: str = "input",
) -> Tuple[Dict[Any, Any], int]:
    if fits_event(value):
        return safe_deep_copy(value), 0

    list_items = [
        (key, list_value)
        for key, list_value in value.items()
        if isinstance(list_value, list) and len(list_value) > 0
    ]
    list_items.sort(key=lambda item: _json_size_bytes(item[1]), reverse=True)

    for key, list_value in list_items:
        candidate_dict = safe_deep_copy(value)

        def fits_nested_list(candidate_list: Any) -> bool:
            nested_candidate = dict(candidate_dict)
            nested_candidate[key] = candidate_list
            return fits_event(nested_candidate)

        truncated_list, omitted_count = _truncate_langfuse_list_input(
            list_value,
            original_input_size_bytes=original_input_size_bytes,
            fits_event=fits_nested_list,
            field_name=field_name,
        )
        candidate_dict[key] = truncated_list
        if fits_event(candidate_dict):
            return candidate_dict, omitted_count

    string_items = [
        (key, string_value)
        for key, string_value in value.items()
        if isinstance(string_value, str) and len(string_value) > 0
    ]
    string_items.sort(key=lambda item: _json_size_bytes(item[1]), reverse=True)

    for key, string_value in string_items:
        candidate_dict = safe_deep_copy(value)

        def fits_nested_string(candidate_string: Any) -> bool:
            nested_candidate = dict(candidate_dict)
            nested_candidate[key] = candidate_string
            return fits_event(nested_candidate)

        truncated_string, omitted_count = _truncate_langfuse_string_input(
            string_value,
            original_input_size_bytes=_json_size_bytes(string_value),
            fits_event=fits_nested_string,
            field_name=field_name,
        )
        candidate_dict[key] = truncated_string
        if fits_event(candidate_dict):
            return candidate_dict, omitted_count

    return _langfuse_structured_truncation_marker(
        omitted_bytes_estimate=original_input_size_bytes,
        omitted_count=len(value),
        field_name=field_name,
    ), len(value)


def _truncate_langfuse_input_to_fit_event(
    input_value: Any,
    *,
    original_input_size_bytes: int,
    fits_event: Callable[[Any], bool],
    field_name: str = "input",
) -> Tuple[Any, int]:
    if isinstance(input_value, str):
        return _truncate_langfuse_string_input(
            input_value,
            original_input_size_bytes=original_input_size_bytes,
            fits_event=fits_event,
            field_name=field_name,
        )
    if isinstance(input_value, list):
        return _truncate_langfuse_list_input(
            input_value,
            original_input_size_bytes=original_input_size_bytes,
            fits_event=fits_event,
            field_name=field_name,
        )
    if isinstance(input_value, dict):
        return _truncate_langfuse_dict_input(
            input_value,
            original_input_size_bytes=original_input_size_bytes,
            fits_event=fits_event,
            field_name=field_name,
        )
    return safe_deep_copy(input_value), 0


def _langfuse_field_omitted_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if value is None:
        return 0
    return 1


def _langfuse_generation_field_fit_order(
    generation_params: Dict[str, Any],
) -> List[str]:
    ordered_fields: List[str] = []
    for field_name in _LANGFUSE_FIELD_FIT_PRIORITY:
        if field_name in generation_params:
            ordered_fields.append(field_name)

    for field_name in generation_params:
        if (
            field_name not in ordered_fields
            and field_name not in _LANGFUSE_FIELD_FIT_EXCLUDED_KEYS
        ):
            ordered_fields.append(field_name)
    return ordered_fields


def _fit_langfuse_generation_field_to_event_size(
    fitted_generation_params: Dict[str, Any],
    *,
    field_name: str,
    target_bytes: int,
    input_shape_hash_only: bool = False,
) -> Optional[Dict[str, Any]]:
    if field_name not in fitted_generation_params:
        return None

    original_value = fitted_generation_params.get(field_name)
    original_size_bytes = _json_size_bytes(original_value)
    omission_marker = _langfuse_field_omission_marker(
        field_name=field_name,
        omitted_bytes_estimate=original_size_bytes,
        omitted_count=_langfuse_field_omitted_count(original_value),
    )
    omission_marker_size_bytes = _json_size_bytes(omission_marker)
    if original_size_bytes <= omission_marker_size_bytes:
        return None

    def fits_event(candidate_value: Any) -> bool:
        candidate_generation_params = dict(fitted_generation_params)
        candidate_generation_params[field_name] = candidate_value
        return _json_size_bytes(candidate_generation_params) <= target_bytes

    if fits_event(original_value):
        return None

    if field_name in {"metadata", "model_parameters"}:
        truncated_value = omission_marker
        final_size_bytes = omission_marker_size_bytes
        omitted_count = _langfuse_field_omitted_count(original_value)
        strategy = "omitted"
    elif field_name == "input" and input_shape_hash_only:
        metadata = fitted_generation_params.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else None
        truncated_value = _build_langfuse_input_shape_hash_summary(
            original_value,
            original_input_size_bytes=original_size_bytes,
            metadata=metadata_dict,
        )
        final_size_bytes = _json_size_bytes(truncated_value)
        omitted_count = max(0, _langfuse_input_item_count(original_value) - len(truncated_value.get("head", [])) - len(truncated_value.get("tail", [])))
        strategy = "shape_hash_summary"
        if not fits_event(truncated_value):
            truncated_value = omission_marker
            final_size_bytes = omission_marker_size_bytes
            omitted_count = _langfuse_field_omitted_count(original_value)
            strategy = "omitted"
    else:
        truncated_value, omitted_count = _truncate_langfuse_input_to_fit_event(
            original_value,
            original_input_size_bytes=original_size_bytes,
            fits_event=fits_event,
            field_name=field_name,
        )
        final_size_bytes = _json_size_bytes(truncated_value)
        strategy = "truncated"

        if final_size_bytes >= original_size_bytes or not fits_event(truncated_value):
            truncated_value = omission_marker
            final_size_bytes = omission_marker_size_bytes
            omitted_count = _langfuse_field_omitted_count(original_value)
            strategy = "omitted"

    fitted_generation_params[field_name] = truncated_value
    return {
        "field": field_name,
        "strategy": strategy,
        "original_size_bytes": original_size_bytes,
        "final_size_bytes": final_size_bytes,
        "truncated_bytes": max(0, original_size_bytes - final_size_bytes),
        "omitted_count": omitted_count,
    }


def _build_langfuse_generation_fit_summary(
    *,
    target_bytes: int,
    field_summaries: List[Dict[str, Any]],
    final_total_size_bytes: int,
) -> Optional[Dict[str, Any]]:
    if not field_summaries and final_total_size_bytes <= target_bytes:
        return None

    summary: Dict[str, Any] = {
        "event_fit_target_bytes": target_bytes,
        "field_truncations": field_summaries,
        "truncated_fields": [
            field_summary["field"] for field_summary in field_summaries
        ],
        "omitted_fields": [
            field_summary["field"]
            for field_summary in field_summaries
            if field_summary.get("strategy") == "omitted"
        ],
        "final_total_size_bytes": final_total_size_bytes,
        "event_fit_failed": final_total_size_bytes > target_bytes,
    }

    input_summary = next(
        (
            field_summary
            for field_summary in field_summaries
            if field_summary.get("field") == "input"
        ),
        None,
    )
    if input_summary is not None:
        summary.update(
            {
                "input_truncated": input_summary.get("strategy") != "shape_hash_summary",
                "input_shape_hash_summary": input_summary.get("strategy")
                == "shape_hash_summary",
                "original_input_size_bytes": input_summary["original_size_bytes"],
                "final_input_size_bytes": input_summary["final_size_bytes"],
                "truncated_input_bytes": input_summary["truncated_bytes"],
                "omitted_input_count": input_summary["omitted_count"],
            }
        )
        if input_summary.get("strategy") == "shape_hash_summary":
            summary["input_summary_type"] = _LANGFUSE_INPUT_SUMMARY_TYPE
    return summary


def _fit_langfuse_generation_params_to_event_size(
    generation_params: Dict[str, Any],
    *,
    max_event_size_bytes: Optional[int] = None,
    input_shape_hash_only: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    max_bytes = max_event_size_bytes or _get_langfuse_max_event_size_bytes()
    target_bytes = _langfuse_event_fit_target_bytes(max_bytes)
    if _json_size_bytes(generation_params) < target_bytes:
        return generation_params, None

    fitted_generation_params = dict(generation_params)
    field_summaries: List[Dict[str, Any]] = []
    use_input_shape_hash_only = (
        _langfuse_input_shape_hash_only_enabled()
        if input_shape_hash_only is None
        else input_shape_hash_only
    )

    for field_name in _langfuse_generation_field_fit_order(fitted_generation_params):
        if _json_size_bytes(fitted_generation_params) <= target_bytes:
            break

        field_summary = _fit_langfuse_generation_field_to_event_size(
            fitted_generation_params,
            field_name=field_name,
            target_bytes=target_bytes,
            input_shape_hash_only=use_input_shape_hash_only,
        )
        if field_summary is not None:
            field_summaries.append(field_summary)

    final_total_size_bytes = _json_size_bytes(fitted_generation_params)
    fit_summary = _build_langfuse_generation_fit_summary(
        target_bytes=target_bytes,
        field_summaries=field_summaries,
        final_total_size_bytes=final_total_size_bytes,
    )
    if fit_summary is None:
        return generation_params, None
    return fitted_generation_params, fit_summary


def _build_langfuse_payload_size_summary(
    generation_params: Dict[str, Any],
    *,
    trace_id: Optional[str],
    call_type: Optional[str],
    max_event_size_bytes: Optional[int] = None,
    input_truncation_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    max_bytes = max_event_size_bytes or _get_langfuse_max_event_size_bytes()
    threshold_bytes = int(max_bytes * _LANGFUSE_SIZE_AUDIT_THRESHOLD_RATIO)
    total_size_bytes = _json_size_bytes(generation_params)

    if total_size_bytes < threshold_bytes and input_truncation_summary is None:
        return None

    metadata = generation_params.get("metadata")
    summary = {
        "trace_id": trace_id,
        "generation_id": _langfuse_summary_identifier(generation_params.get("id")),
        "generation_name": _langfuse_summary_identifier(
            generation_params.get("name")
        ),
        "model": _langfuse_summary_identifier(generation_params.get("model")),
        "call_type": call_type,
        "total_size_bytes": total_size_bytes,
        "max_event_size_bytes": max_bytes,
        "warning_threshold_bytes": threshold_bytes,
        "input_size_bytes": _json_size_bytes(generation_params.get("input")),
        "output_size_bytes": _json_size_bytes(generation_params.get("output")),
        "metadata_size_bytes": _json_size_bytes(metadata),
        "model_parameters_size_bytes": _json_size_bytes(
            generation_params.get("model_parameters")
        ),
        "largest_metadata_keys": _largest_metadata_keys_by_size(metadata),
    }
    if input_truncation_summary is not None:
        summary.update(input_truncation_summary)

    if total_size_bytes >= threshold_bytes or input_truncation_summary is not None:
        summary["compaction_savings_audit"] = _build_langfuse_compaction_savings_audit(
            generation_params,
            trace_id=trace_id,
            call_type=call_type,
            fit_summary=input_truncation_summary,
            max_event_size_bytes=max_bytes,
        )
    return summary


def _log_langfuse_payload_size_if_needed(
    generation_params: Dict[str, Any],
    *,
    trace_id: Optional[str],
    call_type: Optional[str],
    input_truncation_summary: Optional[Dict[str, Any]] = None,
) -> None:
    size_summary = _build_langfuse_payload_size_summary(
        generation_params=generation_params,
        trace_id=trace_id,
        call_type=call_type,
        input_truncation_summary=input_truncation_summary,
    )
    if size_summary is None:
        return

    record_langfuse_enqueue_size_audit(size_summary)

    event_fit_failed = bool(size_summary.get("event_fit_failed"))
    total_size_bytes = size_summary.get("total_size_bytes")
    max_event_size_bytes = size_summary.get("max_event_size_bytes")
    still_over_limit = (
        isinstance(total_size_bytes, int)
        and isinstance(max_event_size_bytes, int)
        and total_size_bytes > max_event_size_bytes
    )
    if not event_fit_failed and not still_over_limit:
        verbose_logger.debug(
            "Langfuse event size audit below SDK limit before enqueue: %s",
            json.dumps(size_summary, sort_keys=True),
        )
        return

    verbose_logger.warning(
        "Langfuse event near/exceeds size limit before SDK enqueue: %s",
        json.dumps(size_summary, sort_keys=True),
    )


def _explicit_openrouter_model_for_langfuse(
    metadata: Optional[dict],
    standard_logging_object: Optional[StandardLoggingPayload],
) -> Optional[str]:
    candidates: List[Any] = []
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("anthropic_adapter_original_model"),
                metadata.get("codex_adapter_original_model"),
                metadata.get("model"),
            ]
        )
    if isinstance(standard_logging_object, dict):
        standard_metadata = standard_logging_object.get("metadata")
        if isinstance(standard_metadata, dict):
            candidates.extend(
                [
                    standard_metadata.get("anthropic_adapter_original_model"),
                    standard_metadata.get("codex_adapter_original_model"),
                    standard_metadata.get("model"),
                ]
            )
        candidates.append(standard_logging_object.get("model"))

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if cleaned.lower().startswith("openrouter/") and len(cleaned) > len(
            "openrouter/"
        ):
            return cleaned
    return None


def _extract_cache_read_input_tokens(usage_obj) -> int:
    """
    Extract cache_read_input_tokens from usage object.

    Checks both:
    1. Top-level cache_read_input_tokens (Anthropic format)
    2. prompt_tokens_details.cached_tokens (Gemini, OpenAI format)

    See: https://github.com/BerriAI/litellm/issues/18520

    Args:
        usage_obj: Usage object from LLM response

    Returns:
        int: Number of cached tokens read, defaults to 0
    """
    cache_read_input_tokens = usage_obj.get("cache_read_input_tokens") or 0

    # Check prompt_tokens_details.cached_tokens (used by Gemini and other providers)
    if hasattr(usage_obj, "prompt_tokens_details"):
        prompt_tokens_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_tokens_details is not None and hasattr(
            prompt_tokens_details, "cached_tokens"
        ):
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
            if (
                cached_tokens is not None
                and isinstance(cached_tokens, (int, float))
                and cached_tokens > 0
            ):
                cache_read_input_tokens = cached_tokens

    return cache_read_input_tokens


def _coerce_langfuse_span_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            normalized_value = stripped_value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized_value)
        except ValueError:
            return None
    return None


def _log_metadata_spans_as_span(
    trace: StatefulTraceClient,
    metadata_spans: Any,
) -> None:
    if not isinstance(metadata_spans, list):
        return

    for span_descriptor in metadata_spans:
        if not isinstance(span_descriptor, dict):
            continue

        span_name = span_descriptor.get("name")
        if not isinstance(span_name, str) or not span_name.strip():
            continue

        span_kwargs: Dict[str, Any] = {
            "name": span_name.strip(),
            "input": span_descriptor.get("input"),
            "output": span_descriptor.get("output"),
            "metadata": span_descriptor.get("metadata"),
        }
        start_time = _coerce_langfuse_span_time(span_descriptor.get("start_time"))
        end_time = _coerce_langfuse_span_time(span_descriptor.get("end_time"))
        if start_time is not None:
            span_kwargs["start_time"] = start_time
        if end_time is not None:
            span_kwargs["end_time"] = end_time

        span = trace.span(**span_kwargs)
        if hasattr(span, "end"):
            span.end()


class LangFuseLogger:
    # Class variables or attributes
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        flush_interval=1,
    ):
        try:
            import langfuse
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
        # Instance variables
        self.secret_key = langfuse_secret or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = langfuse_host or os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        if not (
            self.langfuse_host.startswith("http://")
            or self.langfuse_host.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.langfuse_host = "http://" + self.langfuse_host
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.langfuse_flush_interval = LangFuseLogger._get_langfuse_flush_interval(
            flush_interval
        )

        if should_use_langfuse_mock():
            self.langfuse_client = create_mock_langfuse_client()
            self.is_mock_mode = True
        else:
            http_client = _get_httpx_client()
            self.langfuse_client = http_client.client
            self.is_mock_mode = False

        parameters = {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
            "host": self.langfuse_host,
            "release": self.langfuse_release,
            "debug": self.langfuse_debug,
            "flush_interval": self.langfuse_flush_interval,  # flush interval in seconds
            "httpx_client": self.langfuse_client,
        }
        self.langfuse_sdk_version: str = langfuse.version.__version__

        if Version(self.langfuse_sdk_version) >= Version("2.6.0"):
            parameters["sdk_integration"] = "litellm"
        self.Langfuse: Langfuse = self.safe_init_langfuse_client(parameters)

        # set the current langfuse project id in the environ
        # this is used by Alerting to link to the correct project
        if self.is_mock_mode:
            os.environ["LANGFUSE_PROJECT_ID"] = "mock-project-id"
            verbose_logger.debug("Langfuse Mock: Using mock project ID")
        else:
            try:
                project_id = self.Langfuse.client.projects.get().data[0].id
                os.environ["LANGFUSE_PROJECT_ID"] = project_id
            except Exception:
                project_id = None

        if os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY") is not None:
            upstream_langfuse_debug = (
                str_to_bool(self.upstream_langfuse_debug)
                if self.upstream_langfuse_debug is not None
                else None
            )
            self.upstream_langfuse_secret_key = os.getenv(
                "UPSTREAM_LANGFUSE_SECRET_KEY"
            )
            self.upstream_langfuse_public_key = os.getenv(
                "UPSTREAM_LANGFUSE_PUBLIC_KEY"
            )
            self.upstream_langfuse_host = os.getenv("UPSTREAM_LANGFUSE_HOST")
            self.upstream_langfuse_release = os.getenv("UPSTREAM_LANGFUSE_RELEASE")
            self.upstream_langfuse_debug = os.getenv("UPSTREAM_LANGFUSE_DEBUG")
            self.upstream_langfuse = Langfuse(
                public_key=self.upstream_langfuse_public_key,
                secret_key=self.upstream_langfuse_secret_key,
                host=self.upstream_langfuse_host,
                release=self.upstream_langfuse_release,
                debug=(
                    upstream_langfuse_debug
                    if upstream_langfuse_debug is not None
                    else False
                ),
            )
        else:
            self.upstream_langfuse = None

    def safe_init_langfuse_client(self, parameters: dict) -> Langfuse:
        """
        Safely init a langfuse client if the number of initialized clients is less than the max

        Note:
            - Langfuse initializes 1 thread everytime a client is initialized.
            - We've had an incident in the past where we reached 100% cpu utilization because Langfuse was initialized several times.
        """
        from langfuse import Langfuse

        if litellm.initialized_langfuse_clients >= MAX_LANGFUSE_INITIALIZED_CLIENTS:
            raise Exception(
                f"Max langfuse clients reached: {litellm.initialized_langfuse_clients} is greater than {MAX_LANGFUSE_INITIALIZED_CLIENTS}"
            )
        langfuse_client = Langfuse(**parameters)
        litellm.initialized_langfuse_clients += 1
        verbose_logger.debug(
            f"Created langfuse client number {litellm.initialized_langfuse_clients}"
        )
        return langfuse_client

    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict:
        """
        Adds metadata from proxy request headers to Langfuse logging if keys start with "langfuse_"
        and overwrites litellm_params.metadata if already included.

        For example if you want to append your trace to an existing `trace_id` via header, send
        `headers: { ..., langfuse_existing_trace_id: your-existing-trace-id }` via proxy request.
        """
        if litellm_params is None:
            return metadata

        if litellm_params.get("proxy_server_request") is None:
            return metadata

        if metadata is None:
            metadata = {}

        proxy_headers = (
            litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}
        )

        for metadata_param_key in proxy_headers:
            if metadata_param_key.startswith("langfuse_"):
                trace_param_key = metadata_param_key.replace("langfuse_", "", 1)
                if trace_param_key in metadata:
                    verbose_logger.debug(
                        f"Overwriting Langfuse `{trace_param_key}` from request header"
                    )
                else:
                    verbose_logger.debug(
                        f"Found Langfuse `{trace_param_key}` in request header"
                    )
                metadata[trace_param_key] = proxy_headers.get(metadata_param_key)

        return metadata

    def log_event_on_langfuse(
        self,
        kwargs: dict,
        response_obj: Union[
            None,
            dict,
            EmbeddingResponse,
            ModelResponse,
            TextCompletionResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
            HttpxBinaryResponseContent,
            ResponsesAPIResponse,
        ],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ) -> dict:
        """
        Logs a success or error event on Langfuse
        """
        try:
            verbose_logger.debug(
                f"Langfuse Logging - Enters logging function for model {kwargs}"
            )

            # set default values for input/output for langfuse logging
            input = None
            output = None

            litellm_params = kwargs.get("litellm_params", {})
            litellm_call_id = kwargs.get("litellm_call_id", None)
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            optional_params = safe_deep_copy(kwargs.get("optional_params", {}))

            prompt = {"messages": kwargs.get("messages")}

            functions = optional_params.pop("functions", None)
            tools = optional_params.pop("tools", None)
            # Remove secret_fields to prevent leaking sensitive data (e.g., authorization headers)
            optional_params.pop("secret_fields", None)
            if functions is not None:
                prompt["functions"] = functions
            if tools is not None:
                prompt["tools"] = tools

            # langfuse only accepts str, int, bool, float for logging
            for param, value in optional_params.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        optional_params[param] = str(value)
                    except Exception:
                        # if casting value to str fails don't block logging
                        pass

            input, output = self._get_langfuse_input_output_content(
                kwargs=kwargs,
                response_obj=response_obj,
                prompt=prompt,
                level=level,
                status_message=status_message,
            )
            verbose_logger.debug(
                f"OUTPUT IN LANGFUSE: {output}; original: {response_obj}"
            )
            trace_id = None
            generation_id = None
            if self._is_langfuse_v2():
                trace_id, generation_id = self._log_langfuse_v2(
                    user_id=user_id,
                    metadata=metadata,
                    litellm_params=litellm_params,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    kwargs=kwargs,
                    optional_params=optional_params,
                    input=input,
                    response_obj=response_obj,
                    level=level,
                    litellm_call_id=litellm_call_id,
                )
            elif response_obj is not None:
                self._log_langfuse_v1(
                    user_id=user_id,
                    metadata=metadata,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    kwargs=kwargs,
                    optional_params=optional_params,
                    input=input,
                    response_obj=response_obj,
                )
            verbose_logger.debug(
                f"Langfuse Layer Logging - final response object: {response_obj}"
            )
            verbose_logger.info("Langfuse Layer Logging - logging success")

            return {"trace_id": trace_id, "generation_id": generation_id}
        except Exception as e:
            verbose_logger.exception(
                "Langfuse Layer Error(): Exception occured - {}".format(str(e))
            )
            return {"trace_id": None, "generation_id": None}

    def _get_langfuse_input_output_content(
        self,
        kwargs: dict,
        response_obj: Union[
            None,
            dict,
            EmbeddingResponse,
            ModelResponse,
            TextCompletionResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
            HttpxBinaryResponseContent,
            ResponsesAPIResponse,
        ],
        prompt: dict,
        level: str,
        status_message: Optional[str],
    ) -> Tuple[Optional[dict], Optional[Union[str, dict, list]]]:
        """
        Get the input and output content for Langfuse logging

        Args:
            kwargs: The keyword arguments passed to the function
            response_obj: The response object returned by the function
            prompt: The prompt used to generate the response
            level: The level of the log message
            status_message: The status message of the log message

        Returns:
            input: The input content for Langfuse logging
            output: The output content for Langfuse logging
        """
        input = None
        output: Optional[Union[str, dict, List[Any]]] = None
        if (
            level == "ERROR"
            and status_message is not None
            and isinstance(status_message, str)
        ):
            input = prompt
            output = status_message
        elif response_obj is not None and (
            kwargs.get("call_type", None) == "embedding"
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            input = prompt
            output = None
        elif response_obj is not None and isinstance(
            response_obj, litellm.ModelResponse
        ):
            input = prompt
            output = self._get_chat_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(
            response_obj, litellm.HttpxBinaryResponseContent
        ):
            input = prompt
            output = "speech-output"
        elif response_obj is not None and isinstance(
            response_obj, litellm.TextCompletionResponse
        ):
            input = prompt
            output = self._get_text_completion_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(
            response_obj, litellm.ImageResponse
        ):
            input = prompt
            output = response_obj.get("data", None)
        elif response_obj is not None and isinstance(
            response_obj, litellm.TranscriptionResponse
        ):
            input = prompt
            output = response_obj.get("text", None)
        elif response_obj is not None and isinstance(
            response_obj, litellm.RerankResponse
        ):
            input = prompt
            output = response_obj.results
        elif response_obj is not None and isinstance(
            response_obj, litellm.ResponsesAPIResponse
        ):
            input = prompt
            output = self._get_responses_api_content_for_langfuse(response_obj)
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "_arealtime"
            and response_obj is not None
            and isinstance(response_obj, list)
        ):
            input = kwargs.get("input")
            output = response_obj
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "pass_through_endpoint"
            and response_obj is not None
            and isinstance(response_obj, dict)
        ):
            input = prompt
            output = response_obj.get("response", "")
        return input, output

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, user_id
    ):
        """
        Langfuse SDK uses a background thread to log events

        This approach does not impact latency and runs in the background
        """

    def _is_langfuse_v2(self):
        import langfuse

        return Version(langfuse.version.__version__) >= Version("2.0.0")

    def _log_langfuse_v1(
        self,
        user_id,
        metadata,
        output,
        start_time,
        end_time,
        kwargs,
        optional_params,
        input,
        response_obj,
    ):
        from langfuse.model import CreateGeneration, CreateTrace  # type: ignore

        verbose_logger.warning(
            "Please upgrade langfuse to v2.0.0 or higher: https://github.com/langfuse/langfuse-python/releases/tag/v2.0.1"
        )

        trace = self.Langfuse.trace(  # type: ignore
            CreateTrace(  # type: ignore
                name=metadata.get("generation_name", "litellm-completion"),
                input=input,
                output=output,
                userId=user_id,
            )
        )

        custom_llm_provider = cast(Optional[str], kwargs.get("custom_llm_provider"))
        model_name = reconstruct_model_name(
            kwargs.get("model", ""), custom_llm_provider, metadata
        )

        trace.generation(
            CreateGeneration(
                name=metadata.get("generation_name", "litellm-completion"),
                startTime=start_time,
                endTime=end_time,
                model=model_name,
                modelParameters=optional_params,
                prompt=input,
                completion=output,
                usage={
                    "prompt_tokens": response_obj.usage.prompt_tokens,
                    "completion_tokens": response_obj.usage.completion_tokens,
                },
                metadata=metadata,
            )
        )

    def _log_langfuse_v2(  # noqa: PLR0915
        self,
        user_id: Optional[str],
        metadata: dict,
        litellm_params: dict,
        output: Optional[Union[str, dict, list]],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        kwargs: dict,
        optional_params: dict,
        input: Optional[dict],
        response_obj,
        level: str,
        litellm_call_id: Optional[str],
    ) -> tuple:
        verbose_logger.debug("Langfuse Layer Logging - logging to langfuse v2")

        try:
            standard_logging_object: Optional[StandardLoggingPayload] = cast(
                Optional[StandardLoggingPayload],
                kwargs.get("standard_logging_object", None),
            )
            tags = (
                self._get_langfuse_tags(
                    standard_logging_object=standard_logging_object,
                    metadata=metadata,
                )
                if self._supports_tags()
                else []
            )

            if standard_logging_object is None:
                end_user_id = None
                prompt_management_metadata: Optional[
                    StandardLoggingPromptManagementMetadata
                ] = None
            else:
                end_user_id = standard_logging_object["metadata"].get(
                    "user_api_key_end_user_id", None
                )

                prompt_management_metadata = cast(
                    Optional[StandardLoggingPromptManagementMetadata],
                    standard_logging_object["metadata"].get(
                        "prompt_management_metadata", None
                    ),
                )

            if end_user_id is None and isinstance(metadata, dict):
                end_user_id = (
                    metadata.get("user_api_key_end_user_id")
                    or metadata.get("trace_user_id")
                    or user_id
                )

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata: Dict[str, Any] = {}
            if prompt_management_metadata is not None:
                clean_metadata[
                    "prompt_management_metadata"
                ] = prompt_management_metadata
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

                    # clean litellm metadata before logging
                    if key in [
                        "headers",
                        "endpoint",
                        "caching_groups",
                        "previous_models",
                    ]:
                        continue
                    else:
                        clean_metadata[key] = value

            # Add default langfuse tags
            tags = self.add_default_langfuse_tags(
                tags=tags, kwargs=kwargs, metadata=metadata
            )

            session_id = clean_metadata.pop("session_id", None)
            trace_name = cast(Optional[str], clean_metadata.pop("trace_name", None))
            trace_id = clean_metadata.pop("trace_id", None)
            langfuse_spans = clean_metadata.pop("langfuse_spans", None)
            # Use standard_logging_object.trace_id if available (when trace_id from metadata is None)
            # This allows standard trace_id to be used when provided in standard_logging_object
            if trace_id is None and standard_logging_object is not None:
                trace_id = cast(Optional[str], standard_logging_object.get("trace_id"))
            # Fallback to litellm_call_id if no trace_id found
            if trace_id is None:
                trace_id = kwargs.get("litellm_trace_id") or litellm_call_id
            existing_trace_id = clean_metadata.pop("existing_trace_id", None)
            # If existing_trace_id is provided, use it as the trace_id to return
            # This allows continuing an existing trace while still returning the correct trace_id
            if existing_trace_id is not None:
                trace_id = existing_trace_id
            update_trace_keys = cast(list, clean_metadata.pop("update_trace_keys", []))
            debug = clean_metadata.pop("debug_langfuse", None)
            mask_input = clean_metadata.pop("mask_input", False)
            mask_output = clean_metadata.pop("mask_output", False)
            # Look for masking function in the dedicated location first (set by scrub_sensitive_keys_in_metadata)
            # Fall back to metadata for backwards compatibility
            masking_function = litellm_params.get(
                "_langfuse_masking_function"
            ) or clean_metadata.pop("langfuse_masking_function", None)

            # Apply custom masking function if provided
            if masking_function is not None and callable(masking_function):
                input = self._apply_masking_function(input, masking_function)
                output = self._apply_masking_function(output, masking_function)

            clean_metadata = redact_user_api_key_info(metadata=clean_metadata)

            if trace_name is None and existing_trace_id is None:
                # just log `litellm-{call_type}` as the trace name
                ## DO NOT SET TRACE_NAME if trace-id set. this can lead to overwriting of past traces.
                trace_name = f"litellm-{kwargs.get('call_type', 'completion')}"

            if existing_trace_id is not None:
                trace_params: Dict[str, Any] = {"id": existing_trace_id}

                # Update the following keys for this trace
                for metadata_param_key in update_trace_keys:
                    trace_param_key = metadata_param_key.replace("trace_", "")
                    if trace_param_key not in trace_params:
                        updated_trace_value = clean_metadata.pop(
                            metadata_param_key, None
                        )
                        if updated_trace_value is not None:
                            trace_params[trace_param_key] = updated_trace_value

                # Pop the trace specific keys that would have been popped if there were a new trace
                for key in list(
                    filter(lambda key: key.startswith("trace_"), clean_metadata.keys())
                ):
                    clean_metadata.pop(key, None)

                # Special keys that are found in the function arguments and not the metadata
                if "input" in update_trace_keys:
                    trace_params["input"] = (
                        input if not mask_input else "redacted-by-litellm"
                    )
                if "output" in update_trace_keys:
                    trace_params["output"] = (
                        output if not mask_output else "redacted-by-litellm"
                    )
            else:  # don't overwrite an existing trace
                trace_params = {
                    "id": trace_id,
                    "name": trace_name,
                    "session_id": session_id,
                    "input": input if not mask_input else "redacted-by-litellm",
                    "version": clean_metadata.pop(
                        "trace_version", clean_metadata.get("version", None)
                    ),  # If provided just version, it will applied to the trace as well, if applied a trace version it will take precedence
                    "user_id": end_user_id,
                }
                for key in list(
                    filter(lambda key: key.startswith("trace_"), clean_metadata.keys())
                ):
                    trace_params[key.replace("trace_", "")] = clean_metadata.pop(
                        key, None
                    )

                if level == "ERROR":
                    trace_params["status_message"] = output
                else:
                    trace_params["output"] = (
                        output if not mask_output else "redacted-by-litellm"
                    )

            if debug is True or (isinstance(debug, str) and debug.lower() == "true"):
                metadata_passed_to_litellm = _strip_langfuse_generation_metadata(
                    metadata
                )
                if "metadata" in trace_params:
                    # log the raw_metadata in the trace
                    trace_params["metadata"]["metadata_passed_to_litellm"] = (
                        metadata_passed_to_litellm
                    )
                else:
                    trace_params["metadata"] = {
                        "metadata_passed_to_litellm": metadata_passed_to_litellm
                    }

            cost = kwargs.get("response_cost", None)
            verbose_logger.debug(f"trace: {cost}")

            clean_metadata["litellm_response_cost"] = cost
            if standard_logging_object is not None:
                hidden_params = standard_logging_object.get("hidden_params", {})
                clean_metadata["hidden_params"] = filter_exceptions_from_params(
                    hidden_params
                )

            if (
                litellm.langfuse_default_tags is not None
                and isinstance(litellm.langfuse_default_tags, list)
                and "proxy_base_url" in litellm.langfuse_default_tags
            ):
                proxy_base_url = os.environ.get("PROXY_BASE_URL", None)
                if proxy_base_url is not None:
                    tags.append(f"proxy_base_url:{proxy_base_url}")

            api_base = litellm_params.get("api_base", None)
            if api_base:
                clean_metadata["api_base"] = api_base

            vertex_location = kwargs.get("vertex_location", None)
            if vertex_location:
                clean_metadata["vertex_location"] = vertex_location

            aws_region_name = kwargs.get("aws_region_name", None)
            if aws_region_name:
                clean_metadata["aws_region_name"] = aws_region_name

            if self._supports_tags():
                if "cache_hit" in kwargs:
                    if kwargs["cache_hit"] is None:
                        kwargs["cache_hit"] = False
                    clean_metadata["cache_hit"] = kwargs["cache_hit"]
                if existing_trace_id is None:
                    trace_params.update({"tags": tags})

            proxy_server_request = litellm_params.get("proxy_server_request", None)
            if proxy_server_request:
                proxy_server_request.get("method", None)
                proxy_server_request.get("url", None)
                headers = proxy_server_request.get("headers", None)
                clean_headers = {}
                if headers:
                    for key, value in headers.items():
                        # these headers can leak our API keys and/or JWT tokens
                        if key.lower() not in ["authorization", "cookie", "referer"]:
                            clean_headers[key] = value

            trace_params, trace_fit_summary = (
                _fit_langfuse_generation_params_to_event_size(trace_params)
            )
            _log_langfuse_payload_size_if_needed(
                generation_params=trace_params,
                trace_id=trace_id,
                call_type=f"{kwargs.get('call_type', 'completion')}.trace",
                input_truncation_summary=trace_fit_summary,
            )

            trace: StatefulTraceClient = self.Langfuse.trace(**trace_params)

            _log_metadata_spans_as_span(trace, langfuse_spans)

            # Log provider specific information as a span
            log_provider_specific_information_as_span(trace, clean_metadata)

            # Log guardrail information as a span
            self._log_guardrail_information_as_span(
                trace=trace,
                standard_logging_object=standard_logging_object,
            )

            generation_id = None
            usage = None
            usage_details = None
            cost_details: Optional[LangfuseCostDetails] = None
            fallback_prompt_tokens = 0
            fallback_completion_tokens = 0
            fallback_total_tokens = 0
            fallback_cost = cost
            fallback_model_name: Optional[str] = None
            if standard_logging_object is not None:
                fallback_prompt_tokens = (
                    standard_logging_object.get("prompt_tokens", 0) or 0
                )
                fallback_completion_tokens = (
                    standard_logging_object.get("completion_tokens", 0) or 0
                )
                fallback_total_tokens = (
                    standard_logging_object.get("total_tokens", 0) or 0
                )
                fallback_cost = standard_logging_object.get("response_cost", cost)
                fallback_model_name = standard_logging_object.get("model")

            if response_obj is not None:
                if (
                    hasattr(response_obj, "id")
                    and response_obj.get("id", None) is not None
                ):
                    generation_id = litellm.utils.get_logging_id(
                        start_time, response_obj
                    )
                _usage_obj = getattr(response_obj, "usage", None)

                if _usage_obj:
                    # Safely get usage values, defaulting None to 0 for Langfuse compatibility.
                    # Some providers may return null for token counts.
                    prompt_tokens = getattr(_usage_obj, "prompt_tokens", None) or 0
                    completion_tokens = (
                        getattr(_usage_obj, "completion_tokens", None) or 0
                    )
                    total_tokens = getattr(_usage_obj, "total_tokens", None) or 0

                    cache_creation_input_tokens = (
                        _usage_obj.get("cache_creation_input_tokens") or 0
                    )
                    cache_read_input_tokens = _extract_cache_read_input_tokens(
                        _usage_obj
                    )

                    # Langfuse prefers the generic usage shape over provider-specific
                    # token keys for persisting totalCost and trace rollups.
                    input_tokens = prompt_tokens - cache_read_input_tokens
                    usage = {
                        "input": input_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                        "unit": "TOKENS",
                        "total_cost": cost if self._supports_costs() else None,
                    }
                    # According to langfuse documentation: "the input value must be reduced by the number of cache_read_input_tokens"
                    usage_details = LangfuseUsageDetails(
                        input=input_tokens,
                        output=completion_tokens,
                        total=total_tokens,
                        cache_creation_input_tokens=cache_creation_input_tokens,
                        cache_read_input_tokens=cache_read_input_tokens,
                    )
                    if self._supports_costs() and cost is not None:
                        cost_details = LangfuseCostDetails(total=float(cost))

            if (
                usage is None
                or (
                    usage.get("prompt_tokens", 0) == 0
                    and usage.get("completion_tokens", 0) == 0
                    and fallback_total_tokens > 0
                )
            ) and fallback_total_tokens > 0:
                usage = {
                    "input": fallback_prompt_tokens,
                    "output": fallback_completion_tokens,
                    "total": fallback_total_tokens,
                    "unit": "TOKENS",
                    "total_cost": fallback_cost if self._supports_costs() else None,
                }
                usage_details = LangfuseUsageDetails(
                    input=fallback_prompt_tokens,
                    output=fallback_completion_tokens,
                    total=fallback_total_tokens,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                )
                if self._supports_costs() and fallback_cost is not None:
                    cost_details = LangfuseCostDetails(total=float(fallback_cost))

            generation_name = clean_metadata.pop("generation_name", None)
            if generation_name is None:
                # if `generation_name` is None, use sensible default values
                # If using litellm proxy user `key_alias` if not None
                # If `key_alias` is None, just log `litellm-{call_type}` as the generation name
                _user_api_key_alias = cast(
                    Optional[str], clean_metadata.get("user_api_key_alias", None)
                )
                generation_name = (
                    f"litellm-{cast(str, kwargs.get('call_type', 'completion'))}"
                )
                if _user_api_key_alias is not None:
                    generation_name = f"litellm:{_user_api_key_alias}"

            if response_obj is not None:
                system_fingerprint = getattr(response_obj, "system_fingerprint", None)
            else:
                system_fingerprint = None

            if system_fingerprint is not None:
                optional_params["system_fingerprint"] = system_fingerprint

            custom_llm_provider = cast(Optional[str], kwargs.get("custom_llm_provider"))
            model_name = reconstruct_model_name(
                kwargs.get("model", ""), custom_llm_provider, metadata
            )
            explicit_openrouter_model = _explicit_openrouter_model_for_langfuse(
                metadata,
                standard_logging_object,
            )
            if explicit_openrouter_model is not None:
                model_name = explicit_openrouter_model
            if (
                (not model_name or model_name == "unknown")
                and fallback_model_name is not None
                and len(fallback_model_name) > 0
            ):
                model_name = fallback_model_name

            generation_params = {
                "name": generation_name,
                "id": clean_metadata.pop("generation_id", generation_id),
                "start_time": start_time,
                "end_time": end_time,
                "model": model_name,
                "model_parameters": optional_params,
                "input": input if not mask_input else "redacted-by-litellm",
                "output": output if not mask_output else "redacted-by-litellm",
                "usage": usage,
                "usage_details": usage_details,
                "cost_details": cost_details,
                "metadata": _strip_langfuse_generation_metadata(
                    log_requester_metadata(clean_metadata)
                ),
                "level": level,
                "version": clean_metadata.pop("version", None),
            }

            parent_observation_id = metadata.get("parent_observation_id", None)
            if parent_observation_id is not None:
                generation_params["parent_observation_id"] = parent_observation_id

            if self._supports_prompt():
                generation_params = _add_prompt_to_generation_params(
                    generation_params=generation_params,
                    clean_metadata=clean_metadata,
                    prompt_management_metadata=prompt_management_metadata,
                    langfuse_client=self.Langfuse,
                )
            if output is not None and isinstance(output, str) and level == "ERROR":
                generation_params["status_message"] = output

            if self._supports_completion_start_time():
                generation_params["completion_start_time"] = kwargs.get(
                    "completion_start_time", None
                )

            (
                generation_params,
                input_truncation_summary,
            ) = _fit_langfuse_generation_params_to_event_size(generation_params)

            _log_langfuse_payload_size_if_needed(
                generation_params=generation_params,
                trace_id=trace_id,
                call_type=cast(Optional[str], kwargs.get("call_type")),
                input_truncation_summary=input_truncation_summary,
            )

            generation_client = trace.generation(**generation_params)

            # Return the trace_id we set (which should be litellm_call_id when no explicit trace_id provided)
            # We explicitly set trace_id in trace_params["id"], so langfuse should use it
            # Verify langfuse accepted our trace_id; if it differs, log a warning but still return our intended value
            # to match expected test behavior
            if hasattr(generation_client, "trace_id") and generation_client.trace_id:
                if generation_client.trace_id != trace_id:
                    verbose_logger.warning(
                        f"Langfuse trace_id mismatch: set {trace_id}, but langfuse returned {generation_client.trace_id}. "
                        "Using our intended trace_id for consistency."
                    )
            return trace_id, generation_id
        except Exception:
            verbose_logger.error(f"Langfuse Layer Error - {traceback.format_exc()}")
            return None, None

    @staticmethod
    def _get_chat_content_for_langfuse(
        response_obj: ModelResponse,
    ):
        """
        Get the chat content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            output = response_obj["choices"][0]["message"].json()
            return output
        else:
            return None

    @staticmethod
    def _get_text_completion_content_for_langfuse(
        response_obj: TextCompletionResponse,
    ):
        """
        Get the text completion content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            return response_obj.choices[0].text
        else:
            return None

    @staticmethod
    def _get_responses_api_content_for_langfuse(
        response_obj: ResponsesAPIResponse,
    ):
        """
        Get the responses API content for Langfuse logging
        """
        if hasattr(response_obj, "output") and response_obj.output:
            # ResponsesAPIResponse.output is a list of strings
            return response_obj.output
        else:
            return None

    @staticmethod
    def _get_langfuse_tags(
        standard_logging_object: Optional[StandardLoggingPayload],
        metadata: Optional[dict] = None,
    ) -> List[str]:
        tags: List[str] = []

        if standard_logging_object is not None:
            request_tags = standard_logging_object.get("request_tags", []) or []
            if isinstance(request_tags, list):
                for tag in request_tags:
                    if isinstance(tag, str) and tag and tag not in tags:
                        tags.append(tag)

        if isinstance(metadata, dict):
            for key in ("request_tags", "tags"):
                metadata_tags = metadata.get(key) or []
                if not isinstance(metadata_tags, list):
                    continue
                for tag in metadata_tags:
                    if isinstance(tag, str) and tag and tag not in tags:
                        tags.append(tag)

        return tags

    def add_default_langfuse_tags(self, tags, kwargs, metadata):
        """
        Helper function to add litellm default langfuse tags

        - Special LiteLLM tags:
            - cache_hit
            - cache_key

        """
        if litellm.langfuse_default_tags is not None and isinstance(
            litellm.langfuse_default_tags, list
        ):
            if "cache_hit" in litellm.langfuse_default_tags:
                _cache_hit_value = kwargs.get("cache_hit", False)
                tags.append(f"cache_hit:{_cache_hit_value}")
            if "cache_key" in litellm.langfuse_default_tags:
                _hidden_params = metadata.get("hidden_params", {}) or {}
                _cache_key = _hidden_params.get("cache_key", None)
                if _cache_key is None and litellm.cache is not None:
                    # fallback to using "preset_cache_key"
                    _preset_cache_key = litellm.cache._get_preset_cache_key_from_kwargs(
                        **kwargs
                    )
                    _cache_key = _preset_cache_key
                tags.append(f"cache_key:{_cache_key}")
        return tags

    def _supports_tags(self):
        """Check if current langfuse version supports tags"""
        return Version(self.langfuse_sdk_version) >= Version("2.6.3")

    def _supports_prompt(self):
        """Check if current langfuse version supports prompt"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    def _supports_costs(self):
        """Check if current langfuse version supports costs"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    def _supports_completion_start_time(self):
        """Check if current langfuse version supports completion start time"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    @staticmethod
    def _apply_masking_function(
        data: Any, masking_function: Callable[[Any], Any]
    ) -> Any:
        """
        Apply a masking function to data, handling different data types.

        Args:
            data: The data to mask (can be str, dict, list, or None)
            masking_function: A callable that takes data and returns masked data

        Returns:
            The masked data
        """
        if data is None:
            return None

        try:
            if isinstance(data, str):
                return masking_function(data)
            elif isinstance(data, dict):
                masked_dict = {}
                for key, value in data.items():
                    masked_dict[key] = LangFuseLogger._apply_masking_function(
                        value, masking_function
                    )
                return masked_dict
            elif isinstance(data, list):
                return [
                    LangFuseLogger._apply_masking_function(item, masking_function)
                    for item in data
                ]
            else:
                # For other types, try to apply the function directly
                return masking_function(data)
        except Exception as e:
            verbose_logger.warning(
                f"Failed to apply masking function: {e}. Returning original data."
            )
            return data

    @staticmethod
    def _get_langfuse_flush_interval(flush_interval: int) -> int:
        """
        Get the langfuse flush interval to initialize the Langfuse client

        Reads `LANGFUSE_FLUSH_INTERVAL` from the environment variable.
        If not set, uses the flush interval passed in as an argument.

        Args:
            flush_interval: The flush interval to use if LANGFUSE_FLUSH_INTERVAL is not set

        Returns:
            [int] The flush interval to use to initialize the Langfuse client
        """
        return int(os.getenv("LANGFUSE_FLUSH_INTERVAL") or flush_interval)

    def _log_guardrail_information_as_span(
        self,
        trace: StatefulTraceClient,
        standard_logging_object: Optional[StandardLoggingPayload],
    ):
        """
        Log guardrail information as a span
        """
        if standard_logging_object is None:
            verbose_logger.debug(
                "Not logging guardrail information as span because standard_logging_object is None"
            )
            return

        guardrail_information = standard_logging_object.get(
            "guardrail_information", None
        )
        if not guardrail_information:
            verbose_logger.debug(
                "Not logging guardrail information as span because guardrail_information is empty"
            )
            return

        if not isinstance(guardrail_information, list):
            verbose_logger.debug(
                "Not logging guardrail information as span because guardrail_information is not a list: %s",
                type(guardrail_information),
            )
            return

        for guardrail_entry in guardrail_information:
            if not isinstance(guardrail_entry, dict):
                verbose_logger.debug(
                    "Skipping guardrail entry with unexpected type: %s",
                    type(guardrail_entry),
                )
                continue

            span = trace.span(
                name="guardrail",
                input=guardrail_entry.get("guardrail_request", None),
                output=guardrail_entry.get("guardrail_response", None),
                metadata={
                    "guardrail_name": guardrail_entry.get("guardrail_name", None),
                    "guardrail_mode": guardrail_entry.get("guardrail_mode", None),
                    "guardrail_masked_entity_count": guardrail_entry.get(
                        "masked_entity_count", None
                    ),
                },
                start_time=guardrail_entry.get("start_time", None),  # type: ignore
                end_time=guardrail_entry.get("end_time", None),  # type: ignore
            )

            verbose_logger.debug(f"Logged guardrail information as span: {span}")
            span.end()


def _add_prompt_to_generation_params(
    generation_params: dict,
    clean_metadata: dict,
    prompt_management_metadata: Optional[StandardLoggingPromptManagementMetadata],
    langfuse_client: Any,
) -> dict:
    from langfuse import Langfuse
    from langfuse.model import (
        ChatPromptClient,
        Prompt_Chat,
        Prompt_Text,
        TextPromptClient,
    )

    langfuse_client = cast(Langfuse, langfuse_client)

    user_prompt = clean_metadata.pop("prompt", None)
    if user_prompt is None and prompt_management_metadata is None:
        pass
    elif isinstance(user_prompt, dict):
        if user_prompt.get("type", "") == "chat":
            _prompt_chat = Prompt_Chat(**user_prompt)
            generation_params["prompt"] = ChatPromptClient(prompt=_prompt_chat)
        elif user_prompt.get("type", "") == "text":
            _prompt_text = Prompt_Text(**user_prompt)
            generation_params["prompt"] = TextPromptClient(prompt=_prompt_text)
        elif "version" in user_prompt and "prompt" in user_prompt:
            # prompts
            if isinstance(user_prompt["prompt"], str):
                prompt_text_params = getattr(
                    Prompt_Text, "model_fields", Prompt_Text.__fields__
                )
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_text_params and "tags" in prompt_text_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []
                _prompt_obj = Prompt_Text(**_data)  # type: ignore
                generation_params["prompt"] = TextPromptClient(prompt=_prompt_obj)

            elif isinstance(user_prompt["prompt"], list):
                prompt_chat_params = getattr(
                    Prompt_Chat, "model_fields", Prompt_Chat.__fields__
                )
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_chat_params and "tags" in prompt_chat_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []

                _prompt_obj = Prompt_Chat(**_data)  # type: ignore

                generation_params["prompt"] = ChatPromptClient(prompt=_prompt_obj)
            else:
                verbose_logger.error(
                    "[Non-blocking] Langfuse Logger: Invalid prompt format"
                )
        else:
            verbose_logger.error(
                "[Non-blocking] Langfuse Logger: Invalid prompt format. No prompt logged to Langfuse"
            )
    elif (
        prompt_management_metadata is not None
        and prompt_management_metadata["prompt_integration"] == "langfuse"
    ):
        try:
            generation_params["prompt"] = langfuse_client.get_prompt(
                prompt_management_metadata["prompt_id"]
            )
        except Exception as e:
            verbose_logger.debug(
                f"[Non-blocking] Langfuse Logger: Error getting prompt client for logging: {e}"
            )
            pass

    else:
        generation_params["prompt"] = user_prompt

    return generation_params


def log_provider_specific_information_as_span(
    trace,
    clean_metadata,
):
    """
    Logs provider-specific information as spans.

    Parameters:
        trace: The tracing object used to log spans.
        clean_metadata: A dictionary containing metadata to be logged.

    Returns:
        None
    """

    _hidden_params = clean_metadata.get("hidden_params", None)
    if _hidden_params is None:
        return

    vertex_ai_grounding_metadata = _hidden_params.get(
        "vertex_ai_grounding_metadata", None
    )

    if vertex_ai_grounding_metadata is not None:
        if isinstance(vertex_ai_grounding_metadata, list):
            for elem in vertex_ai_grounding_metadata:
                if isinstance(elem, dict):
                    for key, value in elem.items():
                        trace.span(
                            name=key,
                            input=value,
                        )
                else:
                    trace.span(
                        name="vertex_ai_grounding_metadata",
                        input=elem,
                    )
        else:
            trace.span(
                name="vertex_ai_grounding_metadata",
                input=vertex_ai_grounding_metadata,
            )


def log_requester_metadata(clean_metadata: dict):
    returned_metadata = {}
    requester_metadata = clean_metadata.get("requester_metadata") or {}
    for k, v in clean_metadata.items():
        if k not in requester_metadata:
            returned_metadata[k] = v

    returned_metadata.update({"requester_metadata": requester_metadata})

    return returned_metadata


def _strip_langfuse_generation_metadata(full_metadata: dict) -> dict:
    """Return Langfuse generation metadata without bulky durable snapshots."""

    return _compact_langfuse_generation_metadata_for_enqueue(full_metadata)
