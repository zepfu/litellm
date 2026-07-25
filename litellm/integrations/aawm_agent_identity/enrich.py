"""Enrichment orchestrators, worker-context exhaustion, usage breakout, thinking/signature decoding.

Behavior-preserving Wave A4B extraction from the identity package
``__init__``. This is the high-fan-in module that moves last within A4B.
Function bodies resolve free names through the identity host namespace
after :func:`install` rebinds ``__globals__`` (record.py contract), so
module-level imports of identity helpers are intentionally absent here."""

import base64
import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from litellm._logging import verbose_logger

if TYPE_CHECKING:

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict: ...

    def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]: ...

    def _enrich_provider_cache_metadata(kwargs: Dict[str, Any], result: Any) -> None: ...

    def _enrich_session_runtime_identity_metadata(kwargs: Dict[str, Any]) -> None: ...

    def _enrich_token_count_usage_metadata(kwargs: Dict[str, Any], result: Any) -> None: ...

    def _extract_agent_context(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]: ...

    def _extract_agent_id_from_kwargs(
        kwargs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        standard_logging_object: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        repository: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]: ...

    def _extract_agent_name(kwargs: Dict[str, Any]) -> str: ...

    def _extract_cache_creation_input_tokens(usage_obj: Any) -> int: ...

    def _extract_cache_read_input_tokens(usage_obj: Any) -> int: ...

    def _extract_first_response_message(result: Any) -> Any: ...

    def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]: ...

    def _extract_reported_reasoning_tokens(usage_obj: Any) -> Optional[int]: ...

    def _extract_repository_identity_from_kwargs(
        kwargs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        standard_logging_object: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]: ...

    def _extract_response_output_tool_call_info(
        result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, List[str]]: ...

    def _extract_session_id(kwargs: Dict[str, Any]) -> Optional[str]: ...

    def _extract_tenant_identity_from_kwargs(
        kwargs: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        standard_logging_object: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]: ...

    def _extract_tool_call_info(message: Any) -> Tuple[int, List[str]]: ...

    def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any: ...

    def _fallback_gemini_reasoning_tokens_from_signatures(
        metadata: Dict[str, Any],
        message: Any = None,
    ) -> Optional[int]: ...

    def _is_claude_permission_check_metadata(metadata: Any) -> bool: ...

    def _is_codex_default_agent_context(kwargs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool: ...

    def _is_generic_grok_trace_name(value: Any) -> bool: ...

    def _is_native_grok_passthrough_context(metadata: Dict[str, Any], headers: Dict[str, Any]) -> bool: ...

    def _iter_litellm_metadata_sources(
        kwargs: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _merge_tags(metadata: Dict[str, Any], tags_to_add: List[str]) -> None: ...

    def _append_langfuse_span(
        metadata: Dict[str, Any],
        *,
        name: str,
        span_metadata: Optional[Dict[str, Any]] = None,
        input_data: Any = None,
        output_data: Any = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None: ...

    def _promote_codex_repository_trace_user_id(
        kwargs: Dict[str, Any],
        metadata: Dict[str, Any],
        headers: Dict[str, Any],
    ) -> None: ...

    def _promote_grok_repository_trace_identity(
        kwargs: Dict[str, Any],
        metadata: Dict[str, Any],
        headers: Dict[str, Any],
    ) -> None: ...

    def _sync_standard_logging_object(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> None: ...

    def _apply_claude_auto_review_metadata(
        metadata: Dict[str, Any],
        *,
        repository: Optional[str] = None,
        tenant_id: Optional[str] = None,
        source_model: Optional[str] = None,
    ) -> None: ...

    def _enrich_claude_permission_check_metadata(
        kwargs: Dict[str, Any],
        metadata: Dict[str, Any],
        result: Any,
        *,
        standard_logging_object: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def _extract_claude_auto_review_source_model(
        metadata: Dict[str, Any],
        fallback_model: Optional[str] = None,
    ) -> Optional[str]: ...

_WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS = (
    "worker_context_exhaustion_failure_class",
    "worker_context_exhaustion_failure_reason",
    "worker_context_exhaustion_partial_output_summary",
    "worker_context_exhaustion_changed_paths_hint",
    "worker_context_exhaustion_attempted_patch_scope",
    "worker_context_exhaustion_last_visible_message",
    "worker_context_exhaustion_success",
    "worker_context_exhaustion_completed",
)

_WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN = {
    "worker_context_exhaustion_failure_class": 128,
    "worker_context_exhaustion_failure_reason": 512,
    "worker_context_exhaustion_partial_output_summary": 2000,
    "worker_context_exhaustion_changed_paths_hint": 2000,
    "worker_context_exhaustion_attempted_patch_scope": 2000,
    "worker_context_exhaustion_last_visible_message": 2000,
}

_WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS = frozenset(
    {
        "worker_context_exhaustion_success",
        "worker_context_exhaustion_completed",
    }
)

_GEMINI_MARKER = bytes.fromhex("8f3d6b5f")


def _extract_reasoning_content(message: Any, thinking_blocks: List[dict]) -> str:
    reasoning_content = _maybe_get(message, "reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content

    thinking_parts: List[str] = []
    for block in thinking_blocks:
        thinking_text = _maybe_get(block, "thinking")
        if isinstance(thinking_text, str) and thinking_text:
            thinking_parts.append(thinking_text)
    return "\n".join(thinking_parts)


def _extract_thinking_blocks(message: Any) -> List[dict]:
    thinking_blocks = _maybe_get(message, "thinking_blocks")
    if not isinstance(thinking_blocks, list):
        provider_specific_fields = _extract_provider_specific_fields(message)
        thinking_blocks = provider_specific_fields.get("thinking_blocks")
    if not isinstance(thinking_blocks, list):
        return []
    return [block for block in thinking_blocks if isinstance(block, dict)]


def _normalize_base64_text(value: str) -> str:
    return "".join(value.split())


def _decode_base64_bytes(value: str) -> bytes:
    normalized_value = _normalize_base64_text(value)
    padding = (-len(normalized_value)) % 4
    if padding:
        normalized_value += "=" * padding
    return base64.b64decode(normalized_value)


def _short_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:12]


def _get_reasoning_state_tags(
    provider_prefix: str,
    reasoning_content: str,
    thinking_blocks: List[dict],
) -> List[str]:
    stripped_reasoning = reasoning_content.strip()
    tags: List[str] = []
    if stripped_reasoning:
        tags.append("reasoning-present")
        tags.append(f"{provider_prefix}-reasoning-present")
    else:
        tags.append("reasoning-empty")
        tags.append(f"{provider_prefix}-reasoning-empty")

    if thinking_blocks:
        tags.append("thinking-blocks-present")
        tags.append(f"{provider_prefix}-thinking-blocks-present")
    else:
        tags.append("thinking-blocks-empty")
        tags.append(f"{provider_prefix}-thinking-blocks-empty")
    return tags


def _extract_claude_experiment_ids(decoded_bytes: bytes) -> List[str]:
    experiment_ids: List[str] = []
    for offset, current_byte in enumerate(decoded_bytes[:-2]):
        if current_byte != 0x32:
            continue
        candidate_length = decoded_bytes[offset + 1]
        candidate_start = offset + 2
        candidate_end = candidate_start + candidate_length
        if candidate_end > len(decoded_bytes):
            continue
        candidate_bytes = decoded_bytes[candidate_start:candidate_end]
        if not all(32 <= byte <= 126 for byte in candidate_bytes):
            continue
        decoded_match = candidate_bytes.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)

    if experiment_ids:
        return experiment_ids

    for match in _CLAUDE_EXPERIMENT_ID_RE.findall(decoded_bytes):  # type: ignore[name-defined]  # noqa: F821
        decoded_match = match.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)
    return experiment_ids


def _bound_worker_context_exhaustion_string(
    key: str,
    value: Any,
) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if cleaned is None:
        return None
    max_len = _WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN.get(key, 512)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def _normalize_worker_context_exhaustion_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _sanitize_worker_context_exhaustion_metadata(metadata: Dict[str, Any]) -> None:
    """Bound orchestrator worker exhaustion fields; never infer success from LLM output."""
    for key in _WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS:
        if key not in metadata:
            continue
        raw_value = metadata.get(key)
        if key in _WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS:
            normalized_bool = _normalize_worker_context_exhaustion_bool(raw_value)
            if normalized_bool is None:
                metadata.pop(key, None)
            else:
                metadata[key] = normalized_bool
            continue

        if isinstance(raw_value, list):
            bounded_items = []
            for item in raw_value[:50]:
                item_text = _bound_worker_context_exhaustion_string(key, item)
                if item_text is not None:
                    bounded_items.append(item_text)
            if bounded_items:
                metadata[key] = bounded_items
            else:
                metadata.pop(key, None)
            continue

        bounded = _bound_worker_context_exhaustion_string(key, raw_value)
        if bounded is None:
            metadata.pop(key, None)
        else:
            metadata[key] = bounded

    if metadata.get("worker_context_exhaustion_failure_class"):
        metadata["worker_context_exhaustion_success"] = False
        metadata["worker_context_exhaustion_completed"] = False


def _promote_worker_context_exhaustion_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """Copy allowlisted worker exhaustion keys from upstream litellm_metadata without overwriting."""
    for source in _iter_litellm_metadata_sources(kwargs, metadata):
        for key in _WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS:
            if key in metadata:
                continue
            if key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            metadata[key] = value
    _sanitize_worker_context_exhaustion_metadata(metadata)


def _infer_usage_breakout_provider_prefix(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if route_family_lower == "codex_responses" or route_family_lower.startswith("codex_"):
            return "codex"
        if "gemini" in route_family_lower:
            return "gemini"

    provider = kwargs.get("custom_llm_provider")
    if isinstance(provider, str) and provider.strip():
        provider_lower = provider.lower()
        if provider_lower == "gemini":
            return "gemini"

    model = kwargs.get("model")
    if isinstance(model, str) and model.strip():
        model_lower = model.lower()
        if "gemini" in model_lower:
            return "gemini"
        if "codex" in model_lower:
            return "codex"

    return None


def _enrich_usage_breakout_metadata(kwargs: Dict[str, Any], result: Any) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    provider_prefix = _infer_usage_breakout_provider_prefix(kwargs, metadata)
    if provider_prefix is None:
        return

    usage_obj = _extract_usage_object(kwargs, result)
    if usage_obj is None:
        return

    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)
    reasoning_tokens_source: Optional[str] = None
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)

    message = _extract_first_response_message(result)
    if reported_reasoning_tokens is not None:
        reasoning_tokens_source = "provider_reported"
    elif provider_prefix == "gemini":
        reported_reasoning_tokens = _fallback_gemini_reasoning_tokens_from_signatures(
            metadata,
            message,
        )
        if reported_reasoning_tokens is not None:
            reasoning_tokens_source = "provider_signature_present"

    tool_call_count, tool_names = _extract_tool_call_info(message)
    if tool_call_count == 0:
        tool_call_count, tool_names = _extract_response_output_tool_call_info(
            result,
            kwargs.get("standard_logging_object"),
        )

    metadata["usage_cache_read_input_tokens"] = cache_read_input_tokens
    metadata["usage_cache_creation_input_tokens"] = cache_creation_input_tokens
    metadata["usage_tool_call_count"] = tool_call_count
    metadata["usage_tool_names"] = tool_names
    metadata[f"{provider_prefix}_cache_read_input_tokens"] = cache_read_input_tokens
    metadata[f"{provider_prefix}_cache_creation_input_tokens"] = cache_creation_input_tokens
    metadata[f"{provider_prefix}_tool_call_count"] = tool_call_count
    metadata[f"{provider_prefix}_tool_names"] = tool_names

    if reported_reasoning_tokens is not None:
        metadata["usage_reasoning_tokens_reported"] = reported_reasoning_tokens
        metadata["usage_reasoning_tokens_source"] = reasoning_tokens_source or "provider_reported"
        metadata[f"{provider_prefix}_reasoning_tokens_reported"] = reported_reasoning_tokens

    tags_to_add = [f"{provider_prefix}-usage-breakout"]
    if reported_reasoning_tokens is not None:
        tags_to_add.extend(["reasoning-tokens-reported", f"{provider_prefix}-reasoning-tokens-reported"])
    if cache_read_input_tokens > 0:
        tags_to_add.extend(["cache-read-input-tokens", f"{provider_prefix}-cache-read-input-tokens"])
    if cache_creation_input_tokens > 0:
        tags_to_add.extend(
            [
                "cache-creation-input-tokens",
                f"{provider_prefix}-cache-creation-input-tokens",
            ]
        )
    if tool_call_count > 0:
        tags_to_add.extend(["tool-calls-present", f"{provider_prefix}-tool-calls-present"])
    _merge_tags(metadata, tags_to_add)

    _append_langfuse_span(
        metadata,
        name=f"{provider_prefix}.usage_breakout",
        span_metadata={
            "reported_reasoning_tokens": reported_reasoning_tokens,
            "reported_reasoning_tokens_source": reasoning_tokens_source,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "tool_call_count": tool_call_count,
            "tool_names": tool_names,
        },
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )


def _enrich_claude_thinking_metadata(metadata: Dict[str, Any], message: Any) -> None:
    span_started_at = datetime.now(timezone.utc)
    thinking_blocks = _extract_thinking_blocks(message)
    if not thinking_blocks:
        return
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    signatures: List[str] = []
    for block in thinking_blocks:
        if _maybe_get(block, "type") != "thinking":
            continue
        signature = _maybe_get(block, "signature")
        if isinstance(signature, str) and signature.strip():
            signatures.append(signature)

    if not signatures:
        return

    decoded_hashes: List[str] = []
    experiment_ids: List[str] = []
    decode_errors: List[str] = []
    decoded_any = False

    for signature in signatures:
        try:
            decoded_bytes = _decode_base64_bytes(signature)
            decoded_hashes.append(_short_hash(decoded_bytes))
            decoded_any = True
            for experiment_id in _extract_claude_experiment_ids(decoded_bytes):
                if experiment_id not in experiment_ids:
                    experiment_ids.append(experiment_id)
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["claude_thinking_signature_present"] = len(signatures) > 0
    metadata["claude_thinking_signature_count"] = len(signatures)
    metadata["claude_thinking_signature_hashes"] = decoded_hashes
    metadata["claude_thinking_signature_decoded"] = decoded_any
    metadata["claude_thinking_decode_version"] = "v1"
    metadata["claude_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["claude_reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    if experiment_ids:
        metadata["claude_thinking_experiment_ids"] = experiment_ids
        if len(experiment_ids) == 1:
            metadata["claude_thinking_experiment_id"] = experiment_ids[0]
    if decode_errors:
        metadata["claude_thinking_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = decoded_any
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["claude-thinking-signature", "thinking-signature-present"]
    if decoded_any:
        tags_to_add.extend(["claude-thinking-decoded", "thinking-signature-decoded"])
    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="claude",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    tags_to_add.extend(f"claude-exp:{experiment_id}" for experiment_id in experiment_ids)
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="claude.thinking_signature_decode",
        span_metadata={
            "signature_count": len(signatures),
            "decoded_signature_count": len(decoded_hashes),
            "thinking_block_count": len(thinking_blocks),
            "reasoning_content_present": bool(reasoning_content.strip()),
            "experiment_ids": experiment_ids,
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _read_varint(data: bytes, offset: int) -> Tuple[Optional[int], int]:
    value = 0
    shift = 0
    current_offset = offset
    while current_offset < len(data):
        current_byte = data[current_offset]
        value |= (current_byte & 0x7F) << shift
        current_offset += 1
        if current_byte < 0x80:
            return value, current_offset
        shift += 7
        if shift > 63:
            break
    return None, offset


def _extract_gemini_signature_summary(signature: str) -> Dict[str, Any]:
    decoded_bytes = _decode_base64_bytes(signature)
    signature_hash = _short_hash(decoded_bytes)

    record_sizes: List[int] = []
    prefixes: List[str] = []
    marker_offsets: List[int] = []
    indexed_fields: Dict[str, Any] = {}

    offset = 0
    record_index = 0
    while offset < len(decoded_bytes):
        if decoded_bytes[offset] != 0x0A:
            break
        record_size, payload_offset = _read_varint(decoded_bytes, offset + 1)
        if record_size is None:
            break
        payload_end = payload_offset + record_size
        if payload_end > len(decoded_bytes):
            break

        payload = decoded_bytes[payload_offset:payload_end]
        marker_index = payload.find(_GEMINI_MARKER)
        prefix_hex = ""
        absolute_marker_offset = None
        if marker_index >= 0:
            prefix_hex = payload[:marker_index].hex()
            absolute_marker_offset = payload_offset + marker_index
            marker_offsets.append(absolute_marker_offset)

        record_sizes.append(record_size)
        prefixes.append(prefix_hex)
        indexed_fields[f"gemini_tsig_0_record_{record_index}_size"] = record_size
        indexed_fields[f"gemini_tsig_0_record_{record_index}_prefix"] = prefix_hex
        if absolute_marker_offset is not None:
            indexed_fields[f"gemini_tsig_0_record_{record_index}_marker_offset"] = absolute_marker_offset

        record_index += 1
        offset = payload_end

    shape_components = {
        "decoded_bytes": len(decoded_bytes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
    }
    shape_hash = _short_hash(str(shape_components).encode("utf-8"))

    summary: Dict[str, Any] = {
        "decoded_bytes": len(decoded_bytes),
        "record_count": len(record_sizes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
        "marker_hex": _GEMINI_MARKER.hex(),
        "shape_hash": shape_hash,
        "signature_hash": signature_hash,
        "indexed_fields": indexed_fields,
    }
    return summary


def _enrich_gemini_thought_signature_metadata(  # noqa: PLR0915
    metadata: Dict[str, Any], message: Any
) -> None:
    span_started_at = datetime.now(timezone.utc)
    provider_specific_fields = _extract_provider_specific_fields(message)
    thought_signatures = provider_specific_fields.get("thought_signatures")
    thinking_blocks = _extract_thinking_blocks(message)
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    if not isinstance(thought_signatures, list):
        thought_signatures = []
    thought_signatures = [
        signature for signature in thought_signatures if isinstance(signature, str) and signature.strip()
    ]

    if not thought_signatures:
        return

    summaries: List[Dict[str, Any]] = []
    decode_errors: List[str] = []
    signature_hashes: List[str] = []
    shape_hashes: List[str] = []

    for index, signature in enumerate(thought_signatures):
        try:
            summary = _extract_gemini_signature_summary(signature)
            summaries.append(summary)
            signature_hashes.append(summary["signature_hash"])
            shape_hashes.append(summary["shape_hash"])
            metadata[f"gemini_tsig_{index}_decoded_bytes"] = summary["decoded_bytes"]
            metadata[f"gemini_tsig_{index}_record_count"] = summary["record_count"]
            metadata[f"gemini_tsig_{index}_record_sizes"] = summary["record_sizes"]
            metadata[f"gemini_tsig_{index}_prefixes"] = summary["prefixes"]
            metadata[f"gemini_tsig_{index}_marker_offsets"] = summary["marker_offsets"]
            metadata[f"gemini_tsig_{index}_marker_hex"] = summary["marker_hex"]
            metadata[f"gemini_tsig_{index}_shape_hash"] = summary["shape_hash"]

            indexed_fields = summary["indexed_fields"]
            for key, value in list(indexed_fields.items()):
                if key.startswith("gemini_tsig_0_"):
                    metadata[key.replace("gemini_tsig_0_", f"gemini_tsig_{index}_")] = value
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["gemini_thought_signature_present"] = len(thought_signatures) > 0
    metadata["gemini_thought_signature_count"] = len(thought_signatures)
    metadata["gemini_tsig_signature_hashes"] = signature_hashes
    metadata["gemini_tsig_shape_hashes"] = sorted(set(shape_hashes))
    metadata["gemini_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["gemini_reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["gemini_thinking_blocks_present"] = len(thinking_blocks) > 0
    if summaries:
        first_summary = summaries[0]
        metadata["gemini_tsig_decoded_bytes"] = first_summary["decoded_bytes"]
        metadata["gemini_tsig_record_count"] = first_summary["record_count"]
        metadata["gemini_tsig_record_sizes"] = first_summary["record_sizes"]
        metadata["gemini_tsig_prefixes"] = first_summary["prefixes"]
        metadata["gemini_tsig_marker_offsets"] = first_summary["marker_offsets"]
        metadata["gemini_tsig_marker_hex"] = first_summary["marker_hex"]
        metadata["gemini_tsig_shape_hash"] = first_summary["shape_hash"]
    if decode_errors:
        metadata["gemini_tsig_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = len(summaries) > 0
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["gemini-thought-signature", "thinking-signature-present"]
    if summaries:
        tags_to_add.extend(["gemini-thought-signature-decoded", "thinking-signature-decoded"])
        for shape_hash in sorted(set(shape_hashes)):
            tags_to_add.append(f"gemini-tsig-shape:{shape_hash}")
        for record_count in sorted({summary["record_count"] for summary in summaries}):
            tags_to_add.append(f"gemini-tsig-records:{record_count}")

    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="gemini",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="gemini.thought_signature_decode",
        span_metadata={
            "signature_count": len(thought_signatures),
            "decoded_signature_count": len(summaries),
            "shape_hashes": sorted(set(shape_hashes)),
            "record_counts": sorted({summary["record_count"] for summary in summaries} if summaries else []),
            "reasoning_content_present": bool(reasoning_content.strip()),
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _enrich_agent_identity_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    if (
        _is_codex_default_agent_context(kwargs, metadata)
        and not _clean_non_empty_string(metadata.get("agent_name"))
        and not _clean_non_empty_string(metadata.get("aawm_claude_agent_name"))
    ):
        metadata["agent_name"] = _DEFAULT_AGENT  # type: ignore[name-defined]  # noqa: F821

    agent_context_name, agent_context_tenant_id = _extract_agent_context(kwargs)
    agent_id_repository = _extract_repository_identity_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=kwargs.get("standard_logging_object") or {},
    )
    agent_id, agent_id_source = _extract_agent_id_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=kwargs.get("standard_logging_object") or {},
        agent_name=agent_context_name,
        tenant_id=agent_context_tenant_id,
        repository=agent_id_repository,
    )
    if agent_id:
        metadata["agent_id"] = agent_id
        if agent_id_source:
            metadata["agent_id_source"] = agent_id_source
    else:
        metadata.pop("agent_id", None)
        metadata.pop("agent_id_source", None)


def _enrich_trace_name_and_provider_metadata(kwargs: Dict[str, Any], result: Any) -> Tuple[dict, Any]:
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)
    metadata = _ensure_mutable_metadata(kwargs)
    session_id = _extract_session_id(kwargs)
    is_grok_context = _is_native_grok_passthrough_context(metadata, headers)
    _enrich_claude_permission_check_metadata(kwargs, metadata, result)
    if _is_claude_permission_check_metadata(metadata):
        direct_repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        direct_tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        _apply_claude_auto_review_metadata(
            metadata,
            repository=direct_repository,
            tenant_id=direct_tenant_id,
            source_model=_extract_claude_auto_review_source_model(
                metadata,
                _clean_non_empty_string(kwargs.get("model")),
            ),
        )

    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
    elif is_grok_context and (not current_trace_name or _is_generic_grok_trace_name(current_trace_name)):
        metadata["trace_name"] = (
            f"grok-build.{agent_name}" if agent_name and agent_name != _DEFAULT_AGENT else "grok-build"  # type: ignore[name-defined]  # noqa: F821
        )
    elif not current_trace_name:
        metadata["trace_name"] = agent_name
    child_trace_user_id = _clean_non_empty_string(metadata.get("trace_user_id"))
    child_trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    if headers and child_trace_name and child_trace_name.startswith("claude-code."):
        current_trace_name_header = _clean_non_empty_string(headers.get("langfuse_trace_name"))
        if (
            current_trace_name_header is None
            or current_trace_name_header == "claude-code"
            or current_trace_name_header.startswith("claude-code.")
        ) and current_trace_name_header != child_trace_name:
            headers["langfuse_trace_name"] = child_trace_name
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to %s",
                child_trace_name,
            )
    if headers and is_grok_context and child_trace_name:
        current_trace_name_header = _clean_non_empty_string(headers.get("langfuse_trace_name"))
        if (
            current_trace_name_header is None or _is_generic_grok_trace_name(current_trace_name_header)
        ) and current_trace_name_header != child_trace_name:
            headers["langfuse_trace_name"] = child_trace_name
            verbose_logger.debug(
                "AawmAgentIdentity: enriched Grok header trace_name to %s",
                child_trace_name,
            )
    if headers and child_trace_user_id and child_trace_name and child_trace_name.startswith("claude-code."):
        current_trace_user_id = headers.get("langfuse_trace_user_id")
        if current_trace_user_id != child_trace_user_id:
            headers["langfuse_trace_user_id"] = child_trace_user_id
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_user_id to %s",
                child_trace_user_id,
            )
    if session_id and not metadata.get("session_id"):
        metadata["session_id"] = session_id

    _promote_codex_repository_trace_user_id(kwargs, metadata, headers)
    _promote_grok_repository_trace_identity(kwargs, metadata, headers)
    _enrich_agent_identity_metadata(kwargs, metadata)
    _enrich_session_runtime_identity_metadata(kwargs)

    message = _extract_first_response_message(result)
    if message is not None:
        _enrich_claude_thinking_metadata(metadata, message)
        _enrich_gemini_thought_signature_metadata(metadata, message)
    _enrich_token_count_usage_metadata(kwargs, result)
    _enrich_usage_breakout_metadata(kwargs, result)
    _enrich_provider_cache_metadata(kwargs, result)

    _sync_standard_logging_object(kwargs, metadata)

    verbose_logger.debug(
        "AawmAgentIdentity: agent=%s, trace_name=%s, tags=%s",
        agent_name,
        metadata.get("trace_name"),
        metadata.get("tags"),
    )
    return kwargs, result


_HOST_FUNCTION_NAMES = (
    "_bound_worker_context_exhaustion_string",
    "_normalize_worker_context_exhaustion_bool",
    "_sanitize_worker_context_exhaustion_metadata",
    "_promote_worker_context_exhaustion_metadata",
    "_infer_usage_breakout_provider_prefix",
    "_enrich_usage_breakout_metadata",
    "_enrich_claude_thinking_metadata",
    "_read_varint",
    "_extract_gemini_signature_summary",
    "_enrich_gemini_thought_signature_metadata",
    "_enrich_agent_identity_metadata",
    "_enrich_trace_name_and_provider_metadata",
    "_get_reasoning_state_tags",
    "_extract_claude_experiment_ids",
    "_extract_reasoning_content",
    "_extract_thinking_blocks",
    "_normalize_base64_text",
    "_decode_base64_bytes",
    "_short_hash",
)


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def _rebind_installable_callable(value, host_globals):
    if isinstance(value, _FunctionType):
        return _rebind_to_host_globals(value, host_globals)

    wrapped = getattr(value, "__wrapped__", None)
    cache_parameters = getattr(value, "cache_parameters", None)
    if not isinstance(wrapped, _FunctionType) or not callable(cache_parameters):
        return value

    parameters = cache_parameters()
    if not isinstance(parameters, dict) or not {"maxsize", "typed"} <= parameters.keys():
        return value

    rebound_wrapped = _rebind_to_host_globals(wrapped, host_globals)
    rebound = lru_cache(
        maxsize=parameters["maxsize"],
        typed=bool(parameters["typed"]),
    )(rebound_wrapped)
    for attribute, attribute_value in getattr(value, "__dict__", {}).items():
        if attribute != "__wrapped__":
            setattr(rebound, attribute, attribute_value)
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
