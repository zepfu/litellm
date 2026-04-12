"""AAWM observability callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the langfuse_trace_name request header so
each agent's API calls can be distinguished in Langfuse.

The hook injects: "You are '<agent-name>' and you are working..."
When no agent designation is found, defaults to "orchestrator".

Enriches langfuse_trace_name from "claude-code" to "claude-code.<agent>"
(e.g. "claude-code.ops").

Uses BOTH logging_hook() (sync) and async_logging_hook() (async) to modify
headers BEFORE Langfuse's add_metadata_from_header() reads them. The sync
hook is critical for pass-through endpoints because Langfuse runs as a string
callback ("langfuse") in the sync success_handler - the async hook alone
would race with the thread-pool-submitted sync handler.

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["aawm_litellm_callbacks.agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]
"""

import base64
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_DEFAULT_AGENT = "orchestrator"
_CLAUDE_EXPERIMENT_ID_RE = re.compile(
    rb"(?<![A-Za-z0-9._-])([A-Za-z][A-Za-z0-9._-]{11,})(?![A-Za-z0-9._-])"
)
_GEMINI_MARKER = bytes.fromhex("8f3d6b5f")


def _content_to_text(content: Any) -> str:
    """Convert message content (string or Anthropic content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content) if content else ""


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    """Extract agent name from request content.

    Checks four sources:
    1. kwargs["messages"] - system messages only (standard LLM call path).
    2. kwargs["system"] - Anthropic pass-through transforms system to top-level kwarg.
    3. kwargs["passthrough_logging_payload"]["request_body"]["system"] - pass-through system field.
    4. kwargs["passthrough_logging_payload"]["request_body"]["messages"] - first user message.
       Claude Code SubagentStart hook injects identity into the task prompt (first user message),
       not the system prompt. The tightened regex avoids false positives.
    """
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                continue
            text = _content_to_text(message.get("content", ""))
            match = _AGENT_RE.search(text)
            if match:
                return match.group(1)

    system_direct = kwargs.get("system")
    if system_direct:
        text = _content_to_text(system_direct)
        match = _AGENT_RE.search(text)
        if match:
            return match.group(1)

    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            system = request_body.get("system")
            if system:
                text = _content_to_text(system)
                match = _AGENT_RE.search(text)
                if match:
                    return match.group(1)

            pt_messages = request_body.get("messages")
            if pt_messages and isinstance(pt_messages, list):
                for msg in pt_messages[:3]:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    text = _content_to_text(msg.get("content", ""))
                    match = _AGENT_RE.search(text)
                    if match:
                        return match.group(1)
                    break

    return _DEFAULT_AGENT


def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict:
    """Ensure proxy_server_request.headers is a mutable dict."""
    litellm_params = kwargs.get("litellm_params") or {}
    psr = litellm_params.get("proxy_server_request") or {}
    headers = psr.get("headers")

    if headers is None:
        return {}

    if not isinstance(headers, dict):
        headers = dict(headers)
        psr["headers"] = headers

    return headers


def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    litellm_params["metadata"] = metadata
    kwargs["litellm_params"] = litellm_params
    return metadata


def _merge_tags(metadata: Dict[str, Any], tags_to_add: List[str]) -> None:
    existing_tags = metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = list(existing_tags)
    for tag in tags_to_add:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)
    metadata["tags"] = merged_tags


def _sync_standard_logging_object(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        return

    standard_logging_metadata = standard_logging_object.get("metadata")
    if not isinstance(standard_logging_metadata, dict):
        standard_logging_metadata = {}
    standard_logging_metadata.update(metadata)
    standard_logging_object["metadata"] = standard_logging_metadata

    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    existing_request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []

    merged_request_tags = list(existing_request_tags)
    for tag in tags:
        if isinstance(tag, str) and tag and tag not in merged_request_tags:
            merged_request_tags.append(tag)
    standard_logging_object["request_tags"] = merged_request_tags
    kwargs["standard_logging_object"] = standard_logging_object


def _maybe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_first_response_message(result: Any) -> Any:
    choices = _maybe_get(result, "choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None

    first_choice = choices[0]
    message = _maybe_get(first_choice, "message")
    if message is not None:
        return message
    return _maybe_get(first_choice, "delta")


def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]:
    provider_specific_fields = _maybe_get(message, "provider_specific_fields")
    if isinstance(provider_specific_fields, dict):
        return provider_specific_fields
    return {}


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


def _format_langfuse_span_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_langfuse_span(
    metadata: Dict[str, Any],
    *,
    name: str,
    span_metadata: Optional[Dict[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []

    span_descriptor: Dict[str, Any] = {"name": name}
    if input_data is not None:
        span_descriptor["input"] = input_data
    if output_data is not None:
        span_descriptor["output"] = output_data
    if span_metadata:
        span_descriptor["metadata"] = span_metadata
    if start_time is not None:
        span_descriptor["start_time"] = _format_langfuse_span_timestamp(start_time)
    if end_time is not None:
        span_descriptor["end_time"] = _format_langfuse_span_timestamp(end_time)

    existing_spans.append(span_descriptor)
    metadata["langfuse_spans"] = existing_spans


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

    for match in _CLAUDE_EXPERIMENT_ID_RE.findall(decoded_bytes):
        decoded_match = match.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)
    return experiment_ids


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
    metadata["claude_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
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
            indexed_fields[
                f"gemini_tsig_0_record_{record_index}_marker_offset"
            ] = absolute_marker_offset

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


def _enrich_gemini_thought_signature_metadata(
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
        signature
        for signature in thought_signatures
        if isinstance(signature, str) and signature.strip()
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
            for key, value in indexed_fields.items():
                if key.startswith("gemini_tsig_0_"):
                    metadata[key.replace("gemini_tsig_0_", f"gemini_tsig_{index}_")] = value
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["gemini_thought_signature_present"] = len(thought_signatures) > 0
    metadata["gemini_thought_signature_count"] = len(thought_signatures)
    metadata["gemini_tsig_signature_hashes"] = signature_hashes
    metadata["gemini_tsig_shape_hashes"] = sorted(set(shape_hashes))
    metadata["gemini_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["gemini_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
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
        tags_to_add.extend(
            ["gemini-thought-signature-decoded", "thinking-signature-decoded"]
        )
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
            "record_counts": sorted(
                {summary["record_count"] for summary in summaries} if summaries else []
            ),
            "reasoning_content_present": bool(reasoning_content.strip()),
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _enrich_trace_name_and_provider_metadata(
    kwargs: Dict[str, Any], result: Any
) -> Tuple[dict, Any]:
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)
    metadata = _ensure_mutable_metadata(kwargs)

    if headers:
        current = headers.get("langfuse_trace_name")
        if current == "claude-code":
            headers["langfuse_trace_name"] = f"claude-code.{agent_name}"
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to claude-code.%s",
                agent_name,
            )

    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
    elif not current_trace_name:
        metadata["trace_name"] = agent_name

    message = _extract_first_response_message(result)
    if message is not None:
        _enrich_claude_thinking_metadata(metadata, message)
        _enrich_gemini_thought_signature_metadata(metadata, message)

    _sync_standard_logging_object(kwargs, metadata)

    verbose_logger.debug(
        "AawmAgentIdentity: agent=%s, trace_name=%s, tags=%s",
        agent_name,
        metadata.get("trace_name"),
        metadata.get("tags"),
    )
    return kwargs, result


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths:
    - Sync: pass-through endpoints run Langfuse in sync success_handler (thread pool)
    - Async: standard LLM calls run Langfuse in async_success_handler
    """

    def logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync hook - runs before Langfuse in sync success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning("AawmAgentIdentity.logging_hook failed: %s", exc)
            return kwargs, result

    async def async_logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Async hook - runs before Langfuse in async success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_logging_hook failed: %s", exc
            )
            return kwargs, result


# Module-level instance for config registration via get_instance_fn().
# Config must reference this instance name, not the class name:
#   callbacks: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]
aawm_agent_identity_instance = AawmAgentIdentity()
