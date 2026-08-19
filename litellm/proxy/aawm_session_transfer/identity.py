"""Content-free identity extraction for session-transfer records."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from litellm.proxy.aawm_session_transfer.schema import (
    coerce_non_negative_int,
    empty_prompt_category_tokens,
    sanitize_identity,
    sanitize_label,
    sanitize_prompt_category_tokens,
    sanitize_route_label,
)

_PROMPT_CATEGORY_SOURCE_MAP = {
    "system": "input_system_tokens_estimated",
    "tool_advertisement": "input_tool_advertisement_tokens_estimated",
    "conversation": "input_conversation_tokens_estimated",
    "other": "input_other_tokens_estimated",
    "residual": "input_breakdown_residual_tokens",
    "system_behavior": "system_behavior_tokens_estimated",
    "system_safety": "system_safety_tokens_estimated",
    "system_instructional": "system_instructional_tokens_estimated",
    "system_unclassified": "system_unclassified_tokens_estimated",
}


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_value(sources: list[Mapping[str, Any]], *keys: str) -> Optional[str]:
    for source in sources:
        for key in keys:
            cleaned = sanitize_identity(source.get(key))
            if cleaned:
                return cleaned
    return None


def _metadata_from_kwargs(kwargs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(kwargs, Mapping):
        return {}
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, Mapping):
            return dict(metadata)
    metadata = kwargs.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _metadata_from_request_body(request_body: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(request_body, Mapping):
        return {}
    merged: Dict[str, Any] = {}
    for key in ("litellm_metadata", "metadata"):
        value = request_body.get(key)
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def _logging_details(logging_obj: Any) -> Dict[str, Any]:
    details = getattr(logging_obj, "model_call_details", None)
    return dict(details) if isinstance(details, Mapping) else {}


def extract_transfer_identity(
    *,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    logging_obj: Any = None,
    kwargs: Optional[Mapping[str, Any]] = None,
    litellm_call_id: Optional[str] = None,
    url_route: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    stream_path: str = "unknown",
) -> Dict[str, Any]:
    """Collect queryable identities without copying request/response content."""
    body = _as_mapping(request_body)
    metadata = _metadata_from_request_body(body)
    kwargs_metadata = _metadata_from_kwargs(kwargs)
    metadata.update(kwargs_metadata)
    details = _logging_details(logging_obj)
    request_state = getattr(request, "state", None)
    state_values: Dict[str, Any] = {}
    if request_state is not None:
        for attr in (
            "litellm_call_id",
            "litellm_trace_id",
            "litellm_session_id",
            "aawm_canonical_session_id",
            "aawm_codex_session_id",
        ):
            value = getattr(request_state, attr, None)
            if value is not None:
                state_values[attr] = value

    sources = [state_values, metadata, details, _as_mapping(kwargs), body]
    call_id = sanitize_identity(
        litellm_call_id
        or _first_value(sources, "litellm_call_id", "call_id", "request_id")
        or getattr(logging_obj, "litellm_call_id", None)
    )
    session_id = _first_value(
        sources,
        "session_id",
        "aawm_session_id",
        "codex_session_id",
        "claude_session_id",
        "litellm_session_id",
    )
    codex_session_id = _first_value(
        sources,
        "codex_session_id",
        "aawm_codex_session_id",
        "session_id",
        "aawm_session_id",
    )
    canonical_session_id = _first_value(
        sources,
        "canonical_session_id",
        "aawm_canonical_session_id",
        "effective_session_identity",
    ) or session_id or codex_session_id

    prompt_categories = empty_prompt_category_tokens()
    overhead = metadata.get("prompt_overhead")
    if isinstance(overhead, Mapping):
        prompt_categories = sanitize_prompt_category_tokens(
            {
                field: overhead.get(source_key)
                for field, source_key in _PROMPT_CATEGORY_SOURCE_MAP.items()
            }
        )
    else:
        prompt_categories = sanitize_prompt_category_tokens(
            {
                field: metadata.get(source_key)
                for field, source_key in _PROMPT_CATEGORY_SOURCE_MAP.items()
            }
        )

    estimated_input = coerce_non_negative_int(
        metadata.get("estimated_input_tokens")
        or metadata.get("prompt_tokens_estimated")
        or details.get("prompt_tokens")
    )
    estimated_output = coerce_non_negative_int(
        metadata.get("estimated_output_tokens")
        or metadata.get("completion_tokens_estimated")
    )
    provider_input = coerce_non_negative_int(
        metadata.get("provider_input_tokens") or metadata.get("prompt_tokens")
    )
    provider_output = coerce_non_negative_int(
        metadata.get("provider_output_tokens") or metadata.get("completion_tokens")
    )
    context_window = coerce_non_negative_int(
        metadata.get("context_window") or metadata.get("max_input_tokens")
    )
    remaining = None
    used_input = provider_input if provider_input is not None else estimated_input
    if context_window is not None and used_input is not None and context_window >= used_input:
        remaining = context_window - used_input

    return {
        "litellm_call_id": call_id,
        "trace_id": _first_value(sources, "trace_id", "litellm_trace_id"),
        "canonical_session_id": canonical_session_id,
        "session_id": session_id,
        "codex_session_id": codex_session_id,
        "agent_id": _first_value(
            sources,
            "agent_id",
            "aawm_agent_id",
            "codex_agent_id",
            "claude_agent_id",
            "subagent_id",
            "source_agent_id",
        ),
        "agent_name": sanitize_label(
            _first_value(sources, "agent_name", "aawm_agent_name", "codex_agent_name")
        ),
        "parent_agent_id": _first_value(
            sources,
            "parent_agent_id",
            "aawm_parent_agent_id",
            "parent_actor_id",
        ),
        "parent_session_id": _first_value(
            sources,
            "parent_session_id",
            "aawm_parent_session_id",
            "parent_session_identity",
        ),
        "provider": sanitize_label(
            custom_llm_provider
            or _first_value(sources, "custom_llm_provider", "provider")
        ),
        "model": sanitize_label(
            _first_value(
                sources,
                "model",
                "anthropic_auto_agent_selected_model",
                "codex_auto_agent_selected_model",
            )
        ),
        "route": sanitize_route_label(
            url_route
            or _first_value(sources, "route", "url_route", "endpoint")
        ),
        "stream_path": stream_path,
        "context_window": context_window,
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "provider_input_tokens": provider_input,
        "provider_output_tokens": provider_output,
        "remaining_tokens": remaining,
        "request_count": coerce_non_negative_int(metadata.get("request_count")) or 1,
        "cumulative_input_tokens": coerce_non_negative_int(
            metadata.get("cumulative_input_tokens")
        ),
        "repeated_prefix_tokens": coerce_non_negative_int(
            metadata.get("repeated_prefix_tokens")
            or metadata.get("cached_tokens")
            or metadata.get("prompt_cache_tokens")
        ),
        "prompt_category_tokens": prompt_categories,
    }
