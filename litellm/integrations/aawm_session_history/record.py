"""AAWM session_history record builders and persist entrypoints.

Owns live/Langfuse/spend-log record construction, shared field derivation,
DB payload shaping, and asyncpg persist helpers for session_history.

Identity extraction primitives remain in `aawm_agent_identity`. Record APIs are
defined here as ordinary Python functions (real source, real line numbers), then
rebound at install time so each function's ``__globals__`` is the identity
module dict. That preserves the historical monkeypatch contract:

- tests patch helpers on `litellm.integrations.aawm_agent_identity`
- free-name lookup inside record APIs resolves those patches
- callables are re-exported on both this module and the identity module

This is intentional dual-binding of *function objects*, not source-string
relocation via compile/exec. ``inspect.getsource`` / coverage / tracebacks use
this file path.
"""

# Host free-names (e.g. _safe_int, extractors) resolve at runtime after
# FunctionType rebind onto aawm_agent_identity globals. Static F821 is expected.
# ruff: noqa: F821
# ruff: noqa: PLR0915


from __future__ import annotations

import inspect
import sys
from types import FunctionType
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger

# SQL constants (package-owned)
from litellm.integrations.aawm_session_history.sql import (  # noqa: F401
    _AAWM_SESSION_HISTORY_INSERT_SQL,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL,
    _AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOT_INSERT_SQL,
    _AAWM_RATE_LIMIT_OBSERVATION_INSERT_SQL,
    _AAWM_RATE_LIMIT_TRANSITION_INSERT_SQL,
    _AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL,
    _AAWM_ALIAS_ROUTING_AUDIT_INSERT_SQL,
    _AAWM_SESSION_HISTORY_PREVIOUS_GAP_UPDATE_SQL,
    _AAWM_CLAUDE_AUTO_REVIEW_PARENT_IDENTITY_SQL,
    _AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY,
    _SESSION_HISTORY_PREVIOUS_GAP_FIELD,
)


def _host():
    from litellm.integrations import aawm_agent_identity as identity

    return identity


def _h(name: str):
    return getattr(_host(), name)


# Names installed onto this module and re-exported by aawm_agent_identity.
# Keep in sync with the function/class definitions below.
_RECORD_API_NAMES: Tuple[str, ...] = (
    "_build_provider_error_observation_only_record",
    "_build_structured_output_failure_session_history_record",
    "_build_failure_observation_only_record",
    "_derive_session_history_reasoning_fields",
    "_derive_session_history_tool_fields",
    "_derive_session_history_provider_cache_fields",
    "_build_kimi_code_reference_cost_metadata",
    "_build_alibaba_token_plan_cost_metadata",
    "_build_session_history_record_from_spend_log_row",
    "_build_session_history_record_from_langfuse_trace_observation",
    "_build_session_history_metadata",
    "_build_session_history_record",
    "_build_session_history_db_payload",
    "_build_tool_activity_db_payloads",
    "_build_tool_definition_snapshot_db_payloads",
    "_persist_tool_definition_snapshots_best_effort",
    "_persist_alias_routing_audit_best_effort",
    "_build_rate_limit_observation_only_record",
    "_persist_rate_limit_observations_best_effort",
    "_persist_provider_error_observations_best_effort",
    "_session_history_transaction",
    "_persist_session_history_record",
    "_persist_session_history_records",
    "_handle_session_history_success_event",
    "_handle_session_history_failure_event",
    "_NullSessionHistoryTransaction",
)

# --- _build_provider_error_observation_only_record ---
def _build_provider_error_observation_only_record(
    kwargs: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = _merged_rate_limit_metadata(kwargs)
    model = _resolve_rate_limit_model(kwargs, None, metadata)
    return {
        "_skip_session_history": True,
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "session_id": _extract_session_id(kwargs),
        "model": model,
        "provider_error_observations": [observation],
    }

# --- _build_structured_output_failure_session_history_record ---
def _build_structured_output_failure_session_history_record(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> Optional[Dict[str, Any]]:
    metadata = _merged_rate_limit_metadata(kwargs)
    request_body = _extract_provider_cache_request_body(kwargs)
    structured_output_state = _detect_structured_output_request(request_body, metadata)
    if not structured_output_state.get("structured_output_attempted"):
        return None

    failure_reason = _first_non_empty_string(
        structured_output_state.get("structured_output_failure_reason"),
        _classify_structured_output_failure(result),
    )
    structured_output_state["structured_output_failed"] = bool(
        structured_output_state.get("structured_output_failed") or failure_reason
    )
    structured_output_state["structured_output_failure_reason"] = failure_reason
    _ensure_mutable_metadata(kwargs)["source_status"] = "failure"

    record = _build_session_history_record(
        kwargs=kwargs,
        result={},
        start_time=start_time,
        end_time=end_time,
    )
    if record is None:
        return None

    record.update(structured_output_state)
    return _normalize_session_history_record(record)

# --- _build_failure_observation_only_record ---
def _build_failure_observation_only_record(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> Optional[Dict[str, Any]]:
    failure_result = result
    if failure_result is None:
        failure_result = kwargs.get("exception") or kwargs.get("original_exception")
    rate_limit_observations = _build_rate_limit_observations(
        kwargs=kwargs,
        result=failure_result,
        start_time=start_time,
        end_time=end_time,
    )
    provider_error_observation = _build_provider_error_observation(
        kwargs=kwargs,
        result=failure_result,
        start_time=start_time,
        end_time=end_time,
    )
    structured_output_record = _build_structured_output_failure_session_history_record(
        kwargs=kwargs,
        result=failure_result,
        start_time=start_time,
        end_time=end_time,
    )
    if (
        not rate_limit_observations
        and provider_error_observation is None
        and structured_output_record is None
    ):
        return None

    if structured_output_record is not None:
        record = structured_output_record
        if rate_limit_observations:
            record["rate_limit_observations"] = rate_limit_observations
    elif rate_limit_observations:
        record = _build_rate_limit_observation_only_record(
            kwargs,
            rate_limit_observations,
        )
    else:
        if provider_error_observation is None:
            return None
        record = _build_provider_error_observation_only_record(
            kwargs,
            provider_error_observation,
        )
    if provider_error_observation is not None:
        record["provider_error_observations"] = [provider_error_observation]
    return record

# --- _derive_session_history_reasoning_fields ---
def _derive_session_history_reasoning_fields(
    *,
    metadata: Dict[str, Any],
    message: Any,
    resolved_model: str,
    provider: Optional[str],
    reported_reasoning_tokens: Optional[int],
    provider_reported_reasoning_tokens: Optional[int],
) -> Dict[str, Any]:
    """Shared reasoning-token derivation for live and backfill builders."""
    provider_name = str(provider or "").strip().lower()
    effective_reported = reported_reasoning_tokens
    if effective_reported is None and provider_name in {"gemini", "google"}:
        effective_reported = _fallback_gemini_reasoning_tokens_from_signatures(
            metadata,
            message,
        )
    thinking_blocks = _extract_thinking_blocks(message) if message is not None else []
    reasoning_text = (
        _extract_reasoning_content(message, thinking_blocks)
        if message is not None
        else ""
    )
    reasoning_present = bool(
        (isinstance(reasoning_text, str) and reasoning_text.strip())
        or thinking_blocks
        or metadata.get("reasoning_content_present")
        or (effective_reported and effective_reported > 0)
    )
    estimated_reasoning_tokens = None
    if effective_reported is None and reasoning_present:
        estimated_reasoning_tokens = _estimate_reasoning_tokens(
            model=resolved_model,
            reasoning_text=reasoning_text,
        )
    reasoning_tokens_source = _determine_reasoning_tokens_source(
        provider_reported_reasoning_tokens=provider_reported_reasoning_tokens,
        reported_reasoning_tokens=effective_reported,
        estimated_reasoning_tokens=estimated_reasoning_tokens,
        reasoning_present=reasoning_present,
    )
    return {
        "reported_reasoning_tokens": effective_reported,
        "estimated_reasoning_tokens": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_blocks": thinking_blocks,
        "reasoning_text": reasoning_text,
    }

# --- _derive_session_history_tool_fields ---
def _derive_session_history_tool_fields(
    *,
    message: Any,
    request_body: Any,
    metadata: Dict[str, Any],
    output_payload: Any = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
    observation_tool_name_fallbacks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Shared tool-activity derivation for live and backfill builders."""
    tool_call_count, tool_names = _extract_tool_call_info(message)
    tool_activity = (
        _extract_tool_activity_from_message(message) if message is not None else []
    )
    if tool_call_count == 0:
        if standard_logging_object is not None:
            output_tool_call_count, output_tool_names = (
                _extract_response_output_tool_call_info(
                    output_payload,
                    standard_logging_object,
                )
            )
        else:
            output_tool_call_count, output_tool_names = (
                _extract_response_output_tool_call_info(output_payload)
            )
        if output_tool_call_count > 0:
            tool_call_count, tool_names = output_tool_call_count, output_tool_names
    if not tool_activity:
        if standard_logging_object is not None:
            tool_activity = _extract_response_output_tool_activity(
                output_payload,
                standard_logging_object,
            )
        else:
            tool_activity = _extract_response_output_tool_activity(output_payload)

    if tool_call_count == 0 and observation_tool_name_fallbacks:
        for fallback_tool_names in observation_tool_name_fallbacks:
            if not isinstance(fallback_tool_names, list):
                continue
            normalized_tool_names = [
                str(tool_name)
                for tool_name in fallback_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names
                break

    if tool_call_count == 0:
        metadata_tool_names = metadata.get("usage_tool_names")
        if isinstance(metadata_tool_names, list):
            normalized_tool_names = [
                str(tool_name)
                for tool_name in metadata_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names

    invalid_tool_call_count = max(
        _extract_invalid_tool_call_count_from_request_body(request_body),
        _safe_int(metadata.get("usage_invalid_tool_call_count")) or 0,
    )
    structured_output_state = _detect_structured_output_request(request_body, metadata)
    tool_activity_summary = _summarize_tool_activity(tool_activity)
    return {
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "tool_activity": tool_activity,
        "tool_activity_summary": tool_activity_summary,
        "invalid_tool_call_count": invalid_tool_call_count,
        "structured_output_state": structured_output_state,
    }

# --- _derive_session_history_provider_cache_fields ---
def _derive_session_history_provider_cache_fields(
    *,
    provider: Any,
    model: str,
    usage_obj: Any,
    metadata: Dict[str, Any],
    request_body: Any,
    response_cost_usd: Optional[float],
    provider_family: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared provider-cache + miss-cost derivation for live and backfill builders."""
    provider_cache_state = _resolve_provider_cache_state(
        provider=provider,
        model=model,
        usage_obj=usage_obj,
        metadata=metadata,
        request_body=request_body,
    )
    provider_cache_state = dict(provider_cache_state or {})
    if provider_cache_state:
        family = provider_family
        if family is None:
            family = _normalize_provider_cache_family(provider, model, metadata)
        provider_cache_state.update(
            _compute_provider_cache_miss_cost_state(
                provider_family=family,
                model=model,
                usage_obj=usage_obj,
                cache_state=provider_cache_state,
                metadata=metadata,
                response_cost_usd=response_cost_usd,
            )
        )
    return {
        "provider_cache_state": provider_cache_state,
        "provider_cache_attempted": bool(
            provider_cache_state and provider_cache_state.get("attempted")
        ),
        "provider_cache_status": (
            provider_cache_state.get("status") if provider_cache_state else None
        ),
        "provider_cache_miss": bool(
            provider_cache_state and provider_cache_state.get("miss")
        ),
        "provider_cache_miss_reason": (
            provider_cache_state.get("miss_reason") if provider_cache_state else None
        ),
        "provider_cache_miss_token_count": (
            provider_cache_state.get("miss_token_count")
            if provider_cache_state
            else None
        ),
        "provider_cache_miss_cost_usd": (
            provider_cache_state.get("miss_cost_usd") if provider_cache_state else None
        ),
    }


# --- _build_kimi_code_reference_cost_metadata ---
def _build_kimi_code_reference_cost_metadata(
    *,
    provider: Optional[str],
    model: str,
    prompt_tokens: int,
    cache_read_input_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    """Build non-invoice reference-cost metadata for managed Kimi Code usage."""
    from litellm.llms.kimi_code.reference_cost import (
        build_kimi_code_reference_cost_metadata,
    )

    return build_kimi_code_reference_cost_metadata(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        completion_tokens=completion_tokens,
    )


def _build_alibaba_token_plan_cost_metadata(
    *,
    provider: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """Record honest cost provenance for Token Plan subscription usage."""
    if str(provider or "").strip().lower() != "alibaba_token_plan":
        return {}

    normalized_model = str(model or "").strip()
    if normalized_model.lower().startswith("alibaba_token_plan/"):
        normalized_model = normalized_model.split("/", 1)[1]
    if not normalized_model or normalized_model == "unknown":
        return {}

    return {
        "billing_mode": "alibaba_token_plan_subscription",
        "actual_invoice_cost_known": False,
        "reference_cost_kind": "provider_token_plan_no_public_per_token_rate",
        "reference_cost_currency": "USD",
        "reference_cost_model": f"alibaba_token_plan/{normalized_model}",
        "reference_cost_source": (
            "https://www.alibabacloud.com/help/en/model-studio/coding-plan"
        ),
    }


# --- _build_session_history_record_from_spend_log_row ---
def _build_session_history_record_from_spend_log_row(
    spend_log_row: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prepared = _build_backfill_kwargs_from_spend_log_row(spend_log_row)
    if prepared is None:
        return None

    kwargs, result, provenance = prepared
    kwargs, result = _enrich_trace_name_and_provider_metadata(kwargs, result)

    record = _build_session_history_record(
        kwargs=kwargs,
        result=result,
        start_time=_parse_datetime_value(spend_log_row.get("startTime")),
        end_time=_parse_datetime_value(spend_log_row.get("endTime")),
        allow_runtime_identity=False,
    )
    if record is None:
        return None

    metadata = record.get("metadata") or {}
    metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LiteLLM_SpendLogs",
            "backfill_run_id": backfill_run_id,
            "source_request_id": provenance["source_request_id"],
            "source_spend_log_session_field": provenance["source_spend_log_session_field"],
            "session_id_source": provenance["session_id_source"],
            "trace_id_source": provenance["trace_id_source"],
            "source_status": spend_log_row.get("status"),
        }
    )
    if spend_log_row.get("agent_id") is not None:
        source_agent_id = spend_log_row.get("agent_id")
        metadata["source_agent_id"] = source_agent_id
        if not record.get("agent_id"):
            agent_id = _normalize_agent_id_identity(
                source_agent_id,
                disallowed_values=_agent_id_disallowed_values(
                    record.get("session_id"),
                    record.get("trace_id"),
                    record.get("litellm_call_id"),
                    record.get("agent_name"),
                    record.get("tenant_id"),
                    record.get("repository"),
                    metadata.get("session_id"),
                    metadata.get("trace_id"),
                    metadata.get("trace_user_id"),
                    metadata.get("agent_name"),
                    metadata.get("tenant_id"),
                    metadata.get("repository"),
                ),
            )
            if agent_id:
                record["agent_id"] = agent_id
                metadata["agent_id"] = agent_id
                metadata["agent_id_source"] = "spend_log.agent_id"
    record["metadata"] = metadata
    _enrich_backfill_anthropic_context_window_metadata(record)
    record["trace_id"] = kwargs.get("litellm_trace_id") or record.get("trace_id")
    return _normalize_session_history_record(record)

# --- _build_session_history_record_from_langfuse_trace_observation ---
def _build_session_history_record_from_langfuse_trace_observation(  # noqa: PLR0915
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if observation.get("type") != "GENERATION":
        return None

    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    session_id, session_id_source = _extract_langfuse_session_id(trace, metadata)
    if not session_id:
        return None

    trace_id = trace.get("id") or observation.get("traceId")
    if trace_id is not None and str(trace_id).strip():
        trace_id = str(trace_id).strip()
    else:
        trace_id = None

    usage_object = _build_usage_object_from_langfuse_observation(observation)
    prompt_tokens = _extract_prompt_tokens(usage_object)
    completion_tokens = _extract_completion_tokens(usage_object)
    total_tokens = _extract_total_tokens(usage_object, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_object)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_object)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_object)
    provider_reported_reasoning_tokens = reported_reasoning_tokens
    if usage_object.get("token_count_response"):
        metadata["usage_token_count_response"] = True
    provider = _infer_provider_from_langfuse_observation(observation, metadata)
    output_payload = observation.get("output")
    output_response_payload = _extract_responses_completed_response_from_langfuse_output(
        output_payload
    )
    output_model = _first_non_empty_string(
        _maybe_get(output_response_payload, "model"),
        _extract_model_from_langfuse_output(output_payload),
    )
    input_model = _extract_model_from_langfuse_input(observation.get("input"))
    explicit_openrouter_model = _first_explicit_openrouter_model_string(
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("aawm_auto_agent_selected_model"),
        metadata.get("anthropic_adapter_original_model"),
        metadata.get("codex_adapter_original_model"),
        metadata.get("model"),
        observation.get("model"),
        output_model,
        input_model,
    )
    resolved_model = _first_known_model_string(
        explicit_openrouter_model,
        _session_history_adapter_model(metadata),
        _session_history_metadata_model(metadata),
        observation.get("model"),
        output_model,
        input_model,
        _extract_codex_model_from_response_headers(metadata),
    ) or _first_non_empty_string(
        _session_history_adapter_model(metadata),
        _session_history_metadata_model(metadata),
        observation.get("model"),
        output_model,
        input_model,
        _extract_codex_model_from_response_headers(metadata),
    ) or "unknown"
    model_group = _normalize_session_history_model_group(
        _clean_non_empty_string(metadata.get("model_group")),
        metadata,
        resolved_model,
    )
    call_type = metadata.get("user_api_key_request_route") or observation.get("name")
    api_base = (
        metadata.get("api_base")
        or _maybe_get(metadata.get("hidden_params"), "api_base")
        or observation.get("apiBase")
    )
    api_base_provider = _session_history_provider_from_api_base(
        api_base,
        call_type=call_type,
    )
    if (
        api_base_provider is not None
        and provider != "antigravity"
        and (provider in {None, "openai"} or api_base_provider != "openai")
    ):
        provider = api_base_provider
    provider, resolved_model = _apply_local_embedding_route_metadata(
        metadata=metadata,
        resolved_provider=provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    provider, resolved_model = _apply_local_llm_route_metadata(
        metadata=metadata,
        resolved_provider=provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    provider, resolved_model, model_group = _apply_local_biomed_route_metadata(
        metadata=metadata,
        resolved_provider=provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    inbound_model_alias = _resolve_inbound_model_alias_from_langfuse(
        observation=observation,
        metadata=metadata,
        input_model=input_model,
        output_model=output_model,
        resolved_model=resolved_model,
    )
    _enrich_anthropic_context_window_metadata(
        dict(),
        metadata,
        resolved_model=resolved_model,
        inbound_model_alias=inbound_model_alias,
        provider=provider,
        allow_implicit_default=False,
    )
    permission_decision = _extract_claude_permission_check_decision_from_value(
        output_payload
    )
    if permission_decision is not None:
        permission_blocked = permission_decision == "yes"
        metadata["claude_internal_check"] = True
        metadata["claude_internal_check_type"] = "permission_check"
        metadata["claude_permission_check"] = True
        metadata["claude_permission_check_decision"] = permission_decision
        metadata["claude_permission_check_blocked"] = permission_blocked
        observation_model = str(observation.get("model") or "").strip()
        if observation_model:
            metadata["claude_permission_check_response_model"] = observation_model
        _merge_tags(
            metadata,
            [
                "claude-internal-check",
                "claude-permission-check",
                f"claude-permission-check:{permission_decision}",
                "claude-permission-check:block"
                if permission_blocked
                else "claude-permission-check:allow",
            ],
        )

    message = _extract_first_langfuse_response_message(output_payload)
    request_body = _extract_request_body_from_langfuse_input(observation.get("input"))
    reasoning_fields = _derive_session_history_reasoning_fields(
        metadata=metadata,
        message=message,
        resolved_model=resolved_model,
        provider=provider,
        reported_reasoning_tokens=reported_reasoning_tokens,
        provider_reported_reasoning_tokens=provider_reported_reasoning_tokens,
    )
    reported_reasoning_tokens = reasoning_fields["reported_reasoning_tokens"]
    estimated_reasoning_tokens = reasoning_fields["estimated_reasoning_tokens"]
    reasoning_tokens_source = reasoning_fields["reasoning_tokens_source"]
    reasoning_present = reasoning_fields["reasoning_present"]
    tool_fields = _derive_session_history_tool_fields(
        message=message,
        request_body=request_body,
        metadata=metadata,
        output_payload=output_payload,
        observation_tool_name_fallbacks=[
            observation.get("toolCallNames"),
            observation.get("tool_call_names"),
        ],
    )
    tool_call_count = tool_fields["tool_call_count"]
    tool_names = tool_fields["tool_names"]
    tool_activity = tool_fields["tool_activity"]
    tool_activity_summary = tool_fields["tool_activity_summary"]
    invalid_tool_call_count = tool_fields["invalid_tool_call_count"]
    structured_output_state = tool_fields["structured_output_state"]
    agent_name, tenant_id = _extract_agent_context_from_langfuse_trace_observation(
        trace,
        observation,
    )
    explicit_tenant_id, tenant_source = _extract_tenant_identity_from_langfuse_trace_observation(
        trace,
        observation,
        metadata,
    )
    if explicit_tenant_id:
        tenant_id = explicit_tenant_id
    if tenant_id and tenant_source:
        metadata["tenant_id_source"] = tenant_source
    elif tenant_id:
        metadata["tenant_id_source"] = "agent_context_text"
    repository, repository_source = _extract_repository_identity_from_langfuse_trace_observation_with_source(
        trace,
        observation,
        metadata,
    )
    if repository:
        metadata["repository"] = repository
        if repository_source:
            metadata["repository_source"] = repository_source
    agent_id, agent_id_source = _extract_agent_id_from_langfuse_trace_observation(
        trace,
        observation,
        metadata,
        agent_name=agent_name,
        tenant_id=tenant_id,
        repository=repository,
    )
    if agent_id:
        metadata["agent_id"] = agent_id
        if agent_id_source:
            metadata["agent_id_source"] = agent_id_source
    request_tags = _derive_request_tags_from_langfuse_metadata(metadata)
    response_cost_usd = _safe_float(
        _first_non_none(
            _maybe_get(observation.get("costDetails"), "total"),
            observation.get("calculatedTotalCost"),
            metadata.get("litellm_response_cost"),
            metadata.get("response_cost"),
            metadata.get("usage_openrouter_cost"),
            trace.get("totalCost"),
        )
    )
    if str(provider or "").strip().lower() == "alibaba_token_plan":
        response_cost_usd = None
    cache_fields = _derive_session_history_provider_cache_fields(
        provider=provider,
        model=resolved_model,
        usage_obj=usage_object,
        metadata=metadata,
        request_body=request_body,
        response_cost_usd=response_cost_usd,
        provider_family=provider,
    )
    provider_cache_state = cache_fields["provider_cache_state"]

    permission_usage_fields = _build_permission_usage_fields(
        metadata=metadata,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_cost_usd=response_cost_usd,
    )

    history_metadata = _build_session_history_metadata(
        metadata=metadata,
        request_tags=request_tags,
        tenant_id=tenant_id,
    )
    history_metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LangfuseTraces",
            "backfill_run_id": backfill_run_id,
            "source_trace_id": trace_id,
            "source_observation_id": observation.get("id"),
            "session_id_source": session_id_source,
            "trace_id_source": "trace.id" if trace_id else "missing",
            "source_trace_environment": trace.get("environment"),
            "source_status": "failure"
            if observation.get("statusMessage")
            else "success",
        }
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=history_metadata,
        trace_environment=trace.get("environment"),
        allow_runtime=False,
    )
    compact_summary_state = _classify_compact_summary_state(
        metadata=metadata,
        request_body=request_body,
        output_payload=output_payload,
        session_id=session_id,
        litellm_call_id=observation.get("id"),
        trace_id=trace_id,
    )
    tool_definition_snapshot = _tool_definition_snapshot_from_metadata(metadata)

    record = {
        "litellm_call_id": observation.get("id"),
        "session_id": session_id,
        "trace_id": trace_id,
        "provider_response_id": _first_non_empty_string(
            _maybe_get(output_payload, "id"),
            _maybe_get(output_response_payload, "id"),
        ),
        "provider": provider,
        "model": resolved_model,
        "inbound_model_alias": inbound_model_alias,
        "model_group": model_group,
        "agent_name": agent_name,
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "repository": repository,
        "call_type": call_type,
        "start_time": _parse_datetime_value(observation.get("startTime")),
        "end_time": _parse_datetime_value(observation.get("endTime")),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "provider_cache_attempted": bool(
            provider_cache_state and provider_cache_state.get("attempted")
        ),
        "provider_cache_status": (
            provider_cache_state.get("status") if provider_cache_state else None
        ),
        "provider_cache_miss": bool(
            provider_cache_state and provider_cache_state.get("miss")
        ),
        "provider_cache_miss_reason": (
            provider_cache_state.get("miss_reason") if provider_cache_state else None
        ),
        "provider_cache_miss_token_count": (
            provider_cache_state.get("miss_token_count") if provider_cache_state else None
        ),
        "provider_cache_miss_cost_usd": (
            provider_cache_state.get("miss_cost_usd") if provider_cache_state else None
        ),
        "tool_call_count": tool_call_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        **structured_output_state,
        "tool_names": tool_names,
        "file_read_count": tool_activity_summary["file_read_count"],
        "file_modified_count": tool_activity_summary["file_modified_count"],
        "changed_pre_commit_config": tool_activity_summary[
            "changed_pre_commit_config"
        ],
        "changed_env_file": tool_activity_summary["changed_env_file"],
        "changed_pyproject_toml": tool_activity_summary["changed_pyproject_toml"],
        "changed_gitignore": tool_activity_summary["changed_gitignore"],
        "git_commit_count": tool_activity_summary["git_commit_count"],
        "git_push_count": tool_activity_summary["git_push_count"],
        "tool_activity": tool_activity,
        "response_cost_usd": response_cost_usd,
        **compact_summary_state,
        **permission_usage_fields,
        "litellm_environment": runtime_identity["litellm_environment"],
        "litellm_version": runtime_identity["litellm_version"],
        "litellm_fork_version": runtime_identity["litellm_fork_version"],
        "litellm_wheel_versions": runtime_identity["litellm_wheel_versions"],
        "client_name": runtime_identity["client_name"],
        "client_version": runtime_identity["client_version"],
        "client_user_agent": runtime_identity["client_user_agent"],
        "metadata": history_metadata,
    }
    if tool_definition_snapshot is not None:
        record[_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY] = tool_definition_snapshot
    return _normalize_session_history_record(record)

# --- _build_session_history_metadata ---
def _build_session_history_metadata(
    *,
    metadata: Dict[str, Any],
    request_tags: List[str],
    tenant_id: Optional[str],
) -> Dict[str, Any]:
    _sanitize_worker_context_exhaustion_metadata(metadata)
    history_metadata: Dict[str, Any] = {"request_tags": request_tags}
    if tenant_id:
        history_metadata["tenant_id"] = tenant_id

    for key in _AAWM_SESSION_HISTORY_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            history_metadata[key] = _json_safe_rate_limit_value(value)

    return history_metadata

# --- _build_session_history_record ---
def _build_session_history_record(  # noqa: PLR0915
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
    allow_runtime_identity: bool = True,
) -> Optional[Dict[str, Any]]:
    session_id = _extract_session_id(kwargs)
    if not session_id:
        return None

    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params
    _promote_worker_context_exhaustion_metadata(kwargs, metadata)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    _enrich_claude_permission_check_metadata(
        kwargs,
        metadata,
        result,
        standard_logging_object=standard_logging_object,
    )
    _sync_standard_logging_object(kwargs, metadata)
    standard_logging_object = kwargs.get("standard_logging_object") or standard_logging_object
    request_tags = (
        standard_logging_object.get("request_tags") or metadata.get("tags") or []
    )
    if not isinstance(request_tags, list):
        request_tags = []

    resolved_model = _resolve_session_history_model(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
        result=result,
    )

    usage_obj = _extract_usage_object(kwargs, result)
    hidden_params = getattr(result, "_hidden_params", {}) or {}
    usage_obj = _merge_estimated_rerank_tokens_into_usage(
        kwargs=kwargs,
        result=result,
        usage_obj=usage_obj,
        model=resolved_model,
    )
    usage_dict = _coerce_usage_object_to_dict(usage_obj) or {}
    if usage_dict.get("token_count_response"):
        metadata["usage_token_count_response"] = True
    search_units = _safe_int(usage_dict.get("search_units"))
    if search_units:
        metadata["usage_search_units"] = search_units
    usage_cost = _safe_float(usage_dict.get("cost"))
    if usage_cost is not None:
        metadata["usage_openrouter_cost"] = usage_cost
    openrouter_usage = hidden_params.get("openrouter_usage")
    if isinstance(openrouter_usage, dict):
        search_units = _safe_int(openrouter_usage.get("search_units"))
        if search_units:
            metadata["usage_search_units"] = search_units
        openrouter_cost = _safe_float(openrouter_usage.get("cost"))
        if openrouter_cost is not None:
            metadata["usage_openrouter_cost"] = openrouter_cost
    if hidden_params.get("openrouter_provider") is not None:
        metadata["openrouter_provider"] = hidden_params.get("openrouter_provider")
    if hidden_params.get("openrouter_response_model") is not None:
        metadata["openrouter_response_model"] = hidden_params.get(
            "openrouter_response_model"
        )

    prompt_tokens = _extract_prompt_tokens(usage_obj)
    completion_tokens = _extract_completion_tokens(usage_obj)
    total_tokens = _extract_total_tokens(usage_obj, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)
    provider_reported_reasoning_tokens = reported_reasoning_tokens
    provider_prefix = _infer_usage_breakout_provider_prefix(kwargs, metadata)
    resolved_provider = _normalize_session_history_provider(
        kwargs.get("custom_llm_provider"),
        resolved_model,
        metadata,
    )
    model_group = _get_session_history_model_group(metadata, standard_logging_object)
    api_base = _extract_session_history_api_base(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
    )
    call_type = kwargs.get("call_type") or standard_logging_object.get("call_type")
    api_base_provider = _session_history_provider_from_api_base(
        api_base,
        call_type=call_type,
    )
    if (
        api_base_provider is not None
        and resolved_provider != "antigravity"
        and (resolved_provider in {None, "openai"} or api_base_provider != "openai")
    ):
        resolved_provider = api_base_provider
    provider_for_cache = resolved_provider
    model_group = _normalize_session_history_model_group(
        model_group,
        metadata,
        resolved_model,
    )
    resolved_provider, resolved_model = _apply_local_embedding_route_metadata(
        metadata=metadata,
        resolved_provider=resolved_provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    resolved_provider, resolved_model = _apply_local_llm_route_metadata(
        metadata=metadata,
        resolved_provider=resolved_provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    resolved_provider, resolved_model, model_group = _apply_local_biomed_route_metadata(
        metadata=metadata,
        resolved_provider=resolved_provider,
        resolved_model=resolved_model,
        model_group=model_group,
        call_type=call_type,
        api_base=api_base,
    )
    inbound_model_alias = _resolve_inbound_model_alias(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
        resolved_model=resolved_model,
    )
    _enrich_anthropic_context_window_metadata(
        kwargs,
        metadata,
        resolved_model=resolved_model,
        inbound_model_alias=inbound_model_alias,
        provider=resolved_provider,
        allow_implicit_default=True,
    )

    message = _extract_first_response_message(result)
    request_body = _extract_provider_cache_request_body(kwargs)
    reasoning_fields = _derive_session_history_reasoning_fields(
        metadata=metadata,
        message=message,
        resolved_model=resolved_model,
        provider=provider_prefix or resolved_provider,
        reported_reasoning_tokens=reported_reasoning_tokens,
        provider_reported_reasoning_tokens=provider_reported_reasoning_tokens,
    )
    reported_reasoning_tokens = reasoning_fields["reported_reasoning_tokens"]
    estimated_reasoning_tokens = reasoning_fields["estimated_reasoning_tokens"]
    reasoning_tokens_source = reasoning_fields["reasoning_tokens_source"]
    reasoning_present = reasoning_fields["reasoning_present"]
    tool_fields = _derive_session_history_tool_fields(
        message=message,
        request_body=request_body,
        metadata=metadata,
        output_payload=result,
        standard_logging_object=standard_logging_object,
    )
    tool_call_count = tool_fields["tool_call_count"]
    tool_names = tool_fields["tool_names"]
    tool_activity = tool_fields["tool_activity"]
    tool_activity_summary = tool_fields["tool_activity_summary"]
    invalid_tool_call_count = tool_fields["invalid_tool_call_count"]
    structured_output_state = tool_fields["structured_output_state"]
    agent_name, tenant_id = _extract_agent_context(kwargs)
    explicit_tenant_id, tenant_source = _extract_tenant_identity_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    if explicit_tenant_id:
        tenant_id = explicit_tenant_id
    if tenant_id and tenant_source:
        metadata["tenant_id_source"] = tenant_source
    elif tenant_id:
        metadata["tenant_id_source"] = "agent_context_text"
    repository, repository_source = _extract_repository_identity_from_kwargs_with_source(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    repository_before_memory_workflow = repository
    repository = _apply_codex_memory_workflow_repository(
        kwargs,
        metadata,
        repository,
        request_body=request_body,
    )
    if repository:
        metadata["repository"] = repository
        if repository_source:
            if repository != repository_before_memory_workflow:
                repository_source = f"{repository_source}.codex_memory_workflow"
            metadata["repository_source"] = repository_source
    agent_id, agent_id_source = _extract_agent_id_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
        agent_name=agent_name,
        tenant_id=tenant_id,
        repository=repository,
    )
    if agent_id:
        metadata["agent_id"] = agent_id
        if agent_id_source:
            metadata["agent_id_source"] = agent_id_source
    if metadata.get("workload_type") == "agent_memory":
        request_tags = list(request_tags)
        for tag in metadata.get("tags") or []:
            if isinstance(tag, str) and tag and tag not in request_tags:
                request_tags.append(tag)

    response_cost_usd = None
    if resolved_provider not in {"kimi_code", "alibaba_token_plan"}:
        if resolved_provider == "openrouter":
            response_cost_usd = _first_reported_openrouter_cost(metadata, usage_dict)
        if response_cost_usd is None:
            response_cost_usd = _safe_float(
                _first_non_none(
                    kwargs.get("response_cost"),
                    standard_logging_object.get("response_cost"),
                    hidden_params.get("response_cost"),
                    _maybe_get_path(
                        hidden_params,
                        "additional_headers",
                        "llm_provider-x-litellm-response-cost",
                    ),
                    metadata.get("litellm_response_cost"),
                    metadata.get("response_cost"),
                    metadata.get("usage_openrouter_cost"),
                    usage_dict.get("cost"),
                )
            )
        if (
            (response_cost_usd is None or response_cost_usd == 0)
            and prompt_tokens > 0
            and resolved_model != "unknown"
            and not usage_dict.get("token_count_response")
        ):
            try:
                litellm = _get_litellm_module()
                ResponseAPILoggingUtils = _get_response_api_logging_utils()

                usage_for_cost = None
                if (
                    ResponseAPILoggingUtils is not None
                    and isinstance(usage_obj, dict)
                    and {
                        "input_tokens",
                        "output_tokens",
                    }.issubset(usage_obj.keys())
                ):
                    usage_for_cost = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                        dict(usage_obj)
                    )
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=resolved_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    usage_object=usage_for_cost,
                    call_type="responses",
                )
                response_cost_usd = prompt_cost + completion_cost
            except Exception as exc:
                response_cost_usd = _calculate_response_cost_from_bundled_model_cost_map(
                    model=resolved_model,
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usage_obj=usage_obj,
                )
                verbose_logger.debug(
                    "AawmAgentIdentity: failed to backfill response cost for model=%s: %s",
                    resolved_model,
                    exc,
                )

            if response_cost_usd == 0:
                bundled_response_cost = _calculate_response_cost_from_bundled_model_cost_map(
                    model=resolved_model,
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usage_obj=usage_obj,
                )
                if bundled_response_cost is not None:
                    response_cost_usd = bundled_response_cost

    permission_usage_fields = _build_permission_usage_fields(
        metadata=metadata,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_cost_usd=response_cost_usd,
    )

    prompt_overhead_breakdown = _build_prompt_overhead_breakdown(
        kwargs=kwargs,
        metadata=metadata,
        model=resolved_model,
        prompt_tokens=prompt_tokens,
        request_body=request_body,
    )

    cache_fields = _derive_session_history_provider_cache_fields(
        provider=provider_for_cache,
        model=resolved_model,
        usage_obj=usage_obj,
        metadata=metadata,
        request_body=request_body,
        response_cost_usd=response_cost_usd,
        provider_family=_normalize_provider_cache_family(
            resolved_provider,
            resolved_model,
            metadata,
        ),
    )
    provider_cache_state = cache_fields["provider_cache_state"]
    kimi_code_reference_cost_metadata = _build_kimi_code_reference_cost_metadata(
        provider=resolved_provider,
        model=resolved_model,
        prompt_tokens=prompt_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        completion_tokens=completion_tokens,
    )
    if kimi_code_reference_cost_metadata:
        metadata.update(kimi_code_reference_cost_metadata)
    alibaba_token_plan_cost_metadata = _build_alibaba_token_plan_cost_metadata(
        provider=resolved_provider,
        model=resolved_model,
    )
    if alibaba_token_plan_cost_metadata:
        metadata.update(alibaba_token_plan_cost_metadata)

    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=allow_runtime_identity,
    )
    trace_id = _extract_trace_id(kwargs)
    compact_summary_state = _classify_compact_summary_state(
        metadata=metadata,
        request_body=request_body,
        output_payload=result,
        session_id=session_id,
        litellm_call_id=kwargs.get("litellm_call_id"),
        trace_id=trace_id,
    )
    tool_definition_snapshot = _tool_definition_snapshot_from_metadata(metadata)

    record = {
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "session_id": session_id,
        "trace_id": trace_id,
        "provider_response_id": _maybe_get(result, "id"),
        "provider": resolved_provider,
        "model": resolved_model,
        "inbound_model_alias": inbound_model_alias,
        "model_group": model_group,
        "agent_name": agent_name,
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "repository": repository,
        "call_type": kwargs.get("call_type") or standard_logging_object.get("call_type"),
        "start_time": _normalize_datetime(start_time),
        "end_time": _normalize_datetime(end_time),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "provider_cache_attempted": bool(
            provider_cache_state and provider_cache_state.get("attempted")
        ),
        "provider_cache_status": (
            provider_cache_state.get("status") if provider_cache_state else None
        ),
        "provider_cache_miss": bool(
            provider_cache_state and provider_cache_state.get("miss")
        ),
        "provider_cache_miss_reason": (
            provider_cache_state.get("miss_reason") if provider_cache_state else None
        ),
        "provider_cache_miss_token_count": (
            provider_cache_state.get("miss_token_count") if provider_cache_state else None
        ),
        "provider_cache_miss_cost_usd": (
            provider_cache_state.get("miss_cost_usd") if provider_cache_state else None
        ),
        "tool_call_count": tool_call_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        **structured_output_state,
        "tool_names": tool_names,
        "file_read_count": tool_activity_summary["file_read_count"],
        "file_modified_count": tool_activity_summary["file_modified_count"],
        "changed_pre_commit_config": tool_activity_summary[
            "changed_pre_commit_config"
        ],
        "changed_env_file": tool_activity_summary["changed_env_file"],
        "changed_pyproject_toml": tool_activity_summary["changed_pyproject_toml"],
        "changed_gitignore": tool_activity_summary["changed_gitignore"],
        "git_commit_count": tool_activity_summary["git_commit_count"],
        "git_push_count": tool_activity_summary["git_push_count"],
        "tool_activity": tool_activity,
        "response_cost_usd": response_cost_usd,
        **compact_summary_state,
        **permission_usage_fields,
        "litellm_environment": runtime_identity["litellm_environment"],
        "litellm_version": runtime_identity["litellm_version"],
        "litellm_fork_version": runtime_identity["litellm_fork_version"],
        "litellm_wheel_versions": runtime_identity["litellm_wheel_versions"],
        "client_name": runtime_identity["client_name"],
        "client_version": runtime_identity["client_version"],
        "client_user_agent": runtime_identity["client_user_agent"],
        **_extract_session_host_attribution(metadata),
        **prompt_overhead_breakdown,
        "metadata": _build_session_history_metadata(
            metadata=metadata,
            request_tags=[tag for tag in request_tags if isinstance(tag, str)],
            tenant_id=tenant_id,
        ),
    }
    if tool_definition_snapshot is not None:
        record[_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY] = tool_definition_snapshot
    _apply_runtime_agent_quality_scores(
        record=record,
        request_body=request_body,
        result=result,
        tool_activity=tool_activity,
    )
    _apply_claude_auto_review_identity_to_record(record)
    return _normalize_session_history_record(record)

# --- _build_session_history_db_payload ---
def _build_session_history_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    record = _strip_postgres_nul_bytes(_normalize_session_history_record(dict(record)))
    return (
        record.get("litellm_call_id"),
        record.get("session_id"),
        record.get("trace_id"),
        record.get("provider_response_id"),
        record.get("provider"),
        record.get("model"),
        record.get("model_group"),
        record.get("agent_name"),
        record.get("tenant_id"),
        record.get("call_type"),
        record.get("start_time"),
        record.get("end_time"),
        record.get("input_tokens"),
        record.get("output_tokens"),
        record.get("total_tokens"),
        record.get("cache_read_input_tokens"),
        record.get("cache_creation_input_tokens"),
        record.get("reasoning_tokens_reported"),
        record.get("reasoning_tokens_estimated"),
        record.get("reasoning_tokens_source"),
        record.get("reasoning_present"),
        record.get("thinking_signature_present"),
        record.get("provider_cache_attempted", False),
        record.get("provider_cache_status"),
        record.get("provider_cache_miss", False),
        record.get("provider_cache_miss_reason"),
        record.get("provider_cache_miss_token_count"),
        record.get("provider_cache_miss_cost_usd"),
        record.get("tool_call_count", 0),
        record.get("invalid_tool_call_count", 0),
        json.dumps(record.get("tool_names", [])),
        record.get("file_read_count", 0),
        record.get("file_modified_count", 0),
        record.get("changed_pre_commit_config"),
        record.get("changed_env_file"),
        record.get("changed_pyproject_toml"),
        record.get("changed_gitignore"),
        record.get("git_commit_count", 0),
        record.get("git_push_count", 0),
        record.get("response_cost_usd"),
        record.get("litellm_environment"),
        record.get("litellm_version"),
        record.get("litellm_fork_version"),
        json.dumps(record.get("litellm_wheel_versions") or {}),
        record.get("client_name"),
        record.get("client_version"),
        record.get("client_user_agent"),
        record.get("client_ip"),
        record.get("host_name"),
        record.get("token_permission_input", 0),
        record.get("token_permission_output", 0),
        record.get("permission_usd_cost", 0),
        json.dumps(record.get("metadata", {})),
        record.get("repository"),
        record.get("input_system_tokens_estimated", 0),
        record.get("input_tool_advertisement_tokens_estimated", 0),
        record.get("input_conversation_tokens_estimated", 0),
        record.get("input_other_tokens_estimated", 0),
        record.get("input_breakdown_residual_tokens", 0),
        record.get("system_behavior_tokens_estimated", 0),
        record.get("system_safety_tokens_estimated", 0),
        record.get("system_instructional_tokens_estimated", 0),
        record.get("system_unclassified_tokens_estimated", 0),
        record.get("litellm_processing_ms"),
        record.get("llm_upstream_elapsed_ms"),
        record.get("total_server_elapsed_ms"),
        record.get("ttft_ms"),
        record.get("litellm_pre_send_ms"),
        record.get("litellm_post_response_ms"),
        record.get("llm_upstream_time_to_first_byte_ms"),
        record.get("llm_upstream_stream_ms"),
        record.get("latency_unclassified_ms"),
        record.get(_SESSION_HISTORY_PREVIOUS_GAP_FIELD),
        record.get("structured_output_attempted", False),
        record.get("structured_output_failed", False),
        record.get("structured_output_mode"),
        record.get("structured_output_schema_hash"),
        record.get("structured_output_failure_reason"),
        record.get("trace_quality_score"),
        record.get("empty_completion_failure"),
        record.get("large_tool_result_payload_risk"),
        record.get("destructive_checkout_after_work"),
        record.get("invalid_tool_call_error"),
        record.get("read_only_policy_compliance_score"),
        record.get("read_only_policy_violation_count"),
        record.get("response_meaningfulness_score"),
        record.get("instruction_adherence_score"),
        record.get("answer_completeness_score"),
        record.get("evidence_fidelity_score"),
        record.get("tool_result_fidelity_score"),
        record.get("error_attribution_quality_score"),
        record.get("repetition_loop_risk_score"),
        record.get("context_retention_score"),
        record.get("tool_use_validity_score"),
        record.get("tool_error_recovery_score"),
        record.get("stall_risk_score"),
        record.get("output_contract_compliance_score"),
        record.get("task_progress_score"),
        record.get("scope_control_score"),
        record.get("destructive_action_policy_score"),
        record.get("ignored_path_tracking_policy_score"),
        record.get("ignored_path_tracking_violation_count"),
        record.get("baseline_deflection_attempted_score"),
        record.get("baseline_deflection_incident_score"),
        record.get("baseline_deflection_attempt_count"),
        record.get("baseline_deflection_tool_call_count"),
        record.get("baseline_deflection_input_tokens"),
        record.get("baseline_deflection_elapsed_ms"),
        record.get("quality_gate_trigger_count"),
        record.get("quality_gate_fix_attempt_count"),
        record.get("quality_gate_rerun_count"),
        record.get("sleep_wellness_interruption_attempted_score"),
        record.get("sleep_wellness_interruption_incident_score"),
        record.get("sleep_wellness_interruption_count"),
        record.get("sleep_wellness_interruption_output_tokens"),
        record.get("sleep_wellness_interruption_input_tokens"),
        record.get("sleep_wellness_interruption_elapsed_ms"),
        record.get("sleep_wellness_interruption_after_user_pushback_count"),
        record.get("sleep_wellness_interruption_repeated_count"),
        record.get("terminal_completion_score"),
        record.get("discovery_inventory_coverage_score"),
        record.get("discovery_inventory_missing_count"),
        json.dumps(record.get("agent_score_reasons") or {}),
        record.get("is_compact_summary", False),
        record.get("compact_summary_source"),
        record.get("compact_summary_id"),
        record.get("compact_summary_role"),
        record.get("inbound_model_alias"),
        record.get("agent_id"),
    )

# --- _build_tool_activity_db_payloads ---
def _build_tool_activity_db_payloads(record: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    record = _strip_postgres_nul_bytes(record)
    tool_activity = record.get("tool_activity") or []
    if not isinstance(tool_activity, list):
        return []
    agent_id = _clean_non_empty_string(record.get("agent_id"))

    payloads: List[Tuple[Any, ...]] = []
    for index, item in enumerate(tool_activity):
        if not isinstance(item, dict):
            continue
        payloads.append(
            (
                record["litellm_call_id"],
                record["session_id"],
                record.get("trace_id"),
                record.get("provider"),
                record["model"],
                record.get("agent_name"),
                _safe_int(item.get("tool_index"))
                if _safe_int(item.get("tool_index")) is not None
                else index,
                item.get("tool_call_id"),
                item.get("tool_name"),
                item.get("tool_kind"),
                json.dumps(item.get("file_paths_read") or []),
                json.dumps(item.get("file_paths_modified") or []),
                _safe_int(item.get("git_commit_count")) or 0,
                _safe_int(item.get("git_push_count")) or 0,
                item.get("command_text"),
                json.dumps(item.get("arguments") or {}),
                json.dumps(item.get("metadata") or {}),
                agent_id,
            )
        )
    return payloads

# --- _build_tool_definition_snapshot_db_payloads ---
def _build_tool_definition_snapshot_db_payloads(
    records: List[Dict[str, Any]],
) -> List[Tuple[Any, ...]]:
    payloads_by_key: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
    for record in records:
        payload = _build_tool_definition_snapshot_db_payload(record)
        if payload is None:
            continue
        key = (str(payload[0]), str(payload[1]))
        payloads_by_key.setdefault(key, payload)
    return list(payloads_by_key.values())

# --- _persist_tool_definition_snapshots_best_effort ---
async def _persist_tool_definition_snapshots_best_effort(
    conn: Any,
    records: List[Dict[str, Any]],
) -> None:
    try:
        snapshot_payloads = _build_tool_definition_snapshot_db_payloads(records)
        if not snapshot_payloads:
            return

        await conn.executemany(
            _AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOT_INSERT_SQL,
            snapshot_payloads,
        )
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: best-effort tool definition snapshot persist failed: %s",
            _format_exception_for_warning(exc),
        )

# --- _persist_alias_routing_audit_best_effort ---
async def _persist_alias_routing_audit_best_effort(
    conn: Any,
    records: List[Dict[str, Any]],
) -> None:
    try:
        payloads: List[Tuple[Any, ...]] = []
        for record in records:
            events = _extract_alias_routing_audit_events(record)
            payloads.extend(
                _build_alias_routing_audit_db_payload(record, event, index)
                for index, event in enumerate(events)
            )
        if not payloads:
            return
        await conn.executemany(_AAWM_ALIAS_ROUTING_AUDIT_INSERT_SQL, payloads)
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: best-effort alias routing audit persist failed: %s",
            _format_exception_for_warning(exc),
        )

# --- _build_rate_limit_observation_only_record ---
def _build_rate_limit_observation_only_record(
    kwargs: Dict[str, Any],
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = _merged_rate_limit_metadata(kwargs)
    model = _resolve_rate_limit_model(kwargs, None, metadata)
    return {
        "_skip_session_history": True,
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "session_id": _extract_session_id(kwargs),
        "model": model,
        "rate_limit_observations": observations,
    }

# --- _persist_rate_limit_observations_best_effort ---
async def _persist_rate_limit_observations_best_effort(
    conn: Any,
    records: List[Dict[str, Any]],
    *,
    history_records: List[Dict[str, Any]],
) -> None:
    try:
        openrouter_free_daily_observations = (
            await _build_openrouter_free_daily_observations_for_records(
                conn,
                history_records,
            )
        )
        rate_limit_observations: List[Dict[str, Any]] = []
        for record in records:
            observations = record.get("rate_limit_observations")
            if isinstance(observations, list):
                rate_limit_observations.extend(
                    observation
                    for observation in observations
                    if isinstance(observation, dict)
                )
        rate_limit_observations.extend(openrouter_free_daily_observations)
        if rate_limit_observations:
            (
                rate_limit_observations,
                initial_previous_by_limit_key,
            ) = await _filter_meaningful_rate_limit_observations(
                conn,
                rate_limit_observations,
            )
        if not rate_limit_observations:
            return
        transitions = await _derive_rate_limit_transitions(
            conn,
            rate_limit_observations,
            initial_previous_by_limit_key,
        )
        await conn.executemany(
            _AAWM_RATE_LIMIT_OBSERVATION_INSERT_SQL,
            [
                _build_rate_limit_observation_db_payload(observation)
                for observation in rate_limit_observations
            ],
        )
        if transitions:
            await conn.executemany(
                _AAWM_RATE_LIMIT_TRANSITION_INSERT_SQL,
                [
                    _build_rate_limit_transition_db_payload(transition)
                    for transition in transitions
                ],
            )
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: best-effort rate limit observation persist failed: %s",
            _format_exception_for_warning(exc),
        )

# --- _persist_provider_error_observations_best_effort ---
async def _persist_provider_error_observations_best_effort(
    conn: Any,
    records: List[Dict[str, Any]],
    *,
    identity_by_session: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    try:
        provider_error_observations: List[Dict[str, Any]] = []
        for record in records:
            observations = record.get("provider_error_observations")
            if isinstance(observations, list):
                provider_error_observations.extend(
                    observation
                    for observation in observations
                    if isinstance(observation, dict)
                )
        if not provider_error_observations:
            return
        for observation in provider_error_observations:
            await _apply_claude_auto_review_parent_identity_from_store(
                conn,
                observation,
                identity_by_session,
            )
        await conn.executemany(
            _AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL,
            [
                _build_provider_error_observation_db_payload(observation)
                for observation in provider_error_observations
            ],
        )
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: best-effort provider error observation persist failed: %s",
            _format_exception_for_warning(exc),
        )

# --- _NullSessionHistoryTransaction ---
class _NullSessionHistoryTransaction:
    async def __aenter__(self) -> "_NullSessionHistoryTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

# --- _session_history_transaction ---
def _session_history_transaction(conn: Any) -> Any:
    """Return an async transaction CM when asyncpg-style, else a no-op CM.

    Production asyncpg connections expose `conn.transaction()` as a real async
    context manager object. AsyncMock/test doubles often return awaitable
    coroutines or non-CM placeholders; fall back to a no-op so primary write
    ordering remains testable without a live DB transaction implementation.
    """
    transaction = getattr(conn, "transaction", None)
    if not callable(transaction):
        return _NullSessionHistoryTransaction()
    try:
        cm = transaction()
    except Exception:
        return _NullSessionHistoryTransaction()
    # AsyncMock() attribute access yields awaitables; those are not CMs.
    if inspect.isawaitable(cm):
        try:
            cm.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        return _NullSessionHistoryTransaction()
    if hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__"):
        return cm
    return _NullSessionHistoryTransaction()

# --- _persist_session_history_record ---
async def _persist_session_history_record(record: Dict[str, Any]) -> None:
    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        await _ensure_session_history_schema(conn)

        history_records = [] if record.get("_skip_session_history") else [record]
        if history_records:
            # Primary session_history + tool_activity are one unit of work.
            # Best-effort side tables stay outside so optional writes cannot
            # roll back durable primary history (and match helper names).
            async with _session_history_transaction(conn):
                await _apply_claude_auto_review_parent_identity_from_store(conn, record)
                history_payload = _build_session_history_db_payload(record)
                tool_activity_payloads = _build_tool_activity_db_payloads(record)

                await conn.execute(_AAWM_SESSION_HISTORY_INSERT_SQL, *history_payload)
                await _update_session_history_previous_gap_ms(conn, [history_payload])
                if tool_activity_payloads:
                    await conn.executemany(
                        _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL,
                        tool_activity_payloads,
                    )

        await _persist_tool_definition_snapshots_best_effort(conn, history_records)
        await _persist_rate_limit_observations_best_effort(
            conn,
            [record],
            history_records=history_records,
        )
        await _persist_provider_error_observations_best_effort(conn, [record])
        await _persist_alias_routing_audit_best_effort(conn, [record])

# --- _persist_session_history_records ---
async def _persist_session_history_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        await _ensure_session_history_schema(conn)

        history_records = [
            record for record in records if not record.get("_skip_session_history")
        ]
        identity_by_session = _build_session_identity_cache(history_records)
        if history_records:
            async with _session_history_transaction(conn):
                for record in history_records:
                    await _apply_claude_auto_review_parent_identity_from_store(
                        conn,
                        record,
                        identity_by_session,
                    )
                payloads = [
                    _build_session_history_db_payload(record)
                    for record in history_records
                ]
                tool_activity_payloads: List[Tuple[Any, ...]] = []
                for record in history_records:
                    tool_activity_payloads.extend(
                        _build_tool_activity_db_payloads(record)
                    )

                if payloads:
                    await conn.executemany(_AAWM_SESSION_HISTORY_INSERT_SQL, payloads)
                    await _update_session_history_previous_gap_ms(conn, payloads)
                if tool_activity_payloads:
                    await conn.executemany(
                        _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL,
                        tool_activity_payloads,
                    )
        await _persist_tool_definition_snapshots_best_effort(conn, history_records)
        await _persist_rate_limit_observations_best_effort(
            conn,
            records,
            history_records=history_records,
        )
        await _persist_provider_error_observations_best_effort(
            conn,
            records,
            identity_by_session=identity_by_session,
        )
        await _persist_alias_routing_audit_best_effort(conn, records)

# --- _handle_session_history_success_event ---
def _handle_session_history_success_event(
    kwargs: Any,
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    *,
    log_label: str,
) -> None:
    try:
        rate_limit_observations = _build_rate_limit_observations(
            kwargs=kwargs,
            result=response_obj,
            start_time=start_time,
            end_time=end_time,
        )
        if rate_limit_observations and _rate_limit_observation_only_requested(kwargs):
            _enqueue_session_history_record(
                _build_rate_limit_observation_only_record(
                    kwargs,
                    rate_limit_observations,
                )
            )
            return
        record = _build_session_history_record(
            kwargs=kwargs,
            result=response_obj,
            start_time=start_time,
            end_time=end_time,
        )
        if record is None:
            if rate_limit_observations:
                _enqueue_session_history_record(
                    _build_rate_limit_observation_only_record(
                        kwargs,
                        rate_limit_observations,
                    )
                )
            return

        if rate_limit_observations:
            record["rate_limit_observations"] = rate_limit_observations
        _enqueue_session_history_record(record)
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity.%s failed: %s",
            log_label,
            exc,
        )

# --- _handle_session_history_failure_event ---
def _handle_session_history_failure_event(
    kwargs: Any,
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    *,
    log_label: str,
) -> None:
    try:
        record = _build_failure_observation_only_record(
            kwargs,
            response_obj,
            start_time,
            end_time,
        )
        if record is None:
            return
        _enqueue_session_history_record(record)
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity.%s failed: %s",
            log_label,
            exc,
        )


def _rebind_to_host_globals(fn: FunctionType, host_globals: Dict[str, Any]) -> FunctionType:
    """Return a function object identical to *fn* but using *host_globals*.

    Free-name lookup (``_safe_int``, extractors, etc.) then resolves through
    `aawm_agent_identity`, matching historical monkeypatch behavior without
    compile/exec of source strings.
    """
    rebound = FunctionType(
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


def _install_record_functions() -> None:
    """Publish record APIs on this module and the identity host.

    Each API is a real function defined in this file. Install rebinds
    ``__globals__`` to the identity module dict so helper monkeypatches on
    `aawm_agent_identity` remain effective, then assigns the same function
    object onto both namespaces.
    """
    host = _host()
    host_globals = host.__dict__
    # Modules used by record APIs that may not already be identity globals.
    host_globals.setdefault("inspect", inspect)
    # verbose_logger is imported on this module; ensure host can see the same
    # object if not already present (tests often patch host.verbose_logger).
    host_globals.setdefault("verbose_logger", verbose_logger)

    mod_globals = sys.modules[__name__].__dict__
    for name in _RECORD_API_NAMES:
        original = mod_globals.get(name)
        if original is None:
            raise RuntimeError(f"session_history.record missing API definition: {name}")
        if isinstance(original, FunctionType):
            installed = _rebind_to_host_globals(original, host_globals)
        else:
            # Classes (e.g. _NullSessionHistoryTransaction) keep their own
            # globals; methods resolve via class body. Publish shared class.
            installed = original
            try:
                installed.__module__ = __name__
            except Exception:
                pass
        mod_globals[name] = installed
        host_globals[name] = installed


def install() -> None:
    _install_record_functions()


_installed = False


def _ensure_installed() -> None:
    global _installed
    if _installed:
        return
    _install_record_functions()
    _installed = True


def __getattr__(name: str):
    # Lazy install if identity bind did not run first.
    if name in _RECORD_API_NAMES:
        _ensure_installed()
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(_RECORD_API_NAMES))
