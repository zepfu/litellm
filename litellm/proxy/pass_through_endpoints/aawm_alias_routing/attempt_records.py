"""Attempt-record mutation, reasoning-effort normalization, and alias metadata composition.

Wave 5C extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_attempt_records_runtime`.
Signal helpers are consumed from ``error_signals.py`` through explicit seams.
Direct imports from sibling Wave 4/5 modules (``lane_keys``, ``policy``) are
used where those modules own the symbols.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional

from fastapi import Request

from .lane_keys import _CODEX_REASONING_EFFORT_TIER_INDEX
from .policy import (
    ANTHROPIC_AUTO_AGENT_MODEL_ALIAS as _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    CODEX_AUTO_AGENT_MODEL_ALIAS as _CODEX_AUTO_AGENT_MODEL_ALIAS,
)

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / error_signals / classification / state)
# ---------------------------------------------------------------------------

# --- error_signals.py seams ---
_extract_codex_auto_agent_error_tokens: Optional[Callable[..., set[str]]] = None
_extract_codex_auto_agent_error_type_and_code: Optional[
    Callable[..., tuple[Optional[str], Optional[str]]]
] = None
_parse_codex_auto_agent_header_wait_seconds: Optional[Callable[..., Optional[float]]] = None
_get_codex_auto_agent_source_error_summary: Optional[Callable[..., Optional[str]]] = None
_build_safe_kimi_code_selection_telemetry: Optional[Callable[..., dict[str, Any]]] = None

# --- god-module / host seams ---
_extract_exception_status_code: Optional[Callable[..., Optional[int]]] = None
_safe_set_request_parsed_body: Optional[Callable[..., None]] = None
_emit_auto_agent_alias_route_event: Optional[Callable[..., None]] = None
_build_auto_agent_alias_audit_event: Optional[Callable[..., dict[str, Any]]] = None
_build_auto_agent_alias_audit_events: Optional[Callable[..., list[dict[str, Any]]]] = None
_persist_auto_agent_alias_audit_only_events_best_effort: Optional[Callable[..., None]] = None
# Default to owner-concrete implementations from audit_persist (D1-591).
from .audit_persist import (
    _aawm_alias_route_healthy_json_enabled as _default_healthy_json_enabled,
    _aawm_alias_route_verbose_json_enabled as _default_verbose_json_enabled,
)

_aawm_alias_route_verbose_json_enabled: Callable[[], bool] = _default_verbose_json_enabled
_aawm_alias_route_healthy_json_enabled: Callable[[], bool] = _default_healthy_json_enabled
_merge_litellm_metadata: Optional[Callable[..., dict[str, Any]]] = None
_normalize_low_cardinality_tag_value: Optional[Callable[..., Optional[str]]] = None
_normalize_codex_auto_agent_alias_model: Optional[Callable[..., Optional[str]]] = None
_normalize_anthropic_auto_agent_alias_model: Optional[Callable[..., Optional[str]]] = None
_load_bundled_model_cost_map_for_codex_policy: Optional[Callable[[], dict[str, Any]]] = None

# --- model catalog seams ---
_get_model_info: Optional[Callable[..., Any]] = None
_model_cost: Optional[dict[str, Any]] = None
_openai_provider_value: Optional[str] = None

# --- classification / read-pilot gate seams ---
_classify_failure: Optional[Callable[..., Any]] = None
_read_pilot_gate_record: Optional[Callable[..., Any]] = None


# Reference to host_globals set by install(); configure updates it too.
_host_globals_ref: dict | None = None
_MISSING = object()
_RUNTIME_STATE_NAMES = (
    "_extract_codex_auto_agent_error_tokens",
    "_extract_codex_auto_agent_error_type_and_code",
    "_parse_codex_auto_agent_header_wait_seconds",
    "_get_codex_auto_agent_source_error_summary",
    "_build_safe_kimi_code_selection_telemetry",
    "_extract_exception_status_code",
    "_safe_set_request_parsed_body",
    "_emit_auto_agent_alias_route_event",
    "_build_auto_agent_alias_audit_event",
    "_build_auto_agent_alias_audit_events",
    "_persist_auto_agent_alias_audit_only_events_best_effort",
    "_aawm_alias_route_verbose_json_enabled",
    "_aawm_alias_route_healthy_json_enabled",
    "_merge_litellm_metadata",
    "_normalize_low_cardinality_tag_value",
    "_normalize_codex_auto_agent_alias_model",
    "_normalize_anthropic_auto_agent_alias_model",
    "_load_bundled_model_cost_map_for_codex_policy",
    "_get_model_info",
    "_model_cost",
    "_openai_provider_value",
    "_classify_failure",
    "_read_pilot_gate_record",
)
_runtime_restore_stacks: dict[str, list[tuple[object, object, object]]] = {}


def _update_host_runtime_callbacks(
    callbacks: Mapping[str, object],
    previous_module_values: Mapping[str, object],
) -> None:
    if _host_globals_ref is None:
        return
    for name, callback in callbacks.items():
        _runtime_restore_stacks.setdefault(name, []).append(
            (
                callback,
                previous_module_values[name],
                _host_globals_ref.get(name, _MISSING),
            )
        )
        _host_globals_ref[name] = callback


def configure_attempt_records_runtime(  # noqa: PLR0915
    *,
    # error_signals.py
    extract_error_tokens: Callable[..., set[str]],
    extract_error_type_and_code: Callable[..., tuple[Optional[str], Optional[str]]],
    parse_header_wait_seconds: Callable[..., Optional[float]],
    get_source_error_summary: Callable[..., Optional[str]],
    build_kimi_telemetry: Callable[..., dict[str, Any]],
    # god-module / host
    extract_status_code: Callable[..., Optional[int]],
    safe_set_parsed_body: Callable[..., None],
    emit_route_event: Callable[..., None],
    build_audit_event: Callable[..., dict[str, Any]],
    build_audit_events: Callable[..., list[dict[str, Any]]],
    persist_audit_only_events: Callable[..., Any],
    verbose_json_enabled: Optional[Callable[[], bool]] = None,
    healthy_json_enabled: Optional[Callable[[], bool]] = None,
    merge_metadata: Callable[..., dict[str, Any]],
    normalize_tag_value: Callable[..., Optional[str]],
    normalize_codex_alias_model: Callable[..., Optional[str]],
    normalize_anthropic_alias_model: Callable[..., Optional[str]],
    load_bundled_model_cost: Callable[[], dict[str, Any]],
    # model catalog
    get_model_info: Callable[..., Any],
    model_cost: dict[str, Any],
    openai_provider_value: str,
    # classification / read-pilot
    classify_failure: Callable[..., Any],
    read_pilot_gate_record: Callable[..., Any],


) -> None:
    """Bind god-module / error_signals / classification owned dependencies."""
    previous_module_values = {
        name: globals()[name] for name in _RUNTIME_STATE_NAMES
    }
    global _extract_codex_auto_agent_error_tokens
    _extract_codex_auto_agent_error_tokens = extract_error_tokens
    global _extract_codex_auto_agent_error_type_and_code
    _extract_codex_auto_agent_error_type_and_code = extract_error_type_and_code
    global _parse_codex_auto_agent_header_wait_seconds
    _parse_codex_auto_agent_header_wait_seconds = parse_header_wait_seconds
    global _get_codex_auto_agent_source_error_summary
    _get_codex_auto_agent_source_error_summary = get_source_error_summary
    global _build_safe_kimi_code_selection_telemetry
    _build_safe_kimi_code_selection_telemetry = build_kimi_telemetry
    global _extract_exception_status_code
    _extract_exception_status_code = extract_status_code
    global _safe_set_request_parsed_body
    _safe_set_request_parsed_body = safe_set_parsed_body
    global _emit_auto_agent_alias_route_event
    _emit_auto_agent_alias_route_event = emit_route_event
    global _build_auto_agent_alias_audit_event
    _build_auto_agent_alias_audit_event = build_audit_event
    global _build_auto_agent_alias_audit_events
    _build_auto_agent_alias_audit_events = build_audit_events
    global _persist_auto_agent_alias_audit_only_events_best_effort
    _persist_auto_agent_alias_audit_only_events_best_effort = persist_audit_only_events
    global _aawm_alias_route_verbose_json_enabled
    if verbose_json_enabled is not None:
        _aawm_alias_route_verbose_json_enabled = verbose_json_enabled
    global _aawm_alias_route_healthy_json_enabled
    if healthy_json_enabled is not None:
        _aawm_alias_route_healthy_json_enabled = healthy_json_enabled
    global _merge_litellm_metadata
    _merge_litellm_metadata = merge_metadata
    global _normalize_low_cardinality_tag_value
    _normalize_low_cardinality_tag_value = normalize_tag_value
    global _normalize_codex_auto_agent_alias_model
    _normalize_codex_auto_agent_alias_model = normalize_codex_alias_model
    global _normalize_anthropic_auto_agent_alias_model
    _normalize_anthropic_auto_agent_alias_model = normalize_anthropic_alias_model
    global _load_bundled_model_cost_map_for_codex_policy
    _load_bundled_model_cost_map_for_codex_policy = load_bundled_model_cost
    global _get_model_info
    _get_model_info = get_model_info
    global _model_cost
    _model_cost = model_cost
    global _openai_provider_value
    _openai_provider_value = openai_provider_value
    global _classify_failure
    _classify_failure = classify_failure
    global _read_pilot_gate_record
    _read_pilot_gate_record = read_pilot_gate_record
    # If install() has been called, also update host_globals so configured
    # callbacks remain live for facades published there.
    _mod = globals()
    _update_host_runtime_callbacks(
        {name: _mod[name] for name in _RUNTIME_STATE_NAMES},
        previous_module_values,
    )


# ---------------------------------------------------------------------------
# Retryable attempt record mutation
# ---------------------------------------------------------------------------


def _update_codex_auto_agent_retryable_attempt_record(
    *,
    attempt_record: dict[str, Any],
    exc: Any,
    error_class: str,
    cooldown_seconds: float,
    cooldown_scope: Optional[str] = None,
    alias_model: Optional[str] = None,
    candidate: Optional[dict[str, Any]] = None,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> set[str]:
    assert _extract_codex_auto_agent_error_tokens is not None
    assert _extract_exception_status_code is not None
    assert _extract_codex_auto_agent_error_type_and_code is not None
    assert _parse_codex_auto_agent_header_wait_seconds is not None
    assert _get_codex_auto_agent_source_error_summary is not None

    error_tokens = _extract_codex_auto_agent_error_tokens(exc)
    error_status_code = _extract_exception_status_code(exc)
    error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
    retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    source_error = _get_codex_auto_agent_source_error_summary(
        exc,
        status_code=error_status_code,
    )
    update: dict[str, Any] = {
        "status": ("retryable_no_cooldown" if cooldown_scope == "none" else "cooldown_set"),
        "error_class": error_class,
        "error_tokens": sorted(error_tokens),
        "failure_phase": "provider_attempt",
        "attempted_provider_call": True,
        "source_error": source_error,
    }
    if cooldown_scope != "none":
        update["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if cooldown_scope is not None:
        update["cooldown_scope"] = cooldown_scope
    if error_status_code is not None:
        update["error_status_code"] = error_status_code
    if error_type is not None:
        update["error_type"] = error_type
    if error_code is not None:
        update["error_code"] = str(error_code)
    if retry_after_seconds is not None:
        update["retry_after_seconds"] = round(float(retry_after_seconds), 3)
    if alias_model is not None and candidate is not None and kimi_failure_metadata is not None:
        assert _build_safe_kimi_code_selection_telemetry is not None
        update["kimi_code_failure"] = _build_safe_kimi_code_selection_telemetry(
            alias_model=alias_model,
            candidate=candidate,
            metadata=kimi_failure_metadata,
        )
    attempt_record.update(update)
    return error_tokens


# ---------------------------------------------------------------------------
# Attempt-start / failure records
# ---------------------------------------------------------------------------


def _record_auto_agent_alias_attempt_started(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    assert _safe_set_request_parsed_body is not None
    assert _emit_auto_agent_alias_route_event is not None

    candidate_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, candidate_body)
    candidate_metadata = candidate_body.get("litellm_metadata")
    audit_events = (
        candidate_metadata.get("aawm_alias_routing_audit_events")
        if isinstance(candidate_metadata, dict)
        else None
    )
    if (
        isinstance(audit_events, list)
        and audit_events
        and (_aawm_alias_route_verbose_json_enabled() or _aawm_alias_route_healthy_json_enabled())
    ):
        latest_event = audit_events[-1]
        if isinstance(latest_event, dict):
            _emit_auto_agent_alias_route_event(latest_event)
    return candidate_body


# ---------------------------------------------------------------------------
# Read-pilot evidence (exactly-once per event)
# ---------------------------------------------------------------------------


def _record_read_pilot_cooldown_evidence(
    *,
    cooldown_key: Optional[str],
    exc: Any,
    attempt_record: dict[str, Any],
) -> None:
    """Classify + record the CURRENT read-pilot attempt's failure evidence.

    Called from the retry loop BEFORE the cooldown is applied for the same
    attempt, so a structured failure cools immediately (N=1) and a marker
    failure counts toward its N-of-M threshold on this attempt. The evidence
    is keyed on the live ``provider:model:lane`` cooldown key so the gate and
    the applied cooldown share one authoritative key. Classification inputs
    (status code, source-error text, retry-after) are extracted directly from
    the raised exception rather than the post-apply attempt record, because
    those record fields are not populated until after the cooldown decision.

    ``origin`` (upstream/client/unknown; only ``upstream`` ever advances a key
    toward cooling) is stamped on the attempt record for downstream audit.
    """
    assert _extract_exception_status_code is not None
    assert _get_codex_auto_agent_source_error_summary is not None
    assert _parse_codex_auto_agent_header_wait_seconds is not None
    assert _classify_failure is not None
    assert _read_pilot_gate_record is not None

    error_status_code = _extract_exception_status_code(exc)
    source_error = _get_codex_auto_agent_source_error_summary(exc, status_code=error_status_code)
    retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    event = _classify_failure(
        status_code=error_status_code,
        provider=None,
        message=str(source_error or ""),
        retry_after_seconds=retry_after_seconds,
    )
    attempt_record["origin"] = event.origin
    _read_pilot_gate_record(
        cooldown_key=cooldown_key or "read_pilot:unknown",
        event=event,
    )


# ---------------------------------------------------------------------------
# Attempt-failure record
# ---------------------------------------------------------------------------


def _record_auto_agent_alias_attempt_failure(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    error_class: str,
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
    redispatch_required: bool = False,
) -> dict[str, Any]:
    assert _safe_set_request_parsed_body is not None
    assert _build_auto_agent_alias_audit_event is not None
    assert _emit_auto_agent_alias_route_event is not None
    assert _persist_auto_agent_alias_audit_only_events_best_effort is not None

    # Read-pilot cooldown evidence is recorded in the retry loop BEFORE the
    # cooldown is applied (see ``_record_read_pilot_cooldown_evidence``), so it
    # is intentionally NOT re-recorded here -- doing so would double-count
    # marker evidence and double-advance the structured attempt counter.
    failure_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, failure_body)
    failure_metadata = failure_body.get("litellm_metadata")
    full_audit_events = (
        failure_metadata.get("aawm_alias_routing_audit_events")
        if isinstance(failure_metadata, dict)
        else None
    )
    audit_events = [event for event in full_audit_events or [] if isinstance(event, dict)]
    audit_event = audit_events[-1] if audit_events else None
    if audit_event is None:
        audit_event = _build_auto_agent_alias_audit_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            selection=selection,
            candidate=attempt_record,
            event_type="redispatch_required" if redispatch_required else "candidate_retryable_failure",
            candidate_status=attempt_record.get("status") or "cooldown_set",
            attempt_number=len(attempts),
            selected=True,
            selection_reason=selection.get("selection_reason"),
            lane_key=selection.get("lane_key"),
            cooldown_key=selection.get("cooldown_key"),
            cooldown_seconds=attempt_record.get("cooldown_seconds"),
            cooldown_scope=attempt_record.get("cooldown_scope"),
            failure_class=error_class,
            error_status_code=attempt_record.get("error_status_code"),
            error_type=attempt_record.get("error_type"),
            error_code=attempt_record.get("error_code"),
            error_tokens=attempt_record.get("error_tokens"),
            source_error=attempt_record.get("source_error"),
            retry_after_seconds=attempt_record.get("retry_after_seconds"),
            failure_phase=attempt_record.get("failure_phase"),
            attempted_provider_call=attempt_record.get("attempted_provider_call"),
            redispatch_required=redispatch_required,
        )
        audit_events = [audit_event]
    _emit_auto_agent_alias_route_event(
        audit_event,
        level="warning",
    )
    # Only terminal redispatch outcomes use audit-only persistence. Mid-loop
    # retryable 429s that continue failover still reach a normal success or
    # no-candidate write path and must not double-write audit rows.
    if redispatch_required:
        _persist_auto_agent_alias_audit_only_events_best_effort(
            audit_events,
            request_body=prepared_request_body,
        )
    return failure_body


# ---------------------------------------------------------------------------
# Reasoning-effort extraction and normalization
# ---------------------------------------------------------------------------


def _extract_codex_reasoning_effort(
    request_body: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        value = reasoning.get("effort")
        return (value if isinstance(value, str) else None), "reasoning.effort"
    if "reasoning_effort" in request_body:
        value = request_body.get("reasoning_effort")
        return (value if isinstance(value, str) else None), "reasoning_effort"
    return None, None


def _get_codex_reasoning_effort_ceiling(
    resolved_route: dict[str, Any],
) -> Optional[str]:
    assert _openai_provider_value is not None
    assert _get_model_info is not None
    assert _model_cost is not None
    assert _load_bundled_model_cost_map_for_codex_policy is not None

    if (
        resolved_route.get("provider") != _openai_provider_value
        or resolved_route.get("route_family") != "codex_responses"
    ):
        return None

    model = resolved_route.get("model")
    if not isinstance(model, str) or not model:
        return None
    model_info_sources: list[Mapping[str, Any]] = []
    try:
        resolved_model_info = _get_model_info(
            model=model,
            custom_llm_provider=_openai_provider_value,
        )
        if isinstance(resolved_model_info, dict):
            model_info_sources.append(resolved_model_info)
    except Exception:
        pass
    for model_cost in (
        _model_cost,
        _load_bundled_model_cost_map_for_codex_policy(),
    ):
        catalog_model_info = model_cost.get(model)
        if isinstance(catalog_model_info, dict):
            model_info_sources.append(catalog_model_info)

    if any(model_info.get("supports_max_reasoning_effort") is True for model_info in model_info_sources):
        return "max"
    if any(model_info.get("supports_xhigh_reasoning_effort") is True for model_info in model_info_sources):
        return "xhigh"
    if any(model_info.get("supports_reasoning") is True for model_info in model_info_sources) and any(
        model_info.get("supports_xhigh_reasoning_effort") is False for model_info in model_info_sources
    ):
        return "high"
    return None


def _normalize_codex_reasoning_effort_for_resolved_route(
    request_body: dict[str, Any],
    *,
    resolved_route: dict[str, Any],
    attempt_number: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert _merge_litellm_metadata is not None

    requested_effort, native_field = _extract_codex_reasoning_effort(request_body)
    if requested_effort not in _CODEX_REASONING_EFFORT_TIER_INDEX or native_field is None:
        return request_body, {}

    supported_ceiling = _get_codex_reasoning_effort_ceiling(resolved_route)
    if supported_ceiling is None:
        return request_body, {}

    emitted_effort = requested_effort
    mapping_reason = "within_supported_ceiling"
    if _CODEX_REASONING_EFFORT_TIER_INDEX[requested_effort] > _CODEX_REASONING_EFFORT_TIER_INDEX[supported_ceiling]:
        emitted_effort = supported_ceiling
        mapping_reason = "requested_effort_above_model_supported_ceiling"

    updated_body = dict(request_body)
    if emitted_effort != requested_effort:
        if native_field == "reasoning.effort":
            reasoning = dict(updated_body.get("reasoning") or {})
            reasoning["effort"] = emitted_effort
            updated_body["reasoning"] = reasoning
        else:
            updated_body["reasoning_effort"] = emitted_effort

    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    existing_tags = litellm_metadata.get("tags")
    if isinstance(existing_tags, list):
        litellm_metadata["tags"] = [
            tag
            for tag in existing_tags
            if not (
                isinstance(tag, str)
                and (
                    tag == "reasoning-effort-clamped"
                    or tag.startswith("codex-effort:")
                    or tag.startswith("effort:")
                    or tag.startswith("reasoning-effort-ceiling:")
                    or tag.startswith("reasoning-effort-map:")
                    or tag.startswith("codex-auto-agent-attempt:")
                )
            )
        ]
    updated_body["litellm_metadata"] = litellm_metadata

    provider = str(resolved_route["provider"])
    model = str(resolved_route["model"])
    mapping_metadata: dict[str, Any] = {
        "codex_reasoning_effort": emitted_effort,
        "reasoning_effort_requested": requested_effort,
        "reasoning_effort_source": native_field,
        "reasoning_effort_native_provider": provider,
        "reasoning_effort_native_value": emitted_effort,
        "reasoning_effort_native_field": native_field,
        "reasoning_effort_supported_ceiling": supported_ceiling,
        "reasoning_effort_resolved_model": model,
        "reasoning_effort_resolved_provider": provider,
        "reasoning_effort_mapping_reason": mapping_reason,
        "openai_reasoning_effort": emitted_effort,
    }
    tags_to_add = [
        f"codex-effort:{emitted_effort}",
        f"effort:{emitted_effort}",
        f"reasoning-effort-ceiling:{supported_ceiling}",
    ]
    if attempt_number is not None:
        mapping_metadata["reasoning_effort_candidate_attempt"] = attempt_number
        tags_to_add.append(f"codex-auto-agent-attempt:{attempt_number}")
    if emitted_effort != requested_effort:
        mapping_metadata.update(
            {
                "reasoning_effort_clamped_from": requested_effort,
                "reasoning_effort_clamp_reason": mapping_reason,
            }
        )
        tags_to_add.extend(
            [
                "reasoning-effort-clamped",
                f"reasoning-effort-map:{requested_effort}-to-{emitted_effort}",
            ]
        )

    return (
        _merge_litellm_metadata(
            updated_body,
            tags_to_add=tags_to_add,
            extra_fields=mapping_metadata,
        ),
        mapping_metadata,
    )


# ---------------------------------------------------------------------------
# Codex / Anthropic alias metadata composition
# ---------------------------------------------------------------------------


def _add_codex_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    assert _normalize_codex_auto_agent_alias_model is not None
    assert _normalize_low_cardinality_tag_value is not None
    assert _merge_litellm_metadata is not None
    assert _build_auto_agent_alias_audit_events is not None

    candidate = selection["candidate"]
    alias_model = (
        selection.get("alias_model")
        or _normalize_codex_auto_agent_alias_model(request_body.get("model"))
        or _CODEX_AUTO_AGENT_MODEL_ALIAS
    )
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    default_reasoning_effort = _normalize_low_cardinality_tag_value(candidate.get("default_reasoning_effort"))
    default_reasoning_applied = False
    if default_reasoning_effort and "reasoning_effort" not in updated_body:
        reasoning = updated_body.get("reasoning")
        if not isinstance(reasoning, dict):
            updated_body["reasoning"] = {"effort": default_reasoning_effort}
            default_reasoning_applied = True
        elif not reasoning.get("effort"):
            updated_body["reasoning"] = {
                **reasoning,
                "effort": default_reasoning_effort,
            }
            default_reasoning_applied = True
    attempt_number = max(1, len(attempts))
    (
        updated_body,
        reasoning_effort_metadata,
    ) = _normalize_codex_reasoning_effort_for_resolved_route(
        updated_body,
        resolved_route=candidate,
        attempt_number=attempt_number,
    )
    audit_selection = selection
    if reasoning_effort_metadata:
        if attempts:
            attempts[-1].update(reasoning_effort_metadata)
        else:
            audit_selection = {
                **selection,
                "candidate": {
                    **candidate,
                    **reasoning_effort_metadata,
                },
            }
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=audit_selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "codex-auto-agent-alias",
            f"codex-auto-agent-selected:{target_model}",
            f"codex-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(["codex-auto-agent-last-resort"] if candidate.get("last_resort") else []),
            *([f"codex-auto-agent-default-effort:{default_reasoning_effort}"] if default_reasoning_applied else []),
            f"codex-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "codex_auto_agent_alias": alias_model,
            "codex_auto_agent_selected_provider": candidate["provider"],
            "codex_auto_agent_selected_model": target_model,
            "codex_auto_agent_selected_route_family": candidate["route_family"],
            "codex_auto_agent_selected_last_resort": bool(candidate.get("last_resort")),
            **(
                {
                    "codex_auto_agent_default_reasoning_effort": (default_reasoning_effort),
                    "codex_reasoning_effort": (
                        reasoning_effort_metadata.get("codex_reasoning_effort") or default_reasoning_effort
                    ),
                }
                if default_reasoning_applied
                else {}
            ),
            "codex_auto_agent_selection_reason": selection.get("selection_reason"),
            "codex_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "codex_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "codex_auto_agent_lane_key": selection.get("lane_key"),
            "codex_auto_agent_attempts": attempts,
            "codex_auto_agent_skipped_candidates": skipped,
            "codex_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


def _add_anthropic_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    assert _normalize_anthropic_auto_agent_alias_model is not None
    assert _merge_litellm_metadata is not None
    assert _build_auto_agent_alias_audit_events is not None

    candidate = selection["candidate"]
    alias_model = (
        selection.get("alias_model")
        or _normalize_anthropic_auto_agent_alias_model(request_body.get("model"))
        or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "anthropic-auto-agent-alias",
            f"anthropic-auto-agent-selected:{target_model}",
            f"anthropic-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(["anthropic-auto-agent-last-resort"] if candidate.get("last_resort") else []),
            f"anthropic-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "anthropic_auto_agent_alias": alias_model,
            "anthropic_auto_agent_selected_provider": candidate["provider"],
            "anthropic_auto_agent_selected_model": target_model,
            "anthropic_auto_agent_selected_route_family": candidate["route_family"],
            "anthropic_auto_agent_selected_last_resort": bool(candidate.get("last_resort")),
            "anthropic_auto_agent_selection_reason": selection.get("selection_reason"),
            "anthropic_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "anthropic_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "anthropic_auto_agent_lane_key": selection.get("lane_key"),
            "anthropic_auto_agent_attempts": attempts,
            "anthropic_auto_agent_skipped_candidates": skipped,
            "anthropic_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


# ---------------------------------------------------------------------------
# God-module facade installation (Wave 5C)
# ---------------------------------------------------------------------------

_HOST_FUNCTION_NAMES = (
    "_update_codex_auto_agent_retryable_attempt_record",
    "_record_auto_agent_alias_attempt_started",
    "_record_read_pilot_cooldown_evidence",
    "_record_auto_agent_alias_attempt_failure",
    "_extract_codex_reasoning_effort",
    "_get_codex_reasoning_effort_ceiling",
    "_normalize_codex_reasoning_effort_for_resolved_route",
    "_add_codex_auto_agent_alias_metadata",
    "_add_anthropic_auto_agent_alias_metadata",
)


def _host_callback_delegates_to_module(
    name: str,
    callback: object,
    owner_module: object,
) -> bool:
    code = getattr(callback, "__code__", None)
    callback_globals = getattr(callback, "__globals__", None)
    if code is None or not isinstance(callback_globals, dict):
        return False

    owner_callback = getattr(owner_module, name, _MISSING)
    if callback is owner_callback:
        return False
    if (
        callback_globals.get(name) is callback
        and getattr(callback, "__name__", None) != name
    ):
        return True

    referenced_values = [
        callback_globals.get(global_name, _MISSING)
        for global_name in code.co_names
    ]
    closure_values = []
    for cell in getattr(callback, "__closure__", None) or ():
        try:
            closure_values.append(cell.cell_contents)
        except ValueError:
            continue

    if any(value is owner_callback for value in (*referenced_values, *closure_values)):
        return True
    references_seam = name in code.co_names or name in code.co_consts
    return references_seam and any(
        value is owner_module for value in (*referenced_values, *closure_values)
    )


def install(host_globals: dict) -> None:
    """Publish same-object god-module facades for the moved functions.

    Functions retain this module's globals. Host-owned dependencies remain
    late-bound through the callbacks configured by
    :func:`configure_attempt_records_runtime`.
    """
    global _host_globals_ref
    _mod = globals()
    owner_module = _sys.modules[__name__]
    for _name in _RUNTIME_STATE_NAMES:
        host_callback = host_globals.get(_name, _MISSING)
        if host_callback is _MISSING or _host_callback_delegates_to_module(
            _name,
            host_callback,
            owner_module,
        ):
            continue
        _mod[_name] = host_callback
    _host_globals_ref = host_globals
    for _name in _HOST_FUNCTION_NAMES:
        host_globals[_name] = _mod[_name]

    from . import audit_context as _audit_context

    for _name in _audit_context._SEAM_NAMES:
        host_callback = host_globals.get(_name, _MISSING)
        if host_callback is _MISSING or _host_callback_delegates_to_module(
            _name,
            host_callback,
            _audit_context,
        ):
            continue
        setattr(_audit_context, _name, host_callback)


# ---------------------------------------------------------------------------
# Module __setattr__ propagation for callback restores and monkeypatches
# ---------------------------------------------------------------------------

import sys as _sys
import types as _types

_SEAM_NAMES = frozenset(_RUNTIME_STATE_NAMES)


class _SeamPropagatingModule(_types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        seam_names = self.__dict__.get("_SEAM_NAMES")
        if seam_names is None or name not in seam_names:
            return
        host_globals = self.__dict__.get("_host_globals_ref")
        if host_globals is None:
            return
        restore_stacks = self.__dict__.get("_runtime_restore_stacks", {})
        restore_stack = restore_stacks.get(name)
        if restore_stack and value is restore_stack[-1][1]:
            _, _, prior_host_value = restore_stack.pop()
            if prior_host_value is self.__dict__["_MISSING"]:
                host_globals.pop(name, None)
            else:
                host_globals[name] = prior_host_value
            return
        host_globals[name] = value


_sys.modules[__name__].__class__ = _SeamPropagatingModule
