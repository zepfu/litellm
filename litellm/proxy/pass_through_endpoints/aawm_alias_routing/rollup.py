"""Rollup policy and orchestration for auto-agent alias route status.

Wave 7 extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Cross-module dependency ``_get_anthropic_adapter_access_log_target_label`` is
injected via :func:`configure_rollup_runtime`.  Direct imports from sibling
modules (``codex_oauth``) and proxy logging (``aawm_route_logging``) are used
where those modules own the symbols.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import httpx

from litellm.proxy.aawm_route_logging import (
    _normalize_aawm_route_log_reasoning_effort,
    build_aawm_route_rollup_group_header_label,
    emit_aawm_route_status_event,
    record_aawm_route_rollup,
)

from .codex_oauth import _clean_codex_auth_value

# ---------------------------------------------------------------------------
# Injected runtime seams (cross-module)
# ---------------------------------------------------------------------------

_get_anthropic_adapter_access_log_target_label: Optional[
    Callable[[Union[str, httpx.URL]], str]
] = None

_host_globals: Optional[dict] = None


def configure_rollup_runtime(
    *,
    get_access_log_target_label: Callable[[Union[str, httpx.URL]], str],
) -> None:
    """Inject cross-module dependencies.  Must be called before frozen functions."""
    global _get_anthropic_adapter_access_log_target_label

    _get_anthropic_adapter_access_log_target_label = get_access_log_target_label

    if _host_globals is not None:
        _host_globals.update(
            {
                "_resolve_auto_agent_alias_route_rollup_outgoing_target": _resolve_auto_agent_alias_route_rollup_outgoing_target,
                "_auto_agent_alias_model_rollup_label": _auto_agent_alias_model_rollup_label,
                "_auto_agent_alias_event_request_identity": _auto_agent_alias_event_request_identity,
                "_auto_agent_alias_request_outcome_is_recovered": _auto_agent_alias_request_outcome_is_recovered,
                "_auto_agent_alias_request_outcome_is_pending_failover": _auto_agent_alias_request_outcome_is_pending_failover,
                "_auto_agent_alias_route_rollup_status": _auto_agent_alias_route_rollup_status,
                "_auto_agent_alias_route_status_message": _auto_agent_alias_route_status_message,
                "_build_auto_agent_alias_rollup_group_header_label": _build_auto_agent_alias_rollup_group_header_label,
                "_resolve_auto_agent_alias_route_rollup_group_header_label": _resolve_auto_agent_alias_route_rollup_group_header_label,
                "_record_auto_agent_alias_route_status_rollup": _record_auto_agent_alias_route_status_rollup,
            }
        )


# ---------------------------------------------------------------------------
# Frozen functions (baseline llm_passthrough_endpoints.py)
# ---------------------------------------------------------------------------


def _resolve_auto_agent_alias_route_rollup_outgoing_target(
    *,
    route_family: Optional[str],
    target_url: Optional[Union[str, httpx.URL]] = None,
) -> Optional[str]:
    cleaned_route_family = _clean_codex_auth_value(route_family)
    if target_url is not None:
        assert _get_anthropic_adapter_access_log_target_label is not None
        return _get_anthropic_adapter_access_log_target_label(target_url)
    route_family_target_labels = {
        "codex_cohere_chat_completions_adapter": "api.cohere.com/v2/chat",
        "codex_zai_coding_plan_chat_completions_adapter": (
            "api.z.ai/api/coding/paas/v4/chat/completions"
        ),
        "codex_cursor_agent_aiserver_adapter": (
            "agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
        ),
        "anthropic_cursor_agent_aiserver_adapter": (
            "agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
        ),
        "codex_opencode_zen_adapter": "opencode.ai/zen/v1/chat/completions",
        "codex_opencode_go_adapter": "opencode.ai/zen/go/v1/chat/completions",
        "codex_openrouter_completion_adapter": "openrouter.ai/api/v1/chat/completions",
        "anthropic_opencode_zen_responses_adapter": "opencode.ai/zen/v1/responses",
        "anthropic_opencode_zen_completion_adapter": "opencode.ai/zen/v1/chat/completions",
    }
    return route_family_target_labels.get(cleaned_route_family or "", cleaned_route_family)


def _auto_agent_alias_model_rollup_label(event: dict[str, Any]) -> Optional[str]:
    model = _clean_codex_auth_value(event.get("model"))
    alias_model = _clean_codex_auth_value(event.get("alias_model"))
    if (
        model is not None
        and _clean_codex_auth_value(event.get("provider")) == "xai"
        and model.startswith("oa_xai/")
    ):
        # Keep the provider-bound audit model intact; this is display-only.
        model = model.removeprefix("oa_xai/")
    if model and alias_model and model != alias_model:
        return f"{model}({alias_model})"
    return model or alias_model


def _auto_agent_alias_event_request_identity(event: dict[str, Any]) -> Optional[str]:
    for key in ("request_identity", "litellm_call_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _auto_agent_alias_request_outcome_is_recovered(event: dict[str, Any]) -> bool:
    request_outcome = str(event.get("request_outcome") or "")
    candidate_status = str(event.get("candidate_status") or "")
    event_type = str(event.get("event_type") or "")
    return (
        request_outcome == "recovered"
        or candidate_status == "recovered"
        or event_type == "candidate_recovered"
    )


def _auto_agent_alias_request_outcome_is_pending_failover(event: dict[str, Any]) -> bool:
    request_outcome = str(event.get("request_outcome") or "")
    return bool(
        request_outcome == "pending_failover"
        or event.get("account_failover_planned")
    ) and not _auto_agent_alias_request_outcome_is_recovered(event)


def _auto_agent_alias_route_rollup_status(event: dict[str, Any]) -> Optional[str]:
    event_type = str(event.get("event_type") or "")
    candidate_status = str(event.get("candidate_status") or "")
    selection_reason = str(event.get("selection_reason") or "")
    failure_class = str(event.get("failure_class") or "")
    cooldown_scope = str(event.get("cooldown_scope") or "")
    if event_type == "no_candidate_available":
        return "Exhausted"
    if _auto_agent_alias_request_outcome_is_recovered(event):
        return "Recovered"
    if _auto_agent_alias_request_outcome_is_pending_failover(event):
        # Same-request account failover is diagnostic until the request ends.
        return None
    if "auth_degraded" in candidate_status or "auth_degraded" in selection_reason:
        return "Degraded"
    # request-local / no-cooldown failures must not look like durable cool-downs.
    # Note: do not substring-match "cooldown" — retryable_no_cooldown contains it.
    if candidate_status == "retryable_no_cooldown" or cooldown_scope == "none":
        if event.get("error_status_code") or failure_class:
            return "Failed"
        return None
    if cooldown_scope == "request_local":
        if event.get("error_status_code") or failure_class or event.get("redispatch_required"):
            return "Failed"
        return None
    if candidate_status in {
        "cooldown_set",
        "terminal_in_flight_cooldown_set",
        "skipped_cooldown",
    } or (
        candidate_status.startswith("skipped_")
        and "cooldown" in candidate_status
        and "auth_degraded" not in candidate_status
    ):
        return "Cooling Down"
    if cooldown_scope == "candidate" or (event.get("redispatch_required") and cooldown_scope != "request_local"):
        return "Cooling Down"
    if failure_class in {"rate_limited", "capacity_exhausted", "transient_error"}:
        return "Cooling Down"
    if event.get("error_status_code") or failure_class:
        return "Failed"
    return None


def _auto_agent_alias_route_status_message(event: dict[str, Any]) -> str:
    parts: list[str] = []
    source_error = _clean_codex_auth_value(event.get("source_error"))
    if source_error is not None:
        parts.append(f"source_error={source_error}")
    for key in (
        "failure_class",
        "error_type",
        "error_code",
        "error_status_code",
        "candidate_status",
        "selection_reason",
    ):
        value = event.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    error_tokens = event.get("error_tokens")
    if isinstance(error_tokens, list) and error_tokens:
        parts.append("error_tokens={}".format(",".join(str(v) for v in error_tokens[:5])))
    return "; ".join(parts) or "route status changed"


def _should_emit_auto_agent_alias_route_status_event(
    event: dict[str, Any],
    *,
    status: str,
) -> bool:
    """Keep routine successful OAuth failover recovery out of standalone logs."""

    if (
        status != "Recovered"
        or event.get("selection_reason") != "codex_oauth_account_failover"
    ):
        return True
    if event.get("redispatch_required") or event.get("redispatch_threshold_crossed"):
        return True
    if event.get("error_status_code") is not None or event.get("status_code") is not None:
        return True
    if isinstance(event.get("error_tokens"), list) and event["error_tokens"]:
        return True
    return any(
        _clean_codex_auth_value(event.get(key)) is not None
        for key in (
            "source_error",
            "failure_class",
            "error_type",
            "error_code",
            "error_message",
            "error",
            "failure_phase",
        )
    )


def _build_auto_agent_alias_rollup_group_header_label(
    *,
    repository: Optional[str],
    client_product_label: Optional[str],
    host_name: Optional[str],
) -> Optional[str]:
    return build_aawm_route_rollup_group_header_label(
        repository=repository,
        client_product_label=client_product_label,
        host_name=host_name,
    )


def _resolve_auto_agent_alias_route_rollup_group_header_label(
    event: dict[str, Any],
) -> Optional[str]:
    group_header_label = _clean_codex_auth_value(event.get("rollup_group_header_label"))
    if not group_header_label:
        return None
    host_name = _clean_codex_auth_value(event.get("host_name"))
    if "@" in group_header_label or not host_name:
        return group_header_label
    return f"{group_header_label}@{host_name}"


def _record_auto_agent_alias_route_status_rollup(event: dict[str, Any]) -> None:
    status = _auto_agent_alias_route_rollup_status(event)
    if status is None:
        return
    request_identity = _auto_agent_alias_event_request_identity(event)
    alias_model = _clean_codex_auth_value(event.get("alias_model"))
    model_labels: list[str] = []
    model_label = _auto_agent_alias_model_rollup_label(event)
    if model_label:
        model_labels.append(model_label)
    candidates = event.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_label = _auto_agent_alias_model_rollup_label(
                {
                    "model": candidate.get("model"),
                    "alias_model": alias_model,
                    "provider": candidate.get("provider") or event.get("provider"),
                }
            )
            if candidate_label and candidate_label not in model_labels:
                model_labels.append(candidate_label)
    if not model_labels:
        return

    message = _auto_agent_alias_route_status_message(event)
    if _should_emit_auto_agent_alias_route_status_event(event, status=status):
        for label in model_labels:
            emit_aawm_route_status_event(
                alias_model=alias_model,
                model_label=label.split("(", 1)[0],
                status=status,
                message=message,
            )
    group_header_label = _resolve_auto_agent_alias_route_rollup_group_header_label(event)
    incoming_endpoint = _clean_codex_auth_value(event.get("incoming_endpoint"))
    outgoing_target = (
        _clean_codex_auth_value(event.get("outgoing_target"))
        or _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family=_clean_codex_auth_value(event.get("route_family")),
            target_url=event.get("target_url"),
        )
        or "candidate_selection"
    )
    if not group_header_label or not incoming_endpoint:
        return
    # Zero-turn status rollups use same-request native effort only.
    # Core normalization renders absent/invalid values as "none"; never consult
    # candidate config, defaults, or model capabilities here.
    effort = _normalize_aawm_route_log_reasoning_effort(
        event.get("reasoning_effort_native_value")
    )
    origin_kwargs: dict[str, Any] = {}
    if request_identity:
        from litellm.proxy.aawm_route_logging import _AawmRouteRollupOriginIdentity

        origin_kwargs["origin_identity"] = _AawmRouteRollupOriginIdentity(
            litellm_call_id=request_identity,
        )
    for label in model_labels:
        record_aawm_route_rollup(
            group_header_label=group_header_label,
            incoming_endpoint=incoming_endpoint,
            outgoing_target=outgoing_target,
            model_label=label,
            effort=effort,
            turns=0,
            status=status,
            message=_clean_codex_auth_value(event.get("source_error")),
            **origin_kwargs,
        )


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 7)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType  # noqa: E402

_HOST_FUNCTION_NAMES = (
    "_resolve_auto_agent_alias_route_rollup_outgoing_target",
    "_auto_agent_alias_model_rollup_label",
    "_auto_agent_alias_route_rollup_status",
    "_auto_agent_alias_route_status_message",
    "_build_auto_agent_alias_rollup_group_header_label",
    "_resolve_auto_agent_alias_route_rollup_group_header_label",
    "_record_auto_agent_alias_route_status_rollup",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    global _host_globals
    _host_globals = host_globals
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = _FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound
    # Rebound functions resolve imported helpers through host_globals.
    host_globals.setdefault(
        "_normalize_aawm_route_log_reasoning_effort",
        _normalize_aawm_route_log_reasoning_effort,
    )
    host_globals.setdefault(
        "_auto_agent_alias_event_request_identity",
        _auto_agent_alias_event_request_identity,
    )
    host_globals.setdefault(
        "_auto_agent_alias_request_outcome_is_recovered",
        _auto_agent_alias_request_outcome_is_recovered,
    )
    host_globals.setdefault(
        "_auto_agent_alias_request_outcome_is_pending_failover",
        _auto_agent_alias_request_outcome_is_pending_failover,
    )
    host_globals.setdefault(
        "_should_emit_auto_agent_alias_route_status_event",
        _should_emit_auto_agent_alias_route_status_event,
    )

    # Copy seam variables into host_globals so rebound functions resolve them.
    for _sk, _sv in (
        ("_get_anthropic_adapter_access_log_target_label", _get_anthropic_adapter_access_log_target_label),
    ):
        if _sv is not None and _sk not in host_globals:
            host_globals[_sk] = _sv
