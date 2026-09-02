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

from .policy import (
    CODEX_AUTO_AGENT_NATIVE_PROVIDER as _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
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
        "codex_nous_chat_completions_adapter": (
            "inference-api.nousresearch.com/v1/chat/completions"
        ),
        "codex_zai_coding_plan_chat_completions_adapter": (
            "api.z.ai/api/coding/paas/v4/chat/completions"
        ),
        "codex_cursor_agent_aiserver_adapter": (
            "agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
        ),
        "anthropic_cursor_agent_aiserver_adapter": (
            "agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
        ),
        "codex_nvidia_completion_adapter": (
            "integrate.api.nvidia.com/v1/chat/completions"
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
    attempt_status = str(event.get("status") or "")
    selection_reason = str(event.get("selection_reason") or "")
    failure_class = str(event.get("failure_class") or "")
    cooldown_scope = str(event.get("cooldown_scope") or "")
    if event_type == "no_candidate_available":
        return "Exhausted"
    if (
        candidate_status in {"ineligible", "candidate_ineligible_no_cooldown"}
        or attempt_status == "candidate_ineligible_no_cooldown"
        or failure_class == "candidate_deterministically_ineligible"
    ):
        return "Ineligible"
    if _auto_agent_alias_request_outcome_is_recovered(event):
        return "Recovered"
    if _auto_agent_alias_request_outcome_is_pending_failover(event):
        # Same-request account failover is diagnostic until the request ends.
        if (
            event.get("attempted_provider_call") is True
            and _clean_codex_auth_value(event.get("provider"))
            == _CODEX_AUTO_AGENT_NATIVE_PROVIDER
            and _clean_codex_auth_value(event.get("account_hash"))
        ):
            return "Failed"
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
        "ineligibility_reason",
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
        if (
            status == "Failed"
            and _auto_agent_alias_request_outcome_is_pending_failover(event)
            and event.get("attempted_provider_call") is True
            and _clean_codex_auth_value(event.get("provider"))
            == _CODEX_AUTO_AGENT_NATIVE_PROVIDER
            and _clean_codex_auth_value(event.get("account_hash"))
        ):
            return False
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


def _record_auto_agent_alias_route_status_rollup(  # noqa: PLR0915
    event: dict[str, Any],
) -> None:
    status = _auto_agent_alias_route_rollup_status(event)
    request_identity = _auto_agent_alias_event_request_identity(event)
    alias_model = _clean_codex_auth_value(event.get("alias_model"))

    candidates = event.get("candidates")
    attempts = event.get("attempts")
    normalized_attempts = [
        attempt for attempt in attempts or [] if isinstance(attempt, dict)
    ]
    has_terminal_inventory = isinstance(candidates, list) and any(
        isinstance(candidate, dict)
        and candidate.get("terminal_disposition") in {"attempted", "skipped"}
        for candidate in candidates
    )

    # Terminal events carry one event-level failure plus a candidate inventory.
    # Keep legacy events without dispositions on their existing broadcast path,
    # but use each candidate's own attempt/skip data when the inventory is
    # available.
    candidate_entries: list[tuple[str, dict[str, Any]]] = []
    if has_terminal_inventory or normalized_attempts:
        seen_labels: set[str] = set()
        attempts_by_identity: dict[
            tuple[str, str, str], list[dict[str, Any]]
        ] = {}

        def _identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
            return tuple(
                _clean_codex_auth_value(candidate.get(key)) or ""
                for key in ("provider", "model", "route_family")
            )

        for attempt in normalized_attempts:
            attempts_by_identity.setdefault(_identity(attempt), []).append(attempt)

        for attempt in normalized_attempts:
            label = _auto_agent_alias_model_rollup_label(
                {
                    **attempt,
                    "alias_model": alias_model,
                }
            )
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            candidate_entries.append(
                (label, attempts_by_identity[_identity(attempt)][-1])
            )

        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            label = _auto_agent_alias_model_rollup_label(
                {
                    **candidate,
                    "alias_model": alias_model,
                    "provider": candidate.get("provider") or event.get("provider"),
                }
            )
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            candidate_event = dict(candidate)
            reason = _clean_codex_auth_value(
                candidate_event.get("reason") or candidate_event.get("skip_reason")
            )
            if candidate_event.get("terminal_disposition") == "skipped":
                if candidate_event.get("candidate_status") is None:
                    if reason == "candidate_ineligible":
                        candidate_event["candidate_status"] = "ineligible"
                    elif reason:
                        candidate_event["candidate_status"] = f"skipped_{reason}"
                if candidate_event.get("selection_reason") is None:
                    candidate_event["selection_reason"] = reason
                candidate_event.setdefault("attempted_provider_call", False)
            elif candidate_event.get("terminal_disposition") == "attempted":
                if candidate_event.get("status") is None:
                    candidate_event["status"] = candidate_event.get("outcome")
            candidate_entries.append((label, candidate_event))

    has_candidate_local_attribution = bool(candidate_entries)
    if not candidate_entries:
        if status is None:
            return
        model_labels: list[str] = []
        model_label = _auto_agent_alias_model_rollup_label(event)
        if model_label:
            model_labels.append(model_label)
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
        candidate_entries = [(label, event) for label in model_labels]

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
    # Zero-turn status rollups use same-request native effort only.
    # Core normalization renders absent/invalid values as "none"; never consult
    # candidate config, defaults, or model capabilities here.
    effort = _normalize_aawm_route_log_reasoning_effort(
        event.get("reasoning_effort_native_value")
    )
    from litellm.proxy.aawm_route_logging import _AawmRouteRollupOriginIdentity

    for candidate_index, (label, candidate_event) in enumerate(candidate_entries):
        candidate_status = (
            status
            if candidate_event is event
            else _auto_agent_alias_route_rollup_status(candidate_event)
        )
        candidate_message = _auto_agent_alias_route_status_message(candidate_event)
        if (
            candidate_status is not None
            and _should_emit_auto_agent_alias_route_status_event(
                candidate_event,
                status=candidate_status,
            )
        ):
            emit_aawm_route_status_event(
                alias_model=alias_model,
                model_label=label.split("(", 1)[0],
                status=candidate_status,
                message=candidate_message,
            )
        if not group_header_label or not incoming_endpoint:
            continue

        candidate_attempted_provider_call = (
            candidate_event.get("attempted_provider_call") is True
        )
        candidate_provider = (
            _clean_codex_auth_value(candidate_event.get("provider"))
            if candidate_attempted_provider_call
            else None
        )
        candidate_is_native_openai_provider = (
            candidate_provider == _CODEX_AUTO_AGENT_NATIVE_PROVIDER
        )
        candidate_account_hash = _clean_codex_auth_value(
            candidate_event.get("account_hash")
        )
        candidate_account_display = _clean_codex_auth_value(
            candidate_event.get("account_display")
        )
        candidate_origin_identity = None
        if request_identity or (
            candidate_attempted_provider_call and candidate_account_hash
        ):
            candidate_origin_identity = _AawmRouteRollupOriginIdentity(
                litellm_call_id=request_identity,
                provider=candidate_provider,
                account_identity=(
                    candidate_account_hash
                    if candidate_attempted_provider_call
                    and candidate_is_native_openai_provider
                    else None
                ),
                account_display=(
                    candidate_account_display
                    if candidate_attempted_provider_call
                    and candidate_is_native_openai_provider
                    and candidate_account_hash
                    else None
                ),
            )
        candidate_origin_kwargs = (
            {}
            if candidate_origin_identity is None
            else {"origin_identity": candidate_origin_identity}
        )
        request_status_kwargs = (
            {"request_status": status}
            if has_candidate_local_attribution
            and candidate_index == len(candidate_entries) - 1
            and status is not None
            else {}
        )
        record_aawm_route_rollup(
            group_header_label=group_header_label,
            incoming_endpoint=incoming_endpoint,
            outgoing_target=outgoing_target,
            model_label=label,
            effort=effort,
            turns=0,
            status=candidate_status,
            message=_clean_codex_auth_value(candidate_event.get("source_error")),
            **request_status_kwargs,
            **candidate_origin_kwargs,
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
