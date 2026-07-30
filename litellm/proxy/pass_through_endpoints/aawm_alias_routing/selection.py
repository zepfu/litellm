"""Candidate selection, state construction, availability shaping, and selection errors.

Wave 5B extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_selection_runtime`.
Direct imports from sibling Wave 4/5A modules (``lane_keys``, ``snapshot_select``,
``openrouter_quota``, ``policy``) are used where those modules own the symbols.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Optional, Sequence

from fastapi import HTTPException, Request

from . import cooldown_state as _cooldown_state
from .cooldown_state import _attach_aawm_alias_routing_state_sources
from .lane_keys import (
    _codex_auto_agent_candidate_key,
    _resolve_anthropic_auto_agent_native_cooldown_lane_key,
    _resolve_codex_auto_agent_openai_cooldown_lane_key,
    _resolve_codex_auto_agent_xai_lane_key,
)
from .openrouter_quota import _apply_openrouter_durable_quota_candidate_cooldown
from .policy import (
    ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS as _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS,
    ANTHROPIC_AUTO_AGENT_MODEL_ALIAS as _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER as _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY as _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER as _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY as _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER as _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    CODEX_AUTO_AGENT_MODEL_ALIAS as _CODEX_AUTO_AGENT_MODEL_ALIAS,
    CODEX_AUTO_AGENT_NATIVE_PROVIDER as _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    CODEX_AUTO_AGENT_OPENCODE_LANE_KEY as _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY,
    CODEX_AUTO_AGENT_OPENCODE_PROVIDER as _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
    CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY as _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY,
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER as _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    CODEX_AUTO_AGENT_XAI_PROVIDER as _CODEX_AUTO_AGENT_XAI_PROVIDER,
)
from .snapshot_select import (
    _commit_round_robin_selection,
    _get_codex_auto_agent_candidates_for_alias,
    _is_alias_config_startup_failed,
    _READ_PILOT_ALIAS_NAME,
    _resolve_aawm_alias_selection_enumeration,
    _routing_candidate_to_anthropic_public_dict,
    _routing_candidate_to_public_dict,
    _select_read_pilot_snapshot_candidates_anthropic,
    get_active_routing_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / cooldown_state.py dependencies)
# ---------------------------------------------------------------------------
_get_codex_active_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_get_anthropic_active_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_get_anthropic_merged_codex_openai_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_set_codex_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None
_set_anthropic_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None
_get_codex_session_affinity: Optional[
    Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]]
] = None
_get_anthropic_session_affinity: Optional[
    Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]]
] = None
_get_openrouter_adapter_active_cooldown_seconds: Optional[
    Callable[[Optional[str]], Awaitable[float]]
] = None
_normalize_codex_alias_model: Optional[Callable[[Any], Optional[str]]] = None
_extract_client_product_label: Optional[
    Callable[[Request, dict[str, Any]], Optional[str]]
] = None
_resolve_codex_session_key: Optional[Callable[..., Optional[str]]] = None
_resolve_anthropic_session_key: Optional[Callable[..., Optional[str]]] = None
_has_continuation_state: Optional[Callable[[Any], bool]] = None
_get_anthropic_candidates_for_alias: Optional[
    Callable[[str], tuple[dict[str, Any], ...]]
] = None
_is_grok_account_quota_candidate: Optional[
    Callable[[Optional[dict[str, Any]]], bool]
] = None
_get_grok_account_quota_lane_cooldown_key: Optional[
    Callable[[Any, Optional[str]], Optional[str]]
] = None
_is_kimi_code_candidate: Optional[Callable[[Optional[dict[str, Any]]], bool]] = None
_get_kimi_managed_account_cooldown_key: Optional[Callable[[], str]] = None


def configure_selection_runtime(
    *,
    get_codex_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    get_anthropic_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    get_anthropic_merged_codex_openai_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    set_codex_cooldown: Callable[[str, float], Awaitable[object]],
    set_anthropic_cooldown: Callable[[str, float], Awaitable[object]],
    get_codex_session_affinity: Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]],
    get_anthropic_session_affinity: Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]],
    get_openrouter_adapter_active_cooldown_seconds: Callable[[Optional[str]], Awaitable[float]],
    normalize_codex_alias_model: Callable[[Any], Optional[str]],
    extract_client_product_label: Callable[[Request, dict[str, Any]], Optional[str]],
    resolve_codex_session_key: Callable[..., Optional[str]],
    resolve_anthropic_session_key: Callable[..., Optional[str]],
    has_continuation_state: Callable[[Any], bool],
    get_anthropic_candidates_for_alias: Callable[[str], tuple[dict[str, Any], ...]],
    is_grok_account_quota_candidate: Callable[[Optional[dict[str, Any]]], bool],
    get_grok_account_quota_lane_cooldown_key: Callable[[Any, Optional[str]], Optional[str]],
    is_kimi_code_candidate: Callable[[Optional[dict[str, Any]]], bool],
    get_kimi_managed_account_cooldown_key: Callable[[], str],
) -> None:
    """Bind god-module / cooldown_state.py owned dependencies."""
    global _get_codex_active_cooldown_state
    _get_codex_active_cooldown_state = get_codex_active_cooldown_state
    global _get_anthropic_active_cooldown_state
    _get_anthropic_active_cooldown_state = get_anthropic_active_cooldown_state
    global _get_anthropic_merged_codex_openai_cooldown_state
    _get_anthropic_merged_codex_openai_cooldown_state = get_anthropic_merged_codex_openai_cooldown_state
    global _set_codex_cooldown
    _set_codex_cooldown = set_codex_cooldown
    global _set_anthropic_cooldown
    _set_anthropic_cooldown = set_anthropic_cooldown
    global _get_codex_session_affinity
    _get_codex_session_affinity = get_codex_session_affinity
    global _get_anthropic_session_affinity
    _get_anthropic_session_affinity = get_anthropic_session_affinity
    global _get_openrouter_adapter_active_cooldown_seconds
    _get_openrouter_adapter_active_cooldown_seconds = get_openrouter_adapter_active_cooldown_seconds
    global _normalize_codex_alias_model
    _normalize_codex_alias_model = normalize_codex_alias_model
    global _extract_client_product_label
    _extract_client_product_label = extract_client_product_label
    global _resolve_codex_session_key
    _resolve_codex_session_key = resolve_codex_session_key
    global _resolve_anthropic_session_key
    _resolve_anthropic_session_key = resolve_anthropic_session_key
    global _has_continuation_state
    _has_continuation_state = has_continuation_state
    global _get_anthropic_candidates_for_alias
    _get_anthropic_candidates_for_alias = get_anthropic_candidates_for_alias
    global _is_grok_account_quota_candidate
    _is_grok_account_quota_candidate = is_grok_account_quota_candidate
    global _get_grok_account_quota_lane_cooldown_key
    _get_grok_account_quota_lane_cooldown_key = get_grok_account_quota_lane_cooldown_key
    global _is_kimi_code_candidate
    _is_kimi_code_candidate = is_kimi_code_candidate
    global _get_kimi_managed_account_cooldown_key
    _get_kimi_managed_account_cooldown_key = get_kimi_managed_account_cooldown_key

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _auto_agent_alias_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_anthropic_auto_agent_alias_model(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    for alias in _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS:
        if normalized == alias.lower():
            return alias
    # CFG-001: recognize config-driven aliases from the active snapshot so
    # the logical `read` alias resolves on Anthropic ingress too.
    snapshot = get_active_routing_snapshot()
    if snapshot is not None:
        for alias_name in snapshot.aliases:
            if normalized == alias_name.lower():
                return alias_name
    return None


# ---------------------------------------------------------------------------
# Candidate public shaping
# ---------------------------------------------------------------------------


def _codex_auto_agent_candidate_public_shape(
    candidate: dict[str, Any],
    *,
    lane_key: Optional[str] = None,
    cooldown_seconds: Optional[float] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    shaped: dict[str, Any] = {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "route_family": candidate["route_family"],
        "last_resort": bool(candidate.get("last_resort")),
    }
    if lane_key is not None:
        shaped["lane_key"] = lane_key
    if cooldown_seconds is not None:
        shaped["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if reason is not None:
        shaped["reason"] = reason
    return shaped


# ---------------------------------------------------------------------------
# Candidate availability
# ---------------------------------------------------------------------------


def _is_auto_agent_candidate_state_available(state: dict[str, Any]) -> bool:
    return state["cooldown_seconds"] <= 0 and state.get("skip_reason") is None


def _build_auto_agent_skipped_candidates_from_states(
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for state in states:
        if _is_auto_agent_candidate_state_available(state):
            continue
        shaped = _codex_auto_agent_candidate_public_shape(
            state["candidate"],
            lane_key=state["lane_key"],
            cooldown_seconds=(state["cooldown_seconds"] if state["cooldown_seconds"] > 0 else None),
            reason=state.get("skip_reason") or "cooldown",
        )
        for field in (
            "cooldown_state_source",
            "cooldown_scope",
            "failure_phase",
            "attempted_provider_call",
        ):
            if field in state:
                shaped[field] = state[field]
        skipped.append(shaped)
    return skipped


# ---------------------------------------------------------------------------
# Request-local cooldown / exclusion helpers
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_request_local_cooldown_key(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
) -> str:
    return _codex_auto_agent_candidate_key(
        candidate,
        lane_key or "__default__",
        epoch_tag=candidate.get("config_epoch_tag"),
    )


def _get_codex_auto_agent_request_local_cooldown_state(
    request: Request,
) -> dict[str, float]:
    state = getattr(request.state, "aawm_alias_request_local_cooldown_until", None)
    if isinstance(state, dict):
        return state
    state = {}
    setattr(request.state, "aawm_alias_request_local_cooldown_until", state)
    return state


def _get_codex_auto_agent_request_local_cooldown_seconds(
    request: Request,
    *,
    cooldown_key: str,
) -> float:
    until = _get_codex_auto_agent_request_local_cooldown_state(request).get(
        cooldown_key,
        0.0,
    )
    remaining = max(0.0, until - time.monotonic())
    if remaining <= 0:
        _get_codex_auto_agent_request_local_cooldown_state(request).pop(
            cooldown_key,
            None,
        )
        return 0.0
    return remaining


def _set_codex_auto_agent_request_local_cooldown(
    request: Request,
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    ttl_seconds = max(0.0, float(cooldown_seconds))
    if ttl_seconds <= 0:
        return
    until = time.monotonic() + ttl_seconds
    state = _get_codex_auto_agent_request_local_cooldown_state(request)
    current_until = state.get(cooldown_key, 0.0)
    if until > current_until:
        state[cooldown_key] = until


def _get_codex_auto_agent_request_local_excluded_keys(
    request: Request,
) -> set[str]:
    excluded = getattr(request.state, "aawm_alias_request_local_excluded_keys", None)
    if isinstance(excluded, set):
        return excluded
    excluded = set()
    setattr(request.state, "aawm_alias_request_local_excluded_keys", excluded)
    return excluded


def _exclude_codex_auto_agent_request_local_candidate(
    request: Request,
    *,
    cooldown_key: str,
) -> None:
    _get_codex_auto_agent_request_local_excluded_keys(request).add(cooldown_key)


def _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
) -> None:
    _exclude_codex_auto_agent_request_local_candidate(
        request,
        cooldown_key=_get_codex_auto_agent_request_local_cooldown_key(
            candidate=candidate,
            lane_key=lane_key,
        ),
    )


def _apply_request_local_cooldown_from_plan(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    """Apply a request-local cooldown + exclusion for a request-local plan (R3-1)."""
    request_local_key = _get_codex_auto_agent_request_local_cooldown_key(
        candidate=candidate,
        lane_key=lane_key,
    )
    _set_codex_auto_agent_request_local_cooldown(
        request,
        cooldown_key=request_local_key,
        cooldown_seconds=cooldown_seconds,
    )
    _exclude_codex_auto_agent_request_local_candidate(
        request,
        cooldown_key=request_local_key,
    )


def _apply_codex_auto_agent_request_local_candidate_state(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
) -> tuple[float, Optional[str], Optional[str]]:
    request_local_cooldown_key = _get_codex_auto_agent_request_local_cooldown_key(
        candidate=candidate,
        lane_key=lane_key,
    )
    request_local_cooldown_seconds = _get_codex_auto_agent_request_local_cooldown_seconds(
        request,
        cooldown_key=request_local_cooldown_key,
    )
    if request_local_cooldown_seconds > cooldown_seconds:
        cooldown_seconds = request_local_cooldown_seconds
        cooldown_state_source = "request_local"
    if request_local_cooldown_key in _get_codex_auto_agent_request_local_excluded_keys(request):
        if request_local_cooldown_seconds <= 0:
            cooldown_seconds = max(cooldown_seconds, 0.001)
        cooldown_state_source = "request_local"
        skip_reason = "request_local_transient_failure"
    return cooldown_seconds, cooldown_state_source, skip_reason


# ---------------------------------------------------------------------------
# Forced / adapter / Kimi / Grok lane cooldown application
# ---------------------------------------------------------------------------


async def _apply_codex_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    assert _set_codex_cooldown is not None
    await _set_codex_cooldown(cooldown_key, cooldown_seconds)


async def _apply_anthropic_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    assert _set_anthropic_cooldown is not None
    await _set_anthropic_cooldown(cooldown_key, cooldown_seconds)


async def _apply_codex_auto_agent_adapter_local_candidate_cooldown(
    *,
    candidate: dict[str, Any],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
) -> tuple[float, Optional[str], Optional[str]]:
    """Merge process-local adapter cooldown evidence into alias selection state."""
    provider = candidate.get("provider")
    model = candidate.get("model")
    adapter_seconds = 0.0
    if provider == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        assert _get_openrouter_adapter_active_cooldown_seconds is not None
        adapter_seconds = await _get_openrouter_adapter_active_cooldown_seconds(model)
    if adapter_seconds <= 0:
        return cooldown_seconds, cooldown_state_source, skip_reason
    if adapter_seconds > cooldown_seconds:
        cooldown_seconds = adapter_seconds
        cooldown_state_source = "adapter_local"
    if cooldown_seconds > 0 and skip_reason is None:
        skip_reason = "adapter_cooldown"
    return cooldown_seconds, cooldown_state_source, skip_reason


async def _apply_kimi_code_managed_account_lane_cooldown(
    *,
    candidate: dict[str, Any],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
    get_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
) -> tuple[float, Optional[str], Optional[str], Optional[str]]:
    """Apply one Kimi managed-account lane without conflating candidate gates."""
    assert _is_kimi_code_candidate is not None
    assert _get_kimi_managed_account_cooldown_key is not None
    if not _is_kimi_code_candidate(candidate):
        return cooldown_seconds, cooldown_state_source, skip_reason, None
    managed_seconds, managed_source = await get_active_cooldown_state(_get_kimi_managed_account_cooldown_key())
    if managed_seconds <= 0:
        return cooldown_seconds, cooldown_state_source, skip_reason, None
    if managed_seconds >= cooldown_seconds:
        cooldown_seconds = managed_seconds
        cooldown_state_source = f"kimi_managed_account:{managed_source}"
    return cooldown_seconds, cooldown_state_source, skip_reason, "managed_account"


async def _apply_codex_auto_agent_grok_account_lane_cooldown(
    *,
    candidate: Any,
    lane_key: Optional[str],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
    get_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
) -> tuple[float, Optional[str], Optional[str]]:
    assert _get_grok_account_quota_lane_cooldown_key is not None
    lane_cooldown_key = _get_grok_account_quota_lane_cooldown_key(
        candidate,
        lane_key,
    )
    if lane_cooldown_key is None:
        return cooldown_seconds, cooldown_state_source, skip_reason
    lane_seconds, lane_source = await get_active_cooldown_state(lane_cooldown_key)
    if lane_seconds > cooldown_seconds:
        cooldown_seconds = lane_seconds
        cooldown_state_source = lane_source
    if lane_seconds > 0 and skip_reason is None:
        skip_reason = "account_quota_cooldown"
    return cooldown_seconds, cooldown_state_source, skip_reason


# ---------------------------------------------------------------------------
# Candidate lookup
# ---------------------------------------------------------------------------


def _find_codex_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
    client_product_label: Optional[str] = None,
    request: Optional[Request] = None,
) -> Optional[dict[str, Any]]:
    if request is not None:
        candidates: Sequence[dict[str, Any]] = _resolve_aawm_alias_selection_enumeration(
            request,
            alias_model,
            client_product_label=client_product_label,
        ).candidates
    else:
        candidates = _get_codex_auto_agent_candidates_for_alias(
            alias_model,
            client_product_label=client_product_label,
        )
    for candidate in candidates:
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


def _find_codex_auto_agent_affinity_candidate(
    affinity: dict[str, Any],
    *,
    alias_model: str,
    client_product_label: Optional[str],
    request: Request,
) -> Optional[dict[str, Any]]:
    """Resolve a pinned candidate without applying new-request eligibility gates.

    Snapshot-established affinity is checked against the active snapshot's full
    alias membership so schedule-only changes do not evict an in-flight
    continuation. Static/legacy affinity keeps the existing lookup path.
    """
    if (
        alias_model == _READ_PILOT_ALIAS_NAME
        and affinity.get("config_hash") is not None
    ):
        snapshot = get_active_routing_snapshot()
        if snapshot is None:
            return None
        alias = snapshot.aliases.get(alias_model)
        if alias is None:
            return None
        for candidate in alias.candidates:
            if (
                candidate.provider == affinity.get("provider")
                and candidate.model == affinity.get("model")
            ):
                return _routing_candidate_to_public_dict(
                    candidate,
                    epoch_tag=snapshot.config_hash,
                )
        return None
    return _find_codex_auto_agent_candidate(
        affinity.get("provider"),
        affinity.get("model"),
        alias_model=alias_model,
        client_product_label=client_product_label,
        request=request,
    )


def _get_anthropic_candidates_for_alias_snapshot_aware(
    alias_model: str,
    *,
    client_product_label: Optional[str] = None,
) -> tuple[dict[str, Any], ...]:
    """Resolve Anthropic-ingress candidates, preferring the active snapshot for `read`.

    CFG-001: when the alias is the config-driven `read` pilot and a snapshot
    is active, return the snapshot's anthropic-projected candidates. Falls
    back to the injected static-table seam for all other aliases or when no
    snapshot is active.

    When a snapshot is active, arbitrary aliases that are neither the
    config-driven pilot nor explicitly registered legacy aliases fail
    closed (empty tuple) rather than delegating to the generic static
    table.  This mirrors the Codex-side ``_get_codex_auto_agent_candidates_for_alias``
    behavior.

    CFG-002 Finding 2: failure state is checked FIRST, before any snapshot
    or static branch.  Once failure is published, all paths return empty.
    """
    # CFG-002 Finding 2: check failure state FIRST.
    if _is_alias_config_startup_failed():
        return ()
    if alias_model == _READ_PILOT_ALIAS_NAME:
        snapshot_candidates = _select_read_pilot_snapshot_candidates_anthropic(
            client_product_label=client_product_label,
        )
        if snapshot_candidates is not None:
            return snapshot_candidates
    # CFG-001: fail closed for unsupported aliases when a snapshot is active.
    if get_active_routing_snapshot() is not None:
        candidates = _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(alias_model)
        if candidates is not None:
            return candidates
        return ()
    assert _get_anthropic_candidates_for_alias is not None
    return _get_anthropic_candidates_for_alias(alias_model)


def _find_anthropic_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    client_product_label: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    for candidate in _get_anthropic_candidates_for_alias_snapshot_aware(
        alias_model, client_product_label=client_product_label,
    ):
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


def _find_anthropic_auto_agent_affinity_candidate(
    affinity: dict[str, Any],
    *,
    alias_model: str,
    client_product_label: Optional[str],
) -> Optional[dict[str, Any]]:
    """Resolve a pinned Anthropic candidate without applying new-request eligibility gates.

    Mirrors _find_codex_auto_agent_affinity_candidate: snapshot-established
    affinity is checked against the active snapshot's full alias membership so
    schedule/TUI-only changes do not evict an in-flight continuation.
    """
    if (
        alias_model == _READ_PILOT_ALIAS_NAME
        and affinity.get("config_hash") is not None
    ):
        snapshot = get_active_routing_snapshot()
        if snapshot is None:
            return None
        alias = snapshot.aliases.get(alias_model)
        if alias is None:
            return None
        for candidate in alias.candidates:
            if (
                candidate.provider == affinity.get("provider")
                and candidate.model == affinity.get("model")
            ):
                return _routing_candidate_to_anthropic_public_dict(
                    candidate,
                    epoch_tag=snapshot.config_hash,
                )
        return None
    return _find_anthropic_auto_agent_candidate(
        affinity.get("provider"),
        affinity.get("model"),
        alias_model=alias_model,
        client_product_label=client_product_label,
    )


# ---------------------------------------------------------------------------
# Candidate state construction
# ---------------------------------------------------------------------------


async def _get_anthropic_auto_agent_candidate_cooldown_state(
    *,
    provider: str,
    cooldown_key: str,
) -> tuple[float, str]:
    """OpenAI/Codex candidates merge Anthropic + Codex cooldown; others use Anthropic-only."""
    assert _get_anthropic_merged_codex_openai_cooldown_state is not None
    assert _get_anthropic_active_cooldown_state is not None
    if provider in {
        _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    }:
        return await _get_anthropic_merged_codex_openai_cooldown_state(cooldown_key)
    return await _get_anthropic_active_cooldown_state(cooldown_key)


async def _build_codex_auto_agent_candidate_state(  # noqa: PLR0915
    request: Request,
    *,
    candidate_template: dict[str, Any],
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
    openai_lane_key: Optional[str] = None,
) -> dict[str, Any]:
    assert _get_codex_active_cooldown_state is not None
    candidate = dict(candidate_template)
    if openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _resolve_codex_auto_agent_xai_lane_key(candidate)
    elif candidate["provider"] == _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    else:
        lane_key = openai_lane_key
    # Wave 3 R3-4: snapshot-resolved candidates carry config_epoch_tag;
    # static/legacy candidates do not, keeping bare keys.
    _epoch_tag = candidate.get("config_epoch_tag")
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key, epoch_tag=_epoch_tag)
    (
        cooldown_seconds,
        initial_cooldown_state_source,
    ) = await _get_codex_active_cooldown_state(cooldown_key)
    cooldown_state_source: Optional[str] = initial_cooldown_state_source
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_grok_account_lane_cooldown(
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_codex_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
        managed_account_cooldown_scope,
    ) = await _apply_kimi_code_managed_account_lane_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_codex_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = _apply_codex_auto_agent_request_local_candidate_state(
        request,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_adapter_local_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_openrouter_durable_quota_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    if forced_cooldown_seconds is not None and forced_cooldown_seconds > cooldown_seconds:
        await _apply_codex_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = cooldown_state_source_override or "forced_candidate_cooldown"
    state: dict[str, Any] = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if _epoch_tag is not None:
        state["config_epoch_tag"] = _epoch_tag
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    if managed_account_cooldown_scope is not None:
        state["cooldown_scope"] = managed_account_cooldown_scope
    return state


async def _build_anthropic_auto_agent_candidate_state(  # noqa: PLR0915
    request: Request,
    *,
    candidate_template: dict[str, Any],
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    openai_lane_key: Optional[str] = None,
    anthropic_lane_key: Optional[str] = None,
) -> dict[str, Any]:
    assert _get_anthropic_active_cooldown_state is not None
    candidate = dict(candidate_template)
    if openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    if anthropic_lane_key is None:
        anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(request)
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _resolve_codex_auto_agent_xai_lane_key(candidate)
    elif candidate["provider"] == _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    elif candidate["provider"] == _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER:
        lane_key = anthropic_lane_key
    else:
        lane_key = openai_lane_key
    _epoch_tag = candidate.get("config_epoch_tag")
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key, epoch_tag=_epoch_tag)
    (
        cooldown_seconds,
        initial_cooldown_state_source,
    ) = await _get_anthropic_auto_agent_candidate_cooldown_state(
        provider=candidate["provider"],
        cooldown_key=cooldown_key,
    )
    cooldown_state_source: Optional[str] = initial_cooldown_state_source
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_grok_account_lane_cooldown(
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_anthropic_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
        managed_account_cooldown_scope,
    ) = await _apply_kimi_code_managed_account_lane_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=lambda cooldown_key: (
            _get_anthropic_auto_agent_candidate_cooldown_state(
                provider=_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
                cooldown_key=cooldown_key,
            )
        ),
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = _apply_codex_auto_agent_request_local_candidate_state(
        request,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_adapter_local_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_openrouter_durable_quota_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    if forced_cooldown_seconds is not None and forced_cooldown_seconds > cooldown_seconds:
        await _apply_anthropic_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = cooldown_state_source_override or "forced_candidate_cooldown"
    state: dict[str, Any] = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if _epoch_tag is not None:
        state["config_epoch_tag"] = _epoch_tag
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    if managed_account_cooldown_scope is not None:
        state["cooldown_scope"] = managed_account_cooldown_scope
    return state


async def _build_codex_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
    client_product_label: Optional[str] = None,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    states: list[dict[str, Any]] = []
    for candidate_template in _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        client_product_label=client_product_label,
    ).candidates:
        states.append(
            await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
                alias_model=alias_model,
                openai_lane_key=openai_lane_key,
            )
        )
    return states


async def _build_anthropic_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    client_product_label: Optional[str] = None,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(request)
    states: list[dict[str, Any]] = []
    for candidate_template in _get_anthropic_candidates_for_alias_snapshot_aware(
        alias_model, client_product_label=client_product_label,
    ):
        states.append(
            await _build_anthropic_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
                alias_model=alias_model,
                openai_lane_key=openai_lane_key,
                anthropic_lane_key=anthropic_lane_key,
            )
        )
    return states


# ---------------------------------------------------------------------------
# Selection errors: in-flight cooldown + redispatch
# ---------------------------------------------------------------------------


def _raise_codex_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Codex auto-agent alias target is cooling down for an in-flight "
                    "session; provider switching is disabled for stateful agent "
                    "continuations. Redispatch a fresh agent attempt to re-run the "
                    "auto selector."
                ),
                "type": "invalid_request_error",
                "code": "aawm_codex_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _raise_anthropic_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Anthropic auto-agent alias target is cooling down for an "
                    "in-flight session; provider switching is disabled for "
                    "stateful Claude continuations. Redispatch a fresh agent "
                    "attempt to re-run the auto selector."
                ),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _build_auto_agent_redispatch_http_exception_detail(
    *,
    alias_family: str,
    alias_model: str,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
    code: str,
    message: str,
) -> dict[str, Any]:
    retry_after = int(
        max(
            1.0,
            float(retry_after_seconds) if retry_after_seconds is not None else cooldown_seconds,
        )
    )
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_retryable_provider_exhaustion",
    )
    detail: dict[str, Any] = {
        "error": {
            "message": message,
            "type": "rate_limit_error",
            "code": code,
        },
        "alias_family": alias_family,
        "alias_model": alias_model,
        "redispatch_model": alias_model,
        "redispatch_reason": "in_flight_retryable_provider_exhaustion",
        "redispatch_required": True,
        "selected_provider": candidate.get("provider"),
        "selected_model": candidate.get("model"),
        "selected_route_family": candidate.get("route_family"),
        "cooldown_seconds": round(float(cooldown_seconds), 3),
        "cooldown_scope": cooldown_scope,
        "retry_after_seconds": retry_after,
        "error_tokens": sorted(error_tokens),
        "candidate": shaped_candidate,
    }
    if error_class is not None:
        detail["failure_class"] = error_class
    if error_status_code is not None:
        detail["error_status_code"] = error_status_code
    if error_type is not None:
        detail["error_type"] = error_type
    if error_code is not None:
        detail["error_code"] = str(error_code)
    if failure_phase is not None:
        detail["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        detail["attempted_provider_call"] = attempted_provider_call
    if isinstance(audit_events, list):
        detail["aawm_alias_routing_audit_events"] = audit_events
    if isinstance(attempts, list):
        detail["attempts"] = attempts
    if isinstance(skipped_candidates, list):
        detail["skipped_candidates"] = skipped_candidates
    return detail


def _raise_codex_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
) -> None:
    detail = _build_auto_agent_redispatch_http_exception_detail(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        error_tokens=error_tokens,
        error_class=error_class,
        cooldown_scope=cooldown_scope,
        error_status_code=error_status_code,
        error_type=error_type,
        error_code=error_code,
        retry_after_seconds=retry_after_seconds,
        failure_phase=failure_phase,
        attempted_provider_call=attempted_provider_call,
        audit_events=audit_events,
        attempts=attempts,
        skipped_candidates=skipped_candidates,
        code="aawm_codex_auto_agent_redispatch_required",
        message=(
            "Codex auto-agent alias target hit retryable provider exhaustion "
            "for an in-flight session. Do not continue this child agent. "
            f"Redispatch a fresh subagent using model {alias_model} "
            "so the auto selector can choose the next available candidate."
        ),
    )
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(detail["retry_after_seconds"])},
    )


def _raise_anthropic_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
) -> None:
    detail = _build_auto_agent_redispatch_http_exception_detail(
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        error_tokens=error_tokens,
        error_class=error_class,
        cooldown_scope=cooldown_scope,
        error_status_code=error_status_code,
        error_type=error_type,
        error_code=error_code,
        retry_after_seconds=retry_after_seconds,
        failure_phase=failure_phase,
        attempted_provider_call=attempted_provider_call,
        audit_events=audit_events,
        attempts=attempts,
        skipped_candidates=skipped_candidates,
        code="aawm_anthropic_auto_agent_redispatch_required",
        message=(
            "Anthropic auto-agent alias target hit retryable provider "
            "exhaustion for an in-flight session. Do not continue this "
            f"child agent. Redispatch a fresh subagent using model {alias_model} "
            "so the auto selector can choose the next available candidate."
        ),
    )
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(detail["retry_after_seconds"])},
    )


# ---------------------------------------------------------------------------
# Codex selector
# ---------------------------------------------------------------------------


async def _select_codex_auto_agent_candidate(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    assert _normalize_codex_alias_model is not None
    assert _extract_client_product_label is not None
    assert _resolve_codex_session_key is not None
    assert _has_continuation_state is not None
    assert _get_codex_session_affinity is not None
    alias_model = _normalize_codex_alias_model(request_body.get("model")) or _CODEX_AUTO_AGENT_MODEL_ALIAS
    client_product_label = _extract_client_product_label(request, request_body)
    session_key = _resolve_codex_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _has_continuation_state(request_body)

    enumeration = _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        client_product_label=client_product_label,
    )
    commit_token = enumeration.commit_token

    affinity = await _get_codex_session_affinity(session_key)
    if affinity is not None and not has_continuation_state:
        affinity = None
    if affinity is not None and has_continuation_state:
        affinity_candidate = _find_codex_auto_agent_affinity_candidate(
            affinity,
            alias_model=alias_model,
            client_product_label=client_product_label,
            request=request,
        )
        # Wave 3 R3-4: continuation-safe affinity.  If the pinned candidate
        # was removed from the active enumeration or its route_family changed
        # (route-incompatible), fail closed with redispatch-required BEFORE
        # any alternate upstream call.  Compatible candidates (same
        # provider/model/route_family) remain pinned regardless of
        # priority/weight/schedule changes.
        if affinity_candidate is None or (
            affinity_candidate.get("route_family") != affinity.get("route_family")
        ):
            _pinned_candidate_shape = {
                "provider": affinity.get("provider"),
                "model": affinity.get("model"),
                "route_family": affinity.get("route_family"),
                "last_resort": bool(affinity.get("last_resort")),
            }
            _raise_codex_auto_agent_redispatch_required(
                candidate=_pinned_candidate_shape,
                lane_key=None,
                cooldown_seconds=0.0,
                error_tokens=set(),
                alias_model=alias_model,
                failure_phase="affinity_continuation_removed",
                attempted_provider_call=False,
            )
        if affinity_candidate is not None:
            affinity_state = await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=affinity_candidate,
                alias_model=alias_model,
            )
            if _is_auto_agent_candidate_state_available(affinity_state):
                return _attach_aawm_alias_routing_state_sources(
                    {
                        **affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": [],
                        "in_flight_session": has_continuation_state,
                    },
                    affinity=affinity,
                    selected_state=affinity_state,
                )
            if affinity_state["cooldown_seconds"] > 0:
                _raise_codex_auto_agent_in_flight_cooldown(
                    candidate=affinity_candidate,
                    lane_key=affinity_state.get("lane_key"),
                    cooldown_seconds=affinity_state["cooldown_seconds"],
                )

    states = await _build_codex_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
        client_product_label=client_product_label,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    if affinity is not None:
        affinity_candidate = _find_codex_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
            client_product_label=client_product_label,
            request=request,
        )
        if affinity_candidate is not None:
            matched_affinity_state: Optional[dict[str, Any]] = None
            for state in states:
                if (
                    state["candidate"]["provider"] == affinity_candidate["provider"]
                    and state["candidate"]["model"] == affinity_candidate["model"]
                ):
                    matched_affinity_state = state
                    break
            if matched_affinity_state is not None:
                if not _is_auto_agent_candidate_state_available(matched_affinity_state):
                    if has_continuation_state:
                        if matched_affinity_state["cooldown_seconds"] > 0:
                            _raise_codex_auto_agent_in_flight_cooldown(
                                candidate=affinity_candidate,
                                lane_key=matched_affinity_state.get("lane_key"),
                                cooldown_seconds=matched_affinity_state["cooldown_seconds"],
                            )
                    skipped.append(
                        _codex_auto_agent_candidate_public_shape(
                            affinity_candidate,
                            lane_key=matched_affinity_state.get("lane_key"),
                            cooldown_seconds=(
                                matched_affinity_state["cooldown_seconds"]
                                if matched_affinity_state["cooldown_seconds"] > 0
                                else None
                            ),
                            reason=matched_affinity_state.get("skip_reason") or "session_affinity_cooldown",
                        )
                    )
                else:
                    return _attach_aawm_alias_routing_state_sources(
                        {
                            **matched_affinity_state,
                            "alias_model": alias_model,
                            "session_key": session_key,
                            "selection_reason": "session_affinity",
                            "skipped": skipped,
                            "in_flight_session": has_continuation_state,
                        },
                        affinity=affinity,
                        selected_state=matched_affinity_state,
                    )
            preferred_available = any(
                not state["candidate"].get("last_resort") and _is_auto_agent_candidate_state_available(state)
                for state in states
            )
            if (
                matched_affinity_state is not None
                and _is_auto_agent_candidate_state_available(matched_affinity_state)
                and (not affinity_candidate.get("last_resort") or not preferred_available)
            ):
                return _attach_aawm_alias_routing_state_sources(
                    {
                        **matched_affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": skipped,
                    },
                    affinity=affinity,
                    selected_state=matched_affinity_state,
                )

    for state in states:
        if state["candidate"].get("last_resort") or not _is_auto_agent_candidate_state_available(state):
            continue
        _commit_round_robin_selection(commit_token, selected_candidate=state["candidate"])
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "first_available",
                "skipped": skipped,
            },
            selected_state=state,
        )

    for state in states:
        if not state["candidate"].get("last_resort") or not _is_auto_agent_candidate_state_available(state):
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "last_resort",
                "skipped": skipped,
            },
            selected_state=state,
        )

    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": ("All Codex auto-agent alias candidates are currently cooled down."),
                "type": "rate_limit_error",
                "code": "aawm_codex_auto_agent_all_candidates_cooling_down",
            },
            "candidates": skipped,
        },
    )


# ---------------------------------------------------------------------------
# Anthropic selector
# ---------------------------------------------------------------------------


async def _select_anthropic_auto_agent_candidate(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    assert _resolve_anthropic_session_key is not None
    assert _has_continuation_state is not None
    assert _get_anthropic_session_affinity is not None
    assert _extract_client_product_label is not None
    alias_model = (
        _normalize_anthropic_auto_agent_alias_model(request_body.get("model")) or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )
    client_product_label = _extract_client_product_label(request, request_body)
    session_key = _resolve_anthropic_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _has_continuation_state(request_body)

    affinity = await _get_anthropic_session_affinity(session_key)
    if affinity is not None and not has_continuation_state:
        affinity = None
    if affinity is not None and has_continuation_state:
        affinity_candidate = _find_anthropic_auto_agent_affinity_candidate(
            affinity,
            alias_model=alias_model,
            client_product_label=client_product_label,
        )
        # CFG-001: continuation-safe affinity.  If the pinned candidate
        # was removed from the active enumeration or its route_family changed
        # (route-incompatible), fail closed with redispatch-required BEFORE
        # any alternate upstream call.  Mirrors the Codex selector guard.
        if affinity_candidate is None or (
            affinity_candidate.get("route_family") != affinity.get("route_family")
        ):
            _pinned_candidate_shape = {
                "provider": affinity.get("provider"),
                "model": affinity.get("model"),
                "route_family": affinity.get("route_family"),
                "last_resort": bool(affinity.get("last_resort")),
            }
            _raise_anthropic_auto_agent_redispatch_required(
                candidate=_pinned_candidate_shape,
                lane_key=None,
                cooldown_seconds=0.0,
                error_tokens=set(),
                alias_model=alias_model,
                failure_phase="affinity_continuation_removed",
                attempted_provider_call=False,
            )
        if affinity_candidate is not None:
            affinity_state = await _build_anthropic_auto_agent_candidate_state(
                request,
                candidate_template=affinity_candidate,
                alias_model=alias_model,
            )
            if _is_auto_agent_candidate_state_available(affinity_state):
                return _attach_aawm_alias_routing_state_sources(
                    {
                        **affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": [],
                        "in_flight_session": has_continuation_state,
                    },
                    affinity=affinity,
                    selected_state=affinity_state,
                )
            if affinity_state["cooldown_seconds"] > 0:
                _raise_anthropic_auto_agent_in_flight_cooldown(
                    candidate=affinity_candidate,
                    lane_key=affinity_state.get("lane_key"),
                    cooldown_seconds=affinity_state["cooldown_seconds"],
                )

    states = await _build_anthropic_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
        client_product_label=client_product_label,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    if affinity is not None:
        affinity_candidate = _find_anthropic_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
            client_product_label=client_product_label,
        )
        if affinity_candidate is not None:
            matched_affinity_state: Optional[dict[str, Any]] = None
            for state in states:
                if (
                    state["candidate"]["provider"] == affinity_candidate["provider"]
                    and state["candidate"]["model"] == affinity_candidate["model"]
                ):
                    matched_affinity_state = state
                    break
            if matched_affinity_state is not None:
                if not _is_auto_agent_candidate_state_available(matched_affinity_state):
                    if has_continuation_state:
                        if matched_affinity_state["cooldown_seconds"] > 0:
                            _raise_anthropic_auto_agent_in_flight_cooldown(
                                candidate=affinity_candidate,
                                lane_key=matched_affinity_state.get("lane_key"),
                                cooldown_seconds=matched_affinity_state["cooldown_seconds"],
                            )
                    skipped.append(
                        _codex_auto_agent_candidate_public_shape(
                            affinity_candidate,
                            lane_key=matched_affinity_state.get("lane_key"),
                            cooldown_seconds=(
                                matched_affinity_state["cooldown_seconds"]
                                if matched_affinity_state["cooldown_seconds"] > 0
                                else None
                            ),
                            reason=matched_affinity_state.get("skip_reason") or "session_affinity_cooldown",
                        )
                    )
                else:
                    return _attach_aawm_alias_routing_state_sources(
                        {
                            **matched_affinity_state,
                            "alias_model": alias_model,
                            "session_key": session_key,
                            "selection_reason": "session_affinity",
                            "skipped": skipped,
                            "in_flight_session": has_continuation_state,
                        },
                        affinity=affinity,
                        selected_state=matched_affinity_state,
                    )

    for state in states:
        if state["candidate"].get("last_resort") or not _is_auto_agent_candidate_state_available(state):
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "first_available",
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
            },
            selected_state=state,
        )

    for state in states:
        if not state["candidate"].get("last_resort") or not _is_auto_agent_candidate_state_available(state):
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "last_resort",
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
            },
            selected_state=state,
        )

    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": ("All Anthropic auto-agent alias candidates are currently cooled down."),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_all_candidates_cooling_down",
            },
            "candidates": skipped,
        },
    )


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5B)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_auto_agent_alias_float",
    "_normalize_anthropic_auto_agent_alias_model",
    "_codex_auto_agent_candidate_public_shape",
    "_is_auto_agent_candidate_state_available",
    "_build_auto_agent_skipped_candidates_from_states",
    "_get_codex_auto_agent_request_local_cooldown_key",
    "_get_codex_auto_agent_request_local_cooldown_state",
    "_get_codex_auto_agent_request_local_cooldown_seconds",
    "_set_codex_auto_agent_request_local_cooldown",
    "_get_codex_auto_agent_request_local_excluded_keys",
    "_exclude_codex_auto_agent_request_local_candidate",
    "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
    "_apply_request_local_cooldown_from_plan",
    "_apply_codex_auto_agent_request_local_candidate_state",
    "_apply_codex_auto_agent_forced_candidate_cooldown",
    "_apply_anthropic_auto_agent_forced_candidate_cooldown",
    "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
    "_apply_kimi_code_managed_account_lane_cooldown",
    "_apply_codex_auto_agent_grok_account_lane_cooldown",
    "_find_codex_auto_agent_candidate",
    "_find_codex_auto_agent_affinity_candidate",
    "_find_anthropic_auto_agent_candidate",
    "_find_anthropic_auto_agent_affinity_candidate",
    "_get_anthropic_auto_agent_candidate_cooldown_state",
    "_build_codex_auto_agent_candidate_state",
    "_build_anthropic_auto_agent_candidate_state",
    "_build_codex_auto_agent_candidate_states",
    "_build_anthropic_auto_agent_candidate_states",
    "_raise_codex_auto_agent_in_flight_cooldown",
    "_raise_anthropic_auto_agent_in_flight_cooldown",
    "_build_auto_agent_redispatch_http_exception_detail",
    "_raise_codex_auto_agent_redispatch_required",
    "_raise_anthropic_auto_agent_redispatch_required",
    "_select_codex_auto_agent_candidate",
    "_select_anthropic_auto_agent_candidate",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
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
    _mod["_attach_aawm_alias_routing_state_sources"] = (
        _cooldown_state._attach_aawm_alias_routing_state_sources
    )
    host_globals["_attach_aawm_alias_routing_state_sources"] = (
        _cooldown_state._attach_aawm_alias_routing_state_sources
    )
    # Copy seam variables into host_globals so rebound functions resolve them.
    host_globals.update({
        "_get_codex_active_cooldown_state": _get_codex_active_cooldown_state,
        "_get_anthropic_active_cooldown_state": _get_anthropic_active_cooldown_state,
        "_get_anthropic_merged_codex_openai_cooldown_state": _get_anthropic_merged_codex_openai_cooldown_state,
        "_set_codex_cooldown": _set_codex_cooldown,
        "_set_anthropic_cooldown": _set_anthropic_cooldown,
        "_get_codex_session_affinity": _get_codex_session_affinity,
        "_get_anthropic_session_affinity": _get_anthropic_session_affinity,
        "_normalize_codex_alias_model": _normalize_codex_alias_model,
        "_extract_client_product_label": _extract_client_product_label,
        "_resolve_codex_session_key": _resolve_codex_session_key,
        "_resolve_anthropic_session_key": _resolve_anthropic_session_key,
        "_has_continuation_state": _has_continuation_state,
        "_get_anthropic_candidates_for_alias": _get_anthropic_candidates_for_alias,
        "_get_anthropic_candidates_for_alias_snapshot_aware": _get_anthropic_candidates_for_alias_snapshot_aware,
        "_find_anthropic_auto_agent_affinity_candidate": _find_anthropic_auto_agent_affinity_candidate,
        "_routing_candidate_to_anthropic_public_dict": _routing_candidate_to_anthropic_public_dict,
        "_is_grok_account_quota_candidate": _is_grok_account_quota_candidate,
        "_get_grok_account_quota_lane_cooldown_key": _get_grok_account_quota_lane_cooldown_key,
        "_is_kimi_code_candidate": _is_kimi_code_candidate,
        "_get_kimi_managed_account_cooldown_key": _get_kimi_managed_account_cooldown_key,
    })
