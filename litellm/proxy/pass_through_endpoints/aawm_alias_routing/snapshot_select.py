"""Snapshot-backed alias lookup, ordering, distribution, and candidate selection."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

from fastapi import Request

from .config_snapshot import (
    AliasReference as _AliasReference,
    RoutingCandidate as _RoutingSnapshotCandidate,
    RoutingSnapshot as _RoutingSnapshot,
    active_routing_snapshot_holder as _active_routing_snapshot_holder,
)
from .request_metadata import _normalize_tui_family

# CFG-008: route families that require Anthropic-native credentials. The
# Codex/OpenAI Responses ingress has no native-Anthropic egress, so
# candidates carrying one of these codex-ingress route families are excluded
# from the Codex projection (Anthropic ingress owns them via their
# anthropic_route_family projection). This preserves the TOS boundary:
# Anthropic/Claude models never egress through Codex/OpenAI credentials.
_ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "anthropic_messages",
    }
)

# ---------------------------------------------------------------------------
# Injected runtime state
# ---------------------------------------------------------------------------
_rr_cursor: dict[tuple[str, str], int] = {}
_REQUEST_ROUTING_SNAPSHOT_STATE_KEY = "aawm_alias_routing_snapshot"
_REQUEST_ROUTING_SNAPSHOT_UNSET = object()


def configure_snapshot_runtime(
    *,
    round_robin_cursor: dict[tuple[str, str], int],
) -> None:
    """Bind the state-manager-owned round-robin cursor dict."""
    global _rr_cursor
    _rr_cursor = round_robin_cursor


# ---------------------------------------------------------------------------
# Snapshot holder accessors
# ---------------------------------------------------------------------------


def get_active_routing_snapshot() -> Optional[_RoutingSnapshot]:
    """Return the process-local active config-driven routing snapshot, if any."""
    return _active_routing_snapshot_holder.get()


def set_active_routing_snapshot(
    snapshot: Optional[_RoutingSnapshot],
) -> Optional[_RoutingSnapshot]:
    """Atomically activate ``snapshot`` (or clear it with ``None``)."""
    if snapshot is None:
        return _active_routing_snapshot_holder.swap(None)  # type: ignore[arg-type]
    return _active_routing_snapshot_holder.swap(snapshot)


def _is_alias_config_startup_failed() -> bool:
    """Return ``True`` if alias-config startup was attempted and failed.

    Lazy import avoids the ``config_startup -> snapshot_select`` circular
    dependency at module-load time.
    """
    from .config_startup import is_startup_failed

    return is_startup_failed()


def _get_request_routing_snapshot(
    request: Optional[Request] = None,
) -> Optional[_RoutingSnapshot]:
    """Capture one active snapshot reference for the lifetime of a request."""
    if request is None:
        if _is_alias_config_startup_failed():
            return None
        return get_active_routing_snapshot()

    state = getattr(request, "state", None)
    if state is None:
        if _is_alias_config_startup_failed():
            return None
        return get_active_routing_snapshot()

    cached = getattr(
        state,
        _REQUEST_ROUTING_SNAPSHOT_STATE_KEY,
        _REQUEST_ROUTING_SNAPSHOT_UNSET,
    )
    if cached is not _REQUEST_ROUTING_SNAPSHOT_UNSET:
        return cached if isinstance(cached, _RoutingSnapshot) else None

    snapshot = (
        None
        if _is_alias_config_startup_failed()
        else get_active_routing_snapshot()
    )
    setattr(state, _REQUEST_ROUTING_SNAPSHOT_STATE_KEY, snapshot)
    return snapshot


def _lookup_active_snapshot_canonical_alias(
    model: Any,
    *,
    request: Optional[Request] = None,
) -> Optional[str]:
    """Return the configured alias spelling for a case-insensitive model name."""
    if not isinstance(model, str):
        return None
    normalized = model.strip().casefold()
    if not normalized:
        return None
    snapshot = _get_request_routing_snapshot(request)
    if snapshot is None:
        return None
    for alias_name in snapshot.aliases:
        if alias_name.casefold() == normalized:
            return alias_name

    if normalized.startswith("aawm-"):
        stripped_alias = normalized.removeprefix("aawm-")
        for alias_name in snapshot.aliases:
            if alias_name.casefold() == stripped_alias:
                return alias_name
    return None


# ---------------------------------------------------------------------------
# Public shaping
# ---------------------------------------------------------------------------


def _routing_candidate_to_public_dict(
    candidate: _RoutingSnapshotCandidate,
    *,
    epoch_tag: Optional[str] = None,
) -> dict[str, Any]:
    """Shape a compiled ``RoutingCandidate`` into the legacy candidate-dict form.

    When ``epoch_tag`` is provided (snapshot-resolved candidates), it is
    carried as ``config_epoch_tag`` so downstream state-key construction
    can prefix cooldown/evidence/probe keys with the semantic digest.
    """
    shaped: dict[str, Any] = {
        "provider": candidate.provider,
        "model": candidate.model,
        "route_family": candidate.route_family,
        "last_resort": candidate.priority == 0,
    }
    # CFG-006: carry the optional authoritative candidate reasoning effort so
    # both ingress candidate-shaping paths can apply it uniformly.
    if candidate.reasoning_effort is not None:
        shaped["reasoning_effort"] = candidate.reasoning_effort
    if epoch_tag:
        shaped["config_epoch_tag"] = epoch_tag
    return shaped


# ---------------------------------------------------------------------------
# Ordering / distribution
# ---------------------------------------------------------------------------


def _order_snapshot_candidates_by_priority(
    candidates: Sequence[_RoutingSnapshotCandidate],
) -> list[_RoutingSnapshotCandidate]:
    """Pure ordering: descending priority; ``priority: 0`` always placed last."""
    non_zero = [c for c in candidates if c.priority != 0]
    zero = [c for c in candidates if c.priority == 0]
    non_zero_sorted = sorted(non_zero, key=lambda c: c.priority, reverse=True)
    return non_zero_sorted + zero


def _select_proportional_snapshot_candidate(
    candidates: Sequence[_RoutingSnapshotCandidate],
    weights: Mapping[str, float],
    rng: random.Random,
) -> _RoutingSnapshotCandidate:
    """Pure weighted tie-break among ``candidates`` using ``weights`` (by model)."""
    ordered = list(candidates)
    total = sum(max(0.0, weights.get(c.model, 0.0)) for c in ordered) or 1.0
    pick = rng.random() * total
    cumulative = 0.0
    for candidate in ordered:
        cumulative += max(0.0, weights.get(candidate.model, 0.0))
        if pick <= cumulative:
            return candidate
    return ordered[-1]


class RoundRobinCommitToken(NamedTuple):
    """Immutable receipt describing how a round-robin rotation must commit.

    Captured once per request (in the selection context) at enumeration time so
    the actual selection -- not any getter call multiplicity -- drives the single
    cursor advance. ``tied_candidate_ids`` is the stable, priority-ordered tied
    top-tier identity tuple; ``start_index`` is the cursor value read when the
    enumeration resolved (Wave 3 seam / diagnostics -- the commit itself keys off
    the actually selected member's position, never blindly ``start_index + 1``).
    """

    alias_name: str
    epoch_tag: str
    tied_candidate_ids: Tuple[Tuple[str, str], ...]
    start_index: int


class SelectionEnumeration(NamedTuple):
    """Per-request memoized alias enumeration + optional round-robin commit token."""

    candidates: Tuple[dict[str, Any], ...]
    commit_token: Optional[RoundRobinCommitToken]


def _select_round_robin_snapshot_candidate(
    tied: Sequence[_RoutingSnapshotCandidate],
    *,
    alias_name: str,
    epoch_tag: str = "",
) -> _RoutingSnapshotCandidate:
    """Pick the next tied candidate by READING the per-alias rotation cursor.

    Distinct from ``proportional``: this is a deterministic round-robin over the
    equal-top-priority candidates (in their stable priority ordering). It is a
    PURE read of the cursor -- it does NOT advance it. The cursor only advances
    via ``_commit_round_robin_selection`` once per request, keyed on the member
    that is actually selected, so repeated enumeration getter calls within a
    single request cannot desync the rotation from live traffic.
    """
    cursor = _rr_cursor.get((epoch_tag, alias_name), 0)
    return tied[cursor % len(tied)]


def _commit_round_robin_selection(
    token: Optional[RoundRobinCommitToken],
    *,
    selected_candidate: dict[str, Any],
) -> None:
    """Advance the rotation cursor to the slot AFTER the actually selected member.

    No-op when ``token`` is ``None`` (non-round-robin alias) or when the selected
    candidate is not a member of the rotated tied tier (affinity, last-resort, or
    lower-priority fallback selections must never consume top-tier rotation). The
    next cursor points immediately after the actual selected candidate's stable
    position -- never a blind ``start_index + 1`` -- so a fallback pick inside the
    tier (e.g. B chosen while A cools) rotates to C, not back to B.
    """
    if token is None:
        return
    identity = (selected_candidate.get("provider"), selected_candidate.get("model"))
    try:
        index = token.tied_candidate_ids.index(identity)
    except ValueError:
        return
    _rr_cursor[(token.epoch_tag, token.alias_name)] = (index + 1) % len(token.tied_candidate_ids)


def _apply_snapshot_alias_distribution_strategy(
    ordered: Sequence[_RoutingSnapshotCandidate],
    *,
    distribution_strategy: Optional[str],
    rng: random.Random,
    alias_name: str = "",
    epoch_tag: str = "",
) -> list[_RoutingSnapshotCandidate]:
    """Reorder the top priority-tier of ``ordered`` per ``distribution_strategy``.

    ``ordered`` is already sorted descending by priority with ``priority: 0``
    last (see ``_order_snapshot_candidates_by_priority``). When more than one
    candidate shares the top (highest, non-zero) priority tier:

    - ``proportional`` uses a weighted random pick to decide which tied
      candidate leads the returned ordering.
    - ``round_robin`` rotates a per-alias process-local cursor across the tied
      candidates so successive selections cycle through them deterministically.

    In both cases the remaining candidates (including any other tiers) keep
    their existing relative order as the fallback chain. Any other/absent
    strategy leaves the ordering untouched.
    """
    if distribution_strategy not in ("proportional", "round_robin") or len(ordered) < 2:
        return list(ordered)
    top_priority = ordered[0].priority
    tied = [c for c in ordered if c.priority == top_priority]
    if len(tied) < 2:
        return list(ordered)
    if distribution_strategy == "round_robin":
        winner = _select_round_robin_snapshot_candidate(tied, alias_name=alias_name, epoch_tag=epoch_tag)
    else:
        weights = {c.model: c.weight for c in tied}
        winner = _select_proportional_snapshot_candidate(tied, weights, rng)
    remainder = [c for c in ordered if c is not winner]
    return [winner, *remainder]


# ---------------------------------------------------------------------------
# TUI / schedule gates
# ---------------------------------------------------------------------------


def _is_tui_attached_candidate_eligible(
    candidate: _RoutingSnapshotCandidate | _AliasReference,
    *,
    client_product_label: Optional[str],
) -> bool:
    """Per-model TUI gate: an undetermined TUI excludes only ``tui_attached`` candidates."""
    if not candidate.tui_attached:
        return True
    if not client_product_label:
        return False
    return _normalize_tui_family(client_product_label) == _normalize_tui_family(
        candidate.tui_attached
    )


def _is_tui_excluded_candidate_eligible(
    candidate: _RoutingSnapshotCandidate | _AliasReference,
    *,
    client_product_label: Optional[str],
) -> bool:
    """Per-model TUI exclusion gate (CFG-008).

    Complementary to :func:`_is_tui_attached_candidate_eligible`: a candidate
    carrying ``tui_excluded`` is ineligible ONLY for requests whose identified
    client product name matches (product name only, version-insensitive).
    Missing/unknown/undetermined origins keep the candidate eligible, so an
    excluded-for-one-TUI candidate can serve as the default branch tail.
    """
    if not candidate.tui_excluded:
        return True
    if not client_product_label:
        return True
    return _normalize_tui_family(client_product_label) != _normalize_tui_family(
        candidate.tui_excluded
    )


def _is_snapshot_candidate_in_schedule_window(
    candidate: _RoutingSnapshotCandidate,
    *,
    now_utc: datetime,
) -> bool:
    """Schedule gate: only prevents NEW affinity, never evicts existing state."""
    schedule = candidate.schedule
    if schedule is None:
        return True
    return schedule.start <= now_utc <= schedule.end


# ---------------------------------------------------------------------------
# Snapshot-driven resolution
# ---------------------------------------------------------------------------
def _resolve_dispatch_target(
    alias_name: str,
    *,
    client_product_label: Optional[str],
    snapshot: Optional[_RoutingSnapshot],
) -> Optional[str]:
    """Resolve a TUI dispatch target, returning ``None`` for blocked/no target."""
    if snapshot is None or alias_name not in snapshot.aliases:
        return None

    alias = snapshot.aliases[alias_name]
    if alias.dispatch is None:
        return None

    tui_family = _normalize_tui_family(client_product_label)
    if tui_family in alias.dispatch.blocked_tui_families:
        return None
    for rule in alias.dispatch.by_tui:
        if rule.tui_family == tui_family:
            return rule.target_alias
    return alias.dispatch.default


def _order_snapshot_entries_by_priority(
    entries: Sequence[_RoutingSnapshotCandidate | _AliasReference],
) -> list[_RoutingSnapshotCandidate | _AliasReference]:
    non_zero = [entry for entry in entries if entry.priority != 0]
    zero = [entry for entry in entries if entry.priority == 0]
    return sorted(
        non_zero,
        key=lambda entry: entry.priority,
        reverse=True,
    ) + zero


def _shape_snapshot_candidate(
    candidate: _RoutingSnapshotCandidate,
    *,
    ingress: str,
    epoch_tag: str,
) -> Optional[dict[str, Any]]:
    if ingress == "codex":
        if candidate.route_family in _ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES:
            return None
        return _routing_candidate_to_public_dict(candidate, epoch_tag=epoch_tag)
    try:
        return _routing_candidate_to_anthropic_public_dict(
            candidate,
            epoch_tag=epoch_tag,
        )
    except ValueError:
        return None


def _resolve_snapshot_alias_candidates(
    alias_name: str,
    *,
    ingress: str,
    client_product_label: Optional[str],
    now_utc: datetime,
    snapshot: _RoutingSnapshot,
    include_out_of_schedule: bool = False,
    path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Resolve one config alias to concrete candidates without nested loops."""
    if alias_name in path:
        return []
    alias = snapshot.aliases.get(alias_name)
    if alias is None:
        return []

    next_path = (*path, alias_name)
    if alias.dispatch is not None:
        target = _resolve_dispatch_target(
            alias_name,
            client_product_label=client_product_label,
            snapshot=snapshot,
        )
        if target is None:
            return []
        return _resolve_snapshot_alias_candidates(
            target,
            ingress=ingress,
            client_product_label=client_product_label,
            now_utc=now_utc,
            snapshot=snapshot,
            include_out_of_schedule=include_out_of_schedule,
            path=next_path,
        )

    resolved: list[dict[str, Any]] = []
    for entry in _order_snapshot_entries_by_priority(alias.candidates):
        if not _is_tui_attached_candidate_eligible(
            entry, client_product_label=client_product_label
        ) or not _is_tui_excluded_candidate_eligible(
            entry, client_product_label=client_product_label
        ):
            continue
        if isinstance(entry, _AliasReference):
            children = _resolve_snapshot_alias_candidates(
                entry.alias_name,
                ingress=ingress,
                client_product_label=client_product_label,
                now_utc=now_utc,
                snapshot=snapshot,
                include_out_of_schedule=include_out_of_schedule,
                path=next_path,
            )
            for child in children:
                shaped = dict(child)
                shaped["selection_priority"] = entry.priority
                shaped["last_resort"] = entry.priority == 0
                shaped["alias_reference"] = entry.alias_name
                shaped["alias_path"] = [*next_path, entry.alias_name]
                if alias.distribution_strategy is not None:
                    shaped["selection_group"] = alias.name
                    shaped["selection_strategy"] = alias.distribution_strategy
                    shaped["selection_choice"] = entry.alias_name
                    shaped["selection_weight"] = entry.weight
                resolved.append(shaped)
            continue

        if (
            not include_out_of_schedule
            and not _is_snapshot_candidate_in_schedule_window(entry, now_utc=now_utc)
        ):
            continue
        shaped = _shape_snapshot_candidate(
            entry,
            ingress=ingress,
            epoch_tag=snapshot.config_hash,
        )
        if shaped is None:
            continue
        shaped["selection_priority"] = entry.priority
        shaped["resolved_alias"] = alias.name
        shaped["alias_path"] = list(next_path)
        if alias.distribution_strategy is not None:
            shaped["selection_group"] = alias.name
            shaped["selection_strategy"] = alias.distribution_strategy
            shaped["selection_choice"] = f"{entry.provider}:{entry.model}"
            shaped["selection_weight"] = entry.weight
        resolved.append(shaped)
    return resolved


def _select_snapshot_candidates(
    canonical_alias: str,
    *,
    ingress: str,
    client_product_label: Optional[str] = None,
    now_utc: Optional[datetime] = None,
    request: Optional[Request] = None,
    include_out_of_schedule: bool = False,
) -> tuple[dict[str, Any], ...]:
    snapshot = _get_request_routing_snapshot(request)
    if snapshot is None:
        return ()
    return tuple(
        _resolve_snapshot_alias_candidates(
            canonical_alias,
            ingress=ingress,
            client_product_label=client_product_label,
            now_utc=now_utc or datetime.now(timezone.utc),
            snapshot=snapshot,
            include_out_of_schedule=include_out_of_schedule,
        )
    )


def _routing_candidate_to_anthropic_public_dict(
    candidate: _RoutingSnapshotCandidate,
    *,
    epoch_tag: Optional[str] = None,
) -> dict[str, Any]:
    """Shape a compiled candidate for Anthropic ingress using its anthropic_route_family.

    Fail closed: if the candidate has no resolved anthropic_route_family
    (should not happen after compile-time validation), raise ValueError.
    """
    anthropic_rf = candidate.anthropic_route_family
    if anthropic_rf is None:
        raise ValueError(
            f"candidate model {candidate.model!r} has no anthropic_route_family; "
            f"this indicates a compile-time validation gap"
        )
    shaped: dict[str, Any] = {
        "provider": candidate.provider,
        "model": candidate.model,
        "route_family": anthropic_rf,
        "last_resort": candidate.priority == 0,
    }
    if candidate.reasoning_effort is not None:
        shaped["reasoning_effort"] = candidate.reasoning_effort
    if epoch_tag:
        shaped["config_epoch_tag"] = epoch_tag
    return shaped


def _derive_round_robin_commit_token(
    canonical_alias: str,
    *,
    request: Request,
    ingress: str,
    client_product_label: Optional[str],
    now_utc: Optional[datetime] = None,
    candidates: Optional[Sequence[dict[str, Any]]] = None,
) -> Optional[RoundRobinCommitToken]:
    """Build the commit token for a snapshot round-robin alias."""
    snapshot = _get_request_routing_snapshot(request)
    if snapshot is None or canonical_alias not in snapshot.aliases:
        return None
    alias = snapshot.aliases[canonical_alias]
    if alias.distribution_strategy != "round_robin":
        return None
    resolved_candidates = candidates
    if resolved_candidates is None:
        resolved_candidates = _select_snapshot_candidates(
            canonical_alias,
            ingress=ingress,
            client_product_label=client_product_label,
            now_utc=now_utc,
            request=request,
        )
    if len(resolved_candidates) < 2:
        return None
    top_priority = resolved_candidates[0].get("selection_priority", 0)
    tied = [
        candidate
        for candidate in resolved_candidates
        if candidate.get("selection_priority", 0) == top_priority
    ]
    if len(tied) < 2:
        return None
    epoch_tag = snapshot.config_hash
    start_index = _rr_cursor.get((epoch_tag, canonical_alias), 0)
    return RoundRobinCommitToken(
        alias_name=canonical_alias,
        epoch_tag=epoch_tag,
        tied_candidate_ids=tuple(
            (str(candidate["provider"]), str(candidate["model"]))
            for candidate in tied
        ),
        start_index=start_index,
    )


# ---------------------------------------------------------------------------
# Selection-context memoization
# ---------------------------------------------------------------------------


def _get_aawm_alias_selection_context(
    request: Request,
) -> dict[tuple[str, str], SelectionEnumeration]:
    """Per-request cache of ingress/alias enumerations on ``request.state``.

    Mirrors the ``aawm_alias_request_local_*`` request-state cache pattern so the
    alias enumeration resolves exactly once per request even though the wrapper,
    the candidate-state builder, and the affinity resolver each need it.
    """
    context = getattr(request.state, "aawm_alias_selection_context", None)
    if isinstance(context, dict):
        return context
    context = {}
    setattr(request.state, "aawm_alias_selection_context", context)
    return context


def _resolve_aawm_alias_selection_enumeration(
    request: Request,
    canonical_alias: str,
    *,
    ingress: str,
    client_product_label: Optional[str] = None,
) -> SelectionEnumeration:
    """Resolve (and memoize) the ordered candidate enumeration + commit token.

    The generic snapshot selector is invoked exactly once per request, ingress,
    and canonical alias. Every subsequent call site consumes the cached
    enumeration.
    """
    context = _get_aawm_alias_selection_context(request)
    cache_key = (ingress, canonical_alias)
    cached = context.get(cache_key)
    if cached is not None:
        return cached
    candidates = _select_snapshot_candidates(
        canonical_alias,
        ingress=ingress,
        client_product_label=client_product_label,
        request=request,
    )
    token = _derive_round_robin_commit_token(
        canonical_alias,
        request=request,
        ingress=ingress,
        client_product_label=client_product_label,
        candidates=candidates,
    )
    enumeration = SelectionEnumeration(candidates=candidates, commit_token=token)
    context[cache_key] = enumeration
    return enumeration
