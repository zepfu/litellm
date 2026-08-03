"""Snapshot ordering, distribution, TUI/schedule gates, and alias-candidate getters.

Wave 5A extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.  The round-robin cursor dict
(``_round_robin_cursor_by_alias``) remains owned by the god module and is
injected via :func:`configure_snapshot_runtime` (deferred to Wave 5B for
state-manager ownership).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

from fastapi import Request

from .config_snapshot import (
    RoutingCandidate as _RoutingSnapshotCandidate,
    RoutingSnapshot as _RoutingSnapshot,
    active_routing_snapshot_holder as _active_routing_snapshot_holder,
)
from .policy import (
    CODEX_AUTO_AGENT_CANDIDATES as _CODEX_AUTO_AGENT_CANDIDATES,
    CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS as _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_READ_PILOT_ALIAS_NAME = "read"

# ---------------------------------------------------------------------------
# Injected runtime state (round-robin cursor stays in god module for Wave 5A)
# ---------------------------------------------------------------------------
_rr_cursor: dict[tuple[str, str], int] = {}
_candidates_getter: Optional[Callable[..., tuple[dict[str, Any], ...]]] = None


def configure_snapshot_runtime(
    *,
    round_robin_cursor: dict[tuple[str, str], int],
    get_candidates_for_alias: Optional[Callable[..., tuple[dict[str, Any], ...]]] = None,
) -> None:
    """Bind the god-module-owned round-robin cursor dict."""
    global _rr_cursor, _candidates_getter
    _rr_cursor = round_robin_cursor
    _candidates_getter = get_candidates_for_alias


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
    candidate: _RoutingSnapshotCandidate,
    *,
    client_product_label: Optional[str],
) -> bool:
    """Per-model TUI gate: an undetermined TUI excludes only ``tui_attached`` candidates."""
    if not candidate.tui_attached:
        return True
    if not client_product_label:
        return False
    product_name = client_product_label.split("/", 1)[0]
    return product_name == candidate.tui_attached


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


def _resolve_read_pilot_eligible_candidates(
    *,
    client_product_label: Optional[str],
    now_utc: datetime,
    snapshot: Optional[_RoutingSnapshot] = None,
) -> Optional[list[_RoutingSnapshotCandidate]]:
    """Return the eligibility-filtered, priority-ordered ``read`` alias candidates.

    ``None`` means there is no active snapshot ``read`` alias (callers use the
    static fallback table). An empty list means every candidate was gated out
    (TUI/schedule) -- callers fail closed rather than dispatching a rejected
    candidate. Shared by the enumeration getter and the round-robin commit-token
    derivation so both observe the identical tied top-tier ordering.

    When ``snapshot`` is provided, it is used directly instead of fetching
    the global holder (Finding 5: single-capture coherence).
    """
    if snapshot is None:
        snapshot = get_active_routing_snapshot()
    if snapshot is None or _READ_PILOT_ALIAS_NAME not in snapshot.aliases:
        return None
    alias = snapshot.aliases[_READ_PILOT_ALIAS_NAME]
    ordered = _order_snapshot_candidates_by_priority(alias.candidates)
    return [
        candidate
        for candidate in ordered
        if _is_tui_attached_candidate_eligible(candidate, client_product_label=client_product_label)
        and _is_snapshot_candidate_in_schedule_window(candidate, now_utc=now_utc)
    ]


def _select_read_pilot_snapshot_candidates(
    *,
    client_product_label: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> tuple[dict[str, Any], ...]:
    """Resolve the ordered ``read`` alias candidate tuple from the active snapshot.

    Falls back to the hard-coded ``_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS`` table
    when no snapshot has been activated (or the snapshot has no ``read`` alias),
    so the pilot degrades gracefully instead of raising.

    CFG-002 Finding 2: failure state is checked FIRST, before any snapshot
    or static branch.  Once failure is published, all paths return empty.
    CFG-002 Finding 5: exactly one snapshot reference is captured and used
    for eligibility, distribution, hash, and shaping.
    """
    # Finding 2: fail-closed check before any snapshot/static branch.
    if _is_alias_config_startup_failed():
        return ()
    # Finding 5: capture exactly one snapshot reference.
    snapshot = get_active_routing_snapshot()
    resolved_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    eligible = _resolve_read_pilot_eligible_candidates(
        client_product_label=client_product_label,
        now_utc=resolved_now,
        snapshot=snapshot,
    )
    if eligible is None:
        if snapshot is None:
            # Genuine legacy / no-config state: degrade to the static table.
            return _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(
                _READ_PILOT_ALIAS_NAME,
                _CODEX_AUTO_AGENT_CANDIDATES,
            )
        # Snapshot active but has no read alias: fail closed.
        return ()
    if not eligible:
        return ()
    assert snapshot is not None
    alias = snapshot.aliases[_READ_PILOT_ALIAS_NAME]
    epoch_tag = snapshot.config_hash
    distributed = _apply_snapshot_alias_distribution_strategy(
        eligible,
        distribution_strategy=alias.distribution_strategy,
        rng=random.Random(),
        alias_name=_READ_PILOT_ALIAS_NAME,
        epoch_tag=epoch_tag,
    )
    return tuple(_routing_candidate_to_public_dict(c, epoch_tag=epoch_tag) for c in distributed)


def _select_read_pilot_snapshot_candidates_anthropic(
    *,
    client_product_label: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> Optional[tuple[dict[str, Any], ...]]:
    """Resolve the ordered ``read`` alias candidates for Anthropic Messages ingress.

    Returns the same snapshot-resolved, priority-ordered, eligibility-filtered
    candidate set as the Codex ingress path, but with each candidate's
    ``route_family`` replaced by its ``anthropic_route_family`` projection.

    Returns `None` when there is no active snapshot ``read`` alias (callers
    fall back to the static table). Returns an empty tuple when every
    candidate is gated out (fail closed, same as Codex ingress).

    Ingress isolation: Codex and Anthropic ingress each see only their own
    route-family projection. No cross-provider fallback is introduced.

    CFG-002 Finding 2: failure state is checked FIRST.
    CFG-002 Finding 5: exactly one snapshot reference captured per call.
    """
    # Finding 2: fail-closed check before any snapshot/static branch.
    if _is_alias_config_startup_failed():
        return ()
    # Finding 5: capture exactly one snapshot reference.
    snapshot = get_active_routing_snapshot()
    resolved_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    eligible = _resolve_read_pilot_eligible_candidates(
        client_product_label=client_product_label,
        now_utc=resolved_now,
        snapshot=snapshot,
    )
    if eligible is None:
        if snapshot is None:
            # Genuine legacy / no-config state: callers fall back to static table.
            return None
        # Snapshot active but has no read alias: fail closed.
        return ()
    if not eligible:
        return ()
    assert snapshot is not None
    alias = snapshot.aliases[_READ_PILOT_ALIAS_NAME]
    epoch_tag = snapshot.config_hash
    distributed = _apply_snapshot_alias_distribution_strategy(
        eligible,
        distribution_strategy=alias.distribution_strategy,
        rng=random.Random(),
        alias_name=_READ_PILOT_ALIAS_NAME,
        epoch_tag=epoch_tag,
    )
    return tuple(
        _routing_candidate_to_anthropic_public_dict(c, epoch_tag=epoch_tag)
        for c in distributed
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
    alias_model: str,
    *,
    client_product_label: Optional[str],
    now_utc: Optional[datetime] = None,
) -> Optional[RoundRobinCommitToken]:
    """Build the commit token for a ``round_robin`` snapshot alias, else ``None``.

    Only the snapshot-driven ``read`` alias participates in round-robin rotation;
    every other alias (static-table lanes, non-round-robin strategies, single-
    candidate tiers) yields ``None`` so the commit path is a no-op for them.
    """
    if alias_model != _READ_PILOT_ALIAS_NAME:
        return None
    snapshot = get_active_routing_snapshot()
    if snapshot is None or _READ_PILOT_ALIAS_NAME not in snapshot.aliases:
        return None
    alias = snapshot.aliases[_READ_PILOT_ALIAS_NAME]
    if alias.distribution_strategy != "round_robin":
        return None
    resolved_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    eligible = _resolve_read_pilot_eligible_candidates(
        client_product_label=client_product_label,
        now_utc=resolved_now,
    )
    if not eligible or len(eligible) < 2:
        return None
    top_priority = eligible[0].priority
    tied = [c for c in eligible if c.priority == top_priority]
    if len(tied) < 2:
        return None
    epoch_tag = snapshot.config_hash
    start_index = _rr_cursor.get((epoch_tag, _READ_PILOT_ALIAS_NAME), 0)
    return RoundRobinCommitToken(
        alias_name=_READ_PILOT_ALIAS_NAME,
        epoch_tag=epoch_tag,
        tied_candidate_ids=tuple((c.provider, c.model) for c in tied),
        start_index=start_index,
    )


# ---------------------------------------------------------------------------
# Selection-context memoization
# ---------------------------------------------------------------------------


def _get_aawm_alias_selection_context(
    request: Request,
) -> dict[str, SelectionEnumeration]:
    """Per-request cache of ``alias_model -> SelectionEnumeration`` on ``request.state``.

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
    alias_model: str,
    *,
    client_product_label: Optional[str] = None,
) -> SelectionEnumeration:
    """Resolve (and memoize) the ordered candidate enumeration + commit token.

    The underlying getter ``_get_codex_auto_agent_candidates_for_alias`` is
    invoked exactly once per ``(request, alias_model)`` -- every subsequent call
    site consumes the cached enumeration, so cursor reads stay consistent and the
    getter cannot advance rotation multiple times per live request.
    """
    context = _get_aawm_alias_selection_context(request)
    cached = context.get(alias_model)
    if cached is not None:
        return cached
    getter = _candidates_getter or _get_codex_auto_agent_candidates_for_alias
    candidates = tuple(
        getter(
            alias_model,
            client_product_label=client_product_label,
        )
    )
    token = _derive_round_robin_commit_token(
        alias_model,
        client_product_label=client_product_label,
    )
    enumeration = SelectionEnumeration(candidates=candidates, commit_token=token)
    context[alias_model] = enumeration
    return enumeration


# ---------------------------------------------------------------------------
# Alias-candidate getters
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_candidates_for_alias(
    alias_model: str,
    *,
    client_product_label: Optional[str] = None,
) -> tuple[dict[str, Any], ...]:
    # CFG-002 Finding 2: check failure state FIRST, before any snapshot or
    # static branch.  Once failure is published, all aliases return empty.
    if _is_alias_config_startup_failed():
        return ()
    if alias_model == _READ_PILOT_ALIAS_NAME:
        return _select_read_pilot_snapshot_candidates(client_product_label=client_product_label)
    # When a snapshot is active, only config-defined aliases (read) and
    # explicitly registered legacy aliases are supported.  Arbitrary
    # normalized aliases fail closed rather than receiving the generic
    # static OpenAI fallback.
    if get_active_routing_snapshot() is not None:
        candidates = _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(alias_model)
        if candidates is not None:
            return candidates
        return ()
    candidates = _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(
        alias_model,
        _CODEX_AUTO_AGENT_CANDIDATES,
    )
    return candidates
