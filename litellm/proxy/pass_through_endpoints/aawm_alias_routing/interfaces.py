"""Typed seam contracts for the shared alias-routing candidate loop (Wave 2).

The candidate retry loop (``candidate_loop.handle_alias_route``) used to live
inline in ``llm_passthrough_endpoints.py`` and consumed a set of type-erased
``Callable[..., ...]`` seams. That erasure let a production applicator gain a
new required/optional kwarg that a test stub silently fell behind on (the
``b9c97f9540``-class silent stub-rot failure). This module replaces those
erased seams with explicit ``Protocol`` contracts whose required keyword
parameters are spelled out, plus the frozen value objects the R3-1
single-flight restructure passes across the seam boundary.

Design notes:

- The ``Protocol`` ``__call__`` signatures use EXPLICIT keyword parameters
  (never ``**kwargs``). The seam-contract test verifies these with
  ``inspect.signature``; a runtime-checkable ``isinstance`` is only an
  attribute-presence smoke check, not signature proof.
- ``CandidateSelection.from_legacy_dict`` is the transitional bridge that lets
  the existing selector (which still returns a plain dict) feed the typed loop
  without rewriting the selector internals in this wave.
- ``CooldownPublicationPlan`` is the single immutable artifact the R3-1
  restructure produces from one failure: the loop derives the synchronous
  memory targets, the post-release durable targets, the applied scope, and the
  request-local action from the SAME plan so telemetry, waiter visibility, and
  Redis state cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request
    from starlette.responses import Response


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasCandidate:
    """One routable alias candidate (typed view of the legacy candidate dict)."""

    provider: str
    model: str
    route_family: str
    last_resort: bool = False
    metadata_gate: Optional[str] = None
    default_reasoning_effort: Optional[str] = None
    # CFG-006: optional authoritative candidate-level reasoning effort
    # (canonical vocabulary); carried verbatim from the compiled snapshot
    # candidate shape.
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class CandidateSelection:
    """Typed result of candidate selection for one loop attempt.

    Mirrors the keys the retry loop consumes off the legacy selector's dict
    return value. ``config_epoch_tag`` is reserved for the Wave-3 semantic
    cooldown-epoch work and defaults to ``None`` (bare/legacy keys) here.
    """

    candidate: AliasCandidate
    lane_key: str
    cooldown_key: str
    session_key: Optional[str] = None
    selection_reason: Optional[str] = None
    skipped: list[Any] = field(default_factory=list)
    in_flight_session: bool = False
    cooldown_seconds: float = 0.0
    cooldown_state_source: Optional[str] = None
    affinity_state_source: Optional[str] = None
    config_epoch_tag: Optional[str] = None

    @classmethod
    def from_legacy_dict(cls, selection: dict[str, Any]) -> "CandidateSelection":
        """Build a typed selection from the legacy selector dict.

        Transitional bridge so the selector migrates to the typed seam without
        rewriting its internals in this wave. The legacy ``candidate`` value is
        itself a dict (provider/model/route_family/last_resort/...).
        """
        raw_candidate = selection.get("candidate") or {}
        candidate = AliasCandidate(
            provider=str(raw_candidate.get("provider") or ""),
            model=str(raw_candidate.get("model") or ""),
            route_family=str(raw_candidate.get("route_family") or ""),
            last_resort=bool(raw_candidate.get("last_resort")),
            metadata_gate=raw_candidate.get("metadata_gate"),
            default_reasoning_effort=raw_candidate.get("default_reasoning_effort"),
            reasoning_effort=raw_candidate.get("reasoning_effort"),
        )
        return cls(
            candidate=candidate,
            lane_key=str(selection.get("lane_key") or ""),
            cooldown_key=str(selection.get("cooldown_key") or ""),
            session_key=selection.get("session_key"),
            selection_reason=selection.get("selection_reason"),
            skipped=list(selection.get("skipped") or []),
            in_flight_session=bool(selection.get("in_flight_session")),
            cooldown_seconds=float(selection.get("cooldown_seconds") or 0.0),
            cooldown_state_source=selection.get("cooldown_state_source"),
            affinity_state_source=selection.get("affinity_state_source"),
            config_epoch_tag=selection.get("config_epoch_tag"),
        )


@dataclass(frozen=True)
class CooldownPublicationPlan:
    """Immutable publication plan for one candidate failure (R3-1).

    Produced once by the publication-plan resolver from a classified failure.
    The loop publishes ``memory_keys`` synchronously (direct ``state.py``
    writes) BEFORE releasing the probe lock, then persists exactly
    ``durable_keys`` AFTER release. ``applied_scope`` is the single scope
    string recorded on the attempt record so telemetry reports the scope that
    was actually applied. ``request_local_action`` carries the request-local
    exclusion/cooldown directive (applied post-release) for scopes that do not
    publish shared state.
    """

    memory_keys: tuple[str, ...] = ()
    durable_keys: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    applied_scope: str = "none"
    request_local_action: Optional[str] = None
    grok_account_quota_exhausted: bool = False
    kimi_failure_metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Typed seam protocols (explicit keyword parameters -- no ``**kwargs``)
# ---------------------------------------------------------------------------


@runtime_checkable
class GetActiveCooldownStateFn(Protocol):
    """Query active cooldown remaining for a key (may await durable cache).

    Returns ``(remaining_seconds, source_label)`` where ``source_label`` is
    one of ``"memory"``, ``"negative_cache"``, ``"durable"``, or
    ``"local_fallback"``.
    """

    def __call__(self, cooldown_key: str) -> Awaitable[tuple[float, str]]:
        ...


@runtime_checkable
class SelectCandidateFn(Protocol):
    """Select the next candidate for one loop attempt (legacy dict return)."""

    async def __call__(
        self,
        *,
        request: "Request",
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class PerformCandidateRequestFn(Protocol):
    """Perform the upstream provider call for a selected candidate."""

    async def __call__(
        self,
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> "Response":
        ...


@runtime_checkable
class ResolveCooldownPublicationFn(Protocol):
    """Classify one failure and produce the immutable publication plan.

    Synchronous and pure: it records read-pilot evidence and resolves the scope
    + target keys, but performs NO I/O. The loop owns all publishing.
    """

    def __call__(
        self,
        *,
        request: Optional["Request"],
        candidate: dict[str, Any],
        lane_key: Optional[str],
        selected_cooldown_key: str,
        cooldown_seconds: float,
        error_class: Optional[str],
        grok_account_quota_exhausted: bool = False,
        kimi_failure_metadata: Optional[dict[str, Any]] = None,
        is_read_pilot_lane: bool = False,
    ) -> CooldownPublicationPlan:
        ...


@runtime_checkable
class PublishCooldownMemoryFn(Protocol):
    """Synchronously publish cooldowns into process-local memory.

    Called INSIDE the probe lock so waiters observe the cooldown before the
    lock is released (R3-1 single-flight). Must not await.
    """

    def __call__(self, *, keys: Sequence[str], seconds: float) -> None:
        ...


@runtime_checkable
class RecordReadPilotEvidenceFn(Protocol):
    """Synchronously record one read-pilot failure observation."""

    def __call__(
        self,
        *,
        cooldown_key: str,
        exc: Exception,
        attempt_record: dict[str, Any],
    ) -> None:
        ...


@runtime_checkable
class GetKimiFailureMetadataFn(Protocol):
    """Extract allowlisted Kimi failure metadata without suspension."""

    def __call__(
        self,
        exc: Exception,
        *,
        candidate: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        ...


@runtime_checkable
class ClassifyKimiFailureFn(Protocol):
    """Classify allowlisted Kimi failure metadata synchronously."""

    def __call__(
        self,
        metadata: Optional[dict[str, Any]],
    ) -> Optional[str]:
        ...


@runtime_checkable
class ClassifyRetryableFailureFn(Protocol):
    """Classify a provider exception into the retry vocabulary."""

    def __call__(self, exc: Exception) -> Optional[str]:
        ...


@runtime_checkable
class IsGrokAccountQuotaFailureFn(Protocol):
    """Return whether a failure exhausted the selected Grok account lane."""

    def __call__(
        self,
        exc: Exception,
        *,
        candidate: dict[str, Any],
    ) -> bool:
        ...


@runtime_checkable
class GetCooldownSecondsFn(Protocol):
    """Resolve the cooldown duration for one candidate failure."""

    def __call__(
        self,
        exc: Exception,
        *,
        candidate: dict[str, Any],
    ) -> float:
        ...


@runtime_checkable
class PersistCooldownFn(Protocol):
    """Persist cooldown keys to durable Redis (post-release, may await)."""

    async def __call__(self, *, keys: Sequence[str], seconds: float) -> None:
        ...


@runtime_checkable
class SetSessionAffinityFn(Protocol):
    """Pin session affinity after a successful candidate probe."""

    async def __call__(
        self,
        session_key: Optional[str],
        candidate: dict[str, Any],
    ) -> object:
        ...


@runtime_checkable
class AddAliasMetadataFn(Protocol):
    """Attach alias-routing metadata to the outgoing candidate request body."""

    def __call__(
        self,
        request_body: dict[str, Any],
        *,
        request: "Request",
        selection: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class RaiseRedispatchFn(Protocol):
    """Raise the redispatch-required terminal error for in-flight cooldowns."""

    def __call__(
        self,
        *,
        candidate: dict[str, Any],
        lane_key: Optional[str],
        cooldown_seconds: float,
        error_tokens: set[str],
        alias_model: str,
        error_class: str,
        cooldown_scope: Optional[str],
        error_status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        failure_phase: Optional[str] = None,
        attempted_provider_call: Optional[bool] = None,
        audit_events: Optional[list[Any]] = None,
        attempts: Optional[list[Any]] = None,
        skipped_candidates: Optional[list[Any]] = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class AliasRouteServices:
    """Immutable bundle of typed seams consumed by the candidate loop.

    The two wrappers (Codex / Anthropic) assemble this from the existing
    production functions; the loop depends only on these contracts, never on
    the god-module directly.
    """

    select_candidate_fn: SelectCandidateFn
    perform_candidate_request_fn: PerformCandidateRequestFn
    resolve_cooldown_publication_fn: ResolveCooldownPublicationFn
    publish_cooldown_memory_fn: PublishCooldownMemoryFn
    persist_cooldown_fn: PersistCooldownFn
    set_session_affinity_fn: SetSessionAffinityFn
    add_alias_metadata_fn: AddAliasMetadataFn
    raise_redispatch_fn: RaiseRedispatchFn
