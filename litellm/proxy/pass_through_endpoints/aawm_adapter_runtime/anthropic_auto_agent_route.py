"""Wave 7 extraction: Anthropic auto-agent alias route owner.

Behavior-preserving extraction of the ``_handle_auto_agent_alias_route``
(legacy seam facade, god-module lines ~7025-7170) and
``_handle_anthropic_auto_agent_alias_route`` (production wrapper,
god-module lines ~7172-7227) bodies from ``llm_passthrough_endpoints.py``.

Do not import llm_passthrough_endpoints at module scope.

Google Code Assist and Antigravity-only paths are omitted per operator
removal scope.  Retained provider paths: Codex/OpenAI, Grok, OpenRouter,
NVIDIA, OpenCode, Kimi, Alibaba, and native Anthropic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Sequence,
    cast,
)

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
    PerformCandidateRequestFn,
    SelectCandidateFn,
    SetSessionAffinityFn,
)

if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request
    from starlette.responses import Response

    from litellm.proxy._types import UserAPIKeyAuth

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasFamilyState,
    )

Payload = dict[str, Any]

# ---------------------------------------------------------------------------
# Callable type aliases matching the legacy seam contract
# ---------------------------------------------------------------------------

_AutoAgentAliasSelectionFn = Callable[..., Awaitable[dict[str, Any]]]
_AutoAgentAliasMetadataFn = Callable[..., dict[str, Any]]

HandleAliasRouteFn = Callable[..., Awaitable["Response"]]
"""Typed reference to ``candidate_loop.handle_alias_route``."""

ResolveCooldownPublicationFn = Callable[..., CooldownPublicationPlan]
"""Typed reference to ``_resolve_auto_agent_cooldown_publication_plan``."""

NormalizeAliasModelFn = Callable[[Any], Optional[str]]
"""``_normalize_anthropic_auto_agent_alias_model``."""

PerformCandidateRequestDirectFn = Callable[..., Awaitable["Response"]]
"""``_perform_anthropic_auto_agent_alias_candidate_request``."""

SelectCandidateDirectFn = Callable[..., Awaitable[dict[str, Any]]]
"""``_select_anthropic_auto_agent_candidate``."""

PublishCooldownMemoryFn = Callable[..., None]
"""``_publish_anthropic_cooldown_memory``."""

PersistCooldownFn = Callable[..., Awaitable[None]]
"""``_persist_anthropic_cooldown_durable``."""

SetSessionAffinityDirectFn = Callable[..., Awaitable[object]]
"""``_set_anthropic_auto_agent_session_affinity``."""

AddAliasMetadataDirectFn = Callable[..., dict[str, Any]]
"""``_add_anthropic_auto_agent_alias_metadata``."""

RaiseRedispatchFn = Callable[..., None]
"""``_raise_anthropic_auto_agent_redispatch_required``."""

GetCandidatesForAliasFn = Callable[[str], list[Any]]
"""``_get_anthropic_auto_agent_candidates_for_alias``."""

GetActiveCooldownStateFn = Callable[[str], Awaitable[tuple[float, str]]]
"""``_get_anthropic_auto_agent_active_cooldown_state``."""


# ---------------------------------------------------------------------------
# Runtime dependency bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicAutoAgentRouteRuntime:
    """Host-owned callbacks/state for the auto-agent alias route functions.

    Every field is injected by the integrator so that this module never
    imports the god module at module scope.
    """

    # -- shared infrastructure --
    handle_alias_route: HandleAliasRouteFn
    resolve_cooldown_publication: ResolveCooldownPublicationFn

    # -- legacy facade (_handle_auto_agent_alias_route) --
    anthropic_family_state: "AliasFamilyState"
    codex_family_state: "AliasFamilyState"

    # -- production wrapper (_handle_anthropic_auto_agent_alias_route) --
    normalize_alias_model: NormalizeAliasModelFn
    default_alias_model: str
    perform_candidate_request: PerformCandidateRequestDirectFn
    select_candidate: SelectCandidateDirectFn
    publish_cooldown_memory: PublishCooldownMemoryFn
    persist_cooldown_durable: PersistCooldownFn
    set_session_affinity: SetSessionAffinityDirectFn
    add_alias_metadata: AddAliasMetadataDirectFn
    raise_redispatch_required: RaiseRedispatchFn
    get_candidates_for_alias: GetCandidatesForAliasFn
    get_active_cooldown_state: GetActiveCooldownStateFn


# ---------------------------------------------------------------------------
# Seam disposition (for test parity and documentation)
# ---------------------------------------------------------------------------

ANTHROPIC_AUTO_AGENT_ROUTE_SEAM_DISPOSITION: dict[str, str] = {
    "handle_alias_route": "runtime.handle_alias_route",
    "resolve_cooldown_publication": "runtime.resolve_cooldown_publication",
    "anthropic_family_state": "runtime.anthropic_family_state",
    "codex_family_state": "runtime.codex_family_state",
    "normalize_alias_model": "runtime.normalize_alias_model",
    "default_alias_model": "runtime.default_alias_model",
    "perform_candidate_request": "runtime.perform_candidate_request",
    "select_candidate": "runtime.select_candidate",
    "publish_cooldown_memory": "runtime.publish_cooldown_memory",
    "persist_cooldown_durable": "runtime.persist_cooldown_durable",
    "set_session_affinity": "runtime.set_session_affinity",
    "add_alias_metadata": "runtime.add_alias_metadata",
    "raise_redispatch_required": "runtime.raise_redispatch_required",
    "get_candidates_for_alias": "runtime.get_candidates_for_alias",
    "get_active_cooldown_state": "runtime.get_active_cooldown_state",
}


# ---------------------------------------------------------------------------
# Legacy seam facade
# ---------------------------------------------------------------------------


async def handle_auto_agent_alias_route(
    runtime: AnthropicAutoAgentRouteRuntime,
    *,
    alias_family: str,
    alias_model: str,
    request: "Request",
    prepared_request_body: Payload,
    max_candidate_attempts: int,
    select_candidate_fn: _AutoAgentAliasSelectionFn,
    add_alias_metadata_fn: _AutoAgentAliasMetadataFn,
    perform_candidate_request_fn: Callable[..., Awaitable["Response"]],
    get_active_cooldown_state_fn: Callable[[str], Awaitable[tuple[float, str]]],
    set_session_affinity_fn: Callable[..., Awaitable[object]],
    apply_cooldown_fn: Callable[..., Awaitable[str]],
    raise_redispatch_required_fn: Callable[..., None],
    attempts_metadata_key: str,
    skipped_candidates_metadata_key: str,
    no_candidate_detail: str,
    log_label: str,
) -> "Response":
    """Shared Anthropic/Codex auto-agent alias candidate loop (RR-054 #10).

    Thin facade that adapts the legacy per-call seam callables into the typed
    :class:`AliasRouteServices` bundle and delegates to the injected
    ``handle_alias_route``, which owns the R3-1 widened-lock single-flight
    publication.  The production wrappers build the services directly; this
    facade keeps the legacy seam contract for the RR-054 single-flight tests.
    Process-local publication uses the same synchronous family-memory writer
    as production.  The legacy async applicator is isolated in the
    post-release persistence callback and never enters the typed synchronous
    publisher contract.
    """
    legacy_request: Optional["Request"] = None
    legacy_candidate: dict[str, Any] = {}
    legacy_lane_key: Optional[str] = None
    legacy_selected_cooldown_key = ""
    legacy_cooldown_seconds = 0.0
    legacy_error_class: Optional[str] = None
    legacy_grok_account_quota_exhausted = False
    legacy_kimi_failure_metadata: Optional[dict[str, Any]] = None
    legacy_is_read_pilot_lane = False
    family_state = (
        runtime.anthropic_family_state
        if alias_family == "anthropic_auto_agent"
        else runtime.codex_family_state
    )

    def _legacy_resolve_publication(
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
        nonlocal legacy_request
        nonlocal legacy_candidate
        nonlocal legacy_lane_key
        nonlocal legacy_selected_cooldown_key
        nonlocal legacy_cooldown_seconds
        nonlocal legacy_error_class
        nonlocal legacy_grok_account_quota_exhausted
        nonlocal legacy_kimi_failure_metadata
        nonlocal legacy_is_read_pilot_lane
        legacy_request = request
        legacy_candidate = candidate
        legacy_lane_key = lane_key
        legacy_selected_cooldown_key = selected_cooldown_key
        legacy_cooldown_seconds = cooldown_seconds
        legacy_error_class = error_class
        legacy_grok_account_quota_exhausted = grok_account_quota_exhausted
        legacy_kimi_failure_metadata = kimi_failure_metadata
        legacy_is_read_pilot_lane = is_read_pilot_lane
        return runtime.resolve_cooldown_publication(
            request=request,
            candidate=candidate,
            lane_key=lane_key,
            selected_cooldown_key=selected_cooldown_key,
            cooldown_seconds=cooldown_seconds,
            error_class=error_class,
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
            is_read_pilot_lane=is_read_pilot_lane,
        )

    def _legacy_publish_memory(*, keys: Sequence[str], seconds: float) -> None:
        for key in keys:
            family_state.set_cooldown_memory(key, seconds)

    async def _legacy_persist(*, keys: Sequence[str], seconds: float) -> None:
        if legacy_request is None:
            raise RuntimeError("legacy cooldown resolver did not capture a request")
        await apply_cooldown_fn(
            request=legacy_request,
            candidate=legacy_candidate,
            lane_key=legacy_lane_key,
            selected_cooldown_key=legacy_selected_cooldown_key,
            cooldown_seconds=legacy_cooldown_seconds,
            error_class=legacy_error_class,
            grok_account_quota_exhausted=legacy_grok_account_quota_exhausted,
            kimi_failure_metadata=legacy_kimi_failure_metadata,
            is_read_pilot_lane=legacy_is_read_pilot_lane,
        )

    async def _legacy_get_active_cooldown_state(
        cooldown_key: str,
    ) -> tuple[float, str]:
        memory_seconds = family_state.get_memory_cooldown_remaining(cooldown_key)
        if memory_seconds > 0:
            return memory_seconds, "memory"
        return await get_active_cooldown_state_fn(cooldown_key)

    # The legacy seam callables are type-erased (``Callable[..., ...]``); cast
    # them to the typed protocols at this bridge boundary.  The production
    # wrappers pass conforming functions directly and need no cast.
    services = AliasRouteServices(
        select_candidate_fn=cast(SelectCandidateFn, select_candidate_fn),
        perform_candidate_request_fn=cast(
            PerformCandidateRequestFn, perform_candidate_request_fn
        ),
        resolve_cooldown_publication_fn=_legacy_resolve_publication,
        publish_cooldown_memory_fn=_legacy_publish_memory,
        persist_cooldown_fn=_legacy_persist,
        set_session_affinity_fn=cast(SetSessionAffinityFn, set_session_affinity_fn),
        add_alias_metadata_fn=add_alias_metadata_fn,
        raise_redispatch_fn=raise_redispatch_required_fn,
    )
    return await runtime.handle_alias_route(
        services,
        alias_family=alias_family,
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=max_candidate_attempts,
        get_active_cooldown_state_fn=_legacy_get_active_cooldown_state,
        attempts_metadata_key=attempts_metadata_key,
        skipped_candidates_metadata_key=skipped_candidates_metadata_key,
        no_candidate_detail=no_candidate_detail,
        log_label=log_label,
    )


# ---------------------------------------------------------------------------
# Production Anthropic wrapper
# ---------------------------------------------------------------------------


async def handle_anthropic_auto_agent_alias_route(
    runtime: AnthropicAutoAgentRouteRuntime,
    *,
    endpoint: str,
    request: "Request",
    fastapi_response: "Response",
    user_api_key_dict: "UserAPIKeyAuth",
    prepared_request_body: dict[str, Any],
    target_url: str,
    custom_headers: dict[str, Any],
) -> "Response":
    """Production Anthropic auto-agent alias route (native-Anthropic fail-closed).

    Assembles the typed :class:`AliasRouteServices` from runtime-injected
    production functions and delegates to the candidate loop.  Alias
    selection, cooldown, affinity, attempt metadata, request mutation,
    and redispatch behavior are preserved exactly.
    """
    alias_model = (
        runtime.normalize_alias_model(prepared_request_body.get("model"))
        or runtime.default_alias_model
    )

    async def _perform_candidate_request(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> "Response":
        return await runtime.perform_candidate_request(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            candidate=candidate,
            candidate_body=candidate_body,
            target_url=target_url,
            custom_headers=custom_headers,
        )

    services = AliasRouteServices(
        select_candidate_fn=runtime.select_candidate,
        perform_candidate_request_fn=_perform_candidate_request,
        resolve_cooldown_publication_fn=runtime.resolve_cooldown_publication,
        publish_cooldown_memory_fn=runtime.publish_cooldown_memory,
        persist_cooldown_fn=runtime.persist_cooldown_durable,
        set_session_affinity_fn=runtime.set_session_affinity,
        add_alias_metadata_fn=runtime.add_alias_metadata,
        raise_redispatch_fn=runtime.raise_redispatch_required,
    )
    return await runtime.handle_alias_route(
        services,
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=len(runtime.get_candidates_for_alias(alias_model)),
        get_active_cooldown_state_fn=runtime.get_active_cooldown_state,
        attempts_metadata_key="anthropic_auto_agent_attempts",
        skipped_candidates_metadata_key="anthropic_auto_agent_skipped_candidates",
        no_candidate_detail="No Anthropic auto-agent alias candidates were available.",
        log_label="Anthropic",
    )
