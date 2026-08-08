"""Wave 7 owner: Codex auto-agent alias route handler.

Behavior-preserving extraction of ``_handle_codex_auto_agent_alias_route``
from ``llm_passthrough_endpoints.py`` (god module, lines 8915-8977).

Defines:
- ``CodexAutoAgentRouteRuntime``: frozen typed dependency bundle for all host
  callbacks and state consumed by the handler.
- ``handle_codex_auto_agent_alias_route``: the extracted async handler.
- ``build_runtime_from_host``: lazy factory that imports the god module ONLY
  when called (never at module scope).

Do NOT import ``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    handle_alias_route,
)

if TYPE_CHECKING:
    from fastapi import Request, Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
        SelectionEnumeration,
    )


# ---------------------------------------------------------------------------
# Typed runtime / dependency bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodexAutoAgentRouteRuntime:
    """Immutable bundle of every host callback/state the handler needs.

    Construction is deferred to ``build_runtime_from_host`` so the god module
    is never imported at module scope.
    """

    # Client product label extraction
    extract_client_product_label_fn: Callable[
        ["Request", "dict[str, Any]"], Optional[str]
    ]

    # Candidate request performer (raw, before closure wrapping)
    perform_candidate_request_fn: Callable[..., Awaitable["Response"]]

    # Alias-route service seams (passed through to AliasRouteServices)
    select_candidate_fn: Any
    resolve_cooldown_publication_fn: Callable[..., CooldownPublicationPlan]
    publish_cooldown_memory_fn: Callable[..., None]
    persist_cooldown_fn: Callable[..., Awaitable[None]]
    set_session_affinity_fn: Callable[..., Awaitable[Any]]
    add_alias_metadata_fn: Callable[..., "dict[str, Any]"]
    raise_redispatch_fn: Callable[..., None]

    # Cooldown state query
    get_active_cooldown_state_fn: Callable[[str], Awaitable["tuple[float, str]"]]

    # Selection enumeration resolver
    resolve_selection_enumeration_fn: Callable[..., "SelectionEnumeration"]


# ---------------------------------------------------------------------------
# Extracted handler
# ---------------------------------------------------------------------------


async def handle_codex_auto_agent_alias_route(
    runtime: CodexAutoAgentRouteRuntime,
    *,
    endpoint: str,
    request: "Request",
    fastapi_response: "Response",
    user_api_key_dict: "UserAPIKeyAuth",
    prepared_request_body: "dict[str, Any]",
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
    canonical_alias: str,
) -> "Response":
    """Handle a Codex auto-agent alias route request.

    Exact behavioral equivalent of the god-module
    ``_handle_codex_auto_agent_alias_route`` (lines 8915-8977).
    """
    alias_model = canonical_alias
    client_product_label = runtime.extract_client_product_label_fn(
        request, prepared_request_body
    )

    async def _perform_candidate_request(
        *,
        candidate: "dict[str, Any]",
        candidate_body: "dict[str, Any]",
    ) -> "Response":
        return await runtime.perform_candidate_request_fn(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            candidate=candidate,
            candidate_body=candidate_body,
            target_url=target_url,
            api_key=api_key,
            forward_headers=forward_headers,
        )

    services = AliasRouteServices(
        select_candidate_fn=runtime.select_candidate_fn,
        perform_candidate_request_fn=_perform_candidate_request,
        resolve_cooldown_publication_fn=runtime.resolve_cooldown_publication_fn,
        publish_cooldown_memory_fn=runtime.publish_cooldown_memory_fn,
        persist_cooldown_fn=runtime.persist_cooldown_fn,
        set_session_affinity_fn=runtime.set_session_affinity_fn,
        add_alias_metadata_fn=runtime.add_alias_metadata_fn,
        raise_redispatch_fn=runtime.raise_redispatch_fn,
    )
    return await handle_alias_route(
        services,
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=len(
            runtime.resolve_selection_enumeration_fn(
                request,
                alias_model,
                ingress="codex",
                client_product_label=client_product_label,
            ).candidates
        ),
        get_active_cooldown_state_fn=runtime.get_active_cooldown_state_fn,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="No Codex auto-agent alias candidates were available.",
        log_label="Codex",
    )


# ---------------------------------------------------------------------------
# Lazy runtime factory (god module imported ONLY here, at call time)
# ---------------------------------------------------------------------------


def build_runtime_from_host() -> CodexAutoAgentRouteRuntime:
    """Construct the runtime bundle from the god module's live namespace.

    Imports ``llm_passthrough_endpoints`` lazily so this module never creates
    a module-scope import cycle.
    """
    from litellm.proxy.pass_through_endpoints import (  # noqa: PLC0415
        llm_passthrough_endpoints as _host,
    )

    return CodexAutoAgentRouteRuntime(
        extract_client_product_label_fn=_host._extract_auto_agent_alias_client_product_label,
        perform_candidate_request_fn=_host._perform_codex_auto_agent_alias_candidate_request,
        select_candidate_fn=_host._select_codex_auto_agent_candidate,
        resolve_cooldown_publication_fn=_host._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=_host._publish_codex_cooldown_memory,
        persist_cooldown_fn=_host._persist_codex_cooldown_durable,
        set_session_affinity_fn=_host._set_codex_auto_agent_session_affinity,
        add_alias_metadata_fn=_host._add_codex_auto_agent_alias_metadata,
        raise_redispatch_fn=_host._raise_codex_auto_agent_redispatch_required,
        get_active_cooldown_state_fn=_host._get_codex_auto_agent_active_cooldown_state,
        resolve_selection_enumeration_fn=_host._resolve_aawm_alias_selection_enumeration,
    )
