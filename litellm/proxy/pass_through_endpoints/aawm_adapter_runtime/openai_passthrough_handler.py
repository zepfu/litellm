"""Wave 7 owner: BaseOpenAIPassThroughHandler extraction.

Behavior-preserving extraction of the ``BaseOpenAIPassThroughHandler`` class
from ``llm_passthrough_endpoints.py`` (god module, lines 8979-9297).

Defines:
- ``OpenAIPassThroughHandlerRuntime``: frozen typed dependency bundle for all
  host callbacks consumed by the handler methods.
- ``BaseOpenAIPassThroughHandler``: facade class with identical static-method
  API.  Pure utility methods (``_join_url_paths``,
  ``_normalize_endpoint_for_target``) are self-contained; DI-dependent methods
  delegate through the class-level ``_runtime``.
- ``build_runtime_from_host``: lazy factory that imports the god module ONLY
  when called (never at module scope).

Do NOT import ``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Optional,
    Union,
)

import httpx

import litellm

if TYPE_CHECKING:
    from fastapi import Request, Response

    from litellm.proxy._types import UserAPIKeyAuth


def _identity_codex_tool_adapter(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    return request_body, []


# ---------------------------------------------------------------------------
# Typed runtime / dependency bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAIPassThroughHandlerRuntime:
    """Immutable bundle of every host callback the handler class needs.

    Construction is deferred to ``build_runtime_from_host`` so the god module
    is never imported at module scope.
    """

    # -- xAI / Grok request preparation ------------------------------------
    prepare_oa_xai_passthrough_request_fn: Callable[
        ..., Awaitable[tuple[bool, Optional[str], Optional[str]]]
    ]
    is_openai_responses_endpoint_fn: Callable[[str], bool]
    to_xai_native_passthrough_model_fn: Callable[[Any], str]
    get_openai_passthrough_route_family_fn: Callable[[str], str]
    merge_litellm_metadata_fn: Callable[..., dict[str, Any]]
    prepare_grok_native_oauth_passthrough_request_fn: Callable[
        ..., Awaitable[tuple[bool, Optional[str], dict[str, Any], dict[str, Any]]]
    ]
    join_grok_passthrough_url_fn: Callable[..., str]

    # -- Codex dispatch chain ----------------------------------------------
    request_uses_codex_native_auth_fn: Callable[["Request"], bool]
    resolve_codex_auto_agent_alias_model_fn: Callable[
        ..., Optional[str]
    ]
    resolve_auto_agent_alias_route_host_attribution_fn: Callable[
        ["Request"], Awaitable[dict[str, Optional[str]]]
    ]
    add_route_family_logging_metadata_fn: Callable[
        [dict[str, Any], str], dict[str, Any]
    ]
    apply_codex_tool_description_patches_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ]
    drop_unsupported_codex_hosted_tools_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ]
    drop_unsupported_codex_request_params_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ]
    drop_unsupported_codex_input_items_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ]
    is_oa_xai_request_body_fn: Callable[[dict[str, Any]], bool]
    is_grok_native_oauth_request_body_fn: Callable[[dict[str, Any]], bool]
    drop_tool_choice_without_tools_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ]
    add_codex_request_breakout_logging_metadata_fn: Callable[
        [dict[str, Any]], dict[str, Any]
    ]

    # -- Observability / request body --------------------------------------
    prepare_request_body_for_passthrough_observability_fn: Callable[
        ..., dict[str, Any]
    ]
    safe_set_request_parsed_body_fn: Callable[..., None]
    get_request_body_fn: Callable[..., Awaitable[dict[str, Any]]]

    # -- Pass-through infrastructure ---------------------------------------
    create_pass_through_route_fn: Callable[..., Any]
    try_dispatch_codex_request_fn: Callable[
        ..., Awaitable[Optional["Response"]]
    ]
    validate_codex_auto_agent_responses_payload_fn: Callable[
        ..., Awaitable["Response"]
    ]

    # -- Route checks ------------------------------------------------------
    is_assistants_api_request_fn: Callable[["Request"], bool]

    # -- Direct managed xAI tool adaptation --------------------------------
    adapt_codex_custom_tools_to_functions_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ] = _identity_codex_tool_adapter
    adapt_codex_namespace_tools_to_functions_fn: Callable[
        [dict[str, Any]], tuple[dict[str, Any], Any]
    ] = _identity_codex_tool_adapter


# ---------------------------------------------------------------------------
# Facade class (identical static-method API to the god-module original)
# ---------------------------------------------------------------------------


class BaseOpenAIPassThroughHandler:
    """Drop-in replacement for the god-module ``BaseOpenAIPassThroughHandler``.

    Pure utility methods are self-contained.  DI-dependent methods read
    ``_runtime``; call ``install_runtime`` (or ``build_runtime_from_host``)
    before invoking them.
    """

    _runtime: ClassVar[Optional[OpenAIPassThroughHandlerRuntime]] = None

    @classmethod
    def _get_runtime(cls) -> OpenAIPassThroughHandlerRuntime:
        rt = cls._runtime
        if rt is None:
            raise RuntimeError(
                "BaseOpenAIPassThroughHandler._runtime is not configured. "
                "Call install_runtime() or build_runtime_from_host() first."
            )
        return rt

    # -- Pure utility methods (no DI needed) --------------------------------

    @staticmethod
    def _join_url_paths(
        base_url: httpx.URL,
        path: str,
        custom_llm_provider: Union[litellm.LlmProviders, str],
    ) -> str:
        """Properly joins a base URL with a path, preserving any existing
        path in the base URL."""
        if not base_url.path or base_url.path == "/":
            joined_path_str = str(base_url.copy_with(path=path))
        else:
            base_path = base_url.path.rstrip("/")
            clean_path = path.lstrip("/")
            full_path = f"{base_path}/{clean_path}"
            joined_path_str = str(base_url.copy_with(path=full_path))

        if (
            custom_llm_provider == litellm.LlmProviders.OPENAI
            and "/v1/" not in joined_path_str
        ):
            joined_path_str = joined_path_str.replace(
                "api.openai.com/", "api.openai.com/v1/"
            )

        return joined_path_str

    @staticmethod
    def _normalize_endpoint_for_target(
        endpoint: str, base_target_url: str
    ) -> str:
        normalized_endpoint = httpx.URL(endpoint).path
        if not normalized_endpoint.startswith("/"):
            normalized_endpoint = "/" + normalized_endpoint

        base_url = httpx.URL(base_target_url)
        if (
            base_url.host
            and "chatgpt.com" in base_url.host
            and base_url.path.rstrip("/") == "/backend-api/codex"
            and normalized_endpoint.startswith("/v1/")
        ):
            return normalized_endpoint[len("/v1"):]
        if base_url.path.rstrip("/") == "/v1" and normalized_endpoint.startswith(
            "/v1/"
        ):
            return normalized_endpoint[len("/v1"):]
        return normalized_endpoint

    # -- DI-dependent utility methods ---------------------------------------

    @staticmethod
    def _append_openai_beta_header(
        headers: dict, request: "Request"
    ) -> dict:
        """Appends the OpenAI-Beta header if the request is an OpenAI
        Assistants API request."""
        rt = BaseOpenAIPassThroughHandler._get_runtime()
        if rt.is_assistants_api_request_fn(request) is True and "OpenAI-Beta" not in headers:
            headers["OpenAI-Beta"] = "assistants=v2"
        return headers

    @staticmethod
    def _assemble_headers(
        api_key: Optional[str],
        request: "Request",
        extra_headers: Optional[dict] = None,
    ) -> dict:
        base_headers: dict[str, str] = {}
        if api_key is not None:
            base_headers = {
                "authorization": "Bearer {}".format(api_key),
                "api-key": "{}".format(api_key),
            }
        if extra_headers is not None:
            base_headers.update(extra_headers)
        return BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers=base_headers,
            request=request,
        )

    # -- Async preparation methods ------------------------------------------

    @staticmethod
    async def _prepare_openai_oa_xai_context(
        *,
        endpoint: str,
        request_body: dict[str, Any],
    ) -> Optional[tuple[str, str, dict[str, Any], str]]:
        rt = BaseOpenAIPassThroughHandler._get_runtime()
        (
            prepared_oa_xai,
            oa_xai_api_base,
            oa_xai_api_key,
        ) = await rt.prepare_oa_xai_passthrough_request_fn(
            request_body,
            sanitize_responses_request=rt.is_openai_responses_endpoint_fn(
                endpoint
            ),
        )
        if not prepared_oa_xai:
            return None
        if oa_xai_api_base is None or oa_xai_api_key is None:
            raise Exception(
                "OpenAI passthrough requests for xAI OAuth models require "
                "a managed xAI OAuth credential."
            )

        request_body["model"] = rt.to_xai_native_passthrough_model_fn(
            request_body.get("model")
        )
        openai_route_family = rt.get_openai_passthrough_route_family_fn(
            endpoint
        )
        encoded_endpoint = (
            BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
                endpoint=endpoint,
                base_target_url=oa_xai_api_base,
            )
        )
        updated_url = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url=httpx.URL(oa_xai_api_base),
            path=encoded_endpoint,
            custom_llm_provider=litellm.LlmProviders.XAI,
        )
        prepared_request_body = rt.merge_litellm_metadata_fn(
            request_body,
            tags_to_add=[
                f"openai-passthrough-route:{openai_route_family}",
            ],
            extra_fields={
                "openai_passthrough_route_family": openai_route_family,
            },
        )
        return (
            oa_xai_api_base,
            oa_xai_api_key,
            prepared_request_body,
            updated_url,
        )

    @staticmethod
    async def _prepare_openai_grok_native_oauth_context(
        *,
        endpoint: str,
        request: "Request",
        request_body: dict[str, Any],
        extra_headers: Optional[dict],
    ) -> Optional[tuple[str, dict[str, Any], dict[str, Any], str]]:
        rt = BaseOpenAIPassThroughHandler._get_runtime()
        (
            prepared_grok_native,
            grok_target_base_url,
            grok_headers,
            grok_prepared_body,
        ) = await rt.prepare_grok_native_oauth_passthrough_request_fn(
            request_body,
            request=request,
            tags_to_add=[
                "openai-grok-native-responses-adapter",
            ],
            extra_fields={
                "openai_passthrough_route_family": (
                    rt.get_openai_passthrough_route_family_fn(endpoint)
                ),
                "grok_native_entrypoint": "openai_responses",
            },
        )
        if not prepared_grok_native:
            return None
        if grok_target_base_url is None:
            raise Exception(
                "OpenAI passthrough requests for Grok native OAuth models "
                "require a Grok target base URL."
            )

        merged_headers = {
            **(extra_headers or {}),
            **grok_headers,
        }
        updated_url = rt.join_grok_passthrough_url_fn(
            base_target_url=grok_target_base_url,
            endpoint="/v1/responses",
        )
        return (
            grok_target_base_url,
            merged_headers,
            grok_prepared_body,
            updated_url,
        )

    # -- Main handler -------------------------------------------------------

    @staticmethod
    async def _base_openai_pass_through_handler(  # noqa: PLR0915
        endpoint: str,
        request: "Request",
        fastapi_response: "Response",
        user_api_key_dict: "UserAPIKeyAuth",
        base_target_url: str,
        api_key: Optional[str],
        custom_llm_provider: litellm.LlmProviders,
        extra_headers: Optional[dict] = None,
        forward_headers: bool = False,
    ):
        rt = BaseOpenAIPassThroughHandler._get_runtime()

        encoded_endpoint = (
            BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
                endpoint=endpoint,
                base_target_url=base_target_url,
            )
        )

        base_url = httpx.URL(base_target_url)
        updated_url = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url=base_url,
            path=encoded_endpoint,
            custom_llm_provider=custom_llm_provider,
        )
        egress_credential_family: Optional[str] = None
        expected_target_family: Optional[str] = None
        endpoint_custom_body: Optional[dict[str, Any]] = None
        canonical_managed_oa_xai_request_body: Optional[dict[str, Any]] = None
        bound_codex_oauth_identity: Optional[dict[str, str]] = None
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
                _get_bound_codex_oauth_candidate_identity as _get_bound_identity,
            )

            bound_codex_oauth_identity = _get_bound_identity(request)
        except Exception:  # noqa: BLE001
            bound_codex_oauth_identity = None

        if request.method == "POST":
            request_body = await rt.get_request_body_fn(request)
            prepared_request_body = request_body
            body_was_prepared = False
            is_codex_responses_request = (
                rt.request_uses_codex_native_auth_fn(request)
                and rt.is_openai_responses_endpoint_fn(endpoint)
            ) or bound_codex_oauth_identity is not None
            if (
                rt.resolve_codex_auto_agent_alias_model_fn(
                    prepared_request_body,
                    endpoint=endpoint,
                    request=request,
                )
                is not None
            ):
                is_codex_responses_request = True
            is_managed_oa_xai_request = rt.is_oa_xai_request_body_fn(
                prepared_request_body
            )
            if is_managed_oa_xai_request:
                canonical_managed_oa_xai_request_body = copy.deepcopy(
                    prepared_request_body
                )
            if is_codex_responses_request:
                codex_route_host_attribution = (
                    await rt.resolve_auto_agent_alias_route_host_attribution_fn(request)
                )
                litellm_metadata = prepared_request_body.get("litellm_metadata")
                if not isinstance(litellm_metadata, dict):
                    litellm_metadata = {}
                for key in (
                    "client_ip",
                    "client_ip_source",
                    "host_name",
                    "host_name_source",
                ):
                    value = codex_route_host_attribution.get(key)
                    if value is None:
                        continue
                    if key == "client_ip" and litellm_metadata.get("client_ip"):
                        continue
                    litellm_metadata[key] = value
                prepared_request_body = {
                    **prepared_request_body,
                    "litellm_metadata": litellm_metadata,
                }
                prepared_request_body = (
                    rt.add_route_family_logging_metadata_fn(
                        prepared_request_body,
                        "codex_responses",
                    )
                )
            if is_codex_responses_request or is_managed_oa_xai_request:
                if is_managed_oa_xai_request:
                    (
                        prepared_request_body,
                        _codex_adapted_custom_tools,
                    ) = rt.adapt_codex_custom_tools_to_functions_fn(
                        prepared_request_body
                    )
                    (
                        prepared_request_body,
                        _codex_adapted_namespace_tools,
                    ) = rt.adapt_codex_namespace_tools_to_functions_fn(
                        prepared_request_body
                    )
                (
                    prepared_request_body,
                    _codex_tool_description_patch_events,
                ) = rt.apply_codex_tool_description_patches_fn(
                    prepared_request_body
                )
                (
                    prepared_request_body,
                    _codex_unsupported_hosted_tools,
                ) = rt.drop_unsupported_codex_hosted_tools_fn(
                    prepared_request_body
                )
                (
                    prepared_request_body,
                    _codex_unsupported_request_params,
                ) = rt.drop_unsupported_codex_request_params_fn(
                    prepared_request_body
                )
                (
                    prepared_request_body,
                    _codex_unsupported_input_items,
                ) = rt.drop_unsupported_codex_input_items_fn(
                    prepared_request_body
                )
                if is_managed_oa_xai_request or rt.is_grok_native_oauth_request_body_fn(
                    prepared_request_body
                ):
                    (
                        prepared_request_body,
                        _codex_removed_empty_tool_choice,
                    ) = rt.drop_tool_choice_without_tools_fn(
                        prepared_request_body
                    )
                if is_codex_responses_request:
                    prepared_request_body = (
                        rt.add_codex_request_breakout_logging_metadata_fn(
                            prepared_request_body
                        )
                    )
            oa_xai_context = (
                await BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                    endpoint=endpoint,
                    request_body=prepared_request_body,
                )
            )
            if oa_xai_context is not None:
                body_was_prepared = True
                (
                    base_target_url,
                    api_key,
                    prepared_request_body,
                    updated_url,
                ) = oa_xai_context
                custom_llm_provider = litellm.LlmProviders.XAI
                forward_headers = False
                egress_credential_family = "xai"
                expected_target_family = "xai"
            elif rt.is_openai_responses_endpoint_fn(endpoint):
                grok_native_context = await BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                    endpoint=endpoint,
                    request=request,
                    request_body=prepared_request_body,
                    extra_headers=extra_headers,
                )
                if grok_native_context is not None:
                    body_was_prepared = True
                    (
                        base_target_url,
                        extra_headers,
                        prepared_request_body,
                        updated_url,
                    ) = grok_native_context
                    api_key = None
                    custom_llm_provider = litellm.LlmProviders.XAI
                    forward_headers = False
                    egress_credential_family = "xai"
                    expected_target_family = "xai"
                elif is_codex_responses_request:
                    dispatched_response = await rt.try_dispatch_codex_request_fn(
                        endpoint=endpoint,
                        request=request,
                        request_body=request_body,
                        prepared_request_body=prepared_request_body,
                        fastapi_response=fastapi_response,
                        user_api_key_dict=user_api_key_dict,
                        target_url=str(updated_url),
                        api_key=api_key,
                        forward_headers=forward_headers,
                    )
                    if dispatched_response is not None:
                        return dispatched_response
            else:
                prepared_request_body = (
                    rt.add_route_family_logging_metadata_fn(
                        prepared_request_body,
                        rt.get_openai_passthrough_route_family_fn(endpoint),
                    )
                )
            prepared_request_body = (
                rt.prepare_request_body_for_passthrough_observability_fn(
                    request=request,
                    request_body=prepared_request_body,
                )
            )
            if body_was_prepared or prepared_request_body is not request_body:
                rt.safe_set_request_parsed_body_fn(
                    request, prepared_request_body
                )
                endpoint_custom_body = prepared_request_body

        ## check for streaming
        is_streaming_request = "stream" in str(updated_url)

        ## CREATE PASS-THROUGH
        endpoint_func = rt.create_pass_through_route_fn(
            endpoint=endpoint,
            target=str(updated_url),
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=api_key, request=request, extra_headers=extra_headers
            ),
            _forward_headers=forward_headers,
            is_streaming_request=is_streaming_request,  # type: ignore
            custom_llm_provider=custom_llm_provider.value
            if isinstance(custom_llm_provider, litellm.LlmProviders)
            else custom_llm_provider,
            egress_credential_family=egress_credential_family,
            expected_target_family=expected_target_family,
        )

        # D1-612: request-scoped session-owner guard for direct OpenAI fallthrough.
        # Does not mutate the egress body (preserves caller body identity). Nested
        # alias routes return earlier; pass_through_request also renews pre-send.
        import sys

        _sa = sys.modules.get(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
        )
        if _sa is None:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
                session_affinity as _sa,
            )

        session_owner_lease = None
        if not _sa.should_skip_session_owner_for_openai_models_discovery(
            request,
            endpoint=endpoint,
            url=updated_url,
        ) and not _sa.request_session_owner_already_guarded(request):
            direct_body = endpoint_custom_body if isinstance(endpoint_custom_body, dict) else {}
            canonical_session_identity = _sa.resolve_canonical_session_identity(
                request,
                direct_body,
            )
            if canonical_session_identity is not None:
                # Concrete resolved provider/model/route — not generic defaults.
                provider_name = (
                    custom_llm_provider.value
                    if isinstance(custom_llm_provider, litellm.LlmProviders)
                    else str(custom_llm_provider or "openai")
                )
                requested_model = (
                    direct_body.get("model")
                    if isinstance(direct_body, dict)
                    else None
                )
                is_responses_endpoint = bool(
                    rt.is_openai_responses_endpoint_fn(endpoint)
                )
                is_codex_native = bool(
                    rt.request_uses_codex_native_auth_fn(request)
                )
                # Prefer explicit resolved credential/target families when the
                # prepare path set them (xAI/Grok/etc). For direct Codex/OpenAI
                # fallthrough, pin an account-scoped codex_oauth family so the
                # selected ChatGPT account lane is part of owner identity.
                if egress_credential_family:
                    route_family = str(egress_credential_family)
                elif expected_target_family:
                    route_family = str(expected_target_family)
                elif (
                    bound_codex_oauth_identity is not None
                    or (is_codex_native and is_responses_endpoint)
                ):
                    route_family = "codex_oauth"
                elif is_responses_endpoint:
                    route_family = "openai_responses"
                else:
                    route_family = str(provider_name)
                endpoint_contract = (
                    "openai_responses"
                    if is_responses_endpoint
                    else "openai_passthrough"
                )
                state_format = (
                    "openai_responses" if is_responses_endpoint else "openai"
                )
                account_identity = _sa.extract_account_identity_from_context(
                    request=request,
                    request_body=direct_body,
                )
                # Prefer the server-selected inventory identity over any inbound
                # chatgpt-account-id digest so D1-612 pins the actual account lane.
                if isinstance(bound_codex_oauth_identity, dict):
                    account_identity = {
                        **account_identity,
                        **{
                            key: value
                            for key, value in {
                                "account_label": bound_codex_oauth_identity.get(
                                    "account_label"
                                ),
                                "account_hash": bound_codex_oauth_identity.get(
                                    "account_hash"
                                ),
                                "account_lane": bound_codex_oauth_identity.get(
                                    "lane_key"
                                ),
                                "credential_affinity": bound_codex_oauth_identity.get(
                                    "credential_affinity"
                                ),
                            }.items()
                            if value
                        },
                    }
                requested_attributes = _sa.build_session_owner_attributes(
                    provider=provider_name,
                    model=requested_model,
                    route_family=route_family,
                    endpoint_contract=endpoint_contract,
                    state_format=state_format,
                    ingress="openai_passthrough",
                    requested_model=requested_model,
                    extra=account_identity,
                )
                # Account-scoped direct Codex/OpenAI routes must fail closed
                # before send when safe selected account identity is missing.
                # Non-account-scoped direct routes remain valid without it.
                if _sa.route_requires_account_identity(requested_attributes):
                    incomplete = _sa.incomplete_owner_attribute_reason(
                        requested_attributes, for_promotion=True
                    )
                    if incomplete is not None:
                        _sa.raise_session_owner_redispatch_required(
                            session_identity=canonical_session_identity,
                            failure_phase=(
                                "session_owner_direct_openai_missing_account_identity"
                            ),
                            candidate=requested_attributes,
                            guard=_sa.SessionOwnerGuardResult(
                                decision=(
                                    _sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
                                ),
                                session_identity=canonical_session_identity,
                                mismatch_reason=incomplete,
                                provenance=_sa.build_session_owner_provenance(
                                    session_identity=canonical_session_identity,
                                    decision="redispatch_required",
                                    mismatch_reason=incomplete,
                                ),
                            ),
                            request=request,
                        )
                guard = await _sa.ensure_session_owner_guard_for_request(
                    request=request,
                    request_body=direct_body,
                    session_identity=canonical_session_identity,
                    requested_attributes=requested_attributes,
                    alias_model=str(requested_model)
                    if requested_model is not None
                    else None,
                    require_exact_attributes=True,
                    failure_phase="session_owner_direct_openai_pre_egress",
                    raise_on_redispatch=False,
                )
                if (
                    guard.decision
                    is _sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
                ):
                    can_retry_with_effective_identity = (
                        _sa.is_exact_owned_session_owner_route_mismatch(
                            guard=guard,
                            requested_attributes=requested_attributes,
                        )
                        and _sa.is_replay_safe_session_owner_redispatch_body(
                            direct_body
                        )
                        and not _sa.request_has_effective_session_identity(request)
                    )
                    if can_retry_with_effective_identity and _sa.clear_non_held_request_session_owner_lease(
                        request
                    ):
                        effective_identity = _sa.activate_session_owner_redispatch_effective_identity(
                            request=request,
                            base_session_identity=canonical_session_identity,
                        )
                        if effective_identity is not None:
                            guard = await _sa.ensure_session_owner_guard_for_request(
                                request=request,
                                request_body=direct_body,
                                session_identity=effective_identity,
                                requested_attributes=requested_attributes,
                                alias_model=str(requested_model)
                                if requested_model is not None
                                else None,
                                require_exact_attributes=True,
                                failure_phase="session_owner_direct_openai_pre_egress",
                                raise_on_redispatch=False,
                            )
                    if (
                        guard.decision
                        is _sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
                    ):
                        _sa.raise_session_owner_redispatch_required(
                            session_identity=guard.session_identity
                            or canonical_session_identity,
                            guard=guard,
                            alias_model=str(requested_model)
                            if requested_model is not None
                            else None,
                            candidate=requested_attributes,
                            failure_phase="session_owner_direct_openai_pre_egress",
                            request=request,
                        )
        session_owner_lease = _sa.get_request_session_owner_lease(request)

        try:
            response = await endpoint_func(
                request,
                fastapi_response,
                user_api_key_dict,
                custom_body=endpoint_custom_body,
            )
            status_code = getattr(response, "status_code", None)
            if (
                isinstance(status_code, int)
                and 200 <= status_code < 300
                and canonical_managed_oa_xai_request_body is not None
                and rt.is_openai_responses_endpoint_fn(endpoint)
            ):
                provider_bound_body = (
                    endpoint_custom_body
                    if isinstance(endpoint_custom_body, dict)
                    else {}
                )
                response = await rt.validate_codex_auto_agent_responses_payload_fn(
                    response,
                    adapter_model=str(
                        provider_bound_body.get("model")
                        or canonical_managed_oa_xai_request_body.get("model")
                        or "unknown-model"
                    ),
                    adapter="direct_managed_xai_responses",
                    adapter_label="xAI OAuth",
                    request_body=canonical_managed_oa_xai_request_body,
                )
        except Exception:
            await _sa.finalize_session_owner_lease_on_failure(session_owner_lease)
            raise

        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and 200 <= status_code < 300:
            promote_result = await _sa.finalize_session_owner_lease_on_success(
                session_owner_lease,
                attributes=(
                    session_owner_lease.attributes
                    if session_owner_lease is not None
                    else None
                ),
            )
            if promote_result is not None and promote_result.outcome in {
                _sa.SessionOwnerMutationOutcome.CONFLICT,
                _sa.SessionOwnerMutationOutcome.ERROR,
                _sa.SessionOwnerMutationOutcome.NOT_HELD,
            }:
                _sa.raise_session_owner_redispatch_required(
                    session_identity=(
                        session_owner_lease.session_identity
                        if session_owner_lease is not None
                        else None
                    ),
                    mutation=promote_result,
                    failure_phase="session_owner_direct_openai_promote",
                    request=request,
                )
        else:
            await _sa.finalize_session_owner_lease_on_failure(session_owner_lease)
        return response


# ---------------------------------------------------------------------------
# Runtime installation
# ---------------------------------------------------------------------------


def install_runtime(runtime: OpenAIPassThroughHandlerRuntime) -> None:
    """Attach *runtime* to the facade class so DI-dependent methods work."""
    BaseOpenAIPassThroughHandler._runtime = runtime


# ---------------------------------------------------------------------------
# Lazy runtime factory (god module imported ONLY here, at call time)
# ---------------------------------------------------------------------------


def _late_bound_callback(owner: Any, owner_name: str, attribute_name: str) -> Callable[..., Any]:
    """Resolve a required host callback on every invocation.

    The runtime remains stable after installation while host monkeypatches and
    serial integration publications remain live.
    """

    def _call(*args: Any, **kwargs: Any) -> Any:
        callback = getattr(owner, attribute_name, None)
        if not callable(callback):
            raise RuntimeError(
                f"Required {owner_name} callback {attribute_name!r} is not published."
            )
        return callback(*args, **kwargs)

    return _call


def build_runtime_from_host() -> OpenAIPassThroughHandlerRuntime:
    """Construct the runtime bundle from the god module's live namespace.

    Imports ``llm_passthrough_endpoints`` lazily so this module never creates
    a module-scope import cycle. Every callback remains late-bound so host
    monkeypatches made after runtime installation are observed.
    """
    from litellm.proxy.pass_through_endpoints import (  # noqa: PLC0415
        llm_passthrough_endpoints as _host,
    )
    from litellm.proxy.auth.route_checks import RouteChecks  # noqa: PLC0415

    return OpenAIPassThroughHandlerRuntime(
        prepare_oa_xai_passthrough_request_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_prepare_oa_xai_passthrough_request",
        ),
        is_openai_responses_endpoint_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "_is_openai_responses_endpoint"
        ),
        to_xai_native_passthrough_model_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_to_xai_native_passthrough_model",
        ),
        get_openai_passthrough_route_family_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_get_openai_passthrough_route_family",
        ),
        merge_litellm_metadata_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "_merge_litellm_metadata"
        ),
        prepare_grok_native_oauth_passthrough_request_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_prepare_grok_native_oauth_passthrough_request",
        ),
        join_grok_passthrough_url_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "_join_grok_passthrough_url"
        ),
        request_uses_codex_native_auth_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_request_uses_codex_native_auth",
        ),
        resolve_codex_auto_agent_alias_model_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_resolve_codex_auto_agent_alias_model",
        ),
        add_route_family_logging_metadata_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_add_route_family_logging_metadata",
        ),
        apply_codex_tool_description_patches_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_apply_codex_tool_description_patches_to_request_body",
        ),
        drop_unsupported_codex_hosted_tools_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_drop_unsupported_codex_hosted_tools_from_request_body",
        ),
        drop_unsupported_codex_request_params_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_drop_unsupported_codex_request_params_from_request_body",
        ),
        drop_unsupported_codex_input_items_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_drop_unsupported_codex_input_items_from_request_body",
        ),
        is_oa_xai_request_body_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "_is_oa_xai_request_body"
        ),
        is_grok_native_oauth_request_body_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_is_grok_native_oauth_request_body",
        ),
        drop_tool_choice_without_tools_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_drop_tool_choice_without_tools_from_request_body",
        ),
        add_codex_request_breakout_logging_metadata_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_add_codex_request_breakout_logging_metadata",
        ),
        prepare_request_body_for_passthrough_observability_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_prepare_request_body_for_passthrough_observability",
        ),
        resolve_auto_agent_alias_route_host_attribution_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_aresolve_auto_agent_alias_route_host_attribution",
        ),
        safe_set_request_parsed_body_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "_safe_set_request_parsed_body"
        ),
        get_request_body_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "get_request_body"
        ),
        create_pass_through_route_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "create_pass_through_route"
        ),
        try_dispatch_codex_request_fn=_late_bound_callback(
            _host, "llm_passthrough_endpoints", "try_dispatch_codex_request"
        ),
        validate_codex_auto_agent_responses_payload_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_validate_codex_auto_agent_responses_payload",
        ),
        is_assistants_api_request_fn=_late_bound_callback(
            RouteChecks, "RouteChecks", "_is_assistants_api_request"
        ),
        adapt_codex_custom_tools_to_functions_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_adapt_codex_custom_tools_to_functions_from_request_body",
        ),
        adapt_codex_namespace_tools_to_functions_fn=_late_bound_callback(
            _host,
            "llm_passthrough_endpoints",
            "_adapt_codex_namespace_tools_to_functions_from_request_body",
        ),
    )
