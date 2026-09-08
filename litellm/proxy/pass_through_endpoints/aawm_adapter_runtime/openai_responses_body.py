"""OPENAI-044 canonical OpenAI Responses wire-body compiler.

One pure, request-local compiler produces the exact body serialized to the
provider for both direct and alias OpenAI Responses egress. It composes the
existing shaping primitives exactly once per compiled body and returns the
final wire body together with diagnostics/restoration state, so internal
server context (route identity, watermark state, encrypted-reasoning
disposition, function-name restoration maps) is separated from the object
handed to HTTPX and cannot re-enter the wire body.

Design contract (from the OPENAI-044 Oracle egress evidence):

* Sanitation is scoped to protocol-owned locations (the Responses envelope,
  top-level ``input`` items, and known internal metadata wrappers) instead of
  recursively deleting internal-looking names from user/tool data.
* Each retained transformation (legacy function-history id normalization,
  function-name sanitization, route-identity removal, watermark egress,
  encrypted-reasoning preparation, resolved-model parameter shaping) is
  applied exactly once to the compiled body.
* The returned immutable wire body is the single object the caller must
  serialize; callers copy it back into the caller-owned send dict when the
  transport requires an in-place object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NoReturn, Optional

from litellm.types.utils import all_litellm_params

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.direct_openai_function_call_history import (
    normalize_direct_openai_legacy_function_call_history_ids,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_collaboration_dispatch import (
    _NormalizedCodexAgentMessage,
    bind_codex_collaboration_tool_identities,
    build_codex_collaboration_wire_aliases,
    collect_codex_collaboration_advertised_tool_names,
    get_bound_codex_collaboration_tool_identities,
    normalize_codex_collaboration_dispatch_body,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
    PROVENANCE_ITEM_FIELD,
    ROUTE_IDENTITY_FIELD,
    guard_openai_encrypted_reasoning_egress,
)
from litellm.responses.function_name_sanitization import (
    ResponsesFunctionNameRewrite,
    sanitize_responses_function_names,
)

# Internal provenance sidecar carried on items for local observability. It is
# server-owned and must never reach the provider wire body.
_PROVENANCE_ITEM_FIELD = PROVENANCE_ITEM_FIELD

#: Known internal server-owned top-level keys removed from the wire body.
#: These are LiteLLM/AAWM server context, never caller Responses fields. This
#: is deliberately an explicit removal list, NOT a stale OpenAI allowlist:
#: every supported caller Responses field is preserved.
_INTERNAL_ENVELOPE_KEYS: tuple[str, ...] = (
    "litellm_logging_obj",
    ROUTE_IDENTITY_FIELD,
    "aawm_session_id",
    "canonical_session_identity",
    "codex_session_id",
    "session_id",
    "codex_oauth_account_label",
    "codex_oauth_account_hash",
    "codex_oauth_lane_key",
    "codex_auto_agent_selected_account_label",
    "codex_auto_agent_selected_account_hash",
    "codex_auto_agent_selected_account_lane",
    "codex_auto_agent_selected_account_display",
)

_SERVER_CONTEXT_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *_INTERNAL_ENVELOPE_KEYS,
            *(key for key in all_litellm_params if key != "metadata"),
        )
    )
)
_SERVER_GUARDRAIL_METADATA_KEYS: tuple[str, ...] = (
    "guardrails",
    "applied_guardrails",
    "standard_logging_guardrail_information",
)


def _raise_wire_body_immutable(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject mutation of the exact provider-bound JSON payload."""
    _ = (args, kwargs)
    raise TypeError("OpenAI Responses wire body is immutable")


class _FrozenWireDict(dict[str, Any]):
    """JSON-serializable dict that rejects all mutation operations."""

    __setitem__ = _raise_wire_body_immutable
    __delitem__ = _raise_wire_body_immutable
    clear = _raise_wire_body_immutable
    pop = _raise_wire_body_immutable
    popitem = _raise_wire_body_immutable
    setdefault = _raise_wire_body_immutable
    update = _raise_wire_body_immutable
    __ior__ = _raise_wire_body_immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        _ = memo
        return {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in self.items()}


class _FrozenWireList(list[Any]):
    """JSON-serializable list that rejects all mutation operations."""

    __setitem__ = _raise_wire_body_immutable
    __delitem__ = _raise_wire_body_immutable
    append = _raise_wire_body_immutable
    clear = _raise_wire_body_immutable
    extend = _raise_wire_body_immutable
    insert = _raise_wire_body_immutable
    pop = _raise_wire_body_immutable
    remove = _raise_wire_body_immutable
    reverse = _raise_wire_body_immutable
    sort = _raise_wire_body_immutable
    __iadd__ = _raise_wire_body_immutable
    __imul__ = _raise_wire_body_immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        _ = memo
        return [copy.deepcopy(value, memo) for value in self]


def _freeze_wire_value(value: Any) -> Any:
    """Recursively freeze JSON containers while keeping them serializable."""
    if isinstance(value, Mapping):
        return _FrozenWireDict(
            {
                copy.deepcopy(key): _freeze_wire_value(nested)
                for key, nested in value.items()
            }
        )
    if isinstance(value, list):
        return _FrozenWireList(_freeze_wire_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_wire_value(item) for item in value)
    return value


def _strip_item_internal_fields(item: Any) -> Any:
    """Remove server-owned fields from a single Responses input item.

    Only the protocol-owned item envelope is touched; nested user/tool data
    (prompts, tool schemas, function arguments, output payloads) is preserved
    byte-for-byte even when it contains similarly named keys.
    """
    if not isinstance(item, dict):
        return item
    updated: Optional[dict[str, Any]] = None
    for key in (ROUTE_IDENTITY_FIELD, _PROVENANCE_ITEM_FIELD):
        if key in item:
            if updated is None:
                updated = (
                    _NormalizedCodexAgentMessage(item)
                    if isinstance(item, _NormalizedCodexAgentMessage)
                    else dict(item)
                )
            updated.pop(key, None)
    return updated if updated is not None else item


def sanitize_wire_envelope(
    body: Any,
    *,
    preserve_top_level_keys: tuple[str, ...] = (),
    client_metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, bool]:
    """Strip known server state at protocol-owned surfaces only.

    Removes server-owned top-level context and per-item route-identity/
    provenance sidecars from top-level ``input``/``output`` items. Does not
    recurse into user or tool structures. ``preserve_top_level_keys`` keeps
    adapter-required context such as ``litellm_metadata`` available before a
    non-OpenAI translator runs. When supplied, ``client_metadata`` restores
    genuine client guardrail fields while removing server-added values.
    Returns ``(body, changed)``.
    """
    if not isinstance(body, dict):
        return body, False

    changed = False
    updated: Optional[dict[str, Any]] = None
    context_keys = tuple(
        key for key in _SERVER_CONTEXT_KEYS if key not in preserve_top_level_keys
    )
    for key in context_keys:
        if key in body:
            if updated is None:
                updated = dict(body)
            updated.pop(key, None)
            changed = True

    source = updated if updated is not None else body
    metadata = source.get("metadata")
    if isinstance(metadata, dict) and client_metadata is not None:
        sanitized_metadata: Optional[dict[str, Any]] = None
        for key in _SERVER_GUARDRAIL_METADATA_KEYS:
            if isinstance(client_metadata, Mapping) and key in client_metadata:
                client_value = copy.deepcopy(client_metadata[key])
                if metadata.get(key) != client_value:
                    if sanitized_metadata is None:
                        sanitized_metadata = dict(metadata)
                    sanitized_metadata[key] = client_value
            elif key in metadata:
                if sanitized_metadata is None:
                    sanitized_metadata = dict(metadata)
                sanitized_metadata.pop(key, None)
        if sanitized_metadata is not None:
            if updated is None:
                updated = dict(body)
            updated["metadata"] = sanitized_metadata
            source = updated
            changed = True

    for item_key in ("input", "output"):
        items = source.get(item_key)
        if not isinstance(items, list):
            continue
        new_items: Optional[list[Any]] = None
        for index, item in enumerate(items):
            clean_item = _strip_item_internal_fields(item)
            if clean_item is not item:
                if new_items is None:
                    new_items = list(items)
                new_items[index] = clean_item
        if new_items is not None:
            if updated is None:
                updated = dict(body)
            updated[item_key] = new_items
            source = updated
            changed = True

    return (updated if updated is not None else body), changed


@dataclass(frozen=True)
class OpenAIResponsesWireDiagnostics:
    """Immutable logging/restoration state kept beside the wire payload."""

    observability_body: Optional[dict[str, Any]]
    encrypted_reasoning_disposition: Mapping[str, Any]
    function_name_rewrite: Optional[ResponsesFunctionNameRewrite]
    dropped_codex_request_params: tuple[str, ...]
    watermark_audit: Any


@dataclass(frozen=True)
class OpenAIResponsesWireBody:
    """Immutable exact provider payload plus separated request diagnostics.

    ``body`` is the exact object to serialize to the provider. Diagnostics
    and restoration state are kept in a separate immutable object so they
    cannot re-enter the wire body.
    """

    body: dict[str, Any]
    context: "OpenAIResponsesWireContext"
    diagnostics: OpenAIResponsesWireDiagnostics

    @property
    def observability_body(self) -> Optional[dict[str, Any]]:
        return self.diagnostics.observability_body

    @property
    def encrypted_reasoning_disposition(self) -> Mapping[str, Any]:
        return self.diagnostics.encrypted_reasoning_disposition

    @property
    def function_name_rewrite(self) -> Optional[ResponsesFunctionNameRewrite]:
        return self.diagnostics.function_name_rewrite

    @property
    def dropped_codex_request_params(self) -> tuple[str, ...]:
        return self.diagnostics.dropped_codex_request_params

    @property
    def watermark_audit(self) -> Any:
        return self.diagnostics.watermark_audit

    @property
    def function_name_rewrite_changed(self) -> bool:
        return bool(
            self.function_name_rewrite is not None
            and self.function_name_rewrite.changed
        )


@dataclass(frozen=True)
class OpenAIResponsesWireContext:
    """Immutable server context kept out of the serialized provider body."""

    endpoint: str
    resolved_model: Optional[str]
    url: Optional[str]
    egress_credential_family: Optional[str]
    custom_llm_provider: Optional[str]
    expected_target_family: Optional[str]
    session_identity: Optional[str]


_WIRE_RESULT_STATE_FIELD = "_aawm_openai_responses_wire_result"


def bind_openai_responses_wire_body(
    request: Any,
    wire_body: OpenAIResponsesWireBody,
) -> None:
    """Bind a compiled body to the request without mutating the body."""

    state = getattr(request, "state", None)
    if state is None:
        return
    try:
        setattr(state, _WIRE_RESULT_STATE_FIELD, wire_body)
    except Exception:
        return


def get_bound_openai_responses_wire_body(
    request: Any,
    body: Any = None,
) -> Optional[OpenAIResponsesWireBody]:
    """Return the body compilation bound to *request* when identity matches."""

    state = getattr(request, "state", None)
    result = getattr(state, _WIRE_RESULT_STATE_FIELD, None)
    if not isinstance(result, OpenAIResponsesWireBody):
        return None
    if body is not None and result.body is not body:
        return None
    return result


def _apply_watermark_egress(
    *,
    body: dict[str, Any],
    request: Any,
    endpoint: str,
    metadata: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], Any]:
    """Run watermark egress once for the final provider-bound body."""
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_request_watermark_egress,
    )

    intake = None
    try:
        intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        intake = None
    watermark_metadata = metadata if isinstance(metadata, dict) else None
    payload = None
    try:
        from litellm.proxy.proxy_server import general_settings

        if isinstance(general_settings, dict):
            payload = general_settings.get("openai_passthrough_text_watermark")
        else:
            payload = getattr(
                general_settings,
                "openai_passthrough_text_watermark",
                None,
            )
    except Exception:
        payload = None
    combined = str(endpoint or "").lower()
    watermark_endpoint = (
        "chat_completions"
        if "chat/completions" in combined or "chat_completions" in combined
        else "responses"
    )
    result = apply_request_watermark_egress(
        body=body,
        intake=intake,
        config=load_text_watermark_config(payload),
        endpoint=watermark_endpoint,
        direction="request",
        metadata=watermark_metadata,
        litellm_metadata=watermark_metadata,
    )
    new_body = getattr(result, "body", None)
    audit = getattr(result, "audit", None)
    if isinstance(new_body, dict):
        return new_body, audit
    return body, audit


def compile_openai_responses_wire_body(
    source_body: Mapping[str, Any],
    *,
    request: Any = None,
    resolved_model: Optional[str] = None,
    client_stream: Optional[bool] = None,
    store: Optional[bool] = None,
    url: Any = None,
    egress_credential_family: Any = None,
    custom_llm_provider: Any = None,
    expected_target_family: Any = None,
    endpoint: str = "responses",
    session_identity: Any = None,
    client_metadata: Optional[Mapping[str, Any]] = None,
    strip_function_output_ciphertext_without_plaintext: Optional[bool] = None,
    drop_codex_request_params_fn: Optional[
        Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
    ] = None,
) -> OpenAIResponsesWireBody:
    """Compile the exact OpenAI Responses wire body from a source request body.

    Applies every retained transformation exactly once, in canonical order:

    1. legacy function-history id normalization;
    2. resolved-model unsupported Codex request-parameter removal;
    3. request-local function-name sanitization (restoration map separated);
    4. watermark egress;
    5. encrypted-reasoning egress preparation and provenance validation;
    6. one scoped internal-state sanitation traversal.

    Diagnostics and restoration state are returned on the result and never
    re-attached to the wire body. The caller MUST serialize ``result.body``
    (copying it back into the caller-owned send dict when an in-place object
    is required).
    """
    source_snapshot: dict[str, Any] = (
        copy.deepcopy(dict(source_body)) if isinstance(source_body, Mapping) else {}
    )
    body: dict[str, Any] = copy.deepcopy(source_snapshot)
    discovered_collaboration_identities = []
    body = normalize_codex_collaboration_dispatch_body(
        body,
        identity_collector=discovered_collaboration_identities,
    )
    if request is not None:
        bind_codex_collaboration_tool_identities(
            request,
            discovered_collaboration_identities,
        )
    collaboration_identities = (
        get_bound_codex_collaboration_tool_identities(request)
        if request is not None
        else tuple(discovered_collaboration_identities)
    )
    collaboration_aliases = build_codex_collaboration_wire_aliases(
        collaboration_identities,
        reserved_names=collect_codex_collaboration_advertised_tool_names(body),
    )
    forced_identity_rewrites = {
        alias.original: alias.upstream_name for alias in collaboration_aliases
    }

    # 1. Legacy function-history id normalization (direct + alias contract).
    body = normalize_direct_openai_legacy_function_call_history_ids(body)

    # 2. Resolved-model unsupported Codex request-parameter removal.
    dropped_params: tuple[str, ...] = ()
    if drop_codex_request_params_fn is not None:
        if (
            isinstance(resolved_model, str)
            and resolved_model
            and body.get("model") != resolved_model
        ):
            body = dict(body)
            body["model"] = resolved_model
        body, dropped = drop_codex_request_params_fn(body)
        if isinstance(dropped, (list, tuple)):
            dropped_params = tuple(str(item) for item in dropped)

    # 3. Function-name sanitization; restoration map stays off the wire body.
    name_rewrite = sanitize_responses_function_names(
        body,
        forced_identity_rewrites=forced_identity_rewrites,
    )
    if name_rewrite.changed and isinstance(name_rewrite.body, dict):
        body = name_rewrite.body

    # 4. Watermark egress.
    body, watermark_audit = _apply_watermark_egress(
        body=body,
        request=request,
        endpoint=endpoint,
        metadata=body.get("litellm_metadata") if isinstance(body, dict) else None,
    )

    # 5. Encrypted-reasoning preparation; validate provenance before sanitation.
    body, disposition = guard_openai_encrypted_reasoning_egress(
        body,
        session_identity=session_identity,
        target_route_family=egress_credential_family or expected_target_family,
        strip_function_output_ciphertext_without_plaintext=(
            strip_function_output_ciphertext_without_plaintext
        ),
        url=url,
        egress_credential_family=egress_credential_family,
        custom_llm_provider=custom_llm_provider,
        model=resolved_model,
    )

    # 6. The sole canonical sanitation traversal. It runs after provenance
    # validation so foreign sidecars cannot be erased before compatibility
    # checks observe them.
    body, _ = sanitize_wire_envelope(body, client_metadata=client_metadata)

    # Route-owned stream/store shaping is applied before the result is bound,
    # so the returned dict remains the exact serialized provider body.
    if client_stream is not None and isinstance(body, dict):
        body = dict(body)
        body["stream"] = bool(client_stream)
    if store is not None and isinstance(body, dict):
        body = dict(body)
        body["store"] = bool(store)

    frozen_rewrite = (
        replace(name_rewrite, body=_freeze_wire_value(name_rewrite.body))
        if name_rewrite.changed
        else None
    )
    return OpenAIResponsesWireBody(
        body=_freeze_wire_value(body),
        context=OpenAIResponsesWireContext(
            endpoint=endpoint,
            resolved_model=resolved_model,
            url=str(url) if url is not None else None,
            egress_credential_family=(
                str(egress_credential_family)
                if egress_credential_family is not None
                else None
            ),
            custom_llm_provider=(
                str(custom_llm_provider)
                if custom_llm_provider is not None
                else None
            ),
            expected_target_family=(
                str(expected_target_family)
                if expected_target_family is not None
                else None
            ),
            session_identity=(
                str(session_identity) if session_identity is not None else None
            ),
        ),
        diagnostics=OpenAIResponsesWireDiagnostics(
            observability_body=_freeze_wire_value(source_snapshot),
            encrypted_reasoning_disposition=_freeze_wire_value(disposition),
            function_name_rewrite=frozen_rewrite,
            dropped_codex_request_params=dropped_params,
            watermark_audit=_freeze_wire_value(watermark_audit),
        ),
    )


def replace_send_body(send_body: dict[str, Any], wire_body: Mapping[str, Any]) -> None:
    """Copy the compiled wire body back into a caller-owned send dict.

    The canonical repair for the direct-fallthrough replacement-loss defect:
    callers that must preserve an in-place dict reference clear and update it
    with the compiled wire body.
    """
    if not isinstance(send_body, dict) or not isinstance(wire_body, Mapping):
        return
    if send_body is wire_body:
        return
    send_body.clear()
    send_body.update(wire_body)
