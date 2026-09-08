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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from litellm.types.utils import all_litellm_params

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.direct_openai_function_call_history import (
    normalize_direct_openai_legacy_function_call_history_ids,
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
                updated = dict(item)
            updated.pop(key, None)
    return updated if updated is not None else item


def sanitize_wire_envelope(body: Any) -> tuple[Any, bool]:
    """Strip known server state at protocol-owned surfaces only.

    Removes internal envelope keys and per-item route-identity/provenance
    sidecars from top-level ``input``/``output`` items. Does not recurse into
    user or tool structures. Returns ``(body, changed)``.
    """
    if not isinstance(body, dict):
        return body, False

    changed = False
    updated: Optional[dict[str, Any]] = None
    for key in _INTERNAL_ENVELOPE_KEYS:
        if key in body:
            if updated is None:
                updated = dict(body)
            updated.pop(key, None)
            changed = True

    source = updated if updated is not None else body
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
class OpenAIResponsesWireBody:
    """Immutable result of the canonical Responses wire-body compile.

    ``body`` is the exact object to serialize to the provider. Diagnostics
    and restoration state are carried separately so they cannot re-enter the
    wire body. ``observability_body`` is a request-local copy for hooks and
    logging; it is never used for provider serialization.
    """

    body: dict[str, Any]
    context: "OpenAIResponsesWireContext"
    observability_body: Optional[dict[str, Any]] = None
    encrypted_reasoning_disposition: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    function_name_rewrite: Optional[ResponsesFunctionNameRewrite] = None
    dropped_codex_request_params: tuple[str, ...] = ()
    watermark_audit: Any = None

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


def _strip_litellm_context_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Remove server context while preserving client OpenAI ``metadata``."""

    # ``metadata`` is both a LiteLLM logging input and a supported OpenAI
    # Responses request field. It must remain on the provider wire; the
    # server-owned ``litellm_metadata`` namespace is removed below.
    context_keys = tuple(
        key for key in all_litellm_params if key != "metadata"
    )
    if not any(key in body for key in context_keys):
        return body
    updated = dict(body)
    for key in context_keys:
        updated.pop(key, None)
    return updated


def _scoped_route_identity_sanitizer(body: dict[str, Any]) -> dict[str, Any]:
    """Adapter for the encrypted-reasoning guard's request-body callback."""
    sanitized, _ = sanitize_wire_envelope(body)
    return sanitized if isinstance(sanitized, dict) else body


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
    strip_function_output_ciphertext_without_plaintext: Optional[bool] = None,
    drop_codex_request_params_fn: Optional[
        Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
    ] = None,
) -> OpenAIResponsesWireBody:
    """Compile the exact OpenAI Responses wire body from a source request body.

    Applies every retained transformation exactly once, in canonical order:

    1. scoped internal-state removal (envelope + input-item surfaces);
    2. legacy function-history id normalization;
    3. resolved-model unsupported Codex request-parameter removal;
    4. request-local function-name sanitization (restoration map separated);
    5. watermark egress;
    6. encrypted-reasoning egress preparation (fail-closed guard).

    Diagnostics and restoration state are returned on the result and never
    re-attached to the wire body. The caller MUST serialize ``result.body``
    (copying it back into the caller-owned send dict when an in-place object
    is required).
    """
    source_snapshot: dict[str, Any] = (
        copy.deepcopy(dict(source_body)) if isinstance(source_body, Mapping) else {}
    )
    body: dict[str, Any] = copy.deepcopy(source_snapshot)

    # 1. Scoped sanitation of known server state at protocol-owned surfaces.
    body, _ = sanitize_wire_envelope(body)

    # 2. Legacy function-history id normalization (direct + alias contract).
    body = normalize_direct_openai_legacy_function_call_history_ids(body)

    # 3. Resolved-model unsupported Codex request-parameter removal.
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

    # 4. Function-name sanitization; restoration map stays off the wire body.
    name_rewrite = sanitize_responses_function_names(body)
    if name_rewrite.changed and isinstance(name_rewrite.body, dict):
        body = name_rewrite.body

    # 5. Watermark egress.
    body, watermark_audit = _apply_watermark_egress(
        body=body,
        request=request,
        endpoint=endpoint,
        metadata=body.get("litellm_metadata") if isinstance(body, dict) else None,
    )

    # 6. Encrypted-reasoning preparation; fail-closed guard before send.
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
        strip_route_identity_fn=_scoped_route_identity_sanitizer,
    )
    # Re-assert the scoped envelope invariant after provider-state preparation
    # so the returned body carries no server state at protocol-owned surfaces.
    body, _ = sanitize_wire_envelope(body)
    body = _strip_litellm_context_fields(body)

    # Route-owned stream/store shaping is applied before the result is bound,
    # so the returned dict remains the exact serialized provider body.
    if client_stream is not None and isinstance(body, dict):
        body = dict(body)
        body["stream"] = bool(client_stream)
    if store is not None and isinstance(body, dict):
        body = dict(body)
        body["store"] = bool(store)

    return OpenAIResponsesWireBody(
        body=body,
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
        observability_body=source_snapshot,
        encrypted_reasoning_disposition=MappingProxyType(dict(disposition)),
        function_name_rewrite=name_rewrite if name_rewrite.changed else None,
        dropped_codex_request_params=dropped_params,
        watermark_audit=(
            MappingProxyType(dict(watermark_audit))
            if isinstance(watermark_audit, dict)
            else watermark_audit
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
