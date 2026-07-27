"""xAI and Grok-native request preparation.

This module is intentionally independent of ``llm_passthrough_endpoints``.
The later integration step configures host-owned helpers through
``XAIRequestPrepRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, cast
from uuid import uuid4 as _uuid4

from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization as _anthropic_grok_normalization,
)
from litellm.llms.xai.oauth import (
    build_grok_native_oauth_metadata as _build_grok_native_oauth_metadata,
)
from litellm.llms.xai.oauth import (
    get_grok_native_oauth_access_token as _get_grok_native_oauth_access_token,
)
from litellm.llms.xai.oauth import (
    is_grok_native_oauth_model as _is_grok_native_oauth_model,
)
from litellm.llms.xai.oauth import is_oa_xai_model as _is_oa_xai_model
from litellm.llms.xai.oauth import (
    normalize_grok_native_oauth_model as _normalize_grok_native_oauth_model,
)
from litellm.llms.xai.oauth import prepare_oa_xai_request as _prepare_oa_xai_request
from litellm.llms.xai.oauth import (
    resolve_oa_xai_upstream_model as _resolve_oa_xai_upstream_model,
)
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.secret_managers.main import get_secret_str as _get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

if TYPE_CHECKING:
    from starlette.requests import Request


Payload = dict[str, Any]
DropRequestFields = Callable[[Payload], tuple[Payload, list[Payload]]]


@dataclass(frozen=True)
class XAIRequestPrepRuntime:
    """Host and OAuth callbacks required by the extracted request preparers."""

    is_oa_xai_model: Callable[[Any], bool]
    is_grok_native_oauth_model: Callable[[Any], bool]
    normalize_grok_native_oauth_model: Callable[[Any], Optional[str]]
    resolve_oa_xai_upstream_model: Callable[[str], str]
    prepare_oa_xai_request: Callable[[Payload], Awaitable[bool]]
    build_grok_native_oauth_metadata: Callable[[str], Payload]
    get_grok_native_oauth_access_token: Callable[[], Awaitable[str]]
    get_secret_str: Callable[[str], Optional[str]]
    uuid4: Callable[[], Any]
    _get_model_metadata_entry: Callable[[Any], Optional[Payload]]
    _get_openai_tool_type: Callable[[Payload], Optional[str]]
    _normalize_low_cardinality_tag_value: Callable[[Any], Optional[str]]
    _dedupe_sorted_str_list: Callable[[list[str]], list[str]]
    _merge_litellm_metadata: Callable[..., Payload]
    _build_langfuse_span_descriptor: Callable[..., Payload]
    _drop_unsupported_codex_hosted_tools_from_request_body: DropRequestFields
    _drop_unsupported_codex_request_params_from_request_body: DropRequestFields
    _drop_unsupported_codex_input_items_from_request_body: DropRequestFields
    _drop_tool_choice_without_tools_from_request_body: DropRequestFields
    _replace_request_body_in_place: Callable[[Payload, Payload], None]
    _safe_get_request_headers: Callable[[Request], dict[str, Any]]
    _get_case_insensitive_header: Callable[[dict[str, Any], str], Optional[str]]
    _get_rewrite_input_item_types_for_model: Callable[[Any], set[str]]
    _get_grok_passthrough_target_base: Callable[[], str]
    _sanitize_xai_responses_request_body_in_place: Callable[
        [Payload], tuple[list[str], list[dict[str, Any]]]
    ]


XAI_REQUEST_PREP_SEAM_DISPOSITION = {
    "is_oa_xai_model": "runtime.is_oa_xai_model",
    "is_grok_native_oauth_model": "runtime.is_grok_native_oauth_model",
    "normalize_grok_native_oauth_model": "runtime.normalize_grok_native_oauth_model",
    "resolve_oa_xai_upstream_model": "runtime.resolve_oa_xai_upstream_model",
    "prepare_oa_xai_request": "runtime.prepare_oa_xai_request",
    "build_grok_native_oauth_metadata": "runtime.build_grok_native_oauth_metadata",
    "get_grok_native_oauth_access_token": "runtime.get_grok_native_oauth_access_token",
    "get_secret_str": "runtime.get_secret_str",
    "uuid4": "runtime.uuid4",
    "_get_model_metadata_entry": "runtime._get_model_metadata_entry",
    "_get_openai_tool_type": "runtime._get_openai_tool_type",
    "_normalize_low_cardinality_tag_value": (
        "runtime._normalize_low_cardinality_tag_value"
    ),
    "_dedupe_sorted_str_list": "runtime._dedupe_sorted_str_list",
    "_merge_litellm_metadata": "runtime._merge_litellm_metadata",
    "_build_langfuse_span_descriptor": (
        "runtime._build_langfuse_span_descriptor"
    ),
    "_drop_unsupported_codex_hosted_tools_from_request_body": (
        "runtime._drop_unsupported_codex_hosted_tools_from_request_body"
    ),
    "_drop_unsupported_codex_request_params_from_request_body": (
        "runtime._drop_unsupported_codex_request_params_from_request_body"
    ),
    "_drop_unsupported_codex_input_items_from_request_body": (
        "runtime._drop_unsupported_codex_input_items_from_request_body"
    ),
    "_drop_tool_choice_without_tools_from_request_body": (
        "runtime._drop_tool_choice_without_tools_from_request_body"
    ),
    "_replace_request_body_in_place": "runtime._replace_request_body_in_place",
    "_safe_get_request_headers": "runtime._safe_get_request_headers",
    "_get_case_insensitive_header": "runtime._get_case_insensitive_header",
    "_get_rewrite_input_item_types_for_model": (
        "runtime._get_rewrite_input_item_types_for_model"
    ),
    "_get_grok_passthrough_target_base": (
        "runtime._get_grok_passthrough_target_base"
    ),
    "_sanitize_xai_responses_request_body_in_place": (
        "runtime._sanitize_xai_responses_request_body_in_place"
    ),
}


_request_prep_runtime: Optional[XAIRequestPrepRuntime] = None


def _host_sanitize_xai_responses_request_body_in_place(
    request_body: Payload,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Late-binding default that resolves through the host module seam.

    Mirrors the lambda-wired drop-helper callbacks: the host module
    global is looked up at call time so ``patch.object`` on the host
    module is honoured without requiring the host to pass an explicit
    callback.
    """
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as _host,
    )

    return _host._sanitize_xai_responses_request_body_in_place(request_body)


def build_default_xai_request_prep_runtime(
    *,
    get_model_metadata_entry: Callable[[Any], Optional[Payload]],
    get_openai_tool_type: Callable[[Payload], Optional[str]],
    normalize_low_cardinality_tag_value: Callable[[Any], Optional[str]],
    dedupe_sorted_str_list: Callable[[list[str]], list[str]],
    merge_litellm_metadata: Callable[..., Payload],
    build_langfuse_span_descriptor: Callable[..., Payload],
    drop_unsupported_codex_hosted_tools_from_request_body: DropRequestFields,
    drop_unsupported_codex_request_params_from_request_body: DropRequestFields,
    drop_unsupported_codex_input_items_from_request_body: DropRequestFields,
    drop_tool_choice_without_tools_from_request_body: DropRequestFields,
    replace_request_body_in_place: Callable[[Payload, Payload], None],
    safe_get_request_headers: Callable[[Request], dict[str, Any]],
    get_case_insensitive_header: Callable[
        [dict[str, Any], str], Optional[str]
    ],
    get_rewrite_input_item_types_for_model: Callable[[Any], set[str]],
    get_grok_passthrough_target_base: Callable[[], str],
    sanitize_xai_responses_request_body_in_place: Optional[
        Callable[[Payload], tuple[list[str], list[dict[str, Any]]]]
    ] = None,
) -> XAIRequestPrepRuntime:
    """Build production defaults while keeping every host callback explicit."""

    if sanitize_xai_responses_request_body_in_place is None:
        sanitize_xai_responses_request_body_in_place = (
            _host_sanitize_xai_responses_request_body_in_place
        )

    return XAIRequestPrepRuntime(
        is_oa_xai_model=_is_oa_xai_model,
        is_grok_native_oauth_model=_is_grok_native_oauth_model,
        normalize_grok_native_oauth_model=_normalize_grok_native_oauth_model,
        resolve_oa_xai_upstream_model=_resolve_oa_xai_upstream_model,
        prepare_oa_xai_request=_prepare_oa_xai_request,
        build_grok_native_oauth_metadata=_build_grok_native_oauth_metadata,
        get_grok_native_oauth_access_token=_get_grok_native_oauth_access_token,
        get_secret_str=_get_secret_str,
        uuid4=_uuid4,
        _get_model_metadata_entry=get_model_metadata_entry,
        _get_openai_tool_type=get_openai_tool_type,
        _normalize_low_cardinality_tag_value=(
            normalize_low_cardinality_tag_value
        ),
        _dedupe_sorted_str_list=dedupe_sorted_str_list,
        _merge_litellm_metadata=merge_litellm_metadata,
        _build_langfuse_span_descriptor=build_langfuse_span_descriptor,
        _drop_unsupported_codex_hosted_tools_from_request_body=(
            drop_unsupported_codex_hosted_tools_from_request_body
        ),
        _drop_unsupported_codex_request_params_from_request_body=(
            drop_unsupported_codex_request_params_from_request_body
        ),
        _drop_unsupported_codex_input_items_from_request_body=(
            drop_unsupported_codex_input_items_from_request_body
        ),
        _drop_tool_choice_without_tools_from_request_body=(
            drop_tool_choice_without_tools_from_request_body
        ),
        _replace_request_body_in_place=replace_request_body_in_place,
        _safe_get_request_headers=safe_get_request_headers,
        _get_case_insensitive_header=get_case_insensitive_header,
        _get_rewrite_input_item_types_for_model=(
            get_rewrite_input_item_types_for_model
        ),
        _get_grok_passthrough_target_base=get_grok_passthrough_target_base,
        _sanitize_xai_responses_request_body_in_place=(
            sanitize_xai_responses_request_body_in_place
        ),
    )


def configure_xai_request_prep_runtime(runtime: XAIRequestPrepRuntime) -> None:
    """Configure callbacks used by the extracted functions."""

    global _request_prep_runtime
    _request_prep_runtime = runtime


def reset_xai_request_prep_runtime_for_tests() -> None:
    """Clear configured callbacks so missing integration fails loudly."""

    global _request_prep_runtime
    _request_prep_runtime = None


def _require_runtime() -> XAIRequestPrepRuntime:
    runtime = _request_prep_runtime
    if runtime is None:
        raise RuntimeError(
            "xAI request preparation runtime is not configured; "
            "call configure_xai_request_prep_runtime() first"
        )
    return runtime


def _is_oa_xai_request_body(request_body: dict[str, Any]) -> bool:
    runtime = _require_runtime()
    return runtime.is_oa_xai_model(request_body.get("model"))


def _is_grok_native_oauth_request_body(request_body: dict[str, Any]) -> bool:
    runtime = _require_runtime()
    return runtime.is_grok_native_oauth_model(request_body.get("model"))


def _is_oa_xai_responses_model(model: Any) -> bool:
    runtime = _require_runtime()
    if not runtime.is_oa_xai_model(model):
        return False

    candidate_models = [model]
    try:
        candidate_models.append(
            runtime.resolve_oa_xai_upstream_model(cast(str, model))
        )
    except Exception:
        pass

    for candidate_model in candidate_models:
        model_info = runtime._get_model_metadata_entry(candidate_model)
        if isinstance(model_info, dict) and model_info.get("mode") == "responses":
            return True
    return False


def _to_xai_native_passthrough_model(model: Any) -> Any:
    if isinstance(model, str) and model.startswith("xai/"):
        return model[len("xai/") :]
    return model


def _xai_responses_sanitized_tool_changes(
    original_tools: Any,
    sanitized_tools: Any,
) -> list[dict[str, Any]]:
    runtime = _require_runtime()
    if not isinstance(original_tools, list) or not isinstance(
        sanitized_tools, list
    ):
        return []

    tool_changes: list[dict[str, Any]] = []
    for index, original_tool in enumerate(original_tools):
        sanitized_tool = (
            sanitized_tools[index] if index < len(sanitized_tools) else None
        )
        if original_tool == sanitized_tool:
            continue

        change: dict[str, Any] = {"index": index}
        if isinstance(original_tool, dict):
            tool_type = runtime._get_openai_tool_type(original_tool)
            if tool_type:
                change["type"] = tool_type
            if isinstance(sanitized_tool, dict):
                removed_fields = [
                    key for key in original_tool.keys() if key not in sanitized_tool
                ]
                if removed_fields:
                    change["removed_fields"] = sorted(removed_fields)
        elif isinstance(original_tool, str):
            change["type"] = original_tool

        tool_changes.append(change)
    return tool_changes


def _sanitize_xai_responses_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    runtime = _require_runtime()
    sanitized_body = XAIResponsesAPIConfig().map_openai_params(
        cast(ResponsesAPIOptionalRequestParams, request_body),
        model=str(request_body.get("model") or ""),
        drop_params=True,
    )
    removed_params = [
        key
        for key in request_body.keys()
        if key not in sanitized_body and key != "litellm_metadata"
    ]
    tool_changes = _xai_responses_sanitized_tool_changes(
        request_body.get("tools"),
        sanitized_body.get("tools"),
    )
    decoded_previous_response_id = False
    previous_response_id = sanitized_body.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        decoded = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            previous_response_id
        )
        if decoded != previous_response_id:
            sanitized_body = dict(sanitized_body)
            sanitized_body["previous_response_id"] = decoded
            decoded_previous_response_id = True

    if not removed_params and not tool_changes and not decoded_previous_response_id:
        return request_body, [], []

    tool_types = runtime._dedupe_sorted_str_list(
        [
            tool_change["type"]
            for tool_change in tool_changes
            if isinstance(tool_change.get("type"), str)
        ]
    )
    normalized_removed_params = runtime._dedupe_sorted_str_list(
        [
            normalized
            for param in removed_params
            if (
                normalized
                := runtime._normalize_low_cardinality_tag_value(param)
            )
        ]
    )
    updated_body = runtime._merge_litellm_metadata(
        sanitized_body,
        tags_to_add=[
            "xai-responses-request-sanitized",
            *(
                ["xai-responses-previous-response-id-decoded"]
                if decoded_previous_response_id
                else []
            ),
            *(
                f"xai-responses-removed-param:{param}"
                for param in normalized_removed_params
            ),
            *(
                f"xai-responses-sanitized-tool:{tool}"
                for tool in tool_types
            ),
        ],
        extra_fields={
            "xai_responses_request_sanitized": True,
            "xai_responses_sanitized_removed_params": (
                normalized_removed_params
            ),
            "xai_responses_sanitized_tool_count": len(tool_changes),
            "xai_responses_sanitized_tool_types": tool_types,
            "xai_responses_sanitized_tools": tool_changes,
            "xai_responses_previous_response_id_decoded": (
                decoded_previous_response_id
            ),
            "langfuse_spans": [
                runtime._build_langfuse_span_descriptor(
                    name="xai.responses_request_sanitized",
                    metadata={
                        "removed_params": normalized_removed_params,
                        "tool_count": len(tool_changes),
                        "tool_types": tool_types,
                        "previous_response_id_decoded": (
                            decoded_previous_response_id
                        ),
                    },
                )
            ],
        },
    )
    return updated_body, removed_params, tool_changes


def _coerce_grok_native_function_call_arguments_value(
    arguments_value: Any,
) -> tuple[dict[str, Any], Optional[str]]:
    return _anthropic_grok_normalization.coerce_function_call_arguments_value(
        arguments_value
    )


def _get_anthropic_grok_normalization_runtime() -> (
    _anthropic_grok_normalization.Runtime
):
    runtime = _require_runtime()
    return _anthropic_grok_normalization.Runtime(
        normalize_tag=runtime._normalize_low_cardinality_tag_value,
        dedupe_sorted=runtime._dedupe_sorted_str_list,
        merge_metadata=runtime._merge_litellm_metadata,
        build_span=runtime._build_langfuse_span_descriptor,
        get_rewrite_input_item_types=(
            runtime._get_rewrite_input_item_types_for_model
        ),
    )


def _sanitize_grok_native_function_call_arguments_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        _anthropic_grok_normalization.sanitize_function_call_arguments_request_body(
            request_body
        )
    )


def _sanitize_grok_native_function_call_arguments_in_place(
    request_body: dict[str, Any],
) -> list[dict[str, Any]]:
    return _anthropic_grok_normalization.sanitize_function_call_arguments_in_place(
        _get_anthropic_grok_normalization_runtime(),
        request_body,
    )


def _sanitize_xai_responses_request_body_in_place(
    request_body: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    updated_body, removed_params, tool_changes = (
        _sanitize_xai_responses_request_body(request_body)
    )
    if updated_body is not request_body:
        request_body.clear()
        request_body.update(updated_body)
    return removed_params, tool_changes


async def _prepare_oa_xai_passthrough_request(
    request_body: dict[str, Any],
    *,
    sanitize_responses_request: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    runtime = _require_runtime()
    if runtime.is_oa_xai_model(request_body.get("model")) and not isinstance(
        request_body.get("litellm_metadata"), dict
    ):
        request_body["litellm_metadata"] = {}
    prepared = await runtime.prepare_oa_xai_request(request_body)
    if not prepared:
        return False, None, None

    if sanitize_responses_request:
        (
            updated_body,
            _xai_unsupported_hosted_tools,
        ) = runtime._drop_unsupported_codex_hosted_tools_from_request_body(
            request_body
        )
        runtime._replace_request_body_in_place(request_body, updated_body)
        (
            updated_body,
            _xai_unsupported_request_params,
        ) = runtime._drop_unsupported_codex_request_params_from_request_body(
            request_body
        )
        runtime._replace_request_body_in_place(request_body, updated_body)
        (
            updated_body,
            _xai_unsupported_input_items,
        ) = runtime._drop_unsupported_codex_input_items_from_request_body(
            request_body
        )
        runtime._replace_request_body_in_place(request_body, updated_body)
        runtime._sanitize_xai_responses_request_body_in_place(request_body)
        (
            updated_body,
            _removed_tool_choice,
        ) = runtime._drop_tool_choice_without_tools_from_request_body(
            request_body
        )
        runtime._replace_request_body_in_place(request_body, updated_body)

    api_base = request_body.pop("api_base", None)
    api_key = request_body.pop("api_key", None)
    request_body.pop("custom_llm_provider", None)
    return (
        True,
        api_base
        if isinstance(api_base, str) and api_base.strip()
        else None,
        api_key if isinstance(api_key, str) and api_key.strip() else None,
    )


def _get_grok_native_oauth_client_version() -> str:
    runtime = _require_runtime()
    return (
        runtime.get_secret_str("LITELLM_XAI_GROK_CLIENT_VERSION")
        or runtime.get_secret_str("GROK_CLIENT_VERSION")
        or "0.1.210"
    )


def _get_grok_native_oauth_session_id(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> Optional[str]:
    runtime = _require_runtime()
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()

    for header_name in (
        "x-grok-session-id",
        "session_id",
        "x-session-id",
        "x-grok-conv-id",
    ):
        header_value = runtime._get_case_insensitive_header(
            runtime._safe_get_request_headers(request),
            header_name,
        )
        if header_value:
            return header_value
    return None


def _get_grok_native_oauth_request_id(request: Request) -> str:
    runtime = _require_runtime()
    for header_name in ("x-grok-req-id", "x-request-id", "request_id"):
        header_value = runtime._get_case_insensitive_header(
            runtime._safe_get_request_headers(request),
            header_name,
        )
        if header_value:
            return header_value
    return str(runtime.uuid4())


def _build_grok_native_oauth_headers(
    *,
    access_token: str,
    model: str,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    runtime = _require_runtime()
    client_version = _get_grok_native_oauth_client_version()
    request_id = _get_grok_native_oauth_request_id(request)
    headers: dict[str, Any] = {
        "accept": "application/json",
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "user-agent": (
            runtime.get_secret_str("LITELLM_XAI_GROK_USER_AGENT")
            or f"grok/{client_version}"
        ),
        "x-grok-client-identifier": (
            runtime.get_secret_str("LITELLM_XAI_GROK_CLIENT_IDENTIFIER")
            or "grok-cli"
        ),
        "x-grok-client-version": client_version,
        "x-grok-model-override": model,
        "x-grok-req-id": request_id,
        "x-request-id": request_id,
        "x-xai-token-auth": (
            runtime.get_secret_str("LITELLM_XAI_GROK_XAI_TOKEN_AUTH")
            or "xai-grok-cli"
        ),
    }
    session_id = _get_grok_native_oauth_session_id(
        request=request,
        request_body=request_body,
    )
    if session_id:
        headers["x-grok-session-id"] = session_id
    return headers


def _add_grok_native_oauth_metadata(
    request_body: dict[str, Any],
    *,
    model: str,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    runtime = _require_runtime()
    metadata = runtime.build_grok_native_oauth_metadata(model)
    metadata_tags = metadata.pop("tags", [])
    existing_litellm_metadata = request_body.get("litellm_metadata")
    preserved_route_family: Optional[str] = None
    if isinstance(existing_litellm_metadata, dict):
        for route_family_key in ("passthrough_route_family", "route_family"):
            route_family_value = existing_litellm_metadata.get(route_family_key)
            if isinstance(route_family_value, str) and route_family_value.strip():
                preserved_route_family = route_family_value.strip()
                break

    merged_extra_fields = {
        **metadata,
        **(extra_fields or {}),
    }
    if preserved_route_family:
        merged_extra_fields.setdefault(
            "source_passthrough_route_family", preserved_route_family
        )
        merged_extra_fields.setdefault(
            "source_route_family", preserved_route_family
        )
        merged_extra_fields["grok_cli_chat_proxy_used"] = True
    return runtime._merge_litellm_metadata(
        request_body,
        tags_to_add=[
            *(metadata_tags if isinstance(metadata_tags, list) else []),
            *(tags_to_add or []),
        ],
        extra_fields=merged_extra_fields,
    )


async def _prepare_grok_native_oauth_passthrough_request(
    request_body: dict[str, Any],
    *,
    request: Request,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> tuple[bool, Optional[str], dict[str, Any], dict[str, Any]]:
    runtime = _require_runtime()
    model = runtime.normalize_grok_native_oauth_model(request_body.get("model"))
    if model is None:
        return False, None, {}, request_body

    prepared_body = dict(request_body)
    prepared_body["model"] = model
    prepared_body = _add_grok_native_oauth_metadata(
        prepared_body,
        model=model,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )
    (
        prepared_body,
        _grok_unsupported_hosted_tools,
    ) = runtime._drop_unsupported_codex_hosted_tools_from_request_body(
        prepared_body
    )
    (
        prepared_body,
        _grok_unsupported_request_params,
    ) = runtime._drop_unsupported_codex_request_params_from_request_body(
        prepared_body
    )
    (
        prepared_body,
        _grok_unsupported_input_items,
    ) = runtime._drop_unsupported_codex_input_items_from_request_body(
        prepared_body
    )
    _sanitize_grok_native_function_call_arguments_in_place(prepared_body)
    _rewrite_grok_native_unsupported_input_items_in_place(prepared_body)
    runtime._sanitize_xai_responses_request_body_in_place(prepared_body)
    (
        prepared_body,
        _removed_tool_choice,
    ) = runtime._drop_tool_choice_without_tools_from_request_body(prepared_body)
    access_token = await runtime.get_grok_native_oauth_access_token()
    headers = _build_grok_native_oauth_headers(
        access_token=access_token,
        model=model,
        request=request,
        request_body=prepared_body,
    )
    return (
        True,
        runtime._get_grok_passthrough_target_base(),
        headers,
        prepared_body,
    )


def _rewrite_grok_native_unsupported_input_items_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _anthropic_grok_normalization.rewrite_unsupported_input_items_from_request_body(
        _get_anthropic_grok_normalization_runtime(), request_body
    )


def _rewrite_grok_native_unsupported_input_items_in_place(
    request_body: dict[str, Any],
) -> list[dict[str, Any]]:
    return _anthropic_grok_normalization.rewrite_unsupported_input_items_in_place(
        _get_anthropic_grok_normalization_runtime(),
        request_body,
    )
