"""OpenCode Zen provider runtime extracted from the passthrough god module.

Host-owned utilities are supplied through :func:`configure_runtime`.  The
shared integrator can use :func:`install` to publish the same function objects
into the host module while retaining live monkeypatch lookups.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse

from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    constants as _constants,
)
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    normalization as _normalization,
)
from litellm.proxy.pass_through_endpoints.providers.common import (
    _raise_opencode_zen_auto_agent_candidate_unavailable as _common_raise_opencode_zen_unavailable,
)


Payload = dict[str, Any]


@dataclass(frozen=True)
class Runtime:
    """Callbacks owned by the passthrough host."""

    get_secret_str: Callable[[str], Optional[str]]
    assemble_headers: Callable[..., dict[str, str]]
    normalize_endpoint_for_target: Callable[[str, str], str]
    join_url_paths: Callable[[httpx.URL, str, str], str]
    extract_exception_status_code: Callable[[Exception], Optional[int]]
    extract_exception_detail: Callable[[Exception], Any]
    merge_metadata: Callable[..., Payload]
    add_route_family_logging_metadata: Callable[[Payload, str], Payload]
    build_langfuse_span_descriptor: Callable[..., Payload]
    normalization_runtime_factory: Callable[[], _normalization.Runtime]
    is_openai_responses_endpoint: Callable[[str], bool]
    has_anthropic_responses_adapter_endpoint: Callable[[str], bool]
    get_anthropic_adapter_model_candidates: Callable[[Payload], list[str]]
    load_local_api_key: Optional[Callable[[], Awaitable[str]]] = None
    raise_candidate_unavailable: Optional[Callable[[Exception], Any]] = None
    load_candidate_api_key: Optional[Callable[..., Awaitable[str]]] = None


_runtime: Optional[Runtime] = None


_HOST_FUNCTION_NAMES = (
    "_get_opencode_zen_target_base",
    "_get_opencode_go_target_base",
    "_get_opencode_zen_auth_file_path",
    "_load_local_opencode_zen_api_key",
    "_load_opencode_zen_api_key_for_candidate",
    "_build_opencode_zen_headers",
    "_add_opencode_zen_logging_metadata",
    "_get_anthropic_opencode_zen_normalization_runtime",
    "_get_opencode_zen_responses_tool_name",
    "_ordered_unique_str_values",
    "_strip_opencode_zen_unsupported_responses_tools",
    "_opencode_zen_chat_message_role",
    "_opencode_zen_chat_tool_call_id",
    "_opencode_zen_chat_message_tool_call_ids",
    "_opencode_zen_chat_message_tool_result_id",
    "_collect_opencode_zen_following_tool_block",
    "_sanitize_opencode_zen_completion_messages_for_chat_completion",
    "_opencode_zen_responses_sse_event",
    "_opencode_zen_response_payload_for_stream",
    "_opencode_zen_message_item_for_stream",
    "_opencode_zen_completed_response_for_stream",
    "_normalize_opencode_zen_responses_stream_for_codex",
    "_build_codex_opencode_zen_streaming_response",
    "_join_opencode_zen_passthrough_url",
)


def configure_runtime(runtime: Runtime) -> None:
    """Configure callbacks without importing the passthrough host module."""

    global _runtime
    _runtime = runtime
    _get_anthropic_opencode_zen_normalization_runtime.cache_clear()


def _require_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("OpenCode Zen runtime callbacks have not been configured")
    return _runtime


def install(host_globals: dict[str, Any]) -> None:
    """Configure live host lookups and publish same-object facades."""

    def _host(name: str) -> Any:
        return host_globals[name]

    def _normalization_runtime_factory() -> _normalization.Runtime:
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        return _normalization.Runtime(
            clean_secret_string=lambda value: _host("_clean_secret_string")(
                value if isinstance(value, str) else None
            ),
            merge_metadata=lambda *args, **kwargs: _host(
                "_merge_litellm_metadata"
            )(*args, **kwargs),
            add_logging_metadata=lambda *args, **kwargs: _host(
                "_add_opencode_zen_logging_metadata"
            )(*args, **kwargs),
            build_span=lambda *args, **kwargs: _host(
                "_build_langfuse_span_descriptor"
            )(*args, **kwargs),
            transform_responses_api_request_to_chat_completion_request=(
                LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
            ),
            async_responses_api_session_handler=(
                LiteLLMCompletionResponsesConfig.async_responses_api_session_handler
            ),
            iterate_responses_sse_events=lambda iterator: _host(
                "_iterate_responses_sse_events"
            )(iterator),
            coerce_namespace_to_mapping=lambda value: _host(
                "_coerce_namespace_to_mapping"
            )(value),
            responses_output_item_has_meaningful_content=lambda item: _host(
                "_responses_output_item_has_meaningful_content"
            )(item),
            streaming_response_factory=_host("StreamingResponse"),
        )

    configure_runtime(
        Runtime(
            get_secret_str=lambda name: _host("get_secret_str")(name),
            assemble_headers=lambda **kwargs: _host(
                "BaseOpenAIPassThroughHandler"
            )._assemble_headers(**kwargs),
            normalize_endpoint_for_target=lambda endpoint, base_target_url: _host(
                "BaseOpenAIPassThroughHandler"
            )._normalize_endpoint_for_target(
                endpoint=endpoint,
                base_target_url=base_target_url,
            ),
            join_url_paths=lambda base_url, path, provider: _host(
                "BaseOpenAIPassThroughHandler"
            )._join_url_paths(base_url, path, provider),
            extract_exception_status_code=lambda exc: _host(
                "_extract_adapter_exception_status_code"
            )(exc),
            extract_exception_detail=lambda exc: _host(
                "_extract_adapter_exception_detail"
            )(exc),
            merge_metadata=lambda *args, **kwargs: _host("_merge_litellm_metadata")(
                *args, **kwargs
            ),
            add_route_family_logging_metadata=lambda body, route_family: _host(
                "_add_route_family_logging_metadata"
            )(body, route_family),
            build_langfuse_span_descriptor=lambda *args, **kwargs: _host(
                "_build_langfuse_span_descriptor"
            )(*args, **kwargs),
            normalization_runtime_factory=_normalization_runtime_factory,
            is_openai_responses_endpoint=lambda endpoint: _host(
                "_is_openai_responses_endpoint"
            )(endpoint),
            has_anthropic_responses_adapter_endpoint=lambda endpoint: _host(
                "_has_anthropic_responses_adapter_endpoint"
            )(endpoint),
            get_anthropic_adapter_model_candidates=lambda body: _host(
                "_get_anthropic_adapter_model_candidates"
            )(body),
            load_local_api_key=lambda: _host(
                "_load_local_opencode_zen_api_key"
            )(),
            raise_candidate_unavailable=lambda exc: _host(
                "_raise_opencode_zen_auto_agent_candidate_unavailable"
            )(exc),
            load_candidate_api_key=lambda **kwargs: _host(
                "_load_opencode_zen_api_key_for_candidate"
            )(**kwargs),
        )
    )
    for name in _HOST_FUNCTION_NAMES:
        host_globals[name] = globals()[name]


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    runtime = _require_runtime()
    for secret_name in secret_names:
        value = _clean_secret_string(runtime.get_secret_str(secret_name))
        if value:
            return value
    return None


def _get_opencode_zen_target_base() -> str:
    runtime = _require_runtime()
    cleaned = (
        _clean_secret_string(runtime.get_secret_str("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(
            runtime.get_secret_str("AAWM_OPENCODE_ZEN_API_BASE")
        )
        or _clean_secret_string(os.getenv("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_OPENCODE_ZEN_API_BASE"))
        or _constants._OPENCODE_ZEN_DEFAULT_BASE_URL
    ).rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_opencode_go_target_base() -> str:
    runtime = _require_runtime()
    cleaned = (
        _clean_secret_string(runtime.get_secret_str("OPENCODE_GO_API_BASE"))
        or _clean_secret_string(
            runtime.get_secret_str("AAWM_OPENCODE_GO_API_BASE")
        )
        or _clean_secret_string(os.getenv("OPENCODE_GO_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_OPENCODE_GO_API_BASE"))
        or _constants._OPENCODE_GO_DEFAULT_BASE_URL
    ).rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_opencode_zen_auth_file_path() -> Optional[Path]:
    """Resolve the OpenCode Zen auth file path.

    The first nonempty auth-file environment variable is authoritative.
    If it is set but the path is missing or not a regular file, raise
    immediately without consulting later variables or HOME defaults.
    HOME-relative defaults are used only when no auth-file variable is
    configured.
    """
    for env_name in _constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS:
        value = _clean_secret_string(os.getenv(env_name))
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_file():
                raise ValueError(
                    f"OpenCode Zen auth file configured via {env_name} "
                    "is missing or not a regular file."
                )
            return candidate

    for candidate_str in _constants._OPENCODE_ZEN_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.is_file():
            return candidate
    return None


async def _load_local_opencode_zen_api_key() -> str:
    explicit_key = _get_first_secret_value(
        _constants._OPENCODE_ZEN_API_KEY_ENV_VARS
    )
    if explicit_key is not None:
        return explicit_key

    # Identify the configured source for error attribution (env var name
    # only, never the path value).
    configured_source: Optional[str] = None
    for env_name in _constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS:
        if _clean_secret_string(os.getenv(env_name)):
            configured_source = env_name
            break

    auth_path = _get_opencode_zen_auth_file_path()
    if auth_path is None:
        raise FileNotFoundError(
            "OpenCode Zen auth file not found. Expected "
            "'~/.local/share/opencode/auth.json' or set "
            "'LITELLM_OPENCODE_AUTH_FILE'."
        )

    source_label = configured_source or "default"

    try:
        raw_text = auth_path.read_text(encoding="utf-8")
    except Exception:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "is not readable."
        ) from None

    try:
        auth_data = json.loads(raw_text)
    except Exception:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "does not contain valid JSON."
        ) from None

    provider_auth = None
    if isinstance(auth_data, dict):
        provider_auth = auth_data.get("opencode-go") or auth_data.get("opencode")
    api_key = (
        _clean_secret_string(provider_auth.get("key"))
        if isinstance(provider_auth, dict)
        else None
    )
    auth_type = (
        _clean_secret_string(provider_auth.get("type"))
        if isinstance(provider_auth, dict)
        else None
    )
    if api_key is None or auth_type not in {None, "api"}:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "must contain provider 'opencode' with API-key auth."
        )
    return api_key


async def _load_opencode_zen_api_key_for_candidate(
    *,
    use_alias_candidate_probe: bool = False,
) -> str:
    runtime = _require_runtime()
    load_api_key = runtime.load_local_api_key
    try:
        if load_api_key is not None:
            return await load_api_key()
        return await _load_local_opencode_zen_api_key()
    except (FileNotFoundError, ValueError) as exc:
        if use_alias_candidate_probe:
            if runtime.raise_candidate_unavailable is not None:
                runtime.raise_candidate_unavailable(exc)
            _common_raise_opencode_zen_unavailable(exc)
        raise


async def _build_opencode_zen_headers(
    request: Request,
    *,
    use_alias_candidate_probe: bool = False,
) -> dict[str, str]:
    runtime = _require_runtime()
    load_candidate = runtime.load_candidate_api_key
    if load_candidate is not None:
        api_key = await load_candidate(
            use_alias_candidate_probe=use_alias_candidate_probe,
        )
    else:
        api_key = await _load_opencode_zen_api_key_for_candidate(
            use_alias_candidate_probe=use_alias_candidate_probe,
        )
    return runtime.assemble_headers(
        api_key=api_key,
        request=request,
    )


def _add_opencode_zen_logging_metadata(
    request_body: Payload,
    *,
    route_family: str,
    tag_prefix: str,
    requested_model: Any,
    adapter_model: Optional[str] = None,
    input_shape: Optional[str] = None,
    output_shape: Optional[str] = None,
    client_name: Optional[str] = None,
) -> Payload:
    runtime = _require_runtime()
    extra_fields: Payload = {
        "opencode_zen": True,
        "opencode_zen_requested_model": requested_model,
    }
    if client_name is not None:
        extra_fields["client_name"] = client_name
    if adapter_model is not None:
        extra_fields["opencode_zen_adapter_model"] = adapter_model
    if input_shape is not None:
        extra_fields["codex_adapter_input_shape"] = input_shape
    if output_shape is not None:
        extra_fields["codex_adapter_output_shape"] = output_shape

    tags = [tag_prefix, "opencode-zen"]
    if adapter_model is not None:
        tags.append(f"opencode-zen-model:{adapter_model}")

    return runtime.merge_metadata(
        runtime.add_route_family_logging_metadata(request_body, route_family),
        tags_to_add=tags,
        extra_fields=extra_fields,
    )


@lru_cache(maxsize=1)
def _get_anthropic_opencode_zen_normalization_runtime() -> (
    _normalization.Runtime
):
    return _require_runtime().normalization_runtime_factory()


def _get_opencode_zen_responses_tool_name(tool: Any) -> Optional[str]:
    return _normalization.get_responses_tool_name(
        _get_anthropic_opencode_zen_normalization_runtime(),
        tool,
    )


def _ordered_unique_str_values(
    values: list[Optional[str]],
) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _strip_opencode_zen_unsupported_responses_tools(
    request_body: Payload,
) -> Payload:
    return _normalization.strip_unsupported_responses_tools(
        _get_anthropic_opencode_zen_normalization_runtime(),
        request_body,
    )


def _opencode_zen_chat_message_role(message: Any) -> Optional[str]:
    return _normalization.chat_message_role(message)


def _opencode_zen_chat_tool_call_id(tool_call: Any) -> Optional[str]:
    return _normalization.chat_tool_call_id(tool_call)


def _opencode_zen_chat_message_tool_call_ids(message: Any) -> list[str]:
    return _normalization.chat_message_tool_call_ids(message)


def _opencode_zen_chat_message_tool_result_id(
    message: Any,
) -> Optional[str]:
    return _normalization.chat_message_tool_result_id(message)


def _collect_opencode_zen_following_tool_block(
    messages: list[Any],
    start_index: int,
) -> tuple[list[Any], list[Optional[str]], int]:
    return _normalization.collect_following_tool_block(messages, start_index)


def _sanitize_opencode_zen_completion_messages_for_chat_completion(
    completion_kwargs: Payload,
) -> tuple[Payload, Payload]:
    return _normalization.sanitize_completion_messages_for_chat_completion(
        completion_kwargs
    )


def _opencode_zen_responses_sse_event(
    event_type: str,
    payload: Payload,
) -> str:
    return _normalization.responses_sse_event(event_type, payload)


def _opencode_zen_response_payload_for_stream(
    *,
    response_id: str,
    model: str,
    status: str,
    output: Optional[list[Payload]] = None,
    usage: Optional[Payload] = None,
) -> Payload:
    return _normalization.response_payload_for_stream(
        response_id=response_id,
        model=model,
        status=status,
        output=output,
        usage=usage,
    )


def _opencode_zen_message_item_for_stream(
    *,
    message_id: str,
    status: str,
    output_text: str = "",
) -> Payload:
    return _normalization.message_item_for_stream(
        message_id=message_id,
        status=status,
        output_text=output_text,
    )


def _opencode_zen_completed_response_for_stream(
    *,
    response_event: Payload,
    response_id: str,
    model: str,
    message_id: Optional[str],
    output_text: str,
) -> Payload:
    return _normalization.completed_response_for_stream(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response_event=response_event,
        response_id=response_id,
        model=model,
        message_id=message_id,
        output_text=output_text,
    )


async def _normalize_opencode_zen_responses_stream_for_codex(
    response: Any,
    *,
    adapter_model: str,
) -> AsyncIterator[str]:
    async for chunk in _normalization.normalize_responses_stream_for_codex(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response,
        adapter_model=adapter_model,
    ):
        yield chunk


def _build_codex_opencode_zen_streaming_response(
    response: Any,
    *,
    adapter_model: str,
) -> StreamingResponse:
    return _normalization.build_codex_streaming_response(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response,
        adapter_model=adapter_model,
    )


def _join_opencode_zen_passthrough_url(
    base_target_url: str,
    endpoint: str,
) -> str:
    runtime = _require_runtime()
    normalized_endpoint = runtime.normalize_endpoint_for_target(
        endpoint,
        base_target_url,
    )
    return str(
        runtime.join_url_paths(
            httpx.URL(base_target_url),
            normalized_endpoint,
            _constants._OPENCODE_ZEN_PROVIDER,
        )
    )
