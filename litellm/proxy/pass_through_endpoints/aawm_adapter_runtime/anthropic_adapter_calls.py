"""Wave 6F extraction: Anthropic adapter-call machinery.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``.
Do not import ``llm_passthrough_endpoints`` at module scope.

Owns the perform/finalize/build/prepare/policy/auth/response pipeline
that the 12 god-module route-pair delegates call into.  Provider runtime
configs, route pairs, the combined OpenCode wrapper, native Anthropic
routes, and aawm.2/aawm.5 patches remain in the god module.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import FunctionType, SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
    cast,
)
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse
from typing_extensions import TypeGuard

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.anthropic.experimental_pass_through.providers import (
    common as _anthropic_provider_common,
)
from litellm.proxy.aawm_route_logging import (
    emit_aawm_route_access_log,
    record_aawm_route_rollup_turn,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config as _aawm_adapter_config,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    responses_finalize as _aawm_responses_finalize,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _build_langfuse_span_descriptor,
    _dedupe_sorted_str_list,
    _merge_litellm_metadata,
    _normalize_low_cardinality_tag_value,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
    load_text_watermark_config,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
    apply_request_watermark_egress,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

if TYPE_CHECKING:
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth

    # -- host-global dependencies (resolved via install()) --
    passthrough_endpoint_router: Any
    BaseOpenAIPassThroughHandler: Any
    pass_through_request: Callable[..., Awaitable[Response]]
    _anthropic_adapter_request_has_openai_client_auth: Callable[..., bool]
    _anthropic_adapter_request_uses_codex_native_auth: Callable[..., bool]
    _load_local_codex_auth_headers: Callable[..., Awaitable[Optional[dict[str, Any]]]]
    _anthropic_adapter_should_forward_direct_auth_headers: Callable[..., bool]
    _is_failed_responses_body: Callable[..., bool]
    _is_empty_success_responses_body: Callable[..., bool]
    _raise_responses_adapter_failed_response: Callable[..., None]
    _is_codex_auto_agent_malformed_tool_call_text_output: Callable[..., bool]
    _raise_codex_auto_agent_malformed_tool_call_text_payload: Callable[..., None]
    _build_empty_success_responses_diagnostic: Callable[..., dict[str, Any]]
    _build_anthropic_streaming_response_from_completion_adapter_stream: Callable[..., StreamingResponse]
    _get_openrouter_api_key: Callable[[], Optional[str]]
    _get_openrouter_target_base: Callable[[], str]
    _get_first_secret_value: Callable[..., Optional[str]]
    _clean_secret_string: Callable[..., Optional[str]]
    _apply_codex_tool_description_patches_to_request_body: Callable[..., Any]
    _ANTHROPIC_PROVIDER_SHAPING_RUNTIME: Any


# ---------------------------------------------------------------------------
# Constants (published back to host via install())
# ---------------------------------------------------------------------------

_AAWM_REQUEST_BODY_WALK_MAX_DEPTH = 64

_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS: tuple[str, ...] = (
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_API_KEY",
)

# Keep in sync with pass_through_endpoints.PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES.
# Do not import that parent-module constant at load time (RR-093 cycle).
_PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES = frozenset(
    {500, 502, 503, 504, 529}
)

_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES = sorted(
    _PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES - {429}
)

_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT = [
    429,
    500,
    502,
    503,
    504,
]

_OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS = """You are an OpenAI Responses function-calling agent for Claude Code.

Parallel tool calls are enabled. When the current user task asks for multiple independent tool calls, emit all independent function calls together in one response output array before receiving any tool result. Do not serialize independent Read, Glob, Grep, Bash, WebSearch, or WebFetch calls when their arguments are already specified or can be determined from the current task.

Follow the latest user task exactly. Use the provided tool schemas as the source of truth for arguments. Emit no assistant text before tool calls when the task asks for tool calls only. After tool results return, provide the requested final answer.

Do NOT Write report/summary/findings/analysis .md files unless EXPLICITLY asked to do. Regardless of a file write-- you need to return findings directly as your final assistant message."""


def _watermark_endpoint_from_path(*parts: Any) -> str:
    combined = " ".join(
        str(part or "") for part in parts if part is not None
    ).lower()
    if "chat/completions" in combined or "chat_completions" in combined:
        return "chat_completions"
    return "responses"


def _get_runtime_text_watermark_config() -> Any:
    payload = None
    try:
        from litellm.proxy.proxy_server import general_settings as _gs

        if isinstance(_gs, dict):
            payload = _gs.get("openai_passthrough_text_watermark")
        else:
            payload = getattr(_gs, "openai_passthrough_text_watermark", None)
    except Exception:
        payload = None
    return load_text_watermark_config(payload)


# ---------------------------------------------------------------------------
# Decode helper
# ---------------------------------------------------------------------------


def _decode_http_response_body(body: Any) -> str:
    # RR-054 #43: never raise UnicodeDecodeError into JSON parse call sites.
    return bytes(body).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Rollup / access-log helpers
# ---------------------------------------------------------------------------


def _build_adapted_route_rollup_kwargs(
    litellm_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "litellm_params": {
            "metadata": dict(litellm_metadata),
        }
    }


def _emit_adapted_route_access_log(
    *,
    request: Request,
    target_url: str,
    request_body: dict[str, Any],
    rollup_kwargs: dict[str, Any],
    adapter_label: str,
    provider_bound_body: Optional[dict[str, Any]] = None,
) -> None:
    try:
        emit_aawm_route_access_log(
            request=request,
            target=target_url,
            request_body=request_body,
            kwargs=rollup_kwargs,
            provider_bound_body=provider_bound_body,
        )
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to emit AAWM route access log for %s adapter",
            adapter_label,
            exc_info=True,
        )


def _record_adapted_completed_route_rollup_turn(
    rollup_kwargs: dict[str, Any],
    *,
    adapter_label: str,
) -> None:
    try:
        record_aawm_route_rollup_turn(rollup_kwargs)
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to record AAWM route rollup turn for %s adapter",
            adapter_label,
            exc_info=True,
        )


def _record_adapted_completed_route_rollup_after_stream(
    response: StreamingResponse,
    rollup_kwargs: dict[str, Any],
    *,
    adapter_label: str,
) -> StreamingResponse:
    original_iterator = response.body_iterator
    recorded = False

    async def _wrapped_iterator() -> Any:
        nonlocal recorded
        async for chunk in original_iterator:
            yield chunk
        if not recorded:
            recorded = True
            _record_adapted_completed_route_rollup_turn(
                rollup_kwargs,
                adapter_label=adapter_label,
            )

    response.body_iterator = _wrapped_iterator()
    return response


# ---------------------------------------------------------------------------
# Tool / schema normalization
# ---------------------------------------------------------------------------


def _add_codex_native_tool_alias_adapter_metadata(
    adapter_tags: list[str],
    adapter_extra_fields: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    adapter_tags.append("anthropic-openai-codex-native-tools")
    adapter_extra_fields["anthropic_adapter_codex_native_tool_aliases"] = True


def _normalize_openai_function_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    normalized_parameters = dict(parameters)
    if normalized_parameters.get("type") is None:
        normalized_parameters["type"] = "object"
    _sanitize_openai_object_schema_properties(normalized_parameters)

    return normalized_parameters


def _sanitize_openai_object_schema_properties(schema_node: Any) -> int:
    fix_count = 0
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "object" and not isinstance(schema_node.get("properties"), dict):
            schema_node["properties"] = {}
            fix_count += 1
        for value in schema_node.values():
            fix_count += _sanitize_openai_object_schema_properties(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_openai_object_schema_properties(item)
    return fix_count


def _normalize_openai_function_tool_schemas(translated_body: dict[str, Any]) -> None:
    tools = translated_body.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        if "parameters" in tool:
            tool["parameters"] = _normalize_openai_function_tool_parameters(tool.get("parameters"))

        function_block = tool.get("function")
        if isinstance(function_block, dict):
            function_block["parameters"] = _normalize_openai_function_tool_parameters(function_block.get("parameters"))


# ---------------------------------------------------------------------------
# Parallel instruction policy
# ---------------------------------------------------------------------------


def _get_openai_adapter_function_tool_names(
    request_body: dict[str, Any],
) -> list[str]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return []

    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _apply_responses_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
    *,
    tag_prefix: str,
    metadata_prefix: str,
    span_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request_body.get("parallel_tool_calls") is not True:
        return request_body, {}

    function_tool_names = _get_openai_adapter_function_tool_names(request_body)
    if len(function_tool_names) < 2:
        return request_body, {}

    existing_instructions = request_body.get("instructions")
    if not isinstance(existing_instructions, str) or not existing_instructions.strip():
        return request_body, {}

    policy_block = _OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS
    # RR-054 #20: prepend the parallel-tool policy; never wipe the caller system prompt.
    if policy_block in existing_instructions:
        return request_body, {}

    rewritten_instructions = f"{policy_block}\n\n{existing_instructions}"
    updated_body = dict(request_body)
    updated_body["instructions"] = rewritten_instructions
    original_hash = hashlib.sha256(existing_instructions.encode("utf-8", errors="replace")).hexdigest()
    changes = {
        f"{metadata_prefix}_parallel_instruction_policy_applied": True,
        f"{metadata_prefix}_parallel_instruction_original_chars": len(existing_instructions),
        f"{metadata_prefix}_parallel_instruction_rewritten_chars": len(rewritten_instructions),
        f"{metadata_prefix}_parallel_instruction_original_hash": original_hash,
        f"{metadata_prefix}_parallel_instruction_tool_names": function_tool_names,
        f"{metadata_prefix}_parallel_instruction_mode": "prepend",
    }
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            f"{tag_prefix}-parallel-instruction-policy",
            *[
                f"{tag_prefix}-parallel-tool:{_normalize_low_cardinality_tag_value(tool_name) or 'unknown'}"
                for tool_name in function_tool_names
            ],
        ],
        extra_fields={
            **changes,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata={
                        "tool_names": function_tool_names,
                        "original_chars": len(existing_instructions),
                        "rewritten_chars": len(rewritten_instructions),
                        "mode": "prepend",
                    },
                )
            ],
        },
    )
    return updated_body, changes


def _apply_openai_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _apply_responses_adapter_parallel_instruction_policy(
        request_body,
        tag_prefix="openai-adapter",
        metadata_prefix="openai_adapter",
        span_name="openai_adapter.parallel_instruction_policy",
    )


def _apply_openrouter_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _apply_responses_adapter_parallel_instruction_policy(
        request_body,
        tag_prefix="openrouter-adapter",
        metadata_prefix="openrouter_adapter",
        span_name="openrouter_adapter.parallel_instruction_policy",
    )


# ---------------------------------------------------------------------------
# Bash tool-choice forcing
# ---------------------------------------------------------------------------


def _get_latest_adapter_user_prompt_text(request_body: dict[str, Any]) -> Optional[str]:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _prompt_explicitly_requests_bash_tool(prompt_text: Optional[str]) -> bool:
    if not isinstance(prompt_text, str) or not prompt_text:
        return False
    lowered_prompt = prompt_text.lower()
    return "bash tool" in lowered_prompt or "run the bash command" in lowered_prompt


def _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
    request_body: dict[str, Any],
    translated_body: dict[str, Any],
) -> dict[str, Any]:
    if translated_body.get("tool_choice") is not None:
        return {}

    tools = translated_body.get("tools")
    if not isinstance(tools, list):
        return {}

    latest_user_prompt = _get_latest_adapter_user_prompt_text(request_body)
    if not _prompt_explicitly_requests_bash_tool(latest_user_prompt):
        return {}

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tool_name = tool.get("name")
        if tool_name in {"Bash", "run_shell_command"}:
            translated_body["tool_choice"] = {"type": "function", "name": tool_name}
            return {"forced_explicit_bash_tool_choice": tool_name}
    return {}


def _apply_forced_bash_tool_choice_for_responses_adapter(
    request_body: dict[str, Any],
    translated_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
        request_body,
        translated_body,
    )
    if not forced_tool_choice_changes:
        return translated_body, {}
    return (
        _merge_litellm_metadata(
            translated_body,
            extra_fields=forced_tool_choice_changes,
        ),
        forced_tool_choice_changes,
    )


def _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    if request_body.get("tool_choice") is not None:
        return {}

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return {}

    latest_user_prompt = _get_latest_adapter_user_prompt_text(request_body)
    if not _prompt_explicitly_requests_bash_tool(latest_user_prompt):
        return {}

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name")
        if tool_name in {"Bash", "run_shell_command"}:
            request_body["tool_choice"] = {"type": "tool", "name": tool_name}
            return {"forced_explicit_bash_tool_choice": tool_name}
    return {}


# ---------------------------------------------------------------------------
# MCP tool check
# ---------------------------------------------------------------------------


def _responses_request_contains_mcp_tools(request_body: dict[str, Any]) -> bool:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "mcp":
            return True
    return False


# ---------------------------------------------------------------------------
# Namespace coercion
# ---------------------------------------------------------------------------


def _coerce_mapping_to_namespace(
    value: Any,
    *,
    _depth: int = 0,
    _max_depth: int = _AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
) -> Any:
    # RR-054 #27: bound recursion so pathological SSE payloads cannot explode CPU/stack.
    if _depth > _max_depth:
        return value
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: _coerce_mapping_to_namespace(val, _depth=_depth + 1, _max_depth=_max_depth)
                for key, val in value.items()
            }
        )
    if isinstance(value, list):
        return [_coerce_mapping_to_namespace(item, _depth=_depth + 1, _max_depth=_max_depth) for item in value]
    return value


# ---------------------------------------------------------------------------
# Grok native prior function-call replay drop
# ---------------------------------------------------------------------------


def _drop_anthropic_grok_native_prior_function_call_replay(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    # First pass: collect call_ids for prior function_call items to drop.
    drop_call_ids: set[str] = set()
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id.strip():
            drop_call_ids.add(call_id.strip())

    updated_input_items: list[Any] = []
    dropped_items: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        cleaned_call_id = call_id.strip() if isinstance(call_id, str) and call_id.strip() else None
        # RR-054 #21: drop prior function_call items and any paired outputs so the
        # provider does not see orphaned function_call_output rows.
        if item_type == "function_call" or (
            item_type == "function_call_output" and cleaned_call_id is not None and cleaned_call_id in drop_call_ids
        ):
            metadata_item: dict[str, Any] = {
                "type": item_type,
                "index": index,
            }
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                metadata_item["name"] = name.strip()
            if cleaned_call_id is not None:
                metadata_item["call_id_hash"] = hashlib.sha256(
                    cleaned_call_id.encode("utf-8", errors="replace")
                ).hexdigest()[:12]
            dropped_items.append(metadata_item)
            continue
        updated_input_items.append(item)

    if not dropped_items:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    dropped_names = _dedupe_sorted_str_list(
        [item["name"] for item in dropped_items if isinstance(item.get("name"), str) and item["name"]]
    )
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "anthropic-grok-native-prior-function-call-replay-dropped",
        ],
        extra_fields={
            "anthropic_grok_native_prior_function_call_replay_dropped_count": len(dropped_items),
            "anthropic_grok_native_prior_function_call_replay_dropped_items": dropped_items,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.grok_native_prior_function_call_replay_dropped",
                    metadata={
                        "dropped_count": len(dropped_items),
                        "dropped_names": dropped_names,
                    },
                )
            ],
        },
    )
    return updated_body, dropped_items


# ---------------------------------------------------------------------------
# Response building
# ---------------------------------------------------------------------------


def _build_anthropic_response_from_responses_response(
    response_body: dict[str, Any],
    *,
    reject_empty_success: bool = False,
    diagnostic_context: Optional[dict[str, Any]] = None,
    use_codex_native_tools: bool = False,
    retryable_failed_response: bool = False,
    failed_response_adapter_model: Optional[str] = None,
    failed_response_adapter: str = "anthropic_responses_adapter",
    failed_response_adapter_label: str = "Responses",
    malformed_intake_context: Optional[dict[str, Any]] = None,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    if _is_failed_responses_body(response_body):  # noqa: F821
        _raise_responses_adapter_failed_response(  # noqa: F821
            response_body=response_body,
            adapter_model=failed_response_adapter_model or str(response_body.get("model") or "unknown-model"),
            adapter=failed_response_adapter,
            adapter_label=failed_response_adapter_label,
            retryable_alias_candidate=retryable_failed_response,
        )

    if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):  # noqa: F821
        _raise_codex_auto_agent_malformed_tool_call_text_payload(  # noqa: F821
            response_body=response_body,
            adapter_model=failed_response_adapter_model or str(response_body.get("model") or "unknown-model"),
            adapter=failed_response_adapter,
            adapter_label=failed_response_adapter_label,
            intake_context=malformed_intake_context,
        )

    if reject_empty_success and _is_empty_success_responses_body(response_body):  # noqa: F821
        diagnostic = _build_empty_success_responses_diagnostic(  # noqa: F821
            response_body=response_body,
            diagnostic_context=diagnostic_context,
        )
        verbose_proxy_logger.warning(
            "OpenRouter Responses adapter returned empty successful response: %s",
            json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)[:8000],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "OpenRouter Responses adapter returned empty successful response",
                "diagnostic": diagnostic,
            },
        )

    adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    translated_response = adapter.translate_response(
        ResponsesAPIResponse(**response_body),
        use_codex_native_tools=use_codex_native_tools,
    )
    translated_response_any = cast(Any, translated_response)
    if hasattr(translated_response_any, "model_dump_json"):
        serialized_response = translated_response_any.model_dump_json(exclude_none=True)
    elif hasattr(translated_response_any, "json"):
        serialized_response = translated_response_any.json(exclude_none=True)
    else:
        serialized_response = json.dumps(translated_response)
    return Response(
        content=serialized_response,
        media_type="application/json",
    )


def _build_completion_adapter_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(request_body.get("metadata") or {})
    litellm_metadata = request_body.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return metadata

    # Normal completion callbacks turn metadata.trace_* into Langfuse trace
    # fields. Keep provider-specific litellm_metadata intact, but mirror the
    # trace context into metadata so completion adapters match passthrough logs.
    for key in (
        "session_id",
        "trace_id",
        "existing_trace_id",
        "trace_name",
        "trace_user_id",
        "trace_environment",
    ):
        value = litellm_metadata.get(key)
        if value and (key in {"trace_name", "trace_user_id"} or not metadata.get(key)):
            metadata[key] = value
    for key in (
        "source_trace_name",
        "agent_name",
        "aawm_claude_agent_name",
        "tenant_id",
        "aawm_tenant_id",
        "aawm_claude_project",
    ):
        value = litellm_metadata.get(key)
        if value:
            metadata[key] = value
    for key in (
        "passthrough_route_family",
        "anthropic_adapter_model",
        "anthropic_adapter_original_model",
        "anthropic_adapter_target_endpoint",
        "langfuse_spans",
    ):
        value = litellm_metadata.get(key)
        if value:
            metadata[key] = value
    litellm_tags = litellm_metadata.get("tags")
    if isinstance(litellm_tags, list):
        existing_tags = metadata.get("tags")
        if not isinstance(existing_tags, list):
            existing_tags = []
        metadata["tags"] = [
            *existing_tags,
            *[tag for tag in litellm_tags if tag not in existing_tags],
        ]
    return metadata


def _copy_translated_anthropic_adapter_response_headers(
    *,
    translated_response: Response,
    upstream_response: Response,
) -> None:
    for header_name, header_value in upstream_response.headers.items():
        if header_name.lower() in {
            "content-length",
            "content-encoding",
            "transfer-encoding",
        }:
            continue
        translated_response.headers[header_name] = header_value


# ---------------------------------------------------------------------------
# Access-log target annotation
# ---------------------------------------------------------------------------


def _get_anthropic_adapter_access_log_target_label(
    target_url: Union[str, httpx.URL],
) -> str:
    parsed_url = urlparse(str(target_url))
    hostname = parsed_url.hostname or "unknown-target"
    path = parsed_url.path or "/"
    query = f"?{parsed_url.query}" if parsed_url.query else ""
    return f"{hostname}{path}{query}"


def _annotate_request_scope_for_adapted_access_log(request: Request, target_url: Union[str, httpx.URL]) -> None:
    """Record adapted target for access logs without mutating live query_string.

    RR-054 #39: cosmetic logging must not corrupt ASGI ``query_string`` / path
    observed by later middleware, guardrails, or exception handlers. The target
    label lives on private scope keys and is consumed by access-log builders.
    """
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return

    target_label = _get_anthropic_adapter_access_log_target_label(target_url)
    if scope.get("_aawm_adapted_access_log_target") == target_label:
        return
    scope["_aawm_adapted_access_log_target"] = target_label
    request_url = getattr(request, "url", None)
    if request_url is not None:
        original_path = getattr(request_url, "path", None) or scope.get("path", "")
        original_query = getattr(request_url, "query", None) or ""
    else:
        original_path = scope.get("path", "")
        raw_query_string = scope.get("query_string", b"")
        if isinstance(raw_query_string, bytes):
            original_query = raw_query_string.decode("utf-8", errors="replace")
        else:
            original_query = str(raw_query_string or "")
    if isinstance(original_query, bytes):
        original_query = original_query.decode("utf-8", errors="replace")
    display_query = f"{original_query} -> {target_label}" if original_query else f"adapted_to={target_label}"
    scope["_aawm_adapted_access_log_display_path"] = (
        f"{original_path}?{display_query}" if original_path else display_query
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_anthropic_adapter_response(response_obj: Any) -> str:
    if hasattr(response_obj, "model_dump_json"):
        return response_obj.model_dump_json(exclude_none=True)
    if hasattr(response_obj, "json"):
        return response_obj.json(exclude_none=True)
    return json.dumps(response_obj)


def _build_anthropic_response_from_completion_adapter_response(
    response_obj: Any,
) -> Response:
    return Response(
        content=_serialize_anthropic_adapter_response(response_obj),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Provider API keys / target bases
# ---------------------------------------------------------------------------


def _get_anthropic_adapter_openai_target_base(
    request: Request,
    *,
    prefer_chatgpt_codex_backend: bool = False,
) -> str:
    if prefer_chatgpt_codex_backend or _anthropic_adapter_request_uses_codex_native_auth(request):  # noqa: F821
        return os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
    return os.getenv("OPENAI_API_BASE") or "https://api.openai.com/"


def _get_anthropic_adapter_openrouter_api_key() -> Optional[str]:
    return _get_openrouter_api_key()  # noqa: F821


def _get_anthropic_adapter_nvidia_api_key() -> Optional[str]:
    return _get_first_secret_value(_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS)  # noqa: F821


def _get_anthropic_adapter_nvidia_target_base() -> str:
    cleaned = (
        _clean_secret_string(os.getenv("NVIDIA_NIM_API_BASE"))  # noqa: F821
        or _clean_secret_string(os.getenv("AAWM_NVIDIA_API_BASE"))  # noqa: F821
        or "https://integrate.api.nvidia.com/v1"
    )
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_anthropic_adapter_openrouter_target_base() -> str:
    return _get_openrouter_target_base()  # noqa: F821


# ---------------------------------------------------------------------------
# Auth context resolution
# ---------------------------------------------------------------------------


async def _resolve_anthropic_openai_responses_adapter_auth_context(
    request: Request,
) -> tuple[dict[str, Any], bool, bool, Optional[str]]:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
        _get_bound_codex_oauth_candidate_identity,
        _load_bound_codex_oauth_auth,
    )

    if _get_bound_codex_oauth_candidate_identity(request) is not None:
        selected_auth = await _load_bound_codex_oauth_auth(request)
        return (
            selected_auth.headers,
            False,
            True,
            "openai",
        )

    local_codex_headers = None
    has_client_auth = _anthropic_adapter_request_has_openai_client_auth(request)  # noqa: F821
    uses_codex_native_auth = _anthropic_adapter_request_uses_codex_native_auth(request)  # noqa: F821
    if not has_client_auth:
        local_codex_headers = await _load_local_codex_auth_headers(request)  # noqa: F821

    custom_headers: dict[str, Any] = {}
    forward_headers = _anthropic_adapter_should_forward_direct_auth_headers(request)  # noqa: F821
    if local_codex_headers is not None:
        custom_headers = local_codex_headers
        forward_headers = False
    elif not has_client_auth:
        openai_api_key = passthrough_endpoint_router.get_credentials(  # noqa: F821
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            region_name=None,
        )
        if openai_api_key is None:
            raise Exception(
                "Anthropic adapter requests for OpenAI/Codex models require forwarded OpenAI/Codex auth headers or 'OPENAI_API_KEY' in environment."
            )
        custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(  # noqa: F821
            api_key=openai_api_key,
            request=request,
        )
        forward_headers = False

    use_chatgpt_codex_defaults = uses_codex_native_auth or local_codex_headers is not None
    egress_credential_family = "openai" if local_codex_headers is not None else None
    return (
        custom_headers,
        forward_headers,
        use_chatgpt_codex_defaults,
        egress_credential_family,
    )


# ---------------------------------------------------------------------------
# Responses adapter request body
# ---------------------------------------------------------------------------


def _build_anthropic_responses_adapter_request_body(
    request_body: dict[str, Any],
    *,
    adapter_model: str,
    route_family: str = "anthropic_openai_responses_adapter",
    tag_prefix: str = "anthropic-openai-responses-adapter",
    span_name: str = "anthropic.openai_responses_adapter",
    target_endpoint: str = "/v1/responses",
    use_chatgpt_codex_defaults: bool = False,
) -> dict[str, Any]:
    return _anthropic_provider_common.build_responses_request_body(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,  # noqa: F821
        request_body,
        adapter_model=adapter_model,
        route_family=route_family,
        tag_prefix=tag_prefix,
        span_name=span_name,
        target_endpoint=target_endpoint,
        use_chatgpt_codex_defaults=use_chatgpt_codex_defaults,
    )


# ---------------------------------------------------------------------------
# Completion adapter request body
# ---------------------------------------------------------------------------


def _prepare_anthropic_completion_adapter_request_body(
    prepared_request_body: Payload,
    *,
    adapter_model: str,
    route_family: str,
    tag_prefix: str,
    span_name: str,
    target_endpoint_label: str,
    span_metadata_extra: Optional[Payload] = None,
) -> Payload:
    return _anthropic_provider_common.prepare_completion_request_body(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,  # noqa: F821
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=route_family,
        tag_prefix=tag_prefix,
        span_name=span_name,
        target_endpoint_label=target_endpoint_label,
        span_metadata_extra=span_metadata_extra,
    )


# ---------------------------------------------------------------------------
# Policy application
# ---------------------------------------------------------------------------


def _apply_anthropic_responses_adapter_common_request_policies(
    prepared_request_body: Payload,
    translated_request_body: Payload,
    *,
    parallel_policy_log_label: str,
    forced_tool_choice_log_label: str,
) -> Payload:
    config = _aawm_adapter_config.AnthropicResponsesAdapterConfig(
        adapter="compat",
        adapter_label="compat",
        provider="openai",
        unexpected_detail="compat",
        parallel_policy_log_label=parallel_policy_log_label,
        forced_tool_choice_log_label=forced_tool_choice_log_label,
    )
    return _anthropic_provider_common.apply_responses_policies(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,  # noqa: F821
        prepared_request_body,
        translated_request_body,
        config=config,
    )


def _apply_anthropic_responses_adapter_policies_from_config(
    prepared_request_body: Payload,
    translated_request_body: Payload,
    *,
    config: _aawm_adapter_config.AnthropicResponsesAdapterConfig,
) -> Payload:
    return _anthropic_provider_common.apply_responses_policies(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,  # noqa: F821
        prepared_request_body,
        translated_request_body,
        config=config,
    )


# ---------------------------------------------------------------------------
# Finalize / perform pipeline
# ---------------------------------------------------------------------------


async def _finalize_anthropic_responses_adapter_upstream_response(
    *,
    upstream_response: object,
    request: Request,
    translated_request_body: Payload,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    provider: str,
    target_url: object,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool,
    use_codex_native_tools: bool = False,
    unexpected_detail: str,
    response_builder_kwargs: Optional[Payload] = None,
    stream_builder_kwargs: Optional[Payload] = None,
    malformed_upstream_url: Optional[object] = None,
    skip_stream_probe_validation: bool = False,
) -> Response:
    """Thin compatibility wrapper around package-owned response finalization."""
    return await _aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response(
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        adapter=adapter,
        adapter_label=adapter_label,
        provider=provider,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        use_codex_native_tools=use_codex_native_tools,
        unexpected_detail=unexpected_detail,
        response_builder_kwargs=response_builder_kwargs,
        stream_builder_kwargs=stream_builder_kwargs,
        malformed_upstream_url=malformed_upstream_url,
        skip_stream_probe_validation=skip_stream_probe_validation,
    )


async def _finalize_anthropic_responses_adapter_from_config(
    *,
    config: _aawm_adapter_config.AnthropicResponsesAdapterConfig,
    upstream_response: object,
    request: Request,
    translated_request_body: Payload,
    adapter_model: str,
    target_url: object,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool,
    use_codex_native_tools: Optional[bool] = None,
    malformed_upstream_url: Optional[object] = None,
) -> Response:
    """Config-driven Responses finalize entry (RR-054 #9)."""
    kwargs = _aawm_adapter_config.responses_finalize_kwargs(
        config,
        adapter_model=adapter_model,
        translated_request_body=translated_request_body,
    )
    if use_codex_native_tools is not None:
        kwargs["use_codex_native_tools"] = use_codex_native_tools
    if malformed_upstream_url is not None:
        kwargs["malformed_upstream_url"] = malformed_upstream_url
    return await _finalize_anthropic_responses_adapter_upstream_response(
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        **kwargs,
    )



def _session_affinity_mod():
    """Lazy session_affinity import (safe under module rebinding / tests)."""
    import sys as _sys

    _sa = _sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if _sa is None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as _sa,
        )
    return _sa


async def _ensure_anthropic_nested_session_owner_pre_egress(
    *,
    request: Request,
    request_body: Optional[Payload],
    provider: Any,
    model: Any,
    route_family: Any,
    endpoint_contract: str,
    state_format: str,
    failure_phase: str,
) -> None:
    """Reserve/renew nested session ownership with concrete resolved attrs.

    Called at the last common pre-egress points after adapter/provider/model/
    route resolution and before actual provider send. Uses exact provider,
    concrete model, route_family, endpoint_contract, state_format, and any
    safe account-lane labels already present on the request.

    Idempotent:
    - alias ``candidate_loop`` / pass-through that already reserved a concrete
      owner identity is renewed in place (attributes preserved)
    - generic nested placeholders (e.g. ``anthropic_nested``) are upgraded to
      the concrete resolved identity before send/finalize
    """

    _sa = _session_affinity_mod()
    body = request_body if isinstance(request_body, dict) else {}
    existing = _sa.get_request_session_owner_lease(request)
    prior = (
        existing.attributes
        if existing is not None and isinstance(existing.attributes, dict)
        else None
    )
    prior_route = str((prior or {}).get("route_family") or "").strip().lower()
    prior_is_generic_placeholder = (not prior) or prior_route in {
        "",
        "anthropic_nested",
        "codex_nested",
    }
    if (
        existing is not None
        and existing.held_reservation
        and prior
        and not prior_is_generic_placeholder
    ):
        # Already reserved with concrete candidate/handler attrs — renew only.
        await _sa.ensure_session_owner_guard_for_request(
            request=request,
            request_body=body,
            session_identity=existing.session_identity
            or _sa.resolve_canonical_session_identity(request, body),
            requested_attributes=prior,
            alias_model=str(model) if model is not None else None,
            failure_phase=failure_phase,
        )
        return

    account_identity = _sa.extract_account_identity_from_context(
        request=request,
        request_body=body,
    )
    attrs = _sa.build_session_owner_attributes(
        provider=provider,
        model=model,
        route_family=route_family,
        endpoint_contract=endpoint_contract,
        state_format=state_format,
        ingress="anthropic_nested_pre_egress",
        requested_model=body.get("model") if isinstance(body, dict) else None,
        extra=account_identity,
    )
    if existing is not None and existing.held_reservation:
        # Upgrade generic nested placeholder reservation before send.
        _sa.refresh_request_session_owner_lease_attributes(request, attrs)
    await _sa.ensure_session_owner_guard_for_request(
        request=request,
        request_body=body,
        session_identity=(
            existing.session_identity
            if existing is not None and existing.session_identity
            else _sa.resolve_canonical_session_identity(request, body)
        ),
        requested_attributes=attrs,
        alias_model=str(model) if model is not None else None,
        failure_phase=failure_phase,
    )


async def _perform_anthropic_responses_adapter_pass_through(
    *,
    config: _aawm_adapter_config.AnthropicResponsesAdapterConfig,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    translated_request_body: Payload,
    adapter_model: str,
    target_url: object,
    custom_headers: Payload,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool = False,
    forward_headers: bool = False,
    allowed_forward_headers: Optional[list[str]] = None,
    allowed_pass_through_prefixed_headers: Optional[list[str]] = None,
    custom_llm_provider: Optional[str] = None,
    egress_credential_family: Optional[str] = None,
    expected_target_family: Optional[str] = None,
    retryable_upstream_status_codes: Optional[list[int]] = None,
    pass_through_fn: Optional[Callable[..., Awaitable[object]]] = None,
    use_codex_native_tools: Optional[bool] = None,
    malformed_upstream_url: Optional[object] = None,
    extra_pass_through_kwargs: Optional[Payload] = None,
) -> Response:
    """Shared Responses adapter pass-through + finalize driver (RR-054 #9)."""
    transport = pass_through_fn or pass_through_request  # noqa: F821
    retry_codes = retryable_upstream_status_codes
    if retry_codes is None:
        retry_codes = list(
            _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
            if use_alias_candidate_probe
            else _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT
        )
    pt_kwargs: dict[str, Any] = {
        "request": request,
        "target": str(target_url),
        "custom_headers": custom_headers,
        "user_api_key_dict": user_api_key_dict,
        "custom_body": translated_request_body,
        "forward_headers": forward_headers,
        "stream": bool(translated_request_body.get("stream")),
        "custom_llm_provider": custom_llm_provider or config.provider,
        "egress_credential_family": egress_credential_family or config.provider,
        "expected_target_family": expected_target_family or config.provider,
        "retryable_upstream_status_codes": retry_codes,
        "caller_managed_hidden_retry": use_alias_candidate_probe,
    }
    if allowed_forward_headers is not None:
        pt_kwargs["allowed_forward_headers"] = allowed_forward_headers
    if allowed_pass_through_prefixed_headers is not None:
        pt_kwargs["allowed_pass_through_prefixed_headers"] = allowed_pass_through_prefixed_headers
    if extra_pass_through_kwargs:
        pt_kwargs.update(extra_pass_through_kwargs)
    # D1-612: last common pre-egress point for nested Responses adapters.
    # Concrete provider/model/route are known here; reserve before send so
    # promotion cannot pin generic anthropic_nested inbound placeholders.
    await _ensure_anthropic_nested_session_owner_pre_egress(
        request=request,
        request_body=translated_request_body
        if isinstance(translated_request_body, dict)
        else None,
        provider=pt_kwargs.get("custom_llm_provider") or config.provider,
        model=adapter_model,
        route_family=(
            pt_kwargs.get("egress_credential_family")
            or pt_kwargs.get("expected_target_family")
            or config.adapter
            or config.provider
        ),
        endpoint_contract="openai_responses",
        state_format="openai_responses",
        failure_phase="session_owner_anthropic_nested_pre_egress",
    )
    upstream_response = await transport(**pt_kwargs)
    return await _finalize_anthropic_responses_adapter_from_config(
        config=config,
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        use_codex_native_tools=use_codex_native_tools,
        malformed_upstream_url=malformed_upstream_url,
    )


async def _perform_normalized_anthropic_completion_adapter_stream(
    *,
    handler: Any,
    handler_call_kwargs: dict[str, Any],
    handler_extra_kwargs: dict[str, Any],
    completion_stream_normalizer: Callable[[Any], Any],
    completion_kwargs: Optional[dict[str, Any]] = None,
    tool_name_mapping: Optional[dict[str, str]] = None,
) -> object:
    if completion_kwargs is None or tool_name_mapping is None:
        completion_kwargs, tool_name_mapping = handler._prepare_completion_kwargs(
            **handler_call_kwargs,
            extra_kwargs=handler_extra_kwargs,
        )
    _watermark_metadata = None
    if isinstance(completion_kwargs, dict):
        _watermark_metadata = completion_kwargs.get("litellm_metadata") or completion_kwargs.get(
            "metadata"
        )
    if not isinstance(_watermark_metadata, dict) and isinstance(handler_extra_kwargs, dict):
        _watermark_metadata = handler_extra_kwargs.get("litellm_metadata")
    if not isinstance(_watermark_metadata, dict):
        _watermark_metadata = {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions"),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body
    raw_completion_stream = await litellm.acompletion(**completion_kwargs)
    normalized_completion_stream = completion_stream_normalizer(raw_completion_stream)
    return handler._transform_completion_response(
        normalized_completion_stream,
        model=handler_call_kwargs["model"],
        stream=True,
        tool_name_mapping=tool_name_mapping,
    )


def _is_anthropic_messages_response(
    value: object,
) -> TypeGuard[AnthropicMessagesResponse]:
    return isinstance(value, dict)


def _finalize_anthropic_completion_adapter_response(
    *,
    completion_response: object,
    stream_flag: bool,
    fake_stream: bool,
    rollup_kwargs: dict[str, Any],
    adapter_label: str,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    if stream_flag:
        if fake_stream:
            if not _is_anthropic_messages_response(completion_response):
                raise TypeError("Fake Anthropic streaming requires a non-streaming response")
            response_stream = FakeAnthropicMessagesStreamIterator(completion_response)
        else:
            response_stream = completion_response  # type: ignore[assignment]
        streaming_response = _build_anthropic_streaming_response_from_completion_adapter_stream(  # noqa: F821
            response_stream,
        )
        return _record_adapted_completed_route_rollup_after_stream(
            streaming_response,
            rollup_kwargs,
            adapter_label=adapter_label,
        )

    response = _build_anthropic_response_from_completion_adapter_response(
        completion_response,
    )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label=adapter_label,
    )
    return response


def _build_anthropic_completion_adapter_handler_call_kwargs(
    *,
    prepared_request_body: Payload,
    model_name: str,
    upstream_stream: bool,
) -> dict[str, Any]:
    """Build completion-handler kwargs from a prepared Anthropic messages body."""
    raw_max_tokens = prepared_request_body.get("max_tokens")
    raw_messages = prepared_request_body.get("messages")
    raw_stop_sequences = prepared_request_body.get("stop_sequences")
    raw_system = prepared_request_body.get("system")
    raw_temperature = prepared_request_body.get("temperature")
    raw_thinking = prepared_request_body.get("thinking")
    raw_reasoning_effort = prepared_request_body.get("reasoning_effort")
    raw_tool_choice = prepared_request_body.get("tool_choice")
    raw_tools = prepared_request_body.get("tools")
    raw_top_k = prepared_request_body.get("top_k")
    raw_top_p = prepared_request_body.get("top_p")
    raw_output_format = prepared_request_body.get("output_format")
    raw_output_config = prepared_request_body.get("output_config")
    return {
        "max_tokens": (
            raw_max_tokens
            if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool)
            else 1024
        ),
        "messages": raw_messages if isinstance(raw_messages, list) else [],
        "model": model_name,
        "metadata": _build_completion_adapter_metadata(prepared_request_body),
        "stop_sequences": (
            [item for item in raw_stop_sequences if isinstance(item, str)]
            if isinstance(raw_stop_sequences, list)
            else None
        ),
        "stream": upstream_stream,
        "system": raw_system if isinstance(raw_system, str) else None,
        "temperature": (
            float(raw_temperature)
            if isinstance(raw_temperature, (int, float)) and not isinstance(raw_temperature, bool)
            else None
        ),
        "thinking": raw_thinking if isinstance(raw_thinking, dict) else None,
        "reasoning_effort": (
            raw_reasoning_effort
            if isinstance(raw_reasoning_effort, str) and raw_reasoning_effort
            else None
        ),
        "tool_choice": raw_tool_choice if isinstance(raw_tool_choice, dict) else None,
        "tools": raw_tools if isinstance(raw_tools, list) else None,
        "top_k": (
            raw_top_k if isinstance(raw_top_k, int) and not isinstance(raw_top_k, bool) else None
        ),
        "top_p": (
            float(raw_top_p)
            if isinstance(raw_top_p, (int, float)) and not isinstance(raw_top_p, bool)
            else None
        ),
        "output_format": raw_output_format if isinstance(raw_output_format, dict) else None,
        "output_config": raw_output_config if isinstance(raw_output_config, dict) else None,
    }


async def _perform_anthropic_completion_adapter_messages_call(
    *,
    config: _aawm_adapter_config.AnthropicCompletionAdapterConfig,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    custom_llm_provider: Optional[str] = None,
    client_requested_stream: Optional[bool] = None,
    model_for_upstream: Optional[str] = None,
    stream_override: Optional[bool] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    operation_wrapper: Optional[Callable[[Callable[[], Awaitable[object]]], Awaitable[object]]] = None,
    fake_stream: bool = False,
    extra_handler_kwargs: Optional[Payload] = None,
    completion_stream_normalizer: Optional[Callable[[Any], Any]] = None,
) -> Response:
    """Shared completion-adapter messages handler + response branch (RR-054 #9)."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    stream_flag = (
        bool(prepared_request_body.get("stream")) if client_requested_stream is None else bool(client_requested_stream)
    )
    upstream_stream = stream_flag if stream_override is None else bool(stream_override)
    requested_model = prepared_request_body.get("model")
    model_name = model_for_upstream or (requested_model if isinstance(requested_model, str) else adapter_model)
    handler_extra_kwargs: dict[str, Any] = {
        "custom_llm_provider": custom_llm_provider or config.custom_llm_provider,
        "api_key": api_key,
        "api_base": api_base,
        "litellm_metadata": prepared_request_body.get("litellm_metadata") or {},
        "proxy_server_request": {
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
    }
    if timeout is not None:
        handler_extra_kwargs["timeout"] = timeout
    if max_retries is not None:
        handler_extra_kwargs["max_retries"] = max_retries
    if extra_handler_kwargs:
        handler_extra_kwargs.update(extra_handler_kwargs)

    handler_call_kwargs = _build_anthropic_completion_adapter_handler_call_kwargs(
        prepared_request_body=prepared_request_body,
        model_name=model_name,
        upstream_stream=upstream_stream,
    )

    # D1-521: prepare the exact final translated/clamped completion kwargs once
    # before access logging, pass them as provider_bound_body, and reuse them for
    # the upstream call. Keep prepared_request_body for request_body/model label.
    completion_kwargs, tool_name_mapping = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            **handler_call_kwargs,
            extra_kwargs=handler_extra_kwargs,
        )
    )
    _watermark_metadata = prepared_request_body.get("litellm_metadata")
    if not isinstance(_watermark_metadata, dict):
        _watermark_metadata = handler_extra_kwargs.get("litellm_metadata")
    if not isinstance(_watermark_metadata, dict):
        _watermark_metadata = {}
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body

    async def _operation() -> object:
        if upstream_stream and completion_stream_normalizer is not None:
            return await _perform_normalized_anthropic_completion_adapter_stream(
                handler=LiteLLMMessagesToCompletionTransformationHandler,
                handler_call_kwargs=handler_call_kwargs,
                handler_extra_kwargs=handler_extra_kwargs,
                completion_stream_normalizer=completion_stream_normalizer,
                completion_kwargs=completion_kwargs,
                tool_name_mapping=tool_name_mapping,
            )
        completion_response = await litellm.acompletion(**completion_kwargs)
        return LiteLLMMessagesToCompletionTransformationHandler._transform_completion_response(
            completion_response,
            model=handler_call_kwargs["model"],
            stream=upstream_stream,
            tool_name_mapping=tool_name_mapping,
        )

    litellm_metadata = prepared_request_body.get("litellm_metadata")
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata if isinstance(litellm_metadata, dict) else {})
    _annotate_request_scope_for_adapted_access_log(request, target_url)
    _emit_adapted_route_access_log(
        request=request,
        target_url=str(target_url),
        request_body=prepared_request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label=config.adapter_label,
        provider_bound_body=completion_kwargs,
    )
    _sa = _session_affinity_mod()
    # D1-612: last common pre-egress point for nested completion adapters
    # (acompletion path has no pass_through_request pre-send hook).
    await _ensure_anthropic_nested_session_owner_pre_egress(
        request=request,
        request_body=prepared_request_body
        if isinstance(prepared_request_body, dict)
        else None,
        provider=custom_llm_provider
        or getattr(config, "custom_llm_provider", None),
        model=model_name,
        route_family=(
            getattr(config, "route_family", None)
            or getattr(config, "adapter", None)
        ),
        endpoint_contract="anthropic_messages",
        state_format="anthropic",
        failure_phase="session_owner_anthropic_nested_pre_egress",
    )

    try:
        if operation_wrapper is not None:
            completion_response = await operation_wrapper(_operation)
        else:
            completion_response = await _operation()
        response = _finalize_anthropic_completion_adapter_response(
            completion_response=completion_response,
            stream_flag=stream_flag,
            fake_stream=fake_stream,
            rollup_kwargs=rollup_kwargs,
            adapter_label=config.adapter_label,
        )
    except Exception as _exc:
        # Release tokenized reservation on acompletion/stream failure.
        # Nested dispatch may also finalize; SessionOwnerLease is one-shot.
        await _sa.finalize_request_session_owner_lease(
            request,
            exc=_exc,
            failure_phase="session_owner_anthropic_completion_release",
        )
        raise
    # Authoritative success / first-byte stream object: promote held reservation.
    # Nested dispatch finalizer is a no-op once promoted/released.
    await _sa.finalize_request_session_owner_lease(
        request,
        response,
        failure_phase="session_owner_anthropic_completion_promote",
    )
    return response


# ---------------------------------------------------------------------------
# Route family metadata
# ---------------------------------------------------------------------------


def _add_route_family_logging_metadata(request_body: dict[str, Any], route_family: str) -> dict[str, Any]:
    normalized_route_family = _normalize_low_cardinality_tag_value(route_family)
    if not normalized_route_family:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=[f"route:{normalized_route_family}"],
        extra_fields={"passthrough_route_family": normalized_route_family},
    )


# ---------------------------------------------------------------------------
# install() -- rebind extracted functions to host globals for live lookup
# ---------------------------------------------------------------------------

_EXTRACTED_FUNCTION_NAMES: tuple[str, ...] = (
    "_decode_http_response_body",
    "_build_adapted_route_rollup_kwargs",
    "_emit_adapted_route_access_log",
    "_record_adapted_completed_route_rollup_turn",
    "_record_adapted_completed_route_rollup_after_stream",
    "_add_codex_native_tool_alias_adapter_metadata",
    "_normalize_openai_function_tool_parameters",
    "_sanitize_openai_object_schema_properties",
    "_normalize_openai_function_tool_schemas",
    "_get_openai_adapter_function_tool_names",
    "_apply_responses_adapter_parallel_instruction_policy",
    "_apply_openai_adapter_parallel_instruction_policy",
    "_apply_openrouter_adapter_parallel_instruction_policy",
    "_get_latest_adapter_user_prompt_text",
    "_prompt_explicitly_requests_bash_tool",
    "_maybe_force_explicit_bash_tool_choice_for_responses_adapter",
    "_apply_forced_bash_tool_choice_for_responses_adapter",
    "_maybe_force_explicit_bash_tool_choice_for_completion_adapter",
    "_responses_request_contains_mcp_tools",
    "_coerce_mapping_to_namespace",
    "_drop_anthropic_grok_native_prior_function_call_replay",
    "_build_anthropic_response_from_responses_response",
    "_build_completion_adapter_metadata",
    "_copy_translated_anthropic_adapter_response_headers",
    "_get_anthropic_adapter_access_log_target_label",
    "_annotate_request_scope_for_adapted_access_log",
    "_serialize_anthropic_adapter_response",
    "_build_anthropic_response_from_completion_adapter_response",
    "_get_anthropic_adapter_openai_target_base",
    "_get_anthropic_adapter_openrouter_api_key",
    "_get_anthropic_adapter_nvidia_api_key",
    "_get_anthropic_adapter_nvidia_target_base",
    "_get_anthropic_adapter_openrouter_target_base",
    "_resolve_anthropic_openai_responses_adapter_auth_context",
    "_build_anthropic_responses_adapter_request_body",
    "_prepare_anthropic_completion_adapter_request_body",
    "_apply_anthropic_responses_adapter_common_request_policies",
    "_apply_anthropic_responses_adapter_policies_from_config",
    "_finalize_anthropic_responses_adapter_upstream_response",
    "_finalize_anthropic_responses_adapter_from_config",
    "_perform_anthropic_responses_adapter_pass_through",
    "_perform_normalized_anthropic_completion_adapter_stream",
    "_is_anthropic_messages_response",
    "_finalize_anthropic_completion_adapter_response",
    "_build_anthropic_completion_adapter_handler_call_kwargs",
    "_session_affinity_mod",
    "_ensure_anthropic_nested_session_owner_pre_egress",
    "_perform_anthropic_completion_adapter_messages_call",
    "_add_route_family_logging_metadata",
)

_EXTRACTED_CONSTANT_NAMES: tuple[str, ...] = (
    "_AAWM_REQUEST_BODY_WALK_MAX_DEPTH",
    "_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS",
    "_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES",
    "_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT",
    "_OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS",
)


def install(host_globals: dict[str, Any]) -> None:
    """Rebind extracted functions to *host_globals* for live lookup.

    Each named function's ``__globals__`` is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    Constants are published as-is.
    """
    _mod = globals()
    for _name in _EXTRACTED_FUNCTION_NAMES:
        _obj = _mod[_name]
        if not isinstance(_obj, FunctionType):
            host_globals[_name] = _obj
            continue
        rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        rebound.__kwdefaults__ = _obj.__kwdefaults__
        rebound.__annotations__ = _obj.__annotations__
        rebound.__doc__ = _obj.__doc__
        rebound.__module__ = _obj.__module__
        rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = rebound
        host_globals[_name] = rebound
    for _name in _EXTRACTED_CONSTANT_NAMES:
        host_globals[_name] = _mod[_name]
    for _name, _value in (
        ("apply_request_watermark_egress", apply_request_watermark_egress),
        ("load_text_watermark_config", load_text_watermark_config),
        ("_get_runtime_text_watermark_config", _get_runtime_text_watermark_config),
        ("_watermark_endpoint_from_path", _watermark_endpoint_from_path),
    ):
        host_globals.setdefault(_name, _value)
