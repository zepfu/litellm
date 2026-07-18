"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import Request

import litellm


@dataclass(frozen=True)
class Runtime:
    _ChangeAccumulator: Any
    _annotate_google_code_assist_claude_tool_response_ids: Any
    _annotate_google_code_assist_duplicate_tool_responses: Any
    _apply_codex_google_code_assist_tool_contract_policy: Any
    _apply_google_adapter_completion_message_window: Any
    _apply_google_adapter_system_prompt_policy: Any
    _apply_google_code_assist_native_tool_aliases: Any
    _ensure_codex_google_code_assist_tool_results_have_calls: Any
    _inject_google_adapter_fallback_text_context: Any
    _inject_google_adapter_tool_call_context_text: Any
    _insert_google_code_assist_missing_claude_function_call_pairs: Any
    _normalize_codex_google_code_assist_anthropic_tool_replay: Any
    _normalize_codex_google_code_assist_reasoning_effort: Any
    _normalize_codex_openai_chat_kwargs_for_google_code_assist: Any
    _normalize_google_code_assist_httpx_payload: Any
    _normalize_google_code_assist_thinking_max_tokens: Any
    _normalize_google_completion_adapter_model_name: Any
    _repair_anthropic_tool_use_ids_for_passthrough: Any
    _repair_codex_google_code_assist_openai_tool_call_ids: Any
    _resolve_codex_google_code_assist_tool_call_scope_key: Any
    _resolve_google_adapter_session_id: Any
    _resolve_google_adapter_user_prompt_id: Any
    _sanitize_codex_google_code_assist_orphan_tool_results: Any
    _validate_anthropic_tool_blocks_for_passthrough: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _ChangeAccumulator=namespace["_ChangeAccumulator"],
        _annotate_google_code_assist_claude_tool_response_ids=namespace[
            "_annotate_google_code_assist_claude_tool_response_ids"
        ],
        _annotate_google_code_assist_duplicate_tool_responses=namespace[
            "_annotate_google_code_assist_duplicate_tool_responses"
        ],
        _apply_codex_google_code_assist_tool_contract_policy=namespace[
            "_apply_codex_google_code_assist_tool_contract_policy"
        ],
        _apply_google_adapter_completion_message_window=namespace["_apply_google_adapter_completion_message_window"],
        _apply_google_adapter_system_prompt_policy=namespace["_apply_google_adapter_system_prompt_policy"],
        _apply_google_code_assist_native_tool_aliases=namespace["_apply_google_code_assist_native_tool_aliases"],
        _ensure_codex_google_code_assist_tool_results_have_calls=namespace[
            "_ensure_codex_google_code_assist_tool_results_have_calls"
        ],
        _inject_google_adapter_fallback_text_context=namespace["_inject_google_adapter_fallback_text_context"],
        _inject_google_adapter_tool_call_context_text=namespace["_inject_google_adapter_tool_call_context_text"],
        _insert_google_code_assist_missing_claude_function_call_pairs=namespace[
            "_insert_google_code_assist_missing_claude_function_call_pairs"
        ],
        _normalize_codex_google_code_assist_anthropic_tool_replay=namespace[
            "_normalize_codex_google_code_assist_anthropic_tool_replay"
        ],
        _normalize_codex_google_code_assist_reasoning_effort=namespace[
            "_normalize_codex_google_code_assist_reasoning_effort"
        ],
        _normalize_codex_openai_chat_kwargs_for_google_code_assist=namespace[
            "_normalize_codex_openai_chat_kwargs_for_google_code_assist"
        ],
        _normalize_google_code_assist_httpx_payload=namespace["_normalize_google_code_assist_httpx_payload"],
        _normalize_google_code_assist_thinking_max_tokens=namespace[
            "_normalize_google_code_assist_thinking_max_tokens"
        ],
        _normalize_google_completion_adapter_model_name=namespace["_normalize_google_completion_adapter_model_name"],
        _repair_anthropic_tool_use_ids_for_passthrough=namespace["_repair_anthropic_tool_use_ids_for_passthrough"],
        _repair_codex_google_code_assist_openai_tool_call_ids=namespace[
            "_repair_codex_google_code_assist_openai_tool_call_ids"
        ],
        _resolve_codex_google_code_assist_tool_call_scope_key=namespace[
            "_resolve_codex_google_code_assist_tool_call_scope_key"
        ],
        _resolve_google_adapter_session_id=namespace["_resolve_google_adapter_session_id"],
        _resolve_google_adapter_user_prompt_id=namespace["_resolve_google_adapter_user_prompt_id"],
        _sanitize_codex_google_code_assist_orphan_tool_results=namespace[
            "_sanitize_codex_google_code_assist_orphan_tool_results"
        ],
        _validate_anthropic_tool_blocks_for_passthrough=namespace["_validate_anthropic_tool_blocks_for_passthrough"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


async def _build_google_code_assist_request_from_completion_kwargs(  # noqa: PLR0915
    *,
    completion_kwargs: dict[str, Any],
    adapter_model: str,
    project: str,
    request: Request,
    completion_kwargs_are_openai_chat: bool = False,
    scope_key: Optional[str] = None,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body

    google_model = _runtime()._normalize_google_completion_adapter_model_name(adapter_model)
    tool_call_scope_key = _runtime()._resolve_codex_google_code_assist_tool_call_scope_key(
        request=request,
        request_body=completion_kwargs,
        explicit_scope_key=scope_key,
    )
    if completion_kwargs_are_openai_chat:
        completion_kwargs = dict(completion_kwargs)
        (
            completion_kwargs,
            openai_chat_shape_changes,
        ) = _runtime()._normalize_codex_openai_chat_kwargs_for_google_code_assist(completion_kwargs)
        (
            completion_kwargs,
            codex_anthropic_tool_replay_changes,
        ) = _runtime()._normalize_codex_google_code_assist_anthropic_tool_replay(completion_kwargs)
        (
            completion_kwargs,
            codex_openai_tool_call_id_changes,
        ) = _runtime()._repair_codex_google_code_assist_openai_tool_call_ids(completion_kwargs)
        (
            completion_kwargs,
            codex_orphan_tool_result_changes,
        ) = _runtime()._sanitize_codex_google_code_assist_orphan_tool_results(
            completion_kwargs,
            scope_key=tool_call_scope_key,
        )
        (
            completion_kwargs,
            codex_tool_pair_changes,
        ) = _runtime()._ensure_codex_google_code_assist_tool_results_have_calls(
            completion_kwargs,
            scope_key=tool_call_scope_key,
        )
        tool_name_mapping: dict[str, str] = {}
        anthropic_native_tool_replay_changes: dict[str, Any] = {}
    else:
        from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        (
            completion_kwargs,
            anthropic_native_tool_use_id_repaired_count,
        ) = _runtime()._repair_anthropic_tool_use_ids_for_passthrough(completion_kwargs)
        _runtime()._validate_anthropic_tool_blocks_for_passthrough(completion_kwargs)
        anthropic_native_tool_replay_changes = {}
        if anthropic_native_tool_use_id_repaired_count:
            anthropic_native_tool_replay_changes[
                "google_adapter_repaired_anthropic_native_tool_use_id_count"
            ] = anthropic_native_tool_use_id_repaired_count

        (
            completion_kwargs,
            tool_name_mapping,
        ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=completion_kwargs["max_tokens"],
            messages=completion_kwargs.get("messages") or [],
            model=google_model,
            metadata=completion_kwargs.get("metadata"),
            stop_sequences=completion_kwargs.get("stop_sequences"),
            stream=completion_kwargs.get("stream"),
            system=completion_kwargs.get("system"),
            temperature=completion_kwargs.get("temperature"),
            thinking=completion_kwargs.get("thinking"),
            tool_choice=completion_kwargs.get("tool_choice"),
            tools=completion_kwargs.get("tools"),
            top_k=completion_kwargs.get("top_k"),
            top_p=completion_kwargs.get("top_p"),
            output_format=completion_kwargs.get("output_format"),
            output_config=completion_kwargs.get("output_config"),
            extra_kwargs={
                "custom_llm_provider": litellm.LlmProviders.GEMINI.value,
                "metadata": completion_kwargs.get("metadata"),
                "parallel_tool_calls": completion_kwargs.get("parallel_tool_calls"),
                "response_format": completion_kwargs.get("response_format"),
                "reasoning_effort": completion_kwargs.get("reasoning_effort"),
                "frequency_penalty": completion_kwargs.get("frequency_penalty"),
                "presence_penalty": completion_kwargs.get("presence_penalty"),
                "seed": completion_kwargs.get("seed"),
                "n": completion_kwargs.get("n"),
            },
        )
        openai_chat_shape_changes = {}
        codex_tool_pair_changes = {}
        codex_orphan_tool_result_changes = {}
        codex_anthropic_tool_replay_changes = {}
        codex_openai_tool_call_id_changes = {}
    if tool_call_scope_key is not None:
        tool_name_mapping["__aawm_scope_key__"] = tool_call_scope_key
    (
        completion_kwargs,
        thinking_max_tokens_changes,
    ) = _runtime()._normalize_google_code_assist_thinking_max_tokens(completion_kwargs)
    (
        completion_kwargs,
        system_prompt_policy_changes,
    ) = _runtime()._apply_google_adapter_system_prompt_policy(completion_kwargs)
    codex_tool_contract_policy_changes: dict[str, Any] = {}
    if completion_kwargs_are_openai_chat:
        (
            completion_kwargs,
            codex_tool_contract_policy_changes,
        ) = _runtime()._apply_codex_google_code_assist_tool_contract_policy(completion_kwargs)
    completion_messages = list(completion_kwargs.get("messages") or [])
    (
        completion_messages,
        completion_message_window_changes,
    ) = _runtime()._apply_google_adapter_completion_message_window(completion_messages)
    (
        completion_messages,
        tool_call_context_changes,
    ) = _runtime()._inject_google_adapter_tool_call_context_text(completion_messages)
    if tool_call_context_changes:
        completion_message_window_changes = {
            **completion_message_window_changes,
            **tool_call_context_changes,
        }
    completion_kwargs["messages"] = completion_messages
    (
        completion_kwargs,
        native_tool_alias_changes,
    ) = _runtime()._apply_google_code_assist_native_tool_aliases(
        completion_kwargs,
        tool_name_mapping,
    )
    completion_messages = list(completion_kwargs.get("messages") or [])

    mappable_params = {
        key: value
        for key, value in completion_kwargs.items()
        if key
        not in {
            "model",
            "messages",
            "metadata",
            "stream",
            "stream_options",
            "litellm_logging_obj",
            "custom_llm_provider",
            "api_key",
            "api_base",
            "user",
        }
        and value is not None
    }
    (
        mappable_params,
        reasoning_effort_policy_changes,
    ) = _runtime()._normalize_codex_google_code_assist_reasoning_effort(mappable_params)
    gemini_optional_params = litellm.GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params=mappable_params,
        optional_params={},
        model=google_model,
        drop_params=False,
    )
    litellm_params: dict[str, Any] = {}
    metadata = completion_kwargs.get("metadata")
    if isinstance(metadata, dict):
        litellm_params["metadata"] = metadata

    google_request = _transform_request_body(
        messages=completion_messages,
        model=google_model,
        optional_params=gemini_optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )
    google_request_dict = _runtime()._normalize_google_code_assist_httpx_payload(dict(google_request))
    claude_tool_response_id_changes = _runtime()._annotate_google_code_assist_claude_tool_response_ids(
        google_request_dict,
        completion_messages,
        google_model=google_model,
    )
    claude_tool_pair_changes = _runtime()._insert_google_code_assist_missing_claude_function_call_pairs(
        google_request_dict,
        google_model=google_model,
        scope_key=tool_call_scope_key,
    )
    duplicate_tool_response_changes = _runtime()._annotate_google_code_assist_duplicate_tool_responses(
        google_request_dict,
        completion_messages,
    )
    fallback_context_changes = _runtime()._inject_google_adapter_fallback_text_context(
        google_request_dict,
        completion_messages,
    )
    system_instruction = google_request_dict.pop("system_instruction", None)
    if system_instruction is not None:
        google_request_dict["systemInstruction"] = system_instruction
    session_id, session_id_source = _runtime()._resolve_google_adapter_session_id(
        request,
        completion_messages,
        google_model=google_model,
    )
    google_request_dict["session_id"] = session_id
    user_prompt_id = _runtime()._resolve_google_adapter_user_prompt_id(
        request,
        completion_messages,
        google_model=google_model,
        session_id=session_id,
    )

    wrapped_request = {
        "model": google_model,
        "project": project,
        "user_prompt_id": user_prompt_id,
        "request": google_request_dict,
    }
    session_id_hash = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8]
    if isinstance(metadata, dict) and metadata:
        wrapped_request["litellm_metadata"] = dict(metadata)
    litellm_metadata = wrapped_request.setdefault("litellm_metadata", {})
    litellm_metadata.setdefault("session_id", session_id)
    litellm_metadata["google_adapter_session_id"] = session_id
    litellm_metadata["google_adapter_session_id_source"] = session_id_source
    litellm_metadata["google_adapter_session_id_hash"] = session_id_hash
    # RR-054 #8: accumulate transform metadata via one tracker to avoid silent
    # merge-order overwrites / forgotten steps.
    change_acc = _runtime()._ChangeAccumulator()
    change_acc.record("completion_message_window", completion_message_window_changes)
    change_acc.record("fallback_context", fallback_context_changes)
    change_acc.record("openai_chat_shape", openai_chat_shape_changes)
    change_acc.record("codex_anthropic_tool_replay", codex_anthropic_tool_replay_changes)
    change_acc.record("codex_openai_tool_call_id", codex_openai_tool_call_id_changes)
    change_acc.record("codex_orphan_tool_result", codex_orphan_tool_result_changes)
    change_acc.record("codex_tool_pair", codex_tool_pair_changes)
    change_acc.record("anthropic_native_tool_replay", anthropic_native_tool_replay_changes)
    change_acc.record("reasoning_effort_policy", reasoning_effort_policy_changes)
    change_acc.record("thinking_max_tokens", thinking_max_tokens_changes)
    change_acc.record("native_tool_alias", native_tool_alias_changes)
    change_acc.record("claude_tool_response_id", claude_tool_response_id_changes)
    change_acc.record("claude_tool_pair", claude_tool_pair_changes)
    change_acc.record("duplicate_tool_response", duplicate_tool_response_changes)
    change_acc.record("system_prompt_policy", system_prompt_policy_changes)
    change_acc.record("codex_tool_contract_policy", codex_tool_contract_policy_changes)
    completion_message_window_changes = change_acc.as_dict()
    completion_message_window_changes["google_adapter_session_id_source"] = session_id_source
    completion_message_window_changes["google_adapter_session_id_hash"] = session_id_hash
    return (
        wrapped_request,
        tool_name_mapping,
        completion_messages,
        gemini_optional_params,
        litellm_params,
        completion_message_window_changes,
    )
