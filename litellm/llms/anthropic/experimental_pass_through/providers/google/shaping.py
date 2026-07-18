"""Compatibility facade for provider-owned Google shaping modules."""

from __future__ import annotations

import sys
from collections.abc import Mapping

from . import content_selection as _content_selection
from . import content_compaction as _content_compaction
from . import error_and_schema as _error_and_schema
from . import schema_and_prompt as _schema_and_prompt
from . import anthropic_replay as _anthropic_replay
from . import tool_pairing as _tool_pairing
from . import request_assembly as _request_assembly
from . import request_building as _request_building
from . import tool_aliasing as _tool_aliasing
from . import response_translation as _response_translation
from . import response_streaming as _response_streaming
from . import request_preparation as _request_preparation
from . import persisted_output as _persisted_output


_EXPORTS_BY_MODULE = (
    (_content_selection, (
        "_normalize_anthropic_google_completion_adapter_model_name",
        "_normalize_codex_google_code_assist_adapter_model_name",
        "_extract_google_adapter_agent_name_from_completion_messages",
        "_extract_google_adapter_latest_user_prompt_text",
        "_extract_google_adapter_latest_tool_result_fingerprint",
        "_estimate_google_content_text_chars",
        "_selected_google_contents_have_paired_function_responses",
        "_selected_google_contents_have_complete_function_exchanges",
        "_add_required_google_function_call_pair_indices",
        "_trim_google_content_indices_to_window",
        "_trim_google_adapter_followup_tools",
        "_is_google_function_call_allowed_predecessor",
        "_merge_google_model_content_parts",
        "_repair_google_adapter_function_call_turn_adjacency",
    )),
    (_content_compaction, (
        "_split_google_adapter_inline_context_and_prompt",
        "_compact_google_adapter_oversized_text_part",
        "_compact_google_adapter_oversized_text_parts",
        "_apply_google_adapter_contents_window_policy",
        "_apply_google_adapter_generation_config_policy",
        "_apply_google_adapter_request_shape_policy",
    )),
    (_error_and_schema, (
        "_extract_google_adapter_exception_status_code",
        "_extract_google_adapter_exception_detail",
        "_extract_google_adapter_error_payloads",
        "_extract_google_adapter_error_reason",
        "_extract_google_adapter_error_payload_for_logging",
        "_is_google_adapter_transient_retryable_failure",
        "_build_google_adapter_terminal_error_log_context",
        "_normalize_google_completion_adapter_model_name",
        "_sanitize_google_schema_array_items",
        "_merge_google_code_assist_schema_annotations",
        "_sanitize_google_code_assist_union_schemas",
        "_sanitize_google_code_assist_tool_schema",
    )),
    (_schema_and_prompt, (
        "_is_google_adapter_synthetic_tool_context_text",
        "_is_google_adapter_synthetic_tool_context_message",
        "_inject_google_adapter_fallback_text_context",
        "_extract_google_adapter_system_text_from_content",
        "_replace_google_adapter_system_message_text",
        "_append_codex_google_code_assist_tool_contract_to_system_text",
        "_apply_codex_google_code_assist_tool_contract_policy",
        "_is_google_adapter_claude_overhead_block",
        "_strip_google_adapter_claude_system_overhead",
        "_build_google_adapter_system_prompt_policy_text",
        "_apply_google_adapter_system_prompt_policy",
    )),
    (_anthropic_replay, (
        "_normalize_codex_openai_chat_kwargs_for_google_code_assist",
        "_has_codex_google_code_assist_anthropic_tool_replay_blocks",
        "_normalize_codex_google_code_assist_anthropic_assistant_message",
        "_normalize_codex_google_code_assist_anthropic_user_message",
        "_build_codex_google_code_assist_anthropic_replay_changes",
        "_normalize_codex_google_code_assist_anthropic_tool_replay",
        "_deterministic_codex_google_code_assist_tool_call_id",
        "_next_codex_google_code_assist_tool_messages",
        "_paired_codex_google_code_assist_tool_message",
        "_repair_codex_google_code_assist_tool_call_id",
        "_repair_codex_google_code_assist_openai_tool_call_ids",
    )),
    (_tool_pairing, (
        "_normalize_codex_google_code_assist_reasoning_effort",
        "_normalize_google_code_assist_thinking_max_tokens",
        "_normalize_codex_google_code_assist_tool_call_arguments",
        "_infer_single_codex_google_code_assist_function_tool_name",
        "_is_codex_google_code_assist_empty_text_content",
        "_previous_codex_google_code_assist_assistant_index",
        "_previous_codex_google_code_assist_contiguous_assistant_index",
        "_previous_codex_google_code_assist_tool_call",
        "_build_codex_google_code_assist_synthetic_tool_call",
        "_append_codex_google_code_assist_tool_call_to_assistant",
        "_build_codex_google_code_assist_tool_pair_repair_changes",
        "_append_codex_google_code_assist_orphan_tool_result_context",
        "_sanitize_codex_google_code_assist_orphan_tool_results",
        "_ensure_codex_google_code_assist_tool_results_have_calls",
    )),
    (_request_assembly, (
        "_build_google_code_assist_request_from_completion_kwargs",
    )),
    (_request_building, (
        "_drop_codex_google_code_assist_non_function_tools",
        "_build_codex_google_code_assist_completion_kwargs",
        "_prepare_codex_google_code_assist_adapter_request",
    )),
    (_tool_aliasing, (
        "_apply_google_code_assist_alias_to_function_block",
        "_apply_google_code_assist_alias_to_tool",
        "_apply_google_code_assist_aliases_to_tool_calls",
        "_apply_google_code_assist_aliases_to_message",
        "_apply_google_code_assist_native_tool_aliases",
        "_inject_google_adapter_tool_call_context_text",
        "_extract_google_adapter_preserved_task_excerpt",
        "_build_google_adapter_preserved_task_state_message",
        "_apply_google_adapter_completion_message_window",
        "_normalize_google_code_assist_httpx_payload",
        "_annotate_google_code_assist_duplicate_tool_response_parts",
        "_annotate_google_code_assist_duplicate_tool_responses",
        "_annotate_google_code_assist_claude_tool_response_ids",
        "_insert_google_code_assist_missing_claude_function_call_pairs",
    )),
    (_response_translation, (
        "_extract_google_code_assist_text_metrics",
        "_summarize_google_code_assist_content_preview_entry",
        "_summarize_google_code_assist_request_contents_shape",
        "_summarize_google_code_assist_generation_config_shape",
        "_extract_google_code_assist_function_names",
        "_summarize_google_code_assist_request_shape",
        "_unwrap_google_code_assist_response_payload",
        "_translate_google_code_assist_response_to_anthropic",
    )),
    (_response_streaming, (
        "_iterate_google_code_assist_unwrapped_stream",
        "_build_anthropic_streaming_response_from_google_code_assist_stream",
        "_restore_google_adapter_tool_call_names",
        "_restore_google_adapter_tool_call_names_stream",
        "_collect_google_code_assist_model_response_from_stream",
        "_collect_google_code_assist_response_from_stream",
        "_build_codex_streaming_response_from_google_code_assist_stream",
        "_build_google_debug_header_summary",
        "_build_google_adapter_native_headers",
        "_is_codex_google_code_assist_empty_success_model_response",
        "_sanitize_google_code_assist_request_schemas",
    )),
    (_request_preparation, (
        "_prepare_anthropic_google_completion_adapter_request",
    )),
    (_persisted_output, (
        "_compact_google_adapter_persisted_output_preview_and_expanded_text",
        "_compact_expanded_claude_persisted_output_text_for_google_adapter",
        "_compact_google_adapter_text_part_sequence",
        "_compact_google_adapter_followup_request_contents",
        "_compact_google_adapter_persisted_output_value",
        "_compact_google_adapter_persisted_output_in_anthropic_request_body",
    )),
)
_IMPLEMENTATION_MODULES = tuple(
    implementation_module
    for implementation_module, _ in _EXPORTS_BY_MODULE
)
_OWNED_FUNCTION_NAMES = tuple(
    name
    for _, owned_names in _EXPORTS_BY_MODULE
    for name in owned_names
)
_FACADE_MODULE = sys.modules[__name__]

for _implementation_module, _owned_names in _EXPORTS_BY_MODULE:
    for _owned_name in _owned_names:
        setattr(
            _FACADE_MODULE,
            _owned_name,
            getattr(_implementation_module, _owned_name),
        )


def bind_runtime(namespace: Mapping[str, object]) -> None:
    provider_namespace = dict(namespace)
    provider_namespace.update(
        {
            name: getattr(_FACADE_MODULE, name)
            for name in _OWNED_FUNCTION_NAMES
        }
    )
    for implementation_module in _IMPLEMENTATION_MODULES:
        implementation_module.bind_runtime(provider_namespace)


__all__ = ["bind_runtime", *_OWNED_FUNCTION_NAMES]
