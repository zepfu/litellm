"""Wave A2 (identity leaf extractions) ownership/parity tests.

`litellm/integrations/aawm_agent_identity/__init__.py` is CURRENTLY the
package form landed by Wave A1 (verbatim `git mv` of the single-file module).
Wave A2's engineer will move bounded, behavior-preserving leaf bands OUT of
`__init__.py` into new sibling submodules:

  - `constants.py`         <- env tables/regexes/field tuples
  - `coerce.py`             <- generic coercers + DSN build
  - `cost_map.py`           <- bundled model-cost-map helpers
  - `identity_tenant_agent.py` <- tenant/agent-id extraction
  - `identity_repository.py`   <- repo patterns/extractors/memory-workflow
  - `identity_runtime.py`      <- versions/user-agent/client-IP/host
  - `agent_context.py`         <- "You are '<agent>'" paths + codex/grok
                                  predicates + trace promotion

with façade rebinds left behind in `__init__.py` (`name = new_module.name`)
so every existing importer / monkeypatch target keeps working.

These tests are written BEFORE the move (TDD extraction, per
`.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md`
### Wave A2). They are EXPECTED to be RED until the engineer performs the
move: the ownership/facade-identity/rebind-order tests can only pass once
the target submodules exist and `__init__.py` no longer defines the moved
names directly. Do not weaken these assertions to make them pass early --
red-before-green is the intended TDD signal for this wave.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import builtins
import importlib
import json
import inspect
import sys
import typing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_PKG_DIR = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity"
INIT_PATH = IDENTITY_PKG_DIR / "__init__.py"

# Representative, defensible sampled set per target module (>= 5 names for the
# modules that have >= 5 candidate helpers in the plan's line-range map;
# `cost_map.py` has exactly 4 helpers in the current source, so all 4 are
# used for that module).
_MOVED_NAMES_BY_MODULE: Dict[str, List[str]] = {
    "constants": [
        "_AAWM_LITELLM_ENVIRONMENT_ENV_VARS",
        "_AAWM_LITELLM_VERSION_ENV_VARS",
        "_AAWM_ASSOCIATED_VERSION_ENV_VARS",
        "_AAWM_AGENT_ID_UUID_RE",
        "_AAWM_AGENT_ID_HEX_RE",
        "_AAWM_AGENT_ID_PREFIXED_RE",
        "_AAWM_SESSION_HISTORY_METADATA_KEYS",
        "_AAWM_TENANT_ID_METADATA_KEYS",
        "_AAWM_REPOSITORY_METADATA_KEYS",
    ],
    "coerce": [
        "_clean_secret_string",
        "_get_first_secret_value",
        "_normalize_aawm_sslmode",
        "_build_aawm_dsn",
        "_append_aawm_dsn_query_params",
        "_clean_non_empty_string",
        "_first_non_empty_string",
        "_coerce_string_dict",
    ],
    "cost_map": [
        "_load_bundled_model_cost_map",
        "_bundled_model_cost_casefold_lookup",
        "_lookup_bundled_model_cost_info",
        "_calculate_response_cost_from_bundled_model_cost_map",
    ],
    "identity_tenant_agent": [
        "_extract_claude_trace_agent_name",
        "_extract_claude_trace_user_identity_from_metadata_sources",
        "_extract_tenant_identity_from_kwargs",
        "_extract_tenant_identity_from_langfuse_trace_observation",
        "_is_agent_id_like",
        "_normalize_agent_id_identity",
        "_extract_agent_id_from_metadata_sources",
        "_extract_agent_id_from_kwargs",
        "_extract_agent_id_from_langfuse_trace_observation",
    ],
    "identity_repository": [
        "_normalize_repository_identity",
        "_normalize_repository_identity_from_absolute_path",
        "_extract_repository_identity_from_text",
        "_extract_repository_identity_from_value",
        "_extract_repository_identity_from_metadata_sources",
        "_extract_repository_identity_from_kwargs",
        "_extract_repository_identity_from_langfuse_trace_observation",
        "_is_codex_memory_workflow_request",
        "_apply_codex_memory_workflow_repository",
    ],
    "identity_runtime": [
        "_parse_client_identity_from_user_agent",
        "_extract_claude_code_version_from_metadata",
        "_clean_session_history_client_ip_candidate",
        "_canonical_session_history_client_ip",
    ],
    "agent_context": [
        "_extract_agent_name",
        "_is_native_codex_passthrough_context",
        "_is_codex_client_identity",
        "_is_codex_default_agent_context",
        "_is_codex_subagent_context",
        "_is_native_grok_passthrough_context",
        "_promote_grok_repository_trace_identity",
        "_promote_codex_repository_trace_user_id",
    ],
}

# Flat sample used by the AST-ownership / facade-identity tests.
_ALL_MOVED_NAMES: List[str] = [name for names in _MOVED_NAMES_BY_MODULE.values() for name in names]


def _parse_init_module() -> ast.Module:
    source = INIT_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(INIT_PATH))


def _function_def_names(tree: ast.Module) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _module_level_assign_targets(tree: ast.Module) -> Dict[str, ast.AST]:
    """Map assigned name -> the AST node of its (first) module-level Assign."""
    targets: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in targets:
                    targets[target.id] = node
    return targets


def test_moved_names_not_defined_in_init() -> None:
    """None of the A2-moved helper names may remain as `FunctionDef` in `__init__.py`.

    They may still appear as facade *assignments* (`name = submodule.name`),
    just not as function/async-function definitions. This is RED until the
    engineer performs the extraction.
    """
    tree = _parse_init_module()
    defined_functions = _function_def_names(tree)

    still_defined = sorted(
        name
        for name in _ALL_MOVED_NAMES
        if name in defined_functions
        # constants are never FunctionDefs to begin with; only check names
        # that are actually helper functions in the pre-move source.
        and name not in _MOVED_NAMES_BY_MODULE["constants"]
    )
    assert not still_defined, (
        "expected these A2-moved helper functions to no longer be defined "
        f"directly in __init__.py (they should live in their target "
        f"submodule with only a facade assignment remaining): {still_defined}"
    )


def test_facade_identity() -> None:
    """Sampled facade identity check per target module.

    `getattr(pkg, name) is getattr(submodule, name)` for every A2-moved name,
    once the target submodules exist. Import failures (submodule doesn't
    exist yet) are surfaced as an assertion failure naming the missing
    module, not a bare collection error, so the RED signal is legible.
    """
    import litellm.integrations.aawm_agent_identity as identity_pkg

    missing_modules = []
    mismatched = []
    for module_name, names in _MOVED_NAMES_BY_MODULE.items():
        try:
            submodule = importlib.import_module(f"litellm.integrations.aawm_agent_identity.{module_name}")
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        for name in names:
            pkg_value = getattr(identity_pkg, name, None)
            sub_value = getattr(submodule, name, None)
            if pkg_value is None or sub_value is None or pkg_value is not sub_value:
                mismatched.append((module_name, name))

    assert not missing_modules, f"expected these A2 target submodules to exist: {missing_modules}"
    assert not mismatched, (
        "expected `getattr(pkg, name) is getattr(submodule, name)` for each "
        f"facade-bound name; mismatches (module, name): {mismatched}"
    )


def test_rebind_order_facades_before_record_install() -> None:
    """Every facade assignment for a moved helper precedes the
    `_bind_session_history_record_apis()` call in `__init__.py`'s AST.

    This preserves the `__globals__`-rebinding contract: record APIs are
    installed via `_bind_session_history_record_apis()`, and any free-name
    helper they reference by module-global lookup must already be bound to
    its facade value (the submodule's live function object) by that point.
    """
    tree = _parse_init_module()

    call_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_bind_session_history_record_apis"
        ):
            call_line = node.lineno
            break

    assert call_line is not None, "expected a call to _bind_session_history_record_apis() in __init__.py"

    # Restrict to moved *function* names (exclude the `constants` module's
    # names): constants are already plain module-level Assign nodes before
    # the move, which would make this assertion vacuously true pre-move.
    # Function facades (`name = submodule.name`) only exist as Assign nodes
    # AFTER the engineer performs the extraction, so this is the real RED
    # gate for this test.
    defined_functions = _function_def_names(tree)
    moved_function_names = [
        name for module_name, names in _MOVED_NAMES_BY_MODULE.items() if module_name != "constants" for name in names
    ]

    assign_targets = _module_level_assign_targets(tree)
    facade_names_present = [
        name for name in moved_function_names if name in assign_targets and name not in defined_functions
    ]

    assert facade_names_present, (
        "expected at least some A2-moved helper functions to appear as "
        "module-level facade assignments (`name = submodule.name`, with no "
        "remaining FunctionDef of the same name) in __init__.py once the "
        "extraction has landed (none found -- this is the expected RED "
        "state before the move)"
    )

    late_facades = [name for name in facade_names_present if assign_targets[name].lineno >= call_line]
    assert not late_facades, (
        "expected every facade assignment for a moved helper to precede the "
        f"_bind_session_history_record_apis() call (line {call_line}); "
        f"these facades assign at or after that call: {late_facades}"
    )


def test_moved_submodules_do_not_import_init_at_module_scope() -> None:
    """New leaf submodules must not import the `__init__` package at module
    scope (would risk a circular import); they should use the lazy
    `_get_litellm_module()`-style pattern if they need `litellm` itself.
    """
    missing_modules = []
    offending = []
    for module_name in _MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        if not module_path.is_file():
            missing_modules.append(module_name)
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "litellm.integrations.aawm_agent_identity":
                offending.append(module_name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "litellm.integrations.aawm_agent_identity":
                        offending.append(module_name)

    assert not missing_modules, f"expected these A2 target submodule files to exist: {missing_modules}"
    assert not offending, f"these submodules import the identity __init__ package at module scope: {offending}"


# =========================================================================
# Wave A4A: usage_extract / provider_normalize / request_signals /
#           prompt_overhead ownership inventory
#
# Bands are the ORIGINAL single-file line ranges from
# .analysis/aawm-agent-identity-and-oversized-units-decomposition-2026-07-23.md
# (table at lines 68-126) and the plan A4 table. The A4A description is
# "Metadata extraction/normalization WITHOUT trust/storage code", so every
# enrichment / trust / storage / backfill / context-window / claude-review /
# gemini-signature symbol is explicitly EXCLUDED here (those are A4B/A4C/A4D).
# =========================================================================

_A4A_MOVED_NAMES_BY_MODULE: Dict[str, List[str]] = {
    # Original bands 7759-8348 (usage objects from all sources + token field
    # extractors) and 10094-10267 (reasoning tokens + rerank payloads).
    # EXCLUDES gemini thought-signature helpers (A4B enrich), and spend-log/
    # langfuse BACKFILL synthesis (A4D backfill.py).
    "usage_extract": [
        "_build_usage_object_from_metadata",
        "_build_usage_object_from_token_count_payload",
        "_extract_responses_completed_response_from_langfuse_output",
        "_build_usage_object_from_langfuse_output",
        "_extract_codex_model_from_response_headers",
        "_session_history_metadata_model",
        "_session_history_model_from_request_tags",
        "_extract_model_from_langfuse_input",
        "_extract_model_from_langfuse_output",
        "_first_known_model_string",
        "_first_explicit_openrouter_model_string",
        "_coerce_usage_object_to_dict",
        "_extract_metadata_usage_object",
        "_merge_usage_object_with_metadata",
        "_extract_usage_object",
        "_enrich_token_count_usage_metadata",
        "_extract_prompt_tokens",
        "_extract_completion_tokens",
        "_extract_total_tokens",
        "_extract_prompt_tokens_details",
        "_extract_completion_tokens_details",
        "_extract_cache_read_input_tokens",
        "_extract_cache_creation_input_tokens",
        "_has_nested_path",
        # 10094-10267: reasoning tokens + rerank payloads
        "_extract_reported_reasoning_tokens",
        "_fallback_gemini_reasoning_tokens_from_signatures",
        "_determine_reasoning_tokens_source",
        "_estimate_reasoning_tokens",
        "_extract_rerank_request_payload",
        "_coerce_rerank_text",
        "_extract_rerank_document_text",
        # constants owned by this band
        "_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE",
    ],
    # Original bands 8349-8742 (provider/model/route-family/cache-family/
    # prompt-caching normalization) and 14797-15306 (api-base sanitize/local
    # detection, model-group + inbound alias resolve, local embedding/llm/
    # biomed route metadata, session-history model resolve + xai override).
    # EXCLUDES worker-context-exhaustion + anthropic 1M context-window
    # (A4B enrich / A4C context_window) and rate-limit storage mappers (A4D).
    "provider_normalize": [
        # 8349-8742
        "_normalize_session_history_provider_name",
        "_session_history_provider_from_model_catalog",
        "_session_history_provider_from_model",
        "_session_history_provider_from_route_family",
        "_session_history_adapter_target_provider",
        "_session_history_auto_agent_selected_provider",
        "_session_history_adapter_model",
        "_normalize_session_history_provider",
        # 14797-15306
        "_sanitize_session_history_api_base",
        "_is_local_session_history_api_base",
        "_extract_session_history_api_base",
        "_get_session_history_model_group",
        "_resolve_inbound_model_alias",
        "_resolve_inbound_model_alias_from_langfuse",
        "_normalize_session_history_model_group",
        "_is_completion_call_type",
        "_is_embedding_call_type",
        "_strip_local_provider_model_prefix",
        "_session_history_provider_from_api_base",
        "_apply_local_embedding_route_metadata",
        "_apply_local_llm_route_metadata",
        "_resolve_local_biomed_session_history_route",
        "_apply_local_biomed_route_metadata",
        "_resolve_session_history_model",
        "_resolve_xai_grok_model_override",
        # constants
        "_LOCAL_BIOMED_SESSION_HISTORY_ROUTES",
    ],
    # Original band 8743-9539: invalid-tool-call detection, structured-output
    # detection/classification, request-payload scanning (cache hints), and
    # compact-summary classification.
    # EXCLUDES claude auto-review parent identity (A4B claude_review.py) and
    # rate-limit storage / tool-definition DB payload / previous-gap (A4D
    # storage_fields.py).
    "request_signals": [
        # invalid-tool-call detection
        "_invalid_tool_call_error_text_seen",
        "_iter_tool_result_error_candidates",
        "_iter_request_message_payloads",
        "_extract_invalid_tool_call_count_from_request_body",
        # structured-output detection/classification
        "_empty_structured_output_state",
        "_merge_structured_output_state",
        "_structured_output_schema_hash",
        "_structured_output_state_from_format",
        "_structured_output_state_from_generation_config",
        "_detect_structured_output_request",
        "_collect_structured_output_failure_texts",
        "_classify_structured_output_failure",
        # request-payload scanning (cache hints)
        "_extract_request_body_from_langfuse_input",
        "_request_payload_contains",
        # compact-summary classification
        "_append_request_content_text",
        "_extract_request_user_texts",
        "_join_compact_request_user_texts",
        "_extract_codex_compact_thread_id",
        "_extract_gemini_compact_prompt_id",
        "_base_gemini_compact_prompt_id",
        "_extract_compact_output_text",
        "_is_claude_code_compact_context",
        "_is_codex_compact_context",
        "_is_gemini_cli_compact_context",
        "_classify_compact_summary_state",
        # constants
        "_INVALID_TOOL_CALL_ERROR_RE",
        "_TOOL_RESULT_ERROR_BLOCK_TYPES",
        "_STRUCTURED_OUTPUT_JSON_MODE_VALUES",
        "_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS",
        "_STRUCTURED_OUTPUT_FAILURE_PATTERNS",
        "_CODEX_THREAD_ID_RE",
        "_GEMINI_COMPACT_PROMPT_ID_RE",
        "_CLAUDE_CODE_COMPACT_REQUEST_MARKERS",
    ],
    # Original band 10268-11019: prompt-overhead buckets/components/breakdown
    # plus rerank token estimate/merge.
    # EXCLUDES reasoning-token + rerank-payload parsing (usage_extract) and
    # compact-summary classification (request_signals).
    "prompt_overhead": [
        "_fallback_text_token_estimate",
        "_empty_prompt_overhead_breakdown",
        "_serialize_prompt_overhead_component",
        "_estimate_prompt_overhead_tokens",
        "_extract_prompt_text_blocks",
        "_classify_system_prompt_block",
        "_estimate_system_prompt_bucket_tokens",
        "_append_prompt_component",
        "_append_prompt_text_components",
        "_extract_responses_visible_text_blocks",
        "_responses_message_component_path",
        "_record_responses_excluded_fields",
        "_append_openai_responses_input_component",
        "_append_openai_responses_input_components",
        "_split_chat_prompt_messages",
        "_extract_prompt_overhead_components",
        "_build_prompt_overhead_breakdown",
        "_estimate_rerank_request_tokens",
        "_usage_has_positive_tokens",
        "_merge_estimated_rerank_tokens_into_usage",
        "_positive_int_or_none",
        # constants
        "_RESPONSES_SYSTEM_ROLES",
        "_RESPONSES_CONVERSATION_ROLES",
        "_RESPONSES_TEXT_CONTENT_TYPES",
        "_RESPONSES_OPAQUE_CONTENT_TYPES",
        "_RESPONSES_OPAQUE_ITEM_TYPES",
    ],
}

_A4A_ALL_MOVED_NAMES: List[str] = [
    name for names in _A4A_MOVED_NAMES_BY_MODULE.values() for name in names
]

# Names that are module-level constants (Assign), not FunctionDefs.
_A4A_CONSTANT_NAMES = frozenset(
    {
        "_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE",
        "_LOCAL_BIOMED_SESSION_HISTORY_ROUTES",
        "_INVALID_TOOL_CALL_ERROR_RE",
        "_TOOL_RESULT_ERROR_BLOCK_TYPES",
        "_STRUCTURED_OUTPUT_JSON_MODE_VALUES",
        "_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS",
        "_STRUCTURED_OUTPUT_FAILURE_PATTERNS",
        "_CODEX_THREAD_ID_RE",
        "_GEMINI_COMPACT_PROMPT_ID_RE",
        "_CLAUDE_CODE_COMPACT_REQUEST_MARKERS",
        "_RESPONSES_SYSTEM_ROLES",
        "_RESPONSES_CONVERSATION_ROLES",
        "_RESPONSES_TEXT_CONTENT_TYPES",
        "_RESPONSES_OPAQUE_CONTENT_TYPES",
        "_RESPONSES_OPAQUE_ITEM_TYPES",
    }
)

_A4A_FUNCTION_NAMES: List[str] = [
    name for name in _A4A_ALL_MOVED_NAMES if name not in _A4A_CONSTANT_NAMES
]

# Symbols that belong to OTHER subwaves and must NEVER appear in the A4A
# inventory. Used by the boundary-guard tests below.
_A4B_SYMBOLS = frozenset(
    {
        # enrich.py (worker-context-exhaustion + anthropic 1M ctx + gemini sig)
        "_bound_worker_context_exhaustion_string",
        "_normalize_worker_context_exhaustion_bool",
        "_sanitize_worker_context_exhaustion_metadata",
        "_promote_worker_context_exhaustion_metadata",
        "_is_anthropic_session_history_context",
        "_iter_anthropic_beta_header_candidates",
        "_split_anthropic_beta_values",
        "_extract_context_1m_beta_values",
        "_model_strings_indicate_context_1m_suffix",
        "_select_safe_anthropic_context_window_beta",
        "_apply_anthropic_context_window_metadata_fields",
        "_classify_anthropic_context_window_from_retained_evidence",
        "_enrich_anthropic_context_window_metadata",
        "_enrich_backfill_anthropic_context_window_metadata",
        "_read_varint",
        "_extract_gemini_signature_summary",
        "_enrich_gemini_thought_signature_metadata",
        "_enrich_agent_identity_metadata",
        "_enrich_trace_name_and_provider_metadata",
        # tool_activity.py
        "_extract_tool_activity_from_message",
        "_summarize_tool_activity",
        "_build_tool_activity_entry",
        "_classify_tool_kind",
        # claude_review.py
        "_lookup_claude_auto_review_parent_identity",
        "_apply_claude_auto_review_parent_identity_from_store",
        "_apply_claude_auto_review_metadata",
        "_extract_claude_auto_review_source_model",
    }
)

_A4C_SYMBOLS = frozenset(
    {
        # aawm_session_history/normalize.py (record normalization + trust chains)
        "_normalize_session_history_record",
        "_sync_session_history_record_metadata",
        "_normalize_session_repository_on_record",
        "_normalize_session_tenant_on_record",
        # aawm_session_history/context_window.py
        "_enrich_anthropic_context_window_metadata",
    }
)

_A4D_SYMBOLS = frozenset(
    {
        # aawm_session_history/backfill.py (spend-log/langfuse synthesis)
        "_split_spend_log_proxy_server_request",
        "_extract_trace_id_from_spend_log_row",
        "_coerce_nested_session_id",
        "_extract_session_id_from_spend_log_row",
        "_coerce_spend_log_request_tags",
        "_synthesize_result_from_spend_log_row",
        "_build_backfill_kwargs_from_spend_log_row",
        "_derive_langfuse_trace_tags_from_spend_log_row",
        "_serialize_searchable_text",
        "_extract_agent_context_from_langfuse_trace_observation",
        "_extract_langfuse_session_id",
        "_build_usage_object_from_langfuse_observation",
        "_extract_first_langfuse_response_message",
        "_infer_provider_from_langfuse_observation",
        "_derive_request_tags_from_langfuse_metadata",
        "_derive_langfuse_trace_tags_from_langfuse_trace",
        "_iter_litellm_metadata_sources",
        # aawm_session_history/storage_fields.py (rate-limit storage + DB payloads)
        "_rate_limit_storage_provider",
        "_rate_limit_storage_client",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_quota_type",
        "_rate_limit_storage_remaining_pct",
        "_rate_limit_storage_numeric_detail",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_quota_remaining",
        "_tool_definition_snapshot_from_metadata",
        "_build_tool_definition_snapshot_db_payload",
        "_update_session_history_previous_gap_ms",
        "_extract_session_history_call_ids_from_payloads",
        "_strip_postgres_nul_bytes",
    }
)


# =========================================================================
# Wave A4A RED ownership tests (fail until engineer performs extraction)
# =========================================================================


def test_a4a_moved_functions_not_defined_in_init() -> None:
    """A4A-moved helper functions must not remain as FunctionDef in __init__.py.

    RED until the engineer extracts them into the four target submodules.
    """
    tree = _parse_init_module()
    defined_functions = _function_def_names(tree)

    still_defined = sorted(
        name for name in _A4A_FUNCTION_NAMES if name in defined_functions
    )
    assert not still_defined, (
        "expected these A4A-moved functions to no longer be defined directly "
        f"in __init__.py (they should live in their target submodule with only "
        f"a facade assignment remaining): {still_defined}"
    )


def test_a4a_target_submodules_exist() -> None:
    """The four A4A target submodule files must exist after extraction."""
    missing = [
        mod
        for mod in _A4A_MOVED_NAMES_BY_MODULE
        if not (IDENTITY_PKG_DIR / f"{mod}.py").is_file()
    ]
    assert not missing, f"expected these A4A target submodule files to exist: {missing}"


def test_a4a_facade_identity() -> None:
    """getattr(pkg, name) is getattr(submodule, name) for every A4A-moved name."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    missing_modules: List[str] = []
    mismatched: List[tuple] = []
    for module_name, names in _A4A_MOVED_NAMES_BY_MODULE.items():
        try:
            submodule = importlib.import_module(
                f"litellm.integrations.aawm_agent_identity.{module_name}"
            )
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        for name in names:
            pkg_value = getattr(identity_pkg, name, None)
            sub_value = getattr(submodule, name, None)
            if pkg_value is None or sub_value is None or pkg_value is not sub_value:
                mismatched.append((module_name, name))

    assert not missing_modules, (
        f"expected these A4A target submodules to exist: {missing_modules}"
    )
    assert not mismatched, (
        "expected facade identity for each A4A-moved name; "
        f"mismatches (module, name): {mismatched}"
    )


def test_a4a_host_global_install_pattern() -> None:
    """Each A4A submodule must expose an ``install(globals_dict)`` callable
    that rebinds moved helpers' ``__globals__`` to the host namespace,
    matching the existing A2 extraction pattern."""
    missing_modules: List[str] = []
    missing_install: List[str] = []
    for module_name in _A4A_MOVED_NAMES_BY_MODULE:
        try:
            submodule = importlib.import_module(
                f"litellm.integrations.aawm_agent_identity.{module_name}"
            )
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        if not callable(getattr(submodule, "install", None)):
            missing_install.append(module_name)

    assert not missing_modules, (
        f"expected these A4A target submodules to exist: {missing_modules}"
    )
    assert not missing_install, (
        "expected each A4A submodule to expose an install(globals_dict) "
        f"callable for __globals__ rebinding: {missing_install}"
    )


def test_a4a_cached_helper_rebinds_wrapped_globals_and_preserves_cache(
    monkeypatch,
) -> None:
    """Cached moved helpers must resolve dependencies through package facades."""
    import litellm.integrations.aawm_agent_identity as identity_pkg
    import litellm.integrations.aawm_agent_identity.provider_normalize as provider_normalize
    import litellm.utils as litellm_utils

    helper = identity_pkg._session_history_provider_from_model_catalog
    assert helper is provider_normalize._session_history_provider_from_model_catalog
    assert helper.__wrapped__.__globals__ is identity_pkg.__dict__
    assert helper.cache_parameters() == {"maxsize": 512, "typed": False}

    calls = {"catalog": 0, "normalize": 0}

    def fake_get_model_info(*, model: str):
        calls["catalog"] += 1
        return {"litellm_provider": f"catalog:{model}"}

    def fake_normalize(candidate):
        calls["normalize"] += 1
        return f"patched:{candidate}"

    monkeypatch.setattr(litellm_utils, "get_model_info", fake_get_model_info)
    monkeypatch.setattr(
        identity_pkg,
        "_normalize_session_history_provider_name",
        fake_normalize,
    )
    helper.cache_clear()
    try:
        expected = "patched:catalog:a4a-cache-rebind-probe"
        assert helper("a4a-cache-rebind-probe") == expected
        assert helper("a4a-cache-rebind-probe") == expected
        assert calls == {"catalog": 1, "normalize": 1}
        assert helper.cache_info().hits == 1
        assert helper.cache_info().misses == 1
    finally:
        helper.cache_clear()


def test_a4a_moved_annotations_are_runtime_objects() -> None:
    """Moved annotations retain the baseline module's evaluated-object form."""
    string_annotations = []
    for module_name, names in _A4A_MOVED_NAMES_BY_MODULE.items():
        submodule = importlib.import_module(
            f"litellm.integrations.aawm_agent_identity.{module_name}"
        )
        for name in names:
            if name in _A4A_CONSTANT_NAMES:
                continue
            annotations = getattr(getattr(submodule, name), "__annotations__", {})
            for annotation_name, annotation_value in annotations.items():
                if isinstance(annotation_value, str):
                    string_annotations.append(
                        (module_name, name, annotation_name, annotation_value)
                    )

    assert not string_annotations, (
        "A4A moved functions must expose evaluated annotation objects, not "
        f"postponed strings: {string_annotations}"
    )

    provider_normalize = importlib.import_module(
        "litellm.integrations.aawm_agent_identity.provider_normalize"
    )
    annotations = (
        provider_normalize._session_history_provider_from_model_catalog
        .__wrapped__.__annotations__
    )
    assert annotations["model"] is str
    assert annotations["return"] == Optional[str]
    assert (
        provider_normalize._session_history_provider_from_model_catalog
        .__wrapped__.__globals__
        is sys.modules["litellm.integrations.aawm_agent_identity"].__dict__
    )


def test_a4a_type_checking_host_callable_signatures_match_runtime() -> None:
    """Type-only host declarations must match runtime callable contracts."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    annotation_namespace = dict(vars(typing))
    annotation_namespace.update(vars(builtins))
    annotation_namespace["datetime"] = datetime
    mismatches = []

    def evaluate_annotation(annotation: ast.expr | None):
        if annotation is None:
            return inspect.Signature.empty
        expression = ast.Expression(body=annotation)
        ast.fix_missing_locations(expression)
        value = eval(
            compile(expression, "<a4a-type-contract>", "eval"),
            annotation_namespace,
        )
        return type(None) if value is None else value

    def declared_parameters(function: ast.FunctionDef):
        arguments = function.args
        positional = [
            (argument, inspect.Parameter.POSITIONAL_ONLY)
            for argument in arguments.posonlyargs
        ] + [
            (argument, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for argument in arguments.args
        ]
        positional_defaults = [None] * (
            len(positional) - len(arguments.defaults)
        ) + list(arguments.defaults)
        parameters = []
        for (argument, kind), default_node in zip(
            positional,
            positional_defaults,
        ):
            default = (
                inspect.Parameter.empty
                if default_node is None
                else ast.literal_eval(default_node)
            )
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    kind,
                    default=default,
                    annotation=evaluate_annotation(argument.annotation),
                )
            )
        if arguments.vararg is not None:
            parameters.append(
                inspect.Parameter(
                    arguments.vararg.arg,
                    inspect.Parameter.VAR_POSITIONAL,
                    annotation=evaluate_annotation(arguments.vararg.annotation),
                )
            )
        for argument, default_node in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
        ):
            default = (
                inspect.Parameter.empty
                if default_node is None
                else ast.literal_eval(default_node)
            )
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=evaluate_annotation(argument.annotation),
                )
            )
        if arguments.kwarg is not None:
            parameters.append(
                inspect.Parameter(
                    arguments.kwarg.arg,
                    inspect.Parameter.VAR_KEYWORD,
                    annotation=evaluate_annotation(arguments.kwarg.annotation),
                )
            )
        return parameters

    for module_name in _A4A_MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )
        for node in tree.body:
            if not (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                continue
            for declaration in node.body:
                if not isinstance(declaration, ast.FunctionDef):
                    continue
                runtime_callable = getattr(identity_pkg, declaration.name)
                declared_signature = inspect.Signature(
                    declared_parameters(declaration),
                    return_annotation=evaluate_annotation(declaration.returns),
                )
                runtime_signature = inspect.signature(runtime_callable)
                runtime_hints = typing.get_type_hints(runtime_callable)
                runtime_signature = runtime_signature.replace(
                    parameters=[
                        parameter.replace(
                            annotation=runtime_hints.get(
                                parameter.name,
                                inspect.Signature.empty,
                            )
                        )
                        for parameter in runtime_signature.parameters.values()
                    ],
                    return_annotation=runtime_hints.get(
                        "return",
                        inspect.Signature.empty,
                    ),
                )
                if declared_signature != runtime_signature:
                    mismatches.append(
                        (
                            module_name,
                            declaration.name,
                            declared_signature,
                            runtime_signature,
                        )
                    )

    assert not mismatches, (
        "A4A TYPE_CHECKING host callable declarations drifted from runtime "
        f"signatures: {mismatches}"
    )


def test_a4a_rebind_order_facades_before_record_install() -> None:
    """Every A4A facade assignment in __init__.py must precede the
    _bind_session_history_record_apis() call."""
    tree = _parse_init_module()

    call_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_bind_session_history_record_apis"
        ):
            call_line = node.lineno
            break

    assert call_line is not None, (
        "expected a call to _bind_session_history_record_apis() in __init__.py"
    )

    defined_functions = _function_def_names(tree)
    assign_targets = _module_level_assign_targets(tree)

    facade_names_present = [
        name
        for name in _A4A_FUNCTION_NAMES
        if name in assign_targets and name not in defined_functions
    ]

    assert facade_names_present, (
        "expected at least some A4A-moved functions to appear as module-level "
        "facade assignments in __init__.py once the extraction has landed "
        "(none found -- expected RED state before the move)"
    )

    late_facades = [
        name
        for name in facade_names_present
        if assign_targets[name].lineno >= call_line
    ]
    assert not late_facades, (
        "expected every A4A facade assignment to precede the "
        f"_bind_session_history_record_apis() call (line {call_line}); "
        f"these facades assign at or after that call: {late_facades}"
    )


def test_a4a_submodules_do_not_import_init_at_module_scope() -> None:
    """A4A submodules must not import the __init__ package at module scope."""
    missing_modules: List[str] = []
    offending: List[str] = []
    for module_name in _A4A_MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        if not module_path.is_file():
            missing_modules.append(module_name)
            continue
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "litellm.integrations.aawm_agent_identity"
            ):
                offending.append(module_name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "litellm.integrations.aawm_agent_identity":
                        offending.append(module_name)

    assert not missing_modules, (
        f"expected these A4A target submodule files to exist: {missing_modules}"
    )
    assert not offending, (
        f"these A4A submodules import the identity __init__ package at "
        f"module scope: {offending}"
    )


# =========================================================================
# Wave A4A boundary guards (GREEN now and must stay GREEN): the inventory
# must not claim any symbol owned by A4B/A4C/A4D. These catch the exact
# misclassifications called out in the review (spend-log/langfuse backfill,
# gemini signatures, worker-context/anthropic-1M, claude auto-review,
# rate-limit storage / tool-definition DB payloads, compact-summary vs
# reasoning/rerank split).
# =========================================================================


def test_a4a_inventory_excludes_a4b_symbols() -> None:
    overlap = sorted(_A4B_SYMBOLS & set(_A4A_ALL_MOVED_NAMES))
    assert not overlap, f"A4A inventory must not claim A4B symbols: {overlap}"


def test_a4a_inventory_excludes_a4c_symbols() -> None:
    overlap = sorted(_A4C_SYMBOLS & set(_A4A_ALL_MOVED_NAMES))
    assert not overlap, f"A4A inventory must not claim A4C symbols: {overlap}"


def test_a4a_inventory_excludes_a4d_symbols() -> None:
    overlap = sorted(_A4D_SYMBOLS & set(_A4A_ALL_MOVED_NAMES))
    assert not overlap, f"A4A inventory must not claim A4D symbols: {overlap}"


def test_a4a_inventory_has_no_duplicate_across_modules() -> None:
    """Each A4A symbol must be owned by exactly one target module."""
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for module_name, names in _A4A_MOVED_NAMES_BY_MODULE.items():
        for name in names:
            if name in seen:
                duplicates.append(f"{name} (in {seen[name]} and {module_name})")
            else:
                seen[name] = module_name
    assert not duplicates, f"A4A symbols owned by >1 module: {duplicates}"


def test_a4a_compact_summary_in_request_signals_not_prompt_overhead() -> None:
    """Compact-summary classification belongs to request_signals (8743-9539),
    NOT prompt_overhead (10268-11019)."""
    assert "_classify_compact_summary_state" in _A4A_MOVED_NAMES_BY_MODULE["request_signals"]
    assert "_classify_compact_summary_state" not in _A4A_MOVED_NAMES_BY_MODULE["prompt_overhead"]


def test_a4a_reasoning_and_rerank_in_usage_extract_not_prompt_overhead() -> None:
    """Reasoning-token + rerank-payload parsing (10094-10267) belongs to
    usage_extract, NOT prompt_overhead."""
    for name in (
        "_extract_reported_reasoning_tokens",
        "_fallback_gemini_reasoning_tokens_from_signatures",
        "_determine_reasoning_tokens_source",
        "_estimate_reasoning_tokens",
        "_extract_rerank_request_payload",
        "_coerce_rerank_text",
        "_extract_rerank_document_text",
    ):
        assert name in _A4A_MOVED_NAMES_BY_MODULE["usage_extract"], name
        assert name not in _A4A_MOVED_NAMES_BY_MODULE["prompt_overhead"], name


# =========================================================================
# Wave A4B: tool_activity / claude_review / enrich ownership inventory
#
# Bands are the ORIGINAL single-file line ranges from the plan:
#   tool_activity.py  (:12660-13454)
#   claude_review.py  (:3192-3626, :15395-15446)
#   enrich.py         (:3052-3191, :13455-13671, :14393-14491,
#                      :16158-16482, orchestrators :16483-16641)
#
# enrich.py is the high-fan-in module that moves LAST within A4B.
# The two orchestrators (_enrich_agent_identity_metadata,
# _enrich_trace_name_and_provider_metadata) may stay in __init__ as
# thin delegates -- engineer's call, documented in the A4B landing note.
#
# EXCLUDES all A4A (usage_extract, provider_normalize, request_signals,
# prompt_overhead), A4C (normalize, context_window), and A4D (backfill,
# storage_fields) symbols.
# =========================================================================

_A4B_MOVED_NAMES_BY_MODULE: Dict[str, List[str]] = {
    # Tool activity detection, classification, extraction, summarization,
    # and sensitive-config handling. Original band :12660-13454.
    "tool_activity": [
        # helpers
        "_dedupe_strings",
        "_normalize_changed_file_path",
        "_changed_file_basename",
        "_sensitive_config_change_flags_from_paths",
        "_text_mentions_env_file",
        "_redact_sensitive_config_argument_value",
        "_sanitize_tool_activity_arguments_for_sensitive_config",
        "_normalize_sensitive_config_change_state_on_record",
        "_parse_tool_arguments",
        "_is_empty_claude_read_pages_value",
        "_sanitize_tool_activity_arguments",
        "_extract_paths_from_patch_text",
        "_extract_file_paths_from_tool_arguments",
        "_extract_command_text_from_tool_arguments",
        "_count_git_subcommand",
        "_collect_file_paths_from_value",
        "_find_command_text_in_value",
        # primary API
        "_classify_tool_kind",
        "_build_tool_activity_entry",
        "_extract_tool_activity_from_message",
        "_extract_response_output_items",
        "_resolve_response_output_tool_name",
        "_extract_response_output_tool_activity",
        "_summarize_tool_activity",
        "_extract_tool_call_info",
        "_extract_response_output_tool_call_info",
        # constants
        "_TOOL_ACTIVITY_READ_NAMES",
        "_TOOL_ACTIVITY_MODIFY_NAMES",
        "_TOOL_ACTIVITY_COMMAND_NAMES",
        "_TOOL_ACTIVITY_SKIP_PATH_KEYS",
        "_APPLY_PATCH_FILE_RE",
        "_APPLY_PATCH_MOVE_TO_RE",
        "_GIT_COMMAND_RE",
        "_GIT_GLOBAL_OPTIONS_WITH_VALUES",
        "_TOOL_ACTIVITY_COMMAND_TEXT_KEYS",
        "_TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS",
        "_SENSITIVE_CONFIG_CHANGE_FIELDS",
        "_SENSITIVE_CONFIG_ENV_REDACTION",
        "_SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS",
        "_SENSITIVE_CONFIG_ENV_COMMAND_RE",
        "_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES",
        "_RESPONSE_OUTPUT_TOOL_ITEM_TYPES",
    ],
    # Claude permission-check detection, auto-review identity, and
    # parent-identity inheritance. Original bands :3192-3626, :15395-15446.
    "claude_review": [
        # permission-check detection
        "_permission_check_probeable_value",
        "_extract_claude_permission_check_decision_from_value",
        "_extract_claude_permission_check_decision",
        "_extract_claude_permission_check_models",
        "_enrich_claude_permission_check_metadata",
        # metadata helpers
        "_metadata_bool",
        "_metadata_request_tags",
        "_is_claude_permission_check_metadata",
        "_extract_claude_project_from_metadata_tags",
        # auto-review identity
        "_extract_claude_auto_review_source_model",
        "_apply_claude_auto_review_metadata",
        "_apply_claude_auto_review_identity_to_record",
        "_extract_claude_auto_review_identity_from_row",
        "_apply_claude_auto_review_parent_identity",
        "_build_session_identity_cache",
        "_build_permission_usage_fields",
        # async store helpers
        "_lookup_claude_auto_review_parent_identity",
        "_apply_claude_auto_review_parent_identity_from_store",
        # constants
        "_CLAUDE_PERMISSION_CHECK_OUTPUT_RE",
        "_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL",
        "_CLAUDE_AUTO_REVIEW_TRACE_NAME",
        "_CLAUDE_AUTO_REVIEW_AGENT_NAME",
    ],
    # Enrichment orchestrators, worker-context exhaustion, usage breakout,
    # thinking/signature decoding. enrich.py moves LAST within A4B.
    # Original bands :3052-3191, :13455-13671, :14393-14491,
    # :16158-16482, orchestrators :16483-16641.
    "enrich": [
        # worker context exhaustion
        "_bound_worker_context_exhaustion_string",
        "_normalize_worker_context_exhaustion_bool",
        "_sanitize_worker_context_exhaustion_metadata",
        "_promote_worker_context_exhaustion_metadata",
        # usage breakout
        "_infer_usage_breakout_provider_prefix",
        "_enrich_usage_breakout_metadata",
        # claude thinking signature decode
        "_enrich_claude_thinking_metadata",
        # gemini thought signature decode
        "_read_varint",
        "_extract_gemini_signature_summary",
        "_enrich_gemini_thought_signature_metadata",
        # orchestrators (may stay as thin delegates in __init__)
        "_enrich_agent_identity_metadata",
        "_enrich_trace_name_and_provider_metadata",
        # shared enrichment helpers
        "_get_reasoning_state_tags",
        "_extract_claude_experiment_ids",
        "_extract_reasoning_content",
        "_extract_thinking_blocks",
        "_normalize_base64_text",
        "_decode_base64_bytes",
        "_short_hash",
        # constants
        "_WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS",
        "_WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN",
        "_WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS",
        "_GEMINI_MARKER",
    ],
}

_A4B_ALL_MOVED_NAMES: List[str] = [
    name for names in _A4B_MOVED_NAMES_BY_MODULE.values() for name in names
]

_A4B_CONSTANT_NAMES = frozenset(
    {
        # tool_activity constants
        "_TOOL_ACTIVITY_READ_NAMES",
        "_TOOL_ACTIVITY_MODIFY_NAMES",
        "_TOOL_ACTIVITY_COMMAND_NAMES",
        "_TOOL_ACTIVITY_SKIP_PATH_KEYS",
        "_APPLY_PATCH_FILE_RE",
        "_APPLY_PATCH_MOVE_TO_RE",
        "_GIT_COMMAND_RE",
        "_GIT_GLOBAL_OPTIONS_WITH_VALUES",
        "_TOOL_ACTIVITY_COMMAND_TEXT_KEYS",
        "_TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS",
        "_SENSITIVE_CONFIG_CHANGE_FIELDS",
        "_SENSITIVE_CONFIG_ENV_REDACTION",
        "_SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS",
        "_SENSITIVE_CONFIG_ENV_COMMAND_RE",
        "_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES",
        "_RESPONSE_OUTPUT_TOOL_ITEM_TYPES",
        # claude_review constants
        "_CLAUDE_PERMISSION_CHECK_OUTPUT_RE",
        "_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL",
        "_CLAUDE_AUTO_REVIEW_TRACE_NAME",
        "_CLAUDE_AUTO_REVIEW_AGENT_NAME",
        # enrich constants
        "_WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS",
        "_WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN",
        "_WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS",
        "_GEMINI_MARKER",
    }
)

_A4B_FUNCTION_NAMES = frozenset(
    set(_A4B_ALL_MOVED_NAMES) - _A4B_CONSTANT_NAMES
)

_A4B_ASYNC_FUNCTION_NAMES = frozenset(
    {
        "_lookup_claude_auto_review_parent_identity",
        "_apply_claude_auto_review_parent_identity_from_store",
    }
)


# =========================================================================
# Wave A4B golden parity tests (GREEN now, must stay GREEN after extraction)
#
# These pin current pre-extraction behavior. Golden values were captured
# by running the current code on 2026-07-25.
# =========================================================================


def test_a4b_golden_classify_tool_kind() -> None:
    """Pin _classify_tool_kind classification for representative tool names."""
    from litellm.integrations.aawm_agent_identity import _classify_tool_kind

    goldens = {
        "bash": "command",
        "write": "modify",
        "read": "read",
        "mcp__server__tool": "mcp",
        "unknown_thing": "other",
        "exec_command": "command",
        "apply_patch": "modify",
        "grep": "read",
        "shell": "command",
        "notebookedit": "modify",
        "webfetch": "read",
        "custom_xyz": "other",
    }
    for tool_name, expected_kind in goldens.items():
        assert _classify_tool_kind(tool_name) == expected_kind, tool_name


def test_a4b_golden_build_tool_activity_entry_command() -> None:
    """Pin _build_tool_activity_entry for a bash command with git ops."""
    from litellm.integrations.aawm_agent_identity import _build_tool_activity_entry

    entry = _build_tool_activity_entry(
        tool_index=0,
        tool_name="bash",
        arguments=json.dumps(
            {"command": 'git commit -m "test" && git push origin main'}
        ),
        tool_call_id="tc1",
        source="message.tool_calls",
    )
    assert entry["tool_kind"] == "command"
    assert entry["git_commit_count"] == 1
    assert entry["git_push_count"] == 1
    assert entry["command_text"] == 'git commit -m "test" && git push origin main'
    assert entry["file_paths_read"] == []
    assert entry["file_paths_modified"] == []
    assert entry["metadata"] == {"source": "message.tool_calls"}


def test_a4b_golden_build_tool_activity_entry_read() -> None:
    """Pin _build_tool_activity_entry for a read tool."""
    from litellm.integrations.aawm_agent_identity import _build_tool_activity_entry

    entry = _build_tool_activity_entry(
        tool_index=1,
        tool_name="read",
        arguments=json.dumps({"file_path": "src/main.py"}),
        tool_call_id="tc2",
        source="message.tool_calls",
    )
    assert entry["tool_kind"] == "read"
    assert entry["file_paths_read"] == ["src/main.py"]
    assert entry["file_paths_modified"] == []
    assert entry["git_commit_count"] == 0
    assert entry["git_push_count"] == 0
    assert entry["command_text"] is None


def test_a4b_golden_build_tool_activity_entry_apply_patch() -> None:
    """Pin _build_tool_activity_entry for apply_patch with move-to."""
    from litellm.integrations.aawm_agent_identity import _build_tool_activity_entry

    entry = _build_tool_activity_entry(
        tool_index=2,
        tool_name="apply_patch",
        arguments=json.dumps(
            {"patch": "*** Update File: src/app.py\n*** Move to: src/new_app.py\n"}
        ),
        tool_call_id="tc3",
        source="message.tool_calls",
    )
    assert entry["tool_kind"] == "modify"
    assert entry["file_paths_modified"] == ["src/app.py", "src/new_app.py"]
    assert entry["file_paths_read"] == []
    assert entry["git_commit_count"] == 0
    assert entry["git_push_count"] == 0


def test_a4b_golden_extract_tool_activity_openai_format() -> None:
    """Pin _extract_tool_activity_from_message for OpenAI tool_calls format."""
    from litellm.integrations.aawm_agent_identity import (
        _extract_tool_activity_from_message,
    )

    message = {
        "tool_calls": [
            {
                "id": "tc1",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "ls -la"}),
                },
            }
        ]
    }
    activity = _extract_tool_activity_from_message(message)
    assert len(activity) == 1
    assert activity[0]["tool_name"] == "bash"
    assert activity[0]["tool_kind"] == "command"
    assert activity[0]["command_text"] == "ls -la"
    assert activity[0]["metadata"] == {"source": "message.tool_calls"}


def test_a4b_golden_extract_tool_activity_claude_format() -> None:
    """Pin _extract_tool_activity_from_message for Anthropic content blocks."""
    from litellm.integrations.aawm_agent_identity import (
        _extract_tool_activity_from_message,
    )

    message = {
        "content": [
            {
                "type": "tool_use",
                "id": "tc2",
                "name": "write",
                "input": {"file_path": "test.py", "content": "print(1)"},
            }
        ]
    }
    activity = _extract_tool_activity_from_message(message)
    assert len(activity) == 1
    assert activity[0]["tool_name"] == "write"
    assert activity[0]["tool_kind"] == "modify"
    assert activity[0]["file_paths_modified"] == ["test.py"]
    assert activity[0]["metadata"] == {"source": "message.content"}


def test_a4b_golden_extract_tool_activity_absent() -> None:
    """Pin _extract_tool_activity_from_message for a message with no tools."""
    from litellm.integrations.aawm_agent_identity import (
        _extract_tool_activity_from_message,
    )

    assert _extract_tool_activity_from_message({"content": "just text"}) == []
    assert _extract_tool_activity_from_message({}) == []
    assert _extract_tool_activity_from_message(None) == []


def test_a4b_golden_summarize_tool_activity() -> None:
    """Pin _summarize_tool_activity rollup shape."""
    from litellm.integrations.aawm_agent_identity import _summarize_tool_activity

    activity = [
        {
            "file_paths_read": [],
            "file_paths_modified": [],
            "git_commit_count": 0,
            "git_push_count": 0,
        },
        {
            "file_paths_read": [],
            "file_paths_modified": ["test.py"],
            "git_commit_count": 0,
            "git_push_count": 0,
        },
    ]
    summary = _summarize_tool_activity(activity)
    assert summary == {
        "file_read_count": 0,
        "file_modified_count": 1,
        "changed_pre_commit_config": False,
        "changed_env_file": False,
        "changed_pyproject_toml": False,
        "changed_gitignore": False,
        "git_commit_count": 0,
        "git_push_count": 0,
    }


def test_a4b_golden_extract_tool_call_info() -> None:
    """Pin _extract_tool_call_info for present and absent tool calls."""
    from litellm.integrations.aawm_agent_identity import _extract_tool_call_info

    msg_openai = {
        "tool_calls": [
            {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}
        ]
    }
    count, names = _extract_tool_call_info(msg_openai)
    assert count == 1
    assert names == ["bash"]

    count_empty, names_empty = _extract_tool_call_info({"content": "text"})
    assert count_empty == 0
    assert names_empty == []


def test_a4b_golden_claude_auto_review_source_model() -> None:
    """Pin _extract_claude_auto_review_source_model priority chain."""
    from litellm.integrations.aawm_agent_identity import (
        _extract_claude_auto_review_source_model,
    )

    assert (
        _extract_claude_auto_review_source_model(
            {"source_model": "claude-sonnet-4-20250514"}
        )
        == "claude-sonnet-4-20250514"
    )
    assert (
        _extract_claude_auto_review_source_model(
            {"claude_permission_check_response_model": "claude-3-opus"}
        )
        == "claude-3-opus"
    )
    assert (
        _extract_claude_auto_review_source_model({}, "fallback-model")
        == "fallback-model"
    )
    assert _extract_claude_auto_review_source_model({}) is None


def test_a4b_golden_apply_claude_auto_review_metadata_with_identity() -> None:
    """Pin _apply_claude_auto_review_metadata with repository inheritance."""
    from litellm.integrations.aawm_agent_identity import (
        _apply_claude_auto_review_metadata,
    )

    metadata: Dict[str, Any] = {}
    _apply_claude_auto_review_metadata(
        metadata,
        repository="my-repo",
        tenant_id="my-tenant",
        source_model="claude-sonnet-4-20250514",
    )
    assert metadata["trace_name"] == "claude-code.auto-reviewer"
    assert metadata["agent_name"] == "auto-reviewer"
    assert metadata["aawm_claude_agent_name"] == "auto-reviewer"
    assert metadata["logical_model"] == "claude-auto-review"
    assert metadata["source_model"] == "claude-sonnet-4-20250514"
    assert metadata["repository"] == "my-repo"
    assert metadata["tenant_id"] == "my-repo"
    assert metadata["aawm_tenant_id"] == "my-repo"
    assert metadata["aawm_claude_project"] == "my-repo"
    assert metadata["trace_user_id"] == "my-repo"
    assert "claude-internal-check" in metadata["tags"]
    assert "claude-permission-check" in metadata["tags"]
    assert "claude-agent:auto-reviewer" in metadata["tags"]
    assert "claude-project:my-repo" in metadata["tags"]
    assert metadata["request_tags"] == metadata["tags"]


def test_a4b_golden_apply_claude_auto_review_metadata_no_identity() -> None:
    """Pin _apply_claude_auto_review_metadata without repository."""
    from litellm.integrations.aawm_agent_identity import (
        _apply_claude_auto_review_metadata,
    )

    metadata: Dict[str, Any] = {}
    _apply_claude_auto_review_metadata(metadata)
    assert metadata["trace_name"] == "claude-code.auto-reviewer"
    assert metadata["agent_name"] == "auto-reviewer"
    assert "repository" not in metadata
    assert "tenant_id" not in metadata
    assert "claude-project:" not in str(metadata.get("tags", []))


def test_a4b_golden_apply_claude_auto_review_identity_to_record() -> None:
    """Pin _apply_claude_auto_review_identity_to_record for permission-check
    and non-permission-check records."""
    from litellm.integrations.aawm_agent_identity import (
        _apply_claude_auto_review_identity_to_record,
    )

    # Permission-check record: identity is applied
    record: Dict[str, Any] = {
        "model": "claude-sonnet-4-20250514",
        "repository": "test-repo",
        "tenant_id": "test-tenant",
        "metadata": {
            "claude_permission_check": True,
            "source_model": "claude-sonnet-4-20250514",
        },
    }
    _apply_claude_auto_review_identity_to_record(record)
    assert record["model"] == "claude-auto-review"
    assert record["agent_name"] == "auto-reviewer"
    assert record["repository"] == "test-repo"
    assert record["metadata"]["trace_name"] == "claude-code.auto-reviewer"

    # Non-permission-check record: unchanged
    record2: Dict[str, Any] = {"model": "gpt-4", "metadata": {"foo": "bar"}}
    _apply_claude_auto_review_identity_to_record(record2)
    assert record2["model"] == "gpt-4"
    assert record2["metadata"] == {"foo": "bar"}


def test_a4b_golden_extract_claude_auto_review_identity_from_row() -> None:
    """Pin _extract_claude_auto_review_identity_from_row for present and
    absent repository."""
    from litellm.integrations.aawm_agent_identity import (
        _extract_claude_auto_review_identity_from_row,
    )

    row_with_repo = {
        "id": "row-1",
        "repository": "my-repo",
        "tenant_id": "my-tenant",
        "metadata": {},
    }
    identity = _extract_claude_auto_review_identity_from_row(row_with_repo)
    assert identity == {
        "repository": "my-repo",
        "tenant_id": "my-repo",
        "source_row_id": "row-1",
        "source": "same_session.session_history",
    }

    row_without_repo = {"id": "row-2", "metadata": {}}
    assert _extract_claude_auto_review_identity_from_row(row_without_repo) is None


def test_a4b_golden_is_claude_permission_check_metadata() -> None:
    """Pin _is_claude_permission_check_metadata for various inputs."""
    from litellm.integrations.aawm_agent_identity import (
        _is_claude_permission_check_metadata,
    )

    assert _is_claude_permission_check_metadata({"claude_permission_check": True})
    assert _is_claude_permission_check_metadata(
        {"request_tags": ["claude-permission-check"]}
    )
    assert not _is_claude_permission_check_metadata({"foo": "bar"})
    assert not _is_claude_permission_check_metadata(None)
    assert not _is_claude_permission_check_metadata("not-a-dict")


def test_a4b_golden_enrich_claude_permission_check_metadata() -> None:
    """Pin _enrich_claude_permission_check_metadata for block/allow/absent."""
    from litellm.integrations.aawm_agent_identity import (
        _enrich_claude_permission_check_metadata,
    )

    # Decision: yes (blocked)
    metadata_yes: Dict[str, Any] = {}
    _enrich_claude_permission_check_metadata(
        {"model": "claude-sonnet-4-20250514"},
        metadata_yes,
        {"content": [{"type": "text", "text": "<block>yes"}]},
    )
    metadata_yes.pop("langfuse_spans", None)
    assert metadata_yes["claude_permission_check"] is True
    assert metadata_yes["claude_permission_check_decision"] == "yes"
    assert metadata_yes["claude_permission_check_blocked"] is True
    assert "claude-permission-check:block" in metadata_yes["tags"]

    # Decision: no (allowed)
    metadata_no: Dict[str, Any] = {}
    _enrich_claude_permission_check_metadata(
        {"model": "claude-sonnet-4-20250514"},
        metadata_no,
        {"content": [{"type": "text", "text": "<block>no"}]},
    )
    metadata_no.pop("langfuse_spans", None)
    assert metadata_no["claude_permission_check_decision"] == "no"
    assert metadata_no["claude_permission_check_blocked"] is False
    assert "claude-permission-check:allow" in metadata_no["tags"]

    # No decision: metadata unchanged
    metadata_none: Dict[str, Any] = {}
    _enrich_claude_permission_check_metadata(
        {"model": "gpt-4"},
        metadata_none,
        {"content": [{"type": "text", "text": "Hello world"}]},
    )
    assert metadata_none == {}


def test_a4b_golden_sanitize_worker_context_exhaustion_metadata() -> None:
    """Pin _sanitize_worker_context_exhaustion_metadata bounding behavior."""
    from litellm.integrations.aawm_agent_identity import (
        _sanitize_worker_context_exhaustion_metadata,
    )

    # failure_class forces success/completed to False
    meta1: Dict[str, Any] = {
        "worker_context_exhaustion_success": "true",
        "worker_context_exhaustion_completed": "false",
        "worker_context_exhaustion_failure_class": "rate_limit",
        "worker_context_exhaustion_failure_reason": "x" * 600,
    }
    _sanitize_worker_context_exhaustion_metadata(meta1)
    assert meta1["worker_context_exhaustion_success"] is False
    assert meta1["worker_context_exhaustion_completed"] is False
    assert meta1["worker_context_exhaustion_failure_class"] == "rate_limit"
    # detail is bounded to 512 (default max)
    assert len(meta1["worker_context_exhaustion_failure_reason"]) == 512

    # Invalid bool value is dropped
    meta2: Dict[str, Any] = {"worker_context_exhaustion_success": "invalid_value"}
    _sanitize_worker_context_exhaustion_metadata(meta2)
    assert "worker_context_exhaustion_success" not in meta2

    # Empty metadata is unchanged
    meta3: Dict[str, Any] = {}
    _sanitize_worker_context_exhaustion_metadata(meta3)
    assert meta3 == {}


def test_a4b_golden_normalize_worker_context_exhaustion_bool() -> None:
    """Pin _normalize_worker_context_exhaustion_bool for all input types."""
    from litellm.integrations.aawm_agent_identity import (
        _normalize_worker_context_exhaustion_bool,
    )

    goldens = [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
        (2, None),
        ("maybe", None),
        (None, None),
    ]
    for value, expected in goldens:
        assert _normalize_worker_context_exhaustion_bool(value) == expected, repr(
            value
        )


def test_a4b_golden_infer_usage_breakout_provider_prefix() -> None:
    """Pin _infer_usage_breakout_provider_prefix resolution order."""
    from litellm.integrations.aawm_agent_identity import (
        _infer_usage_breakout_provider_prefix,
    )

    assert (
        _infer_usage_breakout_provider_prefix({"custom_llm_provider": "gemini"}, {})
        == "gemini"
    )
    assert (
        _infer_usage_breakout_provider_prefix({"model": "codex-mini"}, {}) == "codex"
    )
    assert (
        _infer_usage_breakout_provider_prefix(
            {}, {"passthrough_route_family": "codex_responses"}
        )
        == "codex"
    )
    assert _infer_usage_breakout_provider_prefix({"model": "gpt-4"}, {}) is None
    assert (
        _infer_usage_breakout_provider_prefix(
            {}, {"passthrough_route_family": "gemini_cli"}
        )
        == "gemini"
    )


def test_a4b_golden_enrich_usage_breakout_metadata() -> None:
    """Pin the high-fan-in usage-breakout orchestrator's current output."""
    from litellm.integrations.aawm_agent_identity import (
        _enrich_usage_breakout_metadata,
    )

    kwargs: Dict[str, Any] = {
        "model": "gemini/gemini-2.5-pro",
        "custom_llm_provider": "gemini",
        "litellm_params": {"metadata": {}},
    }
    result = {
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ],
    }

    _enrich_usage_breakout_metadata(kwargs, result)
    metadata = kwargs["litellm_params"]["metadata"]
    metadata.pop("langfuse_spans", None)
    assert metadata == {
        "gemini_cache_creation_input_tokens": 0,
        "gemini_cache_read_input_tokens": 3,
        "gemini_reasoning_tokens_reported": 2,
        "gemini_tool_call_count": 1,
        "gemini_tool_names": ["read"],
        "tags": [
            "gemini-usage-breakout",
            "reasoning-tokens-reported",
            "gemini-reasoning-tokens-reported",
            "cache-read-input-tokens",
            "gemini-cache-read-input-tokens",
            "tool-calls-present",
            "gemini-tool-calls-present",
        ],
        "usage_cache_creation_input_tokens": 0,
        "usage_cache_read_input_tokens": 3,
        "usage_reasoning_tokens_reported": 2,
        "usage_reasoning_tokens_source": "provider_reported",
        "usage_tool_call_count": 1,
        "usage_tool_names": ["read"],
    }


def test_a4b_golden_bound_worker_context_exhaustion_string() -> None:
    """Pin _bound_worker_context_exhaustion_string truncation."""
    from litellm.integrations.aawm_agent_identity import (
        _bound_worker_context_exhaustion_string,
    )

    assert (
        _bound_worker_context_exhaustion_string(
            "worker_context_exhaustion_detail", "short"
        )
        == "short"
    )
    assert (
        _bound_worker_context_exhaustion_string(
            "worker_context_exhaustion_detail", None
        )
        is None
    )
    assert (
        _bound_worker_context_exhaustion_string(
            "worker_context_exhaustion_detail", ""
        )
        is None
    )
    long_value = "x" * 600
    bounded = _bound_worker_context_exhaustion_string(
        "worker_context_exhaustion_detail", long_value
    )
    assert bounded is not None
    assert len(bounded) == 512


def test_a4b_golden_enrich_gemini_thought_signature_metadata() -> None:
    """Pin _enrich_gemini_thought_signature_metadata for present and absent
    signatures."""
    from litellm.integrations.aawm_agent_identity import (
        _enrich_gemini_thought_signature_metadata,
    )

    # Construct a minimal valid gemini signature with the known marker
    marker = bytes.fromhex("8f3d6b5f")
    payload = b"\x00\x01\x02" + marker + b"\x03\x04"
    record = b"\x0A" + bytes([len(payload)]) + payload
    gemini_sig_b64 = base64.b64encode(record).decode()

    message = {
        "content": [{"type": "text", "text": "response"}],
        "provider_specific_fields": {"thought_signatures": [gemini_sig_b64]},
    }
    metadata: Dict[str, Any] = {}
    _enrich_gemini_thought_signature_metadata(metadata, message)
    metadata.pop("langfuse_spans", None)

    assert metadata["gemini_thought_signature_present"] is True
    assert metadata["gemini_thought_signature_count"] == 1
    assert metadata["gemini_tsig_decoded_bytes"] == 11
    assert metadata["gemini_tsig_record_count"] == 1
    assert metadata["gemini_tsig_record_sizes"] == [9]
    assert metadata["gemini_tsig_prefixes"] == ["000102"]
    assert metadata["gemini_tsig_marker_offsets"] == [5]
    assert metadata["gemini_tsig_marker_hex"] == "8f3d6b5f"
    assert metadata["thinking_signature_present"] is True
    assert metadata["thinking_signature_decoded"] is True
    assert "gemini-thought-signature" in metadata["tags"]

    # Absent signatures: metadata unchanged
    metadata_empty: Dict[str, Any] = {}
    _enrich_gemini_thought_signature_metadata(
        metadata_empty, {"content": [{"type": "text", "text": "hello"}]}
    )
    assert metadata_empty == {}


# =========================================================================
# Wave A4B async golden parity (GREEN now, must stay GREEN)
# =========================================================================


def test_a4b_golden_async_lookup_claude_auto_review_parent_identity() -> None:
    """Pin _lookup_claude_auto_review_parent_identity awaitability and
    result shape."""
    from litellm.integrations.aawm_agent_identity import (
        _lookup_claude_auto_review_parent_identity,
    )

    class _MockConn:
        async def fetch(self, sql: str, *args: Any) -> list:
            return [
                {
                    "id": "row-1",
                    "repository": "parent-repo",
                    "tenant_id": "parent-tenant",
                    "agent_name": "dev",
                    "metadata": {},
                }
            ]

    async def _run() -> None:
        conn = _MockConn()
        # With session_id: returns identity
        result = await _lookup_claude_auto_review_parent_identity(
            conn, {"session_id": "sess-1", "start_time": "2026-01-01T00:00:00Z"}
        )
        assert result == {
            "repository": "parent-repo",
            "tenant_id": "parent-repo",
            "source_row_id": "row-1",
            "source": "same_session.session_history",
        }
        # Without session_id: returns None
        result_none = await _lookup_claude_auto_review_parent_identity(conn, {})
        assert result_none is None

    asyncio.run(_run())


def test_a4b_golden_async_apply_from_store_cached_and_noop() -> None:
    """Pin _apply_claude_auto_review_parent_identity_from_store for cached
    identity hit and non-permission-check no-op."""
    from litellm.integrations.aawm_agent_identity import (
        _apply_claude_auto_review_parent_identity_from_store,
    )

    class _MockConn:
        async def fetch(self, sql: str, *args: Any) -> list:
            return []

    async def _run() -> None:
        conn = _MockConn()

        # Cached identity hit
        payload: Dict[str, Any] = {
            "session_id": "sess-1",
            "model": "claude-sonnet-4-20250514",
            "metadata": {"claude_permission_check": True},
        }
        identity_cache = {
            "sess-1": {
                "repository": "cached-repo",
                "tenant_id": "cached-repo",
                "source_row_id": "r1",
                "source": "same_session.session_history",
            }
        }
        await _apply_claude_auto_review_parent_identity_from_store(
            conn, payload, identity_by_session=identity_cache
        )
        assert payload["repository"] == "cached-repo"
        assert payload["tenant_id"] == "cached-repo"
        assert payload["metadata"]["trace_name"] == "claude-code.auto-reviewer"
        assert (
            payload["metadata"]["claude_auto_review_parent_identity_source"]
            == "same_session.session_history"
        )
        assert (
            payload["metadata"]["claude_auto_review_parent_identity_source_row_id"]
            == "r1"
        )

        # Non-permission-check: no-op
        payload_noop: Dict[str, Any] = {
            "session_id": "sess-2",
            "metadata": {"foo": "bar"},
        }
        await _apply_claude_auto_review_parent_identity_from_store(conn, payload_noop)
        assert payload_noop == {"session_id": "sess-2", "metadata": {"foo": "bar"}}

    asyncio.run(_run())


# =========================================================================
# Wave A4B monkeypatch-through-facade tests (GREEN now, must stay GREEN)
# =========================================================================


def test_a4b_tool_activity_monkeypatchable_through_facade(monkeypatch: Any) -> None:
    """Moved tool_activity functions remain monkeypatchable via the identity
    namespace after extraction."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    sentinel = {"called_with": None}
    original = identity_pkg._classify_tool_kind

    def _stub_classify(tool_name: str) -> str:
        sentinel["called_with"] = tool_name
        return "stubbed"

    monkeypatch.setattr(identity_pkg, "_classify_tool_kind", _stub_classify)
    # _build_tool_activity_entry calls _classify_tool_kind by free name;
    # after extraction the facade rebind must route through the patched value.
    entry = identity_pkg._build_tool_activity_entry(
        tool_index=0,
        tool_name="bash",
        arguments="{}",
        tool_call_id="tc-probe",
        source="test",
    )
    assert sentinel["called_with"] == "bash"
    assert entry["tool_kind"] == "stubbed"
    monkeypatch.setattr(identity_pkg, "_classify_tool_kind", original)


def test_a4b_claude_review_monkeypatchable_through_facade(monkeypatch: Any) -> None:
    """Moved claude_review functions remain monkeypatchable via the identity
    namespace after extraction."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    sentinel = {"called": False}
    original = identity_pkg._is_claude_permission_check_metadata

    def _stub_is_pc(metadata: Any) -> bool:
        sentinel["called"] = True
        return original(metadata)

    monkeypatch.setattr(
        identity_pkg, "_is_claude_permission_check_metadata", _stub_is_pc
    )
    # _apply_claude_auto_review_identity_to_record calls
    # _is_claude_permission_check_metadata by free name
    record: Dict[str, Any] = {
        "model": "claude-sonnet-4-20250514",
        "metadata": {"claude_permission_check": True},
    }
    identity_pkg._apply_claude_auto_review_identity_to_record(record)
    assert sentinel["called"] is True
    monkeypatch.setattr(
        identity_pkg, "_is_claude_permission_check_metadata", original
    )


def test_a4b_enrich_usage_breakout_resolves_a4a_dependency_through_facade(
    monkeypatch: Any,
) -> None:
    """The enrich orchestrator resolves A4A-owned helpers via façade globals."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    calls: List[tuple] = []

    def _stub_extract_usage_object(kwargs: Dict[str, Any], result: Any) -> dict:
        calls.append((kwargs, result))
        return {
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 4},
        }

    monkeypatch.setattr(
        identity_pkg,
        "_extract_usage_object",
        _stub_extract_usage_object,
    )
    kwargs: Dict[str, Any] = {
        "model": "gemini/gemini-2.5-pro",
        "custom_llm_provider": "gemini",
        "litellm_params": {"metadata": {}},
    }
    result = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    identity_pkg._enrich_usage_breakout_metadata(kwargs, result)

    assert calls == [(kwargs, result)]
    metadata = kwargs["litellm_params"]["metadata"]
    assert metadata["usage_cache_read_input_tokens"] == 7
    assert metadata["usage_reasoning_tokens_reported"] == 4
    assert metadata["usage_reasoning_tokens_source"] == "provider_reported"
    assert metadata["gemini_cache_read_input_tokens"] == 7
    assert metadata["gemini_reasoning_tokens_reported"] == 4


# =========================================================================
# Wave A4B RED structural ownership tests (fail until engineer extracts)
# =========================================================================


def test_a4b_moved_functions_not_defined_in_init() -> None:
    """A4B-moved helper functions must not remain as FunctionDef in
    __init__.py. RED until the engineer extracts them."""
    tree = _parse_init_module()
    defined_functions = _function_def_names(tree)

    still_defined = sorted(
        name for name in _A4B_FUNCTION_NAMES if name in defined_functions
    )
    assert not still_defined, (
        "expected these A4B-moved functions to no longer be defined directly "
        f"in __init__.py (they should live in their target submodule with only "
        f"a facade assignment remaining): {still_defined}"
    )


def test_a4b_target_submodules_exist() -> None:
    """The three A4B target submodule files must exist after extraction."""
    missing = [
        mod
        for mod in _A4B_MOVED_NAMES_BY_MODULE
        if not (IDENTITY_PKG_DIR / f"{mod}.py").is_file()
    ]
    assert not missing, f"expected these A4B target submodule files to exist: {missing}"


def test_a4b_facade_identity() -> None:
    """getattr(pkg, name) is getattr(submodule, name) for every A4B-moved
    name."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    missing_modules: List[str] = []
    mismatched: List[tuple] = []
    for module_name, names in _A4B_MOVED_NAMES_BY_MODULE.items():
        try:
            submodule = importlib.import_module(
                f"litellm.integrations.aawm_agent_identity.{module_name}"
            )
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        for name in names:
            pkg_value = getattr(identity_pkg, name, None)
            sub_value = getattr(submodule, name, None)
            if pkg_value is None or sub_value is None or pkg_value is not sub_value:
                mismatched.append((module_name, name))

    assert not missing_modules, (
        f"expected these A4B target submodules to exist: {missing_modules}"
    )
    assert not mismatched, (
        "expected facade identity for each A4B-moved name; "
        f"mismatches (module, name): {mismatched}"
    )


def test_a4b_host_global_install_pattern() -> None:
    """Each A4B submodule must expose an install(globals_dict) callable for
    __globals__ rebinding, matching the A2/A4A extraction pattern."""
    missing_modules: List[str] = []
    missing_install: List[str] = []
    for module_name in _A4B_MOVED_NAMES_BY_MODULE:
        try:
            submodule = importlib.import_module(
                f"litellm.integrations.aawm_agent_identity.{module_name}"
            )
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        if not callable(getattr(submodule, "install", None)):
            missing_install.append(module_name)

    assert not missing_modules, (
        f"expected these A4B target submodules to exist: {missing_modules}"
    )
    assert not missing_install, (
        "expected each A4B submodule to expose an install(globals_dict) "
        f"callable for __globals__ rebinding: {missing_install}"
    )


def test_a4b_moved_annotations_are_runtime_objects() -> None:
    """Moved A4B annotations retain the baseline module's evaluated-object
    form (no postponed string annotations)."""
    string_annotations = []
    for module_name, names in _A4B_MOVED_NAMES_BY_MODULE.items():
        try:
            submodule = importlib.import_module(
                f"litellm.integrations.aawm_agent_identity.{module_name}"
            )
        except ModuleNotFoundError:
            continue
        for name in names:
            if name in _A4B_CONSTANT_NAMES:
                continue
            annotations = getattr(getattr(submodule, name), "__annotations__", {})
            for annotation_name, annotation_value in annotations.items():
                if isinstance(annotation_value, str):
                    string_annotations.append(
                        (module_name, name, annotation_name, annotation_value)
                    )

    assert not string_annotations, (
        "A4B moved functions must expose evaluated annotation objects, not "
        f"postponed strings: {string_annotations}"
    )


def test_a4b_type_checking_host_callable_signatures_match_runtime() -> None:  # noqa: PLR0915
    """TYPE_CHECKING host declarations exactly match live façade callables."""
    import litellm.integrations.aawm_agent_identity as identity_pkg

    annotation_namespace = dict(vars(typing))
    annotation_namespace.update(vars(builtins))
    annotation_namespace["datetime"] = datetime
    missing_modules: List[str] = []
    missing_type_checking_blocks: List[str] = []
    missing_declarations: List[tuple] = []
    missing_runtime_callables: List[tuple] = []
    annotation_errors: List[tuple] = []
    mismatches: List[tuple] = []

    def evaluate_annotation(annotation: ast.expr | None) -> Any:
        if annotation is None:
            return inspect.Signature.empty
        expression = ast.Expression(body=annotation)
        ast.fix_missing_locations(expression)
        value = eval(
            compile(expression, "<a4b-type-contract>", "eval"),
            annotation_namespace,
        )
        return type(None) if value is None else value

    def declared_parameters(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> List[inspect.Parameter]:
        arguments = function.args
        positional = [
            (argument, inspect.Parameter.POSITIONAL_ONLY)
            for argument in arguments.posonlyargs
        ] + [
            (argument, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for argument in arguments.args
        ]
        positional_defaults = [None] * (
            len(positional) - len(arguments.defaults)
        ) + list(arguments.defaults)
        parameters: List[inspect.Parameter] = []
        for (argument, kind), default_node in zip(
            positional,
            positional_defaults,
        ):
            default = (
                inspect.Parameter.empty
                if default_node is None
                else ast.literal_eval(default_node)
            )
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    kind,
                    default=default,
                    annotation=evaluate_annotation(argument.annotation),
                )
            )
        if arguments.vararg is not None:
            parameters.append(
                inspect.Parameter(
                    arguments.vararg.arg,
                    inspect.Parameter.VAR_POSITIONAL,
                    annotation=evaluate_annotation(arguments.vararg.annotation),
                )
            )
        for argument, default_node in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
        ):
            default = (
                inspect.Parameter.empty
                if default_node is None
                else ast.literal_eval(default_node)
            )
            parameters.append(
                inspect.Parameter(
                    argument.arg,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=evaluate_annotation(argument.annotation),
                )
            )
        if arguments.kwarg is not None:
            parameters.append(
                inspect.Parameter(
                    arguments.kwarg.arg,
                    inspect.Parameter.VAR_KEYWORD,
                    annotation=evaluate_annotation(arguments.kwarg.annotation),
                )
            )
        return parameters

    for module_name in _A4B_MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        if not module_path.is_file():
            missing_modules.append(module_name)
            continue

        tree = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )
        local_callable_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(
                    alias.asname or alias.name for alias in node.names
                )

        type_checking_blocks = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            )
        ]
        if not type_checking_blocks:
            missing_type_checking_blocks.append(module_name)
            continue

        declarations: Dict[
            str,
            ast.FunctionDef | ast.AsyncFunctionDef,
        ] = {}
        for block in type_checking_blocks:
            for declaration in block.body:
                if isinstance(
                    declaration,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    declarations[declaration.name] = declaration

        expected_host_callables: set[str] = set()
        for function in tree.body:
            if not isinstance(
                function,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            if function.name == "install":
                continue
            for call in ast.walk(function):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                ):
                    continue
                name = call.func.id
                if (
                    name in local_callable_names
                    or name in imported_names
                    or hasattr(builtins, name)
                ):
                    continue
                if callable(getattr(identity_pkg, name, None)):
                    expected_host_callables.add(name)

        for name in sorted(expected_host_callables - declarations.keys()):
            missing_declarations.append((module_name, name))
        if not declarations:
            missing_declarations.append(
                (module_name, "<all>", "no TYPE_CHECKING callables")
            )

        for declaration_name, declaration in declarations.items():
            runtime_callable = getattr(identity_pkg, declaration_name, None)
            if not callable(runtime_callable):
                missing_runtime_callables.append(
                    (module_name, declaration_name)
                )
                continue
            try:
                declared_signature = inspect.Signature(
                    declared_parameters(declaration),
                    return_annotation=evaluate_annotation(declaration.returns),
                )
                runtime_hints = typing.get_type_hints(
                    runtime_callable,
                    globalns=identity_pkg.__dict__,
                    localns=identity_pkg.__dict__,
                )
            except Exception as exc:
                annotation_errors.append(
                    (module_name, declaration_name, repr(exc))
                )
                continue

            runtime_signature = inspect.signature(runtime_callable).replace(
                parameters=[
                    parameter.replace(
                        annotation=runtime_hints.get(
                            parameter.name,
                            inspect.Signature.empty,
                        )
                    )
                    for parameter in inspect.signature(
                        runtime_callable
                    ).parameters.values()
                ],
                return_annotation=runtime_hints.get(
                    "return",
                    inspect.Signature.empty,
                ),
            )
            declared_async = isinstance(declaration, ast.AsyncFunctionDef)
            runtime_async = inspect.iscoroutinefunction(runtime_callable)
            if (
                declared_signature != runtime_signature
                or declared_async != runtime_async
            ):
                mismatches.append(
                    (
                        module_name,
                        declaration_name,
                        declared_signature,
                        runtime_signature,
                        declared_async,
                        runtime_async,
                    )
                )

    assert not missing_modules, (
        f"expected these A4B target submodules to exist: {missing_modules}"
    )
    assert not missing_type_checking_blocks, (
        "expected each A4B submodule to contain a TYPE_CHECKING host-contract "
        f"block: {missing_type_checking_blocks}"
    )
    assert not missing_declarations, (
        "A4B callable free-name dependencies are missing TYPE_CHECKING "
        f"declarations: {missing_declarations}"
    )
    assert not missing_runtime_callables, (
        "A4B TYPE_CHECKING declarations have no callable identity façade "
        f"target: {missing_runtime_callables}"
    )
    assert not annotation_errors, (
        "A4B TYPE_CHECKING/runtime annotations could not be resolved: "
        f"{annotation_errors}"
    )
    assert not mismatches, (
        "A4B TYPE_CHECKING host callable declarations drifted from runtime "
        "signatures or sync/async status: "
        f"{mismatches}"
    )


def test_a4b_rebind_order_facades_before_record_install() -> None:
    """Every A4B facade assignment in __init__.py must precede the
    _bind_session_history_record_apis() call."""
    tree = _parse_init_module()

    call_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_bind_session_history_record_apis"
        ):
            call_line = node.lineno
            break

    assert call_line is not None, (
        "expected a call to _bind_session_history_record_apis() in __init__.py"
    )

    defined_functions = _function_def_names(tree)
    assign_targets = _module_level_assign_targets(tree)

    facade_names_present = [
        name
        for name in _A4B_FUNCTION_NAMES
        if name in assign_targets and name not in defined_functions
    ]

    assert facade_names_present, (
        "expected at least some A4B-moved functions to appear as module-level "
        "facade assignments in __init__.py once the extraction has landed "
        "(none found -- expected RED state before the move)"
    )

    late_facades = [
        name
        for name in facade_names_present
        if assign_targets[name].lineno >= call_line
    ]
    assert not late_facades, (
        "expected every A4B facade assignment to precede the "
        f"_bind_session_history_record_apis() call (line {call_line}); "
        f"these facades assign at or after that call: {late_facades}"
    )


def test_a4b_submodules_do_not_import_init_at_module_scope() -> None:
    """A4B submodules must not import the __init__ package at module scope."""
    missing_modules: List[str] = []
    offending: List[str] = []
    for module_name in _A4B_MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        if not module_path.is_file():
            missing_modules.append(module_name)
            continue
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "litellm.integrations.aawm_agent_identity"
            ):
                offending.append(module_name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "litellm.integrations.aawm_agent_identity":
                        offending.append(module_name)

    assert not missing_modules, (
        f"expected these A4B target submodule files to exist: {missing_modules}"
    )
    assert not offending, (
        f"these A4B submodules import the identity __init__ package at "
        f"module scope: {offending}"
    )


# =========================================================================
# Wave A4B boundary guards (GREEN now, must stay GREEN): the A4B inventory
# must not claim any symbol owned by A4A, A4C, or A4D.
# =========================================================================


def test_a4b_inventory_excludes_a4a_symbols() -> None:
    overlap = sorted(set(_A4A_ALL_MOVED_NAMES) & set(_A4B_ALL_MOVED_NAMES))
    assert not overlap, f"A4B inventory must not claim A4A symbols: {overlap}"


def test_a4b_inventory_excludes_a4c_symbols() -> None:
    overlap = sorted(_A4C_SYMBOLS & set(_A4B_ALL_MOVED_NAMES))
    assert not overlap, f"A4B inventory must not claim A4C symbols: {overlap}"


def test_a4b_inventory_excludes_a4d_symbols() -> None:
    overlap = sorted(_A4D_SYMBOLS & set(_A4B_ALL_MOVED_NAMES))
    assert not overlap, f"A4B inventory must not claim A4D symbols: {overlap}"


def test_a4b_inventory_has_no_duplicate_across_modules() -> None:
    """Each A4B symbol must be owned by exactly one target module."""
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for module_name, names in _A4B_MOVED_NAMES_BY_MODULE.items():
        for name in names:
            if name in seen:
                duplicates.append(f"{name} (in {seen[name]} and {module_name})")
            else:
                seen[name] = module_name
    assert not duplicates, f"A4B symbols owned by >1 module: {duplicates}"


def test_a4b_enrich_moves_last_documented() -> None:
    """enrich.py is the high-fan-in module that moves last within A4B.
    The two orchestrators may stay as thin delegates in __init__."""
    assert "_enrich_agent_identity_metadata" in _A4B_MOVED_NAMES_BY_MODULE["enrich"]
    assert (
        "_enrich_trace_name_and_provider_metadata"
        in _A4B_MOVED_NAMES_BY_MODULE["enrich"]
    )
    # enrich.py must not claim tool_activity or claude_review primaries
    assert "_classify_tool_kind" not in _A4B_MOVED_NAMES_BY_MODULE["enrich"]
    assert (
        "_apply_claude_auto_review_metadata"
        not in _A4B_MOVED_NAMES_BY_MODULE["enrich"]
    )
