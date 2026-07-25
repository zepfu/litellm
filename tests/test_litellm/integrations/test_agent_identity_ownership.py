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
import builtins
import importlib
import inspect
import sys
import typing
from pathlib import Path
from typing import Dict, List, Optional

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
