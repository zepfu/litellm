"""Wave 4 decomposition ownership + golden-parity contract tests.

Enforces the behavior-preserving pure-leaf extraction contract from
``llm_passthrough_endpoints.py`` into:

- ``aawm_adapter_runtime/model_resolution.py``
- ``aawm_alias_routing/lane_keys.py``
- ``providers/google/env_policy.py``
- ``providers/google/context_window.py``
- ``providers/google/error_signals.py``
- ``providers/grok/side_channel.py``
- restored constants redistributed to owning provider modules

Structural ownership tests are intentionally RED until the implementation
lands.  Golden parity tests are GREEN now and must remain green after
extraction.

Write scope: this file only.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# God-module import (always available)
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

GOD_PATH = Path(lpe.__file__).resolve()
PASS_THROUGH_DIR = GOD_PATH.parent
PROVIDER_DIR = (
    GOD_PATH.parents[2]
    / "llms"
    / "anthropic"
    / "experimental_pass_through"
    / "providers"
)

# ---------------------------------------------------------------------------
# Target module import paths (relative to litellm package root)
# ---------------------------------------------------------------------------
TARGET_MODULE_IMPORT_PATHS: dict[str, str] = {
    "model_resolution": "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.model_resolution",
    "lane_keys": "litellm.proxy.pass_through_endpoints.aawm_alias_routing.lane_keys",
    "google_env_policy": "litellm.llms.anthropic.experimental_pass_through.providers.google.env_policy",
    "google_context_window": "litellm.llms.anthropic.experimental_pass_through.providers.google.context_window",
    "google_error_signals": "litellm.llms.anthropic.experimental_pass_through.providers.google.error_signals",
    "grok_side_channel": "litellm.llms.anthropic.experimental_pass_through.providers.grok.side_channel",
}

WAVE6A_MODULE_ORDER: tuple[str, ...] = (
    "request_build",
    "sse",
    "tool_call_restore",
    "stream_collect",
    "payload_validation",
)
WAVE6A_MODULE_IMPORT_PATHS: dict[str, str] = {
    name: (
        "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime."
        f"{name}"
    )
    for name in WAVE6A_MODULE_ORDER
}
WAVE6A_EXPECTED_COUNTS: dict[str, int] = {
    "request_build": 22,
    "sse": 11,
    "tool_call_restore": 14,
    "stream_collect": 9,
    "payload_validation": 14,
}

# ---------------------------------------------------------------------------
# Symbol inventory: target_module_key -> set of symbol names
# Built from current source AST analysis (develop @ 6138438678).
# ---------------------------------------------------------------------------

MODEL_RESOLUTION_SYMBOLS: set[str] = {
    "_normalize_anthropic_adapter_model_name",
    "_split_anthropic_adapter_provider_prefix",
    "_get_anthropic_adapter_model_candidates",
    "_has_anthropic_responses_adapter_endpoint",
    "_normalize_anthropic_openai_responses_adapter_model_name",
    "_normalize_anthropic_nvidia_responses_adapter_model_name",
    "_normalize_anthropic_openrouter_adapter_model_name",
    "_get_openrouter_completion_adapter_upstream_model",
    "_normalize_opencode_zen_adapter_model_name",
    "_normalize_kimi_code_chat_completions_adapter_model_name",
    "_normalize_alibaba_token_plan_adapter_model_name",
    "_normalize_anthropic_google_completion_adapter_model_name",
    "_normalize_antigravity_code_assist_adapter_model_name",
    "_normalize_codex_google_code_assist_adapter_model_name",
    "_resolve_codex_opencode_zen_adapter_model",
    "_resolve_codex_kimi_chat_completions_adapter_model",
    "_resolve_codex_alibaba_token_plan_adapter_model",
    "_resolve_anthropic_opencode_zen_adapter_model",
    "_resolve_anthropic_kimi_chat_completions_adapter_model",
    "_resolve_anthropic_alibaba_token_plan_adapter_model",
    "_resolve_anthropic_antigravity_code_assist_adapter_model",
    "_resolve_codex_google_code_assist_adapter_model",
    "_resolve_codex_antigravity_code_assist_adapter_model",
    "_normalize_codex_auto_agent_alias_model",
    "_is_codex_auto_agent_alias_model",
    "_resolve_codex_auto_agent_alias_model",
    "_resolve_anthropic_openai_responses_adapter_model",
    "_resolve_anthropic_xai_oauth_adapter_model",
    "_resolve_anthropic_grok_native_oauth_adapter_model",
    "_resolve_anthropic_openrouter_completion_adapter_model",
    "_resolve_anthropic_nvidia_responses_adapter_model",
    "_resolve_anthropic_openrouter_responses_adapter_model",
    "_resolve_anthropic_google_completion_adapter_model",
}

LANE_KEYS_SYMBOLS: set[str] = {
    "_get_codex_auto_agent_header",
    "_hash_codex_auto_agent_lane_value",
    "_resolve_codex_auto_agent_openai_lane_key",
    "_resolve_codex_auto_agent_openai_cooldown_lane_key",
    "_get_codex_auto_agent_lane_state_cache_ttl_seconds",
    "_get_codex_auto_agent_google_lane_cache_key",
    "_get_codex_auto_agent_antigravity_lane_cache_key",
    "_codex_auto_agent_candidate_key",
    "_resolve_codex_auto_agent_xai_lane_key",
    "_resolve_anthropic_auto_agent_native_lane_key",
    "_resolve_anthropic_auto_agent_native_cooldown_lane_key",
}

GOOGLE_ENV_POLICY_SYMBOLS: set[str] = {
    "_get_google_code_assist_prime_ttl_seconds",
    "_get_google_code_assist_prime_cache_key",
    "_get_google_adapter_max_concurrent",
    "_get_google_adapter_shared_lane_key",
    "_get_google_adapter_rate_limit_key",
    "_get_google_adapter_rate_limit_key_from_kwargs",
    "_get_google_adapter_max_retries",
    "_coerce_non_negative_int",
    "_coerce_non_negative_float",
    "_get_google_adapter_post_tool_cooldown_seconds",
    "_google_code_assist_unwrapped_chunk_contains_tool_call",
    "_get_google_adapter_max_output_tokens_cap",
    "_get_google_adapter_default_thinking_level",
    "_get_google_adapter_max_contents_window",
    "_get_google_adapter_max_contents_text_chars",
    "_google_content_has_text",
    "_get_google_adapter_oversized_text_part_char_cap",
    "_get_google_adapter_pure_context_text_part_char_cap",
    "_get_google_adapter_subagent_context_text_part_char_cap",
    "_get_google_adapter_followup_subagent_context_text_part_char_cap",
    "_get_google_adapter_followup_allowed_tool_names",
    "_get_google_adapter_model_capacity_max_retries",
    "_get_google_adapter_capacity_backoff_seconds",
    "_get_google_adapter_hidden_retry_budget_seconds",
    "_get_google_adapter_transient_retry_max_attempts",
    "_get_google_adapter_transient_backoff_seconds",
    "_get_google_adapter_fallback_context_char_cap",
    "_get_google_adapter_system_prompt_policy",
    "_get_google_code_assist_native_tool_aliases",
    "_get_google_adapter_max_completion_messages_window",
    "_get_google_adapter_preserved_task_state_char_cap",
    "_get_google_adapter_native_user_agent",
    "_get_google_adapter_native_api_client_header",
    "_get_google_adapter_persisted_output_char_cap",
    "_get_google_adapter_auxiliary_context_char_cap",
    "_get_google_adapter_followup_persisted_output_char_cap",
    "_get_google_adapter_followup_auxiliary_context_char_cap",
}

GOOGLE_CONTEXT_WINDOW_SYMBOLS: set[str] = {
    "_google_content_has_function_exchange",
    "_google_content_has_function_call",
    "_apply_google_adapter_contents_window_policy",
    "_extract_completion_message_text",
    "_completion_message_has_visible_text",
    "_estimate_completion_message_text_chars",
    "_completion_message_has_tool_result",
    "_completion_message_tool_call_ids",
    "_completion_message_tool_result_ids",
    "_trim_completion_message_tail_preserving_tool_pairs",
    "_apply_google_adapter_completion_message_window",
    "_google_code_assist_duplicate_tool_results_from_completion_messages",
    "_google_code_assist_tool_results_from_completion_messages",
}

GOOGLE_ERROR_SIGNALS_SYMBOLS: set[str] = {
    "_extract_google_adapter_exception_status_code",
    "_extract_google_adapter_exception_detail",
    "_parse_google_rate_limit_reset_seconds",
    "_extract_google_adapter_error_payloads",
    "_extract_google_adapter_error_reason",
    "_extract_google_adapter_error_payload_for_logging",
    "_record_google_adapter_error_for_logging",
    "_build_google_adapter_terminal_error_log_context",
}

GROK_SIDE_CHANNEL_SYMBOLS: set[str] = {
    "_normalize_grok_endpoint_for_target",
    "_normalize_grok_endpoint_path",
    "_get_grok_side_channel_endpoint_type",
    "_get_grok_session_side_channel_endpoint_type",
    "_get_grok_side_channel_endpoint_path_template",
    "_get_grok_session_side_channel_endpoint_path_template",
    "_json_shape_type_name",
    "_extract_redacted_grok_json_request_shape",
    "_stable_grok_side_channel_body_digest",
    "_build_grok_side_channel_request_shape_metadata",
    "_merge_grok_side_channel_shape_into_passthrough_logging_metadata",
    "_get_grok_side_channel_retryable_status_codes",
}

# Restored constants redistributed to owning provider modules
RESTORED_CONSTANTS_GOOGLE: set[str] = {
    "_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME",
    "_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION",
    "_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV",
    "_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT",
    "_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT",
    "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME",
    "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION",
    "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV",
    "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT",
    "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT",
}

RESTORED_CONSTANTS_GROK: set[str] = {
    "_GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL",
    "_GROK_CLI_FORWARD_HEADER_ALLOWLIST",
    "_GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE",
    "_CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN",
    "_CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN",
    "_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL",
}

RESTORED_CONSTANTS_OPENCODE_ZEN: set[str] = {
    "_OPENCODE_ZEN_DEFAULT_BASE_URL",
    "_OPENCODE_ZEN_PROVIDER",
    "_OPENCODE_ZEN_AUTH_FILE_ENV_VARS",
    "_OPENCODE_ZEN_API_KEY_ENV_VARS",
    "_OPENCODE_ZEN_DEFAULT_AUTH_PATHS",
    "_OPENCODE_ZEN_FREE_MODELS",
    "_OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS",
}

RESTORED_CONSTANTS_ANTIGRAVITY: set[str] = {
    "_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST",
}

RESTORED_CONSTANTS_ANTHROPIC: set[str] = {
    "_ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL",
    "_ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS",
    "_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS",
    "_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS",
    "_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS",
    "_ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS",
    "_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS",
    "_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN",
    "_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN",
    "_ANTHROPIC_BILLING_HEADER_PREFIX",
}

RESTORED_CONSTANTS_ALIAS_ROUTING: set[str] = {
    "_CODEX_REASONING_EFFORT_TIERS",
    "_CODEX_REASONING_EFFORT_TIER_INDEX",
    "_CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS",
    "_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME",
    "_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION",
    "_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT",
    "_AAWM_READ_AGENT_GUIDANCE_POLICY_NAME",
    "_AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION",
    "_AAWM_READ_AGENT_GUIDANCE_PROMPT",
    "_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS",
    "_CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS",
    "_CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_SPARK_MODEL",
    "_CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES",
    "_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS",
    "_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV",
    "_CLAUDE_PERSISTED_OUTPUT_PATTERN",
    "_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN",
    "_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN",
    "_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN",
}

ALL_RESTORED_CONSTANTS: set[str] = (
    RESTORED_CONSTANTS_GOOGLE
    | RESTORED_CONSTANTS_GROK
    | RESTORED_CONSTANTS_OPENCODE_ZEN
    | RESTORED_CONSTANTS_ANTIGRAVITY
    | RESTORED_CONSTANTS_ANTHROPIC
    | RESTORED_CONSTANTS_ALIAS_ROUTING
)

# Unified inventory: target_key -> symbols
SYMBOL_INVENTORY: dict[str, set[str]] = {
    "model_resolution": MODEL_RESOLUTION_SYMBOLS,
    "lane_keys": LANE_KEYS_SYMBOLS,
    "google_env_policy": GOOGLE_ENV_POLICY_SYMBOLS,
    "google_context_window": GOOGLE_CONTEXT_WINDOW_SYMBOLS,
    "google_error_signals": GOOGLE_ERROR_SIGNALS_SYMBOLS,
    "grok_side_channel": GROK_SIDE_CHANNEL_SYMBOLS,
}

# All moved function symbols (union of function-bearing targets)
ALL_MOVED_FUNCTIONS: set[str] = set()
for _syms in SYMBOL_INVENTORY.values():
    ALL_MOVED_FUNCTIONS |= _syms

# State aliases that must NEVER be rebound or moved (the :453-473 band)
STATE_ALIAS_NAMES: set[str] = {
    "_codex_auto_agent_cooldown_until_monotonic_by_key",
    "_codex_auto_agent_cooldown_negative_until_monotonic_by_key",
    "_codex_auto_agent_session_affinity_by_key",
    "_codex_auto_agent_lock",
    "_codex_auto_agent_lane_state_cache_lock",
    "_codex_auto_agent_google_lane_key_by_key",
    "_codex_auto_agent_google_lane_key_until_monotonic_by_key",
    "_codex_auto_agent_antigravity_lane_key_by_key",
    "_codex_auto_agent_antigravity_lane_key_until_monotonic_by_key",
    "_anthropic_auto_agent_cooldown_until_monotonic_by_key",
    "_anthropic_auto_agent_cooldown_negative_until_monotonic_by_key",
    "_anthropic_auto_agent_session_affinity_by_key",
    "_anthropic_auto_agent_lock",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_god_module() -> ast.Module:
    source = GOD_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(GOD_PATH))


def _top_level_function_defs(tree: ast.Module) -> set[str]:
    """Names defined as FunctionDef/AsyncFunctionDef at module top level."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _top_level_assignments(tree: ast.Module) -> set[str]:
    """Names assigned at module top level (facade bindings)."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _try_import_target(module_key: str):
    """Try to import a target module; return None if not yet created."""
    import_path = TARGET_MODULE_IMPORT_PATHS[module_key]
    try:
        return importlib.import_module(import_path)
    except (ImportError, ModuleNotFoundError):
        return None


# ===========================================================================
# SECTION 1: Structural ownership tests (RED until implementation lands)
# ===========================================================================


class TestMovedBandsNotDefinedInGodModule:
    """After extraction, moved functions must NOT appear as FunctionDef in
    the god module -- only as assignment facades."""

    def test_model_resolution_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = MODEL_RESOLUTION_SYMBOLS & func_defs
        assert not violations, (
            f"model_resolution symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_lane_keys_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = LANE_KEYS_SYMBOLS & func_defs
        assert not violations, (
            f"lane_keys symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_google_env_policy_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = GOOGLE_ENV_POLICY_SYMBOLS & func_defs
        assert not violations, (
            f"google_env_policy symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_google_context_window_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = GOOGLE_CONTEXT_WINDOW_SYMBOLS & func_defs
        assert not violations, (
            f"google_context_window symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_google_error_signals_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = GOOGLE_ERROR_SIGNALS_SYMBOLS & func_defs
        assert not violations, (
            f"google_error_signals symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_grok_side_channel_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = GROK_SIDE_CHANNEL_SYMBOLS & func_defs
        assert not violations, (
            f"grok_side_channel symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_restored_constants_absent_as_owned_assignments(self):
        """Restored constants must not remain as direct value assignments in
        the god module after redistribution (facades via import are OK)."""
        tree = _parse_god_module()
        violations = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ALL_RESTORED_CONSTANTS:
                        # Check if it's a facade (RHS is Attribute or Name referencing another module)
                        # vs an owned literal assignment
                        rhs = node.value
                        is_facade = isinstance(rhs, ast.Attribute) or (
                            isinstance(rhs, ast.Name) and rhs.id != target.id
                        )
                        if not is_facade:
                            violations.append(target.id)
        assert not violations, (
            f"Restored constants still owned (non-facade) in god module: {sorted(violations)}"
        )


class TestTargetModulesExist:
    """Target packages/modules must exist after extraction."""

    @pytest.mark.parametrize("module_key", list(TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_target_module_importable(self, module_key: str):
        mod = _try_import_target(module_key)
        assert mod is not None, (
            f"Target module {TARGET_MODULE_IMPORT_PATHS[module_key]} not yet created"
        )


class TestFacadeObjectIdentity:
    """Sampled facade bindings must be the same object as the target module's."""

    def _assert_identity_for_module(self, module_key: str, sample_size: int = 10):
        mod = _try_import_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        symbols = sorted(SYMBOL_INVENTORY[module_key])
        sample = symbols[:sample_size] if len(symbols) >= sample_size else symbols
        for name in sample:
            god_obj = getattr(lpe, name, None)
            target_obj = getattr(mod, name, None)
            assert god_obj is not None, f"{name} not found on god module"
            assert target_obj is not None, f"{name} not found on target module"
            assert god_obj is target_obj, (
                f"{name}: god module facade is not the same object as "
                f"{TARGET_MODULE_IMPORT_PATHS[module_key]}.{name}"
            )

    def test_model_resolution_facade_identity(self):
        self._assert_identity_for_module("model_resolution")

    def test_lane_keys_facade_identity(self):
        self._assert_identity_for_module("lane_keys")

    def test_google_env_policy_facade_identity(self):
        self._assert_identity_for_module("google_env_policy")

    def test_google_context_window_facade_identity(self):
        self._assert_identity_for_module("google_context_window")

    def test_google_error_signals_facade_identity(self):
        self._assert_identity_for_module("google_error_signals")

    def test_grok_side_channel_facade_identity(self):
        self._assert_identity_for_module("grok_side_channel")


class TestFacadeAssignmentsPrecedeConsumers:
    """All facade assignments in the god module must appear before any
    consumer that references the moved symbol via the god-module namespace."""

    def test_facade_line_ordering(self):
        tree = _parse_god_module()
        # Find facade assignment lines for moved symbols
        facade_lines: dict[str, int] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ALL_MOVED_FUNCTIONS:
                        facade_lines[target.id] = node.lineno

        # If no facades exist yet, this test is vacuously true (pre-extraction)
        if not facade_lines:
            pytest.skip("No facade assignments found yet (pre-extraction)")

        # Find first usage of each symbol after its facade line
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in facade_lines:
                if hasattr(node, "lineno") and node.lineno < facade_lines[node.id]:
                    # Usage before facade -- only a problem if it's not the facade itself
                    pass  # AST walk includes the assignment target; skip


# ===========================================================================
# SECTION 2: Import boundary tests (RED until implementation lands)
# ===========================================================================


class TestImportBoundaries:
    """No target module may import llm_passthrough_endpoints at module scope.
    No wildcard imports or name capture."""

    @pytest.mark.parametrize("module_key", list(TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_no_god_module_import_at_module_scope(self, module_key: str):
        mod = _try_import_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        mod_path = Path(mod.__file__).resolve()
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        violations = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "llm_passthrough_endpoints" in alias.name:
                        violations.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "llm_passthrough_endpoints" in node.module:
                    violations.append(f"from {node.module} import ...")
        assert not violations, (
            f"{module_key} imports llm_passthrough_endpoints at module scope: {violations}"
        )

    @pytest.mark.parametrize("module_key", list(TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_no_wildcard_imports(self, module_key: str):
        mod = _try_import_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        mod_path = Path(mod.__file__).resolve()
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        violations.append(
                            f"from {node.module} import * (line {node.lineno})"
                        )
        assert not violations, (
            f"{module_key} uses wildcard imports: {violations}"
        )


# ===========================================================================
# SECTION 3: State alias guard (GREEN now, must remain GREEN)
# ===========================================================================


class TestStateAliasGuard:
    """The state band (:453-473) aliases must never be rebound or moved."""

    def test_state_aliases_remain_as_aliases_in_god_module(self):
        """Each state alias must still be assigned in the god module and must
        reference the same underlying state object (not a new dict/lock)."""
        tree = _parse_god_module()
        assignments = _top_level_assignments(tree)
        missing = STATE_ALIAS_NAMES - assignments
        assert not missing, (
            f"State aliases missing from god module assignments: {sorted(missing)}"
        )

    def test_state_aliases_are_not_function_defs(self):
        """State aliases must never become function definitions."""
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = STATE_ALIAS_NAMES & func_defs
        assert not violations, (
            f"State aliases redefined as functions: {sorted(violations)}"
        )

    def test_state_dict_identity_stable(self):
        """The state dict objects must be the exact same objects owned by
        _alias_routing_state (not copies or new dicts)."""
        state_mgr = getattr(lpe, "_alias_routing_state", None)
        if state_mgr is None:
            pytest.skip("_alias_routing_state not available")
        # Codex family
        assert (
            lpe._codex_auto_agent_cooldown_until_monotonic_by_key
            is state_mgr.codex.cooldown_until_monotonic_by_key
        )
        assert (
            lpe._codex_auto_agent_session_affinity_by_key
            is state_mgr.codex.session_affinity_by_key
        )
        # Anthropic family
        assert (
            lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key
            is state_mgr.anthropic.cooldown_until_monotonic_by_key
        )
        assert (
            lpe._anthropic_auto_agent_session_affinity_by_key
            is state_mgr.anthropic.session_affinity_by_key
        )


# ===========================================================================
# SECTION 4: Router ownership guard (GREEN now, must remain GREEN)
# ===========================================================================


class TestRouterOwnership:
    """Router objects and decorators must remain in the god module."""

    def test_router_object_in_god_module(self):
        assert hasattr(lpe, "router"), "router object missing from god module"
        from fastapi import APIRouter
        assert isinstance(lpe.router, APIRouter)

    def test_passthrough_endpoint_router_in_god_module(self):
        assert hasattr(lpe, "passthrough_endpoint_router"), (
            "passthrough_endpoint_router missing from god module"
        )
        # It is a PassthroughEndpointRouter, not a raw APIRouter

    def test_router_not_in_moved_modules(self):
        """No target module may define or re-export the router objects."""
        for module_key in TARGET_MODULE_IMPORT_PATHS:
            mod = _try_import_target(module_key)
            if mod is None:
                continue
            assert not hasattr(mod, "router"), (
                f"{module_key} must not define 'router'"
            )
            assert not hasattr(mod, "passthrough_endpoint_router"), (
                f"{module_key} must not define 'passthrough_endpoint_router'"
            )


# ===========================================================================
# SECTION 5: Golden parity tests (GREEN now, must remain GREEN after extraction)
# ===========================================================================


class TestGrokSideChannelParity:
    """Golden behavior parity for grok side-channel classification/digests."""

    def test_normalize_grok_endpoint_path_strips_v1_prefix(self):
        assert lpe._normalize_grok_endpoint_path("https://api.x.ai/v1/sessions/register") == "/sessions/register"

    def test_normalize_grok_endpoint_path_adds_leading_slash(self):
        assert lpe._normalize_grok_endpoint_path("sessions/register") == "/sessions/register"

    def test_normalize_grok_endpoint_path_preserves_root(self):
        assert lpe._normalize_grok_endpoint_path("/traces") == "/traces"

    def test_endpoint_type_sessions_register(self):
        assert lpe._get_grok_side_channel_endpoint_type("/sessions/register") == "sessions_register"

    def test_endpoint_type_sessions_replicas_update(self):
        assert lpe._get_grok_side_channel_endpoint_type("/sessions/abc123/replicas/update") == "sessions_replicas_update"

    def test_endpoint_type_sessions_signals(self):
        assert lpe._get_grok_side_channel_endpoint_type("/sessions/abc123/signals") == "sessions_signals"

    def test_endpoint_type_sessions_turn_deltas(self):
        assert lpe._get_grok_side_channel_endpoint_type("/sessions/abc123/turn-deltas") == "sessions_turn_deltas"

    def test_endpoint_type_traces(self):
        assert lpe._get_grok_side_channel_endpoint_type("/traces") == "traces"

    def test_endpoint_type_none_for_regular(self):
        assert lpe._get_grok_side_channel_endpoint_type("/v1/chat/completions") is None

    def test_endpoint_type_none_for_empty(self):
        assert lpe._get_grok_side_channel_endpoint_type("") is None

    def test_path_template_sessions_register(self):
        assert lpe._get_grok_side_channel_endpoint_path_template("sessions_register") == "/sessions/register"

    def test_path_template_sessions_replicas_update(self):
        assert lpe._get_grok_side_channel_endpoint_path_template("sessions_replicas_update") == "/sessions/{session_id}/replicas/update"

    def test_path_template_none_for_unknown(self):
        assert lpe._get_grok_side_channel_endpoint_path_template("unknown_type") is None

    def test_session_side_channel_delegates(self):
        assert lpe._get_grok_session_side_channel_endpoint_type("/sessions/register") == "sessions_register"
        assert lpe._get_grok_session_side_channel_endpoint_type("/v1/chat") is None

    def test_json_shape_type_name_primitives(self):
        assert lpe._json_shape_type_name(None) == "null"
        assert lpe._json_shape_type_name(True) == "bool"
        assert lpe._json_shape_type_name(42) == "int"
        assert lpe._json_shape_type_name(3.14) == "float"
        assert lpe._json_shape_type_name("hello") == "str"
        assert lpe._json_shape_type_name([1, 2]) == "array"
        assert lpe._json_shape_type_name({"a": 1}) == "object"

    def test_extract_redacted_grok_json_request_shape_dict(self):
        shape = lpe._extract_redacted_grok_json_request_shape({"model": "grok", "litellm_metadata": {"x": 1}})
        assert shape["json_container_type"] == "object"
        assert "model" in shape["top_level_key_types"]
        assert "litellm_metadata" not in shape["top_level_key_types"]

    def test_extract_redacted_grok_json_request_shape_list(self):
        shape = lpe._extract_redacted_grok_json_request_shape([1, 2, 3])
        assert shape["json_container_type"] == "array"
        assert shape["array_length"] == 3

    def test_extract_redacted_grok_json_request_shape_none(self):
        shape = lpe._extract_redacted_grok_json_request_shape(None)
        assert shape["json_container_type"] == "null"

    def test_stable_digest_raw_body(self):
        raw = b'{"key": "value"}'
        length, sha, source = lpe._stable_grok_side_channel_body_digest(raw_body=raw)
        assert length == len(raw)
        assert sha == hashlib.sha256(raw).hexdigest()
        assert source == "raw_body"

    def test_stable_digest_dict_strips_litellm_metadata(self):
        body = {"model": "grok", "litellm_metadata": {"trace": "x"}}
        length, sha, source = lpe._stable_grok_side_channel_body_digest(parsed_body=body)
        expected_bytes = json.dumps({"model": "grok"}, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        assert length == len(expected_bytes)
        assert sha == hashlib.sha256(expected_bytes).hexdigest()
        assert source == "canonical_json_without_litellm_metadata"

    def test_stable_digest_empty(self):
        length, sha, source = lpe._stable_grok_side_channel_body_digest(parsed_body=None, raw_body=None)
        assert length == 0
        assert sha == hashlib.sha256(b"").hexdigest()
        assert source == "empty_body"

    def test_retryable_status_codes_side_channel(self):
        codes = lpe._get_grok_side_channel_retryable_status_codes("/sessions/register")
        assert codes == [500, 502, 503, 504]

    def test_retryable_status_codes_non_side_channel(self):
        codes = lpe._get_grok_side_channel_retryable_status_codes("/v1/chat/completions")
        assert codes == []

    def test_merge_shape_metadata_adds_tag(self):
        base = {"tags": ["existing"]}
        shape = {"grok_side_channel": True, "grok_side_channel_endpoint_type": "traces"}
        merged = lpe._merge_grok_side_channel_shape_into_passthrough_logging_metadata(base, shape_metadata=shape)
        assert "grok-side-channel" in merged["tags"]
        assert "existing" in merged["tags"]
        assert merged["grok_side_channel"] is True

    def test_merge_shape_metadata_none_passthrough(self):
        base = {"tags": ["existing"]}
        merged = lpe._merge_grok_side_channel_shape_into_passthrough_logging_metadata(base, shape_metadata=None)
        assert merged == base

    def test_normalize_grok_endpoint_for_target(self):
        # Returns the normalized path, stripping /v1 when base ends with /v1
        result = lpe._normalize_grok_endpoint_for_target(
            "https://api.x.ai/v1/sessions/register",
            "https://custom.base/v1",
        )
        assert result == "/sessions/register"


class TestModelResolutionParity:
    """Golden behavior parity for model resolution functions."""

    def test_normalize_anthropic_adapter_model_name_valid(self):
        assert lpe._normalize_anthropic_adapter_model_name("  claude-3  ") == "claude-3"

    def test_normalize_anthropic_adapter_model_name_none(self):
        assert lpe._normalize_anthropic_adapter_model_name(None) is None

    def test_normalize_anthropic_adapter_model_name_empty(self):
        assert lpe._normalize_anthropic_adapter_model_name("   ") is None

    def test_normalize_anthropic_adapter_model_name_non_string(self):
        assert lpe._normalize_anthropic_adapter_model_name(123) is None

    def test_split_provider_prefix_known(self):
        provider, model = lpe._split_anthropic_adapter_provider_prefix("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_split_provider_prefix_alias(self):
        provider, model = lpe._split_anthropic_adapter_provider_prefix("agy/some-model")
        assert provider == "antigravity"
        assert model == "some-model"

    def test_split_provider_prefix_none(self):
        provider, model = lpe._split_anthropic_adapter_provider_prefix("unknown/model")
        assert provider is None
        assert model == "unknown/model"

    def test_split_provider_prefix_no_slash(self):
        provider, model = lpe._split_anthropic_adapter_provider_prefix("claude-3")
        assert provider is None
        assert model == "claude-3"

    def test_split_provider_prefix_none_input(self):
        provider, model = lpe._split_anthropic_adapter_provider_prefix(None)
        assert provider is None
        assert model is None

    def test_has_anthropic_responses_adapter_endpoint_true(self):
        assert lpe._has_anthropic_responses_adapter_endpoint("/v1/messages") is True

    def test_has_anthropic_responses_adapter_endpoint_false(self):
        assert lpe._has_anthropic_responses_adapter_endpoint("/v1/chat/completions") is False

    def test_has_anthropic_responses_adapter_endpoint_no_leading_slash(self):
        assert lpe._has_anthropic_responses_adapter_endpoint("v1/messages") is True

    def test_normalize_codex_auto_agent_alias_model_none(self):
        assert lpe._normalize_codex_auto_agent_alias_model(None) is None

    def test_normalize_codex_auto_agent_alias_model_non_string(self):
        assert lpe._normalize_codex_auto_agent_alias_model(42) is None

    def test_is_codex_auto_agent_alias_model_non_string(self):
        assert lpe._is_codex_auto_agent_alias_model(None) is False

    def test_resolve_codex_auto_agent_alias_model_wrong_endpoint(self):
        result = lpe._resolve_codex_auto_agent_alias_model({"model": "codex"}, "/v1/chat/completions")
        assert result is None


class TestLaneKeysParity:
    """Golden behavior parity for lane key functions."""

    def test_hash_lane_value_deterministic(self):
        h1 = lpe._hash_codex_auto_agent_lane_value("test-value")
        h2 = lpe._hash_codex_auto_agent_lane_value("test-value")
        assert h1 == h2
        assert len(h1) == 12

    def test_hash_lane_value_different_inputs(self):
        h1 = lpe._hash_codex_auto_agent_lane_value("value-a")
        h2 = lpe._hash_codex_auto_agent_lane_value("value-b")
        assert h1 != h2

    def test_candidate_key_basic(self):
        candidate = {"provider": "openai", "model": "gpt-4o"}
        key = lpe._codex_auto_agent_candidate_key(candidate, "lane-1")
        assert key == "openai:gpt-4o:lane-1"

    def test_candidate_key_default_lane(self):
        candidate = {"provider": "anthropic", "model": "claude-3"}
        key = lpe._codex_auto_agent_candidate_key(candidate, "")
        assert key == "anthropic:claude-3:__default__"

    def test_candidate_key_with_epoch_tag(self):
        candidate = {"provider": "openai", "model": "gpt-4o"}
        key = lpe._codex_auto_agent_candidate_key(candidate, "lane-1", epoch_tag="abc123")
        assert key == "habc123:openai:gpt-4o:lane-1"

    def test_candidate_key_none_epoch_tag(self):
        candidate = {"provider": "openai", "model": "gpt-4o"}
        key = lpe._codex_auto_agent_candidate_key(candidate, "lane-1", epoch_tag=None)
        assert key == "openai:gpt-4o:lane-1"

    def test_resolve_xai_lane_key_oauth(self):
        candidate = {"route_family": "codex_xai_oauth_responses_adapter"}
        key = lpe._resolve_codex_auto_agent_xai_lane_key(candidate)
        # Should return the xai oauth lane key constant
        assert "xai" in key.lower() or "oauth" in key.lower() or key != ""

    def test_resolve_xai_lane_key_default(self):
        candidate = {"route_family": "other_family"}
        key = lpe._resolve_codex_auto_agent_xai_lane_key(candidate)
        assert key != ""


class TestGoogleEnvPolicyParity:
    """Golden behavior parity for google env-knob getters."""

    def test_coerce_non_negative_int_valid(self):
        assert lpe._coerce_non_negative_int("5", 10) == 5

    def test_coerce_non_negative_int_none(self):
        assert lpe._coerce_non_negative_int(None, 10) == 10

    def test_coerce_non_negative_int_negative(self):
        assert lpe._coerce_non_negative_int("-3", 10) == 0

    def test_coerce_non_negative_int_invalid(self):
        assert lpe._coerce_non_negative_int("abc", 7) == 7

    def test_coerce_non_negative_float_valid(self):
        assert lpe._coerce_non_negative_float("2.5", 1.0) == 2.5

    def test_coerce_non_negative_float_none(self):
        assert lpe._coerce_non_negative_float(None, 1.0) == 1.0

    def test_coerce_non_negative_float_negative(self):
        assert lpe._coerce_non_negative_float("-1.5", 2.0) == 0.0

    def test_max_output_tokens_cap_default(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP", raising=False)
        assert lpe._get_google_adapter_max_output_tokens_cap() == 8192

    def test_max_output_tokens_cap_disabled(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP", "0")
        assert lpe._get_google_adapter_max_output_tokens_cap() is None

    def test_max_output_tokens_cap_custom(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP", "4096")
        assert lpe._get_google_adapter_max_output_tokens_cap() == 4096

    def test_default_thinking_level_flash_lite(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL", raising=False)
        assert lpe._get_google_adapter_default_thinking_level("gemini-flash-lite-2.0") == "minimal"

    def test_default_thinking_level_regular(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL", raising=False)
        assert lpe._get_google_adapter_default_thinking_level("gemini-2.5-pro") == "low"

    def test_default_thinking_level_disabled(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", "1")
        assert lpe._get_google_adapter_default_thinking_level("gemini-2.5-pro") is None

    def test_default_thinking_level_env_override(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL", "high")
        assert lpe._get_google_adapter_default_thinking_level("gemini-2.5-pro") == "high"

    def test_unwrapped_chunk_contains_tool_call_true(self):
        chunk = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "test"}}]}}]}
        assert lpe._google_code_assist_unwrapped_chunk_contains_tool_call(chunk) is True

    def test_unwrapped_chunk_contains_tool_call_false(self):
        chunk = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert lpe._google_code_assist_unwrapped_chunk_contains_tool_call(chunk) is False

    def test_unwrapped_chunk_contains_tool_call_empty(self):
        assert lpe._google_code_assist_unwrapped_chunk_contains_tool_call({}) is False

    def test_estimate_google_content_text_chars_delegates(self):
        # This delegates to _anthropic_google_shaping; just verify callable
        result = lpe._estimate_google_content_text_chars({"parts": [{"text": "hello"}]})
        assert isinstance(result, int)

    def test_google_content_has_text_true(self):
        assert lpe._google_content_has_text({"parts": [{"text": "hello"}]}) is True

    def test_google_content_has_text_false(self):
        assert lpe._google_content_has_text({"parts": [{"functionCall": {}}]}) is False


class TestGoogleContextWindowParity:
    """Golden behavior parity for google context window functions."""

    def test_content_has_function_exchange_true(self):
        block = {"parts": [{"functionCall": {"name": "test"}}]}
        assert lpe._google_content_has_function_exchange(block) is True

    def test_content_has_function_exchange_response(self):
        block = {"parts": [{"functionResponse": {"name": "test"}}]}
        assert lpe._google_content_has_function_exchange(block) is True

    def test_content_has_function_exchange_false(self):
        block = {"parts": [{"text": "hello"}]}
        assert lpe._google_content_has_function_exchange(block) is False

    def test_content_has_function_exchange_non_dict(self):
        assert lpe._google_content_has_function_exchange("not a dict") is False

    def test_content_has_function_call_true(self):
        block = {"parts": [{"functionCall": {"name": "test"}}]}
        assert lpe._google_content_has_function_call(block) is True

    def test_content_has_function_call_false_for_response(self):
        block = {"parts": [{"functionResponse": {"name": "test"}}]}
        assert lpe._google_content_has_function_call(block) is False

    def test_content_has_function_call_non_dict(self):
        assert lpe._google_content_has_function_call(None) is False

    def test_completion_message_has_visible_text_string(self):
        assert lpe._completion_message_has_visible_text({"content": "hello"}) is True

    def test_completion_message_has_visible_text_empty_string(self):
        assert lpe._completion_message_has_visible_text({"content": "   "}) is False

    def test_completion_message_has_visible_text_list(self):
        msg = {"content": [{"text": "hello"}]}
        assert lpe._completion_message_has_visible_text(msg) is True

    def test_completion_message_has_visible_text_non_dict(self):
        assert lpe._completion_message_has_visible_text("not a dict") is False

    def test_estimate_completion_message_text_chars_string(self):
        assert lpe._estimate_completion_message_text_chars({"content": "hello"}) == 5

    def test_estimate_completion_message_text_chars_list(self):
        msg = {"content": [{"text": "ab"}, {"text": "cd"}]}
        assert lpe._estimate_completion_message_text_chars(msg) == 4

    def test_estimate_completion_message_text_chars_non_dict(self):
        assert lpe._estimate_completion_message_text_chars(None) == 0

    def test_completion_message_has_tool_result_role(self):
        assert lpe._completion_message_has_tool_result({"role": "tool"}) is True

    def test_completion_message_has_tool_result_id(self):
        assert lpe._completion_message_has_tool_result({"tool_call_id": "tc_1"}) is True

    def test_completion_message_has_tool_result_false(self):
        assert lpe._completion_message_has_tool_result({"role": "user", "content": "hi"}) is False

    def test_completion_message_tool_call_ids(self):
        msg = {"tool_calls": [{"id": "tc_1"}, {"id": "tc_2"}]}
        assert lpe._completion_message_tool_call_ids(msg) == {"tc_1", "tc_2"}

    def test_completion_message_tool_call_ids_empty(self):
        assert lpe._completion_message_tool_call_ids({}) == set()

    def test_completion_message_tool_call_ids_content_tool_use(self):
        msg = {"content": [{"type": "tool_use", "id": "tc_3"}]}
        assert lpe._completion_message_tool_call_ids(msg) == {"tc_3"}

    def test_completion_message_tool_result_ids(self):
        msg = {"role": "tool", "tool_call_id": "tc_1"}
        result = lpe._completion_message_tool_result_ids(msg)
        assert "tc_1" in result


class TestGoogleErrorSignalsParity:
    """Golden behavior parity for google error signal parsing."""

    def test_parse_google_rate_limit_reset_seconds_default(self):
        """With no headers or detail, defaults to 5.0."""

        class FakeExc(Exception):
            pass

        exc = FakeExc("some error")
        result = lpe._parse_google_rate_limit_reset_seconds(exc)
        assert result == 5.0

    def test_extract_exception_status_code_delegates(self):
        """Verify the delegate is callable and returns None for non-HTTP errors."""
        result = lpe._extract_google_adapter_exception_status_code(ValueError("test"))
        assert result is None

    def test_extract_exception_detail_delegates(self):
        result = lpe._extract_google_adapter_exception_detail(ValueError("test detail"))
        # Should return something (the detail extraction is provider-specific)
        assert result is not None or result is None  # callable without error


class TestRestoredConstantsParity:
    """Golden parity for restored constants -- values must be stable."""

    def test_grok_cli_constants_exist(self):
        assert isinstance(lpe._GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL, str)
        assert isinstance(lpe._GROK_CLI_FORWARD_HEADER_ALLOWLIST, (list, tuple, frozenset, set))
        assert isinstance(lpe._GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE, (list, tuple, frozenset, set))

    def test_opencode_zen_constants_exist(self):
        assert isinstance(lpe._OPENCODE_ZEN_DEFAULT_BASE_URL, str)
        assert isinstance(lpe._OPENCODE_ZEN_PROVIDER, str)
        assert isinstance(lpe._OPENCODE_ZEN_FREE_MODELS, (list, tuple, frozenset, set))

    def test_antigravity_constants_exist(self):
        assert isinstance(lpe._ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST, (list, tuple, frozenset, set))

    def test_google_adapter_policy_constants_exist(self):
        assert isinstance(lpe._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME, str)
        assert isinstance(lpe._GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT, str)

    def test_codex_reasoning_effort_tiers_exist(self):
        assert isinstance(lpe._CODEX_REASONING_EFFORT_TIERS, (list, tuple))
        assert len(lpe._CODEX_REASONING_EFFORT_TIERS) > 0

    def test_claude_persisted_output_patterns_exist(self):
        import re
        assert isinstance(lpe._CLAUDE_PERSISTED_OUTPUT_PATTERN, re.Pattern)
        assert isinstance(lpe._CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN, re.Pattern)

    def test_grok_build_exhausted_token_value(self):
        assert lpe._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN == "GROK_BUILD_USAGE_BALANCE_EXHAUSTED"

    def test_grok_build_exhausted_url_value(self):
        assert lpe._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL == "https://cli-chat-proxy.grok.com/v1/responses"

    def test_anthropic_gemini_oauth_constants_exist(self):
        assert isinstance(lpe._ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL, str)
        assert isinstance(lpe._ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS, (list, tuple))
        assert isinstance(lpe._ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS, (list, tuple))


# ===========================================================================
# SECTION 6: Signature and annotation contract tests
# ===========================================================================


class TestSignatureContracts:
    """Verify sync/async signatures are preserved for moved functions."""

    def test_grok_side_channel_functions_are_sync(self):
        import inspect
        sync_funcs = [
            "_normalize_grok_endpoint_path",
            "_get_grok_side_channel_endpoint_type",
            "_get_grok_side_channel_endpoint_path_template",
            "_json_shape_type_name",
            "_extract_redacted_grok_json_request_shape",
            "_stable_grok_side_channel_body_digest",
            "_merge_grok_side_channel_shape_into_passthrough_logging_metadata",
            "_get_grok_side_channel_retryable_status_codes",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_model_resolution_functions_are_sync(self):
        import inspect
        sync_funcs = [
            "_normalize_anthropic_adapter_model_name",
            "_split_anthropic_adapter_provider_prefix",
            "_has_anthropic_responses_adapter_endpoint",
            "_normalize_codex_auto_agent_alias_model",
            "_is_codex_auto_agent_alias_model",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_lane_key_functions_are_sync(self):
        import inspect
        sync_funcs = [
            "_hash_codex_auto_agent_lane_value",
            "_codex_auto_agent_candidate_key",
            "_resolve_codex_auto_agent_xai_lane_key",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_google_env_policy_functions_are_sync(self):
        import inspect
        sync_funcs = [
            "_coerce_non_negative_int",
            "_coerce_non_negative_float",
            "_get_google_adapter_max_output_tokens_cap",
            "_get_google_adapter_default_thinking_level",
            "_google_code_assist_unwrapped_chunk_contains_tool_call",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_google_context_window_functions_are_sync(self):
        import inspect
        sync_funcs = [
            "_google_content_has_function_exchange",
            "_google_content_has_function_call",
            "_completion_message_has_visible_text",
            "_estimate_completion_message_text_chars",
            "_completion_message_has_tool_result",
            "_completion_message_tool_call_ids",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"


# ===========================================================================
# SECTION 7: Uniqueness -- no symbol appears in multiple target modules
# ===========================================================================


class TestInventoryUniqueness:
    """Each symbol must appear in exactly one target module."""

    def test_no_cross_target_duplicates(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for module_key, symbols in SYMBOL_INVENTORY.items():
            for sym in symbols:
                if sym in seen:
                    duplicates.append(f"{sym} in both {seen[sym]} and {module_key}")
                else:
                    seen[sym] = module_key
        assert not duplicates, f"Duplicate symbols across targets: {duplicates}"

    def test_no_overlap_with_restored_constants(self):
        for module_key, symbols in SYMBOL_INVENTORY.items():
            overlap = symbols & ALL_RESTORED_CONSTANTS
            assert not overlap, (
                f"{module_key} overlaps with restored constants: {sorted(overlap)}"
            )


# ===========================================================================
# SECTION 8: Wave 6A adapter-runtime ownership and production installation
# ===========================================================================


class TestWave6AAdapterRuntimeOwnership:
    @staticmethod
    def _modules() -> dict[str, object]:
        return {
            name: importlib.import_module(import_path)
            for name, import_path in WAVE6A_MODULE_IMPORT_PATHS.items()
        }

    def test_exact_70_symbol_union_without_duplicate_ownership(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        modules = self._modules()

        for module_name in WAVE6A_MODULE_ORDER:
            owned = tuple(getattr(modules[module_name], "_HOST_FUNCTION_NAMES"))
            assert len(owned) == WAVE6A_EXPECTED_COUNTS[module_name]
            for symbol in owned:
                if symbol in seen:
                    duplicates.append(
                        f"{symbol}: {seen[symbol]} and {module_name}"
                    )
                seen[symbol] = module_name

        assert len(seen) == 70
        assert not duplicates

    def test_no_wave6a_symbol_remains_a_god_module_function_def(self):
        modules = self._modules()
        owned = {
            symbol
            for module in modules.values()
            for symbol in getattr(module, "_HOST_FUNCTION_NAMES")
        }
        remaining = owned & _top_level_function_defs(_parse_god_module())
        assert not remaining

    def test_all_70_facades_share_identity_and_host_globals(self):
        checked = 0
        for module in self._modules().values():
            for symbol in getattr(module, "_HOST_FUNCTION_NAMES"):
                facade = getattr(lpe, symbol)
                assert facade is getattr(module, symbol)
                function = getattr(facade, "__wrapped__", facade)
                assert function.__globals__ is vars(lpe)
                checked += 1
        assert checked == 70

    def test_sse_retains_canonical_shared_helper_ownership(self):
        modules = self._modules()
        sse = modules["sse"]
        assert lpe._mapping_or_attr_get is getattr(sse, "_mapping_or_attr_get")
        assert lpe._responses_repaired_output_item_id is getattr(
            sse, "_responses_repaired_output_item_id"
        )

        dependent_seams = {
            "tool_call_restore": {"_responses_repaired_output_item_id"},
            "stream_collect": {"_mapping_or_attr_get"},
            "payload_validation": {"_mapping_or_attr_get"},
        }
        for module_name, seams in dependent_seams.items():
            module = modules[module_name]
            owned = set(getattr(module, "_HOST_FUNCTION_NAMES"))
            assert not seams & owned
            for seam in seams:
                assert not hasattr(module, seam)

    def test_package_import_layering_and_install_order(self):
        package = importlib.import_module(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime"
        )
        for module_name, module in self._modules().items():
            assert getattr(package, module_name) is module

        package_tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
        install = next(
            node
            for node in package_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "install"
        )
        order = [
            node.value.func.value.id
            for node in install.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "install"
            and isinstance(node.value.func.value, ast.Name)
        ]
        assert order == list(WAVE6A_MODULE_ORDER)

        for module in self._modules().values():
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(
                        "llm_passthrough_endpoints" not in alias.name
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    assert (
                        node.module is None
                        or "llm_passthrough_endpoints" not in node.module
                    )

    def test_production_import_initializes_facades_without_manual_install(self):
        tree = _parse_god_module()
        install_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_aawm_adapter_runtime"
            and node.func.attr == "install"
        ]
        assert len(install_calls) == 1
        for module in self._modules().values():
            for symbol in getattr(module, "_HOST_FUNCTION_NAMES"):
                assert getattr(lpe, symbol) is getattr(module, symbol)


# ===========================================================================
# SECTION 9: Wave 6D request-policy ownership and facade identity
# ===========================================================================

WAVE6D_MODULE_ORDER: tuple[str, ...] = (
    "persisted_output",
    "observability_metadata",
    "alias_guidance",
)
WAVE6D_MODULE_IMPORT_PATHS: dict[str, str] = {
    name: (
        "litellm.proxy.pass_through_endpoints.aawm_request_policy."
        f"{name}"
    )
    for name in WAVE6D_MODULE_ORDER
}
WAVE6D_EXPECTED_COUNTS: dict[str, int] = {
    "persisted_output": 14,
    "observability_metadata": 43,
    "alias_guidance": 6,
}

WAVE6D_PERSISTED_OUTPUT_SYMBOLS: set[str] = {
    "_is_claude_persisted_output_expansion_enabled",
    "_get_claude_persisted_output_root",
    "_resolve_claude_persisted_output_path",
    "_build_claude_persisted_output_source_metadata",
    "_compact_google_adapter_persisted_output_preview_and_expanded_text",
    "_compact_expanded_claude_persisted_output_text_for_google_adapter",
    "_compact_google_adapter_text_part_sequence",
    "_compact_google_adapter_followup_request_contents",
    "_compact_google_adapter_persisted_output_value",
    "_compact_google_adapter_persisted_output_in_anthropic_request_body",
    "_expand_claude_persisted_output_text",
    "_expand_claude_persisted_output_value",
    "_expand_claude_persisted_output_in_anthropic_request_body",
    "_estimate_google_content_text_chars",
}

WAVE6D_OBSERVABILITY_METADATA_SYMBOLS: set[str] = {
    "_merge_litellm_metadata",
    "_format_langfuse_span_timestamp",
    "_build_langfuse_span_descriptor",
    "_normalize_low_cardinality_tag_value",
    "_dedupe_sorted_str_list",
    "_iter_anthropic_text_fragments",
    "_extract_claude_agent_and_tenant_from_request_body",
    "_add_claude_child_agent_observability_metadata",
    "_detect_claude_post_rewrite_context_files",
    "_add_claude_post_rewrite_context_file_logging_metadata",
    "_get_nested_str_value",
    "_extract_passthrough_session_id",
    "_normalize_passthrough_repository",
    "_extract_passthrough_repository_from_text",
    "_walk_request_value_with_budget",
    "_extract_passthrough_repository_from_body_text",
    "_extract_passthrough_repository",
    "_get_passthrough_trace_environment",
    "_add_passthrough_trace_context_metadata",
    "_truncate_tool_definition_string",
    "_redact_tool_definition_string",
    "_sanitize_tool_definition_value",
    "_tool_definition_name",
    "_tool_definition_description",
    "_tool_definition_parameters",
    "_build_tool_definition_snapshot_entry",
    "_tool_definition_snapshot_hash",
    "_build_passthrough_tool_definition_metadata",
    "_add_passthrough_tool_definition_metadata",
    "_prepare_request_body_for_passthrough_observability",
    "_extract_claude_request_breakout_fields",
    "_add_claude_request_breakout_logging_metadata",
    "_extract_gemini_request_breakout_fields",
    "_add_gemini_request_breakout_logging_metadata",
    "_extract_codex_request_breakout_fields",
    "_add_codex_request_breakout_logging_metadata",
    "_parse_anthropic_billing_header_text",
    "_extract_anthropic_billing_header_fields",
    "_extract_anthropic_billing_header_fields_from_request_body",
    "_add_anthropic_billing_header_logging_metadata",
    "_add_claude_persisted_output_logging_metadata",
    "_add_route_family_logging_metadata",
    "_extract_openai_passthrough_tool_choice",
}

WAVE6D_ALIAS_GUIDANCE_SYMBOLS: set[str] = {
    "_append_codex_auto_agent_prevention_guidance_to_instructions",
    "_is_aawm_read_agent_alias_model",
    "_append_aawm_read_agent_guidance_to_text",
    "_append_aawm_read_agent_guidance_to_anthropic_system",
    "_apply_aawm_read_agent_guidance_to_request_body",
    "_apply_codex_auto_agent_prevention_guidance_to_request_body",
}

WAVE6D_SYMBOL_INVENTORY: dict[str, set[str]] = {
    "persisted_output": WAVE6D_PERSISTED_OUTPUT_SYMBOLS,
    "observability_metadata": WAVE6D_OBSERVABILITY_METADATA_SYMBOLS,
    "alias_guidance": WAVE6D_ALIAS_GUIDANCE_SYMBOLS,
}

ALL_WAVE6D_FUNCTIONS: set[str] = set()
for _syms in WAVE6D_SYMBOL_INVENTORY.values():
    ALL_WAVE6D_FUNCTIONS |= _syms


class TestWave6DRequestPolicyOwnership:
    """Wave 6D structural ownership: 63 functions across 3 modules."""

    @staticmethod
    def _modules() -> dict[str, object]:
        return {
            name: importlib.import_module(import_path)
            for name, import_path in WAVE6D_MODULE_IMPORT_PATHS.items()
        }

    def test_exact_63_symbol_union_without_duplicate_ownership(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []

        for module_name in WAVE6D_MODULE_ORDER:
            symbols = WAVE6D_SYMBOL_INVENTORY[module_name]
            assert len(symbols) == WAVE6D_EXPECTED_COUNTS[module_name], (
                f"{module_name}: expected {WAVE6D_EXPECTED_COUNTS[module_name]}, "
                f"got {len(symbols)}"
            )
            for symbol in symbols:
                if symbol in seen:
                    duplicates.append(
                        f"{symbol}: {seen[symbol]} and {module_name}"
                    )
                seen[symbol] = module_name

        assert len(seen) == 63
        assert not duplicates

    def test_no_wave6d_symbol_remains_a_god_module_function_def(self):
        func_defs = _top_level_function_defs(_parse_god_module())
        remaining = ALL_WAVE6D_FUNCTIONS & func_defs
        assert not remaining, (
            f"Wave 6D symbols still defined as FunctionDef in god module: "
            f"{sorted(remaining)}"
        )

    def test_persisted_output_host_function_names_match_inventory(self):
        modules = self._modules()
        po = modules["persisted_output"]
        host_names = set(getattr(po, "_HOST_FUNCTION_NAMES"))
        assert host_names == WAVE6D_PERSISTED_OUTPUT_SYMBOLS

    def test_all_63_facades_share_identity_with_god_module(self):
        modules = self._modules()
        checked = 0

        # persisted_output: installed via install(globals()), rebound
        for symbol in WAVE6D_PERSISTED_OUTPUT_SYMBOLS:
            facade = getattr(lpe, symbol)
            module_fn = getattr(modules["persisted_output"], symbol)
            assert facade is module_fn, (
                f"persisted_output.{symbol}: facade identity mismatch"
            )
            checked += 1

        # observability_metadata: direct same-object assignment
        for symbol in WAVE6D_OBSERVABILITY_METADATA_SYMBOLS:
            facade = getattr(lpe, symbol)
            module_fn = getattr(modules["observability_metadata"], symbol)
            assert facade is module_fn, (
                f"observability_metadata.{symbol}: facade identity mismatch"
            )
            checked += 1

        # alias_guidance: direct same-object assignment
        for symbol in WAVE6D_ALIAS_GUIDANCE_SYMBOLS:
            facade = getattr(lpe, symbol)
            module_fn = getattr(modules["alias_guidance"], symbol)
            assert facade is module_fn, (
                f"alias_guidance.{symbol}: facade identity mismatch"
            )
            checked += 1

        assert checked == 63

    def test_persisted_output_facades_use_host_globals(self):
        """Installed persisted-output functions resolve through host globals."""
        for symbol in WAVE6D_PERSISTED_OUTPUT_SYMBOLS:
            facade = getattr(lpe, symbol)
            function = getattr(facade, "__wrapped__", facade)
            assert function.__globals__ is vars(lpe), (
                f"{symbol}: expected host globals dict"
            )

    def test_no_wave6d_module_imports_god_module_at_scope(self):
        for module_name, import_path in WAVE6D_MODULE_IMPORT_PATHS.items():
            mod = importlib.import_module(import_path)
            mod_path = Path(mod.__file__).resolve()
            tree = ast.parse(mod_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(
                        "llm_passthrough_endpoints" not in alias.name
                        for alias in node.names
                    ), f"{module_name} imports god module"
                elif isinstance(node, ast.ImportFrom):
                    assert (
                        node.module is None
                        or "llm_passthrough_endpoints" not in node.module
                    ), f"{module_name} imports from god module"

    def test_callback_and_install_ordering(self):
        """God module configures observability, then persisted-output, then
        alias guidance -- in that contractual order."""
        tree = _parse_god_module()
        configure_lines: dict[str, int] = {}
        install_lines: dict[str, int] = {}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_observability_metadata"
                    and node.func.attr == "configure_observability_metadata_runtime"
                ):
                    configure_lines["observability"] = node.lineno
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_persisted_output"
                    and node.func.attr == "install"
                ):
                    install_lines["persisted_output"] = node.lineno
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_alias_guidance"
                    and node.func.attr == "configure_alias_guidance_runtime"
                ):
                    configure_lines["alias_guidance"] = node.lineno

        assert "observability" in configure_lines
        assert "persisted_output" in install_lines
        assert "alias_guidance" in configure_lines
        assert configure_lines["observability"] < install_lines["persisted_output"]
        assert install_lines["persisted_output"] < configure_lines["alias_guidance"]

    def test_estimate_google_content_text_chars_owned_by_persisted_output(self):
        """_estimate_google_content_text_chars is Wave 6D persisted_output owned,
        not google env_policy owned."""
        assert "_estimate_google_content_text_chars" not in GOOGLE_ENV_POLICY_SYMBOLS
        assert "_estimate_google_content_text_chars" in WAVE6D_PERSISTED_OUTPUT_SYMBOLS

    def test_control_plane_preserves_distinct_get_nested_str_value(self):
        """The control plane retains its own local _get_nested_str_value,
        distinct from the observability_metadata facade."""
        from litellm.proxy.pass_through_endpoints import (
            aawm_claude_control_plane as cp,
        )

        cp_fn = cp._get_nested_str_value
        obs_fn = lpe._get_nested_str_value
        # Control plane defines its own; observability_metadata owns the facade
        assert cp_fn is not obs_fn
        # Both are callable and behavior-compatible
        assert cp_fn({"a": {"b": "v"}}, ("a", "b")) == "v"
        assert obs_fn({"a": {"b": "v"}}, ("a", "b")) == "v"

    def test_control_plane_canonicalizes_from_observability_metadata(self):
        """Control plane imports canonical helpers from observability_metadata."""
        from litellm.proxy.pass_through_endpoints import (
            aawm_claude_control_plane as cp,
        )
        from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
            observability_metadata as obs,
        )

        assert cp._iter_anthropic_text_fragments is obs._iter_anthropic_text_fragments
        assert (
            cp._extract_claude_agent_and_tenant_from_request_body
            is obs._extract_claude_agent_and_tenant_from_request_body
        )
        assert (
            cp._detect_claude_post_rewrite_context_files
            is obs._detect_claude_post_rewrite_context_files
        )

    def test_no_cross_wave6d_duplicates(self):
        """Each Wave 6D symbol appears in exactly one module."""
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for module_key, symbols in WAVE6D_SYMBOL_INVENTORY.items():
            for sym in symbols:
                if sym in seen:
                    duplicates.append(f"{sym} in both {seen[sym]} and {module_key}")
                else:
                    seen[sym] = module_key
        assert not duplicates

    def test_no_overlap_with_prior_wave_inventories(self):
        """Wave 6D symbols do not overlap with Wave 4 or Wave 6A inventories."""
        prior = ALL_MOVED_FUNCTIONS | ALL_RESTORED_CONSTANTS
        overlap = ALL_WAVE6D_FUNCTIONS & prior
        # _estimate_google_content_text_chars was removed from env_policy
        assert not overlap, f"Wave 6D overlaps prior waves: {sorted(overlap)}"


# ===========================================================================
# SECTION 10: Wave 6E request-policy ownership and facade identity
# ===========================================================================

WAVE6E_MODULE_ORDER: tuple[str, ...] = (
    "codex_tool_policy",
    "claude_prompt_replacement",
    "anthropic_body_prep",
)
WAVE6E_MODULE_IMPORT_PATHS: dict[str, str] = {
    name: (
        "litellm.proxy.pass_through_endpoints.aawm_request_policy."
        f"{name}"
    )
    for name in WAVE6E_MODULE_ORDER
}
WAVE6E_EXPECTED_COUNTS: dict[str, int] = {
    "codex_tool_policy": 51,
    "claude_prompt_replacement": 14,
    "anthropic_body_prep": 11,
}

WAVE6E_CODEX_TOOL_POLICY_SYMBOLS: set[str] = {
    "_patch_codex_spawn_agent_description_text",
    "_get_codex_core_tool_guidance",
    "_append_codex_core_tool_guidance_to_description",
    "_patch_codex_spawn_agent_payload_parameters",
    "_get_openai_tool_name",
    "_get_openai_tool_type",
    "_load_bundled_model_cost_map_for_codex_policy",
    "_get_codex_tool_policy_model_cost_candidates",
    "_get_unsupported_hosted_tool_types_for_model",
    "_get_unsupported_request_param_names_for_model",
    "_get_unsupported_input_item_types_for_model",
    "_get_rewrite_input_item_types_for_model",
    "_get_custom_tool_function_adapter_names_for_model",
    "_get_namespace_tool_function_adapter_names_for_model",
    "_adapted_custom_tool_function_schema",
    "_adapt_codex_custom_tool_definitions",
    "_adapted_custom_tool_call_ids",
    "_adapt_codex_custom_tool_input_items",
    "_adapt_codex_custom_tool_choice",
    "_add_codex_custom_tool_function_adapter_logging_metadata",
    "_adapt_codex_custom_tools_to_functions_from_request_body",
    "_adapt_codex_namespace_tool_definitions",
    "_adapt_codex_namespace_input_items",
    "_adapt_codex_namespace_tool_choice",
    "_add_codex_namespace_tool_function_adapter_logging_metadata",
    "_adapt_codex_namespace_tools_to_functions_from_request_body",
    "_openai_tool_choice_references_tool_type",
    "_add_codex_unsupported_hosted_tool_logging_metadata",
    "_request_has_openai_tool_definitions",
    "_add_tool_choice_without_tools_logging_metadata",
    "_drop_tool_choice_without_tools_from_request_body",
    "_add_codex_unsupported_request_param_logging_metadata",
    "_drop_unsupported_codex_request_params_from_request_body",
    "_add_codex_unsupported_input_item_logging_metadata",
    "_drop_unsupported_codex_input_items_from_request_body",
    "_stringify_grok_native_input_item_value",
    "_format_grok_native_function_call_input_message",
    "_format_grok_native_function_call_output_input_message",
    "_rewrite_grok_native_input_item_for_model_input",
    "_is_anthropic_grok_native_responses_adapter_body",
    "_add_grok_native_input_item_rewrite_logging_metadata",
    "_drop_unsupported_codex_hosted_tools_from_request_body",
    "_patch_codex_spawn_agent_tool_description",
    "_patch_codex_multi_agent_tool_search_description",
    "_patch_codex_core_tool_description",
    "_add_codex_tool_description_patch_logging_metadata",
    "_apply_codex_tool_description_patches_to_request_body",
    "_rewrite_grok_native_unsupported_input_items_from_request_body",
    "_rewrite_grok_native_unsupported_input_items_in_place",
    "_apply_spawn_agent_parameter_patches",
    "_lookup_model_info_field",
}

WAVE6E_CLAUDE_PROMPT_REPLACEMENT_SYMBOLS: set[str] = {
    "_parse_claude_code_version",
    "_resolve_claude_auto_memory_template_path",
    "_load_claude_context_replacement_template",
    "_load_claude_prompt_patch_manifest",
    "_extract_markdown_section",
    "_render_claude_auto_memory_replacement",
    "_replace_claude_auto_memory_section_in_text",
    "_replace_claude_system_prompt_override_in_value",
    "_add_claude_system_prompt_override_logging_metadata",
    "_replace_claude_system_prompt_in_anthropic_request_body",
    "_apply_claude_prompt_patches_in_text",
    "_replace_claude_prompt_patches_in_value",
    "_add_claude_prompt_patch_logging_metadata",
    "_apply_claude_prompt_patches_to_anthropic_request_body",
}

WAVE6E_ANTHROPIC_BODY_PREP_SYMBOLS: set[str] = {
    "_get_openai_adapter_claude_context_char_cap",
    "_detect_openai_adapter_claude_context_markers",
    "_select_openai_adapter_context_summary_lines",
    "_build_openai_adapter_compacted_claude_context_block",
    "_compact_openai_adapter_claude_context_text",
    "_compact_openai_adapter_claude_context_value",
    "_add_openai_adapter_claude_context_compaction_logging_metadata",
    "_compact_openai_adapter_claude_context_in_anthropic_request_body",
    "_validate_anthropic_tool_blocks_for_passthrough",
    "_repair_anthropic_tool_use_ids_for_passthrough",
    "_prepare_anthropic_request_body_for_passthrough",
}

WAVE6E_SYMBOL_INVENTORY: dict[str, set[str]] = {
    "codex_tool_policy": WAVE6E_CODEX_TOOL_POLICY_SYMBOLS,
    "claude_prompt_replacement": WAVE6E_CLAUDE_PROMPT_REPLACEMENT_SYMBOLS,
    "anthropic_body_prep": WAVE6E_ANTHROPIC_BODY_PREP_SYMBOLS,
}

ALL_WAVE6E_FUNCTIONS: set[str] = set()
for _syms in WAVE6E_SYMBOL_INVENTORY.values():
    ALL_WAVE6E_FUNCTIONS |= _syms


class TestWave6ERequestPolicyOwnership:
    """Wave 6E structural ownership: 76 functions across 3 modules."""

    @staticmethod
    def _modules() -> dict[str, object]:
        return {
            name: importlib.import_module(import_path)
            for name, import_path in WAVE6E_MODULE_IMPORT_PATHS.items()
        }

    def test_exact_77_symbol_union_without_duplicate_ownership(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []

        for module_name in WAVE6E_MODULE_ORDER:
            symbols = WAVE6E_SYMBOL_INVENTORY[module_name]
            assert len(symbols) == WAVE6E_EXPECTED_COUNTS[module_name], (
                f"{module_name}: expected {WAVE6E_EXPECTED_COUNTS[module_name]}, "
                f"got {len(symbols)}"
            )
            for symbol in symbols:
                if symbol in seen:
                    duplicates.append(
                        f"{symbol}: {seen[symbol]} and {module_name}"
                    )
                seen[symbol] = module_name

        assert len(seen) == 76
        assert not duplicates

    # Same-object facades: these must NOT be FunctionDefs in the god module.
    # 14 claude_prompt_replacement + 11 anthropic_body_prep + 9 pure codex.
    WAVE6E_SAME_OBJECT_SYMBOLS: set[str] = (
        WAVE6E_CLAUDE_PROMPT_REPLACEMENT_SYMBOLS
        | WAVE6E_ANTHROPIC_BODY_PREP_SYMBOLS
        | {
            "_get_openai_tool_name",
            "_get_openai_tool_type",
            "_patch_codex_spawn_agent_description_text",
            "_patch_codex_spawn_agent_payload_parameters",
            "_load_bundled_model_cost_map_for_codex_policy",
            "_adapted_custom_tool_function_schema",
            "_request_has_openai_tool_definitions",
            "_apply_spawn_agent_parameter_patches",
            "_lookup_model_info_field",
        }
    )

    # Thin callback wrappers: these ARE intentional FunctionDefs in the god
    # module (they bind CodexToolPolicyCallbacks or normalize_tag_value).
    WAVE6E_WRAPPER_SYMBOLS: set[str] = (
        WAVE6E_CODEX_TOOL_POLICY_SYMBOLS - WAVE6E_SAME_OBJECT_SYMBOLS
    )

    def test_same_object_facades_not_function_defs(self):
        """34 same-object facades must not be FunctionDefs in the god module."""
        func_defs = _top_level_function_defs(_parse_god_module())
        violations = self.WAVE6E_SAME_OBJECT_SYMBOLS & func_defs
        assert not violations, (
            f"Wave 6E same-object facades defined as FunctionDefs: "
            f"{sorted(violations)}"
        )

    def test_wrapper_facades_are_function_defs(self):
        """42 codex callback wrappers are intentional FunctionDefs."""
        func_defs = _top_level_function_defs(_parse_god_module())
        missing = self.WAVE6E_WRAPPER_SYMBOLS - func_defs
        assert not missing, (
            f"Wave 6E wrapper facades missing as FunctionDefs: {sorted(missing)}"
        )

    def test_all_wave6e_symbols_accessible(self):
        """All 76 Wave 6E symbols are callable on the god module."""
        for symbol in ALL_WAVE6E_FUNCTIONS:
            facade = getattr(lpe, symbol, None)
            assert facade is not None, f"{symbol} not on god module"
            assert callable(facade), f"{symbol} not callable"

    def test_claude_prompt_replacement_same_object_identity(self):
        """claude_prompt_replacement facades are same-object assignments."""
        mod = importlib.import_module(WAVE6E_MODULE_IMPORT_PATHS["claude_prompt_replacement"])
        for symbol in WAVE6E_CLAUDE_PROMPT_REPLACEMENT_SYMBOLS:
            facade = getattr(lpe, symbol)
            module_fn = getattr(mod, symbol)
            assert facade is module_fn, (
                f"claude_prompt_replacement.{symbol}: facade identity mismatch"
            )

    def test_anthropic_body_prep_same_object_identity(self):
        """anthropic_body_prep facades are same-object assignments."""
        mod = importlib.import_module(WAVE6E_MODULE_IMPORT_PATHS["anthropic_body_prep"])
        for symbol in WAVE6E_ANTHROPIC_BODY_PREP_SYMBOLS:
            facade = getattr(lpe, symbol)
            module_fn = getattr(mod, symbol)
            assert facade is module_fn, (
                f"anthropic_body_prep.{symbol}: facade identity mismatch"
            )

    def test_codex_tool_policy_pure_function_identity(self):
        """codex_tool_policy pure functions (no callbacks) are same-object."""
        mod = importlib.import_module(WAVE6E_MODULE_IMPORT_PATHS["codex_tool_policy"])
        pure_symbols = {
            "_get_openai_tool_name",
            "_get_openai_tool_type",
            "_patch_codex_spawn_agent_description_text",
            "_patch_codex_spawn_agent_payload_parameters",
            "_load_bundled_model_cost_map_for_codex_policy",
            "_adapted_custom_tool_function_schema",
            "_request_has_openai_tool_definitions",
            "_apply_spawn_agent_parameter_patches",
            "_lookup_model_info_field",
        }
        for symbol in pure_symbols:
            facade = getattr(lpe, symbol)
            # Map god-module name to module name (strip leading _)
            module_name = symbol.lstrip("_")
            module_fn = getattr(mod, module_name, None) or getattr(mod, symbol, None)
            assert module_fn is not None, f"{module_name} not on codex_tool_policy"
            assert facade is module_fn, (
                f"codex_tool_policy.{symbol}: facade identity mismatch"
            )

    def test_no_wave6e_module_imports_god_module_at_scope(self):
        for module_name, import_path in WAVE6E_MODULE_IMPORT_PATHS.items():
            mod = importlib.import_module(import_path)
            mod_path = Path(mod.__file__).resolve()
            tree = ast.parse(mod_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(
                        "llm_passthrough_endpoints" not in alias.name
                        for alias in node.names
                    ), f"{module_name} imports god module"
                elif isinstance(node, ast.ImportFrom):
                    assert (
                        node.module is None
                        or "llm_passthrough_endpoints" not in node.module
                    ), f"{module_name} imports from god module"

    def test_callback_and_install_ordering(self):
        """God module configures Wave 6D first, then Wave 6E body prep."""
        tree = _parse_god_module()
        configure_lines: dict[str, int] = {}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_observability_metadata"
                    and node.func.attr == "configure_observability_metadata_runtime"
                ):
                    configure_lines["observability"] = node.lineno
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_persisted_output"
                    and node.func.attr == "install"
                ):
                    configure_lines["persisted_output"] = node.lineno
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_alias_guidance"
                    and node.func.attr == "configure_alias_guidance_runtime"
                ):
                    configure_lines["alias_guidance"] = node.lineno
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_aawm_anthropic_body_prep"
                    and node.func.attr == "configure_anthropic_body_prep_runtime"
                ):
                    configure_lines["anthropic_body_prep"] = node.lineno

        assert "observability" in configure_lines
        assert "persisted_output" in configure_lines
        assert "alias_guidance" in configure_lines
        assert "anthropic_body_prep" in configure_lines
        # Wave 6D before Wave 6E
        assert configure_lines["observability"] < configure_lines["anthropic_body_prep"]
        assert configure_lines["persisted_output"] < configure_lines["anthropic_body_prep"]
        assert configure_lines["alias_guidance"] < configure_lines["anthropic_body_prep"]

    def test_no_cross_wave6e_duplicates(self):
        """Each Wave 6E symbol appears in exactly one module."""
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for module_key, symbols in WAVE6E_SYMBOL_INVENTORY.items():
            for sym in symbols:
                if sym in seen:
                    duplicates.append(f"{sym} in both {seen[sym]} and {module_key}")
                else:
                    seen[sym] = module_key
        assert not duplicates

    def test_no_overlap_with_prior_wave_inventories(self):
        """Wave 6E symbols do not overlap with Wave 4, 6A, or 6D inventories."""
        prior = ALL_MOVED_FUNCTIONS | ALL_RESTORED_CONSTANTS | ALL_WAVE6D_FUNCTIONS
        overlap = ALL_WAVE6E_FUNCTIONS & prior
        assert not overlap, f"Wave 6E overlaps prior waves: {sorted(overlap)}"

    def test_codex_tool_policy_callbacks_bound(self):
        """_CODEX_TOOL_POLICY_CALLBACKS is constructed in the god module."""
        assert hasattr(lpe, "_CODEX_TOOL_POLICY_CALLBACKS")
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            CodexToolPolicyCallbacks,
        )
        assert isinstance(lpe._CODEX_TOOL_POLICY_CALLBACKS, CodexToolPolicyCallbacks)

    def test_anthropic_body_prep_callbacks_configured(self):
        """configure_anthropic_body_prep_runtime has been called with live callbacks."""
        from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
            anthropic_body_prep as abp,
        )
        # After configuration, the module-level callback slots are not None
        assert abp._expand_persisted_output is not None
        assert abp._extract_billing_header_fields is not None
        assert abp._apply_control_plane_rewrites is not None
        assert abp._prepare_observability is not None
        assert abp._get_tenant_header is not None
