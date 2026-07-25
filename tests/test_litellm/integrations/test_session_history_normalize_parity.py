"""Wave A4C golden-output parity, structural, and AST-baseline tests.

Pins CURRENT behavior of the two A4C target modules so the engineer's
behavior-preserving extraction can be verified by re-running these tests
post-move. Parity and immutable-baseline guards are GREEN before extraction;
structural ownership, packaging, and installed-wheel guards are intentionally
RED until the A4C implementation lands.

Target modules and their ORIGINAL line bands (per
.analysis/aawm-agent-identity-and-oversized-units-decomposition-2026-07-23.md):

  aawm_session_history/normalize.py      11114-12880 (record normalization
      chain incl. trust chains :11267-11888 + _sync_session_history_record_metadata
      :11889-12077)
  aawm_session_history/context_window.py 14492-14796 (Anthropic 1M context-window
      beta classification)

The trust-chain AST fixture is derived from exact Git source at e3dc89f634.
The runtime golden fixture was captured from c69c9d2587 after A4B and before
A4C extraction. Do not regenerate or modify either fixture after A4C lands.
"""

from __future__ import annotations

import ast
import copy
import json
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
IDENTITY_INIT = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity" / "__init__.py"
SESSION_HISTORY_DIR = (
    REPO_ROOT / "litellm" / "integrations" / "aawm_session_history"
)
NORMALIZE_PATH = SESSION_HISTORY_DIR / "normalize.py"
CONTEXT_WINDOW_PATH = SESSION_HISTORY_DIR / "context_window.py"
WHEEL_BUILD_PYPROJECT = REPO_ROOT / ".wheel-build" / "pyproject.toml"
PACKAGE_CONVERSION_TEST = (
    Path(__file__).resolve().parent / "test_agent_identity_package_conversion.py"
)

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_PARITY_GOLDEN_PATH = FIXTURES_DIR / "a4c_parity_golden.json"
_AST_BASELINE_PATH = FIXTURES_DIR / "a4c_trust_chain_ast_baseline.json"


def _load_parity_golden() -> Dict[str, Any]:
    with open(_PARITY_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_ast_baseline() -> Dict[str, Any]:
    with open(_AST_BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _normalize_source_path() -> Path:
    """Return the post-move owner when present, otherwise the pre-move host."""
    return NORMALIZE_PATH if NORMALIZE_PATH.is_file() else IDENTITY_INIT


def _context_window_source_path() -> Path:
    """Return the post-move owner when present, otherwise the pre-move host."""
    return CONTEXT_WINDOW_PATH if CONTEXT_WINDOW_PATH.is_file() else IDENTITY_INIT


def _module_level_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# ---------------------------------------------------------------------------
# Full A4C inventories (derived from baseline e3dc89f634)
# ---------------------------------------------------------------------------

# normalize.py: baseline :11114-12880 minus _positive_int_or_none (A4A prompt_overhead)
_A4C_NORMALIZE_FUNCTIONS: List[str] = [
    # record-state normalizers (:11121-11266)
    "_normalize_reasoning_state",
    "_row_usage_object_from_record",
    "_normalize_provider_cache_state_on_record",
    "_normalize_session_runtime_identity_on_record",
    # trust band (:11267-11888) — 33 functions
    "_is_harness_tenant_identity",
    "_normalize_request_header_tenant_repository",
    "_normalize_repository_trust_source",
    "_repository_source_has_codex_memory_workflow",
    "_is_repository_source_trusted_common",
    "_is_repository_source_trusted_for_tenant",
    "_is_codex_trace_user_tenant_source",
    "_is_codex_passthrough_tenant_extraction_context",
    "_is_repository_source_trusted_for_codex_tenant",
    "_is_codex_session_history_record",
    "_is_claude_session_history_record",
    "_is_claude_project_repository_source",
    "_is_claude_metadata_tenant_source",
    "_claude_project_identity_is_trusted",
    "_codex_repository_source_trusted_for_record",
    "_clear_untrusted_codex_trace_user_tenant_on_record",
    "_mark_codex_trace_user_tenant_skipped",
    "_codex_untrusted_repository_reason",
    "_mark_repository_unresolved_metadata",
    "_session_history_missing_repository_reason",
    "_mark_missing_repository_unresolved",
    "_clear_untrusted_claude_project_repository_on_record",
    "_clear_untrusted_claude_metadata_tenant_on_record",
    "_clear_repository_unresolved_metadata",
    "_mark_codex_repository_tenant_skipped",
    "_clear_codex_trace_user_tenant_source_on_record",
    "_clear_untrusted_codex_tenant_on_record",
    "_codex_tenant_source_trusted_for_record",
    "_clear_untrusted_codex_repository_tenant_on_record",
    "_normalize_session_repository_on_record",
    "_can_promote_known_codex_repository_to_tenant",
    "_normalize_session_tenant_on_record",
    # _sync_session_history_record_metadata (:11889-12077)
    "_sync_session_history_record_metadata",
    # record-side state normalizers (:12080-12159)
    "_normalize_prompt_overhead_state_on_record",
    "_normalize_invalid_tool_call_state_on_record",
    "_normalize_structured_output_state_on_record",
    "_normalize_compact_summary_state_on_record",
    # agent-quality scoring glue (:12160-12498)
    "_optional_metadata_bool",
    "_normalize_agent_score_reasons",
    "_append_agent_quality_text",
    "_append_agent_quality_command_from_arguments",
    "_append_agent_quality_commands_from_message",
    "_collect_agent_quality_context_from_request_body",
    "_collect_agent_quality_response_texts",
    "_agent_quality_commands_from_tool_activity",
    "_apply_runtime_agent_quality_scores",
    "_normalize_agent_score_state_on_record",
    # latency + zero-token (:12499-12655)
    "_normalize_session_latency_state_on_record",
    "_extract_gemini_control_plane_method_from_record",
    "_session_history_record_provider_usage_token_total",
    "_classify_zero_token_session_history_record",
    # orchestrator (:12656)
    "_normalize_session_history_record",
    # post-orchestrator record normalizers (:12687-12880)
    "_normalize_agent_id_on_record",
    "_normalize_inbound_model_alias_on_record",
    "_extract_inline_tool_definition_snapshot_from_metadata",
    "_normalize_reporting_exclusion_state_on_record",
]

# context_window.py: baseline :14492-14796 — exactly 10 functions
_A4C_CONTEXT_WINDOW_FUNCTIONS: List[str] = [
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
]

_A4C_ALL_FUNCTIONS: List[str] = _A4C_NORMALIZE_FUNCTIONS + _A4C_CONTEXT_WINDOW_FUNCTIONS

# Trust band only (for AST baseline comparison)
_A4C_TRUST_BAND_FUNCTIONS: List[str] = [
    "_is_harness_tenant_identity",
    "_normalize_request_header_tenant_repository",
    "_normalize_repository_trust_source",
    "_repository_source_has_codex_memory_workflow",
    "_is_repository_source_trusted_common",
    "_is_repository_source_trusted_for_tenant",
    "_is_codex_trace_user_tenant_source",
    "_is_codex_passthrough_tenant_extraction_context",
    "_is_repository_source_trusted_for_codex_tenant",
    "_is_codex_session_history_record",
    "_is_claude_session_history_record",
    "_is_claude_project_repository_source",
    "_is_claude_metadata_tenant_source",
    "_claude_project_identity_is_trusted",
    "_codex_repository_source_trusted_for_record",
    "_clear_untrusted_codex_trace_user_tenant_on_record",
    "_mark_codex_trace_user_tenant_skipped",
    "_codex_untrusted_repository_reason",
    "_mark_repository_unresolved_metadata",
    "_session_history_missing_repository_reason",
    "_mark_missing_repository_unresolved",
    "_clear_untrusted_claude_project_repository_on_record",
    "_clear_untrusted_claude_metadata_tenant_on_record",
    "_clear_repository_unresolved_metadata",
    "_mark_codex_repository_tenant_skipped",
    "_clear_codex_trace_user_tenant_source_on_record",
    "_clear_untrusted_codex_tenant_on_record",
    "_codex_tenant_source_trusted_for_record",
    "_clear_untrusted_codex_repository_tenant_on_record",
    "_normalize_session_repository_on_record",
    "_can_promote_known_codex_repository_to_tenant",
    "_normalize_session_tenant_on_record",
    "_sync_session_history_record_metadata",
]

# A4D symbols that must NOT appear in A4C inventory
_A4D_SYMBOLS = frozenset(
    {
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
# 1. Exact symbol inventories
# =========================================================================


class TestA4CSymbolInventories:
    """Exact inventory checks for normalize.py and context_window.py."""

    def test_normalize_inventory_count(self) -> None:
        assert len(_A4C_NORMALIZE_FUNCTIONS) == 60, (
            f"expected 60 normalize.py functions, got {len(_A4C_NORMALIZE_FUNCTIONS)}"
        )

    def test_context_window_inventory_count(self) -> None:
        assert len(_A4C_CONTEXT_WINDOW_FUNCTIONS) == 10, (
            f"expected 10 context_window.py functions, got {len(_A4C_CONTEXT_WINDOW_FUNCTIONS)}"
        )

    def test_no_a4d_backfill_storage_symbols(self) -> None:
        overlap = sorted(_A4D_SYMBOLS & set(_A4C_ALL_FUNCTIONS))
        assert not overlap, f"A4C inventory must not claim A4D symbols: {overlap}"

    def test_no_duplicate_across_modules(self) -> None:
        overlap = sorted(
            set(_A4C_NORMALIZE_FUNCTIONS) & set(_A4C_CONTEXT_WINDOW_FUNCTIONS)
        )
        assert not overlap, f"symbols in both modules: {overlap}"

    def test_all_a4c_functions_exist_in_current_source(self) -> None:
        """Every A4C function must be defined in the current __init__.py."""
        import litellm.integrations.aawm_agent_identity as identity_pkg

        missing = [
            name
            for name in _A4C_ALL_FUNCTIONS
            if not callable(getattr(identity_pkg, name, None))
        ]
        assert not missing, f"A4C functions missing from identity package: {missing}"

    def test_selected_normalize_source_contains_full_inventory(self) -> None:
        source_path = _normalize_source_path()
        defined = _module_level_function_names(source_path)
        missing = sorted(set(_A4C_NORMALIZE_FUNCTIONS) - defined)
        assert not missing, (
            f"selected normalize owner {source_path} is missing: {missing}"
        )

    def test_selected_context_source_contains_full_inventory(self) -> None:
        source_path = _context_window_source_path()
        defined = _module_level_function_names(source_path)
        missing = sorted(set(_A4C_CONTEXT_WINDOW_FUNCTIONS) - defined)
        assert not missing, (
            f"selected context-window owner {source_path} is missing: {missing}"
        )


# =========================================================================
# 2. test_normalize_record_parity_golden
# =========================================================================


class TestNormalizeRecordParityGolden:
    """Non-vacuous representative matrix through _normalize_session_history_record.

    Drives the CURRENT production function and pins deep-equal outputs in an
    immutable pre-move fixture. Post-move, the same inputs must produce the
    same outputs.
    """

    @pytest.fixture(autouse=True)
    def _load_golden(self) -> None:
        self.golden = _load_parity_golden()
        self.records = self.golden["normalize_record_goldens"]

    def test_golden_fixture_is_non_vacuous(self) -> None:
        """Fixture must contain the required representative cases."""
        required_cases = {
            "codex_trusted_repo",
            "claude_untrusted_project",
            "grok_basic",
            "openrouter_free",
            "backfill_synthesized",
            "failure_observation",
            "conflicting_tenant_repo",
            "trusted_promotion",
        }
        present = set(self.records.keys())
        missing = required_cases - present
        assert not missing, f"golden fixture missing required cases: {missing}"

    def test_golden_outputs_differ_from_inputs(self) -> None:
        """At least some golden outputs must differ from their inputs,
        proving the function does real work (not a tautology)."""
        changed = 0
        for name, case in self.records.items():
            if case["input"] != case["output"]:
                changed += 1
        assert changed >= 4, (
            f"expected at least 4 of {len(self.records)} golden cases to show "
            f"input->output mutation, got {changed}"
        )

    @pytest.mark.parametrize(
        "case_name",
        [
            "codex_trusted_repo",
            "claude_untrusted_project",
            "grok_basic",
            "openrouter_free",
            "backfill_synthesized",
            "failure_observation",
            "conflicting_tenant_repo",
            "trusted_promotion",
        ],
    )
    def test_normalize_record_matches_golden(self, case_name: str) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _normalize_session_history_record,
        )

        case = self.records[case_name]
        input_record = copy.deepcopy(case["input"])
        expected_output = case["output"]

        actual_output = _normalize_session_history_record(input_record)
        assert actual_output == expected_output, (
            f"normalize_record parity mismatch for '{case_name}':\n"
            f"  expected keys: {sorted(expected_output.keys())}\n"
            f"  actual keys:   {sorted(actual_output.keys())}"
        )


# =========================================================================
# 3. test_trust_chain_functions_ast_identical
# =========================================================================


class TestTrustChainFunctionsAstIdentical:
    """Immutable pre-move AST dumps for all trust predicates/clear/mark helpers.

    Future extracted functions must compare to this baseline with exact
    inventory and no weakening.
    """

    @pytest.fixture(autouse=True)
    def _load_baseline(self) -> None:
        self.baseline = _load_ast_baseline()
        self.baseline_functions: Dict[str, str] = self.baseline["functions"]

    def test_baseline_inventory_exact(self) -> None:
        """Baseline must contain exactly the 33 trust-band functions."""
        expected = set(_A4C_TRUST_BAND_FUNCTIONS)
        actual = set(self.baseline_functions.keys())
        assert actual == expected, (
            f"baseline inventory mismatch.\n"
            f"  missing: {sorted(expected - actual)}\n"
            f"  extra:   {sorted(actual - expected)}"
        )

    def test_current_ast_matches_baseline(self) -> None:
        """Current source AST for each trust-band function must be identical
        to the immutable baseline (no logic drift pre-move)."""
        source_path = _normalize_source_path()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        current_dumps: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in self.baseline_functions:
                    current_dumps[node.name] = ast.dump(
                        node, annotate_fields=True, include_attributes=False
                    )

        missing = sorted(set(self.baseline_functions) - set(current_dumps))
        assert not missing, (
            f"trust-band functions missing from {source_path}: {missing}"
        )

        drifted = []
        for name in sorted(self.baseline_functions):
            if current_dumps.get(name) != self.baseline_functions[name]:
                drifted.append(name)
        assert not drifted, (
            f"trust-band functions with AST drift from baseline: {drifted}"
        )


# =========================================================================
# 4. Context-window golden parity
# =========================================================================


class TestContextWindowGoldenParity:
    """Golden parity for context-window helpers/classifier."""

    @pytest.fixture(autouse=True)
    def _load_golden(self) -> None:
        self.golden = _load_parity_golden()
        self.cw = self.golden["context_window_goldens"]

    def test_select_safe_beta_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _select_safe_anthropic_context_window_beta,
        )

        for case in self.cw["select_safe_beta"]:
            result = _select_safe_anthropic_context_window_beta(case["input"][0])
            assert result == case["output"], (
                f"select_safe_beta mismatch for {case['input']}: "
                f"got {result!r}, expected {case['output']!r}"
            )

    def test_model_1m_suffix_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _model_strings_indicate_context_1m_suffix,
        )

        for case in self.cw["model_1m_suffix"]:
            result = _model_strings_indicate_context_1m_suffix(*case["input"][0])
            assert result == case["output"], (
                f"model_1m_suffix mismatch for {case['input']}: "
                f"got {result!r}, expected {case['output']!r}"
            )

    def test_is_anthropic_context_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _is_anthropic_session_history_context,
        )

        for case in self.cw["is_anthropic_context"]:
            inp = case["input"]
            result = _is_anthropic_session_history_context(
                provider=inp["provider"],
                resolved_model=inp["resolved_model"],
                metadata=inp["metadata"],
            )
            assert result == case["output"], (
                f"is_anthropic_context mismatch for {inp}: "
                f"got {result!r}, expected {case['output']!r}"
            )

    def test_enrich_context_window_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _enrich_anthropic_context_window_metadata,
        )

        for i, case in enumerate(self.cw["enrich_context_window"]):
            inp = case["input"]
            md = copy.deepcopy(inp["metadata"])
            kw = copy.deepcopy(inp["kwargs"])
            _enrich_anthropic_context_window_metadata(
                kw,
                md,
                resolved_model=inp["resolved_model"],
                inbound_model_alias=inp["inbound_model_alias"],
                provider=inp["provider"],
                allow_implicit_default=inp["allow_implicit_default"],
            )
            assert md == case["output_metadata"], (
                f"enrich_context_window case {i} mismatch:\n"
                f"  got:      {md}\n"
                f"  expected: {case['output_metadata']}"
            )

    def test_enrich_backfill_context_window_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _enrich_backfill_anthropic_context_window_metadata,
        )

        for i, case in enumerate(self.cw["enrich_backfill_context_window"]):
            rec = copy.deepcopy(case["input"])
            _enrich_backfill_anthropic_context_window_metadata(rec)
            assert rec == case["output_record"], (
                f"enrich_backfill_context_window case {i} mismatch:\n"
                f"  got:      {rec}\n"
                f"  expected: {case['output_record']}"
            )

    def test_classify_context_window_golden(self) -> None:
        from litellm.integrations.aawm_agent_identity import (
            _classify_anthropic_context_window_from_retained_evidence,
        )

        for i, case in enumerate(self.cw["classify_context_window"]):
            inp = case["input"]
            result = _classify_anthropic_context_window_from_retained_evidence(
                copy.deepcopy(inp["metadata"]),
                resolved_model=inp["resolved_model"],
                inbound_model_alias=inp["inbound_model_alias"],
                headers=inp["headers"],
                allow_implicit_default=inp["allow_implicit_default"],
            )
            assert result == case["output"], (
                f"classify_context_window case {i} mismatch:\n"
                f"  got:      {result}\n"
                f"  expected: {case['output']}"
            )


# =========================================================================
# 5. _sync_session_history_record_metadata verbatim + no U17 refactor
# =========================================================================


class TestSyncMetadataVerbatim:
    """_sync_session_history_record_metadata must remain verbatim in A4C.

    The U17 declarative field-spec refactor is explicitly OUT OF SCOPE.
    """

    def test_sync_metadata_in_trust_band_baseline(self) -> None:
        """_sync_session_history_record_metadata is in the immutable AST
        baseline, so any logic change will be caught by the AST test."""
        baseline = _load_ast_baseline()
        assert "_sync_session_history_record_metadata" in baseline["functions"]

    def test_no_declarative_field_spec_refactor(self) -> None:
        """The current _sync_session_history_record_metadata must NOT use a
        declarative field-spec table pattern (U17 refactor is deferred).

        Checks that the function body does not contain a single loop over a
        list of (record_key, metadata_key, coercer) tuples — the hallmark of
        the U17 refactor.
        """
        source_path = _normalize_source_path()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        sync_fn = None
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_sync_session_history_record_metadata"
            ):
                sync_fn = node
                break

        assert sync_fn is not None, (
            f"_sync_session_history_record_metadata not found in {source_path}"
        )

        # U17 hallmark: a module-level or function-level list of tuples named
        # something like _FIELD_SPEC or FIELD_SPEC, iterated in a single loop.
        # The current code uses multiple distinct for-loops over separate
        # constant tuples (_PROMPT_OVERHEAD_TOKEN_FIELDS, etc.).
        fn_source = ast.get_source_segment(source, sync_fn) or ""
        u17_markers = [
            "FIELD_SPEC",
            "field_spec",
            "_RECORD_FIELD_SPEC",
            "declarative",
        ]
        found_markers = [m for m in u17_markers if m in fn_source]
        assert not found_markers, (
            f"U17 declarative field-spec refactor detected in "
            f"_sync_session_history_record_metadata: {found_markers}"
        )

        # Verify the function still has multiple distinct for-loops (not a
        # single unified loop), confirming no U17 collapse.
        for_loop_count = sum(
            1 for node in ast.walk(sync_fn) if isinstance(node, ast.For)
        )
        assert for_loop_count >= 8, (
            f"expected >= 8 distinct for-loops in "
            f"_sync_session_history_record_metadata (current hand-mirrored "
            f"pattern), got {for_loop_count} — possible U17 collapse"
        )


# =========================================================================
# 6. Installed-package / layout expectations (tests-only, source unchanged)
# =========================================================================


class TestA4CPackageLayoutExpectations:
    """Extend installed-package/layout expectations for A4C target modules.

    These are RED until the engineer creates the target files.
    """

    def test_normalize_module_file_exists(self) -> None:
        assert NORMALIZE_PATH.is_file(), (
            f"expected A4C target module to exist: {NORMALIZE_PATH}"
        )

    def test_context_window_module_file_exists(self) -> None:
        assert CONTEXT_WINDOW_PATH.is_file(), (
            f"expected A4C target module to exist: {CONTEXT_WINDOW_PATH}"
        )

    def test_session_history_init_exports_normalize(self) -> None:
        """aawm_session_history/__init__.py must re-export normalize APIs."""
        import litellm.integrations.aawm_session_history as sh_pkg

        missing = [
            name
            for name in ("_normalize_session_history_record", "_sync_session_history_record_metadata")
            if not hasattr(sh_pkg, name)
        ]
        assert not missing, (
            f"aawm_session_history must re-export these A4C APIs: {missing}"
        )

    def test_wheel_and_sdist_force_include_a4c_modules(self) -> None:
        """Both callback artifacts must package both A4C modules exactly."""
        pyproject = tomllib.loads(
            WHEEL_BUILD_PYPROJECT.read_text(encoding="utf-8")
        )
        targets = pyproject["tool"]["hatch"]["build"]["targets"]
        expected = {
            "../litellm/integrations/aawm_session_history/normalize.py": (
                "litellm/integrations/aawm_session_history/normalize.py"
            ),
            "../litellm/integrations/aawm_session_history/context_window.py": (
                "litellm/integrations/aawm_session_history/context_window.py"
            ),
        }

        failures: Dict[str, Dict[str, Any]] = {}
        for artifact in ("wheel", "sdist"):
            force_include = targets[artifact]["force-include"]
            missing = {
                source: destination
                for source, destination in expected.items()
                if force_include.get(source) != destination
            }
            duplicate_destinations = {
                destination: sum(
                    1
                    for mapped_destination in force_include.values()
                    if mapped_destination == destination
                )
                for destination in expected.values()
            }
            duplicate_destinations = {
                destination: count
                for destination, count in duplicate_destinations.items()
                if count != 1
            }
            if missing or duplicate_destinations:
                failures[artifact] = {
                    "missing_or_wrong": missing,
                    "destination_counts": duplicate_destinations,
                }

        assert not failures, (
            "A4C wheel/sdist force-include mappings are incomplete or "
            f"duplicated: {failures}"
        )

    def test_paired_installed_wheel_smoke_imports_a4c_modules(self) -> None:
        """The existing paired-wheel smoke must import identity plus A4C."""
        source = PACKAGE_CONVERSION_TEST.read_text(encoding="utf-8")
        required_fragments = {
            "identity package": (
                "import litellm.integrations.aawm_agent_identity as main_identity"
            ),
            "normalize module": (
                "'litellm.integrations.aawm_session_history.normalize'"
            ),
            "context-window module": (
                "'litellm.integrations.aawm_session_history.context_window'"
            ),
        }
        missing = [
            label
            for label, fragment in required_fragments.items()
            if fragment not in source
        ]
        assert not missing, (
            "paired installed-wheel smoke does not import required A4C "
            f"surfaces: {missing}"
        )

    def test_session_history_init_exports_context_window(self) -> None:
        """aawm_session_history/__init__.py must re-export context-window APIs."""
        import litellm.integrations.aawm_session_history as sh_pkg

        missing = [
            name
            for name in ("_enrich_anthropic_context_window_metadata",)
            if not hasattr(sh_pkg, name)
        ]
        assert not missing, (
            f"aawm_session_history must re-export these A4C APIs: {missing}"
        )


# =========================================================================
# 7. Structural RED tests (fail until extraction)
# =========================================================================


class TestA4CStructuralRed:
    """Structural extraction assertions — intentionally RED before the move."""

    def test_normalize_functions_not_defined_in_identity_init(self) -> None:
        """A4C normalize functions must not remain as FunctionDef in
        aawm_agent_identity/__init__.py after extraction."""
        source = IDENTITY_INIT.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(IDENTITY_INIT))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        still_defined = sorted(
            name for name in _A4C_NORMALIZE_FUNCTIONS if name in defined
        )
        assert not still_defined, (
            "expected these A4C normalize functions to no longer be defined "
            f"in aawm_agent_identity/__init__.py: {still_defined}"
        )

    def test_context_window_functions_not_defined_in_identity_init(self) -> None:
        """A4C context-window functions must not remain as FunctionDef in
        aawm_agent_identity/__init__.py after extraction."""
        source = IDENTITY_INIT.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(IDENTITY_INIT))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        still_defined = sorted(
            name for name in _A4C_CONTEXT_WINDOW_FUNCTIONS if name in defined
        )
        assert not still_defined, (
            "expected these A4C context-window functions to no longer be "
            f"defined in aawm_agent_identity/__init__.py: {still_defined}"
        )

    def test_facade_same_object_identity(self) -> None:
        """getattr(identity_pkg, name) is getattr(sh_module, name) for every
        A4C function after extraction."""
        import importlib

        import litellm.integrations.aawm_agent_identity as identity_pkg

        modules_to_check = {
            "litellm.integrations.aawm_session_history.normalize": _A4C_NORMALIZE_FUNCTIONS,
            "litellm.integrations.aawm_session_history.context_window": _A4C_CONTEXT_WINDOW_FUNCTIONS,
        }

        missing_modules: List[str] = []
        mismatched: List[tuple] = []
        for module_path, names in modules_to_check.items():
            try:
                mod = importlib.import_module(module_path)
            except ModuleNotFoundError:
                missing_modules.append(module_path)
                continue
            for name in names:
                pkg_val = getattr(identity_pkg, name, None)
                mod_val = getattr(mod, name, None)
                if pkg_val is None or mod_val is None or pkg_val is not mod_val:
                    mismatched.append((module_path, name))

        assert not missing_modules, (
            f"expected these A4C target modules to exist: {missing_modules}"
        )
        assert not mismatched, (
            f"facade identity mismatches (module, name): {mismatched}"
        )

    def test_host_globals_rebind_install_pattern(self) -> None:
        """Every A4C export must be rebound to identity host globals."""
        import importlib

        import litellm.integrations.aawm_agent_identity as identity_pkg

        missing_modules: List[str] = []
        missing_install: List[str] = []
        wrong_globals: List[tuple] = []
        modules_to_check = {
            "litellm.integrations.aawm_session_history.normalize": _A4C_NORMALIZE_FUNCTIONS,
            "litellm.integrations.aawm_session_history.context_window": _A4C_CONTEXT_WINDOW_FUNCTIONS,
        }
        for module_path, names in modules_to_check.items():
            try:
                mod = importlib.import_module(module_path)
            except ModuleNotFoundError:
                missing_modules.append(module_path)
                continue
            if not callable(getattr(mod, "install", None)):
                missing_install.append(module_path)
            for name in names:
                function = getattr(mod, name, None)
                if (
                    function is None
                    or getattr(function, "__globals__", None)
                    is not identity_pkg.__dict__
                ):
                    wrong_globals.append((module_path, name))

        assert not missing_modules, (
            f"expected these A4C target modules to exist: {missing_modules}"
        )
        assert not missing_install, (
            "expected each A4C submodule to expose an install() callable "
            f"for __globals__ rebinding: {missing_install}"
        )
        assert not wrong_globals, (
            "A4C functions not rebound to identity package globals: "
            f"{wrong_globals}"
        )

    def test_rebind_order_facades_before_record_install(self) -> None:
        """Every A4C facade assignment in aawm_agent_identity/__init__.py
        must precede the _bind_session_history_record_apis() call."""
        source = IDENTITY_INIT.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(IDENTITY_INIT))

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
            "expected _bind_session_history_record_apis() call in __init__.py"
        )

        defined_functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assign_targets: Dict[str, ast.Assign] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in assign_targets:
                        assign_targets[target.id] = node

        expected_facades = set(_A4C_ALL_FUNCTIONS)
        remaining_definitions = sorted(expected_facades & defined_functions)
        present_facades = expected_facades & set(assign_targets)
        missing_facades = sorted(expected_facades - present_facades)

        late_facades = [
            name
            for name in sorted(present_facades)
            if assign_targets[name].lineno >= call_line
        ]
        assert not remaining_definitions and not missing_facades and not late_facades, (
            "A4C facade/rebind contract failed: "
            f"remaining FunctionDefs={remaining_definitions}; "
            f"missing facade assignments={missing_facades}; "
            f"late facade assignments (must precede line {call_line})={late_facades}"
        )
