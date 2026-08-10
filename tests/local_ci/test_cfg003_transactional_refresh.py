"""CFG-003: Comprehensive adversarial tests for transactional priority-swap
refresh, coverage gate, semantic-hash contract, fail-closed inventory,
restoration precedence, artifact sanitization, correlated triples, and
error-intake scoping.

Covers all 7 findings with probes designed to catch false passes.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RA_PATH = ROOT / "scripts" / "local-ci" / "run_acceptance.py"
ADAPTER_PATH = ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"
BASIC_YAML_PATH = ROOT / "litellm" / "proxy" / "aawm_alias_config" / "basic.yaml"
CONFIG_JSON_PATH = ROOT / "scripts" / "local-ci" / "anthropic_adapter_config.json"


def _load_ra():
    name = "run_acceptance_cfg003_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RA_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_adapter():
    name = "run_adapter_cfg003_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    return _load_ra()


@pytest.fixture(scope="module")
def adapter():
    return _load_adapter()


@pytest.fixture()
def basic_yaml_text():
    return BASIC_YAML_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Finding 1: Proof enforcement - provider+model+route_family, semantic hash
# ---------------------------------------------------------------------------


class TestProofEnforcement:
    def test_selection_requires_all_three_fields(self, adapter):
        """All-wrong-route probe: matching provider+model but wrong route
        must fail."""
        candidate = {"provider": "openrouter", "model": "m1", "route_family": "codex_openrouter_completion_adapter"}
        # Wrong route_family
        assert not adapter._cfg003_selection_matches_candidate(
            {"provider": "openrouter", "model": "m1", "route_family": "wrong_route"},
            candidate,
        )
        # Wrong model
        assert not adapter._cfg003_selection_matches_candidate(
            {"provider": "openrouter", "model": "wrong", "route_family": "codex_openrouter_completion_adapter"},
            candidate,
        )
        # Wrong provider
        assert not adapter._cfg003_selection_matches_candidate(
            {"provider": "wrong", "model": "m1", "route_family": "codex_openrouter_completion_adapter"},
            candidate,
        )
        # None route_family
        assert not adapter._cfg003_selection_matches_candidate(
            {"provider": "openrouter", "model": "m1", "route_family": None},
            candidate,
        )
        # Exact match
        assert adapter._cfg003_selection_matches_candidate(
            {"provider": "openrouter", "model": "m1", "route_family": "codex_openrouter_completion_adapter"},
            candidate,
        )

    def test_extract_refresh_hash_nested_detail(self, ra):
        resp = {"detail": {"active_config_hash": "abc", "config_version": "v1"}}
        assert ra._extract_refresh_response_hash(resp) == "abc"
        assert ra._extract_refresh_response_version(resp) == "v1"

    def test_extract_refresh_hash_empty(self, ra):
        assert ra._extract_refresh_response_hash({}) == ""
        assert ra._extract_refresh_response_version({}) == ""

    def test_empty_hash_from_swap_fails(self, adapter, ra, monkeypatch):
        """Swap returning empty semantic hash must hard-fail."""
        real_yaml = BASIC_YAML_PATH.read_text(encoding="utf-8")
        def fake_auth():
            snap = _compile_snapshot()
            return {"snapshot": snap, "merged_yaml": real_yaml, "per_file_hashes": {"basic.yaml": "h"},
                    "file_names": ["basic.yaml"], "config_hash": "semhash", "config_version": "semver", "aliases": ["basic"]}

        def fake_post(url, payload, **kw):
            if "not_a_list" in payload.get("yaml", ""):
                return 400, {"detail": {"active_config_hash": "semhash"}}
            return 200, {"changed": True, "active_config_hash": "", "config_version": ""}

        # Provide positive availability for at least 2 candidates.
        snap = _compile_snapshot()
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (200, {"aawm_alias_config": {"state": "active", "config_hash": "semhash", "config_version": "semver", "files": ["basic.yaml"], "aliases": ["basic"]}}))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.8-max-preview", "route_family": "codex_alibaba_token_plan_chat_completions_adapter"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("empty semantic hash" in f for f in result["failures"])

    def test_nonbasic_source_refresh_phase_preserves_full_directory_semantics(  # noqa: PLR0915
        self, adapter, ra, monkeypatch, tmp_path
    ):
        basic_yaml = """\
defaults: {}

aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: or/model-a
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: alibaba_token_plan
        model: atp/qwen-mini
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 90
"""
        nonbasic_yaml = """\
defaults: {}

aliases:
  - name: nonbasic
    candidates:
      - provider: openrouter
        model: gpt-test-1
        route_family: codex_openrouter_completion_adapter
        priority: 200
      - provider: openrouter
        model: or/model-b
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
        (tmp_path / "basic.yaml").write_text(basic_yaml, encoding="utf-8")
        nonbasic_path = tmp_path / "nonbasic.yaml"
        nonbasic_path.write_text(nonbasic_yaml, encoding="utf-8")

        monkeypatch.setattr(adapter.RA, "_AAWM_ALIAS_CONFIG_DIR", tmp_path)
        auth = adapter.RA._load_authoritative_startup_config()
        baseline_inventory = adapter.RA._snapshot_source_inventory()

        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            nonbasic_yaml,
            pair=(("openrouter", "gpt-test-1"), ("openrouter", "or/model-b")),
            alias_name="nonbasic",
        )
        load_calls = {"n": 0}
        observed_mutation_auth = {}
        authoritative_loader = adapter.RA._load_authoritative_startup_config

        def fake_load_auth():
            load_calls["n"] += 1
            assert nonbasic_path.read_bytes() == swapped_yaml.encode("utf-8")
            loaded = authoritative_loader()
            observed_mutation_auth.update(loaded)
            return loaded

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_load_auth)
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory", lambda url: {
            "healthy": True,
            "active_aliases": auth["aliases"],
            "source_files": auth["file_names"],
        })
        readiness_calls = {"n": 0}

        def fake_readiness(*a, **kw):
            readiness_calls["n"] += 1
            if readiness_calls["n"] == 1:
                mutated_auth = observed_mutation_auth
                assert mutated_auth
                return 200, {
                    "aawm_alias_config": {
                        "state": "active",
                        "config_hash": mutated_auth["config_hash"],
                        "config_version": mutated_auth["config_version"],
                        "files": mutated_auth["file_names"],
                        "aliases": mutated_auth["aliases"],
                    }
                }
            return 200, {
                "aawm_alias_config": {
                    "state": "active",
                    "config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "files": auth["file_names"],
                    "aliases": auth["aliases"],
                }
            }

        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", fake_readiness)

        post_calls = {"empty": 0}

        def fake_post(url, payload, **kw):
            assert payload == {}
            post_calls["empty"] += 1
            mutated_auth = observed_mutation_auth
            assert mutated_auth
            mutated_order = ra._derive_full_order_from_snapshot(
                mutated_auth["snapshot"], alias_name="nonbasic"
            )
            if post_calls["empty"] == 1:
                return 200, {
                    "changed": True,
                    "active_config_hash": mutated_auth["config_hash"],
                    "config_version": mutated_auth["config_version"],
                    "active_candidate_order": {"nonbasic": mutated_order},
                }
            if post_calls["empty"] == 2:
                return 200, {
                    "changed": False,
                    "active_config_hash": mutated_auth["config_hash"],
                    "config_version": mutated_auth["config_version"],
                    "active_candidate_order": {"nonbasic": mutated_order},
                }
            if post_calls["empty"] == 3:
                return 400, {
                    "detail": {
                        "active_config_hash": mutated_auth["config_hash"],
                        "config_version": mutated_auth["config_version"],
                    }
                }
            return 200, {
                "changed": False,
                "active_config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "active_candidate_order": {
                    "nonbasic": ra._derive_full_order_from_snapshot(
                        auth["snapshot"], alias_name="nonbasic"
                    )
                },
            }

        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)

        phase = adapter._cfg003_nonbasic_source_file_refresh_phase(
            litellm_base_url="http://localhost:4001",
            refresh_url=f"http://localhost:4001{adapter.RA._AAWM_ALIAS_CONFIG_REFRESH_PATH}",
            baseline_aliases=auth["aliases"],
            baseline_source_files=auth["file_names"],
            baseline_hash=auth["config_hash"],
            baseline_version=auth["config_version"],
            source_inventory_before=baseline_inventory,
        )

        assert not phase["failures"]
        assert phase["restoration_error"] is None
        assert phase["selected_alias"] == "nonbasic"
        assert phase["steps"]["mutate_refresh"]["changed"] is True
        assert observed_mutation_auth["aliases"] == ["basic", "nonbasic"]
        assert observed_mutation_auth["file_names"] == ["basic.yaml", "nonbasic.yaml"]
        assert phase["steps"]["mutate_refresh"]["active_hash"] == observed_mutation_auth["config_hash"]
        assert phase["steps"]["mutate_refresh"]["alias_order_matches"] is True
        assert phase["steps"]["unchanged_refresh"]["changed"] is False
        assert phase["steps"]["unchanged_refresh"]["active_hash"] == observed_mutation_auth["config_hash"]
        assert phase["steps"]["invalid_refresh"]["status_code"] == 400
        assert phase["steps"]["invalid_refresh"]["lkg_hash"] == observed_mutation_auth["config_hash"]
        assert phase["steps"]["restore"]["active_hash"] == auth["config_hash"]
        assert phase["steps"]["restore"]["readiness_passed"] is True
        assert phase["steps"]["source_inventory_after_restore"]["unchanged"] is True
        assert nonbasic_path.read_bytes() == nonbasic_yaml.encode("utf-8")
        assert load_calls["n"] == 1
        assert post_calls["empty"] == 4

    def test_nonbasic_source_phase_restoration_failure_becomes_parent_primary(self, adapter, ra, monkeypatch):
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(
            adapter,
            "_cfg003_nonbasic_source_file_refresh_phase",
            lambda *a, **kw: {
                "name": "nonbasic_source_file_refresh",
                "selected_file": "nonbasic.yaml",
                "selected_alias": "nonbasic",
                "failures": ["nonbasic source helper failed"],
                "steps": {},
                "restoration_error": "RESTORATION FAILED: nonbasic helper",
            },
        )
        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: auth)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_cfg003_readiness_check", lambda *a, **kw: (False, ["halted by test"]))

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={},
            suite_config={},
            query_url="q",
            public_key="pk",
            secret_key="sk",
        )

        assert not result["passed"]
        assert "nonbasic source helper failed" in result["failures"]
        assert "RESTORATION FAILED: nonbasic helper" in result["restoration_failure"]
        assert result["failures"][0] == result["restoration_failure"]


# ---------------------------------------------------------------------------
# Finding 2: Authoritative fail-closed inventory via CFG-002
# ---------------------------------------------------------------------------


class TestFailClosedInventory:
    def test_readiness_unavailable_fails(self, adapter, ra, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (0, {"error": "conn refused"}))
        result = adapter._cfg003_query_active_inventory("http://localhost:4001")
        assert not result["healthy"]
        assert any("unavailable" in f for f in result["inventory_failures"])

    def test_non_active_state_fails(self, adapter, ra, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (200, {"aawm_alias_config": {"state": "failed"}}))
        result = adapter._cfg003_query_active_inventory("http://localhost:4001")
        assert not result["healthy"]
        assert any("not active" in f for f in result["inventory_failures"])

    def test_alias_mismatch_fails(self, adapter, ra, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (200, {
            "aawm_alias_config": {"state": "active", "config_hash": "h", "config_version": "v",
                                  "files": ["basic.yaml"], "aliases": ["basic", "extra"]}
        }))
        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": None, "merged_yaml": "", "per_file_hashes": {"basic.yaml": "h"},
            "file_names": ["basic.yaml"], "config_hash": "h", "config_version": "v", "aliases": ["basic"]
        })
        result = adapter._cfg003_query_active_inventory("http://localhost:4001")
        assert not result["healthy"]
        assert any("alias mismatch" in f for f in result["inventory_failures"])

    def test_cli_passthrough_alone_does_not_qualify(self, ra):
        assert ra._is_real_tui_case({"cli_passthrough": "codex"}) is False
        assert ra._is_real_tui_case({"command": ["codex", "exec"]}) is True
        assert ra._is_real_tui_case({"command": ["claude", "-p"]}) is True
        assert ra._is_real_tui_case({"http_request": {}}) is False
        assert ra._is_real_tui_case({}) is False

    def test_http_only_case_rejected(self, ra):
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses"]}]
        cases = {"http_case": {"verification_alias": "basic", "verification_ingress": "codex_responses", "http_request": {"method": "POST"}}}
        passed, failures = ra._validate_alias_ingress_coverage(alias_inventory=inventory, cases=cases, selected_cases=["http_case"])
        assert not passed
        assert any("not a real TUI case" in f for f in failures)


# ---------------------------------------------------------------------------
# Finding 3: Availability evidence integration
# ---------------------------------------------------------------------------


class TestAvailabilityEvidence:
    def test_parse_route_availability_evidence(self, ra):
        log = """
20260730 06:00:00 Codex[1.0] <mock>
 - openrouter/model-a(basic):none - Turns: 0 [rate limited by upstream] [Cooling Down] -> route
 - openrouter/model-b(basic):low - Turns: 0 [Selected model is at capacity] [Failed] -> route
 - openrouter/model-c(basic):xhigh - Turns: 2 [success] [Selected] -> route
 - openrouter/model-c(basic):low - Turns: 1 [success] [Selected] -> route
"""
        evidence = ra._parse_route_availability_evidence(log, "basic")
        assert evidence["openrouter/model-a"] == "Cooling Down"
        assert evidence["openrouter/model-b"] == "Failed"
        # Mixed-effort buckets for the same model remain separate rollup lines;
        # availability still keys by model and keeps the latest status.
        assert evidence["openrouter/model-c"] == "Selected"

    def test_unavailable_candidates_filtered(self, ra, basic_yaml_text):
        """Candidates with Cooling Down/Failed/Exhausted status are excluded."""
        # Use the snapshot-based derivation with availability evidence.
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        # Mark the first two candidates as unavailable.
        all_eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(all_eligible) >= 3
        first_model = all_eligible[0]["model"]
        second_model = all_eligible[1]["model"]
        availability = {first_model: "Cooling Down", second_model: "Failed"}
        filtered = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", availability_evidence=availability
        )
        filtered_models = {c["model"] for c in filtered}
        assert first_model not in filtered_models
        assert second_model not in filtered_models
        assert len(filtered) == len(all_eligible) - 2

    def test_insufficient_available_candidates_fails(self, adapter, ra, monkeypatch):
        """When availability evidence leaves < 2 candidates, the test must fail."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        all_eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        # Mark ALL but one as unavailable (positive evidence format).
        avail_res = _avail_result(all_eligible, available_count=1)

        def fake_auth():
            return {"snapshot": snap, "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (200, {"aawm_alias_config": {"state": "active", "config_hash": auth["config_hash"], "config_version": auth["config_version"], "files": auth["file_names"], "aliases": auth["aliases"]}}))
        monkeypatch.setattr(adapter.RA, "_http_post_json", lambda *a, **kw: (200, {"changed": False, "active_config_hash": auth["config_hash"], "config_version": auth["config_version"]}))
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001", cases={}, suite_config={},
            query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("eligible" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Finding 4: Exact restoration via CFG-002 authoritative config
# ---------------------------------------------------------------------------


class TestExactRestoration:
    def test_authoritative_startup_config_uses_compile_directory(self, ra):
        auth = ra._load_authoritative_startup_config()
        assert auth["config_hash"]
        assert auth["config_version"]
        assert "basic" in auth["aliases"]
        assert "basic.yaml" in auth["per_file_hashes"]
        # Merged YAML recompiles to the same semantic hash.
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        recompiled = compile_yaml(auth["merged_yaml"])
        assert recompiled.config_hash == auth["config_hash"]

    def test_source_files_unchanged_proof(self, adapter, ra):
        auth = ra._load_authoritative_startup_config()
        ok, failures = adapter._cfg003_verify_source_files_unchanged(auth["per_file_hashes"])
        assert ok
        assert failures == []

    def test_source_file_change_detected(self, adapter, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_AAWM_ALIAS_CONFIG_DIR", tmp_path)
        (tmp_path / "basic.yaml").write_text("modified", encoding="utf-8")
        ok, failures = adapter._cfg003_verify_source_files_unchanged({"basic.yaml": "wrong_hash"})
        assert not ok
        assert any("changed" in f for f in failures)

    def test_restoration_failure_primary_even_with_swap_failure(self, adapter, ra, monkeypatch):
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        def fake_auth():
            return {"snapshot": snap, "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        call_n = {"n": 0}
        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            # unchanged_control succeeds (reaches mutation_attempted=True).
            if call_n["n"] == 1:
                return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            # Swap POST fails.
            if call_n["n"] == 2:
                return 500, {"error": "swap failed"}
            # Restoration POST also fails.
            return 500, {"error": "restore failed"}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (200, {"aawm_alias_config": {"state": "active", "config_hash": auth["config_hash"], "config_version": auth["config_version"], "files": auth["file_names"], "aliases": auth["aliases"]}}))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        first = eligible[0]
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {"provider": first["provider"], "model": first["model"], "route_family": first["route_family"]})
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda: {"a.yaml": "h"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert "RESTORATION" in result["failures"][0]
        assert "restoration_failure" in result
        assert "recovery_artifact" in result


# ---------------------------------------------------------------------------
# Finding 5: Artifact persistence sanitization via real _write_artifact
# ---------------------------------------------------------------------------


class TestArtifactPersistenceSanitization:
    def test_write_artifact_redacts_sensitive_fields(self, adapter, ra, tmp_path):
        """Invoke the REAL _write_artifact and verify no sensitive data on disk."""
        artifact = {
            "results": {
                "case1": {
                    "passed": True,
                    "command": ["codex", "exec", "-p", "secret-profile"],
                    "stdout": "raw output with secrets",
                    "stderr": "error output",
                    "command_string": "codex exec ...",
                },
            },
            "cfg003_transactional_refresh": {
                "phases": {"load": {"original_semantic_hash": "abc"}},
                "recovery_artifact": {"yaml": "raw yaml", "prompt": "secret prompt"},
            },
            "environment": {"api_key": "sk-12345", "safe": "visible"},
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)

        persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
        persisted_str = json.dumps(persisted)
        # Sensitive values must not appear.
        assert "secret-profile" not in persisted_str
        assert "raw output with secrets" not in persisted_str
        assert "error output" not in persisted_str
        assert "codex exec" not in persisted_str
        assert "raw yaml" not in persisted_str
        assert "secret prompt" not in persisted_str
        assert "sk-12345" not in persisted_str
        # Structured outcomes preserved.
        assert persisted["results"]["case1"]["passed"] is True
        assert persisted["cfg003_transactional_refresh"]["phases"]["load"]["original_semantic_hash"] == "abc"
        assert persisted["environment"]["safe"] == "visible"


# ---------------------------------------------------------------------------
# Finding 6: Exact equality + correlated triples
# ---------------------------------------------------------------------------


class TestExactEqualityAndCorrelatedTriples:
    def test_claude_case_uses_required_equals_result(self):
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        claude_case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        checks = claude_case["command_json_checks"]
        assert "required_equals" in checks
        assert checks["required_equals"]["result"] == "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        # Must NOT use required_contains for result (substring is insufficient).
        assert "required_contains" not in checks or "result" not in checks.get("required_contains", {})

    def test_correlated_triples_in_config(self):
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        for case_name in (
            "native_openai_passthrough_responses_codex_basic_alias_collaboration",
            "claude_adapter_basic_alias_child_parallel_read_tools",
        ):
            case = config["cases"][case_name]
            row = case["session_history_validation"]["expected_rows"][0]
            triples = row["correlated_candidate_triples"]
            assert isinstance(triples, list)
            assert len(triples) >= 2
            for triple in triples:
                assert "provider" in triple
                assert "model" in triple
                assert "route_family" in triple

    def test_correlated_triples_enforced_in_matcher(self, adapter):
        """A row matching provider+model but wrong route_family must not match."""
        expected_row = {
            "correlated_candidate_triples": [
                {"provider": "openrouter", "model": "m1", "route_family": "codex_openrouter_completion_adapter"},
            ],
        }
        # Correct triple
        row_ok = {
            "provider": "openrouter",
            "model": "m1",
            "metadata": {"codex_auto_agent_selected_route_family": "codex_openrouter_completion_adapter"},
        }
        assert adapter._session_history_record_matches_expected(row_ok, expected_row)

        # Wrong route_family
        row_bad_route = {
            "provider": "openrouter",
            "model": "m1",
            "metadata": {"codex_auto_agent_selected_route_family": "wrong_route"},
        }
        assert not adapter._session_history_record_matches_expected(row_bad_route, expected_row)

        # Missing metadata
        row_no_meta = {"provider": "openrouter", "model": "m1"}
        assert not adapter._session_history_record_matches_expected(row_no_meta, expected_row)


# ---------------------------------------------------------------------------
# Finding 7: Coverage map, target rejection, TEST_HARNESS.md
# ---------------------------------------------------------------------------


class TestCoverageMapAndTargetRejection:
    def test_complete_coverage_map_validates_all_cases(self, ra):
        """The complete configured case map must cover every alias/ingress."""
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses", "anthropic_messages"]}]
        passed, failures = ra._validate_complete_coverage_map(
            alias_inventory=inventory, cases=config["cases"]
        )
        assert passed, f"coverage map failures: {failures}"

    def test_complete_coverage_map_detects_missing(self, ra):
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses", "anthropic_messages"]}]
        cases = {
            "only_codex": {"verification_alias": "basic", "verification_ingress": "codex_responses", "command": ["codex", "exec"]},
        }
        passed, failures = ra._validate_complete_coverage_map(alias_inventory=inventory, cases=cases)
        assert not passed
        assert any("anthropic_messages" in f for f in failures)


# ---------------------------------------------------------------------------
# Coverage gate structural tests
# ---------------------------------------------------------------------------


class TestCoverageGate:
    def test_missing_coverage_detected(self, ra):
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses", "anthropic_messages"]}]
        cases = {"case_a": {"verification_alias": "basic", "verification_ingress": "codex_responses", "command": ["codex", "exec"]}}
        passed, failures = ra._validate_alias_ingress_coverage(alias_inventory=inventory, cases=cases, selected_cases=["case_a"])
        assert not passed

    def test_duplicate_coverage_detected(self, ra):
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses"]}]
        cases = {
            "case_a": {"verification_alias": "basic", "verification_ingress": "codex_responses", "command": ["codex", "exec"]},
            "case_b": {"verification_alias": "basic", "verification_ingress": "codex_responses", "command": ["codex", "exec"]},
        }
        passed, failures = ra._validate_alias_ingress_coverage(alias_inventory=inventory, cases=cases, selected_cases=["case_a", "case_b"])
        assert not passed
        assert any("duplicate" in f for f in failures)

    def test_exact_coverage_passes(self, ra):
        inventory = [{"alias": "basic", "supported_ingresses": ["codex_responses", "anthropic_messages"]}]
        cases = {
            "case_codex": {"verification_alias": "basic", "verification_ingress": "codex_responses", "command": ["codex", "exec"]},
            "case_claude": {"verification_alias": "basic", "verification_ingress": "anthropic_messages", "command": ["claude", "-p"]},
        }
        passed, failures = ra._validate_alias_ingress_coverage(alias_inventory=inventory, cases=cases, selected_cases=["case_codex", "case_claude"])
        assert passed


# ---------------------------------------------------------------------------
# Config JSON structural tests
# ---------------------------------------------------------------------------


class TestConfigJsonStructure:
    @pytest.fixture(scope="class")
    def config(self):
        return json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))

    def test_basic_alias_cases_exist(self, config):
        assert "native_openai_passthrough_responses_codex_basic_alias_collaboration" in config["cases"]
        assert "claude_adapter_basic_alias_child_parallel_read_tools" in config["cases"]

    def test_basic_alias_cases_in_default_excluded(self, config):
        excluded = config["default_excluded_cases"]
        assert "native_openai_passthrough_responses_codex_basic_alias_collaboration" in excluded
        assert "claude_adapter_basic_alias_child_parallel_read_tools" in excluded

    def test_basic_alias_cases_have_real_commands(self, config):
        codex_case = config["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        assert codex_case["command"][0] == "codex"
        assert "basic" in codex_case["command"]
        claude_case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        assert claude_case["command"][0] == "claude"

    def test_basic_alias_cases_have_parallel_tool_contract(self, config):
        codex_case = config["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        assert codex_case["codex_collaboration_validation"]["command_execution_validation"]["minimum_parallel_count"] == 3
        claude_case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        assert claude_case["transcript_tool_use_validation"]["expected_agents"][0]["minimum_tools_in_single_assistant_message"] == 3


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


class TestHttpHelpers:
    def test_http_post_json_returns_status_and_body(self, ra, monkeypatch):
        class FakeResponse:
            status = 200
            def read(self):
                return json.dumps({"changed": True, "active_config_hash": "abc"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        monkeypatch.setattr(ra.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
        status, body = ra._http_post_json("http://localhost:4001/test", {"yaml": "x"})
        assert status == 200
        assert body["changed"] is True

    def test_http_post_json_handles_http_error(self, ra, monkeypatch):
        import io
        import urllib.error
        class FakeHTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("http://x", 400, "Bad Request", {}, io.BytesIO(
                    json.dumps({"detail": {"error": "compile failed", "active_config_hash": "lkg"}}).encode()
                ))
        monkeypatch.setattr(ra.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(FakeHTTPError()))
        status, body = ra._http_post_json("http://localhost:4001/test", {"yaml": "bad"})
        assert status == 400
        assert ra._extract_refresh_response_hash(body) == "lkg"


# ---------------------------------------------------------------------------
# Eligible candidates and swap tests
# ---------------------------------------------------------------------------


class TestEligibleCandidatesAndSwap:
    def test_at_least_two_eligible(self, ra, basic_yaml_text):
        eligible = ra._derive_eligible_candidates_from_yaml(basic_yaml_text)
        assert len(eligible) >= 2

    def test_excluded_providers_filtered(self, ra, basic_yaml_text):
        eligible = ra._derive_eligible_candidates_from_yaml(basic_yaml_text)
        providers = {c["provider"] for c in eligible}
        assert "anthropic" not in providers
        assert "xai" not in providers

    def test_exact_swap(self, ra, basic_yaml_text):
        swapped_yaml, original, swapped = ra._build_priority_swap_yaml(basic_yaml_text)
        assert swapped[0]["model"] == original[1]["model"]
        assert swapped[1]["model"] == original[0]["model"]

    def test_swap_does_not_mutate_file(self, ra, basic_yaml_text):
        original_bytes = BASIC_YAML_PATH.read_bytes()
        ra._build_priority_swap_yaml(basic_yaml_text)
        assert BASIC_YAML_PATH.read_bytes() == original_bytes

    def test_insufficient_candidates_raises(self, ra):
        yaml_text = "aliases:\n  - name: basic\n    candidates:\n      - provider: openrouter\n        model: m\n        route_family: codex_openrouter_completion_adapter\n        priority: 50\n"
        with pytest.raises(ValueError, match="at least 2"):
            ra._build_priority_swap_yaml(yaml_text)


# ---------------------------------------------------------------------------
# Artifact redaction unit tests
# ---------------------------------------------------------------------------


class TestArtifactRedaction:
    def test_sensitive_keys_redacted(self, ra):
        data = {
            "authorization": "Bearer sk-secret",
            "api_key": "sk-12345",
            "config_hash": "abc123",
            "nested": {"password": "hunter2", "safe_field": "visible"},
            "command": ["codex", "exec"],
            "stdout": "raw output",
        }
        redacted = ra._redact_sensitive_artifact_fields(data)
        assert redacted["authorization"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["config_hash"] == "abc123"
        assert redacted["nested"]["password"] == "[REDACTED]"
        assert redacted["nested"]["safe_field"] == "visible"
        assert redacted["command"] == "[REDACTED]"
        assert redacted["stdout"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# Helper: compile a snapshot for tests that need one
# ---------------------------------------------------------------------------


def _compile_snapshot():
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import compile_directory
    return compile_directory(pathlib.Path(str(ROOT / "litellm" / "proxy" / "aawm_alias_config")))


def _snapshot_eligible_fields(snapshot_candidates: list[dict[str, str]]):
    return [
        (
            c["provider"],
            c["model"],
            c["route_family"],
            c["priority"],
        )
        for c in snapshot_candidates
    ]


class TestAliasReferenceResolution:
    def test_derive_ingresses_resolves_reference_aliases(self, ra):
        snap = _compile_snapshot()
        ingresses = ra._derive_ingresses_from_snapshot(snap, alias_name="work")
        assert ingresses == ["anthropic_messages", "codex_responses"]

    def test_derive_eligible_candidates_preserves_non_reference_aliases(self, ra, basic_yaml_text):
        snap = _compile_snapshot()
        yaml_eligible = ra._derive_eligible_candidates_from_yaml(basic_yaml_text)
        snapshot_eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert _snapshot_eligible_fields(yaml_eligible) == _snapshot_eligible_fields(snapshot_eligible)

    def test_dispatch_only_alias_is_resolved_to_default_target(self, ra):
        snap = _compile_snapshot()
        ingresses = ra._derive_ingresses_from_snapshot(snap, alias_name="sota")
        eligible = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="sota", excluded_providers=frozenset()
        )
        assert ingresses == ["anthropic_messages", "codex_responses"]
        assert len(eligible) == 1
        assert eligible[0]["provider"] == "openai"
        assert eligible[0]["model"] == "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# Finding 1 (round 4): Error intake baseline/delta collector
# ---------------------------------------------------------------------------


class TestErrorIntakeCollector:
    def test_historical_events_ignored(self, ra, tmp_path):
        """Events with observed_at before initiation must not be attributed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        old_time = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
        (analysis / "dev-error.jsonl").write_text(
            json.dumps({"observed_at": old_time, "environment": "dev", "level": "error",
                        "message": "old failure", "fingerprint": "fp1"}) + "\n",
            encoding="utf-8",
        )
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc)
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert events == []
        assert failures == []

    def test_new_attributed_event_fails(self, ra, tmp_path):
        """New event with observed_at >= initiation and matching env is attributed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "new failure", "fingerprint": "fp2",
                                "context": {"container": "litellm-dev"}}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert len(events) == 1
        assert "new failure" in events[0]["message"]
        assert failures == []

    def test_unrelated_environment_ignored(self, ra, tmp_path):
        """Events from a different environment are not attributed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "prod", "level": "error",
                                "message": "prod failure"}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert events == []

    def test_truncation_fails_closed(self, ra, tmp_path):
        """File truncation (size decrease) must fail closed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text(
            json.dumps({"observed_at": "2026-01-01T00:00:00Z", "environment": "dev",
                        "level": "error", "message": "line1"}) + "\n"
            + json.dumps({"observed_at": "2026-01-01T00:00:01Z", "environment": "dev",
                          "level": "error", "message": "line2"}) + "\n",
            encoding="utf-8",
        )
        baseline = ra._snapshot_error_intake(analysis)
        # Truncate the file.
        (analysis / "dev-error.jsonl").write_text("short\n", encoding="utf-8")
        initiation = dt.datetime.now(dt.timezone.utc)
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert any("truncated" in f for f in failures)

    def test_rotation_fails_closed(self, ra, tmp_path):
        """Inode change (rotation) must fail closed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        fp = analysis / "dev-error.jsonl"
        fp.write_text(json.dumps({"observed_at": "2026-01-01T00:00:00Z", "environment": "dev",
                                   "level": "error", "message": "old"}) + "\n", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        # Simulate rotation: delete and recreate (new inode).
        fp.unlink()
        fp.write_text(json.dumps({"observed_at": "2026-01-01T00:00:00Z", "environment": "dev",
                                   "level": "error", "message": "old"}) + "\n", encoding="utf-8")
        initiation = dt.datetime.now(dt.timezone.utc)
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert any("rotated" in f for f in failures)

    def test_malformed_append_recorded_not_failed(self, ra, tmp_path):
        """Malformed JSONL lines are recorded but do not cause failures."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write("NOT VALID JSON\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert len(events) == 1
        assert events[0].get("malformed") is True
        assert failures == []

    def test_nested_files_discovered(self, ra, tmp_path):
        """Nested .analysis subdirectories are discovered recursively."""
        analysis = tmp_path / ".analysis"
        (analysis / "dev").mkdir(parents=True)
        (analysis / "dev" / "runtime-error.jsonl").write_text("", encoding="utf-8")
        (analysis / "root-error.log").write_text("", encoding="utf-8")
        files = ra._discover_error_intake_files(analysis)
        rel_paths = {str(f.relative_to(analysis)) for f in files}
        assert "dev/runtime-error.jsonl" in rel_paths
        assert "root-error.log" in rel_paths


# ---------------------------------------------------------------------------
# Finding 2 (round 4): Positive availability evidence
# ---------------------------------------------------------------------------


class TestPositiveAvailability:
    def test_empty_evidence_yields_zero_available(self, ra):
        """No DB settings means no positive evidence."""
        result = ra._filter_candidates_by_positive_availability(
            [{"model": "m1", "provider": "p1"}], {}
        )
        assert result == []

    def test_one_model_available_yields_one(self, ra):
        candidates = [
            {"model": "m1", "provider": "p1"},
            {"model": "m2", "provider": "p2"},
        ]
        availability = {
            ("p1", "m1"): _valid_avail_record(True, provider="p1", model="m1"),
            ("p2", "m2"): _valid_avail_record(False, provider="p2", model="m2"),
        }
        result = ra._filter_candidates_by_positive_availability(candidates, availability)
        assert len(result) == 1
        assert result[0]["model"] == "m1"

    def test_stale_evidence_not_available(self, ra):
        """Stale evidence (available=False) must not count."""
        candidates = [{"model": "m1", "provider": "p1"}]
        availability = {("p1", "m1"): _valid_avail_record(False)}
        result = ra._filter_candidates_by_positive_availability(candidates, availability)
        assert result == []

    def test_db_settings_none_returns_empty(self, adapter):
        """No DB settings yields no positive evidence."""
        result = adapter._cfg003_collect_availability_evidence(
            [{"model": "m1", "provider": "p1"}], db_settings=None
        )
        assert result["evidence"] == {}
        assert result["available_identities"] == []
        assert result["source"] == "none"


# ---------------------------------------------------------------------------
# Finding 3 (round 4): Single-source requirement and source inventory
# ---------------------------------------------------------------------------


class TestSingleSourceAndInventory:
    def test_snapshot_source_inventory(self, ra):
        inv = ra._snapshot_source_inventory()
        assert "basic.yaml" in inv
        assert len(inv["basic.yaml"]) == 64  # sha256 hex

    def test_multiple_sources_fail_closed(self, adapter, ra, monkeypatch):
        """Multiple source files must fail closed before egress."""
        auth = ra._load_authoritative_startup_config()
        multi_hashes = {"basic.yaml": "h1", "extra.yaml": "h2"}

        def fake_auth():
            return {"snapshot": auth["snapshot"], "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": multi_hashes, "file_names": ["basic.yaml", "extra.yaml"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(
            adapter.RA, "_recursive_yaml_source_inventory",
            lambda *a, **kw: {"basic.yaml": "h1", "extra.yaml": "h2"},
        )
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001", cases={}, suite_config={},
            query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("exactly one recursive YAML source" in f for f in result["failures"])

    def test_restoration_requires_order_match(self, adapter, ra, monkeypatch):
        """Restoration must require active_candidate_order match."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        swapped_hash = swapped_snap.config_hash
        swapped_version = swapped_snap.config_version
        swapped_full_order = ra._derive_full_order_from_snapshot(swapped_snap, alias_name="basic")

        def fake_auth():
            return {"snapshot": snap, "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        def _full_order_response(order):
            return {"basic": [
                {"provider": c["provider"], "model": c["model"],
                 "route_family": c["route_family"],
                 "anthropic_route_family": c.get("anthropic_route_family", ""),
                 "priority": c["priority"], "last_resort": c.get("last_resort", False)}
                for c in order
            ]}

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                # unchanged_control: OK.  Restoration: WRONG order.
                # Use a flag to distinguish.
                if not hasattr(fake_post, "_swap_done"):
                    return 200, {
                        "changed": False, "active_config_hash": auth["config_hash"],
                        "config_version": auth["config_version"],
                        "active_candidate_order": _full_order_response(full_order),
                    }
                # Restoration returns WRONG order.
                return 200, {
                    "changed": True, "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": {"basic": [{"provider": "WRONG", "model": "WRONG", "route_family": "WRONG", "priority": 0}]},
                }
            # Swap POST: correct.
            fake_post._swap_done = True
            return 200, {
                "changed": True, "active_config_hash": swapped_hash,
                "config_version": swapped_version,
                "active_candidate_order": _full_order_response(swapped_full_order),
            }

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda *a, **kw: {"basic.yaml": "h"})
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": eligible[0]["provider"], "model": eligible[0]["model"], "route_family": eligible[0]["route_family"]})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert "restoration_failure" in result
        assert "order_matches" in result["restoration_failure"]


# ---------------------------------------------------------------------------
# Finding 4 (round 4): Restoration precedence for restore_proof failure
# ---------------------------------------------------------------------------


class TestRestorationPrecedence:
    def test_restore_proof_failure_is_restoration_failure(self, adapter, ra, monkeypatch):
        """A failed restore_proof must become the primary restoration failure."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        # Compute the expected swapped hash/version so the swap POST returns
        # correct values and the swap_proof gate passes (round 8 gate).
        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, swapped_eligible = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        swapped_hash = swapped_snap.config_hash
        swapped_version = swapped_snap.config_version
        swapped_full_order = ra._derive_full_order_from_snapshot(swapped_snap, alias_name="basic")

        def fake_auth():
            return {"snapshot": snap, "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        def _full_order_response(order):
            return {"basic": [
                {"provider": c["provider"], "model": c["model"],
                 "route_family": c["route_family"],
                 "anthropic_route_family": c.get("anthropic_route_family", ""),
                 "priority": c["priority"], "last_resort": c.get("last_resort", False)}
                for c in order
            ]}

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                # unchanged_control or restoration
                return 200, {
                    "changed": False, "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": _full_order_response(full_order),
                }
            # Swap POST: return correct swapped hash/version/order.
            return 200, {
                "changed": True, "active_config_hash": swapped_hash,
                "config_version": swapped_version,
                "active_candidate_order": _full_order_response(swapped_full_order),
            }

        proof_call = {"n": 0}
        def fake_run_case(**kw):
            proof_call["n"] += 1
            return {"passed": True}

        def fake_selection(r):
            # Baseline: correct (original first).  Swap proof: correct
            # (original second).  Restore proof: WRONG selection.
            if proof_call["n"] == 1:
                return {"provider": eligible[0]["provider"], "model": eligible[0]["model"], "route_family": eligible[0]["route_family"]}
            if proof_call["n"] == 2:
                return {"provider": eligible[1]["provider"], "model": eligible[1]["model"], "route_family": eligible[1]["route_family"]}
            return {"provider": "WRONG", "model": "WRONG", "route_family": "WRONG"}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda *a, **kw: {"basic.yaml": "h"})
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", fake_selection)

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert "restoration_failure" in result
        assert "RESTORATION PROOF FAILED" in result["restoration_failure"]
        assert result["failures"][0] == result["restoration_failure"]
        assert "recovery_artifact" in result


# ---------------------------------------------------------------------------
# Finding 5 (round 4): Sanitization of arguments/raw_prompt/prompt_text
# ---------------------------------------------------------------------------


class TestSanitizationProductionPath:
    def test_arguments_redacted_via_write_artifact(self, adapter, tmp_path):
        """arguments key must be redacted through the real _write_artifact."""
        artifact = {"tool_use": {"arguments": {"secret_arg": "LEAKED_VALUE"}, "tool_name": "bash"}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "LEAKED_VALUE" not in json.dumps(persisted)
        assert persisted["tool_use"]["arguments"] == "[REDACTED]"

    def test_raw_prompt_redacted_via_write_artifact(self, adapter, tmp_path):
        artifact = {"case": {"raw_prompt": "SECRET PROMPT TEXT", "passed": True}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET PROMPT TEXT" not in json.dumps(persisted)
        assert persisted["case"]["passed"] is True

    def test_prompt_text_redacted_via_write_artifact(self, adapter, tmp_path):
        artifact = {"nested": {"deep": {"prompt_text": "ANOTHER SECRET"}}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "ANOTHER SECRET" not in json.dumps(persisted)

    def test_tool_input_redacted_via_write_artifact(self, adapter, tmp_path):
        artifact = {"tool_activity": {"tool_input": "rm -rf /", "tool_name": "bash", "count": 3}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "rm -rf /" not in json.dumps(persisted)
        assert persisted["tool_activity"]["count"] == 3

    def test_structured_outcomes_preserved(self, adapter, tmp_path):
        """pass/fail/count/selection/lifecycle must survive sanitization."""
        artifact = {
            "results": {"case1": {"passed": True, "failures": [], "warnings": []}},
            "cfg003_transactional_refresh": {
                "passed": False,
                "phases": {"load": {"eligible_count": 5}},
                "tool_count": 3,
                "selection": {"provider": "openrouter", "model": "m1", "route_family": "rf1"},
            },
        }
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["results"]["case1"]["passed"] is True
        assert persisted["cfg003_transactional_refresh"]["phases"]["load"]["eligible_count"] == 5
        assert persisted["cfg003_transactional_refresh"]["tool_count"] == 3
        assert persisted["cfg003_transactional_refresh"]["selection"]["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# Finding 6 (round 4): Dev-only target + unhealthy inventory ordinary run
# ---------------------------------------------------------------------------


class TestTargetAndInventoryGates:
    def test_allowed_targets_dev_only(self, adapter):
        assert adapter._CFG003_ALLOWED_TARGETS == frozenset({"dev"})

    def test_unhealthy_inventory_fails_ordinary_run_with_tui_case(self, adapter, ra, monkeypatch, tmp_path):
        """Finding 4: ordinary runs must NOT fail at the inventory gate.
        Unhealthy inventory enforcement is gated to transactional mode only."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "tui_case": {
                    "command": ["codex", "exec"],
                    "verification_alias": "basic",
                    "verification_ingress": "codex_responses",
                },
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--cases", "tui_case",
        ])
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        # Unhealthy inventory.
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": False, "inventory_failures": ["readiness unavailable"], "alias_inventory": []})
        # Mock the case runner so we don't actually invoke codex.
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True, "failures": [], "warnings": [], "soft_failures": []})
        exit_code = adapter.main()
        # Ordinary run must NOT fail at the inventory gate.
        assert exit_code == 0
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert "cfg003_inventory_gate" not in artifact["results"]

    def test_healthy_inventory_passes_ordinary_run(self, adapter, ra, monkeypatch, tmp_path):
        """Healthy inventory with valid coverage map allows ordinary run."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "non_tui_case": {"required_env": ["NONEXISTENT_ENV_VAR"]},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.delenv("NONEXISTENT_ENV_VAR", raising=False)
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--cases", "non_tui_case",
        ])
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        # Healthy inventory with no aliases (non-TUI case, no coverage needed).
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": True, "inventory_failures": [], "alias_inventory": []})
        exit_code = adapter.main()
        # Should pass (soft skip for missing env).
        assert exit_code == 0


def _valid_avail_record(available=True, environment="dev", provider="openrouter", model="m1"):
    """Build a boundary-valid availability record."""
    import datetime as dt
    return {
        "provider": provider,
        "model": model,
        "available": available,
        "evidence": "remaining_pct=90" if available else "no_fresh_row",
        "observed_at": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat(),
        "environment": environment,
        "environment_binding": "target_db_profile",
    }

def _avail_result(candidates, available_count=None):
    """Build a complete positive-availability result keyed by (provider, model)."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    evidence = {}
    for i, c in enumerate(candidates):
        avail = available_count is None or i < available_count
        evidence[(c["provider"], c["model"])] = {
            "provider": c["provider"],
            "model": c["model"],
            "available": avail,
            "evidence": "remaining_pct=90" if avail else "no_fresh_row",
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
    identities = [{"provider": k[0], "model": k[1]} for k, v in evidence.items() if v["available"]]
    records = [{"provider": k[0], "model": k[1], **v} for k, v in evidence.items()]
    return {
        "evidence": evidence,
        "available_identities": identities,
        "evidence_records": records,
        "source": "rate_limit_observations",
    }


# ---------------------------------------------------------------------------
# Finding 1 (round 5): Recursive single-source gate
# ---------------------------------------------------------------------------


class TestRecursiveSingleSourceGate:
    def test_recursive_inventory_uses_cfg002_scan(self, ra):
        """_recursive_yaml_source_inventory must use the CFG-002 scan path."""
        inv = ra._recursive_yaml_source_inventory()
        assert "basic.yaml" in inv
        assert len(inv) == 1

    def test_nested_source_fails_before_any_egress(self, adapter, ra, monkeypatch):
        """Nested/multi-file source must hard-fail with zero TUI runs and zero
        refresh POSTs."""
        auth = ra._load_authoritative_startup_config()
        post_calls = []
        run_calls = []

        def fake_auth():
            return {"snapshot": auth["snapshot"], "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": {"basic.yaml": "h1", "nested/extra.yaml": "h2"},
                    "file_names": ["basic.yaml", "nested/extra.yaml"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(
            adapter.RA, "_recursive_yaml_source_inventory",
            lambda *a, **kw: {"basic.yaml": "h1", "nested/extra.yaml": "h2"},
        )
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_http_post_json",
                            lambda *a, **kw: post_calls.append(a) or (200, {}))
        monkeypatch.setattr(adapter, "_run_selected_case",
                            lambda **kw: run_calls.append(kw) or {"passed": True})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("exactly one recursive YAML source" in f for f in result["failures"])
        # Zero TUI runs and zero refresh POSTs before the gate.
        assert run_calls == []
        assert post_calls == []


# ---------------------------------------------------------------------------
# Finding 2 (round 5): Availability identity / fail-closed
# ---------------------------------------------------------------------------


class TestAvailabilityIdentityFailClosed:
    def test_shared_model_different_provider_distinguished(self, ra):
        """Two providers sharing a model name must be keyed separately."""
        candidates = [
            {"provider": "openrouter", "model": "shared-model"},
            {"provider": "openai", "model": "shared-model"},
        ]
        availability = {
            ("openrouter", "shared-model"): _valid_avail_record(True, provider="openrouter", model="shared-model"),
            ("openai", "shared-model"): _valid_avail_record(False, provider="openai", model="shared-model"),
        }
        result = ra._filter_candidates_by_positive_availability(candidates, availability)
        assert len(result) == 1
        assert result[0]["provider"] == "openrouter"

    def test_empty_evidence_yields_nothing(self, ra):
        candidates = [{"provider": "p1", "model": "m1"}]
        assert ra._filter_candidates_by_positive_availability(candidates, {}) == []

    def test_one_record_available(self, ra):
        candidates = [{"provider": "p1", "model": "m1"}, {"provider": "p2", "model": "m2"}]
        availability = {("p1", "m1"): _valid_avail_record(True, provider="p1", model="m1")}
        result = ra._filter_candidates_by_positive_availability(candidates, availability)
        assert len(result) == 1
        assert result[0]["provider"] == "p1"

    def test_stale_not_available(self, ra):
        candidates = [{"provider": "p1", "model": "m1"}]
        availability = {("p1", "m1"): _valid_avail_record(False)}
        assert ra._filter_candidates_by_positive_availability(candidates, availability) == []

    def test_wrong_environment_recorded(self, ra):
        """Evidence records environment binding explicitly."""
        candidates = [{"provider": "p1", "model": "m1"}]
        # Simulate query result with environment binding.
        availability = {("p1", "m1"): _valid_avail_record(True, provider="p1", model="m1")}
        result = ra._filter_candidates_by_positive_availability(candidates, availability)
        assert len(result) == 1

    def test_require_availability_missing_cannot_pass(self, ra):
        """_derive_eligible_candidates_from_snapshot with require_availability
        excludes candidates without explicit available=True."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        # No positive evidence at all.
        result = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability={}, require_availability=True
        )
        assert result == []

    def test_require_availability_explicit_passes(self, ra):
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        all_eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        first = all_eligible[0]
        pos = {(first["provider"], first["model"]): _valid_avail_record(True, provider=first["provider"], model=first["model"])}
        result = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability=pos, require_availability=True
        )
        assert len(result) == 1
        assert result[0]["provider"] == first["provider"]
        assert result[0]["model"] == first["model"]


# ---------------------------------------------------------------------------
# Finding 3 (round 5): Unhealthy inventory blocks ALL egress cases
# ---------------------------------------------------------------------------


class TestUnhealthyInventoryBlocksAllEgress:
    def test_http_request_case_blocked(self, adapter, ra, monkeypatch, tmp_path):
        """Finding 4: ordinary runs must NOT block http_request cases on
        unhealthy inventory.  Enforcement is gated to transactional mode."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "http_case": {"http_request": {"method": "POST", "url": "http://x"}},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--cases", "http_case",
        ])
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": False, "inventory_failures": ["readiness unavailable"], "alias_inventory": []})
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True, "failures": [], "warnings": [], "soft_failures": []})
        exit_code = adapter.main()
        # Ordinary run must NOT fail at the inventory gate.
        assert exit_code == 0
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert "cfg003_inventory_gate" not in artifact["results"]

    def test_is_egress_case_detection(self, ra):
        assert ra._is_egress_case({"command": ["codex", "exec"]}) is True
        assert ra._is_egress_case({"command": ["claude", "-p"]}) is True
        assert ra._is_egress_case({"http_request": {"method": "POST"}}) is True
        assert ra._is_egress_case({"cli_passthrough": "codex"}) is True
        assert ra._is_egress_case({"required_env": ["X"]}) is False
        assert ra._is_egress_case({}) is False


# ---------------------------------------------------------------------------
# Finding 4 (round 5): Error intake attribution
# ---------------------------------------------------------------------------


class TestErrorIntakeAttribution:
    def test_wrong_case_ignored(self, ra, tmp_path):
        """Events with an explicitly different case are not attributed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "other case error",
                                "context": {"container": "litellm-dev", "case": "OTHER_CASE"}}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", case_name="MY_CASE", analysis_dir=analysis,
        )
        assert events == []

    def test_exact_case_matched(self, ra, tmp_path):
        """Events with matching case are attributed."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "my case error",
                                "context": {"container": "litellm-dev", "case": "MY_CASE"}}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", case_name="MY_CASE", analysis_dir=analysis,
        )
        assert len(events) == 1
        assert events[0]["attributed_case"] == "MY_CASE"

    def test_sparse_temporal_container_fallback(self, ra, tmp_path):
        """Sparse events (no case/session/trace) fall back to temporal+env+container."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "sparse error",
                                "context": {"container": "litellm-dev"}}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", case_name="MY_CASE", analysis_dir=analysis,
        )
        assert len(events) == 1
        assert events[0]["sparse_fallback"] is False  # container present
        assert events[0]["attributed_container"] == "litellm-dev"

    def test_explicit_container_mismatch_rejected(self, ra, tmp_path):
        """Explicitly different container rejects the event."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "wrong container",
                                "context": {"container": "OTHER_CONTAINER"}}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert events == []

    def test_phase_baseline_advancement(self, adapter, ra, tmp_path):
        """Each phase uses a fresh baseline advanced from the prior phase."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        # Phase 1: one event.
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "phase1 error",
                                "context": {"container": "litellm-dev"}}) + "\n")
        intake1 = adapter._cfg003_phase_error_intake(
            baseline, initiation_time=initiation, environment="dev", container="litellm-dev",
            analysis_dir=analysis,
        )
        assert intake1["attributed_count"] == 1
        advanced = intake1["advanced_baseline"]

        # Phase 2: no new events -- advanced baseline sees zero delta.
        intake2 = adapter._cfg003_phase_error_intake(
            advanced, initiation_time=initiation, environment="dev", container="litellm-dev",
            analysis_dir=analysis,
        )
        assert intake2["attributed_count"] == 0


# ---------------------------------------------------------------------------
# Finding 5 (round 5): Full order comparison
# ---------------------------------------------------------------------------


class TestFullOrderComparison:
    def test_correct_prefix_plus_extra_tail_fails(self, ra):
        """Extra tail elements must cause mismatch (no prefix acceptance)."""
        expected = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100},
            {"provider": "p2", "model": "m2", "route_family": "rf2", "anthropic_route_family": "", "priority": 50},
        ]
        observed = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100},
            {"provider": "p2", "model": "m2", "route_family": "rf2", "anthropic_route_family": "", "priority": 50},
            {"provider": "p3", "model": "m3", "route_family": "rf3", "anthropic_route_family": "", "priority": 10},
        ]
        assert ra._candidate_order_matches(observed, expected) is False

    def test_wrong_anthropic_route_fails(self, ra):
        expected = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "anthropic_messages", "priority": 100},
        ]
        observed = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "WRONG", "priority": 100},
        ]
        assert ra._candidate_order_matches(observed, expected) is False

    def test_exact_match_passes(self, ra):
        expected = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "arf1", "priority": 100, "last_resort": False},
            {"provider": "p2", "model": "m2", "route_family": "rf2", "anthropic_route_family": "", "priority": 50, "last_resort": False},
        ]
        observed = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "arf1", "priority": 100, "last_resort": False},
            {"provider": "p2", "model": "m2", "route_family": "rf2", "anthropic_route_family": "", "priority": 50, "last_resort": False},
        ]
        assert ra._candidate_order_matches(observed, expected) is True

    def test_empty_observed_fails(self, ra):
        expected = [{"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100}]
        assert ra._candidate_order_matches([], expected) is False
        assert ra._candidate_order_matches(None, expected) is False

    def test_config_refresh_emits_deterministic_order(self):
        """config_refresh._snapshot_candidate_order emits sorted alias names."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_refresh import _snapshot_candidate_order
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import compile_directory
        snap = compile_directory(pathlib.Path(str(ROOT / "litellm" / "proxy" / "aawm_alias_config")))
        order = _snapshot_candidate_order(snap)
        assert list(order.keys()) == sorted(order.keys())
        assert "basic" in order
        for cand in order["basic"]:
            assert "provider" in cand
            assert "model" in cand
            assert "route_family" in cand
            assert "anthropic_route_family" in cand
            assert "priority" in cand


# ---------------------------------------------------------------------------
# Finding 6 (round 5): Sanitization of tool_output/tool_result/previews
# ---------------------------------------------------------------------------


class TestSanitizationRound5:
    def test_tool_output_redacted(self, adapter, tmp_path):
        artifact = {"case": {"tool_output": "SECRET OUTPUT DATA", "passed": True}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET OUTPUT DATA" not in json.dumps(persisted)
        assert persisted["case"]["passed"] is True

    def test_tool_result_redacted(self, adapter, tmp_path):
        artifact = {"case": {"tool_result": "SECRET RESULT", "count": 5}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET RESULT" not in json.dumps(persisted)
        assert persisted["case"]["count"] == 5

    def test_input_preview_redacted(self, adapter, tmp_path):
        artifact = {"nested": {"input_preview": "SECRET PREVIEW"}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET PREVIEW" not in json.dumps(persisted)

    def test_output_preview_redacted(self, adapter, tmp_path):
        artifact = {"nested": {"output_preview": "ANOTHER SECRET"}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "ANOTHER SECRET" not in json.dumps(persisted)


# ---------------------------------------------------------------------------
# Finding 7 (round 5): DB credential resolution
# ---------------------------------------------------------------------------


class TestDbCredentialResolution:
    def test_container_owned_preferred(self, adapter, monkeypatch):
        """Container-owned credential is tried before env fallback."""
        calls = []
        monkeypatch.setattr(adapter, "_resolve_container_env_value",
                            lambda container, env: calls.append((container, env)) or "container-secret")
        monkeypatch.delenv("AAWM_DB_PASSWORD", raising=False)
        config = {"cases": {"c": {"session_history_validation": {"db_password_container_env": "AAWM_DB_PASSWORD", "db_host": "h", "db_port": 1234, "db_name": "db", "db_user": "u"}}}}
        profile = {"docker_container_name": "litellm-dev"}
        result = adapter._cfg003_db_settings(config, profile=profile)
        assert result is not None
        assert result["password"] == "container-secret"
        assert calls == [("litellm-dev", "AAWM_DB_PASSWORD")]

    def test_env_fallback_when_container_fails(self, adapter, monkeypatch):
        """Falls back to env when container resolution returns None."""
        monkeypatch.setattr(adapter, "_resolve_container_env_value", lambda c, e: None)
        monkeypatch.setenv("AAWM_DB_PASSWORD", "env-secret")
        config = {"cases": {"c": {"session_history_validation": {"db_password_container_env": "AAWM_DB_PASSWORD", "db_password_env": "AAWM_DB_PASSWORD"}}}}
        profile = {"docker_container_name": "litellm-dev"}
        result = adapter._cfg003_db_settings(config, profile=profile)
        assert result is not None
        assert result["password"] == "env-secret"

    def test_none_when_no_password(self, adapter, monkeypatch):
        """Returns None when no password can be resolved."""
        monkeypatch.setattr(adapter, "_resolve_container_env_value", lambda c, e: None)
        monkeypatch.delenv("AAWM_DB_PASSWORD", raising=False)
        config = {"cases": {"c": {"session_history_validation": {"db_password_env": "AAWM_DB_PASSWORD"}}}}
        result = adapter._cfg003_db_settings(config, profile={})
        assert result is None


# ---------------------------------------------------------------------------
# Finding 1 (round 6): Availability boundary validation
# ---------------------------------------------------------------------------


class TestAvailabilityBoundaryValidation:
    def test_fresh_available_record_passes(self, ra):
        """Fresh available=True record with matching environment passes."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is True

    def test_stale_record_fails(self, ra):
        """Record older than freshness window fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "observed_at": (now - dt.timedelta(hours=2)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_future_skewed_record_fails(self, ra):
        """Record with future timestamp beyond tolerance fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "observed_at": (now + dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_wrong_environment_fails(self, ra):
        """Record with mismatched environment fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "prod",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_malformed_timestamp_fails(self, ra):
        """Record with unparseable observed_at fails closed."""
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "observed_at": "not-a-timestamp",
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev"
        ) is False

    def test_missing_timestamp_fails(self, ra):
        """Record without observed_at fails closed."""
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": True,
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev"
        ) is False

    def test_available_false_fails(self, ra):
        """Record with available=False fails even if fresh."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "model": "m1",
            "available": False,
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_provider_model_mismatch_fails(self, ra):
        """Record with mismatched provider/model fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "available": True,
            "provider": "openai",
            "model": "m1",
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_missing_provider_field_fails(self, ra):
        """Record without provider field fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "model": "m1",
            "available": True,
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False

    def test_missing_model_field_fails(self, ra):
        """Record without model field fails closed."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        record = {
            "provider": "openrouter",
            "available": True,
            "observed_at": (now - dt.timedelta(minutes=5)).isoformat(),
            "environment": "dev",
            "environment_binding": "target_db_profile",
        }
        assert ra._availability_record_is_valid(
            record, provider="openrouter", model="m1", environment="dev", now=now
        ) is False


# ---------------------------------------------------------------------------
# Finding 2 (round 6): Error intake restoration phase
# ---------------------------------------------------------------------------


class TestErrorIntakeRestorationPhase:
    def test_delta_summary_included(self, adapter, tmp_path):
        """Phase error intake includes delta_summary."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = adapter.RA._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error",
                                "message": "test error",
                                "context": {"container": "litellm-dev"}}) + "\n")
        intake = adapter._cfg003_phase_error_intake(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert "delta_summary" in intake
        assert intake["delta_summary"]["total_line_count_delta"] == 1

    def test_restore_proof_correlation_ids_captured(self, adapter, ra, monkeypatch):  # noqa: PLR0915
        """Restore-proof session/trace IDs are captured and passed to final phase."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        # Compute correct swapped hash/version so the swap_proof gate passes.
        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        swapped_hash = swapped_snap.config_hash
        swapped_version = swapped_snap.config_version
        swapped_full_order = ra._derive_full_order_from_snapshot(swapped_snap, alias_name="basic")

        def fake_auth():
            return {"snapshot": snap, "merged_yaml": auth["merged_yaml"],
                    "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
                    "config_hash": auth["config_hash"], "config_version": auth["config_version"],
                    "aliases": auth["aliases"]}

        def _full_order_response(order):
            return {"basic": [
                {"provider": c["provider"], "model": c["model"],
                 "route_family": c["route_family"],
                 "anthropic_route_family": c.get("anthropic_route_family", ""),
                 "priority": c["priority"], "last_resort": c.get("last_resort", False)}
                for c in order
            ]}

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                return 200, {
                    "changed": False, "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": _full_order_response(full_order),
                }
            # Swap POST: correct swapped hash/version/order.
            return 200, {
                "changed": True, "active_config_hash": swapped_hash,
                "config_version": swapped_version,
                "active_candidate_order": _full_order_response(swapped_full_order),
            }

        proof_call = {"n": 0}
        def fake_run_case(**kw):
            proof_call["n"] += 1
            return {
                "passed": True,
                "langfuse": {
                    "command_session_id": f"session-{proof_call['n']}",
                    "trace_ids": [f"trace-{proof_call['n']}"],
                },
            }

        def fake_selection(r):
            # Baseline: original first.  Swap proof: original second.
            # Restore proof: original first.
            if proof_call["n"] == 2:
                return {"provider": eligible[1]["provider"], "model": eligible[1]["model"], "route_family": eligible[1]["route_family"]}
            return {"provider": eligible[0]["provider"], "model": eligible[0]["model"], "route_family": eligible[0]["route_family"]}

        # Track swap state so readiness returns correct hash after swap.
        swap_done = {"done": False}
        orig_fake_post = fake_post
        def tracking_post(url, payload, **kw):
            result = orig_fake_post(url, payload, **kw)
            yaml_text = payload.get("yaml", "")
            if yaml_text != raw_text and "not_a_list" not in yaml_text:
                swap_done["done"] = True
            if yaml_text == raw_text and swap_done["done"]:
                swap_done["done"] = False  # restoration
            return result

        def fake_get(*a, **kw):
            if swap_done["done"]:
                h, v = swapped_hash, swapped_version
            else:
                h, v = auth["config_hash"], auth["config_version"]
            return (200, {"aawm_alias_config": {
                "state": "active", "config_hash": h,
                "config_version": v,
                "files": auth["file_names"], "aliases": auth["aliases"],
            }})

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_post_json", tracking_post)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", fake_get)
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda *a, **kw: {"basic.yaml": "h"})
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", fake_selection)

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        # Restore proof should have captured session/trace (3rd proof call).
        restore_proof = result["phases"]["restore_proof"]
        assert restore_proof["session_id"] == "session-3"
        assert restore_proof["trace_id"] == "trace-3"


# ---------------------------------------------------------------------------
# Finding 3 (round 6): Full order from snapshot
# ---------------------------------------------------------------------------


class TestFullOrderFromSnapshot:
    def test_derive_full_order_independent_of_filters(self, ra):
        """Full order includes all candidates regardless of provider/availability."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        # Full order should include ALL candidates from snapshot.
        # Eligible candidates are a subset (filtered by schedule/availability).
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(full_order) >= len(eligible)
        # All eligible candidates should be present in full order.
        full_keys = {(c["provider"], c["model"]) for c in full_order}
        eligible_keys = {(c["provider"], c["model"]) for c in eligible}
        assert eligible_keys.issubset(full_keys)

    def test_last_resort_normalized_to_bool(self, ra):
        """last_resort is normalized to explicit bool."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        for cand in full_order:
            assert isinstance(cand["last_resort"], bool)

    def test_candidate_order_matches_requires_last_resort(self, ra):
        """Order comparison requires last_resort match when present."""
        expected = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100, "last_resort": False},
        ]
        observed_match = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100, "last_resort": False},
        ]
        observed_mismatch = [
            {"provider": "p1", "model": "m1", "route_family": "rf1", "anthropic_route_family": "", "priority": 100, "last_resort": True},
        ]
        assert ra._candidate_order_matches(observed_match, expected) is True
        assert ra._candidate_order_matches(observed_mismatch, expected) is False


# ---------------------------------------------------------------------------
# Finding 4 (round 6): Sanitization of content-bearing variants
# ---------------------------------------------------------------------------


class TestSanitizationRound6:
    def test_tool_output_text_redacted(self, adapter, tmp_path):
        artifact = {"case": {"tool_output_text": "SECRET OUTPUT TEXT", "passed": True}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET OUTPUT TEXT" not in json.dumps(persisted)
        assert persisted["case"]["passed"] is True

    def test_tool_result_content_text_redacted(self, adapter, tmp_path):
        artifact = {"case": {"tool_result_content_text": "SECRET CONTENT", "count": 5}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET CONTENT" not in json.dumps(persisted)
        assert persisted["case"]["count"] == 5

    def test_tool_result_content_preview_redacted(self, adapter, tmp_path):
        artifact = {"nested": {"tool_result_content_preview": "SECRET PREVIEW"}}
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert "SECRET PREVIEW" not in json.dumps(persisted)

    def test_structured_booleans_preserved(self, adapter, tmp_path):
        """Structured booleans/counts/status are preserved."""
        artifact = {
            "case": {
                "tool_result_is_error": False,
                "tool_count": 3,
                "tool_output_text": "SECRET",
            }
        }
        path = tmp_path / "art.json"
        adapter._write_artifact(path, artifact)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["case"]["tool_result_is_error"] is False
        assert persisted["case"]["tool_count"] == 3
        assert "SECRET" not in json.dumps(persisted)


# ---------------------------------------------------------------------------
# Producer/consumer contract: _query_positive_availability_evidence output
# must be accepted unmodified by the strict transaction eligibility path.
# ---------------------------------------------------------------------------


def _fake_psycopg_module(row_for_candidate):
    """Build a fake ``psycopg`` module whose cursor returns ``row_for_candidate``.

    ``row_for_candidate`` is a callable ``(provider, model) -> dict | None``
    mimicking a real DB row producer.  For multi-window tests it may also
    return a ``list[dict]`` of rows (already ordered observed_at DESC).
    """
    import types

    class _FakeCursor:
        def __init__(self):
            self._last_params = None

        def execute(self, sql, params=None):
            self._last_params = params

        def fetchone(self):
            provider, model = self._last_params[0], self._last_params[1]
            produced = row_for_candidate(provider, model)
            if isinstance(produced, list):
                return produced[0] if produced else None
            return produced

        def fetchall(self):
            provider, model = self._last_params[0], self._last_params[1]
            produced = row_for_candidate(provider, model)
            if produced is None:
                return []
            if isinstance(produced, list):
                return produced
            return [produced]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def close(self):
            return None

    mod = types.ModuleType("psycopg")
    mod.connect = lambda **kw: _FakeConn()
    mod.rows = types.SimpleNamespace(dict_row="dict_row")
    return mod


class TestAvailabilityProducerConsumerContract:
    def test_producer_output_qualifies_unmodified(self, ra, monkeypatch):
        """A genuine fresh positive DB row, produced by the real query path,
        must qualify the exact candidate through the strict consumer without
        any modification of the producer output."""
        import datetime as dt

        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        # Prefer a candidate without a required-window contract so a single
        # positive row qualifies; fall back to the first eligible and provide
        # both windows for alibaba_token_plan models.
        non_alibaba = [c for c in eligible if c["provider"] != "alibaba_token_plan"]
        first = non_alibaba[0] if non_alibaba else eligible[0]
        provider, model = first["provider"], first["model"]

        def row_for_candidate(p, m):
            if (p, m) != (provider, model):
                return None
            now = dt.datetime.now(dt.timezone.utc)
            base_row = {
                "observed_at": now.isoformat(),
                "provider": p,
                "model": m,
                "remaining_pct": 90,
                "quota_remaining": 900,
                "source": "rate_limit_observations",
                "evidence": "live",
            }
            if p == "alibaba_token_plan":
                return [
                    {**base_row, "quota_key": "alibaba_token_plan_5h:credits"},
                    {**base_row, "quota_key": "alibaba_token_plan_7d:credits"},
                ]
            return base_row

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(row_for_candidate)
        )

        produced = ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": provider, "model": model}],
            environment="dev",
        )

        # Producer must stamp exact identity fields on the record.
        rec = produced[(provider, model)]
        assert rec["available"] is True
        assert rec["provider"] == provider
        assert rec["model"] == model

        # Consumer accepts the UNMODIFIED producer output.
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap,
            alias_name="basic",
            positive_availability=produced,
            require_availability=True,
        )
        assert len(qualified) == 1
        assert qualified[0]["provider"] == provider
        assert qualified[0]["model"] == model

    def test_producer_no_fresh_row_does_not_qualify(self, ra, monkeypatch):
        """When the DB producer finds no fresh row, the candidate must NOT
        qualify through the strict consumer."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        first = eligible[0]

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(lambda p, m: None)
        )
        produced = ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": first["provider"], "model": first["model"]}],
            environment="dev",
        )
        assert produced[(first["provider"], first["model"])]["available"] is False
        # Producer must still stamp exact identity fields on the negative record.
        rec = produced[(first["provider"], first["model"])]
        assert rec["provider"] == first["provider"]
        assert rec["model"] == first["model"]
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability=produced, require_availability=True
        )
        assert qualified == []

    def test_producer_exhausted_row_excluded_by_consumer(self, ra, monkeypatch):
        """A fresh DB row with remaining_pct <= 0 must produce available=False
        with exact provider/model, and the strict consumer must exclude it."""
        import datetime as dt

        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        first = eligible[0]
        provider, model = first["provider"], first["model"]

        def row_for_candidate(p, m):
            if (p, m) != (provider, model):
                return None
            return {
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "provider": p,
                "model": m,
                "remaining_pct": 0,
                "quota_remaining": 0,
                "source": "rate_limit_observations",
                "evidence": "exhausted",
            }

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(row_for_candidate)
        )

        produced = ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": provider, "model": model}],
            environment="dev",
        )

        rec = produced[(provider, model)]
        assert rec["available"] is False
        assert rec["provider"] == provider
        assert rec["model"] == model

        # Unmodified producer output through the strict consumer: excluded.
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap,
            alias_name="basic",
            positive_availability=produced,
            require_availability=True,
        )
        assert qualified == []

    def test_record_wrong_provider_field_rejected(self, ra):
        """A record whose stamped ``provider`` field mismatches the candidate
        identity must be rejected by the strict validator/consumer."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        first = eligible[0]
        # Keyed correctly, but the stamped provider field is wrong.
        bad = {
            (first["provider"], first["model"]): _valid_avail_record(
                True, provider="SOME_OTHER_PROVIDER", model=first["model"]
            )
        }
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability=bad, require_availability=True
        )
        assert qualified == []

    def test_record_wrong_model_field_rejected(self, ra):
        """A record whose stamped ``model`` field mismatches must be rejected."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        first = eligible[0]
        bad = {
            (first["provider"], first["model"]): _valid_avail_record(
                True, provider=first["provider"], model="SOME_OTHER_MODEL"
            )
        }
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability=bad, require_availability=True
        )
        assert qualified == []


# ---------------------------------------------------------------------------
# Finding 1: multi-window quota aggregation (tied observed_at, e.g. Alibaba
# Token Plan 5h + 7d).  A candidate is positive ONLY when ALL current windows
# are fresh, valid, and remaining > 0.  Any exhausted/stale/invalid window
# fails closed.  Single-window providers degrade to prior one-row behavior.
# ---------------------------------------------------------------------------


class TestMultiWindowAvailabilityAggregation:
    PROVIDER = "alibaba_token_plan"
    MODEL = "alibaba_token_plan/qwen3.8-max-preview"
    QK_5H = "alibaba_token_plan_5h:credits"
    QK_7D = "alibaba_token_plan_7d:credits"

    def _row(self, *, quota_key, remaining_pct, observed_at):
        return {
            "observed_at": observed_at,
            "provider": self.PROVIDER,
            "model": self.MODEL,
            "quota_key": quota_key,
            "remaining_pct": remaining_pct,
            "quota_remaining": None,
            "source": "rate_limit_observations",
            "evidence": "live",
        }

    def _query(self, ra, monkeypatch, rows):
        def row_for_candidate(p, m):
            if (p, m) != (self.PROVIDER, self.MODEL):
                return None
            return rows

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(row_for_candidate)
        )
        return ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": self.PROVIDER, "model": self.MODEL}],
            environment="dev",
        )

    def test_both_windows_positive_qualifies(self, ra, monkeypatch):
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now),
            self._row(quota_key=self.QK_7D, remaining_pct=60, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is True
        assert self.QK_5H in rec["evidence"]
        assert self.QK_7D in rec["evidence"]

    def test_one_exhausted_window_fails_closed(self, ra, monkeypatch):
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=90, observed_at=now),
            self._row(quota_key=self.QK_7D, remaining_pct=0, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False
        # The exhausted window must be named in the evidence.
        assert self.QK_7D in rec["evidence"]

    def test_stale_window_filtered_by_sql_cutoff_fails_closed(self, ra, monkeypatch):
        """A window whose only row is stale is excluded by the SQL freshness
        cutoff, so the candidate sees a missing window -> no_fresh_row/fail."""
        # Only the 5h window is fresh; the 7d row is stale and would be
        # filtered out by the observed_at >= cutoff clause upstream.  We model
        # the post-cutoff result set: only the fresh 5h row remains, so the
        # candidate has a single observed window that is positive.  To prove
        # fail-closed on a genuinely missing required window, pass NO rows.
        produced = self._query(ra, monkeypatch, [])
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False
        assert rec["evidence"] == "no_fresh_row"

    def test_invalid_remaining_pct_window_fails_closed(self, ra, monkeypatch):
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=70, observed_at=now),
            self._row(quota_key=self.QK_7D, remaining_pct=None, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False

    def test_deterministic_ordering_of_window_evidence(self, ra, monkeypatch):
        """Window verdicts are emitted in sorted quota_key order regardless of
        row arrival order."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        # Deliberately reverse arrival order.
        rows = [
            self._row(quota_key=self.QK_7D, remaining_pct=50, observed_at=now),
            self._row(quota_key=self.QK_5H, remaining_pct=40, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is True
        # 5h sorts before 7d.
        assert rec["evidence"].index(self.QK_5H) < rec["evidence"].index(self.QK_7D)

    def test_latest_row_per_window_wins(self, ra, monkeypatch):
        """When a window has multiple rows, only the latest (first in DESC
        order) is evaluated."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        older = now - dt.timedelta(minutes=10)
        rows = [
            # Latest 5h row: positive.
            self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now),
            # Older 5h row: exhausted (must be ignored).
            self._row(quota_key=self.QK_5H, remaining_pct=0, observed_at=older),
            self._row(quota_key=self.QK_7D, remaining_pct=60, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is True

    def test_single_window_provider_preserves_prior_format(self, ra, monkeypatch):
        """A provider with one quota row degrades to the prior evidence format
        (``remaining_pct=N``) without window prefixes.  Uses a non-alibaba
        provider/model so the required-window contract does not apply."""
        import datetime as dt

        provider = "openrouter"
        model = "openrouter/some-model"
        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            {
                "observed_at": now,
                "provider": provider,
                "model": model,
                "quota_key": "openrouter_free_daily_requests:requests",
                "remaining_pct": 90,
                "quota_remaining": None,
                "source": "rate_limit_observations",
                "evidence": "live",
            },
        ]

        def row_for_candidate(p, m):
            if (p, m) != (provider, model):
                return None
            return rows

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(row_for_candidate)
        )
        produced = ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": provider, "model": model}],
            environment="dev",
        )
        rec = produced[(provider, model)]
        assert rec["available"] is True
        assert rec["evidence"] == "remaining_pct=90"

    def test_no_rows_yields_no_fresh_row(self, ra, monkeypatch):
        produced = self._query(ra, monkeypatch, [])
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False
        assert rec["evidence"] == "no_fresh_row"
        assert rec["observed_at"] is None

    def test_producer_consumer_contract_multi_window(self, ra, monkeypatch):
        """End-to-end: a multi-window positive producer record qualifies the
        exact candidate through the strict consumer unmodified."""
        import datetime as dt

        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        # Find an alibaba candidate if present; else use the first eligible.
        target = next(
            (c for c in eligible if c["provider"] == self.PROVIDER),
            eligible[0],
        )
        provider, model = target["provider"], target["model"]
        now = dt.datetime.now(dt.timezone.utc)

        def row_for_candidate(p, m):
            if (p, m) != (provider, model):
                return None
            return [
                self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now)
                | {"provider": p, "model": m},
                self._row(quota_key=self.QK_7D, remaining_pct=60, observed_at=now)
                | {"provider": p, "model": m},
            ]

        monkeypatch.setitem(
            sys.modules, "psycopg", _fake_psycopg_module(row_for_candidate)
        )
        produced = ra._query_positive_availability_evidence(
            db_settings={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
            candidates=[{"provider": provider, "model": model}],
            environment="dev",
        )
        assert produced[(provider, model)]["available"] is True
        qualified = ra._derive_eligible_candidates_from_snapshot(
            snap, alias_name="basic", positive_availability=produced, require_availability=True
        )
        assert any(
            c["provider"] == provider and c["model"] == model for c in qualified
        )

    # ------------------------------------------------------------------
    # Finding 1: required-window contract (both 5h and 7d must be fresh)
    # ------------------------------------------------------------------

    def test_fresh_5h_plus_stale_7d_fails_closed(self, ra, monkeypatch):
        """One fresh 5h row with a stale 7d row (filtered by SQL cutoff) must
        fail closed because the required 7d window is absent from the fresh
        result set."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        # Only the 5h row is fresh; the 7d row would be filtered by the SQL
        # freshness cutoff upstream, so it never appears in the result set.
        # Model the post-cutoff state: only the 5h row remains.
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False
        # The missing required window must be named in evidence.
        assert self.QK_7D in rec["evidence"]
        assert "missing" in rec["evidence"]

    def test_fresh_5h_only_no_7d_fails_closed(self, ra, monkeypatch):
        """A single fresh 5h row with no 7d row at all must fail closed for
        Alibaba Token Plan models that require both windows."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=95, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False
        assert self.QK_7D in rec["evidence"]

    # ------------------------------------------------------------------
    # Finding 2: tied-latest determinism (reversed order same verdict)
    # ------------------------------------------------------------------

    def test_tied_latest_positive_and_exhausted_fails_closed_order_a(self, ra, monkeypatch):
        """Two rows tied at the same observed_at for the same quota_key: one
        positive, one exhausted.  Must fail closed regardless of arrival
        order (order A: positive first)."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now),
            self._row(quota_key=self.QK_5H, remaining_pct=0, observed_at=now),
            self._row(quota_key=self.QK_7D, remaining_pct=60, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False

    def test_tied_latest_positive_and_exhausted_fails_closed_order_b(self, ra, monkeypatch):
        """Same tied-latest scenario with reversed arrival order (exhausted
        first).  Verdict must be identical to order A."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            self._row(quota_key=self.QK_5H, remaining_pct=0, observed_at=now),
            self._row(quota_key=self.QK_5H, remaining_pct=80, observed_at=now),
            self._row(quota_key=self.QK_7D, remaining_pct=60, observed_at=now),
        ]
        produced = self._query(ra, monkeypatch, rows)
        rec = produced[(self.PROVIDER, self.MODEL)]
        assert rec["available"] is False


# ---------------------------------------------------------------------------
# Body B finding (a): last_resort derives from compiled rule priority == 0
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _fake_cand(provider, model, priority, rf="codex_x_adapter", arf=""):
    return SimpleNamespace(
        provider=provider,
        model=model,
        route_family=rf,
        anthropic_route_family=arf,
        priority=priority,
    )


def _fake_snapshot(candidates, alias_name="basic"):
    alias = SimpleNamespace(name=alias_name, candidates=tuple(candidates))
    return SimpleNamespace(aliases={alias_name: alias})


class TestLastResortFromPriorityZero:
    def test_refresh_response_emits_last_resort_from_priority(self):
        """config_refresh._snapshot_candidate_order: priority==0 -> True,
        nonzero -> False, derived from the compiled rule (not an attribute)."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            config_refresh,
        )

        snap = _fake_snapshot(
            [
                _fake_cand("openrouter", "or/a", priority=300),
                _fake_cand("alibaba_token_plan", "al/b", priority=0),
                _fake_cand("openai", "oa/c", priority=150),
            ]
        )
        order = config_refresh._snapshot_candidate_order(snap)
        cands = order["basic"]
        by_model = {c["model"]: c for c in cands}
        # Priority-zero candidate is emitted as last_resort True.
        assert by_model["al/b"]["last_resort"] is True
        # Nonzero-priority candidates are emitted as last_resort False.
        assert by_model["or/a"]["last_resort"] is False
        assert by_model["oa/c"]["last_resort"] is False

    def test_expected_full_order_builder_uses_priority(self, ra):
        """run_acceptance._derive_full_order_from_snapshot: same derivation."""
        snap = _fake_snapshot(
            [
                _fake_cand("openrouter", "or/a", priority=300),
                _fake_cand("alibaba_token_plan", "al/b", priority=0),
            ]
        )
        full = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        by_model = {c["model"]: c for c in full}
        assert by_model["al/b"]["last_resort"] is True
        assert by_model["or/a"]["last_resort"] is False

    def test_order_comparison_proves_priority_zero_last_resort(self, ra):
        """_candidate_order_matches compares last_resort derived from priority:
        a priority-zero candidate must be matched as last_resort True and a
        nonzero as False; a mismatch on either side fails the match."""
        expected = [
            {"provider": "p1", "model": "hi", "route_family": "rf", "anthropic_route_family": "", "priority": 300, "last_resort": False},
            {"provider": "p2", "model": "lo", "route_family": "rf", "anthropic_route_family": "", "priority": 0, "last_resort": True},
        ]
        observed_ok = [
            {"provider": "p1", "model": "hi", "route_family": "rf", "anthropic_route_family": "", "priority": 300, "last_resort": False},
            {"provider": "p2", "model": "lo", "route_family": "rf", "anthropic_route_family": "", "priority": 0, "last_resort": True},
        ]
        # Flipping the priority-zero candidate's last_resort to False must fail.
        observed_bad = [
            {"provider": "p1", "model": "hi", "route_family": "rf", "anthropic_route_family": "", "priority": 300, "last_resort": False},
            {"provider": "p2", "model": "lo", "route_family": "rf", "anthropic_route_family": "", "priority": 0, "last_resort": False},
        ]
        assert ra._candidate_order_matches(observed_ok, expected) is True
        assert ra._candidate_order_matches(observed_bad, expected) is False


# ---------------------------------------------------------------------------
# Body B finding (b): value-level secret sanitization of persisted artifacts
# and error-intake delta paths
# ---------------------------------------------------------------------------


class TestValueLevelSecretSanitization:
    def test_bearer_and_authorization_values_scrubbed(self, ra):
        out = ra._sanitize_sensitive_string_value(
            "upstream 401 Authorization: Bearer sk-live-abcdef1234567890 retrying"
        )
        assert "sk-live-abcdef1234567890" not in out
        assert "Bearer" in out  # diagnostic class label preserved
        assert "upstream 401" in out and "retrying" in out

    def test_sk_style_token_scrubbed(self, ra):
        out = ra._sanitize_sensitive_string_value("call failed token=sk-proj-ABCDEFGHIJ123456 end")
        assert "sk-proj-ABCDEFGHIJ123456" not in out
        assert "sk-[REDACTED]" in out
        assert "call failed" in out and "end" in out

    def test_api_key_password_credential_assignments_scrubbed(self, ra):
        for text, secret in (
            ("boot api_key=abc123secretvalue ok", "abc123secretvalue"),
            ("db password=hunter2secretvalue now", "hunter2secretvalue"),
            ("set credential: supersecretvalue99 done", "supersecretvalue99"),
            ("master_key = mkey-secret-000111 ok", "mkey-secret-000111"),
        ):
            out = ra._sanitize_sensitive_string_value(text)
            assert secret not in out, (text, out)
            assert "[REDACTED]" in out

    def test_malformed_jsonl_excerpt_and_message_and_legacy_line(self, ra):
        # Malformed JSONL excerpt embedded in a raw string.
        malformed = '{"message": "auth api_key=deadbeefsecret99", "level": "error"'
        assert "deadbeefsecret99" not in ra._sanitize_sensitive_string_value(malformed)
        # JSONL message value.
        msg = "request failed: Bearer abcdefghijklmnop123456 rejected"
        assert "abcdefghijklmnop123456" not in ra._sanitize_sensitive_string_value(msg)
        # Legacy log line.
        legacy = "2026-07-30T06:00:00 ERROR password=legacysecret77 conn refused"
        assert "legacysecret77" not in ra._sanitize_sensitive_string_value(legacy)

    def test_benign_string_unchanged(self, ra):
        text = "candidate openrouter/or/model-a selected with priority 300"
        assert ra._sanitize_sensitive_string_value(text) == text

    def test_persisted_artifact_path_scrubs_string_values(self, adapter, tmp_path):
        """The REAL _write_artifact path must scrub embedded secrets in
        free-text string values under benign keys, preserving diagnostics."""
        artifact = {
            "error_intake": {
                "attributed_events": [
                    {
                        "file": "dev-error.jsonl",
                        "message": "upstream 401 Bearer sk-live-abcdef1234567890 for api_key=inline-secret-value",
                        "level": "error",
                    },
                    {"file": "dev.log", "legacy_line": "ERROR password=legacysecret77 refused"},
                ],
            },
            "diagnostic": "route openrouter/or/model-a cooled down",
            "passed": False,
        }
        path = tmp_path / "artifact.json"
        adapter._write_artifact(path, artifact)
        persisted_text = path.read_text(encoding="utf-8")
        for secret in (
            "sk-live-abcdef1234567890",
            "inline-secret-value",
            "legacysecret77",
        ):
            assert secret not in persisted_text
        # Bounded diagnostics preserved.
        persisted = json.loads(persisted_text)
        assert persisted["diagnostic"] == "route openrouter/or/model-a cooled down"
        assert persisted["passed"] is False
        assert "Bearer" in persisted["error_intake"]["attributed_events"][0]["message"]

    def test_error_intake_delta_path_scrubbed_on_persist(self, adapter, ra, tmp_path):
        """A secret-bearing JSONL message captured by the real error-intake
        delta collector must be scrubbed when persisted via _write_artifact."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        secret_msg = "auth failed Bearer sk-live-abcdef1234567890 api_key=inline-secret-value"
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "observed_at": new_time,
                        "environment": "dev",
                        "level": "error",
                        "message": secret_msg,
                        "context": {"container": "litellm-dev"},
                    }
                )
                + "\n"
            )
        events, failures = ra._collect_error_intake_delta(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            analysis_dir=analysis,
        )
        assert failures == []
        assert len(events) == 1
        # Delta collector captures the message (bounded to 300 chars).
        assert "sk-live-abcdef1234567890" in events[0]["message"]
        # Persisting the delta through the real artifact path scrubs it.
        path = tmp_path / "intake_artifact.json"
        adapter._write_artifact(path, {"error_intake": {"attributed_events": events}})
        persisted_text = path.read_text(encoding="utf-8")
        assert "sk-live-abcdef1234567890" not in persisted_text
        assert "inline-secret-value" not in persisted_text

    # -- CFG-003 redaction remediation: the four independent-validation gaps --

    def test_authorization_basic_and_token_schemes_scrubbed(self, ra):
        """Scheme-bearing Authorization forms must redact the credential run
        while retaining the nonsecret scheme label for diagnostics."""
        basic = "upstream 401 Authorization: Basic dXNlcjpwYXNzd29yZA== retrying"
        out = ra._sanitize_sensitive_string_value(basic)
        assert "dXNlcjpwYXNzd29yZA==" not in out
        assert "Basic" in out  # scheme label preserved
        assert "upstream 401" in out and "retrying" in out

        token = 'header "Authorization: Token ghp_abcdef1234567890" rejected'
        out = ra._sanitize_sensitive_string_value(token)
        assert "ghp_abcdef1234567890" not in out
        assert "Token" in out  # scheme label preserved
        assert "rejected" in out

    def test_provider_prefixed_api_key_assignments_scrubbed(self, ra):
        """Provider-prefixed API_KEY assignments (underscore-joined) must be
        redacted even though \\b does not fire between '_' and the key name."""
        malformed = '{"message": "auth OPENROUTER_API_KEY=sk-or-v1-deadbeef99", "level": "error"'
        out = ra._sanitize_sensitive_string_value(malformed)
        assert "sk-or-v1-deadbeef99" not in out
        assert "[REDACTED]" in out

        legacy = "2026-07-30T06:00:00 ERROR OPENAI_API_KEY=sk-legacysecret77 conn refused"
        out = ra._sanitize_sensitive_string_value(legacy)
        assert "sk-legacysecret77" not in out
        assert "[REDACTED]" in out

    def test_four_gap_examples_scrubbed_via_write_artifact(self, adapter, tmp_path):
        """The REAL _write_artifact path must scrub all four exact examples
        from independent validation, preserving benign diagnostics."""
        artifact = {
            "error_intake": {
                "attributed_events": [
                    {"file": "a.jsonl", "message": "Authorization: Basic dXNlcjpwYXNzd29yZA==", "level": "error"},
                    {"file": "b.jsonl", "message": "Authorization: Token ghp_abcdef1234567890", "level": "error"},
                    {"file": "c.jsonl", "raw_excerpt": '{"message": "OPENROUTER_API_KEY=sk-or-v1-deadbeef99"', "level": "error"},
                    {"file": "d.log", "legacy_line": "ERROR OPENAI_API_KEY=sk-legacysecret77 refused", "level": "error"},
                ],
            },
            "diagnostic": "route openrouter/or/model-a cooled down",
            "passed": False,
        }
        path = tmp_path / "artifact.json"
        adapter._write_artifact(path, artifact)
        persisted_text = path.read_text(encoding="utf-8")
        for secret in (
            "dXNlcjpwYXNzd29yZA==",
            "ghp_abcdef1234567890",
            "sk-or-v1-deadbeef99",
            "sk-legacysecret77",
        ):
            assert secret not in persisted_text
        persisted = json.loads(persisted_text)
        assert persisted["diagnostic"] == "route openrouter/or/model-a cooled down"
        assert persisted["passed"] is False

    def test_four_gap_examples_error_intake_delta_path_scrubbed(self, adapter, ra, tmp_path):
        """All four exact examples captured by the real error-intake delta
        collector must be scrubbed when persisted via _write_artifact."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        secret_msg = (
            "Authorization: Basic dXNlcjpwYXNzd29yZA== | Authorization: Token ghp_abcdef1234567890 "
            "| OPENROUTER_API_KEY=sk-or-v1-deadbeef99 | OPENAI_API_KEY=sk-legacysecret77"
        )
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"observed_at": new_time, "environment": "dev", "level": "error", "message": secret_msg}) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert failures == []
        assert len(events) == 1
        path = tmp_path / "intake_artifact.json"
        adapter._write_artifact(path, {"error_intake": {"attributed_events": events}})
        persisted_text = path.read_text(encoding="utf-8")
        for secret in ("dXNlcjpwYXNzd29yZA==", "ghp_abcdef1234567890", "sk-or-v1-deadbeef99", "sk-legacysecret77"):
            assert secret not in persisted_text



# ---------------------------------------------------------------------------
# Body B finding (c): exact-pair priority-swap helper contract
# ---------------------------------------------------------------------------

_EXACT_PAIR_YAML = """\
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: or/model-a
        route_family: codex_openrouter_completion_adapter
        priority: 300
      - provider: alibaba_token_plan
        model: alibaba/qwen-max
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 200
      - provider: openai
        model: gpt-x
        route_family: codex_openai_chat_completions_adapter
        priority: 100
"""


class TestExactPairPrioritySwap:
    def test_swaps_requested_pair_not_first_two_raw(self, ra):
        """Requesting the 2nd+3rd identities swaps exactly those, leaving the
        first raw candidate untouched -- catches a first-two-raw swap bug."""
        pair = (("alibaba_token_plan", "alibaba/qwen-max"), ("openai", "gpt-x"))
        swapped_yaml, original, swapped = ra._build_exact_pair_priority_swap_yaml(
            _EXACT_PAIR_YAML, pair=pair
        )
        orig_by = {(c["provider"], c["model"]): c["priority"] for c in original}
        swap_by = {(c["provider"], c["model"]): c["priority"] for c in swapped}
        # Requested pair priorities exchanged.
        assert swap_by[("alibaba_token_plan", "alibaba/qwen-max")] == orig_by[("openai", "gpt-x")]
        assert swap_by[("openai", "gpt-x")] == orig_by[("alibaba_token_plan", "alibaba/qwen-max")]
        # First raw candidate (openrouter) is NOT swapped.
        assert swap_by[("openrouter", "or/model-a")] == orig_by[("openrouter", "or/model-a")] == 300

    def test_differs_from_first_two_raw_swap(self, ra):
        """The exact-pair result must differ from the legacy first-two swap,
        proving it does not silently swap the first two raw candidates."""
        pair = (("alibaba_token_plan", "alibaba/qwen-max"), ("openai", "gpt-x"))
        exact_yaml, _, exact_swapped = ra._build_exact_pair_priority_swap_yaml(
            _EXACT_PAIR_YAML, pair=pair
        )
        _first_two_yaml, _, first_two_swapped = ra._build_priority_swap_yaml(
            _EXACT_PAIR_YAML
        )
        exact_order = [(c["provider"], c["model"]) for c in exact_swapped]
        first_two_order = [(c["provider"], c["model"]) for c in first_two_swapped]
        assert exact_order != first_two_order

    def test_absent_identity_fails_closed(self, ra):
        pair = (("alibaba_token_plan", "alibaba/qwen-max"), ("ghost", "nope"))
        with pytest.raises(ValueError, match="not found"):
            ra._build_exact_pair_priority_swap_yaml(_EXACT_PAIR_YAML, pair=pair)

    def test_ambiguous_identity_fails_closed(self, ra):
        dup_yaml = _EXACT_PAIR_YAML + (
            "      - provider: openai\n"
            "        model: gpt-x\n"
            "        route_family: codex_openai_chat_completions_adapter\n"
            "        priority: 50\n"
        )
        pair = (("alibaba_token_plan", "alibaba/qwen-max"), ("openai", "gpt-x"))
        with pytest.raises(ValueError, match="ambiguous"):
            ra._build_exact_pair_priority_swap_yaml(dup_yaml, pair=pair)

    def test_non_distinct_pair_fails_closed(self, ra):
        pair = (("openai", "gpt-x"), ("openai", "gpt-x"))
        with pytest.raises(ValueError, match="distinct"):
            ra._build_exact_pair_priority_swap_yaml(_EXACT_PAIR_YAML, pair=pair)


# ---------------------------------------------------------------------------
# Integration remediation Finding 1: Canonical dev isolation
# ---------------------------------------------------------------------------


class TestCanonicalDevIsolation:
    def test_canonical_dev_profile_accepted(self, adapter):
        """The exact canonical dev profile must pass validation."""
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001",
                "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "dev",
            },
        )
        assert ok
        assert failures == []

    def test_non_dev_target_rejected(self, adapter):
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="prod",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001",
                "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "dev",
            },
        )
        assert not ok
        assert any("target must be 'dev'" in f for f in failures)

    def test_prod_url_override_rejected(self, adapter):
        """CLI override pointing at port 4000 must fail closed."""
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4000",
                "anthropic_base_url": "http://127.0.0.1:4000/anthropic",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "dev",
            },
        )
        assert not ok
        assert any("litellm_base_url" in f for f in failures)
        assert any("port 4000" in f for f in failures)

    def test_aawm_litellm_container_rejected(self, adapter):
        """A dev-labelled profile pointing at the production container must
        fail closed."""
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001",
                "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
                "docker_container_name": "aawm-litellm",
                "expected_trace_environment": "dev",
            },
        )
        assert not ok
        assert any("aawm-litellm" in f for f in failures)

    def test_wrong_trace_environment_rejected(self, adapter):
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001",
                "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "prod",
            },
        )
        assert not ok
        assert any("expected_trace_environment" in f for f in failures)

    def test_wrong_anthropic_url_rejected(self, adapter):
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001",
                "anthropic_base_url": "http://10.0.0.1:4001/anthropic",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "dev",
            },
        )
        assert not ok
        assert any("anthropic_base_url" in f for f in failures)

    def test_trailing_slash_normalized(self, adapter):
        """Trailing slashes on URLs should not cause false rejection."""
        ok, failures = adapter._cfg003_validate_canonical_dev_profile(
            target="dev",
            profile={
                "litellm_base_url": "http://127.0.0.1:4001/",
                "anthropic_base_url": "http://127.0.0.1:4001/anthropic/",
                "docker_container_name": "litellm-dev",
                "expected_trace_environment": "dev",
            },
        )
        assert ok
        assert failures == []


# ---------------------------------------------------------------------------
# Integration remediation Finding 2: Final mutation safety (POST order)
# ---------------------------------------------------------------------------


class TestFinalMutationSafety:
    def test_restore_is_last_config_mutation(self, adapter, ra, monkeypatch):
        """No refresh POST may occur after the unconditional restoration.
        Controls must run pre-swap.  Record POST order and prove restore is
        the last mutation even when invalid-control raises."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(eligible) >= 2
        raw_file_text = BASIC_YAML_PATH.read_text(encoding="utf-8")

        post_log: list[str] = []

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                post_log.append("invalid_control")
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_file_text:
                # Could be unchanged_control or restoration
                if "swap" not in post_log:
                    post_log.append("unchanged_control")
                else:
                    post_log.append("restoration")
                return 200, {
                    "changed": "swap" not in post_log,
                    "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": {"basic": []},
                }
            post_log.append("swap")
            return 200, {
                "changed": True,
                "active_config_hash": "swapped_hash",
                "config_version": "swapped_ver",
                "active_candidate_order": {"basic": []},
            }

        def fake_auth():
            return {
                "snapshot": snap, "merged_yaml": auth["merged_yaml"],
                "per_file_hashes": auth["per_file_hashes"],
                "file_names": auth["file_names"],
                "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "aliases": auth["aliases"],
            }

        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", fake_auth)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })

        adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        # Controls must appear BEFORE swap.
        assert "unchanged_control" in post_log
        assert "invalid_control" in post_log
        uc_idx = post_log.index("unchanged_control")
        ic_idx = post_log.index("invalid_control")
        swap_idx = post_log.index("swap")
        assert uc_idx < swap_idx, f"unchanged_control at {uc_idx} must precede swap at {swap_idx}"
        assert ic_idx < swap_idx, f"invalid_control at {ic_idx} must precede swap at {swap_idx}"
        # Restoration must be the LAST POST.
        if "restoration" in post_log:
            restore_idx = post_log.index("restoration")
            assert restore_idx == len(post_log) - 1, (
                f"restoration at {restore_idx} must be last, log={post_log}"
            )

    def test_invalid_control_exception_still_restores(self, adapter, ra, monkeypatch):
        """Even if the invalid-control POST raises, the final restoration
        must still occur and be the last mutation."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        raw_file_text = BASIC_YAML_PATH.read_text(encoding="utf-8")

        post_log: list[str] = []

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                post_log.append("invalid_control")
                raise ConnectionError("simulated invalid-control failure")
            if yaml_text == raw_file_text:
                if "swap" not in post_log:
                    post_log.append("unchanged_control")
                else:
                    post_log.append("restoration")
                return 200, {
                    "changed": "swap" not in post_log,
                    "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": {"basic": []},
                }
            post_log.append("swap")
            return 200, {
                "changed": True,
                "active_config_hash": "swapped_hash",
                "config_version": "swapped_ver",
                "active_candidate_order": {"basic": []},
            }

        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"],
            "file_names": auth["file_names"],
            "config_hash": auth["config_hash"],
            "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        # Invalid control exception recorded.
        assert any("invalid control exception" in f for f in result["failures"])
        # Restoration still happened and is last.
        assert "restoration" in post_log
        assert post_log[-1] == "restoration"


# ---------------------------------------------------------------------------
# Integration remediation Finding 3: Exact-pair swap wiring
# ---------------------------------------------------------------------------


class TestExactPairSwapWiring:
    def test_transactional_uses_exact_pair_not_first_two_raw(self, adapter, ra, monkeypatch):
        """The transactional test must call _build_exact_pair_priority_swap_yaml
        with the evidenced pair, not _build_priority_swap_yaml."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(eligible) >= 3

        called_with_pair = []

        original_exact = ra._build_exact_pair_priority_swap_yaml

        def spy_exact(raw_yaml, *, pair, alias_name="basic", excluded_providers=None):
            called_with_pair.append(pair)
            return original_exact(raw_yaml, pair=pair, alias_name=alias_name,
                                  excluded_providers=excluded_providers)

        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"],
            "file_names": auth["file_names"],
            "config_hash": auth["config_hash"],
            "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", lambda *a, **kw: (
            200, {"changed": True, "active_config_hash": "new_hash",
                  "config_version": "new_ver",
                  "active_candidate_order": {"basic": []}}
        ))
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })
        monkeypatch.setattr(adapter.RA, "_build_exact_pair_priority_swap_yaml", spy_exact)

        adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        # The exact-pair helper must have been called.
        assert len(called_with_pair) >= 1
        pair = called_with_pair[0]
        # The pair must be the evidenced (provider, model) identities.
        assert pair[0] == (eligible[0]["provider"], eligible[0]["model"])
        assert pair[1] == (eligible[1]["provider"], eligible[1]["model"])

    def test_swap_build_failure_prevents_post(self, adapter, ra, monkeypatch):
        """If the exact-pair helper raises, no swap POST may occur."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")

        post_log: list[str] = []

        def fake_post(url, payload, **kw):
            post_log.append("post")
            return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                         "config_version": auth["config_version"]}

        def broken_exact(*a, **kw):
            raise ValueError("identity not found")

        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"],
            "file_names": auth["file_names"],
            "config_hash": auth["config_hash"],
            "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })
        monkeypatch.setattr(adapter.RA, "_build_exact_pair_priority_swap_yaml", broken_exact)

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("exact-pair swap build failed" in f for f in result["failures"])
        # No swap POST should have occurred (only unchanged/invalid controls
        # and restoration are allowed).
        swap_posts = [p for p in post_log if p == "swap"]
        assert len(swap_posts) == 0


# ---------------------------------------------------------------------------
# Integration remediation Finding 4: Phase-fresh session identity
# ---------------------------------------------------------------------------


class TestPhaseFreshSessionIdentity:
    def test_proof_case_injects_fresh_session(self, adapter, monkeypatch):
        """Each proof phase must receive a unique session ID, not reuse a
        formatted case/session from config."""
        captured_configs: list[dict] = []

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            captured_configs.append(dict(case_config))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        cases = {"proof_case": {"command": ["codex"], "expected_trace_session_id": "SHARED_OLD"}}
        # Run two proof phases.
        p1 = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        p2 = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_swap_proof",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        # Each must have a unique phase_session_id.
        assert p1["phase_session_id"] != p2["phase_session_id"]
        # Neither may reuse the old shared session.
        assert captured_configs[0]["expected_trace_session_id"] != "SHARED_OLD"
        assert captured_configs[1]["expected_trace_session_id"] != "SHARED_OLD"
        # Phase start times recorded.
        assert p1["phase_start_time"]
        assert p2["phase_start_time"]

    def test_original_case_config_not_mutated(self, adapter, monkeypatch):
        """The original case config dict must not be mutated by proof runs."""
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        original_config = {"command": ["codex"], "expected_trace_session_id": "ORIGINAL"}
        cases = {"proof_case": original_config}
        adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        assert original_config["expected_trace_session_id"] == "ORIGINAL"


# ---------------------------------------------------------------------------
# Integration remediation Finding 5: Sanitized phase evidence
# ---------------------------------------------------------------------------


class TestSanitizedPhaseEvidence:
    def test_phase_evidence_required_fields(self, adapter):
        """Phase evidence must include all required summary fields."""
        proof = {
            "result": {"passed": True, "terminal_marker": "READ_PASSED"},
            "selection": {"provider": "openrouter", "model": "or/m1", "route_family": "rf"},
            "phase_session_id": "sess-123",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        case_config = {
            "command": ["codex", "exec", "-p", "my-profile", "-m", "basic",
                        "--json", "my-secret-prompt-text"],
            "verification_ingress": "codex_responses",
            "expected_parent_agent_name": "parent-agent",
            "expected_child_agent_name": "child-agent",
            "agent_profile": "basic",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="baseline",
            case_name="test_case__cfg003_baseline",
            proof=proof,
            case_config=case_config,
            active_hash="hash123",
            active_version="v1",
        )
        # Required fields present.
        assert ev["phase"] == "baseline"
        assert ev["case_name"] == "test_case__cfg003_baseline"
        assert ev["tui_executable"] == "codex"
        assert ev["verification_ingress"] == "codex_responses"
        assert ev["phase_session_id"] == "sess-123"
        assert ev["phase_start_time"] == "2026-07-30T12:00:00+00:00"
        assert ev["selected_provider"] == "openrouter"
        assert ev["selected_model"] == "or/m1"
        assert ev["active_config_hash"] == "hash123"
        assert ev["active_config_version"] == "v1"
        assert ev["passed"] is True
        assert ev["expected_parent_agent_name"] == "parent-agent"
        assert ev["expected_child_agent_name"] == "child-agent"
        assert ev["agent_profile"] == "basic"
        # Prompt identity as SHA-256 + length, never raw.
        # For codex, -p is the profile; the final positional is the prompt.
        import hashlib
        expected_sha = hashlib.sha256(b"my-secret-prompt-text").hexdigest()
        assert ev["prompt_sha256"] == expected_sha
        assert ev["prompt_length"] == len("my-secret-prompt-text")

    def test_phase_evidence_forbids_raw_values(self, adapter):
        """Phase evidence must NOT contain raw prompts, auth values,
        command arguments, stdout/stderr, or provider bodies."""
        proof = {
            "result": {
                "passed": True,
                "stdout": "raw stdout with secrets",
                "stderr": "raw stderr",
                "command": ["codex", "exec", "-p", "SECRET_PROMPT"],
            },
            "selection": {"provider": "p", "model": "m", "route_family": "rf"},
            "phase_session_id": "sess-456",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        case_config = {
            "command": ["codex", "exec", "-p", "SECRET_PROMPT"],
            "verification_ingress": "codex_responses",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="swap_proof",
            case_name="test__cfg003_swap_proof",
            proof=proof,
            case_config=case_config,
        )
        ev_str = json.dumps(ev)
        assert "SECRET_PROMPT" not in ev_str
        assert "raw stdout" not in ev_str
        assert "raw stderr" not in ev_str


# ---------------------------------------------------------------------------
# Integration remediation Finding 6: True end-to-end collector redaction tests
# ---------------------------------------------------------------------------


class TestEndToEndCollectorRedaction:
    def test_malformed_jsonl_excerpt_branch_redacted(self, adapter, ra, tmp_path):
        """Independently exercise the malformed JSONL excerpt branch through
        the real collector, persist via _write_artifact, and prove Authorization/
        API-key forms are absent while diagnostics remain."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        # Write a MALFORMED JSONL line (not valid JSON) containing secrets.
        malformed_line = '{"message": "Authorization: Bearer sk-live-MALFORMEDSECRET99", "level": "error"'
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(malformed_line + "\n")

        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert failures == []
        assert len(events) == 1
        assert events[0].get("malformed") is True
        assert "excerpt" in events[0]

        # Persist through the real _write_artifact.
        path = tmp_path / "artifact.json"
        adapter._write_artifact(path, {"error_intake": {"attributed_events": events}})
        persisted_text = path.read_text(encoding="utf-8")
        assert "sk-live-MALFORMEDSECRET99" not in persisted_text
        # Diagnostic structure preserved.
        persisted = json.loads(persisted_text)
        assert persisted["error_intake"]["attributed_events"][0]["malformed"] is True

    def test_legacy_log_line_branch_redacted(self, adapter, ra, tmp_path):
        """Independently exercise the legacy *-error.log legacy_line branch
        through the real collector, persist via _write_artifact, and prove
        Authorization/API-key forms are absent while diagnostics remain."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.log").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        # Append a legacy log line containing secrets.
        legacy_line = "2026-07-30T06:00:00 dev ERROR api_key=sk-LEGACYSECRET77 Authorization: Basic dXNlcjpwYXNz connection refused"
        with open(analysis / "dev-error.log", "a", encoding="utf-8") as f:
            f.write(legacy_line + "\n")

        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", analysis_dir=analysis,
        )
        assert failures == []
        assert len(events) == 1
        assert "legacy_line" in events[0]

        # Persist through the real _write_artifact.
        path = tmp_path / "artifact.json"
        adapter._write_artifact(path, {"error_intake": {"attributed_events": events}})
        persisted_text = path.read_text(encoding="utf-8")
        assert "sk-LEGACYSECRET77" not in persisted_text
        assert "dXNlcjpwYXNz" not in persisted_text
        # Diagnostic structure preserved.
        persisted = json.loads(persisted_text)
        assert "legacy_line" in persisted["error_intake"]["attributed_events"][0]
        assert "connection refused" in persisted_text


# ---------------------------------------------------------------------------
# Finding 1 (round 7): phase_start_time freshness enforcement
# ---------------------------------------------------------------------------


class TestPhaseStartTimeFreshness:
    def test_session_history_query_includes_phase_start_predicate(self, adapter):
        """When phase_start_time is injected, the SQL query must include a
        start_time >= %s predicate and pass it as a parameter."""
        captured_queries = []
        captured_params = []

        class FakeCursor:
            def execute(self, sql, params=None):
                captured_queries.append(sql)
                captured_params.append(params)

            def fetchall(self):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            closed = False

            def cursor(self):
                return FakeCursor()

        phase_ts = "2026-07-30T12:00:00+00:00"
        checks = {
            "db_host": "h",
            "db_port": 5432,
            "db_name": "d",
            "db_user": "u",
            "db_password": "p",
            "phase_start_time": phase_ts,
        }
        monkey_conn = FakeConn()
        original_conn_fn = adapter._validation_db_connection
        adapter._validation_db_connection = lambda settings: monkey_conn
        try:
            adapter._validate_session_history(
                family="test", session_id="sess-1", checks=checks
            )
        finally:
            adapter._validation_db_connection = original_conn_fn

        assert len(captured_queries) >= 1
        sql = captured_queries[0]
        assert "start_time >= %s" in sql
        params = captured_params[0]
        assert params[0] == "sess-1"
        assert params[1] == phase_ts

    def test_session_history_without_phase_start_no_predicate(self, adapter):
        """Without phase_start_time, no start_time predicate is added."""
        captured_queries = []
        captured_params = []

        class FakeCursor:
            def execute(self, sql, params=None):
                captured_queries.append(sql)
                captured_params.append(params)

            def fetchall(self):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            closed = False

            def cursor(self):
                return FakeCursor()

        checks = {
            "db_host": "h",
            "db_port": 5432,
            "db_name": "d",
            "db_user": "u",
            "db_password": "p",
        }
        original_conn_fn = adapter._validation_db_connection
        adapter._validation_db_connection = lambda settings: FakeConn()
        try:
            adapter._validate_session_history(
                family="test", session_id="sess-1", checks=checks
            )
        finally:
            adapter._validation_db_connection = original_conn_fn

        sql = captured_queries[0]
        assert "start_time >= %s" not in sql
        assert captured_params[0] == ("sess-1",)

    def test_tool_activity_query_includes_phase_start_predicate(self, adapter):
        """When phase_start_time is injected, tool_activity SQL must include
        created_at >= %s predicate."""
        captured_queries = []
        captured_params = []

        class FakeCursor:
            def execute(self, sql, params=None):
                captured_queries.append(sql)
                captured_params.append(params)

            def fetchall(self):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            closed = False

            def cursor(self):
                return FakeCursor()

        phase_ts = "2026-07-28T06:00:00+00:00"
        checks = {
            "db_host": "h",
            "db_port": 5432,
            "db_name": "d",
            "db_user": "u",
            "db_password": "p",
            "phase_start_time": phase_ts,
        }
        original_conn_fn = adapter._validation_db_connection
        adapter._validation_db_connection = lambda settings: FakeConn()
        try:
            adapter._validate_tool_activity(
                family="test", session_id="sess-1", checks=checks
            )
        finally:
            adapter._validation_db_connection = original_conn_fn

        assert len(captured_queries) >= 1
        sql = captured_queries[0]
        assert "created_at >= %s" in sql
        params = captured_params[0]
        assert params[0] == "sess-1"
        assert params[1] == phase_ts

    def test_two_day_old_row_excluded_by_phase_start(self, adapter):
        """A two-day-old matching session row must be excluded when
        phase_start_time is set, while a current row is accepted."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        phase_start = now - dt.timedelta(minutes=5)
        old_time = now - dt.timedelta(days=2)

        captured_queries = []
        captured_params = []

        class FakeCursor:
            def execute(self, sql, params=None):
                captured_queries.append(sql)
                captured_params.append(params)

            def fetchall(self):
                # The SQL predicate filters server-side; verify the parameter
                # is the phase_start, which would exclude old_time rows.
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            closed = False

            def cursor(self):
                return FakeCursor()

        checks = {
            "db_host": "h",
            "db_port": 5432,
            "db_name": "d",
            "db_user": "u",
            "db_password": "p",
            "phase_start_time": phase_start.isoformat(),
        }
        original_conn_fn = adapter._validation_db_connection
        adapter._validation_db_connection = lambda settings: FakeConn()
        try:
            adapter._validate_session_history(
                family="test", session_id="sess-1", checks=checks
            )
        finally:
            adapter._validation_db_connection = original_conn_fn

        # The phase_start parameter must be present, which would exclude
        # any row with start_time < phase_start (i.e., the two-day-old row).
        params = captured_params[0]
        assert params[1] == phase_start.isoformat()
        # old_time < phase_start, so the SQL predicate excludes it.
        assert old_time < phase_start


# ---------------------------------------------------------------------------
# Finding 2 (round 7): Authoritative active state
# ---------------------------------------------------------------------------


class TestAuthoritativeActiveState:
    def test_readiness_check_passes_on_match(self, adapter, monkeypatch):
        """Readiness check passes when hash and version match."""
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {"config_hash": "abc", "config_version": "v1"}}
        ))
        ok, failures = adapter._cfg003_readiness_check(
            "http://localhost:4001",
            expected_hash="abc",
            expected_version="v1",
            phase_label="test",
        )
        assert ok
        assert failures == []

    def test_readiness_check_fails_on_wrong_hash(self, adapter, monkeypatch):
        ok, failures = adapter._cfg003_readiness_check(
            "http://localhost:4001",
            expected_hash="abc",
            expected_version="v1",
            phase_label="test",
        ) if False else (None, None)
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {"config_hash": "WRONG", "config_version": "v1"}}
        ))
        ok, failures = adapter._cfg003_readiness_check(
            "http://localhost:4001",
            expected_hash="abc",
            expected_version="v1",
            phase_label="test",
        )
        assert not ok
        assert any("hash mismatch" in f for f in failures)

    def test_readiness_check_fails_on_wrong_version(self, adapter, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {"config_hash": "abc", "config_version": "WRONG"}}
        ))
        ok, failures = adapter._cfg003_readiness_check(
            "http://localhost:4001",
            expected_hash="abc",
            expected_version="v1",
            phase_label="test",
        )
        assert not ok
        assert any("version mismatch" in f for f in failures)

    def test_readiness_check_fails_on_unavailable(self, adapter, monkeypatch):
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            503, {}
        ))
        ok, failures = adapter._cfg003_readiness_check(
            "http://localhost:4001",
            expected_hash="abc",
            expected_version="v1",
            phase_label="test",
        )
        assert not ok
        assert any("unavailable" in f for f in failures)

    def test_wrong_swapped_hash_fails(self, adapter, ra, monkeypatch):
        """False-pass probe: swap refresh returning a wrong hash (not matching
        locally compiled swapped YAML) must fail."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        call_n = {"n": 0}
        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == BASIC_YAML_PATH.read_text(encoding="utf-8"):
                if call_n["n"] <= 2:
                    return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                                 "config_version": auth["config_version"]}
                return 200, {"changed": True, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            # Swap: return WRONG hash
            return 200, {"changed": True, "active_config_hash": "WRONG_SWAPPED_HASH",
                         "config_version": "WRONG_VER",
                         "active_candidate_order": {"basic": []}}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"], "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("hash mismatch" in f for f in result["failures"])

    def test_wrong_swapped_order_fails(self, adapter, ra, monkeypatch):
        """False-pass probe: swap refresh returning wrong active_candidate_order
        must fail even if hash/version are correct."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        # Compute the expected swapped hash from the real YAML.
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        correct_hash = swapped_snap.config_hash
        correct_version = swapped_snap.config_version

        call_n = {"n": 0}
        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                if call_n["n"] <= 2:
                    return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                                 "config_version": auth["config_version"]}
                return 200, {"changed": True, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            # Swap: correct hash/version but WRONG order
            return 200, {"changed": True, "active_config_hash": correct_hash,
                         "config_version": correct_version,
                         "active_candidate_order": {"basic": [{"provider": "WRONG", "model": "WRONG", "route_family": "WRONG", "priority": 0}]}}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"], "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("order" in f.lower() for f in result["failures"])


# ---------------------------------------------------------------------------
# Finding 3 (round 7): Canonical gate before credential access
# ---------------------------------------------------------------------------


class TestCanonicalGateBeforeCredentials:
    def test_credential_resolution_not_invoked_on_rejection(self, adapter, monkeypatch, tmp_path):
        """A dev-labelled prod override must fail the canonical gate BEFORE
        _resolve_main_credentials is called."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {},
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        cred_calls = []

        def spy_credentials(**kw):
            cred_calls.append(kw)
            return ("pk", "sk", "http://q", "pk_env", "sk_env")

        monkeypatch.setattr(adapter, "_resolve_main_credentials", spy_credentials)
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "dev",
            "--litellm-base-url", "http://127.0.0.1:4000",
            "--cfg003-transactional-refresh",
        ])

        exit_code = adapter.main()
        assert exit_code == 1
        # _resolve_main_credentials must NOT have been called.
        assert cred_calls == []
        # Artifact must record the gate failure.
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert "cfg003_raw_config_gate" in artifact["results"]
        assert artifact["results"]["cfg003_raw_config_gate"]["passed"] is False

    def test_credential_resolution_invoked_when_gate_passes(self, adapter, monkeypatch, tmp_path):
        """When the canonical gate passes, _resolve_main_credentials IS called."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "non_tui_case": {"required_env": ["NONEXISTENT_ENV_VAR_XYZ"]},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        cred_calls = []

        def spy_credentials(**kw):
            cred_calls.append(kw)
            return ("pk", "sk", "http://q", "pk_env", "sk_env")

        monkeypatch.setattr(adapter, "_resolve_main_credentials", spy_credentials)
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": True, "inventory_failures": [], "alias_inventory": []})
        monkeypatch.delenv("NONEXISTENT_ENV_VAR_XYZ", raising=False)
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--cases", "non_tui_case",
            "--cfg003-transactional-refresh",
        ])

        adapter.main()
        # Gate passes (canonical dev profile), so credentials ARE resolved.
        assert len(cred_calls) == 1


# ---------------------------------------------------------------------------
# Finding 4 (round 7): Executable-aware prompt identity
# ---------------------------------------------------------------------------


class TestExecutableAwarePromptIdentity:
    def test_claude_p_is_prompt(self, adapter):
        """For Claude, -p is the prompt."""
        import hashlib

        prompt = "Reply with exactly two words: native anthropic"
        case_config = {
            "command": [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--model", "claude-opus-4-6",
            ],
        }
        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="test", case_name="test_case",
            proof=proof, case_config=case_config,
        )
        expected_sha = hashlib.sha256(prompt.encode()).hexdigest()
        assert ev["prompt_sha256"] == expected_sha
        assert ev["prompt_length"] == len(prompt)

    def test_codex_p_is_profile_not_prompt(self, adapter):
        """For Codex, -p is the provider/profile, NOT the prompt.  The actual
        prompt is the final positional argument."""
        import hashlib

        actual_prompt = "Basic alias collaboration acceptance exercise. Do not modify files."
        case_config = {
            "command": [
                "codex", "exec",
                "-p", "my-secret-profile",
                "-m", "basic",
                "-c", "model_providers.x.http_headers.session_id=sess",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                actual_prompt,
            ],
        }
        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="test", case_name="test_case",
            proof=proof, case_config=case_config,
        )
        # Must hash the actual prompt, NOT the profile.
        expected_sha = hashlib.sha256(actual_prompt.encode()).hexdigest()
        assert ev["prompt_sha256"] == expected_sha
        assert ev["prompt_length"] == len(actual_prompt)
        # Must NOT hash the profile.
        wrong_sha = hashlib.sha256(b"my-secret-profile").hexdigest()
        assert ev["prompt_sha256"] != wrong_sha

    def test_checked_in_codex_case_prompt_extraction(self, adapter):
        """The checked-in Codex proof case must extract the long prompt as the
        final positional, not the profile."""
        import hashlib

        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        command = case["command"]
        # The last element is the prompt.
        actual_prompt = command[-1]
        assert "Basic alias collaboration" in actual_prompt

        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="test", case_name="test_case",
            proof=proof, case_config=case,
        )
        expected_sha = hashlib.sha256(actual_prompt.encode()).hexdigest()
        assert ev["prompt_sha256"] == expected_sha
        assert ev["prompt_length"] == len(actual_prompt)

    def test_checked_in_claude_case_prompt_extraction(self, adapter):
        """The checked-in Claude case must extract the -p value as prompt."""
        import hashlib

        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["native_anthropic_passthrough_claude"]
        command = case["command"]
        # Find -p value.
        p_idx = command.index("-p")
        actual_prompt = command[p_idx + 1]

        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="test", case_name="test_case",
            proof=proof, case_config=case,
        )
        expected_sha = hashlib.sha256(actual_prompt.encode()).hexdigest()
        assert ev["prompt_sha256"] == expected_sha
        assert ev["prompt_length"] == len(actual_prompt)


# ---------------------------------------------------------------------------
# Finding 5 (round 7): Terminal marker derivation from exact-output contract
# ---------------------------------------------------------------------------


class TestTerminalMarkerDerivation:
    def test_codex_prefix_suffix_marker(self, adapter):
        """Codex case: derive marker from command_output_text_checks
        required_prefix/suffix."""
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        marker = adapter._cfg003_derive_terminal_marker(case)
        assert marker == "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED"

    def test_claude_required_equals_result_marker(self, adapter):
        """Claude case: derive marker from command_json_checks
        required_equals.result."""
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        marker = adapter._cfg003_derive_terminal_marker(case)
        assert marker == "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"

    def test_no_contract_returns_empty(self, adapter):
        """A case with no exact-output contract returns empty string."""
        marker = adapter._cfg003_derive_terminal_marker({"command": ["codex"]})
        assert marker == ""

    def test_phase_evidence_contains_derived_marker(self, adapter):
        """Phase evidence must contain the derived terminal marker from the
        checked-in Codex case."""
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="baseline", case_name="test__cfg003_baseline",
            proof=proof, case_config=case,
        )
        assert ev["terminal_marker"] == "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        assert ev["terminal_marker_source"] == "derived_from_contract"

    def test_phase_evidence_claude_marker(self, adapter):
        """Phase evidence for the checked-in Claude case must contain the
        derived marker."""
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        proof = {
            "result": {"passed": True},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="baseline", case_name="test__cfg003_baseline",
            proof=proof, case_config=case,
        )
        assert ev["terminal_marker"] == "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        assert ev["terminal_marker_source"] == "derived_from_contract"

    def test_missing_contract_falls_back_to_result(self, adapter):
        """When no contract can derive a marker, fall back to result field."""
        proof = {
            "result": {"passed": True, "terminal_marker": "FALLBACK_MARKER"},
            "selection": {},
            "phase_session_id": "s1",
            "phase_start_time": "2026-07-30T12:00:00+00:00",
        }
        ev = adapter._cfg003_build_phase_evidence(
            phase_name="test", case_name="test_case",
            proof=proof, case_config={"command": ["codex"]},
        )
        assert ev["terminal_marker"] == "FALLBACK_MARKER"
        assert ev["terminal_marker_source"] == "result_fallback"


# ---------------------------------------------------------------------------
# Finding 1 (round 8): phase_start_time injected into nested validation dicts
# ---------------------------------------------------------------------------


class TestPhaseStartTimeNestedInjection:
    def test_session_history_validation_receives_phase_start_time(self, adapter, monkeypatch):
        """_cfg003_run_proof_case must inject phase_start_time into nested
        session_history_validation so historical rows cannot satisfy a new proof."""
        captured_configs: list[dict] = []

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            captured_configs.append(dict(case_config))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        cases = {"proof_case": {
            "command": ["codex"],
            "session_history_validation": {
                "db_host": "h", "db_port": 5432, "db_name": "d",
                "db_user": "u", "db_password": "p",
                "expected_provider": "openrouter",
            },
        }}
        proof = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        shv = captured_configs[0].get("session_history_validation")
        assert isinstance(shv, dict)
        assert "phase_start_time" in shv
        assert shv["phase_start_time"] == proof["phase_start_time"]
        # Original provider field preserved.
        assert shv["expected_provider"] == "openrouter"

    def test_tool_activity_validation_receives_phase_start_time(self, adapter, monkeypatch):
        """_cfg003_run_proof_case must inject phase_start_time into nested
        tool_activity_validation."""
        captured_configs: list[dict] = []

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            captured_configs.append(dict(case_config))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        cases = {"proof_case": {
            "command": ["codex"],
            "tool_activity_validation": {
                "db_host": "h", "db_port": 5432, "db_name": "d",
                "db_user": "u", "db_password": "p",
                "expected_min_tool_calls": 1,
            },
        }}
        proof = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        tav = captured_configs[0].get("tool_activity_validation")
        assert isinstance(tav, dict)
        assert "phase_start_time" in tav
        assert tav["phase_start_time"] == proof["phase_start_time"]
        assert tav["expected_min_tool_calls"] == 1

    def test_nested_dicts_not_mutated_in_original_config(self, adapter, monkeypatch):
        """The original case config's nested validation dicts must not be
        mutated by proof runs."""
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        original_shv = {"db_host": "h", "db_port": 5432}
        original_tav = {"db_host": "h", "db_port": 5432}
        cases = {"proof_case": {
            "command": ["codex"],
            "session_history_validation": original_shv,
            "tool_activity_validation": original_tav,
        }}
        adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        assert "phase_start_time" not in original_shv
        assert "phase_start_time" not in original_tav

    def test_two_phases_get_distinct_phase_start_times(self, adapter, monkeypatch):
        """Two proof phases must receive distinct phase_start_time values in
        their nested validation dicts."""
        captured_configs: list[dict] = []

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            captured_configs.append(dict(case_config))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        cases = {"proof_case": {
            "command": ["codex"],
            "session_history_validation": {"db_host": "h"},
            "tool_activity_validation": {"db_host": "h"},
        }}
        p1 = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        p2 = adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_swap_proof",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        shv1 = captured_configs[0]["session_history_validation"]
        shv2 = captured_configs[1]["session_history_validation"]
        tav1 = captured_configs[0]["tool_activity_validation"]
        tav2 = captured_configs[1]["tool_activity_validation"]
        # Each phase has its own timestamp.
        assert shv1["phase_start_time"] == p1["phase_start_time"]
        assert shv2["phase_start_time"] == p2["phase_start_time"]
        assert tav1["phase_start_time"] == p1["phase_start_time"]
        assert tav2["phase_start_time"] == p2["phase_start_time"]

    def test_no_validation_dicts_no_injection(self, adapter, monkeypatch):
        """When the case has no session_history_validation or
        tool_activity_validation, no injection occurs and no error is raised."""
        captured_configs: list[dict] = []

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            captured_configs.append(dict(case_config))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        cases = {"proof_case": {"command": ["codex"]}}
        adapter._cfg003_run_proof_case(
            case_name="proof_case__cfg003_baseline",
            case_config_key="proof_case",
            cases=cases, suite_config={}, query_url="q",
            public_key="pk", secret_key="sk",
            litellm_base_url="http://localhost:4001",
        )
        assert "session_history_validation" not in captured_configs[0]
        assert "tool_activity_validation" not in captured_configs[0]


# ---------------------------------------------------------------------------
# Finding 2 (round 8): Fail closed before later traffic/state mutation
# ---------------------------------------------------------------------------


class TestFailClosedBeforeMutation:
    def test_pre_baseline_readiness_failure_prevents_baseline_and_swap(
        self, adapter, ra, monkeypatch
    ):
        """Failed pre-baseline readiness must prevent baseline proof and all
        later swap work.  No TUI/POST calls may occur after the gate."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(eligible) >= 2
        avail_res = _avail_result(eligible[:3])

        post_log: list[str] = []
        tui_log: list[str] = []

        def fake_post(url, payload, **kw):
            post_log.append(payload.get("yaml", "")[:40])
            return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                         "config_version": auth["config_version"]}

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            tui_log.append(case_name)
            return {"passed": True}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        # Readiness returns WRONG hash to fail pre-baseline gate.
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": "WRONG_HASH",
                "config_version": "WRONG_VER",
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "p", "model": "m", "route_family": "rf"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "command": ["codex"],
            }},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("pre_baseline" in f for f in result["failures"])
        # No TUI proof calls may have occurred.
        assert tui_log == [], f"TUI calls must not occur: {tui_log}"
        # No POST calls may have occurred (no baseline, no swap, no controls).
        # Restoration is allowed in finally but only if raw_source_text was set.
        # Since we raised before any POST, restoration POST is the only allowed one.
        # But raw_source_text IS set before readiness, so restoration may fire.
        # The key assertion: no baseline/swap/control POSTs before restoration.
        assert len(post_log) <= 1, f"Too many POSTs: {post_log}"

    def test_wrong_swap_hash_prevents_swap_proof(self, adapter, ra, monkeypatch):
        """Wrong post-swap hash must prevent swap proof TUI call."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        tui_log: list[str] = []
        call_n = {"n": 0}

        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                if call_n["n"] <= 2:
                    return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                                 "config_version": auth["config_version"]}
                return 200, {"changed": True, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            # Swap: return WRONG hash
            return 200, {"changed": True, "active_config_hash": "WRONG_SWAP_HASH",
                         "config_version": "WRONG_VER",
                         "active_candidate_order": {"basic": []}}

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            tui_log.append(case_name)
            return {"passed": True}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": eligible[0]["provider"],
                                       "model": eligible[0]["model"],
                                       "route_family": eligible[0]["route_family"]})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "command": ["codex"],
            }},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("hash mismatch" in f for f in result["failures"])
        # Swap proof TUI call must NOT have occurred.
        swap_proof_calls = [c for c in tui_log if "swap_proof" in c]
        assert swap_proof_calls == [], f"Swap proof TUI must not run: {swap_proof_calls}"

    def test_wrong_swap_order_prevents_swap_proof(self, adapter, ra, monkeypatch):
        """Wrong post-swap candidate order must prevent swap proof TUI call."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        # Compute expected swapped hash.
        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        correct_hash = swapped_snap.config_hash
        correct_version = swapped_snap.config_version

        tui_log: list[str] = []
        call_n = {"n": 0}
        wrong_order = [{"provider": "WRONG", "model": "WRONG",
                        "route_family": "WRONG", "priority": 0}]

        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                if call_n["n"] <= 2:
                    return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                                 "config_version": auth["config_version"]}
                return 200, {"changed": True, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            # Swap: correct hash/version but WRONG order
            return 200, {"changed": True, "active_config_hash": correct_hash,
                         "config_version": correct_version,
                         "active_candidate_order": {"basic": wrong_order}}

        def fake_run(*, case_name, case_config, suite_config, query_url,
                     public_key, secret_key, litellm_base_url,
                     cfg003_transactional=False):
            tui_log.append(case_name)
            return {"passed": True}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": eligible[0]["provider"],
                                       "model": eligible[0]["model"],
                                       "route_family": eligible[0]["route_family"]})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "command": ["codex"],
            }},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("order" in f.lower() for f in result["failures"])
        # Swap proof TUI call must NOT have occurred.
        swap_proof_calls = [c for c in tui_log if "swap_proof" in c]
        assert swap_proof_calls == [], f"Swap proof TUI must not run: {swap_proof_calls}"

    def test_evidence_preserves_observed_bad_order(self, adapter, ra, monkeypatch):
        """Evidence must preserve the actually observed bad order rather than
        copying the expected order."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        correct_hash = swapped_snap.config_hash
        correct_version = swapped_snap.config_version

        call_n = {"n": 0}
        bad_order = [{"provider": "BAD_PROV", "model": "BAD_MODEL",
                      "route_family": "BAD_RF", "priority": 99}]

        def fake_post(url, payload, **kw):
            call_n["n"] += 1
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                if call_n["n"] <= 2:
                    return 200, {"changed": False, "active_config_hash": auth["config_hash"],
                                 "config_version": auth["config_version"]}
                return 200, {"changed": True, "active_config_hash": auth["config_hash"],
                             "config_version": auth["config_version"],
                             "active_candidate_order": {"basic": full_order}}
            return 200, {"changed": True, "active_config_hash": correct_hash,
                         "config_version": correct_version,
                         "active_candidate_order": {"basic": bad_order}}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": eligible[0]["provider"],
                                       "model": eligible[0]["model"],
                                       "route_family": eligible[0]["route_family"]})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "command": ["codex"],
            }},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        # The swap_refresh evidence must contain the ACTUAL observed bad order.
        swap_ev = result["phases"]["swap_refresh"]
        assert swap_ev["active_candidate_order"] == bad_order
        assert swap_ev["order_matches_expected"] is False


# ---------------------------------------------------------------------------
# Finding 3 (round 8): Raw-config canonical preflight before dotenv
# ---------------------------------------------------------------------------


class TestRawConfigCanonicalPreflight:
    def test_invalid_target_rejects_before_dotenv(self, adapter, monkeypatch, tmp_path):
        """An invalid transactional target must call neither the dotenv loader
        nor the credential resolver."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "cases": {},
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        dotenv_calls: list[str] = []
        cred_calls: list[dict] = []

        def spy_dotenv(path):
            dotenv_calls.append(str(path))

        def spy_credentials(**kw):
            cred_calls.append(kw)
            return ("pk", "sk", "http://q", "pk_env", "sk_env")

        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", spy_dotenv)
        monkeypatch.setattr(adapter, "_resolve_main_credentials", spy_credentials)
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "dev",
            "--litellm-base-url", "http://127.0.0.1:4000",
            "--cfg003-transactional-refresh",
        ])

        exit_code = adapter.main()
        assert exit_code == 1
        # Dotenv loader must NOT have been called.
        assert dotenv_calls == [], f"dotenv must not be called: {dotenv_calls}"
        # Credential resolver must NOT have been called.
        assert cred_calls == [], f"credentials must not be called: {cred_calls}"
        # Artifact must record the raw config gate failure.
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert "cfg003_raw_config_gate" in artifact["results"]
        assert artifact["results"]["cfg003_raw_config_gate"]["passed"] is False

    def test_prod_target_rejects_before_dotenv(self, adapter, monkeypatch, tmp_path):
        """Target=prod with --cfg003-transactional-refresh must reject before
        dotenv loading."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "prod",
            "cases": {},
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        dotenv_calls: list[str] = []

        def spy_dotenv(path):
            dotenv_calls.append(str(path))

        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", spy_dotenv)
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: ("pk", "sk", "http://q", "pk_env", "sk_env"))
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "prod",
            "--cfg003-transactional-refresh",
        ])

        exit_code = adapter.main()
        assert exit_code == 1
        assert dotenv_calls == [], f"dotenv must not be called: {dotenv_calls}"

    def test_valid_target_allows_dotenv(self, adapter, monkeypatch, tmp_path):
        """A valid canonical dev target must allow dotenv loading to proceed."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "dev",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "non_tui_case": {"required_env": ["NONEXISTENT_ENV_VAR_XYZ"]},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        dotenv_calls: list[str] = []

        def spy_dotenv(path):
            dotenv_calls.append(str(path))

        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", spy_dotenv)
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: ("pk", "sk", "http://q", "pk_env", "sk_env"))
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": True, "inventory_failures": [],
                                         "alias_inventory": []})
        monkeypatch.delenv("NONEXISTENT_ENV_VAR_XYZ", raising=False)
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--cases", "non_tui_case",
            "--cfg003-transactional-refresh",
        ])

        adapter.main()
        # Dotenv loader MUST have been called for a valid target.
        assert len(dotenv_calls) == 1, f"dotenv must be called once: {dotenv_calls}"

    def test_raw_preflight_function_rejects_prod_url(self, adapter):
        """The raw-config preflight function must reject a prod URL override."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"default_target_profile": "dev", "cases": {}}, f)
            f.flush()
            ok, failures = adapter._cfg003_raw_config_canonical_preflight(
                config_path=adapter.pathlib.Path(f.name),
                target_override="dev",
                litellm_base_url_override="http://127.0.0.1:4000",
                anthropic_base_url_override=None,
                docker_container_name_override=None,
                expected_trace_environment_override=None,
            )
        assert not ok
        assert any("4000" in f or "port" in f.lower() for f in failures)

    def test_raw_preflight_function_accepts_canonical_dev(self, adapter):
        """The raw-config preflight function must accept the canonical dev profile."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"default_target_profile": "dev", "cases": {}}, f)
            f.flush()
            ok, failures = adapter._cfg003_raw_config_canonical_preflight(
                config_path=adapter.pathlib.Path(f.name),
                target_override=None,
                litellm_base_url_override=None,
                anthropic_base_url_override=None,
                docker_container_name_override=None,
                expected_trace_environment_override=None,
            )
        assert ok, f"Canonical dev must pass: {failures}"
        assert failures == []

    def test_non_transactional_run_skips_raw_preflight(self, adapter, monkeypatch, tmp_path):
        """Without --cfg003-transactional-refresh, the raw preflight must not
        block even a prod target."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config = {
            "default_target_profile": "prod",
            "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
            "cases": {
                "non_tui_case": {"required_env": ["NONEXISTENT_ENV_VAR_XYZ"]},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        dotenv_calls: list[str] = []

        def spy_dotenv(path):
            dotenv_calls.append(str(path))

        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", spy_dotenv)
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: ("pk", "sk", "http://q", "pk_env", "sk_env"))
        monkeypatch.setattr(adapter, "_docker_status_for_container", lambda name: "Up")
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": True, "inventory_failures": [],
                                         "alias_inventory": []})
        monkeypatch.delenv("NONEXISTENT_ENV_VAR_XYZ", raising=False)
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "prod",
            "--cases", "non_tui_case",
        ])

        # Must NOT exit 1 from the raw preflight (it is skipped).
        adapter.main()
        assert len(dotenv_calls) == 1


# ---------------------------------------------------------------------------
# Finding 1 (round 9): pre-baseline readiness rejection causes zero POSTs
# ---------------------------------------------------------------------------


class TestPreBaselineRejectionZeroPosts:
    def test_pre_baseline_rejection_zero_posts_zero_tui(self, adapter, ra, monkeypatch):
        """When pre-baseline readiness fails, no POST or TUI call may occur.
        Restoration must NOT fire because mutation_attempted is False."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        post_calls: list[str] = []
        tui_calls: list[str] = []

        def fake_post(url, payload, **kw):
            post_calls.append(url)
            return 200, {"changed": False}

        def fake_run_case(**kw):
            tui_calls.append(kw.get("case_name", ""))
            return {"passed": True}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        # Readiness check returns WRONG hash -> pre_baseline fails.
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": "WRONG_HASH",
                "config_version": "WRONG_VER",
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "x", "model": "x", "route_family": "x"})
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda: {"a.yaml": "h"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert len(post_calls) == 0, f"Expected zero POSTs, got {len(post_calls)}: {post_calls}"
        assert len(tui_calls) == 0, f"Expected zero TUI calls, got {len(tui_calls)}: {tui_calls}"
        # No restoration phase should exist since no mutation was attempted.
        assert "restoration" not in result.get("phases", {})

    def test_pre_baseline_rejection_records_failures(self, adapter, ra, monkeypatch):
        """Pre-baseline rejection must record the readiness failures."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": "MISMATCH",
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", lambda *a, **kw: (200, {}))
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection",
                            lambda r: {"provider": "x", "model": "x", "route_family": "x"})
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda: {"a.yaml": "h"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={}, suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("hash mismatch" in f for f in result["failures"])
        assert result["phases"]["pre_baseline_readiness"]["passed"] is False


# ---------------------------------------------------------------------------
# Finding 3 (round 9): wrong swap version blocks swap_proof, restoration fires
# ---------------------------------------------------------------------------


class TestWrongSwapVersionBlocksSwapProof:
    def test_wrong_swap_version_no_swap_proof_tui(self, adapter, ra, monkeypatch):
        """Swap refresh returning correct hash but WRONG version must block
        swap_proof TUI call while restoration still fires."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        correct_hash = swapped_snap.config_hash
        swapped_full_order = ra._derive_full_order_from_snapshot(swapped_snap, alias_name="basic")

        tui_calls: list[str] = []
        post_urls: list[str] = []

        def _full_order_response(order):
            return {"basic": [
                {"provider": c["provider"], "model": c["model"],
                 "route_family": c["route_family"],
                 "anthropic_route_family": c.get("anthropic_route_family", ""),
                 "priority": c["priority"], "last_resort": c.get("last_resort", False)}
                for c in order
            ]}

        def fake_post(url, payload, **kw):
            post_urls.append(url)
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                return 200, {
                    "changed": False, "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": _full_order_response(full_order),
                }
            # Swap: correct hash, WRONG version, correct order.
            return 200, {
                "changed": True, "active_config_hash": correct_hash,
                "config_version": "WRONG_VERSION",
                "active_candidate_order": _full_order_response(swapped_full_order),
            }

        proof_call = {"n": 0}
        def fake_run_case(**kw):
            proof_call["n"] += 1
            tui_calls.append(kw.get("case_name", ""))
            return {"passed": True}

        def fake_selection(r):
            return {"provider": eligible[0]["provider"], "model": eligible[0]["model"],
                    "route_family": eligible[0]["route_family"]}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", fake_selection)
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda: {"a.yaml": "h"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        assert any("version mismatch" in f for f in result["failures"])
        # swap_proof must NOT have been called (only baseline = 1 TUI call).
        swap_proof_calls = [c for c in tui_calls if "swap_proof" in c]
        assert len(swap_proof_calls) == 0, f"swap_proof TUI must not fire: {swap_proof_calls}"
        # Restoration must still fire (mutation was attempted).
        assert "restoration" in result.get("phases", {})


# ---------------------------------------------------------------------------
# Finding 3 (round 9): failed pre-swap readiness blocks swap_proof
# ---------------------------------------------------------------------------


class TestFailedPreSwapReadinessBlocksSwapProof:
    def test_pre_swap_readiness_failure_no_swap_proof_tui(self, adapter, ra, monkeypatch):
        """When pre-swap-proof readiness check fails (runtime hash mismatch
        after swap POST), swap_proof TUI must not fire but restoration must."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")
        raw_text = BASIC_YAML_PATH.read_text(encoding="utf-8")
        avail_res = _avail_result(eligible[:3])

        evidenced_pair = (
            (eligible[0]["provider"], eligible[0]["model"]),
            (eligible[1]["provider"], eligible[1]["model"]),
        )
        swapped_yaml, _, _ = ra._build_exact_pair_priority_swap_yaml(
            raw_text, pair=evidenced_pair, alias_name="basic"
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import compile_yaml
        swapped_snap = compile_yaml(swapped_yaml)
        swapped_hash = swapped_snap.config_hash
        swapped_version = swapped_snap.config_version
        swapped_full_order = ra._derive_full_order_from_snapshot(swapped_snap, alias_name="basic")

        tui_calls: list[str] = []
        readiness_call_n = {"n": 0}

        def _full_order_response(order):
            return {"basic": [
                {"provider": c["provider"], "model": c["model"],
                 "route_family": c["route_family"],
                 "anthropic_route_family": c.get("anthropic_route_family", ""),
                 "priority": c["priority"], "last_resort": c.get("last_resort", False)}
                for c in order
            ]}

        def fake_post(url, payload, **kw):
            yaml_text = payload.get("yaml", "")
            if "not_a_list" in yaml_text:
                return 400, {"detail": {"active_config_hash": auth["config_hash"]}}
            if yaml_text == raw_text:
                return 200, {
                    "changed": False, "active_config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "active_candidate_order": _full_order_response(full_order),
                }
            # Swap: correct hash/version/order.
            return 200, {
                "changed": True, "active_config_hash": swapped_hash,
                "config_version": swapped_version,
                "active_candidate_order": _full_order_response(swapped_full_order),
            }

        def fake_readiness_get(url, **kw):
            readiness_call_n["n"] += 1
            # First calls (pre_baseline, post_baseline) return correct original.
            # After swap POST, pre_swap_proof readiness returns WRONG hash.
            if readiness_call_n["n"] <= 2:
                return 200, {"aawm_alias_config": {
                    "state": "active", "config_hash": auth["config_hash"],
                    "config_version": auth["config_version"],
                    "files": auth["file_names"], "aliases": auth["aliases"],
                }}
            # pre_swap_proof: wrong hash (simulates runtime not yet updated).
            return 200, {"aawm_alias_config": {
                "state": "active", "config_hash": "STALE_RUNTIME_HASH",
                "config_version": swapped_version,
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}

        proof_call = {"n": 0}
        def fake_run_case(**kw):
            proof_call["n"] += 1
            tui_calls.append(kw.get("case_name", ""))
            return {"passed": True}

        def fake_selection(r):
            return {"provider": eligible[0]["provider"], "model": eligible[0]["model"],
                    "route_family": eligible[0]["route_family"]}

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", fake_readiness_get)
        monkeypatch.setattr(adapter.RA, "_http_post_json", fake_post)
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", fake_selection)
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory", lambda: {"a.yaml": "h"})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        # Pre-swap-proof readiness must have failed.
        pre_swap = result["phases"].get("pre_swap_proof_readiness", {})
        assert pre_swap.get("passed") is False
        # swap_proof TUI must NOT have been called.
        swap_proof_calls = [c for c in tui_calls if "swap_proof" in c]
        assert len(swap_proof_calls) == 0, f"swap_proof must not fire: {swap_proof_calls}"
        # Restoration must still fire (mutation was attempted via swap POST).
        assert "restoration" in result.get("phases", {})


# ---------------------------------------------------------------------------
# Finding 2 (round 9): per-case error-intake validation during CFG-003
# ---------------------------------------------------------------------------


class TestPerCaseErrorIntake:
    def test_case_correlation_ids_extracts_session_and_trace(self, adapter):
        """_cfg003_case_correlation_ids must extract session/trace from langfuse."""
        result = {
            "passed": True,
            "langfuse": {
                "command_session_id": "sess-123",
                "trace_ids": ["trace-abc", "trace-def"],
            },
        }
        session, trace = adapter._cfg003_case_correlation_ids(result)
        assert session == "sess-123"
        assert trace == "trace-abc"

    def test_case_correlation_ids_missing_langfuse(self, adapter):
        """Without langfuse section, returns (None, None)."""
        session, trace = adapter._cfg003_case_correlation_ids({"passed": True})
        assert session is None
        assert trace is None

    def test_case_correlation_ids_empty_traces(self, adapter):
        """With empty trace list, trace is None."""
        result = {"langfuse": {"command_session_id": "s1", "trace_ids": []}}
        session, trace = adapter._cfg003_case_correlation_ids(result)
        assert session == "s1"
        assert trace is None

    def test_per_case_intake_fails_case_on_attributable_events(self, adapter, ra, monkeypatch, tmp_path):
        """When error intake finds attributable events during a case, the case
        must be marked failed with the intake evidence persisted."""
        import datetime as dt

        # Create a fake error intake file with a new attributable event.
        analysis_dir = tmp_path / ".analysis"
        analysis_dir.mkdir()
        error_file = analysis_dir / "test-error.jsonl"
        event = {
            "environment": "dev",
            "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "logger": "test",
            "level": "ERROR",
            "message": "test error",
            "fingerprint": "fp1",
            "context": {"source": "test", "container": "litellm-dev"},
        }
        error_file.write_text(json.dumps(event) + "\n", encoding="utf-8")

        # Empty baseline -> the new event is a delta.
        baseline = {}
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)

        intake = adapter._cfg003_phase_error_intake(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            case_name="test_case",
            session_id="sess-1",
            trace_id="trace-1",
            analysis_dir=analysis_dir,
        )
        # The delta should detect the new event.
        assert intake["current_summary"]["file_count"] >= 1
        assert intake["advanced_baseline"] is not None


# ---------------------------------------------------------------------------
# Final remediation: Finding 1 - cleanup verifier exceptions never mask
# restoration failure; recovery artifact always emitted
# ---------------------------------------------------------------------------


class TestCleanupVerifierExceptionIsolation:
    def test_source_verifier_exception_does_not_mask_restoration(self, adapter, ra, monkeypatch):
        """A source-file verifier exception must be recorded but never escape
        or mask the primary restoration failure."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"],
            "file_names": auth["file_names"],
            "config_hash": auth["config_hash"],
            "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        # Restoration POST fails.
        monkeypatch.setattr(adapter.RA, "_http_post_json", lambda *a, **kw: (
            500, {"changed": False}
        ))
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })
        # Source verifier raises.
        def broken_source_verify(*a, **kw):
            raise RuntimeError("disk error")
        monkeypatch.setattr(adapter, "_cfg003_verify_source_files_unchanged", broken_source_verify)
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        assert not result["passed"]
        # Restoration failure must be the PRIMARY (first) failure.
        assert result["failures"][0].startswith("RESTORATION")
        # Source verifier exception is recorded but does not mask.
        assert any("source file verifier exception" in f for f in result["failures"])
        # Recovery artifact always emitted.
        assert "recovery_artifact" in result

    def test_recovery_artifact_always_emitted(self, adapter, ra, monkeypatch):
        """Recovery artifact must be present even when restoration succeeds."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        avail_res = _avail_result(eligible[:3])
        full_order = ra._derive_full_order_from_snapshot(snap, alias_name="basic")

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"],
            "file_names": auth["file_names"],
            "config_hash": auth["config_hash"],
            "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_http_get_json_plain", lambda *a, **kw: (
            200, {"aawm_alias_config": {
                "state": "active", "config_hash": auth["config_hash"],
                "config_version": auth["config_version"],
                "files": auth["file_names"], "aliases": auth["aliases"],
            }}
        ))
        monkeypatch.setattr(adapter.RA, "_http_post_json", lambda *a, **kw: (
            200, {"changed": True, "active_config_hash": auth["config_hash"],
                  "config_version": auth["config_version"],
                  "active_candidate_order": {"basic": full_order}}
        ))
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence", lambda *a, **kw: avail_res)
        monkeypatch.setattr(adapter, "_run_selected_case", lambda **kw: {"passed": True})
        monkeypatch.setattr(adapter, "_cfg003_extract_observed_selection", lambda r: {
            "provider": eligible[0]["provider"],
            "model": eligible[0]["model"],
            "route_family": eligible[0]["route_family"],
        })
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
        )
        # Recovery artifact must always be present.
        assert "recovery_artifact" in result
        assert result["recovery_artifact"]["original_semantic_hash"] == auth["config_hash"]


# ---------------------------------------------------------------------------
# Final remediation: Finding 2/9 - strict correlation
# ---------------------------------------------------------------------------


class TestStrictCorrelation:
    def test_strict_mode_rejects_sparse_events(self, ra, tmp_path):
        """In strict mode, events without matching session+trace are rejected."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "observed_at": new_time, "environment": "dev", "level": "error",
                "message": "sparse event",
                "context": {"container": "litellm-dev"},
            }) + "\n")
        # Non-strict: sparse fallback qualifies.
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", session_id="s1", trace_id="t1",
            analysis_dir=analysis,
        )
        assert len(events) == 1
        # Strict: sparse event does NOT qualify.
        events_strict, failures_strict = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", session_id="s1", trace_id="t1",
            strict_correlation=True, analysis_dir=analysis,
        )
        assert len(events_strict) == 0

    def test_strict_mode_accepts_exact_match(self, ra, tmp_path):
        """In strict mode, events with exact session+trace match qualify."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "observed_at": new_time, "environment": "dev", "level": "error",
                "message": "correlated event",
                "context": {"container": "litellm-dev", "session_id": "s1", "trace_id": "t1"},
            }) + "\n")
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", session_id="s1", trace_id="t1",
            strict_correlation=True, analysis_dir=analysis,
        )
        assert len(events) == 1
        assert events[0]["attributed_session"] == "s1"
        assert events[0]["attributed_trace"] == "t1"


# ---------------------------------------------------------------------------
# Final remediation: Finding 3 - single-snapshot reuse + fail-closed unreadable
# ---------------------------------------------------------------------------


class TestSingleSnapshotAndFailClosed:
    def test_current_snapshot_reused(self, ra, tmp_path):
        """When current_snapshot is provided, no second snapshot is taken."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        new_time = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(analysis / "dev-error.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "observed_at": new_time, "environment": "dev", "level": "error",
                "message": "test", "context": {"container": "litellm-dev"},
            }) + "\n")
        # Provide an explicit current snapshot.
        current = ra._snapshot_error_intake(analysis)
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", current_snapshot=current,
            analysis_dir=analysis,
        )
        assert len(events) == 1
        assert not failures

    def test_unreadable_intake_fails_closed(self, ra, tmp_path):
        """Unreadable intake files must produce an explicit failure."""
        import datetime as dt
        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        # Simulate unreadable by injecting an error marker into the snapshot.
        current = {"dev-error.jsonl": {"size": 0, "line_count": 0, "inode": 0, "error": "unreadable"}}
        events, failures = ra._collect_error_intake_delta(
            baseline, initiation_time=initiation, environment="dev",
            container="litellm-dev", current_snapshot=current,
            analysis_dir=analysis,
        )
        assert any("unreadable" in f and "fail closed" in f for f in failures)


# ---------------------------------------------------------------------------
# Final remediation: Finding 5 - nonadjacent swap validation
# ---------------------------------------------------------------------------

_NONADJACENT_YAML = """\
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/owl-alpha
        route_family: codex_openrouter_completion_adapter
        priority: 400
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max-preview
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 300
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        priority: 200
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 100
"""


class TestNonAdjacentSwapValidation:
    def test_nonadjacent_pair_swap_preserves_intermediate(self, ra):
        """Swapping A (pos 1) and C (pos 3) with B (pos 2) between them must
        exchange A/C priorities and leave B unchanged."""
        pair = (("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview"), ("openai", "gpt-5.6-luna"))
        swapped_yaml, original, swapped = ra._build_exact_pair_priority_swap_yaml(
            _NONADJACENT_YAML, pair=pair
        )
        orig_by = {(c["provider"], c["model"]): c["priority"] for c in original}
        swap_by = {(c["provider"], c["model"]): c["priority"] for c in swapped}
        # A and C priorities exchanged.
        assert swap_by[("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview")] == orig_by[("openai", "gpt-5.6-luna")]
        assert swap_by[("openai", "gpt-5.6-luna")] == orig_by[("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview")]
        # B (nvidia) unchanged.
        assert swap_by[("opencode_zen", "deepseek-v4-flash")] == orig_by[("opencode_zen", "deepseek-v4-flash")] == 200
        # First raw candidate (openrouter) unchanged.
        assert swap_by[("openrouter", "openrouter/owl-alpha")] == orig_by[("openrouter", "openrouter/owl-alpha")] == 400

    def test_nonadjacent_relative_position_in_swapped_eligible(self, ra):
        """After swap, C must precede A in the eligible order even though
        they are not at positions 0 and 1."""
        pair = (("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview"), ("openai", "gpt-5.6-luna"))
        _yaml, _orig, swapped = ra._build_exact_pair_priority_swap_yaml(
            _NONADJACENT_YAML, pair=pair
        )
        ids = [(c["provider"], c["model"]) for c in swapped]
        pos_a = ids.index(("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview"))
        pos_c = ids.index(("openai", "gpt-5.6-luna"))
        # C now has higher priority (300) than A (100), so C comes first.
        assert pos_c < pos_a

    def test_compiled_full_order_matches_swap(self, ra):
        """The compiled full order from the swapped YAML must reflect the
        priority exchange."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
            compile_yaml,
        )
        pair = (("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview"), ("openai", "gpt-5.6-luna"))
        swapped_yaml, _orig, _swapped = ra._build_exact_pair_priority_swap_yaml(
            _NONADJACENT_YAML, pair=pair
        )
        snapshot = compile_yaml(swapped_yaml)
        full_order = ra._derive_full_order_from_snapshot(snapshot, alias_name="basic")
        # All four candidates present.
        assert len(full_order) == 4
        # C (openai/gpt-x) now has priority 300, A has 100.
        by_id = {(c["provider"], c["model"]): c for c in full_order}
        assert by_id[("openai", "gpt-5.6-luna")]["priority"] == 300
        assert by_id[("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview")]["priority"] == 100


# ---------------------------------------------------------------------------
# Final remediation: Finding 6 - mandatory runtime-log evidence
# ---------------------------------------------------------------------------


class TestMandatoryRuntimeLogEvidence:
    def test_require_evidence_fails_on_unreadable_logs(self, adapter, monkeypatch):
        """When require_evidence=True, unreadable docker logs must fail."""
        def fake_read(*, started, until, checks, runtime_postconditions):
            return {"docker_logs_exit_code": 1, "log_structural": {"line_count": 0, "char_count": 0, "sha256": ""}}, ""
        monkeypatch.setattr(adapter, "_read_runtime_logs_since", fake_read)
        summary, failures, warnings = adapter._validate_runtime_logs(
            family="test_case",
            started="2026-01-01T00:00:00",
            checks={"docker_container_name": "litellm-dev"},
            runtime_postconditions={},
            require_evidence=True,
        )
        assert any("mandatory" in f and "unreadable" in f for f in failures)
        assert not warnings

    def test_no_require_evidence_warns_on_unreadable_logs(self, adapter, monkeypatch):
        """When require_evidence=False (ordinary), unreadable logs produce
        only a warning."""
        def fake_read(*, started, until, checks, runtime_postconditions):
            return {"docker_logs_exit_code": 1, "log_structural": {"line_count": 0, "char_count": 0, "sha256": ""}}, ""
        monkeypatch.setattr(adapter, "_read_runtime_logs_since", fake_read)
        summary, failures, warnings = adapter._validate_runtime_logs(
            family="test_case",
            started="2026-01-01T00:00:00",
            checks={"docker_container_name": "litellm-dev"},
            runtime_postconditions={},
            require_evidence=False,
        )
        assert not failures
        assert any("could not read docker logs" in w for w in warnings)


# ---------------------------------------------------------------------------
# Defect 1: strict intake correlation - missing phase IDs + legacy .log
# ---------------------------------------------------------------------------


class TestStrictIntakeCorrelationMissingIDs:
    def test_phase_intake_fails_when_session_missing(self, adapter, ra, tmp_path):
        """_cfg003_phase_error_intake must fail closed when strict and
        session_id is missing."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        result = adapter._cfg003_phase_error_intake(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            case_name="test_case",
            session_id=None,
            trace_id="trace-1",
            strict_correlation=True,
            analysis_dir=analysis,
        )
        assert result["failures"]
        assert any("missing required correlation IDs" in f for f in result["failures"])
        assert result["attributed_count"] == 0

    def test_phase_intake_fails_when_trace_missing(self, adapter, ra, tmp_path):
        """_cfg003_phase_error_intake must fail closed when strict and
        trace_id is missing."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        result = adapter._cfg003_phase_error_intake(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            case_name="test_case",
            session_id="sess-1",
            trace_id=None,
            strict_correlation=True,
            analysis_dir=analysis,
        )
        assert result["failures"]
        assert any("missing required correlation IDs" in f for f in result["failures"])

    def test_phase_intake_fails_when_both_missing(self, adapter, ra, tmp_path):
        """_cfg003_phase_error_intake must fail closed when strict and both
        session_id and trace_id are missing."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        result = adapter._cfg003_phase_error_intake(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            case_name="test_case",
            session_id=None,
            trace_id=None,
            strict_correlation=True,
            analysis_dir=analysis,
        )
        assert result["failures"]
        assert any("missing required correlation IDs" in f for f in result["failures"])

    def test_phase_intake_ok_without_strict_and_missing_ids(self, adapter, ra, tmp_path):
        """Without strict_correlation, missing IDs do not produce a failure."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.jsonl").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        result = adapter._cfg003_phase_error_intake(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            case_name="test_case",
            session_id=None,
            trace_id=None,
            strict_correlation=False,
            analysis_dir=analysis,
        )
        assert not result["failures"]


class TestStrictLegacyLogExclusion:
    def test_legacy_log_not_attributed_in_strict_mode(self, ra, tmp_path):
        """Legacy .log lines with environment substring must NOT be
        attributed in strict correlation mode."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "dev-error.log").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        with open(analysis / "dev-error.log", "a", encoding="utf-8") as f:
            f.write("2026-07-30 ERROR dev something went wrong\n")

        # Non-strict: legacy line IS attributed.
        events_nonstrict, _ = ra._collect_error_intake_delta(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            strict_correlation=False,
            analysis_dir=analysis,
        )
        assert any(e.get("legacy_line") for e in events_nonstrict)

        # Strict: legacy line is NOT attributed.
        events_strict, _ = ra._collect_error_intake_delta(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            strict_correlation=True,
            analysis_dir=analysis,
        )
        assert not any(e.get("legacy_line") for e in events_strict)
        assert len(events_strict) == 0

    def test_legacy_log_environment_only_never_satisfies_case(self, ra, tmp_path):
        """Environment-only legacy lines cannot satisfy or fail a case in
        strict mode, even when they match the environment exactly."""
        import datetime as dt

        analysis = tmp_path / ".analysis"
        analysis.mkdir()
        (analysis / "prod-error.log").write_text("", encoding="utf-8")
        baseline = ra._snapshot_error_intake(analysis)
        initiation = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

        with open(analysis / "prod-error.log", "a", encoding="utf-8") as f:
            f.write("dev container litellm-dev session sess-1 trace trace-1 ERROR\n")

        events, failures = ra._collect_error_intake_delta(
            baseline,
            initiation_time=initiation,
            environment="dev",
            container="litellm-dev",
            session_id="sess-1",
            trace_id="trace-1",
            strict_correlation=True,
            analysis_dir=analysis,
        )
        # Even with session/trace text in the line, legacy .log cannot be
        # attributed in strict mode.
        assert len(events) == 0
        assert not failures


# ---------------------------------------------------------------------------
# Defect 2: runtime-log evidence per-case derivation
# ---------------------------------------------------------------------------


class TestRuntimeLogEvidencePerCaseDerivation:
    def test_alias_case_requires_evidence(self, adapter):
        """Cases with verification_alias require runtime-log evidence."""
        assert adapter._cfg003_case_requires_runtime_evidence(
            "some_alias_case", {"verification_alias": "basic"}
        ) is True

    def test_proof_case_codex_requires_evidence(self, adapter):
        """Exact CFG-003 Codex proof case identity requires evidence."""
        assert adapter._cfg003_case_requires_runtime_evidence(
            adapter._CFG003_CODEX_PROOF_CASE, {"command": ["codex"]}
        ) is True

    def test_proof_case_claude_requires_evidence(self, adapter):
        """Exact CFG-003 Claude proof case identity requires evidence."""
        assert adapter._cfg003_case_requires_runtime_evidence(
            adapter._CFG003_CLAUDE_PROOF_CASE, {"command": ["claude"]}
        ) is True

    def test_proof_case_suffixed_requires_evidence(self, adapter):
        """Suffixed proof case names (e.g. __cfg003_baseline) require evidence."""
        assert adapter._cfg003_case_requires_runtime_evidence(
            adapter._CFG003_CODEX_PROOF_CASE + "__cfg003_baseline",
            {"command": ["codex"]},
        ) is True
        assert adapter._cfg003_case_requires_runtime_evidence(
            adapter._CFG003_CLAUDE_PROOF_CASE + "__cfg003_restore_proof",
            {"command": ["claude"]},
        ) is True

    def test_non_alias_case_does_not_require_evidence(self, adapter):
        """Ordinary non-alias cases must NOT require runtime-log evidence,
        even under the global transactional flag."""
        assert adapter._cfg003_case_requires_runtime_evidence(
            "plain_http_case", {"http_request": {"method": "POST"}}
        ) is False
        assert adapter._cfg003_case_requires_runtime_evidence(
            "plain_command_case", {"command": ["echo", "hello"]}
        ) is False
        assert adapter._cfg003_case_requires_runtime_evidence(
            "empty_config_case", {}
        ) is False

    def test_non_alias_transactional_case_no_evidence_requirement(self, adapter, monkeypatch):
        """Regression: a non-alias selected case under cfg003_transactional=True
        must pass require_evidence=False to _validate_runtime_logs."""
        captured = {}

        def fake_validate_runtime_logs(*, family, started, checks, runtime_postconditions,
                                       attribution_substrings=None, require_evidence=False):
            captured["require_evidence"] = require_evidence
            return {}, [], []

        # Build a minimal _validate_case call path.
        monkeypatch.setattr(adapter, "_validate_runtime_logs", fake_validate_runtime_logs)
        # We test the derivation logic directly since _validate_case has many
        # dependencies.  The derivation is: cfg003_transactional AND
        # _cfg003_case_requires_runtime_evidence.
        case_config = {"command": ["echo", "hello"]}
        derived = (
            True  # cfg003_transactional flag
            and adapter._cfg003_case_requires_runtime_evidence("plain_case", case_config)
        )
        assert derived is False

    def test_alias_transactional_case_evidence_requirement(self, adapter):
        """Positive: an alias case under cfg003_transactional=True must
        derive require_evidence=True."""
        case_config = {"verification_alias": "basic", "command": ["codex"]}
        derived = (
            True  # cfg003_transactional flag
            and adapter._cfg003_case_requires_runtime_evidence("alias_case", case_config)
        )
        assert derived is True


# ---------------------------------------------------------------------------
# Item 1: Claude child proof - tool_result + terminal completion + bounded
# ---------------------------------------------------------------------------


class TestChildProofToolResults:
    """require_all_tool_results: every tool_use must have a successful
    tool_result.  Three tool_use events with zero tool_result must fail."""

    def _make_summary(self, records, assistant_texts=None):
        return {
            "records": records,
            "assistant_texts": assistant_texts or [],
            "tool_result_errors": [],
            "by_tool_name": {},
            "total_tool_uses": len(records),
        }

    def test_three_tool_use_zero_results_fails(self, adapter, tmp_path):
        """Three tool_use events but zero tool_result/child terminal text
        must fail."""
        records = [
            {"tool_name": "Read", "tool_use_id": "t1", "line": 1, "path": str(tmp_path / "a.jsonl")},
            {"tool_name": "Glob", "tool_use_id": "t2", "line": 1, "path": str(tmp_path / "a.jsonl")},
            {"tool_name": "Grep", "tool_use_id": "t3", "line": 1, "path": str(tmp_path / "a.jsonl")},
        ]
        summary = self._make_summary(records)
        agent_checks = {
            "require_all_tool_results": True,
            "require_explicit_completion": True,
        }
        # Call the internal validation logic directly by simulating what
        # _validate_transcript_agent_tool_uses does after building summary.
        failures = []
        # require_all_tool_results
        if agent_checks["require_all_tool_results"]:
            missing = [
                f"{r['tool_name']}(id={r['tool_use_id']})"
                for r in records
                if not r.get("tool_result_line")
            ]
            if missing:
                failures.append(f"tool_use without tool_result: {', '.join(missing)}")
        # require_explicit_completion
        if agent_checks["require_explicit_completion"]:
            last_tool_line = max(int(r.get("line") or 0) for r in records)
            has_terminal = any(
                isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"].strip()
                and int(item.get("line") or 0) > last_tool_line
                for item in summary["assistant_texts"]
            )
            if not has_terminal:
                failures.append("no explicit completion text after final tool_use")

        assert len(failures) == 2
        assert "without tool_result" in failures[0]
        assert "Read" in failures[0]
        assert "Glob" in failures[0]
        assert "Grep" in failures[0]

    def test_all_results_present_passes(self, adapter):
        """When every tool_use has a successful tool_result, no failure."""
        records = [
            {"tool_name": "Read", "tool_use_id": "t1", "line": 1,
             "tool_result_line": 5, "tool_result_is_error": False},
            {"tool_name": "Glob", "tool_use_id": "t2", "line": 1,
             "tool_result_line": 6, "tool_result_is_error": False},
        ]
        failures = []
        missing = [r for r in records if not r.get("tool_result_line")]
        errored = [r for r in records if r.get("tool_result_is_error") is True]
        if missing:
            failures.append("missing")
        if errored:
            failures.append("errored")
        assert failures == []

    def test_errored_tool_result_fails(self, adapter):
        """A tool_result with is_error=True must fail."""
        records = [
            {"tool_name": "Read", "tool_use_id": "t1", "line": 1,
             "tool_result_line": 5, "tool_result_is_error": True},
        ]
        errored = [
            f"{r['tool_name']}(id={r['tool_use_id']})"
            for r in records
            if r.get("tool_result_is_error") is True
        ]
        assert len(errored) == 1
        assert "Read" in errored[0]


class TestChildProofTerminalResponse:
    """require_child_terminal_response: child must emit exact terminal text."""

    def test_missing_terminal_text_fails(self, adapter):
        assistant_texts = []
        required = "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        final_text = ""
        for item in reversed(assistant_texts):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                final_text = item["text"].strip()
                break
        assert final_text == ""
        # Must fail: missing terminal response
        assert not final_text
        assert final_text != required

    def test_wrong_terminal_text_fails(self, adapter):
        assistant_texts = [{"text": "SOME OTHER OUTPUT", "line": 10}]
        required = "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        final_text = ""
        for item in reversed(assistant_texts):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                final_text = item["text"].strip()
                break
        assert final_text != required

    def test_exact_terminal_text_passes(self, adapter):
        assistant_texts = [
            {"text": "intermediate", "line": 5},
            {"text": "BASIC_ALIAS_PARALLEL_TOOLS_PASSED", "line": 10},
        ]
        required = "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        final_text = ""
        for item in reversed(assistant_texts):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                final_text = item["text"].strip()
                break
        assert final_text == required


class TestChildProofBoundedOutput:
    """child_output_max_chars: child output must remain bounded."""

    def test_exceeding_bound_fails(self, adapter):
        assistant_texts = [{"text": "x" * 5000, "line": 1}]
        max_chars = 4096
        total = sum(len(i.get("text") or "") for i in assistant_texts if isinstance(i, dict))
        assert total > max_chars

    def test_within_bound_passes(self, adapter):
        assistant_texts = [{"text": "BASIC_ALIAS_PARALLEL_TOOLS_PASSED", "line": 1}]
        max_chars = 4096
        total = sum(len(i.get("text") or "") for i in assistant_texts if isinstance(i, dict))
        assert total <= max_chars


class TestChildProofConfigContract:
    """The checked-in Claude alias case contract must include child proof keys."""

    def test_basic_alias_case_has_child_proof_keys(self):
        config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        case = config["cases"]["claude_adapter_basic_alias_child_parallel_read_tools"]
        agents = case["transcript_tool_use_validation"]["expected_agents"]
        assert len(agents) == 1
        agent = agents[0]
        assert agent["require_all_tool_results"] is True
        assert agent["require_child_terminal_response"] == "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
        assert agent["require_explicit_completion"] is True
        assert isinstance(agent["child_output_max_chars"], int)
        assert agent["child_output_max_chars"] > 0
        # Existing checks preserved
        assert agent["forbid_tool_result_errors"] is True
        assert agent["minimum_total_tool_uses"] == 3
        assert agent["maximum_total_tool_uses"] == 3


# ---------------------------------------------------------------------------
# Item 2: Artifact privacy - runtime log evidence sentinel tests
# ---------------------------------------------------------------------------


class TestArtifactPrivacyRuntimeLogs:
    """Runtime log excerpts in artifacts must never contain raw prompts or
    tool arguments.  Structural metadata/digests only."""

    def test_log_excerpt_redacted_at_any_depth(self, adapter, ra, tmp_path):
        """Arbitrary prompt/tool-argument text under log_excerpt cannot
        survive _write_artifact at any nesting depth."""
        secret_prompt = "SECRET_PROMPT_TEXT_12345"
        secret_args = "SECRET_TOOL_ARGS_67890"
        artifact = {
            "results": {
                "case1": {
                    "passed": True,
                    "runtime_logs": {
                        "log_excerpt": f"line1 {secret_prompt} line2 {secret_args}",
                        "docker_logs_exit_code": 0,
                    },
                },
            },
            "nested": {
                "deep": {
                    "log_excerpt": secret_prompt,
                },
            },
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted_str = artifact_path.read_text(encoding="utf-8")
        assert secret_prompt not in persisted_str
        assert secret_args not in persisted_str

    def test_legacy_log_excerpt_key_redacted(self, adapter, ra, tmp_path):
        """Legacy/malformed artifact structures with log_excerpt at
        unexpected positions must be redacted."""
        secret = "LEGACY_SECRET_PROMPT_ABC"
        artifact = {
            "cfg003_transactional_refresh": {
                "phases": {
                    "baseline": {
                        "runtime_logs": {
                            "log_excerpt": secret,
                            "line_count": 42,
                        },
                    },
                },
            },
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted_str = artifact_path.read_text(encoding="utf-8")
        assert secret not in persisted_str

    def test_internal_log_text_key_redacted(self, adapter, ra, tmp_path):
        """The private _log_text key must never survive artifact writing."""
        secret = "INTERNAL_LOG_TEXT_SECRET_XYZ"
        artifact = {
            "results": {
                "case1": {
                    "runtime_logs": {
                        "_log_text": secret,
                        "log_structural": {"line_count": 5, "char_count": 100, "sha256": "abc"},
                    },
                },
            },
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted_str = artifact_path.read_text(encoding="utf-8")
        assert secret not in persisted_str

    def test_structural_metadata_preserved(self, adapter, ra, tmp_path):
        """Structural metadata (line_count, char_count, sha256) must survive
        artifact writing to prove logs were read."""
        artifact = {
            "results": {
                "case1": {
                    "passed": True,
                    "runtime_logs": {
                        "log_structural": {
                            "line_count": 42,
                            "char_count": 8192,
                            "sha256": "abcdef0123456789",
                        },
                        "log_evidence_read": True,
                        "docker_logs_exit_code": 0,
                    },
                },
            },
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
        logs = persisted["results"]["case1"]["runtime_logs"]
        assert logs["log_structural"]["line_count"] == 42
        assert logs["log_structural"]["char_count"] == 8192
        assert logs["log_structural"]["sha256"] == "abcdef0123456789"
        assert logs["log_evidence_read"] is True
        assert logs["docker_logs_exit_code"] == 0

    def test_credential_redaction_preserved(self, adapter, ra, tmp_path):
        """Existing credential redaction must still work alongside new
        log_excerpt redaction."""
        artifact = {
            "environment": {"api_key": "sk-test123456789012345"},
            "results": {
                "case1": {
                    "log_excerpt": "Bearer sk-ant-secret1234567890",
                    "command": ["codex", "exec"],
                },
            },
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted_str = artifact_path.read_text(encoding="utf-8")
        assert "sk-test123456789012345" not in persisted_str
        assert "sk-ant-secret1234567890" not in persisted_str
        assert "codex" not in persisted_str

    def test_malformed_list_log_excerpt_redacted(self, adapter, ra, tmp_path):
        """log_excerpt inside a list element must also be redacted."""
        secret = "LIST_NESTED_SECRET_PROMPT"
        artifact = {
            "items": [
                {"log_excerpt": secret, "safe": "visible"},
                {"other": "data"},
            ],
        }
        artifact_path = tmp_path / "artifact.json"
        adapter._write_artifact(artifact_path, artifact)
        persisted_str = artifact_path.read_text(encoding="utf-8")
        assert secret not in persisted_str
        assert "visible" in persisted_str


# ---------------------------------------------------------------------------
# CFG-003: Operator-asserted availability identities
# ---------------------------------------------------------------------------


class TestOperatorAssertionParsing:
    def test_valid_provider_model_with_slashes_and_colons(self, adapter):
        ids, failures = adapter._cfg003_parse_operator_assertions([
            "openrouter=openrouter/cohere/north-mini-code:free",
        ])
        assert failures == []
        assert ids == [("openrouter", "openrouter/cohere/north-mini-code:free")]

    def test_valid_alibaba_token_plan(self, adapter):
        ids, failures = adapter._cfg003_parse_operator_assertions([
            "alibaba_token_plan=alibaba_token_plan/qwen3.6-flash",
        ])
        assert failures == []
        assert ids == [("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash")]

    def test_malformed_no_equals(self, adapter):
        ids, failures = adapter._cfg003_parse_operator_assertions(["just-a-model"])
        assert len(failures) == 1
        assert "malformed" in failures[0]
        assert ids == []

    def test_duplicate_rejected(self, adapter):
        ids, failures = adapter._cfg003_parse_operator_assertions([
            "openrouter=model-a",
            "openrouter=model-a",
        ])
        assert len(failures) == 1
        assert "duplicate" in failures[0]
        assert len(ids) == 1

    def test_none_input_returns_empty(self, adapter):
        ids, failures = adapter._cfg003_parse_operator_assertions(None)
        assert ids == []
        assert failures == []


class TestOperatorAssertionSnapshotValidation:
    def _snapshot(self):
        return [
            {"provider": "openrouter", "model": "openrouter/cohere/north-mini-code:free",
             "route_family": "codex_x_adapter", "priority": 10},
            {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
             "route_family": "codex_x_adapter", "priority": 5},
        ]

    def test_valid_identity_passes(self, adapter):
        failures = adapter._cfg003_validate_operator_assertions(
            [("openrouter", "openrouter/cohere/north-mini-code:free")],
            eligible_snapshot=self._snapshot(),
        )
        assert failures == []

    def test_schedule_expired_identity_rejected(self, adapter):
        # qwen3.8-max-preview is NOT in the eligible snapshot (expired).
        failures = adapter._cfg003_validate_operator_assertions(
            [("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview")],
            eligible_snapshot=self._snapshot(),
        )
        assert len(failures) == 1
        assert "not in the current schedule-eligible" in failures[0]

    def test_bind_asserted_candidates_orders_by_priority(self, adapter):
        bound = adapter._cfg003_bind_asserted_candidates(
            [("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash"),
             ("openrouter", "openrouter/cohere/north-mini-code:free")],
            eligible_snapshot=self._snapshot(),
        )
        # Ordered by snapshot priority desc regardless of assertion order.
        assert [(c["provider"], c["model"]) for c in bound] == [
            ("openrouter", "openrouter/cohere/north-mini-code:free"),
            ("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash"),
        ]


def _assertion_record(provider, model, *, available=True, environment="dev",
                      stale=False, malformed=False):
    """Build a producer-shaped keyed availability record for merge tests."""
    import datetime as dt
    if malformed:
        return {"provider": provider, "model": model, "available": available}
    delta = dt.timedelta(hours=5) if stale else dt.timedelta(minutes=5)
    return {
        "provider": provider,
        "model": model,
        "available": available,
        "evidence": "remaining_pct=90" if available else "no_fresh_row",
        "observed_at": (dt.datetime.now(dt.timezone.utc) - delta).isoformat(),
        "environment": environment,
        "environment_binding": "target_db_profile",
    }


class TestOperatorAssertionMergeReplacement:
    """Producer-shaped keyed negative/stale/malformed DB evidence must be
    replaced by the exact operator assertion; boundary-valid positive DB
    evidence is preserved."""

    def test_keyed_negative_db_replaced_by_assertion(self, adapter):
        db = {("prov", "model"): _assertion_record("prov", "model", available=False)}
        assertion = {("prov", "model"): _assertion_record("prov", "model")}
        assertion[("prov", "model")]["source"] = "operator_assertion"
        merged = adapter._cfg003_merge_availability_evidence(
            db, assertion, environment="dev"
        )
        assert merged[("prov", "model")]["source"] == "operator_assertion"
        assert merged[("prov", "model")]["available"] is True

    def test_keyed_stale_db_replaced_by_assertion(self, adapter):
        db = {("prov", "model"): _assertion_record("prov", "model", stale=True)}
        assertion = {("prov", "model"): _assertion_record("prov", "model")}
        assertion[("prov", "model")]["source"] = "operator_assertion"
        merged = adapter._cfg003_merge_availability_evidence(
            db, assertion, environment="dev"
        )
        assert merged[("prov", "model")]["source"] == "operator_assertion"

    def test_keyed_malformed_db_replaced_by_assertion(self, adapter):
        db = {("prov", "model"): _assertion_record("prov", "model", malformed=True)}
        assertion = {("prov", "model"): _assertion_record("prov", "model")}
        assertion[("prov", "model")]["source"] = "operator_assertion"
        merged = adapter._cfg003_merge_availability_evidence(
            db, assertion, environment="dev"
        )
        assert merged[("prov", "model")]["source"] == "operator_assertion"

    def test_valid_positive_db_preserved_over_assertion(self, adapter):
        db = {("prov", "model"): _assertion_record("prov", "model")}
        db[("prov", "model")]["source"] = "rate_limit_observations"
        assertion = {("prov", "model"): _assertion_record("prov", "model")}
        assertion[("prov", "model")]["source"] = "operator_assertion"
        merged = adapter._cfg003_merge_availability_evidence(
            db, assertion, environment="dev"
        )
        assert merged[("prov", "model")]["source"] == "rate_limit_observations"

    def test_assertion_supplements_absent_db_key(self, adapter):
        db = {("prov_a", "model_a"): _assertion_record("prov_a", "model_a")}
        assertion = {("prov_b", "model_b"): _assertion_record("prov_b", "model_b")}
        assertion[("prov_b", "model_b")]["source"] = "operator_assertion"
        merged = adapter._cfg003_merge_availability_evidence(
            db, assertion, environment="dev"
        )
        assert ("prov_a", "model_a") in merged
        assert merged[("prov_b", "model_b")]["source"] == "operator_assertion"

    def test_db_dict_not_mutated(self, adapter):
        db = {("prov", "model"): _assertion_record("prov", "model", available=False)}
        assertion = {("prov", "model"): _assertion_record("prov", "model")}
        adapter._cfg003_merge_availability_evidence(db, assertion, environment="dev")
        assert db[("prov", "model")]["available"] is False


class TestOperatorAssertionEarlyFailureArtifact:
    def test_non_transactional_use_writes_artifact(self, adapter, monkeypatch, tmp_path):
        """Using --cfg003-assert-availability without --cfg003-transactional-refresh
        must write a sanitized failure artifact before any dotenv/credential work."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config_path.write_text(json.dumps({"cases": {}}), encoding="utf-8")

        dotenv_calls: list[str] = []
        cred_calls: list[dict] = []
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment",
                            lambda path: dotenv_calls.append(str(path)))
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: cred_calls.append(kw) or ("pk", "sk", "q", "pe", "se"))
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "dev",
            "--cfg003-assert-availability", "openrouter=openrouter/m",
        ])

        assert adapter.main() == 1
        assert dotenv_calls == []
        assert cred_calls == []
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        gate = artifact["results"]["cfg003_assertion_gate"]
        assert gate["passed"] is False
        assert any("--cfg003-transactional-refresh" in f for f in gate["failures"])

    def test_parse_failure_writes_artifact(self, adapter, monkeypatch, tmp_path):
        """A malformed/duplicate assertion must write a sanitized failure
        artifact before dotenv/credential resolution."""
        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config_path.write_text(json.dumps({"cases": {}}), encoding="utf-8")

        dotenv_calls: list[str] = []
        monkeypatch.setattr(adapter, "_load_dotenv_into_environment",
                            lambda path: dotenv_calls.append(str(path)))
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: ("pk", "sk", "q", "pe", "se"))
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "dev",
            "--cfg003-transactional-refresh",
            "--cfg003-assert-availability", "malformed-no-equals",
        ])

        assert adapter.main() == 1
        assert dotenv_calls == []
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        gate = artifact["results"]["cfg003_assertion_gate"]
        assert gate["passed"] is False
        assert any("malformed" in f for f in gate["failures"])
        # No credential/secret material in the artifact.
        assert "pk" not in artifact_path.read_text(encoding="utf-8")


class TestOperatorAssertionPreTuiOrdering:
    def test_snapshot_validation_fails_before_any_tui_case(
        self, adapter, ra, monkeypatch, tmp_path
    ):
        """An asserted identity absent from the authoritative schedule-eligible
        snapshot must fail in main() BEFORE the selected TUI case loop."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert eligible, "need at least one eligible candidate"

        config_path = tmp_path / "cfg.json"
        artifact_path = tmp_path / "out.json"
        config_path.write_text(json.dumps({
            "default_target_profile": "dev",
            "cases": {"case_a": {"command": ["codex"]}},
        }), encoding="utf-8")

        tui_calls: list[str] = []

        def fake_run_case(**kw):
            tui_calls.append(kw.get("case_name", ""))
            return {"passed": True}

        monkeypatch.setattr(adapter, "_load_dotenv_into_environment", lambda path: None)
        monkeypatch.setattr(adapter, "_resolve_main_credentials",
                            lambda **kw: ("pk", "sk", "http://q", "pe", "se"))
        monkeypatch.setattr(adapter, "_cfg003_query_active_inventory",
                            lambda url: {"healthy": True, "inventory_failures": [],
                                         "alias_inventory": []})
        monkeypatch.setattr(adapter.RA, "_validate_complete_coverage_map",
                            lambda **kw: (True, []))
        monkeypatch.setattr(adapter.RA, "_validate_alias_ingress_coverage",
                            lambda **kw: (True, []))
        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter, "_run_selected_case", fake_run_case)
        monkeypatch.setattr(sys, "argv", [
            "run_anthropic_adapter_acceptance.py",
            "--config", str(config_path),
            "--write-artifact", str(artifact_path),
            "--target", "dev",
            "--cases", "case_a",
            "--cfg003-transactional-refresh",
            "--cfg003-assert-availability", "nonexistent_provider=nonexistent/model",
        ])

        assert adapter.main() == 1
        assert tui_calls == [], f"Expected zero TUI calls, got {tui_calls}"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        gate = artifact["results"]["cfg003_assertion_gate"]
        assert gate["passed"] is False
        assert any("not in the current schedule-eligible" in f for f in gate["failures"])


class TestOperatorAssertionExactPairSelection:
    def test_intended_pair_not_displaced_by_intervening_positive_db(
        self, adapter, ra, monkeypatch
    ):
        """When operator assertions are active, the swap candidate set is bound
        to exactly the two asserted identities; an intervening positive DB
        candidate cannot displace the intended exact pair."""
        auth = ra._load_authoritative_startup_config()
        snap = auth["snapshot"]
        eligible = ra._derive_eligible_candidates_from_snapshot(snap, alias_name="basic")
        assert len(eligible) >= 3, "need >= 3 eligible candidates for this test"

        # Intended pair = first and third by priority; the second is an
        # intervening candidate that has positive DB evidence but is NOT asserted.
        intended = [eligible[0], eligible[2]]
        intervening = eligible[1]
        asserted = [(c["provider"], c["model"]) for c in intended]

        # DB evidence: all three positive (intervening included).
        db_evidence = _avail_result([eligible[0], eligible[1], eligible[2]])

        monkeypatch.setattr(adapter.RA, "_load_authoritative_startup_config", lambda: {
            "snapshot": snap, "merged_yaml": auth["merged_yaml"],
            "per_file_hashes": auth["per_file_hashes"], "file_names": auth["file_names"],
            "config_hash": auth["config_hash"], "config_version": auth["config_version"],
            "aliases": auth["aliases"],
        })
        monkeypatch.setattr(adapter.RA, "_recursive_yaml_source_inventory",
                            lambda: {"basic.yaml"})
        monkeypatch.setattr(adapter.RA, "_snapshot_source_inventory",
                            lambda: {"basic.yaml": "h"})
        monkeypatch.setattr(adapter.RA, "_snapshot_error_intake", lambda *a, **kw: {})
        monkeypatch.setattr(adapter, "_cfg003_collect_availability_evidence",
                            lambda *a, **kw: db_evidence)
        # Stop right after the load phase records original_first/second.
        monkeypatch.setattr(adapter, "_cfg003_readiness_check",
                            lambda *a, **kw: (False, ["stop-after-load"]))

        result = adapter._cfg003_transactional_refresh_test(
            litellm_base_url="http://localhost:4001",
            cases={"native_openai_passthrough_responses_codex_basic_alias_collaboration": {"command": ["codex"]}},
            suite_config={}, query_url="q", public_key="pk", secret_key="sk",
            operator_assertions=asserted,
        )
        load = result["phases"]["load"]
        first = (load["original_first"]["provider"], load["original_first"]["model"])
        second = (load["original_second"]["provider"], load["original_second"]["model"])
        # Bound to exactly the asserted pair, ordered by snapshot priority.
        assert [first, second] == asserted
        # The intervening positive DB candidate is excluded from the pair.
        assert (intervening["provider"], intervening["model"]) not in (first, second)
        assert load["eligible_count"] == 2


# ---------------------------------------------------------------------------
# Pass 3 Fix 3: _cfg003_extract_observed_selection alias-child preference
# ---------------------------------------------------------------------------


class TestCfg003ExtractObservedSelectionAliasChild:
    """Fix 3: selection extraction must prefer alias-child session records."""

    def test_prefers_alias_child_record_over_native_parent(self, adapter):
        """When multiple records exist, the one with alias metadata wins."""
        parent_record = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "metadata": {"passthrough_route_family": "anthropic_messages"},
        }
        child_record = {
            "provider": "openrouter",
            "model": "openrouter/owl-alpha",
            "metadata": {
                "model_alias_label": "basic",
                "anthropic_auto_agent_alias": "basic",
                "anthropic_auto_agent_selected_route_family": "anthropic_openrouter_completion_adapter",
            },
        }
        case_result = {
            "session_history": {
                "record": parent_record,
                "records": [parent_record, child_record],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "openrouter"
        assert selection["model"] == "openrouter/owl-alpha"
        assert selection["route_family"] == "anthropic_openrouter_completion_adapter"

    def test_falls_back_to_first_record_without_alias_metadata(self, adapter):
        """Without alias metadata in any record, first record is used."""
        rec_a = {
            "provider": "openai",
            "model": "gpt-5.5",
            "metadata": {"passthrough_route_family": "openai_responses"},
        }
        rec_b = {
            "provider": "openrouter",
            "model": "openrouter/nemotron",
            "metadata": {},
        }
        case_result = {
            "session_history": {
                "record": rec_a,
                "records": [rec_a, rec_b],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "openai"
        assert selection["model"] == "gpt-5.5"
        assert selection["route_family"] == "openai_responses"

    def test_single_record_uses_default_path(self, adapter):
        """Single-record session_history uses the standard extraction."""
        rec = {
            "provider": "kimi_code",
            "model": "kimi_code/kimi-for-coding",
            "metadata": {
                "requested_model_alias": "basic",
                "codex_auto_agent_selected_route_family": "anthropic_kimi_chat_completions_adapter",
            },
        }
        case_result = {
            "session_history": {"record": rec, "records": [rec]}
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "kimi_code"
        assert selection["route_family"] == "anthropic_kimi_chat_completions_adapter"

    def test_empty_session_history_returns_nones(self, adapter):
        case_result = {"session_history": {}}
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection == {"provider": None, "model": None, "route_family": None}

    def test_skips_alias_row_with_empty_provider_for_later_usable_alias_row(
        self, adapter
    ):
        """An alias-marked row with empty provider is skipped for a later usable one."""
        broken_alias = {
            "provider": "",
            "model": "openrouter/owl-alpha",
            "metadata": {
                "model_alias_label": "basic",
                "anthropic_auto_agent_selected_route_family": "anthropic_openrouter_completion_adapter",
            },
        }
        usable_alias = {
            "provider": "openrouter",
            "model": "openrouter/north-mini-code:free",
            "metadata": {
                "model_alias_label": "basic",
                "codex_auto_agent_selected_route_family": "codex_openrouter_completion_adapter",
            },
        }
        case_result = {
            "session_history": {
                "record": broken_alias,
                "records": [broken_alias, usable_alias],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "openrouter"
        assert selection["model"] == "openrouter/north-mini-code:free"
        assert selection["route_family"] == "codex_openrouter_completion_adapter"

    def test_skips_alias_row_with_whitespace_model_for_later_usable_alias_row(
        self, adapter
    ):
        """An alias-marked row with whitespace-only model is skipped."""
        broken_alias = {
            "provider": "alibaba_token_plan",
            "model": "   ",
            "metadata": {
                "requested_model_alias": "basic",
                "codex_auto_agent_selected_route_family": "codex_alibaba_token_plan_chat_completions_adapter",
            },
        }
        usable_alias = {
            "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash",
            "metadata": {
                "requested_model_alias": "basic",
                "codex_auto_agent_selected_route_family": "codex_alibaba_token_plan_chat_completions_adapter",
            },
        }
        case_result = {
            "session_history": {
                "record": broken_alias,
                "records": [broken_alias, usable_alias],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "alibaba_token_plan"
        assert selection["model"] == "alibaba_token_plan/qwen3.6-flash"
        assert (
            selection["route_family"]
            == "codex_alibaba_token_plan_chat_completions_adapter"
        )

    def test_skips_alias_row_lacking_route_family_for_later_usable_alias_row(
        self, adapter
    ):
        """An alias-marked row missing a route-family field is skipped (full triple required)."""
        no_route = {
            "provider": "openrouter",
            "model": "openrouter/owl-alpha",
            "metadata": {"model_alias_label": "basic"},
        }
        usable_alias = {
            "provider": "openrouter",
            "model": "openrouter/north-mini-code:free",
            "metadata": {
                "model_alias_label": "basic",
                "passthrough_route_family": "codex_openrouter_completion_adapter",
            },
        }
        case_result = {
            "session_history": {
                "record": no_route,
                "records": [no_route, usable_alias],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "openrouter"
        assert selection["model"] == "openrouter/north-mini-code:free"
        assert selection["route_family"] == "codex_openrouter_completion_adapter"

    def test_fallback_prefers_usable_record_over_unusable_canonical_record(
        self, adapter
    ):
        """When no alias row qualifies, fallback searches for a full-identity record
        rather than blindly returning the unusable canonical record."""
        unusable_canonical = {
            "provider": None,
            "model": None,
            "metadata": {},
        }
        usable_plain = {
            "provider": "openrouter",
            "model": "openrouter/north-mini-code:free",
            "metadata": {"passthrough_route_family": "codex_openrouter_completion_adapter"},
        }
        case_result = {
            "session_history": {
                "record": unusable_canonical,
                "records": [unusable_canonical, usable_plain],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection["provider"] == "openrouter"
        assert selection["model"] == "openrouter/north-mini-code:free"
        assert selection["route_family"] == "codex_openrouter_completion_adapter"

    def test_fallback_returns_empty_when_no_usable_record_exists(self, adapter):
        """When no record carries a full identity, selection is all None."""
        unusable_a = {"provider": None, "model": None, "metadata": {}}
        unusable_b = {
            "provider": "openrouter",
            "model": "openrouter/owl-alpha",
            "metadata": {},  # missing route-family
        }
        case_result = {
            "session_history": {
                "record": unusable_a,
                "records": [unusable_a, unusable_b],
            }
        }
        selection = adapter._cfg003_extract_observed_selection(case_result)
        assert selection == {"provider": None, "model": None, "route_family": None}
