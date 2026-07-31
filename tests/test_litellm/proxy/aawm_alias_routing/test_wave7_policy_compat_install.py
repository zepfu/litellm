"""D1-591: policy compatibility publication API tests.

Pins the full alias inventory, same-object identity, idempotence,
no god-module import, and no Google Code Assist / Antigravity reintroduction.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy

# ---------------------------------------------------------------------------
# Full expected inventory: local_name -> policy public constant name.
# Must match llm_passthrough_endpoints.py lines ~405-470 exactly.
# ---------------------------------------------------------------------------
EXPECTED_INVENTORY: dict[str, str] = {
    "_CODEX_AUTO_AGENT_MODEL_ALIAS": "CODEX_AUTO_AGENT_MODEL_ALIAS",
    "_CODEX_AUTO_AGENT_NATIVE_PROVIDER": "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "_CODEX_AUTO_AGENT_XAI_PROVIDER": "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY": "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_LANE_KEY": "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY": "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY",
    "_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY": "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY": "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY",
    "_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY": "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_CANDIDATES": "CODEX_AUTO_AGENT_CANDIDATES",
    "_CODEX_AAWM_READ_ALIAS": "CODEX_AAWM_READ_ALIAS",
    "_CODEX_AAWM_SOTA_ALIAS": "CODEX_AAWM_SOTA_ALIAS",
    "_CODEX_AAWM_CODE_ALIAS": "CODEX_AAWM_CODE_ALIAS",
    "_CODEX_AAWM_LOW_ALIAS": "CODEX_AAWM_LOW_ALIAS",
    "_CODEX_AAWM_ORCHESTRATION_ALIAS": "CODEX_AAWM_ORCHESTRATION_ALIAS",
    "_CODEX_AAWM_SOTA_CANDIDATES": "CODEX_AAWM_SOTA_CANDIDATES",
    "_CODEX_AAWM_SOTA_OPENAI_ALIAS": "CODEX_AAWM_SOTA_OPENAI_ALIAS",
    "_CODEX_AAWM_SOTA_XAI_ALIAS": "CODEX_AAWM_SOTA_XAI_ALIAS",
    "_CODEX_AAWM_SOTA_MOONSHOT_ALIAS": "CODEX_AAWM_SOTA_MOONSHOT_ALIAS",
    "_CODEX_AAWM_SOTA_ALIBABA_ALIAS": "CODEX_AAWM_SOTA_ALIBABA_ALIAS",
    "_CODEX_AAWM_SOTA_DEEPSEEK_ALIAS": "CODEX_AAWM_SOTA_DEEPSEEK_ALIAS",
    "_CODEX_AAWM_SOTA_GLM_ALIAS": "CODEX_AAWM_SOTA_GLM_ALIAS",
    "_CODEX_AAWM_SOTA_OPENAI_CANDIDATES": "CODEX_AAWM_SOTA_OPENAI_CANDIDATES",
    "_CODEX_AAWM_SOTA_XAI_CANDIDATES": "CODEX_AAWM_SOTA_XAI_CANDIDATES",
    "_CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES": "CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES",
    "_CODEX_AAWM_SOTA_ALIBABA_CANDIDATES": "CODEX_AAWM_SOTA_ALIBABA_CANDIDATES",
    "_CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES": "CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES",
    "_CODEX_AAWM_SOTA_GLM_CANDIDATES": "CODEX_AAWM_SOTA_GLM_CANDIDATES",
    "_CODEX_AAWM_CODE_CANDIDATES": "CODEX_AAWM_CODE_CANDIDATES",
    "_CODEX_AAWM_LOW_CANDIDATES": "CODEX_AAWM_LOW_CANDIDATES",
    "_CODEX_AAWM_ORCHESTRATION_CANDIDATES": "CODEX_AAWM_ORCHESTRATION_CANDIDATES",
    "_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS": "CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS",
    "_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS": "ANTHROPIC_AUTO_AGENT_MODEL_ALIAS",
    "_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER": "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL": "ANTHROPIC_AUTO_AGENT_HAIKU_MODEL",
    "_ANTHROPIC_AUTO_AGENT_CANDIDATES": "ANTHROPIC_AUTO_AGENT_CANDIDATES",
    "_ANTHROPIC_AAWM_READ_ALIAS": "ANTHROPIC_AAWM_READ_ALIAS",
    "_ANTHROPIC_AAWM_SOTA_ALIAS": "ANTHROPIC_AAWM_SOTA_ALIAS",
    "_ANTHROPIC_AAWM_CODE_ALIAS": "ANTHROPIC_AAWM_CODE_ALIAS",
    "_ANTHROPIC_AAWM_LOW_ALIAS": "ANTHROPIC_AAWM_LOW_ALIAS",
    "_ANTHROPIC_AAWM_ORCHESTRATION_ALIAS": "ANTHROPIC_AAWM_ORCHESTRATION_ALIAS",
    "_ANTHROPIC_AAWM_SOTA_CANDIDATES": "ANTHROPIC_AAWM_SOTA_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES": "ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES": "ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES": "ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES": "ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES",
    "_ANTHROPIC_AAWM_CODE_CANDIDATES": "ANTHROPIC_AAWM_CODE_CANDIDATES",
    "_ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES": "ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES",
    "_ANTHROPIC_AAWM_LOW_CANDIDATES": "ANTHROPIC_AAWM_LOW_CANDIDATES",
    "_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS": "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS",
    "_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS",
    "_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS": "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS",
    "_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS": "ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS",
}

GOD_MODULE = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"


class TestCompatAliasInventory:
    """Pin the full alias inventory."""

    def test_map_matches_expected_inventory(self) -> None:
        assert policy.COMPAT_ALIAS_MAP == EXPECTED_INVENTORY

    def test_count_is_65(self) -> None:
        assert policy.COMPAT_ALIAS_COUNT == 65
        assert len(policy.COMPAT_ALIAS_MAP) == 65

    def test_all_policy_targets_exist(self) -> None:
        """Every policy constant referenced by the map must exist."""
        ns = vars(policy)
        missing = [
            pub for pub in policy.COMPAT_ALIAS_MAP.values() if pub not in ns
        ]
        assert missing == [], f"Missing policy constants: {missing}"

    def test_deterministic_order(self) -> None:
        """Map iteration order is stable across re-imports."""
        # Snapshot the original module namespace so we can restore exact
        # object identities after the reload (avoids polluting downstream
        # tests that rely on `is` identity with previously-imported refs).
        original_dict = dict(vars(policy))
        fresh = importlib.reload(policy)
        try:
            assert list(fresh.COMPAT_ALIAS_MAP.keys()) == list(
                EXPECTED_INVENTORY.keys()
            )
        finally:
            # Restore original objects in-place to preserve identity for
            # any module that previously imported from policy.
            vars(policy).clear()
            vars(policy).update(original_dict)


class TestInstallSameObjectIdentity:
    """Installed values must be the exact same objects, not copies."""

    def test_identity_for_all_entries(self) -> None:
        host: dict[str, Any] = {}
        policy.install_policy_compat_aliases(host)
        for local_name, policy_name in policy.COMPAT_ALIAS_MAP.items():
            assert host[local_name] is getattr(policy, policy_name), (
                f"{local_name} is not the same object as policy.{policy_name}"
            )

    def test_candidate_table_identity(self) -> None:
        """Candidate tuples must share object identity (not equal copies)."""
        host: dict[str, Any] = {}
        policy.install_policy_compat_aliases(host)
        assert (
            host["_CODEX_AUTO_AGENT_CANDIDATES"]
            is policy.CODEX_AUTO_AGENT_CANDIDATES
        )
        assert (
            host["_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS"]
            is policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
        )
        assert (
            host["_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS"]
            is policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
        )


class TestIdempotence:
    """Calling install twice must produce identical results."""

    def test_double_install(self) -> None:
        host: dict[str, Any] = {}
        policy.install_policy_compat_aliases(host)
        snapshot = dict(host)
        policy.install_policy_compat_aliases(host)
        assert host == snapshot
        for key in policy.COMPAT_ALIAS_MAP:
            assert host[key] is snapshot[key]

    def test_no_extra_keys(self) -> None:
        host: dict[str, Any] = {"pre_existing": 42}
        policy.install_policy_compat_aliases(host)
        expected_keys = set(policy.COMPAT_ALIAS_MAP.keys()) | {"pre_existing"}
        assert set(host.keys()) == expected_keys


class TestInvalidHostMapping:
    """Fail clearly on invalid host mapping."""

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="host_globals must be a dict"):
            policy.install_policy_compat_aliases(None)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        with pytest.raises(TypeError, match="host_globals must be a dict"):
            policy.install_policy_compat_aliases([])  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        with pytest.raises(TypeError, match="host_globals must be a dict"):
            policy.install_policy_compat_aliases("globals")  # type: ignore[arg-type]


class TestNoGodModuleImport:
    """install_policy_compat_aliases must not import the god module."""

    def test_no_god_module_import(self) -> None:
        # Remove god module if cached so we can detect import.
        saved = sys.modules.pop(GOD_MODULE, None)
        try:
            host: dict[str, Any] = {}
            policy.install_policy_compat_aliases(host)
            assert GOD_MODULE not in sys.modules, (
                "install_policy_compat_aliases imported the god module"
            )
        finally:
            if saved is not None:
                sys.modules[GOD_MODULE] = saved


class TestNoGoogleAntigravityReintroduction:
    """Policy module must not reintroduce Google Code Assist or Antigravity."""

    _BANNED_SUBSTRINGS = (
        "google_code_assist",
        "antigravity",
        "GOOGLE_PROVIDER",
        "ANTIGRAVITY_PROVIDER",
        "GOOGLE_COMPLETION_ADAPTER",
        "CODE_ASSIST_ADAPTER",
    )

    def test_no_banned_names_in_policy_namespace(self) -> None:
        ns = vars(policy)
        violations = [
            name
            for name in ns
            if any(sub in name.upper() for sub in
                   ("GOOGLE_CODE_ASSIST", "ANTIGRAVITY"))
        ]
        assert violations == [], f"Banned names in policy: {violations}"

    def test_no_banned_names_in_compat_map(self) -> None:
        all_names = list(policy.COMPAT_ALIAS_MAP.keys()) + list(
            policy.COMPAT_ALIAS_MAP.values()
        )
        violations = [
            n
            for n in all_names
            if any(sub.upper() in n.upper() for sub in self._BANNED_SUBSTRINGS)
        ]
        assert violations == [], f"Banned names in compat map: {violations}"

    def test_no_banned_values_in_candidate_tables(self) -> None:
        """Candidate table provider strings must not reference banned providers."""
        tables = [
            policy.CODEX_AUTO_AGENT_CANDIDATES,
            policy.ANTHROPIC_AUTO_AGENT_CANDIDATES,
        ]
        for table in tables:
            for entry in table:
                provider = entry.get("provider", "")
                assert "google" not in provider.lower(), (
                    f"Google provider in candidate: {entry}"
                )
                assert "antigravity" not in provider.lower(), (
                    f"Antigravity provider in candidate: {entry}"
                )
