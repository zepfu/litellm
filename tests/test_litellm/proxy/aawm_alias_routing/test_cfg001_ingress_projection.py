"""CFG-001: ingress-specific route-family projection tests.

Verifies that the logical config alias ``read`` resolves from the same active
snapshot on both Codex/OpenAI Responses and Claude Code/Anthropic Messages
ingress, preserving provider-native credential boundaries.

No provider egress, no synthetic LLM calls.
"""

from __future__ import annotations

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    ConfigCompileError,
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    AMBIGUOUS_CODEX_ROUTE_FAMILIES,
    CODEX_TO_ANTHROPIC_ROUTE_FAMILY,
    resolve_anthropic_route_family,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingSnapshot,
    active_routing_snapshot_holder,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    _routing_candidate_to_anthropic_public_dict,
    _select_read_pilot_snapshot_candidates_anthropic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_READ_YAML = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 40
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
        route_family: codex_openrouter_completion_adapter
        priority: 80
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        anthropic_route_family: anthropic_opencode_zen_responses_adapter
        priority: 60
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""

_SNAPSHOT: RoutingSnapshot | None = None


@pytest.fixture()
def read_snapshot() -> RoutingSnapshot:
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = compile_yaml(_MINIMAL_READ_YAML)
    return _SNAPSHOT


@pytest.fixture(autouse=True)
def _swap_snapshot(read_snapshot: RoutingSnapshot):
    previous = active_routing_snapshot_holder.swap(read_snapshot)
    yield
    active_routing_snapshot_holder.swap(previous)


# ---------------------------------------------------------------------------
# Schema: closed projection mapping
# ---------------------------------------------------------------------------


class TestClosedProjectionMapping:
    def test_all_mapped_families_are_registered(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
            REGISTERED_ROUTE_FAMILIES,
        )

        for codex_rf, anthropic_rf in CODEX_TO_ANTHROPIC_ROUTE_FAMILY.items():
            assert codex_rf in REGISTERED_ROUTE_FAMILIES, f"codex {codex_rf} not registered"
            assert anthropic_rf in REGISTERED_ROUTE_FAMILIES, f"anthropic {anthropic_rf} not registered"

    def test_ambiguous_families_not_in_closed_mapping(self):
        for ambiguous in AMBIGUOUS_CODEX_ROUTE_FAMILIES:
            assert ambiguous not in CODEX_TO_ANTHROPIC_ROUTE_FAMILY

    def test_resolve_explicit_override_wins(self):
        assert (
            resolve_anthropic_route_family("codex_responses", "anthropic_messages")
            == "anthropic_messages"
        )

    def test_resolve_closed_mapping(self):
        assert (
            resolve_anthropic_route_family("codex_responses", None)
            == "anthropic_openai_responses_adapter"
        )

    def test_resolve_ambiguous_returns_none(self):
        assert resolve_anthropic_route_family("codex_opencode_zen_adapter", None) is None

    def test_resolve_none_returns_none(self):
        assert resolve_anthropic_route_family(None, None) is None


# ---------------------------------------------------------------------------
# Compiler: fail-closed on missing anthropic route family
# ---------------------------------------------------------------------------


class TestCompilerFailClosed:
    def test_ambiguous_without_override_compiles_with_none_anthropic_rf(self):
        """Ambiguous route families compile with anthropic_route_family=None.

        Backward compatibility: existing configs with codex_opencode_zen_adapter
        and no explicit override must not be rejected at compile time.  The
        fail-closed safety moves to dispatch shaping (_routing_candidate_to_anthropic_public_dict).
        """
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        priority: 60
"""
        snapshot = compile_yaml(yaml_str)
        candidate = snapshot.aliases["read"].candidates[0]
        assert candidate.anthropic_route_family is None
        # Dispatch shaping fails closed:
        with pytest.raises(ValueError, match="no anthropic_route_family"):
            _routing_candidate_to_anthropic_public_dict(candidate)

    def test_ambiguous_with_override_compiles(self):
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        anthropic_route_family: anthropic_opencode_zen_responses_adapter
        priority: 60
"""
        snapshot = compile_yaml(yaml_str)
        candidate = snapshot.aliases["read"].candidates[0]
        assert candidate.anthropic_route_family == "anthropic_opencode_zen_responses_adapter"

    def test_unregistered_anthropic_route_family_rejected(self):
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        anthropic_route_family: not_a_real_family
        priority: 40
"""
        with pytest.raises(Exception, match="not a registered code behavior"):
            compile_yaml(yaml_str)


# ---------------------------------------------------------------------------
# Snapshot: anthropic_route_family populated
# ---------------------------------------------------------------------------


class TestSnapshotAnthropicRouteFamily:
    def test_closed_mapping_candidates_get_anthropic_rf(self, read_snapshot: RoutingSnapshot):
        alias = read_snapshot.aliases["read"]
        by_model = {c.model: c for c in alias.candidates}
        assert by_model["gpt-5.6-luna"].anthropic_route_family == "anthropic_openai_responses_adapter"
        assert (
            by_model["openrouter/cohere/north-mini-code:free"].anthropic_route_family
            == "anthropic_openrouter_completion_adapter"
        )

    def test_explicit_override_preserved(self, read_snapshot: RoutingSnapshot):
        alias = read_snapshot.aliases["read"]
        by_model = {c.model: c for c in alias.candidates}
        assert by_model["deepseek-v4-flash"].anthropic_route_family == "anthropic_opencode_zen_responses_adapter"

    def test_last_resort_gets_anthropic_rf(self, read_snapshot: RoutingSnapshot):
        alias = read_snapshot.aliases["read"]
        by_model = {c.model: c for c in alias.candidates}
        assert by_model["gpt-5.4-mini"].anthropic_route_family == "anthropic_openai_responses_adapter"


# ---------------------------------------------------------------------------
# Anthropic public dict shaping
# ---------------------------------------------------------------------------


class TestAnthropicPublicDict:
    def test_uses_anthropic_route_family(self, read_snapshot: RoutingSnapshot):
        candidate = read_snapshot.aliases["read"].candidates[0]
        shaped = _routing_candidate_to_anthropic_public_dict(candidate, epoch_tag="abc123")
        assert shaped["route_family"] == candidate.anthropic_route_family
        assert shaped["route_family"] != candidate.route_family
        assert shaped["config_epoch_tag"] == "abc123"

    def test_missing_anthropic_rf_raises(self, read_snapshot: RoutingSnapshot):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
            RoutingCandidate,
        )

        bare = RoutingCandidate(
            provider="openai",
            model="test-model",
            route_family="codex_responses",
            priority=10,
            weight=1.0,
            tui_attached=None,
            schedule=None,
            error_rules=(),
            anthropic_route_family=None,
        )
        with pytest.raises(ValueError, match="no anthropic_route_family"):
            _routing_candidate_to_anthropic_public_dict(bare)


# ---------------------------------------------------------------------------
# Anthropic ingress snapshot resolution (no egress)
# ---------------------------------------------------------------------------


class TestAnthropicIngressResolution:
    def test_returns_anthropic_projected_candidates(self):
        result = _select_read_pilot_snapshot_candidates_anthropic()
        assert result is not None
        assert len(result) > 0
        for candidate in result:
            rf = candidate["route_family"]
            assert rf.startswith("anthropic_"), f"expected anthropic route family, got {rf}"

    def test_same_provider_model_set_as_codex(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _select_read_pilot_snapshot_candidates,
        )

        codex = _select_read_pilot_snapshot_candidates()
        anthropic = _select_read_pilot_snapshot_candidates_anthropic()
        assert anthropic is not None
        codex_ids = {(c["provider"], c["model"]) for c in codex}
        anthropic_ids = {(c["provider"], c["model"]) for c in anthropic}
        assert codex_ids == anthropic_ids

    def test_route_families_differ_by_ingress(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _select_read_pilot_snapshot_candidates,
        )

        codex = _select_read_pilot_snapshot_candidates()
        anthropic = _select_read_pilot_snapshot_candidates_anthropic()
        assert anthropic is not None
        codex_by_model = {c["model"]: c for c in codex}
        anthropic_by_model = {c["model"]: c for c in anthropic}
        for model in codex_by_model:
            assert codex_by_model[model]["route_family"] != anthropic_by_model[model]["route_family"]

    def test_no_snapshot_returns_none(self):
        previous = active_routing_snapshot_holder.swap(None)
        try:
            result = _select_read_pilot_snapshot_candidates_anthropic()
            assert result is None
        finally:
            active_routing_snapshot_holder.swap(previous)

    def test_epoch_tag_carried(self):
        result = _select_read_pilot_snapshot_candidates_anthropic()
        assert result is not None
        for candidate in result:
            assert "config_epoch_tag" in candidate

    def test_priority_ordering_preserved(self):
        result = _select_read_pilot_snapshot_candidates_anthropic()
        assert result is not None
        # Last resort (priority 0) must be last
        assert result[-1]["last_resort"] is True
        for c in result[:-1]:
            assert c["last_resort"] is False


# ---------------------------------------------------------------------------
# Legacy aliases unchanged
# ---------------------------------------------------------------------------


class TestLegacyAliasesUnchanged:
    def test_aawm_read_not_in_snapshot(self, read_snapshot: RoutingSnapshot):
        assert "aawm-read" not in read_snapshot.aliases

    def test_aawm_read_anthropic_not_in_snapshot(self, read_snapshot: RoutingSnapshot):
        assert "aawm-read-anthropic" not in read_snapshot.aliases

    def test_config_hash_stable(self, read_snapshot: RoutingSnapshot):
        snapshot2 = compile_yaml(_MINIMAL_READ_YAML)
        assert read_snapshot.config_hash == snapshot2.config_hash


# ---------------------------------------------------------------------------
# Ingress isolation: no cross-provider fallback
# ---------------------------------------------------------------------------


class TestIngressIsolation:
    def test_anthropic_candidates_never_use_codex_route_families(self):
        result = _select_read_pilot_snapshot_candidates_anthropic()
        assert result is not None
        for candidate in result:
            assert not candidate["route_family"].startswith("codex_")

    def test_codex_candidates_never_use_anthropic_route_families(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _select_read_pilot_snapshot_candidates,
        )

        codex = _select_read_pilot_snapshot_candidates()
        for candidate in codex:
            assert not candidate["route_family"].startswith("anthropic_")


# ---------------------------------------------------------------------------
# Fix 1: install() host globals include snapshot_aware getter
# ---------------------------------------------------------------------------


class TestInstallHostGlobals:
    @pytest.fixture(autouse=True)
    def _restore_selection_identity(self):
        """Restore selection module function identities after install() rebinds them."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _HOST_FUNCTION_NAMES,
        )

        saved = {name: vars(selection)[name] for name in _HOST_FUNCTION_NAMES}
        saved["_attach_aawm_alias_routing_state_sources"] = vars(selection).get(
            "_attach_aawm_alias_routing_state_sources"
        )
        yield
        for name, fn in saved.items():
            if fn is not None:
                vars(selection)[name] = fn

    def test_snapshot_aware_getter_published_to_host_globals(self):
        """install() publishes the snapshot-aware getter to host_globals.

        Uses a separate dict as the host so the selection module namespace
        is not polluted with seam-variable copies.
        """
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection

        host_globals: dict = {}
        selection.install(host_globals)
        assert "_get_anthropic_candidates_for_alias_snapshot_aware" in host_globals
        assert callable(host_globals["_get_anthropic_candidates_for_alias_snapshot_aware"])

    def test_install_publishes_anthropic_affinity_candidate(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection

        host_globals: dict = {}
        selection.install(host_globals)
        assert "_find_anthropic_auto_agent_affinity_candidate" in host_globals

    def test_install_publishes_anthropic_public_dict_shaper(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection

        host_globals: dict = {}
        selection.install(host_globals)
        assert "_routing_candidate_to_anthropic_public_dict" in host_globals


# ---------------------------------------------------------------------------
# Fix 2: Compile-time credential-domain compatibility negatives
# ---------------------------------------------------------------------------


class TestCredentialDomainCompatibility:
    def test_anthropic_provider_with_codex_responses_rejected(self):
        """Anthropic-native provider + codex_responses route = TOS violation."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: anthropic
        model: claude-4-sonnet
        route_family: codex_responses
        priority: 40
"""
        with pytest.raises(ConfigCompileError, match="incompatible"):
            compile_yaml(yaml_str)

    def test_openai_provider_with_anthropic_messages_rejected(self):
        """OpenAI provider + anthropic_messages route = credential mismatch."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        anthropic_route_family: anthropic_messages
        priority: 40
"""
        with pytest.raises(ConfigCompileError, match="incompatible"):
            compile_yaml(yaml_str)

    def test_compatible_provider_route_compiles(self):
        """OpenAI provider + codex_responses + anthropic_openai_responses_adapter is fine."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 40
"""
        snapshot = compile_yaml(yaml_str)
        candidate = snapshot.aliases["read"].candidates[0]
        assert candidate.anthropic_route_family == "anthropic_openai_responses_adapter"


# ---------------------------------------------------------------------------
# Fix 3: Anthropic affinity bypasses eligibility gates
# ---------------------------------------------------------------------------


class TestAnthropicAffinityBypass:
    def test_affinity_finds_schedule_gated_candidate(self, read_snapshot: RoutingSnapshot):
        """Pinned candidate outside schedule window is still found via affinity."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _find_anthropic_auto_agent_affinity_candidate,
        )

        # Use the first candidate from the snapshot
        candidate = read_snapshot.aliases["read"].candidates[0]
        affinity = {
            "provider": candidate.provider,
            "model": candidate.model,
            "config_hash": read_snapshot.config_hash,
            "route_family": candidate.anthropic_route_family,
        }
        result = _find_anthropic_auto_agent_affinity_candidate(
            affinity,
            alias_model="read",
            client_product_label=None,
        )
        assert result is not None
        assert result["provider"] == candidate.provider
        assert result["model"] == candidate.model

    def test_affinity_returns_none_for_removed_candidate(self, read_snapshot: RoutingSnapshot):
        """Candidate removed from snapshot yields None (redispatch required)."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _find_anthropic_auto_agent_affinity_candidate,
        )

        affinity = {
            "provider": "nonexistent",
            "model": "nonexistent-model",
            "config_hash": read_snapshot.config_hash,
            "route_family": "anthropic_openai_responses_adapter",
        }
        result = _find_anthropic_auto_agent_affinity_candidate(
            affinity,
            alias_model="read",
            client_product_label=None,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Fix 4: Non-read aliases fail closed when snapshot active
# ---------------------------------------------------------------------------


class TestNonReadAliasFailClosed:
    def test_arbitrary_alias_fails_closed_with_snapshot(self, read_snapshot: RoutingSnapshot):
        """Arbitrary alias 'other' gets empty candidates when snapshot is active."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _get_codex_auto_agent_candidates_for_alias,
        )

        result = _get_codex_auto_agent_candidates_for_alias("other")
        assert result == ()

    def test_known_legacy_alias_still_resolves_with_snapshot(self, read_snapshot: RoutingSnapshot):
        """Known legacy aliases (aawm-low) still resolve from static tables."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _get_codex_auto_agent_candidates_for_alias,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
            CODEX_AAWM_LOW_ALIAS,
        )

        result = _get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_LOW_ALIAS)
        assert len(result) > 0

    def test_arbitrary_alias_resolves_without_snapshot(self):
        """Without a snapshot, arbitrary aliases fall back to static table."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _get_codex_auto_agent_candidates_for_alias,
        )

        previous = active_routing_snapshot_holder.swap(None)
        try:
            result = _get_codex_auto_agent_candidates_for_alias("other")
            assert len(result) > 0  # Falls back to generic static table
        finally:
            active_routing_snapshot_holder.swap(previous)


# ---------------------------------------------------------------------------
# Fix 5: client_product_label wired through Anthropic path
# ---------------------------------------------------------------------------


class TestClientProductLabelWiring:
    def test_tui_attached_candidate_excluded_without_label(self):
        """TUI-attached candidate is excluded when no client_product_label."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        tui_attached: claude
        priority: 40
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
        route_family: codex_openrouter_completion_adapter
        priority: 30
"""
        snapshot = compile_yaml(yaml_str)
        previous = active_routing_snapshot_holder.swap(snapshot)
        try:
            result = _select_read_pilot_snapshot_candidates_anthropic(
                client_product_label=None,
            )
            assert result is not None
            models = [c["model"] for c in result]
            assert "gpt-5.6-luna" not in models
            assert "openrouter/cohere/north-mini-code:free" in models
        finally:
            active_routing_snapshot_holder.swap(previous)

    def test_tui_attached_candidate_included_with_matching_label(self):
        """TUI-attached candidate is included when client_product_label matches."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        tui_attached: claude
        priority: 40
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
        route_family: codex_openrouter_completion_adapter
        priority: 30
"""
        snapshot = compile_yaml(yaml_str)
        previous = active_routing_snapshot_holder.swap(snapshot)
        try:
            result = _select_read_pilot_snapshot_candidates_anthropic(
                client_product_label="claude/1.0",
            )
            assert result is not None
            models = [c["model"] for c in result]
            assert "gpt-5.6-luna" in models
        finally:
            active_routing_snapshot_holder.swap(previous)

    def test_snapshot_aware_getter_accepts_client_product_label(self):
        """The snapshot-aware getter accepts and forwards client_product_label."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _get_anthropic_candidates_for_alias_snapshot_aware,
        )

        # Should not raise with the kwarg
        result = _get_anthropic_candidates_for_alias_snapshot_aware(
            "read", client_product_label=None,
        )
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Fix 6: Backward compat for ambiguous OpenCode at compile
# ---------------------------------------------------------------------------


class TestAmbiguousOpenCodeBackwardCompat:
    def test_opencode_zen_without_override_compiles(self):
        """codex_opencode_zen_adapter without override compiles (backward compat)."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        priority: 60
"""
        snapshot = compile_yaml(yaml_str)
        assert snapshot.aliases["read"].candidates[0].anthropic_route_family is None

    def test_opencode_zen_anthropic_dispatch_fails_closed(self):
        """Anthropic dispatch shaping fails closed for ambiguous without override."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        priority: 60
"""
        snapshot = compile_yaml(yaml_str)
        candidate = snapshot.aliases["read"].candidates[0]
        with pytest.raises(ValueError, match="no anthropic_route_family"):
            _routing_candidate_to_anthropic_public_dict(candidate)

    def test_opencode_zen_with_override_compiles_and_dispatches(self):
        """Explicit override compiles and dispatches correctly."""
        yaml_str = """\
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: opencode_zen
        model: deepseek-v4-flash
        route_family: codex_opencode_zen_adapter
        anthropic_route_family: anthropic_opencode_zen_responses_adapter
        priority: 60
"""
        snapshot = compile_yaml(yaml_str)
        candidate = snapshot.aliases["read"].candidates[0]
        shaped = _routing_candidate_to_anthropic_public_dict(candidate)
        assert shaped["route_family"] == "anthropic_opencode_zen_responses_adapter"

    def test_read_yaml_compiles_with_opencode_candidates(self):
        """The production read.yaml with OpenCode candidates compiles."""
        import os

        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "litellm",
            "proxy",
            "aawm_alias_config",
            "read.yaml",
        )
        with open(yaml_path) as f:
            raw = f.read()
        snapshot = compile_yaml(raw)
        assert "read" in snapshot.aliases
        # OpenCode candidates with explicit overrides compile fine
        by_model = {c.model: c for c in snapshot.aliases["read"].candidates}
        assert by_model["deepseek-v4-flash"].anthropic_route_family == "anthropic_opencode_zen_responses_adapter"
        assert by_model["big-pickle"].anthropic_route_family == "anthropic_opencode_zen_completion_adapter"


# ---------------------------------------------------------------------------
# Fix 7: Anthropic affinity setter persists config_hash (production facade)
# ---------------------------------------------------------------------------


class TestAnthropicAffinityProductionFacade:
    """Production-facade tests: real affinity setter + live selector, no manual config_hash."""

    @pytest.fixture()
    def fresh_manager(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import cooldown_state
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            configure_cooldown_state_runtime,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            AliasRoutingStateManager,
        )

        previous_manager = cooldown_state._manager
        mgr = AliasRoutingStateManager()
        configure_cooldown_state_runtime(manager=mgr)
        try:
            yield mgr
        finally:
            cooldown_state._manager = previous_manager

    @pytest.fixture(autouse=True)
    def _no_durable(self):
        from unittest.mock import AsyncMock, patch

        _mod = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state"
        with (
            patch(f"{_mod}.get_aawm_alias_routing_dual_cache", return_value=None),
            patch(
                f"{_mod}.write_aawm_alias_routing_durable_payload",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            yield

    @pytest.fixture()
    def _selection_runtime(self, fresh_manager):
        """Wire real cooldown_state affinity getter into selection runtime."""
        from unittest.mock import AsyncMock

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            _get_anthropic_auto_agent_session_affinity,
        )

        runtime_names = {
            "_get_codex_active_cooldown_state",
            "_get_anthropic_active_cooldown_state",
            "_get_anthropic_merged_codex_openai_cooldown_state",
            "_set_codex_cooldown",
            "_set_anthropic_cooldown",
            "_get_codex_session_affinity",
            "_get_anthropic_session_affinity",
            "_get_openrouter_adapter_active_cooldown_seconds",
            "_normalize_codex_alias_model",
            "_extract_client_product_label",
            "_resolve_codex_session_key",
            "_resolve_anthropic_session_key",
            "_has_continuation_state",
            "_get_anthropic_candidates_for_alias",
            "_is_grok_account_quota_candidate",
            "_get_grok_account_quota_lane_cooldown_key",
            "_is_kimi_code_candidate",
            "_get_kimi_managed_account_cooldown_key",
        }
        previous = {name: getattr(selection, name) for name in runtime_names}

        async def _zero_cooldown_state(key: str) -> tuple:
            return (0.0, "local_fallback")

        async def _noop_cooldown(key: str, seconds: float) -> None:
            pass

        async def _zero_adapter(model) -> float:
            return 0.0

        selection.configure_selection_runtime(
            get_codex_active_cooldown_state=_zero_cooldown_state,
            get_anthropic_active_cooldown_state=_zero_cooldown_state,
            get_anthropic_merged_codex_openai_cooldown_state=_zero_cooldown_state,
            set_codex_cooldown=_noop_cooldown,
            set_anthropic_cooldown=_noop_cooldown,
            get_codex_session_affinity=AsyncMock(return_value=None),
            get_anthropic_session_affinity=_get_anthropic_auto_agent_session_affinity,
            get_openrouter_adapter_active_cooldown_seconds=_zero_adapter,
            normalize_codex_alias_model=lambda m: None,
            extract_client_product_label=lambda r, b: None,
            resolve_codex_session_key=lambda r, b, **kw: None,
            resolve_anthropic_session_key=lambda r, b, **kw: "test-session",
            has_continuation_state=lambda v: True,
            get_anthropic_candidates_for_alias=lambda alias: (),
            is_grok_account_quota_candidate=lambda c: False,
            get_grok_account_quota_lane_cooldown_key=lambda c, lk: None,
            is_kimi_code_candidate=lambda c: False,
            get_kimi_managed_account_cooldown_key=lambda: "kimi:__managed__",
        )
        # Propagate runtime stubs into the function's actual __globals__ so
        # they take effect even when lpe has rebound selection functions to
        # resolve through vars(lpe) instead of vars(selection).
        runtime_globals = selection._select_anthropic_auto_agent_candidate.__globals__
        runtime_stub_names = {
            "_get_codex_active_cooldown_state": _zero_cooldown_state,
            "_get_anthropic_active_cooldown_state": _zero_cooldown_state,
            "_get_anthropic_merged_codex_openai_cooldown_state": _zero_cooldown_state,
            "_set_codex_cooldown": _noop_cooldown,
            "_set_anthropic_cooldown": _noop_cooldown,
            "_get_codex_session_affinity": AsyncMock(return_value=None),
            "_get_anthropic_session_affinity": _get_anthropic_auto_agent_session_affinity,
            "_get_openrouter_adapter_active_cooldown_seconds": _zero_adapter,
            "_normalize_codex_alias_model": lambda m: None,
            "_extract_client_product_label": lambda r, b: None,
            "_resolve_codex_session_key": lambda r, b, **kw: None,
            "_resolve_anthropic_session_key": lambda r, b, **kw: "test-session",
            "_has_continuation_state": lambda v: True,
            "_get_anthropic_candidates_for_alias": lambda alias: (),
            "_is_grok_account_quota_candidate": lambda c: False,
            "_get_grok_account_quota_lane_cooldown_key": lambda c, lk: None,
            "_is_kimi_code_candidate": lambda c: False,
            "_get_kimi_managed_account_cooldown_key": lambda: "kimi:__managed__",
        }
        previous_runtime_globals = {k: runtime_globals.get(k) for k in runtime_stub_names}
        runtime_globals.update(runtime_stub_names)
        # Patch lane-key resolvers in function globals to avoid god-module dependency
        lane_stubs = {
            "_resolve_codex_auto_agent_openai_cooldown_lane_key": (
                lambda request: "openai:primary"
            ),
            "_resolve_anthropic_auto_agent_native_cooldown_lane_key": (
                lambda request: "anthropic:primary"
            ),
            "_resolve_codex_auto_agent_xai_lane_key": (
                lambda candidate: "xai:default"
            ),
            "_codex_auto_agent_candidate_key": (
                lambda candidate, lane_key, epoch_tag=None: (
                    f"{candidate.get('provider')}:{candidate.get('model')}:{lane_key}"
                )
            ),
        }

        async def _noop_openrouter_quota(*, candidate, cooldown_seconds, cooldown_state_source, skip_reason):
            return cooldown_seconds, cooldown_state_source, skip_reason

        lane_stubs["_apply_openrouter_durable_quota_candidate_cooldown"] = _noop_openrouter_quota
        previous_globals = {k: runtime_globals.get(k) for k in lane_stubs}
        runtime_globals.update(lane_stubs)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(selection, name, value)
            for k, v in previous_globals.items():
                if v is None:
                    runtime_globals.pop(k, None)
                else:
                    runtime_globals[k] = v
            for k, v in previous_runtime_globals.items():
                if v is None:
                    runtime_globals.pop(k, None)
                else:
                    runtime_globals[k] = v

    @pytest.mark.asyncio
    async def test_setter_persists_config_hash_from_epoch_tag(
        self, fresh_manager, read_snapshot
    ):
        """Real setter persists config_hash from candidate config_epoch_tag."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            _get_anthropic_auto_agent_session_affinity,
            _set_anthropic_auto_agent_session_affinity,
        )

        candidate = {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "route_family": "anthropic_openai_responses_adapter",
            "last_resort": False,
            "config_epoch_tag": read_snapshot.config_hash,
        }
        await _set_anthropic_auto_agent_session_affinity("hash-session", candidate)
        affinity = await _get_anthropic_auto_agent_session_affinity("hash-session")
        assert affinity is not None
        assert affinity["config_hash"] == read_snapshot.config_hash

    @pytest.mark.asyncio
    async def test_setter_stores_none_config_hash_without_epoch_tag(self, fresh_manager):
        """Real setter stores None config_hash when candidate lacks config_epoch_tag."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            _get_anthropic_auto_agent_session_affinity,
            _set_anthropic_auto_agent_session_affinity,
        )

        candidate = {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "route_family": "anthropic_openai_responses_adapter",
            "last_resort": True,
        }
        await _set_anthropic_auto_agent_session_affinity("no-hash-session", candidate)
        affinity = await _get_anthropic_auto_agent_session_affinity("no-hash-session")
        assert affinity is not None
        assert affinity["config_hash"] is None

    @pytest.mark.asyncio
    async def test_schedule_changed_candidate_preserved_via_affinity(
        self, fresh_manager, _selection_runtime, read_snapshot
    ):
        """Real setter + live selector: still-present candidate selected via affinity."""
        from fastapi import Request

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            _set_anthropic_auto_agent_session_affinity,
        )

        # Set affinity via real setter for the first snapshot candidate
        candidate = read_snapshot.aliases["read"].candidates[0]
        shaped = _routing_candidate_to_anthropic_public_dict(
            candidate, epoch_tag=read_snapshot.config_hash
        )
        await _set_anthropic_auto_agent_session_affinity("test-session", shaped)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)
        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "read", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert result["selection_reason"] == "session_affinity"
        assert result["candidate"]["provider"] == candidate.provider
        assert result["candidate"]["model"] == candidate.model

    @pytest.mark.asyncio
    async def test_removed_candidate_raises_redispatch(
        self, fresh_manager, _selection_runtime, read_snapshot
    ):
        """Real setter + live selector: removed candidate triggers redispatch-required."""
        from fastapi import HTTPException, Request

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
            _set_anthropic_auto_agent_session_affinity,
        )

        # Set affinity for a candidate not present in the snapshot
        removed_candidate = {
            "provider": "openrouter",
            "model": "openrouter/removed-model",
            "route_family": "anthropic_openrouter_completion_adapter",
            "last_resort": False,
            "config_epoch_tag": read_snapshot.config_hash,
        }
        await _set_anthropic_auto_agent_session_affinity("test-session", removed_candidate)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)
        with pytest.raises(HTTPException) as exc_info:
            await selection._select_anthropic_auto_agent_candidate(
                request=request,
                request_body={"model": "read", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert exc_info.value.status_code == 429
        detail = exc_info.value.detail
        assert detail["redispatch_required"] is True
        assert detail["failure_phase"] == "affinity_continuation_removed"

    def test_other_alias_fails_closed_with_snapshot(self, read_snapshot):
        """Arbitrary alias 'other' returns empty candidates when snapshot is active."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _get_anthropic_candidates_for_alias_snapshot_aware,
        )

        result = _get_anthropic_candidates_for_alias_snapshot_aware("other")
        assert result == ()

    def test_known_legacy_anthropic_alias_still_resolves_with_snapshot(self, read_snapshot):
        """Known legacy Anthropic aliases still resolve from static tables with snapshot active."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
            ANTHROPIC_AAWM_READ_ALIAS,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _get_anthropic_candidates_for_alias_snapshot_aware,
        )

        result = _get_anthropic_candidates_for_alias_snapshot_aware(ANTHROPIC_AAWM_READ_ALIAS)
        assert len(result) > 0
