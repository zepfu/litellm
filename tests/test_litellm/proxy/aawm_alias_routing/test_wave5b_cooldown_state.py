"""Wave 5B module-local tests for cooldown_state.py.

Exercises the extracted cooldown/affinity/publication functions directly with
a fresh ``AliasRoutingStateManager`` -- no ambient god-module state, no import
of ``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import cooldown_state
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
    _attach_aawm_alias_routing_state_sources,
    _format_merged_alias_family_cooldown_state_source,
    _get_anthropic_auto_agent_active_cooldown_seconds,
    _get_anthropic_auto_agent_active_cooldown_state,
    _get_anthropic_auto_agent_merged_codex_openai_cooldown_state,
    _get_anthropic_auto_agent_session_affinity,
    _get_codex_auto_agent_active_cooldown_seconds,
    _get_codex_auto_agent_active_cooldown_state,
    _get_codex_auto_agent_session_affinity,
    _publish_anthropic_cooldown_memory,
    _publish_codex_cooldown_memory,
    _set_anthropic_auto_agent_cooldown,
    _set_anthropic_auto_agent_session_affinity,
    _set_codex_auto_agent_cooldown,
    _set_codex_auto_agent_session_affinity,
    configure_cooldown_state_runtime,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DURABLE_MODULE = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state"


@pytest.fixture()
def fresh_manager() -> AliasRoutingStateManager:
    """Create a fresh manager and bind it as the module runtime."""
    previous_manager = cooldown_state._manager
    mgr = AliasRoutingStateManager()
    configure_cooldown_state_runtime(manager=mgr)
    try:
        yield mgr
    finally:
        cooldown_state._manager = previous_manager


@pytest.fixture(autouse=True)
def _no_durable_cache():
    """Ensure no durable Redis is reachable so tests exercise memory paths."""
    with patch(
        f"{_DURABLE_MODULE}.get_aawm_alias_routing_dual_cache",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _no_durable_write():
    """Stub durable writes to no-ops (no Redis in unit tests)."""
    with patch(
        f"{_DURABLE_MODULE}.write_aawm_alias_routing_durable_payload",
        new_callable=AsyncMock,
        return_value=False,
    ):
        yield


# ---------------------------------------------------------------------------
# Codex cooldown read/write
# ---------------------------------------------------------------------------


class TestCodexCooldown:
    @pytest.mark.asyncio()
    async def test_no_cooldown_returns_zero_local_fallback(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        seconds, source = await _get_codex_auto_agent_active_cooldown_state("key-a")
        assert seconds == 0.0
        assert source == "local_fallback"

    @pytest.mark.asyncio()
    async def test_no_durable_cache_does_not_publish_negative_cache(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        first = await _get_codex_auto_agent_active_cooldown_state("key-no-cache")
        second = await _get_codex_auto_agent_active_cooldown_state("key-no-cache")
        assert first == (0.0, "local_fallback")
        assert second == (0.0, "local_fallback")
        assert (
            "key-no-cache"
            not in fresh_manager.codex.cooldown_negative_until_monotonic_by_key
        )

    @pytest.mark.asyncio()
    async def test_configured_durable_cache_miss_publishes_negative_cache(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        runtime_globals = _get_codex_auto_agent_active_cooldown_state.__globals__
        with patch.dict(
            runtime_globals,
            {
                "get_aawm_alias_routing_dual_cache": lambda: object(),
                "read_aawm_alias_routing_durable_payload": AsyncMock(
                    return_value=None
                ),
            },
        ):
            first = await _get_codex_auto_agent_active_cooldown_state(
                "key-cache-miss"
            )
            second = await _get_codex_auto_agent_active_cooldown_state(
                "key-cache-miss"
            )
        assert first == (0.0, "local_fallback")
        assert second == (0.0, "negative_cache")

    @pytest.mark.asyncio()
    async def test_set_then_read_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_cooldown("key-b", 120.0)
        seconds, source = await _get_codex_auto_agent_active_cooldown_state("key-b")
        assert source == "memory"
        assert 119.0 <= seconds <= 120.0

    @pytest.mark.asyncio()
    async def test_cooldown_seconds_convenience(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_cooldown("key-c", 60.0)
        seconds = await _get_codex_auto_agent_active_cooldown_seconds("key-c")
        assert 59.0 <= seconds <= 60.0

    @pytest.mark.asyncio()
    async def test_zero_ttl_does_not_write(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_cooldown("key-d", 0.0)
        seconds, source = await _get_codex_auto_agent_active_cooldown_state("key-d")
        assert seconds == 0.0
        assert source == "local_fallback"

    @pytest.mark.asyncio()
    async def test_max_expiry_semantics(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """A shorter cooldown must not truncate a longer existing one."""
        await _set_codex_auto_agent_cooldown("key-e", 300.0)
        await _set_codex_auto_agent_cooldown("key-e", 10.0)
        seconds, source = await _get_codex_auto_agent_active_cooldown_state("key-e")
        assert source == "memory"
        assert seconds > 200.0  # still near 300, not 10

    @pytest.mark.asyncio()
    async def test_expired_cooldown_evicted(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """Manually plant an already-expired entry; read should evict it."""
        fresh_manager.codex.cooldown_until_monotonic_by_key["key-f"] = (
            time.monotonic() - 1.0
        )
        seconds, source = await _get_codex_auto_agent_active_cooldown_state("key-f")
        assert seconds == 0.0
        assert source == "local_fallback"
        assert "key-f" not in fresh_manager.codex.cooldown_until_monotonic_by_key


# ---------------------------------------------------------------------------
# Anthropic cooldown read/write
# ---------------------------------------------------------------------------


class TestAnthropicCooldown:
    @pytest.mark.asyncio()
    async def test_no_cooldown_returns_zero_local_fallback(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        seconds, source = await _get_anthropic_auto_agent_active_cooldown_state("key-a")
        assert seconds == 0.0
        assert source == "local_fallback"

    @pytest.mark.asyncio()
    async def test_set_then_read_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_anthropic_auto_agent_cooldown("key-b", 90.0)
        seconds, source = await _get_anthropic_auto_agent_active_cooldown_state("key-b")
        assert source == "memory"
        assert 89.0 <= seconds <= 90.0

    @pytest.mark.asyncio()
    async def test_cooldown_seconds_convenience(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_anthropic_auto_agent_cooldown("key-c", 45.0)
        seconds = await _get_anthropic_auto_agent_active_cooldown_seconds("key-c")
        assert 44.0 <= seconds <= 45.0

    @pytest.mark.asyncio()
    async def test_negative_cache_hit(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """Plant a negative-cache entry; read should return negative_cache."""
        fresh_manager.anthropic.cooldown_negative_until_monotonic_by_key["key-nc"] = (
            time.monotonic() + 10.0
        )
        seconds, source = await _get_anthropic_auto_agent_active_cooldown_state("key-nc")
        assert seconds == 0.0
        assert source == "negative_cache"


# ---------------------------------------------------------------------------
# Codex session affinity
# ---------------------------------------------------------------------------

_CANDIDATE: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-5",
    "route_family": "codex",
    "last_resort": False,
    "config_epoch_tag": "epoch-abc",
}


class TestCodexAffinity:
    @pytest.mark.asyncio()
    async def test_none_session_key_returns_none(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        result = await _get_codex_auto_agent_session_affinity(None)
        assert result is None

    @pytest.mark.asyncio()
    async def test_set_then_read_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_session_affinity("sess-1", _CANDIDATE)
        affinity = await _get_codex_auto_agent_session_affinity("sess-1")
        assert affinity is not None
        assert affinity["provider"] == "openai"
        assert affinity["model"] == "gpt-5"
        assert affinity["route_family"] == "codex"
        assert affinity["config_hash"] == "epoch-abc"
        assert affinity["affinity_state_source"] == "memory"

    @pytest.mark.asyncio()
    async def test_set_none_session_key_is_noop(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_session_affinity(None, _CANDIDATE)
        assert len(fresh_manager.codex.session_affinity_by_key) == 0

    @pytest.mark.asyncio()
    async def test_expired_affinity_evicted(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        fresh_manager.codex.session_affinity_by_key["sess-exp"] = {
            "provider": "openai",
            "model": "gpt-5",
            "route_family": "codex",
            "last_resort": False,
            "expires_at_monotonic": time.monotonic() - 1.0,
            "affinity_state_source": "memory",
        }
        result = await _get_codex_auto_agent_session_affinity("sess-exp")
        assert result is None
        assert "sess-exp" not in fresh_manager.codex.session_affinity_by_key


# ---------------------------------------------------------------------------
# Anthropic session affinity
# ---------------------------------------------------------------------------

_ANTHROPIC_CANDIDATE: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "route_family": "anthropic",
    "last_resort": True,
}


class TestAnthropicAffinity:
    @pytest.mark.asyncio()
    async def test_none_session_key_returns_none(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        result = await _get_anthropic_auto_agent_session_affinity(None)
        assert result is None

    @pytest.mark.asyncio()
    async def test_set_then_read_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_anthropic_auto_agent_session_affinity("sess-a1", _ANTHROPIC_CANDIDATE)
        affinity = await _get_anthropic_auto_agent_session_affinity("sess-a1")
        assert affinity is not None
        assert affinity["provider"] == "anthropic"
        assert affinity["model"] == "claude-sonnet-4-20250514"
        assert affinity["last_resort"] is True
        assert affinity["affinity_state_source"] == "memory"

    @pytest.mark.asyncio()
    async def test_no_config_hash_in_anthropic_affinity(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """Anthropic affinity does not carry config_hash (baseline behavior)."""
        await _set_anthropic_auto_agent_session_affinity("sess-a2", _ANTHROPIC_CANDIDATE)
        affinity = await _get_anthropic_auto_agent_session_affinity("sess-a2")
        assert affinity is not None
        assert "config_hash" not in affinity


# ---------------------------------------------------------------------------
# Merged Codex/OpenAI cooldown state
# ---------------------------------------------------------------------------


class TestMergedCooldownState:
    def test_format_both_zero(self) -> None:
        seconds, source = _format_merged_alias_family_cooldown_state_source(
            anthropic_seconds=0.0,
            anthropic_source="memory",
            codex_seconds=0.0,
            codex_source="memory",
        )
        assert seconds == 0.0
        assert source == "local_fallback"

    def test_format_anthropic_only(self) -> None:
        seconds, source = _format_merged_alias_family_cooldown_state_source(
            anthropic_seconds=100.0,
            anthropic_source="memory",
            codex_seconds=0.0,
            codex_source="local_fallback",
        )
        assert seconds == 100.0
        assert source == "anthropic_family:memory"

    def test_format_codex_only(self) -> None:
        seconds, source = _format_merged_alias_family_cooldown_state_source(
            anthropic_seconds=0.0,
            anthropic_source="local_fallback",
            codex_seconds=50.0,
            codex_source="durable_cache",
        )
        assert seconds == 50.0
        assert source == "codex_family:durable_cache"

    def test_format_both_active_picks_max(self) -> None:
        seconds, source = _format_merged_alias_family_cooldown_state_source(
            anthropic_seconds=80.0,
            anthropic_source="memory",
            codex_seconds=120.0,
            codex_source="durable_cache",
        )
        assert seconds == 120.0
        # Sorted descending by seconds: codex first, then anthropic
        assert "codex_family:durable_cache" in source
        assert "anthropic_family:memory" in source

    @pytest.mark.asyncio()
    async def test_merged_state_with_fresh_manager(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """With no durable cache, both families return local_fallback => 0."""
        seconds, source = await _get_anthropic_auto_agent_merged_codex_openai_cooldown_state("mk-1")
        assert seconds == 0.0
        assert source == "local_fallback"

    @pytest.mark.asyncio()
    async def test_merged_state_with_memory_cooldowns(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_anthropic_auto_agent_cooldown("mk-2", 200.0)
        await _set_codex_auto_agent_cooldown("mk-2", 100.0)
        seconds, source = await _get_anthropic_auto_agent_merged_codex_openai_cooldown_state("mk-2")
        assert seconds > 190.0
        assert "anthropic_family:memory" in source
        assert "codex_family:memory" in source


# ---------------------------------------------------------------------------
# Synchronous memory publication (R3-1)
# ---------------------------------------------------------------------------


class TestMemoryPublication:
    def test_publish_codex_cooldown_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        _publish_codex_cooldown_memory(keys=["pk-1", "pk-2"], seconds=60.0)
        now = time.monotonic()
        assert fresh_manager.codex.cooldown_until_monotonic_by_key["pk-1"] > now
        assert fresh_manager.codex.cooldown_until_monotonic_by_key["pk-2"] > now
        # Anthropic family unaffected
        assert "pk-1" not in fresh_manager.anthropic.cooldown_until_monotonic_by_key

    def test_publish_anthropic_cooldown_memory(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        _publish_anthropic_cooldown_memory(keys=["ak-1"], seconds=90.0)
        now = time.monotonic()
        assert fresh_manager.anthropic.cooldown_until_monotonic_by_key["ak-1"] > now
        # Codex family unaffected
        assert "ak-1" not in fresh_manager.codex.cooldown_until_monotonic_by_key

    def test_publish_clears_negative_cache(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        """set_cooldown_memory clears negative-cache entries (baseline)."""
        fresh_manager.codex.cooldown_negative_until_monotonic_by_key["nk-1"] = (
            time.monotonic() + 100.0
        )
        _publish_codex_cooldown_memory(keys=["nk-1"], seconds=30.0)
        assert "nk-1" not in fresh_manager.codex.cooldown_negative_until_monotonic_by_key

    def test_publish_empty_keys_is_noop(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        _publish_codex_cooldown_memory(keys=[], seconds=60.0)
        assert len(fresh_manager.codex.cooldown_until_monotonic_by_key) == 0


# ---------------------------------------------------------------------------
# State-source attachment
# ---------------------------------------------------------------------------


class TestStateSourceAttachment:
    def test_no_affinity_no_state(self) -> None:
        selection = {"candidate": "x", "lane_key": "lk"}
        result = _attach_aawm_alias_routing_state_sources(selection)
        assert result == selection
        assert result is not selection  # must be a copy

    def test_affinity_source_attached(self) -> None:
        selection = {"lane_key": "lk"}
        affinity = {"affinity_state_source": "durable_cache"}
        result = _attach_aawm_alias_routing_state_sources(
            selection, affinity=affinity
        )
        assert result["affinity_state_source"] == "durable_cache"

    def test_affinity_missing_source_defaults(self) -> None:
        selection = {"lane_key": "lk"}
        affinity: dict[str, Any] = {"provider": "openai"}
        result = _attach_aawm_alias_routing_state_sources(
            selection, affinity=affinity
        )
        assert result["affinity_state_source"] == "local_fallback"

    def test_cooldown_state_source_attached(self) -> None:
        selection = {"lane_key": "lk"}
        selected_state = {"cooldown_state_source": "memory"}
        result = _attach_aawm_alias_routing_state_sources(
            selection, selected_state=selected_state
        )
        assert result["cooldown_state_source"] == "memory"

    def test_both_sources_attached(self) -> None:
        selection = {"lane_key": "lk"}
        result = _attach_aawm_alias_routing_state_sources(
            selection,
            affinity={"affinity_state_source": "memory"},
            selected_state={"cooldown_state_source": "durable_cache"},
        )
        assert result["affinity_state_source"] == "memory"
        assert result["cooldown_state_source"] == "durable_cache"

    def test_original_selection_not_mutated(self) -> None:
        selection = {"lane_key": "lk"}
        _attach_aawm_alias_routing_state_sources(
            selection,
            affinity={"affinity_state_source": "memory"},
        )
        assert "affinity_state_source" not in selection


# ---------------------------------------------------------------------------
# Family isolation
# ---------------------------------------------------------------------------


class TestFamilyIsolation:
    @pytest.mark.asyncio()
    async def test_codex_cooldown_does_not_leak_to_anthropic(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_codex_auto_agent_cooldown("iso-key", 120.0)
        codex_s, _ = await _get_codex_auto_agent_active_cooldown_state("iso-key")
        anthro_s, _ = await _get_anthropic_auto_agent_active_cooldown_state("iso-key")
        assert codex_s > 100.0
        assert anthro_s == 0.0

    @pytest.mark.asyncio()
    async def test_anthropic_affinity_does_not_leak_to_codex(
        self, fresh_manager: AliasRoutingStateManager
    ) -> None:
        await _set_anthropic_auto_agent_session_affinity("iso-sess", _ANTHROPIC_CANDIDATE)
        codex_aff = await _get_codex_auto_agent_session_affinity("iso-sess")
        anthro_aff = await _get_anthropic_auto_agent_session_affinity("iso-sess")
        assert codex_aff is None
        assert anthro_aff is not None


# ---------------------------------------------------------------------------
# Import boundary guard
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def test_no_god_module_import_at_module_scope(self) -> None:
        """cooldown_state.py must not import llm_passthrough_endpoints."""
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state as mod

        # The module's __dict__ must not reference the god module
        for value in vars(mod).values():
            if hasattr(value, "__module__"):
                assert "llm_passthrough_endpoints" not in str(
                    getattr(value, "__module__", "")
                ), f"god-module reference found: {value}"

    def test_no_wildcard_imports(self) -> None:
        """cooldown_state.py must not use wildcard imports."""
        import ast
        import pathlib

        src = pathlib.Path(
            "litellm/proxy/pass_through_endpoints/aawm_alias_routing/cooldown_state.py"
        ).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "*", "wildcard import found"
