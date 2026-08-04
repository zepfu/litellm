"""Wave 7: provider-neutral OAuth token-cache semantics, alias routing
state-manager families, OpenRouter cooldowns, lock identity, and
state-manager reset behavior.
"""

from __future__ import annotations

import asyncio

import pytest

from litellm.proxy.pass_through_endpoints import aawm_alias_routing
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    oauth_token_cache as oauth_token_cache_module,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.oauth_token_cache import (
    OAuthAccessTokenCache,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

# ---------------------------------------------------------------------------
# Provider-neutral OAuth token cache exports
# ---------------------------------------------------------------------------


class TestOAuthTokenCacheModuleExports:
    def test_package_exports_provider_neutral_module(self):
        assert aawm_alias_routing.oauth_token_cache is oauth_token_cache_module
        assert "oauth_token_cache" in aawm_alias_routing.__all__

    def test_class_still_exported(self):
        """The provider-neutral dataclass must remain importable."""
        cache = OAuthAccessTokenCache()
        assert isinstance(cache.lock, asyncio.Lock)
        assert cache.tokens == {}


# ---------------------------------------------------------------------------
# Retained: Codex / Anthropic family states
# ---------------------------------------------------------------------------


class TestRetainedFamilyStates:
    def test_codex_family(self):
        mgr = AliasRoutingStateManager()
        fam = mgr.family("codex")
        assert fam is mgr.codex

    def test_anthropic_family(self):
        mgr = AliasRoutingStateManager()
        fam = mgr.family("anthropic")
        assert fam is mgr.anthropic

    def test_unknown_family_defaults_to_codex(self):
        mgr = AliasRoutingStateManager()
        assert mgr.family("unknown") is mgr.codex


# ---------------------------------------------------------------------------
# Retained: OpenRouter cooldown maps + locks
# ---------------------------------------------------------------------------


class TestRetainedOpenRouterState:
    def test_rate_limit_map(self):
        mgr = AliasRoutingStateManager()
        mgr.openrouter_rate_limit.extend("k", 60.0)
        assert mgr.openrouter_rate_limit.remaining("k") > 0

    def test_failure_circuit_map(self):
        mgr = AliasRoutingStateManager()
        mgr.openrouter_failure_circuit.extend("k", 30.0)
        assert mgr.openrouter_failure_circuit.remaining("k") > 0


# ---------------------------------------------------------------------------
# Retained: lock identity and lane_state_cache_lock
# ---------------------------------------------------------------------------


class TestRetainedLockIdentity:
    def test_lane_state_cache_lock_is_asyncio_lock(self):
        mgr = AliasRoutingStateManager()
        assert isinstance(mgr.lane_state_cache_lock, asyncio.Lock)

    def test_candidate_probe_locks_guard_is_asyncio_lock(self):
        mgr = AliasRoutingStateManager()
        assert isinstance(mgr.candidate_probe_locks_guard, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_candidate_probe_lock_returns_lock(self):
        mgr = AliasRoutingStateManager()
        lock = await mgr.candidate_probe_lock(
            alias_family="codex", cooldown_key="test-key"
        )
        assert isinstance(lock, asyncio.Lock)
        # Same key returns the same lock object (identity).
        lock2 = await mgr.candidate_probe_lock(
            alias_family="codex", cooldown_key="test-key"
        )
        assert lock is lock2


# ---------------------------------------------------------------------------
# Retained: reset_for_tests clears surviving state
# ---------------------------------------------------------------------------


class TestRetainedResetBehavior:
    def test_reset_preserves_retained_map_identity(self):
        mgr = AliasRoutingStateManager()
        codex_cooldowns = mgr.codex.cooldown_until_monotonic_by_key
        anthropic_affinity = mgr.anthropic.session_affinity_by_key
        log_timestamps = mgr.log_until_monotonic_by_key
        probe_locks = mgr.candidate_probe_locks
        openrouter_limits = mgr.openrouter_rate_limit.until_monotonic_by_key
        round_robin_cursor = mgr.round_robin_cursor

        codex_cooldowns["codex"] = 1.0
        anthropic_affinity["anthropic"] = {"provider": "anthropic"}
        log_timestamps["log"] = 1.0
        probe_locks["codex:key"] = asyncio.Lock()
        openrouter_limits["openrouter"] = 1.0
        round_robin_cursor[("codex", "alias")] = 1

        mgr.reset_for_tests()

        assert mgr.codex.cooldown_until_monotonic_by_key is codex_cooldowns
        assert mgr.anthropic.session_affinity_by_key is anthropic_affinity
        assert mgr.log_until_monotonic_by_key is log_timestamps
        assert mgr.candidate_probe_locks is probe_locks
        assert mgr.openrouter_rate_limit.until_monotonic_by_key is openrouter_limits
        assert mgr.round_robin_cursor is round_robin_cursor
        assert not any(
            (
                codex_cooldowns,
                anthropic_affinity,
                log_timestamps,
                probe_locks,
                openrouter_limits,
                round_robin_cursor,
            )
        )

    def test_reset_clears_codex_and_anthropic(self):
        mgr = AliasRoutingStateManager()
        mgr.codex.cooldown_until_monotonic_by_key["x"] = 999.0
        mgr.anthropic.session_affinity_by_key["y"] = {"provider": "p"}
        mgr.reset_for_tests()
        assert mgr.codex.cooldown_until_monotonic_by_key == {}
        assert mgr.anthropic.session_affinity_by_key == {}

    def test_reset_clears_openrouter_maps(self):
        mgr = AliasRoutingStateManager()
        mgr.openrouter_rate_limit.extend("k", 60.0)
        mgr.openrouter_failure_circuit.extend("k", 60.0)
        mgr.reset_for_tests()
        assert mgr.openrouter_rate_limit.remaining("k") == 0.0
        assert mgr.openrouter_failure_circuit.remaining("k") == 0.0

    def test_reset_clears_log_and_probe_locks(self):
        mgr = AliasRoutingStateManager()
        mgr.log_until_monotonic_by_key["z"] = 1.0
        mgr.candidate_probe_locks["a:b"] = asyncio.Lock()
        mgr.reset_for_tests()
        assert mgr.log_until_monotonic_by_key == {}
        assert mgr.candidate_probe_locks == {}

    def test_reset_clears_round_robin_cursor(self):
        mgr = AliasRoutingStateManager()
        mgr.round_robin_cursor[("a", "b")] = 3
        mgr.reset_for_tests()
        assert mgr.round_robin_cursor == {}

    def test_reset_resets_openrouter_free_quota(self):
        mgr = AliasRoutingStateManager()
        mgr.set_openrouter_free_quota_cache((100.0, 999.0))
        mgr.reset_for_tests()
        assert mgr.get_openrouter_free_quota_cache() == (None, 0.0)


# ---------------------------------------------------------------------------
# Retained: OAuthAccessTokenCache provider-neutral semantics
# ---------------------------------------------------------------------------


class TestOAuthCacheNeutralSemantics:
    def test_set_get_roundtrip(self):
        cache = OAuthAccessTokenCache()
        cache.set("key1", "tok_abc", 9_999_999_999_999)
        assert cache.get_if_valid("key1") == "tok_abc"

    def test_expired_token_returns_none(self):
        cache = OAuthAccessTokenCache()
        cache.set("key1", "tok_abc", 1)  # 1 ms epoch -> long expired
        assert cache.get_if_valid("key1") is None

    @pytest.mark.parametrize(
        ("now", "skew_seconds", "expected"),
        (
            (69.9, 30.0, "tok_abc"),
            (70.0, 30.0, None),
            (99.9, 0.0, "tok_abc"),
            (100.0, 0.0, None),
        ),
    )
    def test_custom_now_and_skew(
        self, now: float, skew_seconds: float, expected: str | None
    ):
        cache = OAuthAccessTokenCache()
        cache.set("key1", "tok_abc", 100)
        assert (
            cache.get_if_valid(
                "key1",
                now=now,
                skew_seconds=skew_seconds,
                expiry_is_millis=False,
            )
            == expected
        )

    def test_clear_single_key(self):
        cache = OAuthAccessTokenCache()
        cache.set("a", "t1", 9_999_999_999_999)
        cache.set("b", "t2", 9_999_999_999_999)
        cache.clear("a")
        assert cache.get_if_valid("a") is None
        assert cache.get_if_valid("b") == "t2"

    def test_clear_all(self):
        cache = OAuthAccessTokenCache()
        cache.set("a", "t1", 9_999_999_999_999)
        cache.clear()
        assert cache.tokens == {}
