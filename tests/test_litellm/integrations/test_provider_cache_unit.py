"""Wave A3B (provider-cache typed split) unit + parity tests.

``litellm/integrations/aawm_agent_identity/__init__.py`` CURRENTLY holds the
provider-cache state machine inline (``_resolve_provider_cache_state``,
``_compute_provider_cache_miss_cost_state``, and the metadata enrichment in
``_enrich_provider_cache_metadata``).  Wave A3B's engineer will extract it
into ``provider_cache.py`` behind a typed ``ProviderCacheState`` frozen
dataclass (added to ``interfaces.py``) and a pure ``_price_cache_miss``
function, leaving facade rebinds in the package ``__init__``.

These tests are written BEFORE the move.

GREEN tests (pass on develop today -- parity / golden):
  * Cache attempt/outcome matrix for OpenAI-style and Gemini-style
    cached-token fields via ``_resolve_provider_cache_state``.
  * Cache-miss pricing parity via ``_compute_provider_cache_miss_cost_state``
    with the ``grok-build`` model whose bundled cost-map pricing is known.
  * Stable record key contract: the exact dict keys that the record builder
    in ``aawm_session_history/record.py`` maps to the CLAUDE.md-documented
    session_history columns (``provider_cache_status``,
    ``provider_cache_miss_reason``, ``provider_cache_miss_token_count``,
    ``provider_cache_miss_cost_usd``).

RED tests (fail until the A3B engineer lands):
  * ``ProviderCacheState`` frozen dataclass in ``interfaces.py``.
  * ``_price_cache_miss(state, cost_map)`` pure pricing function.

See ``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``
### Wave A3B -> Test Spec.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import pytest

from litellm.integrations.aawm_agent_identity import (
    _compute_provider_cache_miss_cost_state,
    _resolve_provider_cache_state,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _usage(**overrides: Any) -> Dict[str, Any]:
    """Minimal usage dict with sensible defaults."""
    base: Dict[str, Any] = {
        "prompt_tokens": 2048,
        "completion_tokens": 64,
        "total_tokens": 2112,
    }
    base.update(overrides)
    return base


# ===========================================================================
# GREEN -- cache attempt/outcome matrix (parity with current develop)
# ===========================================================================


class TestResolveCacheStateOpenAIStyle:
    """OpenAI-style cached-token fields: ``prompt_tokens_details.cached_tokens``,
    ``input_tokens_details.cached_tokens``, ``cache_read_input_tokens``, etc."""

    def test_hit_via_cache_read_input_tokens(self) -> None:
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(cache_read_input_tokens=1500),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "hit"
        assert state["miss"] is False
        assert state["miss_reason"] is None
        assert state["source"] == "usage.cache_read_input_tokens"

    def test_miss_zero_cached_tokens_in_prompt_tokens_details(self) -> None:
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(prompt_tokens_details={"cached_tokens": 0}),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "miss"
        assert state["miss"] is True
        assert state["miss_reason"] == "cached_tokens_reported_zero"
        assert state["source"] == "usage.prompt_tokens_details.cached_tokens"

    def test_miss_zero_cached_tokens_in_input_tokens_details(self) -> None:
        """Responses-API shape: ``input_tokens_details.cached_tokens``."""
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(input_tokens_details={"cached_tokens": 0}),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "miss"
        assert state["miss"] is True
        assert state["miss_reason"] == "cached_tokens_reported_zero"
        assert state["source"] == "usage.input_tokens_details.cached_tokens"

    def test_not_attempted_no_cache_fields(self) -> None:
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(),
        )
        assert state is not None
        assert state["attempted"] is False
        assert state["status"] == "not_attempted"
        assert state["miss"] is False
        assert state["miss_reason"] is None
        assert state["source"] is None

    def test_write_via_cache_creation_input_tokens(self) -> None:
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(cache_creation_input_tokens=800),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "write"
        assert state["miss"] is True
        assert state["miss_reason"] == "cache_write_only"
        assert state["source"] == "usage.cache_creation_input_tokens"


class TestResolveCacheStateGeminiStyle:
    """Gemini-style cached-token field: ``cachedContentTokenCount``."""

    def test_hit_via_cached_content_token_count(self) -> None:
        state = _resolve_provider_cache_state(
            provider="google",
            model="gemini-2.5-pro",
            usage_obj=_usage(cachedContentTokenCount=1500),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "hit"
        assert state["miss"] is False
        assert state["miss_reason"] is None
        assert state["source"] == "usage.cache_read_input_tokens"

    def test_miss_zero_cached_content_token_count(self) -> None:
        state = _resolve_provider_cache_state(
            provider="google",
            model="gemini-2.5-pro",
            usage_obj=_usage(cachedContentTokenCount=0),
        )
        assert state is not None
        assert state["attempted"] is True
        assert state["status"] == "miss"
        assert state["miss"] is True
        assert state["miss_reason"] == "cached_tokens_reported_zero"
        assert state["source"] == "usage.cached_content_token_count"

    def test_not_attempted_no_cached_content_field(self) -> None:
        state = _resolve_provider_cache_state(
            provider="google",
            model="gemini-2.5-pro",
            usage_obj=_usage(),
        )
        assert state is not None
        assert state["attempted"] is False
        assert state["status"] == "not_attempted"
        assert state["miss"] is False
        assert state["miss_reason"] is None
        assert state["source"] is None


# ===========================================================================
# GREEN -- cache-miss pricing parity (exact USD via current code path)
# ===========================================================================

# grok-build pricing from the bundled model cost map (verified by the
# existing test_build_session_history_record_marks_xai_partial_cache_hit_miss_cost):
#   input_cost_per_token       = 0.00000125
#   cache_read_input_token_cost = 0.0000002
_GROK_BUILD_INPUT_COST = 0.00000125
_GROK_BUILD_CACHE_READ_COST = 0.0000002
_GROK_BUILD_DELTA = _GROK_BUILD_INPUT_COST - _GROK_BUILD_CACHE_READ_COST  # 0.00000105


class TestComputeMissCostPricing:
    """Pin the exact USD output of ``_compute_provider_cache_miss_cost_state``
    for the ``grok-build`` model whose pricing is known from the bundled cost
    map.  These are parity tests: they must pass today and after the A3B
    extraction."""

    def test_full_miss_exact_usd(self) -> None:
        """Full cache miss: all 2000 prompt tokens are uncached."""
        cache_state: Dict[str, Any] = {
            "attempted": True,
            "status": "miss",
            "miss": True,
            "miss_reason": "cached_tokens_reported_zero",
            "source": "usage.prompt_tokens_details.cached_tokens",
        }
        result = _compute_provider_cache_miss_cost_state(
            provider_family="xai",
            model="grok-build",
            usage_obj=_usage(
                prompt_tokens=2000,
                completion_tokens=64,
                total_tokens=2064,
                prompt_tokens_details={"cached_tokens": 0},
            ),
            cache_state=cache_state,
        )
        assert result["miss_token_count"] == 2000
        assert result["miss_cost_usd"] == pytest.approx(_GROK_BUILD_DELTA * 2000)
        assert result["miss_cost_basis"] == "prompt_vs_cache_read_delta"

    def test_partial_hit_exact_usd(self) -> None:
        """Partial cache hit: 700 of 1000 prompt tokens cached, 300 missed."""
        cache_state: Dict[str, Any] = {
            "attempted": True,
            "status": "hit",
            "miss": True,
            "miss_reason": "partial_cache_hit",
            "source": "usage.cache_read_input_tokens",
        }
        result = _compute_provider_cache_miss_cost_state(
            provider_family="xai",
            model="grok-build",
            usage_obj=_usage(
                prompt_tokens=1000,
                completion_tokens=12,
                total_tokens=1012,
                cache_read_input_tokens=700,
            ),
            cache_state=cache_state,
        )
        assert result["miss_token_count"] == 300
        assert result["miss_cost_usd"] == pytest.approx(_GROK_BUILD_DELTA * 300)
        assert result["miss_cost_basis"] == "prompt_vs_cache_read_delta"

    def test_clean_hit_no_miss_cost(self) -> None:
        """Clean hit with no miss: cost fields stay None."""
        cache_state: Dict[str, Any] = {
            "attempted": True,
            "status": "hit",
            "miss": False,
            "miss_reason": None,
            "source": "usage.cache_read_input_tokens",
        }
        result = _compute_provider_cache_miss_cost_state(
            provider_family="xai",
            model="grok-build",
            usage_obj=_usage(cache_read_input_tokens=2048),
            cache_state=cache_state,
        )
        assert result["miss_token_count"] is None
        assert result["miss_cost_usd"] is None
        assert result["miss_cost_basis"] is None


# ===========================================================================
# GREEN -- stable record key contract
# ===========================================================================

# CLAUDE.md declares these session_history columns a stable surface.
# The record builder in aawm_session_history/record.py maps:
#   cache_state["status"]           -> record["provider_cache_status"]
#   cache_state["miss_reason"]      -> record["provider_cache_miss_reason"]
#   cost_state["miss_token_count"]  -> record["provider_cache_miss_token_count"]
#   cost_state["miss_cost_usd"]     -> record["provider_cache_miss_cost_usd"]
STABLE_RECORD_KEY_MAP = {
    "status": "provider_cache_status",
    "miss_reason": "provider_cache_miss_reason",
    "miss_token_count": "provider_cache_miss_token_count",
    "miss_cost_usd": "provider_cache_miss_cost_usd",
}


class TestStableRecordKeyContract:
    """Pin the exact dict keys that ``_resolve_provider_cache_state`` and
    ``_compute_provider_cache_miss_cost_state`` produce, because the record
    builder maps them 1:1 to the CLAUDE.md-documented stable columns."""

    def test_resolve_state_contains_status_and_miss_reason(self) -> None:
        state = _resolve_provider_cache_state(
            provider="openai",
            model="gpt-4o",
            usage_obj=_usage(prompt_tokens_details={"cached_tokens": 0}),
        )
        assert state is not None
        for key in ("status", "miss_reason", "attempted", "miss", "source"):
            assert key in state, f"Missing key {key!r} in resolved cache state"

    def test_compute_cost_contains_miss_token_count_and_cost(self) -> None:
        cache_state: Dict[str, Any] = {
            "attempted": True,
            "status": "miss",
            "miss": True,
            "miss_reason": "cached_tokens_reported_zero",
            "source": "usage.prompt_tokens_details.cached_tokens",
        }
        result = _compute_provider_cache_miss_cost_state(
            provider_family="xai",
            model="grok-build",
            usage_obj=_usage(
                prompt_tokens=2000,
                total_tokens=2064,
                prompt_tokens_details={"cached_tokens": 0},
            ),
            cache_state=cache_state,
        )
        for key in ("miss_token_count", "miss_cost_usd", "miss_cost_basis"):
            assert key in result, f"Missing key {key!r} in cost state"

    def test_merged_state_maps_to_stable_record_keys(self) -> None:
        """Simulate the record builder's merge + key mapping and verify
        the four CLAUDE.md-stable column names are producible."""
        usage_obj = _usage(
            prompt_tokens=1000,
            completion_tokens=12,
            total_tokens=1012,
            cache_read_input_tokens=700,
        )
        state = _resolve_provider_cache_state(
            provider="xai",
            model="grok-build",
            usage_obj=usage_obj,
            metadata={"passthrough_route_family": "grok_cli_chat_proxy"},
        )
        assert state is not None
        cost = _compute_provider_cache_miss_cost_state(
            provider_family="xai",
            model="grok-build",
            usage_obj=usage_obj,
            cache_state=state,
        )
        merged = {**state, **cost}
        for internal_key, record_key in STABLE_RECORD_KEY_MAP.items():
            assert internal_key in merged, (
                f"Internal key {internal_key!r} (-> {record_key!r}) "
                f"missing from merged cache state"
            )


# ===========================================================================
# RED -- typed ProviderCacheState expectation (fails until A3B engineer lands)
# ===========================================================================


class TestProviderCacheStateTypedExpectation:
    """These tests define the typed contract the A3B engineer must satisfy.
    They FAIL on current develop because ``ProviderCacheState`` and
    ``_price_cache_miss`` do not exist yet."""

    def test_provider_cache_state_importable_and_frozen(self) -> None:
        """``ProviderCacheState`` must be a frozen dataclass in interfaces.py."""
        from litellm.integrations.aawm_agent_identity.interfaces import (
            ProviderCacheState,
        )

        assert dataclasses.is_dataclass(ProviderCacheState)
        params = ProviderCacheState.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen is True

    def test_provider_cache_state_has_stable_fields(self) -> None:
        """``ProviderCacheState`` fields must cover the four CLAUDE.md-stable
        record columns plus the internal bookkeeping fields."""
        from litellm.integrations.aawm_agent_identity.interfaces import (
            ProviderCacheState,
        )

        field_names = {f.name for f in dataclasses.fields(ProviderCacheState)}
        # Stable record columns (mirrored as dataclass fields):
        assert "status" in field_names, "maps to provider_cache_status"
        assert "miss_reason" in field_names, "maps to provider_cache_miss_reason"
        assert "miss_token_count" in field_names, "maps to provider_cache_miss_token_count"
        assert "miss_cost_usd" in field_names, "maps to provider_cache_miss_cost_usd"
        # Internal bookkeeping:
        assert "attempted" in field_names
        assert "miss" in field_names
        assert "source" in field_names
        assert "miss_cost_basis" in field_names

    def test_price_cache_miss_pure(self) -> None:
        """``_price_cache_miss(state, cost_map)`` must compute exact USD from
        literal token counts and per-token pricing, with no model lookup."""
        from litellm.integrations.aawm_agent_identity import _price_cache_miss

        state = {"miss_token_count": 2000}
        cost_map = {
            "input_cost_per_token": 0.00000125,
            "cache_read_input_token_cost": 0.0000002,
        }
        result = _price_cache_miss(state, cost_map)
        assert result == pytest.approx((0.00000125 - 0.0000002) * 2000)

    def test_price_cache_miss_zero_tokens(self) -> None:
        """Zero miss tokens must produce zero cost."""
        from litellm.integrations.aawm_agent_identity import _price_cache_miss

        state = {"miss_token_count": 0}
        cost_map = {
            "input_cost_per_token": 0.00000125,
            "cache_read_input_token_cost": 0.0000002,
        }
        result = _price_cache_miss(state, cost_map)
        assert result == pytest.approx(0.0)

    def test_price_cache_miss_no_cache_read_pricing(self) -> None:
        """When ``cache_read_input_token_cost`` is absent, fall back to
        ``input_cost_per_token`` as the full per-token price."""
        from litellm.integrations.aawm_agent_identity import _price_cache_miss

        state = {"miss_token_count": 1000}
        cost_map = {"input_cost_per_token": 0.000003}
        result = _price_cache_miss(state, cost_map)
        assert result == pytest.approx(0.000003 * 1000)
