"""CFG-013 Body D: canonical ``expert`` alias acceptance tests.

Verifies the checked-in canonical config directory compiles a public
``expert`` alias with exactly two candidates -- Claude-origin native
Anthropic ``claude-opus-5`` first, universal OpenAI/Codex ``gpt-5.6-terra``
last resort -- with authoritative ``reasoning_effort: max`` on both (CFG-006),
and that both ingress projections preserve the provider-native credential
boundary.

Canonical Opus 5 is inherently a 1M-context model; there is deliberately no
``claude-opus-5[1m]`` selector because a second selector would duplicate the
same upstream model.

No provider egress, no synthetic LLM calls, no TUI harness.
"""

from __future__ import annotations

import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingSnapshot,
    active_routing_snapshot_holder,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
    reset_startup_state,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    _select_snapshot_candidates,
)

_EXPERT_ALIAS = "expert"
_OPUS = ("anthropic", "claude-opus-5")
_TERRA = ("openai", "gpt-5.6-terra")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical_snapshot() -> RoutingSnapshot:
    """Compile the real canonical alias-config directory once."""
    return compile_directory(DEFAULT_CONFIG_DIR)


@pytest.fixture()
def active_snapshot(canonical_snapshot: RoutingSnapshot):
    # Guard against a failed-startup state left behind by another test in
    # the same session (fail-closed selection would otherwise return ()).
    reset_startup_state()
    previous = active_routing_snapshot_holder.swap(canonical_snapshot)
    yield canonical_snapshot
    active_routing_snapshot_holder.swap(previous)
    reset_startup_state()


def _make_request() -> Request:
    """Create a minimal Request with a fresh .state namespace."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


def _projected(
    snapshot: RoutingSnapshot,
    *,
    ingress: str,
    client_product_label: str | None,
) -> tuple[dict, ...]:
    return _select_snapshot_candidates(
        _EXPERT_ALIAS,
        ingress=ingress,
        client_product_label=client_product_label,
    )


# ---------------------------------------------------------------------------
# Canonical directory compile
# ---------------------------------------------------------------------------


class TestCanonicalDirectoryCompile:
    def test_expert_is_public_with_exactly_two_candidates_in_order(
        self, canonical_snapshot: RoutingSnapshot
    ) -> None:
        assert _EXPERT_ALIAS in canonical_snapshot.aliases
        alias = canonical_snapshot.aliases[_EXPERT_ALIAS]
        assert alias.visibility == "public"
        assert alias.dispatch is None
        identities = [
            (candidate.provider, candidate.model)
            for candidate in alias.candidates
        ]
        assert identities == [_OPUS, _TERRA]

    def test_both_candidates_carry_authoritative_max_reasoning(
        self, canonical_snapshot: RoutingSnapshot
    ) -> None:
        alias = canonical_snapshot.aliases[_EXPERT_ALIAS]
        efforts = [candidate.reasoning_effort for candidate in alias.candidates]
        assert efforts == ["max", "max"]

    def test_no_duplicate_1m_opus_candidate(
        self, canonical_snapshot: RoutingSnapshot
    ) -> None:
        alias = canonical_snapshot.aliases[_EXPERT_ALIAS]
        models = [candidate.model for candidate in alias.candidates]
        assert len(models) == len(set(models))
        assert "claude-opus-5[1m]" not in models
        assert not any("[1m]" in model for model in models)

    def test_opus_is_claude_attached_anthropic_native(
        self, canonical_snapshot: RoutingSnapshot
    ) -> None:
        opus = canonical_snapshot.aliases[_EXPERT_ALIAS].candidates[0]
        assert opus.route_family == "anthropic_messages"
        assert opus.anthropic_route_family == "anthropic_messages"
        assert opus.tui_attached == "Claude"
        assert opus.tui_excluded is None
        assert opus.priority > 0

    def test_terra_is_unexcluded_universal_last_resort(
        self, canonical_snapshot: RoutingSnapshot
    ) -> None:
        terra = canonical_snapshot.aliases[_EXPERT_ALIAS].candidates[1]
        assert terra.route_family == "codex_responses"
        assert terra.tui_attached is None
        assert terra.tui_excluded is None
        assert terra.priority == 0


# ---------------------------------------------------------------------------
# Anthropic Messages ingress projection
# ---------------------------------------------------------------------------


class TestAnthropicIngressProjection:
    def test_claude_origin_projects_opus_then_terra(
        self, active_snapshot: RoutingSnapshot
    ) -> None:
        candidates = _projected(
            active_snapshot,
            ingress="anthropic",
            client_product_label="claude-code/2.1.0",
        )
        assert [
            (candidate["provider"], candidate["model"]) for candidate in candidates
        ] == [_OPUS, _TERRA]
        assert candidates[0]["route_family"] == "anthropic_messages"
        assert candidates[1]["route_family"] == "anthropic_openai_responses_adapter"
        assert candidates[0]["reasoning_effort"] == "max"
        assert candidates[1]["reasoning_effort"] == "max"
        assert candidates[0]["last_resort"] is False
        assert candidates[1]["last_resort"] is True


# ---------------------------------------------------------------------------
# Codex ingress projection and credential boundary
# ---------------------------------------------------------------------------


class TestCodexIngressProjection:
    @pytest.mark.parametrize(
        "client_product_label",
        [
            "codex/0.50.0",
            "grok/1.0",
            "some-unknown-tui/9.9",
            None,
        ],
        ids=["codex", "non-claude", "unknown", "missing"],
    )
    def test_terra_is_direct_default_without_anthropic_crossover(
        self,
        active_snapshot: RoutingSnapshot,
        client_product_label: str | None,
    ) -> None:
        candidates = _projected(
            active_snapshot,
            ingress="codex",
            client_product_label=client_product_label,
        )
        assert [
            (candidate["provider"], candidate["model"]) for candidate in candidates
        ] == [_TERRA]
        assert candidates[0]["route_family"] == "codex_responses"
        assert candidates[0]["reasoning_effort"] == "max"
        # Anthropic/Claude models must never egress through Codex/OpenAI
        # credentials: no Anthropic-credential route family on this ingress.
        assert all(
            candidate["route_family"] != "anthropic_messages"
            for candidate in candidates
        )

    def test_non_claude_anthropic_ingress_still_projects_terra_only(
        self, active_snapshot: RoutingSnapshot
    ) -> None:
        candidates = _projected(
            active_snapshot,
            ingress="anthropic",
            client_product_label=None,
        )
        assert [
            (candidate["provider"], candidate["model"]) for candidate in candidates
        ] == [_TERRA]
        assert candidates[0]["route_family"] == "anthropic_openai_responses_adapter"


# ---------------------------------------------------------------------------
# Retryable Opus failure reaches Terra
# ---------------------------------------------------------------------------


class TestRetryableOpusFailureFallsBackToTerra:
    """Selection-level fallback after one retryable Opus failure.

    A retryable Opus failure produces an active cooldown on the Opus
    candidate state; the shared ``_select_available_state`` helper then
    skips the cooled-down primary tier and selects the Terra last resort.
    End-to-end dispatch-loop (candidate_loop) coverage belongs to the
    later native-dispatch integration body, not this config acceptance.
    """

    def test_cooled_down_opus_yields_terra_last_resort(self) -> None:
        request = _make_request()
        opus_candidate = {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "route_family": "anthropic_messages",
            "last_resort": False,
            "selection_priority": 100,
            "reasoning_effort": "max",
        }
        terra_candidate = {
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "route_family": "anthropic_openai_responses_adapter",
            "last_resort": True,
            "selection_priority": 0,
            "reasoning_effort": "max",
        }
        states = [
            {
                "candidate": opus_candidate,
                "lane_key": "anthropic:primary",
                "cooldown_key": "anthropic:claude-opus-5:anthropic:primary",
                "cooldown_seconds": 30.0,
                "cooldown_state_source": "memory",
                "skip_reason": "retryable_failure_cooldown",
            },
            {
                "candidate": terra_candidate,
                "lane_key": "openai:primary",
                "cooldown_key": "openai:gpt-5.6-terra:openai:primary",
                "cooldown_seconds": 0.0,
                "cooldown_state_source": None,
            },
        ]

        # Primary (non-last-resort) tier is fully cooled down after the
        # retryable Opus failure.
        assert (
            selection._select_available_state(request, states, last_resort=False)
            is None
        )

        # The fallback tier reaches Terra.
        picked = selection._select_available_state(
            request, states, last_resort=True
        )
        assert picked is not None
        assert picked["candidate"]["provider"] == "openai"
        assert picked["candidate"]["model"] == "gpt-5.6-terra"
