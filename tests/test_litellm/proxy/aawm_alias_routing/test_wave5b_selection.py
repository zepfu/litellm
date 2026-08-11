"""Module-local tests for Wave 5B selection.py extraction.

Drives the new module directly with fresh state/dependency stubs.
Does NOT import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    SelectionEnumeration,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_request() -> Request:
    """Create a minimal Request with a fresh .state namespace."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


def _candidate(provider: str = "openai", model: str = "gpt-4o", last_resort: bool = False) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "route_family": f"{provider}_responses_adapter",
        "last_resort": last_resort,
    }


def _set_selection_runtime(name: str, value: Any) -> None:
    """Update both the package attribute and rebound function globals."""
    setattr(selection, name, value)
    selection._select_codex_auto_agent_candidate.__globals__[name] = value


def _set_selection_candidates(
    candidates: tuple[dict[str, Any], ...],
) -> None:
    enumeration = SelectionEnumeration(
        candidates=candidates,
        commit_token=None,
    )
    _set_selection_runtime(
        "_resolve_aawm_alias_selection_enumeration",
        lambda request, canonical_alias, *, ingress, client_product_label=None: (
            enumeration
        ),
    )


@pytest.fixture(autouse=True)
def _configure_selection():
    """Configure selection runtime with fresh stubs before each test."""
    async def _noop_cooldown(key: str, seconds: float) -> None:
        pass

    async def _zero_cooldown_state(key: str) -> tuple[float, str]:
        return (0.0, "local_fallback")

    async def _zero_adapter(model: Optional[str]) -> float:
        return 0.0

    runtime_names = {
        "_get_codex_active_cooldown_state",
        "_get_anthropic_active_cooldown_state",
        "_get_anthropic_merged_codex_openai_cooldown_state",
        "_set_codex_cooldown",
        "_set_anthropic_cooldown",
        "_get_codex_session_affinity",
        "_get_anthropic_session_affinity",
        "_get_openrouter_adapter_active_cooldown_seconds",
        "_extract_client_product_label",
        "_resolve_codex_session_key",
        "_resolve_anthropic_session_key",
        "_has_continuation_state",
        "_lookup_active_snapshot_canonical_alias",
        "_resolve_aawm_alias_selection_enumeration",
        "_is_grok_account_quota_candidate",
        "_get_grok_account_quota_lane_cooldown_key",
        "_is_kimi_code_candidate",
        "_get_kimi_managed_account_cooldown_key",
    }
    previous_runtime = {
        name: getattr(selection, name)
        for name in runtime_names
    }
    runtime = {
        "_get_codex_active_cooldown_state": _zero_cooldown_state,
        "_get_anthropic_active_cooldown_state": _zero_cooldown_state,
        "_get_anthropic_merged_codex_openai_cooldown_state": _zero_cooldown_state,
        "_set_codex_cooldown": _noop_cooldown,
        "_set_anthropic_cooldown": _noop_cooldown,
        "_get_codex_session_affinity": AsyncMock(return_value=None),
        "_get_anthropic_session_affinity": AsyncMock(return_value=None),
        "_get_openrouter_adapter_active_cooldown_seconds": _zero_adapter,
        "_extract_client_product_label": lambda r, b: None,
        "_resolve_codex_session_key": lambda r, b, *, alias_model: None,
        "_resolve_anthropic_session_key": lambda r, b, *, alias_model: None,
        "_has_continuation_state": lambda v: False,
        "_lookup_active_snapshot_canonical_alias": (
            lambda model, *, request=None: (
                "basic"
                if isinstance(model, str) and model.strip().casefold() == "basic"
                else None
            )
        ),
        "_resolve_aawm_alias_selection_enumeration": (
            lambda request, canonical_alias, *, ingress, client_product_label=None: (
                SelectionEnumeration(candidates=(), commit_token=None)
            )
        ),
        "_is_grok_account_quota_candidate": lambda c: False,
        "_get_grok_account_quota_lane_cooldown_key": lambda c, lk: None,
        "_is_kimi_code_candidate": (
            lambda c: isinstance(c, dict) and c.get("provider") == "kimi_code"
        ),
        "_get_kimi_managed_account_cooldown_key": (
            lambda: "kimi_code:__managed_account__:kimi_code_managed_account"
        ),
    }
    selection.configure_selection_runtime(
        get_codex_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_merged_codex_openai_cooldown_state=_zero_cooldown_state,
        set_codex_cooldown=_noop_cooldown,
        set_anthropic_cooldown=_noop_cooldown,
        get_codex_session_affinity=AsyncMock(return_value=None),
        get_anthropic_session_affinity=AsyncMock(return_value=None),
        get_openrouter_adapter_active_cooldown_seconds=_zero_adapter,
        extract_client_product_label=lambda r, b: None,
        resolve_codex_session_key=lambda r, b, *, alias_model: None,
        resolve_anthropic_session_key=lambda r, b, *, alias_model: None,
        has_continuation_state=lambda v: False,
        is_grok_account_quota_candidate=lambda c: False,
        get_grok_account_quota_lane_cooldown_key=lambda c, lk: None,
        is_kimi_code_candidate=lambda c: isinstance(c, dict) and c.get("provider") == "kimi_code",
        get_kimi_managed_account_cooldown_key=lambda: "kimi_code:__managed_account__:kimi_code_managed_account",
    )
    runtime_globals = selection._build_codex_auto_agent_candidate_state.__globals__
    runtime.update(
        {
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
    )
    try:
        with patch.dict(runtime_globals, runtime):
            yield
    finally:
        for name, value in previous_runtime.items():
            setattr(selection, name, value)


# ---------------------------------------------------------------------------
# Candidate public shaping
# ---------------------------------------------------------------------------


class TestCandidatePublicShape:
    def test_basic_shape(self):
        shaped = selection._codex_auto_agent_candidate_public_shape(
            _candidate(),
            lane_key="openai:primary",
            cooldown_seconds=10.5,
            reason="cooldown",
        )
        assert shaped["provider"] == "openai"
        assert shaped["model"] == "gpt-4o"
        assert shaped["route_family"] == "openai_responses_adapter"
        assert shaped["last_resort"] is False
        assert shaped["lane_key"] == "openai:primary"
        assert shaped["cooldown_seconds"] == 10.5
        assert shaped["reason"] == "cooldown"

    def test_omits_none_fields(self):
        shaped = selection._codex_auto_agent_candidate_public_shape(_candidate())
        assert "lane_key" not in shaped
        assert "cooldown_seconds" not in shaped
        assert "reason" not in shaped


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_when_no_cooldown_no_skip(self):
        state = {"cooldown_seconds": 0.0, "skip_reason": None}
        assert selection._is_auto_agent_candidate_state_available(state) is True

    def test_unavailable_when_cooldown(self):
        state = {"cooldown_seconds": 5.0}
        assert selection._is_auto_agent_candidate_state_available(state) is False

    def test_unavailable_when_skip_reason(self):
        state = {"cooldown_seconds": 0.0, "skip_reason": "auth_degraded"}
        assert selection._is_auto_agent_candidate_state_available(state) is False


# ---------------------------------------------------------------------------
# Skipped shaping
# ---------------------------------------------------------------------------


class TestSkippedShaping:
    def test_skipped_from_states(self):
        states = [
            {
                "candidate": _candidate("openai", "gpt-4o"),
                "lane_key": "openai:primary",
                "cooldown_seconds": 10.0,
                "cooldown_state_source": "memory",
            },
            {
                "candidate": _candidate("xai", "grok-4"),
                "lane_key": "xai:lane1",
                "cooldown_seconds": 0.0,
                "cooldown_state_source": "local_fallback",
            },
        ]
        skipped = selection._build_auto_agent_skipped_candidates_from_states(states)
        assert len(skipped) == 1
        assert skipped[0]["provider"] == "openai"
        assert skipped[0]["reason"] == "cooldown"
        assert skipped[0]["cooldown_state_source"] == "memory"

    def test_skip_reason_propagated(self):
        states = [
            {
                "candidate": _candidate("xai", "grok-4"),
                "lane_key": "xai:auth_degraded",
                "cooldown_seconds": 300.0,
                "skip_reason": "auth_degraded",
                "cooldown_state_source": "auth_degraded",
                "failure_phase": "auth",
                "attempted_provider_call": False,
            },
        ]
        skipped = selection._build_auto_agent_skipped_candidates_from_states(states)
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "auth_degraded"
        assert skipped[0]["failure_phase"] == "auth"
        assert skipped[0]["attempted_provider_call"] is False


# ---------------------------------------------------------------------------
# Request-local cooldown/exclusion
# ---------------------------------------------------------------------------


class TestRequestLocalCooldown:
    def test_set_and_get_cooldown(self):
        request = _make_request()
        key = "openai:gpt-4o:openai:primary"
        selection._set_codex_auto_agent_request_local_cooldown(
            request, cooldown_key=key, cooldown_seconds=30.0
        )
        remaining = selection._get_codex_auto_agent_request_local_cooldown_seconds(
            request, cooldown_key=key
        )
        assert remaining > 29.0

    def test_exclude_candidate(self):
        request = _make_request()
        key = "openai:gpt-4o:openai:primary"
        selection._exclude_codex_auto_agent_request_local_candidate(request, cooldown_key=key)
        assert key in selection._get_codex_auto_agent_request_local_excluded_keys(request)

    def test_apply_request_local_cooldown_from_plan(self):
        request = _make_request()
        candidate = _candidate()
        selection._apply_request_local_cooldown_from_plan(
            request, candidate=candidate, lane_key="openai:primary", cooldown_seconds=15.0
        )
        key = selection._get_codex_auto_agent_request_local_cooldown_key(
            candidate=candidate, lane_key="openai:primary"
        )
        assert key in selection._get_codex_auto_agent_request_local_excluded_keys(request)
        remaining = selection._get_codex_auto_agent_request_local_cooldown_seconds(
            request, cooldown_key=key
        )
        assert remaining > 14.0

    def test_request_local_state_applied_in_candidate_state(self):
        request = _make_request()
        candidate = _candidate()
        lane_key = "openai:primary"
        rl_key = selection._get_codex_auto_agent_request_local_cooldown_key(
            candidate=candidate, lane_key=lane_key
        )
        selection._set_codex_auto_agent_request_local_cooldown(
            request, cooldown_key=rl_key, cooldown_seconds=20.0
        )
        selection._exclude_codex_auto_agent_request_local_candidate(request, cooldown_key=rl_key)
        cd, src, skip = selection._apply_codex_auto_agent_request_local_candidate_state(
            request,
            candidate=candidate,
            lane_key=lane_key,
            cooldown_seconds=0.0,
            cooldown_state_source=None,
            skip_reason=None,
        )
        assert cd > 19.0
        assert src == "request_local"
        assert skip == "request_local_transient_failure"


# ---------------------------------------------------------------------------
# State source attachment
# ---------------------------------------------------------------------------


class TestStateSourceAttachment:
    def test_attach_affinity_and_cooldown_sources(self):
        sel = {"candidate": _candidate()}
        enriched = selection._attach_aawm_alias_routing_state_sources(
            sel,
            affinity={"affinity_state_source": "memory"},
            selected_state={"cooldown_state_source": "durable"},
        )
        assert enriched["affinity_state_source"] == "memory"
        assert enriched["cooldown_state_source"] == "durable"

    def test_defaults_to_local_fallback(self):
        sel = {"candidate": _candidate()}
        enriched = selection._attach_aawm_alias_routing_state_sources(
            sel,
            affinity={},
            selected_state={},
        )
        assert enriched["affinity_state_source"] == "local_fallback"
        assert enriched["cooldown_state_source"] == "local_fallback"


# ---------------------------------------------------------------------------
# In-flight cooldown errors
# ---------------------------------------------------------------------------


class TestInFlightCooldownErrors:
    def test_codex_in_flight_cooldown_raises_429(self):
        with pytest.raises(HTTPException) as exc_info:
            selection._raise_codex_auto_agent_in_flight_cooldown(
                candidate=_candidate(),
                lane_key="openai:primary",
                cooldown_seconds=30.0,
            )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["error"]["code"] == "aawm_codex_auto_agent_in_flight_provider_cooling_down"
        assert exc.headers["Retry-After"] == "30"

    def test_anthropic_in_flight_cooldown_raises_429(self):
        with pytest.raises(HTTPException) as exc_info:
            selection._raise_anthropic_auto_agent_in_flight_cooldown(
                candidate=_candidate("anthropic", "claude-sonnet-4-20250514"),
                lane_key="anthropic:primary",
                cooldown_seconds=15.0,
            )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["error"]["code"] == "aawm_anthropic_auto_agent_in_flight_provider_cooling_down"
        assert exc.headers["Retry-After"] == "15"


# ---------------------------------------------------------------------------
# Redispatch errors
# ---------------------------------------------------------------------------


class TestRedispatchErrors:
    def test_codex_redispatch_required(self):
        with pytest.raises(HTTPException) as exc_info:
            selection._raise_codex_auto_agent_redispatch_required(
                candidate=_candidate(),
                lane_key="openai:primary",
                cooldown_seconds=60.0,
                error_tokens={"429", "RATE_LIMIT_EXCEEDED"},
                alias_model="basic",
                error_class="rate_limited",
            )
        exc = exc_info.value
        assert exc.status_code == 429
        detail = exc.detail
        assert detail["redispatch_required"] is True
        assert detail["error"]["code"] == "aawm_codex_auto_agent_redispatch_required"
        assert detail["alias_family"] == "codex_auto_agent"
        assert detail["failure_class"] == "rate_limited"
        assert "429" in detail["error_tokens"]

    def test_anthropic_redispatch_required(self):
        with pytest.raises(HTTPException) as exc_info:
            selection._raise_anthropic_auto_agent_redispatch_required(
                candidate=_candidate("anthropic", "claude-sonnet-4-20250514"),
                lane_key=None,
                cooldown_seconds=45.0,
                error_tokens=set(),
                alias_model="basic",
            )
        exc = exc_info.value
        assert exc.status_code == 429
        detail = exc.detail
        assert detail["redispatch_required"] is True
        assert detail["error"]["code"] == "aawm_anthropic_auto_agent_redispatch_required"
        assert detail["alias_family"] == "anthropic_auto_agent"


# ---------------------------------------------------------------------------
# Codex selector: first-choice
# ---------------------------------------------------------------------------


class TestCodexSelectorFirstChoice:
    @pytest.mark.asyncio
    async def test_first_available_selected(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4", last_resort=True),
        )
        # Patch the enumeration to return our candidates
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)
        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            result = await selection._select_codex_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
            )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "openai"
        assert result["candidate"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Codex selector: last-resort
# ---------------------------------------------------------------------------


class TestCodexSelectorLastResort:
    @pytest.mark.asyncio
    async def test_last_resort_when_first_cooled(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4", last_resort=True),
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        # Make openai candidate cooled
        async def _codex_cooldown(key: str) -> tuple[float, str]:
            if "gpt-4o" in key:
                return (60.0, "memory")
            return (0.0, "local_fallback")

        _set_selection_runtime(
            "_get_codex_active_cooldown_state",
            _codex_cooldown,
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            result = await selection._select_codex_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "last_resort"
        assert result["candidate"]["provider"] == "xai"

    @pytest.mark.asyncio
    async def test_last_resort_bypasses_its_own_cooldown(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4", last_resort=True),
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            SelectionEnumeration,
        )

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        async def _all_cooled(key: str) -> tuple[float, str]:
            return (120.0, "memory")

        _set_selection_runtime(
            "_get_codex_active_cooldown_state",
            _all_cooled,
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            result = await selection._select_codex_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
            )
        assert result["selection_reason"] == "last_resort"
        assert result["candidate"]["provider"] == "xai"


# ---------------------------------------------------------------------------
# Codex selector: all-cooled
# ---------------------------------------------------------------------------


class TestCodexSelectorAllCooled:
    @pytest.mark.asyncio
    async def test_all_cooled_raises_429(self):
        request = _make_request()
        candidates = (_candidate("openai", "gpt-4o"),)
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        async def _all_cooled(key: str) -> tuple[float, str]:
            return (120.0, "memory")

        _set_selection_runtime(
            "_get_codex_active_cooldown_state",
            _all_cooled,
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["error"]["code"] == "aawm_codex_auto_agent_all_candidates_cooling_down"


# ---------------------------------------------------------------------------
# Codex selector: request-local exclusion
# ---------------------------------------------------------------------------


class TestCodexSelectorRequestLocalExclusion:
    @pytest.mark.asyncio
    async def test_request_local_exclusion_skips_candidate(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4"),
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        # Pre-exclude the openai candidate via request-local state
        rl_key = selection._get_codex_auto_agent_request_local_cooldown_key(
            candidate=_candidate("openai", "gpt-4o"),
            lane_key="openai:primary",
        )
        selection._exclude_codex_auto_agent_request_local_candidate(request, cooldown_key=rl_key)

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            result = await selection._select_codex_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
            )
        # openai excluded, xai selected
        assert result["candidate"]["provider"] == "xai"


class TestCodexSelectorRequestLocalLastResortSkip:
    @pytest.mark.asyncio
    async def test_last_resort_request_local_skipped(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4", last_resort=True),
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            SelectionEnumeration,
        )

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)
        last_resort = _candidate("xai", "grok-4", last_resort=True)
        rl_key = selection._get_codex_auto_agent_request_local_cooldown_key(
            candidate=last_resort,
            lane_key="xai:default",
        )
        selection._exclude_codex_auto_agent_request_local_candidate(
            request,
            cooldown_key=rl_key,
        )

        async def _cooldown(key: str) -> tuple[float, str]:
            return (60.0, "memory")

        _set_selection_runtime(
            "_get_codex_active_cooldown_state",
            _cooldown,
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                )
        assert exc_info.value.status_code == 429
        skipped = exc_info.value.detail["candidates"]
        assert any(
            candidate.get("reason") == "request_local_transient_failure"
            for candidate in skipped
        )


class TestCodexSelectorLastResortSkipBlocking:
    def test_select_available_state_rejects_quota_skipped_last_resort(self):
        request = _make_request()
        states = [
            {
                "candidate": _candidate("xai", "grok-4", last_resort=True),
                "cooldown_seconds": 0.0,
                "skip_reason": "quota_exhausted",
            },
        ]
        assert (
            selection._select_available_state(
                request,
                states,
                ingress="codex",
                last_resort=True,
            )
            is None
        )

    def test_select_available_state_rejects_auth_skipped_last_resort(self):
        request = _make_request()
        states = [
            {
                "candidate": _candidate("xai", "grok-4", last_resort=True),
                "cooldown_seconds": 300.0,
                "skip_reason": "auth_degraded",
            },
        ]
        assert (
            selection._select_available_state(
                request,
                states,
                ingress="codex",
                last_resort=True,
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_codex_last_resort_quota_skip_stays_unavailable(self):
        request = _make_request()
        states = [
            {
                "candidate": _candidate("xai", "grok-4", last_resort=True),
                "lane_key": "xai:default",
                "cooldown_seconds": 0.0,
                "cooldown_state_source": "normalized_quota_observation",
                "skip_reason": "quota_exhausted",
            },
        ]

        async def _states(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return states

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {"_build_codex_auto_agent_candidate_states": _states},
        ):
            with pytest.raises(HTTPException) as exc_info:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                )
        exc = exc_info.value
        assert exc.status_code == 429
        assert (
            exc.detail["error"]["code"]
            == "aawm_codex_auto_agent_all_candidates_cooling_down"
        )
        assert any(
            candidate.get("reason") == "quota_exhausted"
            for candidate in exc.detail["candidates"]
        )


# ---------------------------------------------------------------------------
# Anthropic selector: first-choice
# ---------------------------------------------------------------------------


class TestAnthropicSelectorFirstChoice:
    @pytest.mark.asyncio
    async def test_first_available_selected(self):
        request = _make_request()
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o", last_resort=True),
        )
        _set_selection_candidates(candidates)

        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Anthropic selector: all-cooled
# ---------------------------------------------------------------------------


class TestAnthropicSelectorAllCooled:
    @pytest.mark.asyncio
    async def test_all_cooled_raises_429(self):
        request = _make_request()
        candidates = (_candidate("anthropic", "claude-sonnet-4-20250514"),)
        _set_selection_candidates(candidates)

        async def _all_cooled(key: str) -> tuple[float, str]:
            return (120.0, "memory")

        _set_selection_runtime(
            "_get_anthropic_active_cooldown_state",
            _all_cooled,
        )
        _set_selection_runtime(
            "_get_anthropic_merged_codex_openai_cooldown_state",
            _all_cooled,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_anthropic_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
            )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["error"]["code"] == "aawm_anthropic_auto_agent_all_candidates_cooling_down"


class TestAnthropicSelectorLastResort:
    @pytest.mark.asyncio
    async def test_last_resort_bypasses_its_own_cooldown(self):
        request = _make_request()
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o", last_resort=True),
        )
        _set_selection_candidates(candidates)

        async def _all_cooled(key: str) -> tuple[float, str]:
            return (120.0, "memory")

        _set_selection_runtime(
            "_get_anthropic_active_cooldown_state",
            _all_cooled,
        )
        _set_selection_runtime(
            "_get_anthropic_merged_codex_openai_cooldown_state",
            _all_cooled,
        )

        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "last_resort"
        assert result["candidate"]["provider"] == "openai"


# ---------------------------------------------------------------------------
# Anthropic in-flight cooldown
# ---------------------------------------------------------------------------


class TestAnthropicInFlight:
    @pytest.mark.asyncio
    async def test_in_flight_affinity_cooldown_raises(self):
        request = _make_request()
        affinity = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "route_family": "anthropic_responses_adapter",
            "last_resort": False,
            "affinity_state_source": "memory",
        }
        candidates = (_candidate("anthropic", "claude-sonnet-4-20250514"),)
        _set_selection_candidates(candidates)
        _set_selection_runtime(
            "_get_anthropic_session_affinity",
            AsyncMock(return_value=affinity),
        )
        _set_selection_runtime("_has_continuation_state", lambda v: True)
        _set_selection_runtime(
            "_resolve_anthropic_session_key",
            lambda r, b, *, alias_model: "session:123",
        )

        async def _cooled(key: str) -> tuple[float, str]:
            return (30.0, "memory")

        _set_selection_runtime(
            "_get_anthropic_active_cooldown_state",
            _cooled,
        )
        _set_selection_runtime(
            "_get_anthropic_merged_codex_openai_cooldown_state",
            _cooled,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_anthropic_auto_agent_candidate(
                request=request,
                request_body={"model": "basic"},
            )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["error"]["code"] == "aawm_anthropic_auto_agent_in_flight_provider_cooling_down"


# ---------------------------------------------------------------------------
# Codex redispatch (affinity removed)
# ---------------------------------------------------------------------------


class TestCodexRedispatch:
    @pytest.mark.asyncio
    async def test_affinity_removed_raises_redispatch(self):
        request = _make_request()
        affinity = {
            "provider": "openai",
            "model": "gpt-4o-removed",
            "route_family": "openai_responses_adapter",
            "last_resort": False,
            "affinity_state_source": "memory",
        }
        # Enumeration has no matching candidate
        candidates = (_candidate("openai", "gpt-4o"),)
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)
        _set_selection_runtime(
            "_get_codex_session_affinity",
            AsyncMock(return_value=affinity),
        )
        _set_selection_runtime("_has_continuation_state", lambda v: True)
        _set_selection_runtime(
            "_resolve_codex_session_key",
            lambda r, b, *, alias_model: "session:456",
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                )
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                )
        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail["redispatch_required"] is True
        assert exc.detail["error"]["code"] == "aawm_codex_auto_agent_redispatch_required"
        assert exc.detail["failure_phase"] == "affinity_continuation_removed"

    @pytest.mark.asyncio
    async def test_durable_affinity_removed_raises_before_alternate_selection(self):
        request = _make_request()
        affinity = {
            "provider": "openai",
            "model": "removed-durable-model",
            "route_family": "openai_responses_adapter",
            "last_resort": False,
            "affinity_state_source": "durable_cache",
        }
        candidates = (_candidate("openai", "gpt-4o"),)
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import SelectionEnumeration

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)
        alternate_states = AsyncMock(
            side_effect=AssertionError("alternate selection must not run")
        )
        _set_selection_runtime(
            "_get_codex_session_affinity",
            AsyncMock(return_value=affinity),
        )
        _set_selection_runtime("_has_continuation_state", lambda v: True)
        _set_selection_runtime(
            "_resolve_codex_session_key",
            lambda r, b, *, alias_model: "session:durable-removed",
        )

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: mock_enum
                ),
                "_build_codex_auto_agent_candidate_states": alternate_states,
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                )

        detail = exc_info.value.detail
        assert exc_info.value.status_code == 429
        assert detail["redispatch_required"] is True
        assert detail["failure_phase"] == "affinity_continuation_removed"
        assert detail["attempted_provider_call"] is False
        alternate_states.assert_not_awaited()


# ---------------------------------------------------------------------------
# Kimi managed-account lane
# ---------------------------------------------------------------------------


class TestKimiManagedAccount:
    @pytest.mark.asyncio
    async def test_kimi_managed_account_cooldown(self):
        async def _kimi_cooled(key: str) -> tuple[float, str]:
            if "__managed_account__" in key:
                return (90.0, "memory")
            return (0.0, "local_fallback")

        cd, src, skip, scope = await selection._apply_kimi_code_managed_account_lane_cooldown(
            candidate=_candidate("kimi_code", "kimi-k2"),
            cooldown_seconds=0.0,
            cooldown_state_source=None,
            skip_reason=None,
            get_active_cooldown_state=_kimi_cooled,
        )
        assert cd == 90.0
        assert src == "kimi_managed_account:memory"
        assert scope == "managed_account"

    @pytest.mark.asyncio
    async def test_non_kimi_candidate_unchanged(self):
        async def _zero(key: str) -> tuple[float, str]:
            return (0.0, "local_fallback")

        cd, src, skip, scope = await selection._apply_kimi_code_managed_account_lane_cooldown(
            candidate=_candidate("openai", "gpt-4o"),
            cooldown_seconds=5.0,
            cooldown_state_source="memory",
            skip_reason=None,
            get_active_cooldown_state=_zero,
        )
        assert cd == 5.0
        assert src == "memory"
        assert scope is None


# ---------------------------------------------------------------------------
# Find candidate
# ---------------------------------------------------------------------------


class TestFindCandidate:
    def test_find_anthropic_candidate(self):
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o"),
        )
        _set_selection_candidates(candidates)
        found = selection._find_anthropic_auto_agent_candidate(
            "anthropic",
            "claude-sonnet-4-20250514",
            alias_model="basic",
            request=_make_request(),
        )
        assert found is not None
        assert found["provider"] == "anthropic"

    def test_find_anthropic_candidate_not_found(self):
        _set_selection_candidates(())
        found = selection._find_anthropic_auto_agent_candidate(
            "anthropic",
            "nonexistent",
            alias_model="basic",
            request=_make_request(),
        )
        assert found is None
