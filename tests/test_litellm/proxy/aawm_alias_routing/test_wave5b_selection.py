"""Module-local tests for Wave 5B selection.py extraction.

Drives the new module directly with fresh state/dependency stubs.
Does NOT import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state,
    durable,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    SelectionEnumeration,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
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
    _set_selection_runtime_value(name, value)


def _set_selection_runtime_value(
    name: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    target_globals = selection._select_codex_auto_agent_candidate.__globals__
    if monkeypatch is None:
        setattr(selection, name, value)
        target_globals[name] = value
        return
    monkeypatch.setattr(selection, name, value)
    monkeypatch.setitem(target_globals, name, value)


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
    previous_alias_routing_state = selection.alias_routing_state
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
        "_has_account_bound_state",
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
    runtime_globals = selection._build_codex_auto_agent_candidate_state.__globals__
    _MISSING = object()
    previous_runtime_globals = {
        name: runtime_globals.get(name, _MISSING) for name in runtime_names
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
        "_has_account_bound_state": lambda v: False,
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
        has_account_bound_state=lambda v: False,
        is_grok_account_quota_candidate=lambda c: False,
        get_grok_account_quota_lane_cooldown_key=lambda c, lk: None,
        is_kimi_code_candidate=lambda c: isinstance(c, dict) and c.get("provider") == "kimi_code",
        get_kimi_managed_account_cooldown_key=lambda: "kimi_code:__managed_account__:kimi_code_managed_account",
        get_codex_quota_observation_pool=None,
        get_codex_quota_observation_environment=None,
    )
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
                lambda candidate, lane_key, cooldown_identity_tag=None: (
                    f"{candidate.get('provider')}:{candidate.get('model')}:{lane_key}"
                )
            ),
        }
    )
    try:
        with patch.dict(runtime_globals, runtime):
            yield
    finally:
        selection.alias_routing_state = previous_alias_routing_state
        for name, value in previous_runtime.items():
            setattr(selection, name, value)
        for name, value in previous_runtime_globals.items():
            if value is _MISSING:
                runtime_globals.pop(name, None)
            else:
                runtime_globals[name] = value


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


class TestCodexRequestRedispatchOrdinal:
    @pytest.mark.parametrize(
        "request_body,expected",
        [
            ({"litellm_metadata": {"redispatch_ordinal": 2}}, 2),
            ({"litellm_metadata": {"agent_redispatch_ordinal": "2"}}, 2),
            ({"litellm_metadata": {"redispatch_ordinal": True}}, None),
            ({"litellm_metadata": {"redispatch_ordinal": 0}}, None),
            ({"litellm_metadata": {"redispatch_ordinal": -1}}, None),
            ({"litellm_metadata": {"redispatch_ordinal": "2.5"}}, None),
            ({"litellm_metadata": {"redispatch_ordinal": "bad"}}, None),
            ({"litellm_metadata": {"redispatch_ordinal": float("nan")}}, None),
            ({"litellm_metadata": {"dispatch_ordinal": float("inf")}}, None),
        ],
    )
    def test_extract_codex_request_redispatch_ordinal(self, request_body, expected):
        assert (
            selection._extract_codex_request_redispatch_ordinal(request_body) == expected
        )


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
        _set_selection_runtime("_has_continuation_state", lambda v: True)
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
        assert result["request_mode"] == "ordinary_continuation"
        assert result["redispatch_ordinal"] is None
        assert result["affinity_bypassed"] is False

    @pytest.mark.asyncio
    async def test_fresh_redispatch_ordinal_falls_back_to_next_candidate(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4"),
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            SelectionEnumeration,
        )

        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        async def _codex_cooldown(key: str) -> tuple[float, str]:
            if "gpt-4o" in key:
                return (60.0, "memory")
            return (0.0, "local_fallback")

        _set_selection_runtime("_has_continuation_state", lambda v: True)
        _set_selection_runtime(
            "_get_codex_session_affinity",
            AsyncMock(
                return_value={
                    "provider": "openai",
                    "model": "gpt-4o",
                    "route_family": "openai_responses_adapter",
                    "last_resort": False,
                }
            ),
        )
        _set_selection_runtime("_get_codex_active_cooldown_state", _codex_cooldown)

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
                request_body={"model": "basic", "litellm_metadata": {"redispatch_ordinal": "2"}},
            )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "xai"
        assert result["request_mode"] == "fresh_redispatch"
        assert result["redispatch_ordinal"] == 2
        assert result["affinity_bypassed"] is True

    @pytest.mark.asyncio
    async def test_deterministic_exclusion_advances_real_selector(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4"),
        )
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
                excluded_candidate_keys=frozenset(
                    {"openai:gpt-4o:openai:primary"}
                ),
            )

        assert result["candidate"]["provider"] == "xai"
        assert result["skipped"][0]["reason"] == "candidate_ineligible"
        assert result["skipped"][0]["cooldown_state_source"] == "local_fallback"
        assert (
            getattr(
                request.state,
                "aawm_alias_request_local_cooldown_until",
                None,
            )
            is None
        )
        assert (
            getattr(
                request.state,
                "aawm_alias_request_local_excluded_keys",
                None,
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_excluded_candidate_keeps_active_cooldown_reason(self):
        request = _make_request()
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4"),
        )
        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)

        async def _codex_cooldown(key: str) -> tuple[float, str]:
            if "gpt-4o" in key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        _set_selection_runtime("_get_codex_active_cooldown_state", _codex_cooldown)

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
                excluded_candidate_keys=frozenset(
                    {"openai:gpt-4o:openai:primary"}
                ),
            )

        assert result["candidate"]["provider"] == "xai"
        skipped = next(
            candidate
            for candidate in result["skipped"]
            if candidate["provider"] == "openai"
        )
        assert skipped["reason"] == "cooldown"
        assert skipped["cooldown_seconds"] == 60.0
        assert skipped["cooldown_state_source"] == "durable_cache"

    @pytest.mark.asyncio
    async def test_alibaba_account_cooldown_suppresses_both_models_including_last_resort(
        self,
    ) -> None:
        request = _make_request()
        candidates = (
            {
                "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
                "model": "alibaba_token_plan/qwen3.8-max",
                "route_family": "alibaba_token_plan_chat_completions_adapter",
                "last_resort": False,
            },
            {
                "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
                "model": "alibaba_token_plan/qwen3.7-max",
                "route_family": "alibaba_token_plan_chat_completions_adapter",
                "last_resort": True,
            },
        )
        states: list[dict[str, Any]] = []
        state_keys: list[str] = []

        async def _cooldown_state(cooldown_key: str) -> tuple[float, str]:
            if cooldown_key == CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY:
                return (8434.5, "memory")
            return (0.0, "local_fallback")

        with patch.dict(
            selection._select_codex_auto_agent_candidate.__globals__,
            {
                "_resolve_aawm_alias_selection_enumeration": (
                    lambda request, canonical_alias, *, ingress, client_product_label=None: (
                        SelectionEnumeration(candidates=candidates, commit_token=None)
                    )
                ),
                "_get_codex_active_cooldown_state": _cooldown_state,
            },
        ):
            for candidate in candidates:
                state = await selection._build_codex_auto_agent_candidate_state(
                    request,
                    candidate_template=candidate,
                )
                states.append(state)
                state_keys.append(state["cooldown_key"])

        assert state_keys == [
            "alibaba_token_plan:alibaba_token_plan/qwen3.8-max:alibaba_token_plan",
            "alibaba_token_plan:alibaba_token_plan/qwen3.7-max:alibaba_token_plan",
        ]
        assert all(state["skip_reason"] == "account_quota_cooldown" for state in states)
        assert all(
            state["cooldown_seconds"] == 8434.5
            and state["cooldown_state_source"] == "alibaba_token_plan_account:memory"
            for state in states
        )

    @pytest.mark.asyncio
    async def test_alibaba_account_cooldown_hydrates_from_durable_cache_for_fresh_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        request = _make_request()
        candidates = (
            {
                "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
                "model": "alibaba_token_plan/qwen3.8-max",
                "route_family": "alibaba_token_plan_chat_completions_adapter",
                "last_resort": False,
            },
            {
                "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
                "model": "alibaba_token_plan/qwen3.7-max",
                "route_family": "alibaba_token_plan_chat_completions_adapter",
                "last_resort": True,
            },
        )
        durable_reads: list[str] = []
        durable_cache_reads: list[str] = []
        manager = AliasRoutingStateManager()
        previous_manager = cooldown_state._manager
        payload = {"expires_at_epoch": time.time() + 8434.5}
        canonical_cache_key = durable.build_aawm_alias_routing_durable_cache_key(
            alias_family="codex",
            state_kind="cooldown",
            state_key=CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY,
        )

        class _DurableCache:
            async def async_get_cache(self, *, key: str, **kwargs: Any) -> Any:
                durable_cache_reads.append(key)
                if key == canonical_cache_key:
                    return payload
                return None

        durable_cache = _DurableCache()

        async def _read_state(**kwargs: Any) -> dict[str, Any]:
            state_key = kwargs["state_key"]
            durable_reads.append(state_key)
            return await durable.read_aawm_alias_routing_state(**kwargs)

        async def _active_cooldown_state(key: str) -> tuple[float, str]:
            return await cooldown_state._get_codex_auto_agent_active_cooldown_state(
                key,
                _dual_cache_fn=lambda: durable_cache,
                _read_state_fn=lambda: _read_state,
            )

        cooldown_state.configure_cooldown_state_runtime(manager=manager)
        _set_selection_runtime("_get_codex_active_cooldown_state", _active_cooldown_state)
        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.get_session_owner_record",
            AsyncMock(return_value=(None, None, None)),
        )
        mock_enum = SelectionEnumeration(candidates=candidates, commit_token=None)
        try:
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
        finally:
            cooldown_state._manager = previous_manager

        skipped = {
            candidate["model"]: candidate for candidate in exc_info.value.detail["candidates"]
        }
        assert set(skipped) == {
            "alibaba_token_plan/qwen3.8-max",
            "alibaba_token_plan/qwen3.7-max",
        }
        assert all(
            candidate["reason"] == "account_quota_cooldown"
            and candidate["cooldown_seconds"] > 0
            for candidate in skipped.values()
        )
        assert skipped["alibaba_token_plan/qwen3.8-max"][
            "cooldown_state_source"
        ] == "alibaba_token_plan_account:durable_cache"
        assert skipped["alibaba_token_plan/qwen3.7-max"][
            "cooldown_state_source"
        ] in {
            "alibaba_token_plan_account:durable_cache",
            "alibaba_token_plan_account:memory",
        }
        assert (
            durable_reads.count(
                CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
            )
            >= 1
        )
        assert durable_cache_reads.count(canonical_cache_key) >= 1
        assert (
            manager.codex.cooldown_until_monotonic_by_key[
                CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
            ]
            > time.monotonic()
        )

    @pytest.mark.asyncio
    async def test_excluded_candidate_prefers_normalized_quota_terminal_reset(self):
        request = _make_request()
        candidate = _oauth_account_candidate()
        cooldown_key = "openai:gpt-5.3-codex:codex-oauth:account1:hash-account-1"
        quota_windows = [
            {
                "quota_period": "five_hour",
                "window_minutes": 300,
                "status": "fresh",
                "exhausted": True,
                "remaining_pct": 0,
                "reset_at": "2030-01-01T00:00:00Z",
            }
        ]
        _set_selection_candidates((candidate,))

        async def _candidate_state(
            _request,
            *,
            candidate_template,
            openai_lane_key=None,
            excluded_candidate_keys=None,
        ):
            lane_key = openai_lane_key or candidate_template["codex_oauth_lane_key"]
            state = {
                "candidate": candidate_template,
                "lane_key": lane_key,
                "cooldown_key": cooldown_key,
                "cooldown_seconds": 0.0,
                "cooldown_state_source": "local_fallback",
            }
            if excluded_candidate_keys and cooldown_key in excluded_candidate_keys:
                state["skip_reason"] = "candidate_ineligible"
            return state

        def _quota_state(state, *, account_hash=None):
            if account_hash == "hash-account-1":
                state["quota_exhausted_windows"] = quota_windows
            return state

        async def _candidate_contexts(_request, *, candidate_template, affinity=None):
            return [
                {
                    "candidate": candidate_template,
                    "lane_key": candidate_template["codex_oauth_lane_key"],
                    "auth_status": "ready",
                }
            ]

        restored = {
            "_build_codex_auto_agent_candidate_state": selection._build_codex_auto_agent_candidate_state,
            "_resolve_codex_oauth_account_candidate_contexts": selection._resolve_codex_oauth_account_candidate_contexts,
            "_hydrate_codex_oauth_quota_observations": selection._hydrate_codex_oauth_quota_observations,
            "_attach_normalized_quota_state": selection._attach_normalized_quota_state,
        }
        try:
            _set_selection_runtime(
                "_build_codex_auto_agent_candidate_state", _candidate_state
            )
            _set_selection_runtime(
                "_resolve_codex_oauth_account_candidate_contexts", _candidate_contexts
            )
            _set_selection_runtime(
                "_hydrate_codex_oauth_quota_observations", AsyncMock(return_value=None)
            )
            _set_selection_runtime("_attach_normalized_quota_state", _quota_state)

            with pytest.raises(HTTPException) as caught:
                await selection._select_codex_auto_agent_candidate(
                    request=request,
                    request_body={"model": "basic"},
                    excluded_candidate_keys=frozenset({cooldown_key}),
                )
        finally:
            for name, value in restored.items():
                _set_selection_runtime(name, value)

        assert caught.value.status_code == 429
        detail = caught.value.detail
        assert detail["candidates"][-1]["reason"] == "quota_exhausted"
        assert (
            detail["candidates"][-1]["cooldown_state_source"]
            == "normalized_quota_observation"
        )
        assert detail["terminal_reset"]["reason"] == "codex_oauth_quota_exhausted"
        assert detail["terminal_reset"]["next_reset_at"] == "2030-01-01T00:00:00Z"
        assert detail["terminal_reset"]["accounts"] == [
            {
                "account_hash": "hash-account-1",
                "account_label": "account1",
                "account_lane": "codex-oauth:account1:hash-account-1",
                "exhausted_windows": quota_windows,
            }
        ]

    def test_non_401_provider_terminal_error_does_not_plan_account_rotation(self):
        request = _make_request()
        candidate = _oauth_account_candidate()
        selection_state = {
            "candidate": candidate,
            "failover_ordinal": 0,
            "has_account_bound_state": False,
        }
        attempt = {
            "failure_phase": "direct_openai_provider_response",
            "attempted_provider_call": True,
            "error_status_code": 403,
        }

        assert not selection._plan_codex_oauth_account_failover(
            request,
            candidate=candidate,
            selection=selection_state,
            attempt_record=attempt,
            error_class="provider_terminal_error",
            has_continuation_state=False,
            has_previous_response_id=False,
            has_account_bound_state=False,
            provider_status_code=403,
        )
        assert not hasattr(
            request.state, "aawm_codex_oauth_request_local_failover_context"
        )


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
        _set_selection_runtime("_has_continuation_state", lambda v: True)

        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "anthropic"
        assert result["request_mode"] == "ordinary_continuation"
        assert result["redispatch_ordinal"] is None
        assert result["affinity_bypassed"] is False

    @pytest.mark.asyncio
    async def test_fresh_redispatch_ordinal_falls_back_to_next_candidate(self):
        request = _make_request()
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o"),
        )
        _set_selection_candidates(candidates)

        async def _anthropic_cooldown(key: str) -> tuple[float, str]:
            if "claude" in key:
                return (120.0, "memory")
            return (0.0, "local_fallback")

        async def _openai_merged_cooldown(key: str) -> tuple[float, str]:
            return (0.0, "local_fallback")

        _set_selection_runtime("_has_continuation_state", lambda v: True)
        _set_selection_runtime(
            "_get_anthropic_session_affinity",
            AsyncMock(
                return_value={
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "route_family": "anthropic_responses_adapter",
                    "last_resort": False,
                }
            ),
        )
        _set_selection_runtime("_get_anthropic_active_cooldown_state", _anthropic_cooldown)
        _set_selection_runtime(
            "_get_anthropic_merged_codex_openai_cooldown_state",
            _openai_merged_cooldown,
        )

        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic", "litellm_metadata": {"redispatch_ordinal": "2"}},
        )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "openai"
        assert result["request_mode"] == "fresh_redispatch"
        assert result["redispatch_ordinal"] == 2
        assert result["affinity_bypassed"] is True

    @pytest.mark.asyncio
    async def test_deterministic_exclusion_advances_real_selector(self):
        request = _make_request()
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o"),
        )
        _set_selection_candidates(candidates)
        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic"},
            excluded_candidate_keys=frozenset(
                {"anthropic:claude-sonnet-4-20250514:anthropic:primary"}
            ),
        )

        assert result["candidate"]["provider"] == "openai"
        assert result["skipped"][0]["reason"] == "candidate_ineligible"
        assert result["skipped"][0]["cooldown_state_source"] == "local_fallback"

    @pytest.mark.asyncio
    async def test_excluded_candidate_keeps_active_cooldown_reason(self):
        request = _make_request()
        candidates = (
            _candidate("anthropic", "claude-sonnet-4-20250514"),
            _candidate("openai", "gpt-4o"),
        )
        _set_selection_candidates(candidates)

        async def _cooled(key: str) -> tuple[float, str]:
            if "claude" in key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        _set_selection_runtime(
            "_get_anthropic_active_cooldown_state",
            _cooled,
        )
        _set_selection_runtime(
            "_get_anthropic_merged_codex_openai_cooldown_state",
            _cooled,
        )
        result = await selection._select_anthropic_auto_agent_candidate(
            request=request,
            request_body={"model": "basic"},
            excluded_candidate_keys=frozenset(
                {"anthropic:claude-sonnet-4-20250514:anthropic:primary"}
            ),
        )

        assert result["candidate"]["provider"] == "openai"
        skipped = next(
            candidate
            for candidate in result["skipped"]
            if candidate["provider"] == "anthropic"
        )
        assert skipped["reason"] == "cooldown"
        assert skipped["cooldown_seconds"] == 60.0
        assert skipped["cooldown_state_source"] == "durable_cache"


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

        with patch.dict(
            selection._select_anthropic_auto_agent_candidate.__globals__,
            {
                "_find_anthropic_auto_agent_affinity_candidate": (
                    lambda affinity, *, alias_model, client_product_label=None, request: _candidate(
                        "anthropic", "claude-sonnet-4-20250514"
                    )
                )
            },
        ):
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
        assert skip == "managed_account_cooldown"
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
        assert skip is None
        assert scope is None

    @pytest.mark.asyncio
    async def test_managed_account_cooldown_excludes_last_resort_kimi(self):
        candidates = (
            _candidate("kimi_code", "kimi-k2"),
            _candidate(
                "kimi_code",
                "kimi-for-coding",
                last_resort=True,
            ),
        )
        _set_selection_candidates(candidates)

        async def _kimi_managed_cooldown(key: str) -> tuple[float, str]:
            if "__managed_account__" in key:
                return (90.0, "memory")
            return (0.0, "local_fallback")

        _set_selection_runtime(
            "_get_codex_active_cooldown_state",
            _kimi_managed_cooldown,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic"},
            )

        detail = exc_info.value.detail
        assert exc_info.value.status_code == 429
        assert detail["error"]["code"] == (
            "aawm_codex_auto_agent_all_candidates_cooling_down"
        )
        assert {
            candidate["model"] for candidate in detail["candidates"]
        } == {
            "kimi-k2",
            "kimi-for-coding",
        }
        assert all(
            candidate["reason"] == "managed_account_cooldown"
            for candidate in detail["candidates"]
        )
        assert all(
            candidate["cooldown_scope"] == "managed_account"
            for candidate in detail["candidates"]
        )


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


# ---------------------------------------------------------------------------
# Codex account-bound owner selection
# ---------------------------------------------------------------------------


def _oauth_account_candidate(
    *,
    label: str = "account1",
    account_hash: str = "hash-account-1",
) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "route_family": "codex_responses",
        "last_resort": False,
        "codex_oauth_account_label": label,
        "codex_oauth_account_hash": account_hash,
        "codex_oauth_lane_key": f"codex-oauth:{label}:{account_hash}",
        "codex_oauth_credential_affinity": "interchangeable",
    }


def _codex_oauth_quota_observation(
    *,
    observed_at: float,
    expected_reset_at: float,
    environment: str = "prod",
) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "account_hash": "hash-account-1",
        "environment": environment,
        "quota_key": "codex:seven_day",
        "quota_type": "tokens",
        "limit_scope": "secondary",
        "quota_period": "seven_day",
        "window_minutes": 10080,
        "remaining_pct": 75.0,
        "observed_at": observed_at,
        "expected_reset_at": expected_reset_at,
        "status": "fresh",
        "exhausted": False,
        "source": "codex_quota_poll",
    }


def _alibaba_observation(
    *,
    window: str,
    remaining_pct: float,
    observed_at: float,
    environment: str = "prod",
    account_hash: str = "hash-alibaba-1",
    model: str = "alibaba_token_plan/qwen3.8-max",
) -> dict[str, Any]:
    return {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": model,
        "account_hash": account_hash,
        "environment": environment,
        "quota_key": f"alibaba_token_plan_{window}:credits",
        "quota_period": window,
        "quota_type": "credits",
        "remaining_pct": remaining_pct,
        "observed_at": observed_at,
        "expected_reset_at": observed_at + 3600.0,
        "status": "fresh",
        "exhausted": remaining_pct <= 0,
        "source": "alibaba_token_plan_usage",
    }


def _seed_alibaba_windows(
    *,
    remaining_pct: float,
    observed_at: Optional[float] = None,
    environment: str = "prod",
    models: tuple[str, str] = (
        "alibaba_token_plan/qwen3.8-max",
        "alibaba_token_plan/qwen3.7-max",
    ),
) -> None:
    now = time.time() if observed_at is None else observed_at
    selection.alias_routing_state.record_normalized_quota_observations(
        [
            _alibaba_observation(
                window="5h",
                remaining_pct=remaining_pct,
                observed_at=now,
                environment=environment,
                model=models[0],
            ),
            _alibaba_observation(
                window="7d",
                remaining_pct=remaining_pct,
                observed_at=now,
                environment=environment,
                model=models[1],
            ),
        ]
    )


def _alibaba_candidates() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
            "model": "alibaba_token_plan/qwen3.8-max",
            "route_family": "alibaba_token_plan_chat_completions_adapter",
            "last_resort": False,
        },
        {
            "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
            "model": "alibaba_token_plan/qwen3.7-max",
            "route_family": "alibaba_token_plan_chat_completions_adapter",
            "last_resort": True,
        },
    )


def _alibaba_row(
    *,
    environment: Optional[str] = "prod",
    parser_version: str = "alibaba_token_plan_usage_v3",
    telemetry_status: str = "valid",
) -> dict[str, Any]:
    return {
        "observed_at": time.time() - 10,
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "client": "qwen-cloud-console",
        "model": "alibaba_token_plan/qwen3.8-max",
        "account_hash": "hash-alibaba-1",
        "quota_key": "alibaba_token_plan_5h:credits",
        "quota_period": "5h",
        "quota_type": "credits",
        "expected_reset_at": time.time() + 3600,
        "remaining_pct": 25.0,
        "evidence": {
            "environment": environment,
            "parser_version": parser_version,
            "telemetry_status": telemetry_status,
            "window": "5h",
        },
        "environment": environment,
        "source": "alibaba_token_plan_usage",
    }


@pytest.mark.parametrize(
    ("age_seconds", "reset_offset_seconds", "environment", "expected_valid"),
    [
        (3599.0, 1.0, "prod", True),
        (3601.0, 1.0, "prod", False),
        (1.0, 1.0, "prod", True),
        (1.0, -1.0, "prod", False),
        (-1.0, 1.0, "prod", False),
        (1.0, 1.0, "staging", False),
    ],
)
def test_codex_quota_validity_boundaries_apply_to_both_selection_paths(
    monkeypatch: pytest.MonkeyPatch,
    age_seconds: float,
    reset_offset_seconds: float,
    environment: str,
    expected_valid: bool,
) -> None:
    monkeypatch.setenv(
        "AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS",
        "3600",
    )
    _set_selection_runtime_value(
        "_get_codex_quota_observation_environment",
        lambda: "prod",
        monkeypatch,
    )
    manager = selection.alias_routing_state
    manager.reset_for_tests()
    now = time.time()
    manager.record_normalized_quota_observations(
        [
            _codex_oauth_quota_observation(
                observed_at=now - age_seconds,
                expected_reset_at=now + reset_offset_seconds,
                environment=environment,
            )
        ]
    )
    state = {"candidate": _oauth_account_candidate()}

    dual_family = selection._codex_oauth_dual_family_remaining(state)
    attached = selection._attach_normalized_quota_state(dict(state))

    if expected_valid:
        assert dual_family["overall"] == 75.0
        assert "quota_observation" in attached
    else:
        assert dual_family == {"overall": None, "spark": None}
        assert "quota_observation" not in attached
    manager.reset_for_tests()


def test_codex_quota_hydration_ttl_remains_30_seconds() -> None:
    manager = AliasRoutingStateManager()
    now_monotonic = time.monotonic()

    assert selection._CODEX_OAUTH_QUOTA_CACHE_TTL_SECONDS == 30.0
    assert manager.codex_quota_hydration_due_account_hashes(
        ("hash-account-1",),
        environment="prod",
        now_monotonic=now_monotonic,
    ) == ("hash-account-1",)

    manager.defer_codex_quota_hydration(
        ("hash-account-1",),
        environment="prod",
        ttl_seconds=selection._CODEX_OAUTH_QUOTA_CACHE_TTL_SECONDS,
        now_monotonic=now_monotonic,
    )
    assert manager.codex_quota_hydration_due_account_hashes(
        ("hash-account-1",),
        environment="prod",
        now_monotonic=now_monotonic + 29.999,
    ) == ()
    assert manager.codex_quota_hydration_due_account_hashes(
        ("hash-account-1",),
        environment="prod",
        now_monotonic=now_monotonic + 30.0,
    ) == ("hash-account-1",)


class TestAlibabaTokenPlanQuotaObservations:
    @pytest.fixture(autouse=True)
    def _require_isolated_selection_runtime(self):
        assert selection._get_codex_quota_observation_environment is None
        assert selection._get_codex_quota_observation_pool is None
        yield

    def test_install_exports_hydration_row_identity_helper(self) -> None:
        original_functions = {
            name: getattr(selection, name) for name in selection._HOST_FUNCTION_NAMES
        }
        original_attach = selection._attach_aawm_alias_routing_state_sources
        try:
            host_globals: dict[str, Any] = {}
            selection.install(host_globals)

            hydration = host_globals[
                "_hydrate_alibaba_token_plan_quota_observations"
            ]
            assert hydration.__globals__ is host_globals
            assert (
                host_globals["_alibaba_token_plan_quota_row_account_hash"]
                is selection._alibaba_token_plan_quota_row_account_hash
            )
        finally:
            for name, function in original_functions.items():
                setattr(selection, name, function)
            selection._attach_aawm_alias_routing_state_sources = original_attach

    def test_exact_valid_environment_row_is_normalized(self) -> None:
        observation = selection._alibaba_token_plan_quota_observation_from_row(
            _alibaba_row(),
            expected_environment="prod",
        )

        assert observation is not None
        assert observation["provider"] == "alibaba_token_plan"
        assert observation["account_hash"] == "hash-alibaba-1"
        assert observation["environment"] == "prod"
        assert observation["quota_key"] == "alibaba_token_plan_5h:credits"
        assert observation["exhausted"] is False

    @pytest.mark.parametrize(
        "row_kwargs",
        [
            {"environment": None},
            {"environment": "staging"},
            {"parser_version": "alibaba_token_plan_usage_v2"},
            {"telemetry_status": "unhealthy"},
        ],
    )
    def test_invalid_identity_or_environment_rows_are_unknown(
        self, row_kwargs: dict[str, Any]
    ) -> None:
        observation = selection._alibaba_token_plan_quota_observation_from_row(
            _alibaba_row(**row_kwargs),
            expected_environment="prod",
        )

        assert observation is None

    @pytest.mark.parametrize(
        ("quota_period", "quota_key"),
        [
            ("7d", "alibaba_token_plan_5h:credits"),
            ("5h", "alibaba_token_plan_7d:credits"),
        ],
    )
    def test_mismatched_window_and_quota_key_rows_are_unknown(
        self,
        quota_period: str,
        quota_key: str,
    ) -> None:
        row = _alibaba_row()
        row["quota_period"] = quota_period
        row["quota_key"] = quota_key

        assert (
            selection._alibaba_token_plan_quota_observation_from_row(
                row,
                expected_environment="prod",
            )
            is None
        )

    def test_mismatched_evidence_and_row_window_is_unknown(self) -> None:
        row = _alibaba_row()
        row["quota_period"] = "7d"
        row["quota_key"] = "alibaba_token_plan_7d:credits"

        assert (
            selection._alibaba_token_plan_quota_observation_from_row(
                row,
                expected_environment="prod",
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_hydration_uses_due_gate_and_actual_account_hashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = AliasRoutingStateManager()
        monkeypatch.setattr(selection, "alias_routing_state", manager)
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        rows = [
            _alibaba_row(),
            {
                **_alibaba_row(),
                "model": "alibaba_token_plan/qwen3.7-max",
                "quota_period": "7d",
                "quota_key": "alibaba_token_plan_7d:credits",
                "evidence": {
                    **_alibaba_row()["evidence"],
                    "window": "7d",
                },
            },
        ]
        fetch_args: list[tuple[Any, ...]] = []

        async def _fetch(sql: str, *args: Any):
            fetch_args.append((sql, *args))
            return rows

        fetch = AsyncMock(side_effect=_fetch)

        async def _get_pool():
            return SimpleNamespace(fetch=fetch)

        _set_selection_runtime_value(
            "_get_codex_quota_observation_pool", _get_pool, monkeypatch
        )

        await selection._hydrate_alibaba_token_plan_quota_observations()

        fetch.assert_awaited_once()
        sql, provider, client, source, environment = fetch_args[0]
        assert provider == "alibaba_token_plan"
        assert client == "qwen-cloud-console"
        assert source == "alibaba_token_plan_usage"
        assert environment == "prod"
        assert "NULLIF(BTRIM(evidence->>'environment'), '') = $4" in sql
        assert {
            observation["quota_period"]
            for observation in manager._normalized_quota_observations.values()
        } == {"5h", "7d"}
        assert manager.codex_quota_hydration_due_account_hashes(
            ("alibaba_token_plan",),
            environment="prod",
        ) == ()

        deferred_pool = AsyncMock(side_effect=AssertionError("must stay deferred"))

        async def _get_deferred_pool():
            return SimpleNamespace(fetch=deferred_pool)

        _set_selection_runtime_value(
            "_get_codex_quota_observation_pool",
            _get_deferred_pool,
            monkeypatch,
        )
        await selection._hydrate_alibaba_token_plan_quota_observations()
        deferred_pool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_newer_unavailable_rows_remove_prior_positive_clear_eligibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = AliasRoutingStateManager()
        monkeypatch.setattr(selection, "alias_routing_state", manager)
        now = time.time()
        _seed_alibaba_windows(remaining_pct=40.0, observed_at=now - 30)
        rows = [
            {
                **_alibaba_row(telemetry_status="unavailable"),
                "observed_at": now,
                "model": model,
                "quota_period": window,
                "quota_key": f"alibaba_token_plan_{window}:credits",
                "evidence": {
                    **_alibaba_row(telemetry_status="unavailable")["evidence"],
                    "window": window,
                },
            }
            for window, model in (
                ("5h", "alibaba_token_plan/qwen3.8-max"),
                ("7d", "alibaba_token_plan/qwen3.7-max"),
            )
        ]

        async def _get_pool():
            return SimpleNamespace(fetch=AsyncMock(return_value=rows))

        _set_selection_runtime_value(
            "_get_codex_quota_observation_pool", _get_pool, monkeypatch
        )
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        _set_selection_candidates(_alibaba_candidates())
        account_key = CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY

        async def _cooldown_state(key: str) -> tuple[float, str]:
            if key == account_key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        _set_selection_runtime_value(
            "_get_codex_active_cooldown_state", _cooldown_state, monkeypatch
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        assert [
            candidate["reason"] for candidate in caught.value.detail["candidates"]
        ] == ["account_quota_cooldown", "account_quota_cooldown"]
        assert not manager._normalized_quota_observations
        clear.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_exhausted_windows_block_both_alibaba_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(selection, "alias_routing_state", AliasRoutingStateManager())
        _seed_alibaba_windows(remaining_pct=0.0)
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        detail = caught.value.detail
        assert [candidate["reason"] for candidate in detail["candidates"]] == [
            "quota_exhausted",
            "quota_exhausted",
        ]
        assert all(
            candidate["cooldown_state_source"] == "normalized_quota_observation"
            for candidate in detail["candidates"]
        )

    @pytest.mark.asyncio
    async def test_newer_positive_row_cannot_mask_fresh_exhaustion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(selection, "alias_routing_state", AliasRoutingStateManager())
        _set_selection_runtime_value(
            "_hydrate_alibaba_token_plan_quota_observations",
            AsyncMock(return_value=None),
            monkeypatch,
        )
        now = time.time()
        exhausted_observation = _alibaba_observation(
            window="5h",
            remaining_pct=0.0,
            observed_at=now - 10,
        )
        selection.alias_routing_state.record_normalized_quota_observations(
            [
                exhausted_observation,
                _alibaba_observation(
                    window="5h",
                    remaining_pct=40.0,
                    observed_at=now,
                    model="alibaba_token_plan/qwen3.7-max",
                ),
                _alibaba_observation(
                    window="7d",
                    remaining_pct=40.0,
                    observed_at=now,
                ),
            ]
        )
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        detail = caught.value.detail
        assert [candidate["reason"] for candidate in detail["candidates"]] == [
            "quota_exhausted",
            "quota_exhausted",
        ]
        assert clear.assert_not_awaited() is None

    @pytest.mark.asyncio
    async def test_cross_account_positive_windows_do_not_block_or_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            selection, "alias_routing_state", AliasRoutingStateManager()
        )
        _set_selection_runtime_value(
            "_hydrate_alibaba_token_plan_quota_observations",
            AsyncMock(return_value=None),
            monkeypatch,
        )
        now = time.time()
        selection.alias_routing_state.record_normalized_quota_observations(
            [
                _alibaba_observation(
                    window="5h",
                    remaining_pct=40.0,
                    observed_at=now,
                    account_hash="hash-alibaba-1",
                ),
                _alibaba_observation(
                    window="7d",
                    remaining_pct=40.0,
                    observed_at=now,
                    account_hash="hash-alibaba-2",
                ),
            ]
        )
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body={"model": "basic"},
        )

        assert result["candidate"]["model"] == (
            "alibaba_token_plan/qwen3.8-max"
        )
        assert result["skipped"] == []
        clear.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lone_fresh_exhausted_expired_reset_blocks_and_does_not_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            selection, "alias_routing_state", AliasRoutingStateManager()
        )
        _set_selection_runtime_value(
            "_hydrate_alibaba_token_plan_quota_observations",
            AsyncMock(return_value=None),
            monkeypatch,
        )
        now = time.time()
        observation = _alibaba_observation(
            window="5h",
            remaining_pct=0.0,
            observed_at=now - 30,
        )
        observation["expected_reset_at"] = now - 1
        selection.alias_routing_state.record_normalized_quota_observations(
            [observation]
        )
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        detail = caught.value.detail
        assert [candidate["reason"] for candidate in detail["candidates"]] == [
            "quota_exhausted",
            "quota_exhausted",
        ]
        assert clear.assert_not_awaited() is None

    @pytest.mark.asyncio
    async def test_expired_positive_windows_do_not_clear_active_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            selection, "alias_routing_state", AliasRoutingStateManager()
        )
        _set_selection_runtime_value(
            "_hydrate_alibaba_token_plan_quota_observations",
            AsyncMock(return_value=None),
            monkeypatch,
        )
        now = time.time()
        observations = [
            _alibaba_observation(
                window=window,
                remaining_pct=40.0,
                observed_at=now - 30,
            )
            for window in ("5h", "7d")
        ]
        for observation in observations:
            observation["expected_reset_at"] = now - 1
        selection.alias_routing_state.record_normalized_quota_observations(
            observations
        )
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        account_key = CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY

        async def _cooldown_state(key: str) -> tuple[float, str]:
            if key == account_key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        _set_selection_runtime_value(
            "_get_codex_active_cooldown_state", _cooldown_state, monkeypatch
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        assert [
            candidate["reason"] for candidate in caught.value.detail["candidates"]
        ] == ["account_quota_cooldown", "account_quota_cooldown"]
        clear.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mismatched_window_identity_is_unknown_and_does_not_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            selection, "alias_routing_state", AliasRoutingStateManager()
        )
        _set_selection_runtime_value(
            "_hydrate_alibaba_token_plan_quota_observations",
            AsyncMock(return_value=None),
            monkeypatch,
        )
        now = time.time()
        observation = _alibaba_observation(
            window="5h",
            remaining_pct=40.0,
            observed_at=now,
        )
        observation["quota_period"] = "7d"
        observation["quota_key"] = "alibaba_token_plan_5h:credits"
        selection.alias_routing_state.record_normalized_quota_observations(
            [
                observation,
                _alibaba_observation(
                    window="7d",
                    remaining_pct=40.0,
                    observed_at=now,
                ),
            ]
        )
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        account_key = CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY

        async def _cooldown_state(key: str) -> tuple[float, str]:
            if key == account_key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        _set_selection_runtime_value(
            "_get_codex_active_cooldown_state", _cooldown_state, monkeypatch
        )
        clear = AsyncMock()
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        assert [
            candidate["reason"] for candidate in caught.value.detail["candidates"]
        ] == ["account_quota_cooldown", "account_quota_cooldown"]
        assert clear.assert_not_awaited() is None

    @pytest.mark.asyncio
    async def test_complete_positive_windows_clear_only_account_quota_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(selection, "alias_routing_state", AliasRoutingStateManager())
        _seed_alibaba_windows(remaining_pct=40.0)
        _set_selection_candidates(_alibaba_candidates())
        manager = selection.alias_routing_state
        monkeypatch.setattr(cooldown_state, "_manager", manager)
        account_key = CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
        candidate_key = (
            "alibaba_token_plan:alibaba_token_plan/qwen3.8-max:"
            "alibaba_token_plan"
        )
        manager.codex.cooldown_until_monotonic_by_key[account_key] = (
            time.monotonic() + 60
        )
        manager.codex.cooldown_until_monotonic_by_key[candidate_key] = (
            time.monotonic() + 60
        )

        async def _cooldown_state(key: str) -> tuple[float, str]:
            return (0.0, "local_fallback")

        _set_selection_runtime_value(
            "_get_codex_active_cooldown_state", _cooldown_state, monkeypatch
        )
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        real_clear = cooldown_state.clear_alias_family_cooldown_state
        clear_calls: list[dict[str, Any]] = []

        async def _clear(**kwargs: Any):
            clear_calls.append(kwargs)
            return await real_clear(**{**kwargs, "delete_durable": False})

        monkeypatch.setattr(
            cooldown_state, "clear_alias_family_cooldown_state", _clear
        )

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body={"model": "basic"},
        )

        assert result["candidate"]["model"] in {
            "alibaba_token_plan/qwen3.8-max",
            "alibaba_token_plan/qwen3.7-max",
        }
        assert clear_calls == [
            {
                "alias_family": "codex",
                "canonical_aliases": ["alibaba_token_plan"],
                "cooldown_keys": [account_key],
                "delete_durable": True,
            }
        ]
        assert account_key not in manager.codex.cooldown_until_monotonic_by_key
        assert manager.codex.cooldown_until_monotonic_by_key[candidate_key] > (
            time.monotonic()
        )

    @pytest.mark.parametrize(
        ("remaining_pct", "environment"),
        [
            (25.0, "staging"),
            (25.0, "prod"),
        ],
    )
    @pytest.mark.asyncio
    async def test_stale_and_wrong_environment_evidence_are_unknown(
        self,
        monkeypatch: pytest.MonkeyPatch,
        remaining_pct: float,
        environment: str,
    ) -> None:
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        manager = AliasRoutingStateManager()
        monkeypatch.setattr(selection, "alias_routing_state", manager)
        observed_at = (
            time.time() - 901
            if environment == "prod"
            else time.time()
        )
        _seed_alibaba_windows(
            remaining_pct=remaining_pct,
            observed_at=observed_at,
            environment=environment,
        )
        evidence, windows = selection._alibaba_token_plan_quota_evidence(
            state_manager=manager,
            now_epoch=time.time()
        )

        assert evidence is None
        assert windows == []

    @pytest.mark.asyncio
    async def test_partial_window_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        manager = AliasRoutingStateManager()
        manager.record_normalized_quota_observations(
            [
                _alibaba_observation(
                    window="5h",
                    remaining_pct=25.0,
                    observed_at=time.time(),
                )
            ]
        )

        evidence, windows = selection._alibaba_token_plan_quota_evidence(
            state_manager=manager,
            now_epoch=time.time()
        )

        assert evidence is None
        assert windows == []

    def test_malformed_and_ambiguous_rows_are_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        valid_observation = selection._alibaba_token_plan_quota_observation_from_row(
            _alibaba_row(),
            expected_environment="prod",
        )
        malformed = selection._alibaba_token_plan_quota_observation_from_row(
            {**_alibaba_row(), "evidence": "{invalid"},
            expected_environment="prod",
        )
        unavailable = selection._alibaba_token_plan_quota_observation_from_row(
            {
                **_alibaba_row(),
                "evidence": {
                    **_alibaba_row()["evidence"],
                    "telemetry_status": "unavailable",
                },
            },
            expected_environment="prod",
        )

        assert valid_observation is not None
        assert malformed is None
        assert unavailable is None

        assert valid_observation is not None
        manager = AliasRoutingStateManager()
        manager.record_normalized_quota_observations(
            [
                {
                    **valid_observation,
                    "quota_period": "7d",
                    "quota_key": "alibaba_token_plan_7d:credits",
                },
                {
                    **valid_observation,
                    "account_hash": "hash-alibaba-2",
                    "quota_period": "5h",
                },
            ]
        )
        evidence, windows = selection._alibaba_token_plan_quota_evidence(
            state_manager=manager,
            now_epoch=time.time(),
        )

        assert evidence is None
        assert windows == []

    @pytest.mark.asyncio
    async def test_unknown_sidecar_leaves_ali004_cooldown_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_selection_candidates(_alibaba_candidates())
        _set_selection_runtime_value(
            "_get_codex_quota_observation_environment",
            lambda: "prod",
            monkeypatch,
        )
        account_key = CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY

        async def _cooldown_state(key: str) -> tuple[float, str]:
            if key == account_key:
                return (60.0, "durable_cache")
            return (0.0, "local_fallback")

        clear = AsyncMock()
        _set_selection_runtime_value(
            "_get_codex_active_cooldown_state", _cooldown_state, monkeypatch
        )
        _set_selection_runtime_value(
            "_clear_alibaba_token_plan_account_quota_cooldown",
            clear,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as caught:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={
                    "model": "basic",
                    "litellm_metadata": {"redispatch_ordinal": 1},
                },
            )

        assert [
            candidate["reason"] for candidate in caught.value.detail["candidates"]
        ] == ["account_quota_cooldown", "account_quota_cooldown"]
        clear.assert_not_awaited()


def _owned_interchangeable_record(
    *,
    label: str = "account1",
    account_hash: str = "hash-account-1",
) -> dict[str, Any]:
    return {
        "state": "owned",
        "owner": "owner-1",
        "attributes": {
            "provider": "openai",
            "model": "gpt-5.3-codex",
            "route_family": "codex_responses",
            "account_label": label,
            "account_hash": account_hash,
            "account_lane": f"codex-oauth:{label}:{account_hash}",
            "credential_affinity": "interchangeable",
        },
    }


class TestCodexAccountBound:
    def test_unbound_policy_does_not_create_permanent_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from types import SimpleNamespace

        monkeypatch.setattr(
            "litellm.secret_managers.codex_oauth_inventory.load_codex_oauth_inventory",
            lambda: SimpleNamespace(
                routing=SimpleNamespace(accounts_are_interchangeable=True)
            ),
        )
        adjusted = selection._apply_codex_oauth_inventory_affinity_policy(
            _oauth_account_candidate()
        )
        assert adjusted is not None
        assert adjusted["codex_oauth_credential_affinity"] == "interchangeable"
        assert "codex_oauth_account_label" not in adjusted
        assert "codex_oauth_account_hash" not in adjusted
        assert "codex_oauth_lane_key" not in adjusted

    def test_account_bound_policy_preserves_creating_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from types import SimpleNamespace

        monkeypatch.setattr(
            "litellm.secret_managers.codex_oauth_inventory.load_codex_oauth_inventory",
            lambda: SimpleNamespace(
                routing=SimpleNamespace(accounts_are_interchangeable=True)
            ),
        )
        affinity = _oauth_account_candidate()
        adjusted = selection._apply_codex_oauth_inventory_affinity_policy(
            affinity,
            account_bound=True,
        )
        assert adjusted is not None
        assert adjusted["codex_oauth_credential_affinity"] == "interchangeable"
        assert adjusted["codex_oauth_account_label"] == "account1"
        assert adjusted["codex_oauth_lane_key"] == (
            "codex-oauth:account1:hash-account-1"
        )

    def test_owner_hint_omits_account_identity_by_default(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as sa,
        )

        default_hint = sa.owner_record_as_affinity_hint(
            _owned_interchangeable_record()
        )
        assert default_hint is not None
        assert default_hint["codex_oauth_credential_affinity"] == "interchangeable"
        assert "codex_oauth_account_label" not in default_hint
        assert "codex_oauth_lane_key" not in default_hint

        pinned = sa.owner_record_as_affinity_hint(
            _owned_interchangeable_record(),
            preserve_account_identity=True,
        )
        assert pinned is not None
        assert pinned["codex_oauth_account_label"] == "account1"
        assert pinned["codex_oauth_lane_key"] == (
            "codex-oauth:account1:hash-account-1"
        )

    def test_pre_state_failover_remains_available(self) -> None:
        candidate = _oauth_account_candidate()
        planned = selection._plan_codex_oauth_account_failover(
            _make_request(),
            candidate=candidate,
            selection={"candidate": candidate, "failover_ordinal": 0},
            attempt_record={
                "failure_phase": "direct_openai_provider_response",
                "attempted_provider_call": True,
            },
            error_class="usage_limit_reached",
            has_continuation_state=True,
            has_previous_response_id=False,
            has_account_bound_state=False,
        )
        assert planned is True

    def test_account_bound_state_blocks_failover(self) -> None:
        candidate = _oauth_account_candidate()
        planned = selection._plan_codex_oauth_account_failover(
            _make_request(),
            candidate=candidate,
            selection={
                "candidate": candidate,
                "failover_ordinal": 0,
                "has_account_bound_state": True,
            },
            attempt_record={
                "failure_phase": "direct_openai_provider_response",
                "attempted_provider_call": True,
            },
            error_class="usage_limit_reached",
            has_continuation_state=True,
            has_previous_response_id=False,
            has_account_bound_state=True,
        )
        assert planned is False

    def test_safe_metadata_exposes_lane_not_payload(self) -> None:
        attached = selection._attach_account_bound_selection_metadata(
            {
                "candidate": {
                    **_oauth_account_candidate(),
                    "encrypted_content": "do-not-leak",
                    "Authorization": "Bearer secret",
                }
            },
            has_account_bound_state=True,
            affinity={
                "codex_oauth_lane_key": (
                    "codex-oauth:account1:hash-account-1"
                ),
                "encrypted_content": "do-not-leak",
            },
        )
        assert attached["has_account_bound_state"] is True
        assert attached["account_bound_classification"] == "account_bound"
        assert attached["account_bound_owner_lane"] == (
            "codex-oauth:account1:hash-account-1"
        )
        assert "encrypted_content" not in attached
        assert "Authorization" not in attached

    @pytest.mark.asyncio
    async def test_fresh_unbound_selection_keeps_first_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_selection_runtime_value(
            "_has_account_bound_state",
            lambda _body: False,
            monkeypatch,
        )
        candidates = (
            _candidate("openai", "gpt-4o"),
            _candidate("xai", "grok-4", last_resort=True),
        )
        _set_selection_candidates(candidates)

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "first_available"
        assert result["candidate"]["provider"] == "openai"
        assert result["has_account_bound_state"] is False
        assert result["account_bound_classification"] == "unbound"
        assert "account_bound_owner_lane" not in result

    def _patch_bound_owner_selector(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        affinity_state: dict[str, Any],
        stub_classifier: bool = True,
    ) -> AsyncMock:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as sa,
        )

        async def _owned(**_kwargs):
            return (_owned_interchangeable_record(), "cache-key", None)

        monkeypatch.setattr(sa, "get_session_owner_record", _owned)
        monkeypatch.setattr(
            sa,
            "resolve_canonical_session_identity",
            lambda *_args, **_kwargs: "sess-openai-019",
        )
        if stub_classifier:
            _set_selection_runtime_value(
                "_has_account_bound_state",
                lambda _body: True,
                monkeypatch,
            )
        _set_selection_runtime_value(
            "_apply_codex_oauth_inventory_affinity_policy",
            lambda affinity, *, account_bound=False: affinity,
            monkeypatch,
        )
        _set_selection_runtime_value(
            "_find_codex_auto_agent_affinity_candidate",
            lambda *_args, **_kwargs: {
                "provider": "openai",
                "model": "gpt-5.3-codex",
                "route_family": "codex_responses",
                "last_resort": False,
            },
            monkeypatch,
        )
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            AsyncMock(return_value=affinity_state),
            monkeypatch,
        )
        alternate = AsyncMock(
            side_effect=AssertionError("alternate account selection must not run")
        )
        _set_selection_runtime_value(
            "_build_codex_auto_agent_candidate_states",
            alternate,
            monkeypatch,
        )
        return alternate

    @pytest.mark.asyncio
    async def test_bound_owner_pins_creating_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        affinity_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.0,
            "skip_reason": None,
            "cooldown_state_source": "local_fallback",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=affinity_state,
        )

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body={"model": "basic"},
        )
        assert result["selection_reason"] == "session_affinity"
        assert result["candidate"]["codex_oauth_account_label"] == "account1"
        assert result["has_account_bound_state"] is True
        assert result["account_bound_classification"] == "account_bound"
        assert result["account_bound_owner_lane"] == pinned["codex_oauth_lane_key"]
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unavailable_bound_owner_fails_closed_without_alternate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        affinity_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.0,
            "skip_reason": "quota_exhausted",
            "failure_phase": "quota_exhausted",
            "cooldown_state_source": "local_fallback",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=affinity_state,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic"},
            )
        detail = exc_info.value.detail
        assert exc_info.value.status_code == 429
        assert detail["redispatch_required"] is True
        assert detail["error"]["code"] == (
            "aawm_codex_auto_agent_redispatch_required"
        )
        assert detail["failure_phase"] == "account_bound_owner_unavailable"
        assert detail["attempted_provider_call"] is False
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resp_previous_response_id_pins_creating_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        affinity_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.0,
            "skip_reason": None,
            "cooldown_state_source": "local_fallback",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=affinity_state,
            stub_classifier=False,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build,
        )

        _set_selection_runtime_value(
            "_has_account_bound_state",
            audit_build._aawm_auto_agent_audit_request_has_account_bound_state,
            monkeypatch,
        )

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body={"model": "basic", "previous_response_id": "resp_123"},
        )
        assert result["selection_reason"] == "session_affinity"
        assert result["candidate"]["codex_oauth_account_label"] == "account1"
        assert result["has_account_bound_state"] is True
        assert result["account_bound_classification"] == "account_bound"
        assert result["account_bound_owner_lane"] == pinned["codex_oauth_lane_key"]
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_clears_after_one_wait_on_same_owner_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        hot_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.5,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        clear_state = {**hot_state, "cooldown_seconds": 0.0}
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=hot_state,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        state_builder = AsyncMock(side_effect=[hot_state, clear_state])
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)
        body = {"model": "basic", "previous_response_id": "resp_123"}

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body=body,
        )

        sleep.assert_awaited_once_with(0.5)
        assert state_builder.await_count == 2
        assert result["selection_reason"] == "session_affinity"
        assert result["candidate"] == pinned
        assert result["account_bound_owner_lane"] == pinned["codex_oauth_lane_key"]
        assert body == {"model": "basic", "previous_response_id": "resp_123"}
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_remains_hot_then_uses_existing_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        first_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.5,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        refreshed_state = {**first_state, "cooldown_seconds": 0.25}
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=first_state,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        state_builder = AsyncMock(
            side_effect=[first_state, refreshed_state]
        )
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as sa,
        )

        activate = AsyncMock(
            side_effect=AssertionError("effective identity must not activate")
        )
        monkeypatch.setattr(
            sa,
            "activate_session_owner_redispatch_effective_identity",
            activate,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic"},
            )

        sleep.assert_awaited_once_with(0.5)
        assert state_builder.await_count == 2
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"]["code"] == (
            "aawm_codex_auto_agent_in_flight_provider_cooling_down"
        )
        assert exc_info.value.detail["candidate"]["cooldown_seconds"] == 0.25
        assert exc_info.value.headers["Retry-After"] == "1"
        activate.assert_not_awaited()
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_longer_than_one_second_is_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 1.1,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=state,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        state_builder = AsyncMock(return_value=state)
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic"},
            )

        sleep.assert_not_awaited()
        assert state_builder.await_count == 1
        assert exc_info.value.status_code == 429
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_cancellation_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.5,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=state,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        state_builder = AsyncMock(return_value=state)
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock(side_effect=selection.asyncio.CancelledError)
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)

        with pytest.raises(selection.asyncio.CancelledError):
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic"},
            )

        sleep.assert_awaited_once_with(0.5)
        assert state_builder.await_count == 1
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_does_not_activate_effective_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        hot_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.5,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        clear_state = {**hot_state, "cooldown_seconds": 0.0}
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=hot_state,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        state_builder = AsyncMock(side_effect=[hot_state, clear_state])
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)
        activate = AsyncMock(
            side_effect=AssertionError("effective identity must not activate")
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as sa,
        )

        monkeypatch.setattr(
            sa,
            "activate_session_owner_redispatch_effective_identity",
            activate,
        )

        body = {"model": "basic", "previous_response_id": "resp_123"}
        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body=body,
        )

        assert result["candidate"]["codex_oauth_account_label"] == "account1"
        assert result["account_bound_owner_lane"] == pinned["codex_oauth_lane_key"]
        assert body["previous_response_id"] == "resp_123"
        activate.assert_not_awaited()
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_micro_cooldown_clears_for_alibaba_token_plan_owner_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as sa,
        )

        pinned = {
            "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash",
            "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
            "last_resort": False,
        }
        affinity = dict(pinned)
        hot_state = {
            "candidate": pinned,
            "lane_key": "alibaba_token_plan",
            "cooldown_seconds": 0.5,
            "skip_reason": None,
            "cooldown_state_source": "memory",
        }
        clear_state = {**hot_state, "cooldown_seconds": 0.0}
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=hot_state,
            stub_classifier=False,
        )
        _set_selection_runtime_value(
            "_has_continuation_state",
            lambda _body: True,
            monkeypatch,
        )
        _set_selection_runtime_value(
            "_find_codex_auto_agent_affinity_candidate",
            lambda *_args, **_kwargs: pinned,
            monkeypatch,
        )
        monkeypatch.setattr(
            sa,
            "owner_record_as_affinity_hint",
            lambda *_args, **_kwargs: affinity,
        )
        state_builder = AsyncMock(side_effect=[hot_state, clear_state])
        _set_selection_runtime_value(
            "_build_codex_auto_agent_affinity_candidate_state",
            state_builder,
            monkeypatch,
        )
        sleep = AsyncMock()
        monkeypatch.setattr(selection.asyncio, "sleep", sleep)
        activate = AsyncMock(
            side_effect=AssertionError("effective identity must not activate")
        )
        monkeypatch.setattr(
            sa,
            "activate_session_owner_redispatch_effective_identity",
            activate,
        )
        body = {
            "model": "basic",
            "previous_response_id": "resp_alibaba_123",
        }

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body=body,
        )

        sleep.assert_awaited_once_with(0.5)
        assert state_builder.await_count == 2
        assert result["candidate"] == pinned
        assert result["lane_key"] == "alibaba_token_plan"
        assert result["session_owner_id"] == "owner-1"
        assert body["previous_response_id"] == "resp_alibaba_123"
        activate.assert_not_awaited()
        alternate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resp_previous_response_id_fails_closed_without_alternate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pinned = _oauth_account_candidate()
        affinity_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.0,
            "skip_reason": "quota_exhausted",
            "failure_phase": "quota_exhausted",
            "cooldown_state_source": "local_fallback",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=affinity_state,
            stub_classifier=False,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build,
        )

        _set_selection_runtime_value(
            "_has_account_bound_state",
            audit_build._aawm_auto_agent_audit_request_has_account_bound_state,
            monkeypatch,
        )

        with pytest.raises(HTTPException) as exc_info:
            await selection._select_codex_auto_agent_candidate(
                request=_make_request(),
                request_body={"model": "basic", "previous_response_id": "resp_123"},
            )
        detail = exc_info.value.detail
        assert exc_info.value.status_code == 429
        assert detail["redispatch_required"] is True
        assert detail["error"]["code"] == (
            "aawm_codex_auto_agent_redispatch_required"
        )
        assert detail["failure_phase"] == "account_bound_owner_unavailable"
        assert detail["attempted_provider_call"] is False
        alternate.assert_not_awaited()


    @pytest.mark.asyncio
    async def test_installed_host_classifier_marks_bound_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as lpe,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build,
        )

        host_classifier = getattr(
            lpe,
            "_aawm_auto_agent_audit_request_has_account_bound_state",
        )
        assert callable(host_classifier)
        assert host_classifier is audit_build._aawm_auto_agent_audit_request_has_account_bound_state
        assert not hasattr(
            lpe,
            "_codex_auto_agent_request_has_account_bound_state",
        )
        # Isolate from the autouse fixture's false stub. Bind the canonical
        # installed host classifier into the selection runtime seam.
        _set_selection_runtime_value(
            "_has_account_bound_state",
            host_classifier,
            monkeypatch,
        )

        bound_bodies = (
            {
                "model": "basic",
                "previous_response_id": "resp_123",
            },
            {
                "model": "basic",
                "encrypted_content": "SECRET_ENCRYPTED_BLOB",
            },
            {
                "model": "basic",
                "input": [{"item_reference": {"id": "rs_nested_ref_123"}}],
            },
            {
                "model": "basic",
                "input": [{"type": "reasoning", "summary": []}],
            },
            {
                "model": "basic",
                "input": [
                    {
                        "type": "function_call_output",
                        "output": "SECRET_TOOL_OUTPUT",
                    }
                ],
            },
        )
        fresh_body = {
            "model": "basic",
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        for bound_body in bound_bodies:
            assert host_classifier(bound_body) is True
        assert host_classifier(fresh_body) is False

        # Selector now exercises the canonical installed classifier via the
        # existing selection runtime seam. Do not stub that classifier.
        pinned = _oauth_account_candidate()
        affinity_state = {
            "candidate": pinned,
            "lane_key": pinned["codex_oauth_lane_key"],
            "cooldown_seconds": 0.0,
            "skip_reason": None,
            "cooldown_state_source": "local_fallback",
        }
        alternate = self._patch_bound_owner_selector(
            monkeypatch,
            affinity_state=affinity_state,
            stub_classifier=False,
        )

        result = await selection._select_codex_auto_agent_candidate(
            request=_make_request(),
            request_body=bound_bodies[0],
        )
        assert result["has_account_bound_state"] is True
        assert result["account_bound_classification"] == "account_bound"
        assert result["account_bound_owner_lane"] == pinned["codex_oauth_lane_key"]
        alternate.assert_not_awaited()
