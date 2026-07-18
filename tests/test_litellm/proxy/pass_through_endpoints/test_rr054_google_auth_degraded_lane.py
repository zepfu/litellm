"""RR-054 Google lane failure + Antigravity auth_degraded sentinel regressions.

Covers:
- real Google Code Assist lane-resolution failure without NameError
- process-local negative-cache suppression on the second resolve call
- Antigravity auth_degraded sentinel lane key
- forced cooldown / unavailable candidate state for auth_degraded

Production is not modified by this module; failures document current regressions
(especially missing module-level process-local bindings).
"""

from __future__ import annotations

import time
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


def _minimal_request(path: str = "/v1/responses") -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _stale_antigravity_oauth_http_exception() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=(
            "Antigravity OAuth token is expired or invalid. The "
            "provider-status sidecar owns Antigravity auth refresh; confirm the "
            "sidecar can write the configured token file and refresh "
            "/root/.litellm/antigravity/antigravity-oauth-token."
        ),
    )


def _antigravity_candidate_template() -> dict[str, Any]:
    return {
        "provider": lpe._CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER,
        "model": "gemini-3-flash",
        "route_family": "codex_antigravity_code_assist_adapter",
        "last_resort": False,
    }


def _reset_google_lane_negative_cache() -> None:
    """Best-effort reset of the google-lane negative cache across ownership seams."""
    if hasattr(lpe, "_codex_auto_agent_google_lane_negative_until_monotonic"):
        lpe._codex_auto_agent_google_lane_negative_until_monotonic = 0.0
    state = getattr(lpe, "_alias_routing_state", None)
    if state is not None and hasattr(state, "google_lane_negative_until_monotonic"):
        state.google_lane_negative_until_monotonic = 0.0
    lpe._codex_auto_agent_google_lane_key_by_key.clear()
    lpe._codex_auto_agent_google_lane_key_until_monotonic_by_key.clear()


def _reset_antigravity_lane_cache() -> None:
    lpe._codex_auto_agent_antigravity_lane_key_by_key.clear()
    lpe._codex_auto_agent_antigravity_lane_key_until_monotonic_by_key.clear()
    if hasattr(lpe, "_codex_auto_agent_antigravity_auth_degraded_log_until_monotonic"):
        lpe._codex_auto_agent_antigravity_auth_degraded_log_until_monotonic = 0.0


@pytest.mark.asyncio
async def test_rr054_google_lane_real_resolution_failure_returns_degraded_without_nameerror() -> None:
    """Real OAuth/project failure must yield an unavailable degraded lane."""
    _reset_google_lane_negative_cache()
    load_token = AsyncMock(side_effect=RuntimeError("google oauth unavailable"))

    with patch.object(
        lpe,
        "_load_valid_local_google_oauth_access_token",
        new=load_token,
    ), patch.object(
        lpe.verbose_proxy_logger,
        "warning",
    ):
        try:
            lane_key = await lpe._resolve_codex_auto_agent_google_lane_key()
        except NameError as exc:  # pragma: no cover - documents active regression
            pytest.fail(
                "RR-054 regression: Google lane resolve raised NameError on real "
                f"failure path: {exc}"
            )

    assert lane_key == lpe._CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY
    assert load_token.await_count == 1


@pytest.mark.asyncio
async def test_rr054_google_lane_negative_cache_suppresses_second_resolve_call() -> None:
    """Second resolve within negative-cache TTL must not re-enter OAuth load."""
    _reset_google_lane_negative_cache()
    load_token = AsyncMock(side_effect=RuntimeError("google oauth unavailable"))

    with patch.object(
        lpe,
        "_load_valid_local_google_oauth_access_token",
        new=load_token,
    ), patch.object(
        lpe.verbose_proxy_logger,
        "warning",
    ):
        try:
            first = await lpe._resolve_codex_auto_agent_google_lane_key()
            second = await lpe._resolve_codex_auto_agent_google_lane_key()
        except NameError as exc:  # pragma: no cover - documents active regression
            pytest.fail(
                "RR-054 regression: Google lane negative-cache path raised "
                f"NameError: {exc}"
            )

    assert first == lpe._CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY
    assert second == lpe._CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY
    assert load_token.await_count == 1, (
        "second resolve should be suppressed by the google-lane negative cache"
    )
    # Negative-cache TTL should still be active after the first failure.
    if hasattr(lpe, "_codex_auto_agent_google_lane_negative_until_monotonic"):
        assert (
            lpe._codex_auto_agent_google_lane_negative_until_monotonic
            > time.monotonic()
        )


def test_rr054_antigravity_auth_degraded_exception_classifier() -> None:
    assert lpe._is_codex_auto_agent_antigravity_auth_degraded_exception(
        _stale_antigravity_oauth_http_exception()
    )
    assert lpe._is_codex_auto_agent_antigravity_auth_degraded_exception(
        HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token file does not contain a usable access token"
            ),
        )
    )
    assert not lpe._is_codex_auto_agent_antigravity_auth_degraded_exception(
        RuntimeError("project lookup failed")
    )
    assert not lpe._is_codex_auto_agent_antigravity_auth_degraded_exception(
        HTTPException(status_code=400, detail="request envelope invalid")
    )


@pytest.mark.asyncio
async def test_rr054_antigravity_auth_degraded_lane_key_is_sentinel_without_nameerror() -> None:
    """Auth-degraded Antigravity resolve must return the stable sentinel lane key."""
    _reset_antigravity_lane_cache()
    stale = _stale_antigravity_oauth_http_exception()
    load_token = AsyncMock(side_effect=stale)

    with patch.object(
        lpe,
        "_load_valid_local_antigravity_access_token",
        new=load_token,
    ), patch.object(
        lpe.verbose_proxy_logger,
        "warning",
    ):
        try:
            lane_key = await lpe._resolve_codex_auto_agent_antigravity_lane_key()
        except NameError as exc:  # pragma: no cover - documents active regression
            pytest.fail(
                "RR-054 regression: Antigravity auth_degraded resolve raised "
                f"NameError: {exc}"
            )

    assert lane_key == lpe._CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY
    assert lane_key == "antigravity:auth_degraded"
    assert load_token.await_count == 1


@pytest.mark.asyncio
async def test_rr054_antigravity_auth_degraded_lane_state_forces_cooldown_fields() -> None:
    """auth_degraded lane state must advertise forced cooldown + skip_reason."""
    _reset_antigravity_lane_cache()
    stale = _stale_antigravity_oauth_http_exception()

    with patch.object(
        lpe,
        "_load_valid_local_antigravity_access_token",
        new=AsyncMock(side_effect=stale),
    ), patch.object(
        lpe.verbose_proxy_logger,
        "warning",
    ):
        try:
            state = await lpe._resolve_codex_auto_agent_antigravity_lane_state()
        except NameError as exc:  # pragma: no cover - documents active regression
            pytest.fail(
                "RR-054 regression: Antigravity auth_degraded lane state raised "
                f"NameError: {exc}"
            )

    assert state["lane_key"] == "antigravity:auth_degraded"
    assert state["skip_reason"] == "auth_degraded"
    assert state["cooldown_state_source"] == "auth_degraded"
    assert state["failure_phase"] == "auth"
    assert state["attempted_provider_call"] is False
    assert state["forced_cooldown_seconds"] == pytest.approx(
        lpe._CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS
    )
    assert state["forced_cooldown_seconds"] == pytest.approx(5 * 60.0)


@pytest.mark.asyncio
async def test_rr054_antigravity_auth_degraded_candidate_state_is_unavailable() -> None:
    """Forced auth_degraded cooldown must mark the candidate unavailable."""
    _reset_antigravity_lane_cache()
    request = _minimal_request()
    candidate = _antigravity_candidate_template()
    auth_degraded_lane_state = {
        "lane_key": lpe._CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY,
        "forced_cooldown_seconds": lpe._CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS,
        "skip_reason": "auth_degraded",
        "cooldown_state_source": "auth_degraded",
        "failure_phase": "auth",
        "attempted_provider_call": False,
    }
    expected_cooldown_key = lpe._codex_auto_agent_candidate_key(
        candidate,
        lpe._CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY,
    )

    def _echo_request_local(
        _request: Request,
        *,
        candidate: dict[str, Any],
        lane_key: Optional[str],
        cooldown_seconds: float,
        cooldown_state_source: Optional[str],
        skip_reason: Optional[str],
    ) -> tuple[float, Optional[str], Optional[str]]:
        return cooldown_seconds, cooldown_state_source, skip_reason

    async def _echo_async_cooldown(
        *,
        candidate: dict[str, Any],
        cooldown_seconds: float,
        cooldown_state_source: Optional[str],
        skip_reason: Optional[str],
    ) -> tuple[float, Optional[str], Optional[str]]:
        return cooldown_seconds, cooldown_state_source, skip_reason

    with patch.object(
        lpe,
        "_get_codex_auto_agent_active_cooldown_state",
        new=AsyncMock(return_value=(0.0, "local_fallback")),
    ), patch.object(
        lpe,
        "_apply_codex_auto_agent_request_local_candidate_state",
        side_effect=_echo_request_local,
    ), patch.object(
        lpe,
        "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
        new=AsyncMock(side_effect=_echo_async_cooldown),
    ), patch.object(
        lpe,
        "_apply_openrouter_durable_quota_candidate_cooldown",
        new=AsyncMock(side_effect=_echo_async_cooldown),
    ), patch.object(
        lpe,
        "_apply_codex_auto_agent_forced_candidate_cooldown",
        new=AsyncMock(),
    ) as forced_cd:
        state = await lpe._build_codex_auto_agent_candidate_state(
            request,
            candidate_template=candidate,
            antigravity_lane_state=auth_degraded_lane_state,
        )

    assert state["lane_key"] == "antigravity:auth_degraded"
    assert state["cooldown_key"] == expected_cooldown_key
    assert state["skip_reason"] == "auth_degraded"
    assert state["cooldown_state_source"] == "auth_degraded"
    assert state["failure_phase"] == "auth"
    assert state["attempted_provider_call"] is False
    assert state["cooldown_seconds"] == pytest.approx(
        lpe._CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS
    )
    assert lpe._is_auto_agent_candidate_state_available(state) is False
    forced_cd.assert_awaited_once()
    forced_kwargs = forced_cd.await_args.kwargs
    assert forced_kwargs["cooldown_key"] == expected_cooldown_key
    assert forced_kwargs["cooldown_seconds"] == pytest.approx(
        lpe._CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS
    )


def test_rr054_auth_degraded_rollup_status_is_degraded_not_cooling_down() -> None:
    status = lpe._auto_agent_alias_route_rollup_status(
        {
            "event_type": "candidate_skipped_provider_degraded",
            "candidate_status": "skipped_auth_degraded",
            "selection_reason": "auth_degraded",
        }
    )
    assert status == "Degraded"
