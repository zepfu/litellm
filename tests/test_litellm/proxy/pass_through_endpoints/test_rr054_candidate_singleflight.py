"""RR-054 finding #31: candidate/lane cold-probe single-flight concurrency tests.

Requirements under test:
- concurrent cold requests for the same candidate/lane must single-flight so
  only one upstream probe runs before success/failure/cooldown is visible
- distinct candidates/lanes remain independent (may probe concurrently)

No production edits. Failures here document remaining single-flight gaps
(missing lock use or lock released before cooldown publish).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    alias_routing_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_request(session_id: str) -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {
        "session_id": session_id,
        "user-agent": "codex-cli/1.0",
        "originator": "codex_cli_rs",
    }
    request.query_params = {}
    request.url = MagicMock()
    request.scope = {
        "path": "/openai_passthrough/v1/responses",
        "query_string": b"",
        "parsed_body": None,
    }
    request.state = MagicMock()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


def _candidate(
    *,
    provider: str = "openrouter",
    model: str = "openrouter/cohere/north-mini-code:free",
    route_family: str = "codex_openrouter_completion_adapter",
    last_resort: bool = False,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "last_resort": last_resort,
    }


def _selection(
    *,
    candidate: dict[str, Any],
    lane_key: str,
    cooldown_key: str,
    session_key: str,
) -> dict[str, Any]:
    return {
        "candidate": dict(candidate),
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "session_key": session_key,
        "selection_reason": "first_available",
        "skipped": [],
        "in_flight_session": False,
        "cooldown_seconds": 0.0,
        "cooldown_state_source": "local_fallback",
    }


def _capacity_error() -> RuntimeError:
    return RuntimeError(
        "Selected model is at capacity. Please try a different model."
    )


def _clear_probe_locks() -> None:
    alias_routing_state.candidate_probe_locks.clear()


@pytest.fixture(autouse=True)
def _clear_singleflight_state() -> Any:
    _clear_probe_locks()
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()
    yield
    _clear_probe_locks()
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()


class _ProbeCounter:
    """Tracks total and peak concurrent probe entries for a lane."""

    def __init__(self) -> None:
        self.total = 0
        self.current = 0
        self.max_current = 0
        self._guard = asyncio.Lock()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.hold_seconds = 0.05

    async def run(
        self,
        *,
        hold: bool = True,
        outcome: str = "fail",
        success_response: Optional[Response] = None,
    ) -> Response:
        async with self._guard:
            self.total += 1
            self.current += 1
            self.max_current = max(self.max_current, self.current)
            self.entered.set()
        try:
            if hold:
                # Keep the probe "in flight" long enough for siblings to contend.
                if not self.release.is_set():
                    try:
                        await asyncio.wait_for(
                            self.release.wait(), timeout=self.hold_seconds
                        )
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0)
            if outcome == "success":
                assert success_response is not None
                return success_response
            raise _capacity_error()
        finally:
            async with self._guard:
                self.current -= 1


async def _run_route_once(
    *,
    alias_family: str,
    alias_model: str,
    session_id: str,
    select_candidate_fn: Any,
    perform_candidate_request_fn: Any,
    get_active_cooldown_state_fn: Any,
    apply_cooldown_fn: Any,
    max_candidate_attempts: int = 2,
) -> Any:
    request = _minimal_request(session_id)
    body = {
        "model": alias_model,
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "litellm_metadata": {"session_id": session_id},
    }

    def _add_metadata(
        request_body: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        metadata = dict(request_body.get("litellm_metadata") or {})
        request_body["litellm_metadata"] = metadata
        return request_body

    async def _set_affinity(*_a: Any, **_k: Any) -> None:
        return None

    def _raise_redispatch(**_k: Any) -> None:
        raise AssertionError("unexpected redispatch for cold single-flight test")

    return await lpe._handle_auto_agent_alias_route(
        alias_family=alias_family,
        alias_model=alias_model,
        request=request,
        prepared_request_body=body,
        max_candidate_attempts=max_candidate_attempts,
        select_candidate_fn=select_candidate_fn,
        add_alias_metadata_fn=_add_metadata,
        perform_candidate_request_fn=perform_candidate_request_fn,
        get_active_cooldown_state_fn=get_active_cooldown_state_fn,
        set_session_affinity_fn=_set_affinity,
        apply_cooldown_fn=apply_cooldown_fn,
        raise_redispatch_required_fn=_raise_redispatch,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="No candidates available for single-flight test.",
        log_label="RR054-31",
    )


# ---------------------------------------------------------------------------
# Lock identity / isolation (foundation for single-flight keys)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_31_candidate_probe_lock_is_shared_per_family_and_cooldown_key() -> (
    None
):
    manager = AliasRoutingStateManager(max_size=16)
    lock_a1 = await manager.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="openrouter:model-a:openrouter",
    )
    lock_a2 = await manager.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="openrouter:model-a:openrouter",
    )
    lock_b = await manager.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="openrouter:model-b:openrouter",
    )
    lock_other_family = await manager.candidate_probe_lock(
        alias_family="anthropic_auto_agent",
        cooldown_key="openrouter:model-a:openrouter",
    )

    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b
    assert lock_a1 is not lock_other_family
    assert "codex_auto_agent:openrouter:model-a:openrouter" in manager.candidate_probe_locks
    assert (
        "anthropic_auto_agent:openrouter:model-a:openrouter"
        in manager.candidate_probe_locks
    )


@pytest.mark.asyncio
async def test_rr054_31_process_singleton_probe_lock_matches_alias_routing_state() -> (
    None
):
    lock = await alias_routing_state.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="rr054-31:singleton",
    )
    again = await lpe._alias_routing_state.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="rr054-31:singleton",
    )
    assert lock is again
    assert lpe._alias_routing_state is alias_routing_state


# ---------------------------------------------------------------------------
# Same candidate/lane: single-flight cold probes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_31_same_candidate_lane_serializes_concurrent_cold_probes() -> None:
    """While a cold probe is in flight, siblings must not also enter upstream.

    Peak concurrent probe entries for one cooldown_key must stay at 1.
    """
    primary = _candidate()
    cooldown_key = "openrouter:openrouter/cohere/north-mini-code:free:openrouter"
    probe = _ProbeCounter()
    probe.hold_seconds = 0.15
    success = Response(content='{"ok":true}', media_type="application/json")
    cooldown_until: dict[str, float] = {}

    async def select_candidate_fn(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.monotonic()
        if cooldown_until.get(cooldown_key, 0.0) > now:
            # After the leader publishes failure/cooldown, followers re-select
            # a distinct success candidate rather than re-probing the cold lane.
            return _selection(
                candidate=_candidate(
                    model="openrouter/owl-alpha",
                    route_family="codex_openrouter_completion_adapter",
                ),
                lane_key="openrouter",
                cooldown_key="openrouter:openrouter/owl-alpha:openrouter",
                session_key=str(request_body["litellm_metadata"]["session_id"]),
            )
        return _selection(
            candidate=primary,
            lane_key="openrouter",
            cooldown_key=cooldown_key,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def get_active_cooldown_state_fn(key: str) -> tuple[float, str]:
        remaining = max(0.0, cooldown_until.get(key, 0.0) - time.monotonic())
        return remaining, "local_fallback"

    async def perform_candidate_request_fn(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        if candidate["model"] == primary["model"]:
            return await probe.run(hold=True, outcome="fail")
        return success

    async def apply_cooldown_fn(
        *,
        request: Request,
        candidate: dict[str, Any],
        lane_key: Optional[str],
        selected_cooldown_key: str,
        cooldown_seconds: float,
        error_class: Optional[str],
    ) -> str:
        # Publish durable/local failure state so waiters can observe it.
        cooldown_until[selected_cooldown_key] = time.monotonic() + max(
            1.0, float(cooldown_seconds or 30.0)
        )
        return "candidate"

    async def one(session_id: str) -> Any:
        return await _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id=session_id,
            select_candidate_fn=select_candidate_fn,
            perform_candidate_request_fn=perform_candidate_request_fn,
            get_active_cooldown_state_fn=get_active_cooldown_state_fn,
            apply_cooldown_fn=apply_cooldown_fn,
            max_candidate_attempts=3,
        )

    results = await asyncio.gather(
        one("sf-session-1"),
        one("sf-session-2"),
        one("sf-session-3"),
        one("sf-session-4"),
        return_exceptions=True,
    )

    # Serialization invariant: never more than one concurrent same-lane probe.
    assert probe.max_current == 1, (
        "RR-054 #31 gap: concurrent cold probes for the same candidate/lane "
        f"entered upstream together (max_current={probe.max_current}, "
        f"total={probe.total}). Expected process-local single-flight."
    )

    # Full single-flight invariant: only the leader probes before cooldown is
    # visible; followers wait, observe failure/cooldown, and do not re-probe.
    assert probe.total == 1, (
        "RR-054 #31 gap: more than one upstream probe ran for the same cold "
        f"candidate/lane before success/failure/cooldown was visible "
        f"(total={probe.total}, max_current={probe.max_current}). "
        "Lock must cover probe + cooldown publish, or waiters must re-check "
        "state before probing."
    )

    # At least the followers should recover onto the alternate candidate once
    # the primary cold-probe outcome is published.
    successes = [r for r in results if isinstance(r, Response)]
    assert len(successes) >= 1, (
        "expected at least one request to recover after single-flight cooldown; "
        f"results={results!r}"
    )


@pytest.mark.asyncio
async def test_rr054_31_same_candidate_lane_success_probe_is_not_concurrent() -> None:
    """Concurrent cold successes for one lane must not fan out in parallel.

    Single-flight may still allow sequential probes after unlock if success
    does not publish a shared positive cache; peak concurrency must stay 1.
    """
    primary = _candidate(model="openrouter/owl-alpha")
    cooldown_key = "openrouter:openrouter/owl-alpha:openrouter"
    probe = _ProbeCounter()
    probe.hold_seconds = 0.1
    success = Response(content='{"ok":true}', media_type="application/json")

    async def select_candidate_fn(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return _selection(
            candidate=primary,
            lane_key="openrouter",
            cooldown_key=cooldown_key,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def get_active_cooldown_state_fn(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    async def perform_candidate_request_fn(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await probe.run(
            hold=True,
            outcome="success",
            success_response=success,
        )

    async def apply_cooldown_fn(**_k: Any) -> str:
        raise AssertionError("success path must not apply cooldown")

    results = await asyncio.gather(
        *[
            _run_route_once(
                alias_family="codex_auto_agent",
                alias_model="aawm-low",
                session_id=f"ok-session-{i}",
                select_candidate_fn=select_candidate_fn,
                perform_candidate_request_fn=perform_candidate_request_fn,
                get_active_cooldown_state_fn=get_active_cooldown_state_fn,
                apply_cooldown_fn=apply_cooldown_fn,
                max_candidate_attempts=1,
            )
            for i in range(3)
        ]
    )

    assert all(r is success for r in results)
    assert probe.max_current == 1, (
        "RR-054 #31 gap: concurrent cold success probes for one candidate/lane "
        f"ran in parallel (max_current={probe.max_current}, total={probe.total})."
    )


@pytest.mark.asyncio
async def test_rr054_31_waiter_skips_probe_after_leader_publishes_cooldown() -> None:
    """Follower that acquires the lane lock after cooldown publish must not probe."""
    primary = _candidate()
    cooldown_key = "openrouter:openrouter/cohere/north-mini-code:free:openrouter"
    probe = _ProbeCounter()
    probe.hold_seconds = 0.2
    # Keep the leader blocked until we explicitly release, so the follower is
    # queued on the lane lock for the entire cold probe.
    probe.release = asyncio.Event()
    success = Response(content='{"ok":true}', media_type="application/json")
    cooldown_until: dict[str, float] = {}
    leader_started = asyncio.Event()

    async def select_candidate_fn(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        if cooldown_until.get(cooldown_key, 0.0) > time.monotonic():
            return _selection(
                candidate=_candidate(model="openrouter/owl-alpha"),
                lane_key="openrouter",
                cooldown_key="openrouter:openrouter/owl-alpha:openrouter",
                session_key=str(request_body["litellm_metadata"]["session_id"]),
            )
        return _selection(
            candidate=primary,
            lane_key="openrouter",
            cooldown_key=cooldown_key,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def get_active_cooldown_state_fn(key: str) -> tuple[float, str]:
        remaining = max(0.0, cooldown_until.get(key, 0.0) - time.monotonic())
        return remaining, "local_fallback"

    async def perform_candidate_request_fn(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        if candidate["model"] == primary["model"]:
            leader_started.set()
            return await probe.run(hold=True, outcome="fail")
        return success

    async def apply_cooldown_fn(
        *,
        request: Request,
        candidate: dict[str, Any],
        lane_key: Optional[str],
        selected_cooldown_key: str,
        cooldown_seconds: float,
        error_class: Optional[str],
    ) -> str:
        cooldown_until[selected_cooldown_key] = time.monotonic() + 60.0
        return "candidate"

    leader_task = asyncio.create_task(
        _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id="leader",
            select_candidate_fn=select_candidate_fn,
            perform_candidate_request_fn=perform_candidate_request_fn,
            get_active_cooldown_state_fn=get_active_cooldown_state_fn,
            apply_cooldown_fn=apply_cooldown_fn,
            max_candidate_attempts=3,
        )
    )
    await asyncio.wait_for(leader_started.wait(), timeout=1.0)

    follower_task = asyncio.create_task(
        _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id="follower",
            select_candidate_fn=select_candidate_fn,
            perform_candidate_request_fn=perform_candidate_request_fn,
            get_active_cooldown_state_fn=get_active_cooldown_state_fn,
            apply_cooldown_fn=apply_cooldown_fn,
            max_candidate_attempts=3,
        )
    )
    # Give the follower time to block on the same lane lock.
    await asyncio.sleep(0.05)
    assert probe.total == 1
    assert probe.current == 1

    probe.release.set()
    results = await asyncio.gather(leader_task, follower_task, return_exceptions=True)

    assert probe.total == 1, (
        "RR-054 #31 gap: follower re-probed the same candidate/lane after the "
        f"leader published failure/cooldown (total={probe.total})."
    )
    assert probe.max_current == 1
    assert any(isinstance(r, Response) for r in results)


# ---------------------------------------------------------------------------
# Distinct candidates/lanes remain independent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_31_distinct_candidates_or_lanes_probe_independently() -> None:
    """Different cooldown keys must not share a single-flight barrier."""
    cand_a = _candidate(model="openrouter/cohere/north-mini-code:free")
    cand_b = _candidate(model="openrouter/owl-alpha")
    key_a = "openrouter:openrouter/cohere/north-mini-code:free:openrouter"
    key_b = "openrouter:openrouter/owl-alpha:openrouter"

    probe_a = _ProbeCounter()
    probe_b = _ProbeCounter()
    probe_a.hold_seconds = 0.15
    probe_b.hold_seconds = 0.15
    both_entered = asyncio.Event()
    success = Response(content='{"ok":true}', media_type="application/json")

    async def select_a(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return _selection(
            candidate=cand_a,
            lane_key="openrouter",
            cooldown_key=key_a,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def select_b(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return _selection(
            candidate=cand_b,
            lane_key="openrouter",
            cooldown_key=key_b,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def get_active(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    async def perform_a(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await probe_a.run(
            hold=True,
            outcome="success",
            success_response=success,
        )

    async def perform_b(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await probe_b.run(
            hold=True,
            outcome="success",
            success_response=success,
        )

    async def apply_cooldown(**_k: Any) -> str:
        return "none"

    async def watch_both() -> None:
        while not (probe_a.current >= 1 and probe_b.current >= 1):
            await asyncio.sleep(0)
        both_entered.set()

    watcher = asyncio.create_task(watch_both())
    results = await asyncio.gather(
        _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id="lane-a",
            select_candidate_fn=select_a,
            perform_candidate_request_fn=perform_a,
            get_active_cooldown_state_fn=get_active,
            apply_cooldown_fn=apply_cooldown,
            max_candidate_attempts=1,
        ),
        _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id="lane-b",
            select_candidate_fn=select_b,
            perform_candidate_request_fn=perform_b,
            get_active_cooldown_state_fn=get_active,
            apply_cooldown_fn=apply_cooldown,
            max_candidate_attempts=1,
        ),
    )
    await asyncio.wait_for(watcher, timeout=1.0)

    assert results == [success, success]
    assert probe_a.total == 1
    assert probe_b.total == 1
    assert both_entered.is_set(), (
        "RR-054 #31 regression: distinct candidate/lane probes were serialized "
        "against each other; lanes must remain independent."
    )


@pytest.mark.asyncio
async def test_rr054_31_family_isolation_allows_parallel_probes_for_same_cooldown_key() -> (
    None
):
    """Codex and Anthropic families use independent probe locks for the same key."""
    cand = _candidate()
    shared_key = "openrouter:openrouter/cohere/north-mini-code:free:openrouter"
    probe_codex = _ProbeCounter()
    probe_anth = _ProbeCounter()
    probe_codex.hold_seconds = 0.15
    probe_anth.hold_seconds = 0.15
    success = Response(content='{"ok":true}', media_type="application/json")
    both_entered = asyncio.Event()

    async def select_fn(
        *,
        request: Request,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return _selection(
            candidate=cand,
            lane_key="openrouter",
            cooldown_key=shared_key,
            session_key=str(request_body["litellm_metadata"]["session_id"]),
        )

    async def get_active(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    async def perform_codex(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await probe_codex.run(
            hold=True, outcome="success", success_response=success
        )

    async def perform_anth(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await probe_anth.run(
            hold=True, outcome="success", success_response=success
        )

    async def apply_cooldown(**_k: Any) -> str:
        return "none"

    async def watch_both() -> None:
        while not (probe_codex.current >= 1 and probe_anth.current >= 1):
            await asyncio.sleep(0)
        both_entered.set()

    watcher = asyncio.create_task(watch_both())
    results = await asyncio.gather(
        _run_route_once(
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            session_id="family-codex",
            select_candidate_fn=select_fn,
            perform_candidate_request_fn=perform_codex,
            get_active_cooldown_state_fn=get_active,
            apply_cooldown_fn=apply_cooldown,
            max_candidate_attempts=1,
        ),
        _run_route_once(
            alias_family="anthropic_auto_agent",
            alias_model="aawm-low-anthropic",
            session_id="family-anth",
            select_candidate_fn=select_fn,
            perform_candidate_request_fn=perform_anth,
            get_active_cooldown_state_fn=get_active,
            apply_cooldown_fn=apply_cooldown,
            max_candidate_attempts=1,
        ),
    )
    await asyncio.wait_for(watcher, timeout=1.0)

    assert results == [success, success]
    assert both_entered.is_set()
    assert probe_codex.total == 1
    assert probe_anth.total == 1


# ---------------------------------------------------------------------------
# Integration shape through Codex wrapper wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_31_codex_wrapper_wires_probe_lock_path_for_concurrent_cold_failure() -> (
    None
):
    """Exercise the production Codex wrapper entry with concurrent cold failures.

    Uses real select + cooldown maps; only the upstream perform path is mocked.
    """
    success = Response(content='{"ok":true}', media_type="application/json")
    probe = _ProbeCounter()
    probe.hold_seconds = 0.12

    primary_model = "openrouter/cohere/north-mini-code:free"
    fallback_model = "openrouter/owl-alpha"

    async def openrouter_completion(
        *,
        request: Request,
        adapter_model: str,
        request_body: dict[str, Any],
        use_alias_candidate_probe: bool = False,
    ) -> Response:
        if adapter_model == primary_model:
            return await probe.run(hold=True, outcome="fail")
        if adapter_model == fallback_model:
            return success
        raise AssertionError(f"unexpected adapter_model={adapter_model}")

    # Avoid durable Redis in this pure concurrency unit.
    original_write = lpe._write_aawm_alias_routing_durable_payload
    original_perform = lpe._perform_codex_auto_agent_openrouter_completion_request

    lpe._write_aawm_alias_routing_durable_payload = AsyncMock(return_value=True)  # type: ignore[assignment]
    lpe._perform_codex_auto_agent_openrouter_completion_request = openrouter_completion  # type: ignore[assignment]
    try:

        async def one(session_id: str) -> Any:
            request = _minimal_request(session_id)
            body = {
                "model": "aawm-low",
                "input": [{"role": "user", "content": "hello"}],
                "stream": False,
                "litellm_metadata": {"session_id": session_id},
            }
            return await lpe._handle_codex_auto_agent_alias_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                target_url="https://chatgpt.com/backend-api/codex/responses",
                api_key=None,
                forward_headers=True,
            )

        results = await asyncio.gather(
            one("wrap-1"),
            one("wrap-2"),
            one("wrap-3"),
            return_exceptions=True,
        )
    finally:
        lpe._write_aawm_alias_routing_durable_payload = original_write  # type: ignore[assignment]
        lpe._perform_codex_auto_agent_openrouter_completion_request = original_perform  # type: ignore[assignment]

    assert probe.max_current == 1, (
        "RR-054 #31 gap via Codex wrapper: concurrent cold primary probes "
        f"overlapped (max_current={probe.max_current}, total={probe.total})."
    )
    assert probe.total == 1, (
        "RR-054 #31 gap via Codex wrapper: primary candidate was probed more "
        f"than once before cooldown was visible (total={probe.total})."
    )
    successes = [r for r in results if isinstance(r, Response)]
    assert len(successes) >= 1, f"unexpected wrapper outcomes: {results!r}"
