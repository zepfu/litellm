"""Regression: probe lock must be released unconditionally.

Review finding: ``_resolve_and_publish_failure_memory(...)`` was awaited in
a ``finally`` before ``probe_lock.release()``, so resolver/publisher failure
or cancellation permanently leaked the lock and later same-key requests hung.

These tests drive the production-shaped ``handle_alias_route`` wrapper and
prove:
1. Resolver/publisher errors surface to the caller (not swallowed).
2. The probe lock is unlocked after resolver/publisher failure.
3. A subsequent same-key request does NOT hang (acquires the lock promptly).
4. Production ``get_active_cooldown_state_fn`` callbacks conform to the
   ``GetActiveCooldownStateFn`` protocol signature.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    handle_alias_route,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
    GetActiveCooldownStateFn,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_request(session_id: str = "lock-leak-test") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": session_id}
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


def _candidate() -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "model": "openrouter/test-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    }


def _selection() -> dict[str, Any]:
    return {
        "candidate": _candidate(),
        "lane_key": "openrouter",
        "cooldown_key": "openrouter:openrouter/test-model:openrouter",
        "session_key": "lock-leak-session",
        "selection_reason": "first_available",
        "skipped": [],
        "in_flight_session": False,
        "cooldown_seconds": 0.0,
        "cooldown_state_source": "local_fallback",
    }


def _make_services(
    *,
    perform_exc: Optional[Exception] = None,
    resolve_exc: Optional[Exception] = None,
    publish_exc: Optional[Exception] = None,
    perform_response: Optional[Response] = None,
) -> AliasRouteServices:
    """Build a minimal AliasRouteServices with controllable failure injection."""

    async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
        return _selection()

    async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
        if perform_exc is not None:
            raise perform_exc
        assert perform_response is not None
        return perform_response

    def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
        if resolve_exc is not None:
            raise resolve_exc
        return CooldownPublicationPlan(
            memory_keys=("test-key",),
            durable_keys=(),
            duration_seconds=30.0,
            applied_scope="candidate",
        )

    def _publish(*, keys: Any, seconds: Any) -> None:
        if publish_exc is not None:
            raise publish_exc

    async def _persist(*, keys: Any, seconds: Any) -> None:
        pass

    async def _set_affinity(*_a: Any, **_k: Any) -> None:
        pass

    def _add_metadata(request_body: dict[str, Any], **_k: Any) -> dict[str, Any]:
        return request_body

    def _raise_redispatch(**_k: Any) -> None:
        raise AssertionError("unexpected redispatch")

    return AliasRouteServices(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=_resolve,
        publish_cooldown_memory_fn=_publish,
        persist_cooldown_fn=_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=_add_metadata,
        raise_redispatch_fn=_raise_redispatch,
    )


async def _get_active_cooldown_noop(cooldown_key: str) -> tuple[float, str]:
    return 0.0, "local_fallback"


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    reset_fn = getattr(lpe, "reset_alias_routing_state_for_tests", None)
    if reset_fn is not None:
        reset_fn()
    else:
        alias_routing_state.candidate_probe_locks.clear()
    yield
    if reset_fn is not None:
        reset_fn()
    else:
        alias_routing_state.candidate_probe_locks.clear()


# ---------------------------------------------------------------------------
# Test 1: resolver failure surfaces AND lock is released
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_failure_surfaces_and_lock_released() -> None:
    """When the publication-plan resolver raises, the exception propagates
    and the probe lock is NOT leaked."""
    resolver_error = ValueError("resolver exploded")
    services = _make_services(
        perform_exc=RuntimeError("capacity"),
        resolve_exc=resolver_error,
    )
    request = _minimal_request()
    cooldown_key = "openrouter:openrouter/test-model:openrouter"

    with pytest.raises(ValueError, match="resolver exploded"):
        await handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            request=request,
            prepared_request_body={"model": "aawm-low"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        )

    # Lock must be released: a fresh acquire must succeed immediately.
    lock = await alias_routing_state.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key=cooldown_key,
    )
    acquired = False
    try:
        await asyncio.wait_for(lock.acquire(), timeout=0.5)
        acquired = True
    finally:
        if acquired:
            lock.release()
    assert acquired, "probe lock was leaked after resolver failure"


# ---------------------------------------------------------------------------
# Test 2: publisher failure surfaces AND lock is released
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publisher_failure_surfaces_and_lock_released() -> None:
    """When the memory publisher raises, the exception propagates and the
    probe lock is NOT leaked."""
    publisher_error = OSError("publish failed")
    services = _make_services(
        perform_exc=RuntimeError("capacity"),
        publish_exc=publisher_error,
    )
    request = _minimal_request()
    cooldown_key = "openrouter:openrouter/test-model:openrouter"

    with pytest.raises(OSError, match="publish failed"):
        await handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            request=request,
            prepared_request_body={"model": "aawm-low"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        )

    lock = await alias_routing_state.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key=cooldown_key,
    )
    acquired = False
    try:
        await asyncio.wait_for(lock.acquire(), timeout=0.5)
        acquired = True
    finally:
        if acquired:
            lock.release()
    assert acquired, "probe lock was leaked after publisher failure"


# ---------------------------------------------------------------------------
# Test 3: cancellation during suspended publication releases the lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_during_publisher_releases_lock_and_next_request_completes() -> None:
    """Cancellation while an async legacy publisher is suspended propagates,
    releases the probe lock, and does not hang the next same-key request."""
    perform_call_count = 0
    publisher_started = asyncio.Event()
    publisher_blocker = asyncio.Event()

    async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
        return _selection()

    async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
        nonlocal perform_call_count
        perform_call_count += 1
        if perform_call_count == 1:
            raise RuntimeError("first call fails")
        return Response(content='{"ok":true}', media_type="application/json")

    def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
        return CooldownPublicationPlan(
            memory_keys=("test-key",),
            durable_keys=(),
            duration_seconds=30.0,
            applied_scope="candidate",
        )

    async def _publish(*, keys: Any, seconds: Any) -> None:
        publisher_started.set()
        await publisher_blocker.wait()

    async def _persist(*, keys: Any, seconds: Any) -> None:
        pass

    async def _set_affinity(*_a: Any, **_k: Any) -> None:
        pass

    def _add_metadata(request_body: dict[str, Any], **_k: Any) -> dict[str, Any]:
        return request_body

    def _raise_redispatch(**_k: Any) -> None:
        raise AssertionError("unexpected redispatch")

    services = AliasRouteServices(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=_resolve,
        publish_cooldown_memory_fn=cast(Any, _publish),
        persist_cooldown_fn=_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=_add_metadata,
        raise_redispatch_fn=_raise_redispatch,
    )

    async def _run_route(request: Request) -> Response:
        return await handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            request=request,
            prepared_request_body={"model": "aawm-low"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        )

    first_request = asyncio.create_task(
        _run_route(_minimal_request("cancelled-session"))
    )
    await asyncio.wait_for(publisher_started.wait(), timeout=1.0)
    first_request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_request

    cooldown_key = "openrouter:openrouter/test-model:openrouter"
    lock = await alias_routing_state.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key=cooldown_key,
    )
    assert not lock.locked(), "probe lock was leaked after publisher cancellation"

    response = await asyncio.wait_for(
        _run_route(_minimal_request("post-cancellation-session")),
        timeout=2.0,
    )
    assert isinstance(response, Response)
    assert perform_call_count == 2


# ---------------------------------------------------------------------------
# Test 4: subsequent same-key request does NOT hang after failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subsequent_same_key_request_does_not_hang_after_failure() -> None:
    """After a resolver/publisher failure leaks no lock, a second request
    for the same cooldown_key completes promptly (does not hang)."""
    call_count = 0

    async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
        return _selection()

    async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first call fails")
        return Response(content='{"ok":true}', media_type="application/json")

    resolve_call_count = 0

    def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
        nonlocal resolve_call_count
        resolve_call_count += 1
        if resolve_call_count == 1:
            raise ValueError("resolver fails on first attempt")
        return CooldownPublicationPlan(
            memory_keys=(),
            durable_keys=(),
            duration_seconds=0.0,
            applied_scope="none",
        )

    def _publish(*, keys: Any, seconds: Any) -> None:
        pass

    async def _persist(*, keys: Any, seconds: Any) -> None:
        pass

    async def _set_affinity(*_a: Any, **_k: Any) -> None:
        pass

    def _add_metadata(request_body: dict[str, Any], **_k: Any) -> dict[str, Any]:
        return request_body

    def _raise_redispatch(**_k: Any) -> None:
        raise AssertionError("unexpected redispatch")

    services = AliasRouteServices(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=_resolve,
        publish_cooldown_memory_fn=_publish,
        persist_cooldown_fn=_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=_add_metadata,
        raise_redispatch_fn=_raise_redispatch,
    )

    # First request: resolver fails, exception propagates.
    request1 = _minimal_request("session-1")
    with pytest.raises(ValueError, match="resolver fails"):
        await handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            request=request1,
            prepared_request_body={"model": "aawm-low"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        )

    # Second request: same cooldown_key, must NOT hang.
    request2 = _minimal_request("session-2")
    response = await asyncio.wait_for(
        handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="aawm-low",
            request=request2,
            prepared_request_body={"model": "aawm-low"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        ),
        timeout=2.0,
    )
    assert isinstance(response, Response)
    assert call_count == 2


# ---------------------------------------------------------------------------
# Test 5: production get_active_cooldown_state_fn callbacks conform to
# GetActiveCooldownStateFn protocol signature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_get_active_cooldown_state_fn_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both production cooldown-state callbacks accept a single positional
    ``cooldown_key: str`` and their actual awaited result is a
    ``(float, str)`` tuple, matching ``GetActiveCooldownStateFn``."""
    lpe.reset_alias_routing_state_for_tests()
    monkeypatch.setattr(lpe, "_get_aawm_alias_routing_dual_cache", lambda: None)
    codex_fn = lpe._get_codex_auto_agent_active_cooldown_state
    anthropic_fn = lpe._get_anthropic_auto_agent_active_cooldown_state

    for fn, name in [(codex_fn, "codex"), (anthropic_fn, "anthropic")]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        # Must accept exactly one positional parameter (cooldown_key).
        positional = [
            p
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert len(positional) == 1, (
            f"{name} get_active_cooldown_state_fn must accept exactly one "
            f"positional parameter, got {len(positional)}: {sig}"
        )
        # The parameter must be annotated str (or unannotated, accepting str).
        param = positional[0]
        if param.annotation is not inspect.Parameter.empty:
            assert param.annotation is str or param.annotation == "str", (
                f"{name} cooldown_key parameter must be str, got {param.annotation}"
            )
        # Must be a coroutine function (returns Awaitable).
        assert inspect.iscoroutinefunction(fn), (
            f"{name} get_active_cooldown_state_fn must be async"
        )
        result = await fn(f"qa-contract-isolated-{name}")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)


def test_get_active_cooldown_state_fn_protocol_is_runtime_checkable() -> None:
    """The GetActiveCooldownStateFn protocol is importable and runtime-checkable."""
    assert hasattr(GetActiveCooldownStateFn, "__call__")

    # A conforming callable passes isinstance.
    async def _conforming(cooldown_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    assert isinstance(_conforming, GetActiveCooldownStateFn)
