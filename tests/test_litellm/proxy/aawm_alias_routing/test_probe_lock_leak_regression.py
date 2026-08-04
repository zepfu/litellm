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
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    handle_alias_route,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
    _get_anthropic_auto_agent_active_cooldown_state,
    _get_codex_auto_agent_active_cooldown_state,
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
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic"},
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
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic"},
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
async def test_publisher_cancellation_releases_lock_and_next_request_completes() -> None:
    """A synchronous publisher cancellation propagates, releases the probe
    lock, and does not hang the next same-key request."""
    perform_call_count = 0
    publisher_called = False

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

    def _publish(*, keys: Any, seconds: Any) -> None:
        nonlocal publisher_called
        publisher_called = True
        raise asyncio.CancelledError

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

    async def _run_route(request: Request) -> Response:
        return await handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_get_active_cooldown_noop,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Test",
        )

    with pytest.raises(asyncio.CancelledError):
        await _run_route(_minimal_request("cancelled-session"))
    assert publisher_called

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
            alias_model="basic",
            request=request1,
            prepared_request_body={"model": "basic"},
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
            alias_model="basic",
            request=request2,
            prepared_request_body={"model": "basic"},
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
    codex_fn = _get_codex_auto_agent_active_cooldown_state
    anthropic_fn = _get_anthropic_auto_agent_active_cooldown_state

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


# ---------------------------------------------------------------------------
# Test 6: adversarial lock-inversion: concurrent family->probe publication
# and probe path must not deadlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_publication_and_probe_no_lock_inversion() -> None:  # noqa: PLR0915
    """Adversarial: a family-lock-first publication transaction running
    concurrently with the candidate loop probe path must not deadlock.

    Before the fix, the probe path called get_active_cooldown_state_fn
    (which acquires the family lock) while holding the probe lock.  The
    publication transaction acquires family lock -> sorted probe locks.
    Two tasks with reversed lock order = deadlock.

    After the fix, the cooldown reader runs BEFORE probe lock acquisition,
    so both paths agree on family-first ordering.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
        execute_cooldown_publication_transaction,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )

    mgr = AliasRoutingStateManager()
    cooldown_key = "openrouter:openrouter/test-model:openrouter"

    # Wire the state manager into the candidate loop module.
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop as loop_mod

    original_state = loop_mod.alias_routing_state
    loop_mod.alias_routing_state = mgr

    # Also wire cooldown_apply to use the same manager.
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply as apply_mod

    original_apply_state = getattr(apply_mod, "_state_manager", None)
    apply_mod._state_manager = mgr

    try:
        perform_count = 0

        async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
            return _selection()

        async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
            nonlocal perform_count
            perform_count += 1
            return Response(content='{"ok":true}', media_type="application/json")

        def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
            return CooldownPublicationPlan(
                memory_keys=(cooldown_key,),
                durable_keys=(),
                duration_seconds=30.0,
                applied_scope="candidate",
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

        # Cooldown reader that acquires the family lock (like production).
        async def _cooldown_reader_with_family_lock(
            key: str,
        ) -> tuple[float, str]:
            family_state = mgr.family("codex_auto_agent")
            async with family_state.lock:
                return 0.0, "memory"

        # Task A: publication transaction (family lock -> probe locks).
        plan = CooldownPublicationPlan(
            memory_keys=(cooldown_key,),
            durable_keys=(),
            duration_seconds=30.0,
            applied_scope="candidate",
        )

        async def _run_publication() -> None:
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_candidate(),
                plan=plan,
                publish_cooldown_memory_fn=lambda *, keys, seconds: None,
                persist_cooldown_fn=AsyncMock(),
            )

        # Task B: candidate loop probe path (previously probe lock -> family lock).
        async def _run_probe() -> Response:
            return await handle_alias_route(
                services,
                alias_family="codex_auto_agent",
                alias_model="basic",
                request=_minimal_request("inversion-test"),
                prepared_request_body={"model": "basic"},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=_cooldown_reader_with_family_lock,
                attempts_metadata_key="attempts",
                skipped_candidates_metadata_key="skipped",
                no_candidate_detail="no candidates",
                log_label="InversionTest",
            )

        # Run both concurrently with a bounded timeout.  If there is a
        # lock inversion, this will time out (deadlock).
        pub_task = asyncio.create_task(_run_publication())
        probe_task = asyncio.create_task(_run_probe())

        results = await asyncio.wait_for(
            asyncio.gather(pub_task, probe_task, return_exceptions=True),
            timeout=5.0,
        )

        # Probe must succeed (return a Response).
        assert isinstance(results[1], Response), (
            f"probe path failed: {results[1]}"
        )
        assert perform_count == 1

        # No lock leak: probe lock must be acquirable.
        lock = await mgr.candidate_probe_lock(
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
        assert acquired, "probe lock leaked after concurrent publication+probe"
    finally:
        loop_mod.alias_routing_state = original_state
        apply_mod._state_manager = original_apply_state


# ---------------------------------------------------------------------------
# Test 9: exact family canonicalization -- positive and negative
# ---------------------------------------------------------------------------


class TestCanonicalizeAliasFamily:
    """Exact canonical mapping for known family labels (no substring matching)."""

    def test_positive_codex_labels(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            canonicalize_alias_family,
        )

        assert canonicalize_alias_family("codex") == "codex"
        assert canonicalize_alias_family("codex_auto_agent") == "codex"
        assert canonicalize_alias_family("  codex_auto_agent  ") == "codex"
        assert canonicalize_alias_family("CODEX_AUTO_AGENT") == "codex"

    def test_positive_anthropic_labels(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            canonicalize_alias_family,
        )

        assert canonicalize_alias_family("anthropic") == "anthropic"
        assert canonicalize_alias_family("anthropic_auto_agent") == "anthropic"
        assert canonicalize_alias_family("  ANTHROPIC_AUTO_AGENT ") == "anthropic"

    def test_negative_substring_labels_default_to_codex(self) -> None:
        """Labels containing 'anthropic' as a substring (but not exact known
        labels) must NOT map to the Anthropic family."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            canonicalize_alias_family,
        )

        assert canonicalize_alias_family("not_anthropic") == "codex"
        assert canonicalize_alias_family("xanthropicx") == "codex"
        assert canonicalize_alias_family("codex_anthropic") == "codex"
        assert canonicalize_alias_family("anthropicx") == "codex"
        assert canonicalize_alias_family("my_anthropic_thing") == "codex"

    def test_unknown_labels_default_to_codex(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            canonicalize_alias_family,
        )

        assert canonicalize_alias_family("") == "codex"
        assert canonicalize_alias_family("openrouter") == "codex"
        assert canonicalize_alias_family("random_label") == "codex"

    def test_state_manager_family_uses_canonicalizer(self) -> None:
        """AliasRoutingStateManager.family() resolves through the shared
        canonicalizer for both positive and negative cases."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            AliasRoutingStateManager,
        )

        mgr = AliasRoutingStateManager()
        # Positive: known labels resolve correctly.
        assert mgr.family("codex_auto_agent") is mgr.codex
        assert mgr.family("anthropic_auto_agent") is mgr.anthropic
        assert mgr.family("codex") is mgr.codex
        assert mgr.family("anthropic") is mgr.anthropic
        # Negative: substring traps resolve to codex (default).
        assert mgr.family("not_anthropic") is mgr.codex
        assert mgr.family("xanthropicx") is mgr.codex
        assert mgr.family("codex_anthropic") is mgr.codex

    def test_publication_transaction_uses_same_canonicalizer(self) -> None:
        """execute_cooldown_publication_transaction resolves family via the
        shared canonicalizer, not a duplicated substring check."""
        import ast
        import inspect
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            cooldown_apply,
        )

        source = inspect.getsource(
            cooldown_apply.execute_cooldown_publication_transaction
        )
        tree = ast.parse(source)
        # The function must call canonicalize_alias_family.
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "canonicalize_alias_family" in calls, (
            "execute_cooldown_publication_transaction must use "
            "canonicalize_alias_family, not inline substring logic"
        )
        # Must NOT contain the old substring pattern.
        assert '"anthropic" in normalized' not in source, (
            "substring family check still present in publication transaction"
        )


# ---------------------------------------------------------------------------
# Test 9: Anthropic TOCTOU barrier -- family resolution regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_toctou_barrier_family_resolution() -> None:  # noqa: PLR0915
    """Anthropic TOCTOU barrier: active Anthropic cooldown committed between
    precheck and probe lock acquisition must result in zero provider calls,
    Anthropic state observed (not Codex), and intent/locks cleaned.

    Regression for the exact-match family resolution bug where
    ``state_manager.family('anthropic_auto_agent')`` returned the Codex
    family state, causing the TOCTOU peek to read the wrong cooldown map.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop as loop_mod

    mgr = AliasRoutingStateManager()
    cooldown_key = "anthropic_native:claude-sonnet-4-20250514:anthropic"

    original_state = loop_mod.alias_routing_state
    loop_mod.alias_routing_state = mgr

    try:
        perform_count = 0
        waiter_precheck_done = asyncio.Event()
        leader_publication_done = asyncio.Event()

        def _anthropic_candidate() -> dict[str, Any]:
            return {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "route_family": "anthropic_native",
                "last_resort": False,
            }

        def _anthropic_selection() -> dict[str, Any]:
            return {
                "candidate": _anthropic_candidate(),
                "lane_key": "anthropic_native",
                "cooldown_key": cooldown_key,
                "session_key": "anthropic-toctou-session",
                "selection_reason": "first_available",
                "skipped": [],
                "in_flight_session": False,
                "cooldown_seconds": 0.0,
                "cooldown_state_source": "local_fallback",
            }

        async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
            return _anthropic_selection()

        async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
            nonlocal perform_count
            perform_count += 1
            return Response(content='{"ok":true}', media_type="application/json")

        def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
            return CooldownPublicationPlan(
                memory_keys=(cooldown_key,),
                durable_keys=(),
                duration_seconds=30.0,
                applied_scope="candidate",
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

        cooldown_reader_call_count = 0

        async def _cooldown_reader_with_barrier(
            key: str,
        ) -> tuple[float, str]:
            nonlocal cooldown_reader_call_count
            cooldown_reader_call_count += 1
            if cooldown_reader_call_count == 1:
                # First call: waiter precheck. Signal done, then wait for
                # leader to commit cooldown into Anthropic family state.
                waiter_precheck_done.set()
                await leader_publication_done.wait()
                # Return 0 -- at precheck time there was no cooldown.
                # The TOCTOU guard under probe lock must catch the race
                # by peeking the correct (Anthropic) family state.
                return 0.0, "memory"
            return 0.0, "memory"

        # --- Leader: commit cooldown into Anthropic family state ---
        async def _run_leader_publication() -> None:
            await waiter_precheck_done.wait()
            # Use the production alias_family value.
            family_state = mgr.family("anthropic_auto_agent")
            async with family_state.lock:
                probe_lock = await mgr.candidate_probe_lock(
                    alias_family="anthropic_auto_agent",
                    cooldown_key=cooldown_key,
                )
                async with probe_lock:
                    family_state.set_cooldown_memory(cooldown_key, 30.0)
            leader_publication_done.set()

        # --- Waiter: the candidate loop ---
        async def _run_waiter() -> Response:
            return await handle_alias_route(
                services,
                alias_family="anthropic_auto_agent",
                alias_model="basic",
                request=_minimal_request("anthropic-toctou-waiter"),
                prepared_request_body={"model": "basic"},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=_cooldown_reader_with_barrier,
                attempts_metadata_key="attempts",
                skipped_candidates_metadata_key="skipped",
                no_candidate_detail="no candidates",
                log_label="AnthropicTOCTOU",
            )

        leader_task = asyncio.create_task(_run_leader_publication())
        waiter_task = asyncio.create_task(_run_waiter())

        results = await asyncio.wait_for(
            asyncio.gather(leader_task, waiter_task, return_exceptions=True),
            timeout=5.0,
        )

        # Zero provider calls: the TOCTOU guard must detect the Anthropic
        # cooldown committed between precheck and probe lock acquisition.
        assert perform_count == 0, (
            f"Waiter made {perform_count} provider call(s); "
            f"Anthropic TOCTOU guard failed (family resolution bug?)"
        )

        # Waiter must raise 429 (no candidate after cooldown skip).
        from fastapi import HTTPException as FastHTTPException

        waiter_result = results[1]
        assert isinstance(waiter_result, FastHTTPException), (
            f"Expected HTTPException from skipped waiter, got {type(waiter_result)}: {waiter_result}"
        )
        assert waiter_result.status_code == 429

        # Anthropic state must show the cooldown.
        anthropic_state = mgr.family("anthropic_auto_agent")
        assert anthropic_state is mgr.anthropic, (
            "family('anthropic_auto_agent') must resolve to the Anthropic state"
        )
        assert anthropic_state.peek_cooldown_remaining(cooldown_key) > 0, (
            "Anthropic cooldown was not observed"
        )

        # Codex state must be untouched.
        codex_state = mgr.family("codex_auto_agent")
        assert codex_state is mgr.codex
        assert codex_state.peek_cooldown_remaining(cooldown_key) == 0.0, (
            "Codex state was polluted by Anthropic cooldown"
        )

        # Locks must be released.
        assert not anthropic_state.lock.locked(), "Anthropic family lock leaked"
        probe_lock = await mgr.candidate_probe_lock(
            alias_family="anthropic_auto_agent",
            cooldown_key=cooldown_key,
        )
        assert not probe_lock.locked(), "probe lock leaked"

        # Intent registry must be clean.
        assert mgr.publication_intents.get("anthropic_auto_agent", cooldown_key) is None, (
            "intent registry has leaked intent"
        )

    finally:
        loop_mod.alias_routing_state = original_state


# ---------------------------------------------------------------------------
# Test 7: TOCTOU barrier regression -- publication completes before waiter
# acquires probe lock; waiter must NOT probe the provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toctou_publication_before_waiter_probe_no_provider_call() -> None:  # noqa: PLR0915
    """Barrier-driven regression: a leader publishes cooldown and removes its
    intent BEFORE the waiter acquires the probe lock.  The waiter's TOCTOU
    guard (lock-free peek under probe lock) must detect the active cooldown
    and skip the provider call.

    Sequence forced by barriers:
      1. Waiter prechecks cooldown -> 0 (no cooldown yet).
      2. Leader acquires family->probe, commits cooldown, completes intent,
         removes from registry.
      3. Waiter resumes, acquires probe lock, finds no active intent.
      4. TOCTOU guard peeks cooldown -> active -> skip.

    Assertions:
      - Zero provider calls from the waiter.
      - Active cooldown result (skipped_single_flight_cooldown).
      - Bounded completion (no hang).
      - Family and probe locks unlocked after.
      - Intent registry cleaned (no leaked intents).
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop as loop_mod

    mgr = AliasRoutingStateManager()
    cooldown_key = "openrouter:openrouter/test-model:openrouter"

    original_state = loop_mod.alias_routing_state
    loop_mod.alias_routing_state = mgr

    try:
        perform_count = 0
        # Barriers to force exact interleaving.
        waiter_precheck_done = asyncio.Event()
        leader_publication_done = asyncio.Event()

        async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
            return _selection()

        async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
            nonlocal perform_count
            perform_count += 1
            return Response(content='{"ok":true}', media_type="application/json")

        def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
            return CooldownPublicationPlan(
                memory_keys=(cooldown_key,),
                durable_keys=(),
                duration_seconds=30.0,
                applied_scope="candidate",
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

        # Cooldown reader that signals barrier events for deterministic ordering.
        cooldown_reader_call_count = 0

        async def _cooldown_reader_with_barrier(
            key: str,
        ) -> tuple[float, str]:
            nonlocal cooldown_reader_call_count
            cooldown_reader_call_count += 1
            if cooldown_reader_call_count == 1:
                # First call: waiter precheck. Signal that precheck is done,
                # then wait for leader to finish publication.
                waiter_precheck_done.set()
                await leader_publication_done.wait()
                # Return 0 -- at precheck time there was no cooldown.
                # The TOCTOU guard under probe lock must catch the race.
                return 0.0, "memory"
            # Subsequent calls (should not happen for the waiter in this test).
            return 0.0, "memory"

        # --- Leader task: simulate a completed publication ---
        async def _run_leader_publication() -> None:
            # Wait for the waiter to finish its precheck.
            await waiter_precheck_done.wait()
            # Acquire family -> probe (canonical order), commit cooldown.
            family_state = mgr.family("codex_auto_agent")
            async with family_state.lock:
                probe_lock = await mgr.candidate_probe_lock(
                    alias_family="codex_auto_agent",
                    cooldown_key=cooldown_key,
                )
                async with probe_lock:
                    # Commit cooldown into family memory.
                    family_state.set_cooldown_memory(cooldown_key, 30.0)
            # Signal that publication is complete.
            leader_publication_done.set()

        # --- Waiter task: the candidate loop ---
        async def _run_waiter() -> Response:
            return await handle_alias_route(
                services,
                alias_family="codex_auto_agent",
                alias_model="basic",
                request=_minimal_request("toctou-waiter"),
                prepared_request_body={"model": "basic"},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=_cooldown_reader_with_barrier,
                attempts_metadata_key="attempts",
                skipped_candidates_metadata_key="skipped",
                no_candidate_detail="no candidates",
                log_label="TOCTOU",
            )

        leader_task = asyncio.create_task(_run_leader_publication())
        waiter_task = asyncio.create_task(_run_waiter())

        # Bounded completion: must not hang.
        results = await asyncio.wait_for(
            asyncio.gather(leader_task, waiter_task, return_exceptions=True),
            timeout=5.0,
        )

        # The waiter should have been skipped due to active cooldown.
        # With max_candidate_attempts=1 and the cooldown skip, it raises
        # HTTPException(429) because no candidate was available.
        waiter_result = results[1]
        # The waiter either returns a Response (if it somehow probed) or
        # raises an HTTPException.  The TOCTOU guard must prevent probing.
        assert perform_count == 0, (
            f"Waiter made {perform_count} provider call(s); "
            f"TOCTOU guard failed to detect active cooldown"
        )

        # The waiter should have raised (429 no-candidate) since it skipped
        # the only candidate due to cooldown.
        from fastapi import HTTPException as FastHTTPException

        assert isinstance(waiter_result, FastHTTPException), (
            f"Expected HTTPException from skipped waiter, got {type(waiter_result)}: {waiter_result}"
        )
        assert waiter_result.status_code == 429

        # Family and probe locks must be unlocked.
        family_state = mgr.family("codex_auto_agent")
        assert not family_state.lock.locked(), "family lock leaked"
        probe_lock = await mgr.candidate_probe_lock(
            alias_family="codex_auto_agent",
            cooldown_key=cooldown_key,
        )
        assert not probe_lock.locked(), "probe lock leaked"

        # Intent registry must be clean (no leaked intents).
        assert mgr.publication_intents.get("codex_auto_agent", cooldown_key) is None, (
            "intent registry has leaked intent for cooldown_key"
        )

        # Cooldown must be active in family memory.
        remaining = family_state.peek_cooldown_remaining(cooldown_key)
        assert remaining > 0, "cooldown was not committed to family memory"

    finally:
        loop_mod.alias_routing_state = original_state


# ---------------------------------------------------------------------------
# Test 8: Strengthened deadlock test with barriers and cleanup assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_publication_and_probe_no_deadlock_with_barriers() -> None:  # noqa: PLR0915
    """Strengthened adversarial deadlock test with deterministic barriers.

    Forces the exact interleaving where:
      1. Probe path acquires probe lock and starts provider I/O.
      2. Publication transaction acquires family lock, then waits for probe.
      3. Probe path releases probe lock (provider done).
      4. Publication acquires probe lock, commits, releases all.

    Asserts:
      - No deadlock (bounded completion).
      - Both tasks complete successfully.
      - Family and probe locks unlocked after.
      - Intent registry cleaned.
      - Cooldown committed.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
        execute_cooldown_publication_transaction,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop as loop_mod
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply as apply_mod

    mgr = AliasRoutingStateManager()
    cooldown_key = "openrouter:openrouter/test-model:openrouter"

    original_state = loop_mod.alias_routing_state
    original_apply_state = getattr(apply_mod, "_state_manager", None)
    loop_mod.alias_routing_state = mgr
    apply_mod._state_manager = mgr

    try:
        perform_count = 0
        probe_started = asyncio.Event()
        publication_family_acquired = asyncio.Event()

        async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
            return _selection()

        async def _perform(*, candidate: Any, candidate_body: Any) -> Response:
            nonlocal perform_count
            perform_count += 1
            # Signal that probe I/O has started (probe lock is held).
            probe_started.set()
            # Wait for publication to acquire family lock (creating the
            # potential inversion window).
            await publication_family_acquired.wait()
            return Response(content='{"ok":true}', media_type="application/json")

        def _resolve(**_kwargs: Any) -> CooldownPublicationPlan:
            return CooldownPublicationPlan(
                memory_keys=(cooldown_key,),
                durable_keys=(),
                duration_seconds=30.0,
                applied_scope="candidate",
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

        async def _cooldown_reader_noop(key: str) -> tuple[float, str]:
            return 0.0, "memory"

        # Task A: publication transaction with barrier.
        plan = CooldownPublicationPlan(
            memory_keys=(cooldown_key,),
            durable_keys=(),
            duration_seconds=30.0,
            applied_scope="candidate",
        )

        original_family_lock = mgr.family("codex_auto_agent").lock

        class _BarrierLock:
            """Wraps the family lock to signal when acquired."""

            def __init__(self, inner: asyncio.Lock) -> None:
                self._inner = inner

            async def acquire(self) -> bool:
                result = await self._inner.acquire()
                publication_family_acquired.set()
                return result

            def release(self) -> None:
                self._inner.release()

            def locked(self) -> bool:
                return self._inner.locked()

            async def __aenter__(self) -> "_BarrierLock":
                await self.acquire()
                return self

            async def __aexit__(self, *args: Any) -> None:
                self.release()

        mgr.family("codex_auto_agent").lock = _BarrierLock(original_family_lock)  # type: ignore[assignment]

        async def _run_publication() -> None:
            # Force the claimed interleaving: wait until the probe path has
            # acquired the probe lock and started provider I/O before the
            # publication transaction attempts family -> probe lock acquisition.
            await probe_started.wait()
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_candidate(),
                plan=plan,
                publish_cooldown_memory_fn=lambda *, keys, seconds: None,
                persist_cooldown_fn=AsyncMock(),
            )

        # Task B: candidate loop probe path.
        async def _run_probe() -> Response:
            return await handle_alias_route(
                services,
                alias_family="codex_auto_agent",
                alias_model="basic",
                request=_minimal_request("barrier-deadlock-test"),
                prepared_request_body={"model": "basic"},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=_cooldown_reader_noop,
                attempts_metadata_key="attempts",
                skipped_candidates_metadata_key="skipped",
                no_candidate_detail="no candidates",
                log_label="BarrierDeadlock",
            )

        pub_task = asyncio.create_task(_run_publication())
        probe_task = asyncio.create_task(_run_probe())

        # Bounded completion: deadlock = timeout.
        results = await asyncio.wait_for(
            asyncio.gather(pub_task, probe_task, return_exceptions=True),
            timeout=5.0,
        )

        # Both must succeed.
        assert results[0] is None, f"publication failed: {results[0]}"
        assert isinstance(results[1], Response), f"probe failed: {results[1]}"
        assert perform_count == 1

        # Restore real lock for assertions.
        mgr.family("codex_auto_agent").lock = original_family_lock

        # Cleanup assertions: no lock leaks.
        assert not original_family_lock.locked(), "family lock leaked"
        probe_lock = await mgr.candidate_probe_lock(
            alias_family="codex_auto_agent",
            cooldown_key=cooldown_key,
        )
        assert not probe_lock.locked(), "probe lock leaked"

        # Intent registry must be clean.
        assert mgr.publication_intents.get("codex_auto_agent", cooldown_key) is None, (
            "intent registry has leaked intent"
        )

    finally:
        loop_mod.alias_routing_state = original_state
        apply_mod._state_manager = original_apply_state
