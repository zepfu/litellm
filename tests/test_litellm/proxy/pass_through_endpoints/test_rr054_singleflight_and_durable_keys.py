"""RR-054 strict regressions for #31 candidate single-flight and #56 durable keys."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        }
    )


@pytest.mark.asyncio
async def test_rr054_issue31_candidate_probe_is_single_flight() -> None:
    candidate = {
        "provider": "openai",
        "model": "gpt-test",
        "route_family": "codex_responses",
        "last_resort": False,
    }
    selection = {
        "candidate": candidate,
        "cooldown_key": "openai:gpt-test:lane",
        "lane_key": "lane",
        "session_key": None,
        "selection_reason": "first_available",
        "skipped": [],
    }
    active = 0.0
    provider_entered = asyncio.Event()
    release_provider = asyncio.Event()
    provider_calls = 0

    async def _perform(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        provider_entered.set()
        await release_provider.wait()
        raise HTTPException(status_code=429, detail="capacity")

    async def _active(_key: str):
        return active, "memory" if active else "local_fallback"

    async def _cooldown(**_kwargs):
        nonlocal active
        active = 60.0
        return "candidate"

    async def _run() -> None:
        await lpe._handle_auto_agent_alias_route(
            alias_family="codex_auto_agent",
            alias_model="aawm-test",
            request=_request(),
            prepared_request_body={"model": "aawm-test"},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(return_value=dict(selection)),
            add_alias_metadata_fn=lambda body, **_kwargs: body,
            perform_candidate_request_fn=_perform,
            get_active_cooldown_state_fn=_active,
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=_cooldown,
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="none",
            log_label="Codex",
        )

    with patch.object(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        side_effect=lambda **kwargs: kwargs["prepared_request_body"],
    ), patch.object(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        side_effect=lambda **kwargs: kwargs["prepared_request_body"],
    ):
        first = asyncio.create_task(_run())
        await provider_entered.wait()
        second = asyncio.create_task(_run())
        await asyncio.sleep(0)
        release_provider.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

    assert provider_calls == 1
    assert all(isinstance(result, HTTPException) for result in results)


def test_rr054_issue56_durable_key_is_opaque() -> None:
    raw_session = "tenant-session-secret:lane"
    cache_key = durable.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=raw_session,
    )
    assert raw_session not in cache_key
    assert cache_key.endswith(
        __import__("hashlib").sha256(raw_session.encode("utf-8")).hexdigest()
    )


@pytest.mark.asyncio
async def test_rr054_issue56_durable_affinity_cardinality_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durable._durable_affinity_key_until_epoch.clear()
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT", "1")
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock()
    dual.async_get_cache = AsyncMock(return_value=None)
    dual.async_set_cache = AsyncMock()
    with patch.object(durable, "get_aawm_alias_routing_dual_cache", return_value=dual):
        first = await durable.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="affinity",
            state_key="session-a:lane",
            payload={"provider": "openai"},
            ttl_seconds=60,
        )
        second = await durable.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="affinity",
            state_key="session-b:lane",
            payload={"provider": "openai"},
            ttl_seconds=60,
        )
    assert first is True
    assert second is False
    assert dual.redis_cache.async_set_cache.await_count == 1
    durable._durable_affinity_key_until_epoch.clear()


def test_rr054_issue1_finalize_ownership_is_extracted() -> None:
    wrapper = inspect.getsource(
        lpe._finalize_anthropic_responses_adapter_upstream_response
    )
    assert "_aawm_responses_finalize.finalize_" in wrapper
    assert "_collect_responses_response_from_stream" not in wrapper
    owner = inspect.getsource(
        lpe._aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response
    )
    assert "runtime.collect_stream" in owner
