"""RR-054 #56 durable affinity Redis keys + #58 oa_xai mutation identity.

Finding #56 (security/operational):
  Client-controlled session scopes flow into durable Redis affinity keys. The
  Redis key must hash the session scope (never embed the raw client string) and
  affinity key cardinality must stay bounded/valid under write pressure.

Finding #58 (maintainability/correctness):
  ``_prepare_oa_xai_passthrough_request`` applies several request-body transform
  stages. Mutation consolidation via ``_replace_request_body_in_place`` (or an
  equivalent helper) must preserve the caller's request_body object identity
  and the cumulative drop/sanitize behavior across each stage.

No production edits. Failures here document remaining RR-054 #56/#58 gaps.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dual_cache(*, existing: Any = None) -> MagicMock:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    dual.async_get_cache = AsyncMock(return_value=existing)
    dual.async_set_cache = AsyncMock(return_value=None)
    return dual


def _clear_durable_affinity_cardinality_state() -> None:
    durable_mod._durable_affinity_key_until_epoch.clear()


def _sample_candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "provider": "openai",
        "model": "gpt-5.3-codex-spark",
        "route_family": "openai_responses",
        "last_resort": False,
    }
    candidate.update(overrides)
    return candidate


# ===========================================================================
# RR-054 #56 — durable affinity Redis keys: hashed session scope + bounds
# ===========================================================================


def test_rr054_issue56_durable_affinity_cache_key_hashes_session_scope() -> None:
    """Raw client session scope must not appear in the Redis affinity key."""
    raw_session_scope = (
        "client-session-id-with-sensitive-token-abc123:"
        "auth:deadbeefcafebabe"
    )
    cache_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=raw_session_scope,
    )

    assert raw_session_scope not in cache_key
    assert "client-session-id" not in cache_key
    assert "deadbeefcafebabe" not in cache_key
    assert "sensitive-token" not in cache_key

    expected_digest = _sha256_hex(raw_session_scope)
    assert expected_digest in cache_key
    assert cache_key.endswith(f":affinity:{expected_digest}")
    assert cache_key.startswith(f"{durable_mod.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX}:")


def test_rr054_issue56_durable_affinity_cache_key_is_stable_and_family_scoped() -> None:
    session_scope = "sess-stable-rr054-56:auth:lane-a"
    codex_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope,
    )
    codex_key_again = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope,
    )
    anthropic_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_scope,
    )
    other_session_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope + "-other",
    )
    cooldown_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="cooldown",
        state_key=session_scope,
    )

    assert codex_key == codex_key_again
    assert codex_key != anthropic_key
    assert codex_key != other_session_key
    assert codex_key != cooldown_key
    assert session_scope not in codex_key
    assert session_scope not in anthropic_key
    assert _sha256_hex(session_scope) in codex_key
    assert _sha256_hex(session_scope) in anthropic_key
    assert ":codex:affinity:" in codex_key
    assert ":anthropic:affinity:" in anthropic_key


def test_rr054_issue56_durable_affinity_cache_key_normalizes_kind_and_family() -> None:
    session_scope = "sess-normalize-rr054-56"
    mixed = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="  CoDeX  ",
        state_kind="  AfFiNiTy  ",
        state_key=session_scope,
    )
    lower = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope,
    )
    assert mixed == lower
    assert session_scope not in mixed
    assert ":codex:affinity:" in mixed


def test_rr054_issue56_lpe_wrapper_matches_durable_hashed_affinity_key() -> None:
    """God-file re-export must hash the same way as the durable package helper."""
    session_scope = "aawm-code:session-header-value:auth:lane"
    via_durable = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope,
    )
    via_lpe = lpe._build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_scope,
    )
    assert via_lpe == via_durable
    assert session_scope not in via_lpe
    assert _sha256_hex(session_scope) in via_lpe


def test_rr054_issue56_affinity_cardinality_reserve_accepts_existing_key() -> None:
    _clear_durable_affinity_cardinality_state()
    try:
        cache_key = "aawm:alias-routing:test:codex:affinity:existing"
        assert durable_mod._reserve_durable_affinity_key(
            cache_key, expires_at_epoch=time.time() + 60.0
        )
        # Refreshing the same key must remain allowed even at capacity.
        with patch.object(durable_mod, "_get_durable_affinity_key_limit", return_value=1):
            assert durable_mod._reserve_durable_affinity_key(
                cache_key, expires_at_epoch=time.time() + 120.0
            )
            assert durable_mod._durable_affinity_key_until_epoch[cache_key] >= (
                time.time() + 60.0
            )
    finally:
        _clear_durable_affinity_cardinality_state()


def test_rr054_issue56_affinity_cardinality_reserve_rejects_new_at_cap() -> None:
    _clear_durable_affinity_cardinality_state()
    try:
        with patch.object(durable_mod, "_get_durable_affinity_key_limit", return_value=2):
            assert durable_mod._reserve_durable_affinity_key(
                "aff-key-1", expires_at_epoch=time.time() + 300.0
            )
            assert durable_mod._reserve_durable_affinity_key(
                "aff-key-2", expires_at_epoch=time.time() + 300.0
            )
            assert (
                durable_mod._reserve_durable_affinity_key(
                    "aff-key-3", expires_at_epoch=time.time() + 300.0
                )
                is False
            )
            assert "aff-key-3" not in durable_mod._durable_affinity_key_until_epoch
            assert len(durable_mod._durable_affinity_key_until_epoch) == 2
    finally:
        _clear_durable_affinity_cardinality_state()


def test_rr054_issue56_affinity_cardinality_evicts_expired_before_cap() -> None:
    _clear_durable_affinity_cardinality_state()
    try:
        past = time.time() - 10.0
        future = time.time() + 300.0
        with patch.object(durable_mod, "_get_durable_affinity_key_limit", return_value=1):
            durable_mod._durable_affinity_key_until_epoch["expired-aff"] = past
            assert durable_mod._reserve_durable_affinity_key(
                "fresh-aff", expires_at_epoch=future
            )
            assert "expired-aff" not in durable_mod._durable_affinity_key_until_epoch
            assert "fresh-aff" in durable_mod._durable_affinity_key_until_epoch
    finally:
        _clear_durable_affinity_cardinality_state()


@pytest.mark.asyncio
async def test_rr054_issue56_affinity_write_uses_hashed_key_not_raw_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end affinity durable write must SET the hashed Redis key only."""
    _clear_durable_affinity_cardinality_state()
    monkeypatch.delenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", raising=False)
    raw_session = "raw-client-session-rr054-56:auth:lane-xyz"
    dual = _dual_cache()
    expected_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=raw_session,
    )
    assert raw_session not in expected_key

    try:
        with patch.object(
            durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
        ):
            ok = await durable_mod.write_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="affinity",
                state_key=raw_session,
                payload={
                    "provider": "openai",
                    "model": "gpt-5.3-codex-spark",
                    "route_family": "openai_responses",
                    "last_resort": False,
                },
                ttl_seconds=60.0,
            )

        assert ok is True
        redis_kwargs = dual.redis_cache.async_set_cache.await_args.kwargs
        assert redis_kwargs["key"] == expected_key
        assert raw_session not in redis_kwargs["key"]
        assert _sha256_hex(raw_session) in redis_kwargs["key"]
        # Payload may still identify the candidate; the Redis *key* is opaque.
        written = redis_kwargs["value"]
        assert isinstance(written, dict)
        assert "expires_at_epoch" in written
        assert written["provider"] == "openai"
    finally:
        _clear_durable_affinity_cardinality_state()


@pytest.mark.asyncio
async def test_rr054_issue56_affinity_write_skips_at_cardinality_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Affinity durable writes must refuse new keys once the process cap is hit."""
    _clear_durable_affinity_cardinality_state()
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT", "1")
    dual = _dual_cache()

    try:
        with patch.object(
            durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
        ):
            first = await durable_mod.write_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="affinity",
                state_key="session-cap-first",
                payload={"provider": "openai", "model": "m1", "route_family": "r"},
                ttl_seconds=30.0,
            )
            second = await durable_mod.write_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="affinity",
                state_key="session-cap-second",
                payload={"provider": "openai", "model": "m2", "route_family": "r"},
                ttl_seconds=30.0,
            )

        assert first is True
        assert second is False
        assert dual.redis_cache.async_set_cache.await_count == 1
        written_key = dual.redis_cache.async_set_cache.await_args.kwargs["key"]
        assert "session-cap-first" not in written_key
        assert _sha256_hex("session-cap-first") in written_key
    finally:
        _clear_durable_affinity_cardinality_state()
        monkeypatch.delenv("AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT", raising=False)


@pytest.mark.asyncio
async def test_rr054_issue56_set_codex_affinity_durable_key_is_hashed_not_raw() -> None:
    """``_set_codex_auto_agent_session_affinity`` must durable-write hashed keys."""
    raw_session = "codex-aff-session-raw-rr054-56:auth:lane"
    expected_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=raw_session,
    )
    dual = _dual_cache()
    _clear_durable_affinity_cardinality_state()
    lpe._codex_auto_agent_session_affinity_by_key.clear()

    try:
        with patch.object(
            durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
        ), patch.object(
            lpe, "_get_aawm_alias_routing_dual_cache", return_value=dual
        ):
            await lpe._set_codex_auto_agent_session_affinity(
                raw_session, _sample_candidate()
            )

        assert dual.redis_cache.async_set_cache.await_count >= 1
        redis_key = dual.redis_cache.async_set_cache.await_args.kwargs["key"]
        assert redis_key == expected_key
        assert raw_session not in redis_key
        assert _sha256_hex(raw_session) in redis_key
        # Memory map may still use the process-local session_key identity.
        assert raw_session in lpe._codex_auto_agent_session_affinity_by_key
    finally:
        _clear_durable_affinity_cardinality_state()
        lpe._codex_auto_agent_session_affinity_by_key.clear()


@pytest.mark.asyncio
async def test_rr054_issue56_set_anthropic_affinity_durable_key_is_hashed_not_raw() -> None:
    raw_session = "anthropic-aff-session-raw-rr054-56"
    expected_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=raw_session,
    )
    dual = _dual_cache()
    _clear_durable_affinity_cardinality_state()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()

    try:
        with patch.object(
            durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
        ), patch.object(
            lpe, "_get_aawm_alias_routing_dual_cache", return_value=dual
        ):
            await lpe._set_anthropic_auto_agent_session_affinity(
                raw_session,
                _sample_candidate(
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    route_family="anthropic_messages",
                ),
            )

        redis_key = dual.redis_cache.async_set_cache.await_args.kwargs["key"]
        assert redis_key == expected_key
        assert raw_session not in redis_key
        assert _sha256_hex(raw_session) in redis_key
    finally:
        _clear_durable_affinity_cardinality_state()
        lpe._anthropic_auto_agent_session_affinity_by_key.clear()


def test_rr054_issue56_affinity_key_limit_env_is_positive_and_bounded() -> None:
    """Limit helper must yield a positive int (default or parsed env)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT", None)
        default_limit = durable_mod._get_durable_affinity_key_limit()
    assert isinstance(default_limit, int)
    assert default_limit >= 1

    with patch.dict(
        os.environ, {"AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT": "7"}, clear=False
    ):
        assert durable_mod._get_durable_affinity_key_limit() == 7

    with patch.dict(
        os.environ, {"AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT": "0"}, clear=False
    ):
        # Invalid/zero must clamp to a positive bound rather than unbounded.
        assert durable_mod._get_durable_affinity_key_limit() >= 1

    with patch.dict(
        os.environ,
        {"AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT": "not-an-int"},
        clear=False,
    ):
        assert durable_mod._get_durable_affinity_key_limit() >= 1


# ===========================================================================
# RR-054 #58 — prepare_oa_xai mutation consolidation + object identity
# ===========================================================================


def test_rr054_issue58_replace_request_body_in_place_preserves_identity() -> None:
    original = {"model": "oa_xai/grok-build", "keep": 1, "drop_me": True}
    original_id = id(original)
    updated = {"model": "xai/grok-build", "keep": 1, "added": "yes"}

    lpe._replace_request_body_in_place(original, updated)

    assert id(original) == original_id
    assert original is not updated
    assert original == {"model": "xai/grok-build", "keep": 1, "added": "yes"}
    assert "drop_me" not in original


def test_rr054_issue58_replace_request_body_in_place_noop_when_same_object() -> None:
    body = {"model": "oa_xai/grok-build", "n": 1}
    body_id = id(body)
    lpe._replace_request_body_in_place(body, body)
    assert id(body) == body_id
    assert body == {"model": "oa_xai/grok-build", "n": 1}


@pytest.mark.asyncio
async def test_rr054_issue58_prepare_preserves_request_body_object_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-owned request_body must remain the same object after prepare."""
    monkeypatch.setenv("LITELLM_XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
    request_body: dict[str, Any] = {
        "model": "oa_xai/grok-build",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
    }
    original_id = id(request_body)

    with patch(
        "litellm.llms.xai.oauth.get_xai_oauth_access_token",
        new=AsyncMock(return_value="xai-oauth-token"),
    ):
        prepared, target_base, api_key = await lpe._prepare_oa_xai_passthrough_request(
            request_body,
            sanitize_responses_request=False,
        )

    assert prepared is True
    assert id(request_body) == original_id
    assert request_body is not None
    assert target_base == "https://api.x.ai/v1"
    assert api_key == "xai-oauth-token"
    assert request_body["model"] == "xai/grok-build"
    # Credential fields must be popped off the shared body, not a copy.
    assert "api_base" not in request_body
    assert "api_key" not in request_body
    assert "custom_llm_provider" not in request_body


@pytest.mark.asyncio
async def test_rr054_issue58_prepare_sanitize_preserves_identity_across_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanitize path must keep object identity through every transform stage."""
    monkeypatch.setenv("LITELLM_XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
    request_body: dict[str, Any] = {
        "model": "oa_xai/grok-build",
        "input": [
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "reasoning",
                "id": "rs_rr054_58",
                "summary": [],
                "encrypted_content": "encrypted-rr054-58",
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_rr054_58",
                "arguments": {},
            },
            {
                "type": "function_call_output",
                "call_id": "call_rr054_58",
                "output": "ok",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "medium"},
        "reasoning_effort": "medium",
        "reasoningEffort": "medium",
        "output_config": {"effort": "medium", "verbosity": "low"},
    }
    original_id = id(request_body)
    stage_identities: list[int] = []

    real_replace = lpe._replace_request_body_in_place

    def _spy_replace(
        body: dict[str, Any], updated: dict[str, Any]
    ) -> None:
        real_replace(body, updated)
        stage_identities.append(id(body))
        assert id(body) == original_id, (
            "RR-054 #58: _replace_request_body_in_place must preserve "
            "request_body object identity at every stage"
        )

    real_sanitize_in_place = lpe._sanitize_xai_responses_request_body_in_place

    def _spy_sanitize_in_place(
        body: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        result = real_sanitize_in_place(body)
        stage_identities.append(id(body))
        assert id(body) == original_id, (
            "RR-054 #58: _sanitize_xai_responses_request_body_in_place must "
            "preserve request_body object identity"
        )
        return result

    with patch(
        "litellm.llms.xai.oauth.get_xai_oauth_access_token",
        new=AsyncMock(return_value="xai-oauth-token"),
    ), patch.object(
        lpe, "_replace_request_body_in_place", side_effect=_spy_replace
    ), patch.object(
        lpe,
        "_sanitize_xai_responses_request_body_in_place",
        side_effect=_spy_sanitize_in_place,
    ):
        prepared, target_base, api_key = await lpe._prepare_oa_xai_passthrough_request(
            request_body,
            sanitize_responses_request=True,
        )

    assert prepared is True
    assert id(request_body) == original_id
    assert target_base == "https://api.x.ai/v1"
    assert api_key == "xai-oauth-token"
    # At least the four drop stages + sanitize should have touched the body.
    assert len(stage_identities) >= 4
    assert all(stage_id == original_id for stage_id in stage_identities)

    assert request_body["model"] == "xai/grok-build"
    assert "reasoning" not in request_body
    assert "reasoning_effort" not in request_body
    assert "reasoningEffort" not in request_body
    assert "api_base" not in request_body
    assert "api_key" not in request_body
    assert "custom_llm_provider" not in request_body

    # Reasoning input item dropped; other items retained on the same object.
    input_items = request_body.get("input")
    assert isinstance(input_items, list)
    assert all(
        not (isinstance(item, dict) and item.get("type") == "reasoning")
        for item in input_items
    )
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in input_items
    )

    metadata = request_body.get("litellm_metadata")
    assert isinstance(metadata, dict)
    # Cumulative behavior across stages must remain visible on the same object.
    removed_params = metadata.get("codex_unsupported_request_params_removed")
    if removed_params is not None:
        assert "reasoning" in removed_params or "reasoning_effort" in [
            str(p).lower() for p in removed_params
        ]


@pytest.mark.asyncio
async def test_rr054_issue58_prepare_sanitize_each_stage_receives_same_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each transform stage must be invoked with the same request_body object."""
    monkeypatch.setenv("LITELLM_XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
    request_body: dict[str, Any] = {
        "model": "oa_xai/grok-build",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "reasoning", "summary": []},
        ],
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "function", "name": "missing_tool"},
        "reasoning": {"effort": "low"},
    }
    original_id = id(request_body)
    seen_body_ids: dict[str, int] = {}

    def _wrap_drop(name: str, real_fn: Any) -> Any:
        def _wrapped(body: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            seen_body_ids[name] = id(body)
            assert id(body) == original_id, (
                f"RR-054 #58: stage {name} received a different body object"
            )
            return real_fn(body, *args, **kwargs)

        return _wrapped

    with patch(
        "litellm.llms.xai.oauth.get_xai_oauth_access_token",
        new=AsyncMock(return_value="xai-oauth-token"),
    ), patch.object(
        lpe,
        "_drop_unsupported_codex_hosted_tools_from_request_body",
        side_effect=_wrap_drop(
            "hosted_tools",
            lpe._drop_unsupported_codex_hosted_tools_from_request_body,
        ),
    ), patch.object(
        lpe,
        "_drop_unsupported_codex_request_params_from_request_body",
        side_effect=_wrap_drop(
            "request_params",
            lpe._drop_unsupported_codex_request_params_from_request_body,
        ),
    ), patch.object(
        lpe,
        "_drop_unsupported_codex_input_items_from_request_body",
        side_effect=_wrap_drop(
            "input_items",
            lpe._drop_unsupported_codex_input_items_from_request_body,
        ),
    ), patch.object(
        lpe,
        "_sanitize_xai_responses_request_body_in_place",
        side_effect=_wrap_drop(
            "sanitize_xai",
            lpe._sanitize_xai_responses_request_body_in_place,
        ),
    ), patch.object(
        lpe,
        "_drop_tool_choice_without_tools_from_request_body",
        side_effect=_wrap_drop(
            "tool_choice",
            lpe._drop_tool_choice_without_tools_from_request_body,
        ),
    ):
        prepared, _, _ = await lpe._prepare_oa_xai_passthrough_request(
            request_body,
            sanitize_responses_request=True,
        )

    assert prepared is True
    assert id(request_body) == original_id
    expected_stages = {
        "hosted_tools",
        "request_params",
        "input_items",
        "sanitize_xai",
        "tool_choice",
    }
    assert expected_stages.issubset(set(seen_body_ids))
    assert all(body_id == original_id for body_id in seen_body_ids.values())


@pytest.mark.asyncio
async def test_rr054_issue58_prepare_non_oa_xai_returns_false_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_body: dict[str, Any] = {
        "model": "gpt-4o",
        "input": "hello",
        "keep": True,
    }
    original_id = id(request_body)
    snapshot = dict(request_body)

    prepared, target_base, api_key = await lpe._prepare_oa_xai_passthrough_request(
        request_body,
        sanitize_responses_request=True,
    )

    assert prepared is False
    assert target_base is None
    assert api_key is None
    assert id(request_body) == original_id
    assert request_body == snapshot


@pytest.mark.asyncio
async def test_rr054_issue58_prepare_ensures_litellm_metadata_on_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing litellm_metadata must be created on the caller's body object."""
    monkeypatch.setenv("LITELLM_XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
    request_body: dict[str, Any] = {
        "model": "oa_xai/grok-build",
        "input": "hello",
    }
    original_id = id(request_body)
    assert "litellm_metadata" not in request_body

    with patch(
        "litellm.llms.xai.oauth.get_xai_oauth_access_token",
        new=AsyncMock(return_value="xai-oauth-token"),
    ):
        prepared, _, _ = await lpe._prepare_oa_xai_passthrough_request(
            request_body,
            sanitize_responses_request=False,
        )

    assert prepared is True
    assert id(request_body) == original_id
    assert isinstance(request_body.get("litellm_metadata"), dict)


def test_rr054_issue58_prepare_source_uses_replace_helper_not_inline_clear_update() -> None:
    """Consolidation contract: prepare path should call the shared in-place helper.

    Finding #58 asked to stop repeating clear/update inline four times. This
    source-level regression fails if the prepare function reintroduces the
    duplicated swap idiom without the shared helper.
    """
    import inspect

    source = inspect.getsource(lpe._prepare_oa_xai_passthrough_request)
    assert "_replace_request_body_in_place" in source, (
        "RR-054 #58 gap: _prepare_oa_xai_passthrough_request must consolidate "
        "mutation via _replace_request_body_in_place (or equivalent shared helper)"
    )
    # Inline clear/update swaps should not reappear in the prepare function body.
    # Allow the helper definition itself elsewhere; only the prepare source matters.
    assert "request_body.clear()" not in source, (
        "RR-054 #58 gap: prepare still uses inline request_body.clear() instead "
        "of the shared in-place replace helper"
    )
    assert "request_body.update(updated_body)" not in source, (
        "RR-054 #58 gap: prepare still uses inline request_body.update(updated_body)"
    )
