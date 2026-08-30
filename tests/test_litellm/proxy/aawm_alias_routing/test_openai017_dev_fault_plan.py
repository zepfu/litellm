"""Focused OPENAI-017/018 handler integration tests."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from starlette.responses import Response

from litellm.proxy import aawm_route_logging
from litellm.proxy.common_utils.http_parsing_utils import (
    _safe_get_request_parsed_body,
)
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    attempt_records,
    candidate_loop,
    codex_oauth,
    dev_fault_plan,
    selection,
    session_affinity,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

_MODEL = "gpt-openai017-test"
_FAULT_PLAN_STATE_ATTRIBUTES = (
    "aawm_openai_fault_plan",
    "aawm_openai_fault_plan_slot_index",
    "aawm_openai_fault_plan_slots",
    "aawm_openai_fault_plan_injected_count",
    "aawm_openai_fault_plan_direct_tracking",
    "aawm_openai_fault_plan_direct_attempt",
    "aawm_openai_fault_plan_direct_success_recorded",
    "aawm_openai_fault_plan_direct_terminal_recorded",
)


def _request(
    plan: str | None = None,
    *,
    path: str = "/openai_passthrough/v1/responses",
) -> Request:
    headers = []
    if plan is not None:
        headers.append(
            (
                b"x-aawm-openai-fault-plan",
                plan.encode("latin-1"),
            )
        )
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 4100),
            "server": ("testserver", 80),
            "http_version": "1.1",
        }
    )


def _auth(label: str) -> codex_oauth.CodexOAuthRequestAuth:
    account_hash = f"hash-{label}"
    display_initial = label.rsplit("-", 1)[-1][0]
    return codex_oauth.CodexOAuthRequestAuth(
        account_label=label,
        account_hash=account_hash,
        lane_key=f"codex-oauth:{label}:{account_hash}",
        headers={"Authorization": f"Bearer {label}"},
        account_display=f"{display_initial}*@litellm.invalid",
    )


def _managed_selection(
    label: str,
    *,
    ordinal: int,
    request_mode: str = "fresh",
) -> dict[str, Any]:
    account_hash = f"hash-{label}"
    lane = f"codex-oauth:{label}:{account_hash}"
    display_initial = label.rsplit("-", 1)[-1][0]
    return {
        "candidate": {
            "provider": "openai",
            "model": _MODEL,
            "route_family": "codex_responses",
            "last_resort": False,
            "codex_oauth_account_label": label,
            "codex_oauth_account_hash": account_hash,
            "codex_oauth_lane_key": lane,
            "codex_oauth_account_display": f"{display_initial}*@litellm.invalid",
            "codex_oauth_credential_affinity": "interchangeable",
        },
        "lane_key": lane,
        "cooldown_key": f"openai:{_MODEL}:{lane}",
        "selection_reason": (
            "codex_oauth_account_failover"
            if ordinal
            else "direct_inventory_first_available"
        ),
        "failover_ordinal": ordinal,
        "request_mode": request_mode,
        "alias_model": _MODEL,
        "session_key": None,
        "skipped": [],
    }


def _enable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAWM_OPENAI_FAULT_PLAN_ENABLED", "1")
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-dev")


def _assert_no_fault_plan_state(request: Request) -> None:
    for attribute in _FAULT_PLAN_STATE_ATTRIBUTES:
        assert not hasattr(request.state, attribute)


def _install_direct_route_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    select: Callable[..., Any],
    handler: Callable[..., Any],
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = body or {"model": _MODEL, "input": "test"}

    async def _get_request_body(_request: Request) -> dict[str, Any]:
        return dict(body)

    bind = AsyncMock(side_effect=select)
    publish = MagicMock()
    persist = AsyncMock()
    status_events: list[dict[str, Any]] = []
    rollups: list[dict[str, Any]] = []
    audit_only: list[list[dict[str, Any]]] = []
    access_replacement = MagicMock()

    monkeypatch.setattr(lpe, "get_request_body", _get_request_body)
    monkeypatch.setattr(
        lpe,
        "_resolve_codex_auto_agent_alias_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_is_oa_xai_request_body",
        lambda body: False,
    )
    monkeypatch.setattr(
        lpe,
        "_is_grok_native_oauth_request_body",
        lambda body: False,
    )
    monkeypatch.setattr(
        lpe,
        "_should_use_direct_codex_oauth_inventory",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        codex_oauth,
        "select_and_bind_direct_codex_oauth_inventory",
        bind,
    )
    monkeypatch.setattr(
        lpe.BaseOpenAIPassThroughHandler,
        "_base_openai_pass_through_handler",
        staticmethod(handler),
    )
    monkeypatch.setattr(
        lpe,
        "_publish_codex_cooldown_memory",
        publish,
    )
    monkeypatch.setattr(
        lpe,
        "_persist_codex_cooldown_durable",
        persist,
    )
    monkeypatch.setattr(
        lpe,
        "emit_aawm_route_status_event",
        lambda **kwargs: status_events.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "record_aawm_route_rollup",
        lambda **kwargs: rollups.append(kwargs),
    )
    monkeypatch.setattr(
        aawm_route_logging,
        "register_aawm_route_rollup_access_log_replacement",
        access_replacement,
    )
    monkeypatch.setattr(
        dev_fault_plan._audit_persist,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, **kwargs: audit_only.append(list(events)) or "queued",
    )
    return {
        "bind": bind,
        "publish": publish,
        "persist": persist,
        "status_events": status_events,
        "rollups": rollups,
        "audit_only": audit_only,
        "access_replacement": access_replacement,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fail,success", ("fail", "success")),
        ("fail,fail", ("fail", "fail")),
        ("fail", ()),
        ("success", ()),
        ("success,fail", ()),
        ("fail,success,success", ()),
        ("fail,,success", ()),
        ("fail, success", ()),
        (" fail,success ", ()),
        ("FAIL,SUCCESS", ()),
    ],
)
def test_exact_request_plan_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: tuple[str, ...],
) -> None:
    _enable_gate(monkeypatch)
    request = _request(raw)

    assert dev_fault_plan._resolve_openai_fault_plan(request) == expected
    if expected:
        assert request.state.aawm_openai_fault_plan == expected
        assert not hasattr(
            request.state,
            "aawm_openai_fault_plan_slot_index",
        )
    else:
        _assert_no_fault_plan_state(request)


def test_injected_error_uses_existing_managed_openai_classifiers_without_plan_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request("fail,success")
    candidate = _managed_selection("account-a", ordinal=0)["candidate"]

    with pytest.raises(dev_fault_plan.AawmOpenAIFaultPlanError) as exc_info:
        dev_fault_plan._raise_if_openai_fault_plan_slot_fails(
            request,
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert codex_oauth.is_direct_codex_usage_limit_error(exc) is True
    assert (
        lpe._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
        == "usage_limit_reached"
    )
    emitted_shape = repr(exc.detail)
    assert "fail,success" not in emitted_shape
    assert "fail,fail" not in emitted_shape


@pytest.mark.asyncio
async def test_direct_fail_success_retries_through_real_attempt_and_rollup_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request("fail,success")
    bindings = [
        (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        ),
        (
            _auth("account-b"),
            _managed_selection("account-b", ordinal=1),
            {},
        ),
    ]
    provider_calls: list[str] = []

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return bindings.pop(0)

    async def _handler(**kwargs: Any) -> Response:
        provider_calls.append(
            (kwargs.get("extra_headers") or {}).get("Authorization")
        )
        return Response(content=b"ok", status_code=200)

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=_handler,
    )

    response = await lpe.openai_proxy_route(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),  # type: ignore[arg-type]
    )

    assert response.status_code == 200
    assert provider_calls == ["Bearer account-b"]
    assert captured["bind"].await_count == 2
    captured["publish"].assert_called_once_with(
        keys=(f"openai:{_MODEL}:codex-oauth:account-a:hash-account-a",),
        seconds=1.0,
        allow_ttl_shrink=True,
    )
    assert captured["persist"].await_count == 1
    captured["access_replacement"].assert_called_once_with(request)

    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert outcome["outcome"] == "recovered"
    assert outcome["request_identity"]
    assert [attempt["account_label"] for attempt in outcome["attempts"]] == [
        "account-a",
        "account-b",
    ]
    assert outcome["attempts"][0]["request_outcome"] == "pending_failover"
    assert outcome["attempts"][1]["request_outcome"] == "recovered"
    assert [
        rollup["origin_identity"].account_display
        for rollup in captured["rollups"]
    ] == [
        "a*@litellm.invalid",
        "b*@litellm.invalid",
    ]
    assert captured["status_events"] == []
    assert [rollup["status"] for rollup in captured["rollups"]] == [
        "Failed",
        "Recovered",
    ]


@pytest.mark.asyncio
async def test_direct_without_fault_plan_records_attempt_and_success_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request()
    provider = AsyncMock(return_value=Response(content=b"ok", status_code=200))

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        )

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    response = await lpe.openai_proxy_route(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),  # type: ignore[arg-type]
    )

    assert response.status_code == 200
    provider.assert_awaited_once()
    assert captured["bind"].await_count == 1
    captured["access_replacement"].assert_called_once_with(request)
    assert not hasattr(request.state, "aawm_openai_fault_plan")
    assert not hasattr(request.state, "aawm_openai_fault_plan_slot_index")
    assert not hasattr(
        request.state,
        "aawm_openai_fault_plan_injected_count",
    )

    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert len(outcome["attempts"]) == 1
    attempt = outcome["attempts"][0]
    assert attempt["account_label"] == "account-a"
    assert attempt["attempted_provider_call"] is True
    assert attempt["request_outcome"] == "succeeded"
    assert attempt["account_display"] == "a*@litellm.invalid"

    parsed_body = _safe_get_request_parsed_body(request)
    assert parsed_body is not None
    metadata = parsed_body["litellm_metadata"]
    assert metadata["codex_auto_agent_attempts"] == [attempt]
    assert len(metadata["aawm_alias_routing_audit_events"]) == 1
    event = metadata["aawm_alias_routing_audit_events"][0]
    assert event["event_type"] == "candidate_selected"
    assert event["candidate_status"] == "succeeded"
    assert event["attempted_provider_call"] is True
    assert event["account_display"] == "a*@litellm.invalid"
    assert captured["rollups"] == []


@pytest.mark.asyncio
async def test_direct_normal_usage_limit_failure_recovers_with_attempt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request()
    bindings = [
        (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        ),
        (
            _auth("account-b"),
            _managed_selection("account-b", ordinal=1),
            {},
        ),
    ]
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "quota": {"resets_in_seconds": 1.0},
            "failover_disposition": "usage_limit_reached",
        },
        headers={"Retry-After": "1"},
    )
    provider = AsyncMock(
        side_effect=[
            usage_limit,
            Response(content=b"ok", status_code=200),
        ]
    )

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return bindings.pop(0)

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    response = await lpe.openai_proxy_route(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),  # type: ignore[arg-type]
    )

    assert response.status_code == 200
    assert provider.await_count == 2
    assert captured["bind"].await_count == 2
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert outcome["outcome"] == "recovered"
    assert len(outcome["attempts"]) == 2
    failed_attempt, recovered_attempt = outcome["attempts"]
    assert failed_attempt["error_class"] == "usage_limit_reached"
    assert failed_attempt["error_status_code"] == 429
    assert failed_attempt["error_code"] == "usage_limit_reached"
    assert failed_attempt["attempted_provider_call"] is True
    assert failed_attempt["request_outcome"] == "pending_failover"
    assert failed_attempt["replay_safety"] == "replay_safe"
    assert failed_attempt["credential_reload_outcome"] == "not_attempted"
    assert failed_attempt["guard_reset_outcome"] == "reset"
    assert failed_attempt["failover_decision"] == "move_account"
    assert failed_attempt["terminal_reason"] == "success"
    assert failed_attempt["attempted_account_hashes"] == [
        "hash-account-a",
        "hash-account-b",
    ]
    assert recovered_attempt["failover_decision"] == "completed_after_failover"
    assert recovered_attempt["terminal_reason"] == "success"
    assert recovered_attempt["attempted_account_hashes"] == [
        "hash-account-a",
        "hash-account-b",
    ]
    assert recovered_attempt["attempted_provider_call"] is True
    assert recovered_attempt["request_outcome"] == "recovered"
    assert recovered_attempt["guard_reset_outcome"] == "not_attempted"
    assert [rollup["status"] for rollup in captured["rollups"]] == [
        "Failed",
        "Recovered",
    ]
    assert [
        rollup["origin_identity"].account_display
        for rollup in captured["rollups"]
    ] == [
        "a*@litellm.invalid",
        "b*@litellm.invalid",
    ]
    assert captured["rollups"][1]["origin_identity"].account_display == (
        "b*@litellm.invalid"
    )
    assert not hasattr(request.state, "aawm_openai_fault_plan")
    assert not hasattr(
        request.state,
        "aawm_openai_fault_plan_injected_count",
    )


@pytest.mark.asyncio
async def test_direct_safe_continuation_rebinds_compatible_owner_before_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    body = {
        "model": _MODEL,
        "input": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "reasoned"}],
            },
        ],
    }
    bindings = [
        (
            _auth("account-a"),
            {
                **_managed_selection(
                    "account-a",
                    ordinal=0,
                    request_mode="ordinary_continuation",
                ),
                "has_account_bound_state": False,
            },
            {},
        ),
        (
            _auth("account-b"),
            {
                **_managed_selection(
                    "account-b",
                    ordinal=1,
                    request_mode="ordinary_continuation",
                ),
                "has_account_bound_state": False,
            },
            {},
        ),
    ]
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "quota": {"resets_in_seconds": 1.0},
            "failover_disposition": "usage_limit_reached",
        },
    )
    provider = AsyncMock(
        side_effect=[usage_limit, Response(content=b"ok", status_code=200)]
    )
    helper_calls: list[dict[str, Any]] = []
    current_attributes = {
        "provider": "openai",
        "model": _MODEL,
        "route_family": "codex_oauth",
        "account_label": "account-a",
        "account_hash": "hash-account-a",
        "account_lane": "codex-oauth:account-a:hash-account-a",
        "endpoint_contract": "openai_responses",
        "state_format": "openai_responses",
        "credential_affinity": "interchangeable",
        "ingress": "openai_passthrough",
        "requested_model": _MODEL,
        "hosted_provider": "openai",
    }
    session_affinity.set_request_session_owner_lease(
        request,
        session_affinity.SessionOwnerLease(
            session_identity="sess-openai017",
            owner_id="owner-openai017",
            decision=session_affinity.SessionOwnerGuardDecision.COMPATIBLE_OWNER.value,
            attributes=current_attributes,
        ),
    )

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return bindings.pop(0)

    async def _rebind(**kwargs: Any) -> bool:
        helper_calls.append(kwargs)
        return await real_rebind(**kwargs)

    async def _owned(**kwargs: Any) -> tuple[dict[str, Any], str, None]:
        return (
            {
                "state": "owned",
                "owner": "owner-openai017",
                "attributes": current_attributes,
            },
            "cache-key-openai017",
            None,
        )

    real_rebind = (
        session_affinity.clear_compatible_non_held_request_session_owner_guard_for_failover
    )
    monkeypatch.setattr(session_affinity, "get_session_owner_record", _owned)
    monkeypatch.setattr(
        session_affinity,
        "clear_compatible_non_held_request_session_owner_guard_for_failover",
        _rebind,
    )
    _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
        body=body,
    )

    response = await lpe.openai_proxy_route(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),  # type: ignore[arg-type]
    )

    assert response.status_code == 200
    assert provider.await_count == 2
    assert len(helper_calls) == 1
    helper_call = helper_calls[0]
    assert helper_call["account_failover_planned"] is True
    assert helper_call["account_failover_replay_safe"] is True
    assert helper_call["has_account_bound_state"] is False
    assert helper_call["post_commit_retry"] is False
    assert helper_call["failover_ordinal"] == 1
    assert helper_call["current_attributes"] == current_attributes
    assert helper_call["alternate_attributes"]["account_label"] == "account-b"
    assert helper_call["alternate_attributes"]["account_hash"] == (
        "hash-account-b"
    )
    assert session_affinity.get_request_session_owner_lease(request) is None
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert outcome["attempts"][0]["guard_reset_outcome"] == (
        "rebind_succeeded"
    )
    assert outcome["attempts"][1]["guard_reset_outcome"] == "not_attempted"


@pytest.mark.asyncio
async def test_direct_safe_continuation_guard_rebind_rejection_blocks_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    body = {"model": _MODEL, "input": [{"role": "tool", "content": "history"}]}
    bindings = [
        (
            _auth("account-a"),
            {
                **_managed_selection(
                    "account-a",
                    ordinal=0,
                    request_mode="ordinary_continuation",
                ),
                "has_account_bound_state": False,
            },
            {},
        ),
        (
            _auth("account-b"),
            {
                **_managed_selection(
                    "account-b",
                    ordinal=1,
                    request_mode="ordinary_continuation",
                ),
                "has_account_bound_state": False,
            },
            {},
        ),
    ]
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "failover_disposition": "usage_limit_reached",
        },
    )
    provider = AsyncMock(side_effect=[usage_limit])
    helper = AsyncMock(return_value=False)
    pre_second_selection: dict[str, Any] = {}
    terminal_entry: dict[str, Any] = {}

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if len(bindings) == 1:
            pre_second_selection["bound_account"] = copy.deepcopy(
                request.state.aawm_codex_oauth_selected_account
            )
            pre_second_selection["parsed_body"] = copy.deepcopy(
                request.scope["parsed_body"]
            )
        auth, selected, metadata = bindings.pop(0)
        setattr(
            request.state,
            "aawm_codex_oauth_selected_account",
            {
                "account_label": auth.account_label,
                "account_hash": auth.account_hash,
                "lane_key": auth.lane_key,
                "model": _MODEL,
                "credential_affinity": "interchangeable",
            },
        )
        selected_body = {
            **body,
            "litellm_metadata": {
                "codex_oauth_account_label": auth.account_label,
                "codex_oauth_account_hash": auth.account_hash,
                "codex_oauth_lane_key": auth.lane_key,
                "codex_auto_agent_selected_account_label": auth.account_label,
                "codex_auto_agent_selected_account_hash": auth.account_hash,
                "codex_auto_agent_selected_account_lane": auth.lane_key,
            },
        }
        lpe._safe_set_request_parsed_body(request, selected_body)
        return auth, selected, metadata

    real_terminal = (
        dev_fault_plan.note_direct_openai_managed_terminal_exhaustion
    )

    def _capture_terminal_entry(*args: Any, **kwargs: Any) -> None:
        terminal_entry["bound_account"] = copy.deepcopy(
            request.state.aawm_codex_oauth_selected_account
        )
        terminal_entry["parsed_body"] = copy.deepcopy(
            request.scope["parsed_body"]
        )
        real_terminal(*args, **kwargs)

    monkeypatch.setattr(
        session_affinity,
        "clear_compatible_non_held_request_session_owner_guard_for_failover",
        helper,
    )
    _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
        body=body,
    )
    monkeypatch.setattr(
        dev_fault_plan,
        "note_direct_openai_managed_terminal_exhaustion",
        _capture_terminal_entry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value is usage_limit
    assert provider.await_count == 1
    helper.assert_awaited_once()
    assert terminal_entry == pre_second_selection
    assert request.state.aawm_codex_oauth_selected_account == (
        pre_second_selection["bound_account"]
    )
    parsed_body = _safe_get_request_parsed_body(request)
    assert parsed_body is not None
    metadata = parsed_body["litellm_metadata"]
    assert metadata["codex_oauth_account_label"] == "account-a"
    assert metadata["codex_oauth_account_hash"] == "hash-account-a"
    assert metadata["codex_auto_agent_selected_account_label"] == "account-a"
    assert "account-b" not in repr(parsed_body)
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    attempt = outcome["attempts"][0]
    assert attempt["guard_reset_outcome"] == "rebind_rejected"
    assert attempt["failover_decision"] == "terminal"
    assert attempt["terminal_reason"] == "account_failover_guard_rebind_failed"


@pytest.mark.asyncio
async def test_direct_no_plan_exhaustion_keeps_generic_terminal_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    selected = _managed_selection("account-a", ordinal=0)
    request.state.aawm_codex_oauth_request_local_failover_context = {
        "slot": f"openai:{_MODEL}:codex_responses:",
        "candidate_identity": {
            "provider": "openai",
            "model": _MODEL,
            "route_family": "codex_responses",
        },
        "attempted_account_hashes": ["hash-account-a"],
        "prior_account_hash": "hash-account-a",
        "prior_account_outcomes": [],
        "prior_account_outcome": {},
        "portable_replay": True,
    }
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "failover_disposition": "usage_limit_reached",
        },
    )
    provider = AsyncMock(side_effect=[usage_limit])

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return _auth("account-a"), selected, {}

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value is usage_limit
    assert provider.await_count == 1
    assert captured["bind"].await_count == 1
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    attempt = outcome["attempts"][0]
    assert attempt["account_failover_limit_reached"] is True
    assert attempt["failover_decision"] == "terminal"
    assert attempt["terminal_reason"] == "account_failover_exhausted"


@pytest.mark.asyncio
async def test_direct_safe_continuation_second_selection_failure_preserves_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    body = {"model": _MODEL, "input": [{"role": "tool", "content": "history"}]}
    first_binding = (
        _auth("account-a"),
        {
            **_managed_selection(
                "account-a",
                ordinal=0,
                request_mode="ordinary_continuation",
            ),
            "has_account_bound_state": False,
        },
        {},
    )
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "failover_disposition": "usage_limit_reached",
        },
    )
    unavailable = HTTPException(
        status_code=429,
        detail={
            "error": {
                "code": "aawm_codex_oauth_direct_inventory_unavailable"
            }
        },
    )
    provider = AsyncMock(side_effect=[usage_limit])
    helper = AsyncMock(return_value=True)
    selections = [first_binding]

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if selections:
            return selections.pop(0)
        raise unavailable

    monkeypatch.setattr(
        session_affinity,
        "clear_compatible_non_held_request_session_owner_guard_for_failover",
        helper,
    )
    _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
        body=body,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value is usage_limit
    assert provider.await_count == 1
    helper.assert_not_awaited()
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    attempt = outcome["attempts"][0]
    assert attempt["failover_decision"] == "terminal"
    assert attempt["terminal_reason"] == "account_failover_second_selection_failed"


@pytest.mark.asyncio
async def test_direct_normal_terminal_failures_record_exhausted_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request()
    bindings = [
        (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        ),
        (
            _auth("account-b"),
            _managed_selection("account-b", ordinal=1),
            {},
        ),
    ]
    usage_limit = HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": "Managed OpenAI account usage limit reached.",
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            },
            "quota": {"resets_in_seconds": 1.0},
            "failover_disposition": "usage_limit_reached",
        },
        headers={"Retry-After": "1"},
    )
    token_invalidated = HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": "Managed OpenAI token invalidated.",
                "type": "invalid_request_error",
                "code": "token_invalidated",
            }
        },
    )
    provider = AsyncMock(side_effect=[usage_limit, token_invalidated])
    reload_credential = AsyncMock(return_value=None)
    monkeypatch.setattr(
        codex_oauth,
        "reload_codex_oauth_credential_after_token_invalidated",
        reload_credential,
    )

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if bindings:
            return bindings.pop(0)
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "aawm_codex_oauth_direct_inventory_unavailable"
                }
            },
        )

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value is token_invalidated
    assert provider.await_count == 2
    assert captured["bind"].await_count == 3
    reload_credential.assert_awaited_once()
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert outcome["outcome"] == "failed"
    assert len(outcome["attempts"]) == 2
    usage_attempt, token_attempt = outcome["attempts"]
    assert usage_attempt["error_class"] == "usage_limit_reached"
    assert usage_attempt["request_outcome"] == "pending_failover"
    assert token_attempt["error_class"] == "token_invalidated"
    assert token_attempt["error_status_code"] == 401
    assert token_attempt["error_code"] == "token_invalidated"
    assert token_attempt["attempted_provider_call"] is True
    assert token_attempt["request_outcome"] == "failed"
    assert token_attempt["replay_safety"] == "replay_safe"
    assert token_attempt["credential_reload_outcome"] == (
        "unchanged_already_reloaded"
    )
    assert token_attempt["guard_reset_outcome"] == "reset"
    assert token_attempt["failover_decision"] == "terminal"
    assert token_attempt["terminal_reason"] == (
        "account_failover_second_selection_failed"
    )
    assert token_attempt["attempted_account_hashes"] == [
        "hash-account-a",
        "hash-account-b",
    ]
    assert [
        event["status"]
        for event in captured["status_events"]
        if event["status"] == "Exhausted"
    ] == ["Exhausted"]
    assert [
        rollup["status"]
        for rollup in captured["rollups"]
        if rollup["status"] == "Exhausted"
    ] == ["Exhausted"]
    assert len(captured["audit_only"]) == 1
    terminal_event = captured["audit_only"][0][0]
    assert terminal_event["credential_reload_outcome"] == (
        "unchanged_already_reloaded"
    )
    assert terminal_event["failover_decision"] == "terminal"
    assert terminal_event["terminal_reason"] == (
        "account_failover_second_selection_failed"
    )
    assert terminal_event["attempted_account_hashes"] == [
        "hash-account-a",
        "hash-account-b",
    ]
    assert terminal_event["failure_class"] == "token_invalidated"
    assert terminal_event["attempt_count"] == 2
    assert terminal_event["request_outcome"] == "failed"
    assert not hasattr(request.state, "aawm_openai_fault_plan")
    assert not hasattr(
        request.state,
        "aawm_openai_fault_plan_injected_count",
    )


@pytest.mark.asyncio
async def test_direct_fail_fail_emits_one_terminal_exhausted_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request("fail,fail")
    bindings = [
        (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        ),
        (
            _auth("account-b"),
            _managed_selection("account-b", ordinal=1),
            {},
        ),
    ]
    provider = AsyncMock(return_value=Response(content=b"unexpected"))

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if bindings:
            return bindings.pop(0)
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "aawm_codex_oauth_direct_inventory_unavailable"
                }
            },
        )

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    with pytest.raises(dev_fault_plan.AawmOpenAIFaultPlanError) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert "fail,fail" not in repr(exc_info.value.detail)
    provider.assert_not_awaited()
    assert captured["bind"].await_count == 3
    assert captured["publish"].call_count == 2
    assert captured["persist"].await_count == 2

    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert outcome["outcome"] == "failed"
    assert outcome["request_identity"]
    assert len(outcome["attempts"]) == 2
    assert outcome["attempts"][0]["request_outcome"] == "pending_failover"
    assert outcome["attempts"][1]["request_outcome"] == "failed"
    assert [
        event["status"]
        for event in captured["status_events"]
        if event["status"] == "Exhausted"
    ] == ["Exhausted"]
    assert [
        rollup["status"]
        for rollup in captured["rollups"]
        if rollup["status"] == "Exhausted"
    ] == ["Exhausted"]
    assert len(captured["audit_only"]) == 1


@pytest.mark.asyncio
async def test_disabled_control_preserves_direct_handler_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AAWM_OPENAI_FAULT_PLAN_ENABLED", raising=False)
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-dev")
    request = _request("fail,success")
    provider_calls: list[str] = []

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        )

    async def _handler(**kwargs: Any) -> Response:
        provider_calls.append(
            (kwargs.get("extra_headers") or {}).get("Authorization")
        )
        return Response(content=b"ok", status_code=200)

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=_handler,
    )

    response = await lpe.openai_proxy_route(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),  # type: ignore[arg-type]
    )

    assert response.status_code == 200
    assert provider_calls == ["Bearer account-a"]
    assert captured["bind"].await_count == 1
    captured["publish"].assert_not_called()
    captured["persist"].assert_not_awaited()
    captured["access_replacement"].assert_called_once_with(request)
    assert captured["status_events"] == []
    assert captured["rollups"] == []
    assert not hasattr(request.state, "aawm_openai_fault_plan")
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert len(outcome["attempts"]) == 1
    assert outcome["attempts"][0]["request_outcome"] == "succeeded"


@pytest.mark.asyncio
async def test_malformed_header_preserves_direct_409_and_request_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    request = _request("fail")
    conflict = HTTPException(
        status_code=409,
        detail={
            "error": {"code": "aawm_session_owner_redispatch_required"},
            "redispatch_required": True,
        },
    )

    async def _select(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (
            _auth("account-a"),
            _managed_selection("account-a", ordinal=0),
            {},
        )

    provider = AsyncMock(side_effect=conflict)

    captured = _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=provider,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value is conflict
    provider.assert_awaited_once()
    assert captured["bind"].await_count == 1
    captured["publish"].assert_not_called()
    captured["persist"].assert_not_awaited()
    captured["access_replacement"].assert_called_once_with(request)
    assert captured["status_events"] == []
    assert captured["rollups"] == []
    assert captured["audit_only"] == []
    assert not hasattr(request.state, "aawm_openai_fault_plan")
    assert not hasattr(request.state, "aawm_openai_fault_plan_slot_index")
    assert not hasattr(
        request.state,
        "aawm_openai_fault_plan_injected_count",
    )
    outcome = attempt_records._auto_agent_alias_request_outcome_state(request)
    assert len(outcome["attempts"]) == 1


def _patch_candidate_loop_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        candidate_loop,
        "alias_routing_state",
        AliasRoutingStateManager(),
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: dict(kwargs["prepared_request_body"]),
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **kwargs: {
            "litellm_metadata": {
                "aawm_alias_routing_audit_events": [],
                "codex_auto_agent_attempts": list(kwargs["attempts"]),
                "codex_auto_agent_skipped_candidates": [],
            }
        },
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_success",
        lambda **kwargs: dict(kwargs["prepared_request_body"]),
    )
    monkeypatch.setattr(
        lpe,
        "_record_codex_failure_evidence",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **kwargs: None,
    )

    async def _no_publication(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _no_publication,
    )


def _alias_services(
    *,
    select_candidate: Callable[..., Any],
    perform_candidate: Callable[..., Any],
    classifications: list[str],
    raise_redispatch: Callable[..., None] | None = None,
) -> AliasRouteServices:
    def _resolve_publication(**kwargs: Any) -> CooldownPublicationPlan:
        classifications.append(str(kwargs["error_class"]))
        return CooldownPublicationPlan(
            duration_seconds=1.0,
            applied_scope="candidate",
        )

    async def _persist(**kwargs: Any) -> None:
        return None

    async def _set_affinity(
        session_key: str | None,
        candidate: dict[str, Any],
    ) -> None:
        return None

    return AliasRouteServices(
        select_candidate_fn=select_candidate,
        perform_candidate_request_fn=perform_candidate,
        resolve_cooldown_publication_fn=_resolve_publication,
        publish_cooldown_memory_fn=lambda **kwargs: None,
        persist_cooldown_fn=_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=lambda body, **kwargs: {
            **dict(body),
            "litellm_metadata": {
                "aawm_alias_routing_audit_events": [],
                "codex_auto_agent_attempts": list(
                    kwargs.get("attempts") or []
                ),
                "codex_auto_agent_skipped_candidates": [],
            },
        },
        raise_redispatch_fn=(
            raise_redispatch
            if raise_redispatch is not None
            else lambda **kwargs: pytest.fail("unexpected redispatch")
        ),
    )


async def _zero_cooldown(_key: str) -> tuple[float, str]:
    return 0.0, "local_fallback"


@pytest.mark.asyncio
async def test_alias_injection_is_managed_openai_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    _patch_candidate_loop_runtime(monkeypatch)

    unmanaged_request = _request(
        "fail,success",
        path="/openai/v1/responses",
    )
    unmanaged_candidate = {
        "provider": "test-provider",
        "model": "test-model",
        "route_family": "test_responses",
        "last_resort": False,
    }
    unmanaged_performed: list[str] = []

    async def _select_unmanaged(**kwargs: Any) -> dict[str, Any]:
        return {
            "candidate": unmanaged_candidate,
            "lane_key": "test-provider:test",
            "cooldown_key": "test-provider:test-model:test",
            "selection_reason": "first_available",
            "session_key": None,
            "skipped": [],
        }

    async def _perform_unmanaged(**kwargs: Any) -> Response:
        unmanaged_performed.append(kwargs["candidate"]["provider"])
        return Response(content=b"ok", status_code=200)

    await candidate_loop.handle_alias_route(
        _alias_services(
            select_candidate=_select_unmanaged,
            perform_candidate=_perform_unmanaged,
            classifications=[],
        ),
        alias_family="codex_auto_agent",
        alias_model="basic",
        request=unmanaged_request,
        prepared_request_body={"model": "basic", "input": "test"},
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_zero_cooldown,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key=(
            "codex_auto_agent_skipped_candidates"
        ),
        no_candidate_detail="no candidate",
        log_label="Codex",
    )

    assert unmanaged_performed == ["test-provider"]
    assert not hasattr(
        unmanaged_request.state,
        "aawm_openai_fault_plan_slot_index",
    )

    managed_request = _request(
        "fail,success",
        path="/openai/v1/responses",
    )
    selected: list[str] = []
    performed: list[str] = []
    classifications: list[str] = []

    async def _select_managed(**kwargs: Any) -> dict[str, Any]:
        context = selection._get_codex_oauth_request_local_failover_context(
            managed_request
        )
        label = "account-a" if context is None else "account-b"
        ordinal = 0 if context is None else 1
        selected.append(label)
        return _managed_selection(label, ordinal=ordinal)

    async def _perform_managed(**kwargs: Any) -> Response:
        performed.append(
            kwargs["candidate"]["codex_oauth_account_label"]
        )
        return Response(content=b"ok", status_code=200)

    response = await candidate_loop.handle_alias_route(
        _alias_services(
            select_candidate=_select_managed,
            perform_candidate=_perform_managed,
            classifications=classifications,
        ),
        alias_family="codex_auto_agent",
        alias_model="basic",
        request=managed_request,
        prepared_request_body={"model": "basic", "input": "test"},
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_zero_cooldown,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key=(
            "codex_auto_agent_skipped_candidates"
        ),
        no_candidate_detail="no candidate",
        log_label="Codex",
    )

    assert response.status_code == 200
    assert selected == ["account-a", "account-b"]
    assert performed == ["account-b"]
    assert classifications == ["usage_limit_reached"]


@pytest.mark.asyncio
async def test_malformed_header_preserves_managed_alias_behavior_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    _patch_candidate_loop_runtime(monkeypatch)
    request = _request(
        "success,fail",
        path="/openai/v1/responses",
    )
    classifications: list[str] = []
    provider = AsyncMock(return_value=Response(content=b"ok", status_code=200))

    async def _select(**kwargs: Any) -> dict[str, Any]:
        return _managed_selection("account-a", ordinal=0)

    response = await candidate_loop.handle_alias_route(
        _alias_services(
            select_candidate=_select,
            perform_candidate=provider,
            classifications=classifications,
        ),
        alias_family="codex_auto_agent",
        alias_model="basic",
        request=request,
        prepared_request_body={"model": "basic", "input": "test"},
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_zero_cooldown,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key=(
            "codex_auto_agent_skipped_candidates"
        ),
        no_candidate_detail="no candidate",
        log_label="Codex",
    )

    assert response.status_code == 200
    provider.assert_awaited_once()
    assert classifications == []
    _assert_no_fault_plan_state(request)


@pytest.mark.asyncio
async def test_alias_injected_failure_preserves_existing_redispatch_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    _patch_candidate_loop_runtime(monkeypatch)
    request = _request(
        "fail,success",
        path="/openai/v1/responses",
    )
    conflict = HTTPException(
        status_code=409,
        detail={
            "error": {"code": "aawm_session_owner_redispatch_required"},
            "redispatch_required": True,
        },
    )
    redispatch_calls: list[dict[str, Any]] = []
    classifications: list[str] = []
    provider = AsyncMock(return_value=Response(content=b"unexpected"))

    async def _select(**kwargs: Any) -> dict[str, Any]:
        return _managed_selection(
            "account-a",
            ordinal=0,
            request_mode="ordinary_continuation",
        )

    def _raise_redispatch(**kwargs: Any) -> None:
        redispatch_calls.append(kwargs)
        raise conflict

    with pytest.raises(HTTPException) as exc_info:
        await candidate_loop.handle_alias_route(
            _alias_services(
                select_candidate=_select,
                perform_candidate=provider,
                classifications=classifications,
                raise_redispatch=_raise_redispatch,
            ),
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body={
                "model": "basic",
                "previous_response_id": "response-test",
            },
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_zero_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key=(
                "codex_auto_agent_skipped_candidates"
            ),
            no_candidate_detail="no candidate",
            log_label="Codex",
        )

    assert exc_info.value is conflict
    provider.assert_not_awaited()
    assert classifications == ["usage_limit_reached"]
    assert len(redispatch_calls) == 1
    assert redispatch_calls[0]["error_class"] == "usage_limit_reached"


@pytest.mark.asyncio
async def test_concurrent_direct_requests_keep_fault_state_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate(monkeypatch)
    recovered_request = _request("fail,success")
    terminal_request = _request("fail,fail")
    initial_bind_barrier = asyncio.Event()
    initial_bind_count = 0
    bind_counts: dict[int, int] = {}
    provider_requests: list[Request] = []

    async def _select(
        request: Request,
        *,
        request_body: dict[str, Any],
    ) -> tuple[Any, ...]:
        nonlocal initial_bind_count
        request_key = id(request)
        ordinal = bind_counts.get(request_key, 0)
        bind_counts[request_key] = ordinal + 1
        if ordinal == 0:
            initial_bind_count += 1
            if initial_bind_count == 2:
                initial_bind_barrier.set()
            await initial_bind_barrier.wait()
        label = (
            "recovered-a"
            if request is recovered_request and ordinal == 0
            else "recovered-b"
            if request is recovered_request
            else "terminal-a"
            if ordinal == 0
            else "terminal-b"
        )
        return (
            _auth(label),
            _managed_selection(label, ordinal=ordinal),
            {},
        )

    async def _handler(**kwargs: Any) -> Response:
        provider_requests.append(kwargs["request"])
        await asyncio.sleep(0)
        return Response(content=b"ok", status_code=200)

    _install_direct_route_mocks(
        monkeypatch,
        select=_select,
        handler=_handler,
    )

    recovered_result, terminal_result = await asyncio.gather(
        lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=recovered_request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        ),
        lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=terminal_request,
            fastapi_response=Response(),
            user_api_key_dict=object(),  # type: ignore[arg-type]
        ),
        return_exceptions=True,
    )

    assert isinstance(recovered_result, Response)
    assert recovered_result.status_code == 200
    assert isinstance(
        terminal_result,
        dev_fault_plan.AawmOpenAIFaultPlanError,
    )
    assert provider_requests == [recovered_request]
    assert bind_counts[id(recovered_request)] == 2
    assert bind_counts[id(terminal_request)] == 3

    recovered_outcome = (
        attempt_records._auto_agent_alias_request_outcome_state(
            recovered_request
        )
    )
    terminal_outcome = (
        attempt_records._auto_agent_alias_request_outcome_state(
            terminal_request
        )
    )
    assert recovered_outcome["outcome"] == "recovered"
    assert terminal_outcome["outcome"] == "failed"
    assert recovered_outcome["request_identity"] != terminal_outcome[
        "request_identity"
    ]
    assert recovered_request.state.aawm_openai_fault_plan_slot_index == 2
    assert terminal_request.state.aawm_openai_fault_plan_slot_index == 2
    assert recovered_request.state.aawm_openai_fault_plan_injected_count == 1
    assert terminal_request.state.aawm_openai_fault_plan_injected_count == 2
