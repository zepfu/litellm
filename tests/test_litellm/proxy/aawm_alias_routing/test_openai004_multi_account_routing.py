from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_adapter_calls,
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    candidate_loop,
    codex_oauth,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_NATIVE_PROVIDER,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    SelectionEnumeration,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    alias_routing_state,
)
from litellm.secret_managers import codex_oauth_inventory
from litellm.secret_managers.codex_oauth_inventory import (
    CodexOAuthCredentialRecord,
    CodexOAuthInventory,
)


def _request(*, authorization: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )


def _record(
    label: str,
    account_hash: str,
    priority: int,
    declaration_order: int,
) -> CodexOAuthCredentialRecord:
    return CodexOAuthCredentialRecord(
        label=label,
        auth_path=Path(f"/private/{label}.json"),
        lock_path=Path(f"/private/{label}.lock"),
        priority=priority,
        weight=1.0,
        enabled=True,
        models=("*",),
        expected_account_hash=account_hash,
        declaration_order=declaration_order,
    )


def _inventory() -> CodexOAuthInventory:
    return CodexOAuthInventory(
        records=(
            _record("account1", "hash-account-1", 10, 0),
            _record("account2", "hash-account-2", 20, 1),
        )
    )


def _candidate() -> dict[str, Any]:
    return {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex",
        "route_family": "codex_responses",
        "last_resort": False,
        "selection_priority": 100,
    }


def _quota_observation(
    *,
    account_hash: str,
    observed_at: datetime,
    reset_at: datetime,
    quota_period: str,
    window_minutes: int,
    exhausted: bool,
) -> dict[str, Any]:
    return {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex",
        "account_hash": account_hash,
        "environment": "test",
        "quota_key": f"{account_hash}:{quota_period}",
        "quota_type": "tokens",
        "limit_scope": (
            "primary" if quota_period == "five_hour" else "secondary"
        ),
        "quota_period": quota_period,
        "window_minutes": window_minutes,
        "remaining_pct": 0.0 if exhausted else 50.0,
        "observed_at": observed_at,
        "expected_reset_at": reset_at,
        "status": "fresh",
        "exhausted": exhausted,
        "source": "codex_native_quota_poll",
    }


async def _healthy_auth(
    _request: Request,
    record: CodexOAuthCredentialRecord,
) -> codex_oauth.CodexOAuthRequestAuth:
    return codex_oauth.CodexOAuthRequestAuth(
        account_label=record.label,
        account_hash=record.expected_account_hash,
        lane_key=codex_oauth._codex_oauth_account_lane_key(
            account_label=record.label,
            account_hash=record.expected_account_hash,
        ),
        headers={
            "Authorization": f"Bearer token-{record.label}",
            "chatgpt-account-id": f"id-{record.label}",
        },
    )


def _patch_selector_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    affinity: dict[str, Any] | None = None,
    continuation: bool = False,
) -> None:
    enumeration = SelectionEnumeration(
        candidates=(_candidate(),),
        commit_token=None,
    )

    async def _zero_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    async def _affinity(_session_key: str | None) -> dict[str, Any] | None:
        return affinity

    monkeypatch.setattr(
        lpe,
        "_lookup_active_snapshot_canonical_alias",
        lambda _model, *, request=None: "basic",
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_aawm_alias_selection_enumeration",
        lambda request, alias, *, ingress, client_product_label=None: (
            enumeration
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_select_snapshot_candidates",
        lambda alias, *, ingress, client_product_label, request,
        include_out_of_schedule: list(enumeration.candidates),
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_auto_agent_active_cooldown_state",
        _zero_cooldown,
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_session_affinity",
        _affinity,
    )
    monkeypatch.setattr(
        lpe,
        "_extract_client_product_label",
        lambda request, body: None,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_codex_session_key",
        lambda request, body, *, alias_model: "session-1",
    )
    monkeypatch.setattr(
        lpe,
        "_has_continuation_state",
        lambda body: continuation,
    )


@pytest.fixture(autouse=True)
def _reset_alias_state() -> None:
    alias_routing_state.reset_for_tests()
    yield
    alias_routing_state.reset_for_tests()


def test_account_specific_quota_resolution_preserves_unfiltered_ambiguity() -> None:
    manager = AliasRoutingStateManager()
    now = datetime.now(timezone.utc)
    manager.record_normalized_quota_observations(
        [
            _quota_observation(
                account_hash="hash-account-1",
                observed_at=now,
                reset_at=now + timedelta(hours=2),
                quota_period="five_hour",
                window_minutes=300,
                exhausted=True,
            ),
            _quota_observation(
                account_hash="hash-account-2",
                observed_at=now,
                reset_at=now + timedelta(days=2),
                quota_period="seven_day",
                window_minutes=10080,
                exhausted=False,
            ),
        ]
    )

    assert (
        manager.resolve_normalized_quota_observation(
            provider=CODEX_AUTO_AGENT_NATIVE_PROVIDER,
            model="gpt-5.3-codex",
        )
        is None
    )
    account2 = manager.resolve_normalized_quota_observation(
        provider=CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        model="gpt-5.3-codex",
        account_hash="hash-account-2",
    )
    assert account2 is not None
    assert account2["account_hash"] == "hash-account-2"
    assert account2["remaining_pct"] == 50.0


@pytest.mark.asyncio
async def test_selection_is_deterministic_and_both_exhausted_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory()
    monkeypatch.setattr(
        codex_oauth_inventory,
        "load_codex_oauth_inventory",
        lambda: inventory,
    )
    monkeypatch.setattr(
        codex_oauth,
        "_load_codex_oauth_headers_for_record",
        _healthy_auth,
    )
    _patch_selector_runtime(monkeypatch)
    now = datetime.now(timezone.utc)
    alias_routing_state.record_normalized_quota_observations(
        [
            _quota_observation(
                account_hash="hash-account-1",
                observed_at=now,
                reset_at=now + timedelta(hours=2),
                quota_period="five_hour",
                window_minutes=300,
                exhausted=True,
            ),
            _quota_observation(
                account_hash="hash-account-2",
                observed_at=now - timedelta(hours=1),
                reset_at=now + timedelta(days=2),
                quota_period="seven_day",
                window_minutes=10080,
                exhausted=True,
            ),
        ]
    )

    selected = await lpe._select_codex_auto_agent_candidate(
        request=_request(),
        request_body={"model": "basic"},
    )
    assert selected["candidate"]["codex_oauth_account_label"] == "account2"
    assert selected["candidate"]["codex_oauth_account_hash"] == (
        "hash-account-2"
    )
    assert selected["lane_key"] == (
        "codex-oauth:account2:hash-account-2"
    )
    assert selected["failover_ordinal"] == 0

    alias_routing_state.reset_for_tests()
    alias_routing_state.record_normalized_quota_observations(
        [
            _quota_observation(
                account_hash="hash-account-1",
                observed_at=now,
                reset_at=now + timedelta(hours=2),
                quota_period="five_hour",
                window_minutes=300,
                exhausted=True,
            ),
            _quota_observation(
                account_hash="hash-account-2",
                observed_at=now,
                reset_at=now + timedelta(days=2),
                quota_period="seven_day",
                window_minutes=10080,
                exhausted=True,
            ),
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe._select_codex_auto_agent_candidate(
            request=_request(),
            request_body={"model": "basic"},
        )
    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert {item["account_label"] for item in detail["candidates"]} == {
        "account1",
        "account2",
    }
    assert detail["terminal_reset"]["reason"] == (
        "codex_oauth_quota_exhausted"
    )
    terminal_accounts = detail["terminal_reset"]["accounts"]
    assert {
        account["account_hash"] for account in terminal_accounts
    } == {"hash-account-1", "hash-account-2"}
    assert all(
        account["exhausted_windows"][0]["reset_at"]
        for account in terminal_accounts
    )


@pytest.mark.asyncio
async def test_continuation_affinity_pins_account_and_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory()
    pinned = {
        **_candidate(),
        "codex_oauth_account_label": "account1",
        "codex_oauth_account_hash": "hash-account-1",
        "codex_oauth_lane_key": (
            "codex-oauth:account1:hash-account-1"
        ),
    }
    monkeypatch.setattr(
        codex_oauth_inventory,
        "load_codex_oauth_inventory",
        lambda: inventory,
    )
    loaded_labels: list[str] = []

    async def _load(
        _request: Request,
        record: CodexOAuthCredentialRecord,
    ) -> codex_oauth.CodexOAuthRequestAuth:
        loaded_labels.append(record.label)
        raise HTTPException(status_code=401, detail="unavailable")

    monkeypatch.setattr(
        codex_oauth,
        "_load_codex_oauth_headers_for_record",
        _load,
    )
    _patch_selector_runtime(
        monkeypatch,
        affinity=pinned,
        continuation=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await lpe._select_codex_auto_agent_candidate(
            request=_request(),
            request_body={
                "model": "basic",
                "previous_response_id": "response-1",
            },
        )
    assert exc_info.value.status_code == 429
    assert loaded_labels == ["account1"]
    assert exc_info.value.detail["candidate"]["account_label"] == "account1"
    assert exc_info.value.detail["failure_phase"] == "pre_dispatch_auth"


def _patch_candidate_loop_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lpe,
        "_codex_auto_agent_request_has_continuation_state",
        lambda body: False,
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_auto_agent_native_grok_continuation_transient_max_attempts",
        lambda: 1,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: dict(kwargs["prepared_request_body"]),
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **kwargs: {"litellm_metadata": {}},
    )
    monkeypatch.setattr(
        lpe,
        "_is_auto_agent_alias_in_flight_cooldown_http_exception",
        lambda exc: False,
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_get_safe_kimi_code_probe_failure_metadata",
        lambda exc, *, candidate: None,
    )
    monkeypatch.setattr(
        lpe,
        "_classify_kimi_code_auto_agent_probe_failure",
        lambda metadata: None,
    )
    monkeypatch.setattr(
        lpe,
        "_classify_codex_auto_agent_retryable_exhaustion",
        lambda exc: "capacity_exhausted",
    )
    monkeypatch.setattr(
        lpe,
        "_is_codex_auto_agent_grok_account_quota_exhaustion",
        lambda exc, *, candidate: False,
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_auto_agent_cooldown_seconds",
        lambda exc, *, candidate: 0.0,
    )
    monkeypatch.setattr(
        lpe,
        "_record_codex_failure_evidence",
        lambda **kwargs: None,
    )

    def _update_attempt(**kwargs: Any) -> set[str]:
        kwargs["attempt_record"].update(
            {
                "status": "cooldown_set",
                "error_class": kwargs["error_class"],
                "failure_phase": "provider_attempt",
                "attempted_provider_call": True,
            }
        )
        return {"capacity_exhausted"}

    monkeypatch.setattr(
        lpe,
        "_update_codex_auto_agent_retryable_attempt_record",
        _update_attempt,
    )
    monkeypatch.setattr(
        lpe,
        "_is_codex_auto_agent_native_grok_continuation_transient_retry_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        lpe,
        "_is_codex_auto_agent_native_grok_4_5_candidate",
        lambda candidate: False,
    )
    monkeypatch.setattr(
        lpe,
        "_plan_codex_auto_agent_native_grok_continuation_transient_retry",
        lambda **kwargs: (False, None, None),
    )
    monkeypatch.setattr(
        lpe,
        "_get_codex_auto_agent_source_error_summary",
        lambda exc, *, status_code=None: "capacity exhausted",
    )
    monkeypatch.setattr(
        lpe,
        "_extract_adapter_exception_status_code",
        lambda exc: 429,
    )

    async def _no_publication(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _no_publication,
    )


def _loop_services(
    *,
    select_candidate: Any,
    perform_candidate: Any,
) -> AliasRouteServices:
    def _resolve_publication(**kwargs: Any) -> CooldownPublicationPlan:
        return CooldownPublicationPlan(
            duration_seconds=0.0,
            applied_scope="account",
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
        add_alias_metadata_fn=lambda body, **kwargs: dict(body),
        raise_redispatch_fn=lambda **kwargs: pytest.fail(
            "unexpected continuation redispatch"
        ),
    )


def _account_selection(
    account_label: str,
    account_hash: str,
    *,
    failover_ordinal: int,
) -> dict[str, Any]:
    lane = f"codex-oauth:{account_label}:{account_hash}"
    return {
        "candidate": {
            **_candidate(),
            "codex_oauth_account_label": account_label,
            "codex_oauth_account_hash": account_hash,
            "codex_oauth_lane_key": lane,
        },
        "lane_key": lane,
        "cooldown_key": f"openai:gpt-5.3-codex:{lane}",
        "selection_reason": (
            "codex_oauth_account_failover"
            if failover_ordinal
            else "first_available"
        ),
        "failover_ordinal": failover_ordinal,
        "session_key": None,
        "skipped": [],
    }


@pytest.mark.asyncio
async def test_candidate_loop_allows_exactly_one_account_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_candidate_loop_host(monkeypatch)
    request = _request()
    selected_labels: list[str] = []
    performed_labels: list[str] = []

    async def _select(**kwargs: Any) -> dict[str, Any]:
        context = getattr(
            request.state,
            "aawm_codex_oauth_request_local_failover_context",
            None,
        )
        if context is None:
            selected_labels.append("account1")
            return _account_selection(
                "account1",
                "hash-account-1",
                failover_ordinal=0,
            )
        selected_labels.append("account2")
        return _account_selection(
            "account2",
            "hash-account-2",
            failover_ordinal=1,
        )

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        performed_labels.append(candidate["codex_oauth_account_label"])
        raise RuntimeError("account capacity exhausted")

    async def _zero(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    with pytest.raises(HTTPException) as exc_info:
        await candidate_loop.handle_alias_route(
            _loop_services(
                select_candidate=_select,
                perform_candidate=_perform,
            ),
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_zero,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key=(
                "codex_auto_agent_skipped_candidates"
            ),
            no_candidate_detail="no candidate",
            log_label="Codex",
        )
    assert exc_info.value.status_code == 429
    assert selected_labels == ["account1", "account2"]
    assert performed_labels == ["account1", "account2"]


@pytest.mark.asyncio
async def test_stream_failure_after_response_start_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_candidate_loop_host(monkeypatch)
    request = _request()
    selection_count = 0
    perform_count = 0

    async def _select(**kwargs: Any) -> dict[str, Any]:
        nonlocal selection_count
        selection_count += 1
        return _account_selection(
            "account1",
            "hash-account-1",
            failover_ordinal=0,
        )

    async def _stream():
        yield b"first"
        raise RuntimeError("stream failed after first byte")

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        nonlocal perform_count
        perform_count += 1
        return StreamingResponse(_stream(), media_type="text/event-stream")

    async def _zero(_key: str) -> tuple[float, str]:
        return 0.0, "local_fallback"

    response = await candidate_loop.handle_alias_route(
        _loop_services(
            select_candidate=_select,
            perform_candidate=_perform,
        ),
        alias_family="codex_auto_agent",
        alias_model="basic",
        request=request,
        prepared_request_body={"model": "basic", "stream": True},
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_zero,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key=(
            "codex_auto_agent_skipped_candidates"
        ),
        no_candidate_detail="no candidate",
        log_label="Codex",
    )
    assert isinstance(response, StreamingResponse)
    assert await response.body_iterator.__anext__() == b"first"
    with pytest.raises(
        RuntimeError,
        match="stream failed after first byte",
    ):
        await response.body_iterator.__anext__()
    assert selection_count == 1
    assert perform_count == 1


@pytest.mark.asyncio
async def test_selected_account_drives_both_ingress_auth_and_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        **_candidate(),
        "codex_oauth_account_label": "account2",
        "codex_oauth_account_hash": "hash-account-2",
        "codex_oauth_lane_key": (
            "codex-oauth:account2:hash-account-2"
        ),
        "raw_account_id": "raw-account-id",
        "access_token": "selected-token",
        "auth_path": "/private/oauth.json",
    }
    selected_auth = codex_oauth.CodexOAuthRequestAuth(
        account_label="account2",
        account_hash="hash-account-2",
        lane_key="codex-oauth:account2:hash-account-2",
        headers={
            "Authorization": "Bearer selected-token",
            "chatgpt-account-id": "selected-account-id",
        },
    )

    async def _load_selected(
        request: Request,
    ) -> codex_oauth.CodexOAuthRequestAuth:
        bound = codex_oauth._get_bound_codex_oauth_candidate_identity(
            request
        )
        assert bound is not None
        assert bound["account_label"] == "account2"
        return selected_auth

    real_load_bound = codex_oauth._load_bound_codex_oauth_auth
    monkeypatch.setattr(
        codex_oauth,
        "_load_bound_codex_oauth_auth",
        _load_selected,
    )
    monkeypatch.setattr(
        codex_oauth,
        "_codex_oauth_responses_target_url",
        lambda: "https://chatgpt.test/backend-api/codex/responses",
    )
    native_request = AsyncMock(return_value=Response(content=b"ok"))
    monkeypatch.setitem(
        codex_candidate_calls
        ._perform_codex_auto_agent_alias_candidate_request.__globals__,
        "_perform_codex_auto_agent_native_openai_request",
        native_request,
    )
    request = _request(authorization="Bearer inbound-secret")
    await codex_candidate_calls._perform_codex_auto_agent_alias_candidate_request(
        endpoint="v1/responses",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=object(),
        candidate=candidate,
        candidate_body={"model": "gpt-5.3-codex"},
        target_url="https://api.openai.com/v1/responses",
        api_key="generic-openai-key",
        forward_headers=True,
    )
    native_kwargs = native_request.await_args.kwargs
    assert native_kwargs["target_url"] == (
        "https://chatgpt.test/backend-api/codex/responses"
    )
    assert native_kwargs["api_key"] is None
    assert native_kwargs["forward_headers"] is False
    assert native_kwargs["custom_headers"] == selected_auth.headers

    anthropic_request = _request(
        authorization="Bearer inbound-anthropic-secret"
    )
    codex_oauth._bind_codex_oauth_candidate_to_request(
        anthropic_request,
        candidate,
    )
    custom_headers, forward_headers, codex_defaults, credential_family = (
        await anthropic_adapter_calls._resolve_anthropic_openai_responses_adapter_auth_context(
            anthropic_request
        )
    )
    assert custom_headers == selected_auth.headers
    assert forward_headers is False
    assert codex_defaults is True
    assert credential_family == "openai"

    monkeypatch.setattr(
        codex_oauth,
        "_load_bound_codex_oauth_auth",
        real_load_bound,
    )

    async def _unsafe_failure(
        request: Request,
        *,
        account_label: str | None = None,
        model: str | None = None,
    ) -> codex_oauth.CodexOAuthRequestAuth:
        raise HTTPException(
            status_code=500,
            detail={
                "access_token": "selected-token",
                "account_id": "raw-account-id",
                "auth_path": "/private/oauth.json",
            },
        )

    monkeypatch.setattr(
        codex_oauth,
        "_load_local_codex_auth_selection",
        _unsafe_failure,
    )
    with pytest.raises(HTTPException) as exc_info:
        await codex_oauth._load_bound_codex_oauth_auth(
            anthropic_request
        )
    public_error = json.dumps(exc_info.value.detail, sort_keys=True)
    public_candidate = json.dumps(
        lpe._codex_auto_agent_candidate_public_shape(candidate),
        sort_keys=True,
    )
    for secret in (
        "selected-token",
        "selected-account-id",
        "raw-account-id",
        "/private/oauth.json",
        "inbound-secret",
        "generic-openai-key",
    ):
        assert secret not in public_error
        assert secret not in public_candidate
