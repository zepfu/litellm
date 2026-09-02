"""Focused OPENAI-033 regression tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    candidate_loop,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.classification import (
    CooldownEvidenceGate,
    classify_failure,
)
from litellm.proxy.pass_through_endpoints.providers.common import (
    Runtime,
    _codex_native_openai_candidate_unavailable_detail,
    _raise_codex_native_openai_auto_agent_candidate_unavailable,
)


def _runtime() -> Runtime:
    return Runtime(
        extract_status_code=lpe._extract_adapter_exception_status_code,
        extract_detail=lambda exc: getattr(exc, "detail", None),
    )


def _openai_model_not_found(
    *,
    status_code: int = 400,
    payload: Any = None,
) -> ProxyException:
    message = "The requested model 'gpt-5.6-future' does not exist."
    exc = ProxyException(
        message=message,
        type="invalid_request_error",
        param="model",
        code=status_code,
    )
    exc.status_code = status_code
    exc.detail = payload or {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }
    return exc


def _candidate_unavailable() -> ProxyException:
    source_exc = _openai_model_not_found()
    source_exc._aawm_provider_returned = True  # type: ignore[attr-defined]
    with pytest.raises(ProxyException) as caught:
        _raise_codex_native_openai_auto_agent_candidate_unavailable(
            source_exc,
            runtime=_runtime(),
            target_url="https://api.openai.com/v1/responses",
            custom_llm_provider="openai",
            provider_returned=True,
        )
    return caught.value


@pytest.mark.parametrize("status_code", [400, 404])
def test_openai_model_not_found_accepts_existing_responses_shapes(
    status_code: int,
) -> None:
    exc = _openai_model_not_found(status_code=status_code)

    detail = _codex_native_openai_candidate_unavailable_detail(
        exc,
        runtime=_runtime(),
        target_url="https://api.openai.com/v1/responses",
        custom_llm_provider="openai",
        provider_returned=True,
    )

    assert detail is not None
    assert "model_not_found" in detail
    assert "gpt-5.6-future" in detail


@pytest.mark.parametrize(
    ("exc", "target_url", "custom_llm_provider", "provider_returned"),
    [
        (
            _openai_model_not_found(
                payload={
                    "error": {
                        "message": "The requested model does not exist.",
                        "code": "model_not_found",
                    }
                }
            ),
            "https://api.openai.com/v1/responses",
            "openai",
            True,
        ),
        (
            _openai_model_not_found(),
            "https://api.openai.com/v1/chat/completions",
            "openai",
            True,
        ),
        (
            _openai_model_not_found(),
            "https://api.anthropic.com/v1/responses",
            "anthropic",
            True,
        ),
        (
            ProxyException(
                message="invalid api key",
                type="authentication_error",
                param=None,
                code=401,
            ),
            "https://api.openai.com/v1/responses",
            "openai",
            True,
        ),
        (
            _openai_model_not_found(
                status_code=404,
                payload={
                    "error": {
                        "message": (
                            "Item with id 'rs_missing' not found. "
                            "Items are not persisted when `store` is set to false. "
                            "Try again with `store` set to true, or remove this item from your input."
                        ),
                        "type": "invalid_request_error",
                        "code": "item_not_found",
                    }
                },
            ),
            "https://api.openai.com/v1/responses",
            "openai",
            True,
        ),
        (
            _openai_model_not_found(),
            "https://api.openai.com/v1/responses",
            "openai",
            False,
        ),
    ],
)
def test_openai_model_not_found_rejects_non_matching_or_unattributed_errors(
    exc: ProxyException,
    target_url: str,
    custom_llm_provider: str,
    provider_returned: bool,
) -> None:
    assert (
        _codex_native_openai_candidate_unavailable_detail(
            exc,
            runtime=_runtime(),
            target_url=target_url,
            custom_llm_provider=custom_llm_provider,
            provider_returned=provider_returned,
        )
        is None
    )


def test_chatgpt_codex_entitlement_text_remains_candidate_unavailable() -> None:
    exc = ProxyException(
        message=(
            "Model is not supported when using Codex with a ChatGPT account."
        ),
        type="invalid_request_error",
        param="model",
        code=400,
    )
    exc.status_code = 400
    exc.detail = exc.message

    assert (
        _codex_native_openai_candidate_unavailable_detail(
            exc,
            runtime=_runtime(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            provider_returned=False,
        )
        is not None
    )


@pytest.mark.asyncio
async def test_native_openai_maps_attributed_model_not_found_but_preserves_local_proxy_error(
) -> None:
    source_exc = _openai_model_not_found()
    source_exc._aawm_provider_returned = True  # type: ignore[attr-defined]
    local_exc = _openai_model_not_found()

    async def _raise_attributed(**_kwargs: Any) -> Any:
        raise source_exc

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(lpe, "pass_through_request", _raise_attributed)
        with pytest.raises(ProxyException) as caught:
            await lpe._perform_codex_auto_agent_native_openai_request(
                request=MagicMock(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                target_url="https://api.openai.com/v1/responses",
                api_key=None,
                forward_headers=True,
                request_body={"model": "gpt-5.6-future", "input": "hello"},
                custom_headers={},
            )
    assert caught.value.detail["error"]["code"] == (
        "aawm_codex_auto_agent_candidate_unavailable"
    )

    async def _raise_local(**_kwargs: Any) -> Any:
        raise local_exc

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(lpe, "pass_through_request", _raise_local)
        with pytest.raises(ProxyException) as caught:
            await lpe._perform_codex_auto_agent_native_openai_request(
                request=MagicMock(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                target_url="https://api.openai.com/v1/responses",
                api_key=None,
                forward_headers=True,
                request_body={"model": "gpt-5.6-future", "input": "hello"},
                custom_headers={},
            )
    assert caught.value is local_exc


@pytest.mark.asyncio
async def test_direct_model_not_found_remains_terminal_without_account_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_exc = _openai_model_not_found()
    source_exc._aawm_provider_returned = True  # type: ignore[attr-defined]
    selected_auth = SimpleNamespace(headers={"Authorization": "Bearer account-a"})
    selected = {
        "candidate": {
            "provider": "openai",
            "model": "gpt-5.6-future",
            "route_family": "codex_responses",
            "codex_oauth_account_hash": "account-a-hash",
            "codex_oauth_lane_key": "codex-oauth:account-a",
        },
        "request_mode": "fresh",
        "has_account_bound_state": False,
    }
    retry = AsyncMock()

    monkeypatch.setattr(lpe, "get_request_body", AsyncMock(return_value={"model": "gpt-5.6-future"}))
    monkeypatch.setattr(lpe, "_is_oa_xai_request_body", lambda _body: False)
    monkeypatch.setattr(lpe, "_is_grok_native_oauth_request_body", lambda _body: False)
    monkeypatch.setattr(
        lpe,
        "_resolve_codex_auto_agent_alias_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_should_use_direct_codex_oauth_inventory",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        lpe._aawm_codex_oauth,
        "select_and_bind_direct_codex_oauth_inventory",
        AsyncMock(return_value=(selected_auth, selected, {})),
    )
    monkeypatch.setattr(
        lpe.BaseOpenAIPassThroughHandler,
        "_base_openai_pass_through_handler",
        AsyncMock(side_effect=source_exc),
    )
    monkeypatch.setattr(lpe, "_retry_direct_codex_oauth_after_account_failure", retry)
    monkeypatch.setattr(
        lpe._aawm_dev_fault_plan,
        "note_direct_openai_managed_attempt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe._aawm_dev_fault_plan,
        "note_direct_openai_managed_terminal_exhaustion",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ProxyException) as caught:
        await lpe.openai_proxy_route(
            endpoint="v1/responses",
            request=SimpleNamespace(method="POST", state=SimpleNamespace()),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    assert caught.value is source_exc
    retry.assert_not_awaited()


def test_model_not_found_cooldown_uses_exact_model_credential_route_key() -> None:
    selected_key = "openai:gpt-5.6-future:codex-oauth:account-a"
    plan = lpe._resolve_auto_agent_cooldown_publication_plan(
        request=None,
        candidate={
            "provider": "openai",
            "model": "gpt-5.6-future",
            "route_family": "codex_responses",
            "codex_oauth_account_hash": "account-a-hash",
            "codex_oauth_lane_key": "codex-oauth:account-a",
        },
        lane_key="codex-oauth:account-a",
        selected_cooldown_key=selected_key,
        cooldown_seconds=60.0,
        error_class="candidate_unavailable",
    )

    assert plan.applied_scope == "candidate"
    assert plan.memory_keys == (selected_key,)
    assert plan.durable_keys == (selected_key,)


def _session_affinity_stub() -> SimpleNamespace:
    async def _guard(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    return SimpleNamespace(
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop,
        finalize_session_owner_lease_on_failure=_noop,
        reset_released_request_session_owner_guard=lambda _request: False,
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )


class _Admission:
    async def admit_selected_candidate(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(allowed=True, lease=None)

    async def release_provider_lane_admission(self, _lease: Any) -> None:
        return None


def _configure_candidate_loop_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        candidate_loop,
        "alias_routing_state",
        candidate_loop.alias_routing_state.__class__(),
    )
    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        _session_affinity_stub,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(lpe, "_record_codex_failure_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: dict(kwargs["prepared_request_body"]),
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_success",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )


def _selection(candidate: dict[str, Any], cooldown_key: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "lane_key": candidate["codex_oauth_lane_key"],
        "cooldown_key": cooldown_key,
        "selection_reason": "first_available",
        "failover_ordinal": 0,
    }


@pytest.mark.asyncio
async def test_model_not_found_moves_to_next_configured_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_candidate_loop_test(monkeypatch)
    first = {
        "provider": "openai",
        "model": "gpt-5.6-future",
        "route_family": "codex_responses",
        "codex_oauth_lane_key": "codex-oauth:account-a",
    }
    second = {
        "provider": "openai",
        "model": "gpt-5.5-codex",
        "route_family": "codex_responses",
        "codex_oauth_lane_key": "codex-oauth:account-b",
    }
    first_key = "openai:gpt-5.6-future:codex-oauth:account-a"
    second_key = "openai:gpt-5.5-codex:codex-oauth:account-b"
    selections = [_selection(first, first_key), _selection(second, second_key)]
    calls: list[str] = []
    publications: list[dict[str, Any]] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selections.pop(0)

    async def _perform(*, candidate: dict[str, Any], **_kwargs: Any) -> Any:
        calls.append(candidate["model"])
        if candidate is first:
            raise _candidate_unavailable()
        return {"model": candidate["model"]}

    async def _no_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    def _resolve(**kwargs: Any) -> Any:
        publications.append(kwargs)
        return candidate_loop.CooldownPublicationPlan(
            memory_keys=(kwargs["selected_cooldown_key"],),
            durable_keys=(kwargs["selected_cooldown_key"],),
            duration_seconds=60.0,
            applied_scope="candidate",
        )

    async def _persist(**_kwargs: Any) -> None:
        return None

    async def _set_affinity(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        AsyncMock(return_value=None),
    )

    response = await candidate_loop.handle_alias_route(
        SimpleNamespace(
            select_candidate_fn=_select,
            perform_candidate_request_fn=_perform,
            resolve_cooldown_publication_fn=_resolve,
            publish_cooldown_memory_fn=lambda **_kwargs: None,
            persist_cooldown_fn=_persist,
            set_session_affinity_fn=_set_affinity,
            add_alias_metadata_fn=lambda body, **_kwargs: body,
            raise_redispatch_fn=None,
        ),
        alias_family="codex_auto_agent",
        alias_model="work",
        request=SimpleNamespace(state=SimpleNamespace()),
        prepared_request_body={"model": "work", "input": "hello"},
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_no_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response == {"model": second["model"]}
    assert calls == [first["model"], second["model"]]
    assert publications[0]["selected_cooldown_key"] == first_key


@pytest.mark.asyncio
async def test_repeated_model_not_found_candidate_is_skipped_without_second_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_candidate_loop_test(monkeypatch)
    candidate = {
        "provider": "openai",
        "model": "gpt-5.6-future",
        "route_family": "codex_responses",
        "codex_oauth_lane_key": "codex-oauth:account-a",
    }
    key = "openai:gpt-5.6-future:codex-oauth:account-a"
    selections = [_selection(candidate, key), _selection(candidate, key)]
    calls: list[str] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selections.pop(0)

    async def _perform(*, candidate: dict[str, Any], **_kwargs: Any) -> Any:
        calls.append(candidate["model"])
        raise _candidate_unavailable()

    async def _no_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _persist(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        AsyncMock(return_value=None),
    )

    with pytest.raises(lpe.HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            SimpleNamespace(
                select_candidate_fn=_select,
                perform_candidate_request_fn=_perform,
                resolve_cooldown_publication_fn=lambda **kwargs: candidate_loop.CooldownPublicationPlan(
                    memory_keys=(kwargs["selected_cooldown_key"],),
                    durable_keys=(kwargs["selected_cooldown_key"],),
                    duration_seconds=60.0,
                    applied_scope="candidate",
                ),
                publish_cooldown_memory_fn=lambda **_kwargs: None,
                persist_cooldown_fn=_persist,
                set_session_affinity_fn=lambda *_args, **_kwargs: None,
                add_alias_metadata_fn=lambda body, **_kwargs: body,
                raise_redispatch_fn=None,
            ),
            alias_family="codex_auto_agent",
            alias_model="work",
            request=SimpleNamespace(state=SimpleNamespace()),
            prepared_request_body={"model": "work", "input": "hello"},
            max_candidate_attempts=2,
            get_active_cooldown_state_fn=_no_cooldown,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value.status_code == 503
    assert caught.value.detail["error"]["code"] == "all_candidates_unavailable"
    assert calls == [candidate["model"]]


def test_model_not_found_half_open_probe_recovers_exact_key() -> None:
    key = "openai:gpt-5.6-future:codex-oauth:account-a"
    gate = CooldownEvidenceGate(base_seconds=10.0, max_seconds=60.0)
    event = classify_failure(
        status_code=404,
        provider="openai",
        message="The requested model does not exist.",
    )
    decision = gate.record(cooldown_key=key, event=event, now_monotonic=100.0)

    assert decision.should_cool is True
    assert gate.allow_half_open_probe(
        cooldown_key=key,
        now_monotonic=decision.cooled_until_monotonic + 0.01,
    )
    gate.record_probe_result(cooldown_key=key, success=True)
    assert gate.is_cooled(
        cooldown_key=key,
        now_monotonic=decision.cooled_until_monotonic + 0.02,
    ) is False
