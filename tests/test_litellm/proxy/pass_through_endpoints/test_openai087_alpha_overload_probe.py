"""Focused regression coverage for the alpha synthetic SSE overload helper."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response
from starlette.requests import Request as StarletteRequest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.alpha_probe_control import (
    ALPHA_OVERLOAD_PROBE_ENABLED_ENV,
    ALPHA_OVERLOAD_PROBE_ERROR_CODES,
    _OVERLOAD_RECEIPT_SEQUENCE_STATE_KEY,
    _RECEIPT_SEQUENCE_STATE_KEY,
    _ALPHA_OVERLOAD_PLAN_DEFINITIONS,
    AlphaOverloadProbePlan,
    AlphaOverloadProbeControl,
    alpha_probe_control_enabled,
    alpha_overload_probe_control_enabled,
    build_alpha_overload_probe_receipt,
    build_synthetic_overload_response,
    claim_alpha_overload_slot,
    get_alpha_overload_probe_control,
    get_alpha_overload_probe_receipts,
    resolve_alpha_overload_probe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry import (
    OpenAIAlphaCapacityRetryCoordinator,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.provider_call_ledger import (
    ProviderCallLedger,
    get_request_provider_call_ledger,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


def _request(plan: str | None = None) -> Request:
    headers = []
    if plan is not None:
        headers.append((b"x-aawm-openai-alpha-overload-probe-plan", plan.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/openai_passthrough/v1/responses",
            "raw_path": b"/openai_passthrough/v1/responses",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 4100),
            "server": ("testserver", 80),
            "http_version": "1.1",
        }
    )


def _admin() -> SimpleNamespace:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    return SimpleNamespace(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        token=UserAPIKeyAuth._safe_hash_litellm_api_key("sk-master-key"),
    )


def _bind_control(
    request: Request,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: str = "overload,overload,overload",
    request_body: dict[str, Any] | None = None,
    session_identity: str | None = None,
    allow_nonfresh: bool = False,
) -> AlphaOverloadProbeControl:
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    monkeypatch.setenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "1")
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-master-key")
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing."
        "attempt_records."
        "_bind_auto_agent_alias_request_identity",
        lambda _request: "probe-request-id",
    )
    control = resolve_alpha_overload_probe(
        request,
        user_api_key_dict=_admin(),
        request_body=request_body,
        session_identity=session_identity,
    )
    assert isinstance(control, AlphaOverloadProbeControl)
    if not allow_nonfresh:
        assert control.pre_admission_rejection_reason is None
    assert control.plan == AlphaOverloadProbePlan(actions=_ALPHA_OVERLOAD_PLAN_DEFINITIONS[plan])
    assert control.request_identity == "probe-request-id"
    return control


def test_safe_off_paths_are_public_no_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AAWM_LITELLM_ENVIRONMENT", raising=False)
    monkeypatch.delenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, raising=False)
    request = _request("overload")

    assert not alpha_probe_control_enabled()
    assert not alpha_overload_probe_control_enabled()
    assert resolve_alpha_overload_probe(request, user_api_key_dict=None) is None
    assert claim_alpha_overload_slot(request) is None
    assert get_alpha_overload_probe_control(request) is None
    assert get_alpha_overload_probe_receipts(request) == ()

    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    monkeypatch.delenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, raising=False)
    assert resolve_alpha_overload_probe(request) is None
    assert claim_alpha_overload_slot(request) is None
    assert get_alpha_overload_probe_control(request) is None
    assert get_alpha_overload_probe_receipts(request) == ()

    request = _request()
    monkeypatch.setenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "1")
    assert resolve_alpha_overload_probe(request) is None
    assert get_alpha_overload_probe_control(request) is None
    assert get_alpha_overload_probe_receipts(request) == ()


def test_enabled_plan_claims_fixed_slots_then_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload,success")
    control = _bind_control(request, monkeypatch, plan="overload,success")

    assert claim_alpha_overload_slot(request, control=control) == "overload"
    assert claim_alpha_overload_slot(request, control=control) == "success"
    assert claim_alpha_overload_slot(request, control=control) is None


def test_supported_three_overload_plan_claims_fixed_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload,overload,overload")
    control = _bind_control(request, monkeypatch)

    assert control.plan == AlphaOverloadProbePlan(
        actions=("overload", "overload", "overload")
    )
    assert [claim_alpha_overload_slot(request, control=control) for _ in range(3)] == [
        "overload",
        "overload",
        "overload",
    ]


def test_receipt_sequences_are_isolated_by_state_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload,success")
    control = _bind_control(request, monkeypatch, plan="overload,success")

    sequences = [build_alpha_overload_probe_receipt(request, control=control)["sequence"] for _ in range(3)]

    assert sequences == [1, 2, 3]
    assert getattr(request.state, _OVERLOAD_RECEIPT_SEQUENCE_STATE_KEY) == 3
    assert not hasattr(request.state, _RECEIPT_SEQUENCE_STATE_KEY)


@pytest.mark.asyncio
async def test_codex_dispatch_resolves_and_scrubs_unbound_overload_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    monkeypatch.setenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "1")
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-master-key")
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_dispatch,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as sa,
    )

    request = _request("overload,success")
    request.scope["parsed_body"] = (
        ("model",),
        {"model": "codex-auto-agent"},
    )
    request.state._cached_headers = {
        "content-type": "application/json",
        "x-aawm-openai-alpha-overload-probe-plan": "overload,success",
    }
    monkeypatch.setattr(sa, "request_session_owner_already_guarded", lambda _r: False)
    monkeypatch.setattr(sa, "get_request_session_owner_lease", lambda _r: None)
    monkeypatch.setattr(
        sa,
        "request_has_effective_session_identity",
        lambda _r: False,
    )
    host = {
        "_resolve_codex_auto_agent_alias_model": (
            lambda body, *, endpoint, request: "codex-auto-agent"
        ),
        "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_opencode_go_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_nous_chat_completions_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_kimi_chat_completions_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_alibaba_token_plan_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_zai_coding_plan_adapter_model": (lambda body, *, endpoint: None),
        "_apply_codex_auto_agent_prevention_guidance_to_request_body": (lambda body: (body, [])),
        "_apply_aawm_read_agent_guidance_to_request_body": (lambda body, *, target_field: (body, [])),
        "_prepare_request_body_for_passthrough_observability": (lambda *, request, request_body: request_body),
        "_safe_set_request_parsed_body": lambda request, body: None,
        "_handle_codex_auto_agent_alias_route": AsyncMock(return_value=Response()),
        "_normalize_codex_reasoning_effort_for_resolved_route": (lambda body, *, resolved_route: (body, None)),
    }
    codex_dispatch.install(host)

    result = await host["try_dispatch_codex_request"](
        endpoint="/v1/responses",
        request=request,
        request_body={"model": "codex-auto-agent"},
        prepared_request_body={"model": "codex-auto-agent"},
        fastapi_response=Response(),
        user_api_key_dict=_admin(),
        target_url="https://example.test/responses",
        api_key="sk-test",
        forward_headers=False,
    )

    assert isinstance(result, Response)
    assert isinstance(
        get_alpha_overload_probe_control(request),
        AlphaOverloadProbeControl,
    )
    assert "x-aawm-openai-alpha-overload-probe-plan" not in request.state._cached_headers


@pytest.mark.asyncio
async def test_codex_dispatch_rejects_bound_overload_probe_on_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_dispatch,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        audit_build,
    )

    request = _request("overload,success")
    request.scope["parsed_body"] = (
        ("model", "previous_response_id"),
        {"model": "codex-auto-agent", "previous_response_id": "resp_parent"},
    )
    request.state._cached_headers = {
        "x-aawm-openai-alpha-overload-probe-plan": "overload,success",
    }
    monkeypatch.setattr(
        audit_build,
        "_codex_auto_agent_request_has_continuation_state",
        lambda body: bool(body.get("previous_response_id")),
    )
    _bind_control(
        request,
        monkeypatch,
        plan="overload,success",
        request_body={"model": "codex-auto-agent", "previous_response_id": "resp_parent"},
        allow_nonfresh=True,
    )
    host = {
        "_resolve_codex_auto_agent_alias_model": (lambda body, *, endpoint, request: "codex-auto-agent"),
        "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_opencode_go_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_nous_chat_completions_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_kimi_chat_completions_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_alibaba_token_plan_adapter_model": (lambda body, *, endpoint: None),
        "_resolve_codex_zai_coding_plan_adapter_model": (lambda body, *, endpoint: None),
        "_apply_codex_auto_agent_prevention_guidance_to_request_body": (
            lambda body: (body, [])
        ),
        "_apply_aawm_read_agent_guidance_to_request_body": (
            lambda body, *, target_field: (body, [])
        ),
        "_prepare_request_body_for_passthrough_observability": (
            lambda *, request, request_body: request_body
        ),
        "_safe_set_request_parsed_body": lambda request, body: None,
        "_handle_codex_auto_agent_alias_route": AsyncMock(return_value=Response()),
    }
    codex_dispatch.install(host)

    with pytest.raises(HTTPException) as raised:
        await host["try_dispatch_codex_request"](
            endpoint="/v1/responses",
            request=request,
            request_body={"model": "codex-auto-agent"},
            prepared_request_body={
                "model": "codex-auto-agent",
                "previous_response_id": "resp_parent",
            },
            fastapi_response=Response(),
            user_api_key_dict=_admin(),
            target_url="https://example.test/responses",
            api_key="sk-test",
            forward_headers=False,
        )

    assert raised.value.status_code == 409
    assert raised.value.detail["error"] == (
        "alpha_overload_probe_requires_fresh_unbound_request"
    )
    assert raised.value.detail["reason"] == "non_fresh_or_bound_body"
    assert host["_handle_codex_auto_agent_alias_route"].await_count == 0


@pytest.mark.asyncio
async def test_codex_dispatch_preserves_capture_after_header_scrub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_dispatch,
    )
    request = _request("overload,success")
    request.scope["parsed_body"] = (
        ("model",),
        {"model": "gpt-6.1-sol"},
    )
    request.state._cached_headers = {
        "content-type": "application/json",
        "x-aawm-openai-alpha-overload-probe-plan": "overload,success",
    }
    _bind_control(
        request,
        monkeypatch,
        plan="overload,success",
        request_body={"model": "gpt-6.1-sol", "previous_response_id": "resp_parent"},
        allow_nonfresh=True,
    )
    host = {
        "_resolve_codex_auto_agent_alias_model": (
            lambda body, *, endpoint, request: None
        ),
        "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_opencode_go_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_nous_chat_completions_adapter_model": (
            lambda body, *, endpoint: None
        ),
        "_resolve_codex_kimi_chat_completions_adapter_model": (
            lambda body, *, endpoint: None
        ),
        "_resolve_codex_alibaba_token_plan_adapter_model": (
            lambda body, *, endpoint: None
        ),
        "_resolve_codex_zai_coding_plan_adapter_model": (
            lambda body, *, endpoint: None
        ),
        "_normalize_codex_reasoning_effort_for_resolved_route": (
            lambda body, *, resolved_route: (body, None)
        ),
    }
    codex_dispatch.install(host)

    with pytest.raises(HTTPException) as raised:
        await host["try_dispatch_codex_request"](
            endpoint="/v1/responses",
            request=request,
            request_body={"model": "gpt-6.1-sol"},
            prepared_request_body={"model": "gpt-6.1-sol"},
            fastapi_response=Response(),
            user_api_key_dict=_admin(),
            target_url="https://example.test/responses",
            api_key="sk-test",
            forward_headers=False,
        )

    assert raised.value.status_code == 409
    assert raised.value.detail["reason"] == "non_fresh_or_bound_body"


@pytest.mark.asyncio
async def test_generated_pass_through_route_succeeds_without_probe_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as pte

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    request = _request(None)
    request._parse_request_data = AsyncMock(return_value=(None, None, None, False))
    route = create_pass_through_route(
        endpoint="ordinary-openai",
        target="https://example.test/v1/responses",
        custom_headers={},
        _forward_headers=False,
        custom_llm_provider="openai",
    )
    route_response = Response(content=b"ok", status_code=200)
    pass_through_request = AsyncMock(return_value=route_response)
    monkeypatch.setattr(pte, "pass_through_request", pass_through_request)
    monkeypatch.setattr(
        pte.InitPassThroughEndpointHelpers,
        "_is_mapped_pass_through_route",
        lambda _path: True,
    )

    result = await route(
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=_admin(),
    )

    assert result is route_response
    assert pass_through_request.await_count == 1
    assert pass_through_request.await_args.kwargs["custom_llm_provider"] == "openai"
    assert get_alpha_overload_probe_control(request) is None


@pytest.mark.asyncio
async def test_generated_pass_through_route_reads_nonempty_starlette_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as pte

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.alpha_probe_control import (
        ALPHA_OVERLOAD_PROBE_ENABLED_ENV,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    monkeypatch.setenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "0")
    request = _request(None)
    request.scope["headers"] = [(b"host", b"testserver")]
    request._parse_request_data = AsyncMock(return_value=(None, None, None, False))
    route = create_pass_through_route(
        endpoint="ordinary-openai",
        target="https://example.test/v1/responses",
        custom_headers={},
        _forward_headers=False,
        custom_llm_provider="openai",
    )
    route_response = Response(content=b"ok", status_code=200)
    pass_through_request = AsyncMock(return_value=route_response)
    monkeypatch.setattr(pte, "pass_through_request", pass_through_request)
    monkeypatch.setattr(
        pte.InitPassThroughEndpointHelpers,
        "_is_mapped_pass_through_route",
        lambda _path: True,
    )

    result = await route(
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=_admin(),
    )

    assert result is route_response
    assert pass_through_request.await_count == 1
    assert get_alpha_overload_probe_control(request) is None


@pytest.mark.asyncio
def test_terminal_overload_plan_claims_exact_budget_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload,overload,overload")
    control = _bind_control(request, monkeypatch)

    assert claim_alpha_overload_slot(request, control=control) == "overload"
    assert claim_alpha_overload_slot(request, control=control) == "overload"
    assert claim_alpha_overload_slot(request, control=control) == "overload"
    with pytest.raises(HTTPException) as raised:
        claim_alpha_overload_slot(request, control=control)
    assert raised.value.status_code == 409
    assert raised.value.detail["error"] == ("alpha_overload_probe_terminal_plan_exhausted")
    with pytest.raises(HTTPException) as raised:
        claim_alpha_overload_slot(request, control=control)
    assert raised.value.status_code == 409
    assert raised.value.detail["error"] == ("alpha_overload_probe_terminal_plan_exhausted")


@pytest.mark.parametrize("error_code", sorted(ALPHA_OVERLOAD_PROBE_ERROR_CODES))
def test_synthetic_response_is_terminal_httpx_sse(
    error_code: str,
) -> None:
    response = build_synthetic_overload_response(error_code=error_code)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.request is not None
    event = response.content.decode()
    assert event.startswith("event: response.failed\ndata: ")
    assert event.endswith("\n\n")
    assert event.count("event:") == 1
    assert "data: [DONE]" not in event
    payload = json.loads(event.removeprefix("event: response.failed\ndata: "))
    assert payload["type"] == "response.failed"
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["output"] == []
    assert payload["response"]["error"]["code"] == error_code
    assert payload["response"]["error"]["type"] == "server_error"

    (
        decision,
        error_payload,
        event_type,
        commitment_reason,
    ) = PassThroughStreamingHandler._inspect_responses_pre_commit_chunks([response.content])
    assert decision == "failed"
    assert event_type == "response.failed"
    assert commitment_reason is None
    assert isinstance(error_payload, dict)
    assert error_payload["response"]["error"]["code"] == error_code


def test_receipt_records_synthetic_no_io_bound_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload,overload,overload")
    control = _bind_control(
        request,
        monkeypatch,
        plan="overload,overload,overload",
    )
    assert claim_alpha_overload_slot(request, control=control) == "overload"
    assert claim_alpha_overload_slot(request, control=control) == "overload"

    selection = {
        "alias_model": "gpt-6.1-sol",
        "candidate": {
            "provider": "openai",
            "model": "gpt-6.1-sol",
            "route_family": "codex_responses",
        },
    }
    first_receipt = build_alpha_overload_probe_receipt(
        request,
        control=control,
        selection=selection,
    )
    second_receipt = build_alpha_overload_probe_receipt(
        request,
        control=control,
        selection=selection,
    )

    assert get_alpha_overload_probe_receipts(request) == (
        first_receipt,
        second_receipt,
    )
    assert [receipt["sequence"] for receipt in (first_receipt, second_receipt)] == [1, 2]
    for receipt in (first_receipt, second_receipt):
        assert receipt["schema"] == "aawm_openai_alpha_overload_probe_receipt_v1"
        assert receipt["request_identity"] == "probe-request-id"
        assert receipt["plan"] == "overload,overload,overload"
        assert receipt["action"] == "inject_synthetic_overload"
        assert receipt["event_type"] == "response.failed"
        assert receipt["error_code"] == "server_is_overloaded"
        assert receipt["synthetic"] is True
        assert receipt["attempted_provider_call"] is False
        assert receipt["provider_returned"] is False
        assert receipt["provider"] == "openai"
        assert receipt["model"] == "gpt-6.1-sol"


def test_malformed_plan_fails_closed_without_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("overload ")
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    monkeypatch.setenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "1")
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-master-key")
    with pytest.raises(HTTPException) as raised:
        resolve_alpha_overload_probe(request, user_api_key_dict=_admin())
    assert raised.value.status_code == 400
    assert not hasattr(request.state, "aawm_openai_alpha_overload_probe_control")
    assert claim_alpha_overload_slot(request) is None


def _full_boundary_request(plan: str, body: dict[str, Any]) -> StarletteRequest:
    request = _request(plan)
    request.scope["parsed_body"] = (tuple(body.keys()), body)
    return request


class _SuccessAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


async def _direct_lease_renewal(_lease, operation, **_kwargs):
    return await operation()


async def _direct_run_with_lease_renewal(*, request, operation):
    return await operation()


async def _no_wait_with_wakeup(
    _coordinator,
    _wait_seconds,
    **_metadata,
):
    return "timer"


def _configure_full_boundary_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    monkeypatch.setenv("AAWM_OPENAI_ALPHA_OVERLOAD_PROBE_ENABLED", "1")
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry."
        "_resolve_redis_for_capacity_wakeup",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry." "_CLIENT_DISCONNECT_POLL_SECONDS",
        0.001,
        raising=False,
    )


async def _run_full_boundary_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: str,
    expected_error: str | None = None,
    request_body: dict[str, Any] | None = None,
    expected_freshness_reason: str | None = None,
    allow_nonfresh: bool = False,
) -> SimpleNamespace:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        audit_build,
        session_affinity as sa,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_wire import (
        OpenAIResponsesStreamingResponse,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as pte

    _configure_full_boundary_retry(monkeypatch)

    body = request_body or {"model": "gpt-6.1-sol", "stream": True}
    request = _full_boundary_request(plan, body)
    request.is_disconnected = AsyncMock(return_value=False)
    _bind_control(
        request,
        monkeypatch,
        plan=plan,
        request_body=body,
        allow_nonfresh=allow_nonfresh,
    )
    if expected_freshness_reason is not None:
        monkeypatch.setattr(
            audit_build,
            "_codex_auto_agent_request_has_continuation_state",
            lambda candidate_body: bool(candidate_body.get("previous_response_id")),
        )
        monkeypatch.setattr(
            sa,
            "request_session_owner_already_guarded",
            lambda _request: False,
        )
        monkeypatch.setattr(
            sa,
            "get_request_session_owner_lease",
            lambda _request: None,
        )
        monkeypatch.setattr(
            sa,
            "request_has_effective_session_identity",
            lambda _request: False,
        )
    target = "https://api.openai.com/v1/responses"
    success_chunks = [
        (
            b'event: response.created\ndata: {"type":"response.created",'
            b'"response":{"id":"resp_success","object":"response",'
            b'"status":"in_progress","model":"gpt-6.1-sol","output":[]}}\n\n'
        ),
        (
            "event: response.completed\ndata: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_success",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "aawm_route_identity": {
                            "producer_provider": "openai",
                            "producer_model": "gpt-6.1-sol",
                            "producer_route_family": "openai",
                        },
                    },
                },
                separators=(",", ":"),
            )
            + "\n\n"
        ).encode("utf-8"),
        b"data: [DONE]\n\n",
    ]
    success_response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_SuccessAsyncStream(success_chunks),
        request=httpx.Request("POST", target),
    )
    captured_prepared_requests: list[httpx.Request] = []

    with patch.object(
        pte,
        "_aawm_session_owner_pre_send_guard",
        new=AsyncMock(),
    ), patch.object(
        pte,
        "_aawm_run_with_session_owner_lease_renewal",
        new=_direct_run_with_lease_renewal,
    ), patch.object(
        pte,
        "_aawm_session_owner_on_upstream_result",
        new=AsyncMock(),
    ), patch.object(
        sa,
        "get_request_session_owner_lease",
        return_value=None,
    ), patch.object(
        sa,
        "run_with_session_owner_lease_renewal",
        new=_direct_lease_renewal,
    ), patch.object(
        OpenAIAlphaCapacityRetryCoordinator,
        "sleep_with_wakeup",
        _no_wait_with_wakeup,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
        "get_async_httpx_client",
    ) as mock_get_async_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
    ) as mock_logging_obj, patch.object(
        pte.pass_through_endpoint_logging,
        "pass_through_async_success_handler",
        new_callable=AsyncMock,
    ):
        mock_client = MagicMock()
        mock_client.send = AsyncMock(return_value=success_response)

        def build_prepared_request(**kwargs):
            prepared = httpx.Request(
                kwargs.get("method", "POST"),
                kwargs["url"],
                headers=kwargs["headers"],
                json=kwargs.get("json"),
            )
            captured_prepared_requests.append(prepared)
            return prepared

        mock_client.build_request = MagicMock(side_effect=build_prepared_request)
        mock_get_async_client.return_value = SimpleNamespace(client=mock_client)
        mock_logging_obj.configure_mock(
            pre_call_hook=AsyncMock(return_value=body),
            post_call_success_hook=AsyncMock(),
            post_call_failure_hook=AsyncMock(),
        )

        try:
            response = await pass_through_request(
                request=request,
                target=target,
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=body,
                custom_llm_provider="openai",
                stream=True,
            )
        except ProxyException as exc:
            if expected_freshness_reason is not None:
                assert str(exc.code) == "409"
                assert exc.detail["error"] == (
                    "alpha_overload_probe_requires_fresh_unbound_request"
                )
                assert exc.detail["reason"] == expected_freshness_reason
            else:
                assert expected_error is not None
                assert str(exc.code) == "503"
                assert exc.detail["error"]["code"] == expected_error
            return SimpleNamespace(
                request=request,
                response=None,
                mock_client=mock_client,
                captured_prepared_requests=captured_prepared_requests,
                success_chunks=success_chunks,
            )

    assert isinstance(response, OpenAIResponsesStreamingResponse)
    return SimpleNamespace(
        request=request,
        response=response,
        mock_client=mock_client,
        captured_prepared_requests=captured_prepared_requests,
        success_chunks=success_chunks,
    )


def _assert_overload_events(
    request: Request,
    *,
    plan: str,
    sequences: list[int],
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.alpha_probe_control import (
        get_alpha_overload_probe_receipts,
    )

    receipts = get_alpha_overload_probe_receipts(request)
    assert [receipt["sequence"] for receipt in receipts] == sequences
    assert getattr(request.state, _OVERLOAD_RECEIPT_SEQUENCE_STATE_KEY) == len(
        sequences
    )

    retry_events = getattr(request.state, "aawm_openai_raw_retry_events", [])
    probe_events = [
        event
        for event in retry_events
        if event.get("event_type") == "openai_alpha_overload_probe"
    ]
    assert len(probe_events) == len(sequences)
    for sequence, event in enumerate(probe_events, start=1):
        assert event["schema"] == "aawm_openai_alpha_overload_probe_receipt_v1"
        assert event["sequence"] == sequence
        assert event["action"] == "inject_synthetic_overload"
        assert event["plan"] == plan
        assert event["error_class"] == "server_overloaded"
        assert event["classification"] == "transient_capacity"
        assert event["error_code"] == "server_is_overloaded"
        assert event["synthetic"] is True
        assert event["attempted_provider_call"] is False
        assert event["provider_returned"] is False
        assert event["environment"] == "litellm-alpha"


@pytest.mark.asyncio
async def test_full_boundary_recovers_after_one_synthetic_overload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = "overload,success"
    result = await _run_full_boundary_probe(monkeypatch, plan=plan)
    request = result.request

    wire = b"".join([chunk async for chunk in result.response.body_iterator])
    assert wire == b"".join(result.success_chunks)
    assert result.mock_client.send.await_count == 1
    assert len(result.captured_prepared_requests) == 2
    assert all(
        "x-aawm-openai-alpha-overload-probe-plan" not in prepared.headers
        for prepared in result.captured_prepared_requests
    )
    _assert_overload_events(request, plan=plan, sequences=[1])

    ledger = get_request_provider_call_ledger(request)
    assert isinstance(ledger, ProviderCallLedger)
    assert ledger.logical_provider_calls == 2
    assert [reservation.ordinal for reservation in ledger.reservations] == [1, 2]
    assert ledger.snapshot()["remaining_logical_provider_calls"] == 1
    assert not hasattr(request.state, "aawm_openai_alpha_probe_exhausted")


@pytest.mark.asyncio
async def test_full_boundary_terminal_overloads_send_no_provider_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = "overload,overload,overload"
    result = await _run_full_boundary_probe(
        monkeypatch,
        plan=plan,
        expected_error="server_is_overloaded",
    )

    assert result.mock_client.send.await_count == 0
    request = result.request
    _assert_overload_events(request, plan=plan, sequences=[1, 2, 3])

    ledger = get_request_provider_call_ledger(request)
    assert isinstance(ledger, ProviderCallLedger)
    assert ledger.logical_provider_calls == 3
    assert [reservation.ordinal for reservation in ledger.reservations] == [1, 2, 3]
    snapshot = ledger.snapshot()
    assert snapshot["logical_provider_calls"] == 3
    assert snapshot["remaining_logical_provider_calls"] == 0
    assert snapshot["next_attempt_ordinal"] == 4


@pytest.mark.asyncio
async def test_native_first_send_rejects_continuation_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run_full_boundary_probe(
        monkeypatch,
        plan="overload,success",
        request_body={
            "model": "gpt-6.1-sol",
            "stream": True,
            "previous_response_id": "resp_parent",
        },
        allow_nonfresh=True,
        expected_freshness_reason="non_fresh_or_bound_body",
    )

    assert result.mock_client.send.await_count == 0
    assert get_alpha_overload_probe_receipts(result.request) == ()
