"""Focused regression coverage for the alpha synthetic SSE overload helper."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.alpha_probe_control import (
    ALPHA_OVERLOAD_PROBE_ENABLED_ENV,
    ALPHA_OVERLOAD_PROBE_ERROR_CODES,
    AlphaOverloadProbeControl,
    AlphaOverloadProbePlan,
    alpha_probe_control_enabled,
    alpha_overload_probe_control_enabled,
    build_alpha_overload_probe_receipt,
    build_synthetic_overload_response,
    claim_alpha_overload_slot,
    get_alpha_overload_probe_control,
    get_alpha_overload_probe_receipts,
    resolve_alpha_overload_probe,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


def _request(plan: str | None = None) -> Request:
    headers = []
    if plan is not None:
        headers.append(
            (b"x-aawm-openai-alpha-overload-probe-plan", plan.encode())
        )
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
    control = resolve_alpha_overload_probe(request, user_api_key_dict=_admin())
    assert isinstance(control, AlphaOverloadProbeControl)
    expected_actions: tuple[str, ...] = tuple(
        action.strip() for action in plan.split(",") if action.strip()
    )
    assert control.plan == AlphaOverloadProbePlan(actions=expected_actions)
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
    assert raised.value.detail["error"] == (
        "alpha_overload_probe_terminal_plan_exhausted"
    )
    with pytest.raises(HTTPException) as raised:
        claim_alpha_overload_slot(request, control=control)
    assert raised.value.status_code == 409
    assert raised.value.detail["error"] == (
        "alpha_overload_probe_terminal_plan_exhausted"
    )


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

    decision, error_payload, event_type, commitment_reason = (
        PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(
            [response.content]
        )
    )
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
