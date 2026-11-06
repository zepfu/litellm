"""Focused regression coverage for provider capacity replay admission."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.provider_call_ledger import (
    ProviderCallLedger,
    ProviderCallLedgerExhausted,
    ensure_openai_wire_replay_allowed,
    is_openai_capacity_recovery_safe,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
    ResponsesStreamPreCommitFailure,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry import (
    OpenAIAlphaCapacityRetryBudget,
    OpenAIAlphaCapacityRetryCoordinator,
)


def _request_with_commitment(reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            _aawm_openai_responses_wire_commitment={
                "precommit_replay_committed": True,
                "precommit_commitment_reason": reason,
                "response_start_sent": False,
                "first_body_sent": False,
                "terminal_wire_committed": False,
                "done_wire_committed": False,
            }
        )
    )


def test_provider_429_can_cross_dispatch_commitment_after_ledger_authorization():
    ledger = ProviderCallLedger(
        provider="openai",
        max_logical_provider_calls=3,
        deadline_seconds=60,
    )
    context = {"account_hash": "acct1", "lane_key": "codex-oauth:account1", "model": "gpt-6.1-sol"}
    ledger.reserve(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        prior_response_closed=True,
    )
    assert ledger.authorize_capacity_retry(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        deadline_at_monotonic=time.monotonic() + 60,
    )

    request = _request_with_commitment("provider_execution_possible")
    ensure_openai_wire_replay_allowed(request, ledger=ledger)


def test_semantic_commitment_stays_terminal_for_provider_429():
    request = _request_with_commitment("assistant_output")
    assert not is_openai_capacity_recovery_safe(
        request.state._aawm_openai_responses_wire_commitment,
        provider_returned=True,
        status_code=429,
    )


def test_provider_429_without_capacity_authorization_stays_terminal():
    request = _request_with_commitment("provider_execution_possible")
    assert not is_openai_capacity_recovery_safe(
        request.state._aawm_openai_responses_wire_commitment,
        provider_returned=True,
        capacity_retry_authorized=False,
        status_code=429,
    )


def test_unknown_semantics_empty_response_capacity_can_recover():
    commitment = {
        "precommit_replay_committed": True,
        "precommit_commitment_reason": "unknown_semantics",
        "response_start_sent": False,
        "first_body_sent": False,
        "terminal_wire_committed": False,
        "done_wire_committed": False,
    }
    error_payload = {
        "type": "response.failed",
        "response": {
            "id": "resp_capacity_opaque",
            "status": "failed",
            "provider_extension": {"tool_call": {"name": "opaque"}},
            "error": {
                "code": "server_is_overloaded",
                "message": "capacity",
            },
        },
    }

    assert is_openai_capacity_recovery_safe(
        commitment,
        provider_returned=True,
        capacity_retry_authorized=True,
        error_class="server_overloaded",
        error_payload=error_payload,
        event_type="response.failed",
    )


def test_unknown_response_field_does_not_block_empty_capacity_failure():
    created = {
        "type": "response.created",
        "response": {
            "id": "resp_capacity",
            "status": "in_progress",
            "model": "gpt-6.1-sol",
            "output": [],
        },
    }
    failed = {
        "type": "response.failed",
        "response": {
            "id": "resp_capacity",
            "status": "failed",
            "output": [],
            "error": {
                "type": "server_error",
                "code": "server_overloaded",
                "message": "The selected model is at capacity.",
            },
            "provider_metadata": {"retry_class": "capacity"},
        },
    }

    def sse(payload: dict[str, object]) -> bytes:
        event_type = str(payload["type"])
        return (
            f"event: {event_type}\ndata: "
            + json.dumps(payload, separators=(",", ":"))
            + "\n\n"
        ).encode()

    decision, error_payload, event_type, commitment_reason = (
        PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(
            [sse(created), sse(failed)]
        )
    )
    assert decision == "failed"
    assert event_type == "response.failed"
    assert isinstance(error_payload, dict)
    assert commitment_reason is None


@pytest.mark.asyncio
async def test_hidden_retry_crosses_dispatch_commitment_for_provider_429(monkeypatch):
    request = _request_with_commitment("provider_execution_possible")
    request.is_disconnected = AsyncMock(return_value=False)
    ledger = ProviderCallLedger(
        provider="openai",
        max_logical_provider_calls=3,
        deadline_seconds=60,
    )
    request.state.aawm_openai_provider_call_ledger = ledger
    context = {
        "account_hash": "acct1",
        "lane_key": "codex-oauth:account1",
        "model": "gpt-6.1-sol",
    }
    ledger.reserve(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        prior_response_closed=True,
    )
    assert ledger.authorize_capacity_retry(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        deadline_at_monotonic=time.monotonic() + 60,
    )
    attempts = 0
    coordinator = OpenAIAlphaCapacityRetryCoordinator(
        target_identity="https://api.openai.com/v1/responses",
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=60),
    )

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            failure = HTTPException(status_code=429, detail="rate limited")
            failure._aawm_provider_returned = True
            raise failure
        return "recovered"

    monkeypatch.setattr(pte, "_passthrough_hidden_retry_sleep", AsyncMock())
    monkeypatch.setattr(
        coordinator,
        "sleep_with_wakeup",
        AsyncMock(return_value="timer"),
    )
    result = await pte._execute_passthrough_pre_first_byte_with_hidden_retries(
        kwargs={"litellm_params": {"metadata": {}}},
        operation_name="test_capacity_recovery",
        operation=operation,
        caller_managed_hidden_retry=False,
        request=request,
        openai_capacity_coordinator=coordinator,
    )
    assert result == "recovered"
    assert attempts == 2


@pytest.mark.asyncio
async def test_hidden_retry_relaxes_unknown_semantics_for_typed_capacity_failure(
    monkeypatch,
):
    request = _request_with_commitment("unknown_semantics")
    request.is_disconnected = AsyncMock(return_value=False)
    ledger = ProviderCallLedger(
        provider="openai",
        max_logical_provider_calls=3,
        deadline_seconds=60,
    )
    request.state.aawm_openai_provider_call_ledger = ledger
    context = {
        "account_hash": "acct1",
        "lane_key": "codex-oauth:account1",
        "model": "gpt-6.1-sol",
    }
    ledger.reserve(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        prior_response_closed=True,
    )
    assert ledger.authorize_capacity_retry(
        target="https://api.openai.com/v1/responses",
        candidate_context=context,
        deadline_at_monotonic=time.monotonic() + 60,
    )
    attempts = 0
    coordinator = OpenAIAlphaCapacityRetryCoordinator(
        target_identity="https://api.openai.com/v1/responses",
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=60),
    )

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PassThroughStreamingHandler._build_responses_pre_commit_failure(
                error_payload={
                    "type": "response.failed",
                    "response": {
                        "id": "resp_capacity",
                        "status": "failed",
                        "output": [],
                        "error": {
                            "code": "server_overloaded",
                            "message": "capacity",
                        },
                    },
                },
                event_type="response.failed",
                openai_alpha_capacity_retry_enabled=True,
            )
        return "recovered"

    monkeypatch.setattr(
        coordinator,
        "sleep_with_wakeup",
        AsyncMock(return_value="timer"),
    )
    result = await pte._execute_passthrough_pre_first_byte_with_hidden_retries(
        kwargs={"litellm_params": {"metadata": {}}},
        operation_name="test_unknown_semantics_capacity_recovery",
        operation=operation,
        caller_managed_hidden_retry=False,
        request=request,
        openai_capacity_coordinator=coordinator,
    )
    assert result == "recovered"
    assert attempts == 2
    assert request.state._aawm_openai_responses_wire_commitment[
        "precommit_commitment_reason"
    ] == "provider_execution_possible"


@pytest.mark.asyncio
async def test_hidden_retry_preserves_capacity_after_ledger_exhaustion(monkeypatch):
    """A local reservation denial must not hide the prior provider result."""

    ledger = ProviderCallLedger(
        provider="openai",
        max_logical_provider_calls=3,
        deadline_seconds=60,
    )
    context = {
        "account_hash": "acct1",
        "lane_key": "codex-oauth:account1",
        "model": "gpt-6.1-sol",
    }
    attempts = 0
    coordinator = OpenAIAlphaCapacityRetryCoordinator(
        target_identity="https://api.openai.com/v1/responses",
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=60),
    )

    async def operation():
        nonlocal attempts
        attempts += 1
        ledger.reserve(
            target="https://api.openai.com/v1/responses",
            candidate_context=context,
            prior_response_closed=True,
        )
        raise ResponsesStreamPreCommitFailure(
            error_class="server_overloaded",
            classification="transient_capacity",
            retryable=True,
            status_code=503,
            provider_returned=True,
            failure_origin="provider",
        )

    monkeypatch.setattr(
        coordinator,
        "sleep_with_wakeup",
        AsyncMock(return_value="timer"),
    )
    with pytest.raises(ResponsesStreamPreCommitFailure) as raised:
        await pte._execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs={"litellm_params": {"metadata": {}}},
            operation_name="test_capacity_ledger_handoff",
            operation=operation,
            caller_managed_hidden_retry=False,
            openai_capacity_coordinator=coordinator,
        )

    assert attempts == 4
    assert raised.value.error_class == "server_overloaded"
    assert raised.value.provider_returned is True
    assert not isinstance(raised.value, ProviderCallLedgerExhausted)
    assert raised.value.aawm_capacity_retry_ledger_exhausted is True
