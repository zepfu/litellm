"""Healthy AAWM_XAI_DEFERRED_STREAM INFO is gated by AAWM_ALIAS_ROUTE_LOG_HEALTHY."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity import (
    SessionOwnerLease,
    _make_xai_deferred_stream_observer,
    _xai_deferred_stream_should_emit,
)


def _completed_valid_cancel_payload(phase: str, **fields: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "session_owner_deferred_stream",
        "phase": phase,
        "complete": True,
        "valid": True,
        "terminal_seen": True,
        "terminal_status": "completed",
        "validation_state_present": True,
        "lease_present": True,
        "lease_decision": "compatible_owner",
        "held_reservation": False,
        "promoted": True,
        "released": False,
        "success_finalizer_source": "supplied",
        "wire_disposition": "unknown",
        "wire_terminal_pending": False,
    }
    payload.update(fields)
    return payload


def _xai_request() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            aawm_openai_candidate_context={
                "provider": "xai",
                "route_family": "xai_oauth_api",
            },
            aawm_alias_request_litellm_call_id="call-deferred-1",
            aawm_alias_request_context=None,
        )
    )


def _xai_lease() -> SessionOwnerLease:
    return SessionOwnerLease(
        session_identity="session-deferred-1",
        cache_key="owner-key-deferred-1",
        held_reservation=True,
        decision="reservation_renewed",
        attributes={"provider": "xai", "route_family": "xai_oauth_api"},
    )


def _completed_response() -> SimpleNamespace:
    return SimpleNamespace(
        _aawm_responses_validation_state={
            "complete": True,
            "valid": True,
            "terminal_seen": True,
            "terminal_status": "completed",
        }
    )


def _observer():
    return _make_xai_deferred_stream_observer(
        _xai_request(),
        _xai_lease(),
        _completed_response(),
        success_finalizer=lambda: None,
    )


def test_xai_deferred_stream_should_emit_healthy_only_when_flag_on(monkeypatch) -> None:
    healthy = {
        "event": "session_owner_deferred_stream",
        "phase": "validator_decision",
        "validation_ok": True,
        "terminal_status": "completed",
    }
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    assert _xai_deferred_stream_should_emit(healthy) is False
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    assert _xai_deferred_stream_should_emit(healthy) is True


def test_xai_deferred_stream_should_emit_failure_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    assert (
        _xai_deferred_stream_should_emit(
            {
                "event": "session_owner_deferred_stream",
                "phase": "renewal_failed",
            }
        )
        is True
    )
    assert (
        _xai_deferred_stream_should_emit(
            {
                "event": "session_owner_deferred_stream",
                "phase": "validator_decision",
                "validation_ok": False,
            }
        )
        is True
    )


def test_observer_suppresses_healthy_validator_decision(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    observe = _observer()
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.verbose_aawm_route_logger.info"
    ) as info:
        observe("validator_decision", validation_ok=True)
    info.assert_not_called()


def test_observer_emits_healthy_validator_decision_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    observe = _observer()
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.verbose_aawm_route_logger.info"
    ) as info:
        observe("validator_decision", validation_ok=True)
    info.assert_called_once()
    message = info.call_args.args[0]
    assert message.startswith("AAWM_XAI_DEFERRED_STREAM: ")
    assert '"phase":"validator_decision"' in message
    assert '"validation_ok":true' in message


def test_xai_deferred_stream_suppresses_completed_valid_post_terminal_cancel(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    sequence = [
        _completed_valid_cancel_payload(
            "iterator_cancelled",
            exception_category="cancelled",
            finalization_task_present=True,
            iterator_closed=False,
            iterator_completed=False,
        ),
        _completed_valid_cancel_payload(
            "finalize_enter",
            exception_category="cancelled",
            requested_success=False,
            finalization_task_present=True,
            iterator_closed=False,
            iterator_completed=False,
        ),
        _completed_valid_cancel_payload(
            "finalization_task_reused",
            finalization_basis="failure",
            requested_success=False,
            finalization_task_present=True,
            iterator_closed=False,
            iterator_completed=False,
        ),
        _completed_valid_cancel_payload(
            "finalization_task_returned",
            requested_success=False,
            finalization_task_present=True,
            iterator_closed=False,
            iterator_completed=False,
        ),
        _completed_valid_cancel_payload(
            "stream_response_cancelled",
            exception_category="cancelled",
            finalization_task_present=True,
            iterator_closed=True,
            iterator_completed=False,
        ),
        _completed_valid_cancel_payload(
            "finalize_enter",
            exception_category="cancelled",
            requested_success=False,
            finalization_task_present=True,
            iterator_closed=True,
            iterator_completed=False,
        ),
    ]
    for payload in sequence:
        assert _xai_deferred_stream_should_emit(payload) is False
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    for payload in sequence:
        assert _xai_deferred_stream_should_emit(payload) is True


def test_xai_deferred_stream_emits_mid_stream_cancel_when_flag_off(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    assert (
        _xai_deferred_stream_should_emit(
            _completed_valid_cancel_payload(
                "iterator_cancelled",
                complete=False,
                valid=False,
                terminal_seen=False,
                terminal_status="in_progress",
                exception_category="cancelled",
            )
        )
        is True
    )
    assert (
        _xai_deferred_stream_should_emit(
            {
                "event": "session_owner_deferred_stream",
                "phase": "iterator_cancelled",
                "exception_category": "cancelled",
            }
        )
        is True
    )


def test_observer_suppresses_completed_valid_iterator_cancelled(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    observe = _observer()
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.verbose_aawm_route_logger.info"
    ) as info:
        observe("iterator_cancelled", error=asyncio.CancelledError())
    info.assert_not_called()


def test_observer_emits_completed_valid_iterator_cancelled_when_flag_on(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    observe = _observer()
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.verbose_aawm_route_logger.info"
    ) as info:
        observe("iterator_cancelled", error=asyncio.CancelledError())
    info.assert_called_once()
    message = info.call_args.args[0]
    assert '"phase":"iterator_cancelled"' in message
    assert '"exception_category":"cancelled"' in message
    assert '"terminal_status":"completed"' in message


def test_observer_emits_renewal_failed_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    observe = _observer()
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.verbose_aawm_route_logger.info"
    ) as info:
        observe(
            "renewal_failed",
            site="finalization",
            renewal_error=RuntimeError("reservation renewal failed"),
        )
    info.assert_called_once()
    message = info.call_args.args[0]
    assert '"phase":"renewal_failed"' in message
    assert '"exception_category":"runtime"' in message
