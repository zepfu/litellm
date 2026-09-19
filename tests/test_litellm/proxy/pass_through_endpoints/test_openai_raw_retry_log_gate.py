"""Healthy AAWM_OPENAI_RAW_RETRY INFO is gated by AAWM_ALIAS_ROUTE_LOG_HEALTHY."""

from __future__ import annotations

from unittest.mock import patch

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _openai_raw_retry_console_should_emit,
    _record_openai_raw_retry_event,
)


def _not_retryable_400() -> dict:
    return {
        "event_type": "openai_raw_http_classification",
        "event_stage": "authorization_classification",
        "classification_reason": "status_not_retryable",
        "classification_status_code": 400,
        "observed_http_status_code": 400,
        "authorization_result": None,
        "retryable": None,
        "error_class": None,
    }


def _denied_authorization() -> dict:
    return {
        "event_type": "openai_capacity_retry_authorization",
        "event_stage": "authorization",
        "authorization_result": "denied",
        "authorization_denial_reason": "raw_classification_unavailable",
        "observed_http_status_code": 400,
        "retryable": None,
        "error_class": None,
    }


def test_raw_retry_console_suppresses_non_retryable_400_when_flag_off(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    assert _openai_raw_retry_console_should_emit(_not_retryable_400()) is False
    assert _openai_raw_retry_console_should_emit(_denied_authorization()) is False


def test_raw_retry_console_emits_non_retryable_400_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    assert _openai_raw_retry_console_should_emit(_not_retryable_400()) is True
    assert _openai_raw_retry_console_should_emit(_denied_authorization()) is True


def test_raw_retry_console_emits_retryable_capacity_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    assert (
        _openai_raw_retry_console_should_emit(
            {
                "event_type": "openai_raw_http_classification",
                "retryable": True,
                "error_class": "server_overloaded",
                "classification_status_code": 503,
            }
        )
        is True
    )
    assert (
        _openai_raw_retry_console_should_emit(
            {
                "event_type": "openai_capacity_retry_authorization",
                "authorization_result": "authorized",
                "observed_http_status_code": 429,
            }
        )
        is True
    )
    assert (
        _openai_raw_retry_console_should_emit(
            {
                "event_type": "openai_raw_http_classification",
                "classification_status_code": 429,
                "retryable": None,
            }
        )
        is True
    )


def test_record_openai_raw_retry_event_stays_quiet_for_non_retryable_400(
    monkeypatch, caplog
) -> None:
    monkeypatch.delenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", raising=False)
    caplog.set_level("INFO", logger="LiteLLM Proxy")
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.verbose_proxy_logger.info"
    ) as info:
        _record_openai_raw_retry_event(
            event_type="openai_raw_http_classification",
            event_stage="authorization_classification",
            request=None,
            kwargs=None,
            custom_llm_provider="openai",
            litellm_call_id="call-400",
            failure=None,
            classification_status_code=400,
            classification_reason="status_not_retryable",
            diagnostic_enabled=True,
        )
    info.assert_not_called()


def test_record_openai_raw_retry_event_logs_when_healthy_flag_on(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "1")
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.verbose_proxy_logger.info"
    ) as info:
        _record_openai_raw_retry_event(
            event_type="openai_raw_http_classification",
            event_stage="authorization_classification",
            request=None,
            kwargs=None,
            custom_llm_provider="openai",
            litellm_call_id="call-400-debug",
            failure=None,
            classification_status_code=400,
            classification_reason="status_not_retryable",
            diagnostic_enabled=True,
        )
    info.assert_called_once()
    assert info.call_args.args[0] == "AAWM_OPENAI_RAW_RETRY: %s"
    assert "status_not_retryable" in info.call_args.args[1]
