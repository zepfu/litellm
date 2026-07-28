"""Wave 5C module-local tests for error_signals.py.

Covers representative token/classification behavior, Kimi safe metadata
allowlisting, cooldown scope/key behavior (including model-scoped structured
429 non-widening), header wait/cooldown duration, source summary/type-code
extraction, and native-Grok retry eligibility/budget.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import error_signals
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    _add_codex_auto_agent_text_error_tokens,
    _build_codex_auto_agent_native_grok_continuation_retry_metadata,
    _build_safe_kimi_code_selection_telemetry,
    _classify_codex_auto_agent_retryable_exhaustion,
    _classify_kimi_code_auto_agent_probe_failure,
    _codex_auto_agent_error_text,
    _extract_adapter_upstream_headers,
    _extract_codex_auto_agent_error_tokens,
    _extract_codex_auto_agent_error_type_and_code,
    _extract_embedded_json_payload_candidates,
    _get_adapter_header_value,
    _get_codex_auto_agent_candidate_cooldown_scope,
    _get_codex_auto_agent_cooldown_scope,
    _get_codex_auto_agent_cooldown_seconds,
    _get_codex_auto_agent_grok_account_quota_lane_cooldown_key,
    _get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds,
    _get_codex_auto_agent_native_grok_continuation_transient_max_attempts,
    _get_codex_auto_agent_source_error_summary,
    _get_kimi_code_managed_account_cooldown_key,
    _get_safe_kimi_code_probe_failure_metadata,
    _is_codex_auto_agent_durable_cooldown_error_class,
    _is_codex_auto_agent_grok_4_5_candidate,
    _is_codex_auto_agent_grok_account_quota_candidate,
    _is_codex_auto_agent_native_grok_4_5_candidate,
    _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible,
    _is_codex_auto_agent_retryable_exhaustion,
    _is_codex_auto_agent_spark_candidate,
    _is_codex_auto_agent_transient_internal_error_class,
    _is_codex_auto_agent_xai_candidate,
    _is_kimi_code_auto_agent_candidate,
    _iter_codex_auto_agent_error_blocks,
    _parse_codex_auto_agent_header_wait_seconds,
    _parse_json_payloads_from_text_candidates,
    _parse_rate_limit_reset_wait_seconds_from_headers,
    _parse_retry_after_seconds_from_headers,
    _plan_codex_auto_agent_native_grok_continuation_transient_retry,
    configure_error_signals_runtime,
)


# ---------------------------------------------------------------------------
# Test fixture: configure seams with lightweight stubs
# ---------------------------------------------------------------------------

_DURABLE_COOLDOWN_ERROR_CLASSES = frozenset(
    {
        "capacity_exhausted",
        "candidate_unavailable",
        "malformed_tool_call_text",
        "provider_terminal_error",
        "rate_limited",
        "usage_limit_reached",
        "upstream_overloaded",
        "upstream_timeout",
    }
)
_CAPACITY_ERROR_TOKENS = frozenset(
    {
        "HIGH_DEMAND",
        "MODEL_AT_CAPACITY",
        "MODEL_CAPACITY_EXHAUSTED",
        "MODEL_OVERLOADED",
        "UPSTREAM_BUSY",
    }
)
_RATE_LIMIT_ERROR_TOKENS = frozenset(
    {
        "429",
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT_EXCEEDED",
        "rate_limit_exceeded",
    }
)


class _FakeExc(Exception):
    """Minimal exception stub with configurable attributes."""

    def __init__(self, message: str = "", code: Any = None, detail: Any = None, **kwargs: Any):
        super().__init__(message)
        if message:
            self.message = message
        if code is not None:
            self.code = code
        if detail is not None:
            self.detail = detail
        for key, value in kwargs.items():
            setattr(self, key, value)


def _stub_handled_error_summary(exc: Any, *, status_code: Optional[int] = None) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc)


def _stub_grok_balance_exhausted(**kwargs: Any) -> bool:
    return getattr(kwargs.get("exc"), "_grok_balance_exhausted", False)


def _stub_grok_spending_limit(**kwargs: Any) -> bool:
    return getattr(kwargs.get("exc"), "_grok_spending_limit", False)


@pytest.fixture(autouse=True)
def _configure_seams(request):
    """Configure error_signals runtime seams for every test."""
    if request.node.name == "test_imported_host_namespace_liveness":
        yield
        return

    runtime_names = (
        "_get_passthrough_handled_http_error_summary",
        "_is_known_grok_build_usage_balance_exhausted_response",
        "_is_known_grok_personal_team_spending_limit_response",
        "_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES",
        "_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS",
        "_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS",
        "_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS",
        "_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS",
        "_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS",
    )
    previous_runtime = {
        name: getattr(error_signals, name)
        for name in runtime_names
    }
    previous_host_globals = error_signals._host_globals_ref
    missing = object()
    previous_host_runtime = (
        {
            name: previous_host_globals.get(name, missing)
            for name in runtime_names
        }
        if previous_host_globals is not None
        else {}
    )
    configure_error_signals_runtime(
        get_passthrough_handled_http_error_summary=_stub_handled_error_summary,
        is_known_grok_build_usage_balance_exhausted_response=_stub_grok_balance_exhausted,
        is_known_grok_personal_team_spending_limit_response=_stub_grok_spending_limit,
        durable_cooldown_error_classes=_DURABLE_COOLDOWN_ERROR_CLASSES,
        capacity_error_tokens=_CAPACITY_ERROR_TOKENS,
        rate_limit_error_tokens=_RATE_LIMIT_ERROR_TOKENS,
    )
    yield
    for name, value in previous_runtime.items():
        setattr(error_signals, name, value)
    if previous_host_globals is not None:
        for name, value in previous_host_runtime.items():
            if value is missing:
                previous_host_globals.pop(name, None)
            else:
                previous_host_globals[name] = value
    error_signals._host_globals_ref = previous_host_globals


def test_install_retains_host_globals_and_preserves_unrelated_names() -> None:
    previous_host_globals = error_signals._host_globals_ref
    status_sentinel = object()
    http_exception_sentinel = object()
    logger_sentinel = object()
    host_globals = {
        "status": status_sentinel,
        "HTTPException": http_exception_sentinel,
        "verbose_proxy_logger": logger_sentinel,
    }
    try:
        error_signals.install(host_globals)
        assert error_signals._host_globals_ref is host_globals
        assert host_globals["status"] is status_sentinel
        assert host_globals["HTTPException"] is http_exception_sentinel
        assert host_globals["verbose_proxy_logger"] is logger_sentinel
        for name in error_signals._HOST_FUNCTION_NAMES:
            assert host_globals[name] is getattr(error_signals, name)
    finally:
        error_signals._host_globals_ref = previous_host_globals


# ---------------------------------------------------------------------------
# Token extraction and classification
# ---------------------------------------------------------------------------


class TestErrorTokens:
    def test_add_text_tokens_high_demand(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "currently experiencing high demand")
        assert "HIGH_DEMAND" in tokens

    def test_add_text_tokens_rate_limit(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "too many requests")
        assert "429" in tokens
        assert "RATE_LIMIT_EXCEEDED" in tokens

    def test_add_text_tokens_usage_limit(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "weekly limit reached for this model")
        assert "usage_limit_reached" in tokens

    def test_add_text_tokens_grok_balance(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "grok build usage balance exhausted")
        assert "GROK_BUILD_USAGE_BALANCE_EXHAUSTED" in tokens

    def test_add_text_tokens_grok_spending_limit(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "personal-team-blocked:spending-limit")
        assert "GROK_PERSONAL_TEAM_SPENDING_LIMIT" in tokens

    def test_add_text_tokens_safety_policy(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(
            tokens,
            "permission-denied content violates usage guidelines safety_check_type_cyber",
        )
        assert "safety_policy_denied" in tokens

    def test_add_text_tokens_deepseek_mismatch(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(
            tokens,
            "insufficient tool messages following tool_calls message",
        )
        assert "DEEPSEEK_TOOL_MESSAGE_MISMATCH" in tokens

    def test_add_text_tokens_candidate_unavailable_grok_not_found(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "grok-4.5 model not found")
        assert "aawm_codex_auto_agent_candidate_unavailable" in tokens

    def test_add_text_tokens_no_false_positive(self):
        tokens: set[str] = set()
        _add_codex_auto_agent_text_error_tokens(tokens, "everything is fine")
        assert len(tokens) == 0

    def test_extract_tokens_from_exception_text(self):
        exc = _FakeExc(message="model is overloaded")
        tokens = _extract_codex_auto_agent_error_tokens(exc)
        assert "MODEL_OVERLOADED" in tokens

    def test_error_text_includes_message_and_str(self):
        exc = _FakeExc(message="hello", code=42)
        text = _codex_auto_agent_error_text(exc)
        assert "hello" in text
        assert "42" in text


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_classify_capacity_exhausted(self):
        exc = _FakeExc(message="model capacity exhausted")
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "capacity_exhausted"

    def test_classify_rate_limited_by_token(self):
        exc = _FakeExc(message="too many requests")
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "rate_limited"

    def test_classify_rate_limited_by_status_429(self):
        exc = _FakeExc(message="something")
        exc.status_code = 429
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "rate_limited"

    def test_classify_transient_by_status_502(self):
        exc = _FakeExc(message="bad gateway")
        exc.status_code = 502
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "upstream_transient_internal"

    def test_classify_timeout_504(self):
        exc = _FakeExc(message="gateway timeout")
        exc.status_code = 504
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "upstream_timeout"

    def test_classify_usage_limit(self):
        exc = _FakeExc(message="usage_limit_reached")
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "usage_limit_reached"

    def test_classify_none_for_unknown(self):
        exc = _FakeExc(message="all good")
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) is None

    def test_is_retryable_true(self):
        exc = _FakeExc(message="too many requests")
        assert _is_codex_auto_agent_retryable_exhaustion(exc) is True

    def test_is_retryable_false(self):
        exc = _FakeExc(message="all good")
        assert _is_codex_auto_agent_retryable_exhaustion(exc) is False

    def test_grok_quota_exhaustion_takes_priority(self):
        exc = _FakeExc(message="too many requests")
        exc._grok_balance_exhausted = True
        assert _classify_codex_auto_agent_retryable_exhaustion(exc) == "capacity_exhausted"


# ---------------------------------------------------------------------------
# Kimi safe metadata allowlisting
# ---------------------------------------------------------------------------


class TestKimiMetadata:
    _VALID_METADATA = {
        "kind": "quota",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "reset_reason": "quota_exhausted",
        "status_code": 429,
        "trace_id": "abc-123",
    }

    def test_valid_metadata_passes(self):
        exc = _FakeExc(kimi_code_probe_failure_metadata=dict(self._VALID_METADATA))
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        result = _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate)
        assert result is not None
        assert result["kind"] == "quota"
        assert result["scope"] == "managed_account"

    def test_non_kimi_candidate_returns_none(self):
        exc = _FakeExc(kimi_code_probe_failure_metadata=dict(self._VALID_METADATA))
        candidate = {"provider": "xai", "model": "grok-4.5"}
        assert _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate) is None

    def test_invalid_kind_rejected(self):
        meta = dict(self._VALID_METADATA)
        meta["kind"] = "injection_attack"
        exc = _FakeExc(kimi_code_probe_failure_metadata=meta)
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        assert _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate) is None

    def test_invalid_status_code_rejected(self):
        meta = dict(self._VALID_METADATA)
        meta["status_code"] = 9999
        exc = _FakeExc(kimi_code_probe_failure_metadata=meta)
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        assert _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate) is None

    def test_bool_status_code_rejected(self):
        meta = dict(self._VALID_METADATA)
        meta["status_code"] = True
        exc = _FakeExc(kimi_code_probe_failure_metadata=meta)
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        assert _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate) is None

    def test_invalid_trace_id_rejected(self):
        meta = dict(self._VALID_METADATA)
        meta["trace_id"] = "bad trace id with spaces!"
        exc = _FakeExc(kimi_code_probe_failure_metadata=meta)
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        assert _get_safe_kimi_code_probe_failure_metadata(exc, candidate=candidate) is None

    def test_classify_managed_account(self):
        assert _classify_kimi_code_auto_agent_probe_failure({"scope": "managed_account"}) == "kimi_code_managed_account"

    def test_classify_candidate(self):
        assert _classify_kimi_code_auto_agent_probe_failure({"scope": "candidate"}) == "kimi_code_candidate_failure"

    def test_classify_none_scope(self):
        assert _classify_kimi_code_auto_agent_probe_failure({"scope": "none"}) == "kimi_code_no_cooldown"

    def test_classify_none_metadata(self):
        assert _classify_kimi_code_auto_agent_probe_failure(None) is None

    def test_build_telemetry(self):
        metadata = dict(self._VALID_METADATA)
        candidate = {"model": "kimi-k3"}
        result = _build_safe_kimi_code_selection_telemetry(
            alias_model="aawm-code",
            candidate=candidate,
            metadata=metadata,
        )
        assert result["alias"] == "aawm-code"
        assert result["candidate"] == "kimi-k3"
        assert result["upstream_id"] == "k3"

    def test_managed_account_cooldown_key(self):
        key = _get_kimi_code_managed_account_cooldown_key()
        assert "kimi_code" in key
        assert "__managed_account__" in key
        assert "kimi_code_managed_account" in key


# ---------------------------------------------------------------------------
# Cooldown scope and key behavior
# ---------------------------------------------------------------------------


class TestCooldownScope:
    def test_durable_class_gets_candidate_scope(self):
        assert _get_codex_auto_agent_cooldown_scope("rate_limited") == "candidate"

    def test_unknown_class_gets_request_local(self):
        assert _get_codex_auto_agent_cooldown_scope("unknown_thing") == "request_local"

    def test_none_class_gets_request_local(self):
        assert _get_codex_auto_agent_cooldown_scope(None) == "request_local"

    def test_kimi_managed_account_scope(self):
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        meta = {"scope": "managed_account"}
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "kimi_code_managed_account", candidate=candidate, kimi_failure_metadata=meta
        )
        assert scope == "managed_account"

    def test_kimi_candidate_scope(self):
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        meta = {"scope": "candidate"}
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "kimi_code_candidate_failure", candidate=candidate, kimi_failure_metadata=meta
        )
        assert scope == "candidate"

    def test_kimi_no_cooldown_scope(self):
        candidate = {"provider": "kimi_code", "model": "kimi-k3"}
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "kimi_code_no_cooldown", candidate=candidate
        )
        assert scope == "none"

    def test_native_grok_candidate_unavailable_gets_none(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
        }
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "candidate_unavailable", candidate=candidate
        )
        assert scope == "none"

    def test_non_native_xai_candidate_unavailable_gets_request_local(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_xai_oauth_responses_adapter",
        }
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "candidate_unavailable", candidate=candidate
        )
        assert scope == "request_local"

    def test_native_grok_transient_internal_gets_none(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
        }
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "upstream_transient_internal", candidate=candidate
        )
        assert scope == "none"

    def test_spark_transient_internal_gets_candidate(self):
        candidate = {"provider": "openai", "model": "gpt-5.3-codex-spark"}
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "upstream_transient_internal", candidate=candidate
        )
        assert scope == "candidate"

    def test_native_grok_malformed_tool_call_gets_request_local(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "anthropic_grok_native_responses_adapter",
        }
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "malformed_tool_call_text", candidate=candidate
        )
        assert scope == "request_local"

    def test_non_native_malformed_tool_call_gets_candidate(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_xai_oauth_responses_adapter",
        }
        scope = _get_codex_auto_agent_candidate_cooldown_scope(
            "malformed_tool_call_text", candidate=candidate
        )
        assert scope == "candidate"

    def test_model_scoped_structured_429_does_not_widen(self):
        """A structured 429 with capacity tokens still classifies as capacity_exhausted,
        not rate_limited, preserving non-widening behavior."""
        exc = _FakeExc(message="model capacity exhausted")
        exc.status_code = 429
        result = _classify_codex_auto_agent_retryable_exhaustion(exc)
        # capacity token takes priority over bare 429 status
        assert result == "capacity_exhausted"

    def test_grok_account_quota_lane_key(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
        }
        key = _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(candidate, "my-lane")
        assert key == "xai:__account_quota__:my-lane"

    def test_grok_account_quota_lane_key_none_for_non_quota_candidate(self):
        candidate = {"provider": "openai", "model": "gpt-5"}
        key = _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(candidate, "my-lane")
        assert key is None

    def test_grok_account_quota_lane_key_none_for_empty_lane(self):
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
        }
        key = _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(candidate, "")
        assert key is None


# ---------------------------------------------------------------------------
# Header wait / cooldown duration
# ---------------------------------------------------------------------------


class TestHeaderWaitAndCooldown:
    def test_retry_after_header_respected(self):
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"retry-after": "120"}
        wait = _parse_codex_auto_agent_header_wait_seconds(exc)
        assert wait == 120.0

    def test_retry_after_minimum_1s(self):
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"retry-after": "0.5"}
        wait = _parse_codex_auto_agent_header_wait_seconds(exc)
        assert wait == 1.0

    def test_reset_epoch_header(self):
        future_epoch = time.time() + 300
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"x-codex-primary-reset-at": str(future_epoch)}
        wait = _parse_codex_auto_agent_header_wait_seconds(exc)
        assert wait is not None
        assert 295 <= wait <= 305

    def test_millisecond_epoch_header(self):
        future_epoch_ms = (time.time() + 600) * 1000
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"x-ratelimit-reset": str(future_epoch_ms)}
        wait = _parse_codex_auto_agent_header_wait_seconds(exc)
        assert wait is not None
        assert 595 <= wait <= 605

    def test_no_headers_returns_none(self):
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {}
        wait = _parse_codex_auto_agent_header_wait_seconds(exc)
        assert wait is None

    def test_cooldown_seconds_capacity(self):
        exc = _FakeExc(message="model capacity exhausted")
        seconds = _get_codex_auto_agent_cooldown_seconds(exc)
        # Default capacity cooldown is 3h
        assert seconds == 3 * 60 * 60.0

    def test_cooldown_seconds_transient(self):
        exc = _FakeExc(message="bad gateway")
        exc.status_code = 502
        seconds = _get_codex_auto_agent_cooldown_seconds(exc)
        assert seconds == 30.0

    def test_cooldown_seconds_header_wait_overrides(self):
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"retry-after": "7200"}
        seconds = _get_codex_auto_agent_cooldown_seconds(exc)
        # header wait (7200) < default (10800), so default wins
        assert seconds == 3 * 60 * 60.0

    def test_cooldown_seconds_header_wait_larger(self):
        exc = _FakeExc(message="rate limited")
        exc.upstream_headers = {"retry-after": "14400"}
        seconds = _get_codex_auto_agent_cooldown_seconds(exc)
        assert seconds == 14400.0

    def test_spark_durable_cooldown(self):
        exc = _FakeExc(message="too many requests")
        candidate = {"provider": "openai", "model": "gpt-5.3-codex-spark"}
        seconds = _get_codex_auto_agent_cooldown_seconds(exc, candidate=candidate)
        assert seconds == 300.0

    def test_grok_quota_durable_cooldown(self):
        exc = _FakeExc(message="too many requests")
        exc._grok_balance_exhausted = True
        candidate = {
            "provider": "xai",
            "model": "grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
        }
        seconds = _get_codex_auto_agent_cooldown_seconds(exc, candidate=candidate)
        assert seconds == 3 * 60 * 60.0


# ---------------------------------------------------------------------------
# Source summary and type/code extraction
# ---------------------------------------------------------------------------


class TestSourceSummaryAndTypeCode:
    def test_source_summary_from_openrouter_raw(self):
        import json

        exc = _FakeExc(
            message="outer",
            detail={
                "error": {
                    "message": "Error from provider",
                    "metadata": {
                        "provider_name": "test",
                        "raw": json.dumps({"message": "inner provider error"}),
                    },
                }
            },
        )
        summary = _get_codex_auto_agent_source_error_summary(exc, status_code=502)
        assert "inner provider error" in summary

    def test_source_summary_fallback_to_exc(self):
        exc = _FakeExc(message="outer error")
        summary = _get_codex_auto_agent_source_error_summary(exc, status_code=500)
        assert "outer error" in summary

    def test_source_summary_nested_error_message(self):
        import json

        exc = _FakeExc(
            message="outer",
            detail={
                "error": {
                    "message": "Error from provider",
                    "metadata": {
                        "provider_name": "test",
                        "raw": json.dumps({"error": {"message": "nested msg"}}),
                    },
                }
            },
        )
        summary = _get_codex_auto_agent_source_error_summary(exc, status_code=502)
        assert "nested msg" in summary

    def test_error_type_and_code_from_blocks(self):
        exc = _FakeExc(
            detail={"error": {"type": "overloaded_error", "code": 529}},
        )
        error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
        assert error_type == "overloaded_error"
        assert error_code == 529

    def test_error_type_and_code_fallback(self):
        exc = _FakeExc(message="something")
        exc.type = "fallback_type"
        exc.code = 999
        error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
        assert error_type == "fallback_type"
        assert error_code == 999

    def test_iter_error_blocks_from_detail(self):
        exc = _FakeExc(detail={"error": {"code": 429, "message": "rate limited"}})
        blocks = _iter_codex_auto_agent_error_blocks(exc)
        assert len(blocks) == 1
        assert blocks[0]["code"] == 429


# ---------------------------------------------------------------------------
# Candidate predicates
# ---------------------------------------------------------------------------


class TestCandidatePredicates:
    def test_spark_candidate(self):
        assert _is_codex_auto_agent_spark_candidate({"model": "gpt-5.3-codex-spark"}) is True
        assert _is_codex_auto_agent_spark_candidate({"model": "gpt-5"}) is False
        assert _is_codex_auto_agent_spark_candidate(None) is False

    def test_grok_4_5_candidate(self):
        assert _is_codex_auto_agent_grok_4_5_candidate(
            {"provider": "xai", "model": "grok-4.5", "route_family": "codex_grok_native_responses_adapter"}
        ) is True
        assert _is_codex_auto_agent_grok_4_5_candidate(
            {"provider": "openai", "model": "grok-4.5", "route_family": "codex_grok_native_responses_adapter"}
        ) is False

    def test_native_grok_4_5_candidate(self):
        assert _is_codex_auto_agent_native_grok_4_5_candidate(
            {"provider": "xai", "model": "grok-4.5", "route_family": "codex_grok_native_responses_adapter"}
        ) is True
        assert _is_codex_auto_agent_native_grok_4_5_candidate(
            {"provider": "xai", "model": "grok-4.5", "route_family": "codex_xai_oauth_responses_adapter"}
        ) is False

    def test_xai_candidate(self):
        assert _is_codex_auto_agent_xai_candidate({"provider": "xai"}) is True
        assert _is_codex_auto_agent_xai_candidate({"provider": "openai"}) is False
        assert _is_codex_auto_agent_xai_candidate(None) is False

    def test_kimi_candidate(self):
        assert _is_kimi_code_auto_agent_candidate({"provider": "kimi_code"}) is True
        assert _is_kimi_code_auto_agent_candidate({"provider": "xai"}) is False

    def test_grok_account_quota_candidate(self):
        assert _is_codex_auto_agent_grok_account_quota_candidate(
            {"provider": "xai", "route_family": "codex_grok_native_responses_adapter"}
        ) is True
        assert _is_codex_auto_agent_grok_account_quota_candidate(
            {"provider": "xai", "route_family": "some_other_adapter"}
        ) is False

    def test_durable_cooldown_error_class(self):
        assert _is_codex_auto_agent_durable_cooldown_error_class("rate_limited") is True
        assert _is_codex_auto_agent_durable_cooldown_error_class("upstream_transient_internal") is False
        assert _is_codex_auto_agent_durable_cooldown_error_class(None) is False

    def test_transient_internal_error_class(self):
        assert _is_codex_auto_agent_transient_internal_error_class("upstream_transient_internal") is True
        assert _is_codex_auto_agent_transient_internal_error_class("rate_limited") is False


# ---------------------------------------------------------------------------
# Native Grok retry eligibility and budget
# ---------------------------------------------------------------------------


class TestNativeGrokRetry:
    def test_eligible_with_all_conditions(self):
        assert _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=True,
            error_class="upstream_transient_internal",
            cooldown_scope="none",
        ) is True

    def test_not_eligible_without_continuation_state(self):
        assert _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=False,
            error_class="upstream_transient_internal",
            cooldown_scope="none",
        ) is False

    def test_not_eligible_candidate_unavailable(self):
        assert _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=True,
            error_class="candidate_unavailable",
            cooldown_scope="none",
        ) is False

    def test_not_eligible_wrong_scope(self):
        assert _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=True,
            error_class="upstream_transient_internal",
            cooldown_scope="candidate",
        ) is False

    def test_max_attempts_default(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
        assert result == 8

    def test_max_attempts_env_override(self):
        with patch.dict(
            "os.environ",
            {"AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS": "10"},
        ):
            result = _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
        assert result == 10

    def test_max_attempts_clamped_low(self):
        with patch.dict(
            "os.environ",
            {"AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS": "2"},
        ):
            result = _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
        assert result == 6

    def test_max_attempts_clamped_high(self):
        with patch.dict(
            "os.environ",
            {"AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS": "100"},
        ):
            result = _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
        assert result == 16

    def test_backoff_seconds_bounded(self):
        for attempt in range(1, 20):
            seconds = _get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds(attempt)
            assert 0 <= seconds <= 1.0

    def test_plan_retry_scheduled(self):
        should_retry, backoff, metadata = _plan_codex_auto_agent_native_grok_continuation_transient_retry(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=True,
            error_class="upstream_transient_internal",
            cooldown_scope="none",
            provider_attempt=3,
            provider="xai",
            model="grok-4.5",
            route_family="codex_grok_native_responses_adapter",
            max_attempts=8,
        )
        assert should_retry is True
        assert backoff is not None
        assert backoff > 0
        assert metadata is not None
        assert metadata["status"] == "scheduled_same_candidate_retry"
        assert metadata["provider_attempt"] == 3
        assert metadata["max_attempts"] == 8
        assert "backoff_seconds" in metadata

    def test_plan_retry_exhausted(self):
        should_retry, backoff, metadata = _plan_codex_auto_agent_native_grok_continuation_transient_retry(
            is_native_grok_4_5_candidate=True,
            has_continuation_state=True,
            error_class="upstream_transient_internal",
            cooldown_scope="none",
            provider_attempt=8,
            provider="xai",
            model="grok-4.5",
            route_family="codex_grok_native_responses_adapter",
            max_attempts=8,
        )
        assert should_retry is False
        assert backoff is None
        assert metadata is not None
        assert metadata["status"] == "same_candidate_retry_exhausted"
        assert "backoff_seconds" not in metadata

    def test_plan_retry_ineligible(self):
        should_retry, backoff, metadata = _plan_codex_auto_agent_native_grok_continuation_transient_retry(
            is_native_grok_4_5_candidate=False,
            has_continuation_state=True,
            error_class="upstream_transient_internal",
            cooldown_scope="none",
            provider_attempt=1,
            provider="xai",
            model="grok-4.5",
            route_family="codex_grok_native_responses_adapter",
        )
        assert should_retry is False
        assert backoff is None
        assert metadata is None

    def test_build_retry_metadata_with_backoff(self):
        meta = _build_codex_auto_agent_native_grok_continuation_retry_metadata(
            status="scheduled_same_candidate_retry",
            provider_attempt=2,
            max_attempts=8,
            provider="xai",
            model="grok-4.5",
            route_family="codex_grok_native_responses_adapter",
            backoff_seconds=0.123,
        )
        assert meta["backoff_seconds"] == 0.123
        assert meta["provider_attempt"] == 2

    def test_build_retry_metadata_without_backoff(self):
        meta = _build_codex_auto_agent_native_grok_continuation_retry_metadata(
            status="same_candidate_retry_exhausted",
            provider_attempt=8,
            max_attempts=8,
            provider="xai",
            model="grok-4.5",
            route_family="codex_grok_native_responses_adapter",
        )
        assert "backoff_seconds" not in meta


# ---------------------------------------------------------------------------
# Owner-local helper parity (D1-591 six helpers)
# ---------------------------------------------------------------------------


class TestExtractAdapterUpstreamHeaders:
    def test_upstream_headers_dict_preferred(self):
        exc = _FakeExc()
        exc.upstream_headers = {"X-Foo": "bar", "X-Skip": None}
        result = _extract_adapter_upstream_headers(exc)
        assert result == {"X-Foo": "bar"}
        assert "X-Skip" not in result

    def test_falls_back_to_response_headers(self):
        class _Resp:
            headers = {"Content-Type": "application/json"}
        exc = _FakeExc()
        exc.response = _Resp()
        result = _extract_adapter_upstream_headers(exc)
        assert result == {"Content-Type": "application/json"}

    def test_no_headers_returns_empty(self):
        exc = _FakeExc()
        assert _extract_adapter_upstream_headers(exc) == {}

    def test_non_dict_upstream_headers_ignored(self):
        exc = _FakeExc()
        exc.upstream_headers = "not-a-dict"
        assert _extract_adapter_upstream_headers(exc) == {}


class TestGetAdapterHeaderValue:
    def test_case_insensitive_lookup(self):
        headers = {"Retry-After": "120"}
        assert _get_adapter_header_value(headers, "retry-after") == "120"

    def test_empty_headers_returns_none(self):
        assert _get_adapter_header_value({}, "X-Foo") is None

    def test_none_value_returns_none(self):
        assert _get_adapter_header_value({"X-Foo": None}, "X-Foo") is None

    def test_whitespace_only_returns_none(self):
        assert _get_adapter_header_value({"X-Foo": "   "}, "X-Foo") is None

    def test_strips_whitespace(self):
        assert _get_adapter_header_value({"X-Foo": "  bar  "}, "X-Foo") == "bar"

    def test_non_str_value_coerced(self):
        assert _get_adapter_header_value({"X-Foo": 42}, "X-Foo") == "42"

    def test_non_str_key_skipped(self):
        assert _get_adapter_header_value({123: "val"}, "123") is None

    def test_missing_key_returns_none(self):
        assert _get_adapter_header_value({"X-Other": "1"}, "X-Foo") is None


class TestParseRetryAfterSeconds:
    def test_valid_integer(self):
        assert _parse_retry_after_seconds_from_headers({"Retry-After": "60"}) == 60.0

    def test_negative_clamped_to_zero(self):
        assert _parse_retry_after_seconds_from_headers({"Retry-After": "-5"}) == 0.0

    def test_missing_header_returns_none(self):
        assert _parse_retry_after_seconds_from_headers({}) is None

    def test_non_numeric_returns_none(self):
        assert _parse_retry_after_seconds_from_headers({"Retry-After": "not-a-number"}) is None


class TestParseRateLimitResetWaitSeconds:
    def test_epoch_seconds_future(self):
        future = time.time() + 300
        result = _parse_rate_limit_reset_wait_seconds_from_headers(
            {"X-RateLimit-Reset": str(future)}
        )
        assert result is not None
        assert 295 <= result <= 305

    def test_millisecond_epoch(self):
        future_ms = (time.time() + 600) * 1000
        result = _parse_rate_limit_reset_wait_seconds_from_headers(
            {"X-RateLimit-Reset": str(future_ms)}
        )
        assert result is not None
        assert 595 <= result <= 605

    def test_past_epoch_clamped_to_zero(self):
        past = time.time() - 100
        result = _parse_rate_limit_reset_wait_seconds_from_headers(
            {"X-RateLimit-Reset": str(past)}
        )
        assert result == 0.0

    def test_missing_header_returns_none(self):
        assert _parse_rate_limit_reset_wait_seconds_from_headers({}) is None

    def test_non_numeric_returns_none(self):
        assert _parse_rate_limit_reset_wait_seconds_from_headers(
            {"X-RateLimit-Reset": "garbage"}
        ) is None


class TestExtractEmbeddedJsonPayloadCandidates:
    def test_dict_input(self):
        result = _extract_embedded_json_payload_candidates({"key": "val"})
        assert len(result) == 1
        import json as _json
        assert _json.loads(result[0]) == {"key": "val"}

    def test_bytes_input(self):
        result = _extract_embedded_json_payload_candidates(b'{"a": 1}')
        assert '{"a": 1}' in result

    def test_brace_extraction(self):
        result = _extract_embedded_json_payload_candidates('prefix {"x": 1} suffix')
        assert '{"x": 1}' in result

    def test_bracket_extraction(self):
        result = _extract_embedded_json_payload_candidates('prefix [1, 2] suffix')
        assert '[1, 2]' in result

    def test_none_input(self):
        result = _extract_embedded_json_payload_candidates(None)
        assert result == [""]

    def test_bytes_literal_wrapper(self):
        result = _extract_embedded_json_payload_candidates("detail: b'{\"err\": true}'")
        assert any('{"err": true}' in c for c in result)


class TestParseJsonPayloadsFromTextCandidates:
    def test_valid_json_parsed(self):
        result = _parse_json_payloads_from_text_candidates(['{"a": 1}', "[2, 3]"])
        assert result == [{"a": 1}, [2, 3]]

    def test_invalid_json_skipped(self):
        result = _parse_json_payloads_from_text_candidates(["not json", '{"ok": true}'])
        assert result == [{"ok": True}]

    def test_empty_list(self):
        assert _parse_json_payloads_from_text_candidates([]) == []


class TestOwnerHelperLiveness:
    """Verify production-order configuration and owner helper publication."""

    def test_fresh_process_openrouter_callbacks_do_not_load_eager_runtime(self):
        script = """
import sys

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import error_signals

host_module = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
)
adapter_module = (
    "litellm.llms.anthropic.experimental_pass_through.providers."
    "openrouter.adapter"
)
retry_module = (
    "litellm.llms.anthropic.experimental_pass_through.providers."
    "openrouter.retry_transport"
)
assert host_module not in sys.modules
assert adapter_module not in sys.modules
assert retry_module not in sys.modules

exc = Exception("outer")
exc.detail = {
    "error": {
        "message": "Error from provider",
        "metadata": {
            "provider_name": "test",
            "raw": "  ERROR  ",
        },
    }
}
assert error_signals._extract_openrouter_adapter_raw_message(exc) == "  ERROR  "
assert error_signals._is_openrouter_adapter_provider_raw_error(exc) is True
assert host_module not in sys.modules
assert adapter_module not in sys.modules
assert retry_module not in sys.modules
"""
        env = os.environ.copy()
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[4],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_install_publishes_owner_helpers(self):
        previous = error_signals._host_globals_ref
        host: dict = {}
        try:
            error_signals.install(host)
            for name in (
                "_extract_adapter_exception_detail",
                "_extract_adapter_error_payloads",
                "_extract_adapter_exception_status_code",
                "_extract_openrouter_adapter_raw_message",
                "_is_openrouter_adapter_provider_raw_error",
                "_extract_adapter_upstream_headers",
                "_get_adapter_header_value",
                "_parse_retry_after_seconds_from_headers",
                "_parse_rate_limit_reset_wait_seconds_from_headers",
                "_extract_embedded_json_payload_candidates",
                "_parse_json_payloads_from_text_candidates",
            ):
                assert name in host, f"{name} not published by install()"
                assert host[name] is getattr(error_signals, name)
        finally:
            error_signals._host_globals_ref = previous

    def test_owner_helper_monkeypatch_republishes(self, monkeypatch):
        host: dict[str, Any] = {}

        def patched(headers, name):
            return "patched"

        monkeypatch.setattr(error_signals, "_get_adapter_header_value", patched)
        error_signals.install(host)
        assert host["_get_adapter_header_value"] is patched

    def test_openrouter_error_shape_observes_post_configure_helper_monkeypatch(
        self,
        monkeypatch,
    ):
        payload = {
            "error": {
                "message": "Error from provider",
                "metadata": {
                    "provider_name": "patched",
                    "raw": "ERROR",
                },
            }
        }
        calls: list[tuple[str, object]] = []

        def patched_extract(detail):
            calls.append(("extract", detail))
            return ["patched-candidate"]

        def patched_parse(candidates):
            calls.append(("parse", candidates))
            return [payload]

        monkeypatch.setattr(
            error_signals,
            "_extract_embedded_json_payload_candidates",
            patched_extract,
        )
        monkeypatch.setattr(
            error_signals,
            "_parse_json_payloads_from_text_candidates",
            patched_parse,
        )
        monkeypatch.setattr(
            error_signals,
            "_extract_adapter_upstream_headers",
            lambda exc: {"Retry-After": "17"},
        )
        monkeypatch.setattr(
            error_signals,
            "_parse_retry_after_seconds_from_headers",
            lambda headers: 17.0,
        )
        monkeypatch.setattr(
            error_signals,
            "_get_adapter_header_value",
            lambda headers, name: "late-header",
        )
        monkeypatch.setattr(
            error_signals,
            "_parse_rate_limit_reset_wait_seconds_from_headers",
            lambda headers: 23.0,
        )

        exc = _FakeExc(message="outer")
        assert error_signals._extract_openrouter_adapter_raw_message(exc) == "ERROR"
        assert error_signals._is_openrouter_adapter_provider_raw_error(exc) is True
        runtime = error_signals._OPENROUTER_ERROR_SHAPE_RUNTIME
        assert runtime.extract_upstream_headers(exc) == {"Retry-After": "17"}
        assert runtime.parse_retry_after_seconds_from_headers({}) == 17.0
        assert runtime.get_header_value({}, "X-Test") == "late-header"
        assert runtime.parse_reset_wait_seconds_from_headers({}) == 23.0
        assert ("parse", ["patched-candidate"]) in calls

    def test_imported_host_namespace_liveness(self):
        """Import the real host and invoke every retained error-signals callback."""
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as host,
        )

        detail = {
            "error": {
                "message": "Error from provider",
                "metadata": {
                    "provider_name": "test",
                    "raw": "ERROR",
                },
            }
        }
        exc = _FakeExc(message="429 provider failure", detail=detail)
        exc.status_code = 429
        exc.upstream_headers = {"Retry-After": "10"}

        assert host._extract_adapter_exception_detail(exc) is detail
        assert detail in host._extract_adapter_error_payloads(exc)
        assert host._extract_adapter_exception_status_code(exc) == 429
        assert host._extract_openrouter_adapter_raw_message(exc) == "ERROR"
        assert host._is_openrouter_adapter_provider_raw_error(exc) is True
        assert host._extract_adapter_upstream_headers(exc) == {"Retry-After": "10"}
        assert host._parse_retry_after_seconds_from_headers(
            {"Retry-After": "10"}
        ) == 10.0
        assert host._parse_json_payloads_from_text_candidates(['{"ok": true}']) == [
            {"ok": True}
        ]
        assert (
            host._extract_adapter_exception_status_code
            is error_signals._extract_adapter_exception_status_code
        )

        attempt_status = (
            host._aawm_attempt_records._extract_exception_status_code(exc)
        )
        assert attempt_status == 429

        wave6b_runtime = host._wave6b_common_live_runtime()
        assert wave6b_runtime.extract_status_code(exc) == 429
        assert wave6b_runtime.extract_detail(exc) is detail

        summary = error_signals._get_passthrough_handled_http_error_summary(
            exc,
            status_code=429,
        )
        assert isinstance(summary, str)
        assert (
            error_signals._is_codex_auto_agent_grok_build_usage_balance_exhausted(
                _FakeExc()
            )
            is False
        )
        assert (
            error_signals._is_codex_auto_agent_grok_personal_team_spending_limit(
                _FakeExc()
            )
            is False
        )
