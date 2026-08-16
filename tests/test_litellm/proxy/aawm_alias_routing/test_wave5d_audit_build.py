"""Wave 5D audit_build.py module-local tests.

Covers: exact event shape, skipped/attempt ordering, numeric and token
normalization, cooldown-key fallback, terminal activity inclusion, in-flight
exception detection, continuation-state recursion, and input immutability.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
    _build_auto_agent_alias_audit_event,
    _build_auto_agent_alias_audit_events,
    _codex_auto_agent_request_has_continuation_state,
    _is_auto_agent_alias_in_flight_cooldown_http_exception,
    configure_audit_build_runtime,
)


# ---------------------------------------------------------------------------
# Fake seam implementations
# ---------------------------------------------------------------------------


def _fake_format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fake_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fake_cooldown_until(cooldown_seconds: Optional[float]) -> Optional[str]:
    if cooldown_seconds is None:
        return None
    return "2026-01-01T00:00:00Z"


def _fake_get_request_context(request: Any, request_body: dict) -> dict[str, Any]:
    return {
        "session_id": "sess-123",
        "repository": "test/repo",
        "client_product_label": "Codex/1.0",
        "host_attribution": {
            "client_ip": "10.0.0.1",
            "client_ip_source": "x-forwarded-for",
            "host_name": "testhost",
            "host_name_source": "env",
        },
        "rollup_group_header_label": None,
    }


def _fake_extract_metadata_value(request_body: dict, *keys: str) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        val = metadata.get(key)
        if val:
            return str(val)
    return None


def _fake_extract_incoming_endpoint(request: Any) -> str:
    return "/v1/responses"


def _fake_resolve_outgoing_target(
    *, route_family: Optional[str] = None, target_url: Any = None
) -> Optional[str]:
    return route_family


_terminal_attach_calls: list[dict[str, Any]] = []


def _fake_attach_terminal_context_fields(
    event: dict[str, Any],
    *,
    request: Any = None,
    request_body: Any = None,
    selection: Any = None,
    candidate: Any = None,
    include_activity_status: bool = False,
) -> None:
    _terminal_attach_calls.append(
        {
            "event": event,
            "include_activity_status": include_activity_status,
        }
    )
    event["terminal_attached"] = True


@pytest.fixture(autouse=True)
def _configure_seams():
    """Configure audit_build seams before each test and restore after."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_build as _audit_build_mod

    # Snapshot every prior injected runtime callback so we can restore the
    # exact objects the god-module established at import time.
    _seam_names = (
        "_get_auto_agent_alias_request_context",
        "_attach_auto_agent_alias_terminal_context_fields",
        "_format_auto_agent_alias_timestamp",
        "_extract_auto_agent_alias_metadata_value",
        "_extract_auto_agent_alias_incoming_endpoint",
        "_resolve_auto_agent_alias_route_rollup_outgoing_target",
        "_auto_agent_alias_int",
        "_auto_agent_alias_cooldown_until",
    )
    _previous = {name: getattr(_audit_build_mod, name) for name in _seam_names}

    _terminal_attach_calls.clear()
    configure_audit_build_runtime(
        get_request_context=_fake_get_request_context,
        attach_terminal_context_fields=_fake_attach_terminal_context_fields,
        format_timestamp=_fake_format_timestamp,
        extract_metadata_value=_fake_extract_metadata_value,
        extract_incoming_endpoint=_fake_extract_incoming_endpoint,
        resolve_outgoing_target=_fake_resolve_outgoing_target,
        to_int=_fake_to_int,
        cooldown_until=_fake_cooldown_until,
    )
    yield
    # Restore the exact prior runtime objects, even if the test failed.
    for name, value in _previous.items():
        setattr(_audit_build_mod, name, value)


def _make_request() -> MagicMock:
    req = MagicMock()
    req.url = "http://localhost:4000/v1/responses"
    return req


def _minimal_candidate(**overrides: Any) -> dict[str, Any]:
    base = {
        "provider": "openai",
        "model": "gpt-4.1",
        "route_family": "codex_openai_responses",
        "target_url": "https://api.openai.com/v1/responses",
    }
    base.update(overrides)
    return base


def _minimal_selection(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "lane_key": "openai:gpt-4.1:codex",
        "session_key": "sk-1",
        "selection_reason": "first_available",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _is_auto_agent_alias_in_flight_cooldown_http_exception
# ---------------------------------------------------------------------------


class TestInFlightCooldownException:
    def test_redispatch_required_flag(self):
        exc = HTTPException(status_code=429, detail={"redispatch_required": True})
        assert _is_auto_agent_alias_in_flight_cooldown_http_exception(exc) is True

    @pytest.mark.parametrize(
        "code",
        [
            "aawm_codex_auto_agent_in_flight_provider_cooling_down",
            "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
            "aawm_codex_auto_agent_redispatch_required",
            "aawm_anthropic_auto_agent_redispatch_required",
        ],
    )
    def test_known_error_codes(self, code: str):
        exc = HTTPException(
            status_code=429, detail={"error": {"code": code}}
        )
        assert _is_auto_agent_alias_in_flight_cooldown_http_exception(exc) is True

    def test_unrelated_code_returns_false(self):
        exc = HTTPException(
            status_code=429, detail={"error": {"code": "rate_limited"}}
        )
        assert _is_auto_agent_alias_in_flight_cooldown_http_exception(exc) is False

    def test_non_dict_detail_returns_false(self):
        exc = HTTPException(status_code=500, detail="boom")
        assert _is_auto_agent_alias_in_flight_cooldown_http_exception(exc) is False

    def test_empty_detail_returns_false(self):
        exc = HTTPException(status_code=429, detail={})
        assert _is_auto_agent_alias_in_flight_cooldown_http_exception(exc) is False


# ---------------------------------------------------------------------------
# _build_auto_agent_alias_audit_event - exact shape
# ---------------------------------------------------------------------------


class TestBuildAuditEventShape:
    def test_minimal_event_keys(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        # None-valued keys are omitted
        assert "attempt_number" not in event
        assert "failure_class" not in event
        assert "error_status_code" not in event
        # Always-present keys
        assert event["alias_family"] == "codex"
        assert event["alias_model"] == "codex-auto"
        assert event["event_type"] == "candidate_selected"
        assert event["candidate_status"] == "selected"
        assert event["selected"] is False
        assert event["skipped"] is False
        assert event["redispatch_required"] is False
        assert event["redispatch_threshold_crossed"] is False
        assert event["last_resort"] is False
        assert event["in_flight_session"] is False
        assert event["terminal_attached"] is True

    def test_none_values_omitted(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
            error_type=None,
            failure_class=None,
        )
        assert "error_type" not in event
        assert "failure_class" not in event

    def test_numeric_normalization(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_retryable_failure",
            candidate_status="cooldown_set",
            cooldown_seconds="12.3456789",
            error_status_code="429",
            retry_after_seconds="5.5",
        )
        assert event["cooldown_seconds"] == 12.346
        assert event["error_status_code"] == 429
        assert event["retry_after_seconds"] == 5.5
        assert event["cooldown_until"] == "2026-01-01T00:00:00Z"

    def test_error_tokens_list_preserved(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_retryable_failure",
            candidate_status="cooldown_set",
            error_tokens=["tok_b", "tok_a"],
        )
        assert event["error_tokens"] == ["tok_b", "tok_a"]

    def test_error_tokens_set_sorted(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_retryable_failure",
            candidate_status="cooldown_set",
            error_tokens={"z_token", "a_token"},
        )
        assert event["error_tokens"] == ["a_token", "z_token"]

    def test_error_code_stringified(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_retryable_failure",
            candidate_status="cooldown_set",
            error_code=12345,
        )
        assert event["error_code"] == "12345"

    def test_cooldown_state_source_from_candidate(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(cooldown_state_source="memory"),
            candidate=_minimal_candidate(cooldown_state_source="redis"),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert event["cooldown_state_source"] == "redis"

    def test_cooldown_state_source_fallback_to_selection(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(cooldown_state_source="memory"),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert event["cooldown_state_source"] == "memory"

    def test_reasoning_effort_fields_propagated(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(
                reasoning_effort_requested="high",
                reasoning_effort_source="metadata",
            ),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert event["reasoning_effort_requested"] == "high"
        assert event["reasoning_effort_source"] == "metadata"

    def test_sota_xai_requested_native_xhigh_propagated(self):
        """XAI-008: audit propagation retains requested/native xhigh.

        For the managed sota-xai route, the attempt record's requested and
        native effort values must survive audit construction unclamped and
        name the native field ``reasoning.effort``.
        """
        selection = _minimal_selection(
            lane_key="xai:oa_xai/grok-4.6:production",
        )
        candidate = _minimal_candidate(
            provider="xai",
            model="oa_xai/grok-4.6",
            route_family="codex_xai_oauth_responses_adapter",
            reasoning_effort_requested="xhigh",
            reasoning_effort_native_value="xhigh",
            reasoning_effort_native_field="reasoning.effort",
            reasoning_effort_native_provider="xai",
        )
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="sota-xai",
            request=_make_request(),
            request_body={},
            selection=selection,
            candidate=candidate,
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert event["reasoning_effort_requested"] == "xhigh"
        assert event["reasoning_effort_native_value"] == "xhigh"
        assert event["reasoning_effort_native_field"] == "reasoning.effort"
        assert event["reasoning_effort_native_provider"] == "xai"
        assert event["model"] == "oa_xai/grok-4.6"
        assert event["provider"] == "xai"
        assert event["lane_key"] == "xai:oa_xai/grok-4.6:production"
        assert candidate["model"] == "oa_xai/grok-4.6"
        assert selection["lane_key"] == "xai:oa_xai/grok-4.6:production"


# ---------------------------------------------------------------------------
# Cooldown-key fallback
# ---------------------------------------------------------------------------


class TestCooldownKeyFallback:
    def test_explicit_cooldown_key_used(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
            cooldown_key="explicit:key",
        )
        assert event["cooldown_key"] == "explicit:key"

    def test_cooldown_key_derived_from_lane_key(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(lane_key="openai:gpt-4.1:codex"),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        # Derived key: provider:model:lane_key
        assert event["cooldown_key"] == "openai:gpt-4.1:openai:gpt-4.1:codex"

    def test_cooldown_key_with_epoch_tag(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(lane_key="lane1"),
            candidate=_minimal_candidate(config_epoch_tag="abc"),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert event["cooldown_key"] == "habc:openai:gpt-4.1:lane1"

    def test_lane_key_fallback_to_selection(self):
        event = _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(lane_key="sel-lane"),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
            lane_key=None,
        )
        assert event["lane_key"] == "sel-lane"


# ---------------------------------------------------------------------------
# Terminal activity inclusion
# ---------------------------------------------------------------------------


class TestTerminalActivityInclusion:
    @pytest.mark.parametrize(
        "event_type,expected",
        [
            ("no_candidate_available", True),
            ("redispatch_required", True),
            ("candidate_retryable_failure", True),
            ("candidate_selected", False),
            ("candidate_skipped_cooldown", False),
        ],
    )
    def test_include_activity_status_by_event_type(
        self, event_type: str, expected: bool
    ):
        _terminal_attach_calls.clear()
        _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type=event_type,
            candidate_status="test",
        )
        assert _terminal_attach_calls[-1]["include_activity_status"] is expected

    def test_include_activity_status_on_429(self):
        _terminal_attach_calls.clear()
        _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="test",
            error_status_code=429,
        )
        assert _terminal_attach_calls[-1]["include_activity_status"] is True

    def test_include_activity_status_on_redispatch_flag(self):
        _terminal_attach_calls.clear()
        _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="test",
            redispatch_required=True,
        )
        assert _terminal_attach_calls[-1]["include_activity_status"] is True


# ---------------------------------------------------------------------------
# _build_auto_agent_alias_audit_events - ordering and classification
# ---------------------------------------------------------------------------


class TestBuildAuditEvents:
    def test_skipped_before_attempts(self):
        selection = _minimal_selection(
            skipped=[
                {"provider": "xai", "model": "grok-3", "route_family": "xai", "reason": "cooldown", "lane_key": "xai:grok-3:codex"},
            ]
        )
        attempts = [
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "selected"},
        ]
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=attempts,
        )
        assert len(events) == 2
        assert events[0]["event_type"] == "candidate_skipped_cooldown"
        assert events[0]["skipped"] is True
        assert events[0]["selected"] is False
        assert events[1]["event_type"] == "candidate_selected"
        assert events[1]["selected"] is True
        assert events[1]["attempt_number"] == 1

    def test_auth_degraded_skipped_event_type(self):
        selection = _minimal_selection(
            skipped=[
                {"provider": "anthropic", "model": "claude-4", "route_family": "anthropic", "reason": "auth_degraded", "lane_key": "anth:claude-4:codex"},
            ]
        )
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=[],
        )
        assert events[0]["event_type"] == "candidate_skipped_provider_degraded"
        assert events[0]["candidate_status"] == "skipped_auth_degraded"

    def test_attempt_event_type_classification(self):
        attempts = [
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "selected"},
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "cooldown_set", "error_class": "rate_limit"},
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "terminal_in_flight_cooldown_set"},
        ]
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            attempts=attempts,
        )
        assert events[0]["event_type"] == "candidate_selected"
        assert events[1]["event_type"] == "candidate_retryable_failure"
        assert events[2]["event_type"] == "redispatch_required"
        assert events[2]["redispatch_required"] is True

    def test_fallback_to_selection_candidate(self):
        selection = _minimal_selection(
            candidate={
                "provider": "openai",
                "model": "gpt-4.1",
                "route_family": "openai",
            },
            selection_reason="affinity",
        )
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=[],
        )
        assert len(events) == 1
        assert events[0]["event_type"] == "candidate_selected"
        assert events[0]["selection_reason"] == "affinity"

    def test_cooldown_key_last_attempt_fallback(self):
        """RR-054 #51: last attempt falls back to selection cooldown_key."""
        attempts = [
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "selected", "cooldown_key": "attempt-key"},
        ]
        selection = _minimal_selection(cooldown_key="sel-key")
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=attempts,
        )
        # Single attempt is both first and last; last-attempt fallback applies
        assert events[0]["cooldown_key"] == "attempt-key"

    def test_cooldown_key_last_attempt_uses_selection_when_attempt_missing(self):
        attempts = [
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "selected"},
        ]
        selection = _minimal_selection(cooldown_key="sel-key")
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=attempts,
        )
        assert events[0]["cooldown_key"] == "sel-key"

    def test_non_dict_attempts_skipped(self):
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            attempts=["not-a-dict", 42],  # type: ignore[list-item]
        )
        assert events == []

    def test_non_dict_skipped_candidates_ignored(self):
        selection = _minimal_selection(skipped=["bad", 123])
        events = _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=[],
        )
        assert events == []


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


class TestInputImmutability:
    def test_selection_not_mutated(self):
        selection = _minimal_selection(
            skipped=[{"provider": "xai", "model": "grok-3", "route_family": "xai", "reason": "cooldown", "lane_key": "lk"}]
        )
        selection_copy = copy.deepcopy(selection)
        _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=selection,
            attempts=[{"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "selected"}],
        )
        assert selection == selection_copy

    def test_attempts_not_mutated(self):
        attempts = [
            {"provider": "openai", "model": "gpt-4.1", "route_family": "openai", "status": "cooldown_set", "error_class": "rate_limit"},
        ]
        attempts_copy = copy.deepcopy(attempts)
        _build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body={},
            selection=_minimal_selection(),
            attempts=attempts,
        )
        assert attempts == attempts_copy

    def test_request_body_not_mutated(self):
        body = {"litellm_metadata": {"session_id": "s1", "agent_id": "a1"}}
        body_copy = copy.deepcopy(body)
        _build_auto_agent_alias_audit_event(
            alias_family="codex",
            alias_model="codex-auto",
            request=_make_request(),
            request_body=body,
            selection=_minimal_selection(),
            candidate=_minimal_candidate(),
            event_type="candidate_selected",
            candidate_status="selected",
        )
        assert body == body_copy


# ---------------------------------------------------------------------------
# _codex_auto_agent_request_has_continuation_state
# ---------------------------------------------------------------------------


class TestContinuationState:
    def test_empty_dict(self):
        assert _codex_auto_agent_request_has_continuation_state({}) is False

    def test_scalar(self):
        assert _codex_auto_agent_request_has_continuation_state("hello") is False
        assert _codex_auto_agent_request_has_continuation_state(42) is False
        assert _codex_auto_agent_request_has_continuation_state(None) is False

    def test_previous_response_id(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"previous_response_id": "resp_123"}
        ) is True

    def test_call_id(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"call_id": "call_abc"}
        ) is True

    def test_tool_call_id(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"tool_call_id": "tc_1"}
        ) is True

    def test_item_id(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"item_id": "item_1"}
        ) is True

    @pytest.mark.parametrize(
        "item_type",
        [
            "function_call",
            "function_call_output",
            "mcp_call",
            "mcp_approval_request",
            "mcp_approval_response",
            "reasoning",
            "tool_use",
            "tool_result",
        ],
    )
    def test_type_field_triggers(self, item_type: str):
        assert _codex_auto_agent_request_has_continuation_state(
            {"type": item_type}
        ) is True

    def test_role_tool(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"role": "tool", "content": "result"}
        ) is True

    def test_tool_calls_present(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"role": "assistant", "tool_calls": [{"id": "tc1"}]}
        ) is True

    def test_nested_in_list(self):
        body = {"input": [{"type": "function_call", "name": "exec"}]}
        assert _codex_auto_agent_request_has_continuation_state(body) is True

    def test_deeply_nested(self):
        body = {"messages": [{"content": [{"nested": {"call_id": "deep"}}]}]}
        assert _codex_auto_agent_request_has_continuation_state(body) is True

    def test_cycle_safety(self):
        cyclic: dict[str, Any] = {"key": "value"}
        cyclic["self"] = cyclic
        assert _codex_auto_agent_request_has_continuation_state(cyclic) is False

    def test_list_cycle_safety(self):
        cyclic_list: list[Any] = [1, 2]
        cyclic_list.append(cyclic_list)
        assert _codex_auto_agent_request_has_continuation_state(cyclic_list) is False

    def test_falsy_continuation_keys_ignored(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"previous_response_id": "", "call_id": None, "tool_call_id": 0, "item_id": False}
        ) is False

    def test_empty_tool_calls_ignored(self):
        assert _codex_auto_agent_request_has_continuation_state(
            {"role": "assistant", "tool_calls": []}
        ) is False

    def test_plain_conversation(self):
        body = {
            "model": "gpt-4.1",
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        assert _codex_auto_agent_request_has_continuation_state(body) is False
