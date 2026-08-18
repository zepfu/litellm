"""Module-local tests for Wave 5C attempt_records.py extraction.

Drives the new module directly with fresh state/dependency stubs.
Does NOT import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import attempt_records


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_WAVE5C_AUDIT_EVENT_STUB_KEYS = {
    "codex_auto_agent_audit_events",
    "anthropic_auto_agent_audit_events",
    "aawm_alias_routing_audit_events",
}


def _serialized_recorded_alias_surfaces(
    meta: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> str:
    """Serialize emitted metadata/attempts, not the audit-event stub echo."""
    recorded_meta = {
        key: value
        for key, value in meta.items()
        if key not in _WAVE5C_AUDIT_EVENT_STUB_KEYS
    }
    # Incoming request metadata is preserved by merge; it is not a persist surface.
    recorded_meta.pop("authorization", None)
    return str({"meta": recorded_meta, "attempts": attempts})


def _make_request() -> Request:
    """Create a minimal Request with a fresh .state namespace."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


@dataclass
class _FakeClassificationEvent:
    origin: str = "upstream"


class _StubState:
    """Accumulates calls for assertion."""

    def __init__(self) -> None:
        self.emitted_events: list[Any] = []
        self.persisted_events: list[Any] = []
        self.parsed_bodies: list[Any] = []
        self.gate_records: list[dict[str, Any]] = []
        self.kimi_telemetry_calls: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _configure_attempt_records():  # noqa: PLR0915
    """Configure attempt_records runtime with fresh stubs before each test."""
    runtime_names = (
        "_extract_codex_auto_agent_error_tokens",
        "_extract_codex_auto_agent_error_type_and_code",
        "_parse_codex_auto_agent_header_wait_seconds",
        "_get_codex_auto_agent_source_error_summary",
        "_build_safe_kimi_code_selection_telemetry",
        "_extract_codex_auto_agent_usage_limit_raw_quota_resets",
        "_extract_exception_status_code",
        "_safe_set_request_parsed_body",
        "_emit_auto_agent_alias_route_event",
        "_build_auto_agent_alias_audit_event",
        "_build_auto_agent_alias_audit_events",
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        "_aawm_alias_route_verbose_json_enabled",
        "_aawm_alias_route_healthy_json_enabled",
        "_merge_litellm_metadata",
        "_normalize_low_cardinality_tag_value",
        "_load_bundled_model_cost_map_for_codex_policy",
        "_get_model_info",
        "_model_cost",
        "_openai_provider_value",
        "_classify_failure",
        "_codex_failure_evidence_gate_record",
    )
    previous_runtime = {
        name: getattr(attempt_records, name)
        for name in runtime_names
    }
    previous_host_globals = attempt_records._host_globals_ref
    missing = object()
    previous_host_runtime = (
        {
            name: previous_host_globals.get(name, missing)
            for name in runtime_names
        }
        if previous_host_globals is not None
        else {}
    )
    state = _StubState()

    def _extract_tokens(exc: Any) -> set[str]:
        return getattr(exc, "_tokens", {"rate_limit"})

    def _extract_type_and_code(exc: Any) -> tuple[Optional[str], Optional[str]]:
        return getattr(exc, "_type_code", (None, None))

    def _parse_wait(exc: Any) -> Optional[float]:
        return getattr(exc, "_retry_after", None)

    def _source_summary(exc: Any, *, status_code: Optional[int] = None) -> Optional[str]:
        return f"error:{status_code}"

    def _build_kimi_telemetry(*, alias_model: str, candidate: dict, metadata: dict) -> dict[str, Any]:
        state.kimi_telemetry_calls.append(
            {"alias_model": alias_model, "candidate": candidate, "metadata": metadata}
        )
        return {"kimi": "telemetry", "alias_model": alias_model}

    def _set_parsed_body(request: Request, body: Any) -> None:
        state.parsed_bodies.append(body)

    def _emit_event(event: Any, *, level: str = "info") -> None:
        state.emitted_events.append({"event": event, "level": level})

    def _build_audit_event(**kwargs: Any) -> dict[str, Any]:
        return {"audit_event": True, **kwargs}

    def _build_audit_events(**kwargs: Any) -> list[dict[str, Any]]:
        return [{"audit_events": True, **kwargs}]

    def _persist_audit(events: Any, *, request_body: Any = None) -> None:
        state.persisted_events.append({"events": events, "request_body": request_body})

    def _verbose_enabled() -> bool:
        return True

    def _healthy_enabled() -> bool:
        return False

    def _merge_metadata(body: dict, *, tags_to_add: list, extra_fields: dict) -> dict[str, Any]:
        result = dict(body)
        meta = dict(result.get("litellm_metadata") or {})
        existing_tags = meta.get("tags") or []
        meta["tags"] = list(existing_tags) + list(tags_to_add)
        meta.update(extra_fields)
        result["litellm_metadata"] = meta
        return result

    def _normalize_tag(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value).lower().strip() or None

    def _load_bundled_cost() -> dict[str, Any]:
        return {
            "oa_xai/grok-4.6": {
                "supports_xhigh_reasoning_effort": True,
                "supports_reasoning": True,
            },
            "xai/grok-4.6": {
                "supports_xhigh_reasoning_effort": True,
                "supports_reasoning": True,
            },
        }

    def _get_model_info(*, model: str, custom_llm_provider: str) -> dict[str, Any]:
        if model == "gpt-5":
            return {"supports_max_reasoning_effort": True, "supports_reasoning": True}
        if model == "o3":
            return {"supports_xhigh_reasoning_effort": True, "supports_reasoning": True}
        if model == "o1":
            return {"supports_reasoning": True, "supports_xhigh_reasoning_effort": False}
        raise Exception("model not found")

    def _classify_failure(**kwargs: Any) -> _FakeClassificationEvent:
        return _FakeClassificationEvent(origin="upstream")

    def _gate_record(
        *,
        canonical_alias: str,
        cooldown_key: str,
        event: Any,
    ) -> None:
        state.gate_records.append(
            {
                "canonical_alias": canonical_alias,
                "cooldown_key": cooldown_key,
                "event": event,
            }
        )

    attempt_records.configure_attempt_records_runtime(
        extract_error_tokens=_extract_tokens,
        extract_error_type_and_code=_extract_type_and_code,
        parse_header_wait_seconds=_parse_wait,
        get_source_error_summary=_source_summary,
        build_kimi_telemetry=_build_kimi_telemetry,
        extract_status_code=lambda exc: getattr(exc, '_status_code', None),
        safe_set_parsed_body=_set_parsed_body,
        emit_route_event=_emit_event,
        build_audit_event=_build_audit_event,
        build_audit_events=_build_audit_events,
        persist_audit_only_events=_persist_audit,
        verbose_json_enabled=_verbose_enabled,
        healthy_json_enabled=_healthy_enabled,
        merge_metadata=_merge_metadata,
        normalize_tag_value=_normalize_tag,
        load_bundled_model_cost=_load_bundled_cost,
        get_model_info=_get_model_info,
        model_cost={},
        openai_provider_value="openai",
        classify_failure=_classify_failure,
        codex_failure_evidence_gate_record=_gate_record,
    )
    yield state
    for name, value in previous_runtime.items():
        setattr(attempt_records, name, value)
    if previous_host_globals is not None:
        for name, value in previous_host_runtime.items():
            if value is missing:
                previous_host_globals.pop(name, None)
            else:
                previous_host_globals[name] = value
    attempt_records._host_globals_ref = previous_host_globals


def test_install_retains_host_globals_and_preserves_unrelated_names() -> None:
    previous_host_globals = attempt_records._host_globals_ref
    status_sentinel = object()
    http_exception_sentinel = object()
    logger_sentinel = object()
    host_globals = {
        "status": status_sentinel,
        "HTTPException": http_exception_sentinel,
        "verbose_proxy_logger": logger_sentinel,
    }
    try:
        attempt_records.install(host_globals)
        assert attempt_records._host_globals_ref is host_globals
        assert host_globals["status"] is status_sentinel
        assert host_globals["HTTPException"] is http_exception_sentinel
        assert host_globals["verbose_proxy_logger"] is logger_sentinel
        for name in attempt_records._HOST_FUNCTION_NAMES:
            assert host_globals[name] is getattr(attempt_records, name)
    finally:
        attempt_records._host_globals_ref = previous_host_globals


# ---------------------------------------------------------------------------
# _update_codex_auto_agent_retryable_attempt_record
# ---------------------------------------------------------------------------


class TestRetryableAttemptRecord:
    def test_mutates_record_in_place_with_cooldown(self) -> None:
        record: dict[str, Any] = {"status": "pending"}
        exc = Exception("rate limited")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._tokens = {"rate_limit", "overloaded"}  # type: ignore[attr-defined]
        exc._type_code = ("rate_limit_error", "RL001")  # type: ignore[attr-defined]
        exc._retry_after = 30.5  # type: ignore[attr-defined]

        tokens = attempt_records._update_codex_auto_agent_retryable_attempt_record(
            attempt_record=record,
            exc=exc,
            error_class="rate_limit",
            cooldown_seconds=60.0,
            alias_model="test-alias",
            cooldown_scope="candidate",
        )

        assert tokens == {"rate_limit", "overloaded"}
        assert record["status"] == "cooldown_set"
        assert record["error_class"] == "rate_limit"
        assert record["error_tokens"] == ["overloaded", "rate_limit"]
        assert record["cooldown_seconds"] == 60.0
        assert record["cooldown_scope"] == "candidate"
        assert record["error_status_code"] == 429
        assert record["error_type"] == "rate_limit_error"
        assert record["error_code"] == "RL001"
        assert record["retry_after_seconds"] == 30.5
        assert record["failure_phase"] == "provider_attempt"
        assert record["attempted_provider_call"] is True
        assert record["source_error"] == "error:429"

    def test_no_cooldown_scope_none(self) -> None:
        record: dict[str, Any] = {}
        exc = Exception("transient")
        exc._status_code = None  # type: ignore[attr-defined]
        exc._tokens = set()  # type: ignore[attr-defined]
        exc._type_code = (None, None)  # type: ignore[attr-defined]
        exc._retry_after = None  # type: ignore[attr-defined]

        attempt_records._update_codex_auto_agent_retryable_attempt_record(
            attempt_record=record,
            exc=exc,
            error_class="transient",
            cooldown_seconds=0.0,
            alias_model="test-alias",
            cooldown_scope="none",
        )

        assert record["status"] == "retryable_no_cooldown"
        assert "cooldown_seconds" not in record
        assert record["cooldown_scope"] == "none"

    def test_usage_limit_attaches_raw_quota_resets_while_cooldown_is_capped(self) -> None:
        record: dict[str, Any] = {}
        exc = Exception("usage limit reached")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._tokens = {"usage_limit_reached"}  # type: ignore[attr-defined]
        exc._type_code = (None, None)  # type: ignore[attr-defined]
        exc._retry_after = 200000.0  # type: ignore[attr-defined]
        exc.detail = {  # type: ignore[attr-defined]
            "error": {
                "code": "usage_limit_reached",
                "quota": {
                    "resets_in_seconds": 500000.0,
                    "resets_at": 900000.0,
                },
            }
        }

        cooldown_seconds = 10800.0
        assert cooldown_seconds == 10800.0

        attempt_records._update_codex_auto_agent_retryable_attempt_record(
            attempt_record=record,
            exc=exc,
            error_class="usage_limit_reached",
            cooldown_seconds=cooldown_seconds,
            cooldown_scope="candidate",
            alias_model="codex-auto-agent",
        )

        assert record["cooldown_seconds"] == 10800.0
        assert record["provider_resets_in_seconds"] == 500000.0
        assert record["provider_resets_at"] == 900000.0

    def test_kimi_telemetry_attached_when_all_present(self, _configure_attempt_records: _StubState) -> None:
        state = _configure_attempt_records
        record: dict[str, Any] = {}
        exc = Exception("kimi fail")
        exc._status_code = 500  # type: ignore[attr-defined]
        exc._tokens = {"internal"}  # type: ignore[attr-defined]
        exc._type_code = (None, None)  # type: ignore[attr-defined]
        exc._retry_after = None  # type: ignore[attr-defined]

        candidate = {"provider": "kimi", "model": "kimi-code"}
        attempt_records._update_codex_auto_agent_retryable_attempt_record(
            attempt_record=record,
            exc=exc,
            error_class="internal",
            cooldown_seconds=120.0,
            cooldown_scope="lane",
            alias_model="codex-auto-agent",
            candidate=candidate,
            kimi_failure_metadata={"error_type": "quota"},
        )

        assert record["kimi_code_failure"] == {"kimi": "telemetry", "alias_model": "codex-auto-agent"}
        assert len(state.kimi_telemetry_calls) == 1
        assert state.kimi_telemetry_calls[0]["alias_model"] == "codex-auto-agent"

    def test_kimi_telemetry_not_attached_when_metadata_none(self) -> None:
        record: dict[str, Any] = {}
        exc = Exception("no kimi")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._tokens = set()  # type: ignore[attr-defined]
        exc._type_code = (None, None)  # type: ignore[attr-defined]
        exc._retry_after = None  # type: ignore[attr-defined]

        attempt_records._update_codex_auto_agent_retryable_attempt_record(
            attempt_record=record,
            exc=exc,
            error_class="rate_limit",
            cooldown_seconds=10.0,
            cooldown_scope="candidate",
            alias_model="codex-auto-agent",
            candidate={"provider": "kimi", "model": "kimi-code"},
            kimi_failure_metadata=None,
        )

        assert "kimi_code_failure" not in record


# ---------------------------------------------------------------------------
# _record_auto_agent_alias_attempt_started
# ---------------------------------------------------------------------------


class TestAttemptStarted:
    def test_returns_candidate_body_and_sets_parsed(self, _configure_attempt_records: _StubState) -> None:
        state = _configure_attempt_records
        request = _make_request()
        audit_event = {"event_type": "candidate_started"}

        def _add_metadata(body: dict, *, request: Request, selection: dict, attempts: list) -> dict:
            return {
                **body,
                "litellm_metadata": {"aawm_alias_routing_audit_events": [audit_event]},
            }

        result = attempt_records._record_auto_agent_alias_attempt_started(
            alias_family="codex_auto_agent",
            alias_model="codex-auto-agent",
            request=request,
            prepared_request_body={"model": "gpt-5"},
            selection={"candidate": {"model": "gpt-5"}},
            attempts=[],
            attempt_record={},
            add_alias_metadata_fn=_add_metadata,
        )

        assert result["model"] == "gpt-5"
        assert len(state.parsed_bodies) == 1
        # verbose enabled -> emits latest audit event
        assert len(state.emitted_events) == 1
        assert state.emitted_events[0]["event"] == audit_event

    def test_no_emit_when_no_audit_events(self, _configure_attempt_records: _StubState) -> None:
        state = _configure_attempt_records
        request = _make_request()

        def _add_metadata(body: dict, *, request: Request, selection: dict, attempts: list) -> dict:
            return {**body, "litellm_metadata": {}}

        attempt_records._record_auto_agent_alias_attempt_started(
            alias_family="codex_auto_agent",
            alias_model="codex-auto-agent",
            request=request,
            prepared_request_body={"model": "gpt-5"},
            selection={},
            attempts=[],
            attempt_record={},
            add_alias_metadata_fn=_add_metadata,
        )

        assert len(state.emitted_events) == 0


# ---------------------------------------------------------------------------
# _record_codex_failure_evidence (exactly-once)
# ---------------------------------------------------------------------------


class TestCodexFailureEvidence:
    def test_records_policy_capped_cooldown_for_usage_limit(self) -> None:
        classify_calls: list[dict[str, Any]] = []

        def _capture_classify(**kwargs: Any) -> _FakeClassificationEvent:
            classify_calls.append(dict(kwargs))
            return _FakeClassificationEvent()

        attempt_records._classify_failure = _capture_classify

        record: dict[str, Any] = {}
        exc = Exception("usage limit reached")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._tokens = {"usage_limit_reached"}  # type: ignore[attr-defined]
        exc._retry_after = 422681.0  # type: ignore[attr-defined]
        cooldown_seconds = 10800.0

        attempt_records._record_codex_failure_evidence(
            canonical_alias="test-alias",
            cooldown_key="openai:gpt-5:openai",
            exc=exc,
            attempt_record=record,
            cooldown_seconds=cooldown_seconds,
        )

        assert cooldown_seconds == 10800.0
        assert classify_calls == [
            {
                "status_code": 429,
                "provider": None,
                "message": "error:429",
                "retry_after_seconds": 10800.0,
            }
        ]
        assert record["origin"] == "upstream"

    def test_records_raw_retry_after_for_non_usage_failure(self) -> None:
        classify_calls: list[dict[str, Any]] = []

        def _capture_classify(**kwargs: Any) -> _FakeClassificationEvent:
            classify_calls.append(dict(kwargs))
            return _FakeClassificationEvent()

        attempt_records._classify_failure = _capture_classify

        record: dict[str, Any] = {}
        exc = Exception("rate limited")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._tokens = {"rate_limited"}  # type: ignore[attr-defined]
        exc._retry_after = 5.0  # type: ignore[attr-defined]

        attempt_records._record_codex_failure_evidence(
            canonical_alias="test-alias",
            cooldown_key="openai:gpt-5:openai",
            exc=exc,
            attempt_record=record,
            cooldown_seconds=None,
        )

        assert classify_calls == [
            {
                "status_code": 429,
                "provider": None,
                "message": "error:429",
                "retry_after_seconds": 5.0,
            }
        ]
        assert record["origin"] == "upstream"

    def test_records_once_and_stamps_origin(self, _configure_attempt_records: _StubState) -> None:
        state = _configure_attempt_records
        record: dict[str, Any] = {}
        exc = Exception("429 upstream")
        exc._status_code = 429  # type: ignore[attr-defined]
        exc._retry_after = 5.0  # type: ignore[attr-defined]

        attempt_records._record_codex_failure_evidence(
            canonical_alias="test-alias",
            cooldown_key="openai:gpt-5:lane1",
            exc=exc,
            attempt_record=record,
        )

        assert record["origin"] == "upstream"
        assert len(state.gate_records) == 1
        assert state.gate_records[0]["canonical_alias"] == "test-alias"
        assert state.gate_records[0]["cooldown_key"] == "openai:gpt-5:lane1"

    def test_exactly_once_per_call(self, _configure_attempt_records: _StubState) -> None:
        """Calling twice records two separate events (once per call)."""
        state = _configure_attempt_records
        exc = Exception("fail")
        exc._status_code = 503  # type: ignore[attr-defined]
        exc._retry_after = None  # type: ignore[attr-defined]

        for _ in range(2):
            attempt_records._record_codex_failure_evidence(
                canonical_alias="test-alias",
                cooldown_key="key1",
                exc=exc,
                attempt_record={},
            )

        assert len(state.gate_records) == 2


# ---------------------------------------------------------------------------
# _record_auto_agent_alias_attempt_failure
# ---------------------------------------------------------------------------


class TestAttemptFailure:
    def test_emits_warning_and_no_persist_without_redispatch(
        self, _configure_attempt_records: _StubState
    ) -> None:
        state = _configure_attempt_records
        request = _make_request()

        def _add_metadata(body: dict, *, request: Request, selection: dict, attempts: list) -> dict:
            return {**body, "litellm_metadata": {"aawm_alias_routing_audit_events": [{"e": 1}]}}

        result = attempt_records._record_auto_agent_alias_attempt_failure(
            alias_family="codex_auto_agent",
            alias_model="codex-auto-agent",
            request=request,
            prepared_request_body={"model": "gpt-5"},
            selection={"lane_key": "lk1", "cooldown_key": "ck1"},
            attempts=[{"status": "cooldown_set"}],
            attempt_record={"status": "cooldown_set", "error_class": "rate_limit"},
            error_class="rate_limit",
            add_alias_metadata_fn=_add_metadata,
            redispatch_required=False,
        )

        assert result["model"] == "gpt-5"
        assert len(state.emitted_events) == 1
        assert state.emitted_events[0]["level"] == "warning"
        assert len(state.persisted_events) == 0

    def test_persists_on_redispatch(self, _configure_attempt_records: _StubState) -> None:
        state = _configure_attempt_records
        request = _make_request()

        def _add_metadata(body: dict, *, request: Request, selection: dict, attempts: list) -> dict:
            return {**body, "litellm_metadata": {"aawm_alias_routing_audit_events": [{"e": 1}]}}

        attempt_records._record_auto_agent_alias_attempt_failure(
            alias_family="codex_auto_agent",
            alias_model="codex-auto-agent",
            request=request,
            prepared_request_body={"model": "gpt-5"},
            selection={},
            attempts=[],
            attempt_record={"status": "cooldown_set"},
            error_class="capacity",
            add_alias_metadata_fn=_add_metadata,
            redispatch_required=True,
        )

        assert len(state.persisted_events) == 1

    def test_builds_fallback_audit_event_when_none_in_body(
        self, _configure_attempt_records: _StubState
    ) -> None:
        state = _configure_attempt_records
        request = _make_request()

        def _add_metadata(body: dict, *, request: Request, selection: dict, attempts: list) -> dict:
            return {**body, "litellm_metadata": {}}

        attempt_records._record_auto_agent_alias_attempt_failure(
            alias_family="anthropic_auto_agent",
            alias_model="anthropic-auto-agent",
            request=request,
            prepared_request_body={"model": "claude-4"},
            selection={"lane_key": "lk", "cooldown_key": "ck", "selection_reason": "rr"},
            attempts=[{"a": 1}],
            attempt_record={"status": "cooldown_set", "cooldown_seconds": 30.0},
            error_class="overloaded",
            add_alias_metadata_fn=_add_metadata,
            redispatch_required=False,
        )

        assert len(state.emitted_events) == 1
        emitted = state.emitted_events[0]["event"]
        assert emitted["audit_event"] is True
        assert emitted["event_type"] == "candidate_retryable_failure"


# ---------------------------------------------------------------------------
# Reasoning-effort extraction and normalization
# ---------------------------------------------------------------------------


class TestReasoningEffort:
    def test_extract_from_reasoning_dict(self) -> None:
        body = {"reasoning": {"effort": "high"}}
        effort, field = attempt_records._extract_codex_reasoning_effort(body)
        assert effort == "high"
        assert field == "reasoning.effort"

    def test_extract_from_flat_field(self) -> None:
        body = {"reasoning_effort": "medium"}
        effort, field = attempt_records._extract_codex_reasoning_effort(body)
        assert effort == "medium"
        assert field == "reasoning_effort"

    def test_extract_none_when_absent(self) -> None:
        effort, field = attempt_records._extract_codex_reasoning_effort({"model": "gpt-5"})
        assert effort is None
        assert field is None

    def test_ceiling_max_for_gpt5(self) -> None:
        route = {"provider": "openai", "route_family": "codex_responses", "model": "gpt-5"}
        assert attempt_records._get_codex_reasoning_effort_ceiling(route) == "max"

    def test_ceiling_xhigh_for_o3(self) -> None:
        route = {"provider": "openai", "route_family": "codex_responses", "model": "o3"}
        assert attempt_records._get_codex_reasoning_effort_ceiling(route) == "xhigh"

    def test_ceiling_high_for_o1(self) -> None:
        route = {"provider": "openai", "route_family": "codex_responses", "model": "o1"}
        assert attempt_records._get_codex_reasoning_effort_ceiling(route) == "high"

    def test_ceiling_none_for_non_openai(self) -> None:
        route = {"provider": "anthropic", "route_family": "codex_responses", "model": "gpt-5"}
        assert attempt_records._get_codex_reasoning_effort_ceiling(route) is None

    def test_ceiling_none_for_non_codex_family(self) -> None:
        route = {"provider": "openai", "route_family": "chat_completions", "model": "gpt-5"}
        assert attempt_records._get_codex_reasoning_effort_ceiling(route) is None

    def test_normalize_noop_when_no_effort(self) -> None:
        body = {"model": "gpt-5"}
        route = {"provider": "openai", "route_family": "codex_responses", "model": "gpt-5"}
        result_body, meta = attempt_records._normalize_codex_reasoning_effort_for_resolved_route(
            body, resolved_route=route
        )
        assert result_body is body
        assert meta == {}

    def test_normalize_noop_when_ceiling_none(self) -> None:
        body = {"model": "claude-4", "reasoning_effort": "high"}
        route = {"provider": "anthropic", "route_family": "messages", "model": "claude-4"}
        result_body, meta = attempt_records._normalize_codex_reasoning_effort_for_resolved_route(
            body, resolved_route=route
        )
        assert result_body is body
        assert meta == {}

    def test_normalize_clamps_max_to_high(self) -> None:
        body = {"model": "o1", "reasoning_effort": "max"}
        route = {"provider": "openai", "route_family": "codex_responses", "model": "o1"}
        result_body, meta = attempt_records._normalize_codex_reasoning_effort_for_resolved_route(
            body, resolved_route=route, attempt_number=2
        )

        assert result_body["reasoning_effort"] == "high"
        assert meta["codex_reasoning_effort"] == "high"
        assert meta["reasoning_effort_requested"] == "max"
        assert meta["reasoning_effort_clamped_from"] == "max"
        assert meta["reasoning_effort_candidate_attempt"] == 2
        assert meta["reasoning_effort_mapping_reason"] == "requested_effort_above_model_supported_ceiling"

    def test_normalize_within_ceiling_no_clamp(self) -> None:
        body = {"model": "gpt-5", "reasoning": {"effort": "high"}}
        route = {"provider": "openai", "route_family": "codex_responses", "model": "gpt-5"}
        result_body, meta = attempt_records._normalize_codex_reasoning_effort_for_resolved_route(
            body, resolved_route=route
        )

        assert meta["codex_reasoning_effort"] == "high"
        assert meta["reasoning_effort_mapping_reason"] == "within_supported_ceiling"
        assert "reasoning_effort_clamped_from" not in meta
        # reasoning dict unchanged
        assert result_body["reasoning"]["effort"] == "high"


class TestReasoningEffortSotaXai:
    def test_normalize_noop_for_sota_xai_managed_route_preserves_xhigh(self) -> None:
        """XAI-008: the managed sota-xai route never applies codex-openai clamping.

        The codex reasoning ceiling is gated on the openai provider and
        ``codex_responses`` route family, so a caller ``xhigh`` on the managed
        xAI Responses route passes through unchanged.
        """
        body = {"model": "oa_xai/grok-4.6", "reasoning": {"effort": "xhigh"}}
        route = {
            "provider": "xai",
            "route_family": "codex_xai_oauth_responses_adapter",
            "model": "oa_xai/grok-4.6",
        }
        result_body, meta = attempt_records._normalize_codex_reasoning_effort_for_resolved_route(
            body, resolved_route=route
        )
        assert result_body is body
        assert result_body["reasoning"] == {"effort": "xhigh"}
        assert meta == {}


# ---------------------------------------------------------------------------
# _add_codex_auto_agent_alias_metadata
# ---------------------------------------------------------------------------


class TestCodexAliasMetadata:
    def _selection(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "candidate": {
                "provider": "openai",
                "model": "gpt-5",
                "route_family": "codex_responses",
                "last_resort": False,
            },
            "lane_key": "openai:gpt-5",
            "cooldown_key": "cd:openai:gpt-5",
            "alias_model": "codex-auto-agent",
            "selection_reason": "round_robin",
            "skipped": [],
        }
        base.update(overrides)
        return base

    def test_composes_metadata_and_tags(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent"}
        selection = self._selection()

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["model"] == "gpt-5"
        meta = result["litellm_metadata"]
        assert "codex-auto-agent-alias" in meta["tags"]
        assert "codex-auto-agent-selected:gpt-5" in meta["tags"]
        assert meta["codex_auto_agent_selected_model"] == "gpt-5"
        assert meta["codex_auto_agent_alias"] == "codex-auto-agent"

    def test_applies_default_reasoning_effort(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent"}
        selection = self._selection()
        selection["candidate"]["default_reasoning_effort"] = "medium"

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["reasoning"] == {"effort": "medium"}
        meta = result["litellm_metadata"]
        assert "codex-auto-agent-default-effort:medium" in meta["tags"]
        assert meta["codex_auto_agent_default_reasoning_effort"] == "medium"

    def test_does_not_override_existing_reasoning_effort(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent", "reasoning_effort": "low"}
        selection = self._selection()
        selection["candidate"]["default_reasoning_effort"] = "medium"

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        # flat reasoning_effort present -> default not applied
        assert result.get("reasoning") is None or result["reasoning_effort"] == "low"

    def test_updates_last_attempt_with_reasoning_metadata(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent", "reasoning_effort": "high"}
        selection = self._selection()
        attempts: list[dict[str, Any]] = [{"status": "pending"}]

        attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=attempts
        )

        # reasoning metadata merged into last attempt
        assert "codex_reasoning_effort" in attempts[-1]

    def test_sota_xai_caller_xhigh_preserved_in_body_and_metadata(self) -> None:
        """XAI-008: caller xhigh on oa_xai/grok-4.6 survives onto the provider body.

        The managed sota-xai candidate carries no configured effort override
        (the YAML ``max`` override was removed), so the caller-selected
        ``xhigh`` must remain ``xhigh`` in the provider-bound body and the
        requested/native audit metadata must name the native field
        ``reasoning.effort``.
        """
        request = _make_request()
        body = {"model": "sota-xai", "reasoning": {"effort": "xhigh"}}
        selection = self._selection(
            candidate={
                "provider": "xai",
                "model": "oa_xai/grok-4.6",
                "route_family": "codex_xai_oauth_responses_adapter",
                "last_resort": False,
            },
            lane_key="xai:oa_xai/grok-4.6",
            cooldown_key="cd:xai:oa_xai/grok-4.6",
            alias_model="sota-xai",
        )

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        # Provider-bound body keeps caller xhigh under the native field.
        assert result["model"] == "oa_xai/grok-4.6"
        assert result["reasoning"] == {"effort": "xhigh"}
        assert "reasoning_effort" not in result

        # Caller body is not mutated (deep-copied per attempt).
        assert body["reasoning"] == {"effort": "xhigh"}

        meta = result["litellm_metadata"]
        assert meta["codex_auto_agent_selected_model"] == "oa_xai/grok-4.6"
        assert meta["codex_auto_agent_selected_provider"] == "xai"
        # No codex clamp metadata is produced for the non-codex managed route.
        assert "reasoning_effort_clamped_from" not in meta
        # Already-generated native xAI effort metadata is merged top-level.
        assert meta["reasoning_effort_requested"] == "xhigh"
        assert meta["reasoning_effort_source"] == "reasoning.effort"
        assert meta["reasoning_effort_native_provider"] == "xai"
        assert meta["reasoning_effort_native_value"] == "xhigh"
        assert meta["reasoning_effort_native_field"] == "reasoning.effort"
        assert meta["reasoning_effort_supported_ceiling"] == "xhigh"
        assert meta["reasoning_effort_resolved_model"] == "oa_xai/grok-4.6"
        assert meta["reasoning_effort_resolved_provider"] == "xai"
        assert meta["reasoning_effort_mapping_reason"] == "within_supported_ceiling"
        # Audit construction surfaces requested/native effort metadata.
        audit_events = meta["codex_auto_agent_audit_events"]
        assert audit_events
        event_candidate = audit_events[0]["selection"]["candidate"]
        assert event_candidate["reasoning_effort_requested"] == "xhigh"
        assert event_candidate["reasoning_effort_native_value"] == "xhigh"
        assert event_candidate["reasoning_effort_native_field"] == "reasoning.effort"
        assert event_candidate["reasoning_effort_native_provider"] == "xai"

    def test_sota_xai_caller_xhigh_reaches_attempt_and_top_level_metadata(self) -> None:
        """XAI-008: caller xhigh lands on the attempt record and top-level metadata."""
        request = _make_request()
        body = {"model": "sota-xai", "reasoning": {"effort": "xhigh"}}
        selection = self._selection(
            candidate={
                "provider": "xai",
                "model": "oa_xai/grok-4.6",
                "route_family": "codex_xai_oauth_responses_adapter",
                "last_resort": False,
            },
            lane_key="xai:oa_xai/grok-4.6",
            cooldown_key="cd:xai:oa_xai/grok-4.6",
            alias_model="sota-xai",
        )
        attempts: list[dict[str, Any]] = [{"status": "pending"}]

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=attempts
        )

        assert result["reasoning"] == {"effort": "xhigh"}
        assert attempts[-1]["reasoning_effort_requested"] == "xhigh"
        assert attempts[-1]["reasoning_effort_native_value"] == "xhigh"
        assert attempts[-1]["reasoning_effort_native_field"] == "reasoning.effort"
        assert attempts[-1]["reasoning_effort_native_provider"] == "xai"
        assert attempts[-1]["reasoning_effort_candidate_attempt"] == 1

        meta = result["litellm_metadata"]
        assert meta["reasoning_effort_requested"] == "xhigh"
        assert meta["reasoning_effort_native_value"] == "xhigh"
        assert meta["reasoning_effort_native_field"] == "reasoning.effort"
        assert meta["reasoning_effort_native_provider"] == "xai"
        assert meta["reasoning_effort_candidate_attempt"] == 1
        assert meta["codex_auto_agent_attempts"] is attempts

    def test_sota_xai_xhigh_metadata_survives_when_effective_map_lacks_managed_entry(
        self,
    ) -> None:
        """XAI-008: bundled catalog still supplies xhigh when the live map does not."""
        request = _make_request()
        body = {"model": "sota-xai", "reasoning": {"effort": "xhigh"}}
        selection = self._selection(
            candidate={
                "provider": "xai",
                "model": "oa_xai/grok-4.6",
                "route_family": "codex_xai_oauth_responses_adapter",
                "last_resort": False,
            },
            lane_key="xai:oa_xai/grok-4.6",
            cooldown_key="cd:xai:oa_xai/grok-4.6",
            alias_model="sota-xai",
        )
        attempts: list[dict[str, Any]] = [{"status": "pending"}]
        previous_supports_xhigh = attempt_records._supports_xhigh_reasoning_effort
        previous_model_cost = attempt_records._model_cost
        previous_get_model_info = attempt_records._get_model_info
        previous_bundled = attempt_records._load_bundled_model_cost_map_for_codex_policy

        def _effective_map_lacks_managed_entry(**_kwargs: Any) -> bool:
            return False

        def _missing_model_info(**_kwargs: Any) -> dict[str, Any]:
            raise Exception("managed entry missing from effective runtime map")

        def _repository_catalog() -> dict[str, Any]:
            return {
                "oa_xai/grok-4.6": {"supports_xhigh_reasoning_effort": True},
            }

        attempt_records._supports_xhigh_reasoning_effort = (
            _effective_map_lacks_managed_entry
        )
        attempt_records._model_cost = {"unrelated/model": {"supports_reasoning": True}}
        attempt_records._get_model_info = _missing_model_info
        attempt_records._load_bundled_model_cost_map_for_codex_policy = (
            _repository_catalog
        )
        try:
            result = attempt_records._add_codex_auto_agent_alias_metadata(
                body, request=request, selection=selection, attempts=attempts
            )
        finally:
            attempt_records._supports_xhigh_reasoning_effort = previous_supports_xhigh
            attempt_records._model_cost = previous_model_cost
            attempt_records._get_model_info = previous_get_model_info
            attempt_records._load_bundled_model_cost_map_for_codex_policy = (
                previous_bundled
            )

        assert result["reasoning"] == {"effort": "xhigh"}
        assert attempts[-1]["reasoning_effort_requested"] == "xhigh"
        assert attempts[-1]["reasoning_effort_native_value"] == "xhigh"
        assert attempts[-1]["reasoning_effort_native_field"] == "reasoning.effort"
        assert attempts[-1]["reasoning_effort_native_provider"] == "xai"
        assert attempts[-1]["reasoning_effort_supported_ceiling"] == "xhigh"

        meta = result["litellm_metadata"]
        assert meta["reasoning_effort_requested"] == "xhigh"
        assert meta["reasoning_effort_native_value"] == "xhigh"
        assert meta["reasoning_effort_native_field"] == "reasoning.effort"
        assert meta["reasoning_effort_native_provider"] == "xai"
        assert meta["reasoning_effort_supported_ceiling"] == "xhigh"

    def test_last_resort_tag(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent"}
        selection = self._selection()
        selection["candidate"]["last_resort"] = True

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert "codex-auto-agent-last-resort" in result["litellm_metadata"]["tags"]

    def test_configured_reasoning_effort_overrides_caller(self) -> None:
        """CFG-006: configured candidate reasoning_effort replaces caller reasoning."""
        request = _make_request()
        body = {
            "model": "codex-auto-agent",
            "reasoning": {"effort": "high", "summary": "auto"},
            "reasoning_effort": "high",
        }
        selection = self._selection()
        selection["candidate"]["reasoning_effort"] = "low"

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["reasoning"] == {"effort": "low", "summary": "auto"}
        assert "reasoning_effort" not in result
        # Caller body is not mutated (deep-copied per attempt).
        assert body["reasoning"] == {"effort": "high", "summary": "auto"}
        assert body["reasoning_effort"] == "high"
        meta = result["litellm_metadata"]
        assert "codex-auto-agent-config-effort:low" in meta["tags"]
        assert meta["codex_auto_agent_config_reasoning_effort"] == "low"

    def test_omitted_config_preserves_caller_reasoning(self) -> None:
        """CFG-006: no configured value leaves caller reasoning untouched."""
        request = _make_request()
        body = {"model": "codex-auto-agent", "reasoning": {"effort": "medium"}}
        selection = self._selection()

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["reasoning"] == {"effort": "medium"}
        meta = result["litellm_metadata"]
        assert not any(
            tag.startswith("codex-auto-agent-config-effort:") for tag in meta["tags"]
        )

    def test_failover_attempts_start_from_original_caller_reasoning(self) -> None:
        """CFG-006 regression: each attempt reshapes from the original body.

        A configured first candidate must not leak its provider-shaped
        reasoning into the next attempt; an unset second candidate receives
        the caller's original reasoning intent.
        """
        request = _make_request()
        original_body = {
            "model": "codex-auto-agent",
            "reasoning": {"effort": "high"},
        }

        first = self._selection()
        first["candidate"]["reasoning_effort"] = "low"
        first_body = attempt_records._add_codex_auto_agent_alias_metadata(
            original_body, request=request, selection=first, attempts=[]
        )
        assert first_body["reasoning"] == {"effort": "low"}

        second = self._selection(
            candidate={
                "provider": "openrouter",
                "model": "openrouter/cohere/north-mini-code:free",
                "route_family": "codex_openrouter_completion_adapter",
                "last_resort": False,
            }
        )
        second_body = attempt_records._add_codex_auto_agent_alias_metadata(
            original_body, request=request, selection=second, attempts=[{"status": "started"}]
        )

        # No leak from the prior attempt: caller intent is restored.
        assert second_body["reasoning"] == {"effort": "high"}
        assert original_body["reasoning"] == {"effort": "high"}

    def test_copies_fresh_redispatch_trace_onto_last_attempt(self) -> None:
        request = _make_request()
        body = {"model": "codex-auto-agent"}
        attempts: list[dict[str, Any]] = [{"status": "started"}]
        selection = self._selection(
            request_mode="fresh_redispatch",
            redispatch_ordinal=2,
            affinity_bypassed=True,
        )

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body,
            request=request,
            selection=selection,
            attempts=attempts,
        )

        assert attempts[-1]["request_mode"] == "fresh_redispatch"
        assert attempts[-1]["redispatch_ordinal"] == 2
        assert attempts[-1]["affinity_bypassed"] is True

        meta = result["litellm_metadata"]
        assert meta["codex_auto_agent_request_mode"] == "fresh_redispatch"
        assert meta["codex_auto_agent_redispatch_ordinal"] == 2
        assert meta["codex_auto_agent_affinity_bypassed"] is True

        audit_events = meta["codex_auto_agent_audit_events"]
        assert audit_events
        assert audit_events[0]["attempts"][-1]["request_mode"] == "fresh_redispatch"
        assert audit_events[0]["attempts"][-1]["redispatch_ordinal"] == 2
        assert audit_events[0]["attempts"][-1]["affinity_bypassed"] is True

    def test_three_children_keep_distinct_canonical_thread_ids(self) -> None:
        parent = "01a012a1-2a97-7622-837c-3066ec78f02f"
        children = [
            "01a012a6-c49a-7a42-899d-de19e2af2e9e",
            "01a012b0-e33f-7153-9e20-6af4560b4cec",
            "01a012b0-e58e-7372-b7d9-38bc36db15e7",
        ]
        identities: list[str] = []
        for child in children:
            body = {
                "model": "codex-auto-agent",
                "litellm_metadata": {
                    "session_id": parent,
                    "thread_id": child,
                    "parent_thread_id": parent,
                },
            }
            result = attempt_records._add_codex_auto_agent_alias_metadata(
                body,
                request=_make_request(),
                selection=self._selection(),
                attempts=[],
            )
            meta = result["litellm_metadata"]
            assert meta["canonical_thread_id"] == child
            assert meta["parent_thread_id"] == parent
            identities.append(meta["canonical_thread_id"])
        assert identities == children
        assert len(set(identities)) == 3

    def test_parent_only_body_does_not_invent_child_identity(self) -> None:
        body = {
            "model": "codex-auto-agent",
            "litellm_metadata": {"session_id": "session-parent-only"},
        }
        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body,
            request=_make_request(),
            selection=self._selection(),
            attempts=[],
        )
        meta = result["litellm_metadata"]
        assert meta["canonical_thread_id"] is None
        assert meta["parent_thread_id"] is None

    def test_unbound_classification_and_selected_lane_survive_without_content(
        self,
    ) -> None:
        prompt = "raw user prompt must not leak"
        credential = "sk-secret-credential"
        body = {
            "model": "codex-auto-agent",
            "instructions": prompt,
            "litellm_metadata": {"authorization": credential},
        }
        lane = "codex-oauth:account1:hash-account-1"
        selection = self._selection(
            has_account_bound_state=False,
            account_bound_classification="unbound",
        )
        selection["candidate"]["codex_oauth_lane_key"] = lane
        attempts: list[dict[str, Any]] = [
            {
                "status": "started",
                "provider": "openai",
                "error_class": "item_not_found",
                "litellm_call_id": "call-unbound-1",
            }
        ]

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body,
            request=_make_request(),
            selection=selection,
            attempts=attempts,
        )

        meta = result["litellm_metadata"]
        assert meta["has_account_bound_state"] is False
        assert meta["account_bound_classification"] == "unbound"
        assert meta["codex_auto_agent_selected_account_lane"] == lane
        assert attempts[-1]["has_account_bound_state"] is False
        assert attempts[-1]["account_bound_classification"] == "unbound"
        assert attempts[-1]["error_class"] == "item_not_found"
        assert attempts[-1]["provider"] == "openai"
        assert attempts[-1]["litellm_call_id"] == "call-unbound-1"
        serialized = _serialized_recorded_alias_surfaces(meta, attempts)
        assert prompt not in serialized
        assert credential not in serialized

    def test_account_bound_classification_and_selected_lane_survive_without_content(
        self,
    ) -> None:
        secret = "SECRET_ENCRYPTED_BLOB"
        prompt = "raw user prompt must not leak"
        tool_args = '{"command":"cat /etc/shadow"}'
        credential = "sk-secret-credential"
        body = {
            "model": "codex-auto-agent",
            "encrypted_content": secret,
            "instructions": prompt,
            "input": [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": tool_args,
                }
            ],
            "previous_response_id": "resp_123",
            "litellm_metadata": {"authorization": credential},
        }
        lane = "codex-oauth:account1:hash-account-1"
        selection = self._selection(
            has_account_bound_state=True,
            account_bound_classification="account_bound",
        )
        selection["candidate"]["codex_oauth_lane_key"] = lane
        attempts: list[dict[str, Any]] = [
            {
                "status": "cooldown_set",
                "provider": "openai",
                "error_class": "item_not_found",
                "litellm_call_id": "call-bound-1",
            }
        ]

        result = attempt_records._add_codex_auto_agent_alias_metadata(
            body,
            request=_make_request(),
            selection=selection,
            attempts=attempts,
        )

        meta = result["litellm_metadata"]
        assert meta["has_account_bound_state"] is True
        assert meta["account_bound_classification"] == "account_bound"
        assert meta["codex_auto_agent_selected_account_lane"] == lane
        assert attempts[-1]["has_account_bound_state"] is True
        assert attempts[-1]["account_bound_classification"] == "account_bound"
        assert attempts[-1]["error_class"] == "item_not_found"
        assert attempts[-1]["provider"] == "openai"
        assert attempts[-1]["litellm_call_id"] == "call-bound-1"
        serialized = _serialized_recorded_alias_surfaces(meta, attempts)
        assert secret not in serialized
        assert prompt not in serialized
        assert tool_args not in serialized
        assert credential not in serialized


# ---------------------------------------------------------------------------
# _add_anthropic_auto_agent_alias_metadata
# ---------------------------------------------------------------------------


class TestAnthropicAliasMetadata:
    def _selection(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "candidate": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "route_family": "anthropic_messages",
                "last_resort": False,
            },
            "lane_key": "anthropic:claude-sonnet-4",
            "cooldown_key": "cd:anthropic:claude-sonnet-4",
            "alias_model": "claude-auto-agent",
            "selection_reason": "round_robin",
            "skipped": [],
        }
        base.update(overrides)
        return base

    def test_composes_metadata_and_tags(self) -> None:
        request = _make_request()
        body = {"model": "claude-auto-agent"}
        selection = self._selection()

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["model"] == "claude-sonnet-4-20250514"
        meta = result["litellm_metadata"]
        assert "anthropic-auto-agent-alias" in meta["tags"]
        assert "anthropic-auto-agent-selected:claude-sonnet-4-20250514" in meta["tags"]
        assert meta["anthropic_auto_agent_selected_model"] == "claude-sonnet-4-20250514"

    def test_last_resort_tag(self) -> None:
        request = _make_request()
        body = {"model": "claude-auto-agent"}
        selection = self._selection()
        selection["candidate"]["last_resort"] = True

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert "anthropic-auto-agent-last-resort" in result["litellm_metadata"]["tags"]

    def test_alias_model_from_selection_override(self) -> None:
        request = _make_request()
        body = {"model": "unknown-model"}
        selection = self._selection(alias_model="custom-anthropic-alias")

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        meta = result["litellm_metadata"]
        assert meta["anthropic_auto_agent_alias"] == "custom-anthropic-alias"
        assert "model-alias:custom-anthropic-alias" in meta["tags"]

    def test_configured_reasoning_effort_overrides_caller_thinking(self) -> None:
        """CFG-006: configured value clears caller effort/thinking shapes.

        Conflicting caller representations (thinking, output_config.effort,
        top-level reasoning_effort) are removed only when config is set; the
        canonical top-level value is handed to shared provider translation.
        """
        request = _make_request()
        body = {
            "model": "claude-auto-agent",
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "output_config": {"effort": "high", "verbosity": "medium"},
            "reasoning_effort": "high",
        }
        selection = self._selection()
        selection["candidate"]["reasoning_effort"] = "low"

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body,
            request=request,
            selection=selection,
            attempts=[{"status": "started"}],
        )

        assert "thinking" not in result
        assert result["output_config"] == {"verbosity": "medium"}
        assert result["reasoning_effort"] == "low"
        # Caller body untouched.
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert body["reasoning_effort"] == "high"
        meta = result["litellm_metadata"]
        assert "anthropic-auto-agent-config-effort:low" in meta["tags"]
        assert meta["anthropic_auto_agent_config_reasoning_effort"] == "low"
        # Audit construction consumes the attempt record, so both
        # low-cardinality config fields must survive onto the built event.
        audit_events = meta["anthropic_auto_agent_audit_events"]
        assert audit_events
        event_attempt = audit_events[0]["attempts"][-1]
        assert event_attempt["reasoning_effort_config_value"] == "low"
        assert event_attempt["reasoning_effort_config_source"] == "candidate_yaml"

    def test_omitted_config_preserves_caller_thinking(self) -> None:
        """CFG-006: omission imposes no alias policy on Anthropic ingress."""
        request = _make_request()
        body = {
            "model": "claude-auto-agent",
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {"effort": "high"},
        }
        selection = self._selection()

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body, request=request, selection=selection, attempts=[]
        )

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 8000}
        assert result["output_config"] == {"effort": "high"}
        assert "reasoning_effort" not in result

    def test_copies_fresh_redispatch_trace_onto_last_attempt(self) -> None:
        request = _make_request()
        body = {"model": "claude-auto-agent"}
        attempts: list[dict[str, Any]] = [{"status": "started"}]
        selection = self._selection(
            request_mode="fresh_redispatch",
            redispatch_ordinal=2,
            affinity_bypassed=True,
        )

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body,
            request=request,
            selection=selection,
            attempts=attempts,
        )

        assert attempts[-1]["request_mode"] == "fresh_redispatch"
        assert attempts[-1]["redispatch_ordinal"] == 2
        assert attempts[-1]["affinity_bypassed"] is True

        meta = result["litellm_metadata"]
        assert meta["anthropic_auto_agent_request_mode"] == "fresh_redispatch"
        assert meta["anthropic_auto_agent_redispatch_ordinal"] == 2
        assert meta["anthropic_auto_agent_affinity_bypassed"] is True

        audit_events = meta["anthropic_auto_agent_audit_events"]
        assert audit_events
        assert audit_events[0]["attempts"][-1]["request_mode"] == "fresh_redispatch"
        assert audit_events[0]["attempts"][-1]["redispatch_ordinal"] == 2
        assert audit_events[0]["attempts"][-1]["affinity_bypassed"] is True

    def test_three_children_keep_distinct_canonical_thread_ids(self) -> None:
        parent = "01a012a1-2a97-7622-837c-3066ec78f02f"
        children = [
            "01a012a6-c49a-7a42-899d-de19e2af2e9e",
            "01a012b0-e33f-7153-9e20-6af4560b4cec",
            "01a012b0-e58e-7372-b7d9-38bc36db15e7",
        ]
        identities: list[str] = []
        for child in children:
            body = {
                "model": "claude-auto-agent",
                "litellm_metadata": {
                    "session_id": parent,
                    "thread_id": child,
                    "parent_thread_id": parent,
                },
            }
            result = attempt_records._add_anthropic_auto_agent_alias_metadata(
                body,
                request=_make_request(),
                selection=self._selection(),
                attempts=[],
            )
            meta = result["litellm_metadata"]
            assert meta["canonical_thread_id"] == child
            assert meta["parent_thread_id"] == parent
            identities.append(meta["canonical_thread_id"])
        assert identities == children
        assert len(set(identities)) == 3

    def test_parent_only_body_does_not_invent_child_identity(self) -> None:
        body = {
            "model": "claude-auto-agent",
            "litellm_metadata": {"session_id": "session-parent-only"},
        }
        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body,
            request=_make_request(),
            selection=self._selection(),
            attempts=[],
        )
        meta = result["litellm_metadata"]
        assert meta["canonical_thread_id"] is None
        assert meta["parent_thread_id"] is None

    def test_account_bound_classification_and_selected_lane_survive_without_content(
        self,
    ) -> None:
        secret = "SECRET_ENCRYPTED_BLOB"
        prompt = "raw user prompt must not leak"
        tool_args = '{"command":"cat /etc/shadow"}'
        credential = "sk-secret-credential"
        body = {
            "model": "claude-auto-agent",
            "encrypted_content": secret,
            "instructions": prompt,
            "input": [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": tool_args,
                }
            ],
            "previous_response_id": "resp_123",
            "litellm_metadata": {"authorization": credential},
        }
        lane = "codex-oauth:account1:hash-account-1"
        selection = self._selection(
            has_account_bound_state=True,
            account_bound_classification="account_bound",
        )
        selection["candidate"]["codex_oauth_lane_key"] = lane
        attempts: list[dict[str, Any]] = [
            {
                "status": "cooldown_set",
                "provider": "anthropic",
                "error_class": "item_not_found",
                "litellm_call_id": "call-anth-bound-1",
            }
        ]

        result = attempt_records._add_anthropic_auto_agent_alias_metadata(
            body,
            request=_make_request(),
            selection=selection,
            attempts=attempts,
        )

        meta = result["litellm_metadata"]
        assert meta["has_account_bound_state"] is True
        assert meta["account_bound_classification"] == "account_bound"
        assert meta["anthropic_auto_agent_selected_account_lane"] == lane
        assert attempts[-1]["has_account_bound_state"] is True
        assert attempts[-1]["account_bound_classification"] == "account_bound"
        assert attempts[-1]["error_class"] == "item_not_found"
        assert attempts[-1]["provider"] == "anthropic"
        serialized = _serialized_recorded_alias_surfaces(meta, attempts)
        assert secret not in serialized
        assert prompt not in serialized
        assert tool_args not in serialized
        assert credential not in serialized
