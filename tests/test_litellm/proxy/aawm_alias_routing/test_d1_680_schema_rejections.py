"""Focused D1-680 schema-rejection contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from litellm.proxy import aawm_runtime_error_logging as runtime_error_logging
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    audit_build,
    audit_events,
    candidate_loop,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.attempt_records import (
    _attach_schema_rejection_to_attempt_record,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.schema_rejections import (
    SCHEMA_REJECTION_ERROR_CODE,
    SCHEMA_REJECTION_FAILURE_CLASS,
    SCHEMA_REJECTION_KEY,
    extract_schema_rejection,
    normalize_schema_rejection,
)


class _FakeLocalValidationError(Exception):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("SECRET_EXCEPTION_TEXT")
        self._structured_errors = errors

    def errors(self) -> list[dict[str, Any]]:
        return self._structured_errors


def _diagnostic(
    *,
    provider: str = "openai",
    route_family: str = "codex_responses",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "route_family": route_family,
        "stage": "upstream_4xx",
        "reason": "invalid_function_parameters",
        "category": "tool_schema",
        "object_type": "function",
        "safe_keys": ["name", "parameters"],
        "field_path": "tools[0].function.parameters",
    }


def test_normalizer_is_frozen_and_drops_unbounded_or_sensitive_values() -> None:
    normalized = normalize_schema_rejection(
        {
            **_diagnostic(),
            "safe_keys": [
                "name",
                "authorization",
                "api-key",
                "description",
                "a" * 100,
            ],
            "field_path": "tools[0].function.parameters",
            "item_index": 4097,
            "tool_index": True,
        },
        provider=None,
        route_family=None,
    )

    assert normalized is not None
    assert normalized.safe_keys == ("description", "name")
    assert normalized.item_index is None
    assert normalized.tool_index is None
    with pytest.raises(FrozenInstanceError):
        normalized.provider = "changed"  # type: ignore[misc]


def test_cursor_legacy_bridge_produces_one_bounded_schema_diagnostic() -> None:
    normalized = normalize_schema_rejection(
        {
            "cursor_replay_fresh_dispatch_reject": {
                "stage": "stock_full_history",
                "reason": "output_not_string",
                "item_index": 2,
                "item_type": "function_call_output",
                "item_keys": ["type", "output", "authorization"],
            }
        },
        provider="cursor",
        route_family="codex_cursor_agent_aiserver_adapter",
    )

    assert normalized is not None
    assert normalized.category == "cursor_replay"
    assert normalized.stage == "stock_full_history"
    assert normalized.safe_keys == ("output", "type")
    assert normalized.item_index == 2


def test_upstream_extraction_uses_only_structured_4xx_schema_evidence() -> None:
    exception = SimpleNamespace(
        status_code=422,
        _aawm_provider_returned=True,
        detail={
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_function_parameters",
                "param": "tools[0].function.parameters",
                "category": "tool_schema",
                "object_type": "function",
                "safe_keys": ["name", "parameters"],
                "message": "SECRET_SCHEMA_MESSAGE",
            }
        },
        body="SECRET_UNSTRUCTURED_BODY",
    )

    normalized = extract_schema_rejection(
        exception,
        provider="openai",
        route_family="codex_openai_chat_completions",
    )

    assert normalized is not None
    assert normalized.upstream_status == 422
    assert normalized.upstream_error_class == "invalid_request_error"
    assert normalized.upstream_error_code == "invalid_function_parameters"
    assert normalized.field_path == "tools[0].function.parameters"
    assert "SECRET_SCHEMA_MESSAGE" not in str(normalized.to_dict())
    assert "SECRET_UNSTRUCTURED_BODY" not in str(normalized.to_dict())

    non_schema_exception = SimpleNamespace(
        status_code=400,
        _aawm_provider_returned=True,
        detail={"error": {"type": "invalid_request_error", "message": "SECRET"}},
        body={"message": "SECRET"},
    )
    assert (
        extract_schema_rejection(
            non_schema_exception,
            provider="openai",
            route_family="codex_responses",
        )
        is None
    )


def test_attempt_attachment_sets_bounded_schema_identity_once() -> None:
    exception = SimpleNamespace(
        status_code=400,
        _aawm_provider_returned=True,
        detail={
            "error": {
                "schema_error": True,
                "code": "invalid_schema",
                "param": "tools[0].function.parameters",
            }
        },
    )
    attempt = {
        "provider": "xai",
        "route_family": "codex_xai_oauth_responses_adapter",
    }

    normalized = _attach_schema_rejection_to_attempt_record(
        attempt_record=attempt,
        exc=exception,
    )

    assert normalized is not None
    assert attempt["error_class"] == SCHEMA_REJECTION_FAILURE_CLASS
    assert attempt["error_code"] == SCHEMA_REJECTION_ERROR_CODE
    assert set(key for key in attempt if key == SCHEMA_REJECTION_KEY) == {
        SCHEMA_REJECTION_KEY
    }
    assert attempt[SCHEMA_REJECTION_KEY]["provider"] == "xai"


def test_attempt_attachment_replaces_generic_schema_identity() -> None:
    attempt = {
        **_diagnostic(),
        "schema_rejection": _diagnostic(),
        "error_class": "provider_terminal_error",
        "error_code": "unclassified",
    }

    normalized = _attach_schema_rejection_to_attempt_record(
        attempt_record=attempt,
    )

    assert normalized is not None
    assert attempt["error_class"] == SCHEMA_REJECTION_FAILURE_CLASS
    assert attempt["error_code"] == SCHEMA_REJECTION_ERROR_CODE


@pytest.fixture
def configured_audit_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audit_build,
        "_get_auto_agent_alias_request_context",
        lambda *_args: {"host_attribution": {}},
    )
    monkeypatch.setattr(
        audit_build,
        "_attach_auto_agent_alias_terminal_context_fields",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        audit_build,
        "_extract_auto_agent_alias_metadata_value",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        audit_build,
        "_extract_auto_agent_alias_incoming_endpoint",
        lambda *_args: "/v1/responses",
    )
    monkeypatch.setattr(
        audit_build,
        "_resolve_auto_agent_alias_route_rollup_outgoing_target",
        lambda **kwargs: kwargs.get("route_family"),
    )


def test_audit_and_terminal_paths_propagate_the_same_schema_diagnostic(
    configured_audit_build: None,
) -> None:
    candidate = {
        **_diagnostic(
            provider="xai",
            route_family="codex_xai_oauth_responses_adapter",
        ),
        "model": "grok-4",
        "schema_rejection": _diagnostic(
            provider="xai",
            route_family="codex_xai_oauth_responses_adapter",
        ),
        "error_class": SCHEMA_REJECTION_FAILURE_CLASS,
        "error_code": SCHEMA_REJECTION_ERROR_CODE,
    }
    event = audit_build._build_auto_agent_alias_audit_event(
        alias_family="codex",
        alias_model="codex-auto",
        request=MagicMock(),
        request_body={},
        selection={"lane_key": "xai:managed"},
        candidate=candidate,
        event_type="candidate_retryable_failure",
        candidate_status="failed",
    )
    assert event[SCHEMA_REJECTION_KEY]["route_family"] == (
        "codex_xai_oauth_responses_adapter"
    )

    terminal_event: dict[str, Any] = {}
    attempts = audit_events._enrich_auto_agent_alias_terminal_event_from_attempts(
        terminal_event,
        [candidate],
    )
    assert attempts[0][SCHEMA_REJECTION_KEY] == event[SCHEMA_REJECTION_KEY]
    assert terminal_event[SCHEMA_REJECTION_KEY] == event[SCHEMA_REJECTION_KEY]

    record = runtime_error_logging.build_agent_terminal_error_record(
        error_context={
            "provider": "xai",
            "route_family": "codex_xai_oauth_responses_adapter",
            SCHEMA_REJECTION_KEY: {
                **event[SCHEMA_REJECTION_KEY],
                "safe_keys": ["name", "authorization"],
                "message": "SECRET",
            },
        },
        terminal_outcome="agent_session_terminated",
        agent_session_killed=True,
    )
    assert record["context"][SCHEMA_REJECTION_KEY]["safe_keys"] == ["name"]
    assert "message" not in record["context"][SCHEMA_REJECTION_KEY]


def test_schema_rejection_preserves_specific_identity_across_attempt_and_audit(
    configured_audit_build: None,
) -> None:
    attempt = {
        **_diagnostic(),
        "model": "cursor-work",
        "schema_rejection": _diagnostic(),
        "error_class": "continuation_state_unavailable",
        "error_code": "aawm_codex_auto_agent_candidate_ineligible",
    }

    normalized = _attach_schema_rejection_to_attempt_record(
        attempt_record=attempt,
    )
    event = audit_build._build_auto_agent_alias_audit_event(
        alias_family="codex",
        alias_model="work",
        request=MagicMock(),
        request_body={},
        selection={"lane_key": "cursor"},
        candidate=attempt,
        event_type="candidate_retryable_failure",
        candidate_status="failed",
        failure_class=attempt["error_class"],
        error_code=attempt["error_code"],
    )
    terminal_event: dict[str, Any] = {}
    audit_events._enrich_auto_agent_alias_terminal_event_from_attempts(
        terminal_event,
        [attempt],
    )

    assert normalized is not None
    assert attempt["error_class"] == "continuation_state_unavailable"
    assert attempt["error_code"] == "aawm_codex_auto_agent_candidate_ineligible"
    assert event["failure_class"] == "continuation_state_unavailable"
    assert event["error_code"] == "aawm_codex_auto_agent_candidate_ineligible"
    assert terminal_event["failure_class"] == "continuation_state_unavailable"
    assert terminal_event["error_code"] == (
        "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert terminal_event[SCHEMA_REJECTION_KEY] == attempt[SCHEMA_REJECTION_KEY]


@pytest.mark.parametrize(
    ("provider", "route_family"),
    (
        ("openai", "codex_responses"),
        ("kimi_code", "codex_kimi_chat_completions_adapter"),
        ("xai", "codex_grok_native_responses_adapter"),
        ("xai", "codex_xai_oauth_responses_adapter"),
        ("nvidia", "codex_nvidia_completion_adapter"),
    ),
)
def test_attach_schema_rejection_extracts_local_validation_for_route_family(
    provider: str,
    route_family: str,
) -> None:
    attempt = {
        "provider": provider,
        "route_family": route_family,
        "attempted_provider_call": False,
        "failure_phase": "request_preparation",
    }
    exception = _FakeLocalValidationError(
        [
            {
                "type": "missing",
                "loc": ("request", "tools", 0, "name"),
                "object_type": "tool",
                "keys": ["name", "authorization"],
                "msg": "SECRET_VALIDATION_MESSAGE",
                "input": {"token": "SECRET_INPUT"},
                "ctx": {"error": "SECRET_CONTEXT"},
            }
        ]
    )

    normalized = _attach_schema_rejection_to_attempt_record(
        attempt_record=attempt,
        exc=exception,
    )

    assert normalized is not None
    assert attempt[SCHEMA_REJECTION_KEY] == {
        "provider": provider,
        "route_family": route_family,
        "stage": "request_preparation",
        "reason": "missing",
        "category": "schema_validation",
        "object_type": "tool",
        "safe_keys": ["name", "request", "tools"],
        "field_path": "request.tools[0].name",
    }
    serialized = str(attempt)
    assert "SECRET_VALIDATION_MESSAGE" not in serialized
    assert "SECRET_INPUT" not in serialized
    assert "SECRET_CONTEXT" not in serialized
    assert "SECRET_EXCEPTION_TEXT" not in serialized


def test_resolve_failure_plan_stores_local_validation_state_before_attachment() -> None:
    attempt = {
        "provider": "openai",
        "route_family": "codex_responses",
    }
    exception = _FakeLocalValidationError(
        [
            {
                "type": "missing",
                "loc": ("request", "tools", 0, "name"),
                "object_type": "tool",
            }
        ]
    )
    exception.attempted_provider_call = False
    exception.failure_phase = "request_preparation"
    exception.candidate_status = "ineligible"
    exception.ineligibility_reason = "unsupported"
    exception.code = "aawm_codex_auto_agent_candidate_ineligible"

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=MagicMock(),
        record_codex_failure_evidence_fn=MagicMock(),
        request=MagicMock(),
        candidate={
            "provider": "openai",
            "route_family": "codex_responses",
        },
        selection={},
        attempt_record=attempt,
        exc=exception,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=MagicMock(),
        classify_kimi_fn=MagicMock(),
        classify_retryable_fn=MagicMock(),
        grok_quota_fn=MagicMock(),
        cooldown_seconds_fn=MagicMock(),
    )

    assert plan.applied_scope == "none"
    assert attempt["attempted_provider_call"] is False
    assert attempt["failure_phase"] == "request_preparation"
    assert attempt[SCHEMA_REJECTION_KEY]["stage"] == "request_preparation"
    assert attempt[SCHEMA_REJECTION_KEY]["field_path"] == "request.tools[0].name"


def test_resolve_failure_plan_promotes_generic_schema_classification() -> None:
    attempt = {
        "provider": "openai",
        "route_family": "codex_responses",
    }
    exception = _FakeLocalValidationError(
        [
            {
                "type": "missing",
                "loc": ("request", "tools", 0, "name"),
                "object_type": "tool",
            }
        ]
    )
    exception.attempted_provider_call = False
    exception.failure_phase = "request_preparation"
    resolved: dict[str, Any] = {}

    def _resolve(**kwargs: Any) -> Any:
        resolved.update(kwargs)
        return candidate_loop.CooldownPublicationPlan(applied_scope="none")

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_resolve,
        record_codex_failure_evidence_fn=MagicMock(),
        request=MagicMock(),
        candidate={
            "provider": "openai",
            "route_family": "codex_responses",
        },
        selection={"cooldown_key": "openai:test", "lane_key": None},
        attempt_record=attempt,
        exc=exception,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda _exc, candidate=None: None,
        classify_kimi_fn=lambda _metadata: None,
        classify_retryable_fn=lambda _exc, candidate=None: "provider_terminal_error",
        grok_quota_fn=lambda _exc, candidate=None: False,
        cooldown_seconds_fn=lambda _exc, candidate=None: 0.0,
    )

    assert plan.applied_scope == "none"
    assert resolved["error_class"] == SCHEMA_REJECTION_FAILURE_CLASS
    assert attempt["error_class"] == SCHEMA_REJECTION_FAILURE_CLASS
    assert attempt["error_code"] == SCHEMA_REJECTION_ERROR_CODE


def test_schema_error_drops_arbitrary_code_and_token() -> None:
    normalized = normalize_schema_rejection(
        {
            **_diagnostic(),
            "reason": "arbitrary-private-reason",
            "schema_error": True,
            "code": "arbitrary-private-code",
            "error_code": "arbitrary-private-token",
        },
        provider=None,
        route_family=None,
    )

    assert normalized is not None
    serialized = normalized.to_dict()
    assert serialized["reason"] == "schema_error"
    assert "upstream_error_code" not in serialized
    assert "arbitrary-private-code" not in str(serialized)
    assert "arbitrary-private-token" not in str(serialized)


@pytest.mark.parametrize(
    "field_path",
    (
        "tools[0].authorization.name",
        "tools[0].550e8400-e29b-41d4-a716-446655440000.name",
        "tools[0].abcdef0123456789abcdef0123456789.name",
        "tools[0].opaque_segment.name",
    ),
)
def test_schema_rejection_drops_sensitive_or_opaque_path_segments(
    field_path: str,
) -> None:
    normalized = normalize_schema_rejection(
        {
            **_diagnostic(),
            "field_path": field_path,
        },
        provider=None,
        route_family=None,
    )

    assert normalized is not None
    assert normalized.field_path is None


def test_extract_schema_rejection_rejects_unbounded_local_validation_data():
    class FakeValidationError(Exception):
        def errors(self):
            return [
                {
                    "type": "arbitrary-private-token",
                    "loc": ("request", "user", "550e8400-e29b-41d4-a716-446655440000"),
                    "msg": "must not be serialized",
                    "input": {"secret": "value"},
                    "ctx": {"secret": "value"},
                }
            ]

    assert (
        extract_schema_rejection(
            FakeValidationError(),
            provider="openai",
            route_family="codex_responses",
            attempted_provider_call=False,
            failure_phase="request_preparation",
        )
        is None
    )
