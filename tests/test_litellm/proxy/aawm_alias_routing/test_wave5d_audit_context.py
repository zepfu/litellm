"""Module-local tests for Wave 5D audit_context.py extraction.

Drives the new module directly with fresh state/dependency stubs.
Does NOT import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    headers: Optional[list[tuple[bytes, bytes]]] = None,
    scope_extra: Optional[dict[str, Any]] = None,
) -> Request:
    """Create a minimal Request with a fresh .state namespace."""
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": headers or [],
        "query_string": b"",
    }
    if scope_extra:
        scope.update(scope_extra)
    return Request(scope)


def _clean_secret_string_stub(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _extract_metadata_value_stub(request_body: dict[str, Any], *keys: str) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _extract_client_product_label_stub(request: Any, request_body: dict[str, Any]) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        return metadata.get("client_name_version")
    return None


def _resolve_host_attribution_stub(request: Any) -> dict[str, Optional[str]]:
    return {
        "client_ip": "10.0.0.1",
        "client_ip_source": "x-forwarded-for",
        "host_name": "testhost",
        "host_name_source": "reverse-dns",
    }


def _extract_session_id_stub(request: Any, request_body: dict[str, Any]) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        sid = metadata.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    return None


def _build_rollup_label_stub(
    *,
    repository: Optional[str],
    client_product_label: Optional[str],
    host_name: Optional[str],
) -> Optional[str]:
    parts = [p for p in (repository, client_product_label) if p]
    return "/".join(parts) if parts else None


def _has_continuation_state_stub(value: Any, _seen: Optional[set[int]] = None) -> bool:
    """Simplified continuation-state check for tests."""
    if isinstance(value, dict):
        for key in ("previous_response_id", "call_id", "tool_call_id", "item_id"):
            if value.get(key):
                return True
        return any(_has_continuation_state_stub(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_continuation_state_stub(v) for v in value)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _configure_audit_context():
    """Configure audit_context runtime with fresh stubs before each test."""
    seam_names = (
        "_clean_secret_string",
        "_extract_auto_agent_alias_metadata_value",
        "_extract_auto_agent_alias_client_product_label",
        "_resolve_auto_agent_alias_route_host_attribution",
        "_extract_auto_agent_alias_session_id",
        "_build_auto_agent_alias_rollup_group_header_label",
        "_codex_auto_agent_request_has_continuation_state",
    )
    previous = {name: getattr(audit_context, name) for name in seam_names}

    audit_context.configure_audit_context_runtime(
        clean_secret_string=_clean_secret_string_stub,
        extract_metadata_value=_extract_metadata_value_stub,
        extract_client_product_label=_extract_client_product_label_stub,
        resolve_host_attribution=_resolve_host_attribution_stub,
        extract_session_id=_extract_session_id_stub,
        build_rollup_group_header_label=_build_rollup_label_stub,
        has_continuation_state=_has_continuation_state_stub,
    )
    yield
    for name, value in previous.items():
        setattr(audit_context, name, value)


# ---------------------------------------------------------------------------
# Tests: text blob extraction and role inference
# ---------------------------------------------------------------------------


class TestTextBlobExtraction:
    def test_instructions_string(self):
        body = {"instructions": "You are a 'worker' agent."}
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert blobs == ["You are a 'worker' agent."]

    def test_system_list_with_dicts(self):
        body = {"system": [{"text": "hello"}, {"content": "world"}]}
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert blobs == ["hello\nworld"]

    def test_messages_system_role(self):
        body = {
            "messages": [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "user msg"},
            ]
        }
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert blobs == ["sys prompt"]

    def test_messages_developer_role(self):
        body = {"messages": [{"role": "developer", "content": "dev prompt"}]}
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert blobs == ["dev prompt"]

    def test_messages_content_list(self):
        body = {
            "messages": [
                {"role": "system", "content": [{"text": "a"}, {"text": "b"}]},
            ]
        }
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert blobs == ["a\nb"]

    def test_messages_limited_to_first_five(self):
        body = {
            "messages": [
                {"role": "system", "content": f"msg{i}"} for i in range(10)
            ]
        }
        blobs = audit_context._extract_auto_agent_alias_text_blobs(body)
        assert len(blobs) == 5

    def test_empty_body(self):
        assert audit_context._extract_auto_agent_alias_text_blobs({}) == []


class TestRoleInference:
    def test_explorer_role(self):
        assert audit_context._extract_auto_agent_alias_role_from_text(
            "You are a 'explorer' agent."
        ) == "explorer"

    def test_worker_role(self):
        assert audit_context._extract_auto_agent_alias_role_from_text(
            "You are a 'worker' agent."
        ) == "worker"

    def test_default_role(self):
        assert audit_context._extract_auto_agent_alias_role_from_text(
            "You are a 'default' agent."
        ) == "default"

    def test_unknown_role_rejected(self):
        assert audit_context._extract_auto_agent_alias_role_from_text(
            "You are a 'admin' agent."
        ) is None

    def test_multiline_extraction(self):
        text = "Some preamble\nYou are a 'worker' agent.\nMore text"
        assert audit_context._extract_auto_agent_alias_role_from_text(text) == "worker"

    def test_empty_string(self):
        assert audit_context._extract_auto_agent_alias_role_from_text("") is None

    def test_infer_from_request_body(self):
        body = {"instructions": "You are a 'explorer' agent."}
        assert audit_context._infer_auto_agent_alias_role_from_request_body(body) == "explorer"

    def test_infer_no_role(self):
        body = {"instructions": "Just a normal prompt"}
        assert audit_context._infer_auto_agent_alias_role_from_request_body(body) is None


# ---------------------------------------------------------------------------
# Tests: metadata/header precedence
# ---------------------------------------------------------------------------


class TestMetadataDicts:
    def test_litellm_metadata_first(self):
        request = _make_request()
        body = {
            "litellm_metadata": {"agent_name": "from_litellm"},
            "metadata": {"agent_name": "from_metadata"},
        }
        sources = audit_context._iter_auto_agent_alias_metadata_dicts(request, body)
        assert sources[0] == {"agent_name": "from_litellm"}
        assert sources[1] == {"agent_name": "from_metadata"}

    def test_nested_source(self):
        request = _make_request()
        body = {"metadata": {"agent_name": "outer", "source": {"agent_id": "inner-id"}}}
        sources = audit_context._iter_auto_agent_alias_metadata_dicts(request, body)
        assert {"agent_id": "inner-id"} in sources

    def test_headers_filtered(self):
        headers = [
            (b"x-aawm-agent-name", b"header-agent"),
            (b"authorization", b"Bearer secret"),
        ]
        request = _make_request(headers=headers)
        body: dict[str, Any] = {}
        sources = audit_context._iter_auto_agent_alias_metadata_dicts(request, body)
        # Only the allowed header should be present
        assert len(sources) == 1
        assert "x-aawm-agent-name" in sources[0]
        assert "authorization" not in sources[0]


class TestAgentDispatchFields:
    def test_structured_metadata_precedence(self):
        request = _make_request()
        body = {
            "litellm_metadata": {
                "agent_name": "meta-agent",
                "agent_role": "worker",
                "agent_id": "id-123",
            },
            "instructions": "You are a 'explorer' agent.",
        }
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, body)
        # Structured metadata wins over role inference
        assert fields["agent_name"] == "meta-agent"
        assert fields["agent_role"] == "worker"
        assert fields["agent_id"] == "id-123"

    def test_role_inference_fallback(self):
        request = _make_request()
        body = {"instructions": "You are a 'explorer' agent."}
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, body)
        assert fields["agent_name"] == "explorer"
        assert fields["agent_role"] == "explorer"
        assert fields["agent_profile"] == "explorer"
        assert fields["thread_source"] == "subagent"

    def test_redispatch_ordinal_int(self):
        request = _make_request()
        body = {"litellm_metadata": {"redispatch_ordinal": "3"}}
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, body)
        assert fields["redispatch_ordinal"] == 3

    def test_redispatch_ordinal_non_numeric(self):
        request = _make_request()
        body = {"litellm_metadata": {"redispatch_ordinal": "abc"}}
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, body)
        assert fields["redispatch_ordinal"] == "abc"

    def test_dispatch_id(self):
        request = _make_request()
        body = {"litellm_metadata": {"dispatch_id": "disp-99"}}
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, body)
        assert fields["dispatch_id"] == "disp-99"

    def test_empty_body_no_fields(self):
        request = _make_request()
        fields = audit_context._extract_auto_agent_alias_agent_dispatch_fields(request, {})
        assert fields == {}


# ---------------------------------------------------------------------------
# Tests: request-call ID stability
# ---------------------------------------------------------------------------


class TestRequestCallId:
    def test_metadata_call_id(self):
        request = _make_request()
        body = {"litellm_metadata": {"litellm_call_id": "call-abc"}}
        call_id = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        assert call_id == "call-abc"

    def test_scope_fallback(self):
        request = _make_request(scope_extra={"request_id": "req-xyz"})
        body: dict[str, Any] = {}
        call_id = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        assert call_id == "req-xyz"

    def test_header_fallback(self):
        headers = [(b"x-request-id", b"hdr-123")]
        request = _make_request(headers=headers)
        body: dict[str, Any] = {}
        call_id = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        assert call_id == "hdr-123"

    def test_uuid_generated_when_nothing_available(self):
        request = _make_request()
        body: dict[str, Any] = {}
        call_id = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        # Should be a valid UUID string
        assert len(call_id) == 36
        assert call_id.count("-") == 4

    def test_stability_via_state_memoization(self):
        request = _make_request()
        body: dict[str, Any] = {}
        first = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        second = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        assert first == second

    def test_state_key_set(self):
        request = _make_request()
        body: dict[str, Any] = {}
        call_id = audit_context._get_or_create_auto_agent_alias_request_call_id(request, body)
        stored = getattr(request.state, audit_context._AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY)
        assert stored == call_id


# ---------------------------------------------------------------------------
# Tests: request-state memoization for full context
# ---------------------------------------------------------------------------


class TestRequestContextMemoization:
    def test_context_cached_on_state(self):
        request = _make_request()
        body: dict[str, Any] = {"litellm_metadata": {"session_id": "sess-1"}}
        ctx1 = audit_context._get_auto_agent_alias_request_context(request, body)
        ctx2 = audit_context._get_auto_agent_alias_request_context(request, body)
        assert ctx1["session_id"] == "sess-1"
        assert ctx1["litellm_call_id"] == ctx2["litellm_call_id"]

    def test_host_attribution_populated(self):
        request = _make_request()
        body: dict[str, Any] = {}
        ctx = audit_context._get_auto_agent_alias_request_context(request, body)
        assert ctx["host_attribution"]["client_ip"] == "10.0.0.1"
        assert ctx["host_attribution"]["host_name"] == "testhost"

    def test_rollup_group_header_label(self):
        request = _make_request()
        body: dict[str, Any] = {"litellm_metadata": {"repository": "myrepo"}}
        ctx = audit_context._get_auto_agent_alias_request_context(request, body)
        assert ctx["repository"] == "myrepo"
        assert "myrepo" in (ctx["rollup_group_header_label"] or "")

    def test_include_activity_adds_summary(self):
        request = _make_request()
        body: dict[str, Any] = {"input": [{"type": "function_call", "name": "shell"}]}
        ctx = audit_context._get_auto_agent_alias_request_context(
            request, body, include_activity=True
        )
        assert "prior_tool_activity_summary" in ctx
        assert ctx["prior_tool_activity_summary"]["prior_tool_call_count"] == 1


# ---------------------------------------------------------------------------
# Tests: bounded/cycle-safe prior-tool traversal
# ---------------------------------------------------------------------------


class TestPriorToolTraversal:
    def test_basic_function_call(self):
        tool_names: list[str] = []
        file_edit_names: list[str] = []
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {"type": "function_call", "name": "shell"}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=file_edit_names, counters=counters
        )
        assert counters["prior_tool_call_count"] == 1
        assert "shell" in tool_names

    def test_function_call_output(self):
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {"type": "function_call_output", "output": "done"}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=[], file_edit_tool_names=[], counters=counters
        )
        assert counters["prior_tool_result_count"] == 1

    def test_file_edit_detection(self):
        tool_names: list[str] = []
        file_edit_names: list[str] = []
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {"type": "function_call", "name": "apply_patch"}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=file_edit_names, counters=counters
        )
        assert counters["prior_file_edit_tool_call_count"] == 1
        assert "apply_patch" in file_edit_names

    def test_namespaced_file_edit_tool(self):
        tool_names: list[str] = []
        file_edit_names: list[str] = []
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {"type": "function_call", "name": "mcp.filesystem/write_file"}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=file_edit_names, counters=counters
        )
        assert counters["prior_file_edit_tool_call_count"] == 1
        assert "mcp.filesystem/write_file" in file_edit_names

    def test_tool_calls_in_message(self):
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {
            "role": "assistant",
            "tool_calls": [
                {"name": "shell", "function": {"name": "shell"}},
                {"name": "edit_file", "function": {"name": "edit_file"}},
            ],
        }
        tool_names: list[str] = []
        file_edit_names: list[str] = []
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=file_edit_names, counters=counters
        )
        assert counters["prior_tool_call_count"] == 2
        assert counters["prior_file_edit_tool_call_count"] == 1

    def test_cycle_safety(self):
        """Self-referencing dict must not cause infinite recursion."""
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value: dict[str, Any] = {"type": "function_call", "name": "shell"}
        value["self"] = value  # cycle
        tool_names: list[str] = []
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=[], counters=counters
        )
        # Should count once, not infinitely
        assert counters["prior_tool_call_count"] == 1

    def test_list_cycle_safety(self):
        """Self-referencing list must not cause infinite recursion."""
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        lst: list[Any] = [{"type": "tool_use", "name": "x"}]
        lst.append(lst)  # cycle
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            lst, tool_names=[], file_edit_tool_names=[], counters=counters
        )
        assert counters["prior_tool_call_count"] == 1

    def test_tool_result_role(self):
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        value = {"role": "tool", "content": "result text", "tool_call_id": "tc-1"}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=[], file_edit_tool_names=[], counters=counters
        )
        assert counters["prior_tool_result_count"] == 1

    def test_function_name_from_function_obj(self):
        counters = {"prior_tool_call_count": 0, "prior_tool_result_count": 0, "prior_file_edit_tool_call_count": 0}
        tool_names: list[str] = []
        value = {"type": "function_call", "function": {"name": "create_file"}}
        audit_context._walk_auto_agent_alias_prior_tool_activity(
            value, tool_names=tool_names, file_edit_tool_names=[], counters=counters
        )
        assert "create_file" in tool_names


class TestSummarizeActivity:
    def test_with_tool_calls(self):
        body = {
            "input": [
                {"type": "function_call", "name": "shell"},
                {"type": "function_call_output", "output": "ok"},
            ]
        }
        summary = audit_context._summarize_auto_agent_alias_actual_prior_tool_activity(body)
        assert summary["has_actual_prior_tool_activity"] is True
        assert summary["prior_tool_call_count"] == 1
        assert summary["prior_tool_result_count"] == 1
        assert "shell" in summary["prior_tool_names"]

    def test_no_activity(self):
        body = {"input": [{"type": "message", "content": "hello"}]}
        summary = audit_context._summarize_auto_agent_alias_actual_prior_tool_activity(body)
        assert summary["has_actual_prior_tool_activity"] is False
        assert summary["prior_tool_call_count"] == 0

    def test_continuation_state(self):
        body = {"input": [{"call_id": "c-1"}]}
        summary = audit_context._summarize_auto_agent_alias_actual_prior_tool_activity(body)
        assert summary["has_continuation_state"] is True

    def test_previous_response_id(self):
        body = {"previous_response_id": "resp-123"}
        summary = audit_context._summarize_auto_agent_alias_actual_prior_tool_activity(body)
        assert summary["has_previous_response_id"] is True

    def test_tool_names_bounded_to_20(self):
        body = {
            "input": [
                {"type": "function_call", "name": f"tool_{i}"} for i in range(30)
            ]
        }
        summary = audit_context._summarize_auto_agent_alias_actual_prior_tool_activity(body)
        assert len(summary["prior_tool_names"]) == 20


# ---------------------------------------------------------------------------
# Tests: activity classification
# ---------------------------------------------------------------------------


class TestActivityClassification:
    def test_partial_activity(self):
        summary = {"has_actual_prior_tool_activity": True}
        assert audit_context._classify_auto_agent_alias_terminal_activity_status(summary) == "failed_after_partial_activity"

    def test_no_activity(self):
        summary = {"has_actual_prior_tool_activity": False}
        assert audit_context._classify_auto_agent_alias_terminal_activity_status(summary) == "failed_no_activity"

    def test_none_summary(self):
        assert audit_context._classify_auto_agent_alias_terminal_activity_status(None) == "failed_no_activity"

    def test_empty_dict(self):
        assert audit_context._classify_auto_agent_alias_terminal_activity_status({}) == "failed_no_activity"


# ---------------------------------------------------------------------------
# Tests: terminal field attachment without overwriting
# ---------------------------------------------------------------------------


class TestTerminalContextAttachment:
    def test_attaches_agent_dispatch(self):
        request = _make_request()
        body = {"litellm_metadata": {"agent_name": "w1", "agent_role": "worker"}}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert result["agent_name"] == "w1"
        assert result["agent_role"] == "worker"

    def test_does_not_overwrite_existing(self):
        request = _make_request()
        body = {"litellm_metadata": {"agent_name": "from-meta"}}
        event: dict[str, Any] = {"event_type": "route_attempt", "agent_name": "pre-existing"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert result["agent_name"] == "pre-existing"

    def test_session_id_attached(self):
        request = _make_request()
        body = {"litellm_metadata": {"session_id": "sess-42"}}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert result["session_id"] == "sess-42"

    def test_litellm_call_id_attached(self):
        request = _make_request()
        body = {"litellm_metadata": {"litellm_call_id": "call-77"}}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert result["litellm_call_id"] == "call-77"

    def test_cooldown_state_source_from_candidate(self):
        request = _make_request()
        body: dict[str, Any] = {}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        candidate = {"cooldown_state_source": "redis"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body, candidate=candidate
        )
        assert result["cooldown_state_source"] == "redis"

    def test_cooldown_state_source_from_selection_fallback(self):
        request = _make_request()
        body: dict[str, Any] = {}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        selection = {"cooldown_state_source": "memory"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body, selection=selection
        )
        assert result["cooldown_state_source"] == "memory"

    def test_activity_included_for_terminal_event_type(self):
        request = _make_request()
        body = {"input": [{"type": "function_call", "name": "shell"}]}
        event: dict[str, Any] = {"event_type": "no_candidate_available"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert "actual_prior_tool_activity_summary" in result
        assert result["actual_prior_tool_activity_summary"]["prior_tool_call_count"] == 1

    def test_activity_status_when_include_flag(self):
        request = _make_request()
        body = {"input": [{"type": "function_call", "name": "shell"}]}
        event: dict[str, Any] = {"event_type": "agent_session_terminated"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body, include_activity_status=True
        )
        assert result["terminal_activity_status"] == "failed_after_partial_activity"

    def test_no_activity_for_non_terminal_event(self):
        request = _make_request()
        body = {"input": [{"type": "function_call", "name": "shell"}]}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert "actual_prior_tool_activity_summary" not in result
        assert "terminal_activity_status" not in result

    def test_agent_id_fallback_from_metadata(self):
        request = _make_request()
        body = {"litellm_metadata": {"agent_id": "fallback-id"}}
        event: dict[str, Any] = {"event_type": "route_attempt"}
        result = audit_context._attach_auto_agent_alias_terminal_context_fields(
            event, request=request, request_body=body
        )
        assert result["agent_id"] == "fallback-id"


# ---------------------------------------------------------------------------
# Tests: normalize context
# ---------------------------------------------------------------------------


class TestNormalizeContext:
    def test_basic_normalization(self):
        raw = {
            "agent_dispatch": {"agent_name": "x"},
            "session_id": "  sess  ",
            "litellm_call_id": "call-1",
            "trace_id": None,
            "repository": "repo",
            "client_product_label": "Codex/1.0",
            "host_attribution": {"client_ip": "1.2.3.4"},
            "rollup_group_header_label": "label",
        }
        ctx = audit_context._normalize_auto_agent_alias_request_context(raw)
        assert ctx["agent_dispatch"] == {"agent_name": "x"}
        assert ctx["session_id"] == "sess"
        assert ctx["litellm_call_id"] == "call-1"
        assert ctx["trace_id"] is None
        assert ctx["repository"] == "repo"
        assert ctx["host_attribution"] == {"client_ip": "1.2.3.4"}

    def test_missing_litellm_call_id_generates_uuid(self):
        raw: dict[str, object] = {}
        ctx = audit_context._normalize_auto_agent_alias_request_context(raw)
        assert len(ctx["litellm_call_id"]) == 36

    def test_prior_activity_preserved(self):
        raw = {"prior_tool_activity_summary": {"prior_tool_call_count": 5}}
        ctx = audit_context._normalize_auto_agent_alias_request_context(raw)
        assert ctx["prior_tool_activity_summary"]["prior_tool_call_count"] == 5

    def test_non_dict_agent_dispatch_becomes_empty(self):
        raw = {"agent_dispatch": "not-a-dict"}
        ctx = audit_context._normalize_auto_agent_alias_request_context(raw)
        assert ctx["agent_dispatch"] == {}


# ---------------------------------------------------------------------------
# Tests: clean_optional_string
# ---------------------------------------------------------------------------


class TestCleanOptionalString:
    def test_strips_whitespace(self):
        assert audit_context._clean_optional_string("  hello  ") == "hello"

    def test_strips_quotes(self):
        assert audit_context._clean_optional_string('"quoted"') == "quoted"

    def test_none_for_non_string(self):
        assert audit_context._clean_optional_string(123) is None

    def test_none_for_empty(self):
        assert audit_context._clean_optional_string("") is None
        assert audit_context._clean_optional_string("   ") is None
