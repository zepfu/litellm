"""RR-054 operational residual behavioral tests (#44, #45, #49, #51, #54).

Production-only assertions for trailing-JSON preservation after Grok tool
literals, non-advertised tool preservation, single Vertex auth, per-attempt
cooldown keys on audit events, and adversarial system-reminder regex bounds.
"""

from __future__ import annotations

import inspect
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _exec_command_advertised_tools() -> dict[str, dict[str, Any]]:
    return {
        "exec_command": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "additionalProperties": False,
        }
    }


def _minimal_request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/openai_passthrough/responses",
        "raw_path": b"/openai_passthrough/responses",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# #44 trailing JSON preservation after Grok composer tool payloads
# ---------------------------------------------------------------------------


def test_rr054_issue44_trailing_valid_json_after_newline_not_absorbed_into_payload() -> None:
    """Free-form text that parses as JSON after a tool payload must stay visible.

    RR-054 #44: trailing valid-JSON text after a newline is not part of the tool
    payload and must not be absorbed into the strip span / payload field.
    """
    text = (
        "Intro keeps.\n"
        "Tool label: exec_command\n"
        "Correlation ref: call-1\n"
        'Input payload: {"cmd": "ls"}\n'
        '{"note": "free form json text that must remain visible"}\n'
        "Tool label: exec_command\n"
        "Correlation ref: call-2\n"
        'Input payload: {"cmd": "pwd"}'
    )

    blocks = lpe._parse_grok_composer_literal_tool_label_blocks(text)
    assert len(blocks) == 2
    first = blocks[0]
    # Payload must be only the tool JSON object, not the following freeform JSON.
    assert first["payload"].strip() == '{"cmd": "ls"}'
    first_span = text[int(first["start"]) : int(first["end"])]
    assert '{"note": "free form json text that must remain visible"}' not in first_span
    assert lpe._parse_grok_composer_literal_tool_payload_json(first["payload"]) == {
        "cmd": "ls"
    }

    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert items is not None and len(items) == 2
    assert all(item.get("name") == "exec_command" for item in items)
    assert leftover is not None
    assert "free form json text that must remain visible" in leftover
    assert "Intro keeps." in leftover
    assert "Tool label:" not in leftover


def test_rr054_issue44_end_marker_branch_does_not_absorb_trailing_freeform() -> None:
    """End-marker after decoded JSON should not pull freeform into the payload."""
    text = (
        "Tool label: exec_command\n"
        "Correlation ref: call-1\n"
        'Input payload: {"cmd": "ls"}\n'
        "<|tool_call_end|>\n"
        'Trailing freeform {"note": "keep"}\n'
        "Tool label: exec_command\n"
        "Correlation ref: call-2\n"
        'Input payload: {"cmd": "pwd"}'
    )

    blocks = lpe._parse_grok_composer_literal_tool_label_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["payload"].strip() == '{"cmd": "ls"}'
    first_span = text[int(blocks[0]["start"]) : int(blocks[0]["end"])]
    assert "Trailing freeform" not in first_span
    assert "<|tool_call_end|>" not in blocks[0]["payload"]

    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert len(items) == 2
    assert leftover is not None
    assert "Trailing freeform" in leftover
    assert '"note": "keep"' in leftover or "keep" in leftover


def test_rr054_issue44_non_json_trailing_text_still_preserved() -> None:
    """Control: non-JSON freeform after a payload must remain in repaired text."""
    text = (
        "Tool label: exec_command\n"
        "Correlation ref: call-1\n"
        'Input payload: {"cmd": "ls"}\n'
        "Keep this freeform text\n"
        "Tool label: exec_command\n"
        "Correlation ref: call-2\n"
        'Input payload: {"cmd": "pwd"}'
    )
    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert len(items) == 2
    assert leftover is not None
    assert "Keep this freeform text" in leftover


# ---------------------------------------------------------------------------
# #45 non-advertised tool preservation
# ---------------------------------------------------------------------------


def test_rr054_issue45_non_advertised_tool_literal_left_in_text() -> None:
    """Hallucinated non-advertised tool labels must remain visible in text.

    RR-054 #45: blocks whose tool name is not in advertised_tools must not be
    added to strip_spans (and must not be silently deleted).
    """
    text = (
        "Delegating work.\n"
        "Tool label: spawn_agent\n"
        "Correlation ref: call-spawn\n"
        'Input payload: {"agent_type": "worker", "message": "fix it"}\n'
        "Outcome text: spawn_agent failed: unknown agent type worker\n"
        "Inspecting directly.\n"
        "Tool label: exec_command\n"
        "Correlation ref: call-exec\n"
        'Input payload: {"cmd": "git status -sb"}'
    )

    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert items is not None and len(items) == 1
    assert items[0]["name"] == "exec_command"
    assert items[0]["call_id"] == "call-exec"
    assert leftover is not None
    # Non-advertised spawn_agent block stays in the residual text.
    assert "Tool label: spawn_agent" in leftover
    assert "Input payload: {\"agent_type\": \"worker\"" in leftover
    assert "call-spawn" in leftover
    # Advertised tool markup is stripped.
    assert "Tool label: exec_command" not in leftover
    assert "call-exec" not in leftover


def test_rr054_issue45_only_non_advertised_does_not_claim_success_repair() -> None:
    """If nothing advertised repaired, do not pretend a successful strip/repair."""
    text = (
        "Tool label: spawn_agent\n"
        "Correlation ref: call-spawn\n"
        'Input payload: {"agent_type": "worker"}'
    )
    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert items == []
    assert leftover is None


# ---------------------------------------------------------------------------
# #49 single Vertex auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue49_base_vertex_proxy_route_auth_once() -> None:
    """_base_vertex_proxy_route must call user_api_key_auth exactly once.

    RR-054 #49: previous re-check called auth twice with the same key.
    """
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()
    mock_handler.get_default_base_target_url.return_value = (
        "https://us-central1-aiplatform.googleapis.com/"
    )

    mock_auth = AsyncMock(return_value={"user_id": "rr054-vertex-user"})
    mock_endpoint_func = AsyncMock(return_value={"ok": True})

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
        new=mock_auth,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key",
        return_value="Bearer sk-test",
    ), patch(
        "litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url",
        return_value="proj",
    ), patch(
        "litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url",
        return_value="us-central1",
    ), patch(
        "litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url",
        return_value=None,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
    ) as mock_pt_router, patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
        new=AsyncMock(
            return_value=(
                {},
                "https://us-central1-aiplatform.googleapis.com/",
                False,
                "proj",
                "us-central1",
            )
        ),
    ), patch(
        "litellm.llms.vertex_ai.common_utils.construct_target_url",
        return_value="https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini:generateContent",
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
        return_value=mock_endpoint_func,
    ), patch(
        "litellm.proxy.proxy_server.llm_router",
        None,
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await lpe._base_vertex_proxy_route(
            endpoint="v1/projects/proj/locations/us-central1/publishers/google/models/gemini:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler,
        )

    assert result == {"ok": True}
    assert mock_auth.await_count == 1
    mock_auth.assert_awaited_once()
    assert mock_auth.await_args.kwargs.get("api_key") == "Bearer sk-test"


def test_rr054_issue49_base_vertex_source_has_single_auth_call() -> None:
    source = inspect.getsource(lpe._base_vertex_proxy_route)
    # One await of user_api_key_auth; no dead re-check branch.
    assert source.count("user_api_key_auth(") == 1
    assert "if user_api_key_dict is None" not in source


# ---------------------------------------------------------------------------
# #51 per-attempt cooldown key on audit events
# ---------------------------------------------------------------------------


def test_rr054_issue51_audit_events_use_per_attempt_cooldown_key() -> None:
    """Each attempt event must carry that attempt's cooldown_key, not only the last.

    RR-054 #51: earlier failure attempts must not get cooldown_key=None (and must
    not receive the *selected* candidate's key only because they are last).
    """
    request = _minimal_request()
    selection = {
        "candidate": {
            "provider": "openai",
            "model": "m-selected",
            "route_family": "codex_openai_responses",
            "last_resort": False,
        },
        "cooldown_key": "selected:cooldown",
        "lane_key": "lane-selected",
        "selection_reason": "priority",
        "session_key": "sess-rr054-51",
        "skipped": [],
    }
    attempts = [
        {
            "provider": "openai",
            "model": "m-fail-a",
            "route_family": "codex_openai_responses",
            "last_resort": False,
            "status": "cooldown_set",
            "error_class": "rate_limit",
            "lane_key": "lane-a",
            "cooldown_key": "attempt-a-cooldown",
            "cooldown_seconds": 30,
        },
        {
            "provider": "xai",
            "model": "m-fail-b",
            "route_family": "codex_xai_responses",
            "last_resort": False,
            "status": "cooldown_set",
            "error_class": "transient",
            "lane_key": "lane-b",
            "cooldown_key": "attempt-b-cooldown",
            "cooldown_seconds": 15,
        },
        {
            "provider": "openai",
            "model": "m-selected",
            "route_family": "codex_openai_responses",
            "last_resort": False,
            "status": "selected",
            "lane_key": "lane-selected",
            # Last attempt may omit its own key; fall back to selection only then.
        },
    ]

    with patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_host_attribution",
        return_value={
            "client_ip": "127.0.0.1",
            "client_ip_source": "test",
            "host_name": "localhost",
            "host_name_source": "test",
        },
    ), patch.object(
        lpe, "_extract_auto_agent_alias_session_id", return_value="sid-51"
    ), patch.object(
        lpe, "_extract_auto_agent_alias_metadata_value", return_value=None
    ), patch.object(
        lpe, "_extract_auto_agent_alias_client_product_label", return_value=None
    ), patch.object(
        lpe, "_extract_auto_agent_alias_incoming_endpoint", return_value="/v1"
    ), patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_rollup_outgoing_target",
        return_value=None,
    ), patch.object(
        lpe, "_attach_auto_agent_alias_terminal_context_fields", return_value=None
    ):
        events = lpe._build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="aawm-code",
            request=request,
            request_body={},
            selection=selection,
            attempts=attempts,
        )

    by_attempt = {
        e.get("attempt_number"): e
        for e in events
        if e.get("attempt_number") is not None
    }
    assert by_attempt[1]["cooldown_key"] == "attempt-a-cooldown"
    assert by_attempt[1]["model"] == "m-fail-a"
    assert by_attempt[1]["event_type"] == "candidate_retryable_failure"
    assert by_attempt[2]["cooldown_key"] == "attempt-b-cooldown"
    assert by_attempt[2]["model"] == "m-fail-b"
    # Last attempt falls back to selection cooldown_key when attempt omits it.
    assert by_attempt[3]["cooldown_key"] == "selected:cooldown"
    assert by_attempt[3]["model"] == "m-selected"
    # Earlier attempts must not be labelled with the selected key.
    assert by_attempt[1]["cooldown_key"] != "selected:cooldown"
    assert by_attempt[2]["cooldown_key"] != "selected:cooldown"


def test_rr054_issue51_early_attempt_with_own_key_not_overwritten_by_selection() -> None:
    """If the last attempt is a different candidate, early keys still stick."""
    request = _minimal_request()
    selection = {
        "candidate": {
            "provider": "openai",
            "model": "m-last",
            "route_family": "codex_openai_responses",
            "last_resort": True,
        },
        "cooldown_key": "last:selected-cooldown",
        "lane_key": "lane-last",
        "selection_reason": "fallback",
        "session_key": "sess-rr054-51b",
        "skipped": [],
    }
    attempts = [
        {
            "provider": "openai",
            "model": "m-first",
            "route_family": "codex_openai_responses",
            "last_resort": False,
            "status": "cooldown_set",
            "error_class": "usage_limit_reached",
            "lane_key": "lane-first",
            "cooldown_key": "first:attempt-cooldown",
            "cooldown_seconds": 3600,
        },
        {
            "provider": "openai",
            "model": "m-last",
            "route_family": "codex_openai_responses",
            "last_resort": True,
            "status": "selected",
            "lane_key": "lane-last",
            "cooldown_key": "last:attempt-cooldown",
        },
    ]

    with patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_host_attribution",
        return_value={},
    ), patch.object(
        lpe, "_extract_auto_agent_alias_session_id", return_value="sid-51b"
    ), patch.object(
        lpe, "_extract_auto_agent_alias_metadata_value", return_value=None
    ), patch.object(
        lpe, "_extract_auto_agent_alias_client_product_label", return_value=None
    ), patch.object(
        lpe, "_extract_auto_agent_alias_incoming_endpoint", return_value="/v1"
    ), patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_rollup_outgoing_target",
        return_value=None,
    ), patch.object(
        lpe, "_attach_auto_agent_alias_terminal_context_fields", return_value=None
    ):
        events = lpe._build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="aawm-code",
            request=request,
            request_body={},
            selection=selection,
            attempts=attempts,
        )

    by_attempt = {
        e.get("attempt_number"): e
        for e in events
        if e.get("attempt_number") is not None
    }
    assert by_attempt[1]["cooldown_key"] == "first:attempt-cooldown"
    assert by_attempt[2]["cooldown_key"] == "last:attempt-cooldown"


# ---------------------------------------------------------------------------
# #54 adversarial system-reminder bound
# ---------------------------------------------------------------------------


def test_rr054_issue54_adversarial_unclosed_system_reminders_complete_quickly() -> None:
    """Many unclosed <system-reminder> openers must not quadratic-burn the loop.

    RR-054 #54: non-greedy DOTALL patterns over client text without matching
    closers should be guarded (close-tag count / bound) so adversarial payloads
    stay cheap.
    """
    # ~6k openers + filler (~megabyte-class, no closers). Unbounded non-greedy
    # DOTALL scans over this shape commonly take multi-second / 10s+; a guarded
    # path should return near-linearly in well under half a second.
    adversarial = ("<system-reminder>\n" * 6000) + ("payload " * 20000)
    assert len(adversarial) > 200_000

    t0 = time.perf_counter()
    out, compacted, markers, meta = lpe._compact_openai_adapter_claude_context_text(
        adversarial,
        cap=128,
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(out, str)
    assert compacted == 0  # no closed blocks to compact
    assert elapsed < 0.5, (
        f"RR-054 #54 OpenAI system-reminder compact still expensive on "
        f"adversarial unclosed openers: {elapsed:.3f}s (len={len(adversarial)})"
    )


def test_rr054_issue54_adversarial_unclosed_openers_google_auxiliary_bound() -> None:
    adversarial = ("<system-reminder>\n" * 6000) + ("payload " * 20000)
    assert len(adversarial) > 200_000

    t0 = time.perf_counter()
    (
        out,
        compacted,
        hooks,
        meta,
    ) = lpe._compact_expanded_claude_persisted_output_text_for_google_adapter(
        adversarial,
        auxiliary_context_char_cap=128,
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(out, str)
    assert elapsed < 0.5, (
        f"RR-054 #54 Google auxiliary/system-reminder compact still expensive: "
        f"{elapsed:.3f}s (len={len(adversarial)})"
    )


def test_rr054_issue54_closed_system_reminders_still_compact() -> None:
    """Guard must not break the legitimate closed-tag compaction path."""
    closed = "".join(
        (
            "<system-reminder>\n"
            f"SubagentStart hook additional context: CLAUDE.md body {i} "
            + ("x" * 200)
            + "\n</system-reminder>\n"
        )
        for i in range(5)
    )
    out, compacted, markers, meta = lpe._compact_openai_adapter_claude_context_text(
        closed,
        cap=80,
    )
    assert compacted == 5
    assert isinstance(out, str)
    assert out.count("<system-reminder>") == 5
    assert out.count("</system-reminder>") == 5
