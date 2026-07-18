"""RR-054 regressions for #46 stream/completed output merge and #50 session_key.

Finding #46: final merge of streamed output items with response.completed output
must correlate fallback/output_index-keyed stream items that later gain a real
call_id/id, and must not emit duplicates.

Finding #50: terminal no-candidate audit/terminal events must carry a
lane-scoped resolved session_key (via _resolve_*_auto_agent_session_key), not
raw session_id and not None.

Tests-only packet. Failures here document remaining production gaps.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request
from starlette.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _minimal_request(
    *,
    path: str = "/openai_passthrough/responses",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")


def _function_calls(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


# ---------------------------------------------------------------------------
# #46 final stream/completed output merge correlation
# ---------------------------------------------------------------------------


def test_rr054_issue46_merge_fallback_keyed_stream_item_gaining_call_id_does_not_duplicate() -> None:
    """Stream fallback-keyed item + completed call_id item must merge to one row.

    RR-054 #46: a streamed incomplete item without mid-stream ids must correlate
    with the completed payload item that later surfaces a real call_id, not
    append a second logical tool call.
    """
    streamed_output = [
        {
            "type": "function_call",
            # No call_id/id: stream key becomes type-prefixed fallback/output key.
            "name": "Bash",
            "arguments": '{"cmd":"pwd"}',
            "status": "in_progress",
        }
    ]
    completed_output = [
        {
            "type": "function_call",
            "id": "fc_pwd",
            "call_id": "call_pwd",
            "name": "Bash",
            "arguments": '{"cmd":"pwd"}',
            "status": "completed",
        }
    ]

    merged = lpe._merge_responses_output_lists(completed_output, streamed_output)
    function_calls = _function_calls(merged)

    assert len(function_calls) == 1, (
        "RR-054 #46: fallback/output_index stream item later gaining call_id "
        f"must not duplicate; got {function_calls!r}"
    )
    assert function_calls[0].get("call_id") == "call_pwd"
    assert function_calls[0].get("arguments") == '{"cmd":"pwd"}'
    assert function_calls[0].get("id") == "fc_pwd"


def test_rr054_issue46_merge_with_stream_correlation_maps_does_not_duplicate() -> None:
    """Finalize-shaped correlation maps must keep fallback stream + call_id one row.

    Mirrors how ``_finalize_collected_responses_stream_response`` threads
    ``streamed_ordered_keys`` / ``key_aliases`` / ``key_by_output_index``.
    """
    stream_key = "function_call:output:0"
    streamed_output = [
        {
            "type": "function_call",
            "name": "Bash",
            "arguments": '{"cmd":"ls","via":"stream"}',
            "status": "in_progress",
        }
    ]
    completed_output = [
        {
            "type": "function_call",
            "id": "fc_ls",
            "call_id": "call_ls",
            "name": "Bash",
            "arguments": '{"cmd":"ls"}',
            "status": "completed",
        }
    ]

    merged = lpe._merge_responses_output_lists(
        completed_output,
        streamed_output,
        streamed_ordered_keys=[stream_key],
        key_aliases={},
        key_by_output_index={0: stream_key},
    )
    function_calls = _function_calls(merged)
    assert len(function_calls) == 1, (
        "RR-054 #46: correlation maps must collapse fallback stream key "
        f"{stream_key!r} with completed call_id; got {function_calls!r}"
    )
    assert function_calls[0].get("call_id") == "call_ls"
    # Stream args preserved when completed is thinner / overwrites carefully.
    assert "ls" in str(function_calls[0].get("arguments") or "")


def test_rr054_issue46_finalize_preserves_stream_args_when_completed_gains_call_id() -> None:
    """Finalize path must not leave a fallback twin beside the completed item."""
    stream_key = "function_call:fallback:0"
    ordered_keys = [stream_key]
    output_items = {
        stream_key: {
            "type": "function_call",
            "name": "Bash",
            # Stream accumulated richer args before completed payload arrived.
            "arguments": '{"cmd":"pwd","extra":1}',
        }
    }
    key_aliases: dict[str, str] = {}
    key_by_output_index = {0: stream_key}
    response_dict: dict[str, Any] = {
        "id": "resp_rr054_46",
        "status": "completed",
        "model": "gpt-test",
        "output": [
            {
                "type": "function_call",
                "id": "fc_pwd",
                "call_id": "call_pwd",
                "name": "Bash",
                "arguments": '{"cmd":"pwd"}',
                "status": "completed",
            }
        ],
    }

    finalized = lpe._finalize_collected_responses_stream_response(
        response_dict=response_dict,
        output_text_parts=[],
        output_items=output_items,
        ordered_keys=ordered_keys,
        key_aliases=key_aliases,
        key_by_output_index=key_by_output_index,
    )
    function_calls = _function_calls(finalized.get("output"))
    assert len(function_calls) == 1, (
        "RR-054 #46 finalize: fallback-keyed stream item + completed call_id "
        f"must not duplicate; got {function_calls!r}"
    )
    assert function_calls[0].get("call_id") == "call_pwd"
    assert "pwd" in str(function_calls[0].get("arguments") or "")


@pytest.mark.asyncio
async def test_rr054_issue46_collect_stream_without_midstream_ids_then_completed_call_id() -> None:
    """End-to-end collect: mid-stream item lacks call_id; completed supplies it."""
    chunks = [
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    # Intentionally omit id/call_id so accumulator uses synthetic key.
                    "name": "Bash",
                    "arguments": "",
                },
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": '{"cmd":"pwd"}',
            }
        ),
        _sse(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_rr054_46_stream",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_pwd",
                            "call_id": "call_pwd",
                            "name": "Bash",
                            "arguments": '{"cmd":"pwd"}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        ),
    ]

    async def _gen():
        for chunk in chunks:
            yield chunk

    response = StreamingResponse(_gen(), media_type="text/event-stream")
    collected = await lpe._collect_responses_response_from_stream(response)
    function_calls = _function_calls(collected.get("output"))
    assert len(function_calls) == 1, (
        "RR-054 #46 stream collect: item that only later gains call_id must not "
        f"duplicate; got {function_calls!r}"
    )
    assert function_calls[0].get("call_id") == "call_pwd"
    assert function_calls[0].get("arguments") == '{"cmd":"pwd"}'


def test_rr054_issue46_merge_does_not_duplicate_when_completed_only_has_call_id() -> None:
    """Synthetic stream key shape from helper must still collapse with call_id."""
    stream_item = {
        "type": "function_call",
        "name": "Bash",
        "arguments": '{"cmd":"ls"}',
    }
    synthetic_key = lpe._responses_output_stream_key(
        item=stream_item,
        output_index=0,
        fallback_index=0,
    )
    assert "call_" not in synthetic_key
    assert "fc_" not in synthetic_key

    completed_output = [
        {
            "type": "function_call",
            "id": "fc_ls",
            "call_id": "call_ls",
            "name": "Bash",
            "arguments": '{"cmd":"ls"}',
            "status": "completed",
        }
    ]
    merged = lpe._merge_responses_output_lists(
        completed_output,
        [stream_item],
        streamed_ordered_keys=[synthetic_key],
        key_by_output_index={0: synthetic_key},
    )
    function_calls = _function_calls(merged)
    assert len(function_calls) == 1, (
        "RR-054 #46: synthetic stream key "
        f"{synthetic_key!r} must correlate with completed call_id; got "
        f"{function_calls!r}"
    )
    assert function_calls[0].get("call_id") == "call_ls"


# ---------------------------------------------------------------------------
# #50 terminal no-candidate lane-scoped session_key
# ---------------------------------------------------------------------------


def test_rr054_issue50_no_candidate_audit_uses_lane_scoped_session_key() -> None:
    """Terminal no-candidate path must resolve lane-scoped session_key.

    RR-054 #50: events must not fall back to raw session_id or leave
    session_key=None. Key must match ``_resolve_codex_auto_agent_session_key``
    (session_id + openai lane).
    """
    session_id = "rr054-session-50"
    account_id = "acct-rr054-50"
    alias_model = "aawm-code"
    request = _minimal_request(
        headers=[
            (b"session_id", session_id.encode("utf-8")),
            (b"chatgpt-account-id", account_id.encode("utf-8")),
        ]
    )
    body = {
        "model": alias_model,
        "litellm_metadata": {
            "session_id": session_id,
            "agent_id": "agent-rr054-50",
        },
    }
    expected_session_key = lpe._resolve_codex_auto_agent_session_key(
        request,
        body,
        alias_model=alias_model,
    )
    assert expected_session_key is not None
    assert expected_session_key != session_id
    assert session_id in expected_session_key
    assert "chatgpt-account" in expected_session_key or account_id in expected_session_key

    captured_events: list[dict[str, Any]] = []

    def _capture_persist(events, **kwargs: Any) -> None:
        for event in events or []:
            if isinstance(event, dict):
                captured_events.append(event)

    with patch.object(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        side_effect=_capture_persist,
    ), patch.object(
        lpe, "_emit_auto_agent_alias_route_event", return_value=None
    ), patch(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        return_value=True,
    ):
        lpe._emit_auto_agent_alias_no_candidate_event(
            alias_family="codex_auto_agent",
            alias_model=alias_model,
            request=request,
            request_body=body,
            exc=HTTPException(
                status_code=429,
                detail={
                    "candidates": [
                        {
                            "provider": "openai",
                            "model": "gpt-test",
                            "route_family": "codex_openai_responses",
                        }
                    ],
                    "error": {"code": "all_unavailable"},
                },
            ),
            attempts=[
                {
                    "provider": "openai",
                    "model": "gpt-test",
                    "route_family": "codex_openai_responses",
                    "lane_key": f"chatgpt-account:{account_id}",
                    "error_class": "rate_limit",
                    "status": "cooldown_set",
                }
            ],
        )

    assert captured_events, "expected terminal/audit events to be persisted"
    session_keys = [
        event.get("session_key")
        for event in captured_events
        if event.get("session_key") is not None
    ]
    assert expected_session_key in session_keys, (
        "RR-054 #50: terminal no-candidate events must carry lane-scoped "
        f"session_key={expected_session_key!r}; observed session_keys="
        f"{session_keys!r} events={captured_events!r}"
    )

    for event in captured_events:
        if event.get("event_type") in {
            "no_candidate_available",
            "candidate_retryable_failure",
            "candidate_selected",
            "redispatch_required",
        } or event.get("terminal_outcome") == "agent_session_terminated":
            sk = event.get("session_key")
            assert sk is not None, (
                "RR-054 #50: terminal/audit no-candidate event must not leave "
                f"session_key=None; event={event!r}"
            )
            assert sk != session_id, (
                "RR-054 #50: session_key must be lane-scoped, not raw session_id; "
                f"event={event!r}"
            )
            assert sk == expected_session_key, (
                "RR-054 #50: session_key must match "
                f"_resolve_codex_auto_agent_session_key; expected "
                f"{expected_session_key!r}, got {sk!r}"
            )


def test_rr054_issue50_no_candidate_default_lane_session_key_not_none() -> None:
    """Lane-scoped default-lane key is required (not raw/None session identity)."""
    session_id = "rr054-session-50-default"
    alias_model = lpe._CODEX_AUTO_AGENT_MODEL_ALIAS
    request = _minimal_request(
        headers=[(b"session_id", session_id.encode("utf-8"))]
    )
    body = {
        "model": alias_model,
        "litellm_metadata": {"session_id": session_id},
    }
    expected_session_key = lpe._resolve_codex_auto_agent_session_key(
        request,
        body,
        alias_model=alias_model,
    )
    assert expected_session_key is not None
    assert expected_session_key != session_id
    # With only session_id present, lane falls back to session:<id> (not bare id).
    assert expected_session_key.startswith(f"{session_id}:")

    captured_events: list[dict[str, Any]] = []

    def _capture_persist(events, **kwargs: Any) -> None:
        for event in events or []:
            if isinstance(event, dict):
                captured_events.append(event)

    with patch.object(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        side_effect=_capture_persist,
    ), patch.object(
        lpe, "_emit_auto_agent_alias_route_event", return_value=None
    ), patch(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        return_value=True,
    ):
        lpe._emit_auto_agent_alias_no_candidate_event(
            alias_family="codex_auto_agent",
            alias_model=alias_model,
            request=request,
            request_body=body,
            exc=HTTPException(
                status_code=429,
                detail={"candidates": [], "error": {"code": "all_unavailable"}},
            ),
            attempts=[
                {
                    "provider": "openai",
                    "model": "gpt-test",
                    "route_family": "codex_openai_responses",
                    "lane_key": expected_session_key.split(":", 1)[-1],
                    "error_class": "rate_limit",
                    "status": "cooldown_set",
                }
            ],
        )

    terminalish = [
        event
        for event in captured_events
        if event.get("event_type") == "no_candidate_available"
        or event.get("terminal_outcome") == "agent_session_terminated"
        or event.get("fallback_result") == "no_candidate_available"
    ]
    assert terminalish, f"expected terminal no-candidate event; got {captured_events!r}"
    for event in terminalish:
        assert event.get("session_key") == expected_session_key, (
            "RR-054 #50: terminal no-candidate event must carry lane-scoped "
            f"session_key={expected_session_key!r}, not raw/None; event={event!r}"
        )


def test_rr054_issue50_resolve_helpers_produce_lane_scoped_keys() -> None:
    """Sanity: resolve helpers are the production source of truth for #50."""
    session_id = "sess-lane"
    request = _minimal_request(
        headers=[
            (b"session_id", session_id.encode("utf-8")),
            (b"authorization", b"Bearer tok-rr054"),
        ]
    )
    body = {"litellm_metadata": {"session_id": session_id}}

    codex_key = lpe._resolve_codex_auto_agent_session_key(
        request, body, alias_model=lpe._CODEX_AUTO_AGENT_MODEL_ALIAS
    )
    assert codex_key is not None
    assert codex_key.startswith(f"{session_id}:")
    assert codex_key != session_id
    assert "auth:" in codex_key

    non_default = lpe._resolve_codex_auto_agent_session_key(
        request, body, alias_model="aawm-code"
    )
    assert non_default is not None
    assert non_default.startswith("aawm-code:")
    assert session_id in non_default

    # No credentials / no session header → no key (must not invent raw None alias).
    empty_request = _minimal_request()
    assert (
        lpe._resolve_codex_auto_agent_session_key(empty_request, {}, alias_model="aawm-code")
        is None
    )
