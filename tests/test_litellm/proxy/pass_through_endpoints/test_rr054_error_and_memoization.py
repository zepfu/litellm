"""RR-054 #23 / #29 error classification + request-scoped memoization regressions.

Finding #23: malformed adapted custom-tool-call classification must surface as a
non-rate-limit 502 (Bad Gateway / upstream invalid payload), not 429 rate limit.

Finding #29: host / repository / client / dispatch extraction and prior-tool
activity summarization must execute once across multiple audit events built for
the same request (request-scoped memoization).

Tests only. Failures here document remaining production gaps.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import Request
from fastapi.responses import Response

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_request(
    path: str = "/openai_passthrough/v1/responses",
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] = ("127.0.0.1", 12345),
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
        "client": client,
        "server": ("test", 80),
    }
    return Request(scope)


def _apply_patch_custom_tool() -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch to files in the workspace.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: /.+/",
        },
    }


def _malformed_adapted_custom_tool_response_body() -> dict[str, Any]:
    return {
        "id": "resp_rr054_23_adapted",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_apply_patch",
                "name": "apply_patch",
                # Non-string input triggers adapted-custom restore failure.
                "arguments": '{"input": 123}',
            }
        ],
    }


def _request_body_with_prior_tools_and_identity() -> dict[str, Any]:
    return {
        "model": "aawm-code",
        "instructions": "You are a 'worker' agent.\nContinue.",
        "input": [
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_prior_1",
                "arguments": '{"cmd": "ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_prior_1",
                "output": "ok",
            },
        ],
        "tools": [_apply_patch_custom_tool()],
        "litellm_metadata": {
            "session_id": "sess-rr054-29",
            "repository": "zepfu/litellm",
            "repo_name": "should-not-override-repository-key",
            "agent_id": "agent-rr054-29",
            "agent_name": "worker",
            "agent_role": "worker",
            "dispatch_id": "dispatch-rr054-29",
            "redispatch_ordinal": 2,
            "thread_source": "subagent",
            "client_name_version": "codex-cli/0.142.5",
            "litellm_call_id": "call-rr054-29",
            "trace_id": "trace-rr054-29",
        },
    }


def _selection_and_attempts_for_multi_event_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = {
        "candidate": {
            "provider": "xai",
            "model": "xai/grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
            "last_resort": False,
        },
        "cooldown_key": "xai:xai/grok-4.5:lane",
        "lane_key": "lane-xai",
        "selection_reason": "first_available",
        "session_key": "sess-rr054-29",
        "skipped": [
            {
                "provider": "openai",
                "model": "gpt-5.3-codex-spark",
                "route_family": "codex_responses",
                "lane_key": "lane-openai",
                "reason": "cooldown",
                "cooldown_seconds": 12.0,
                "last_resort": False,
            }
        ],
    }
    attempts = [
        {
            "provider": "xai",
            "model": "xai/grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
            "lane_key": "lane-xai",
            "status": "cooldown_set",
            "error_class": "rate_limited",
            "error_status_code": 429,
            "error_code": "rate_limit_error",
            "error_tokens": ["rate_limit_error"],
            "cooldown_key": "xai:xai/grok-4.5:lane",
            "cooldown_seconds": 30,
            "attempted_provider_call": True,
            "reason": "first_available",
        },
        {
            "provider": "xai",
            "model": "xai/grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
            "lane_key": "lane-xai",
            "status": "terminal_in_flight_cooldown_set",
            "error_class": "malformed_tool_call_text",
            "error_status_code": 502,
            "error_code": "aawm_auto_agent_malformed_tool_call_text",
            "error_tokens": ["aawm_auto_agent_malformed_tool_call_text"],
            "cooldown_key": "xai:xai/grok-4.5:lane",
            "cooldown_seconds": 45,
            "attempted_provider_call": True,
            "reason": "affinity",
        },
    ]
    return selection, attempts


# ---------------------------------------------------------------------------
# #23 malformed adapted custom tool-call classification
# ---------------------------------------------------------------------------


def test_rr054_issue23_malformed_adapted_custom_tool_raise_is_non_rate_limit_502() -> None:
    """Direct raise helper must not classify adapted custom-tool failures as 429.

    Sibling #23 helpers for malformed marker text / empty success / failed
    Responses already use non-rate-limit 502. Adapted custom-tool restore
    failures must match that contract so alias cooldown/redispatch logic does
    not treat a payload-shape bug as capacity.
    """
    adapter_error = {
        "name": "apply_patch",
        "output_index": 0,
        "reason": "input_not_string",
    }
    with pytest.raises(ProxyException) as exc_info:
        lpe._raise_codex_auto_agent_malformed_adapted_custom_tool_call(
            response_body=_malformed_adapted_custom_tool_response_body(),
            adapter_model="grok-build",
            adapter="codex_auto_agent_xai_oauth_responses",
            adapter_label="xAI OAuth",
            adapter_error=adapter_error,
        )

    exc = exc_info.value
    # Status / OpenAI-mapped code must be 502, never capacity 429.
    assert exc.code in {502, "502"}
    assert int(exc.code) == 502
    assert int(exc.code) != 429
    assert exc.type != "rate_limit_error"
    assert exc.type in {"upstream_error", "invalid_request_error", "bad_gateway"}

    detail = exc.detail
    assert isinstance(detail, dict)
    error = detail.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "aawm_auto_agent_malformed_tool_call_text"
    assert error.get("status") == "RESPONSES_MALFORMED_TOOL_CALL"
    assert error.get("type") != "rate_limit_error"
    assert error.get("type") == exc.type
    assert detail.get("diagnostic", {}).get("custom_tool_function_adapter_error") == (adapter_error)


@pytest.mark.parametrize(
    "arguments",
    [
        "not-json",
        '{"input": 123}',
        '{"input": "patch", "extra": true}',
        42,  # non-string arguments
    ],
)
def test_rr054_issue23_restore_adapter_error_paths_feed_non_rate_limit_raise(
    arguments: Any,
) -> None:
    """Restore failure reasons must surface through the #23 raise helper contract."""
    response_body = {
        "id": "resp_rr054_23_param",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_apply_patch",
                "name": "apply_patch",
                "arguments": arguments,
            }
        ],
    }
    request_body = {
        "model": "oa_xai/grok-build",
        "tools": [_apply_patch_custom_tool()],
    }

    restored_body, restored_count, adapter_error = lpe._restore_adapted_custom_tool_calls_in_response_body(
        response_body,
        request_body=request_body,
        adapter_model="grok-build",
    )
    assert restored_body is response_body
    assert restored_count == 0
    assert isinstance(adapter_error, dict)
    assert adapter_error.get("name") == "apply_patch"
    assert adapter_error.get("reason")

    with pytest.raises(ProxyException) as exc_info:
        lpe._raise_codex_auto_agent_malformed_adapted_custom_tool_call(
            response_body=response_body,
            adapter_model="grok-build",
            adapter="codex_auto_agent_xai_oauth_responses",
            adapter_label="xAI OAuth",
            adapter_error=adapter_error,
        )
    assert int(exc_info.value.code) == 502
    assert exc_info.value.type != "rate_limit_error"


@pytest.mark.asyncio
async def test_rr054_issue23_validate_payload_malformed_adapted_custom_is_502_not_429() -> None:
    """End-to-end validate path must preserve non-rate-limit classification."""
    upstream = Response(
        content=json.dumps(_malformed_adapted_custom_tool_response_body()),
        media_type="application/json",
    )
    with pytest.raises(ProxyException) as exc_info:
        await lpe._validate_codex_auto_agent_responses_payload(
            upstream,
            adapter_model="grok-build",
            adapter="codex_auto_agent_xai_oauth_responses",
            adapter_label="xAI OAuth",
            request_body={
                "model": "oa_xai/grok-build",
                "tools": [_apply_patch_custom_tool()],
            },
        )

    exc = exc_info.value
    assert int(exc.code) == 502
    assert int(exc.code) != 429
    assert exc.type != "rate_limit_error"
    detail = exc.detail
    assert isinstance(detail, dict)
    assert detail["error"]["code"] == "aawm_auto_agent_malformed_tool_call_text"
    assert detail["error"]["type"] != "rate_limit_error"
    assert detail["diagnostic"]["custom_tool_function_adapter_error"] == {
        "name": "apply_patch",
        "output_index": 0,
        "reason": "input_not_string",
    }


def test_rr054_issue23_adapted_custom_matches_sibling_non_rate_limit_contract() -> None:
    """Adapted custom raise must align with other #23 non-capacity raise helpers."""
    with pytest.raises(ProxyException) as text_exc:
        lpe._raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body={"output": []},
            adapter_model="m",
            adapter="a",
            adapter_label="L",
        )
    with pytest.raises(ProxyException) as adapted_exc:
        lpe._raise_codex_auto_agent_malformed_adapted_custom_tool_call(
            response_body=_malformed_adapted_custom_tool_response_body(),
            adapter_model="m",
            adapter="a",
            adapter_label="L",
            adapter_error={"name": "apply_patch", "output_index": 0, "reason": "x"},
        )

    assert int(text_exc.value.code) == 502
    assert text_exc.value.type != "rate_limit_error"
    # Contract under test: adapted custom path must match sibling status class.
    assert int(adapted_exc.value.code) == int(text_exc.value.code)
    assert adapted_exc.value.type != "rate_limit_error"
    assert adapted_exc.value.type == text_exc.value.type or adapted_exc.value.type in {
        "upstream_error",
        "invalid_request_error",
        "bad_gateway",
    }


# ---------------------------------------------------------------------------
# #29 request-scoped memoization across multi-event audit builds
# ---------------------------------------------------------------------------


def test_rr054_issue29_prior_tool_activity_summary_once_across_multi_event_audit() -> None:
    """Prior-tool walk must run once per request when many audit events need it.

    RR-054 #29: terminal multi-event audit builds (skipped + retryable +
    redispatch) must not re-walk the full request body for each event.
    """
    request = _minimal_request()
    request_body = _request_body_with_prior_tools_and_identity()
    selection, attempts = _selection_and_attempts_for_multi_event_audit()

    real_summarize = lpe._summarize_auto_agent_alias_actual_prior_tool_activity
    summarize_calls: list[int] = []

    def _counting_summarize(body: dict[str, Any]) -> dict[str, Any]:
        summarize_calls.append(id(body))
        return real_summarize(body)

    with patch.object(
        lpe,
        "_summarize_auto_agent_alias_actual_prior_tool_activity",
        side_effect=_counting_summarize,
    ), patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_host_attribution",
        return_value={
            "client_ip": "100.64.0.10",
            "client_ip_source": "request_client",
            "host_name": "thoth",
            "host_name_source": "magicdns_reverse",
        },
    ):
        events = lpe._build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="aawm-code",
            request=request,
            request_body=request_body,
            selection=selection,
            attempts=attempts,
        )

    # At least one terminal/retryable event should request activity.
    activity_events = [e for e in events if isinstance(e, dict) and "actual_prior_tool_activity_summary" in e]
    assert len(events) >= 2
    assert len(activity_events) >= 1
    # Request-scoped memoization: summarize once even when multiple events need it.
    assert len(summarize_calls) == 1, (
        "RR-054 #29 gap: prior-tool activity summarized "
        f"{len(summarize_calls)} times across {len(activity_events)} "
        "activity-bearing audit events on one request"
    )
    for event in activity_events:
        summary = event["actual_prior_tool_activity_summary"]
        assert summary["has_actual_prior_tool_activity"] is True
        assert summary["prior_tool_call_count"] == 1
        assert "exec_command" in summary["prior_tool_names"]


def test_rr054_issue29_host_repository_client_dispatch_extract_once_per_request() -> None:
    """Host / repository / client / dispatch extraction once across multi-event audit.

    RR-054 #29: building N audit events for one request must not re-run the
    expensive identity extractors N times.
    """
    request = _minimal_request(
        headers=[
            (b"user-agent", b"codex-cli/0.142.5"),
            (b"session_id", b"sess-rr054-29"),
        ]
    )
    request_body = _request_body_with_prior_tools_and_identity()
    selection, attempts = _selection_and_attempts_for_multi_event_audit()

    host_calls = {"n": 0}
    client_calls = {"n": 0}
    dispatch_calls = {"n": 0}
    repository_lookups = {"n": 0}

    real_client = lpe._extract_auto_agent_alias_client_product_label
    real_dispatch = lpe._extract_auto_agent_alias_agent_dispatch_fields
    real_meta = lpe._extract_auto_agent_alias_metadata_value

    def _counting_host(req: Request) -> dict[str, Any]:
        host_calls["n"] += 1
        # Keep attribution deterministic without real reverse DNS.
        return {
            "client_ip": "100.64.0.10",
            "client_ip_source": "request_client",
            "host_name": "thoth",
            "host_name_source": "magicdns_reverse",
        }

    def _counting_client(req: Request, body: dict[str, Any]) -> Any:
        client_calls["n"] += 1
        return real_client(req, body)

    def _counting_dispatch(req: Request, body: dict[str, Any]) -> dict[str, Any]:
        dispatch_calls["n"] += 1
        return real_dispatch(req, body)

    def _counting_meta(body: dict[str, Any], *keys: str, **kwargs: Any) -> Any:
        if keys and keys[0] in {
            "repository",
            "repo",
            "repo_name",
            "repository_name",
        }:
            repository_lookups["n"] += 1
        return real_meta(body, *keys, **kwargs)

    with patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_host_attribution",
        side_effect=_counting_host,
    ), patch.object(
        lpe,
        "_extract_auto_agent_alias_client_product_label",
        side_effect=_counting_client,
    ), patch.object(
        lpe,
        "_extract_auto_agent_alias_agent_dispatch_fields",
        side_effect=_counting_dispatch,
    ), patch.object(
        lpe,
        "_extract_auto_agent_alias_metadata_value",
        side_effect=_counting_meta,
    ):
        events = lpe._build_auto_agent_alias_audit_events(
            alias_family="codex",
            alias_model="aawm-code",
            request=request,
            request_body=request_body,
            selection=selection,
            attempts=attempts,
        )

    assert len(events) >= 3  # 1 skipped + 2 attempts
    event_count = len(events)

    assert host_calls["n"] == 1, (
        "RR-054 #29 gap: host attribution ran "
        f"{host_calls['n']} times for {event_count} audit events on one request "
        f"(expected 1)"
    )
    assert client_calls["n"] == 1, (
        "RR-054 #29 gap: client product extraction ran "
        f"{client_calls['n']} times for {event_count} audit events on one request "
        f"(expected 1)"
    )
    assert dispatch_calls["n"] == 1, (
        "RR-054 #29 gap: agent/dispatch extraction ran "
        f"{dispatch_calls['n']} times for {event_count} audit events on one request "
        f"(expected 1)"
    )
    # Repository is pulled via metadata helper at least once per event today;
    # request-scoped memo should collapse that to a single effective extract.
    assert repository_lookups["n"] == 1, (
        "RR-054 #29 gap: repository metadata extract ran "
        f"{repository_lookups['n']} times for {event_count} audit events on one "
        f"request (expected 1)"
    )

    # Sanity: events still carry identity fields after memoized extract.
    for event in events:
        assert event.get("repository") == "zepfu/litellm"
        assert event.get("client_product_label") in {
            "Codex/0.142.5",
            "codex-cli/0.142.5",
            "Codex",
        } or (isinstance(event.get("client_product_label"), str) and "Codex" in event["client_product_label"])
        assert event.get("host_name") == "thoth"
        assert event.get("agent_name") == "worker"
        assert event.get("dispatch_id") == "dispatch-rr054-29"


def test_rr054_issue29_attach_terminal_context_reuses_request_scoped_prior_summary() -> None:
    """Repeated terminal-context attaches on one request reuse prior-tool memo."""
    request = _minimal_request()
    request_body = _request_body_with_prior_tools_and_identity()

    real_summarize = lpe._summarize_auto_agent_alias_actual_prior_tool_activity
    summarize_calls = {"n": 0}

    def _counting_summarize(body: dict[str, Any]) -> dict[str, Any]:
        summarize_calls["n"] += 1
        return real_summarize(body)

    with patch.object(
        lpe,
        "_summarize_auto_agent_alias_actual_prior_tool_activity",
        side_effect=_counting_summarize,
    ), patch.object(
        lpe,
        "_extract_auto_agent_alias_agent_dispatch_fields",
        wraps=lpe._extract_auto_agent_alias_agent_dispatch_fields,
    ) as dispatch_mock:
        for event_type in (
            "no_candidate_available",
            "redispatch_required",
            "agent_session_terminated",
        ):
            event: dict[str, Any] = {"event_type": event_type}
            lpe._attach_auto_agent_alias_terminal_context_fields(
                event,
                request=request,
                request_body=request_body,
                include_activity_status=True,
            )
            assert event["actual_prior_tool_activity_summary"]["has_actual_prior_tool_activity"] is True
            assert event["terminal_activity_status"] == ("failed_after_partial_activity")

    assert summarize_calls["n"] == 1, (
        "RR-054 #29 gap: prior-tool activity summarized "
        f"{summarize_calls['n']} times across 3 terminal attach calls on one request"
    )
    assert dispatch_mock.call_count == 1, (
        "RR-054 #29 gap: agent/dispatch extraction ran "
        f"{dispatch_mock.call_count} times across 3 terminal attach calls "
        "(expected 1)"
    )


def test_rr054_issue29_build_audit_event_reuses_host_attribution_on_repeated_calls() -> None:
    """Single-event builder still memoizes host attribution for the request.

    When callers emit multiple individual `_build_auto_agent_alias_audit_event`
    calls for the same Request (common for incremental attempt logging), host
    attribution must not re-resolve per event.
    """
    request = _minimal_request()
    request_body = _request_body_with_prior_tools_and_identity()
    host_calls = {"n": 0}

    def _counting_host(req: Request) -> dict[str, Any]:
        host_calls["n"] += 1
        return {
            "client_ip": "10.0.0.8",
            "client_ip_source": "test",
            "host_name": "seshat",
            "host_name_source": "test",
        }

    candidate = {
        "provider": "openai",
        "model": "gpt-test",
        "route_family": "codex_responses",
        "last_resort": False,
    }
    selection = {
        "lane_key": "lane-a",
        "session_key": "sess-rr054-29b",
        "candidate": candidate,
    }

    with patch.object(
        lpe,
        "_resolve_auto_agent_alias_route_host_attribution",
        side_effect=_counting_host,
    ):
        for i in range(3):
            event = lpe._build_auto_agent_alias_audit_event(
                alias_family="codex",
                alias_model="aawm-code",
                request=request,
                request_body=request_body,
                selection=selection,
                candidate=candidate,
                event_type="candidate_retryable_failure",
                candidate_status="cooldown_set",
                attempt_number=i + 1,
                failure_class="rate_limited",
                error_status_code=429,
            )
            assert event["host_name"] == "seshat"
            assert event["repository"] == "zepfu/litellm"

    assert host_calls["n"] == 1, (
        "RR-054 #29 gap: host attribution ran " f"{host_calls['n']} times across 3 audit-event builds on one request"
    )
