import asyncio
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, UploadFile
from starlette.datastructures import Headers, QueryParams
from starlette.datastructures import UploadFile as StarletteUploadFile

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import (
    AawmErrorLogFileHandler,
    _AAWM_ERROR_LOG_CONTEXT_FIELDS,
    _build_aawm_error_log_record,
    AawmRouteAccessLogReplacementFilter,
    clear_aawm_route_access_log_replacements,
    get_egress_guard_alert_state,
    register_aawm_route_access_log_replacement,
    reset_egress_guard_alert_state,
    verbose_aawm_route_logger,
    verbose_proxy_logger,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    flush_aawm_route_rollups,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import success_handler
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    _build_passthrough_input_item_shape_samples,
    build_aawm_route_access_log_line,
    _build_passthrough_request_shape_failure_request_payload,
    _classify_passthrough_request_shape_deserialization_422,
    _direct_capture_xai_passthrough_failure,
    _enrich_passthrough_error_log_context_for_request_shape_422,
    _sanitize_passthrough_request_shape_error_preview,
    _execute_passthrough_pre_first_byte_with_hidden_retries,
    _get_passthrough_handled_http_error_summary,
    _get_passthrough_hidden_retry_budget_seconds,
    _headers_for_json_passthrough_egress,
    _is_aawm_agent_identity_registered_in_litellm_callbacks,
    _is_known_grok_billing_passthrough_timeout_cancel_response,
    _is_known_grok_personal_team_spending_limit_response,
    _get_passthrough_grok_personal_team_spending_limit_failure_kind,
    _is_known_grok_build_usage_balance_exhausted_response,
    _get_passthrough_grok_build_usage_balance_exhausted_failure_kind,
    _is_known_chatgpt_codex_model_not_supported_for_account_response,
    _restore_responses_function_names_in_sse_chunks,
    _get_passthrough_chatgpt_codex_model_not_supported_failure_kind,
    _record_grok_billing_passthrough_request_contract,
    _record_passthrough_hidden_retry_metadata,
    _should_log_passthrough_terminal_failure_without_traceback,
    emit_aawm_route_access_log,
    pass_through_request,
)
from litellm.responses.function_name_sanitization import (
    sanitize_responses_function_names,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _build_aawm_route_log_request(
    *,
    method: str = "POST",
    url: str = "http://127.0.0.1:4001/anthropic/v1/messages?beta=true",
    client: tuple[str, int] = ("172.19.0.1", 52834),
    http_version: str = "1.1",
    headers: Optional[dict[str, str]] = None,
) -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = method
    request.url = url
    request.headers = Headers(headers or {})
    query = httpx.URL(url).query
    request.scope = {
        "type": "http",
        "method": method,
        "path": httpx.URL(url).path,
        "query_string": query if isinstance(query, bytes) else query.encode("utf-8"),
        "client": client,
        "http_version": http_version,
    }
    return request


def test_request_shape_422_modelinput_enrichment_is_sanitized_and_fingerprinted():
    secret_prompt = "SUPER_SECRET_MODELINPUT_PROMPT"
    secret_token = "sk-leaked-modelinput-token"
    base_context = {
        "source": "pass_through_endpoint",
        "endpoint": "/openai_passthrough/responses",
        "upstream_url": "https://chatgpt.com/backend-api/codex/responses",
        "provider": "openai",
        "model": "gpt-5.5",
        "model_alias": "aawm-code",
        "route_family": "codex_responses",
        "status_code": 422,
        "auth_credential_source": "route_custom_header",
        "auth_header_names": ["authorization"],
        "auth_header_sources": ["route_custom_header:authorization"],
        "aawm_passthrough_body_container_type": "object",
        "aawm_passthrough_body_top_level_keys": ["input", "model", "tools"],
        "aawm_passthrough_input_container_type": "array",
        "aawm_passthrough_input_item_count": 3,
        "aawm_passthrough_input_item_type_counts": {
            "message": 1,
            "function_call": 1,
            "function_call_output": 1,
        },
        "aawm_passthrough_input_item_shape_samples": [
            {
                "index": 0,
                "container_type": "object",
                "type": "message",
                "keys": ["content", "role", "type"],
            },
            {
                "index": 1,
                "container_type": "object",
                "type": "function_call",
                "keys": ["arguments", "call_id", "name", "type"],
            },
            {
                "index": 2,
                "container_type": "object",
                "type": "function_call_output",
                "keys": ["call_id", "output", "type"],
            },
        ],
        "aawm_passthrough_tool_count": 2,
        "aawm_passthrough_tool_type_counts": {"function": 2},
    }
    exc = HTTPException(
        status_code=422,
        detail=(
            '{"error":"Failed to deserialize the JSON body into the target type: '
            f"data did not match any variant of untagged enum ModelInput; prompt={secret_prompt}; "
            f'token={secret_token}"}}'
        ),
    )

    enriched = _enrich_passthrough_error_log_context_for_request_shape_422(
        error_log_context=dict(base_context),
        exc=exc,
    )

    assert enriched["failure_kind"] == "request_shape_deserialization_failed"
    assert (
        enriched["aawm_passthrough_request_shape_error_class"]
        == "request_shape_deserialization_failed"
    )
    assert (
        enriched["aawm_passthrough_request_shape_error_message_class"]
        == "model_input_deserialization_failed"
    )
    assert "ModelInput" in enriched["aawm_passthrough_request_shape_error_body_preview"]
    assert enriched["aawm_passthrough_request_shape_summary"]["input_item_count"] == 3
    assert enriched["aawm_passthrough_request_shape_summary"][
        "input_item_shape_samples"
    ][-1] == {
        "index": 2,
        "container_type": "object",
        "type": "function_call_output",
        "keys": ["call_id", "output", "type"],
    }
    assert enriched["aawm_passthrough_request_shape_fingerprint"]
    assert enriched["aawm_passthrough_request_shape_error_fingerprint"]

    repeat = _enrich_passthrough_error_log_context_for_request_shape_422(
        error_log_context=dict(base_context),
        exc=exc,
    )
    assert (
        repeat["aawm_passthrough_request_shape_fingerprint"]
        == enriched["aawm_passthrough_request_shape_fingerprint"]
    )
    assert (
        repeat["aawm_passthrough_request_shape_error_fingerprint"]
        == enriched["aawm_passthrough_request_shape_error_fingerprint"]
    )

    serialized = json.dumps(enriched)
    assert secret_prompt not in serialized
    assert secret_token not in serialized
    assert "sk-leaked" not in serialized


def test_request_shape_422_classification_skips_non_responses_route():
    context = {
        "endpoint": "/anthropic/v1/messages",
        "upstream_url": "https://api.anthropic.com/v1/messages",
        "status_code": 422,
    }
    exc = HTTPException(
        status_code=422,
        detail=(
            '{"error":"Failed to deserialize the JSON body into the target type: '
            'data did not match any variant of untagged enum ModelInput"}'
        ),
    )

    assert (
        _classify_passthrough_request_shape_deserialization_422(
            status_code=422,
            error_log_context=context,
            exc=exc,
        )
        is None
    )


def test_request_shape_422_classification_skips_unrelated_responses_422():
    context = {
        "endpoint": "/openai_passthrough/responses",
        "upstream_url": "https://chatgpt.com/backend-api/codex/responses",
        "status_code": 422,
    }
    exc = HTTPException(
        status_code=422, detail='{"error":"validation failed for stream"}'
    )

    assert (
        _classify_passthrough_request_shape_deserialization_422(
            status_code=422,
            error_log_context=context,
            exc=exc,
        )
        is None
    )


def test_aawm_error_log_context_allowlist_preserves_request_shape_fields(monkeypatch):
    """Structural request-shape fields persist by default; body-preview needs opt-in."""
    from litellm._logging import (
        _AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS,
        _AAWM_ERROR_LOG_DEFAULT_CONTEXT_FIELDS,
    )

    structural_fields = {
        "aawm_passthrough_request_shape_error_class",
        "aawm_passthrough_request_shape_error_message_class",
        "aawm_passthrough_request_shape_summary",
        "aawm_passthrough_request_shape_fingerprint",
        "aawm_passthrough_request_shape_error_fingerprint",
        "aawm_passthrough_input_item_shape_samples",
        "failure_kind",
    }
    content_fields = {
        "aawm_passthrough_request_shape_error_body_preview",
    }

    # Full catalog still names content-bearing fields for discovery/opt-in.
    assert structural_fields.issubset(set(_AAWM_ERROR_LOG_CONTEXT_FIELDS))
    assert content_fields.issubset(set(_AAWM_ERROR_LOG_CONTEXT_FIELDS))
    assert content_fields.issubset(set(_AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS))
    assert not content_fields.intersection(set(_AAWM_ERROR_LOG_DEFAULT_CONTEXT_FIELDS))

    record = logging.LogRecord(
        name="litellm.proxy",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="request-shape failure",
        args=(),
        exc_info=None,
    )
    for field_name in structural_fields | content_fields:
        setattr(record, field_name, field_name)

    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", raising=False)
    default_context = _build_aawm_error_log_record(
        record, formatter=logging.Formatter()
    )["context"]
    for field_name in structural_fields:
        assert default_context[field_name] == field_name
    for field_name in content_fields:
        assert field_name not in default_context

    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", "1")
    opt_in_context = _build_aawm_error_log_record(
        record, formatter=logging.Formatter()
    )["context"]
    for field_name in structural_fields | content_fields:
        assert opt_in_context[field_name] == field_name


def test_passthrough_input_item_shape_samples_capture_head_and_tail_without_values():
    secret_prompt = "PROMPT_VALUE_MUST_NOT_APPEAR"
    secret_arguments = "TOOL_ARGUMENTS_MUST_NOT_APPEAR"
    input_items = [
        {"type": "message", "role": "user", "content": secret_prompt},
        *[
            {
                "type": "message",
                "role": "assistant",
                "content": f"middle-{index}",
            }
            for index in range(30)
        ],
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "call_1",
            "arguments": secret_arguments,
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "SECRET_OUTPUT_MUST_NOT_APPEAR",
        },
    ]

    samples = _build_passthrough_input_item_shape_samples(
        input_items,
        max_samples=6,
    )

    assert [sample["index"] for sample in samples] == [0, 1, 2, 30, 31, 32]
    assert samples[-2] == {
        "index": 31,
        "container_type": "object",
        "type": "function_call",
        "keys": ["arguments", "call_id", "name", "type"],
    }
    assert samples[-1] == {
        "index": 32,
        "container_type": "object",
        "type": "function_call_output",
        "keys": ["call_id", "output", "type"],
    }
    serialized = json.dumps(samples)
    assert secret_prompt not in serialized
    assert secret_arguments not in serialized
    assert "SECRET_OUTPUT" not in serialized


def test_sanitize_passthrough_request_shape_error_preview_redacts_secrets():
    raw = (
        "Failed to deserialize the JSON body into the target type: "
        "data did not match any variant of untagged enum ModelInput; "
        "prompt=SUPER_SECRET_PROMPT; token=sk-leaked-token-abc; "
        "authorization=Bearer secret; api_key=mykey123; "
        '"input": "JSON_INPUT_LEAK", "content": "JSON_CONTENT_LEAK", '
        '"arguments": {"tool": "TOOL_ARG_LEAK"}, "output": "OUTPUT_LEAK"'
    )
    preview = _sanitize_passthrough_request_shape_error_preview(raw)
    assert preview is not None
    assert "ModelInput" in preview or "deserialize" in preview.lower()
    assert "SUPER_SECRET" not in preview
    assert "sk-leaked" not in preview
    assert "mykey123" not in preview
    assert "JSON_INPUT_LEAK" not in preview
    assert "JSON_CONTENT_LEAK" not in preview
    assert "TOOL_ARG_LEAK" not in preview
    assert "OUTPUT_LEAK" not in preview
    assert "prompt" not in preview.lower() or "[REDACTED]" in preview
    assert "api_key" not in preview.lower() or "[REDACTED]" in preview
    assert "[REDACTED]" in preview

    bearer_preview = _sanitize_passthrough_request_shape_error_preview(
        "Failed to deserialize the JSON body into the target type: "
        "data did not match any variant of untagged enum ModelInput; Bearer abc.def"
    )
    assert bearer_preview is not None
    assert "Bearer abc.def" not in bearer_preview
    assert "[REDACTED]" in bearer_preview


def test_build_request_shape_failure_request_payload_omits_raw_body_and_secrets():
    secret_prompt = "RAW_PROMPT_LEAK"
    secret_tool = "tool_output_secret_xyz"
    error_log_context = {
        "endpoint": "/openai_passthrough/responses",
        "upstream_url": "https://chatgpt.com/backend-api/codex/responses",
        "provider": "openai",
        "model": "gpt-5.5",
        "model_alias": "aawm-code",
        "route_family": "codex_responses",
        "status_code": 422,
        "litellm_call_id": "call-shape-422",
        "failure_kind": "request_shape_deserialization_failed",
        "aawm_passthrough_request_shape_error_class": "request_shape_deserialization_failed",
        "aawm_passthrough_request_shape_error_message_class": "model_input_deserialization_failed",
        "aawm_passthrough_request_shape_error_body_preview": "ModelInput variant mismatch",
        "aawm_passthrough_input_item_count": 2,
    }
    kwargs = {
        "call_type": "pass_through_endpoint",
        "litellm_call_id": "call-shape-422",
        "passthrough_logging_payload": {
            "url": "https://chatgpt.com/backend-api/codex/responses",
            "request_method": "POST",
            "cost_per_request": 0.01,
            "start_time": "2026-06-27T12:00:00Z",
            "end_time": "2026-06-27T12:00:01Z",
            "request_body": {
                "input": [{"type": "message", "content": secret_prompt}],
                "tools": [{"type": "function", "name": "x", "output": secret_tool}],
                "authorization": "Bearer sk-hook-leak",
            },
            "request_headers": {"authorization": "Bearer sk-header-leak"},
            "response_body": {"output": secret_tool, "prompt": secret_prompt},
            "raw_body": secret_prompt,
            "body": {"content": secret_prompt},
            "headers": {"authorization": "Bearer sk-header-leak"},
            "extra": {"prompt": secret_prompt, "tool": secret_tool},
        },
        "api_key": "sk-kwargs-leak",
        "authorization": "Bearer kwargs-auth",
    }

    payload = _build_passthrough_request_shape_failure_request_payload(
        error_log_context=error_log_context,
        kwargs=kwargs,
        custom_llm_provider="openai",
    )

    serialized = json.dumps(payload)
    assert secret_prompt not in serialized
    assert secret_tool not in serialized
    assert "sk-hook-leak" not in serialized
    assert "sk-header-leak" not in serialized
    assert "sk-kwargs-leak" not in serialized
    assert "kwargs-auth" not in serialized
    assert "input" not in serialized or "RAW_PROMPT" not in serialized
    assert payload.get("model") == "gpt-5.5"
    assert payload.get("custom_llm_provider") == "openai"
    assert payload["failure_kind"] == "request_shape_deserialization_failed"
    assert payload["litellm_params"]["metadata"]["route_family"] == "codex_responses"
    logging_payload = payload.get("passthrough_logging_payload", {})
    assert set(logging_payload.keys()) <= {
        "url",
        "request_method",
        "cost_per_request",
        "start_time",
        "end_time",
    }
    assert logging_payload.get("url") == (
        "https://chatgpt.com/backend-api/codex/responses"
    )
    assert logging_payload.get("request_method") == "POST"
    assert "request_body" not in logging_payload
    assert "response_body" not in logging_payload
    assert "raw_body" not in logging_payload
    assert "body" not in logging_payload
    assert "headers" not in logging_payload
    assert "extra" not in logging_payload


@pytest.mark.asyncio
async def test_pass_through_request_shape_422_failure_hook_payload_is_sanitized():
    secret_prompt = "HOOK_SECRET_PROMPT_VALUE"
    secret_token = "sk-hook-failure-token"
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4001/openai_passthrough/responses"
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams({})

    target_url = "https://chatgpt.com/backend-api/codex/responses"
    upstream_detail = (
        '{"error":"Failed to deserialize the JSON body into the target type: '
        f"data did not match any variant of untagged enum ModelInput; prompt={secret_prompt}; "
        f'token={secret_token}"}}'
    )
    upstream_response = httpx.Response(
        status_code=422,
        content=upstream_detail.encode("utf-8"),
        request=httpx.Request("POST", target_url),
    )
    request_body = {
        "model": "gpt-5.5",
        "input": [{"type": "message", "content": secret_prompt}],
        "tools": [{"type": "function", "name": "run", "output": "TOOL_OUT_LEAK"}],
        "litellm_metadata": {
            "route_family": "codex_responses",
            "inbound_model_alias": "aawm-code",
            "aawm_passthrough_body_container_type": "object",
            "aawm_passthrough_input_item_count": 1,
        },
    }

    post_failure_mock = AsyncMock()

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=MagicMock(client=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kw: kw["data"]),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.emit_aawm_route_access_log",
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
        new=AsyncMock(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ):
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": f"Bearer {secret_token}"},
                user_api_key_dict=MagicMock(),
                custom_body=request_body,
                custom_llm_provider="openai",
                stream=False,
            )

    assert exc_info.value.code == "422"
    post_failure_mock.assert_awaited_once()
    hook_kwargs = post_failure_mock.await_args.kwargs
    assert hook_kwargs["traceback_str"] is None
    request_data = hook_kwargs["request_data"]
    serialized = json.dumps(request_data)
    assert secret_prompt not in serialized
    assert secret_token not in serialized
    assert "TOOL_OUT_LEAK" not in serialized
    assert "sk-hook" not in serialized
    assert request_data.get("failure_kind") == "request_shape_deserialization_failed"
    assert "aawm_passthrough_request_shape_error_body_preview" in request_data or (
        request_data.get("litellm_params", {})
        .get("metadata", {})
        .get("aawm_passthrough_request_shape_error_body_preview")
    )


def test_headers_for_json_passthrough_egress_preserves_json_content_type():
    headers, removed_content_type = _headers_for_json_passthrough_egress(
        {
            "Content-Type": "application/json",
            "authorization": "Bearer token",
        }
    )

    assert removed_content_type is None
    assert headers["Content-Type"] == "application/json"
    assert headers["authorization"] == "Bearer token"


def test_headers_for_json_passthrough_egress_sets_missing_json_content_type():
    headers, removed_content_type = _headers_for_json_passthrough_egress(
        {"authorization": "Bearer token"}
    )

    assert removed_content_type is None
    assert headers["content-type"] == "application/json"
    assert headers["authorization"] == "Bearer token"


def test_headers_for_json_passthrough_egress_replaces_stale_non_json_content_type():
    headers, removed_content_type = _headers_for_json_passthrough_egress(
        {
            "Content-Type": "multipart/form-data; boundary=stale",
            "authorization": "Bearer token",
        }
    )

    assert removed_content_type == "multipart/form-data; boundary=stale"
    assert "Content-Type" not in headers
    assert headers["content-type"] == "application/json"
    assert headers["authorization"] == "Bearer token"


def _build_uvicorn_access_record(
    *,
    client_addr: str = "172.19.0.1:52834",
    method: str = "POST",
    full_path: str = "/anthropic/v1/messages?beta=true",
    http_version: str = "1.1",
    status_code: int = 200,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client_addr, method, full_path, http_version, status_code),
        exc_info=None,
    )


@contextmanager
def _capture_aawm_route_logs(caplog):
    logger = logging.getLogger("LiteLLM AAWM Route")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            yield
    finally:
        logger.removeHandler(caplog.handler)


def test_build_aawm_route_access_log_line_includes_available_context() -> None:
    request = _build_aawm_route_log_request(
        url=(
            "http://127.0.0.1:4001/anthropic/v1/messages"
            "?beta=true&api_key=should-not-log"
        ),
        headers={"user-agent": "claude-code/1.2.3 (darwin)"},
    )
    request_body = {
        "model": "grok-composer-2.5-fast",
        "litellm_metadata": {
            "agent_name": "W4 engineer",
            "agent_id": "019ed4f2",
            "repository": "dashboard-shell",
            "requested_model_alias": "aawm-code",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://chatgpt.com/backend-api/codex/responses?ignored=1",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line == (
        "20260614 19:31:05 [ROUTE] Claude/1.2.3 - "
        "W4 engineer#019ed4f2@dashboard-shell.grok-composer-2.5-fast(aawm-code) "
        "POST 172.19.0.1:52834 /anthropic/v1/messages?beta=true -> "
        "chatgpt.com/backend-api/codex/responses"
    )
    assert "api_key" not in line
    assert "ignored=1" not in line


def test_build_aawm_route_access_log_line_omits_missing_optional_context() -> None:
    request = _build_aawm_route_log_request(
        method="GET",
        url="http://127.0.0.1:4001/openai_passthrough/responses?token=secret",
        client=("127.0.0.1", 44780),
        http_version="2",
    )

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.openai.com/v1/responses?api_key=secret",
        request_body={},
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line == (
        "20260614 19:31:05 [ROUTE] GET 127.0.0.1:44780 "
        "/openai_passthrough/responses -> api.openai.com/v1/responses"
    )
    assert "secret" not in line
    assert "token" not in line
    assert "None" not in line


def test_build_aawm_route_access_log_line_omits_alias_when_same_as_model() -> None:
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "aawm-code",
        "litellm_metadata": {
            "requested_model_alias": "aawm-code",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.anthropic.com/v1/messages",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line == (
        "20260614 19:31:05 [ROUTE] - aawm-code "
        "POST 172.19.0.1:52834 /anthropic/v1/messages?beta=true -> "
        "api.anthropic.com/v1/messages"
    )
    assert "aawm-code(aawm-code)" not in line


def test_build_aawm_route_access_log_line_rejects_freeform_identity_metadata() -> None:
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "gpt-5.5",
        "litellm_metadata": {
            "agent_name": (
                "general-purpose@mypy fails, fix and retry. "
                "This is prompt-like prose"
            ),
            "repository": ("litellm; reuse_rule=safe for similar queue-shaping work"),
            "requested_model_alias": "aawm-code",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://chatgpt.com/backend-api/codex/responses",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line == (
        "20260614 19:31:05 [ROUTE] - gpt-5.5(aawm-code) "
        "POST 172.19.0.1:52834 /anthropic/v1/messages?beta=true -> "
        "chatgpt.com/backend-api/codex/responses"
    )
    assert "general-purpose@" not in line
    assert "reuse_rule" not in line
    assert "prompt-like" not in line


def test_build_aawm_route_access_log_line_rejects_freeform_identity_headers() -> None:
    request = _build_aawm_route_log_request(
        headers={
            "x-aawm-agent-name": "worker; now run this unrelated shell command",
            "x-aawm-repository": "mypy fails, fix and retry.",
        }
    )

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.openai.com/v1/responses",
        request_body={"model": "gpt-5.3-codex-spark"},
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line == (
        "20260614 19:31:05 [ROUTE] - gpt-5.3-codex-spark "
        "POST 172.19.0.1:52834 /anthropic/v1/messages?beta=true -> "
        "api.openai.com/v1/responses"
    )
    assert "unrelated shell command" not in line
    assert "mypy fails" not in line


def test_build_aawm_route_access_log_line_normalizes_repository_paths() -> None:
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "claude-sonnet-4-6",
        "litellm_metadata": {
            "agent_name": "W4 engineer",
            "repository": "/home/zepfu/projects/dashboard-shell/",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.anthropic.com/v1/messages",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] - W4 engineer@dashboard-shell.")
    assert "/home/zepfu/projects" not in line


def test_build_aawm_route_access_log_line_preserves_owner_repository_slug() -> None:
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "claude-sonnet-4-6",
        "litellm_metadata": {
            "agent_name": "W4 engineer",
            "repository": "zepfu/litellm",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.anthropic.com/v1/messages",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] - W4 engineer@zepfu/litellm.")


def test_build_aawm_route_access_log_line_uses_session_history_repo_aliases() -> None:
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "claude-sonnet-4-6",
        "workspaceRoot": "/home/zepfu/projects/dashboard-shell",
        "metadata": {
            "agent_name": "orchestrator",
            "trace_name": "claude-code.orchestrator",
            "trace_user_id": "should-not-win",
        },
    }

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.anthropic.com/v1/messages",
        request_body=request_body,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] - orchestrator@dashboard-shell.")
    assert "should-not-win" not in line


def test_build_aawm_route_access_log_line_uses_trace_user_repo_when_structured_repo_missing() -> (
    None
):
    request = _build_aawm_route_log_request(
        headers={
            "langfuse_trace_name": "claude-code.orchestrator",
            "langfuse_trace_user_id": "aegis",
        }
    )

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.anthropic.com/v1/messages",
        request_body={"model": "claude-opus-4-8"},
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] - aegis.claude-opus-4-8 ")


def test_build_aawm_route_access_log_line_uses_split_client_identity() -> None:
    request = _build_aawm_route_log_request(
        headers={
            "x-aawm-client-name": "codex-cli",
            "x-aawm-client-version": "0.119.0-alpha.29",
        }
    )

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.openai.com/v1/responses",
        request_body={"model": "gpt-5.3-codex-spark"},
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] Codex/0.119.0-alpha.29 - ")


def test_build_aawm_route_access_log_line_rejects_freeform_client_identity() -> None:
    request = _build_aawm_route_log_request(
        headers={
            "user-agent": "curl/8.0 this prompt-like extra text should not appear",
            "x-aawm-client": "codex; now run this unrelated command",
        }
    )

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.openai.com/v1/responses",
        request_body={"model": "gpt-5.3-codex-spark"},
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith("20260614 19:31:05 [ROUTE] curl/8.0 - ")
    assert "prompt-like" not in line
    assert "unrelated command" not in line


@pytest.mark.parametrize(
    ("route_type", "url", "expected_log_type"),
    (
        ("aembedding", "http://127.0.0.1:4001/v1/embeddings", "EMBED"),
        ("arerank", "http://127.0.0.1:4001/v1/rerank", "RERANK"),
        (None, "http://127.0.0.1:4001/v1/embeddings", "EMBED"),
        (None, "http://127.0.0.1:4001/v1/rerank", "RERANK"),
    ),
)
def test_build_aawm_route_access_log_line_classifies_route_type(
    route_type: Optional[str],
    url: str,
    expected_log_type: str,
) -> None:
    request = _build_aawm_route_log_request(url=url)

    line = build_aawm_route_access_log_line(
        request=request,
        target="https://api.openai.com/v1/responses",
        request_body={"model": "test-model"},
        route_type=route_type,
        now=datetime(2026, 6, 14, 19, 31, 5),
    )

    assert line.startswith(f"20260614 19:31:05 [{expected_log_type}] - test-model ")


def test_emit_aawm_route_access_log_is_scoped_once(caplog, monkeypatch) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
    request = _build_aawm_route_log_request(
        url="http://127.0.0.1:4001/openai_passthrough/responses?stream=false"
    )
    adapted_display_path = (
        "/openai_passthrough/responses?adapted_to=chatgpt.com/backend-api/codex/responses"
    )
    adapted_target = (
        "chatgpt.com/backend-api/codex/responses"
    )
    request.scope["_aawm_adapted_access_log_display_path"] = adapted_display_path
    request.scope["_aawm_adapted_access_log_target"] = adapted_target
    native_path = request.scope["path"]
    native_query_string = request.scope["query_string"]
    request_body = {
        "model": "gpt-5.3-codex-spark",
        "metadata": {"model_alias_label": "aawm-code-anthropic"},
    }

    with _capture_aawm_route_logs(caplog):
        emit_aawm_route_access_log(
            request=request,
            target="https://chatgpt.com/backend-api/codex/responses",
            request_body=request_body,
        )
        emit_aawm_route_access_log(
            request=request,
            target="https://chatgpt.com/backend-api/codex/responses",
            request_body=request_body,
        )

    assert request.scope["path"] == native_path
    assert request.scope["query_string"] == native_query_string
    assert request.scope["_aawm_adapted_access_log_target"] == adapted_target
    assert (
        request.scope["_aawm_adapted_access_log_display_path"]
        == adapted_display_path
    )

    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/v1/models",
            )
        )
        is True
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/openai_passthrough/responses?stream=false",
            )
        )
        is False
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/openai_passthrough/responses?stream=false",
            )
        )
        is True
    )
    route_records = [
        record.getMessage()
        for record in caplog.records
        if " [ROUTE] " in record.getMessage()
    ]
    assert len(route_records) == 1
    assert re.fullmatch(
        r"\d{8} \d{2}:\d{2}:\d{2} \[ROUTE\] - "
        r"gpt-5\.3-codex-spark\(aawm-code-anthropic\) "
        r"POST 172\.19\.0\.1:52834 /openai_passthrough/responses\?stream=false "
        r"-> chatgpt\.com/backend-api/codex/responses",
        route_records[0],
    )


def test_emit_aawm_route_access_log_refreshes_suppression_for_same_scope(
    caplog,
    monkeypatch,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
    request = _build_aawm_route_log_request()
    request_body = {
        "model": "gpt-5.3-codex-spark",
        "metadata": {"model_alias_label": "aawm-code-anthropic"},
    }

    with _capture_aawm_route_logs(caplog):
        emit_aawm_route_access_log(
            request=request,
            target="https://chatgpt.com/backend-api/codex/responses",
            request_body=request_body,
        )
        request.scope[
            "query_string"
        ] = b"beta=true -> cli-chat-proxy.grok.com/v1/responses"
        emit_aawm_route_access_log(
            request=request,
            target="https://cli-chat-proxy.grok.com/v1/responses",
            request_body={
                **request_body,
                "model": "grok-composer-2.5-fast",
            },
        )

    route_records = [
        record.getMessage()
        for record in caplog.records
        if " [ROUTE] " in record.getMessage()
    ]
    assert len(route_records) == 1

    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path=(
                    "/anthropic/v1/messages?beta=true%20-%3E%20"
                    "cli-chat-proxy.grok.com/v1/responses"
                ),
            )
        )
        is False
    )


def test_aawm_route_access_log_filter_suppresses_matching_access_record_once() -> None:
    clear_aawm_route_access_log_replacements()
    access_filter = AawmRouteAccessLogReplacementFilter()
    register_aawm_route_access_log_replacement(
        client_addr="172.19.0.1:52834",
        method="POST",
        full_path="/anthropic/v1/messages?beta=true",
        http_version="1.1",
    )

    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                client_addr="172.19.0.1:52835",
                full_path="/anthropic/v1/messages?beta=true",
            )
        )
        is True
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/anthropic/v1/messages?beta=true",
            )
        )
        is False
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/anthropic/v1/messages?beta=true",
            )
        )
        is True
    )


def test_aawm_route_access_log_filter_suppresses_failed_record_for_rollup() -> None:
    clear_aawm_route_access_log_replacements()
    access_filter = AawmRouteAccessLogReplacementFilter()
    register_aawm_route_access_log_replacement(
        client_addr="172.19.0.1:52834",
        method="POST",
        full_path="/openai_passthrough/responses",
        http_version="1.1",
        suppress_all_statuses=True,
    )

    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/openai_passthrough/responses",
                status_code=400,
            )
        )
        is False
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/openai_passthrough/responses",
                status_code=400,
            )
        )
        is True
    )


def test_aawm_route_access_log_filter_suppresses_escaped_adapted_paths() -> None:
    clear_aawm_route_access_log_replacements()
    access_filter = AawmRouteAccessLogReplacementFilter()
    register_aawm_route_access_log_replacement(
        client_addr="172.19.0.1:41278",
        method="POST",
        full_path=(
            "/anthropic/v1/messages?beta=true -> "
            "cli-chat-proxy.grok.com/v1/responses"
        ),
        http_version="1.1",
    )

    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                client_addr="172.19.0.1:41278",
                full_path=(
                    "/anthropic/v1/messages?beta=true%20-%3E%20"
                    "cli-chat-proxy.grok.com/v1/responses"
                ),
            )
        )
        is False
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                client_addr="172.19.0.1:41278",
                full_path=(
                    "/anthropic/v1/messages?beta=true%20-%3E%20"
                    "cli-chat-proxy.grok.com/v1/responses"
                ),
            )
        )
        is False
    )


def test_aawm_route_access_log_filter_preserves_non_access_records() -> None:
    clear_aawm_route_access_log_replacements()
    access_filter = AawmRouteAccessLogReplacementFilter()
    register_aawm_route_access_log_replacement(
        client_addr="172.19.0.1:52834",
        method="POST",
        full_path="/anthropic/v1/messages?beta=true",
        http_version="1.1",
    )

    non_access_record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="normal proxy log",
        args=(),
        exc_info=None,
    )

    assert access_filter.filter(non_access_record) is True
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                full_path="/anthropic/v1/messages?beta=true",
            )
        )
        is False
    )


def test_aawm_route_logger_emits_info_level_access_lines() -> None:
    assert verbose_aawm_route_logger.getEffectiveLevel() <= logging.INFO
    assert verbose_aawm_route_logger.handlers
    assert all(
        handler.level <= logging.INFO for handler in verbose_aawm_route_logger.handlers
    )


@pytest.mark.asyncio
async def test_pass_through_request_emits_aawm_route_access_log(
    caplog, monkeypatch
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
    request = _build_aawm_route_log_request(
        url="http://127.0.0.1:4001/openai_passthrough/responses?stream=false",
        client=("172.19.0.1", 44766),
        headers={"user-agent": "codex-cli/0.119.0-alpha.29"},
    )
    request_body = {
        "model": "gpt-5.3-codex-spark",
        "input": "redacted",
        "litellm_metadata": {
            "agent_name": "W2 tester",
            "repository": "litellm",
            "requested_model_alias": "aawm-code",
            "codex_auto_agent_selected_model": "gpt-5.3-codex-spark",
        },
    }
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.api_key = "test-api-key"
    mock_user_api_key_dict.key_alias = "test-alias"
    mock_user_api_key_dict.user_email = "test@example.com"
    mock_user_api_key_dict.user_id = "test-user-id"
    mock_user_api_key_dict.team_id = "test-team-id"
    mock_user_api_key_dict.org_id = "test-org-id"
    mock_user_api_key_dict.project_id = "test-project-id"
    mock_user_api_key_dict.team_alias = "test-team-alias"
    mock_user_api_key_dict.end_user_id = "test-end-user-id"
    mock_user_api_key_dict.request_route = "/openai_passthrough/responses"
    mock_user_api_key_dict.spend = 0
    mock_user_api_key_dict.max_budget = None
    mock_user_api_key_dict.budget_reset_at = None
    mock_user_api_key_dict.metadata = {}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.aread = AsyncMock(return_value=b'{"success": true}')
    mock_response.text = '{"success": true}'
    mock_response.content = b'{"success": true}'
    mock_response.raise_for_status = MagicMock()

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value=request_body)
        mock_proxy_logging.post_call_failure_hook = AsyncMock()
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=AsyncMock(return_value=mock_response),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
            return_value={},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(return_value=None),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_response_body",
            return_value={"success": True},
        ), _capture_aawm_route_logs(
            caplog
        ):
            await pass_through_request(
                request=request,
                target="https://api.openai.com/v1/responses?api_key=secret",
                custom_headers={},
                user_api_key_dict=mock_user_api_key_dict,
                custom_body=request_body,
                custom_llm_provider="openai",
            )

    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                client_addr="172.19.0.1:44766",
                method="POST",
                full_path="/openai_passthrough/responses?stream=false",
                http_version="1.1",
            )
        )
        is False
    )
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                client_addr="172.19.0.1:44767",
                method="POST",
                full_path="/openai_passthrough/responses?stream=false",
                http_version="1.1",
            )
        )
        is True
    )
    route_records = [
        record.getMessage()
        for record in caplog.records
        if " [ROUTE] " in record.getMessage()
    ]
    assert len(route_records) == 1
    assert re.fullmatch(
        r"\d{8} \d{2}:\d{2}:\d{2} \[ROUTE\] Codex/0\.119\.0-alpha\.29 - "
        r"W2 tester@litellm\.gpt-5\.3-codex-spark\(aawm-code\) "
        r"POST 172\.19\.0\.1:44766 /openai_passthrough/responses\?stream=false "
        r"-> api\.openai\.com/v1/responses",
        route_records[0],
    )
    assert "redacted" not in route_records[0]
    assert "api_key" not in route_records[0]
    clear_aawm_route_rollups()


@pytest.mark.asyncio
async def test_pass_through_async_success_handler_records_completed_route_rollup_turn(
    monkeypatch,
) -> None:
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")

    kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "group_header_label": "litellm@Codex[0.119.0-alpha.29]",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": "api.openai.com/v1/responses",
                    "model_label": "gpt-5.3-codex-spark(aawm-code)",
                }
            }
        }
    }
    handler = PassThroughEndpointLogging()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"success": true}'
    mock_response.headers = {}
    mock_response.request = MagicMock(
        method="POST", url="https://api.openai.com/v1/responses"
    )

    with patch.object(
        handler,
        "_handle_logging",
        new=AsyncMock(return_value=None),
    ):
        await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body={"success": True},
            logging_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "gpt-5.3-codex-spark"},
            passthrough_logging_payload=MagicMock(),
            custom_llm_provider="openai",
            **kwargs,
        )

    flushed = flush_aawm_route_rollups(force=True)
    rendered = "\n".join(flushed)

    assert len(flushed) == 2
    assert ("litellm@Codex[0.119.0-alpha.29] /openai_passthrough/responses") in rendered
    assert (
        " - gpt-5.3-codex-spark(aawm-code) - Turns: 1 -> " "api.openai.com/v1/responses"
    ) in rendered
    assert (
        kwargs["litellm_params"]["metadata"]["aawm_route_rollup_turn_recorded"] is True
    )
    clear_aawm_route_rollups()


# Test is_multipart
def test_is_multipart():
    # Test with multipart content type
    request = MagicMock(spec=Request)
    request.headers = Headers({"content-type": "multipart/form-data; boundary=123"})
    assert HttpPassThroughEndpointHelpers.is_multipart(request) is True

    # Test with non-multipart content type
    request.headers = Headers({"content-type": "application/json"})
    assert HttpPassThroughEndpointHelpers.is_multipart(request) is False


def test_validate_outgoing_egress_allows_matching_openai_codex_headers():
    headers = {
        "Authorization": "Bearer codex-token",
        "ChatGPT-Account-Id": "acct_123",
        "originator": "codex_cli_rs",
        "user-agent": "codex_cli_rs/0.0.0",
        "session_id": "session_123",
    }

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url="https://chatgpt.com/backend-api/codex/responses",
        headers=headers,
        credential_family="openai",
        expected_target_family="openai",
    )


def test_validate_outgoing_egress_blocks_openai_markers_to_anthropic():
    reset_egress_guard_alert_state()
    headers = {
        "Authorization": "Bearer codex-token",
        "ChatGPT-Account-Id": "acct_123",
        "originator": "codex_cli_rs",
        "user-agent": "codex_cli_rs/0.0.0",
        "session_id": "session_123",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.anthropic.com/v1/messages",
            headers=headers,
            credential_family="openai",
        )

    assert exc_info.value.status_code == 500
    assert "credential family openai cannot be sent to anthropic" in str(
        exc_info.value.detail
    )
    alert_state = get_egress_guard_alert_state()
    assert alert_state["trigger_count"] == 1
    assert alert_state["last_credential_family"] == "openai"
    assert alert_state["last_target_family"] == "anthropic"
    reset_egress_guard_alert_state()


@pytest.mark.asyncio
async def test_passthrough_success_handler_applies_logging_hooks_before_success_callbacks(
    monkeypatch,
):
    events = []

    class ImmediateExecutor:
        def submit(self, fn, *args):
            fn(*args)
            return None

    class Enricher:
        def logging_hook(self, kwargs, result, call_type):
            events.append(("hook", call_type))
            updated_kwargs = dict(kwargs)
            litellm_params = dict(updated_kwargs.get("litellm_params") or {})
            metadata = dict(litellm_params.get("metadata") or {})
            metadata["trace_user_id"] = "pytest-classifier"
            litellm_params["metadata"] = metadata
            updated_kwargs["litellm_params"] = litellm_params
            return updated_kwargs, result

    class Recorder:
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append(
                (
                    "success",
                    kwargs["litellm_params"]["metadata"].get("trace_user_id"),
                )
            )

    monkeypatch.setattr(success_handler, "thread_pool_executor", ImmediateExecutor())
    monkeypatch.setattr(
        success_handler.litellm,
        "success_callback",
        [Enricher(), Recorder()],
    )
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5.4-mini",
        messages=[],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )

    await PassThroughEndpointLogging()._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={},
        result="{}",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        litellm_params={"metadata": {}},
    )

    assert events == [
        ("hook", "pass_through_endpoint"),
        ("success", "pytest-classifier"),
    ]


def test_validate_outgoing_egress_allows_matching_openrouter_headers():
    headers = {
        "Authorization": "Bearer openrouter-token",
        "HTTP-Referer": "https://litellm.ai",
        "X-Title": "liteLLM",
    }

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url="https://openrouter.ai/api/v1/responses",
        headers=headers,
        credential_family="openrouter",
        expected_target_family="openrouter",
    )


def test_validate_outgoing_egress_blocks_openrouter_credentials_to_openai():
    reset_egress_guard_alert_state()
    headers = {
        "Authorization": "Bearer openrouter-token",
        "HTTP-Referer": "https://litellm.ai",
        "X-Title": "liteLLM",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.openai.com/v1/responses",
            headers=headers,
            credential_family="openrouter",
        )

    assert exc_info.value.status_code == 500
    assert "credential family openrouter cannot be sent to openai" in str(
        exc_info.value.detail
    )
    alert_state = get_egress_guard_alert_state()
    assert alert_state["trigger_count"] == 1
    assert alert_state["last_credential_family"] == "openrouter"
    assert alert_state["last_target_family"] == "openai"
    reset_egress_guard_alert_state()


def test_validate_outgoing_egress_allows_matching_google_headers():
    headers = {
        "Authorization": "Bearer ya29.google-token",
        "x-goog-api-client": "anthropic-google-adapter",
    }

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent",
        headers=headers,
        credential_family="google",
        expected_target_family="google",
    )


def test_validate_outgoing_egress_blocks_google_credentials_to_anthropic():
    reset_egress_guard_alert_state()
    headers = {
        "Authorization": "Bearer ya29.google-token",
        "x-goog-api-client": "anthropic-google-adapter",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.anthropic.com/v1/messages",
            headers=headers,
            credential_family="google",
        )

    assert exc_info.value.status_code == 500
    assert "credential family google cannot be sent to anthropic" in str(
        exc_info.value.detail
    )
    alert_state = get_egress_guard_alert_state()
    assert alert_state["trigger_count"] == 1
    assert alert_state["last_credential_family"] == "google"
    assert alert_state["last_target_family"] == "anthropic"
    reset_egress_guard_alert_state()


def test_validate_outgoing_egress_allows_matching_nvidia_headers():
    headers = {
        "Authorization": "Bearer nvapi-test-token",
        "api-key": "nvapi-test-token",
    }

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url="https://integrate.api.nvidia.com/v1/responses",
        headers=headers,
        credential_family="nvidia",
        expected_target_family="nvidia",
    )


def test_validate_outgoing_egress_blocks_nvidia_credentials_to_openai():
    reset_egress_guard_alert_state()
    headers = {
        "Authorization": "Bearer nvapi-test-token",
        "api-key": "nvapi-test-token",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.openai.com/v1/responses",
            headers=headers,
            credential_family="nvidia",
        )

    assert exc_info.value.status_code == 500
    assert "credential family nvidia cannot be sent to openai" in str(
        exc_info.value.detail
    )
    alert_state = get_egress_guard_alert_state()
    assert alert_state["trigger_count"] == 1
    assert alert_state["last_credential_family"] == "nvidia"
    assert alert_state["last_target_family"] == "openai"
    reset_egress_guard_alert_state()


def test_validate_outgoing_egress_allows_matching_xai_grok_headers():
    headers = {
        "Authorization": "Bearer oidc-token",
        "X-XAI-Token-Auth": "xai-grok-cli",
        "x-grok-model-override": "grok-build",
    }

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url="https://cli-chat-proxy.grok.com/v1/responses",
        headers=headers,
        credential_family="xai",
        expected_target_family="xai",
    )


def test_validate_outgoing_egress_blocks_xai_grok_headers_to_openai():
    reset_egress_guard_alert_state()
    headers = {
        "Authorization": "Bearer oidc-token",
        "X-XAI-Token-Auth": "xai-grok-cli",
        "x-grok-model-override": "grok-build",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.openai.com/v1/responses",
            headers=headers,
            credential_family="xai",
        )

    assert exc_info.value.status_code == 500
    assert "credential family xai cannot be sent to openai" in str(
        exc_info.value.detail
    )
    alert_state = get_egress_guard_alert_state()
    assert alert_state["trigger_count"] == 1
    assert alert_state["last_credential_family"] == "xai"
    assert alert_state["last_target_family"] == "openai"
    reset_egress_guard_alert_state()


def test_validate_outgoing_egress_blocks_anthropic_markers_to_openai():
    headers = {
        "x-api-key": "anthropic-secret",
        "anthropic-version": "2023-06-01",
    }

    with pytest.raises(HTTPException) as exc_info:
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url="https://api.openai.com/v1/responses",
            headers=headers,
            expected_target_family="openai",
        )

    assert exc_info.value.status_code == 500
    assert "cross-provider credential/header markers" in str(exc_info.value.detail)


def test_forward_headers_from_request_filters_to_allowlist():
    request_headers = {
        "authorization": "Bearer direct-token",
        "anthropic-version": "2023-06-01",
        "user-agent": "claude-cli/2.1.114",
        "x-pass-chatgpt-account-id": "acct_123",
        "x-pass-anthropic-beta": "tool-use-2026-01-01",
    }

    forwarded = HttpPassThroughEndpointHelpers.forward_headers_from_request(
        request_headers=request_headers,
        headers={},
        forward_headers=True,
        allowed_forward_headers=["authorization", "user-agent"],
        allowed_pass_through_prefixed_headers=["chatgpt-account-id"],
    )

    assert forwarded == {
        "authorization": "Bearer direct-token",
        "user-agent": "claude-cli/2.1.114",
        "chatgpt-account-id": "acct_123",
    }


# Test _build_request_files_from_upload_file
@pytest.mark.asyncio
async def test_build_request_files_from_upload_file():
    # Test with FastAPI UploadFile
    file_content = b"test content"
    file = BytesIO(file_content)
    # Create SpooledTemporaryFile with content type headers
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    result = await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
        upload_file
    )
    assert result == ("test.txt", file_content, "text/plain")

    # Test with Starlette UploadFile
    file2 = BytesIO(file_content)
    starlette_file = StarletteUploadFile(
        file=file2,
        filename="test2.txt",
        headers=Headers({"content-type": "text/plain"}),
    )
    starlette_file.read = AsyncMock(return_value=file_content)

    result = await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
        starlette_file
    )
    assert result == ("test2.txt", file_content, "text/plain")


# Test make_multipart_http_request
@pytest.mark.asyncio
async def test_make_multipart_http_request():
    # Mock request with file and form field
    request = MagicMock(spec=Request)
    request.method = "POST"

    # Mock form data
    file_content = b"test file content"
    file = BytesIO(file_content)
    # Create SpooledTemporaryFile with content type headers
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = {"file": upload_file, "text_field": "test value"}
    request.form = AsyncMock(return_value=form_data)

    # Mock httpx client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    # Test the function
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers={},
        requested_query_params=None,
    )

    # Verify the response
    assert response == mock_response

    # Verify the client call
    async_client.request.assert_called_once()
    call_args = async_client.request.call_args[1]

    assert call_args["method"] == "POST"
    assert str(call_args["url"]) == "http://test.com"
    assert isinstance(call_args["files"], dict)
    assert isinstance(call_args["data"], dict)
    assert call_args["data"]["text_field"] == "test value"


@pytest.mark.asyncio
async def test_make_multipart_http_request_removes_content_type_header():
    """
    Test that make_multipart_http_request removes the content-type header
    to prevent boundary mismatch errors.

    When forwarding multipart requests, the original content-type header contains
    a boundary that doesn't match the new boundary httpx generates. This test
    verifies that the content-type header is removed so httpx can set it correctly.
    """
    # Mock request with form data
    request = MagicMock(spec=Request)
    request.method = "POST"

    # Mock form data with both file and regular field
    file_content = b"test file content"
    file = BytesIO(file_content)
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = {"file": upload_file, "key": "value"}
    request.form = AsyncMock(return_value=form_data)

    # Mock httpx client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    # Headers with content-type containing old boundary (this is what causes the issue)
    original_headers = {
        "content-type": "multipart/form-data; boundary=--------------------------416423083260054165225918",
        "user-agent": "PostmanRuntime/7.49.0",
        "Authorization": "bearer sk-1234",
    }

    # Test the function
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers=original_headers,
        requested_query_params={"param": "value"},
    )

    # Verify the response
    assert response == mock_response

    # Verify the client call
    async_client.request.assert_called_once()
    call_args = async_client.request.call_args[1]

    # CRITICAL ASSERTION: content-type header should be removed
    assert "content-type" not in call_args["headers"]

    # Other headers should be preserved
    assert call_args["headers"]["user-agent"] == "PostmanRuntime/7.49.0"
    assert call_args["headers"]["Authorization"] == "bearer sk-1234"

    # Verify other parameters are correct
    assert call_args["method"] == "POST"
    assert str(call_args["url"]) == "http://test.com"
    assert isinstance(call_args["files"], dict)
    assert isinstance(call_args["data"], dict)
    assert call_args["data"]["key"] == "value"
    assert call_args["params"] == {"param": "value"}

    # Verify the original headers dict was not modified (copy was used)
    assert "content-type" in original_headers


@pytest.mark.asyncio
async def test_pass_through_request_failure_handler():
    """
    Test that the failure handler is called when pass_through_request fails

    Critical Test: When a users pass through endpoint request fails, we must log the failure code, exception in litellm spend logs.
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                # Setup mock for post_call_failure_hook and pre_call_hook
                mock_proxy_logging.post_call_failure_hook = AsyncMock()
                mock_proxy_logging.pre_call_hook = AsyncMock(
                    return_value={"test": "data"}
                )

                # Setup mock for httpx client.
                # Non-stream path uses prefer_stream_for_unknown_content (RR-056 #6),
                # so failures surface through async_client.send, not .request.
                mock_client = MagicMock()
                mock_client.client = MagicMock()
                mock_client.client.build_request = MagicMock(return_value=MagicMock())
                mock_client.client.send = AsyncMock(
                    side_effect=httpx.HTTPError("Request failed")
                )
                mock_client.client.request = AsyncMock(
                    side_effect=httpx.HTTPError("Request failed")
                )
                mock_get_client.return_value = mock_client

                # Mock headers for custom headers
                mock_processing.get_custom_headers.return_value = {}

                # Create mock request
                mock_request = MagicMock(spec=Request)
                mock_request.method = "POST"
                mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
                mock_request.headers = Headers({})

                # Create a simple empty QueryParams
                mock_request.query_params = QueryParams({})

                # Create mock user API key dict
                mock_user_api_key_dict = MagicMock()

                # Call the function with a target that will trigger an HTTPError
                with pytest.raises(Exception):
                    await pass_through_request(
                        request=mock_request,
                        target="http://test.com",
                        custom_headers={},
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                # Assert post_call_failure_hook was called
                mock_proxy_logging.post_call_failure_hook.assert_called_once()

                # Verify the arguments to post_call_failure_hook
                call_args = mock_proxy_logging.post_call_failure_hook.call_args[1]
                assert call_args["user_api_key_dict"] == mock_user_api_key_dict
                assert isinstance(call_args["original_exception"], httpx.HTTPError)
                assert "traceback_str" in call_args


def test_is_anthropic_route_excludes_count_tokens():
    handler = PassThroughEndpointLogging()

    assert (
        handler.is_anthropic_route("http://localhost:4000/anthropic/v1/messages")
        is True
    )
    assert (
        handler.is_anthropic_route(
            "http://localhost:4000/anthropic/v1/messages/batches"
        )
        is True
    )
    assert (
        handler.is_anthropic_route(
            "http://localhost:4000/anthropic/v1/messages/count_tokens?beta=true"
        )
        is False
    )


def test_is_langfuse_route():
    """
    Test that the is_langfuse_route method correctly identifies Langfuse routes
    """
    handler = PassThroughEndpointLogging()

    # Test positive cases
    assert (
        handler.is_langfuse_route("http://localhost:4000/langfuse/api/public/traces")
        is True
    )
    assert (
        handler.is_langfuse_route(
            "https://proxy.example.com/langfuse/api/public/sessions"
        )
        is True
    )
    assert handler.is_langfuse_route("/langfuse/api/public/ingestion") is True
    assert handler.is_langfuse_route("http://localhost:4000/langfuse/") is True

    # Test negative cases
    assert (
        handler.is_langfuse_route("https://api.openai.com/v1/chat/completions") is False
    )
    assert (
        handler.is_langfuse_route("http://localhost:4000/anthropic/v1/messages")
        is False
    )
    assert handler.is_langfuse_route("https://example.com/other") is False
    assert handler.is_langfuse_route("") is False


@pytest.mark.asyncio
async def test_langfuse_passthrough_no_logging():
    """
    Test that langfuse pass-through requests skip logging by returning early
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Mock httpx response for langfuse request
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"status": "success"}'

    # Create langfuse URL
    langfuse_url = "http://localhost:4000/langfuse/api/public/traces"

    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=langfuse_url,
        request_body={"test": "data"},
        request_method="POST",
    )

    # Call the success handler with langfuse route
    result = await handler.pass_through_async_success_handler(
        httpx_response=mock_response,
        response_body={"status": "success"},
        logging_obj=mock_logging_obj,
        url_route=langfuse_url,
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"test": "data"},
        passthrough_logging_payload=passthrough_logging_payload,
    )

    # Should return None (early return) and not proceed with logging
    assert result is None

    # Verify that the passthrough_logging_payload was still set (this happens before the langfuse check)
    assert (
        mock_logging_obj.model_call_details["passthrough_logging_payload"]
        == passthrough_logging_payload
    )


def test_construct_target_url_with_subpath():
    """
    Test that construct_target_url_with_subpath correctly constructs target URLs
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )

    # Test with include_subpath=False
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="api/v1", include_subpath=False
    )
    assert result == "http://example.com"

    # Test with include_subpath=True and no subpath
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="", include_subpath=True
    )
    assert result == "http://example.com"

    # Test with include_subpath=True and subpath
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with base_target already ending with /
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com/", subpath="api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with subpath starting with /
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="/api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with both conditions
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com/", subpath="/api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"


def test_add_exact_path_route():
    """
    Test that add_exact_path_route correctly adds exact path routes
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test data
    path = "/test/path"
    target = "http://example.com"
    custom_headers = {"x-custom": "header"}
    forward_headers = True
    merge_query_params = False
    dependencies = []

    # Call the function
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=mock_app,
        path=path,
        target=target,
        custom_headers=custom_headers,
        forward_headers=forward_headers,
        merge_query_params=merge_query_params,
        dependencies=dependencies,
        cost_per_request=None,
        endpoint_id="test-endpoint-id",
    )

    # Verify add_api_route was called with correct parameters
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    assert call_args["path"] == path
    assert call_args["methods"] == ["GET", "POST", "PUT", "DELETE", "PATCH"]
    assert call_args["dependencies"] == dependencies
    assert callable(call_args["endpoint"])


def test_add_subpath_route():
    """
    Test that add_subpath_route correctly adds wildcard routes for sub-paths
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test data
    path = "/test/path"
    target = "http://example.com"
    custom_headers = {"x-custom": "header"}
    forward_headers = True
    merge_query_params = False
    dependencies = []

    # Call the function
    InitPassThroughEndpointHelpers.add_subpath_route(
        app=mock_app,
        path=path,
        target=target,
        custom_headers=custom_headers,
        forward_headers=forward_headers,
        merge_query_params=merge_query_params,
        dependencies=dependencies,
        cost_per_request=None,
        endpoint_id="test-endpoint-id",
    )

    # Verify add_api_route was called with correct parameters
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Should have wildcard path
    expected_wildcard_path = f"{path}/{{subpath:path}}"
    assert call_args["path"] == expected_wildcard_path
    assert call_args["methods"] == ["GET", "POST", "PUT", "DELETE", "PATCH"]
    assert call_args["dependencies"] == dependencies
    assert callable(call_args["endpoint"])


@pytest.mark.asyncio
async def test_initialize_pass_through_endpoints_with_include_subpath():
    """
    Test that initialize_pass_through_endpoints adds wildcard routes when include_subpath is True
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        initialize_pass_through_endpoints,
    )

    # Mock the helper functions directly
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_exact_path_route"
    ) as mock_add_exact_route:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_subpath_route"
        ) as mock_add_subpath_route:
            with patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ):
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header"
                ) as mock_set_env:
                    mock_set_env.return_value = {}

                    # Test endpoint with include_subpath=True
                    endpoints = [
                        {
                            "path": "/test/endpoint",
                            "target": "http://example.com",
                            "include_subpath": True,
                        }
                    ]

                    await initialize_pass_through_endpoints(endpoints)

                    # Should be called once for exact path and once for subpath
                    mock_add_exact_route.assert_called_once()
                    mock_add_subpath_route.assert_called_once()

                    # Verify exact path route call
                    exact_call_args = mock_add_exact_route.call_args[1]
                    assert exact_call_args["path"] == "/test/endpoint"
                    assert exact_call_args["target"] == "http://example.com"

                    # Verify subpath route call
                    subpath_call_args = mock_add_subpath_route.call_args[1]
                    assert subpath_call_args["path"] == "/test/endpoint"
                    assert subpath_call_args["target"] == "http://example.com"


@pytest.mark.asyncio
async def test_initialize_pass_through_endpoints_without_include_subpath():
    """
    Test that initialize_pass_through_endpoints only adds exact route when include_subpath is False
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        initialize_pass_through_endpoints,
    )

    # Mock the helper functions directly
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_exact_path_route"
    ) as mock_add_exact_route:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_subpath_route"
        ) as mock_add_subpath_route:
            with patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ):
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header"
                ) as mock_set_env:
                    mock_set_env.return_value = {}

                    # Test endpoint with include_subpath=False (default)
                    endpoints = [
                        {
                            "path": "/test/endpoint",
                            "target": "http://example.com",
                            "include_subpath": False,
                        }
                    ]

                    await initialize_pass_through_endpoints(endpoints)

                    # Should be called only once for exact path
                    mock_add_exact_route.assert_called_once()
                    mock_add_subpath_route.assert_not_called()

                    # Verify exact path route call
                    exact_call_args = mock_add_exact_route.call_args[1]
                    assert exact_call_args["path"] == "/test/endpoint"
                    assert exact_call_args["target"] == "http://example.com"


def test_set_cost_per_request():
    """
    Test that _set_cost_per_request correctly sets the cost in logging object and kwargs
    """

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Test with cost_per_request set
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=0.50,
    )

    kwargs = {"some_existing_key": "value"}

    # Call the method
    result_kwargs = handler._set_cost_per_request(
        logging_obj=mock_logging_obj,
        passthrough_logging_payload=passthrough_logging_payload,
        kwargs=kwargs,
    )

    # Verify that response_cost is set in kwargs and logging object
    assert result_kwargs["response_cost"] == 0.50
    assert mock_logging_obj.model_call_details["response_cost"] == 0.50
    assert result_kwargs["some_existing_key"] == "value"  # Existing kwargs preserved


@pytest.mark.asyncio
async def test_handle_logging_skips_plain_function_callbacks():
    handler = PassThroughEndpointLogging()
    mock_logging_obj = MagicMock()
    mock_logging_obj.get_combined_callback_list.side_effect = [
        [lambda *args, **kwargs: None],
        [lambda *args, **kwargs: None],
    ]

    with patch(
        "litellm.proxy.pass_through_endpoints.success_handler.thread_pool_executor.submit"
    ) as mock_submit:
        await handler._handle_logging(
            logging_obj=mock_logging_obj,
            standard_logging_response_object={"ok": True},
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
        )

    mock_submit.assert_not_called()


@pytest.mark.asyncio
async def test_handle_logging_uses_standard_callback_contracts():
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    class DummyAsyncCallback:
        def __init__(self):
            self.calls = []

        async def async_logging_hook(self, kwargs, result, call_type):
            self.calls.append((kwargs, result, call_type))
            next_kwargs = dict(kwargs)
            next_kwargs["mutated"] = True
            return next_kwargs, {"mutated_result": True}

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            self.calls.append((kwargs, response_obj, start_time, end_time))

    class DummySyncCallback:
        def logging_hook(self, kwargs, result, call_type):
            return kwargs, result

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            return None

    async_callback = DummyAsyncCallback()
    sync_callback = DummySyncCallback()
    mock_logging_obj = MagicMock()
    mock_logging_obj.call_type = "acompletion"
    mock_logging_obj.get_combined_callback_list.side_effect = [
        [sync_callback],
        [async_callback],
    ]

    with patch(
        "litellm.proxy.pass_through_endpoints.success_handler.thread_pool_executor.submit"
    ) as mock_submit:
        await handler._handle_logging(
            logging_obj=mock_logging_obj,
            standard_logging_response_object={"ok": True},
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            example="value",
        )

    assert mock_submit.call_count == 1
    success_call = mock_submit.call_args_list[0].args
    assert success_call[0] == sync_callback.log_success_event
    assert success_call[1:] == (
        {"example": "value", "standard_callback_dynamic_params": {}},
        {"ok": True},
        start_time,
        end_time,
    )

    assert async_callback.calls[0] == (
        {"example": "value", "standard_callback_dynamic_params": {}},
        {"ok": True},
        "acompletion",
    )
    assert async_callback.calls[1][0] == {
        "example": "value",
        "standard_callback_dynamic_params": {},
        "mutated": True,
    }
    assert async_callback.calls[1][1] == {"mutated_result": True}
    assert async_callback.calls[1][2] == start_time
    assert async_callback.calls[1][3] == end_time


def test_normalize_llm_passthrough_logging_payload_uses_openai_handler_for_adapted_openrouter_chat():
    handler = PassThroughEndpointLogging()
    mock_response = MagicMock(spec=httpx.Response)
    mock_logging_obj = MagicMock()
    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "openrouter/elephant-alpha",
        "choices": [],
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler",
        return_value={"result": "openai-ok", "kwargs": {"marker": "openai"}},
    ) as mock_openai_handler, patch(
        "litellm.proxy.pass_through_endpoints.success_handler.AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler"
    ) as mock_anthropic_handler:
        result = handler.normalize_llm_passthrough_logging_payload(
            httpx_response=mock_response,
            response_body=response_body,
            request_body={"model": "openrouter/elephant-alpha"},
            logging_obj=mock_logging_obj,
            url_route="http://127.0.0.1:4001/anthropic/v1/messages",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="openrouter",
        )

    assert result["standard_logging_response_object"] == "openai-ok"
    assert result["kwargs"] == {"marker": "openai"}
    assert (
        mock_openai_handler.call_args.kwargs["url_route"]
        == "https://openrouter.ai/api/v1/chat/completions"
    )
    mock_anthropic_handler.assert_not_called()


def test_normalize_llm_passthrough_logging_payload_uses_openai_handler_for_adapted_xai_responses():
    handler = PassThroughEndpointLogging()
    mock_response = MagicMock(spec=httpx.Response)
    mock_logging_obj = MagicMock()
    response_body = {
        "id": "resp-test",
        "object": "response",
        "model": "grok-build",
        "output": [],
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler",
        return_value={"result": "openai-ok", "kwargs": {"marker": "xai"}},
    ) as mock_openai_handler:
        result = handler.normalize_llm_passthrough_logging_payload(
            httpx_response=mock_response,
            response_body=response_body,
            request_body={"model": "grok-build"},
            logging_obj=mock_logging_obj,
            url_route="https://cli-chat-proxy.grok.com/v1/responses",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="xai",
        )

    assert result["standard_logging_response_object"] == "openai-ok"
    assert result["kwargs"] == {"marker": "xai"}
    assert mock_openai_handler.call_args.kwargs["url_route"] == (
        "https://api.x.ai/v1/responses"
    )
    assert mock_openai_handler.call_args.kwargs["custom_llm_provider"] == "xai"


def test_normalize_llm_passthrough_logging_payload_normalizes_xai_embeddings_not_cohere():
    handler = PassThroughEndpointLogging()
    response_body = {
        "id": "grok-embed-test",
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
        "model": "embedding-beta-3-small",
    }
    mock_response = httpx.Response(
        200,
        json=response_body,
        request=httpx.Request(
            "POST",
            "https://cli-chat-proxy.grok.com/v1/embeddings",
        ),
    )
    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {}

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.cohere_passthrough_logging_handler.CoherePassthroughLoggingHandler.cohere_passthrough_handler"
    ) as mock_cohere_handler:
        result = handler.normalize_llm_passthrough_logging_payload(
            httpx_response=mock_response,
            response_body=response_body,
            request_body={"model": "grok-build"},
            logging_obj=mock_logging_obj,
            url_route="https://cli-chat-proxy.grok.com/v1/embeddings",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="xai",
        )

    normalized_response = result["standard_logging_response_object"]
    assert normalized_response.model == "grok-build"
    assert normalized_response.usage.prompt_tokens == 12
    assert result["kwargs"]["custom_llm_provider"] == "xai"
    assert result["kwargs"]["call_type"] == "embedding"
    assert mock_cohere_handler.call_count == 0


def test_set_cost_per_request_none():
    """
    Test that _set_cost_per_request does nothing when cost_per_request is None
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Test with cost_per_request not set (None)
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=None,
    )

    kwargs = {"some_existing_key": "value"}

    # Call the method
    result_kwargs = handler._set_cost_per_request(
        logging_obj=mock_logging_obj,
        passthrough_logging_payload=passthrough_logging_payload,
        kwargs=kwargs,
    )

    # Verify that response_cost is not set
    assert "response_cost" not in result_kwargs
    assert "response_cost" not in mock_logging_obj.model_call_details
    assert result_kwargs["some_existing_key"] == "value"  # Existing kwargs preserved


@pytest.mark.asyncio
async def test_pass_through_success_handler_with_cost_per_request():
    """
    Test that the success handler correctly processes cost_per_request
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Mock the _handle_logging method to capture the call
    handler._handle_logging = AsyncMock()

    # Mock httpx response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"status": "success", "data": "test"}'

    # Create passthrough logging payload with cost_per_request
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=1.25,
    )

    start_time = datetime.now()
    end_time = datetime.now()

    # Call the success handler
    await handler.pass_through_async_success_handler(
        httpx_response=mock_response,
        response_body={"status": "success", "data": "test"},
        logging_obj=mock_logging_obj,
        url_route="http://example.com/api",
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        request_body={"test": "data"},
        passthrough_logging_payload=passthrough_logging_payload,
    )

    # Verify that the logging object has the cost set
    assert mock_logging_obj.model_call_details["response_cost"] == 1.25

    # Verify that _handle_logging was called with the correct kwargs
    handler._handle_logging.assert_called_once()
    call_kwargs = handler._handle_logging.call_args[1]
    assert call_kwargs["response_cost"] == 1.25


@pytest.mark.asyncio
async def test_create_pass_through_route_with_cost_per_request():
    """
    Test that create_pass_through_route correctly passes cost_per_request to the endpoint function
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    # Create the endpoint function with cost_per_request
    unique_path = "/test/path/unique/cost_per_request"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com",
        custom_headers={},
        _forward_headers=True,
        _merge_query_params=False,
        dependencies=[],
        cost_per_request=3.75,
    )

    # Mock the pass_through_request function to capture its call
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
    ) as mock_pass_through, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
    ) as mock_get_registered:
        mock_pass_through.return_value = MagicMock()
        # Single-fetch contract (RR-056 #10): registry hit supplies params.
        mock_get_registered.return_value = {
            "passthrough_params": {},
        }

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})

        # Call the endpoint function
        # Create a proper UserAPIKeyAuth mock
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        await endpoint_func(
            request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
            fastapi_response=MagicMock(),
        )

        # Verify that pass_through_request was called with cost_per_request
        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]
        assert call_kwargs["cost_per_request"] == 3.75


def test_initialize_pass_through_endpoints_with_cost_per_request():
    """
    Test that initialize_pass_through_endpoints correctly passes cost_per_request to route creation
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test exact path route with cost_per_request
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=mock_app,
        path="/test/path",
        target="http://example.com",
        custom_headers={},
        forward_headers=True,
        merge_query_params=False,
        dependencies=[],
        cost_per_request=5.00,
        endpoint_id="test-endpoint-id-1",
    )

    # Verify add_api_route was called
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Verify the endpoint function was created with cost_per_request
    # We can't directly test the internal cost_per_request value, but we can verify
    # that the endpoint function was created properly
    assert call_args["path"] == "/test/path"
    assert callable(call_args["endpoint"])

    # Reset mock for subpath test
    mock_app.reset_mock()

    # Test subpath route with cost_per_request
    InitPassThroughEndpointHelpers.add_subpath_route(
        app=mock_app,
        path="/test/path",
        target="http://example.com",
        custom_headers={},
        forward_headers=True,
        merge_query_params=False,
        dependencies=[],
        cost_per_request=7.50,
        endpoint_id="test-endpoint-id-2",
    )

    # Verify add_api_route was called for subpath
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Verify the wildcard path and endpoint function
    assert call_args["path"] == "/test/path/{subpath:path}"
    assert callable(call_args["endpoint"])


@pytest.mark.asyncio
async def test_pass_through_request_contains_proxy_server_request_in_kwargs():  # noqa: PLR0915
    """
    Test that pass_through_request (parent method) correctly includes proxy_server_request
    in kwargs passed to the success handler.

    Critical Test: Ensures that when pass_through_request is called, the kwargs passed to
    downstream methods contain the proxy server request details (url, method, body).
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler"
        ) as mock_http_handler:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    with patch(
                        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_response_body"
                    ) as mock_get_response_body:
                        # Setup mock for pre_call_hook and post_call_failure_hook
                        mock_proxy_logging.pre_call_hook = AsyncMock(
                            return_value={"test": "data"}
                        )
                        mock_proxy_logging.post_call_failure_hook = AsyncMock()

                        # Setup mock for http response
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.headers = {}
                        mock_response.aread = AsyncMock(
                            return_value=b'{"success": true}'
                        )
                        mock_response.text = '{"success": true}'
                        mock_response.raise_for_status = MagicMock()

                        # Mock the HTTP request handler directly
                        mock_http_handler.return_value = mock_response

                        # Mock response body parser
                        mock_get_response_body.return_value = {"success": True}

                        # Mock headers for custom headers
                        mock_processing.get_custom_headers.return_value = {}

                        # Mock success handler to capture kwargs
                        mock_success_handler.return_value = None

                        # Create mock request
                        mock_request = MagicMock(spec=Request)
                        mock_request.method = "POST"
                        mock_request.url = "http://test-proxy.com/api/endpoint"
                        mock_request.body = AsyncMock(
                            return_value=b'{"message": "test request"}'
                        )
                        mock_request.headers = Headers({})
                        mock_request.query_params = QueryParams({})

                        # Create mock user API key dict
                        mock_user_api_key_dict = MagicMock()
                        mock_user_api_key_dict.api_key = "test-api-key"
                        mock_user_api_key_dict.key_alias = "test-alias"
                        mock_user_api_key_dict.user_email = "test@example.com"
                        mock_user_api_key_dict.user_id = "test-user-id"
                        mock_user_api_key_dict.team_id = "test-team-id"
                        mock_user_api_key_dict.org_id = "test-org-id"
                        mock_user_api_key_dict.team_alias = "test-team-alias"
                        mock_user_api_key_dict.end_user_id = "test-end-user-id"
                        mock_user_api_key_dict.request_route = "/api/endpoint"

                        # Call pass_through_request (the parent method)
                        await pass_through_request(
                            request=mock_request,
                            target="http://target-api.com/endpoint",
                            custom_headers={"X-Custom": "header"},
                            user_api_key_dict=mock_user_api_key_dict,
                        )

                        # Verify the success handler was called
                        mock_success_handler.assert_called_once()

                        # Extract the kwargs passed to the success handler
                        call_kwargs = mock_success_handler.call_args[1]

                        # Verify that litellm_params exists in kwargs
                        assert "litellm_params" in call_kwargs
                        litellm_params = call_kwargs["litellm_params"]

                        # Verify that proxy_server_request exists in litellm_params
                        assert "proxy_server_request" in litellm_params
                        proxy_server_request = litellm_params["proxy_server_request"]

                        # Verify the proxy_server_request contains expected fields
                        assert "url" in proxy_server_request
                        assert "method" in proxy_server_request
                        assert "body" in proxy_server_request

                        # Verify the values match the original request
                        assert proxy_server_request["url"] == str(mock_request.url)
                        assert proxy_server_request["method"] == mock_request.method
                        # The body should be the value returned by pre_call_hook, not the original request body
                        assert proxy_server_request["body"] == {"test": "data"}

                        # Verify other required kwargs are present
                        assert "call_type" in call_kwargs
                        assert call_kwargs["call_type"] == "pass_through_endpoint"
                        assert "litellm_call_id" in call_kwargs
                        assert "passthrough_logging_payload" in call_kwargs

                        # Verify metadata contains user information
                        assert "metadata" in litellm_params
                        metadata = litellm_params["metadata"]
                        assert metadata["user_api_key_hash"] == "test-api-key"
                        assert metadata["user_api_key_alias"] == "test-alias"
                        assert metadata["user_api_key_user_email"] == "test@example.com"
                        assert metadata["user_api_key_user_id"] == "test-user-id"


@pytest.mark.asyncio
async def test_create_pass_through_endpoint():
    """
    Test creating a new pass-through endpoint

    This test verifies that the create_pass_through_endpoints function:
    1. Accepts a PassThroughGenericEndpoint object
    2. Auto-generates an ID if not provided
    3. Adds the endpoint to the database
    4. Returns the created endpoint with the generated ID
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Mock existing config (empty list)
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=[]
            )

            # Create test endpoint data
            test_endpoint = PassThroughGenericEndpoint(
                path="/test/endpoint",
                target="http://example.com/api",
                headers={"Authorization": "Bearer test-token"},
                include_subpath=True,
                cost_per_request=0.50,
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the create function
            result = await create_pass_through_endpoints(
                data=test_endpoint,
                request=MagicMock(spec=Request),
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            created_endpoint = result.endpoints[0]
            assert created_endpoint.path == "/test/endpoint"
            assert created_endpoint.target == "http://example.com/api"
            assert created_endpoint.headers == {"Authorization": "Bearer test-token"}
            assert created_endpoint.include_subpath is True
            assert created_endpoint.cost_per_request == 0.50
            assert created_endpoint.id is not None  # Should be auto-generated

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            assert len(update_call_args["data"].field_value) == 1
            assert update_call_args["data"].field_value[0]["path"] == "/test/endpoint"
            assert update_call_args["data"].field_value[0]["id"] is not None


@pytest.mark.asyncio
async def test_update_pass_through_endpoint():
    """
    Test updating an existing pass-through endpoint

    This test verifies that the update_pass_through_endpoints function:
    1. Finds the existing endpoint by ID
    2. Updates only the provided fields (partial updates)
    3. Preserves the existing ID
    4. Updates the database with the modified endpoint
    5. Returns the updated endpoint
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Create existing endpoint data
            existing_endpoint_id = "test-endpoint-123"
            existing_endpoints = [
                {
                    "id": existing_endpoint_id,
                    "path": "/test/endpoint",
                    "target": "http://example.com/api",
                    "headers": {"Authorization": "Bearer old-token"},
                    "include_subpath": False,
                    "cost_per_request": 0.25,
                }
            ]

            # Mock existing config
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=existing_endpoints
            )

            # Create update data (partial update)
            update_data = PassThroughGenericEndpoint(
                path="/test/endpoint",  # Keep same path
                target="http://newapi.com/v2",  # Update target
                headers={
                    "Authorization": "Bearer new-token",
                    "X-Custom": "header",
                },  # Update headers
                cost_per_request=0.75,  # Update cost
                # include_subpath not provided - should preserve existing value
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the update function
            result = await update_pass_through_endpoints(
                endpoint_id=existing_endpoint_id,
                data=update_data,
                request=MagicMock(spec=Request),
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            updated_endpoint = result.endpoints[0]
            assert updated_endpoint.id == existing_endpoint_id  # ID preserved
            assert updated_endpoint.path == "/test/endpoint"
            assert updated_endpoint.target == "http://newapi.com/v2"  # Updated
            assert updated_endpoint.headers == {
                "Authorization": "Bearer new-token",
                "X-Custom": "header",
            }  # Updated
            assert updated_endpoint.include_subpath is False  # Preserved existing value
            assert updated_endpoint.cost_per_request == 0.75  # Updated

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            assert len(update_call_args["data"].field_value) == 1
            updated_data = update_call_args["data"].field_value[0]
            assert updated_data["id"] == existing_endpoint_id
            assert updated_data["target"] == "http://newapi.com/v2"
            assert updated_data["cost_per_request"] == 0.75


@pytest.mark.asyncio
async def test_update_pass_through_endpoint_not_found():
    """
    Test updating a non-existent pass-through endpoint raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock existing config with different endpoint
        existing_endpoints = [
            {
                "id": "different-endpoint-456",
                "path": "/different/endpoint",
                "target": "http://different.com/api",
                "headers": {},
                "include_subpath": False,
                "cost_per_request": 0.0,
            }
        ]

        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        # Create update data
        update_data = PassThroughGenericEndpoint(
            path="/test/endpoint", target="http://newapi.com/v2"
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the update function with non-existent ID
        with pytest.raises(HTTPException) as exc_info:
            await update_pass_through_endpoints(
                endpoint_id="non-existent-endpoint-123",
                data=update_data,
                request=MagicMock(spec=Request),
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Verify the exception
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint():
    """
    Test deleting an existing pass-through endpoint

    This test verifies that the delete_pass_through_endpoints function:
    1. Finds the existing endpoint by ID
    2. Removes it from the database
    3. Returns the deleted endpoint
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughEndpointResponse,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Create existing endpoint data
            endpoint_to_delete_id = "test-endpoint-123"
            other_endpoint_id = "other-endpoint-456"
            existing_endpoints = [
                {
                    "id": endpoint_to_delete_id,
                    "path": "/test/endpoint",
                    "target": "http://example.com/api",
                    "headers": {"Authorization": "Bearer test-token"},
                    "include_subpath": True,
                    "cost_per_request": 0.50,
                },
                {
                    "id": other_endpoint_id,
                    "path": "/other/endpoint",
                    "target": "http://other.com/api",
                    "headers": {},
                    "include_subpath": False,
                    "cost_per_request": 0.25,
                },
            ]

            # Mock existing config
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=existing_endpoints
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the delete function
            result = await delete_pass_through_endpoints(
                endpoint_id=endpoint_to_delete_id,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            deleted_endpoint = result.endpoints[0]
            assert deleted_endpoint.id == endpoint_to_delete_id
            assert deleted_endpoint.path == "/test/endpoint"
            assert deleted_endpoint.target == "http://example.com/api"
            assert deleted_endpoint.headers == {"Authorization": "Bearer test-token"}
            assert deleted_endpoint.include_subpath is True
            assert deleted_endpoint.cost_per_request == 0.50

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            # Should only have the other endpoint remaining
            assert len(update_call_args["data"].field_value) == 1
            remaining_endpoint = update_call_args["data"].field_value[0]
            assert remaining_endpoint["id"] == other_endpoint_id
            assert remaining_endpoint["path"] == "/other/endpoint"


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint_not_found():
    """
    Test deleting a non-existent pass-through endpoint raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import ConfigFieldInfo, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock existing config with different endpoint
        existing_endpoints = [
            {
                "id": "different-endpoint-456",
                "path": "/different/endpoint",
                "target": "http://different.com/api",
                "headers": {},
                "include_subpath": False,
                "cost_per_request": 0.0,
            }
        ]

        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the delete function with non-existent ID
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="non-existent-endpoint-123",
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Verify the exception
        assert exc_info.value.status_code == 400
        assert "not found" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_get_pass_through_endpoints_includes_config_and_db():
    """
    Test that get_pass_through_endpoints returns both config-defined and DB endpoints,
    with correct is_from_config flag. Config-only endpoints have is_from_config=True,
    DB endpoints have is_from_config=False. When same path exists in both, DB overrides.
    """
    from litellm.proxy._types import (
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        get_pass_through_endpoints,
    )

    # Config-defined endpoints (from config file)
    config_endpoints = [
        {
            "path": "/v1/rerank",
            "target": "https://api.cohere.com/v1/rerank",
            "headers": {"content-type": "application/json"},
        },
        {
            "path": "/v1/config-only",
            "target": "https://config.example.com/api",
            "headers": {},
        },
    ]

    # DB endpoints (one overlaps with config path, one is DB-only)
    db_endpoints = [
        {
            "id": "db-endpoint-1",
            "path": "/v1/rerank",  # Same as config - DB should override
            "target": "https://db-override.com/v1/rerank",
            "headers": {},
            "include_subpath": False,
        },
        {
            "id": "db-endpoint-2",
            "path": "/db/only",
            "target": "https://db-only.example.com/api",
            "headers": {},
            "include_subpath": False,
        },
    ]

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        MagicMock(),
    ):
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._get_pass_through_endpoints_from_db",
            new_callable=AsyncMock,
        ) as mock_get_db:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._get_pass_through_endpoints_from_config"
            ) as mock_get_config:
                db_objects = [
                    PassThroughGenericEndpoint(**ep, is_from_config=False)
                    for ep in db_endpoints
                ]
                config_objects = [
                    PassThroughGenericEndpoint(**ep, is_from_config=True)
                    for ep in config_endpoints
                ]
                mock_get_db.return_value = db_objects
                mock_get_config.return_value = config_objects

                mock_user = MagicMock(spec=UserAPIKeyAuth)

                result = await get_pass_through_endpoints(
                    endpoint_id=None,
                    user_api_key_dict=mock_user,
                    team_id=None,
                )

    assert isinstance(result, PassThroughEndpointResponse)
    # config_only: /v1/config-only (not in db_paths)
    # db: /v1/rerank (overrides config), /db/only
    # So we should have: /v1/config-only (from config) + /v1/rerank + /db/only (from db)
    assert len(result.endpoints) == 3

    # Check is_from_config values
    by_path = {ep.path: ep for ep in result.endpoints}
    assert by_path["/v1/config-only"].is_from_config is True
    assert by_path["/v1/rerank"].is_from_config is False  # DB overrides
    assert by_path["/db/only"].is_from_config is False

    # Verify DB override: /v1/rerank should have DB target
    assert by_path["/v1/rerank"].target == "https://db-override.com/v1/rerank"


def test_get_pass_through_endpoints_from_config_skips_malformed():
    """
    Test that _get_pass_through_endpoints_from_config skips malformed endpoints
    and returns only valid ones, without raising.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _get_pass_through_endpoints_from_config,
    )

    # Mix of valid and malformed config endpoints
    config_passthrough_endpoints = [
        {"path": "/valid/1", "target": "https://valid1.example.com"},
        {},  # Missing required path and target
        {"path": "/missing-target"},  # Missing required target
        {"target": "https://example.com"},  # Missing required path
        {"path": "/valid/2", "target": "https://valid2.example.com", "headers": {}},
    ]

    with patch(
        "litellm.proxy.proxy_server.config_passthrough_endpoints",
        config_passthrough_endpoints,
    ):
        result = _get_pass_through_endpoints_from_config()

    # Only the 2 valid endpoints should be returned
    assert len(result) == 2
    paths = {ep.path for ep in result}
    assert "/valid/1" in paths
    assert "/valid/2" in paths
    for ep in result:
        assert ep.is_from_config is True


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint_empty_list():
    """
    Test deleting from an empty endpoint list raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import ConfigFieldInfo, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock empty config
        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the delete function
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="any-endpoint-123", user_api_key_dict=mock_user_api_key_dict
            )

        # Verify the exception
        assert exc_info.value.status_code == 400
        assert "no pass-through endpoints setup" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_pass_through_request_query_params_forwarding():
    """
    Test that query parameters from the original request are properly forwarded to the target URL.

    This test verifies the fix for the bug where query parameters like api-version were being lost
    when forwarding requests to Azure OpenAI and other pass-through endpoints.
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler"
        ) as mock_http_handler:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    with patch(
                        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_response_body"
                    ) as mock_get_response_body:
                        # Setup mock for pre_call_hook
                        test_body = {"name": "Azure Assistant", "model": "gpt-4o"}
                        mock_proxy_logging.pre_call_hook = AsyncMock(
                            return_value=test_body
                        )

                        # Setup mock for http response
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.headers = {"content-type": "application/json"}
                        mock_response.aread = AsyncMock(
                            return_value=b'{"id": "asst_123", "object": "assistant"}'
                        )
                        mock_response.text = '{"id": "asst_123", "object": "assistant"}'
                        mock_response.raise_for_status = MagicMock()

                        # Mock the HTTP request handler to capture the call
                        mock_http_handler.return_value = mock_response

                        # Mock response body parser
                        mock_get_response_body.return_value = {
                            "id": "asst_123",
                            "object": "assistant",
                        }

                        # Mock headers for custom headers
                        mock_processing.get_custom_headers.return_value = {}

                        # Mock success handler
                        mock_success_handler.return_value = None

                        # Create mock request with query parameters (Azure API version)
                        mock_request = MagicMock(spec=Request)
                        mock_request.method = "POST"
                        mock_request.url = (
                            "http://localhost:4000/azure-assistant/openai/assistants"
                        )
                        mock_request.body = AsyncMock(
                            return_value=json.dumps(test_body).encode()
                        )
                        mock_request.headers = Headers(
                            {"Content-Type": "application/json"}
                        )

                        # Create QueryParams with api-version parameter
                        mock_request.query_params = QueryParams(
                            [("api-version", "2025-01-01-preview")]
                        )

                        # Create mock user API key dict
                        mock_user_api_key_dict = MagicMock()
                        mock_user_api_key_dict.api_key = "sk-1234"

                        # Call pass_through_request
                        await pass_through_request(
                            request=mock_request,
                            target="https://krris-m2f9a9i7-eastus2.openai.azure.com/openai/assistants",
                            custom_headers={"Authorization": "Bearer azure_token"},
                            user_api_key_dict=mock_user_api_key_dict,
                        )

                        # Verify the HTTP handler was called
                        mock_http_handler.assert_called_once()

                        # Extract the call arguments to verify query parameters were passed
                        call_kwargs = mock_http_handler.call_args[1]

                        # The key assertion: query parameters should be preserved and passed to the HTTP handler
                        assert "requested_query_params" in call_kwargs
                        assert call_kwargs["requested_query_params"] == {
                            "api-version": "2025-01-01-preview"
                        }

                        # Verify the target URL is correct
                        assert (
                            str(call_kwargs["url"])
                            == "https://krris-m2f9a9i7-eastus2.openai.azure.com/openai/assistants"
                        )

                        # Verify the request body is preserved
                        assert call_kwargs["_parsed_body"] == test_body


@pytest.mark.asyncio
async def test_pass_through_request_does_not_mutate_custom_body_on_failure():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.headers = Headers({"Content-Type": "application/json"})
    mock_request.query_params = QueryParams("")
    mock_request.body = AsyncMock(return_value=b"{}")

    mock_user_api_key_dict = MagicMock()
    custom_body = {"foo": "bar"}

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = '{"error": "quota"}'
    request_for_error = httpx.Request("POST", "https://example.com/test")
    response_for_error = httpx.Response(
        status_code=429,
        request=request_for_error,
        content=b'{"error": "quota"}',
    )
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429",
        request=request_for_error,
        response=response_for_error,
    )

    mock_async_client = MagicMock()
    mock_async_client_obj = MagicMock(client=mock_async_client)

    post_failure_mock = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_async_client_obj,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=mock_response),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
        new=AsyncMock(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ):
        with pytest.raises(ProxyException):
            await pass_through_request(
                request=mock_request,
                target="https://example.com/test",
                custom_headers={},
                user_api_key_dict=mock_user_api_key_dict,
                custom_body=custom_body,
                stream=False,
            )

    assert custom_body == {"foo": "bar"}
    assert "litellm_logging_obj" not in custom_body
    request_data = post_failure_mock.await_args.kwargs["request_data"]
    logged_body = request_data["passthrough_logging_payload"]["request_body"]
    assert request_data is not logged_body
    assert "passthrough_logging_payload" not in logged_body


@pytest.mark.asyncio
async def test_pass_through_request_adds_xai_context_on_failure_logging():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4000/grok/v1/responses"
    mock_request.headers = Headers(
        {
            "content-type": "application/json",
            "authorization": "Bearer oidc-token",
            "x-grok-model-override": "grok-build",
        }
    )
    mock_request.query_params = QueryParams({})
    mock_user_api_key_dict = MagicMock()
    custom_body = {
        "model": "grok-build",
        "input": "hello",
        "litellm_metadata": {
            "passthrough_route_family": "grok_cli_chat_proxy",
        },
    }
    upstream_request = httpx.Request(
        "POST",
        "https://cli-chat-proxy.grok.com/v1/responses",
    )
    upstream_response = httpx.Response(
        401,
        json={"error": "Invalid or expired credentials"},
        request=upstream_request,
    )
    post_failure_mock = AsyncMock()
    direct_capture_mock = AsyncMock()

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=MagicMock(client=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        # Force the fallback direct-capture path so we still assert xAI context
        # enrichment on the shared request payload (B7 skips when registered).
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._is_aawm_agent_identity_registered_in_litellm_callbacks",
        return_value=False,
    ), patch(
        "litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance.async_post_call_failure_hook",
        new=direct_capture_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ):
        with pytest.raises(ProxyException):
            await pass_through_request(
                request=mock_request,
                target="https://cli-chat-proxy.grok.com/v1/responses",
                custom_headers={},
                user_api_key_dict=mock_user_api_key_dict,
                custom_body=custom_body,
                custom_llm_provider="xai",
                egress_credential_family="xai",
                expected_target_family="xai",
            )

    post_failure_mock.assert_awaited_once()
    direct_capture_mock.assert_awaited_once()
    request_data = post_failure_mock.await_args.kwargs["request_data"]
    direct_request_data = direct_capture_mock.await_args.kwargs["request_data"]
    assert direct_request_data is request_data
    metadata = request_data["litellm_params"]["metadata"]
    assert request_data["custom_llm_provider"] == "xai"
    assert request_data["model"] == "grok-build"
    assert metadata["custom_llm_provider"] == "xai"
    assert metadata["passthrough_route_family"] == "grok_cli_chat_proxy"
    assert metadata["grok_model_override"] == "grok-build"
    assert metadata["xai_cli_chat_proxy"] is True


def test_record_grok_billing_passthrough_request_contract_redacts_header_values():
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url = "http://localhost:4001/grok/v1/billing?format=credits"
    metadata = {
        "passthrough_route_family": "grok_cli_chat_proxy",
        "user_api_key_request_route": "/grok/v1/billing",
    }

    _record_grok_billing_passthrough_request_contract(
        request=request,
        url=httpx.URL("https://cli-chat-proxy.grok.com/v1/billing"),
        headers={
            "authorization": "Bearer oidc-token-secret",
            "content-type": "application/json",
            "user-agent": "grok-pager/0.2.55 grok-shell/0.2.55 (linux; x86_64)",
            "x-email": "user@example.com",
            "x-grok-user-id": "user_123",
            "x-teamid": "team_123",
            "x-userid": "user_123",
            "x-xai-token-auth": "xai-grok-cli",
        },
        requested_query_params={"format": "credits"},
        metadata=metadata,
        custom_llm_provider="xai",
    )

    assert metadata["grok_billing_passthrough_http_client"] == "httpx"
    assert metadata["grok_billing_passthrough_request_method"] == "GET"
    assert metadata["grok_billing_passthrough_target_host"] == (
        "cli-chat-proxy.grok.com"
    )
    assert metadata["grok_billing_passthrough_target_path"] == "/v1/billing"
    assert metadata["grok_billing_passthrough_query_keys"] == ["format"]
    assert metadata["grok_billing_passthrough_query_present"] is True
    assert metadata["grok_billing_passthrough_x_xai_token_auth_configured"] is True
    assert "authorization" in metadata["grok_billing_passthrough_header_names"]
    assert "x-email" in metadata["grok_billing_passthrough_header_names"]
    assert len(metadata["grok_billing_passthrough_request_contract_fingerprint"]) == 64
    dumped = json.dumps(metadata)
    assert "oidc-token-secret" not in dumped
    assert "xai-grok-cli" not in dumped
    assert "user@example.com" not in dumped
    assert "user_123" not in dumped
    assert "team_123" not in dumped


@pytest.mark.asyncio
async def test_direct_capture_xai_passthrough_failure_uses_canonical_agent_identity():
    """RR-056 #7: single canonical import (RR-003 packaging), no dual module probe."""
    direct_capture_mock = AsyncMock()
    request_payload = {"model": "grok-build", "custom_llm_provider": "xai"}
    user_api_key_dict = MagicMock()
    original_exception = HTTPException(status_code=401, detail="xai auth failed")

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._is_aawm_agent_identity_registered_in_litellm_callbacks",
        return_value=False,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._CANONICAL_AAWM_AGENT_IDENTITY_INSTANCE",
        new=SimpleNamespace(async_post_call_failure_hook=direct_capture_mock),
    ):
        await _direct_capture_xai_passthrough_failure(
            request_payload=request_payload,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
            traceback_str="traceback",
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
        )

    direct_capture_mock.assert_awaited_once_with(
        user_api_key_dict=user_api_key_dict,
        original_exception=original_exception,
        request_data=request_payload,
        traceback_str="traceback",
    )


@pytest.mark.asyncio
async def test_direct_capture_xai_passthrough_failure_skips_when_agent_identity_in_callbacks():
    """B7 / RR-056 #2: do not double-invoke agent identity when already registered."""
    import litellm

    direct_capture_mock = AsyncMock()

    class FakeAawmAgentIdentity:
        async def async_post_call_failure_hook(self, *args, **kwargs):
            return await direct_capture_mock(*args, **kwargs)

    FakeAawmAgentIdentity.__module__ = "litellm.integrations.aawm_agent_identity"
    FakeAawmAgentIdentity.__name__ = "AawmAgentIdentity"
    agent_identity_callback = FakeAawmAgentIdentity()

    request_payload = {"model": "grok-build", "custom_llm_provider": "xai"}
    original_exception = HTTPException(status_code=401, detail="xai auth failed")
    original_callbacks = list(getattr(litellm, "callbacks", []) or [])
    original_failure_callbacks = list(getattr(litellm, "failure_callback", []) or [])
    canonical_hook = AsyncMock(
        side_effect=AssertionError(
            "canonical agent identity must not run when already registered"
        )
    )

    try:
        litellm.callbacks = [agent_identity_callback]
        litellm.failure_callback = []
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._CANONICAL_AAWM_AGENT_IDENTITY_INSTANCE",
            new=SimpleNamespace(async_post_call_failure_hook=canonical_hook),
        ):
            assert _is_aawm_agent_identity_registered_in_litellm_callbacks() is True
            await _direct_capture_xai_passthrough_failure(
                request_payload=request_payload,
                original_exception=original_exception,
                user_api_key_dict=MagicMock(),
                traceback_str="traceback",
                url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
                custom_llm_provider="xai",
            )
    finally:
        litellm.callbacks = original_callbacks
        litellm.failure_callback = original_failure_callbacks

    direct_capture_mock.assert_not_awaited()
    canonical_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_capture_xai_passthrough_failure_skips_string_callback_registration():
    """String callback entries (config form) also suppress the direct xAI path."""
    import litellm

    original_callbacks = list(getattr(litellm, "callbacks", []) or [])
    original_failure_callbacks = list(getattr(litellm, "failure_callback", []) or [])
    canonical_hook = AsyncMock(
        side_effect=AssertionError(
            "canonical agent identity must not run when string callback registered"
        )
    )

    try:
        litellm.callbacks = [
            "litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance",
        ]
        litellm.failure_callback = []
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._CANONICAL_AAWM_AGENT_IDENTITY_INSTANCE",
            new=SimpleNamespace(async_post_call_failure_hook=canonical_hook),
        ):
            await _direct_capture_xai_passthrough_failure(
                request_payload={"model": "grok-build", "custom_llm_provider": "xai"},
                original_exception=HTTPException(status_code=500, detail="boom"),
                user_api_key_dict=MagicMock(),
                traceback_str=None,
                url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
                custom_llm_provider="xai",
            )
    finally:
        litellm.callbacks = original_callbacks
        litellm.failure_callback = original_failure_callbacks

    canonical_hook.assert_not_awaited()


class TestGrokPersonalTeamSpendingLimitLogging:
    def test_classifier_matches_known_spending_limit_403_body(self):
        detail = (
            b'{"code":"personal-team-blocked:spending-limit",'
            b'"error":"Personal team spending limit reached."}'
        )
        assert _is_known_grok_personal_team_spending_limit_response(
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
            status_code=403,
            exc=HTTPException(status_code=403, detail=detail),
        )
        assert (
            _get_passthrough_grok_personal_team_spending_limit_failure_kind()
            == "upstream_grok_account_quota_exhaustion"
        )

    @pytest.mark.parametrize(
        "detail",
        [
            b'{"code":"permission-denied","error":"Project access denied."}',
            b'{"code":"personal-team-blocked:other","error":"blocked"}',
            b'{"error":"quota"}',
        ],
    )
    def test_classifier_rejects_unrelated_grok_403_bodies(self, detail):
        assert not _is_known_grok_personal_team_spending_limit_response(
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
            status_code=403,
            exc=HTTPException(status_code=403, detail=detail),
        )

    def test_classifier_rejects_non_xai_target(self):
        detail = (
            b'{"code":"personal-team-blocked:spending-limit",'
            b'"error":"Personal team spending limit reached."}'
        )
        assert not _is_known_grok_personal_team_spending_limit_response(
            url=httpx.URL("https://api.anthropic.com/v1/messages"),
            custom_llm_provider="anthropic",
            status_code=403,
            exc=HTTPException(status_code=403, detail=detail),
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_known_spending_limit_warns_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/grok/v1/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.body = AsyncMock(
            return_value=b'{"model":"grok-composer-2.5-fast"}'
        )

        target_url = "https://cli-chat-proxy.grok.com/v1/responses"
        upstream_body = (
            b'{"code":"personal-team-blocked:spending-limit",'
            b'"error":"Personal team spending limit reached."}'
        )
        upstream_response = httpx.Response(
            status_code=403,
            content=upstream_body,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = TestPassThroughTerminalFailureLogging._install_aawm_error_log_handler(
            tmp_path,
            monkeypatch,
        )

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning, patch.object(
                verbose_proxy_logger,
                "exception",
            ) as mock_exception:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(return_value={})
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                        custom_llm_provider="xai",
                        stream=False,
                    )

                assert exc_info.value.code == "403"
                assert handler.await_count == 1
                mock_warning.assert_called_once()
                assert (
                    mock_warning.call_args.kwargs["extra"]["failure_kind"]
                    == "upstream_grok_account_quota_exhaustion"
                )
                assert (
                    mock_warning.call_args.args[2]
                    == "Personal team spending limit reached."
                )
                mock_exception.assert_not_called()
                mock_logging_obj.post_call_failure_hook.assert_not_awaited()
        finally:
            TestPassThroughTerminalFailureLogging._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()


class TestGrokBuildUsageBalanceExhaustedLogging:
    def test_classifier_matches_known_402_body(self):
        detail = b'{"error":"Grok Build usage balance exhausted"}'
        assert _is_known_grok_build_usage_balance_exhausted_response(
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
            status_code=402,
            exc=HTTPException(status_code=402, detail=detail),
        )
        assert (
            _get_passthrough_grok_build_usage_balance_exhausted_failure_kind()
            == "upstream_grok_account_quota_exhaustion"
        )

    @pytest.mark.parametrize(
        "detail",
        [
            b'{"error":"Grok Build usage balance low"}',
            b'{"error":"quota exhausted"}',
            b'{"code":"payment-required","error":"other"}',
        ],
    )
    def test_classifier_rejects_unrelated_grok_402_bodies(self, detail):
        assert not _is_known_grok_build_usage_balance_exhausted_response(
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
            status_code=402,
            exc=HTTPException(status_code=402, detail=detail),
        )

    def test_classifier_rejects_non_xai_target(self):
        detail = b'{"error":"Grok Build usage balance exhausted"}'
        assert not _is_known_grok_build_usage_balance_exhausted_response(
            url=httpx.URL("https://api.openai.com/v1/responses"),
            custom_llm_provider="openai",
            status_code=402,
            exc=HTTPException(status_code=402, detail=detail),
        )

    def test_classifier_rejects_wrong_status(self):
        detail = b'{"error":"Grok Build usage balance exhausted"}'
        assert not _is_known_grok_build_usage_balance_exhausted_response(
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/responses"),
            custom_llm_provider="xai",
            status_code=403,
            exc=HTTPException(status_code=403, detail=detail),
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_known_402_warns_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/grok/v1/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.body = AsyncMock(
            return_value=b'{"model":"grok-composer-2.5-fast"}'
        )

        target_url = "https://cli-chat-proxy.grok.com/v1/responses"
        upstream_body = b'{"error":"Grok Build usage balance exhausted"}'
        upstream_response = httpx.Response(
            status_code=402,
            content=upstream_body,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = TestPassThroughTerminalFailureLogging._install_aawm_error_log_handler(
            tmp_path,
            monkeypatch,
        )

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning, patch.object(
                verbose_proxy_logger,
                "exception",
            ) as mock_exception:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(return_value={})
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                        custom_llm_provider="xai",
                        stream=False,
                    )

                assert exc_info.value.code == "402"
                assert handler.await_count == 1
                mock_warning.assert_called_once()
                assert (
                    mock_warning.call_args.kwargs["extra"]["failure_kind"]
                    == "upstream_grok_account_quota_exhaustion"
                )
                mock_exception.assert_not_called()
                mock_logging_obj.post_call_failure_hook.assert_not_awaited()
        finally:
            TestPassThroughTerminalFailureLogging._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()


def test_handled_http_error_summary_preserves_proxy_exception_detail():
    exc = ProxyException(
        message="HTTP 402 request rejected",
        type="None",
        param="None",
        code=402,
    )
    setattr(exc, "detail", b'{"error":"Grok Build usage balance exhausted"}')

    assert (
        _get_passthrough_handled_http_error_summary(
            exc,
            status_code=402,
        )
        == "Grok Build usage balance exhausted"
    )


class TestGrokBillingPassthroughTimeoutLogging:
    def test_classifier_matches_known_billing_timeout_cancel_body(self):
        request = MagicMock(spec=Request)
        request.url = "http://localhost:4001/grok/v1/billing?format=credits"
        assert _is_known_grok_billing_passthrough_timeout_cancel_response(
            request=request,
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/billing?format=credits"),
            custom_llm_provider="xai",
            status_code=400,
            exc=HTTPException(
                status_code=400,
                detail='{"code":"The operation was cancelled","error":"Timeout expired"}',
            ),
        )

    def test_classifier_rejects_unexpected_billing_400_body(self):
        request = MagicMock(spec=Request)
        request.url = "http://localhost:4001/grok/v1/billing?format=credits"
        assert not _is_known_grok_billing_passthrough_timeout_cancel_response(
            request=request,
            url=httpx.URL("https://cli-chat-proxy.grok.com/v1/billing?format=credits"),
            custom_llm_provider="xai",
            status_code=400,
            exc=HTTPException(
                status_code=400,
                detail='{"error":"billing unavailable"}',
            ),
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_known_billing_timeout_warns_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = "http://localhost:4001/grok/v1/billing?format=credits"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {"format": "credits"}
        mock_request.body = AsyncMock(return_value=b"")

        target_url = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
        upstream_response = httpx.Response(
            status_code=400,
            content=(
                b'{"code":"The operation was cancelled",' b'"error":"Timeout expired"}'
            ),
            request=httpx.Request("GET", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = TestPassThroughTerminalFailureLogging._install_aawm_error_log_handler(
            tmp_path,
            monkeypatch,
        )

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(return_value={})
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                        custom_llm_provider="xai",
                        stream=False,
                    )

                assert exc_info.value.code == "400"
                assert handler.await_count == 1
                mock_warning.assert_called_once()
                assert (
                    mock_warning.call_args.kwargs["extra"]["failure_kind"]
                    == "degraded_grok_billing_timeout"
                )
                mock_logging_obj.post_call_failure_hook.assert_not_awaited()
        finally:
            TestPassThroughTerminalFailureLogging._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    @pytest.mark.asyncio
    async def test_pass_through_request_unexpected_billing_400_is_handled_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = "http://localhost:4001/grok/v1/billing?format=credits"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {"format": "credits"}
        mock_request.body = AsyncMock(return_value=b"")

        target_url = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
        upstream_response = httpx.Response(
            status_code=400,
            content=b'{"error":"billing unavailable"}',
            request=httpx.Request("GET", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = TestPassThroughTerminalFailureLogging._install_aawm_error_log_handler(
            tmp_path,
            monkeypatch,
        )

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
                new=AsyncMock(),
            ) as mock_direct_capture:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(return_value={})
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                        custom_llm_provider="xai",
                        stream=False,
                    )

                assert exc_info.value.code == "400"
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
                mock_direct_capture.assert_awaited_once()
        finally:
            TestPassThroughTerminalFailureLogging._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "handled client/provider error status=400" in item["message"]
        )
        assert payload["context"]["endpoint"] == "/grok/v1/billing"
        assert payload["context"]["status_code"] == 400
        assert "billing unavailable" in payload["message"]
        assert not payload.get("traceback")


class TestPassThroughTerminalFailureLogging:
    @staticmethod
    def _install_aawm_error_log_handler(tmp_path, monkeypatch):
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

        handler = AawmErrorLogFileHandler()
        saved_handlers = verbose_proxy_logger.handlers[:]
        saved_level = verbose_proxy_logger.level
        saved_propagate = verbose_proxy_logger.propagate
        verbose_proxy_logger.handlers.clear()
        verbose_proxy_logger.addHandler(handler)
        verbose_proxy_logger.setLevel(logging.ERROR)
        verbose_proxy_logger.propagate = False

        return saved_handlers, saved_level, saved_propagate

    @staticmethod
    def _flush_aawm_error_log_handlers() -> None:
        """Drain async AAWM error-log writers before asserting on disk files."""
        for current_handler in list(verbose_proxy_logger.handlers):
            flush = getattr(current_handler, "flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception:
                    pass

    @staticmethod
    def _restore_verbose_proxy_logger(saved_handlers, saved_level, saved_propagate):
        # RR-004/async error log: flush before detaching so tests can read jsonl.
        TestPassThroughTerminalFailureLogging._flush_aawm_error_log_handlers()
        for current_handler in list(verbose_proxy_logger.handlers):
            close = getattr(current_handler, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        verbose_proxy_logger.handlers.clear()
        for saved_handler in saved_handlers:
            verbose_proxy_logger.addHandler(saved_handler)
        verbose_proxy_logger.setLevel(saved_level)
        verbose_proxy_logger.propagate = saved_propagate

    @staticmethod
    def _build_codex_unsupported_content_type_fixture(
        secret_prompt,
        secret_token,
        *,
        model="gpt-5.5",
        upstream_detail=b'{"detail":"Unsupported content type"}',
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = (
            "http://localhost:4001/openai_passthrough/responses?stream=true"
        )
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {secret_token}",
        }
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        request_body = {
            "model": model,
            "stream": True,
            "input": secret_prompt,
            "instructions": f"use token {secret_token}",
            "litellm_metadata": {
                "passthrough_route_family": "codex_responses",
                "session_id": "session-secret-019ed451",
            },
        }
        req = httpx.Request("POST", target_url)
        failing_response = httpx.Response(
            status_code=400,
            content=upstream_detail,
            request=req,
        )

        httpx_error = httpx.HTTPStatusError(
            "Client error '400 Bad Request' for url 'https://chatgpt.com/backend-api/codex/responses'",
            request=req,
            response=failing_response,
        )
        failing_response.aread = AsyncMock(return_value=upstream_detail)
        failing_response.raise_for_status = MagicMock(side_effect=httpx_error)

        async def fake_send(request, stream=True):
            return failing_response

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=req)
        mock_client.send = AsyncMock(side_effect=fake_send)
        return SimpleNamespace(
            mock_client=mock_client,
            mock_request=mock_request,
            request_body=request_body,
            target_url=target_url,
        )

    @staticmethod
    def _assert_codex_unsupported_content_type_error_context(context):
        assert context["endpoint"] == "/openai_passthrough/responses"
        assert (
            context["upstream_url"] == "https://chatgpt.com/backend-api/codex/responses"
        )
        assert context["provider"] == "openai"
        assert context["model"] == "gpt-5.5"
        assert context["route_family"] == "codex_responses"
        assert context["status_code"] == 400
        assert context["auth_credential_source"] == "route_custom_header"
        assert context["auth_header_names"] == ["authorization"]
        assert context["auth_header_sources"] == ["route_custom_header:authorization"]
        assert context["aawm_passthrough_inbound_content_type"] == "application/json"
        assert context["aawm_passthrough_json_egress_content_type_removed"] is None
        assert (
            context["aawm_passthrough_json_egress_content_type_removed_value"] is None
        )
        assert context["aawm_passthrough_body_container_type"] == "object"
        assert context["aawm_passthrough_body_top_level_keys"] == [
            "input",
            "instructions",
            "model",
            "stream",
        ]
        assert context["aawm_passthrough_input_container_type"] == "str"
        assert context["grok_side_channel_request_body_byte_length"] is None

    def test_terminal_failure_classifier_matches_exhausted_hidden_retry_status(self):
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "aawm_passthrough_hidden_retry_final_outcome": "failed_after_retry",
                    "aawm_passthrough_hidden_retry_failure_classification": None,
                    "aawm_passthrough_hidden_retry_count": 6,
                }
            }
        }

        assert _should_log_passthrough_terminal_failure_without_traceback(
            exc=HTTPException(status_code=529, detail="overloaded"),
            kwargs=kwargs,
            status_code=529,
        )

    def test_terminal_failure_classifier_without_hidden_retry_metadata_is_safe(self):
        assert not _should_log_passthrough_terminal_failure_without_traceback(
            exc=HTTPException(status_code=507, detail="buffer limit"),
            kwargs={},
            status_code=507,
        )

    def test_terminal_failure_classifier_keeps_auth_errors_exceptional(self):
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "aawm_passthrough_hidden_retry_final_outcome": "failed_without_retry",
                }
            }
        }

        assert not _should_log_passthrough_terminal_failure_without_traceback(
            exc=HTTPException(status_code=401, detail="unauthorized"),
            kwargs=kwargs,
            status_code=401,
        )

    def test_terminal_failure_classifier_is_side_effect_safe_when_metadata_is_missing(
        self,
    ):
        assert not _should_log_passthrough_terminal_failure_without_traceback(
            exc=HTTPException(status_code=529, detail="overloaded"),
            kwargs=None,
            status_code=529,
        )

    @pytest.mark.parametrize(
        ("status_code", "detail"),
        [
            (
                429,
                "429: b'RESOURCE_EXHAUSTED quotaResetTimeStamp=2026-06-26T11:28:58Z'",
            ),
            (
                507,
                "507: b'exceeded request buffer limit while retrying upstream'",
            ),
        ],
    )
    def test_terminal_failure_classifier_matches_hidden_retry_terminal_statuses(
        self,
        status_code,
        detail,
    ):
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "aawm_passthrough_hidden_retry_final_outcome": "failed_after_retry",
                    "aawm_passthrough_hidden_retry_failure_classification": None,
                    "aawm_passthrough_hidden_retry_count": 1,
                }
            }
        }

        assert _should_log_passthrough_terminal_failure_without_traceback(
            exc=HTTPException(status_code=status_code, detail=detail),
            kwargs=kwargs,
            status_code=status_code,
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_exhausted_529_logs_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        upstream_response = httpx.Response(
            status_code=529,
            content=b'{"error":"overloaded"}',
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "claude-sonnet-4-6"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ):
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-sonnet-4-6"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )

                assert exc_info.value.code == "529"
                assert handler.await_count == 6
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "exhausted hidden retries for upstream failure" in item["message"]
        )

        assert payload["context"]["status_code"] == 529
        assert payload["context"]["model"] == "claude-sonnet-4-6"
        assert "final_outcome=failed_after_retry" in payload["message"]
        assert "retry_count=6" in payload["message"]
        assert payload.get("traceback") in (None, "")

    @pytest.mark.asyncio
    async def test_pass_through_request_429_uses_rollup_without_standalone_warning(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/anthropic/v1/messages?beta=true"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {"beta": "true"}

        target_url = "https://api.anthropic.com/v1/messages"
        upstream_response = httpx.Response(
            status_code=429,
            content=(
                b'{"type":"error","error":{"type":"rate_limit_error",'
                b'"message":"Rate limited"},"request_id":"req_test"}'
            ),
            headers={"retry-after": "17"},
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "claude-opus-4-8"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ) as mock_sleep, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning, patch.object(
                verbose_proxy_logger,
                "debug",
            ) as mock_debug, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.record_aawm_route_rollup_failure"
            ) as mock_rollup_failure:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-opus-4-8"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )

                assert exc_info.value.code == "429"
                assert handler.await_count == 1
                mock_sleep.assert_not_awaited()
                mock_warning.assert_not_called()
                rate_limit_debug = next(
                    call
                    for call in mock_debug.call_args_list
                    if "recorded expected upstream rate limit" in str(call.args[0])
                )
                assert (
                    rate_limit_debug.kwargs["extra"]["failure_kind"]
                    == "expected_provider_rate_limit"
                )
                assert rate_limit_debug.args[2] == "Rate limited"
                mock_rollup_failure.assert_called_once()
                assert mock_rollup_failure.call_args.kwargs["message"] == "Rate limited"
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    @pytest.mark.asyncio
    async def test_pass_through_request_terminal_507_hidden_retry_logs_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_detail = (
            "507: b'exceeded request buffer limit while retrying upstream'"
        )
        upstream_response = httpx.Response(
            status_code=507,
            content=upstream_detail.encode("utf-8"),
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)
        hidden_retry_metadata = {
            "aawm_passthrough_hidden_retry_final_outcome": "failed_after_retry",
            "aawm_passthrough_hidden_retry_failure_classification": None,
            "aawm_passthrough_hidden_retry_count": 1,
        }

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "gpt-5.5"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._ensure_passthrough_metadata",
                return_value=hidden_retry_metadata,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ), patch.object(
                verbose_proxy_logger,
                "exception",
            ) as mock_exception:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "gpt-5.5"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )

                assert exc_info.value.code == "507"
                assert upstream_detail in str(exc_info.value.message)
                mock_exception.assert_not_called()
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "exhausted hidden retries for upstream failure" in item["message"]
        )
        assert payload["context"]["status_code"] == 507
        assert "buffer limit" in payload["message"]
        assert payload.get("traceback") in (None, "")

    @pytest.mark.asyncio
    async def test_pass_through_request_codex_stream_400_error_intake_omits_raw_request_body(
        self,
        monkeypatch,
        tmp_path,
    ):
        secret_prompt = "SUPER_SECRET_CODEX_PROMPT_DO_NOT_LOG"
        secret_token = "sk-leaked-codex-token"
        fixture = self._build_codex_unsupported_content_type_fixture(
            secret_prompt,
            secret_token,
        )

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value=fixture.request_body,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj:
                mock_client_obj = MagicMock()
                mock_client_obj.client = fixture.mock_client
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value=fixture.request_body
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture.mock_request,
                        target=fixture.target_url,
                        custom_headers={"authorization": f"Bearer {secret_token}"},
                        user_api_key_dict=MagicMock(),
                        stream=True,
                        forward_headers=True,
                        custom_llm_provider="openai",
                    )

                assert exc_info.value.code == "400"
                assert "Unsupported content type" in str(exc_info.value.message)
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                request_headers = fixture.mock_client.build_request.call_args.kwargs[
                    "headers"
                ]
                content_type_headers = {
                    str(header_name).lower(): value
                    for header_name, value in request_headers.items()
                }
                assert content_type_headers["content-type"] == "application/json"
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item for item in payloads if "Unsupported content type" in item["message"]
        )
        context = payload["context"]

        self._assert_codex_unsupported_content_type_error_context(context)

        serialized = json.dumps(payload)
        assert secret_prompt not in serialized
        assert secret_token not in serialized
        assert "session-secret-019ed451" not in serialized
        assert "SUPER_SECRET" not in serialized
        assert "sk-leaked" not in serialized

    @pytest.mark.asyncio
    async def test_pass_through_request_codex_stream_invalid_encrypted_content_warns_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        upstream_detail = (
            b'{\n  "error": {\n'
            b'    "message": "The encrypted content A3Sc...4ndF could not be verified.",\n'
            b'    "type": "invalid_request_error",\n'
            b'    "param": null,\n'
            b'    "code": "invalid_encrypted_content"\n'
            b"  }\n}"
        )
        fixture = self._build_codex_unsupported_content_type_fixture(
            "redacted prompt",
            "sk-redacted",
            model="gpt-5.3-codex-spark",
            upstream_detail=upstream_detail,
        )

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value=fixture.request_body,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning, patch.object(
                verbose_proxy_logger,
                "exception",
            ) as mock_exception:
                mock_client_obj = MagicMock()
                mock_client_obj.client = fixture.mock_client
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value=fixture.request_body
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture.mock_request,
                        target=fixture.target_url,
                        custom_headers={"authorization": "Bearer sk-redacted"},
                        user_api_key_dict=MagicMock(),
                        stream=True,
                        forward_headers=True,
                        custom_llm_provider="openai",
                    )

                assert exc_info.value.code == "400"
                assert "invalid_encrypted_content" in str(exc_info.value.detail)
                mock_exception.assert_not_called()
                invalid_warning = next(
                    call
                    for call in mock_warning.call_args_list
                    if "invalid encrypted content" in str(call.args[0])
                )
                assert (
                    invalid_warning.kwargs["extra"]["failure_kind"]
                    == "openai_chatgpt_codex_invalid_encrypted_content"
                )
                assert invalid_warning.kwargs["extra"]["upstream_url"] == (
                    fixture.target_url
                )
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    @pytest.mark.asyncio
    async def test_pass_through_request_chatgpt_codex_block_page_warns_with_failure_kind(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_response = httpx.Response(
            status_code=403,
            content=(
                b"<html><body><p>Unable to load site</p>"
                b"<script src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'>"
                b"</script></body></html>"
            ),
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "gpt-5.5"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch.object(
            verbose_proxy_logger,
            "warning",
        ) as mock_warning, patch.object(
            verbose_proxy_logger,
            "exception",
        ) as mock_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "gpt-5.5"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    custom_llm_provider="openai",
                    stream=False,
                )

        assert exc_info.value.code == "403"
        mock_exception.assert_not_called()
        block_warning = next(
            call
            for call in mock_warning.call_args_list
            if "ChatGPT Codex block page" in str(call.args[0])
        )
        assert (
            block_warning.kwargs["extra"]["failure_kind"]
            == "openai_chatgpt_codex_block_page"
        )
        assert block_warning.kwargs["extra"]["upstream_url"] == target_url
        mock_logging_obj.post_call_failure_hook.assert_awaited_once()
        assert (
            mock_logging_obj.post_call_failure_hook.await_args.kwargs["traceback_str"]
            is None
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_chatgpt_codex_invalid_encrypted_content_warns_with_failure_kind(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_detail = (
            b'{\n  "error": {\n'
            b'    "message": "The encrypted content A3Sc...4ndF could not be verified.",\n'
            b'    "type": "invalid_request_error",\n'
            b'    "param": null,\n'
            b'    "code": "invalid_encrypted_content"\n'
            b"  }\n}"
        )
        upstream_response = httpx.Response(
            status_code=400,
            content=upstream_detail,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "gpt-5.3-codex-spark"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch.object(
            verbose_proxy_logger,
            "warning",
        ) as mock_warning, patch.object(
            verbose_proxy_logger,
            "exception",
        ) as mock_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "gpt-5.3-codex-spark"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    custom_llm_provider="openai",
                    stream=False,
                )

        assert exc_info.value.code == "400"
        assert str(upstream_detail, "utf-8") in str(exc_info.value.detail)
        assert handler.await_count == 1
        mock_exception.assert_not_called()
        invalid_warning = next(
            call
            for call in mock_warning.call_args_list
            if "invalid encrypted content" in str(call.args[0])
        )
        assert (
            invalid_warning.kwargs["extra"]["failure_kind"]
            == "openai_chatgpt_codex_invalid_encrypted_content"
        )
        assert invalid_warning.kwargs["extra"]["upstream_url"] == target_url
        mock_logging_obj.post_call_failure_hook.assert_awaited_once()
        assert (
            mock_logging_obj.post_call_failure_hook.await_args.kwargs["traceback_str"]
            is None
        )

    def test_classifier_matches_chatgpt_codex_unsupported_model_for_account_body(self):
        target_url = "https://chatgpt.com/backend-api/codex/responses"
        exc = HTTPException(
            status_code=400,
            detail=(
                b'{"detail":"The \'gpt-5.6-terra\' model is not supported when using Codex '
                b'with a ChatGPT account."}'
            ),
        )
        assert _is_known_chatgpt_codex_model_not_supported_for_account_response(
            url=httpx.URL(target_url),
            status_code=400,
            exc=exc,
        )
        assert (
            _get_passthrough_chatgpt_codex_model_not_supported_failure_kind()
            == "openai_chatgpt_codex_model_not_supported_for_account"
        )

    def test_classifier_rejects_unrelated_chatgpt_codex_400_body(self):
        target_url = "https://chatgpt.com/backend-api/codex/responses"
        exc = HTTPException(
            status_code=400,
            detail=b'{"detail":"Unsupported content type"}',
        )
        assert not _is_known_chatgpt_codex_model_not_supported_for_account_response(
            url=httpx.URL(target_url),
            status_code=400,
            exc=exc,
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_chatgpt_codex_unsupported_model_warns_with_failure_kind(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_detail = (
            b'{"detail":"The \'gpt-5.6-terra\' model is not supported when using Codex '
            b'with a ChatGPT account."}'
        )
        upstream_response = httpx.Response(
            status_code=400,
            content=upstream_detail,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "gpt-5.6-terra"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch.object(
            verbose_proxy_logger,
            "warning",
        ) as mock_warning, patch.object(
            verbose_proxy_logger,
            "exception",
        ) as mock_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "gpt-5.6-terra"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    custom_llm_provider="openai",
                    stream=False,
                )

        assert exc_info.value.code == "400"
        assert "gpt-5.6-terra" in str(exc_info.value.detail)
        mock_exception.assert_not_called()
        unsupported_warning = next(
            call
            for call in mock_warning.call_args_list
            if "unsupported model for account" in str(call.args[0])
        )
        assert (
            unsupported_warning.kwargs["extra"]["failure_kind"]
            == "openai_chatgpt_codex_model_not_supported_for_account"
        )
        assert unsupported_warning.kwargs["extra"]["upstream_url"] == target_url
        mock_logging_obj.post_call_failure_hook.assert_awaited_once()
        assert (
            mock_logging_obj.post_call_failure_hook.await_args.kwargs["traceback_str"]
            is None
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_chatgpt_codex_unrelated_400_logs_without_traceback(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_detail = b'{"detail":"Unsupported content type"}'
        upstream_response = httpx.Response(
            status_code=400,
            content=upstream_detail,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "gpt-5.5"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch.object(
            verbose_proxy_logger,
            "debug",
        ) as mock_debug, patch.object(
            verbose_proxy_logger,
            "exception",
        ) as mock_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "gpt-5.5"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    custom_llm_provider="openai",
                    stream=False,
                )

        assert exc_info.value.code == "400"
        assert str(upstream_detail, "utf-8") in str(exc_info.value.detail)
        mock_exception.assert_not_called()
        debug_call = next(
            call
            for call in mock_debug.call_args_list
            if "recorded handled client/provider error" in str(call.args[0])
        )
        assert debug_call.args[1] == 400
        assert debug_call.args[2] == "Unsupported content type"
        assert (
            debug_call.kwargs["extra"]["failure_kind"]
            == "handled_upstream_client_error"
        )
        mock_logging_obj.post_call_failure_hook.assert_awaited_once()
        assert (
            mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                "traceback_str"
            ]
            is None
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_openai_model_not_found_warns_without_traceback(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.openai.com/v1/responses"
        upstream_detail = (
            b'{"error":{"message":"The requested model \'aawm-code\' does not '
            b'exist.","type":"invalid_request_error","param":"model",'
            b'"code":"model_not_found"}}'
        )
        upstream_response = httpx.Response(
            status_code=400,
            content=upstream_detail,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "aawm-code"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch.object(
            verbose_proxy_logger,
            "warning",
        ) as mock_warning, patch.object(
            verbose_proxy_logger,
            "exception",
        ) as mock_exception, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.record_aawm_route_rollup_failure"
        ) as mock_rollup_failure:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "aawm-code"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    custom_llm_provider="openai",
                    stream=False,
                )

        assert exc_info.value.code == "400"
        mock_exception.assert_not_called()
        warning_call = next(
            call
            for call in mock_warning.call_args_list
            if "OpenAI model-not-found" in str(call.args[0])
        )
        assert warning_call.args[1] == 400
        assert warning_call.args[2] == (
            "The requested model 'aawm-code' does not exist."
        )
        assert "{" not in warning_call.args[2]
        assert (
            warning_call.kwargs["extra"]["failure_kind"]
            == "openai_model_not_found"
        )
        mock_rollup_failure.assert_called_once()
        assert mock_rollup_failure.call_args.kwargs == {
            "message": "The requested model 'aawm-code' does not exist."
        }
        mock_logging_obj.post_call_failure_hook.assert_awaited_once()
        assert (
            mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                "traceback_str"
            ]
            is None
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_exhausted_chatgpt_codex_503_logs_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/openai_passthrough/codex/responses"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://chatgpt.com/backend-api/codex/responses"
        upstream_detail = (
            b"upstream connect error or disconnect/reset before headers. "
            b"reset reason: connection termination"
        )
        upstream_response = httpx.Response(
            status_code=503,
            content=upstream_detail,
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "gpt-5.4"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(side_effect=fake_sleep),
            ), patch.object(
                verbose_proxy_logger,
                "exception",
            ) as mock_exception:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "gpt-5.4"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        custom_llm_provider="openai",
                        stream=False,
                    )

                assert exc_info.value.code == "503"
                assert handler.await_count == 6
                assert sleep_calls == [5.0, 15.0, 30.0, 60.0, 120.0]
                mock_exception.assert_not_called()
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "exhausted hidden retries for upstream failure" in item["message"]
        )

        assert payload["context"]["status_code"] == 503
        assert payload["context"]["upstream_url"] == target_url
        assert payload["context"]["failure_kind"] == (
            "expected_upstream_capacity_or_internal"
        )
        assert payload["context"]["hidden_retry_final_outcome"] == (
            "failed_after_retry"
        )
        assert payload["context"]["hidden_retry_count"] == 6
        assert "final_outcome=failed_after_retry" in payload["message"]
        assert "retry_count=6" in payload["message"]
        assert "connection termination" in payload["message"]
        assert payload.get("traceback") in (None, "")

    @pytest.mark.asyncio
    async def test_pass_through_request_exhausted_dns_failure_uses_503_without_traceback(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/anthropic/v1/messages?beta=true"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {"beta": "true"}

        target_url = "https://api.anthropic.com/v1/messages"
        dns_error = httpx.ConnectError(
            "Cannot connect to host api.anthropic.com:443 "
            "[Temporary failure in name resolution]",
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(side_effect=dns_error)
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "claude-opus-4-8"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(side_effect=fake_sleep),
            ):
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-opus-4-8"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )

                assert exc_info.value.code == "503"
                assert handler.await_count == 6
                assert sleep_calls == [5.0, 15.0, 30.0, 60.0, 120.0]
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "exhausted hidden retries for upstream failure" in item["message"]
        )

        assert payload["context"]["status_code"] == 503
        assert payload["context"]["failure_kind"] == "transient_provider_connectivity"
        assert payload["context"]["hidden_retry_final_outcome"] == (
            "failed_after_retry"
        )
        assert payload["context"]["hidden_retry_failure_classification"] == (
            "transport_dns_failure"
        )
        assert payload["context"]["hidden_retry_count"] == 6
        assert payload.get("traceback") in (None, "")

    @pytest.mark.asyncio
    async def test_pass_through_request_exhausted_read_error_retries_as_connectivity(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/anthropic/v1/messages?beta=true"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {"beta": "true"}

        target_url = "https://api.anthropic.com/v1/messages"
        read_error = httpx.ReadError(
            "Server disconnected",
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(side_effect=read_error)
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "claude-opus-4-8"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(side_effect=fake_sleep),
            ):
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-opus-4-8"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )

                assert exc_info.value.code == "500"
                assert handler.await_count == 6
                assert sleep_calls == [5.0, 15.0, 30.0, 60.0, 120.0]
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "exhausted hidden retries for upstream failure" in item["message"]
        )

        assert payload["context"]["status_code"] is None
        assert payload["context"]["failure_kind"] == "transient_provider_connectivity"
        assert payload["context"]["hidden_retry_final_outcome"] == (
            "failed_after_retry"
        )
        assert payload["context"]["hidden_retry_failure_classification"] == (
            "upstream_connectivity_failure"
        )
        assert payload["context"]["hidden_retry_count"] == 6
        assert payload.get("traceback") in (None, "")

    def test_terminal_failure_log_record_includes_hidden_retry_context_fields(self):
        metadata = {
            "aawm_passthrough_hidden_retry_final_outcome": "failed_after_retry",
            "aawm_passthrough_hidden_retry_failure_classification": "upstream_connectivity_failure",
            "aawm_passthrough_hidden_retry_count": 6,
        }
        kwargs = {"litellm_params": {"metadata": metadata}}
        hidden_retry_metadata = kwargs["litellm_params"]["metadata"]
        terminal_failure_context = {
            "status_code": 529,
            "model": "claude-sonnet-4-6",
            "failure_kind": "transient_provider_connectivity",
            "hidden_retry_final_outcome": hidden_retry_metadata.get(
                "aawm_passthrough_hidden_retry_final_outcome"
            ),
            "hidden_retry_failure_classification": hidden_retry_metadata.get(
                "aawm_passthrough_hidden_retry_failure_classification"
            ),
            "hidden_retry_count": hidden_retry_metadata.get(
                "aawm_passthrough_hidden_retry_count"
            ),
        }

        record = logging.LogRecord(
            name="LiteLLM Proxy",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=(
                "Pass through endpoint exhausted hidden retries for upstream failure "
                "status=%s error=%s final_outcome=%s retry_count=%s"
            ),
            args=(529, "overloaded", "failed_after_retry", 6),
            exc_info=None,
        )
        for key, value in terminal_failure_context.items():
            setattr(record, key, value)

        payload = _build_aawm_error_log_record(
            record,
            formatter=logging.Formatter(),
        )

        assert getattr(record, "failure_kind") == "transient_provider_connectivity"
        assert getattr(record, "hidden_retry_final_outcome") == "failed_after_retry"
        assert (
            getattr(record, "hidden_retry_failure_classification")
            == "upstream_connectivity_failure"
        )
        assert getattr(record, "hidden_retry_count") == 6
        assert payload["context"]["status_code"] == 529
        assert payload["context"]["model"] == "claude-sonnet-4-6"
        for field, expected in (
            ("failure_kind", "transient_provider_connectivity"),
            ("hidden_retry_final_outcome", "failed_after_retry"),
            (
                "hidden_retry_failure_classification",
                "upstream_connectivity_failure",
            ),
            ("hidden_retry_count", 6),
        ):
            assert getattr(record, field) == expected
            assert payload["context"].get(field) == expected

    @pytest.mark.asyncio
    async def test_pass_through_request_non_classified_failure_keeps_exception_logging(
        self,
        monkeypatch,
        tmp_path,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        handler = AsyncMock(side_effect=RuntimeError("unexpected passthrough failure"))

        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"model": "claude-sonnet-4-6"},
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-sonnet-4-6"}
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=mock_request,
                        target=target_url,
                        custom_headers={"authorization": "Bearer test"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                    )
                assert "unexpected passthrough failure" in str(exc_info.value.message)

                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    "unexpected passthrough failure"
                    in mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "Exception occured - unexpected passthrough failure" in item["message"]
        )
        assert "RuntimeError: unexpected passthrough failure" in payload["traceback"]

    def _build_grok_signals_401_fixture(self, upstream_detail: str):
        session_id = "019ed451-dddf-70d3-a2f9-9e3547b93358"
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = (
            f"http://localhost:4001/grok/v1/sessions/{session_id}/signals"
        )
        raw_body = json.dumps(
            [{"signalType": "heartbeat"}, {"signalType": "ping"}],
            separators=(",", ":"),
        ).encode("utf-8")
        raw_body_length = len(raw_body)
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.body = AsyncMock(return_value=raw_body)

        target_url = f"https://cli-chat-proxy.grok.com/v1/sessions/{session_id}/signals"
        upstream_response = httpx.Response(
            status_code=401,
            content=upstream_detail.encode("utf-8"),
            request=httpx.Request("POST", target_url),
        )
        passthrough_logging_metadata = {
            "client_name": "grok-build",
            "passthrough_route_family": "grok_cli_chat_proxy",
            "route_family": "grok_cli_chat_proxy",
            "grok_side_channel": True,
            "grok_side_channel_endpoint_type": "sessions_signals",
            "grok_side_channel_endpoint_path_template": (
                "/sessions/{session_id}/signals"
            ),
            "grok_side_channel_request_content_type": "application/json",
            "grok_side_channel_request_body_byte_length": raw_body_length,
            "grok_side_channel_request_body_digest_source": "raw_body",
            "grok_side_channel_request_json_container_type": "array",
            "grok_side_channel_request_array_length": 2,
            "grok_side_channel_request_body_sha256": "body-hash-should-not-log",
            "grok_side_channel_request_top_level_key_types": {
                "secretField": "str",
            },
        }
        return {
            "session_id": session_id,
            "request": mock_request,
            "raw_body_length": raw_body_length,
            "target_url": target_url,
            "upstream_response": upstream_response,
            "passthrough_logging_metadata": passthrough_logging_metadata,
        }

    @pytest.mark.asyncio
    async def test_pass_through_request_known_grok_signals_auth_context_warns_without_traceback(  # noqa: PLR0915
        self,
        monkeypatch,
        tmp_path,
    ):
        upstream_detail = (
            '{"error":"Invalid or expired credentials '
            "(auth_kind=bearer, x_xai_token_auth=xai-grok-cli, "
            'upstream=PermissionDenied, reason=no auth context)"}'
        )
        fixture = self._build_grok_signals_401_fixture(upstream_detail)
        handler = AsyncMock(return_value=fixture["upstream_response"])
        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ) as mock_sleep, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
                new=AsyncMock(),
            ) as mock_direct_capture, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    side_effect=lambda **kwargs: kwargs["data"]
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture["request"],
                        target=fixture["target_url"],
                        custom_headers={"authorization": "Bearer stale-grok-token"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                        custom_llm_provider="xai",
                        caller_managed_hidden_retry=True,
                        retryable_upstream_status_codes=[500, 502, 503, 504],
                        raw_body_passthrough=True,
                        passthrough_logging_metadata=fixture[
                            "passthrough_logging_metadata"
                        ],
                    )

                assert exc_info.value.code == "401"
                assert handler.await_count == 1
                mock_sleep.assert_not_awaited()
                mock_logging_obj.post_call_failure_hook.assert_not_awaited()
                mock_direct_capture.assert_not_awaited()
                mock_warning.assert_called_once()
                assert (
                    mock_warning.call_args.kwargs["extra"]["failure_kind"]
                    == "degraded_grok_signals_auth_context"
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    def _build_anthropic_upstream_failure_fixture(
        self,
        *,
        status_code: int,
        upstream_detail: str,
        model: str,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4001/anthropic/v1/messages?beta=true"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-oat01-fake-token-for-testing",
            "anthropic-version": "2023-06-01",
        }
        mock_request.query_params = {"beta": "true"}

        target_url = "https://api.anthropic.com/v1/messages"
        upstream_response = httpx.Response(
            status_code=status_code,
            content=upstream_detail.encode("utf-8"),
            request=httpx.Request("POST", target_url),
        )
        return {
            "request": mock_request,
            "request_body": {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "target_url": target_url,
            "upstream_response": upstream_response,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "upstream_detail", "model", "expected_failure_kind"),
        [
            (
                400,
                (
                    '{"type":"error","error":{"type":"invalid_request_error",'
                    '"message":"prompt is too long: 1002056 tokens > 1000000 maximum"},'
                    '"request_id":"req_context"}'
                ),
                "claude-opus-4-8",
                "anthropic_context_overflow",
            ),
            (
                401,
                (
                    '{"type":"error","error":{"type":"authentication_error",'
                    '"message":"Invalid authentication credentials"},'
                    '"request_id":"req_auth"}'
                ),
                "claude-haiku-4-5-20251001",
                "anthropic_client_authentication_error",
            ),
            (
                404,
                (
                    '{"type":"error","error":{"type":"not_found_error",'
                    '"message":"model: opus-4-8"},'
                    '"request_id":"req_model"}'
                ),
                "opus-4-8",
                "anthropic_model_not_found",
            ),
        ],
    )
    async def test_pass_through_request_known_anthropic_client_failures_warn_without_traceback(  # noqa: PLR0915
        self,
        monkeypatch,
        tmp_path,
        status_code,
        upstream_detail,
        model,
        expected_failure_kind,
    ):
        fixture = self._build_anthropic_upstream_failure_fixture(
            status_code=status_code,
            upstream_detail=upstream_detail,
            model=model,
        )
        handler = AsyncMock(return_value=fixture["upstream_response"])
        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    side_effect=lambda **kwargs: kwargs["data"]
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture["request"],
                        target=fixture["target_url"],
                        custom_headers={},
                        custom_body=fixture["request_body"],
                        user_api_key_dict=MagicMock(),
                        stream=False,
                        custom_llm_provider=None,
                        caller_managed_hidden_retry=True,
                        retryable_upstream_status_codes=[500, 502, 503, 504],
                    )

                assert exc_info.value.code == str(status_code)
                assert handler.await_count == 1
                mock_warning.assert_called_once()
                warning_context = mock_warning.call_args.kwargs["extra"]
                assert warning_context["failure_kind"] == expected_failure_kind
                assert warning_context["upstream_url"] == fixture["target_url"]
                assert warning_context["model"] == model
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    @pytest.mark.asyncio
    async def test_pass_through_request_known_anthropic_context_overflow_preserves_upstream_body(
        self,
        monkeypatch,
        tmp_path,
    ):
        upstream_detail = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"prompt is too long: 1002056 tokens > 1000000 maximum"},'
            '"request_id":"req_011CcMsoxd3XmSY8aU3rsWXj"}'
        )
        fixture = self._build_anthropic_upstream_failure_fixture(
            status_code=400,
            upstream_detail=upstream_detail,
            model="claude-opus-4-8",
        )
        handler = AsyncMock(return_value=fixture["upstream_response"])
        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ) as mock_sleep, patch.object(
                verbose_proxy_logger,
                "warning",
            ) as mock_warning:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    side_effect=lambda **kwargs: kwargs["data"]
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture["request"],
                        target=fixture["target_url"],
                        custom_headers={},
                        custom_body=fixture["request_body"],
                        user_api_key_dict=MagicMock(),
                        stream=False,
                        custom_llm_provider=None,
                        caller_managed_hidden_retry=True,
                        retryable_upstream_status_codes=[500, 502, 503, 504],
                    )

                assert exc_info.value.code == "400"
                assert exc_info.value.detail == upstream_detail
                assert handler.await_count == 1
                mock_sleep.assert_not_awaited()
                mock_warning.assert_called_once()
                warning_context = mock_warning.call_args.kwargs["extra"]
                assert warning_context["failure_kind"] == "anthropic_context_overflow"
                assert warning_context["upstream_url"] == fixture["target_url"]
                assert warning_context["model"] == "claude-opus-4-8"
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        assert not (tmp_path / "dev-error.jsonl").exists()

    @pytest.mark.asyncio
    async def test_pass_through_request_unexpected_grok_signals_401_keeps_redacted_jsonl_intake(  # noqa: PLR0915
        self,
        monkeypatch,
        tmp_path,
    ):
        upstream_detail = '{"error":"grok signals authorization failed"}'
        fixture = self._build_grok_signals_401_fixture(upstream_detail)
        handler = AsyncMock(return_value=fixture["upstream_response"])
        # digest_source is content-bearing and only persisted when explicitly enabled.
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", "1")
        (
            saved_handlers,
            saved_level,
            saved_propagate,
        ) = self._install_aawm_error_log_handler(tmp_path, monkeypatch)

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=handler,
            ), patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client, patch(
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as mock_logging_obj, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
                new=AsyncMock(),
            ) as mock_sleep, patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
                new=AsyncMock(),
            ) as mock_direct_capture:
                mock_client_obj = MagicMock()
                mock_client_obj.client = MagicMock()
                mock_get_client.return_value = mock_client_obj
                mock_logging_obj.pre_call_hook = AsyncMock(
                    side_effect=lambda **kwargs: kwargs["data"]
                )
                mock_logging_obj.post_call_failure_hook = AsyncMock()

                with pytest.raises(ProxyException) as exc_info:
                    await pass_through_request(
                        request=fixture["request"],
                        target=fixture["target_url"],
                        custom_headers={"authorization": "Bearer stale-grok-token"},
                        user_api_key_dict=MagicMock(),
                        stream=False,
                        custom_llm_provider="xai",
                        caller_managed_hidden_retry=True,
                        retryable_upstream_status_codes=[500, 502, 503, 504],
                        raw_body_passthrough=True,
                        passthrough_logging_metadata=fixture[
                            "passthrough_logging_metadata"
                        ],
                    )

                assert exc_info.value.code == "401"
                assert handler.await_count == 1
                mock_sleep.assert_not_awaited()
                mock_logging_obj.post_call_failure_hook.assert_awaited_once()
                assert (
                    mock_logging_obj.post_call_failure_hook.await_args.kwargs[
                        "traceback_str"
                    ]
                    is None
                )
                mock_direct_capture.assert_awaited_once()
        finally:
            self._restore_verbose_proxy_logger(
                saved_handlers,
                saved_level,
                saved_propagate,
            )

        error_log_path = tmp_path / "dev-error.jsonl"
        payloads = [
            json.loads(line)
            for line in error_log_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = next(
            item
            for item in payloads
            if "handled client/provider error status=401" in item["message"]
        )
        context = payload["context"]
        assert not payload.get("traceback")

        assert context["status_code"] == 401
        assert context["endpoint"] == "/grok/v1/sessions/{session_id}/signals"
        assert (
            context["upstream_url"]
            == "https://cli-chat-proxy.grok.com/v1/sessions/{session_id}/signals"
        )
        assert context["grok_side_channel"] is True
        assert context["grok_side_channel_endpoint_type"] == "sessions_signals"
        assert (
            context["grok_side_channel_endpoint_path_template"]
            == "/sessions/{session_id}/signals"
        )
        assert context["grok_side_channel_request_content_type"] == "application/json"
        assert (
            context["grok_side_channel_request_body_byte_length"]
            == fixture["raw_body_length"]
        )
        assert context["grok_side_channel_request_body_digest_source"] == "raw_body"
        assert context["grok_side_channel_request_json_container_type"] == "array"
        assert context["grok_side_channel_request_array_length"] == 2

        serialized_context = json.dumps(context)
        assert fixture["session_id"] not in serialized_context
        assert "body-hash-should-not-log" not in serialized_context
        assert "secretField" not in serialized_context
        assert "grok_side_channel_request_body_sha256" not in serialized_context
        assert "grok_side_channel_request_top_level_key_types" not in serialized_context
        assert "grok signals authorization failed" in payload["message"]


class TestPassThroughHiddenRetry:
    def test_hidden_retry_metadata_initializes_empty_metadata(self):
        kwargs: dict = {}

        _record_passthrough_hidden_retry_metadata(
            kwargs,
            attempt_number=1,
            max_attempts=6,
            status_code=529,
            failure_class="http_status_529",
            wait_seconds=5.0,
            failure_classification=None,
        )

        metadata = kwargs["litellm_params"]["metadata"]
        assert metadata["aawm_passthrough_hidden_retry_count"] == 1
        assert metadata["aawm_passthrough_hidden_retry_attempts"] == [
            {
                "attempt": 1,
                "max_attempts": 6,
                "failure_class": "http_status_529",
                "wait_seconds": 5.0,
                "status_code": 529,
            }
        ]

    @pytest.mark.asyncio
    async def test_hidden_retry_attempt_progress_logs_info_not_warning(self):
        kwargs: dict = {}
        operation = AsyncMock(
            side_effect=[
                HTTPException(status_code=529, detail="overloaded"),
                "ok",
            ]
        )

        with patch.object(verbose_proxy_logger, "info") as mock_info, patch.object(
            verbose_proxy_logger,
            "warning",
        ) as mock_warning, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ):
            result = await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs=kwargs,
                operation_name="non_stream_pre_first_byte",
                operation=operation,
                caller_managed_hidden_retry=False,
            )

        assert result == "ok"
        mock_info.assert_called_once()
        assert "hidden retry attempt" in mock_info.call_args.args[0]
        mock_warning.assert_not_called()

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_pass_through_request_retries_non_stream_transient_status_then_succeeds(
        self,
        status_code,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        failing_response = httpx.Response(
            status_code=status_code,
            content=b'{"error":"internal"}',
            request=httpx.Request("POST", target_url),
        )
        success_response = httpx.Response(
            status_code=200,
            content=b'{"id":"msg_1"}',
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(side_effect=[failing_response, success_response])
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(side_effect=fake_sleep),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new_callable=AsyncMock,
        ) as mock_success_handler:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6"}
            )

            response = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer test"},
                user_api_key_dict=MagicMock(),
                stream=False,
            )

        assert response.status_code == 200
        assert handler.await_count == 2
        assert sleep_calls == [5.0]
        metadata = mock_success_handler.call_args.kwargs["litellm_params"]["metadata"]
        assert metadata["aawm_passthrough_hidden_retry_final_outcome"] == (
            "success_after_retry"
        )
        assert (
            metadata["aawm_passthrough_hidden_retry_attempts"][0]["status_code"]
            == status_code
        )
        assert (
            metadata["aawm_passthrough_hidden_retry_attempts"][-1]["failure_class"]
            == "success"
        )

    @pytest.mark.asyncio
    async def test_pass_through_request_retries_transport_dns_failure(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        success_response = httpx.Response(
            status_code=200,
            content=b'{"id":"msg_1"}',
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(
            side_effect=[
                httpx.ConnectError(
                    "Temporary failure in name resolution",
                    request=httpx.Request("POST", target_url),
                ),
                success_response,
            ]
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new_callable=AsyncMock,
        ):
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6"}
            )

            response = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer test"},
                user_api_key_dict=MagicMock(),
                stream=False,
            )

        assert response.status_code == 200
        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_hidden_retry_exhausts_timeout_failures_with_metadata(
        self, monkeypatch
    ):
        # Disable wall-clock bound so this case exercises the attempt-count ceiling.
        monkeypatch.setenv("AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", "0")
        kwargs: dict = {}
        target_url = "https://api.anthropic.com/v1/messages"
        operation = AsyncMock(
            side_effect=httpx.ReadTimeout(
                "read timed out",
                request=httpx.Request("POST", target_url),
            )
        )
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(side_effect=fake_sleep),
        ):
            with pytest.raises(httpx.ReadTimeout):
                await _execute_passthrough_pre_first_byte_with_hidden_retries(
                    kwargs=kwargs,
                    operation_name="test_operation",
                    operation=operation,
                    caller_managed_hidden_retry=False,
                )

        assert operation.await_count == 6
        assert sleep_calls == [5.0, 15.0, 30.0, 60.0, 120.0]
        metadata = kwargs["litellm_params"]["metadata"]
        assert metadata["aawm_passthrough_hidden_retry_final_outcome"] == (
            "failed_after_retry"
        )
        assert metadata["aawm_passthrough_hidden_retry_failure_classification"] == (
            "upstream_connectivity_failure"
        )
        assert metadata["aawm_passthrough_hidden_retry_count"] == 6

    def test_hidden_retry_budget_defaults_to_backoff_sum(self, monkeypatch):
        monkeypatch.delenv(
            "AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", raising=False
        )
        assert _get_passthrough_hidden_retry_budget_seconds() == 230.0

    def test_hidden_retry_budget_env_override(self, monkeypatch):
        monkeypatch.setenv("AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", "12.5")
        assert _get_passthrough_hidden_retry_budget_seconds() == 12.5
        monkeypatch.setenv("AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", "0")
        assert _get_passthrough_hidden_retry_budget_seconds() == 0.0

    @pytest.mark.asyncio
    async def test_hidden_retry_stops_when_wall_clock_budget_exceeded(
        self, monkeypatch
    ):
        """B2 / RR-056 #1: wall-clock ceiling stops retries before attempt budget."""
        # First backoff is 5s; with a 4s budget the next wait alone exceeds the ceiling.
        monkeypatch.setenv("AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", "4")
        kwargs: dict = {}
        target_url = "https://api.anthropic.com/v1/messages"
        operation = AsyncMock(
            side_effect=httpx.ReadTimeout(
                "read timed out",
                request=httpx.Request("POST", target_url),
            )
        )
        sleep_mock = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=sleep_mock,
        ):
            with pytest.raises(httpx.ReadTimeout):
                await _execute_passthrough_pre_first_byte_with_hidden_retries(
                    kwargs=kwargs,
                    operation_name="test_operation",
                    operation=operation,
                    caller_managed_hidden_retry=False,
                )

        # Only the first attempt runs; no sleep, no further attempts.
        assert operation.await_count == 1
        sleep_mock.assert_not_awaited()
        metadata = kwargs["litellm_params"]["metadata"]
        assert metadata["aawm_passthrough_hidden_retry_final_outcome"] == (
            "failed_without_retry"
        )
        assert metadata["aawm_passthrough_hidden_retry_budget_exhausted"] is True
        assert metadata["aawm_passthrough_hidden_retry_budget_seconds"] == 4.0
        assert metadata["aawm_passthrough_hidden_retry_count"] == 1
        assert metadata["aawm_passthrough_hidden_retry_failure_classification"] == (
            "upstream_connectivity_failure"
        )

    @pytest.mark.asyncio
    async def test_hidden_retry_stops_mid_loop_when_elapsed_exceeds_budget(
        self, monkeypatch
    ):
        """Elapsed wall clock (attempt runtime) can exhaust the budget mid-loop."""
        monkeypatch.setenv("AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS", "20")
        kwargs: dict = {}
        target_url = "https://api.anthropic.com/v1/messages"
        operation = AsyncMock(
            side_effect=httpx.ConnectError(
                "connection refused",
                request=httpx.Request("POST", target_url),
            )
        )
        sleep_mock = AsyncMock()
        # Attempt 1: elapsed 0 → wait 5 allowed → sleep.
        # Attempt 2: elapsed 18 → wait 15 would exceed 20 → stop.
        monotonic_values = iter([100.0, 100.0, 118.0, 118.0])

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=sleep_mock,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            with pytest.raises(httpx.ConnectError):
                await _execute_passthrough_pre_first_byte_with_hidden_retries(
                    kwargs=kwargs,
                    operation_name="test_operation",
                    operation=operation,
                    caller_managed_hidden_retry=False,
                )

        assert operation.await_count == 2
        sleep_mock.assert_awaited_once_with(5.0)
        metadata = kwargs["litellm_params"]["metadata"]
        assert metadata["aawm_passthrough_hidden_retry_final_outcome"] == (
            "failed_after_retry"
        )
        assert metadata["aawm_passthrough_hidden_retry_budget_exhausted"] is True
        assert metadata["aawm_passthrough_hidden_retry_count"] == 2

    @pytest.mark.asyncio
    async def test_pass_through_request_does_not_retry_429(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        upstream_response = httpx.Response(
            status_code=429,
            content=b'{"error":"throttled"}',
            headers={"retry-after": "17"},
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ) as mock_sleep:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    stream=False,
                )

        assert exc_info.value.code == "429"
        assert handler.await_count == 1
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pass_through_request_retries_stream_pre_yield_status(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        req = httpx.Request("POST", target_url)
        failing_response = httpx.Response(
            status_code=529,
            content=b'{"error":"overloaded"}',
            request=req,
        )
        success_response = httpx.Response(
            status_code=200,
            content=b"event: message_start\n\ndata: {}\n\n",
            headers={"content-type": "text/event-stream"},
            request=req,
        )

        async def fake_send(request, stream=True):
            if fake_send.calls == 0:
                fake_send.calls += 1
                return failing_response
            fake_send.calls += 1
            return success_response

        fake_send.calls = 0

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=req)
        mock_client.send = AsyncMock(side_effect=fake_send)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6", "stream": True},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughStreamingHandler.chunk_processor",
            return_value=iter([b"chunk"]),
        ):
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6", "stream": True}
            )

            response = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer test"},
                user_api_key_dict=MagicMock(),
                stream=True,
            )

        assert response.status_code == 200
        assert mock_client.send.await_count == 2

    @pytest.mark.asyncio
    async def test_pass_through_request_does_not_retry_after_streaming_response_returned(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        req = httpx.Request("POST", target_url)
        success_response = httpx.Response(
            status_code=200,
            content=b"event: message_start\n\ndata: {}\n\n",
            headers={"content-type": "text/event-stream"},
            request=req,
        )

        async def post_first_byte_failure():
            yield b"event: message_start\n\n"
            raise RuntimeError("post-first-byte stream failure")

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=req)
        mock_client.send = AsyncMock(return_value=success_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6", "stream": True},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ) as mock_sleep, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughStreamingHandler.chunk_processor",
            return_value=post_first_byte_failure(),
        ):
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6", "stream": True}
            )

            response = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer test"},
                user_api_key_dict=MagicMock(),
                stream=True,
            )
            with pytest.raises(RuntimeError, match="post-first-byte stream failure"):
                async for _chunk in response.body_iterator:
                    pass

        assert mock_client.send.await_count == 1
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pass_through_request_opt_out_skips_hidden_retry(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.anthropic.com/v1/messages"
        upstream_response = httpx.Response(
            status_code=500,
            content=b'{"error":"internal"}',
            request=httpx.Request("POST", target_url),
        )
        handler = AsyncMock(return_value=upstream_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"model": "claude-sonnet-4-6"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=handler,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
            new=AsyncMock(),
        ) as mock_sleep:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"model": "claude-sonnet-4-6"}
            )
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException):
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    stream=False,
                    caller_managed_hidden_retry=True,
                )

        assert handler.await_count == 1
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_pass_through_request_forwards_raw_body_without_json_parse():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.headers = Headers({"Content-Type": "application/x-protobuf"})
    mock_request.query_params = QueryParams("")
    mock_request.body = AsyncMock(return_value=b"\x08\x01native")

    mock_user_api_key_dict = MagicMock()
    request_for_response = httpx.Request("POST", "https://example.com/v1/traces")
    mock_response = httpx.Response(
        status_code=204,
        request=request_for_response,
        content=b"",
        headers={"content-type": "application/json"},
    )

    mock_async_client = MagicMock()
    mock_async_client_obj = MagicMock(client=mock_async_client)

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_async_client_obj,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=mock_response),
    ) as mock_http_handler, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ):
        response = await pass_through_request(
            request=mock_request,
            target="https://example.com/v1/traces",
            custom_headers={},
            user_api_key_dict=mock_user_api_key_dict,
            stream=False,
            raw_body_passthrough=True,
        )

    assert response.status_code == 204
    mock_request.body.assert_awaited_once()
    call_kwargs = mock_http_handler.await_args.kwargs
    assert call_kwargs["raw_body"] == b"\x08\x01native"
    assert call_kwargs["_parsed_body"]["raw_body_passthrough"] is True
    assert (
        call_kwargs["_parsed_body"]["raw_body_content_type"] == "application/x-protobuf"
    )
    assert call_kwargs["_parsed_body"]["raw_body_bytes"] == len(b"\x08\x01native")


@pytest.mark.asyncio
async def test_non_streaming_http_request_handler_preserves_content_type_for_raw_body():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.headers = Headers({"content-type": "application/x-protobuf"})

    mock_response = httpx.Response(
        status_code=204,
        request=httpx.Request("POST", "https://example.com/v1/traces"),
        content=b"",
    )

    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    raw_body = b"\x08\x01native"
    headers = {
        "content-type": "application/x-protobuf",
        "authorization": "Bearer test",
    }

    response = await HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
        request=mock_request,
        async_client=async_client,
        url=httpx.URL("https://example.com/v1/traces"),
        headers=headers,
        raw_body=raw_body,
    )

    assert response.status_code == 204
    async_client.request.assert_awaited_once()
    call_kwargs = async_client.request.await_args.kwargs
    assert call_kwargs["content"] == raw_body
    assert call_kwargs["headers"]["content-type"] == "application/x-protobuf"
    assert call_kwargs["headers"]["authorization"] == "Bearer test"


@pytest.mark.asyncio
async def test_pass_through_request_full_payload_capture_uses_upstream_http_request(
    tmp_path,
    monkeypatch,
):
    control_file = tmp_path / "full-payload.enabled"
    capture_dir = tmp_path / "full-payloads"
    control_file.write_text("1", encoding="utf-8")
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_SHAPES", raising=False)
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", raising=False)
    monkeypatch.setenv(
        "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_CONTROL_FILE",
        str(control_file),
    )
    monkeypatch.setenv(
        "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_DIR",
        str(capture_dir),
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4001/openai_passthrough/v1/responses"
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams("")

    target_url = "https://cli-chat-proxy.grok.com/v1/responses"
    upstream_request = httpx.Request(
        "POST",
        target_url,
        headers={
            "authorization": "Bearer full-upstream-token",
            "x-xai-token-auth": "full-xai-token",
            "content-type": "application/json",
        },
        content=b'{"model":"grok-composer-2.5-fast","input":"wire body"}',
    )
    mock_response = httpx.Response(
        status_code=200,
        headers={
            "authorization": "Bearer response-token",
            "x-ratelimit-remaining-requests": "900",
            "content-type": "application/json",
        },
        content=b'{"id":"resp_1","model":"grok-composer-2.5-fast"}',
        request=upstream_request,
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new_callable=AsyncMock,
    ):
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        # RR-056 #6 non-stream path uses send(stream=True) then aread().
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_obj = MagicMock(client=mock_client)
        mock_get_client.return_value = mock_client_obj

        mock_logging_obj.pre_call_hook = AsyncMock(
            side_effect=lambda **kwargs: kwargs["data"]
        )
        mock_logging_obj.post_call_success_hook = AsyncMock()
        mock_logging_obj.post_call_failure_hook = AsyncMock()

        await pass_through_request(
            request=mock_request,
            target=target_url,
            custom_headers={"authorization": "Bearer full-upstream-token"},
            user_api_key_dict=MagicMock(),
            custom_body={
                "model": "grok-composer-2.5-fast",
                "input": "logging body",
                "litellm_metadata": {"source": "test"},
            },
            stream=False,
            custom_llm_provider="xai",
        )

    artifacts = list(capture_dir.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert artifact["capture_kind"] == "aawm_passthrough_full_payload"
    assert artifact["capture_scope"] == "upstream_http_transaction"
    assert artifact["request"]["method"] == "POST"
    assert artifact["request"]["url"] == target_url
    # A6: full-payload capture drops sensitive auth headers; keep non-secret wire metadata.
    request_headers = {
        str(k).lower(): v for k, v in artifact["request"]["headers"].items()
    }
    response_headers = {
        str(k).lower(): v for k, v in artifact["response"]["headers"].items()
    }
    assert "authorization" not in request_headers
    assert "x-xai-token-auth" not in request_headers
    assert request_headers.get("content-type") == "application/json"
    assert artifact["request"]["body"]["json"]["input"] == "wire body"
    assert "authorization" not in response_headers
    assert response_headers.get("content-type") == "application/json"
    assert artifact["response"]["body"]["json"]["model"] == "grok-composer-2.5-fast"


@pytest.mark.asyncio
async def test_pass_through_with_httpbin_redirect():
    """
    Integration test using httpbin.org redirect endpoint to test real redirect handling.
    This tests the actual redirect handling capability end-to-end using the full pass_through_request function.
    """
    from unittest.mock import MagicMock

    from fastapi import Request
    from starlette.datastructures import Headers, QueryParams

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.headers = Headers({})
    mock_request.query_params = QueryParams("")

    # Mock the body method to return empty bytes for GET request
    async def mock_body():
        return b""

    mock_request.body = mock_body

    # Mock user API key dict
    mock_user_api_key_dict = MagicMock()

    try:
        # Test with httpbin.org redirect endpoint
        # This will redirect to httpbin.org/get
        response = await pass_through_request(
            request=mock_request,
            target="https://httpbin.org/redirect/1",
            custom_headers={},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Should get the final response (200) from /get endpoint, not the redirect (302)
        assert response.status_code == 200

        # The response should be from the /get endpoint
        response_content = bytes(response.body).decode("utf-8")

        # httpbin.org/get returns JSON with info about the request
        assert '"url": "https://httpbin.org/get"' in response_content
    except Exception as e:
        # If httpbin.org is not accessible, skip the test
        import pytest

        pytest.skip(f"Could not reach httpbin.org for integration test: {e}")


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_with_filter():
    """
    Test that _filter_endpoints_by_team_allowed_routes correctly filters endpoints
    when team has allowed_passthrough_routes in metadata
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/allowed1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/allowed2", target="http://example.com/api2"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-3", path="/api/notallowed", target="http://example.com/api3"
        ),
    ]

    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {
        "allowed_passthrough_routes": ["/api/allowed1", "/api/allowed2"]
    }
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should only return allowed endpoints
    assert len(result) == 2
    assert result[0].path == "/api/allowed1"
    assert result[1].path == "/api/allowed2"

    # Verify database call
    mock_prisma_client.db.litellm_teamtable.find_unique.assert_called_once_with(
        where={"team_id": "test-team-123"}
    )


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_team_not_found():
    """
    Test that _filter_endpoints_by_team_allowed_routes raises HTTPException
    when team is not found
    """
    from fastapi import HTTPException

    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test", target="http://example.com/api"
        ),
    ]

    # Mock prisma client to return None (team not found)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Call the function and expect HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await _filter_endpoints_by_team_allowed_routes(
            team_id="non-existent-team",
            pass_through_endpoints=endpoints,
            prisma_client=mock_prisma_client,
        )

    # Verify the exception
    assert exc_info.value.status_code == 404
    assert "Team not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_no_metadata():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns all endpoints
    when team has no metadata
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has None metadata
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = None
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return all endpoints when no metadata
    assert len(result) == 2
    assert result[0].path == "/api/test1"
    assert result[1].path == "/api/test2"


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_no_allowed_routes_key():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns all endpoints
    when team metadata doesn't have allowed_passthrough_routes key
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has metadata but no allowed_passthrough_routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"some_other_key": "some_value"}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return all endpoints when allowed_passthrough_routes key doesn't exist
    assert len(result) == 2
    assert result[0].path == "/api/test1"
    assert result[1].path == "/api/test2"


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_empty_allowed_list():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns empty list
    when team has empty allowed_passthrough_routes list
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has empty allowed_passthrough_routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"allowed_passthrough_routes": []}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return empty list when allowed_passthrough_routes is empty
    assert len(result) == 0


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_partial_match():
    """
    Test that _filter_endpoints_by_team_allowed_routes correctly filters
    when only some endpoints match allowed routes
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/openai", target="http://example.com/openai"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2",
            path="/api/anthropic",
            target="http://example.com/anthropic",
        ),
        PassThroughGenericEndpoint(
            id="endpoint-3", path="/api/azure", target="http://example.com/azure"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-4", path="/api/cohere", target="http://example.com/cohere"
        ),
    ]

    # Mock prisma client with team that allows only 2 routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"allowed_passthrough_routes": ["/api/openai", "/api/azure"]}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return only the 2 allowed endpoints
    assert len(result) == 2
    assert result[0].path == "/api/openai"
    assert result[1].path == "/api/azure"


@pytest.mark.asyncio
async def test_bedrock_router_passthrough_metadata_initialization():
    """
    Test that bedrock router passthrough properly initializes metadata for hooks.

    This test verifies the fix for issue #15826 where metadata.headers and
    litellm_params.proxy_server_request were missing for /bedrock passthrough
    requests with router models.

    The fix ensures router bedrock models use the same common processing path
    as non-router models, which properly initializes all metadata structures.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        handle_bedrock_passthrough_router_model,
    )

    # Mock ProxyBaseLLMRequestProcessing to verify it's used
    with patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing"
    ) as mock_processing_class:
        # Setup mock instance
        mock_processor = MagicMock()
        mock_processing_class.return_value = mock_processor

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aread = AsyncMock(
            return_value=b'{"content": [{"text": "Hello"}]}'
        )
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value=mock_response
        )

        # Create mock request with headers
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4000/bedrock/model/my-model/invoke"
        mock_request.headers = Headers(
            {
                "content-type": "application/json",
                "authorization": "Bearer sk-test-key",
                "x-custom-header": "test-value",
            }
        )
        mock_request.query_params = QueryParams({})

        # Create mock user API key dict with all required fields
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "sk-test-key"
        mock_user_api_key_dict.key_alias = "test-alias"
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.team_id = "team-123"

        # Mock other required dependencies
        mock_router = MagicMock()
        mock_proxy_logging = MagicMock()
        mock_general_settings = {}
        mock_proxy_config = MagicMock()
        mock_select_data_generator = MagicMock()

        request_body = {
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "anthropic_version": "bedrock-2023-05-31",
        }

        # Call the function
        result = await handle_bedrock_passthrough_router_model(
            model="my-bedrock-model",
            endpoint="/model/my-bedrock-model/invoke",
            request=mock_request,
            request_body=request_body,
            llm_router=mock_router,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging,
            general_settings=mock_general_settings,
            proxy_config=mock_proxy_config,
            select_data_generator=mock_select_data_generator,
            user_model=None,
            user_temperature=None,
            user_request_timeout=None,
            user_max_tokens=None,
            user_api_base=None,
            version="1.0",
        )

        # Verify that ProxyBaseLLMRequestProcessing was instantiated
        # This is the KEY assertion - router models now use the common processing path
        mock_processing_class.assert_called_once()

        # Verify that base_passthrough_process_llm_request was called
        # This proves we're using the common processing path that initializes metadata
        mock_processor.base_passthrough_process_llm_request.assert_called_once()

        # Verify the call included all required parameters for proper metadata initialization
        call_kwargs = mock_processor.base_passthrough_process_llm_request.call_args[1]

        # These are the critical parameters that ensure metadata is properly initialized:
        assert (
            call_kwargs["request"] == mock_request
        ), "Request must be passed for header extraction"
        assert (
            call_kwargs["user_api_key_dict"] == mock_user_api_key_dict
        ), "User API key dict needed for metadata"
        assert (
            call_kwargs["proxy_logging_obj"] == mock_proxy_logging
        ), "Logging obj needed for hooks"
        assert (
            call_kwargs["llm_router"] == mock_router
        ), "Router needed for model routing"
        assert call_kwargs["model"] == "my-bedrock-model", "Model name must be passed"

        # Verify response was returned
        assert result == mock_response


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_adds_headers_to_metadata():
    """
    Test that add_litellm_data_to_request adds headers to metadata for guardrails.

    This test verifies the fix for issue #17477 where guardrails couldn't access
    request headers (like User-Agent) on Bedrock pass-through endpoints.

    The fix ensures headers are available in data["metadata"]["headers"] so
    guardrails can validate User-Agent, API keys, and other header-based checks.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Create mock request with headers including User-Agent
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = MagicMock()
    mock_request.url.path = "/bedrock/model/my-model/converse"
    mock_request.headers = Headers(
        {
            "content-type": "application/json",
            "user-agent": "claude-cli/2.0.69 (external, cli)",
            "authorization": "Bearer sk-test-key",
            "x-custom-header": "test-value",
        }
    )
    mock_request.query_params = QueryParams({})

    # Create mock user API key dict
    mock_user_api_key_dict = UserAPIKeyAuth()

    # Create mock proxy config
    mock_proxy_config = MagicMock()
    mock_proxy_config.pass_through_endpoints = []

    # Initial data dict (simulating Bedrock pass-through)
    data = {
        "model": "my-bedrock-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=mock_request,
        user_api_key_dict=mock_user_api_key_dict,
        proxy_config=mock_proxy_config,
        general_settings={},
        version="1.0",
    )

    # Verify headers are added to metadata for guardrails
    assert "metadata" in result, "metadata should be present in result"
    assert "headers" in result["metadata"], "headers should be present in metadata"
    assert isinstance(
        result["metadata"]["headers"], dict
    ), "headers should be a dictionary"

    # Verify specific headers are accessible (important for guardrails)
    headers = result["metadata"]["headers"]
    assert (
        "user-agent" in headers or "User-Agent" in headers
    ), "User-Agent header should be accessible in metadata"

    # Also verify proxy_server_request has headers (original location)
    assert "proxy_server_request" in result
    assert "headers" in result["proxy_server_request"]


@pytest.mark.asyncio
async def test_create_pass_through_route_custom_body_url_target():
    """
    Test that the URL-based endpoint_func created by create_pass_through_route
    accepts a custom_body parameter and forwards it to pass_through_request,
    taking precedence over the request-parsed body.

    This verifies the fix for issue #16999 where bedrock_proxy_route passes
    custom_body=data to the endpoint function, which previously crashed with:
    TypeError: endpoint_func() got an unexpected keyword argument 'custom_body'
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    unique_path = "/test/path/unique/custom_body_url"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="https://bedrock-agent-runtime.us-east-1.amazonaws.com",
        custom_headers={"Content-Type": "application/json"},
        _forward_headers=True,
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
    ) as mock_pass_through, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
    ) as mock_get_registered, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type"
    ) as mock_parse_request:
        mock_pass_through.return_value = MagicMock()
        mock_get_registered.return_value = {
            "passthrough_params": {},
        }
        # Simulate the request parser returning a different body
        mock_parse_request.return_value = (
            {},  # query_params_data
            {"parsed_from_request": True},  # custom_body_data (from request)
            None,  # file_data
            False,  # stream
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        # The caller-supplied body (e.g. from bedrock_proxy_route)
        bedrock_body = {
            "retrievalQuery": {"text": "What is in the knowledge base?"},
        }

        # Call endpoint_func with custom_body — this is the call that
        # used to crash with TypeError before the fix
        await endpoint_func(
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=mock_user_api_key_dict,
            custom_body=bedrock_body,
        )

        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]

        # The critical assertion: custom_body takes precedence over
        # the body parsed from the raw request
        assert call_kwargs["custom_body"] == bedrock_body


@pytest.mark.asyncio
async def test_create_pass_through_route_no_custom_body_falls_back():
    """
    Test that the URL-based endpoint_func falls back to the request-parsed body
    when custom_body is not provided.

    This ensures the default pass-through behavior is preserved — only the
    Bedrock proxy route (and similar callers) supply a pre-built body.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    unique_path = "/test/path/unique/no_custom_body"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com/api",
        custom_headers={},
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
    ) as mock_pass_through, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
    ) as mock_get_registered, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type"
    ) as mock_parse_request:
        mock_pass_through.return_value = MagicMock()
        mock_get_registered.return_value = {
            "passthrough_params": {},
        }
        request_parsed_body = {"key": "from_request"}
        mock_parse_request.return_value = (
            {},  # query_params_data
            request_parsed_body,  # custom_body_data
            None,  # file_data
            False,  # stream
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        # Call without custom_body — should use the request-parsed body
        await endpoint_func(
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]

        # Should fall back to the body parsed from the request
        assert call_kwargs["custom_body"] == request_parsed_body


def test_build_full_path_with_root_default():
    """
    Test _build_full_path_with_root with default root path (/)
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path"
    ) as mock_get_root:
        # Test with default root path
        mock_get_root.return_value = "/"

        result = InitPassThroughEndpointHelpers._build_full_path_with_root(
            "/api/v1/endpoint"
        )
        assert result == "/api/v1/endpoint"


def test_build_full_path_with_root_custom():
    """
    Test _build_full_path_with_root with custom root path
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path"
    ) as mock_get_root:
        # Test with custom root path /proxy
        mock_get_root.return_value = "/proxy"

        result = InitPassThroughEndpointHelpers._build_full_path_with_root(
            "/api/v1/endpoint"
        )
        assert result == "/proxy/api/v1/endpoint"


def test_build_full_path_with_root_nested():
    """
    Test _build_full_path_with_root with nested root path
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path"
    ) as mock_get_root:
        # Test with nested root path /api/v2
        mock_get_root.return_value = "/api/v2"

        result = InitPassThroughEndpointHelpers._build_full_path_with_root("/endpoint")
        assert result == "/api/v2/endpoint"


def test_is_registered_pass_through_route_with_custom_root():
    """
    Test is_registered_pass_through_route correctly handles server root path

    When server has a custom root path like /proxy, the registered path
    should be constructed by prepending the root to match incoming routes.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        _registered_pass_through_routes,
    )

    # Clear the registry + indexes first (lazy rebuild needs empty indexes
    # when tests seed only the flat dict without calling _index_*).
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()

    # Register a pass-through route with endpoint format: {endpoint_id}:exact:{path}
    endpoint_id = "test-endpoint-123"
    path = "/api/endpoint"
    route_key = f"{endpoint_id}:exact:{path}"
    _registered_pass_through_routes[route_key] = {
        "target": "http://example.com",
        "headers": {},
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path"
    ) as mock_get_root:
        # Test with custom root path /proxy
        mock_get_root.return_value = "/proxy"

        # Should match when request route includes the root path
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/proxy/api/endpoint"
            )
            is True
        )

        # Should not match when request route doesn't include root path
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/api/endpoint"
            )
            is False
        )

        # Test with default root path
        mock_get_root.return_value = "/"

        # Should match with default root
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/api/endpoint"
            )
            is True
        )

        # Should not match with root prepended when root is /
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/proxy/api/endpoint"
            )
            is False
        )

    # Clean up
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


def test_get_registered_pass_through_route_with_custom_root():
    """
    Test get_registered_pass_through_route correctly handles server root path

    When server has a custom root path, the method should return the correct
    endpoint configuration by matching the full path including the root.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        _registered_pass_through_routes,
    )

    # Clear the registry + indexes first (lazy rebuild needs empty indexes
    # when tests seed only the flat dict without calling _index_*).
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()

    # Register a pass-through route
    endpoint_id = "test-endpoint-456"
    path = "/chat/completions"
    target_config = {
        "target": "http://api.example.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer token123"},
        "forward_headers": True,
    }
    route_key = f"{endpoint_id}:exact:{path}"
    _registered_pass_through_routes[route_key] = target_config

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path"
    ) as mock_get_root:
        # Test with custom root path /litellm
        mock_get_root.return_value = "/litellm"

        # Should return config when request route includes root path
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/litellm/chat/completions"
        )
        assert result is not None
        assert result["target"] == "http://api.example.com/v1/chat/completions"
        assert result["headers"]["Authorization"] == "Bearer token123"

        # Should return None when route doesn't match
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/chat/completions"
        )
        assert result is None

        # Test with default root path
        mock_get_root.return_value = "/"

        # Should return config with default root
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/chat/completions"
        )
        assert result is not None
        assert result["target"] == "http://api.example.com/v1/chat/completions"

    # Clean up
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


def test_mapped_pass_through_routes_with_server_root_path():
    """
    Mapped passthrough routes (vertex_ai, bedrock, etc) should match
    even when SERVER_ROOT_PATH is set and the incoming route is prefixed.

    Regression test for https://github.com/BerriAI/litellm/issues/22272
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    with patch("litellm.proxy.utils.get_server_root_path") as mock_get_root:
        mock_get_root.return_value = "/litellm"

        # prefixed route should match mapped routes like /vertex_ai
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/litellm/vertex_ai/v1/projects/foo"
            )
            is True
        )
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/litellm/bedrock/model/invoke"
            )
            is True
        )

        # bare route without prefix should not match when root is set
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/vertex_ai/v1/projects/foo"
            )
            is False
        )


@pytest.mark.asyncio
async def test_multipart_passthrough_preserves_boundary():
    """
    Test that multipart/form-data requests through passthrough preserve the boundary
    and can be correctly parsed by the upstream server.

    Regression test for multipart boundary stripping issue.
    """
    from io import BytesIO

    # Mock the httpx request to verify files are passed correctly
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({"content-type": "application/json"})
    mock_response.aread = AsyncMock(
        return_value=b'{"filename": "test.txt", "size": 17}'
    )
    mock_response.text = '{"filename": "test.txt", "size": 17}'

    async def mock_httpx_request(method, url, **kwargs):
        # Verify that files parameter is passed (not json)
        assert "files" in kwargs, "Files should be passed for multipart requests"
        assert "file" in kwargs["files"], "File field should be in files dict"

        # Verify content-type is NOT in headers (httpx will set it with correct boundary)
        headers = kwargs.get("headers", {})
        assert (
            "content-type" not in headers
        ), "content-type should be removed for multipart"

        filename, content, content_type = kwargs["files"]["file"]
        assert filename == "test.txt"
        assert content == b"test file content"
        assert content_type == "text/plain"

        return mock_response

    async_client = MagicMock()
    async_client.request = AsyncMock(side_effect=mock_httpx_request)

    # Create mock request
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = Headers({"content-type": "multipart/form-data; boundary=test123"})

    # Mock form data
    file_content = b"test file content"
    file = BytesIO(file_content)
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = {"file": upload_file}
    request.form = AsyncMock(return_value=form_data)

    # Test the multipart handler directly
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com/upload"),
        headers={},
        requested_query_params=None,
    )

    # Verify the response
    assert response.status_code == 200
    async_client.request.assert_called_once()


class _FakeStreamingResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.headers = {}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_chunk_processor_records_segmented_streaming_metrics():
    response = _FakeStreamingResponse([b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"])
    logging_obj = MagicMock()
    success_handler_kwargs = {
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }

    chunks = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "gpt-5.4"},
        litellm_logging_obj=logging_obj,
        endpoint_type=EndpointType.OPENAI,
        start_time=datetime.now() - timedelta(milliseconds=5),
        passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
        url_route="https://chatgpt.com/backend-api/codex/responses",
        custom_llm_provider="openai",
        success_handler_kwargs=success_handler_kwargs,
        upstream_wait_started_at=datetime.now() - timedelta(milliseconds=4),
        upstream_wait_completed_at=datetime.now() - timedelta(milliseconds=3),
        local_prepare_ms=2.5,
    ):
        chunks.append(chunk)

    await asyncio.sleep(0)

    assert chunks == [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]
    metadata = success_handler_kwargs["litellm_params"]["metadata"]
    assert metadata["aawm_stream_chunk_count"] == 2
    assert metadata["aawm_stream_total_bytes"] > 0
    assert metadata["aawm_time_to_first_token_ms"] >= 0
    assert metadata["aawm_upstream_first_chunk_ms"] >= 0
    assert metadata["aawm_first_emitted_chunk_ms"] >= 0
    assert metadata["aawm_upstream_stream_complete_ms"] >= 0
    assert success_handler_kwargs["completion_start_time"] is not None
    logging_obj._update_completion_start_time.assert_called_once()


@pytest.mark.asyncio
async def test_route_streaming_logging_records_finalize_metrics():
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_stream_emit_gap_ms": 3.0,
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks",
        return_value={"result": {"response": "ok"}, "kwargs": {}},
    ):
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            response=httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://chatgpt.com/backend-api/codex/responses",
                ),
            ),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4"},
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now() - timedelta(milliseconds=10),
            raw_bytes=[b"data: {}\n\n"],
            end_time=datetime.now(),
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
            local_prepare_ms=2.5,
        )

    call_kwargs = logging_obj.async_success_handler.await_args.kwargs
    metadata = call_kwargs["litellm_params"]["metadata"]
    assert metadata["aawm_local_stream_finalize_ms"] >= 0
    assert metadata["aawm_total_proxy_overhead_ms"] >= 5.5
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "proxy.post_response_finalize" in span_names


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "antigravity"])
@pytest.mark.asyncio
async def test_route_streaming_logging_skips_code_assist_control_plane(
    custom_llm_provider,
):
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    success_handler = PassThroughEndpointLogging()

    await PassThroughStreamingHandler._route_streaming_logging_to_handler(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=success_handler,
        response=httpx.Response(
            200,
            request=httpx.Request(
                "POST",
                "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
            ),
        ),
        url_route="https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
        request_body={"request": {"session_id": "gemini-session-123"}},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=datetime.now() - timedelta(milliseconds=10),
        raw_bytes=[b'{"response":{"sessionId":"code-assist-session-123"}}'],
        end_time=datetime.now(),
        custom_llm_provider=custom_llm_provider,
        success_handler_kwargs={},
    )

    logging_obj.async_success_handler.assert_not_awaited()


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "antigravity"])
@pytest.mark.asyncio
async def test_route_streaming_logging_captures_code_assist_quota_observation(
    custom_llm_provider,
):
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler = PassThroughEndpointLogging()
    success_handler_kwargs = {
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }
    quota_payload = {
        "remainingRequests": 1499,
        "totalRequests": 1500,
        "buckets": [
            {
                "quotaPeriod": "DAILY",
                "remainingRequests": 1499,
                "totalRequests": 1500,
                "resetTime": "2026-06-04T00:00:00Z",
            }
        ],
    }

    await PassThroughStreamingHandler._route_streaming_logging_to_handler(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=success_handler,
        response=httpx.Response(
            200,
            request=httpx.Request(
                "POST",
                "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
            ),
        ),
        url_route="https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
        request_body={"request": {"session_id": "gemini-session-123"}},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=datetime.now() - timedelta(milliseconds=10),
        raw_bytes=[f"data: {json.dumps(quota_payload)}\n\n".encode("utf-8")],
        end_time=datetime.now(),
        custom_llm_provider=custom_llm_provider,
        success_handler_kwargs=success_handler_kwargs,
    )

    logging_obj.async_success_handler.assert_awaited_once()
    call_kwargs = logging_obj.async_success_handler.await_args.kwargs
    metadata = call_kwargs["litellm_params"]["metadata"]
    assert metadata["aawm_rate_limit_observation_only"] is True
    expected_source = (
        "antigravity_retrieve_user_quota"
        if custom_llm_provider == "antigravity"
        else "google_retrieve_user_quota"
    )
    assert metadata["google_retrieve_user_quota"]["source"] == expected_source
    assert metadata["google_retrieve_user_quota"]["remainingRequests"] == 1499
    assert (
        metadata["google_retrieve_user_quota"]["buckets"]["items"][0]["resetTime"]
        == "2026-06-04T00:00:00Z"
    )


@pytest.mark.asyncio
async def test_openai_responses_passthrough_sanitizes_and_restores_function_names():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _build_anthropic_responses_adapter_request_body,
    )

    original_name = "mcp__server__" + ("long_function_name_" * 8)
    translated_body = _build_anthropic_responses_adapter_request_body(
        {
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_history",
                            "name": original_name,
                            "input": {"value": 1},
                        }
                    ],
                }
            ],
            "max_tokens": 32,
            "tools": [
                {
                    "name": original_name,
                    "description": "Use the long-named tool.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "tool", "name": original_name},
        },
        adapter_model="gpt-5.5",
        use_chatgpt_codex_defaults=True,
    )
    translated_body["stream"] = False
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = MagicMock()
    mock_request.url.path = "/anthropic/v1/messages"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    captured_body = {}

    async def fake_non_stream_handler(**kwargs):
        provider_body = kwargs["_parsed_body"]
        captured_body.update(provider_body)
        upstream_name = provider_body["tools"][0]["name"]
        return httpx.Response(
            status_code=200,
            content=json.dumps(
                {
                    "id": "resp_d1_500",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_d1_500",
                            "call_id": "call_d1_500",
                            "name": upstream_name,
                            "arguments": "{}",
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "content-length": "1",
                "x-request-id": "request-d1-500",
            },
            request=httpx.Request(
                "POST",
                "https://chatgpt.com/backend-api/codex/responses",
            ),
        )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
        "get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
        "HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=fake_non_stream_handler,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
        "capture_passthrough_shape"
    ):
        mock_client_obj = MagicMock()
        mock_client_obj.client = MagicMock()
        mock_get_client.return_value = mock_client_obj
        mock_logging_obj.pre_call_hook = AsyncMock(return_value=translated_body)

        response = await pass_through_request(
            request=mock_request,
            target="https://chatgpt.com/backend-api/codex/responses",
            custom_headers={"authorization": "Bearer test"},
            user_api_key_dict=MagicMock(),
            custom_body=translated_body,
            custom_llm_provider="openai",
            stream=False,
        )

    historical_name = next(
        item["name"]
        for item in captured_body["input"]
        if item.get("type") == "function_call"
    )
    assert len(historical_name) <= 64
    assert historical_name == captured_body["tools"][0]["name"]
    assert historical_name == captured_body["tool_choice"]["name"]
    assert historical_name != original_name
    assert (
        next(
            item["call_id"]
            for item in captured_body["input"]
            if item.get("type") == "function_call"
        )
        == "toolu_history"
    )

    response_body = json.loads(response.body)
    assert response_body["output"][0]["name"] == original_name
    assert response_body["output"][0]["call_id"] == "call_d1_500"
    assert response.headers["x-request-id"] == "request-d1-500"
    assert int(response.headers["content-length"]) == len(response.body)


@pytest.mark.asyncio
async def test_responses_sse_restoration_is_frame_preserving_across_split_chunks():
    original_name = "stream_tool_" + ("x" * 80)
    rewrite = sanitize_responses_function_names(
        {
            "tools": [{"type": "function", "name": original_name}],
        }
    )
    upstream_name = rewrite.original_to_upstream[original_name]
    raw_stream = (
        "event: response.output_item.added\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_stream",
                    "call_id": "call_stream",
                    "name": upstream_name,
                    "arguments": "",
                },
            }
        )
        + "\n\n"
        + "event: response.function_call_arguments.done\r\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_stream",
                "name": upstream_name,
                "arguments": "{}",
            }
        )
        + "\r\n\r\n"
        + "event: response.completed\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "name": upstream_name,
                            "call_id": "call_stream",
                        }
                    ],
                },
            }
        )
        + "\n\n"
        + "data: not-json\n\n"
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    async def split_chunks():
        for boundary in (7, 53, 127, 211, len(raw_stream)):
            start = getattr(split_chunks, "start", 0)
            if start >= len(raw_stream):
                break
            yield raw_stream[start:boundary]
            split_chunks.start = boundary

    split_chunks.start = 0
    restored = b"".join(
        [
            chunk
            async for chunk in _restore_responses_function_names_in_sse_chunks(
                split_chunks(),
                rewrite,
            )
        ]
    )

    assert original_name.encode("utf-8") in restored
    assert upstream_name.encode("utf-8") not in restored
    assert b"event: response.output_item.added\n" in restored
    assert b"event: response.function_call_arguments.done\r\n" in restored
    assert b"data: not-json\n\n" in restored
    assert restored.endswith(b"data: [DONE]\n\n")


def _assert_request_shape_422_failure_hook_is_sanitized(
    *,
    post_failure_mock: AsyncMock,
) -> None:
    post_failure_mock.assert_awaited_once()
    hook_kwargs = post_failure_mock.await_args.kwargs
    assert hook_kwargs["traceback_str"] is None
    request_data = hook_kwargs["request_data"]
    assert request_data.get("failure_kind") == "request_shape_deserialization_failed"


def _assert_request_shape_422_terminal_call_has_agent_attribution(
    *,
    terminal_kwargs: dict,
    secret_prompt: str,
    secret_token: str,
) -> None:
    error_context = terminal_kwargs["error_context"]
    assert terminal_kwargs["terminal_outcome"] == "request_rejected"
    assert terminal_kwargs["fallback_result"] == "none"
    assert terminal_kwargs["redispatch_required"] is False
    assert terminal_kwargs["agent_session_killed"] is True
    assert error_context["failure_kind"] == "request_shape_deserialization_failed"
    assert error_context["status_code"] == 422
    assert error_context["model_alias"] == "aawm-code"
    assert error_context["route_family"] == "codex_responses"
    assert error_context["endpoint"] == "/openai_passthrough/responses"
    assert error_context["agent_id"] == "agent-route-modelinput"
    assert error_context["agent_name"] == "worker"
    assert error_context["agent_role"] == "worker"
    assert error_context["thread_source"] == "subagent"
    assert error_context["session_id"] == "session-route-modelinput"
    assert error_context["dispatch_id"] == "dispatch-route-modelinput"
    assert error_context["trace_id"] == "trace-route-modelinput"
    assert error_context["aawm_passthrough_request_shape_error_message_class"] == "model_input_deserialization_failed"
    preview = error_context.get("aawm_passthrough_request_shape_error_body_preview") or ""
    assert "ModelInput" in preview
    assert secret_prompt not in preview
    assert secret_token not in preview


def _assert_request_shape_422_terminal_jsonl_record(
    *,
    tmp_path,
    secret_prompt: str,
    secret_token: str,
) -> None:
    log_path = tmp_path / "test-error.jsonl"
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["failure_kind"] == "request_shape_deserialization_failed"
    assert record["status_code"] == 422
    assert record["agent_id"] == "agent-route-modelinput"
    assert record["session_id"] == "session-route-modelinput"
    assert record["terminal_outcome"] == "request_rejected"
    assert record["agent_session_killed"] is True
    assert record["context"]["model_alias"] == "aawm-code"
    assert record["context"]["route_family"] == "codex_responses"
    assert record["context"]["agent_role"] == "worker"
    assert record["context"]["thread_source"] == "subagent"
    assert record["context"]["dispatch_id"] == "dispatch-route-modelinput"
    serialized = json.dumps(record)
    assert secret_prompt not in serialized
    assert secret_token not in serialized
    assert "sk-route-level" not in serialized


def _build_request_shape_422_route_level_fixture(
    *,
    secret_prompt: str,
    secret_token: str,
) -> dict:
    """Build request/upstream fixtures for Responses ModelInput 422 route tests."""
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4001/openai_passthrough/responses"
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams({})

    target_url = "https://chatgpt.com/backend-api/codex/responses"
    upstream_detail = (
        '{"error":"Failed to deserialize the JSON body into the target type: '
        f"data did not match any variant of untagged enum ModelInput; prompt={secret_prompt}; "
        f'token={secret_token}"}}'
    )
    upstream_response = httpx.Response(
        status_code=422,
        content=upstream_detail.encode("utf-8"),
        request=httpx.Request("POST", target_url),
    )
    request_body = {
        "model": "gpt-5.5",
        "input": [{"type": "message", "role": "user", "content": secret_prompt}],
        "tools": [{"type": "function", "name": "exec_command"}],
        "litellm_metadata": {
            "route_family": "codex_responses",
            "inbound_model_alias": "aawm-code",
            "provider": "openai",
            "repository": "litellm",
            "agent_name": "worker",
            "agent_id": "agent-route-modelinput",
            "agent_role": "worker",
            "agent_profile": "worker",
            "thread_source": "subagent",
            "session_id": "session-route-modelinput",
            "trace_id": "trace-route-modelinput",
            "dispatch_id": "dispatch-route-modelinput",
            "aawm_passthrough_body_container_type": "object",
            "aawm_passthrough_body_top_level_keys": ["input", "model", "tools"],
            "aawm_passthrough_input_container_type": "array",
            "aawm_passthrough_input_item_count": 1,
            "aawm_passthrough_input_item_type_counts": {"message": 1},
            "aawm_passthrough_tool_count": 1,
            "aawm_passthrough_tool_type_counts": {"function": 1},
        },
    }
    return {
        "mock_request": mock_request,
        "target_url": target_url,
        "upstream_response": upstream_response,
        "request_body": request_body,
    }


async def _invoke_pass_through_request_shape_422_with_terminal_capture(
    *,
    fixture: dict,
    secret_token: str,
    post_failure_mock: AsyncMock,
    terminal_calls: list,
):
    from litellm.proxy.aawm_runtime_error_logging import (
        persist_agent_terminal_error as real_persist_agent_terminal_error,
    )

    def _capture_terminal(**kwargs):
        terminal_calls.append(kwargs)
        return real_persist_agent_terminal_error(**kwargs)

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=MagicMock(client=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=fixture["upstream_response"]),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kw: kw["data"]),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.emit_aawm_route_access_log",
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
        new=AsyncMock(),
    ), patch(
        # RR-056 #8 hoisted the import into pass_through_endpoints; patch the
        # bound module symbol rather than the source module only.
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.persist_agent_terminal_error",
        side_effect=_capture_terminal,
    ) as mock_terminal, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ):
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=fixture["mock_request"],
                target=fixture["target_url"],
                custom_headers={"authorization": f"Bearer {secret_token}"},
                user_api_key_dict=MagicMock(),
                custom_body=fixture["request_body"],
                custom_llm_provider="openai",
                stream=False,
            )
    return exc_info, mock_terminal


def test_agent_terminal_422_intake_is_redacted_and_agent_correlated(
    tmp_path,
    monkeypatch,
):
    from litellm.proxy.aawm_runtime_error_logging import (
        persist_agent_terminal_error,
    )

    monkeypatch.setenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    secret_prompt = "SUPER_SECRET_TERMINAL_PROMPT"
    secret_token = "sk-terminal-secret-token"
    base_context = {
        "source": "pass_through_endpoint",
        "endpoint": "/openai_passthrough/responses",
        "upstream_url": "https://chatgpt.com/backend-api/codex/responses",
        "provider": "openai",
        "model": "gpt-5.5",
        "model_alias": "aawm-code",
        "route_family": "codex_responses",
        "status_code": 422,
        "repository": "litellm",
        "agent_name": "worker",
        "agent_id": "agent-model-input",
        "agent_role": "worker",
        "agent_profile": "worker",
        "thread_source": "subagent",
        "session_id": "session-model-input",
        "trace_id": "trace-model-input",
        "litellm_call_id": "call-model-input",
        "dispatch_id": "dispatch-model-input",
        "aawm_passthrough_body_container_type": "object",
        "aawm_passthrough_body_top_level_keys": ["input", "model", "tools"],
        "aawm_passthrough_input_container_type": "array",
        "aawm_passthrough_input_item_count": 1,
        "aawm_passthrough_input_item_type_counts": {"message": 1},
        "aawm_passthrough_input_item_shape_samples": [
            {
                "index": 0,
                "container_type": "object",
                "type": "message",
                "keys": ["content", "role", "type"],
            }
        ],
        "aawm_passthrough_tool_count": 1,
        "aawm_passthrough_tool_type_counts": {"function": 1},
        "attempts": [
            {
                "provider": "xai",
                "model": "xai/grok-4.5",
                "route_family": "codex_grok_native_responses_adapter",
                "status": "cooldown_set",
                "error_class": "safety_policy_denied",
                "error_status_code": 403,
                "error_code": "api_key=terminal-secret-key",
                "error_message": secret_prompt,
                "headers": {"authorization": f"Bearer {secret_token}"},
            }
        ],
        "candidates": [
            {
                "provider": "xai",
                "model": "xai/grok-4.5",
                "route_family": "codex_grok_native_responses_adapter",
                "last_resort": False,
                "request_body": {"input": secret_prompt},
            }
        ],
        "actual_prior_tool_activity_summary": {
            "has_actual_prior_tool_activity": True,
            "prior_tool_call_count": 1,
            "prior_tool_names": ["exec_command"],
            "tool_arguments": {"cmd": secret_prompt},
        },
    }
    exc = HTTPException(
        status_code=422,
        detail=(
            '{"error":"Failed to deserialize the JSON body into the target type: '
            f"data did not match any variant of untagged enum ModelInput; prompt={secret_prompt}; "
            f'token={secret_token}"}}'
        ),
    )
    enriched = _enrich_passthrough_error_log_context_for_request_shape_422(
        error_log_context=base_context,
        exc=exc,
    )

    assert persist_agent_terminal_error(
        error_context=enriched,
        terminal_outcome="request_rejected",
        fallback_result="none",
        redispatch_required=False,
        agent_session_killed=True,
    )

    log_path = tmp_path / "test-error.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["failure_kind"] == "request_shape_deserialization_failed"
    assert record["status_code"] == 422
    assert record["agent_id"] == "agent-model-input"
    assert record["session_id"] == "session-model-input"
    assert record["litellm_call_id"] == "call-model-input"
    assert record["terminal_outcome"] == "request_rejected"
    assert record["agent_session_killed"] is True
    assert record["context"]["agent_role"] == "worker"
    assert record["context"]["thread_source"] == "subagent"
    assert record["context"]["dispatch_id"] == "dispatch-model-input"
    assert record["context"]["aawm_passthrough_request_shape_summary"]["input_item_count"] == 1
    assert record["context"]["attempts"] == [
        {
            "provider": "xai",
            "model": "xai/grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
            "status": "cooldown_set",
            "error_class": "safety_policy_denied",
            "error_status_code": 403,
            "error_code": "REDACTED",
        }
    ]
    assert record["context"]["candidates"] == [
        {
            "provider": "xai",
            "model": "xai/grok-4.5",
            "route_family": "codex_grok_native_responses_adapter",
            "last_resort": False,
        }
    ]
    assert record["context"]["actual_prior_tool_activity_summary"] == {
        "has_actual_prior_tool_activity": True,
        "prior_tool_call_count": 1,
        "prior_tool_names": ["exec_command"],
    }
    serialized = json.dumps(record)
    assert secret_prompt not in serialized
    assert secret_token not in serialized
    assert "sk-terminal-secret" not in serialized


@pytest.mark.asyncio
async def test_pass_through_alias_managed_xai_safety_403_skips_generic_failure_hooks():
    """Alias-managed intermediate SAFETY_CHECK_TYPE_CYBER 403 must not fire generic failure paths.

    Alias fallback still depends on the exception being re-raised to the caller.
    """
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4001/openai_passthrough/responses"
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams({})

    target_url = "https://cli-chat-proxy.grok.com/v1/responses"
    upstream_detail = (
        '{"code":"permission-denied","error":"Content violates usage guidelines. '
        'Failed check: SAFETY_CHECK_TYPE_CYBER"}'
    )
    upstream_response = httpx.Response(
        status_code=403,
        content=upstream_detail.encode("utf-8"),
        request=httpx.Request("POST", target_url),
    )
    request_body = {
        "model": "xai/grok-4.5",
        "input": [{"type": "message", "content": "hello"}],
        "litellm_metadata": {
            "inbound_model_alias": "aawm-code",
            "route_family": "codex_grok_native_responses_adapter",
            "provider": "xai",
        },
    }

    post_failure_mock = AsyncMock()
    direct_capture_mock = AsyncMock()
    terminal_intake_mock = MagicMock(return_value=True)

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=MagicMock(client=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kw: kw["data"]),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.emit_aawm_route_access_log",
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
        new=direct_capture_mock,
    ), patch(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        new=terminal_intake_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ), patch.object(
        verbose_proxy_logger,
        "exception",
    ) as mock_exception, patch.object(
        verbose_proxy_logger,
        "error",
    ) as mock_error, patch.object(
        verbose_proxy_logger,
        "warning",
    ) as mock_warning, patch.object(
        verbose_proxy_logger,
        "debug",
    ) as mock_debug:
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer xai-token"},
                user_api_key_dict=MagicMock(),
                custom_body=request_body,
                custom_llm_provider="xai",
                stream=False,
            )

    # Exception is re-raised so alias-level request-local fallback remains possible.
    assert exc_info.value.code == "403"
    assert "SAFETY_CHECK_TYPE_CYBER" in str(exc_info.value.message)
    post_failure_mock.assert_not_awaited()
    direct_capture_mock.assert_not_awaited()
    terminal_intake_mock.assert_not_called()
    mock_exception.assert_not_called()
    mock_error.assert_not_called()
    mock_warning.assert_not_called()
    deferred_debug = [
        call
        for call in mock_debug.call_args_list
        if call.args and "deferred alias-managed safety denial" in str(call.args[0])
    ]
    assert deferred_debug
    assert deferred_debug[0].kwargs["extra"]["failure_kind"] == "safety_policy_denied"
    assert deferred_debug[0].kwargs["extra"]["model_alias"] == "aawm-code"


@pytest.mark.asyncio
async def test_pass_through_direct_xai_safety_403_is_handled_without_traceback():
    """Direct/non-alias SAFETY_CHECK_TYPE_CYBER 403 remains bounded and client-visible."""
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4001/grok/v1/responses"
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams({})

    target_url = "https://cli-chat-proxy.grok.com/v1/responses"
    upstream_detail = (
        '{"code":"permission-denied","error":"Content violates usage guidelines. '
        'Failed check: SAFETY_CHECK_TYPE_CYBER"}'
    )
    upstream_response = httpx.Response(
        status_code=403,
        content=upstream_detail.encode("utf-8"),
        request=httpx.Request("POST", target_url),
    )
    request_body = {
        "model": "grok-4.5",
        "input": [{"type": "message", "content": "hello"}],
        "litellm_metadata": {
            "inbound_model_alias": "direct-grok",
            "route_family": "grok_cli_chat_proxy",
            "provider": "xai",
        },
    }

    post_failure_mock = AsyncMock()
    direct_capture_mock = AsyncMock()

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=MagicMock(client=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(side_effect=lambda **kw: kw["data"]),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.emit_aawm_route_access_log",
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=post_failure_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._direct_capture_xai_passthrough_failure",
        new=direct_capture_mock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ), patch.object(
        verbose_proxy_logger,
        "exception",
    ) as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers={"authorization": "Bearer xai-token"},
                user_api_key_dict=MagicMock(),
                custom_body=request_body,
                custom_llm_provider="xai",
                stream=False,
            )

    assert exc_info.value.code == "403"
    assert "SAFETY_CHECK_TYPE_CYBER" in str(exc_info.value.message)
    post_failure_mock.assert_awaited_once()
    direct_capture_mock.assert_awaited_once()
    mock_exception.assert_not_called()
    hook_kwargs = post_failure_mock.await_args.kwargs
    assert hook_kwargs["traceback_str"] is None


@pytest.mark.asyncio
async def test_pass_through_request_shape_422_calls_terminal_jsonl_writer_with_agent_attribution(
    monkeypatch,
    tmp_path,
):
    """Drive the real Responses ModelInput 422 branch and assert terminal JSONL intake."""
    secret_prompt = "ROUTE_LEVEL_MODELINPUT_SECRET_PROMPT"
    secret_token = "sk-route-level-modelinput-token"
    monkeypatch.setenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    fixture = _build_request_shape_422_route_level_fixture(
        secret_prompt=secret_prompt,
        secret_token=secret_token,
    )
    post_failure_mock = AsyncMock()
    terminal_calls: list = []

    exc_info, mock_terminal = await _invoke_pass_through_request_shape_422_with_terminal_capture(
        fixture=fixture,
        secret_token=secret_token,
        post_failure_mock=post_failure_mock,
        terminal_calls=terminal_calls,
    )

    assert exc_info.value.code == "422"
    mock_terminal.assert_called_once()
    assert len(terminal_calls) == 1
    _assert_request_shape_422_terminal_call_has_agent_attribution(
        terminal_kwargs=terminal_calls[0],
        secret_prompt=secret_prompt,
        secret_token=secret_token,
    )
    _assert_request_shape_422_failure_hook_is_sanitized(
        post_failure_mock=post_failure_mock,
    )
    _assert_request_shape_422_terminal_jsonl_record(
        tmp_path=tmp_path,
        secret_prompt=secret_prompt,
        secret_token=secret_token,
    )


def test_passthrough_agent_terminal_error_context_requires_alias_or_agent_identity():
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _is_passthrough_agent_terminal_error_context,
    )

    assert _is_passthrough_agent_terminal_error_context({"model_alias": "aawm-code"})
    assert _is_passthrough_agent_terminal_error_context({"model_alias": "direct-openai", "agent_id": "agent-123"})
    assert not _is_passthrough_agent_terminal_error_context({"model_alias": "direct-openai"})


def test_passthrough_alias_managed_safety_denial_is_deferred_to_alias_terminal():
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _is_passthrough_alias_managed_safety_policy_denial,
    )

    exc = HTTPException(
        status_code=403,
        detail=(
            '{"code":"permission-denied","error":"Content violates usage guidelines. '
            'Failed check: SAFETY_CHECK_TYPE_CYBER"}'
        ),
    )
    assert _is_passthrough_alias_managed_safety_policy_denial(
        exc=exc,
        status_code=403,
        error_log_context={"model_alias": "aawm-code"},
    )
    assert not _is_passthrough_alias_managed_safety_policy_denial(
        exc=exc,
        status_code=403,
        error_log_context={"model_alias": "direct-grok"},
    )
    assert not _is_passthrough_alias_managed_safety_policy_denial(
        exc=HTTPException(status_code=403, detail="Forbidden"),
        status_code=403,
        error_log_context={"model_alias": "aawm-code"},
    )
