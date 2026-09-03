from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Union, get_args
from urllib.parse import urlparse

import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.requests import Request

from litellm.llms.cursor_agent.common_utils import run_url
from litellm.llms.cursor_agent import connect as cursor_connect
from litellm.llms.cursor_agent.connect import (
    CursorAgentRunResult,
    CursorConnectError,
    CursorConnectProtocolError,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints
from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    flush_aawm_route_rollups,
    record_aawm_route_rollup_failure,
)


def _request(*, selected_account: dict[str, Any] | None = None) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"x-request-id", b"req-1")],
            "client": ("test", 1234),
            "server": ("test", 80),
            "scheme": "http",
        }
    )
    if selected_account is not None:
        request.state.aawm_codex_oauth_selected_account = selected_account
    return request


def _candidate(**overrides: str) -> dict[str, str]:
    candidate = {
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    candidate.update(overrides)
    return candidate


def _call(
    body: dict[str, Any],
    *,
    candidate: dict[str, Any] | None = None,
    target_url: str = "",
    api_key: str | None = "access-token",
    selected_account: dict[str, Any] | None = None,
) -> Response:
    return asyncio.run(
        codex_candidate_calls._perform_codex_auto_agent_cursor_agent_request(
            endpoint="/v1/responses",
            request=_request(selected_account=selected_account),
            fastapi_response=Response(),
            user_api_key_dict=None,
            candidate=candidate or _candidate(),
            candidate_body=body,
            target_url=target_url,
            api_key=api_key,
            forward_headers=False,
        )
    )


@pytest.fixture(autouse=True)
def _clear_replay_registry() -> None:
    codex_candidate_calls._clear_cursor_replay_registry()
    yield
    codex_candidate_calls._clear_cursor_replay_registry()


@pytest.fixture
def _route_rollup_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    clear_aawm_route_rollups()
    yield
    clear_aawm_route_rollups()


def _capture_real_route_log(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        anthropic_adapter_calls,
    )

    captured: dict[str, Any] = {}
    real_emit = anthropic_adapter_calls._emit_adapted_route_access_log

    def _capture(**kwargs: Any) -> None:
        real_emit(**kwargs)
        captured["rollup_kwargs"] = kwargs["rollup_kwargs"]

    monkeypatch.setattr(
        anthropic_adapter_calls,
        "_emit_adapted_route_access_log",
        _capture,
    )
    return captured


def _assert_cursor_route_context(
    captured: dict[str, Any],
) -> None:
    context = captured["rollup_kwargs"]["litellm_params"]["metadata"][
        "aawm_route_rollup_context"
    ]
    target = urlparse(run_url(None))
    assert context["outgoing_target"] == f"{target.netloc}{target.path}"
    assert context["model_label"] == "work(cursor-test)"
    assert context["codex_auto_agent_selected_provider"] == "openai"
    assert context["codex_oauth_account_hash"] == "cursor-account"


def _store_replay_state(response_id: str = "resp-replay") -> None:
    codex_candidate_calls._store_cursor_replay_state(
        response_id,
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    }
                ],
            },
        ],
        tools=[],
    )


def _replay_body(response_id: str = "resp-replay") -> dict[str, Any]:
    return {
        "model": "work",
        "previous_response_id": response_id,
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "pwd output",
            }
        ],
    }


def _cursor_continuation_failure() -> CursorConnectError:
    continuation_exc = CursorConnectError(
        "missing retained session",
        status_code=409,
    )
    setattr(
        continuation_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )
    return continuation_exc


def _stock_tool_search_descriptor() -> dict[str, Any]:
    return {
        "description": (
            "# Tool discovery\n\n"
            "Searches over deferred tool metadata with BM25 and exposes "
            "matching tools for the next model call.\n\n"
            "You have access to tools from the following sources:\n"
            "- Codex: Built-in tools.\n"
            "Some of the tools may not have been provided to you upfront, "
            "and you should use this tool (`tool_search`) to search for "
            "the required tools."
        ),
        "execution": "client",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for deferred tools.",
                },
                "limit": {
                    "type": "number",
                    "description": (
                        "Maximum number of tools to return. Defaults to 8."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "type": "tool_search",
    }


def _stock_codex_tools() -> list[dict[str, Any]]:
    function_names = (
        "exec_command",
        "write_stdin",
        "view_image",
        "get_goal",
        "create_goal",
        "update_goal",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "request_user_input",
        "request_plugin_install",
    )
    function_tools = [
        {
            "type": "function",
            "name": name,
            "description": f"Run the {name} tool.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": False,
        }
        for name in function_names
    ]
    return [
        *function_tools,
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch.",
        },
        _stock_tool_search_descriptor(),
        {
            "type": "web_search",
            "filters": {"allowed_domains": ["example.com"]},
        },
    ]


def _stock_codex_full_history_body(
    *,
    include_metadata: bool = False,
    function_call_item_id: str = "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_0",
    function_call_call_id: str = "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
) -> dict[str, Any]:
    turn_id = "01a06269-1662-7c02-a81a-031c450f8606"
    body = {
        "model": "work",
        "tools": _stock_codex_tools(),
        "input": [
            {
                "type": "message",
                "id": "msg_01a06269-1827-79d2-b3a5-41d50a2fad1a",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "model instructions"},
                    {"type": "input_text", "text": "developer instructions"},
                    {"type": "input_text", "text": "memory instructions"},
                    {"type": "input_text", "text": "skill instructions"},
                    {"type": "input_text", "text": "permission instructions"},
                    {"type": "input_text", "text": "app instructions"},
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.7673376,
                    "content_item_kinds": [
                        "model_switch.instructions",
                        "generic.developer_instructions",
                        "memories.instructions",
                        "host_skills.instructions",
                        "permissions.instructions",
                        "apps.instructions",
                    ],
                },
            },
            {
                "type": "message",
                "id": "msg_01a06269-1827-79d2-b3a5-41ed4566fa70",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "plugin recommendations"},
                    {"type": "input_text", "text": "AGENTS instructions"},
                    {"type": "input_text", "text": "environment context"},
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.7673385,
                    "content_item_kinds": [
                        "plugins.recommendations",
                        "agents_md.instructions",
                        "environments.environment_context",
                    ],
                },
            },
            {
                "type": "message",
                "id": "msg_01a06269-1847-7802-a387-e9c9ef8d2032",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Run pwd, then continue the original assignment.",
                    }
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.79907,
                    "content_item_kinds": ["user.text"],
                },
            },
            {
                "type": "message",
                "id": "msg_resp_277feeae9f69433ab4e4ef2597a25db8",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will run pwd first.",
                    }
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "content_item_kinds": ["unknown"],
                },
            },
            {
                "type": "function_call",
                "id": function_call_item_id,
                "name": "exec_command",
                "arguments": (
                    '{"cmd":"pwd","workdir":"/home/zepfu/projects/litellm"}'
                ),
                "call_id": function_call_call_id,
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
            {
                "type": "function_call_output",
                "id": "fco_01a06269-2d51-7de0-a7e3-bddda588ad80",
                "call_id": function_call_call_id,
                "output": (
                    "Chunk ID: 91fb25\n"
                    "Wall time: 0.0000 seconds\n"
                    "Process exited with code 0\n"
                    "Final output:\n"
                    "/home/zepfu/projects/litellm\n"
                ),
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357455.1854475,
                },
            },
        ],
    }
    if not include_metadata:
        for item in body["input"]:
            item.pop("internal_chat_message_metadata_passthrough", None)
    return body


def _cursor_subagent_args(*, extra: bytes = b"") -> bytes:
    return b"".join(
        (
            cursor_connect._encode_proto_string_field(1, "subagent-call"),
            cursor_connect._encode_proto_string_field(2, "explorer"),
            cursor_connect._encode_proto_string_field(3, "read"),
            cursor_connect._encode_proto_string_field(
                4,
                "Inspect the requested files without changing them.",
            ),
            extra,
        )
    )


def _cursor_subagent_server_message(
    subagent_args: bytes,
    *,
    request_id: int = 314,
    exec_id: str = "exec-014",
    with_metadata: bool = False,
) -> bytes:
    exec_fields = cursor_connect._encode_proto_varint_field(1, request_id)
    if with_metadata:
        exec_fields += cursor_connect._encode_proto_message_field(
            19,
            b"span-context-secret",
        )
    exec_fields += cursor_connect._encode_proto_message_field(28, subagent_args)
    if with_metadata:
        exec_fields += cursor_connect._encode_proto_message_field(
            55,
            b"accept-context-secret",
        )
    exec_fields += cursor_connect._encode_proto_string_field(15, exec_id)
    return cursor_connect._encode_proto_message_field(2, exec_fields)


def _advertised_spawn_agent_definition() -> dict[str, Any]:
    return {
        "name": "spawn_agent",
        "inputSchemaJson": json.dumps(
            {
                "type": "object",
                "properties": {
                    "agent_type": {"type": "string"},
                    "model": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["message"],
            }
        ),
    }


def test_cursor_subagent_operation_selection_ignores_exec_metadata() -> None:
    payload = _cursor_subagent_server_message(
        _cursor_subagent_args(),
        with_metadata=True,
    )

    decoded = cursor_connect.decode_agent_server_message(payload)

    assert decoded == {
        "execServerMessage": {
            "id": 314,
            "execId": "exec-014",
            "messageField": 28,
        }
    }
    serialized = json.dumps(decoded)
    assert "span-context-secret" not in serialized
    assert "accept-context-secret" not in serialized


def test_cursor_subagent_args_map_to_advertised_spawn_agent() -> None:
    external_exec_requests: list[dict[str, Any]] = []

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _cursor_subagent_server_message(_cursor_subagent_args()),
        {},
        spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
        external_exec_requests=external_exec_requests,
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    assert tool_call["callId"] == "subagent-call"
    assert tool_call["toolName"] == "spawn_agent"
    assert json.loads(tool_call["argsJson"]) == {
        "agent_type": "explorer",
        "model": "read",
        "message": "Inspect the requested files without changing them.",
    }
    assert client_messages == []
    assert len(external_exec_requests) == 1
    request = external_exec_requests[0]
    assert request["call_id"] == "subagent-call"
    assert request["message_field"] == 28
    assert request["request_id"] == 314
    assert request["exec_id"] == "exec-014"
    assert request["tool_call_id"] == "subagent-call"
    assert request["exec_fields"] == [
        (1, 0, 314),
        (15, 2, b"exec-014"),
    ]


def test_cursor_subagent_without_outer_identity_preserves_tool_call_id() -> None:
    external_exec_requests: list[dict[str, Any]] = []

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _cursor_subagent_server_message(
            _cursor_subagent_args(),
            request_id=0,
            exec_id="",
        ),
        {},
        spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
        external_exec_requests=external_exec_requests,
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    assert tool_call["callId"] == "subagent-call"
    assert client_messages == []
    assert len(external_exec_requests) == 1
    request = external_exec_requests[0]
    assert request["request_id"] == 0
    assert request["exec_id"] == ""
    assert request["tool_call_id"] == "subagent-call"
    assert request["exec_fields"] == [
        (1, 0, 0),
        (15, 2, b""),
    ]


@pytest.mark.parametrize(
    ("ignored_fields", "secrets"),
    [
        (
            cursor_connect._encode_proto_string_field(
                9,
                "parent-conversation-sensitive-value",
            ),
            ("parent-conversation-sensitive-value",),
        ),
        (
            cursor_connect._encode_proto_string_field(
                16,
                "root-parent-sensitive-value",
            ),
            ("root-parent-sensitive-value",),
        ),
        (
            cursor_connect._encode_proto_string_field(
                9,
                "parent-conversation-sensitive-value",
            )
            + cursor_connect._encode_proto_string_field(
                16,
                "root-parent-sensitive-value",
            ),
            (
                "parent-conversation-sensitive-value",
                "root-parent-sensitive-value",
            ),
        ),
    ],
    ids=["parent-conversation", "root-parent", "both-parent-fields"],
)
def test_cursor_subagent_ignores_parent_state_fields_without_leaking(
    ignored_fields: bytes,
    secrets: tuple[str, ...],
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _cursor_subagent_server_message(
            _cursor_subagent_args(extra=ignored_fields),
        ),
        {},
        spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
        external_exec_requests=external_exec_requests,
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    assert tool_call["callId"] == "subagent-call"
    assert tool_call["toolName"] == "spawn_agent"
    assert json.loads(tool_call["argsJson"]) == {
        "agent_type": "explorer",
        "model": "read",
        "message": "Inspect the requested files without changing them.",
    }
    assert client_messages == []
    assert len(external_exec_requests) == 1
    request = external_exec_requests[0]
    assert request["call_id"] == "subagent-call"
    assert request["message_field"] == 28
    assert request["request_id"] == 314
    assert request["exec_id"] == "exec-014"
    assert request["tool_call_id"] == "subagent-call"
    assert request["exec_fields"] == [
        (1, 0, 314),
        (15, 2, b"exec-014"),
    ]
    serialized = json.dumps(tool_call) + repr(request)
    for secret in secrets:
        assert secret not in serialized


@pytest.mark.parametrize(
    "default_field",
    [
        cursor_connect._encode_proto_varint_field(
            7,
            0,
            include_default=True,
        ),
        cursor_connect._encode_proto_string_field(
            9,
            "",
            include_empty=True,
        ),
        cursor_connect._encode_proto_varint_field(
            19,
            0,
            include_default=True,
        ),
        cursor_connect._encode_proto_varint_field(19, 1),
    ],
    ids=[
        "run-in-background-false",
        "parent-conversation-empty",
        "environment-unspecified",
        "environment-local",
    ],
)
def test_cursor_subagent_accepts_supported_optional_fields_through_spawn_agent(
    default_field: bytes,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _cursor_subagent_server_message(
            _cursor_subagent_args(extra=default_field),
        ),
        {},
        spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
        external_exec_requests=external_exec_requests,
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    assert tool_call["toolName"] == "spawn_agent"
    assert json.loads(tool_call["argsJson"]) == {
        "agent_type": "explorer",
        "model": "read",
        "message": "Inspect the requested files without changing them.",
    }
    assert client_messages == []
    assert external_exec_requests == [
        {
            "call_id": "subagent-call",
            "message_field": 28,
            "exec_fields": [
                (1, 0, 314),
                (15, 2, b"exec-014"),
            ],
            "request_id": 314,
            "exec_id": "exec-014",
            "tool_call_id": "subagent-call",
            "subagent_type": "explorer",
            "model_id": "read",
            "prompt": "Inspect the requested files without changing them.",
            "readonly": False,
        }
    ]


@pytest.mark.parametrize(
    ("duplicate_optional_fields", "not_leaked"),
    [
        (
            cursor_connect._encode_proto_varint_field(
                7,
                0,
                include_default=True,
            )
            + cursor_connect._encode_proto_varint_field(
                7,
                0,
                include_default=True,
            ),
            "Inspect the requested files",
        ),
        (
            cursor_connect._encode_proto_string_field(
                9,
                "",
                include_empty=True,
            )
            + cursor_connect._encode_proto_string_field(
                9,
                "",
                include_empty=True,
            ),
            "Inspect the requested files",
        ),
        (
            cursor_connect._encode_proto_string_field(
                9,
                "parent-conversation-duplicate-secret",
            )
            + cursor_connect._encode_proto_string_field(
                9,
                "parent-conversation-duplicate-secret",
            ),
            "parent-conversation-duplicate-secret",
        ),
        (
            cursor_connect._encode_proto_string_field(
                16,
                "root-parent-duplicate-secret",
            )
            + cursor_connect._encode_proto_string_field(
                16,
                "root-parent-duplicate-secret",
            ),
            "root-parent-duplicate-secret",
        ),
        (
            cursor_connect._encode_proto_varint_field(
                19,
                0,
                include_default=True,
            )
            + cursor_connect._encode_proto_varint_field(19, 1),
            "Inspect the requested files",
        ),
    ],
    ids=[
        "run-in-background-false",
        "parent-conversation-empty",
        "parent-conversation-non-empty",
        "root-parent-non-empty",
        "environment",
    ],
)
def test_cursor_subagent_rejects_repeated_optional_fields_without_leaking(
    duplicate_optional_fields: bytes,
    not_leaked: str,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    with pytest.raises(
        CursorConnectProtocolError,
        match="repeated safe scalar field",
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(
                _cursor_subagent_args(extra=duplicate_optional_fields),
            ),
            {},
            spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert not_leaked not in exc_info.value.message
    assert not_leaked not in json.dumps(exc_info.value.body or {})


def test_cursor_subagent_requires_unambiguous_advertised_spawn_agent() -> None:
    external_exec_requests: list[dict[str, Any]] = []
    definition = _advertised_spawn_agent_definition()
    schema = json.loads(definition["inputSchemaJson"])
    schema["properties"]["subagent_type"] = {"type": "string"}
    definition["inputSchemaJson"] = json.dumps(schema)

    with pytest.raises(
        CursorConnectProtocolError,
        match="ambiguous agent_type field mapping",
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(_cursor_subagent_args()),
            {},
            spawn_agent_tool_definition=definition,
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert "subagent-call" not in exc_info.value.message


def test_cursor_subagent_rejects_duplicate_advertised_spawn_agent() -> None:
    definition = _advertised_spawn_agent_definition()

    with pytest.raises(
        CursorConnectProtocolError,
        match="multiple spawn_agent tools",
    ):
        cursor_connect._advertised_spawn_agent_tool_definition(
            {
                "runRequest": {
                    "mcpTools": {
                        "mcpTools": [definition, definition],
                    }
                }
            }
        )


def test_cursor_subagent_without_advertised_spawn_agent_uses_canonical_arguments() -> None:
    external_exec_requests: list[dict[str, Any]] = []

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _cursor_subagent_server_message(_cursor_subagent_args()),
        {},
        external_exec_requests=external_exec_requests,
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    assert tool_call["callId"] == "subagent-call"
    assert tool_call["toolName"] == "spawn_agent"
    assert json.loads(tool_call["argsJson"]) == {
        "agent_type": "explorer",
        "model": "read",
        "message": "Inspect the requested files without changing them.",
    }
    assert client_messages == []
    assert len(external_exec_requests) == 1
    request = external_exec_requests[0]
    assert request["call_id"] == "subagent-call"
    assert request["message_field"] == 28
    assert request["request_id"] == 314
    assert request["exec_id"] == "exec-014"
    assert request["tool_call_id"] == "subagent-call"
    assert request["exec_fields"] == [
        (1, 0, 314),
        (15, 2, b"exec-014"),
    ]


@pytest.mark.parametrize(
    ("field_number", "secret"),
    [
        (6, "resume-secret"),
        (8, "continuation-secret"),
        (10, "credential-one-secret"),
        (11, "credential-two-secret"),
        (12, "credential-three-secret"),
        (13, "interrupt-secret"),
        (14, "mode-secret"),
        (15, "fork-secret"),
        (17, "selected-context-secret"),
        (18, "direct-parent-secret"),
        (20, "cloud-branch-secret"),
        (21, "model-parameters-secret"),
    ],
    ids=[
        "resume",
        "continuation",
        "credential-one",
        "credential-two",
        "credential-three",
        "interrupt",
        "mode",
        "fork",
        "selected-context",
        "direct-parent",
        "cloud-branch",
        "model-parameters",
    ],
)
def test_cursor_subagent_rejects_prohibited_optional_fields_without_dispatch(
    field_number: int,
    secret: str,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []
    prohibited_field = cursor_connect._encode_proto_message_field(
        field_number,
        cursor_connect._encode_proto_string_field(1, secret),
    )

    with pytest.raises(
        CursorConnectProtocolError,
        match="unsupported optional field",
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(
                _cursor_subagent_args(extra=prohibited_field)
            ),
            {},
            spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert secret not in exc_info.value.message
    assert secret not in json.dumps(exc_info.value.body or {})


@pytest.mark.parametrize(
    ("non_default_field", "not_leaked"),
    [
        (
            cursor_connect._encode_proto_varint_field(7, 1),
            "Inspect the requested files",
        ),
    ],
    ids=["run-in-background-true"],
)
def test_cursor_subagent_rejects_non_default_optional_fields_without_leaking(
    non_default_field: bytes,
    not_leaked: str,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    with pytest.raises(
        CursorConnectProtocolError,
        match="unsupported optional field",
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(
                _cursor_subagent_args(extra=non_default_field),
            ),
            {},
            spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert not_leaked not in exc_info.value.message
    assert not_leaked not in json.dumps(exc_info.value.body or {})


@pytest.mark.parametrize(
    "environment",
    [2, 3, 99],
    ids=["cloud", "unknown-three", "unknown-ninety-nine"],
)
def test_cursor_subagent_rejects_unsupported_execution_environment_without_leaking(
    environment: int,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    with pytest.raises(
        CursorConnectProtocolError,
        match="unsupported optional field",
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(
                _cursor_subagent_args(
                    extra=cursor_connect._encode_proto_varint_field(19, environment),
                ),
            ),
            {},
            spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert str(environment) not in exc_info.value.message
    assert str(environment) not in json.dumps(exc_info.value.body or {})


@pytest.mark.parametrize(
    ("invalid_field", "not_leaked", "error_match"),
    [
        (
            cursor_connect._encode_proto_varint_field(9, 314),
            "parent-conversation-wire-secret",
            "unsupported optional field",
        ),
        (
            cursor_connect._encode_proto_varint_field(16, 314),
            "root-parent-wire-secret",
            "unsupported optional field",
        ),
        (
            cursor_connect._encode_proto_bytes_field(
                9,
                b"parent-conversation-invalid-utf8-secret\xff",
            ),
            "parent-conversation-invalid-utf8-secret",
            "invalid UTF-8",
        ),
        (
            cursor_connect._encode_proto_bytes_field(
                16,
                b"root-parent-invalid-utf8-secret\xff",
            ),
            "root-parent-invalid-utf8-secret",
            "invalid UTF-8",
        ),
        (
            cursor_connect._encode_proto_string_field(
                19,
                "environment-wire-secret",
            ),
            "environment-wire-secret",
            "unsupported optional field",
        ),
    ],
    ids=[
        "parent-conversation-wrong-wire",
        "root-parent-wrong-wire",
        "parent-conversation-malformed-string",
        "root-parent-malformed-string",
        "environment-wrong-wire",
    ],
)
def test_cursor_subagent_rejects_invalid_parent_state_fields_without_leaking(
    invalid_field: bytes,
    not_leaked: str,
    error_match: str,
) -> None:
    external_exec_requests: list[dict[str, Any]] = []

    with pytest.raises(
        CursorConnectProtocolError,
        match=error_match,
    ) as exc_info:
        cursor_connect._process_agent_server_message(
            _cursor_subagent_server_message(
                _cursor_subagent_args(extra=invalid_field),
            ),
            {},
            spawn_agent_tool_definition=_advertised_spawn_agent_definition(),
            external_exec_requests=external_exec_requests,
        )

    assert external_exec_requests == []
    assert not_leaked not in exc_info.value.message
    assert not_leaked not in json.dumps(exc_info.value.body or {})


def test_cursor_subagent_result_encodes_success_and_error_with_identity() -> None:
    exec_fields = cursor_connect._decode_proto_fields(
        cursor_connect._encode_proto_varint_field(1, 314)
        + cursor_connect._encode_proto_string_field(15, "exec-014")
    )
    exec_request = {
        "call_id": "subagent-call",
        "tool_call_id": "subagent-call",
        "message_field": 28,
        "exec_fields": exec_fields,
    }

    success_messages = cursor_connect._encode_subagent_terminal_result(
        exec_request,
        json.dumps(
            {
                "agent_id": "child-agent",
                "final_message": "Child completed.",
                "tool_call_count": 2,
                "background_reason": "",
                "transcript_path": "/tmp/child.jsonl",
            }
        ),
    )
    error_messages = cursor_connect._encode_subagent_terminal_result(
        exec_request,
        json.dumps(
            {
                "agent_id": "child-agent",
                "error": "Child failed.",
            }
        ),
    )

    for messages in (success_messages, error_messages):
        assert len(messages) == 2
        exec_client_message = cursor_connect._proto_last_field(
            cursor_connect._decode_proto_fields(messages[0]),
            2,
            wire_type=2,
        )
        assert isinstance(exec_client_message, bytes)
        assert cursor_connect._proto_last_field(
            cursor_connect._decode_proto_fields(exec_client_message),
            1,
            wire_type=0,
        ) == 314
        assert cursor_connect._proto_last_field(
            cursor_connect._decode_proto_fields(exec_client_message),
            15,
            wire_type=2,
        ) == b"exec-014"

    success_exec = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success_messages[0]),
        2,
        wire_type=2,
    )
    assert isinstance(success_exec, bytes)
    success_result = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success_exec),
        28,
        wire_type=2,
    )
    assert isinstance(success_result, bytes)
    success = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success_result),
        1,
        wire_type=2,
    )
    assert isinstance(success, bytes)
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success),
        1,
        wire_type=2,
    ) == b"child-agent"
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success),
        2,
        wire_type=2,
    ) == b"Child completed."
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success),
        3,
        wire_type=0,
    ) == 2
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(success),
        5,
        wire_type=2,
    ) == b"/tmp/child.jsonl"

    error_exec = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(error_messages[0]),
        2,
        wire_type=2,
    )
    assert isinstance(error_exec, bytes)
    error_result = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(error_exec),
        28,
        wire_type=2,
    )
    assert isinstance(error_result, bytes)
    error = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(error_result),
        2,
        wire_type=2,
    )
    assert isinstance(error, bytes)
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(error),
        1,
        wire_type=2,
    ) == b"child-agent"
    assert cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(error),
        2,
        wire_type=2,
    ) == b"Child failed."

    for messages in (success_messages, error_messages):
        stream_close = cursor_connect._proto_last_field(
            cursor_connect._decode_proto_fields(messages[1]),
            5,
            wire_type=2,
        )
        assert isinstance(stream_close, bytes)
        assert cursor_connect._proto_last_field(
            cursor_connect._decode_proto_fields(stream_close),
            1,
            wire_type=2,
        ) is not None


@pytest.mark.parametrize(
    "output",
    [
        json.dumps(
            {
                "agent_id": "child-agent",
                "final_message": "Child completed.",
            }
        ),
        json.dumps(
            {
                "agent_id": "child-agent",
                "error": "Child failed.",
            }
        ),
    ],
    ids=["success", "error"],
)
def test_cursor_subagent_result_encodes_without_invented_identity(
    output: str,
) -> None:
    messages = cursor_connect._encode_subagent_terminal_result(
        {
            "exec_fields": [],
        },
        output,
    )

    exec_client_message = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(messages[0]),
        2,
        wire_type=2,
    )
    assert isinstance(exec_client_message, bytes)
    exec_client_fields = cursor_connect._decode_proto_fields(exec_client_message)
    assert cursor_connect._proto_last_field(
        exec_client_fields,
        1,
        wire_type=0,
    ) is None
    assert cursor_connect._proto_last_field(
        exec_client_fields,
        15,
        wire_type=2,
    ) is None
    assert cursor_connect._proto_last_field(
        exec_client_fields,
        28,
        wire_type=2,
    ) is not None

    exec_client_control = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(messages[1]),
        5,
        wire_type=2,
    )
    assert isinstance(exec_client_control, bytes)
    stream_close = cursor_connect._proto_last_field(
        cursor_connect._decode_proto_fields(exec_client_control),
        1,
        wire_type=2,
    )
    assert stream_close == b""


class _CountingRetainedSession:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.close()


def test_cursor_codex_path_returns_native_function_call_and_replays_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert codex_candidate_calls._responses_input_to_cursor_messages(
        {"input": "run pwd"}
    ) == [{"role": "user", "content": "run pwd"}]

    class FakeRetainedSession:
        def __init__(self) -> None:
            self.outputs: list[list[tuple[str, Any]]] = []
            self.close_calls = 0
            self.results = [
                CursorAgentRunResult(
                    tool_calls=[
                        {
                            "id": "cursor-item-2",
                            "call_id": "call-2",
                            "name": "exec_command",
                            "arguments": '{"cmd":"date"}',
                        }
                    ],
                    retained_session=self,
                ),
                CursorAgentRunResult(text="final answer", turn_ended=True),
            ]

        async def continue_with_tool_outputs(
            self,
            outputs: list[tuple[str, Any]],
        ) -> CursorAgentRunResult:
            self.outputs.append(outputs)
            return self.results.pop(0)

        def close(self) -> None:
            self.close_calls += 1

        async def aclose(self) -> None:
            self.close()

    retained_session = FakeRetainedSession()

    class FakeCursorClient:
        calls: list[dict[str, Any]] = []
        init_kwargs: list[dict[str, Any]] = []
        results = [
            CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "fc_d772cb13-2b1b-9884-b49a-e0ba78733f62_0",
                        "call_id": "call-3e993636-853f-474a-b20c-439984662b5d-0",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    }
                ],
                retained_session=retained_session,
            )
        ]

        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs.append(kwargs)

        async def run(self, payload: dict[str, Any], **kwargs: Any) -> CursorAgentRunResult:
            self.calls.append({"payload": payload, "kwargs": kwargs})
            return self.results.pop(0)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    first = _call(
        {
            "model": "work",
            "input": "run pwd",
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key="bogus-openai-bearer",
    )
    first_body = json.loads(first.body)
    first_item = first_body["output"][0]
    assert first_item["type"] == "function_call"
    assert (
        first_item["call_id"]
        == "call-3e993636-853f-474a-b20c-439984662b5d-0"
    )
    assert first_item["id"] == "fc_d772cb13-2b1b-9884-b49a-e0ba78733f62_0"
    assert first_item["arguments"] == '{"cmd":"pwd"}'
    assert "mcp_call" not in json.dumps(first_body)
    assert FakeCursorClient.init_kwargs == [{}]
    assert FakeCursorClient.calls[0]["kwargs"]["url"] == run_url(None)
    assert "bogus-openai-bearer" not in json.dumps(
        FakeCursorClient.calls,
        default=str,
    )
    assert "chatgpt.com" not in json.dumps(FakeCursorClient.calls, default=str)

    second = _call(
        {
            "model": "work",
            "previous_response_id": first_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-3e993636-853f-474a-b20c-439984662b5d-0",
                    "output": "pwd output",
                },
            ],
        }
    )
    second_body = json.loads(second.body)
    assert second_body["output"][0]["call_id"] == "call-2"

    third = _call(
        {
            "model": "work",
            "previous_response_id": second_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": "date output",
                }
            ],
        }
    )
    third_body = json.loads(third.body)
    assert third_body["output_text"] == "final answer"

    first_run = FakeCursorClient.calls[0]["payload"]["runRequest"]
    assert first_run["action"]["userMessageAction"]["userMessage"]["text"] == "run pwd"
    assert first_run["action"]["userMessageAction"]["requestContext"] == {}
    assert first_run["requestedModel"] == {
        "modelId": "grok-4.6",
        "parameters": [
            {"id": "effort", "value": "high"},
            {"id": "fast", "value": "false"},
        ],
    }
    assert first_run["mcpTools"]["mcpTools"][0]["name"] == "exec_command"
    assert "rootPromptMessagesJson" not in json.dumps(first_run)

    assert len(FakeCursorClient.calls) == 1
    assert retained_session.outputs == [
        [
            (
                "call-3e993636-853f-474a-b20c-439984662b5d-0",
                "pwd output",
            )
        ],
        [("call-2", "date output")],
    ]
    assert retained_session.close_calls == 1


def test_cursor_local_exec_function_call_replays_through_existing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRetainedSession:
        def __init__(self) -> None:
            self.closed = False

        async def continue_with_tool_outputs(
            self,
            _outputs: list[tuple[str, Any]],
        ) -> CursorAgentRunResult:
            return CursorAgentRunResult(text="command completed", turn_ended=True)

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.close()

    retained_session = FakeRetainedSession()

    class FakeCursorClient:
        calls: list[dict[str, Any]] = []
        results = [
            CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "fc_exec-local",
                        "call_id": "exec-local",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
                    }
                ],
                retained_session=retained_session,
            )
        ]

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            payload: dict[str, Any],
            **kwargs: Any,
        ) -> CursorAgentRunResult:
            self.calls.append({"payload": payload, "kwargs": kwargs})
            return self.results.pop(0)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    first = _call(
        {
            "model": "work",
            "input": "run pwd",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "exec_command"},
                }
            ],
        }
    )
    first_body = json.loads(first.body)
    assert first_body["output"][0] == {
        "id": "fc_exec-local",
        "type": "function_call",
        "status": "completed",
        "call_id": "exec-local",
        "name": "exec_command",
        "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
    }
    assert codex_candidate_calls._peek_cursor_replay_state(
        first_body["id"]
    )["retained_session"] is retained_session

    second = _call(
        {
            "model": "work",
            "previous_response_id": first_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "exec-local",
                    "output": "pwd output",
                }
            ],
        }
    )

    assert json.loads(second.body)["output_text"] == "command completed"
    assert len(FakeCursorClient.calls) == 1
    assert retained_session.closed is True


def test_cursor_unretained_tool_output_continuation_does_not_build_replacement_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent import common_utils

    build_calls: list[dict[str, Any]] = []
    run_calls: list[dict[str, Any]] = []
    real_build_run_request = common_utils.build_run_request

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            payload: dict[str, Any],
            **kwargs: Any,
        ) -> CursorAgentRunResult:
            run_calls.append({"payload": payload, "kwargs": kwargs})
            return CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "cursor-item-1",
                        "call_id": "call-1",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    }
                ]
            )

    def _build_run_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        build_calls.append({"args": args, "kwargs": kwargs})
        return real_build_run_request(*args, **kwargs)

    monkeypatch.setattr(common_utils, "build_run_request", _build_run_request)
    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    first = _call(
        {
            "model": "work",
            "input": "run pwd",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "exec_command"},
                }
            ],
        }
    )
    first_body = json.loads(first.body)

    with pytest.raises(CursorConnectError) as exc_info:
        _call(
            {
                "model": "work",
                "previous_response_id": first_body["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": first_body["output"][0]["call_id"],
                        "output": "pwd output",
                    }
                ],
            }
        )

    exc = exc_info.value
    assert exc.status_code == 409
    assert getattr(
        exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        False,
    )
    assert len(build_calls) == 1
    assert len(run_calls) == 1
    assert (
        codex_candidate_calls._CURSOR_TOOL_CONTINUATION_CUE
        not in json.dumps(build_calls[0])
    )
    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._peek_cursor_replay_state(first_body["id"])


def test_cursor_fresh_full_history_tool_output_emits_continuation_cue() -> None:
    messages = codex_candidate_calls._responses_input_to_cursor_messages(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "run the requested read",
                },
                {
                    "type": "function_call",
                    "call_id": "read-call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"read"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "read-call",
                    "output": "read result",
                },
            ]
        }
    )

    assert messages[-1] == {
        "role": "user",
        "content": codex_candidate_calls._CURSOR_TOOL_CONTINUATION_CUE,
        codex_candidate_calls._CURSOR_TOOL_CONTINUATION_CUE_MARKER: True,
    }


def test_cursor_fresh_full_history_tool_output_does_not_build_replacement_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent import common_utils
    from litellm.proxy._types import ProxyException

    build_calls: list[dict[str, Any]] = []
    run_calls: list[dict[str, Any]] = []
    real_build_run_request = common_utils.build_run_request

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            payload: dict[str, Any],
            **kwargs: Any,
        ) -> CursorAgentRunResult:
            run_calls.append({"payload": payload, "kwargs": kwargs})
            return CursorAgentRunResult(text="unexpected fresh run", turn_ended=True)

    def _build_run_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        build_calls.append({"args": args, "kwargs": kwargs})
        return real_build_run_request(*args, **kwargs)

    monkeypatch.setattr(common_utils, "build_run_request", _build_run_request)
    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    with pytest.raises(ProxyException) as exc_info:
        asyncio.run(
            llm_passthrough_endpoints._perform_codex_auto_agent_alias_candidate_request(
                endpoint="/v1/responses",
                request=_request(),
                fastapi_response=Response(),
                user_api_key_dict=None,
                candidate=_candidate(provider="cursor_agent"),
                candidate_body={
                    "model": "work",
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "exec_command"},
                        }
                    ],
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "Complete the original assignment in /workspace.",
                        },
                        {
                            "type": "function_call",
                            "call_id": "pwd-call",
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "pwd-call",
                            "output": "/workspace",
                        },
                    ],
                },
                target_url="https://chatgpt.com/backend-api/codex/responses",
                api_key="access-token",
                forward_headers=False,
            )
        )

    exc = exc_info.value
    assert exc.status_code == 409
    assert getattr(
        exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        False,
    )
    assert exc.failure_phase == "cursor_session_continuation"
    assert exc.attempted_provider_call is False
    assert build_calls == []
    assert run_calls == []


def test_cursor_retained_session_consumes_function_call_output_without_new_run(
    monkeypatch: pytest.MonkeyPatch,
    _route_rollup_state: None,
) -> None:
    captured = _capture_real_route_log(monkeypatch)

    class FakeRetainedSession:
        def __init__(self) -> None:
            self.outputs: list[list[tuple[str, Any]]] = []
            self.closed = False

        async def continue_with_tool_outputs(
            self,
            outputs: list[tuple[str, Any]],
        ) -> CursorAgentRunResult:
            self.outputs.append(outputs)
            return CursorAgentRunResult(text="read complete", turn_ended=True)

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.close()

    session = FakeRetainedSession()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-retained",
        messages=[
            {"role": "user", "content": "run the requested read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read-call",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"read"}',
                        },
                    }
                ],
            },
        ],
        tools=[],
        retained_session=session,
    )

    response = _call(
        {
            "model": "work",
            "previous_response_id": "resp-retained",
            "litellm_metadata": {
                "codex_auto_agent_alias": "cursor-test",
                "litellm_call_id": "cursor-retained-call",
            },
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "read-call",
                    "output": "LITELLM_CURSOR_READ_V1:envelope",
                }
            ],
        },
        selected_account={
            "provider": "openai",
            "account_hash": "cursor-account",
            "account_display": "selected@example.com",
        },
    )

    assert json.loads(response.body)["output_text"] == "read complete"
    assert session.outputs == [
        [("read-call", "LITELLM_CURSOR_READ_V1:envelope")]
    ]
    assert session.closed is True
    _assert_cursor_route_context(captured)
    flushed = flush_aawm_route_rollups(force=True)
    assert len(flushed) == 2
    rendered = "\n".join(flushed)
    assert rendered.count("Turns: 1") == 1
    assert "selected@example.com" in rendered
    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._peek_cursor_replay_state("resp-retained")


def test_cursor_continuation_requires_matching_function_call() -> None:
    with pytest.raises(ValueError, match="matching"):
        codex_candidate_calls._responses_input_to_cursor_messages(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "missing",
                        "output": "result",
                    }
                ]
            }
        )


def test_cursor_fresh_replay_dispatch_fails_closed_without_complete_state() -> None:
    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-missing")
        )
        is None
    )

    codex_candidate_calls._store_cursor_replay_state(
        "resp-retained",
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    }
                ],
            },
        ],
        tools=[],
        retained_session=object(),
    )
    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-retained")
        )
        is None
    )


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "run commands"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{}"},
                    },
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{}"},
                    },
                ],
            },
        ],
        [
            {"role": "user", "content": "run commands"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
            {"role": "tool", "tool_call_id": "call-1", "content": "duplicate"},
        ],
        [
            {"role": "user", "content": "run commands"},
            {"role": "tool", "tool_call_id": "call-1", "content": "orphan"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{}"},
                    }
                ],
            },
        ],
    ],
    ids=["duplicate-call", "duplicate-output", "output-before-call"],
)
def test_cursor_fresh_replay_dispatch_rejects_invalid_historical_call_state(
    messages: list[dict[str, Any]],
) -> None:
    codex_candidate_calls._store_cursor_replay_state(
        "resp-invalid-history",
        messages=messages,
        tools=[],
    )

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-invalid-history")
        )
        is None
    )


def test_cursor_fresh_replay_dispatch_rejects_completed_historical_output() -> None:
    codex_candidate_calls._store_cursor_replay_state(
        "resp-completed-output",
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "old output"},
        ],
        tools=[],
    )

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-completed-output")
        )
        is None
    )


@pytest.mark.parametrize(
    "output_item",
    [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "pwd output",
            "message_id": "cursor-message",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": {"content": "pwd output"},
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "callId": "call-2",
            "output": "pwd output",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
        },
        {
            "type": "function_call_output",
            "id": "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1",
            "call_id": "call-1",
            "output": "pwd output",
        },
    ],
    ids=[
        "unknown-field",
        "nested-output",
        "conflicting-call-id-aliases",
        "missing-output",
        "id-without-metadata",
    ],
)
def test_cursor_fresh_replay_dispatch_rejects_noncanonical_incremental_output(
    output_item: dict[str, Any],
) -> None:
    _store_replay_state()
    body = _replay_body()
    body["input"] = [output_item]

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
        is None
    )


def test_cursor_fresh_replay_dispatch_canonicalizes_camel_call_id() -> None:
    _store_replay_state()
    body = _replay_body()
    body["input"] = [
        {
            "type": "function_call_output",
            "callId": " call-1 ",
            "output": "pwd output",
        }
    ]

    rebuilt = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
    )

    assert rebuilt is not None
    assert rebuilt["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "pwd output",
    }


@pytest.mark.parametrize(
    ("item_id", "turn_id"),
    [
        (
            "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1",
            "01a06244-8523-79c2-b8ff-59238c523de8",
        ),
        (
            "fco_01a06244-9b3d-7722-a6ba-fbdd312c711b",
            "01a06244-867a-75e0-8714-e6422f086d33",
        ),
        (
            "fco_01a06244-b1e8-7453-bbb8-fb4c7663270b",
            "01a06244-8756-7081-a4a2-4228ecbc5b92",
        ),
    ],
    ids=["work", "expert", "sota-xai"],
)
def test_cursor_fresh_replay_dispatch_canonicalizes_stock_codex_output_metadata(
    item_id: str,
    turn_id: str,
) -> None:
    _store_replay_state()
    body = _replay_body()
    body["input"] = [
        {
            "type": "function_call_output",
            "id": item_id,
            "call_id": "call-1",
            "output": "pwd output",
            "internal_chat_message_metadata_passthrough": {
                "turn_id": turn_id,
                "create_time": 1788355059.5830524,
            },
        }
    ]

    rebuilt = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
    )

    assert rebuilt is not None
    assert rebuilt["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "pwd output",
    }


@pytest.mark.parametrize(
    ("function_call_item_id", "function_call_call_id"),
    [
        (
            "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_0",
            "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
        ),
        (
            "fc_79fe4188-5748-4b27-92ef-7c5b4d50e373",
            "79fe4188-5748-4b27-92ef-7c5b4d50e373",
        ),
    ],
    ids=["legacy-indexed-id", "current-stock-id"],
)
@pytest.mark.parametrize(
    "include_metadata",
    [False, True],
    ids=["metadata-omitted", "metadata-present"],
)
def test_cursor_fresh_replay_dispatch_accepts_stock_codex_full_history(
    function_call_item_id: str,
    function_call_call_id: str,
    include_metadata: bool,
) -> None:
    body = _stock_codex_full_history_body(
        include_metadata=include_metadata,
        function_call_item_id=function_call_item_id,
        function_call_call_id=function_call_call_id,
    )
    expected_item_keys = [
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "name", "arguments", "call_id"},
        {"type", "id", "call_id", "output"},
    ]
    if include_metadata:
        expected_item_keys = [
            item_keys | {"internal_chat_message_metadata_passthrough"}
            for item_keys in expected_item_keys
        ]
    assert [set(item) for item in body["input"]] == expected_item_keys
    assert len(body["tools"]) == 14
    assert [tool["type"] for tool in body["tools"][:11]] == ["function"] * 11
    assert body["tools"][11]["type"] == "custom"
    assert set(body["tools"][12]) == {
        "description",
        "execution",
        "parameters",
        "type",
    }
    assert body["tools"][12]["type"] == "tool_search"
    assert body["tools"][13]["type"] == "web_search"
    continuation_exc = CursorConnectError("missing retained session", status_code=409)
    setattr(
        continuation_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
        is None
    )
    rebuilt = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
    )

    assert rebuilt is not None
    assert rebuilt["tools"] == body["tools"]
    assert rebuilt["input"][:-2] == [
        {
            "role": item["role"],
            "content": "".join(part["text"] for part in item["content"]),
        }
        for item in body["input"][:-2]
    ]
    assert rebuilt["input"][-2:] == [
        {
            "type": "function_call",
            "call_id": function_call_call_id,
            "name": "exec_command",
            "arguments": (
                '{"cmd":"pwd","workdir":"/home/zepfu/projects/litellm"}'
            ),
        },
        {
            "type": "function_call_output",
            "call_id": function_call_call_id,
            "output": (
                "Chunk ID: 91fb25\n"
                "Wall time: 0.0000 seconds\n"
                "Process exited with code 0\n"
                "Final output:\n"
                "/home/zepfu/projects/litellm\n"
            ),
        },
    ]
    assert all(
        "id" not in item
        and "internal_chat_message_metadata_passthrough" not in item
        for item in rebuilt["input"]
    )


@pytest.mark.parametrize(
    "external_web_access",
    [True, False],
    ids=["external-web-access-true", "external-web-access-false"],
)
def test_cursor_fresh_replay_dispatch_accepts_stock_web_search_extension(
    external_web_access: bool,
) -> None:
    body = _stock_codex_full_history_body()
    web_search_tool = {
        "type": "web_search",
        "external_web_access": external_web_access,
    }
    body["tools"][13] = web_search_tool
    rejection: dict[str, Any] = {}

    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        body,
        continuation_exc=_cursor_continuation_failure(),
        rejection_diagnostic_out=rejection,
    )

    assert rebuilt is not None, rejection
    assert len(body["tools"]) == 14
    assert rebuilt["tools"] == body["tools"]
    assert rejection == {}


@pytest.mark.parametrize(
    "web_search_tool",
    [
        {"external_web_access": True},
        {"type": "web_search", "external_web_access": 1},
        {"type": "web_search", "external_web_access": "true"},
        {"type": "web_search", "external_web_access": None},
        {"type": "web_search", "external_web_access": True, "filters": {}},
        {"type": "web_search_preview", "external_web_access": True},
    ],
    ids=[
        "missing-type-key",
        "integer-external-web-access",
        "string-external-web-access",
        "null-external-web-access",
        "extra-key",
        "wrong-tool-type",
    ],
)
def test_cursor_fresh_replay_dispatch_rejects_malformed_stock_web_search_extension(
    web_search_tool: dict[str, Any],
) -> None:
    body = _stock_codex_full_history_body()
    body["tools"][13] = web_search_tool
    rejection: dict[str, Any] = {}

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=_cursor_continuation_failure(),
            rejection_diagnostic_out=rejection,
        )
        is None
    )
    diagnostic = rejection[
        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    ]
    assert diagnostic["stage"] == "provider_neutral_tools"
    assert diagnostic["reason"] == "tool_validation"
    assert diagnostic["tool_index"] == 13


def test_cursor_fresh_replay_dispatch_accepts_stock_tool_search_with_legacy_tool_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai.types.responses import response_create_params
    from pydantic import TypeAdapter, ValidationError

    body = _stock_codex_full_history_body()
    stock_tool_search = body["tools"][12]
    legacy_tool_args = tuple(
        tool_type
        for tool_type in get_args(response_create_params.ToolParam)
        if getattr(tool_type, "__name__", "") != "ToolSearchToolParam"
    )
    legacy_tool_param = Union[legacy_tool_args]
    monkeypatch.setattr(
        response_create_params,
        "ToolParam",
        legacy_tool_param,
    )

    with pytest.raises(ValidationError) as validation_exc:
        TypeAdapter(legacy_tool_param).validate_python(
            stock_tool_search,
            strict=True,
        )
    assert any(
        error["type"] == "literal_error"
        and error["loc"][-1] == "type"
        and error["input"] == "tool_search"
        for error in validation_exc.value.errors()
    )

    rejection: dict[str, Any] = {}
    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        body,
        continuation_exc=_cursor_continuation_failure(),
        rejection_diagnostic_out=rejection,
    )

    assert rebuilt is not None, rejection
    assert rebuilt["tools"][12] == stock_tool_search
    assert rejection == {}


@pytest.mark.parametrize(
    "mutate_tool_search",
    [
        lambda tool: tool.update({"unexpected": "value"}),
        lambda tool: tool.update({"execution": "server"}),
        lambda tool: tool["parameters"]["properties"]["limit"].update(
            {"type": "string"}
        ),
        lambda tool: tool["parameters"].update({"required": []}),
        lambda tool: tool.update({"type": "unknown_tool"}),
    ],
    ids=[
        "extra-top-level-key",
        "wrong-execution",
        "wrong-limit-type",
        "wrong-required",
        "unknown-tool-type",
    ],
)
def test_cursor_fresh_replay_dispatch_rejects_malformed_stock_tool_search(
    mutate_tool_search: Any,
) -> None:
    body = _stock_codex_full_history_body()
    mutate_tool_search(body["tools"][12])
    rejection: dict[str, Any] = {}

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=_cursor_continuation_failure(),
            rejection_diagnostic_out=rejection,
        )
        is None
    )
    diagnostic = rejection[
        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    ]
    assert diagnostic["stage"] == "provider_neutral_tools"
    assert diagnostic["reason"] == "tool_validation"
    assert diagnostic["tool_index"] == 12
    assert diagnostic["tool_type"] in {"tool_search", "unknown_tool"}


def test_cursor_fresh_replay_dispatch_rejects_provider_owned_full_history() -> None:
    body = _stock_codex_full_history_body()
    body["input"].insert(
        -2,
        {
            "type": "reasoning",
            "id": "rs_provider-owned-state",
            "encrypted_content": "opaque-provider-state",
        },
    )
    continuation_exc = CursorConnectError("missing retained session", status_code=409)
    setattr(
        continuation_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
        is None
    )


@pytest.mark.parametrize(
    ("item_index", "item_update", "expected_reason"),
    [
        (0, {"id": "msg_not-a-uuid"}, "id_shape"),
        (4, {"id": "fc_not-a-uuid_0"}, "id_shape"),
        (5, {"id": "fco_not-a-uuid"}, "id_shape"),
        (4, {"id": "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_"}, "id_shape"),
        (
            4,
            {"id": "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_01"},
            "id_shape",
        ),
        (
            4,
            {"id": "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_-1"},
            "id_shape",
        ),
        (
            4,
            {"id": "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_x"},
            "id_shape",
        ),
        (
            4,
            {"id": "fc_A4335D9D-8539-9945-BDC7-F14243B0E9B8"},
            "id_shape",
        ),
        (2, {"unexpected": "value"}, "item_key_set"),
        (
            2,
            {
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": "turn-1",
                    "create_time": 1788357449.79907,
                    "content_item_kinds": ["user.text"],
                }
            },
            "id_shape",
        ),
    ],
    ids=[
        "malformed-message-id",
        "malformed-function-call-id",
        "malformed-function-call-output-id",
        "empty-function-call-id-suffix",
        "leading-zero-function-call-id-suffix",
        "negative-function-call-id-suffix",
        "nonnumeric-function-call-id-suffix",
        "noncanonical-function-call-id-uuid",
        "unknown-item-key",
        "invalid-present-metadata",
    ],
)
def test_cursor_fresh_replay_dispatch_rejects_unsafe_stock_codex_full_history(
    item_index: int,
    item_update: dict[str, Any],
    expected_reason: str,
) -> None:
    body = _stock_codex_full_history_body()
    body["input"][item_index].update(item_update)
    continuation_exc = CursorConnectError("missing retained session", status_code=409)
    setattr(
        continuation_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )
    rejection: dict[str, Any] = {}

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
            rejection_diagnostic_out=rejection,
        )
        is None
    )
    diagnostic = rejection[
        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    ]
    assert diagnostic["stage"] == "stock_full_history"
    assert diagnostic["reason"] == expected_reason


@pytest.mark.parametrize(
    ("mutate_body", "expected_reason"),
    [
        (
            lambda body: body["input"][2].update({"unexpected_key": "secret"}),
            "item_key_set",
        ),
        (
            lambda body: body["input"][0].update({"id": "msg_invalid"}),
            "id_shape",
        ),
        (
            lambda body: body["input"][2]["content"][0].update(
                {"type": "output_text"}
            ),
            "content_part_type",
        ),
        (
            lambda body: [
                item.update({"content": [{"type": "input_text", "text": ""}]})
                for item in body["input"]
                if item.get("role") == "user"
            ],
            "empty_user_text",
        ),
        (
            lambda body: body["input"][4].update({"arguments": "[]"}),
            "arguments_not_object",
        ),
        (
            lambda body: body["input"][5].update({"output": {"secret": "value"}}),
            "output_not_string",
        ),
        (
            lambda body: body["input"][5].update({"call_id": "unresolved"}),
            "unresolved_call_id",
        ),
    ],
    ids=[
        "item-key-set",
        "id-shape",
        "content-part-type",
        "empty-user-text",
        "arguments-not-object",
        "output-not-string",
        "unresolved-call-id",
    ],
)
def test_cursor_fresh_replay_dispatch_reports_stock_history_rejection_reason(
    mutate_body: Any,
    expected_reason: str,
) -> None:
    body = _stock_codex_full_history_body()
    mutate_body(body)
    rejection: dict[str, Any] = {}

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=_cursor_continuation_failure(),
            rejection_diagnostic_out=rejection,
        )
        is None
    )

    diagnostic = rejection[
        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    ]
    assert diagnostic["stage"] == "stock_full_history"
    assert diagnostic["reason"] == expected_reason
    assert "secret" not in json.dumps(diagnostic)
    assert "value" not in json.dumps(diagnostic)


def test_cursor_fresh_replay_dispatch_reports_provider_tool_rejection() -> None:
    codex_candidate_calls._store_cursor_replay_state(
        "resp-provider-tool-rejection",
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    }
                ],
            },
        ],
        tools=["secret tool"],
    )
    rejection: dict[str, Any] = {}

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-provider-tool-rejection"),
            rejection_diagnostic_out=rejection,
        )
        is None
    )

    assert rejection[
        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    ] == {
        "stage": "provider_neutral_tools",
        "reason": "tool_item",
        "tool_index": 0,
    }


def test_cursor_fresh_replay_dispatch_success_emits_no_rejection_diagnostic() -> None:
    _store_replay_state()
    rejection: dict[str, Any] = {"stale": "value"}

    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        _replay_body(),
        rejection_diagnostic_out=rejection,
    )

    assert rebuilt is not None
    assert rejection == {}


def test_cursor_replay_rebuilt_body_rejection_reports_replay_safety_reason() -> None:
    diagnostic = (
        codex_candidate_calls._cursor_replay_fresh_dispatch_reject_for_replay_safety(
            SimpleNamespace(
                safe=False,
                classification="id_only_reasoning_reference",
            )
        )
    )

    assert diagnostic == {
        "stage": "rebuilt_body_replay_unsafe",
        "reason": "id_only_reasoning_reference",
    }


@pytest.mark.parametrize(
    "metadata_update",
    [
        {"status": "completed"},
        {"id": "msg_not-a-function-call-output"},
        {"id": "fco_"},
        {"id": "fco_not-a-uuid"},
        {"id": " fco_01a06244-9f7f-7fe1-869b-23d587ad56f1"},
        {"id": "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1 "},
        {"id": "fco_01A06244-9F7F-7FE1-869B-23D587AD56F1"},
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "turn-1",
                "create_time": 1788355059.5830524,
            }
        },
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": " 01a06244-8523-79c2-b8ff-59238c523de8",
                "create_time": 1788355059.5830524,
            }
        },
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "01a06244-8523-79c2-b8ff-59238c523de8 ",
                "create_time": 1788355059.5830524,
            }
        },
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "01A06244-8523-79C2-B8FF-59238C523DE8",
                "create_time": 1788355059.5830524,
            }
        },
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "01a06244-8523-79c2-b8ff-59238c523de8",
                "create_time": 1788355059.5830524,
                "opaque_state": {"provider": "cursor"},
            }
        },
        {
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "01a06244-8523-79c2-b8ff-59238c523de8",
                "create_time": float("inf"),
            }
        },
    ],
    ids=[
        "unobserved-status",
        "wrong-id-kind",
        "empty-id-suffix",
        "arbitrary-id-suffix",
        "id-leading-space",
        "id-trailing-space",
        "id-uppercase",
        "arbitrary-turn-id",
        "turn-id-leading-space",
        "turn-id-trailing-space",
        "turn-id-uppercase",
        "unknown-nested-metadata",
        "nonfinite-create-time",
    ],
)
def test_cursor_fresh_replay_dispatch_rejects_unsafe_output_metadata(
    metadata_update: dict[str, Any],
) -> None:
    _store_replay_state()
    body = _replay_body()
    output_item = {
        "type": "function_call_output",
        "id": "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1",
        "call_id": "call-1",
        "output": "pwd output",
        "internal_chat_message_metadata_passthrough": {
            "turn_id": "01a06244-8523-79c2-b8ff-59238c523de8",
            "create_time": 1788355059.5830524,
        },
    }
    output_item.update(metadata_update)
    body["input"] = [output_item]

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
        is None
    )


def test_cursor_fresh_replay_dispatch_rejects_duplicate_output_item_ids() -> None:
    codex_candidate_calls._store_cursor_replay_state(
        "resp-duplicate-item-id",
        messages=[
            {"role": "user", "content": "run commands"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"date"}',
                        },
                    },
                ],
            },
        ],
        tools=[],
    )
    duplicate_item_id = "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1"
    body = {
        "model": "work",
        "previous_response_id": "resp-duplicate-item-id",
        "input": [
            {
                "type": "function_call_output",
                "id": duplicate_item_id,
                "call_id": call_id,
                "output": output,
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788355059.5830524,
                },
            }
            for call_id, output, turn_id in (
                (
                    "call-1",
                    "pwd output",
                    "01a06244-8523-79c2-b8ff-59238c523de8",
                ),
                (
                    "call-2",
                    "date output",
                    "01a06244-867a-75e0-8714-e6422f086d33",
                ),
            )
        ],
    }

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(body)
        is None
    )


@pytest.mark.parametrize(
    "tools",
    [
        ["not-a-tool"],
        [{"type": "unsupported"}],
        [{"type": "function", "function": {}}],
        [{"type": "function", "function": {"name": "exec_command", "extra": 1}}],
    ],
    ids=["non-mapping", "unsupported-type", "missing-name", "unknown-field"],
)
def test_cursor_fresh_replay_dispatch_rejects_malformed_stored_tools(
    tools: list[Any],
) -> None:
    codex_candidate_calls._store_cursor_replay_state(
        "resp-malformed-tools",
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    }
                ],
            },
        ],
        tools=tools,
    )

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            _replay_body("resp-malformed-tools")
        )
        is None
    )


def test_cursor_continuation_fails_for_missing_and_expired_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CursorConnectError, match="missing"):
        _call(
            {
                "model": "work",
                "previous_response_id": "resp-missing",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "result",
                    }
                ],
            }
        )

    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_TTL_SECONDS", -1.0)
    codex_candidate_calls._store_cursor_replay_state(
        "resp-expired",
        messages=[{"role": "user", "content": "run pwd"}],
        tools=[],
    )
    with pytest.raises(CursorConnectError, match="expired"):
        _call(
            {
                "model": "work",
                "previous_response_id": "resp-expired",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "result",
                    }
                ],
            }
        )


def test_cursor_replay_registry_enforces_max_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_MAX_SIZE", 1)
    for response_id in ("resp-old", "resp-new"):
        codex_candidate_calls._store_cursor_replay_state(
            response_id,
            messages=[{"role": "user", "content": response_id}],
            tools=[],
        )

    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._take_cursor_replay_state("resp-old")
    assert codex_candidate_calls._take_cursor_replay_state("resp-new")["messages"]


def test_cursor_replay_registry_expires_idle_retained_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_TTL_SECONDS", 0.01)
    session = _CountingRetainedSession()

    async def run_scenario() -> None:
        codex_candidate_calls._store_cursor_replay_state(
            "resp-idle",
            messages=[],
            tools=[],
            retained_session=session,
        )
        state = codex_candidate_calls._CURSOR_REPLAY_REGISTRY["resp-idle"]
        assert state["expiry_handle"] is not None

        await asyncio.sleep(0.03)

        assert "resp-idle" not in codex_candidate_calls._CURSOR_REPLAY_REGISTRY

    asyncio.run(run_scenario())
    assert session.close_calls == 1


def test_cursor_replay_registry_replacement_timer_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_session = _CountingRetainedSession()
    new_session = _CountingRetainedSession()

    async def run_scenario() -> None:
        monkeypatch.setattr(
            codex_candidate_calls,
            "_CURSOR_REPLAY_TTL_SECONDS",
            0.01,
        )
        codex_candidate_calls._store_cursor_replay_state(
            "resp-replaced",
            messages=[],
            tools=[],
            retained_session=old_session,
        )
        old_state = codex_candidate_calls._CURSOR_REPLAY_REGISTRY["resp-replaced"]

        monkeypatch.setattr(
            codex_candidate_calls,
            "_CURSOR_REPLAY_TTL_SECONDS",
            0.1,
        )
        codex_candidate_calls._store_cursor_replay_state(
            "resp-replaced",
            messages=[],
            tools=[],
            retained_session=new_session,
        )
        codex_candidate_calls._expire_cursor_replay_state(
            "resp-replaced",
            old_state,
        )
        await asyncio.sleep(0.03)

        current = codex_candidate_calls._CURSOR_REPLAY_REGISTRY["resp-replaced"]
        assert current["retained_session"] is new_session
        assert new_session.close_calls == 0
        codex_candidate_calls._clear_cursor_replay_registry()

    asyncio.run(run_scenario())
    assert old_session.close_calls == 1
    assert new_session.close_calls == 1


def test_cursor_replay_registry_consume_and_clear_cancel_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_TTL_SECONDS", 0.01)
    consumed_session = _CountingRetainedSession()
    cleared_session = _CountingRetainedSession()

    async def run_scenario() -> None:
        codex_candidate_calls._store_cursor_replay_state(
            "resp-consumed",
            messages=[],
            tools=[],
            retained_session=consumed_session,
        )
        consumed_state = codex_candidate_calls._CURSOR_REPLAY_REGISTRY[
            "resp-consumed"
        ]
        consumed_handle = consumed_state["expiry_handle"]
        codex_candidate_calls._consume_cursor_replay_state(
            "resp-consumed",
            expected_state=consumed_state,
        )
        assert consumed_handle.cancelled()

        codex_candidate_calls._store_cursor_replay_state(
            "resp-cleared",
            messages=[],
            tools=[],
            retained_session=cleared_session,
        )
        cleared_handle = codex_candidate_calls._CURSOR_REPLAY_REGISTRY[
            "resp-cleared"
        ]["expiry_handle"]
        codex_candidate_calls._clear_cursor_replay_registry()
        assert cleared_handle.cancelled()

        await asyncio.sleep(0.03)

    asyncio.run(run_scenario())
    assert consumed_session.close_calls == 1
    assert cleared_session.close_calls == 1


def test_cursor_retained_session_failure_closes_once() -> None:
    class FailingRetainedSession(_CountingRetainedSession):
        async def continue_with_tool_outputs(
            self,
            _outputs: list[tuple[str, Any]],
        ) -> CursorAgentRunResult:
            raise RuntimeError("continuation failed")

    session = FailingRetainedSession()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-failing-retained",
        messages=[
            {"role": "user", "content": "run the requested read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read-call",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"read"}',
                        },
                    }
                ],
            },
        ],
        tools=[],
        retained_session=session,
    )

    with pytest.raises(RuntimeError, match="continuation failed"):
        _call(
            {
                "model": "work",
                "previous_response_id": "resp-failing-retained",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "read-call",
                        "output": "LITELLM_CURSOR_READ_V1:envelope",
                    }
                ],
            }
        )

    assert session.close_calls == 1


def test_cursor_stream_uses_responses_event_schema(
    monkeypatch: pytest.MonkeyPatch,
    _route_rollup_state: None,
) -> None:
    captured = _capture_real_route_log(monkeypatch)

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            return CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "cursor-item-1",
                        "call_id": "call-1",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    }
                ]
            )

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )
    response = _call(
        {
            "model": "work",
            "input": "run pwd",
            "stream": True,
            "litellm_metadata": {
                "codex_auto_agent_alias": "cursor-test",
                "litellm_call_id": "cursor-stream-call",
            },
        },
        selected_account={
            "provider": "openai",
            "account_hash": "cursor-account",
            "account_display": "selected@example.com",
        },
    )
    assert isinstance(response, StreamingResponse)
    flushed = flush_aawm_route_rollups(force=True)
    assert len(flushed) == 2
    rendered = "\n".join(flushed)
    _assert_cursor_route_context(captured)

    async def collect_events() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
            for line in text.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
        return events

    events = asyncio.run(collect_events())
    assert flush_aawm_route_rollups(force=True) == []
    assert rendered.count("Turns: 1") == 1
    assert "selected@example.com" in rendered
    event_types = [event["type"] for event in events]
    assert event_types == [
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[0]["output_index"] == 0
    assert events[1]["output_index"] == 0
    assert events[1]["arguments"] == '{"cmd":"pwd"}'
    assert events[2]["output_index"] == 0
    assert events[2]["item"]["type"] == "function_call"


def test_cursor_completed_tool_turn_survives_failed_continuation(
    monkeypatch: pytest.MonkeyPatch,
    _route_rollup_state: None,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        anthropic_adapter_calls,
    )

    captured_rollup_kwargs: list[dict[str, Any]] = []
    response_bodies: list[dict[str, Any]] = []
    real_emit = anthropic_adapter_calls._emit_adapted_route_access_log
    real_build_response_body = codex_candidate_calls._cursor_responses_response_body

    def _capture_route_log(**kwargs: Any) -> None:
        real_emit(**kwargs)
        captured_rollup_kwargs.append(kwargs["rollup_kwargs"])

    def _capture_response_body(**kwargs: Any) -> dict[str, Any]:
        response_body = real_build_response_body(**kwargs)
        response_bodies.append(response_body)
        return response_body

    monkeypatch.setattr(
        anthropic_adapter_calls,
        "_emit_adapted_route_access_log",
        _capture_route_log,
    )
    monkeypatch.setattr(
        codex_candidate_calls,
        "_cursor_responses_response_body",
        _capture_response_body,
    )

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            return CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "cursor-item-1",
                        "call_id": "cursor-call-1",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    }
                ]
            )

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    metadata = {
        "codex_auto_agent_alias": "cursor-test",
        "canonical_session_identity": "cursor-session-1",
        "repository": "litellm",
    }
    selected_account = {
        "provider": "openai",
        "account_hash": "cursor-account",
        "account_display": "selected@example.com",
    }
    first = _call(
        {
            "model": "work",
            "input": "run pwd",
            "stream": True,
            "litellm_metadata": {
                **metadata,
                "litellm_call_id": "cursor-first-call",
            },
        },
        selected_account=selected_account,
    )
    assert isinstance(first, StreamingResponse)
    assert response_bodies
    assert len(captured_rollup_kwargs) == 1

    with pytest.raises(CursorConnectError, match="live retained session"):
        _call(
            {
                "model": "work",
                "previous_response_id": response_bodies[0]["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "cursor-call-1",
                        "output": "pwd output",
                    }
                ],
                "litellm_metadata": {
                    **metadata,
                    "litellm_call_id": "cursor-continuation-call",
                },
            },
            selected_account=selected_account,
        )

    assert len(captured_rollup_kwargs) == 2
    first_context = captured_rollup_kwargs[0]["litellm_params"]["metadata"][
        "aawm_route_rollup_context"
    ]
    second_context = captured_rollup_kwargs[1]["litellm_params"]["metadata"][
        "aawm_route_rollup_context"
    ]
    assert first_context["canonical_session_identity"] == "cursor-session-1"
    assert second_context["canonical_session_identity"] == "cursor-session-1"
    assert first_context["litellm_call_id"] == "cursor-first-call"
    assert second_context["litellm_call_id"] == "cursor-continuation-call"

    assert record_aawm_route_rollup_failure(
        captured_rollup_kwargs[1],
        message="Cursor Agent continuation state is unavailable.",
        status="Failed",
    )

    async def consume_stream() -> None:
        async for _chunk in first.body_iterator:
            pass

    asyncio.run(consume_stream())
    flushed = flush_aawm_route_rollups(force=True)
    rendered = "\n".join(flushed)
    assert rendered.count("Turns: 1") == 1
    assert rendered.count("Turns: 0") == 1
    assert "Cursor Agent continuation state is unavailable." in rendered
    assert response_bodies[0]["id"] not in rendered


def test_cursor_non_stream_records_route_rollup_once(
    monkeypatch: pytest.MonkeyPatch,
    _route_rollup_state: None,
) -> None:
    captured = _capture_real_route_log(monkeypatch)

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            return CursorAgentRunResult(text="done", turn_ended=True)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    response = _call(
        {
            "model": "work",
            "input": "finish",
            "litellm_metadata": {
                "codex_auto_agent_alias": "cursor-test",
                "litellm_call_id": "cursor-non-stream-call",
            },
        },
        selected_account={
            "provider": "openai",
            "account_hash": "cursor-account",
            "account_display": "selected@example.com",
        },
    )

    assert response.status_code == 200
    _assert_cursor_route_context(captured)
    flushed = flush_aawm_route_rollups(force=True)
    assert len(flushed) == 2
    rendered = "\n".join(flushed)
    assert rendered.count("Turns: 1") == 1
    assert "selected@example.com" in rendered


def test_cursor_text_requires_turn_ended_and_tool_boundary_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursorClient:
        results = [
            CursorAgentRunResult(text="partial"),
            CursorAgentRunResult(
                tool_calls=[
                    {
                        "call_id": "call-1",
                        "name": "exec_command",
                        "arguments": "{}",
                    }
                ]
            ),
        ]

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            return self.results.pop(0)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    with pytest.raises(CursorConnectProtocolError, match="turnEnded"):
        _call({"model": "work", "input": "answer"})

    response = _call({"model": "work", "input": "run"})
    assert json.loads(response.body)["output"][0]["type"] == "function_call"


@pytest.mark.parametrize("omitted_key", ["call_id", "name"])
def test_cursor_malformed_returned_tool_call_is_post_egress_attempt(
    monkeypatch: pytest.MonkeyPatch,
    omitted_key: str,
) -> None:
    from litellm.proxy._types import ProxyException

    ran = False
    tool_call = {
        "id": "cursor-tool-call",
        "call_id": "call-1",
        "name": "exec_command",
        "arguments": "{}",
    }
    tool_call.pop(omitted_key)

    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            nonlocal ran
            ran = True
            return CursorAgentRunResult(tool_calls=[tool_call])

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    with pytest.raises(ProxyException) as exc_info:
        _call({"model": "work", "input": "run"})

    exc = exc_info.value
    assert ran is True
    assert exc.status_code == 502
    assert exc.detail["error"]["code"] == "upstream_transient_internal"
    assert exc.failure_phase == "candidate_post_egress_normalization"
    assert exc.attempted_provider_call is True
    assert codex_candidate_calls._CURSOR_REPLAY_REGISTRY == {}


def test_cursor_route_family_is_codex_only() -> None:
    with pytest.raises(ValueError, match="codex_cursor_agent_aiserver_adapter"):
        _call(
            {"model": "work", "input": "answer"},
            candidate=_candidate(
                route_family="anthropic_cursor_agent_aiserver_adapter"
            ),
        )


def test_cursor_candidate_api_base_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursorClient:
        urls: list[str] = []

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **kwargs: Any,
        ) -> CursorAgentRunResult:
            self.urls.append(kwargs["url"])
            return CursorAgentRunResult(text="done", turn_ended=True)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )
    _call(
        {"model": "work", "input": "answer"},
        candidate=_candidate(api_base="https://cursor.example"),
        target_url="https://chatgpt.com/backend-api/codex/responses",
    )

    assert FakeCursorClient.urls == [
        "https://cursor.example/agent.v1.AgentService/Run"
    ]


def test_cursor_codex_instructions_avoid_unsupported_system_prompt_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursorClient:
        payloads: list[dict[str, Any]] = []

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            self.payloads.append(payload)
            return CursorAgentRunResult(text="done", turn_ended=True)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )

    _call(
        {
            "model": "work",
            "instructions": "Preserve this Codex agent instruction.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Preserve this developer instruction.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Run the requested task.",
                },
            ],
        }
    )

    run_request = FakeCursorClient.payloads[0]["runRequest"]
    dumped = json.dumps(run_request)
    assert "customSystemPrompt" not in run_request
    assert "--system-prompt" not in dumped
    history = run_request["action"]["userMessageAction"]["conversationHistory"]
    instruction_text = history["messages"][0]["user"]["content"][0]["text"]["text"]
    assert "Preserve this Codex agent instruction." in instruction_text
    assert "Preserve this developer instruction." in instruction_text
    assert (
        run_request["action"]["userMessageAction"]["userMessage"]["text"]
        == "Run the requested task."
    )


def test_installed_cursor_candidate_closure_resolves_cursor_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy._types import ProxyException
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints

    async def fail_cursor_request(**_kwargs: Any) -> Response:
        raise CursorConnectError("expected installed Cursor failure")

    installed_dispatch = (
        llm_passthrough_endpoints._perform_codex_auto_agent_alias_candidate_request
    )
    assert installed_dispatch.__globals__ is llm_passthrough_endpoints.__dict__
    assert callable(
        installed_dispatch.__globals__[
            "_perform_codex_auto_agent_cursor_agent_request"
        ]
    )
    assert callable(
        installed_dispatch.__globals__["_raise_cursor_agent_alias_error"]
    )
    monkeypatch.setattr(
        llm_passthrough_endpoints,
        "_perform_codex_auto_agent_cursor_agent_request",
        fail_cursor_request,
    )

    with pytest.raises(ProxyException) as exc_info:
        asyncio.run(
            installed_dispatch(
                endpoint="/v1/responses",
                request=_request(),
                fastapi_response=Response(),
                user_api_key_dict=None,
                candidate={
                    **_candidate(),
                    "provider": "cursor_agent",
                },
                candidate_body={
                    "model": "work",
                    "input": "run pwd",
                },
                target_url="https://chatgpt.com/backend-api/codex/responses",
                api_key="bogus-openai-bearer",
                forward_headers=False,
            )
        )
    assert "expected installed Cursor failure" in exc_info.value.message


@pytest.mark.parametrize(
    ("status_code", "error_type", "error_code"),
    [
        (408, "upstream_timeout", "upstream_timeout"),
        (500, "upstream_error", "upstream_transient_internal"),
        (502, "upstream_error", "upstream_transient_internal"),
        (503, "upstream_error", "upstream_transient_internal"),
        (504, "upstream_timeout", "upstream_timeout"),
        (529, "upstream_error", "upstream_transient_internal"),
    ],
)
def test_cursor_transient_statuses_preserve_retry_classification(
    status_code: int,
    error_type: str,
    error_code: str,
) -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectError(
                "Cursor upstream failure",
                status_code=status_code,
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.code == str(status_code)
    assert exc.status_code == status_code
    assert exc.type == error_type
    assert exc.detail["error"]["type"] == error_type
    assert exc.detail["error"]["code"] == error_code
    assert (
        exc.detail["error"]["code"]
        != "aawm_codex_auto_agent_candidate_unavailable"
    )
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == error_code
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "request_local"
    )


def test_cursor_transport_failure_maps_to_request_local_transient() -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=OSError("connection reset"),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == 502
    assert exc.detail["error"]["code"] == "upstream_transient_internal"
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "upstream_transient_internal"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "request_local"
    )


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_cursor_local_4xx_is_terminal_without_candidate_unavailable(
    status_code: int,
) -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectError(
                "Cursor non-transient failure",
                status_code=status_code,
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == status_code
    assert exc.detail["error"]["code"] == "provider_terminal_error"
    assert exc.attempted_provider_call is False
    assert getattr(exc, "_aawm_provider_returned", False) is False
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
            attempted_provider_call=exc.attempted_provider_call,
        )
    )
    assert classified is None


def test_cursor_local_4xx_preserves_sanitized_error_without_attribution() -> None:
    from litellm.proxy._types import ProxyException

    local_body = {
        "error": {
            "code": "local_proxy_route_not_found",
            "message": "Local proxy route was not found.",
            "type": "invalid_request_error",
            "status": 404,
            "secret": "do-not-leak",
        },
        "request_id": "local-request-secret",
    }
    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectError(
                "Local Cursor proxy returned HTTP 404.",
                status_code=404,
                body=json.dumps(local_body).encode("utf-8"),
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    sanitized_body = {
        "error": {
            "code": "local_proxy_route_not_found",
            "message": "Local proxy route was not found.",
            "status": 404,
            "type": "invalid_request_error",
        }
    }
    assert exc.status_code == 404
    assert exc.attempted_provider_call is False
    assert getattr(exc, "_aawm_provider_returned", False) is False
    assert exc.body == sanitized_body
    assert exc.detail["error"] == sanitized_body["error"]
    assert exc.detail["cursor_sanitized_provider_error"] == sanitized_body
    assert (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
            attempted_provider_call=exc.attempted_provider_call,
        )
        is None
    )
    serialized = json.dumps(exc.detail)
    assert "do-not-leak" not in serialized
    assert "local-request-secret" not in serialized


@pytest.mark.asyncio
async def test_cursor_local_4xx_stops_alias_without_cooldown_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from litellm.proxy._types import ProxyException
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    cursor_candidate = _candidate(provider="cursor_agent")
    fallback_candidate = {
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "route_family": "codex_responses",
    }
    selections = [
        {
            "candidate": cursor_candidate,
            "alias_model": "work",
            "lane_key": "cursor-agent",
            "cooldown_key": "cursor-agent:cursor-grok-4.6-high",
            "selection_reason": "first_choice",
            "skipped": [],
        },
        {
            "candidate": fallback_candidate,
            "alias_model": "work",
            "lane_key": "openai",
            "cooldown_key": "openai:gpt-5.3-codex",
            "selection_reason": "fallback",
            "skipped": [],
        },
    ]
    selection_calls: list[str] = []
    provider_calls: list[str] = []
    memory_publications: list[tuple[str, ...]] = []
    durable_publications: list[tuple[str, ...]] = []
    request_local_cooldowns: list[str] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        selection = selections[len(selection_calls)]
        selection_calls.append(selection["candidate"]["provider"])
        return selection

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        del candidate_body
        provider_calls.append(candidate["provider"])
        if candidate["provider"] == "cursor_agent":
            codex_candidate_calls._raise_cursor_agent_alias_error(
                exc=CursorConnectError(
                    "Local Cursor proxy returned HTTP 404.",
                    status_code=404,
                    body={
                        "error": {
                            "code": "local_proxy_route_not_found",
                            "message": "Local proxy route was not found.",
                        }
                    },
                ),
                candidate=candidate,
            )
        return Response(content="fallback", status_code=200)

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: Any) -> None:
            return None

    async def _ensure_session_owner_guard(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_ensure_session_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )
    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=(
            llm_passthrough_endpoints._resolve_auto_agent_cooldown_publication_plan
        ),
        publish_cooldown_memory_fn=lambda *, keys, **_kwargs: (
            memory_publications.append(tuple(keys))
        ),
        persist_cooldown_fn=lambda *, keys, **_kwargs: (
            durable_publications.append(tuple(keys))
        ),
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=(
            llm_passthrough_endpoints._add_codex_auto_agent_alias_metadata
        ),
        raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("local Cursor 4xx must not redispatch")
        ),
    )

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        llm_passthrough_endpoints,
        "_record_codex_failure_evidence",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        llm_passthrough_endpoints,
        "_apply_request_local_cooldown_from_plan",
        lambda _request, *, candidate, **_kwargs: (
            request_local_cooldowns.append(candidate["provider"])
        ),
    )

    with pytest.raises(ProxyException) as exc_info:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="work",
            request=_request(),
            prepared_request_body={
                "model": "work",
                "input": "hello",
                "stream": False,
            },
            max_candidate_attempts=2,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["code"] == "local_proxy_route_not_found"
    assert selection_calls == ["cursor_agent"]
    assert provider_calls == ["cursor_agent"]
    assert memory_publications == []
    assert durable_publications == []
    assert request_local_cooldowns == []


def test_cursor_provider_http_error_preserves_sanitized_attribution() -> None:
    from litellm.proxy._types import ProxyException

    provider_body = {
        "error": {
            "code": "model_not_found",
            "message": "Unknown model: cursor-grok-4.6-high",
            "type": "invalid_request_error",
            "model": "cursor-grok-4.6-high",
            "secret": "do-not-leak",
        },
        "request_id": "provider-request-secret",
    }
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectError(
                "Cursor Agent Connect request failed with HTTP 404.",
                status_code=404,
                body=json.dumps(provider_body).encode("utf-8"),
                provider_returned=True,
            ),
            candidate=_candidate(provider="cursor_agent"),
        )

    exc = exc_info.value
    sanitized_body = {
        "error": {
            "code": "model_not_found",
            "message": "Unknown model: cursor-grok-4.6-high",
            "model": "cursor-grok-4.6-high",
            "type": "invalid_request_error",
        }
    }
    assert exc.status_code == 404
    assert exc.attempted_provider_call is True
    assert exc._aawm_provider_returned is True
    assert exc.body == sanitized_body
    assert exc.detail["error"]["code"] == "model_not_found"
    assert exc.detail["error"]["message"] == (
        "Unknown model: cursor-grok-4.6-high"
    )
    assert exc.detail["cursor_sanitized_provider_error"] == sanitized_body
    assert (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=_candidate(provider="cursor_agent"),
            attempted_provider_call=exc.attempted_provider_call,
        )
        == "candidate_unavailable"
    )
    serialized = json.dumps(exc.detail)
    assert "do-not-leak" not in serialized
    assert "provider-request-secret" not in serialized


@pytest.mark.parametrize(
    ("status_code", "body", "expected_code", "expected_type"),
    [
        (
            400,
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Unsupported request parameter.",
                }
            },
            "invalid_request",
            "upstream_error",
        ),
        (404, b"", "provider_terminal_error", "upstream_error"),
        (
            404,
            {"error": {"code": "not_found", "message": "Resource not found."}},
            "not_found",
            "upstream_error",
        ),
        (401, b"", "provider_terminal_error", "authentication_error"),
        (403, b"", "provider_terminal_error", "authentication_error"),
    ],
    ids=[
        "generic-400",
        "header-only-404",
        "generic-404",
        "401-auth",
        "403-auth",
    ],
)
def test_cursor_provider_unmatched_4xx_is_terminal_without_candidate_cooldown(
    status_code: int,
    body: Any,
    expected_code: str,
    expected_type: str,
) -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectError(
                f"Cursor provider returned HTTP {status_code}.",
                status_code=status_code,
                body=body,
                provider_returned=True,
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == status_code
    assert exc.type == expected_type
    assert exc.attempted_provider_call is True
    assert exc._aawm_provider_returned is True
    assert exc.detail["error"]["code"] == expected_code
    assert (
        exc.detail["error"]["code"]
        != "aawm_codex_auto_agent_candidate_unavailable"
    )
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "provider_terminal_error"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "request_local"
    )


def test_cursor_preflight_conversion_value_error_maps_to_ineligible() -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=ValueError("Binary or non-text content is not supported."),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.code == "400"
    assert exc.type == "invalid_request_error"
    assert exc.candidate_status == "ineligible"
    assert exc.ineligibility_reason == "unsupported"
    assert exc.failure_phase == "candidate_preflight"
    assert exc.attempted_provider_call is False
    assert (
        exc.detail["error"]["code"]
        == "aawm_codex_auto_agent_candidate_ineligible"
    )
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "candidate_deterministically_ineligible"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "none"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Cursor Agent requested unsupported external exec field 9.",
        "Cursor Agent requested unsupported local exec operation field 4.",
        "Cursor Agent requested an unsupported interactive client response.",
    ],
    ids=["external-exec", "local-exec", "interactive-client"],
)
def test_cursor_unsupported_operation_protocol_error_maps_to_ineligible(
    message: str,
) -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectProtocolError(message),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.code == "400"
    assert exc.type == "invalid_request_error"
    assert exc.candidate_status == "ineligible"
    assert exc.ineligibility_reason == "unsupported"
    assert exc.failure_phase == "candidate_preflight"
    assert exc.attempted_provider_call is True
    assert (
        exc.detail["error"]["code"]
        == "aawm_codex_auto_agent_candidate_ineligible"
    )
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "candidate_deterministically_ineligible"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "none"
    )


def test_cursor_protocol_structure_maps_without_raw_values() -> None:
    from litellm.proxy._types import ProxyException

    raw_command = "secret-command"
    raw_workspace = "/secret/workspace"
    raw_bytes = b"opaque-provider-bytes"
    structure = {
        "fields": [
            {
                "field_number": 1,
                "wire_type": 0,
                "payload_length": 1,
                "value": raw_command,
            },
            {
                "field_number": 4,
                "wire_type": 2,
                "payload_length": 12,
                "nested_fields": [
                    {
                        "field_number": 2,
                        "wire_type": 2,
                        "payload_length": 16,
                        "value": raw_workspace,
                    }
                ],
            },
        ],
        "raw": raw_bytes,
    }
    expected_structure = {
        "fields": [
            {
                "field_number": 1,
                "wire_type": 0,
                "payload_length": 1,
            },
            {
                "field_number": 4,
                "wire_type": 2,
                "payload_length": 12,
                "nested_fields": [
                    {
                        "field_number": 2,
                        "wire_type": 2,
                        "payload_length": 16,
                    }
                ],
            },
        ]
    }
    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectProtocolError(
                "Cursor Agent requested unsupported local exec operation field 4.",
                body=structure,
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    field_name = codex_candidate_calls._CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD
    assert exc.status_code == 400
    assert exc.detail["error"]["code"] == (
        "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert exc.detail[field_name] == expected_structure
    assert getattr(exc, field_name) == expected_structure
    assert exc.detail[field_name] is not structure
    serialized = json.dumps(exc.detail)
    assert raw_command not in serialized
    assert raw_workspace not in serialized
    assert "opaque-provider-bytes" not in serialized
    assert '"value"' not in serialized
    assert '"raw"' not in serialized


def test_cursor_unrelated_protocol_errors_keep_transient_502_semantics() -> None:
    from litellm.proxy._types import ProxyException

    candidate = _candidate(provider="cursor_agent")
    with pytest.raises(ProxyException) as exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectProtocolError(
                "Cursor Connect response ended with an incomplete frame."
            ),
            candidate=candidate,
        )

    exc = exc_info.value
    assert exc.status_code == 502
    assert exc.detail["error"]["code"] == "upstream_transient_internal"
    assert getattr(exc, "candidate_status", None) != "ineligible"
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "upstream_transient_internal"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "request_local"
    )


@pytest.mark.parametrize(
    "failure",
    [
        CursorConnectError("timeout", status_code=408),
        CursorConnectError("internal", status_code=500),
        CursorConnectError("bad gateway", status_code=502),
        CursorConnectError("unavailable", status_code=503),
        CursorConnectError("gateway timeout", status_code=504),
        CursorConnectError("overloaded", status_code=529),
        OSError("transport reset"),
    ],
    ids=["408", "500", "502", "503", "504", "529", "transport"],
)
def test_cursor_transient_failures_preserve_replay_state(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            raise failure

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )
    _store_replay_state()

    with pytest.raises(type(failure)):
        _call(
            {
                "model": "work",
                "previous_response_id": "resp-replay",
                "input": "continue",
            }
        )

    state = codex_candidate_calls._peek_cursor_replay_state("resp-replay")
    assert state["messages"]


def test_cursor_success_consumes_replay_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            return CursorAgentRunResult(text="done", turn_ended=True)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )
    _store_replay_state()

    response = _call(
        {
            "model": "work",
            "previous_response_id": "resp-replay",
            "input": "continue",
        }
    )

    assert response.status_code == 200
    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._peek_cursor_replay_state("resp-replay")


@pytest.mark.parametrize("status_code", [400, 401, 403, 429])
def test_cursor_terminal_failures_consume_replay_state(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    class FakeCursorClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(
            self,
            _payload: dict[str, Any],
            **_kwargs: Any,
        ) -> CursorAgentRunResult:
            raise CursorConnectError("terminal", status_code=status_code)

    monkeypatch.setattr(
        "litellm.llms.cursor_agent.connect.CursorAgentConnectClient",
        FakeCursorClient,
    )
    _store_replay_state()

    with pytest.raises(CursorConnectError):
        _call(_replay_body())

    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._peek_cursor_replay_state("resp-replay")


# =============================================================================
# CURSOR-015: Reject nested cursor continuation identifiers
# =============================================================================

def test_cursor_fresh_replay_dispatch_rejects_nested_continuation_identifiers() -> None:
    """R7: Known Cursor continuation identifiers in replayed input must be
    rejected before asserting portability."""
    body = _stock_codex_full_history_body()
    body["input"][0]["previous_response_id"] = "resp_nested_123"
    continuation_exc = _cursor_continuation_failure()

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
        is None
    )


def test_cursor_fresh_replay_dispatch_rejects_camel_continuation_identifier() -> None:
    """R7: CamelCase variants of Cursor continuation identifiers are also rejected."""
    body = _stock_codex_full_history_body()
    body["input"][2]["conversationId"] = "conv_nested_456"
    continuation_exc = _cursor_continuation_failure()

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
        is None
    )


def test_cursor_fresh_replay_dispatch_rejects_agent_session_id() -> None:
    """R7: agent_session_id in input items is rejected."""
    body = _stock_codex_full_history_body()
    body["input"][1]["agent_session_id"] = "agent_sess_789"
    continuation_exc = _cursor_continuation_failure()

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
        is None
    )


# =============================================================================
# CURSOR-015: Reject unresolved call graph
# =============================================================================

def test_cursor_fresh_replay_dispatch_rejects_unresolved_call_graph() -> None:
    """R7: A partially resolved call graph must be rejected before
    asserting portability."""
    codex_candidate_calls._store_cursor_replay_state(
        "resp-unresolved-call",
        messages=[
            {"role": "user", "content": "run two commands"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}',
                        },
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"date"}',
                        },
                    },
                ],
            },
        ],
        tools=[],
    )
    body = {
        "model": "work",
        "previous_response_id": "resp-unresolved-call",
        "input": [
            {
                "type": "function_call_output",
                "id": "fco_01a06244-9f7f-7fe1-869b-23d587ad56f1",
                "call_id": "call-1",
                "output": "pwd output",
            },
        ],
    }
    continuation_exc = _cursor_continuation_failure()

    assert (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            body,
            continuation_exc=continuation_exc,
        )
        is None
    )


# =============================================================================
# CURSOR-015: Non-streaming 2xx non-JSON rejection
# =============================================================================

def test_cursor_015_non_streaming_non_json_2xx_raises_502() -> None:
    """R5: Non-streaming 2xx response that is not valid JSON must raise
    HTTP 502, not be silently promoted."""
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        payload_validation,
    )

    response = Response(
        content=b"not json",
        status_code=200,
        media_type="application/json",
    )

    try:
        asyncio.run(
            payload_validation._validate_codex_auto_agent_responses_payload(
                response=response,
                adapter="xai",
                adapter_model="xai/grok-4.6",
                adapter_label="native-xai",
            )
        )
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 502
        assert "non-JSON" in exc.detail["error"]["message"]
