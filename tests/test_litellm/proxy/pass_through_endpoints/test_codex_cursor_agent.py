from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.responses import Response, StreamingResponse
from starlette.requests import Request

from litellm.llms.cursor_agent.common_utils import run_url
from litellm.llms.cursor_agent.connect import (
    CursorAgentRunResult,
    CursorConnectError,
    CursorConnectProtocolError,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints


def _request() -> Request:
    return Request(
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
) -> Response:
    return asyncio.run(
        codex_candidate_calls._perform_codex_auto_agent_cursor_agent_request(
            endpoint="/v1/responses",
            request=_request(),
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
    continuation_cue = (
        "Finish the original user request using the completed tool result above. "
        "Do not repeat completed tool calls."
    )
    assert codex_candidate_calls._responses_input_to_cursor_messages(
        {"input": "run pwd"}
    ) == [{"role": "user", "content": "run pwd"}]

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
                ]
            ),
            CursorAgentRunResult(
                tool_calls=[
                    {
                        "id": "cursor-item-2",
                        "call_id": "call-2",
                        "name": "exec_command",
                        "arguments": '{"cmd":"date"}',
                    }
                ]
            ),
            CursorAgentRunResult(text="final answer", turn_ended=True),
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
    assert first_run["requestedModel"] == {
        "modelId": "grok-4.6",
        "parameters": [
            {"id": "effort", "value": "high"},
            {"id": "fast", "value": "false"},
        ],
    }
    assert first_run["mcpTools"]["mcpTools"][0]["name"] == "exec_command"
    assert first_run["excludeWorkspaceContext"] is True
    assert "rootPromptMessagesJson" not in json.dumps(first_run)

    second_run = FakeCursorClient.calls[1]["payload"]["runRequest"]
    history = second_run["action"]["userMessageAction"]["conversationHistory"]
    assert history == {
        "messages": [
            {"user": {"content": [{"text": {"text": "run pwd"}}]}},
            {
                "assistant": {
                    "content": [
                        {
                            "toolCall": {
                                "toolCallId": (
                                    "call-3e993636-853f-474a-b20c-439984662b5d-0"
                                ),
                                "toolName": "exec_command",
                                "argsJson": '{"cmd":"pwd"}',
                            }
                        }
                    ]
                }
            },
            {
                "tool": {
                    "toolCallId": (
                        "call-3e993636-853f-474a-b20c-439984662b5d-0"
                    ),
                    "toolName": "exec_command",
                    "content": [{"text": {"text": "pwd output"}}],
                }
            },
        ]
    }
    assert "rootPromptMessagesJson" not in json.dumps(second_run)
    assert (
        second_run["action"]["userMessageAction"]["userMessage"]["text"]
        == continuation_cue
    )
    assert second_run["mcpTools"]["mcpTools"][0]["name"] == "exec_command"
    assert (
        history["messages"][2]["tool"]["toolCallId"]
        == first_body["output"][0]["call_id"]
    )
    assert (
        history["messages"][2]["tool"]["toolCallId"]
        != first_body["output"][0]["id"]
    )

    third_run = FakeCursorClient.calls[2]["payload"]["runRequest"]
    assert third_run["action"]["userMessageAction"]["conversationHistory"] == {
        "messages": [
            {"user": {"content": [{"text": {"text": "run pwd"}}]}},
            {
                "assistant": {
                    "content": [
                        {
                            "toolCall": {
                                "toolCallId": (
                                    "call-3e993636-853f-474a-b20c-439984662b5d-0"
                                ),
                                "toolName": "exec_command",
                                "argsJson": '{"cmd":"pwd"}',
                            }
                        }
                    ]
                }
            },
            {
                "tool": {
                    "toolCallId": (
                        "call-3e993636-853f-474a-b20c-439984662b5d-0"
                    ),
                    "toolName": "exec_command",
                    "content": [{"text": {"text": "pwd output"}}],
                }
            },
            {
                "assistant": {
                    "content": [
                        {
                            "toolCall": {
                                "toolCallId": "call-2",
                                "toolName": "exec_command",
                                "argsJson": '{"cmd":"date"}',
                            }
                        }
                    ]
                }
            },
            {
                "tool": {
                    "toolCallId": "call-2",
                    "toolName": "exec_command",
                    "content": [{"text": {"text": "date output"}}],
                }
            },
        ]
    }
    assert (
        third_run["action"]["userMessageAction"]["userMessage"]["text"]
        == continuation_cue
    )
    assert third_run["mcpTools"]["mcpTools"][0]["name"] == "exec_command"
    message_ids = [
        run["action"]["userMessageAction"]["userMessage"]["messageId"]
        for run in (first_run, second_run, third_run)
    ]
    assert len(set(message_ids)) == 3
    for field in ("conversationId", "conversationGroupId", "runId"):
        assert len({first_run[field], second_run[field], third_run[field]}) == 3
    for run in (first_run, second_run, third_run):
        assert run["conversationGroupId"] == run["conversationId"]


def test_cursor_local_exec_function_call_replays_through_existing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                ]
            ),
            CursorAgentRunResult(text="command completed", turn_ended=True),
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
    )["retained_session"] is None

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
    assert len(FakeCursorClient.calls) == 2


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


def test_cursor_retained_session_consumes_function_call_output_without_new_run() -> None:
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
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "read-call",
                    "output": "LITELLM_CURSOR_READ_V1:envelope",
                }
            ],
        }
    )

    assert json.loads(response.body)["output_text"] == "read complete"
    assert session.outputs == [
        [("read-call", "LITELLM_CURSOR_READ_V1:envelope")]
    ]
    assert session.closed is True
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
) -> None:
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
    response = _call({"model": "work", "input": "run pwd", "stream": True})
    assert isinstance(response, StreamingResponse)

    async def collect_events() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
            for line in text.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
        return events

    events = asyncio.run(collect_events())
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


@pytest.mark.parametrize("status_code", [400, 401, 403, 429])
def test_cursor_auth_and_non_transient_fail_closed_to_durable_unavailable(
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
    assert (
        exc.detail["error"]["code"]
        == "aawm_codex_auto_agent_candidate_unavailable"
    )
    classified = (
        llm_passthrough_endpoints._classify_codex_auto_agent_retryable_exhaustion(
            exc,
            candidate=candidate,
        )
    )
    assert classified == "candidate_unavailable"
    assert (
        llm_passthrough_endpoints._get_codex_auto_agent_candidate_cooldown_scope(
            classified,
            candidate=candidate,
        )
        == "candidate"
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
        _call(_replay_body())

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

    response = _call(_replay_body())

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
