"""
Focused tests for RR-024: convert Anthropic bash_code_execution_tool_result
blocks into provider_specific_fields["code_interpreter_results"] on the
non-streaming Anthropic chat transformation path.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import httpx
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.utils import ModelResponse


def _bash_tool_use(
    call_id: str = "srvtoolu_01ABC",
    command: str = "python3 << 'EOF'\nprint(2 + 2)\nEOF\n",
) -> Dict[str, Any]:
    return {
        "type": "server_tool_use",
        "id": call_id,
        "name": "bash_code_execution",
        "input": {"command": command},
    }


def _bash_tool_result(
    call_id: str = "srvtoolu_01ABC",
    stdout: str = "4\n",
    stderr: str = "",
    return_code: int = 0,
) -> Dict[str, Any]:
    return {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": call_id,
        "content": {
            "type": "bash_code_execution_result",
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
        },
    }


def _editor_tool_result(call_id: str = "srvtoolu_01DEF") -> Dict[str, Any]:
    return {
        "type": "text_editor_code_execution_tool_result",
        "tool_use_id": call_id,
        "content": {
            "type": "text_editor_code_execution_result",
            "is_file_update": False,
        },
    }


def _anthropic_completion(
    content: List[Dict[str, Any]],
    *,
    container: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "id": "msg_rr024",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": content,
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if container is not None:
        response["container"] = container
    return response


def _raw_response(completion_response: Dict[str, Any]) -> MagicMock:
    mock_raw = MagicMock(spec=httpx.Response)
    mock_raw.json.return_value = completion_response
    mock_raw.status_code = 200
    mock_raw.headers = {}
    return mock_raw


def _transform(completion_response: Dict[str, Any]) -> ModelResponse:
    config = AnthropicConfig()
    return config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=_raw_response(completion_response),
        model_response=ModelResponse(),
        json_mode=False,
        prefix_prompt=None,
    )


class TestRR024BashCodeInterpreterResults:
    def test_bash_result_builds_code_interpreter_results_and_hidden_params(self):
        command = "echo hello"
        completion = _anthropic_completion(
            [
                {"type": "text", "text": "running"},
                _bash_tool_use("srvtoolu_A", command),
                _bash_tool_result("srvtoolu_A", stdout="hello\n", stderr=""),
                {"type": "text", "text": "done"},
            ],
            container={"id": "container_123"},
        )

        transformed = _transform(completion)
        psf = transformed.choices[0].message.provider_specific_fields
        assert psf is not None
        assert "tool_results" in psf
        assert len(psf["tool_results"]) == 1
        assert "code_interpreter_results" in psf

        results = psf["code_interpreter_results"]
        assert len(results) == 1
        item = results[0]
        assert isinstance(item, ResponseCodeInterpreterToolCall)
        assert item.type == "code_interpreter_call"
        assert item.id == "srvtoolu_A"
        assert item.code == command
        assert item.container_id == "container_123"
        assert item.status == "completed"
        assert item.outputs is not None
        assert len(item.outputs) == 1
        assert item.outputs[0].type == "logs"
        assert item.outputs[0].logs == "hello\n"

        # Streaming aggregate / Responses consumers also read hidden_params.
        hidden = transformed._hidden_params
        assert isinstance(hidden, dict)
        assert "provider_specific_fields" in hidden
        assert hidden["provider_specific_fields"]["code_interpreter_results"] is results

    def test_non_bash_tool_results_are_skipped_for_code_interpreter(self):
        completion = _anthropic_completion(
            [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_bash",
                    "name": "bash_code_execution",
                    "input": {"command": "printf ok"},
                },
                _bash_tool_result("srvtoolu_bash", stdout="ok", stderr=""),
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_editor",
                    "name": "text_editor_code_execution",
                    "input": {
                        "command": "create",
                        "path": "test.txt",
                        "file_text": "Hello",
                    },
                },
                _editor_tool_result("srvtoolu_editor"),
            ]
        )

        transformed = _transform(completion)
        psf = transformed.choices[0].message.provider_specific_fields
        assert psf is not None
        assert len(psf["tool_results"]) == 2

        results = psf["code_interpreter_results"]
        assert len(results) == 1
        assert results[0].id == "srvtoolu_bash"
        assert results[0].code == "printf ok"
        # Missing container maps to empty string for ResponseCodeInterpreterToolCall.
        assert results[0].container_id == ""

    def test_stderr_is_included_in_logs_and_empty_content_yields_none_outputs(self):
        config = AnthropicConfig()

        logs = config._build_code_interpreter_log_outputs(
            {"stdout": "out\n", "stderr": "warn\n"}
        )
        assert logs == [{"type": "logs", "logs": "out\nSTDERR: warn\n"}]

        assert config._build_code_interpreter_log_outputs({"stdout": "", "stderr": ""}) is None
        assert config._build_code_interpreter_log_outputs("not-a-dict") is None
        assert config._build_code_interpreter_log_outputs(None) is None

        # Direct helper: only bash_code_execution_tool_result converts.
        code_by_id = {"srv_ok": "echo ok", "srv_editor": "create"}
        results = config._build_code_interpreter_results(
            tool_results=[
                _bash_tool_result("srv_ok", stdout="ok\n"),
                _editor_tool_result("srv_editor"),
                {"type": "bash_code_execution_tool_result", "tool_use_id": "", "content": {}},
                "bad-entry",
                {
                    "type": "bash_code_execution_tool_result",
                    "tool_use_id": "srv_empty",
                    "content": {"stdout": "", "stderr": ""},
                },
            ],
            code_by_id=code_by_id,
            container_id="ctr_1",
        )
        assert [r.id for r in results] == ["srv_ok", "srv_empty"]
        assert results[0].outputs[0].logs == "ok\n"
        assert results[0].container_id == "ctr_1"
        assert results[1].code == ""
        assert results[1].outputs is None

    def test_code_by_id_map_uses_command_from_tool_call_arguments(self):
        config = AnthropicConfig()
        tool_calls = [
            {
                "id": "srv_1",
                "type": "function",
                "function": {
                    "name": "bash_code_execution",
                    "arguments": '{"command":"ls -la"}',
                },
                "index": 0,
            },
            {
                "id": "srv_bad",
                "type": "function",
                "function": {"name": "bash_code_execution", "arguments": "{not-json"},
                "index": 1,
            },
            {
                "id": None,
                "type": "function",
                "function": {"name": "bash_code_execution", "arguments": '{"command":"x"}'},
                "index": 2,
            },
        ]
        mapping = config._build_code_by_id_map(tool_calls)  # type: ignore[arg-type]
        assert mapping == {"srv_1": "ls -la"}

    def test_no_tool_results_omits_code_interpreter_results_key(self):
        completion = _anthropic_completion(
            [{"type": "text", "text": "hello only"}]
        )
        transformed = _transform(completion)
        psf = transformed.choices[0].message.provider_specific_fields
        assert psf is not None
        assert "tool_results" not in psf or psf.get("tool_results") is None
        assert "code_interpreter_results" not in psf

    def test_multiple_bash_results_preserve_order_and_commands(self):
        completion = _anthropic_completion(
            [
                _bash_tool_use("srv_A", "echo first"),
                _bash_tool_result("srv_A", stdout="first\n"),
                _bash_tool_use("srv_B", "echo second"),
                _bash_tool_result("srv_B", stdout="second\n", stderr="e2\n"),
            ],
            container={"id": "ctr_multi"},
        )
        transformed = _transform(completion)
        results = transformed.choices[0].message.provider_specific_fields[
            "code_interpreter_results"
        ]
        assert [r.id for r in results] == ["srv_A", "srv_B"]
        assert results[0].code == "echo first"
        assert results[1].code == "echo second"
        assert results[0].outputs[0].logs == "first\n"
        assert results[1].outputs[0].logs == "second\nSTDERR: e2\n"
        assert all(r.container_id == "ctr_multi" for r in results)
