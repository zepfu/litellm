"""Focused tests for RR-023: streaming bash_code_execution → code_interpreter_results.

Covers the handler-owned streaming conversion in ModelResponseIterator, plus
restored tool_results initialization (no lazy hasattr guard).
"""

from typing import Any, Dict, List, Optional

from litellm.llms.anthropic.chat.handler import ModelResponseIterator


def _psf(parsed) -> Optional[Dict[str, Any]]:
    if parsed.choices and parsed.choices[0].delta:
        return getattr(parsed.choices[0].delta, "provider_specific_fields", None)
    return None


def _emit_code_results(iterator: ModelResponseIterator, chunks: List[dict]) -> List[Any]:
    emissions: List[Any] = []
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        psf = _psf(parsed)
        if psf and "code_interpreter_results" in psf:
            emissions.append(psf["code_interpreter_results"])
    return emissions


def test_tool_results_initialized_on_iterator():
    """RR-023 Low: tool_results must be initialized in __init__, not via hasattr."""
    iterator = ModelResponseIterator(None, sync_stream=True)
    assert hasattr(iterator, "tool_results")
    assert iterator.tool_results == []
    assert isinstance(iterator._server_tool_inputs, dict)
    assert iterator._current_server_tool_id is None
    assert iterator._container_id is None


def test_streaming_code_execution_produces_code_interpreter_results():
    """bash_code_execution_tool_result streaming emits provider-neutral results."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Running code..."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC",
                "name": "bash_code_execution",
                "input": {"command": "echo hello"},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01ABC",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "hello\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    iterator = ModelResponseIterator(None, sync_stream=True)
    found = False
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        psf = _psf(parsed)
        if psf and "code_interpreter_results" in psf:
            found = True
            results = psf["code_interpreter_results"]
            assert len(results) == 1
            item = results[0]
            # Dict-shaped (Responses normalizer accepts dicts) or attribute access.
            item_type = item["type"] if isinstance(item, dict) else item.type
            item_id = item["id"] if isinstance(item, dict) else item.id
            item_code = item["code"] if isinstance(item, dict) else item.code
            item_outputs = item["outputs"] if isinstance(item, dict) else item.outputs
            assert item_type == "code_interpreter_call"
            assert item_id == "srvtoolu_01ABC"
            assert item_code == "echo hello"
            assert item_outputs is not None
            assert len(item_outputs) == 1
            logs = (
                item_outputs[0]["logs"]
                if isinstance(item_outputs[0], dict)
                else item_outputs[0].logs
            )
            assert logs == "hello\n"
            assert "tool_results" in psf
            assert len(psf["tool_results"]) == 1

    assert found, (
        "code_interpreter_results should appear in provider_specific_fields "
        "when bash_code_execution_tool_result is streamed"
    )


def test_streaming_multiple_code_executions_cumulative_emissions():
    """Each tool_result emission is cumulative (stream_chunk_builder last-wins)."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01AAA",
                "name": "bash_code_execution",
                "input": {"command": "echo first"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01AAA",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "first\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01BBB",
                "name": "bash_code_execution",
                "input": {"command": "echo second"},
            },
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01BBB",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "second\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 3},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    emissions = _emit_code_results(ModelResponseIterator(None, sync_stream=True), chunks)
    assert len(emissions) == 2, f"Expected 2 emissions, got {len(emissions)}"

    def _id(item):
        return item["id"] if isinstance(item, dict) else item.id

    def _code(item):
        return item["code"] if isinstance(item, dict) else item.code

    def _logs(item):
        outputs = item["outputs"] if isinstance(item, dict) else item.outputs
        out0 = outputs[0]
        return out0["logs"] if isinstance(out0, dict) else out0.logs

    assert len(emissions[0]) == 1
    assert _id(emissions[0][0]) == "srvtoolu_01AAA"
    assert _code(emissions[0][0]) == "echo first"
    assert _logs(emissions[0][0]) == "first\n"

    assert len(emissions[1]) == 2
    assert _id(emissions[1][0]) == "srvtoolu_01AAA"
    assert _code(emissions[1][0]) == "echo first"
    assert _logs(emissions[1][0]) == "first\n"
    assert _id(emissions[1][1]) == "srvtoolu_01BBB"
    assert _code(emissions[1][1]) == "echo second"
    assert _logs(emissions[1][1]) == "second\n"


def test_streaming_code_execution_input_assembled_from_deltas():
    """Assemble server_tool_use input from input_json_delta at content_block_stop."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01AAA",
                "name": "code_execution",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"comma'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": 'nd": "echo hello"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01AAA",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "hello\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    emissions = _emit_code_results(ModelResponseIterator(None, sync_stream=True), chunks)
    assert emissions, "No code_interpreter_results emitted"
    code_results = emissions[-1]
    assert len(code_results) == 1
    item = code_results[0]
    item_id = item["id"] if isinstance(item, dict) else item.id
    item_code = item["code"] if isinstance(item, dict) else item.code
    outputs = item["outputs"] if isinstance(item, dict) else item.outputs
    logs = outputs[0]["logs"] if isinstance(outputs[0], dict) else outputs[0].logs
    assert item_id == "srvtoolu_01AAA"
    assert item_code == "echo hello"
    assert logs == "hello\n"


def test_empty_output_produces_null_outputs():
    """Empty stdout/stderr → outputs=None (OpenAI parity)."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01AAA",
                "name": "bash_code_execution",
                "input": {"command": "true"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01AAA",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    emissions = _emit_code_results(ModelResponseIterator(None, sync_stream=True), chunks)
    assert emissions
    item = emissions[-1][0]
    outputs = item["outputs"] if isinstance(item, dict) else item.outputs
    assert outputs is None, f"Expected outputs=None, got {outputs}"


def test_non_bash_tool_result_skipped_for_code_interpreter():
    """text_editor results stay in tool_results but not code_interpreter_results."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01AAA",
                "name": "text_editor",
                "input": {"command": "view", "path": "/tmp/test.py"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01AAA",
                "content": [{"type": "text", "text": "file contents here"}],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    iterator = ModelResponseIterator(None, sync_stream=True)
    code_results = None
    tool_results = None
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        psf = _psf(parsed)
        if not psf:
            continue
        if "code_interpreter_results" in psf:
            code_results = psf["code_interpreter_results"]
        if "tool_results" in psf:
            tool_results = psf["tool_results"]

    assert code_results is not None, "Expected code_interpreter_results key to be emitted"
    assert len(code_results) == 0
    assert tool_results is not None
    assert len(tool_results) == 1
    assert tool_results[0]["type"] == "text_editor_code_execution_tool_result"


def test_message_delta_container_id_reemits_code_interpreter_results():
    """container_id arrives on message_delta and is re-emitted on results."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC",
                "name": "bash_code_execution",
                "input": {"command": "echo hello"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01ABC",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "hello\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "container": {
                    "id": "container_011CW9hA9zpZ8xD3bjjShy4p",
                    "expires_at": "2025-12-16T04:57:16.913181Z",
                },
            },
            "usage": {"output_tokens": 50},
        },
    ]

    emissions = _emit_code_results(ModelResponseIterator(None, sync_stream=True), chunks)
    # One emission on tool_result (container_id None) + re-emit on message_delta
    assert len(emissions) >= 2
    final = emissions[-1]
    assert len(final) == 1
    item = final[0]
    container_id = (
        item["container_id"] if isinstance(item, dict) else item.container_id
    )
    assert container_id == "container_011CW9hA9zpZ8xD3bjjShy4p"


def test_stderr_appended_to_logs():
    """stderr is folded into logs with a STDERR: prefix."""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_err",
                "name": "bash_code_execution",
                "input": {"command": "false"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_err",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "out\n",
                    "stderr": "boom\n",
                    "return_code": 1,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
    ]

    emissions = _emit_code_results(ModelResponseIterator(None, sync_stream=True), chunks)
    assert emissions
    item = emissions[-1][0]
    outputs = item["outputs"] if isinstance(item, dict) else item.outputs
    logs = outputs[0]["logs"] if isinstance(outputs[0], dict) else outputs[0].logs
    assert logs == "out\nSTDERR: boom\n"
