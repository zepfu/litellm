from __future__ import annotations

import json
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_events import (
    ResponsesEventSummaryReducer,
)


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _item_added(item: dict[str, Any], output_index: int = 0) -> dict[str, Any]:
    return {"type": "response.output_item.added", "output_index": output_index, "item": item}


def _function_call_item(arguments: str) -> dict[str, Any]:
    return {"type": "function_call", "id": "fc_1", "name": "lookup", "arguments": arguments}


def _snapshot_output(summary: dict[str, Any]) -> dict[str, Any]:
    assert len(summary["output"]) == 1
    return summary["output"][0]


def test_done_only_text_uses_prepopulated_final_snapshot() -> None:
    summary = ResponsesEventSummaryReducer(max_output_bytes=1024)
    text = "done-only text"

    summary.consume(_item_added({"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": text}]}))
    summary.consume({"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0, "text": text})

    snapshot = summary.snapshot()
    assert _snapshot_output(snapshot)["content"][0]["text"] == text
    assert snapshot["stream_text"] == text
    assert snapshot["output_bytes"] == len(_compact_json(_snapshot_output(snapshot)))
    assert snapshot["overflow"] is False


def test_done_only_arguments_uses_prepopulated_final_snapshot() -> None:
    summary = ResponsesEventSummaryReducer(max_output_bytes=1024)
    arguments = '{"query":"a\\"b"}'

    summary.consume(_item_added(_function_call_item(arguments)))
    summary.consume(
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0, "arguments": arguments}
    )

    snapshot = summary.snapshot()
    assert _snapshot_output(snapshot)["arguments"] == arguments
    assert snapshot["output_bytes"] == len(_compact_json(_snapshot_output(snapshot)))
    assert snapshot["overflow"] is False


def test_output_budget_rejects_new_item_exceeding_remaining_space() -> None:
    summary = ResponsesEventSummaryReducer(max_output_bytes=1024)
    retained = _item_added({"type": "message", "id": "retained", "content": [{"type": "output_text", "text": "x"}]}, 0)
    summary.consume(retained)
    retained_json = _compact_json(retained["item"])
    rejected_delta = "y" * 1024

    summary.consume({"type": "response.output_text.delta", "item_id": "msg_2", "output_index": 1, "delta": rejected_delta})

    snapshot = summary.snapshot()
    assert [item["id"] for item in snapshot["output"]] == ["retained"]
    assert snapshot["output_bytes"] == len(retained_json)
    assert snapshot["overflow"] is True
    assert snapshot["omitted_output_bytes"] > 0


@pytest.mark.parametrize("field", ["text", "arguments"])
def test_repeated_escaped_deltas_charge_cumulative_value(field: str) -> None:
    summary = ResponsesEventSummaryReducer(max_output_bytes=1024)
    escaped_value = 'a\\"b'
    base_item = _function_call_item("")
    base_item["type"] = "message" if field == "text" else "function_call"
    if field == "text":
        base_item["id"] = "msg_1"
        base_item["content"] = [{"type": "output_text", "text": ""}]
    else:
        base_item["arguments"] = ""

    summary.consume(_item_added(base_item))
    after_add = len(_compact_json(base_item))
    assert summary.snapshot()["output_bytes"] == after_add

    if field == "text":
        summary.consume({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": escaped_value})
        summary.consume({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": escaped_value})
    else:
        summary.consume(
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": escaped_value}
        )
        summary.consume(
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": escaped_value}
        )

    escaped_bytes = len(json.dumps(escaped_value, ensure_ascii=False).encode("utf-8")) - 2
    assert summary.snapshot()["output_bytes"] == after_add + escaped_bytes * 2

    snapshot = summary.snapshot()
    output = _snapshot_output(snapshot)
    if field == "text":
        assert output["content"][0]["text"] == escaped_value * 2
    else:
        assert output["arguments"] == escaped_value * 2
    assert snapshot["output_bytes"] == after_add + escaped_bytes * 2
    assert snapshot["overflow"] is False


@pytest.mark.parametrize("field", ["text", "arguments"])
def test_delta_then_output_item_done_replaces_active_field(field: str) -> None:
    summary = ResponsesEventSummaryReducer(max_output_bytes=1024)
    base_item = _function_call_item("")
    base_item["type"] = "message" if field == "text" else "function_call"
    if field == "text":
        base_item["id"] = "msg_1"
        base_item["content"] = [{"type": "output_text", "text": ""}]
    else:
        base_item["arguments"] = ""

    summary.consume(_item_added(base_item))
    if field == "text":
        summary.consume({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": "streamed"})
        replacement = dict(base_item, content=[{"type": "output_text", "text": "final"}])
    else:
        summary.consume(
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": "streamed"}
        )
        replacement = dict(base_item, arguments="final")

    summary.consume(_item_added(replacement, 0))

    snapshot = summary.snapshot()
    output = _snapshot_output(snapshot)
    if field == "text":
        assert output["content"][0]["text"] == "final"
    else:
        assert output["arguments"] == "final"
    assert snapshot["output_bytes"] == len(_compact_json(output))
    assert snapshot["overflow"] is False
