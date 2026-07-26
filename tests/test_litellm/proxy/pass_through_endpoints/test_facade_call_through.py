"""Live Wave 6A facade call-through tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    payload_validation,
    request_build,
    sse,
    stream_collect,
    tool_call_restore,
)


def test_request_build_facade_uses_live_metadata_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def metadata_value(body: dict[str, Any], *keys: str) -> str | None:
        calls.append(keys)
        return "repository-live" if "repository" in keys else None

    monkeypatch.setattr(lpe, "_extract_auto_agent_alias_metadata_value", metadata_value)
    context = request_build._build_malformed_tool_call_intake_context(
        None,
        {},
        adapter="wave6a",
    )

    assert request_build._build_malformed_tool_call_intake_context is (
        lpe._build_malformed_tool_call_intake_context
    )
    assert context["repository"] == "repository-live"
    assert calls


def test_sse_facade_uses_live_event_type_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def events() -> AsyncIterator[dict[str, int]]:
        yield {"value": 1}

    def event_type(obj: Any, key: str, default: Any = None) -> Any:
        if key == "type":
            return "response.wave6a"
        return default

    monkeypatch.setattr(lpe, "_mapping_or_attr_get", event_type)

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in sse._responses_sse_from_iterator(events())
        ]

    chunks = asyncio.run(collect())
    assert sse._responses_sse_from_iterator is lpe._responses_sse_from_iterator
    assert chunks[0].startswith("event: response.wave6a\n")


def test_tool_restore_facade_uses_live_advertised_tool_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def advertised(
        request_body: dict[str, Any] | None,
        *,
        adapter_model: str,
    ) -> set[str]:
        calls.append(adapter_model)
        return set()

    monkeypatch.setattr(
        lpe,
        "_advertised_custom_tool_function_adapter_names",
        advertised,
    )
    body = {"output": [{"type": "function_call", "name": "unchanged"}]}
    restored, count, error = (
        tool_call_restore._restore_adapted_custom_tool_calls_in_response_body(
            body,
            request_body={"tools": []},
            adapter_model="live-model",
        )
    )

    assert restored is body
    assert count == 0
    assert error is None
    assert calls == ["live-model"]


def test_stream_collect_facade_uses_canonical_sse_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lpe._mapping_or_attr_get
    calls: list[str] = []

    def recording_lookup(obj: Any, key: str, default: Any = None) -> Any:
        calls.append(key)
        return original(obj, key, default)

    monkeypatch.setattr(lpe, "_mapping_or_attr_get", recording_lookup)
    output_items: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    key_aliases: dict[str, str] = {}
    key_by_output_index: dict[int, str] = {}
    stream_collect._record_collected_responses_output_item_event(
        event={
            "item": {"type": "message", "id": "message-live"},
            "output_index": 0,
        },
        output_items=output_items,
        ordered_keys=ordered_keys,
        key_aliases=key_aliases,
        key_by_output_index=key_by_output_index,
    )

    assert stream_collect._record_collected_responses_output_item_event is (
        lpe._record_collected_responses_output_item_event
    )
    assert output_items["message-live"]["type"] == "message"
    assert calls


def test_payload_validation_facade_uses_live_malformed_text_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def malformed(text: str) -> bool:
        calls.append(text)
        return True

    monkeypatch.setattr(lpe, "is_malformed_composer_call_literal_text", malformed)
    body = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "live marker"}],
            }
        ]
    }

    assert (
        payload_validation._is_codex_auto_agent_malformed_tool_call_text_output(
            body
        )
        is True
    )
    assert payload_validation._is_codex_auto_agent_malformed_tool_call_text_output is (
        lpe._is_codex_auto_agent_malformed_tool_call_text_output
    )
    assert calls == ["live marker"]
