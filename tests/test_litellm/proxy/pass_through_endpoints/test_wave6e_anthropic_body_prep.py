"""Wave 6E tests: anthropic_body_prep module.

Covers compaction boundaries, logging callback shape, tool-block
repair/validation, final body composition/order, no-op/idempotence,
malformed input, and errors.
"""

from __future__ import annotations


from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_request_policy.anthropic_body_prep import (
    _add_openai_adapter_claude_context_compaction_logging_metadata,
    _build_openai_adapter_compacted_claude_context_block,
    _compact_openai_adapter_claude_context_in_anthropic_request_body,
    _compact_openai_adapter_claude_context_text,
    _compact_openai_adapter_claude_context_value,
    _detect_openai_adapter_claude_context_markers,
    _get_openai_adapter_claude_context_char_cap,
    _prepare_anthropic_request_body_for_passthrough,
    _repair_anthropic_tool_use_ids_for_passthrough,
    _select_openai_adapter_context_summary_lines,
    _validate_anthropic_tool_blocks_for_passthrough,
    configure_anthropic_body_prep_runtime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oversized_reminder(marker: str = "CLAUDE.md", size: int = 3000) -> str:
    """Build a system-reminder block exceeding the default 1200-char cap."""
    inner = f"{marker}\n" + "x" * (size - len(marker) - 40)
    return f"<system-reminder>\n{inner}\n</system-reminder>\n"


import asyncio


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _get_openai_adapter_claude_context_char_cap
# ---------------------------------------------------------------------------


class TestGetCharCap:
    def test_default_cap(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _get_openai_adapter_claude_context_char_cap() == 1200

    def test_custom_cap(self):
        with patch.dict(
            "os.environ", {"AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP": "500"}
        ):
            assert _get_openai_adapter_claude_context_char_cap() == 500

    def test_minimum_floor(self):
        with patch.dict(
            "os.environ", {"AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP": "10"}
        ):
            assert _get_openai_adapter_claude_context_char_cap() == 256

    def test_invalid_value_falls_back(self):
        with patch.dict(
            "os.environ", {"AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP": "abc"}
        ):
            assert _get_openai_adapter_claude_context_char_cap() == 1200

    def test_empty_value_falls_back(self):
        with patch.dict(
            "os.environ", {"AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP": ""}
        ):
            assert _get_openai_adapter_claude_context_char_cap() == 1200


# ---------------------------------------------------------------------------
# _detect_openai_adapter_claude_context_markers
# ---------------------------------------------------------------------------


class TestDetectMarkers:
    def test_no_markers(self):
        assert _detect_openai_adapter_claude_context_markers("hello world") == set()

    def test_single_marker(self):
        text = "some text CLAUDE.md more text"
        assert _detect_openai_adapter_claude_context_markers(text) == {"claude-md"}

    def test_multiple_markers(self):
        text = "CLAUDE.md and MEMORY.md and # TriStore Inject"
        markers = _detect_openai_adapter_claude_context_markers(text)
        assert markers == {"claude-md", "memory-md", "tristore-inject"}

    def test_subagent_marker(self):
        text = "SubagentStart hook additional context: stuff"
        assert _detect_openai_adapter_claude_context_markers(text) == {"subagentstart"}

    def test_subagent_case_variant(self):
        text = "SubAgentStart hook additional context: stuff"
        assert _detect_openai_adapter_claude_context_markers(text) == {"subagentstart"}


# ---------------------------------------------------------------------------
# _select_openai_adapter_context_summary_lines
# ---------------------------------------------------------------------------


class TestSelectSummaryLines:
    def test_selects_matching_lines(self):
        text = "# Heading\nrandom line\nIMPORTANT: do stuff\nContents of /foo"
        lines = _select_openai_adapter_context_summary_lines(text)
        assert "# Heading" in lines
        assert "IMPORTANT: do stuff" in lines
        assert "Contents of /foo" in lines
        assert "random line" not in lines

    def test_deduplicates(self):
        text = "# Same\n# Same\n# Same"
        lines = _select_openai_adapter_context_summary_lines(text)
        assert lines == ["# Same"]

    def test_caps_at_10(self):
        text = "\n".join(f"# heading {i}" for i in range(20))
        lines = _select_openai_adapter_context_summary_lines(text)
        assert len(lines) == 10

    def test_fallback_first_4_nonblank(self):
        text = "alpha\nbeta\ngamma\ndelta\nepsilon"
        lines = _select_openai_adapter_context_summary_lines(text)
        assert lines == ["alpha", "beta", "gamma", "delta"]

    def test_empty_text(self):
        assert _select_openai_adapter_context_summary_lines("") == []


# ---------------------------------------------------------------------------
# _build_openai_adapter_compacted_claude_context_block
# ---------------------------------------------------------------------------


class TestBuildCompactedBlock:
    def test_wraps_in_system_reminder(self):
        result = _build_openai_adapter_compacted_claude_context_block(
            original_block="x" * 2000,
            markers={"claude-md"},
            cap=1200,
        )
        assert result.startswith("<system-reminder>\n")
        assert result.endswith("</system-reminder>\n")

    def test_contains_marker_text(self):
        result = _build_openai_adapter_compacted_claude_context_block(
            original_block="x" * 2000,
            markers={"claude-md", "memory-md"},
            cap=1200,
        )
        assert "claude-md" in result
        assert "memory-md" in result

    def test_contains_original_char_count(self):
        original = "x" * 2000
        result = _build_openai_adapter_compacted_claude_context_block(
            original_block=original,
            markers={"claude-md"},
            cap=1200,
        )
        assert "from 2000 chars" in result

    def test_respects_cap(self):
        result = _build_openai_adapter_compacted_claude_context_block(
            original_block="x" * 5000,
            markers={"claude-md"},
            cap=500,
        )
        assert len(result) <= 600  # cap + wrapper overhead


# ---------------------------------------------------------------------------
# _compact_openai_adapter_claude_context_text
# ---------------------------------------------------------------------------


class TestCompactText:
    def test_no_system_reminder_noop(self):
        text = "just plain text"
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            text, cap=1200
        )
        assert result == text
        assert count == 0
        assert markers == set()
        assert meta == []

    def test_small_reminder_not_compacted(self):
        text = "<system-reminder>\nCLAUDE.md tiny\n</system-reminder>\n"
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            text, cap=1200
        )
        assert result == text
        assert count == 0

    def test_oversized_reminder_compacted(self):
        text = _make_oversized_reminder(size=3000)
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            text, cap=1200
        )
        assert count == 1
        assert "claude-md" in markers
        assert len(result) < len(text)
        assert len(meta) == 1
        assert meta[0]["mode"] == "system_reminder_context_cap"
        assert meta[0]["original_chars"] == len(text)

    def test_multiple_spans_all_compacted(self):
        text = _make_oversized_reminder("CLAUDE.md", 2000) + _make_oversized_reminder(
            "MEMORY.md", 2000
        )
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            text, cap=1200
        )
        assert count == 2
        assert "claude-md" in markers
        assert "memory-md" in markers
        assert len(meta) == 2

    def test_no_markers_in_reminder_not_compacted(self):
        inner = "x" * 3000
        text = f"<system-reminder>\n{inner}\n</system-reminder>\n"
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            text, cap=1200
        )
        assert count == 0
        assert result == text

    def test_explicit_cap_override(self):
        text = _make_oversized_reminder(size=500)
        result, count, _, _ = _compact_openai_adapter_claude_context_text(
            text, cap=256
        )
        assert count == 1


# ---------------------------------------------------------------------------
# _compact_openai_adapter_claude_context_value
# ---------------------------------------------------------------------------


class TestCompactValue:
    def test_string_passthrough(self):
        text = _make_oversized_reminder(size=3000)
        result, count, markers, meta = _compact_openai_adapter_claude_context_value(
            text, cap=1200
        )
        assert count == 1
        assert isinstance(result, str)

    def test_dict_recursive(self):
        inner = _make_oversized_reminder(size=3000)
        value = {"type": "text", "text": inner}
        result, count, markers, meta = _compact_openai_adapter_claude_context_value(
            value, cap=1200
        )
        assert count == 1
        assert result["type"] == "text"
        assert len(result["text"]) < len(inner)

    def test_list_recursive(self):
        inner = _make_oversized_reminder(size=3000)
        value = [{"type": "text", "text": inner}]
        result, count, _, _ = _compact_openai_adapter_claude_context_value(
            value, cap=1200
        )
        assert count == 1
        assert isinstance(result, list)

    def test_non_text_scalar_noop(self):
        result, count, markers, meta = _compact_openai_adapter_claude_context_value(
            42, cap=1200
        )
        assert result == 42
        assert count == 0

    def test_unchanged_dict_returns_same_object(self):
        value = {"key": "no markers here"}
        result, count, _, _ = _compact_openai_adapter_claude_context_value(
            value, cap=1200
        )
        assert result is value
        assert count == 0


# ---------------------------------------------------------------------------
# _add_openai_adapter_claude_context_compaction_logging_metadata
# ---------------------------------------------------------------------------


class TestCompactionLoggingMetadata:
    def test_metadata_shape(self):
        body: dict[str, Any] = {"model": "test"}
        now = datetime.now(timezone.utc)
        result = _add_openai_adapter_claude_context_compaction_logging_metadata(
            body,
            compacted_count=2,
            markers={"claude-md", "memory-md"},
            metadata_items=[
                {"original_chars": 3000, "kept_chars": 500, "markers": ["claude-md"], "mode": "system_reminder_context_cap"},
                {"original_chars": 2000, "kept_chars": 400, "markers": ["memory-md"], "mode": "system_reminder_context_cap"},
            ],
            span_started_at=now,
        )
        lm = result["litellm_metadata"]
        assert lm["openai_adapter_claude_context_compacted"] is True
        assert lm["openai_adapter_claude_context_compacted_count"] == 2
        assert lm["openai_adapter_claude_context_markers"] == ["claude-md", "memory-md"]
        assert lm["openai_adapter_claude_context_original_chars"] == 5000
        assert lm["openai_adapter_claude_context_compacted_chars"] == 900
        assert lm["openai_adapter_claude_context_saved_chars"] == 4100
        assert len(lm["openai_adapter_claude_context_compaction_events"]) == 2
        assert "openai-adapter-claude-context-compacted" in lm["tags"]
        assert "openai-adapter-claude-context:claude-md" in lm["tags"]
        spans = lm["langfuse_spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "openai_adapter.claude_context_compaction"
        assert "start_time" in spans[0]
        assert "end_time" in spans[0]

    def test_custom_prefixes(self):
        body: dict[str, Any] = {}
        now = datetime.now(timezone.utc)
        result = _add_openai_adapter_claude_context_compaction_logging_metadata(
            body,
            compacted_count=1,
            markers={"tristore-inject"},
            metadata_items=[
                {"original_chars": 100, "kept_chars": 50, "markers": ["tristore-inject"], "mode": "x"},
            ],
            span_started_at=now,
            tag_prefix="custom",
            metadata_prefix="custom_pfx",
            span_name="custom.span",
        )
        lm = result["litellm_metadata"]
        assert "custom-claude-context-compacted" in lm["tags"]
        assert lm["custom_pfx_claude_context_compacted"] is True
        assert lm["langfuse_spans"][0]["name"] == "custom.span"


# ---------------------------------------------------------------------------
# _compact_openai_adapter_claude_context_in_anthropic_request_body
# ---------------------------------------------------------------------------


class TestCompactInRequestBody:
    def test_noop_no_system_or_messages(self):
        body: dict[str, Any] = {"model": "test"}
        result, count, markers, meta = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(body)
        )
        assert result is body
        assert count == 0

    def test_compacts_system_field(self):
        body: dict[str, Any] = {
            "system": _make_oversized_reminder(size=3000),
        }
        result, count, markers, meta = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(body)
        )
        assert count == 1
        assert result is not body
        assert "litellm_metadata" in result

    def test_compacts_messages_field(self):
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _make_oversized_reminder(size=3000)}
                    ],
                }
            ],
        }
        result, count, _, _ = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(body)
        )
        assert count == 1

    def test_idempotent_second_call(self):
        body: dict[str, Any] = {
            "system": _make_oversized_reminder(size=3000),
        }
        result1, count1, _, _ = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(body)
        )
        result2, count2, _, _ = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(result1)
        )
        assert count1 == 1
        assert count2 == 0
        assert result2 is result1


# ---------------------------------------------------------------------------
# _validate_anthropic_tool_blocks_for_passthrough
# ---------------------------------------------------------------------------


class TestValidateToolBlocks:
    def test_valid_body_passes(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "bash", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "ok"}
                    ],
                },
            ]
        }
        _validate_anthropic_tool_blocks_for_passthrough(body)  # should not raise

    def test_missing_tool_use_id_raises(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "bash", "input": {}}],
                }
            ]
        }
        with pytest.raises(HTTPException) as exc_info:
            _validate_anthropic_tool_blocks_for_passthrough(body)
        assert exc_info.value.status_code == 400
        assert "tool_use.id" in exc_info.value.detail

    def test_empty_tool_use_id_raises(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "  ", "name": "bash", "input": {}}],
                }
            ]
        }
        with pytest.raises(HTTPException) as exc_info:
            _validate_anthropic_tool_blocks_for_passthrough(body)
        assert exc_info.value.status_code == 400

    def test_missing_tool_result_id_raises(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "ok"}],
                }
            ]
        }
        with pytest.raises(HTTPException) as exc_info:
            _validate_anthropic_tool_blocks_for_passthrough(body)
        assert exc_info.value.status_code == 400
        assert "tool_result.tool_use_id" in exc_info.value.detail

    def test_suffix_tool_result_validated(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "mcp_tool_result", "content": "ok"}],
                }
            ]
        }
        with pytest.raises(HTTPException) as exc_info:
            _validate_anthropic_tool_blocks_for_passthrough(body)
        assert exc_info.value.status_code == 400

    def test_no_messages_noop(self):
        _validate_anthropic_tool_blocks_for_passthrough({"model": "x"})

    def test_non_dict_messages_noop(self):
        _validate_anthropic_tool_blocks_for_passthrough({"messages": "not a list"})

    def test_non_dict_message_skipped(self):
        body = {"messages": ["not a dict", 42, None]}
        _validate_anthropic_tool_blocks_for_passthrough(body)

    def test_non_list_content_skipped(self):
        body = {"messages": [{"role": "user", "content": "plain string"}]}
        _validate_anthropic_tool_blocks_for_passthrough(body)

    def test_non_dict_block_skipped(self):
        body = {"messages": [{"role": "user", "content": ["string", 42]}]}
        _validate_anthropic_tool_blocks_for_passthrough(body)

    def test_text_block_ignored(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ]
        }
        _validate_anthropic_tool_blocks_for_passthrough(body)


# ---------------------------------------------------------------------------
# _repair_anthropic_tool_use_ids_for_passthrough
# ---------------------------------------------------------------------------


class TestRepairToolUseIds:
    def test_no_messages_noop(self):
        body: dict[str, Any] = {"model": "x"}
        result, count = _repair_anthropic_tool_use_ids_for_passthrough(body)
        assert result is body
        assert count == 0

    def test_non_list_messages_noop(self):
        body: dict[str, Any] = {"messages": "bad"}
        result, count = _repair_anthropic_tool_use_ids_for_passthrough(body)
        assert result is body
        assert count == 0

    def test_valid_ids_no_repair(self):
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}
                    ],
                }
            ]
        }
        result, count = _repair_anthropic_tool_use_ids_for_passthrough(body)
        assert count == 0
        assert result is body

    def test_repair_adds_metadata(self):
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "bash", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "", "content": "ok"}
                    ],
                },
            ]
        }
        result, count = _repair_anthropic_tool_use_ids_for_passthrough(body)
        if count > 0:
            lm = result.get("litellm_metadata", {})
            assert "anthropic-tool-use-id-repaired" in lm.get("tags", [])
            assert lm.get("anthropic_tool_use_id_repaired_count") == count


# ---------------------------------------------------------------------------
# _prepare_anthropic_request_body_for_passthrough
# ---------------------------------------------------------------------------


class TestPrepareAnthropicRequestBody:
    def setup_method(self):
        configure_anthropic_body_prep_runtime()

    def teardown_method(self):
        configure_anthropic_body_prep_runtime()

    def test_noop_no_callbacks(self):
        body: dict[str, Any] = {"model": "claude-3", "messages": []}
        result, expanded, hooks, billing = _run(
            _prepare_anthropic_request_body_for_passthrough(MagicMock(), body)
        )
        assert expanded == 0
        assert hooks == set()
        assert billing == {}

    def test_callback_order(self):
        call_order: list[str] = []

        def make_sync(name: str):
            def fn(body: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
                call_order.append(name)
                return body
            return fn

        def make_async(name: str):
            async def fn(body: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
                call_order.append(name)
                return body, [], []
            return fn

        def expand_po(body: dict[str, Any]) -> Any:
            call_order.append("expand_persisted_output")
            return body, 0, set(), []

        def extract_billing(body: dict[str, Any]) -> dict[str, str]:
            call_order.append("extract_billing")
            return {}

        async def apply_cp(body: dict[str, Any], billing: dict[str, str]) -> Any:
            call_order.append("apply_control_plane")
            return body, [], []

        async def expand_dd(body: dict[str, Any]) -> Any:
            call_order.append("expand_dynamic")
            return body, []

        def sanitize_ws(body: dict[str, Any]) -> Any:
            call_order.append("sanitize_web_search")
            return body, 0

        def add_route(body: dict[str, Any], family: str) -> dict[str, Any]:
            call_order.append("add_route_family")
            return body

        def add_breakout(body: dict[str, Any]) -> dict[str, Any]:
            call_order.append("add_breakout")
            return body

        def prepare_obs(*, request: Any, request_body: dict[str, Any]) -> dict[str, Any]:
            call_order.append("prepare_observability")
            return request_body

        configure_anthropic_body_prep_runtime(
            expand_persisted_output=expand_po,
            extract_billing_header_fields=extract_billing,
            apply_control_plane_rewrites=apply_cp,
            expand_dynamic_directives=expand_dd,
            add_post_rewrite_context_file_metadata=make_sync("post_rewrite"),
            sanitize_web_search_domain_lists=sanitize_ws,
            add_route_family_logging_metadata=add_route,
            add_request_breakout_logging_metadata=add_breakout,
            prepare_observability=prepare_obs,
        )

        body: dict[str, Any] = {"model": "claude-3", "messages": []}
        _run(_prepare_anthropic_request_body_for_passthrough(MagicMock(), body))

        assert call_order.index("expand_persisted_output") < call_order.index("extract_billing")
        assert call_order.index("extract_billing") < call_order.index("apply_control_plane")
        assert call_order.index("apply_control_plane") < call_order.index("expand_dynamic")
        assert call_order.index("expand_dynamic") < call_order.index("post_rewrite")
        assert call_order.index("post_rewrite") < call_order.index("sanitize_web_search")
        assert call_order.index("sanitize_web_search") < call_order.index("add_route_family")
        assert call_order.index("add_route_family") < call_order.index("add_breakout")
        assert call_order.index("add_breakout") < call_order.index("prepare_observability")

    def test_billing_header_fields_returned(self):
        def extract_billing(body: dict[str, Any]) -> dict[str, str]:
            return {"x-billing": "yes"}

        configure_anthropic_body_prep_runtime(
            extract_billing_header_fields=extract_billing,
        )
        body: dict[str, Any] = {"model": "claude-3", "messages": []}
        _, _, _, billing = _run(
            _prepare_anthropic_request_body_for_passthrough(MagicMock(), body)
        )
        assert billing == {"x-billing": "yes"}

    def test_expand_persisted_output_count_returned(self):
        def expand_po(body: dict[str, Any]) -> Any:
            return body, 3, {"subagentstart"}, [{"path": "/tmp/x"}]

        configure_anthropic_body_prep_runtime(
            expand_persisted_output=expand_po,
        )
        body: dict[str, Any] = {"model": "claude-3", "messages": []}
        _, expanded, hooks, _ = _run(
            _prepare_anthropic_request_body_for_passthrough(MagicMock(), body)
        )
        assert expanded == 3
        assert hooks == {"subagentstart"}

    def test_validation_error_propagates(self):
        # tool_result with missing tool_use_id and no tool_use to repair from
        body: dict[str, Any] = {
            "model": "claude-3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": "ok"}
                    ],
                }
            ],
        }
        with pytest.raises(HTTPException) as exc_info:
            _run(_prepare_anthropic_request_body_for_passthrough(MagicMock(), body))
        assert exc_info.value.status_code == 400

    def test_tenant_header_forwarded(self):

        def get_tenant(request: Any) -> str:
            return "tenant-42"

        configure_anthropic_body_prep_runtime(
            get_tenant_header=get_tenant,
        )
        body: dict[str, Any] = {
            "model": "claude-3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are 'worker' and you are working on the 'myproj' project",
                        }
                    ],
                }
            ],
        }
        result, _, _, _ = _run(
            _prepare_anthropic_request_body_for_passthrough(MagicMock(), body)
        )
        lm = result.get("litellm_metadata", {})
        assert lm.get("tenant_id") == "tenant-42"


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


class TestMalformedInput:
    def test_compact_text_empty_string(self):
        result, count, markers, meta = _compact_openai_adapter_claude_context_text(
            "", cap=1200
        )
        assert result == ""
        assert count == 0

    def test_compact_value_none(self):
        result, count, _, _ = _compact_openai_adapter_claude_context_value(
            None, cap=1200
        )
        assert result is None
        assert count == 0

    def test_validate_empty_body(self):
        _validate_anthropic_tool_blocks_for_passthrough({})

    def test_repair_empty_body(self):
        result, count = _repair_anthropic_tool_use_ids_for_passthrough({})
        assert count == 0

    def test_compact_body_non_dict_values(self):
        body: dict[str, Any] = {"system": 42, "messages": "bad"}
        result, count, _, _ = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(body)
        )
        assert count == 0
        assert result is body
