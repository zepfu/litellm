"""Focused tests for the Wave 6D observability metadata extraction."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    observability_metadata as metadata,
)


def _claude_assignment(agent: str = "worker", tenant: str = "litellm") -> str:
    return (
        f"You are '{agent}' and you are working on the '{tenant}' project "
        "with a bounded ownership scope."
    )


def test_iter_anthropic_text_fragments_preserves_nested_order() -> None:
    value = {
        "first": "plain",
        "second": [
            {"type": "text", "text": "block", "ignored": "not-recursed"},
            {"nested": {"value": "nested"}},
            None,
            17,
        ],
    }

    assert list(metadata._iter_anthropic_text_fragments(value)) == [
        "plain",
        "block",
        "nested",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        (0, []),
        (True, []),
        ({"type": "text", "text": 7}, ["text"]),
        ([None, 3, False, {"type": "image", "source": 8}], ["image"]),
    ],
)
def test_iter_anthropic_text_fragments_preserves_malformed_behavior(
    value: Any,
    expected: list[str],
) -> None:
    assert list(metadata._iter_anthropic_text_fragments(value)) == expected


def test_extract_claude_agent_and_tenant_prefers_messages_and_strips() -> None:
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _claude_assignment("  worker  ", "  tenant-a  "),
                    }
                ],
            }
        ],
        "system": _claude_assignment("system-agent", "system-tenant"),
    }

    assert metadata._extract_claude_agent_and_tenant_from_request_body(
        request_body
    ) == ("worker", "tenant-a")


def test_extract_claude_agent_and_tenant_ignores_malformed_content() -> None:
    request_body = {
        "messages": [
            None,
            {"content": 42},
            {"content": {"type": "text", "text": None}},
            {"content": "You are '' and you are working on the 'x' project"},
        ],
        "system": {"type": "text", "text": ["not", "text"]},
    }

    assert metadata._extract_claude_agent_and_tenant_from_request_body(
        request_body
    ) == (None, None)


def test_merge_litellm_metadata_preserves_copy_and_merges_spans() -> None:
    request_body: dict[str, Any] = {
        "model": "claude",
        "litellm_metadata": {
            "tags": ["existing", "duplicate"],
            "langfuse_spans": [{"name": "existing"}],
        },
    }

    result = metadata._merge_litellm_metadata(
        request_body,
        tags_to_add=["duplicate", "added"],
        extra_fields={
            "langfuse_spans": [{"name": "added"}],
            "trace_name": "trace",
        },
    )

    assert result is not request_body
    assert result["litellm_metadata"] == {
        "tags": ["existing", "duplicate", "added"],
        "langfuse_spans": [{"name": "existing"}, {"name": "added"}],
        "trace_name": "trace",
    }
    assert request_body["litellm_metadata"]["tags"] == ["existing", "duplicate"]


def test_shared_metadata_primitives_preserve_normalization_and_descriptor() -> None:
    eastern = timezone(timedelta(hours=-5))
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=eastern)

    assert metadata._normalize_low_cardinality_tag_value(True) == "true"
    assert metadata._normalize_low_cardinality_tag_value(" Agent ") == "agent"
    assert metadata._normalize_low_cardinality_tag_value(" ") is None
    assert metadata._normalize_low_cardinality_tag_value(1) is None
    assert metadata._dedupe_sorted_str_list(["b", "", "a", "b"]) == ["a", "b"]
    assert metadata._build_langfuse_span_descriptor(
        name="operation",
        metadata={"key": "value"},
        input_data={"in": 1},
        output_data={"out": 2},
        start_time=timestamp,
    ) == {
        "name": "operation",
        "metadata": {"key": "value"},
        "input": {"in": 1},
        "output": {"out": 2},
        "start_time": "2026-01-02T08:04:05Z",
    }


def test_add_child_agent_observability_metadata_preserves_identity_fields() -> None:
    request_body = {
        "messages": [{"content": _claude_assignment("Worker Agent", "Project X")}],
        "litellm_metadata": {
            "tags": ["existing"],
            "trace_name": "parent-trace",
            "trace_user_id": "parent-user",
        },
    }

    result = metadata._add_claude_child_agent_observability_metadata(
        request_body,
        explicit_tenant_id="tenant-header",
    )

    assert result["litellm_metadata"] == {
        "tags": [
            "existing",
            "claude-agent:worker agent",
            "claude-project:project x",
        ],
        "agent_name": "Worker Agent",
        "aawm_claude_agent_name": "Worker Agent",
        "source_trace_name": "parent-trace",
        "trace_name": "claude-code.Worker Agent",
        "tenant_id": "tenant-header",
        "aawm_tenant_id": "tenant-header",
        "aawm_claude_project": "Project X",
        "source_trace_user_id": "parent-user",
        "trace_user_id": "tenant-header",
    }


def test_add_child_agent_observability_uses_request_callback() -> None:
    request = object()
    observed_requests: list[Any] = []

    def get_tenant_id(candidate: Any) -> str:
        observed_requests.append(candidate)
        return "callback-tenant"

    result = metadata._add_claude_child_agent_observability_metadata(
        {"system": _claude_assignment()},
        request=request,
        get_explicit_tenant_id=get_tenant_id,
    )

    assert observed_requests == [request]
    assert result["litellm_metadata"]["tenant_id"] == "callback-tenant"
    assert result["litellm_metadata"]["aawm_claude_project"] == "litellm"


def test_add_child_agent_observability_returns_original_without_assignment() -> None:
    request_body = {
        "messages": [None, {"content": {"type": "text", "text": 7}}],
        "litellm_metadata": "malformed-but-unread",
    }

    assert (
        metadata._add_claude_child_agent_observability_metadata(request_body)
        is request_body
    )


def test_detect_post_rewrite_context_files_preserves_discovery_order() -> None:
    request_body = {
        "system": [{"type": "text", "text": "Read CLAUDE.md first."}],
        "messages": [
            {"content": "Then consult MEMORY.md and CLAUDE.md."},
            {"content": None},
        ],
    }

    assert metadata._detect_claude_post_rewrite_context_files(request_body) == [
        "CLAUDE.md",
        "MEMORY.md",
    ]


def test_add_context_file_logging_metadata_uses_explicit_callbacks() -> None:
    request_body = {"system": "context"}
    observed: list[tuple[dict[str, Any], list[str]]] = []
    merge_calls: list[dict[str, Any]] = []

    def detect(body: dict[str, Any]) -> list[str]:
        assert body is request_body
        return ["MEMORY.md", "CLAUDE.md"]

    def log_files(body: dict[str, Any], files: list[str]) -> None:
        observed.append((body, files))

    def merge(
        body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        merge_calls.append(
            {"tags_to_add": tags_to_add, "extra_fields": extra_fields}
        )
        return {"merged": body}

    result = metadata._add_claude_post_rewrite_context_file_logging_metadata(
        request_body,
        detect_context_files=detect,
        merge_metadata=merge,
        log_context_files=log_files,
    )

    assert result == {"merged": request_body}
    assert observed == [(request_body, ["MEMORY.md", "CLAUDE.md"])]
    assert merge_calls == [
        {
            "tags_to_add": [
                "claude-post-rewrite-context-file-present",
                "claude-post-rewrite-context-file:memory-md",
                "claude-post-rewrite-context-file:claude-md",
            ],
            "extra_fields": {
                "claude_post_rewrite_context_files_present": [
                    "MEMORY.md",
                    "CLAUDE.md",
                ],
                "claude_post_rewrite_context_file_count": 2,
            },
        }
    ]


def test_configured_runtime_callbacks_are_optional_and_resettable() -> None:
    request = object()
    logged: list[list[str]] = []

    metadata.configure_observability_metadata_runtime(
        get_explicit_tenant_id=lambda candidate: (
            "configured-tenant" if candidate is request else None
        ),
        log_context_files=lambda _body, files: logged.append(files),
    )
    try:
        child_result = metadata._add_claude_child_agent_observability_metadata(
            {"messages": [_claude_assignment()]},
            request=request,
        )
        context_result = (
            metadata._add_claude_post_rewrite_context_file_logging_metadata(
                {"system": "Loaded MEMORY.md."}
            )
        )
    finally:
        metadata.configure_observability_metadata_runtime()

    assert child_result["litellm_metadata"]["tenant_id"] == "configured-tenant"
    assert context_result["litellm_metadata"] == {
        "tags": [
            "claude-post-rewrite-context-file-present",
            "claude-post-rewrite-context-file:memory-md",
        ],
        "claude_post_rewrite_context_files_present": ["MEMORY.md"],
        "claude_post_rewrite_context_file_count": 1,
    }
    assert logged == [["MEMORY.md"]]


def test_context_file_logging_returns_original_for_malformed_absent_content() -> None:
    request_body = {
        "system": {"type": "text", "text": None},
        "messages": [None, 4, {"content": ["unrelated"]}],
    }
    logged: list[list[str]] = []

    result = metadata._add_claude_post_rewrite_context_file_logging_metadata(
        request_body,
        log_context_files=lambda _body, files: logged.append(files),
    )

    assert result is request_body
    assert logged == []


# ---------------------------------------------------------------------------
# Owned-symbol inventory
# ---------------------------------------------------------------------------


def test_owned_symbols_inventory_is_exact_and_resolvable() -> None:
    expected_count = 58
    assert len(metadata.OWNED_SYMBOLS) == expected_count
    assert len(set(metadata.OWNED_SYMBOLS)) == expected_count
    for symbol in metadata.OWNED_SYMBOLS:
        assert hasattr(metadata, symbol), f"Missing symbol: {symbol}"


# ---------------------------------------------------------------------------
# No-god-import guard
# ---------------------------------------------------------------------------


def test_no_god_module_import() -> None:
    module = sys.modules[metadata.__name__]
    source_file = module.__file__
    assert source_file is not None
    with open(source_file) as fh:
        source_text = fh.read()
    import_lines = [
        line
        for line in source_text.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for line in import_lines:
        assert "llm_passthrough_endpoints" not in line, f"God import found: {line}"


# ---------------------------------------------------------------------------
# Session / repository extraction
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def test_extract_passthrough_session_id_from_body_paths() -> None:
    request = _FakeRequest()
    assert metadata._extract_passthrough_session_id(
        request, {"session_id": "s-1"}
    ) == "s-1"
    assert metadata._extract_passthrough_session_id(
        request, {"request": {"session_id": "s-2"}}
    ) == "s-2"
    assert metadata._extract_passthrough_session_id(
        request, {"metadata": {"user_id": {"session_id": "s-3"}}}
    ) == "s-3"


def test_extract_passthrough_session_id_from_headers() -> None:
    request = _FakeRequest({"x-session-id": "header-session"})
    assert metadata._extract_passthrough_session_id(request, {}) == "header-session"


def test_extract_passthrough_session_id_returns_none_when_absent() -> None:
    assert metadata._extract_passthrough_session_id(_FakeRequest(), {}) is None


def test_normalize_passthrough_repository_goldens() -> None:
    assert metadata._normalize_passthrough_repository(
        "https://github.com/user/repo.git"
    ) == "user/repo"
    assert metadata._normalize_passthrough_repository(
        "git@github.com:user/repo.git"
    ) == "user/repo"
    assert metadata._normalize_passthrough_repository(
        "/home/user/projects/litellm"
    ) == "litellm"
    assert metadata._normalize_passthrough_repository("unknown") is None
    assert metadata._normalize_passthrough_repository("agent") is None
    assert metadata._normalize_passthrough_repository("agent-abc123") is None
    assert metadata._normalize_passthrough_repository("wave3-engineer") is None
    assert metadata._normalize_passthrough_repository("rollout-2026-abc.jsonl") is None
    assert metadata._normalize_passthrough_repository("file:///home/user/myrepo") == "myrepo"


def test_extract_passthrough_repository_from_body_key() -> None:
    request = _FakeRequest()
    assert metadata._extract_passthrough_repository(
        request, {"repository": "https://github.com/org/proj.git"}
    ) == "org/proj"


def test_extract_passthrough_repository_from_headers() -> None:
    request = _FakeRequest({"x-aawm-repository": "/home/user/projects/aawm"})
    assert metadata._extract_passthrough_repository(request, {}) == "aawm"


def test_extract_passthrough_repository_from_cwd_text() -> None:
    request = _FakeRequest()
    body = {"messages": [{"content": "<cwd>/home/user/projects/litellm</cwd>"}]}
    assert metadata._extract_passthrough_repository(request, body) == "litellm"


def test_get_passthrough_trace_environment_uses_env_callback() -> None:
    metadata.configure_observability_metadata_runtime(
        get_env=lambda name: "staging"
        if name == "LITELLM_LANGFUSE_TRACE_ENVIRONMENT"
        else None,
    )
    try:
        assert metadata._get_passthrough_trace_environment() == "staging"
    finally:
        metadata.configure_observability_metadata_runtime()


def test_add_passthrough_trace_context_metadata_preserves_original() -> None:
    body: dict[str, Any] = {"model": "x"}
    result = metadata._add_passthrough_trace_context_metadata(
        body,
        session_id="sess-1",
        trace_environment="prod",
        repository="org/repo",
    )
    assert result is not body
    assert result["litellm_metadata"] == {
        "session_id": "sess-1",
        "trace_environment": "prod",
        "repository": "org/repo",
    }
    assert "litellm_metadata" not in body


def test_add_passthrough_trace_context_metadata_returns_original_when_no_change() -> None:
    body: dict[str, Any] = {
        "litellm_metadata": {"session_id": "existing", "trace_environment": "prod"}
    }
    result = metadata._add_passthrough_trace_context_metadata(
        body, session_id="existing", trace_environment="prod"
    )
    assert result is body


# ---------------------------------------------------------------------------
# Tool-definition snapshot metadata
# ---------------------------------------------------------------------------


def test_tool_definition_snapshot_entry_and_hash() -> None:
    tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    entry, truncated = metadata._build_tool_definition_snapshot_entry(
        source="tools", index=0, tool=tool
    )
    assert entry is not None
    assert not truncated
    assert entry["name"] == "get_weather"
    assert entry["source"] == "tools"
    assert entry["index"] == 0

    hash_val = metadata._tool_definition_snapshot_hash([entry])
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64


def test_build_passthrough_tool_definition_metadata_empty() -> None:
    assert metadata._build_passthrough_tool_definition_metadata({}) == {}
    assert metadata._build_passthrough_tool_definition_metadata({"tools": []}) == {}


def test_build_passthrough_tool_definition_metadata_basic() -> None:
    body = {
        "tools": [
            {"type": "function", "name": "tool_a", "description": "A"},
            {"type": "function", "name": "tool_b", "description": "B"},
        ]
    }
    result = metadata._build_passthrough_tool_definition_metadata(body)
    assert result["aawm_tool_definition_count"] == 2
    assert result["aawm_tool_definition_captured_count"] == 2
    assert result["aawm_tool_definition_names"] == ["tool_a", "tool_b"]
    assert result["aawm_tool_definition_capture_version"] == "v1"
    assert "aawm_tool_definition_snapshot" not in result


def test_add_passthrough_tool_definition_metadata_merges() -> None:
    body: dict[str, Any] = {"tools": [{"name": "x", "type": "function"}]}
    result = metadata._add_passthrough_tool_definition_metadata(body)
    assert result is not body
    assert "aawm_tool_definition_count" in result["litellm_metadata"]


def test_sanitize_tool_definition_redacts_secrets() -> None:
    value, truncated = metadata._sanitize_tool_definition_value(
        {"api_key": "sk-abc12345678901234567890", "safe": "hello"}
    )
    assert not truncated
    assert value["api_key"] == "redacted-by-litellm"
    assert value["safe"] == "hello"


# ---------------------------------------------------------------------------
# _prepare_request_body_for_passthrough_observability
# ---------------------------------------------------------------------------


def test_prepare_request_body_for_passthrough_observability() -> None:
    request = _FakeRequest({"x-session-id": "sess-99"})
    body: dict[str, Any] = {
        "repository": "/home/user/projects/litellm",
        "tools": [{"name": "t1", "type": "function"}],
    }
    result = metadata._prepare_request_body_for_passthrough_observability(request, body)
    lm = result["litellm_metadata"]
    assert lm["session_id"] == "sess-99"
    assert lm["repository"] == "litellm"
    assert "aawm_tool_definition_count" in lm


def test_shared_observability_does_not_import_langfuse_identity_headers() -> None:
    request = _FakeRequest(
        {
            "x-session-id": "sess-99",
            "langfuse_trace_name": "direct-only-name",
            "Langfuse-Trace-User-Id": "direct-only-user",
        }
    )
    body: dict[str, Any] = {
        "repository": "/home/user/projects/litellm",
        "litellm_metadata": {
            "trace_name": "orchestrator",
            "trace_user_id": "existing-user",
        },
    }

    result = metadata._prepare_request_body_for_passthrough_observability(
        request, body
    )

    assert result["litellm_metadata"] == {
        "trace_name": "orchestrator",
        "trace_user_id": "existing-user",
        "session_id": "sess-99",
        "repository": "litellm",
    }


# ---------------------------------------------------------------------------
# Claude breakout extraction / logging
# ---------------------------------------------------------------------------


def test_extract_claude_request_breakout_fields_full() -> None:
    body = {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "high"},
        "context_management": {
            "edits": [
                {"type": "clear_tool_uses", "keep": "all"},
                {"type": "summarize", "keep": "recent"},
            ]
        },
        "metadata": {"user_id": {"account_uuid": "uuid-1", "device_id": "dev-1"}},
    }
    tags, fields = metadata._extract_claude_request_breakout_fields(body)
    assert "claude-thinking-type:enabled" in tags
    assert "thinking-type:enabled" in tags
    assert "claude-effort:high" in tags
    assert "effort:high" in tags
    assert "claude-context-edit:clear_tool_uses" in tags
    assert "claude-context-edit:summarize" in tags
    assert "claude-context-keep:all" in tags
    assert "claude-context-keep:recent" in tags
    assert fields["claude_thinking_type"] == "enabled"
    assert fields["claude_effort"] == "high"
    assert fields["claude_context_edit_count"] == 2
    assert fields["claude_context_edit_types"] == ["clear_tool_uses", "summarize"]
    assert fields["claude_context_keep_values"] == ["all", "recent"]
    assert fields["claude_account_uuid"] == "uuid-1"
    assert fields["claude_device_id"] == "dev-1"


def test_add_claude_request_breakout_logging_metadata_empty_returns_original() -> None:
    body: dict[str, Any] = {"model": "claude"}
    assert metadata._add_claude_request_breakout_logging_metadata(body) is body


def test_add_claude_request_breakout_logging_metadata_merges() -> None:
    body: dict[str, Any] = {"thinking": {"type": "enabled"}}
    result = metadata._add_claude_request_breakout_logging_metadata(body)
    assert result is not body
    assert "claude-thinking-type:enabled" in result["litellm_metadata"]["tags"]


# ---------------------------------------------------------------------------
# Codex breakout extraction / logging
# ---------------------------------------------------------------------------


def test_extract_codex_request_breakout_fields_full() -> None:
    body = {
        "reasoning": {"effort": "high"},
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "cache-1",
    }
    tags, fields = metadata._extract_codex_request_breakout_fields(body)
    assert "codex-effort:high" in tags
    assert "effort:high" in tags
    assert "codex-tool-choice:auto" in tags
    assert "codex-parallel-tools:true" in tags
    assert "codex-include:reasoning.encrypted_content" in tags
    assert fields["codex_reasoning_effort"] == "high"
    assert fields["codex_tool_choice"] == "auto"
    assert fields["codex_parallel_tool_calls"] is True
    assert fields["codex_include"] == ["reasoning.encrypted_content"]
    assert fields["codex_prompt_cache_key_present"] is True


def test_extract_codex_tool_choice_dict() -> None:
    assert metadata._extract_openai_passthrough_tool_choice(
        {"type": "function", "name": "my_tool"}
    ) == "function"
    assert metadata._extract_openai_passthrough_tool_choice(None) is None
    assert metadata._extract_openai_passthrough_tool_choice(42) is None


def test_add_codex_request_breakout_logging_metadata_empty_returns_original() -> None:
    body: dict[str, Any] = {"model": "codex"}
    assert metadata._add_codex_request_breakout_logging_metadata(body) is body


# ---------------------------------------------------------------------------
# Anthropic billing header
# ---------------------------------------------------------------------------


def test_parse_anthropic_billing_header_text() -> None:
    text = (
        "x-anthropic-billing-header: cc_version=1.0.23; plan=pro; org_id=org-1\n"
        "unrelated line\n"
    )
    result = metadata._parse_anthropic_billing_header_text(text)
    assert result == {"cc_version": "1.0.23", "plan": "pro", "org_id": "org-1"}


def test_extract_anthropic_billing_header_fields_from_request_body() -> None:
    body = {
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.0.0; tier=enterprise",
            }
        ]
    }
    result = metadata._extract_anthropic_billing_header_fields_from_request_body(body)
    assert result == {"cc_version": "2.0.0", "tier": "enterprise"}


def test_extract_anthropic_billing_header_fields_empty() -> None:
    assert metadata._extract_anthropic_billing_header_fields_from_request_body({}) == {}
    assert metadata._extract_anthropic_billing_header_fields(None) == {}


def test_add_anthropic_billing_header_logging_metadata() -> None:
    body: dict[str, Any] = {"model": "claude"}
    fields = {"cc_version": "1.0.0", "plan": "pro"}
    result = metadata._add_anthropic_billing_header_logging_metadata(body, fields)
    lm = result["litellm_metadata"]
    assert "anthropic-billing-header" in lm["tags"]
    assert "anthropic-billing-header-key:cc_version" in lm["tags"]
    assert "anthropic-billing-header:cc_version=1.0.0" in lm["tags"]
    assert "anthropic-billing-header-key:plan" in lm["tags"]
    assert "anthropic-billing-header:plan=pro" in lm["tags"]
    assert lm["anthropic_billing_header_present"] is True
    assert lm["anthropic_billing_header_keys"] == ["cc_version", "plan"]
    assert lm["anthropic_billing_header_fields"] == fields


# ---------------------------------------------------------------------------
# Claude persisted-output logging metadata
# ---------------------------------------------------------------------------


def test_add_claude_persisted_output_logging_metadata_full() -> None:
    body: dict[str, Any] = {"model": "claude"}
    hooks = {"PreToolUse", "PostToolUse"}
    source_items = [
        {
            "path": "/tmp/out1.txt",
            "basename": "out1.txt",
            "content_hash": "abc123",
            "bytes": 42,
        }
    ]
    result = metadata._add_claude_persisted_output_logging_metadata(
        body, 3, hooks, source_items
    )
    lm = result["litellm_metadata"]
    assert "claude-persisted-output-expanded" in lm["tags"]
    assert "claude-persisted-output-hook:PostToolUse" in lm["tags"]
    assert "claude-persisted-output-hook:PreToolUse" in lm["tags"]
    assert lm["claude_persisted_output_expanded"] is True
    assert lm["claude_persisted_output_expanded_count"] == 3
    assert lm["claude_persisted_output_hooks"] == ["PostToolUse", "PreToolUse"]
    assert lm["claude_persisted_output_source_paths"] == ["/tmp/out1.txt"]
    assert lm["claude_persisted_output_source_basenames"] == ["out1.txt"]
    assert lm["claude_persisted_output_source_content_hashes"] == ["abc123"]
    assert lm["claude_persisted_output_source_bytes"] == [42]
    spans = lm["langfuse_spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == "claude.persisted_output_expand"
    assert spans[0]["metadata"]["expanded_count"] == 3
    assert spans[0]["metadata"]["hook_count"] == 2


def test_add_claude_persisted_output_logging_metadata_empty_hooks() -> None:
    body: dict[str, Any] = {}
    result = metadata._add_claude_persisted_output_logging_metadata(body, 1, set(), [])
    lm = result["litellm_metadata"]
    assert lm["tags"] == ["claude-persisted-output-expanded"]
    assert "claude_persisted_output_hooks" not in lm
    assert "claude_persisted_output_source_paths" not in lm


# ---------------------------------------------------------------------------
# Route-family logging metadata
# ---------------------------------------------------------------------------


def test_add_route_family_logging_metadata() -> None:
    body: dict[str, Any] = {"model": "x"}
    result = metadata._add_route_family_logging_metadata(body, "  Anthropic_Native ")
    lm = result["litellm_metadata"]
    assert lm["tags"] == ["route:anthropic_native"]
    assert lm["passthrough_route_family"] == "anthropic_native"


def test_add_route_family_logging_metadata_empty_returns_original() -> None:
    body: dict[str, Any] = {"model": "x"}
    assert metadata._add_route_family_logging_metadata(body, "  ") is body


# ---------------------------------------------------------------------------
# Callback ordering
# ---------------------------------------------------------------------------


def test_callback_ordering_in_prepare_request_body() -> None:
    """Verify session/repo extraction happens before trace-context and tool-def."""
    call_order: list[str] = []
    request = _FakeRequest({"x-session-id": "s-1"})
    body: dict[str, Any] = {"repository": "/home/user/projects/litellm"}

    original_session = metadata._extract_passthrough_session_id
    original_repo = metadata._extract_passthrough_repository
    original_trace_env = metadata._get_passthrough_trace_environment
    original_trace_ctx = metadata._add_passthrough_trace_context_metadata
    original_tool_def = metadata._add_passthrough_tool_definition_metadata

    def track_session(*a: Any, **kw: Any) -> Any:
        call_order.append("session_id")
        return original_session(*a, **kw)

    def track_repo(*a: Any, **kw: Any) -> Any:
        call_order.append("repository")
        return original_repo(*a, **kw)

    def track_trace_env(*a: Any, **kw: Any) -> Any:
        call_order.append("trace_environment")
        return original_trace_env(*a, **kw)

    def track_trace_ctx(*a: Any, **kw: Any) -> Any:
        call_order.append("trace_context")
        return original_trace_ctx(*a, **kw)

    def track_tool_def(*a: Any, **kw: Any) -> Any:
        call_order.append("tool_definition")
        return original_tool_def(*a, **kw)

    metadata._extract_passthrough_session_id = track_session  # type: ignore[assignment]
    metadata._extract_passthrough_repository = track_repo  # type: ignore[assignment]
    metadata._get_passthrough_trace_environment = track_trace_env  # type: ignore[assignment]
    metadata._add_passthrough_trace_context_metadata = track_trace_ctx  # type: ignore[assignment]
    metadata._add_passthrough_tool_definition_metadata = track_tool_def  # type: ignore[assignment]
    try:
        metadata._prepare_request_body_for_passthrough_observability(request, body)
    finally:
        metadata._extract_passthrough_session_id = original_session  # type: ignore[assignment]
        metadata._extract_passthrough_repository = original_repo  # type: ignore[assignment]
        metadata._get_passthrough_trace_environment = original_trace_env  # type: ignore[assignment]
        metadata._add_passthrough_trace_context_metadata = original_trace_ctx  # type: ignore[assignment]
        metadata._add_passthrough_tool_definition_metadata = original_tool_def  # type: ignore[assignment]

    assert call_order == [
        "session_id",
        "repository",
        "trace_environment",
        "trace_context",
        "tool_definition",
    ]


# ---------------------------------------------------------------------------
# _get_nested_str_value
# ---------------------------------------------------------------------------


def test_get_nested_str_value_json_string_traversal() -> None:
    source = {"metadata": '{"user_id": {"session_id": "nested-json"}}'}
    assert metadata._get_nested_str_value(source, ("metadata", "user_id", "session_id")) == "nested-json"
    assert metadata._get_nested_str_value(source, ("metadata", "missing")) is None
    assert metadata._get_nested_str_value("not-a-dict", ("a",)) is None
    assert metadata._get_nested_str_value({"a": "  "}, ("a",)) is None


# ---------------------------------------------------------------------------
# Walk budget
# ---------------------------------------------------------------------------


def test_walk_request_value_with_budget_respects_limits() -> None:
    deep: Any = "leaf"
    for _ in range(100):
        deep = {"child": deep}

    def visitor(node: object, _depth: int) -> str | None:
        if node == "leaf":
            return "found"
        return None

    result = metadata._walk_request_value_with_budget(
        deep, visitor=visitor, max_depth=10
    )
    assert result is None

    result2 = metadata._walk_request_value_with_budget(
        {"a": "target"}, visitor=lambda n, d: "hit" if n == "target" else None
    )
    assert result2 == "hit"
