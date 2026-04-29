import json
import os
import sys
import time
import traceback
from typing import Any
from unittest import mock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    BaseOpenAIPassThroughHandler,
    RouteChecks,
    _apply_google_adapter_completion_message_window,
    _apply_google_adapter_request_shape_policy,
    _apply_google_adapter_system_prompt_policy,
    _apply_google_code_assist_native_tool_aliases,
    _apply_openai_adapter_parallel_instruction_policy,
    _apply_openrouter_adapter_parallel_instruction_policy,
    _build_anthropic_responses_adapter_request_body,
    _build_completion_adapter_metadata,
    _build_google_code_assist_request_from_completion_kwargs,
    _collect_responses_response_from_stream,
    _compact_google_adapter_persisted_output_in_anthropic_request_body,
    _compact_openai_adapter_claude_context_in_anthropic_request_body,
    _get_google_adapter_rate_limit_key,
    _get_google_code_assist_prime_cache_key,
    _get_google_code_assist_prime_ttl_seconds,
    _get_gemini_passthrough_route_family,
    _get_openai_passthrough_route_family,
    _google_adapter_rate_limit_until_monotonic_by_key,
    _google_code_assist_prime_until_monotonic_by_key,
    _get_google_adapter_semaphore,
    _get_openrouter_adapter_hidden_retry_budget_seconds,
    _openrouter_adapter_failure_circuit_until_monotonic_by_key,
    _handle_anthropic_nvidia_completion_adapter_route,
    _perform_google_adapter_pass_through_request,
    _perform_openrouter_adapter_pass_through_request,
    _perform_openrouter_completion_adapter_operation,
    _prime_google_code_assist_session,
    _resolve_google_adapter_session_id,
    _set_google_adapter_cooldown,
    _wait_for_google_adapter_cooldown_if_needed,
    _expand_claude_persisted_output_in_anthropic_request_body,
    _extract_google_adapter_error_reason,
    _expand_claude_persisted_output_text,
    _prepare_anthropic_request_body_for_passthrough,
    _prepare_request_body_for_passthrough_observability,
    _handle_anthropic_google_completion_adapter_route,
    _iterate_responses_sse_events,
    _maybe_force_explicit_bash_tool_choice_for_completion_adapter,
    _maybe_force_explicit_bash_tool_choice_for_responses_adapter,
    _wrap_streaming_response_with_release_callback,
    anthropic_proxy_route,
    bedrock_llm_proxy_route,
    create_pass_through_route,
    cursor_proxy_route,
    gemini_proxy_route,
    llm_passthrough_factory_proxy_route,
    milvus_proxy_route,
    openai_proxy_route,
    vertex_discovery_proxy_route,
    vertex_proxy_route,
    vllm_proxy_route,
)
from litellm.proxy._types import ProxyException
from litellm.types.passthrough_endpoints.vertex_ai import VertexPassThroughCredentials
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


_CLAUDE_CODE_AGENT_PROJECT_TEXT = (
    "You are 'gpt5-5' and you are working on the 'aawm-tap' project.\n"
    "Return ok."
)


@pytest.fixture(autouse=True)
def clear_openrouter_adapter_failure_circuit_state():
    _openrouter_adapter_failure_circuit_until_monotonic_by_key.clear()
    yield
    _openrouter_adapter_failure_circuit_until_monotonic_by_key.clear()


def _build_claude_code_agent_project_request_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 32,
        "litellm_metadata": {
            "tags": ["existing-tag"],
            "trace_name": "claude-code",
        },
        "messages": [
            {
                "role": "user",
                "content": _CLAUDE_CODE_AGENT_PROJECT_TEXT,
            }
        ],
    }


async def _prepare_claude_code_agent_project_request_body(
    model: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    mock_request = MagicMock(spec=Request)
    mock_request.headers = headers or {}
    request_body = _build_claude_code_agent_project_request_body(model)
    updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
        mock_request,
        request_body,
    )
    return updated_body


def _assert_claude_code_agent_project_litellm_metadata(
    litellm_metadata: dict[str, Any],
) -> None:
    assert litellm_metadata["agent_name"] == "gpt5-5"
    assert litellm_metadata["aawm_claude_agent_name"] == "gpt5-5"
    assert litellm_metadata["tenant_id"] == "aawm-tap"
    assert litellm_metadata["aawm_tenant_id"] == "aawm-tap"
    assert litellm_metadata["aawm_claude_project"] == "aawm-tap"
    assert litellm_metadata["trace_user_id"] == "aawm-tap"
    assert litellm_metadata["trace_name"] == "claude-code.gpt5-5"
    assert "existing-tag" in litellm_metadata["tags"]
    assert "claude-agent:gpt5-5" in litellm_metadata["tags"]
    assert "claude-project:aawm-tap" in litellm_metadata["tags"]


@pytest.mark.asyncio
async def test_prepare_anthropic_request_body_uses_explicit_tenant_header_for_child_identity():
    updated_body = await _prepare_claude_code_agent_project_request_body(
        "gpt-5.5",
        headers={"x-aawm-tenant-id": "adapter-harness-tenant"},
    )

    litellm_metadata = updated_body["litellm_metadata"]
    assert litellm_metadata["agent_name"] == "gpt5-5"
    assert litellm_metadata["tenant_id"] == "adapter-harness-tenant"
    assert litellm_metadata["aawm_tenant_id"] == "adapter-harness-tenant"
    assert litellm_metadata["trace_user_id"] == "adapter-harness-tenant"
    assert litellm_metadata["aawm_claude_project"] == "aawm-tap"
    assert litellm_metadata["trace_name"] == "claude-code.gpt5-5"
    assert "claude-project:aawm-tap" in litellm_metadata["tags"]


@pytest.mark.asyncio
async def test_prepare_anthropic_request_body_overrides_orchestrator_trace_name_for_child():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"x-aawm-tenant-id": "adapter-harness-tenant"}
    request_body = _build_claude_code_agent_project_request_body("gpt-5.5")
    request_body["litellm_metadata"]["trace_name"] = "claude-code.orchestrator"

    updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
        mock_request,
        request_body,
    )

    litellm_metadata = updated_body["litellm_metadata"]
    assert litellm_metadata["source_trace_name"] == "claude-code.orchestrator"
    assert litellm_metadata["trace_name"] == "claude-code.gpt5-5"
    assert litellm_metadata["trace_user_id"] == "adapter-harness-tenant"


def test_anthropic_completion_adapter_preserves_trace_metadata():
    completion_kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=32,
            messages=[{"role": "user", "content": "Say ok"}],
            model="deepseek-ai/deepseek-v3.2",
            metadata={
                "existing_key": "existing-value",
                "session_id": "session-1",
                "trace_environment": "prod",
            },
            stream=True,
            extra_kwargs={
                "custom_llm_provider": litellm.LlmProviders.NVIDIA_NIM.value,
            },
        )
    )

    assert completion_kwargs["metadata"]["existing_key"] == "existing-value"
    assert completion_kwargs["metadata"]["session_id"] == "session-1"
    assert completion_kwargs["metadata"]["trace_environment"] == "prod"


def test_completion_adapter_metadata_prefers_child_trace_context():
    metadata = _build_completion_adapter_metadata(
        {
            "metadata": {
                "trace_name": "claude-code.orchestrator",
                "trace_user_id": "stale-user",
                "existing_key": "existing-value",
            },
            "litellm_metadata": {
                "trace_name": "claude-code.harness-nvidia",
                "trace_user_id": "adapter-harness-tenant",
                "source_trace_name": "claude-code.orchestrator",
                "agent_name": "harness-nvidia",
                "aawm_claude_agent_name": "harness-nvidia",
                "tenant_id": "adapter-harness-tenant",
                "aawm_tenant_id": "adapter-harness-tenant",
                "aawm_claude_project": "litellm",
            },
        }
    )

    assert metadata["existing_key"] == "existing-value"
    assert metadata["trace_name"] == "claude-code.harness-nvidia"
    assert metadata["trace_user_id"] == "adapter-harness-tenant"
    assert metadata["source_trace_name"] == "claude-code.orchestrator"
    assert metadata["agent_name"] == "harness-nvidia"
    assert metadata["aawm_claude_project"] == "litellm"


class TestResponsesAdapterToolChoice:
    def test_applies_openai_parallel_instruction_policy_for_multiple_function_tools(
        self,
    ):
        request_body = {
            "instructions": (
                "You are Claude Code, Anthropic's official CLI for Claude. "
                "Return findings directly."
            ),
            "parallel_tool_calls": True,
            "tools": [
                {"type": "function", "name": "Read", "parameters": {}},
                {"type": "function", "name": "Glob", "parameters": {}},
                {"type": "function", "name": "Grep", "parameters": {}},
                {"type": "web_search_preview"},
            ],
            "litellm_metadata": {
                "tags": ["existing-tag"],
                "langfuse_spans": [{"name": "existing.span"}],
            },
        }

        updated_body, changes = _apply_openai_adapter_parallel_instruction_policy(
            request_body
        )

        assert updated_body is not request_body
        assert updated_body["instructions"].startswith(
            "You are an OpenAI Responses function-calling agent for Claude Code."
        )
        assert (
            "emit all independent function calls together"
            in updated_body["instructions"]
        )
        assert "You are Claude Code, Anthropic's official CLI" not in updated_body[
            "instructions"
        ]
        assert changes["openai_adapter_parallel_instruction_policy_applied"] is True
        assert changes["openai_adapter_parallel_instruction_tool_names"] == [
            "Read",
            "Glob",
            "Grep",
        ]

        litellm_metadata = updated_body["litellm_metadata"]
        assert (
            litellm_metadata["openai_adapter_parallel_instruction_policy_applied"]
            is True
        )
        assert litellm_metadata[
            "openai_adapter_parallel_instruction_tool_names"
        ] == ["Read", "Glob", "Grep"]
        assert "existing-tag" in litellm_metadata["tags"]
        assert (
            "openai-adapter-parallel-instruction-policy"
            in litellm_metadata["tags"]
        )
        assert "openai-adapter-parallel-tool:read" in litellm_metadata["tags"]
        assert "openai-adapter-parallel-tool:glob" in litellm_metadata["tags"]
        assert "openai-adapter-parallel-tool:grep" in litellm_metadata["tags"]
        assert any(
            span["name"] == "openai_adapter.parallel_instruction_policy"
            for span in litellm_metadata["langfuse_spans"]
        )

    def test_skips_openai_parallel_instruction_policy_without_parallel_flag(self):
        request_body = {
            "instructions": "Preserve these instructions.",
            "parallel_tool_calls": False,
            "tools": [
                {"type": "function", "name": "Read", "parameters": {}},
                {"type": "function", "name": "Glob", "parameters": {}},
            ],
        }

        updated_body, changes = _apply_openai_adapter_parallel_instruction_policy(
            request_body
        )

        assert updated_body is request_body
        assert updated_body["instructions"] == "Preserve these instructions."
        assert changes == {}

    def test_skips_openai_parallel_instruction_policy_with_single_function_tool(self):
        request_body = {
            "instructions": "Preserve these instructions.",
            "parallel_tool_calls": True,
            "tools": [
                {"type": "function", "name": "Read", "parameters": {}},
                {"type": "web_search_preview"},
            ],
        }

        updated_body, changes = _apply_openai_adapter_parallel_instruction_policy(
            request_body
        )

        assert updated_body is request_body
        assert updated_body["instructions"] == "Preserve these instructions."
        assert changes == {}

    def test_skips_openai_parallel_instruction_policy_when_already_applied(self):
        request_body = {
            "instructions": (
                "You are Claude Code, Anthropic's official CLI for Claude. "
                "Return findings directly."
            ),
            "parallel_tool_calls": True,
            "tools": [
                {"type": "function", "name": "Read", "parameters": {}},
                {"type": "function", "name": "Glob", "parameters": {}},
            ],
        }
        updated_body, changes = _apply_openai_adapter_parallel_instruction_policy(
            request_body
        )
        assert changes["openai_adapter_parallel_instruction_policy_applied"] is True

        second_body, second_changes = _apply_openai_adapter_parallel_instruction_policy(
            updated_body
        )

        assert second_body is updated_body
        assert second_changes == {}

    def test_applies_openrouter_parallel_instruction_policy_with_openrouter_metadata(
        self,
    ):
        request_body = {
            "instructions": "You are Claude Code. Return findings directly.",
            "parallel_tool_calls": True,
            "tools": [
                {"type": "function", "name": "Read", "parameters": {}},
                {"type": "function", "name": "Glob", "parameters": {}},
            ],
        }

        updated_body, changes = _apply_openrouter_adapter_parallel_instruction_policy(
            request_body
        )

        assert updated_body["instructions"].startswith(
            "You are an OpenAI Responses function-calling agent for Claude Code."
        )
        assert (
            changes["openrouter_adapter_parallel_instruction_policy_applied"]
            is True
        )
        litellm_metadata = updated_body["litellm_metadata"]
        assert (
            litellm_metadata["openrouter_adapter_parallel_instruction_policy_applied"]
            is True
        )
        assert (
            "openrouter-adapter-parallel-instruction-policy"
            in litellm_metadata["tags"]
        )
        assert "openrouter-adapter-parallel-tool:read" in litellm_metadata["tags"]
        assert any(
            span["name"] == "openrouter_adapter.parallel_instruction_policy"
            for span in litellm_metadata["langfuse_spans"]
        )

    def test_responses_adapter_normalizes_empty_object_tool_schema(self):
        translated_body = _build_anthropic_responses_adapter_request_body(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "tools": [
                    {
                        "name": "mcp__mcppg__pg_alter_table",
                        "description": "Alters a table.",
                        "input_schema": {"type": "object"},
                    }
                ],
            },
            adapter_model="gpt-5.4",
        )

        assert translated_body["tools"][0]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_responses_adapter_normalizes_nested_object_schemas(self):
        translated_body = _build_anthropic_responses_adapter_request_body(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "tools": [
                    {
                        "name": "mcp__mcppg__pg_alter_table",
                        "description": "Alters a table.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "alter_columns": {
                                    "items": {"$ref": "#/$defs/_AlterColumnSpec"},
                                    "type": "array",
                                }
                            },
                            "$defs": {
                                "_AlterColumnSpec": {
                                    "type": "object",
                                    "additionalProperties": True,
                                }
                            },
                        },
                    }
                ],
            },
            adapter_model="gpt-5.4",
        )

        assert translated_body["tools"][0]["parameters"]["$defs"]["_AlterColumnSpec"] == {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
        }

    def test_responses_adapter_codex_defaults_alias_bash_to_exec_command(self):
        request_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Run pwd with Bash."}],
            "max_tokens": 32,
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }

        default_body = _build_anthropic_responses_adapter_request_body(
            request_body,
            adapter_model="gpt-5.4",
        )
        codex_body = _build_anthropic_responses_adapter_request_body(
            request_body,
            adapter_model="gpt-5.4",
            use_chatgpt_codex_defaults=True,
        )

        assert default_body["tools"][0]["name"] == "Bash"
        assert default_body["tool_choice"] == {"type": "function", "name": "Bash"}

        assert codex_body["tools"][0]["name"] == "exec_command"
        assert codex_body["tool_choice"] == {
            "type": "function",
            "name": "exec_command",
        }
        litellm_metadata = codex_body["litellm_metadata"]
        assert "anthropic-openai-codex-native-tools" in litellm_metadata["tags"]
        assert litellm_metadata["anthropic_adapter_codex_native_tool_aliases"] is True

    def test_forces_explicit_bash_tool_choice_when_prompt_requires_bash(self):
        translated_body = {
            "tools": [
                {"type": "function", "name": "Bash", "parameters": {}},
            ]
        }
        changes = _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Run the Bash command `date -u` exactly once using the Bash tool.",
                    }
                ]
            },
            translated_body,
        )

        assert translated_body["tool_choice"] == {"type": "function", "name": "Bash"}
        assert changes == {"forced_explicit_bash_tool_choice": "Bash"}

    def test_forces_explicit_bash_tool_choice_for_completion_adapter(self):
        request_body = {
            "tools": [
                {"name": "Bash", "input_schema": {}},
                {"name": "Read", "input_schema": {}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "Run the Bash command `date -u` exactly once using the Bash tool.",
                }
            ],
        }

        changes = _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
            request_body,
        )

        assert request_body["tool_choice"] == {"type": "tool", "name": "Bash"}
        assert changes == {"forced_explicit_bash_tool_choice": "Bash"}


class TestGoogleNativeToolAliases:
    def test_google_system_prompt_policy_replace_compact_preserves_project_and_safety(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY", "replace_compact"
        )
        completion_kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
                        "x-anthropic-billing-header: cc_version=2.1.119; cc_entrypoint=cli;\n\n"
                        "# Project\nPreserve workspace constraints and keep API keys secret.\n\n"
                        "IMPORTANT: Assist only with authorized security testing."
                    ),
                },
                {"role": "user", "content": "hello"},
            ],
            "metadata": {"tags": ["existing-tag"]},
        }

        updated_kwargs, changes = _apply_google_adapter_system_prompt_policy(
            completion_kwargs
        )

        system_text = updated_kwargs["messages"][0]["content"]
        metadata = updated_kwargs["metadata"]
        assert "You are a non-interactive CLI software engineering agent." in system_text
        assert "# Preserved Project And Safety Instructions" in system_text
        assert "You are Claude Code" not in system_text
        assert "x-anthropic-billing-header" not in system_text
        assert "Preserve workspace constraints and keep API keys secret." in system_text
        assert "IMPORTANT: Assist only with authorized security testing." in system_text
        assert "Final responses must include visible assistant text." in system_text
        assert changes["google_adapter_system_prompt_policy"] == "replace_compact"
        assert changes["google_adapter_system_prompt_removed_claude_overhead_chars"] > 0
        assert metadata["google_adapter_system_prompt_policy"] == "replace_compact"
        assert (
            metadata["google_adapter_system_prompt_policy_version"]
            == "2026-04-27.v2"
        )
        assert "google-adapter-system-prompt-policy:replace_compact" in metadata["tags"]
        assert "existing-tag" in metadata["tags"]

    def test_google_system_prompt_policy_rewrites_list_text_content(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY", "replace_compact"
        )
        completion_kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
                                "# Project\nKeep this repository constraint."
                            ),
                        },
                        {"type": "image", "source": {"type": "base64", "data": "x"}},
                        {"type": "text", "text": "x-anthropic-billing-header: cc"},
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }

        updated_kwargs, changes = _apply_google_adapter_system_prompt_policy(
            completion_kwargs
        )

        system_content = updated_kwargs["messages"][0]["content"]
        assert isinstance(system_content, list)
        assert system_content[0]["type"] == "text"
        assert (
            "You are a non-interactive CLI software engineering agent."
            in system_content[0]["text"]
        )
        assert "Keep this repository constraint." in system_content[0]["text"]
        assert system_content[1] == {"type": "image", "source": {"type": "base64", "data": "x"}}
        assert len(system_content) == 2
        assert changes["google_adapter_system_prompt_policy"] == "replace_compact"

    def test_google_system_prompt_policy_off_leaves_system_text_unchanged(
        self, monkeypatch
    ):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY", "off")
        original_system_text = "You are Claude Code.\n\n# Project\nKeep constraints."
        completion_kwargs = {
            "messages": [
                {"role": "system", "content": original_system_text},
                {"role": "user", "content": "hello"},
            ]
        }

        updated_kwargs, changes = _apply_google_adapter_system_prompt_policy(
            completion_kwargs
        )

        assert updated_kwargs["messages"][0]["content"] == original_system_text
        assert changes["google_adapter_system_prompt_policy"] == "off"
        assert updated_kwargs["metadata"][
            "google_adapter_system_prompt_policy_applied"
        ] is False

    def test_google_system_prompt_policy_append_keeps_original_for_rollout(
        self, monkeypatch
    ):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY", "append")
        original_system_text = "You are Claude Code.\n\n# Project\nKeep constraints."
        completion_kwargs = {
            "messages": [
                {"role": "system", "content": original_system_text},
                {"role": "user", "content": "hello"},
            ]
        }

        updated_kwargs, changes = _apply_google_adapter_system_prompt_policy(
            completion_kwargs
        )

        system_text = updated_kwargs["messages"][0]["content"]
        assert "You are a non-interactive CLI software engineering agent." in system_text
        assert "# Original Claude System Instructions" in system_text
        assert original_system_text in system_text
        assert changes["google_adapter_system_prompt_policy"] == "append"
        assert changes["google_adapter_system_prompt_removed_claude_overhead_chars"] == 0

    def test_apply_google_code_assist_native_tool_aliases(self):
        expected_aliases = {
            "Bash": "run_shell_command",
            "Read": "read_file",
            "Write": "write_file",
            "Edit": "replace",
            "Glob": "glob",
            "Grep": "grep_search",
            "WebFetch": "web_fetch",
            "WebSearch": "google_web_search",
        }
        completion_kwargs = {
            "tools": [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in expected_aliases
            ] + [
                {"type": "function", "function": {"name": "UnchangedTool", "parameters": {}}},
            ],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                        for index, name in enumerate(expected_aliases)
                    ],
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "WebSearch"}},
        }
        tool_name_mapping = {
            **{name: name for name in expected_aliases},
            "UnchangedTool": "UnchangedTool",
        }

        updated_kwargs, changes = _apply_google_code_assist_native_tool_aliases(
            completion_kwargs,
            tool_name_mapping,
        )

        function_names = [tool["function"]["name"] for tool in updated_kwargs["tools"]]
        assert function_names == [*expected_aliases.values(), "UnchangedTool"]
        tool_call_names = [call["function"]["name"] for call in updated_kwargs["messages"][0]["tool_calls"]]
        assert tool_call_names == list(expected_aliases.values())
        assert updated_kwargs["tool_choice"]["function"]["name"] == "google_web_search"
        for original_name, alias_name in expected_aliases.items():
            assert tool_name_mapping[alias_name] == original_name
        assert changes["google_native_tool_aliases"] == sorted(
            expected_aliases.values()
        )


class TestGoogleCodeAssistPrimeCache:
    def test_google_code_assist_prime_ttl_defaults_to_cli_cache_window(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_CODE_ASSIST_PRIME_TTL_SECONDS", raising=False)

        assert _get_google_code_assist_prime_ttl_seconds() == 30.0

    @pytest.mark.asyncio
    async def test_google_code_assist_prime_cache_skips_repeat_preflight(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_CODE_ASSIST_PRIME_TTL_SECONDS", "300")
        _google_code_assist_prime_until_monotonic_by_key.clear()

        mock_client = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = False

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.httpx.AsyncClient",
            return_value=mock_context,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.HttpPassThroughEndpointHelpers.validate_outgoing_egress",
        ):
            await _prime_google_code_assist_session("token-123", "project-123")
            await _prime_google_code_assist_session("token-123", "project-123")

        assert mock_client.post.await_count == 3
        cache_key = _get_google_code_assist_prime_cache_key("token-123", "project-123")
        assert _google_code_assist_prime_until_monotonic_by_key.get(cache_key, 0.0) > time.monotonic()


class TestGoogleOAuthFallbacks:
    def test_load_google_oauth_client_values_from_local_gemini_cli_bundle(
        self, tmp_path, monkeypatch
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _load_google_oauth_client_values_from_local_gemini_cli_bundle,
        )

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "chunk-auth.js").write_text(
            'var OAUTH_CLIENT_ID = "client-id-123";\n'
            'var OAUTH_CLIENT_SECRET = "client-secret-456";\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("LITELLM_GEMINI_CLI_BUNDLE_PATH", str(bundle_dir))
        monkeypatch.delenv("LITELLM_GEMINI_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("LITELLM_GEMINI_OAUTH_CLIENT_SECRET", raising=False)

        client_id, client_secret = (
            _load_google_oauth_client_values_from_local_gemini_cli_bundle()
        )

        assert client_id == "client-id-123"
        assert client_secret == "client-secret-456"

    @pytest.mark.asyncio
    async def test_refresh_local_google_oauth_credentials_falls_back_to_gemini_cli_bundle(
        self, tmp_path, monkeypatch
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _refresh_local_google_oauth_credentials,
        )

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "chunk-auth.js").write_text(
            'var OAUTH_CLIENT_ID = "client-id-123";\n'
            'var OAUTH_CLIENT_SECRET = "client-secret-456";\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("LITELLM_GEMINI_CLI_BUNDLE_PATH", str(bundle_dir))
        monkeypatch.delenv("LITELLM_GEMINI_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("LITELLM_GEMINI_OAUTH_CLIENT_SECRET", raising=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = httpx.Response(
            200,
            json={"access_token": "ya29.refreshed", "expires_in": 3600},
        )
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = False

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.httpx.AsyncClient",
            return_value=mock_context,
        ):
            refreshed = await _refresh_local_google_oauth_credentials(
                {"refresh_token": "refresh-token-123"}
            )

        assert refreshed["access_token"] == "ya29.refreshed"
        assert refreshed["refresh_token"] == "refresh-token-123"
        assert isinstance(refreshed["expiry_date"], int)
        assert mock_client.post.await_args.kwargs["data"] == {
            "client_id": "client-id-123",
            "client_secret": "client-secret-456",
            "refresh_token": "refresh-token-123",
            "grant_type": "refresh_token",
        }


class TestGoogleAdapterRequestShapePolicy:
    def test_clamps_large_max_output_tokens_and_removes_default_temperature(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP", raising=False)
        payload = {
            "request": {
                "generationConfig": {
                    "max_output_tokens": 32000,
                    "temperature": 1.0,
                    "top_p": 0.95,
                }
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes == {
            "injected_default_thinking_config": True,
            "injected_default_thinking_level": "low",
            "removed_oversized_max_output_tokens_from": 32000,
            "removed_oversized_max_output_tokens_cap": 8192,
            "removed_default_temperature": True,
        }
        assert payload["request"]["generationConfig"] == {
            "top_p": 0.95,
            "thinkingConfig": {"includeThoughts": False, "thinkingLevel": "low"},
        }
        assert changes["injected_default_thinking_config"] is True
        assert changes["injected_default_thinking_level"] == "low"

    def test_injects_default_thinking_config_for_google_adapter(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL", raising=False)
        payload = {"model": "gemini-3-flash-preview", "request": {}}

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["injected_default_thinking_config"] is True
        assert changes["injected_default_thinking_level"] == "low"
        assert payload["request"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": False,
            "thinkingLevel": "low",
        }

    def test_injects_minimal_default_thinking_config_for_flash_lite(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL", raising=False)
        payload = {"model": "gemini-3.1-flash-lite-preview", "request": {}}

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["injected_default_thinking_config"] is True
        assert changes["injected_default_thinking_level"] == "minimal"
        assert payload["request"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": False,
            "thinkingLevel": "minimal",
        }

    def test_keeps_existing_google_thinking_config(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG", raising=False)
        payload = {
            "model": "gemini-3.1-pro-preview",
            "request": {
                "generationConfig": {
                    "thinkingConfig": {"includeThoughts": False, "thinkingLevel": "high"}
                }
            },
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert "injected_default_thinking_config" not in changes
        assert payload["request"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": False,
            "thinkingLevel": "high",
        }

    def test_retains_bounded_followup_reminder_only_context(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_PURE_CONTEXT_TEXT_PART_CHAR_CAP", raising=False)
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP", "9000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP", "9000")
        reminder_text = "<system-reminder>" + ("x" * 7000) + "</system-reminder>"
        payload = {
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": reminder_text}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {"output": "ok"}}}]},
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        retained_text = payload["request"]["contents"][0]["parts"][0]["text"]
        assert len(retained_text) == 6000
        assert changes["retained_followup_reminder_only_context_count"] == 1
        assert changes["compacted_pure_context_text_parts_cap"] == 6000

    def test_compacts_subagent_context_more_aggressively_on_first_turn(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP", raising=False)
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP", "9000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", "9000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP", "9000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP", "9000")
        reminder_text = (
            "<system-reminder>\n"
            "SubagentStart hook additional context: cached context\n"
            + ("x" * 7000)
            + "\n</system-reminder>"
        )
        payload = {
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": reminder_text}]},
                    {"role": "user", "parts": [{"text": "Run date -u once."}]},
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        retained_text = payload["request"]["contents"][0]["parts"][0]["text"]
        assert len(retained_text) == 2000
        assert changes["subagent_context_text_parts_compacted_count"] == 1
        assert changes["subagent_context_text_parts_cap"] == 2000

    def test_compacts_subagent_context_more_aggressively_on_followup(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP", raising=False)
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP", "9000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP", "9000")
        reminder_text = (
            "<system-reminder>\n"
            "SubagentStart hook additional context: cached context\n"
            + ("x" * 7000)
            + "\n</system-reminder>"
        )
        payload = {
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": reminder_text}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {"output": "ok"}}}]},
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        retained_text = payload["request"]["contents"][0]["parts"][0]["text"]
        assert len(retained_text) == 1200
        assert changes["retained_followup_reminder_only_context_count"] == 1
        assert changes["subagent_context_text_parts_compacted_count"] == 1
        assert changes["subagent_context_text_parts_cap"] == 1200
        assert changes["compacted_pure_context_text_parts_cap"] == 1200

    def test_trims_followup_google_tools_to_core_set(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_ALLOWED_TOOL_NAMES", raising=False)
        payload = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "Bash",
                                    "response": {"output": "Mon Apr 20 10:30:47 UTC 2026"},
                                }
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "functionDeclarations": [
                            {"name": "Read"},
                            {"name": "Write"},
                            {"name": "Edit"},
                            {"name": "Glob"},
                            {"name": "Grep"},
                            {"name": "Bash"},
                            {"name": "WebSearch"},
                            {"name": "WebFetch"},
                            {"name": "mcp__aawm__search"},
                            {"name": "mcp__aawm__list_tasks"},
                        ]
                    }
                ],
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["trimmed_followup_function_declarations_from"] == 10
        assert changes["trimmed_followup_function_declarations_to"] == 8
        assert payload["request"]["tools"] == [
            {
                "functionDeclarations": [
                    {"name": "Read"},
                    {"name": "Write"},
                    {"name": "Edit"},
                    {"name": "Glob"},
                    {"name": "Grep"},
                    {"name": "Bash"},
                    {"name": "WebSearch"},
                    {"name": "WebFetch"},
                ]
            }
        ]

    def test_trims_followup_google_tools_to_core_set_with_native_aliases(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_ALLOWED_TOOL_NAMES", raising=False)
        payload = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "run_shell_command",
                                    "response": {"output": "Mon Apr 20 10:30:47 UTC 2026"},
                                }
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "functionDeclarations": [
                            {"name": "read_file"},
                            {"name": "write_file"},
                            {"name": "replace"},
                            {"name": "glob"},
                            {"name": "grep_search"},
                            {"name": "run_shell_command"},
                            {"name": "google_web_search"},
                            {"name": "web_fetch"},
                            {"name": "mcp__aawm__search"},
                            {"name": "mcp__aawm__list_tasks"},
                        ]
                    }
                ],
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["trimmed_followup_function_declarations_from"] == 10
        assert changes["trimmed_followup_function_declarations_to"] == 8
        assert payload["request"]["tools"] == [
            {
                "functionDeclarations": [
                    {"name": "read_file"},
                    {"name": "write_file"},
                    {"name": "replace"},
                    {"name": "glob"},
                    {"name": "grep_search"},
                    {"name": "run_shell_command"},
                    {"name": "google_web_search"},
                    {"name": "web_fetch"},
                ]
            }
        ]

    def test_trims_completion_messages_before_google_transform(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW", raising=False)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}-" + ("x" * 100)}
            for i in range(20)
        ]

        trimmed_messages, changes = _apply_google_adapter_completion_message_window(messages)

        assert changes == {
            "trimmed_completion_messages_from_count": 20,
            "trimmed_completion_messages_to_count": 12,
            "trimmed_completion_messages_from_text_chars": 2130,
            "trimmed_completion_messages_to_text_chars": 1282,
            "trimmed_completion_messages_max_window": 12,
        }
        assert len(trimmed_messages) == 12
        assert trimmed_messages[0]["content"].startswith("msg-8-")
        assert trimmed_messages[-1]["content"].startswith("msg-19-")

    def test_preserves_active_task_state_when_trimming_tool_followup_messages(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW", raising=False)
        task = (
            "<system-reminder>\nSubagentStart hook additional context: cached context\n"
            + ("context\n" * 200)
            + "</system-reminder>\n"
            "Run this numbered script exactly.\n"
            "6. Bash: call Bash exactly once with command exactly `date -u +%Y-%m-%dT%H:%M:%S.%NZ`.\n"
            "7. WebSearch: after step 6 Bash, the next and only valid tool call is WebSearch with query exactly `IANA example domain`.\n"
            "8. WebFetch: fetch https://example.com/.\n"
        )
        messages = [{"role": "user", "content": task}]
        for index, tool_name in enumerate(
            ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebSearch"]
        ):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": f"{tool_name} result",
                }
            )

        trimmed_messages, changes = _apply_google_adapter_completion_message_window(messages)

        preserved_text = trimmed_messages[0]["content"]
        assert len(trimmed_messages) == 11
        assert changes["preserved_active_task_state"] is True
        assert changes["preserved_active_task_state_source_index"] == 0
        assert changes["trimmed_completion_messages_tool_pair_boundary_adjustments"] == 1
        assert "Gemini adapter preserved active child-agent task state" in preserved_text
        assert "SubagentStart hook additional context" not in preserved_text
        assert "Run this numbered script exactly" in preserved_text
        assert "IANA example domain" in preserved_text
        assert trimmed_messages[1]["role"] == "assistant"
        assert trimmed_messages[1]["tool_calls"][0]["id"] == "call_2"
        assert trimmed_messages[2]["role"] == "tool"
        assert trimmed_messages[2]["tool_call_id"] == "call_2"
        assert trimmed_messages[-1]["content"] == "WebSearch result"

    def test_preserved_task_state_does_not_leave_orphan_tool_result(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW", raising=False)
        task = (
            "Run this numbered script exactly.\n"
            "6. Bash: call Bash exactly once with command exactly `date -u +%Y-%m-%dT%H:%M:%S.%NZ`.\n"
            "7. WebSearch: after step 6 Bash, the next and only valid tool call is WebSearch with query exactly `IANA example domain`.\n"
            "8. WebFetch: fetch https://example.com/.\n"
            "A final response immediately after Bash is invalid.\n"
        )
        messages = [{"role": "user", "content": task}]
        for index, tool_name in enumerate(
            ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
        ):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": f"{tool_name} result",
                }
            )

        trimmed_messages, changes = _apply_google_adapter_completion_message_window(messages)

        assert changes["preserved_active_task_state"] is True
        assert changes["trimmed_completion_messages_tool_pair_boundary_adjustments"] == 1
        assert len(trimmed_messages) == 11
        assert "IANA example domain" in trimmed_messages[0]["content"]
        assert trimmed_messages[1]["role"] == "assistant"
        assert trimmed_messages[1]["tool_calls"][0]["id"] == "call_1"
        assert trimmed_messages[2]["role"] == "tool"
        assert trimmed_messages[2]["tool_call_id"] == "call_1"
        assert trimmed_messages[-2]["tool_calls"][0]["function"]["name"] == "Bash"
        assert trimmed_messages[-1]["content"] == "Bash result"

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_preserves_task_state_before_transform(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW", raising=False)
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-xyz"}
        captured_messages = {}
        task = (
            "<system-reminder>\nSubagentStart hook additional context: cached context\n"
            + ("context\n" * 200)
            + "</system-reminder>\n"
            "Run this numbered script exactly.\n"
            "6. Bash: call Bash exactly once with command exactly `date -u +%Y-%m-%dT%H:%M:%S.%NZ`.\n"
            "7. WebSearch: after step 6 Bash, the next and only valid tool call is WebSearch with query exactly `IANA example domain`.\n"
            "8. WebFetch: fetch https://example.com/.\n"
        )
        messages = [{"role": "user", "content": task}]
        for index, tool_name in enumerate(
            ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebSearch"]
        ):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": f"{tool_name} result",
                }
            )

        def _capture_transform_request_body(*args, **kwargs):
            captured_messages["messages"] = kwargs["messages"]
            return {"contents": [{"role": "user", "parts": [{"text": "ok"}]}]}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=({"messages": messages, "max_tokens": 32}, {}),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            side_effect=_capture_transform_request_body,
        ):
            _, _, completion_messages, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        preserved_text = captured_messages["messages"][0]["content"]
        assert completion_messages[0]["content"] == preserved_text
        assert changes["preserved_active_task_state"] is True
        assert "Gemini adapter preserved active child-agent task state" in preserved_text
        assert "Run this numbered script exactly" in preserved_text
        assert "IANA example domain" in preserved_text

    def test_trims_large_contents_window_for_session_scoped_google_requests(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_WINDOW", raising=False)
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_TEXT_CHARS", raising=False)
        payload = {
            "request": {
                "session_id": "sess-123",
                "contents": [
                    {"parts": [{"text": f"block-{i}-" + ("x" * 1200)}]}
                    for i in range(30)
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes == {
            "injected_default_thinking_config": True,
            "injected_default_thinking_level": "low",
            "trimmed_contents_from_count": 30,
            "trimmed_contents_to_count": 9,
            "trimmed_contents_from_text_chars": 36260,
            "trimmed_contents_to_text_chars": 10881,
            "trimmed_contents_max_window": 24,
            "trimmed_contents_max_text_chars": 12000,
            "trimmed_contents_preserved_text_entries": 2,
        }
        assert len(payload["request"]["contents"]) == 9
        assert payload["request"]["contents"][0]["parts"][0]["text"].startswith("block-21-")
        assert payload["request"]["contents"][-1]["parts"][0]["text"].startswith("block-29-")
        assert payload["request"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": False,
            "thinkingLevel": "low",
        }

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_derives_user_prompt_id_from_prompt_not_session(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-abc"}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            side_effect=[
                ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}, {}),
                ({"messages": [{"role": "user", "content": "hi again"}], "max_tokens": 32}, {}),
            ],
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            return_value={},
        ):
            first_request, _, _, _, _, _ = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )
            second_request, _, _, _, _, _ = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "hi again"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert first_request["request"]["session_id"] == second_request["request"]["session_id"]
        assert first_request["user_prompt_id"] != second_request["user_prompt_id"]

    def test_google_adapter_session_id_scopes_direct_session_id_by_model(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "session_id": "parent-session-abc",
            "langfuse_trace_name": "claude-code.gemini-3-flash-preview",
        }

        session_id, source = _resolve_google_adapter_session_id(
            mock_request,
            [{"role": "user", "content": "Run date -u once."}],
            google_model="gemini-3-flash-preview",
        )

        assert source == "direct_session_id"
        assert session_id != "parent-session-abc"
        assert len(session_id) == 36

    def test_google_adapter_session_id_falls_back_to_agent_name_when_trace_name_missing(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "session_id": "parent-session-abc",
            "x-claude-code-session-id": "parent-session-abc",
        }

        session_id, source = _resolve_google_adapter_session_id(
            mock_request,
            [
                {
                    "role": "user",
                    "content": "You are 'gemini-3-flash-preview' and you are working on the 'aawm' project.\nRun date -u once.",
                }
            ],
            google_model="gemini-3-flash-preview",
        )

        assert source == "direct_session_id"
        assert session_id != "parent-session-abc"
        assert len(session_id) == 36

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_keeps_user_prompt_id_stable_across_followup_turns_with_same_trace_id(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "session_id": "header-session-abc",
            "langfuse_trace_id": "trace-123",
        }

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            side_effect=[
                ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}, {}),
                ({
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "date -u"}'}}]},
                        {"role": "tool", "tool_call_id": "call_1", "content": "Mon Apr 20 00:00:00 UTC 2026"},
                    ],
                    "max_tokens": 32,
                }, {}),
            ],
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            return_value={},
        ):
            first_request, _, _, _, _, _ = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )
            second_request, _, _, _, _, _ = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={
                    "max_tokens": 32,
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "date -u"}'}}]},
                        {"role": "tool", "tool_call_id": "call_1", "content": "Mon Apr 20 00:00:00 UTC 2026"},
                    ],
                },
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert first_request["user_prompt_id"] == second_request["user_prompt_id"]

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_leaves_empty_tool_call_text_empty(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-tools"}
        captured_messages = {}

        def _capture_transform_request_body(*args, **kwargs):
            captured_messages["messages"] = kwargs["messages"]
            return {}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=(
                {
                    "messages": [
                        {"role": "user", "content": "Run date -u once."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": '{"command":"date -u"}'},
                                }
                            ],
                        },
                    ],
                    "max_tokens": 32,
                },
                {},
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            side_effect=_capture_transform_request_body,
        ):
            _, _, completion_messages, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert "google_adapter_injected_tool_call_context_count" not in changes
        assert "google_adapter_suppressed_tool_call_context_text_count" not in changes
        assert completion_messages[1]["content"] == ""
        assert captured_messages["messages"][1]["content"] == ""

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_suppresses_synthetic_tool_call_text(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-tools"}
        captured_messages = {}

        def _capture_transform_request_body(*args, **kwargs):
            captured_messages["messages"] = kwargs["messages"]
            return {}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=(
                {
                    "messages": [
                        {"role": "user", "content": "Run date -u once."},
                        {
                            "role": "assistant",
                            "content": "Calling tool Bash.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": '{"command":"date -u"}'},
                                }
                            ],
                        },
                    ],
                    "max_tokens": 32,
                },
                {},
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            side_effect=_capture_transform_request_body,
        ):
            _, _, completion_messages, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert changes["google_adapter_suppressed_tool_call_context_text_count"] == 1
        assert completion_messages[1]["content"] == ""
        assert captured_messages["messages"][1]["content"] == ""

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_normalizes_httpx_part_keys(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-httpx"}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}, {}),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            return_value={
                "contents": [
                    {"role": "model", "parts": [{"function_call": {"name": "Bash", "args": {"command": "date -u"}}}]},
                    {"role": "user", "parts": [{"function_response": {"name": "Bash", "response": {"content": "ok"}}}]},
                ]
            },
        ):
            wrapped_request, _, _, _, _, _ = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        all_parts = [
            part
            for content in wrapped_request["request"]["contents"]
            for part in content.get("parts", [])
            if isinstance(part, dict)
        ]
        assert any("functionCall" in part for part in all_parts)
        assert not any("function_call" in part for part in all_parts)
        assert any("functionResponse" in part for part in all_parts)
        assert not any("function_response" in part for part in all_parts)

    def test_google_request_shape_policy_recompacts_followup_persisted_output_blocks(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP", "256")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP", "512")
        long_block = (
            "<system-reminder>\n"
            "SubagentStart hook additional context: <persisted-output>\n"
            + ("alpha line\n" * 600)
            + "</persisted-output>\n"
            "</system-reminder>\n"
        )
        payload = {
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": long_block}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {"command": "date -u"}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {"content": "ok"}}}]},
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        updated_text = payload["request"]["contents"][0]["parts"][0]["text"]
        assert changes["followup_persisted_output_compacted_count"] >= 1
        assert changes["followup_persisted_output_text_chars_after"] < changes["followup_persisted_output_text_chars_before"]
        assert len(updated_text) < len(long_block)
        assert "Gemini adapter compacted" in updated_text

    def test_google_request_shape_policy_splits_inline_context_and_prompt(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_OVERSIZED_TEXT_PART_CHAR_CAP", "6000")
        user_task = "Use Bash to run `date -u` exactly once and reply with exactly the timestamp it returns."
        payload = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "<system-reminder>\n"
                                    "SessionStart hook additional context: <persisted-output>\n"
                                    + ("A" * 3000)
                                    + "\n</persisted-output>\n"
                                    "</system-reminder>\n\n"
                                    f"{user_task}"
                                )
                            }
                        ],
                    }
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["split_inline_context_prompt_count"] == 1
        assert payload["request"]["contents"][1]["parts"][0]["text"] == user_task

    def test_google_request_shape_policy_compacts_oversized_single_user_text_part(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_OVERSIZED_TEXT_PART_CHAR_CAP", "4000")
        payload = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "<system-reminder>\n"
                                    "SessionStart hook additional context: <persisted-output>\n"
                                    + ("A" * 10000)
                                    + "\n</persisted-output>\n"
                                    "</system-reminder>\n\n"
                                    "Task tail: run `date -u` once and return exactly the output."
                                )
                            }
                        ],
                    }
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        updated_text = payload["request"]["contents"][0]["parts"][0]["text"]
        assert changes["split_inline_context_prompt_count"] == 1
        assert changes["followup_persisted_output_compacted_count"] >= 1
        assert len(updated_text) < 2000
        assert payload["request"]["contents"][1]["parts"][0]["text"] == "Task tail: run `date -u` once and return exactly the output."

    def test_google_request_shape_policy_aggressively_compacts_pure_context_blocks(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_OVERSIZED_TEXT_PART_CHAR_CAP", "6000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PURE_CONTEXT_TEXT_PART_CHAR_CAP", "1200")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP", "8000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP", "8000")
        payload = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "<system-reminder>\n"
                                    "SubagentStart hook additional context: first reminder block\n"
                                    + ("context line a\n" * 300)
                                    + "</system-reminder>\n"
                                    "<system-reminder>\n"
                                    "SessionStart hook additional context: second reminder block\n"
                                    + ("context line b\n" * 300)
                                    + "</system-reminder>\n"
                                )
                            }
                        ],
                    },
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {"command": "date -u"}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {"content": "ok"}}}]},
                ]
            }
        }

        changes = _apply_google_adapter_request_shape_policy(payload)

        assert changes["compacted_pure_context_text_parts_count"] == 1
        assert changes["retained_followup_reminder_only_context_count"] == 1
        assert changes["compacted_pure_context_text_parts_cap"] == 1200
        assert len(payload["request"]["contents"]) == 3
        assert len(payload["request"]["contents"][0]["parts"][0]["text"]) == 1200
        assert payload["request"]["contents"][1]["role"] == "model"
        assert payload["request"]["contents"][2]["role"] == "user"


    @pytest.mark.asyncio
    async def test_google_code_assist_builder_injects_fallback_text_context_when_contents_have_no_text(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FALLBACK_CONTEXT_CHAR_CAP", "500")
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-xyz"}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=(
                {
                    "messages": [
                        {"role": "user", "content": "Run date -u and return only the output."},
                        {"role": "assistant", "content": "I will call Bash."},
                    ],
                    "max_tokens": 32,
                },
                {},
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            return_value={
                "contents": [
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {}}}]},
                ]
            },
        ):
            wrapped_request, _, _, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert wrapped_request["request"]["contents"][0]["parts"][0]["text"] == (
            "Run date -u and return only the output.\n\nI will call Bash."
        )
        assert changes["inserted_fallback_text_context"] is True
        assert changes["inserted_fallback_text_context_sources"] == 2
        assert changes["inserted_fallback_text_context_chars"] == len(
            "Run date -u and return only the output.\n\nI will call Bash."
        )

    @pytest.mark.asyncio
    async def test_google_code_assist_fallback_excludes_synthetic_tool_call_text(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_FALLBACK_CONTEXT_CHAR_CAP", "500")
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-xyz"}

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs",
            return_value=(
                {
                    "messages": [
                        {"role": "user", "content": "Run date -u and return only the output."},
                        {
                            "role": "assistant",
                            "content": "Calling tool Bash.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                    "max_tokens": 32,
                },
                {},
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.GoogleAIStudioGeminiConfig.map_openai_params",
            return_value={},
        ), patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
            return_value={
                "contents": [
                    {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "Bash", "response": {}}}]},
                ]
            },
        ):
            wrapped_request, _, _, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
                completion_kwargs={"max_tokens": 32, "messages": [{"role": "user", "content": "ignored"}]},
                adapter_model="gemini-3-flash-preview",
                project="test-project",
                request=mock_request,
            )

        assert wrapped_request["request"]["contents"][0]["parts"][0]["text"] == (
            "Run date -u and return only the output."
        )
        assert changes["inserted_fallback_text_context"] is True
        assert changes["inserted_fallback_text_context_sources"] == 1
        assert changes["google_adapter_suppressed_tool_call_context_text_count"] == 1

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_applies_system_prompt_policy_before_transform(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY", "replace_compact"
        )
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-policy"}

        wrapped_request, tool_name_mapping, _, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
            completion_kwargs={
                "max_tokens": 32,
                "system": (
                    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
                    "# Project\nKeep workspace and safety constraints."
                ),
                "messages": [{"role": "user", "content": "Read the source file."}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    }
                ],
            },
            adapter_model="gemini-3-flash-preview",
            project="test-project",
            request=mock_request,
        )

        def _collect_text(value: Any) -> list[str]:
            if isinstance(value, dict):
                texts = [value["text"]] if isinstance(value.get("text"), str) else []
                for child in value.values():
                    texts.extend(_collect_text(child))
                return texts
            if isinstance(value, list):
                texts = []
                for child in value:
                    texts.extend(_collect_text(child))
                return texts
            return []

        system_text = "\n".join(
            _collect_text(wrapped_request["request"]["systemInstruction"])
        )
        function_names = [
            declaration["name"]
            for tool_entry in wrapped_request["request"]["tools"]
            for declaration in (
                tool_entry.get("functionDeclarations")
                or tool_entry.get("function_declarations")
                or []
            )
        ]

        assert "You are a non-interactive CLI software engineering agent." in system_text
        assert "You are Claude Code" not in system_text
        assert "Keep workspace and safety constraints." in system_text
        assert function_names == ["read_file"]
        assert tool_name_mapping["read_file"] == "Read"
        assert changes["google_adapter_system_prompt_policy"] == "replace_compact"
        assert (
            wrapped_request["litellm_metadata"]["google_adapter_system_prompt_policy"]
            == "replace_compact"
        )

    @pytest.mark.asyncio
    async def test_google_code_assist_builder_preserves_core_tool_alias_envelope(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "native-session-1"}

        wrapped_request, tool_name_mapping, completion_messages, _, _, changes = await _build_google_code_assist_request_from_completion_kwargs(
            completion_kwargs={
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "Use Bash and Read."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I will call tools."},
                            {
                                "type": "tool_use",
                                "id": "toolu_bash",
                                "name": "Bash",
                                "input": {"command": "date -u"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_read",
                                "name": "Read",
                                "input": {"file_path": "/tmp/a.txt"},
                            },
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": "Bash",
                        "description": "Run shell",
                        "input_schema": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                    {
                        "name": "Read",
                        "description": "Read file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                    {
                        "name": "Grep",
                        "description": "Search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"pattern": {"type": "string"}},
                            "required": ["pattern"],
                        },
                    },
                ],
                "tool_choice": {"type": "tool", "name": "Bash"},
            },
            adapter_model="gemini-3-flash-preview",
            project="test-project",
            request=mock_request,
        )

        request_payload = wrapped_request["request"]
        declarations = request_payload["tools"][0].get(
            "functionDeclarations"
        ) or request_payload["tools"][0].get("function_declarations")
        function_names = [declaration["name"] for declaration in declarations]
        model_parts = request_payload["contents"][1]["parts"]
        tool_call_names = [
            part["functionCall"]["name"] for part in model_parts if "functionCall" in part
        ]

        assert wrapped_request["model"] == "gemini-3-flash-preview"
        assert wrapped_request["project"] == "test-project"
        assert wrapped_request["user_prompt_id"] == "c0c2a79b-f32a-5a0b-be0f-13586c8dc09f"
        assert request_payload["session_id"] == "df8d0bd4-bce2-5ecf-8192-42ecfcb4e058"
        assert function_names == ["run_shell_command", "read_file", "grep_search"]
        assert request_payload["toolConfig"]["functionCallingConfig"] == {
            "mode": "ANY",
            "allowed_function_names": ["run_shell_command"],
        }
        assert tool_call_names == ["run_shell_command", "read_file"]
        assert completion_messages[1]["tool_calls"][0]["function"]["name"] == "run_shell_command"
        assert completion_messages[1]["tool_calls"][1]["function"]["name"] == "read_file"
        assert tool_name_mapping == {
            "run_shell_command": "Bash",
            "read_file": "Read",
            "grep_search": "Grep",
        }
        assert changes["google_native_tool_aliases"] == [
            "grep_search",
            "read_file",
            "run_shell_command",
        ]

    @pytest.mark.asyncio
    async def test_wrap_streaming_response_with_release_callback_releases_after_stream_completion(self):
        released = []

        async def body_iterator():
            yield b"chunk-1"
            yield b"chunk-2"

        response = StreamingResponse(body_iterator(), media_type="text/event-stream")
        wrapped = _wrap_streaming_response_with_release_callback(
            response,
            lambda: released.append("released"),
        )

        chunks = []
        async for chunk in wrapped.body_iterator:
            chunks.append(chunk)

        assert chunks == [b"chunk-1", b"chunk-2"]
        assert released == ["released"]

    def test_google_adapter_semaphore_is_shared_by_account_project_lane(
        self, monkeypatch
    ):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_CONCURRENT", "1")

        lane_key_a = _get_google_adapter_rate_limit_key(
            "gemini-3-flash-preview",
            access_token="token-123",
            companion_project="project-123",
        )
        lane_key_b = _get_google_adapter_rate_limit_key(
            "gemini-3.1-pro-preview",
            access_token="token-123",
            companion_project="project-123",
        )
        other_lane_key = _get_google_adapter_rate_limit_key(
            "gemini-3.1-pro-preview",
            access_token="token-456",
            companion_project="project-123",
        )

        same_lane_a = _get_google_adapter_semaphore(rate_limit_key=lane_key_a)
        same_lane_b = _get_google_adapter_semaphore(rate_limit_key=lane_key_b)
        other_lane = _get_google_adapter_semaphore(rate_limit_key=other_lane_key)

        assert lane_key_a == lane_key_b
        assert same_lane_a is same_lane_b
        assert same_lane_a is not other_lane

    @pytest.mark.asyncio
    async def test_google_adapter_cooldown_is_shared_by_account_project_lane(self):
        _google_adapter_rate_limit_until_monotonic_by_key.clear()
        shared_lane_key = _get_google_adapter_rate_limit_key(
            "gemini-3-flash-preview",
            access_token="token-123",
            companion_project="project-123",
        )
        same_lane_other_model_key = _get_google_adapter_rate_limit_key(
            "gemini-3.1-pro-preview",
            access_token="token-123",
            companion_project="project-123",
        )
        different_lane_key = _get_google_adapter_rate_limit_key(
            "gemini-3.1-pro-preview",
            access_token="token-456",
            companion_project="project-123",
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.time.monotonic",
            return_value=100.0,
        ):
            await _set_google_adapter_cooldown(shared_lane_key, 7.0)

        sleep_mock = AsyncMock()
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.time.monotonic",
            return_value=100.0,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.asyncio.sleep",
            new=sleep_mock,
        ):
            await _wait_for_google_adapter_cooldown_if_needed(different_lane_key)

        sleep_mock.assert_not_awaited()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.time.monotonic",
            return_value=100.0,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.asyncio.sleep",
            new=sleep_mock,
        ):
            await _wait_for_google_adapter_cooldown_if_needed(
                same_lane_other_model_key
            )

        sleep_mock.assert_awaited_once_with(7.0)

    @pytest.mark.asyncio
    async def test_google_adapter_request_retries_on_generic_429(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "1")

        class Generic429Error(Exception):
            def __init__(self):
                self.status_code = 429
                self.detail = b'{"error":{"message":"quota reset after 7s"}}'
                super().__init__("generic 429")

        first_error = Generic429Error()
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with('__default__', 8.0)

    @pytest.mark.asyncio
    async def test_google_adapter_request_strips_internal_wrapper_kwargs(self):
        successful_response = Response(
            content='{"ok": true}', media_type="application/json"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(return_value=successful_response),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ):
            result = await _perform_google_adapter_pass_through_request(
                request=MagicMock(),
                target="https://example.com",
                google_access_token="token-123",
                google_adapter_rate_limit_key="lane-123",
            )

        assert result is successful_response
        forwarded_kwargs = mock_pass_through.await_args.kwargs
        assert forwarded_kwargs["request"] is not None
        assert forwarded_kwargs["target"] == "https://example.com"
        assert "google_access_token" not in forwarded_kwargs
        assert "google_adapter_rate_limit_key" not in forwarded_kwargs

    @pytest.mark.asyncio
    async def test_google_adapter_request_stops_after_retry_budget(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "1")

        class Generic429Error(Exception):
            def __init__(self):
                self.status_code = 429
                self.detail = b'{"error":{"message":"quota reset after 4s"}}'
                super().__init__("generic 429")

        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[Generic429Error(), Generic429Error()]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            with pytest.raises(Generic429Error):
                await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with('__default__', 5.0)

    @pytest.mark.asyncio
    async def test_google_adapter_request_retries_on_proxy_exception_code(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "1")

        first_error = ProxyException(
            message="You have exhausted your capacity on this model. Your quota will reset after 6s.",
            type="None",
            param="None",
            code=429,
        )
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with('__default__', 7.0)

    @pytest.mark.asyncio
    async def test_google_adapter_request_retries_on_model_capacity_exhausted(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "0")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "2")

        first_error = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'{\n  \"error\": {\n    \"code\": 429,\n    \"message\": \"No capacity available for model gemini-3.1-pro-preview on the server\",\n    \"status\": \"RESOURCE_EXHAUSTED\",\n    \"details\": [\n      {\n        \"@type\": \"type.googleapis.com/google.rpc.ErrorInfo\",\n        \"reason\": \"MODEL_CAPACITY_EXHAUSTED\",\n        \"domain\": \"cloudcode-pa.googleapis.com\",\n        \"metadata\": {\n          \"model\": \"gemini-3.1-pro-preview\"\n        }\n      }\n    ]\n  }\n}\n'"""

        second_error = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        second_error.detail = first_error.detail

        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, second_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert result is successful_response
        assert mock_pass_through.await_count == 3
        assert [await_call.args for await_call in set_cooldown.await_args_list] == [('__default__', 6.0), ('__default__', 16.0)]


    def test_extract_google_adapter_error_reason_parses_list_wrapped_payload(self):
        exc = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        exc.detail = """429: b'[{\n  "error": {\n    "code": 429,\n    "message": "No capacity available for model gemini-3.1-pro-preview on the server",\n    "status": "RESOURCE_EXHAUSTED",\n    "details": [\n      {\n        "@type": "type.googleapis.com/google.rpc.ErrorInfo",\n        "reason": "MODEL_CAPACITY_EXHAUSTED",\n        "domain": "cloudcode-pa.googleapis.com",\n        "metadata": {\n          "model": "gemini-3.1-pro-preview"\n        }\n      }\n    ]\n  }\n}]'"""

        assert _extract_google_adapter_error_reason(exc) == "MODEL_CAPACITY_EXHAUSTED"

    @pytest.mark.asyncio
    async def test_google_adapter_request_retries_on_list_wrapped_model_capacity_exhausted(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "0")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "2")

        first_error = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'[{\n  "error": {\n    "code": 429,\n    "message": "No capacity available for model gemini-3.1-pro-preview on the server",\n    "status": "RESOURCE_EXHAUSTED",\n    "details": [\n      {\n        "@type": "type.googleapis.com/google.rpc.ErrorInfo",\n        "reason": "MODEL_CAPACITY_EXHAUSTED",\n        "domain": "cloudcode-pa.googleapis.com",\n        "metadata": {\n          "model": "gemini-3.1-pro-preview"\n        }\n      }\n    ]\n  }\n}]'"""

        second_error = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        second_error.detail = first_error.detail

        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, second_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert result is successful_response
        assert mock_pass_through.await_count == 3
        assert [await_call.args for await_call in set_cooldown.await_args_list] == [('__default__', 6.0), ('__default__', 16.0)]

    @pytest.mark.asyncio
    async def test_google_adapter_request_uses_hidden_retry_budget_after_capacity_limit(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "0")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "0")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_BACKOFF_SECONDS", "2,4")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "7")

        first_error = ProxyException(
            message="No capacity available for model gemini-3.1-pro-preview on the server",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'[{\n  "error": {\n    "code": 429,\n    "message": "No capacity available for model gemini-3.1-pro-preview on the server",\n    "status": "RESOURCE_EXHAUSTED",\n    "details": [\n      {\n        "@type": "type.googleapis.com/google.rpc.ErrorInfo",\n        "reason": "MODEL_CAPACITY_EXHAUSTED",\n        "domain": "cloudcode-pa.googleapis.com",\n        "metadata": {\n          "model": "gemini-3.1-pro-preview"\n        }\n      }\n    ]\n  }\n}]'"""

        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(request=MagicMock())

        assert result is successful_response
        assert mock_pass_through.await_count == 3
        assert [await_call.args for await_call in set_cooldown.await_args_list] == [('__default__', 3.0), ('__default__', 5.0)]

    @pytest.mark.asyncio
    async def test_google_adapter_request_uses_upstream_retry_after_header(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "1")

        first_error = ProxyException(
            message="quota throttled",
            type="None",
            param="None",
            code=429,
        )
        first_error.upstream_headers = {"Retry-After": "9"}
        successful_response = Response(
            content='{"ok": true}', media_type="application/json"
        )
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_google_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_google_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_google_adapter_pass_through_request(
                request=MagicMock()
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        assert (
            mock_pass_through.await_args_list[0].kwargs[
                "retryable_upstream_status_codes"
            ]
            == [429]
        )
        set_cooldown.assert_awaited_once_with("__default__", 10.0)


class TestPassThroughRequestRetryableFailures:
    @pytest.mark.asyncio
    async def test_pass_through_request_preserves_retry_headers_and_skips_failure_hook(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.example.com/v1/test"
        upstream_response = httpx.Response(
            status_code=429,
            headers={"Retry-After": "17", "X-RateLimit-Remaining": "0"},
            content=b'{"error":"throttled"}',
            request=httpx.Request("POST", target_url),
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"test": "data"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=AsyncMock(return_value=upstream_response),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.verbose_proxy_logger.exception"
        ) as mock_log_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(return_value={"test": "data"})
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    stream=False,
                    retryable_upstream_status_codes=[429],
                )

        assert exc_info.value.code == "429"
        assert exc_info.value.detail == '{"error":"throttled"}'
        normalized_upstream_headers = {
            key.lower(): value
            for key, value in exc_info.value.upstream_headers.items()
        }
        assert normalized_upstream_headers["retry-after"] == "17"
        assert normalized_upstream_headers["x-ratelimit-remaining"] == "0"
        mock_logging_obj.post_call_failure_hook.assert_not_awaited()
        mock_log_exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_pass_through_request_skips_failure_hook_for_adapter_managed_502(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        target_url = "https://api.example.com/v1/test"
        upstream_response = httpx.Response(
            status_code=502,
            content=b'{"error":"provider failed"}',
            request=httpx.Request("POST", target_url),
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"test": "data"},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new=AsyncMock(return_value=upstream_response),
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.verbose_proxy_logger.exception"
        ) as mock_log_exception:
            mock_client_obj = MagicMock()
            mock_client_obj.client = MagicMock()
            mock_get_client.return_value = mock_client_obj
            mock_logging_obj.pre_call_hook = AsyncMock(return_value={"test": "data"})
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=target_url,
                    custom_headers={"authorization": "Bearer test"},
                    user_api_key_dict=MagicMock(),
                    stream=False,
                    retryable_upstream_status_codes=[429, 500, 502, 503, 504],
                )

        assert exc_info.value.code == "502"
        assert exc_info.value.detail == '{"error":"provider failed"}'
        mock_logging_obj.post_call_failure_hook.assert_not_awaited()
        mock_log_exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_pass_through_request_normalizes_openai_function_tool_schemas(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}

        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(
            return_value=[b'{"result": "success"}']
        )
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        custom_body = {
            "model": "gpt-5.4",
            "tools": [
                {
                    "type": "function",
                    "name": "mcp__mcppg__pg_alter_table",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "function",
                    "function": {
                        "name": "nested_tool",
                        "parameters": {},
                    },
                },
            ],
            "litellm_metadata": {},
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new_callable=AsyncMock,
        ) as mock_success_handler:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            await pass_through_request(
                request=mock_request,
                target="https://api.openai.com/v1/responses",
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=custom_body,
                stream=False,
            )

            sent_json = mock_client.request.call_args.kwargs["json"]
            assert sent_json["tools"][0]["parameters"] == {
                "type": "object",
                "properties": {},
            }
            assert sent_json["tools"][1]["function"]["parameters"] == {
                "type": "object",
                "properties": {},
            }
            assert "litellm_metadata" not in sent_json

            mock_success_handler.assert_called_once()
            success_kwargs = mock_success_handler.call_args.kwargs
            assert (
                success_kwargs["litellm_params"]["metadata"][
                    "openai_function_tool_schema_fix_count"
                ]
                == 2
            )


class TestOpenRouterAdapterRetry:
    @pytest.mark.asyncio
    async def test_openrouter_adapter_request_retries_on_proxy_exception_code(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly.","provider_name":"Stealth","is_byok":false}},"user_id":"user_test"}'"""
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_adapter_pass_through_request(
                adapter_model="google/gemma-4-31b-it:free",
                request=MagicMock(),
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 2.0)

    @pytest.mark.asyncio
    async def test_openrouter_adapter_request_uses_upstream_retry_after_header(
        self, monkeypatch
    ):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "45")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        first_error.upstream_headers = {"Retry-After": "19"}
        successful_response = Response(
            content='{"ok": true}', media_type="application/json"
        )
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_adapter_pass_through_request(
                adapter_model="google/gemma-4-31b-it:free",
                request=MagicMock(),
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        assert (
            mock_pass_through.await_args_list[0].kwargs[
                "retryable_upstream_status_codes"
            ]
            == [429, 500, 502, 503, 504]
        )
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 20.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_model",
        [
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "minimax/minimax-m2.5:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "qwen/qwen3-coder:free",
        ],
    )
    async def test_openrouter_adapter_request_retries_on_proxy_exception_code_for_free_models(self, monkeypatch, adapter_model):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        detail = (
            "429: b'{\"error\":{\"message\":\"Provider returned error\",\"code\":429,\"metadata\":{\"raw\":\""
            + adapter_model
            + " is temporarily rate-limited upstream. Please retry shortly.\",\"provider_name\":\"Stealth\",\"is_byok\":false}},\"user_id\":\"user_test\"}'"
        )
        first_error.detail = detail
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_adapter_pass_through_request(
                adapter_model=adapter_model,
                request=MagicMock(),
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with(adapter_model, 2.0)

    def test_openrouter_free_model_info_has_zero_cost(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        pricing_map_paths = [
            repo_root / "model_prices_and_context_window.json",
            repo_root
            / "litellm"
            / "bundled_model_prices_and_context_window_fallback.json",
        ]

        for pricing_map_path in pricing_map_paths:
            pricing_map = json.loads(pricing_map_path.read_text())
            model_info = pricing_map["openrouter/free"]
            prefixed_model_info = pricing_map["openrouter/openrouter/free"]

            assert "openrouter/elephant-alpha" not in pricing_map
            assert "openrouter/openrouter/elephant-alpha" not in pricing_map
            assert model_info["litellm_provider"] == "openrouter"
            assert model_info["input_cost_per_token"] == 0
            assert model_info["output_cost_per_token"] == 0
            assert model_info["max_input_tokens"] == 200000
            assert model_info["max_tokens"] == 200000
            assert prefixed_model_info["input_cost_per_token"] == 0
            assert prefixed_model_info["output_cost_per_token"] == 0

    @pytest.mark.asyncio
    async def test_openrouter_adapter_request_raises_after_retry_budget(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly.","provider_name":"Stealth","is_byok":false}},"user_id":"user_test"}'"""
        second_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        second_error.detail = first_error.detail
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, second_error]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            with pytest.raises(ProxyException):
                await _perform_openrouter_adapter_pass_through_request(
                    adapter_model="google/gemma-4-31b-it:free",
                    request=MagicMock(),
                )

        assert mock_pass_through.await_count == 2
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 2.0)

    @pytest.mark.asyncio
    async def test_openrouter_adapter_request_fast_fails_when_failure_circuit_open(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS", "300")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = """429: b'{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly.","provider_name":"Venice","is_byok":false}},"user_id":"user_test"}'"""
        second_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        second_error.detail = first_error.detail
        set_cooldown = AsyncMock()
        _openrouter_adapter_failure_circuit_until_monotonic_by_key.clear()

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
                new=AsyncMock(side_effect=[first_error, second_error]),
            ) as mock_pass_through, patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
                new=AsyncMock(),
            ), patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
                new=set_cooldown,
            ):
                with pytest.raises(ProxyException):
                    await _perform_openrouter_adapter_pass_through_request(
                        adapter_model="qwen/qwen3-coder:free",
                        request=MagicMock(),
                    )

                with pytest.raises(HTTPException) as exc_info:
                    await _perform_openrouter_adapter_pass_through_request(
                        adapter_model="qwen/qwen3-coder:free",
                        request=MagicMock(),
                    )

            assert mock_pass_through.await_count == 2
            assert exc_info.value.status_code == 429
            assert "cooling down" in str(exc_info.value.detail)
        finally:
            _openrouter_adapter_failure_circuit_until_monotonic_by_key.clear()

    @pytest.mark.asyncio
    async def test_openrouter_completion_adapter_operation_retries_on_rate_limit_error_string(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")

        class FakeRateLimitError(Exception):
            def __str__(self) -> str:
                return (
                    'litellm.RateLimitError: RateLimitError: OpenrouterException - '
                    '{"error":{"message":"Provider returned error","code":429,'
                    '"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly.",'
                    '"provider_name":"Stealth","is_byok":false}},"user_id":"user_test"}'
                )

        operation = AsyncMock(side_effect=[FakeRateLimitError(), "ok"])
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_completion_adapter_operation(
                adapter_model="google/gemma-4-31b-it:free",
                operation=operation,
            )

        assert result == "ok"
        assert operation.await_count == 2
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 2.0)

    @pytest.mark.asyncio
    async def test_openrouter_adapter_request_fails_fast_on_long_window_rate_limit(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "3")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2,4,8,12")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "45")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        reset_ms = int((time.time() + 3600) * 1000)
        first_error.detail = json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded: free-models-per-day-stealth. ",
                    "code": 429,
                    "metadata": {
                        "headers": {
                            "X-RateLimit-Limit": "1000",
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_ms),
                        }
                    },
                },
                "user_id": "user_test",
            }
        )
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            with pytest.raises(ProxyException):
                await _perform_openrouter_adapter_pass_through_request(
                    adapter_model="google/gemma-4-31b-it:free",
                    request=MagicMock(),
                )

        assert mock_pass_through.await_count == 1
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 300.0)

    @pytest.mark.asyncio
    async def test_openrouter_free_model_request_uses_retry_after_and_shared_cooldown(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "45")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        first_error.detail = json.dumps(
            {
                "error": {
                    "message": "Provider returned error",
                    "code": 429,
                    "metadata": {
                        "raw": "qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly.",
                        "provider_name": "Venice",
                        "retry_after_seconds": 31,
                    },
                },
                "user_id": "user_test",
            }
        )
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_adapter_pass_through_request(
                adapter_model="qwen/qwen3-coder:free",
                request=MagicMock(),
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        cooldown_keys, cooldown_wait = set_cooldown.await_args.args
        assert cooldown_keys == "qwen/qwen3-coder:free"
        assert cooldown_wait == 32.0

    @pytest.mark.asyncio
    async def test_openrouter_free_model_request_uses_reset_window_and_shared_cooldown(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "45")

        first_error = ProxyException(
            message="Provider returned error",
            type="None",
            param="None",
            code=429,
        )
        reset_ms = int((time.time() + 18) * 1000)
        first_error.detail = json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded: free-models-per-minute. ",
                    "code": 429,
                    "metadata": {
                        "headers": {
                            "X-RateLimit-Limit": "20",
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_ms),
                        }
                    },
                },
                "user_id": "user_test",
            }
        )
        successful_response = Response(content='{"ok": true}', media_type="application/json")
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(side_effect=[first_error, successful_response]),
        ) as mock_pass_through, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_adapter_pass_through_request(
                adapter_model="google/gemma-4-31b-it:free",
                request=MagicMock(),
            )

        assert result is successful_response
        assert mock_pass_through.await_count == 2
        cooldown_keys, cooldown_wait = set_cooldown.await_args.args
        assert cooldown_keys == "google/gemma-4-31b-it:free"
        assert 18.0 <= cooldown_wait <= 60.0

    @pytest.mark.asyncio
    async def test_openrouter_completion_adapter_operation_fails_fast_on_long_window_rate_limit(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "3")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2,4,8,12")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "45")

        reset_ms = int((time.time() + 3600) * 1000)
        payload_text = json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded: free-models-per-day-stealth. ",
                    "code": 429,
                    "metadata": {
                        "headers": {
                            "X-RateLimit-Limit": "1000",
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_ms),
                        }
                    },
                },
                "user_id": "user_test",
            }
        )

        class FakeRateLimitError(Exception):
            def __str__(self) -> str:
                return (
                    'litellm.RateLimitError: RateLimitError: OpenrouterException - '
                    + payload_text
                )

        operation = AsyncMock(side_effect=[FakeRateLimitError()])
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            with pytest.raises(FakeRateLimitError):
                await _perform_openrouter_completion_adapter_operation(
                    adapter_model="google/gemma-4-31b-it:free",
                    operation=operation,
                )

        assert operation.await_count == 1
        set_cooldown.assert_awaited_once_with("google/gemma-4-31b-it:free", 300.0)

    @pytest.mark.asyncio
    async def test_openrouter_completion_adapter_uses_hidden_retry_budget_beyond_attempt_cap(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2,10,20")
        monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "15")

        class FakeRateLimitError(Exception):
            def __str__(self) -> str:
                return (
                    'litellm.RateLimitError: RateLimitError: OpenrouterException - '
                    '{"error":{"message":"Provider returned error","code":429,'
                    '"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly.",'
                    '"provider_name":"Stealth","is_byok":false}},"user_id":"user_test"}'
                )

        operation = AsyncMock(side_effect=[FakeRateLimitError(), FakeRateLimitError(), "ok"])
        set_cooldown = AsyncMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._wait_for_openrouter_adapter_cooldown_if_needed",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._set_openrouter_adapter_cooldown",
            new=set_cooldown,
        ):
            result = await _perform_openrouter_completion_adapter_operation(
                adapter_model="google/gemma-4-31b-it:free",
                operation=operation,
            )

        assert result == "ok"
        assert operation.await_count == 3
        assert [await_call.args for await_call in set_cooldown.await_args_list] == [
            ("google/gemma-4-31b-it:free", 2.0),
            ("google/gemma-4-31b-it:free", 10.0),
        ]


class TestBaseOpenAIPassThroughHandler:
    def test_join_url_paths(self):
        print("\nTesting _join_url_paths method...")

        # Test joining base URL with no path and a path
        base_url = httpx.URL("https://api.example.com")
        path = "/v1/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with no path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test joining base URL with path and another path
        base_url = httpx.URL("https://api.example.com/v1")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with path not starting with slash
        base_url = httpx.URL("https://api.example.com/v1")
        path = "chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Path without leading slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with base URL having trailing slash
        base_url = httpx.URL("https://api.example.com/v1/")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with trailing slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"


def test_build_completion_adapter_metadata_overrides_adapter_owned_keys() -> None:
    metadata = _build_completion_adapter_metadata(
        {
            "metadata": {
                "existing_key": "existing-value",
                "trace_environment": "caller-env",
                "passthrough_route_family": "stale-route",
                "anthropic_adapter_model": "stale-model",
                "anthropic_adapter_original_model": "stale-original",
                "anthropic_adapter_target_endpoint": "stale-target",
                "langfuse_spans": [{"name": "stale.span"}],
                "tags": ["caller-tag", "shared-tag"],
            },
            "litellm_metadata": {
                "trace_environment": "prod",
                "passthrough_route_family": "anthropic_nvidia_completion_adapter",
                "anthropic_adapter_model": "deepseek-ai/deepseek-v3.2",
                "anthropic_adapter_original_model": "nvidia/deepseek-ai/deepseek-v3.2",
                "anthropic_adapter_target_endpoint": "nvidia:/v1/chat/completions",
                "langfuse_spans": [{"name": "anthropic.nvidia_completion_adapter"}],
                "tags": ["shared-tag", "anthropic-nvidia-completion-adapter"],
            },
        }
    )

    assert metadata["existing_key"] == "existing-value"
    assert metadata["trace_environment"] == "caller-env"
    assert metadata["passthrough_route_family"] == "anthropic_nvidia_completion_adapter"
    assert metadata["anthropic_adapter_model"] == "deepseek-ai/deepseek-v3.2"
    assert (
        metadata["anthropic_adapter_original_model"]
        == "nvidia/deepseek-ai/deepseek-v3.2"
    )
    assert (
        metadata["anthropic_adapter_target_endpoint"]
        == "nvidia:/v1/chat/completions"
    )
    assert metadata["langfuse_spans"] == [
        {"name": "anthropic.nvidia_completion_adapter"}
    ]
    assert metadata["tags"] == [
        "caller-tag",
        "shared-tag",
        "anthropic-nvidia-completion-adapter",
    ]


class TestAnthropicAdapterClaudeCodeAgentProjectMetadata:
    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_preserves_agent_project_context_in_litellm_metadata(
        self,
    ):
        prepared_body = await _prepare_claude_code_agent_project_request_body(
            "claude-opus-4-6"
        )

        litellm_metadata = prepared_body["litellm_metadata"]
        _assert_claude_code_agent_project_litellm_metadata(litellm_metadata)
        assert litellm_metadata["passthrough_route_family"] == "anthropic_messages"
        assert "route:anthropic_messages" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_openai_responses_adapter_preserves_agent_project_litellm_metadata(
        self,
    ):
        prepared_body = await _prepare_claude_code_agent_project_request_body(
            "gpt-5.4-mini"
        )

        translated_body = _build_anthropic_responses_adapter_request_body(
            prepared_body,
            adapter_model="gpt-5.4-mini",
        )

        litellm_metadata = translated_body["litellm_metadata"]
        _assert_claude_code_agent_project_litellm_metadata(litellm_metadata)
        assert (
            litellm_metadata["passthrough_route_family"]
            == "anthropic_openai_responses_adapter"
        )
        assert "route:anthropic_messages" in litellm_metadata["tags"]
        assert "route:anthropic_openai_responses_adapter" in litellm_metadata["tags"]
        assert "anthropic-openai-responses-adapter" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_openrouter_responses_adapter_preserves_agent_project_litellm_metadata(
        self,
    ):
        prepared_body = await _prepare_claude_code_agent_project_request_body(
            "openrouter/google/gemma-4-31b-it:free"
        )

        translated_body = _build_anthropic_responses_adapter_request_body(
            prepared_body,
            adapter_model="google/gemma-4-31b-it:free",
            route_family="anthropic_openrouter_responses_adapter",
            tag_prefix="anthropic-openrouter-responses-adapter",
            span_name="anthropic.openrouter_responses_adapter",
            target_endpoint="openrouter:/v1/responses",
        )

        litellm_metadata = translated_body["litellm_metadata"]
        _assert_claude_code_agent_project_litellm_metadata(litellm_metadata)
        assert (
            litellm_metadata["passthrough_route_family"]
            == "anthropic_openrouter_responses_adapter"
        )
        assert "route:anthropic_messages" in litellm_metadata["tags"]
        assert (
            "route:anthropic_openrouter_responses_adapter"
            in litellm_metadata["tags"]
        )
        assert "anthropic-openrouter-responses-adapter" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_google_completion_adapter_preserves_agent_project_litellm_metadata(
        self,
    ):
        prepared_body = await _prepare_claude_code_agent_project_request_body(
            "gemini-3.1-pro-preview"
        )
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}

        translated_response = {
            "id": "chatcmpl-google-agent-project",
            "object": "chat.completion",
            "created": 1744974432,
            "model": "gemini-3.1-pro-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            new=AsyncMock(return_value="ya29.test-google-token"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_or_load_google_code_assist_project",
            new=AsyncMock(return_value="project_123"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prime_google_code_assist_session",
            new=AsyncMock(),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._build_google_code_assist_request_from_completion_kwargs",
            new=AsyncMock(
                return_value=(
                    {
                        "model": "gemini-3.1-pro-preview",
                        "project": "project_123",
                        "user_prompt_id": "prompt-123",
                        "request": {"contents": [{"parts": [{"text": "ok"}]}]},
                    },
                    {},
                    [],
                    {},
                    {},
                    {},
                )
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._perform_google_adapter_pass_through_request",
            new=AsyncMock(
                return_value=StreamingResponse(
                    iter([b"data: [DONE]\n\n"]),
                    media_type="text/event-stream",
                )
            ),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._collect_google_code_assist_response_from_stream",
            new=AsyncMock(return_value=translated_response),
        ):
            await _handle_anthropic_google_completion_adapter_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(),
                prepared_request_body=prepared_body,
                adapter_model="gemini-3.1-pro-preview",
            )

        litellm_metadata = mock_pass_through_request.await_args.kwargs["custom_body"][
            "litellm_metadata"
        ]
        _assert_claude_code_agent_project_litellm_metadata(litellm_metadata)
        assert (
            litellm_metadata["passthrough_route_family"]
            == "anthropic_google_completion_adapter"
        )
        assert "route:anthropic_messages" in litellm_metadata["tags"]
        assert "route:anthropic_google_completion_adapter" in litellm_metadata["tags"]
        assert "anthropic-google-completion-adapter" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_nvidia_completion_adapter_preserves_agent_project_litellm_metadata(
        self,
    ):
        prepared_body = await _prepare_claude_code_agent_project_request_body(
            "nvidia/deepseek-ai/deepseek-v3.2"
        )
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_nvidia_api_key",
            return_value="nvidia-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.HttpPassThroughEndpointHelpers.validate_outgoing_egress",
        ), patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
            new=AsyncMock(
                return_value={
                    "id": "msg_nvidia_agent_project",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-ai/deepseek-v3.2",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                    },
                }
            ),
        ) as mock_completion_adapter:
            await _handle_anthropic_nvidia_completion_adapter_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(),
                prepared_request_body=prepared_body,
                adapter_model="deepseek-ai/deepseek-v3.2",
            )

        litellm_metadata = mock_completion_adapter.await_args.kwargs[
            "litellm_metadata"
        ]
        _assert_claude_code_agent_project_litellm_metadata(litellm_metadata)
        assert (
            litellm_metadata["passthrough_route_family"]
            == "anthropic_nvidia_completion_adapter"
        )
        assert "route:anthropic_messages" in litellm_metadata["tags"]
        assert "route:anthropic_nvidia_completion_adapter" in litellm_metadata["tags"]
        assert "anthropic-nvidia-completion-adapter" in litellm_metadata["tags"]


class TestClaudePersistedOutputExpansion:
    def test_expand_claude_persisted_output_text_subagentstart(self, tmp_path, monkeypatch):
        claude_root = tmp_path / ".claude" / "projects"
        persisted_file = (
            claude_root
            / "project-a"
            / "session-1"
            / "tool-results"
            / "hook-123-1-additionalContext.txt"
        )
        persisted_file.parent.mkdir(parents=True)
        persisted_file.write_text(
            "You are 'engineer' and you are working on the 'aegis' project.\nFull context.",
            encoding="utf-8",
        )
        monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
        monkeypatch.setenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(claude_root)
        )

        text = (
            "<system-reminder>\n"
            "SubagentStart hook additional context: <persisted-output>\n"
            f"Output too large (24.4KB). Full output saved to: {persisted_file}\n\n"
            "Preview (first 2KB):\n"
            "truncated preview\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )

        expanded_text, was_expanded, hook_name, source_metadata = (
            _expand_claude_persisted_output_text(text)
        )

        assert was_expanded is True
        assert hook_name == "subagentstart"
        assert "truncated preview" not in expanded_text
        assert "You are 'engineer'" in expanded_text
        assert expanded_text.startswith("<system-reminder>\nSubagentStart hook")
        assert expanded_text.endswith("</system-reminder>\n")
        assert source_metadata is not None
        assert source_metadata["path"] == str(persisted_file)
        assert source_metadata["basename"] == persisted_file.name

    def test_expand_claude_persisted_output_text_sessionstart(self, tmp_path, monkeypatch):
        claude_root = tmp_path / ".claude" / "projects"
        persisted_file = (
            claude_root
            / "project-a"
            / "session-1"
            / "tool-results"
            / "hook-456-1-additionalContext.txt"
        )
        persisted_file.parent.mkdir(parents=True)
        persisted_file.write_text(
            "SessionStart full persisted output.",
            encoding="utf-8",
        )
        monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
        monkeypatch.setenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(claude_root)
        )

        text = (
            "<system-reminder>\n"
            "SessionStart hook additional context: <persisted-output>\n"
            f"Output too large (111.6KB). Full output saved to: {persisted_file}\n\n"
            "Preview (first 2KB):\n"
            "truncated preview\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )

        expanded_text, was_expanded, hook_name, source_metadata = (
            _expand_claude_persisted_output_text(text)
        )

        assert was_expanded is True
        assert hook_name == "sessionstart"
        assert "SessionStart full persisted output." in expanded_text
        assert expanded_text.startswith("<system-reminder>\nSessionStart hook")
        assert source_metadata is not None
        assert source_metadata["path"] == str(persisted_file)

    def test_expand_claude_persisted_output_text_noop_outside_allowed_root(
        self, tmp_path, monkeypatch
    ):
        claude_root = tmp_path / ".claude" / "projects"
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("outside", encoding="utf-8")
        monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
        monkeypatch.setenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(claude_root)
        )

        text = (
            "<system-reminder>\n"
            "SubAgentStart hook additional context: <persisted-output>\n"
            f"Output too large (24.4KB). Full output saved to: {outside_file}\n\n"
            "Preview (first 2KB):\n"
            "truncated preview\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )

        expanded_text, was_expanded, hook_name, source_metadata = (
            _expand_claude_persisted_output_text(text)
        )

        assert was_expanded is False
        assert hook_name is None
        assert expanded_text == text
        assert source_metadata is None

    def test_expand_claude_persisted_output_in_anthropic_request_body(
        self, tmp_path, monkeypatch
    ):
        claude_root = tmp_path / ".claude" / "projects"
        persisted_file = (
            claude_root
            / "project-a"
            / "session-1"
            / "tool-results"
            / "hook-789-1-additionalContext.txt"
        )
        persisted_file.parent.mkdir(parents=True)
        persisted_file.write_text("expanded body payload", encoding="utf-8")
        monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
        monkeypatch.setenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(claude_root)
        )

        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-dynamic-html-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                f"Output too large (24.4KB). Full output saved to: {persisted_file}\n\n"
                                "Preview (first 2KB):\n"
                                "truncated preview\n"
                                "</persisted-output>\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ],
        }

        updated_body, expanded_count, hooks, source_metadata_items = (
            _expand_claude_persisted_output_in_anthropic_request_body(request_body)
        )

        assert expanded_count == 1
        assert hooks == {"subagentstart"}
        updated_text = updated_body["messages"][0]["content"][0]["text"]
        assert "expanded body payload" in updated_text
        assert "truncated preview" not in updated_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_persisted_output_expanded"] is True
        assert litellm_metadata["claude_persisted_output_expanded_count"] == 1
        assert litellm_metadata["claude_persisted_output_hooks"] == ["subagentstart"]
        assert litellm_metadata["claude_persisted_output_source_paths"] == [
            str(persisted_file)
        ]
        assert litellm_metadata["claude_persisted_output_source_basenames"] == [
            persisted_file.name
        ]
        assert len(litellm_metadata["claude_persisted_output_source_content_hashes"]) == 1
        assert litellm_metadata["claude_persisted_output_source_bytes"] == [
            len("expanded body payload".encode("utf-8"))
        ]
        assert "claude-persisted-output-expanded" in litellm_metadata["tags"]
        assert "claude-persisted-output-hook:subagentstart" in litellm_metadata["tags"]
        assert isinstance(litellm_metadata["langfuse_spans"], list)
        assert litellm_metadata["langfuse_spans"][0]["name"] == (
            "claude.persisted_output_expand"
        )
        assert litellm_metadata["langfuse_spans"][0]["metadata"]["expanded_count"] == 1
        assert litellm_metadata["langfuse_spans"][0]["metadata"]["hooks"] == [
            "subagentstart"
        ]
        assert (
            litellm_metadata["langfuse_spans"][0]["metadata"]["source_paths"]
            == [str(persisted_file)]
        )
        assert "start_time" in litellm_metadata["langfuse_spans"][0]
        assert "end_time" in litellm_metadata["langfuse_spans"][0]
        assert len(source_metadata_items) == 1
        assert source_metadata_items[0]["path"] == str(persisted_file)

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_expands_persisted_output_before_passthrough(
        self, tmp_path, monkeypatch
    ):
        claude_root = tmp_path / ".claude" / "projects"
        persisted_file = (
            claude_root
            / "project-a"
            / "session-1"
            / "tool-results"
            / "hook-999-1-additionalContext.txt"
        )
        persisted_file.parent.mkdir(parents=True)
        persisted_file.write_text("expanded route payload", encoding="utf-8")
        monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
        monkeypatch.setenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(claude_root)
        )

        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                f"Output too large (24.4KB). Full output saved to: {persisted_file}\n\n"
                                "Preview (first 2KB):\n"
                                "truncated preview\n"
                                "</persisted-output>\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new=AsyncMock(return_value=False),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ) as mock_set_parsed_body, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="anthropic-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(return_value={"id": "msg_123"})
            mock_create_route.return_value = mock_endpoint_func

            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            assert result == {"id": "msg_123"}
            mock_set_parsed_body.assert_called_once()
            expanded_body = mock_set_parsed_body.call_args.args[1]
            expanded_text = expanded_body["messages"][0]["content"][0]["text"]
            assert "expanded route payload" in expanded_text
            assert "truncated preview" not in expanded_text
            litellm_metadata = expanded_body["litellm_metadata"]
            assert "claude-persisted-output-expanded" in litellm_metadata["tags"]
            assert litellm_metadata["claude_persisted_output_source_paths"] == [
                str(persisted_file)
            ]
            assert (
                "claude-persisted-output-hook:subagentstart"
                in litellm_metadata["tags"]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_name,declared_model",
        [
            ("formatter", "gpt-5.3-codex-spark"),
            ("peerOmega", "gpt-5.4"),
        ],
    )
    async def test_anthropic_proxy_route_adapts_supported_subagent_from_agent_spec(
        self, tmp_path, monkeypatch, agent_name, declared_model
    ):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / f"{agent_name}.md").write_text(
            "---\n"
            f"name: {agent_name}\n"
            f"model: {declared_model}\n"
            "---\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("LITELLM_CLAUDE_AGENTS_DIR", str(agents_dir))

        request_body = {
            "model": "claude-opus-4-7",
            "max_tokens": 256,
            "system": (
                f"You are '{agent_name}' and you are working on the 'litellm' project."
            ),
            "messages": [{"role": "user", "content": "Say adapter ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "user-agent": "claude-cli/2.1.114 (external, sdk-cli)",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        local_codex_headers = {
            "Authorization": "Bearer codex-access-token",
            "ChatGPT-Account-Id": "acct_local",
            "originator": "codex_cli_rs",
            "user-agent": "codex_cli_rs/0.0.0",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_local_codex_auth_headers",
            return_value=local_codex_headers,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_123",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": declared_model,
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == declared_model
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert (
            call_kwargs["target"]
            == "https://chatgpt.com/backend-api/codex/responses"
        )
        assert call_kwargs["custom_llm_provider"] in {
            litellm.LlmProviders.CHATGPT.value,
            litellm.LlmProviders.OPENAI.value,
        }
        assert call_kwargs["custom_body"]["model"] == declared_model
        assert call_kwargs["custom_headers"] == local_codex_headers
        assert call_kwargs["forward_headers"] is False
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> chatgpt.com/backend-api/codex/responses"
        )

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_adapts_gemini31_model_to_google_completion(
        self,
    ):
        request_body = {
            "model": "gemini-3.1-pro-preview",
            "max_tokens": 128,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Say gemini ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        translated_response = Response(
            content=json.dumps(
                {
                    "id": "msg_google",
                    "type": "message",
                    "role": "assistant",
                    "model": "gemini-3.1-pro-preview",
                    "content": [{"type": "text", "text": "gemini ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                }
            ).encode("utf-8"),
            media_type="application/json",
        )
        mock_streaming_response = StreamingResponse(
            iter([b"data: [DONE]\n\n"]), media_type="text/event-stream"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            new=AsyncMock(return_value="ya29.test-google-token"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_or_load_google_code_assist_project",
            new=AsyncMock(return_value="project_123"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._perform_google_adapter_pass_through_request",
            new=AsyncMock(return_value=mock_streaming_response),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._collect_google_code_assist_response_from_stream",
            new=AsyncMock(return_value=translated_response),
        ) as mock_collect_response:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "gemini-3.1-pro-preview"
        assert translated_body["content"][0]["text"] == "gemini ok"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.GEMINI.value
        assert call_kwargs["forward_headers"] is False
        assert call_kwargs["query_params"] == {"alt": "sse"}
        assert call_kwargs["stream"] is True
        assert call_kwargs["egress_credential_family"] == "google"
        assert call_kwargs["expected_target_family"] == "google"
        assert call_kwargs["custom_headers"]["Authorization"] == "Bearer ya29.test-google-token"
        assert call_kwargs["custom_body"]["model"] == "gemini-3.1-pro-preview"
        assert call_kwargs["custom_body"]["project"] == "project_123"
        assert call_kwargs["custom_body"]["request"]["session_id"]
        assert (
            call_kwargs["custom_body"]["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_google_completion_adapter"
        )
        assert "anthropic-google-completion-adapter" in call_kwargs["custom_body"]["litellm_metadata"]["tags"]
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        )
        collect_kwargs = mock_collect_response.await_args.kwargs
        assert collect_kwargs["adapter_model"] == "gemini-3.1-pro-preview"
        assert collect_kwargs["tool_name_mapping"] == {}


    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_uses_alt_sse_for_google_stream_adapter(
        self,
    ):
        request_body = {
            "model": "gemini-3.1",
            "max_tokens": 128,
            "stream": True,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Say gemini stream ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_streaming_response = StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream")

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            new=AsyncMock(return_value="ya29.test-google-token"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_or_load_google_code_assist_project",
            new=AsyncMock(return_value="project_123"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._perform_google_adapter_pass_through_request",
            new=AsyncMock(return_value=mock_streaming_response),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._build_anthropic_streaming_response_from_google_code_assist_stream",
            return_value=Response(content=b"stream-ok", media_type="text/event-stream"),
        ):
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert result.body == b"stream-ok"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
        assert call_kwargs["query_params"] == {"alt": "sse"}
        assert call_kwargs["stream"] is True
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        )

    @pytest.mark.asyncio
    async def test_google_completion_adapter_uses_streaming_upstream_for_non_stream_clients(self):
        prepared_request_body = {
            "model": "gemini-3-flash-preview",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"content-type": "application/json"}
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_streaming_response = StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream")
        wrapped_request_body = {
            "model": "gemini-3-flash-preview",
            "project": "project_123",
            "user_prompt_id": "prompt-1",
            "request": {
                "session_id": "session-1",
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "tools": [],
            },
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            new=AsyncMock(return_value="ya29.test-google-token"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_or_load_google_code_assist_project",
            new=AsyncMock(return_value="project_123"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prime_google_code_assist_session",
            new=AsyncMock(return_value=None),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._build_google_code_assist_request_from_completion_kwargs",
            new=AsyncMock(return_value=(wrapped_request_body, {}, prepared_request_body["messages"], {}, {}, {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._perform_google_adapter_pass_through_request",
            new=AsyncMock(return_value=mock_streaming_response),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._collect_google_code_assist_response_from_stream",
            new=AsyncMock(return_value=Response(content=b"json-ok", media_type="application/json")),
        ):
            result = await _handle_anthropic_google_completion_adapter_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model="gemini-3-flash-preview",
            )

        assert result.body == b"json-ok"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
        assert call_kwargs["query_params"] == {"alt": "sse"}
        assert call_kwargs["stream"] is True

    @pytest.mark.parametrize(
        "declared_model, translated_model",
        [
            ("gemini-3.1", "gemini-3.1-pro-preview"),
            ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"),
        ],
    )
    def test_normalize_google_completion_adapter_model_name_aliases(
        self,
        declared_model: str,
        translated_model: str,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _normalize_google_completion_adapter_model_name,
        )

        assert _normalize_google_completion_adapter_model_name(declared_model) == translated_model
        assert _normalize_google_completion_adapter_model_name(f"gemini/{declared_model}") == translated_model
        assert _normalize_google_completion_adapter_model_name(f"google/{declared_model}") == translated_model

    @pytest.mark.parametrize(
        ("requested_model", "expected_model"),
        [
            ("openai/gpt-5.4-mini", "gpt-5.4-mini"),
            ("openai/gpt-5.5", "gpt-5.5"),
        ],
    )
    def test_resolve_anthropic_openai_responses_adapter_model_supports_openai_prefix(
        self,
        requested_model: str,
        expected_model: str,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_openai_responses_adapter_model,
        )

        request_body = {"model": requested_model}

        assert (
            _resolve_anthropic_openai_responses_adapter_model(
                request_body, endpoint="v1/messages"
            )
            == expected_model
        )

    def test_anthropic_openai_responses_adapter_merges_cache_litellm_metadata(
        self,
    ):
        request_body = {
            "model": "gpt-5.4-mini",
            "max_tokens": 16,
            "system": "Reply briefly.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "cache me",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "litellm_metadata": {
                "tags": ["route:anthropic_messages"],
                "trace_environment": "dev",
            },
        }

        translated_body = _build_anthropic_responses_adapter_request_body(
            request_body,
            adapter_model="gpt-5.4-mini",
        )

        litellm_metadata = translated_body["litellm_metadata"]
        assert translated_body["prompt_cache_key"].startswith("anthropic-cache-")
        assert set(litellm_metadata["tags"]) >= {
            "route:anthropic_messages",
            "route:anthropic_openai_responses_adapter",
            "anthropic-openai-responses-adapter",
            "anthropic-adapter-model:gpt-5.4-mini",
            "anthropic-adapter-target:/v1/responses",
        }
        assert litellm_metadata["trace_environment"] == "dev"
        assert litellm_metadata["openai_prompt_cache_key_present"] is True
        assert litellm_metadata["anthropic_adapter_cache_control_present"] is True
        assert litellm_metadata["openai_provider_cache_attempted"] is True

    def test_resolve_anthropic_google_completion_adapter_model_supports_google_prefix(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_google_completion_adapter_model,
        )

        request_body = {"model": "google/gemini-3.1"}

        assert (
            _resolve_anthropic_google_completion_adapter_model(
                request_body, endpoint="v1/messages"
            )
            == "gemini-3.1-pro-preview"
        )

    @pytest.mark.parametrize(
        ("requested_model", "expected_model"),
        [
            ("nvidia/deepseek-ai/deepseek-v3.2", "deepseek-ai/deepseek-v3.2"),
            ("nvidia/deepseek-ai/deepseek-v3.1-terminus", "deepseek-ai/deepseek-v3.1-terminus"),
            ("nvidia/mistralai/devstral-2-123b-instruct-2512", "mistralai/devstral-2-123b-instruct-2512"),
            ("nvidia/z-ai/glm4.7", "z-ai/glm4.7"),
            ("nvidia/minimax/minimax-m2.7", "minimaxai/minimax-m2.7"),
        ],
    )
    def test_resolve_anthropic_nvidia_responses_adapter_model_supports_nvidia_prefix_and_aliases(
        self,
        requested_model: str,
        expected_model: str,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_nvidia_responses_adapter_model,
        )

        request_body = {"model": requested_model}

        assert (
            _resolve_anthropic_nvidia_responses_adapter_model(
                request_body, endpoint="v1/messages"
            )
            == expected_model
        )

    def test_resolve_anthropic_google_completion_adapter_model_skips_openrouter_google_namespace(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_google_completion_adapter_model,
            _resolve_anthropic_openrouter_responses_adapter_model,
        )

        request_body = {"model": "google/gemma-4-31b-it:free"}

        assert (
            _resolve_anthropic_google_completion_adapter_model(
                request_body, endpoint="v1/messages"
            )
            is None
        )
        assert (
            _resolve_anthropic_openrouter_responses_adapter_model(
                request_body, endpoint="v1/messages"
            )
            == "google/gemma-4-31b-it:free"
        )

    def test_resolve_anthropic_openrouter_responses_adapter_model_supports_openrouter_prefix(
        self,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_openrouter_responses_adapter_model,
        )

        request_body = {"model": "openrouter/google/gemma-4-31b-it:free"}

        assert (
            _resolve_anthropic_openrouter_responses_adapter_model(
                request_body, endpoint="v1/messages"
            )
            == "google/gemma-4-31b-it:free"
        )

    def test_load_claude_agent_declared_model_tolerates_cp1252_agent_file(
        self, tmp_path, monkeypatch
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _claude_agent_model_cache,
            _load_claude_agent_declared_model,
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_file = agents_dir / "gpt-5-4.md"
        agent_file.write_bytes(
            b"---\n"
            b"name: gpt-5-4\n"
            b"model: gpt-5.4\n"
            b"---\n\n"
            b"Uses smart quotes \x92here.\n"
        )
        monkeypatch.setenv("LITELLM_CLAUDE_AGENTS_DIR", str(agents_dir))
        _claude_agent_model_cache.clear()

        assert _load_claude_agent_declared_model("gpt-5-4") == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_translates_gemini31_alias_to_preview_model(
        self,
    ):
        request_body = {
            "model": "gemini-3.1",
            "max_tokens": 128,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Say gemini alias ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        translated_response = Response(
            content=json.dumps(
                {
                    "id": "msg_google_alias",
                    "type": "message",
                    "role": "assistant",
                    "model": "gemini-3.1-pro-preview",
                    "content": [{"type": "text", "text": "gemini alias ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                }
            ).encode("utf-8"),
            media_type="application/json",
        )
        mock_streaming_response = StreamingResponse(
            iter([b"data: [DONE]\n\n"]), media_type="text/event-stream"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            new=AsyncMock(return_value="ya29.test-google-token"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_or_load_google_code_assist_project",
            new=AsyncMock(return_value="project_123"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._perform_google_adapter_pass_through_request",
            new=AsyncMock(return_value=mock_streaming_response),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._collect_google_code_assist_response_from_stream",
            new=AsyncMock(return_value=translated_response),
        ) as mock_collect_response:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "gemini-3.1-pro-preview"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["custom_body"]["model"] == "gemini-3.1-pro-preview"
        assert call_kwargs["target"] == "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
        assert call_kwargs["query_params"] == {"alt": "sse"}
        assert call_kwargs["stream"] is True
        collect_kwargs = mock_collect_response.await_args.kwargs
        assert collect_kwargs["adapter_model"] == "gemini-3.1-pro-preview"

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_does_not_load_local_google_auth_for_native_anthropic_model(
        self,
    ):
        request_body = {
            "model": "claude-opus-4-7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say anthropic ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_valid_local_google_oauth_access_token",
            side_effect=AssertionError("should not load google auth"),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new=AsyncMock(return_value=False),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="anthropic-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint = AsyncMock(return_value={"id": "msg_native"})
            mock_create_route.return_value = mock_endpoint

            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert result == {"id": "msg_native"}


    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_adapts_openrouter_prefixed_gemma_model_to_responses(
        self,
    ):
        request_body = {
            "model": "openrouter/google/gemma-4-31b-it:free",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Say gemma prefix ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_openrouter_api_key",
            return_value="openrouter-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_gemma_prefix",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "google/gemma-4-31b-it:free",
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "gemma prefix ok",
                                        }
                                    ],
                                }
                            ],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "google/gemma-4-31b-it:free"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://openrouter.ai/api/v1/responses"
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.OPENROUTER.value
        assert call_kwargs["forward_headers"] is False
        assert call_kwargs["custom_body"]["model"] == "google/gemma-4-31b-it:free"
        assert (
            call_kwargs["custom_body"]["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_openrouter_responses_adapter"
        )
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> openrouter.ai/api/v1/responses"
        )

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_adapts_allowlisted_gemma_model_to_responses(
        self,
    ):
        request_body = {
            "model": "google/gemma-4-31b-it:free",
            "max_tokens": 256,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Say gemma ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_openrouter_api_key",
            return_value="openrouter-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_gemma",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "google/gemma-4-31b-it:free",
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "gemma ok",
                                        }
                                    ],
                                }
                            ],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    headers={
                        "content-length": "999",
                        "content-encoding": "br",
                        "x-upstream-trace": "openrouter-test",
                    },
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "google/gemma-4-31b-it:free"
        assert result.headers["content-length"] == str(len(result.body))
        assert result.headers.get("content-encoding") is None
        assert result.headers["x-upstream-trace"] == "openrouter-test"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://openrouter.ai/api/v1/responses"
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.OPENROUTER.value
        assert call_kwargs["custom_body"]["model"] == "google/gemma-4-31b-it:free"
        assert (
            call_kwargs["custom_body"]["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_openrouter_responses_adapter"
        )
        assert mock_request.scope["path"] == "/anthropic/v1/messages"
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> openrouter.ai/api/v1/responses"
        )

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_adapts_prefixed_openrouter_free_model_to_responses(
        self,
    ):
        request_body = {
            "model": "openai/gpt-oss-20b:free",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Say oss ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_openrouter_api_key",
            return_value="openrouter-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_oss",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "openai/gpt-oss-20b:free",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "openai/gpt-oss-20b:free"
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://openrouter.ai/api/v1/responses"
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.OPENROUTER.value
        assert call_kwargs["forward_headers"] is False
        assert call_kwargs["custom_body"]["model"] == "openai/gpt-oss-20b:free"
        assert (
            call_kwargs["custom_body"]["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_openrouter_responses_adapter"
        )
        assert mock_request.scope["query_string"] == b"beta=true -> openrouter.ai/api/v1/responses"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requested_model", "expected_model"),
        [
            ("google/gemma-4-31b-it:free", "google/gemma-4-31b-it:free"),
            ("google/gemma-4-26b-a4b-it:free", "google/gemma-4-26b-a4b-it:free"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-super-120b-a12b:free"),
            ("meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
            ("meta-llama/llama-3.3-70b-instructfree", "meta-llama/llama-3.3-70b-instruct:free"),
            ("minimax/minimax-m2.5:free", "minimax/minimax-m2.5:free"),
            ("qwen/qwen3-coder:free", "qwen/qwen3-coder:free"),
        ],
    )
    async def test_anthropic_proxy_route_adapts_selected_openrouter_free_models_to_responses(
        self,
        requested_model: str,
        expected_model: str,
    ):
        request_body = {
            "model": requested_model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Say model ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_openrouter_api_key",
            return_value="openrouter-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_free",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": requested_model,
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == requested_model
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["target"] == "https://openrouter.ai/api/v1/responses"
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.OPENROUTER.value
        assert call_kwargs["forward_headers"] is False
        assert call_kwargs["custom_body"]["model"] == expected_model
        assert (
            call_kwargs["custom_body"]["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_openrouter_responses_adapter"
        )
        assert f"anthropic-adapter-model:{expected_model}" in call_kwargs["custom_body"]["litellm_metadata"]["tags"]
        assert mock_request.scope["query_string"] == b"beta=true -> openrouter.ai/api/v1/responses"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requested_model", "expected_model", "expected_timeout"),
        [
            ("nvidia/deepseek-ai/deepseek-v3.2", "deepseek-ai/deepseek-v3.2", 120.0),
            ("nvidia/deepseek-ai/deepseek-v3.1-terminus", "deepseek-ai/deepseek-v3.1-terminus", 120.0),
            ("nvidia/mistralai/devstral-2-123b-instruct-2512", "mistralai/devstral-2-123b-instruct-2512", 120.0),
            ("nvidia/z-ai/glm4.7", "z-ai/glm4.7", 120.0),
            ("nvidia/minimax/minimax-m2.7", "minimaxai/minimax-m2.7", 240.0),
        ],
    )
    async def test_anthropic_proxy_route_adapts_selected_nvidia_models_to_completion_adapter(
        self,
        requested_model: str,
        expected_model: str,
        expected_timeout: float,
    ):
        request_body = {
            "model": requested_model,
            "max_tokens": 256,
            "metadata": {"existing_key": "existing-value"},
            "litellm_metadata": {
                "session_id": "nvidia-session-1",
                "trace_environment": "prod",
            },
            "messages": [{"role": "user", "content": "Say model ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_nvidia_api_key",
            return_value="nvidia-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.HttpPassThroughEndpointHelpers.validate_outgoing_egress",
        ) as mock_validate_egress, patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
            new=AsyncMock(
                return_value={
                    "id": "msg_nvidia",
                    "type": "message",
                    "role": "assistant",
                    "model": expected_model,
                    "content": [{"type": "text", "text": "model ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                    },
                }
            ),
        ) as mock_completion_adapter:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == expected_model
        call_kwargs = mock_completion_adapter.await_args.kwargs
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.NVIDIA_NIM.value
        assert call_kwargs["api_key"] == "nvidia-test-key"
        assert call_kwargs["api_base"] == "https://integrate.api.nvidia.com/v1"
        assert call_kwargs["timeout"] == expected_timeout
        assert call_kwargs["max_retries"] == 0
        assert "standard_callback_dynamic_params" not in call_kwargs
        assert call_kwargs["model"] == expected_model
        assert call_kwargs["metadata"]["existing_key"] == "existing-value"
        assert call_kwargs["metadata"]["session_id"] == "nvidia-session-1"
        assert call_kwargs["metadata"]["trace_environment"] == "prod"
        assert (
            call_kwargs["metadata"]["passthrough_route_family"]
            == "anthropic_nvidia_completion_adapter"
        )
        assert call_kwargs["metadata"]["anthropic_adapter_model"] == expected_model
        assert (
            call_kwargs["metadata"]["anthropic_adapter_original_model"]
            == requested_model
        )
        assert (
            call_kwargs["metadata"]["anthropic_adapter_target_endpoint"]
            == "nvidia:/v1/chat/completions"
        )
        assert (
            "anthropic-nvidia-completion-adapter" in call_kwargs["metadata"]["tags"]
        )
        assert call_kwargs["metadata"]["langfuse_spans"][0]["name"] == (
            "anthropic.nvidia_completion_adapter"
        )
        assert (
            call_kwargs["litellm_metadata"]["passthrough_route_family"]
            == "anthropic_nvidia_completion_adapter"
        )
        assert (
            call_kwargs["litellm_metadata"]["anthropic_adapter_model"]
            == expected_model
        )
        assert (
            call_kwargs["litellm_metadata"]["anthropic_adapter_original_model"]
            == requested_model
        )
        assert (
            call_kwargs["litellm_metadata"]["anthropic_adapter_target_endpoint"]
            == "nvidia:/v1/chat/completions"
        )
        assert (
            "anthropic-nvidia-completion-adapter"
            in call_kwargs["litellm_metadata"]["tags"]
        )
        assert (
            f"anthropic-adapter-model:{expected_model}"
            in call_kwargs["litellm_metadata"]["tags"]
        )
        assert (
            f"anthropic-adapter-target:nvidia:/v1/chat/completions"
            in call_kwargs["litellm_metadata"]["tags"]
        )
        assert call_kwargs["litellm_metadata"]["langfuse_spans"][0]["name"] == (
            "anthropic.nvidia_completion_adapter"
        )
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> integrate.api.nvidia.com/v1/chat/completions"
        )
        mock_validate_egress.assert_called_once_with(
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": "Bearer nvidia-test-key"},
            credential_family="nvidia",
            expected_target_family="nvidia",
        )

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_retries_transient_nvidia_gateway_timeout(
        self,
    ):
        class _RetryableNvidiaError(Exception):
            status_code = 504

        request_body = {
            "model": "nvidia/minimaxai/minimax-m2.7",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Say model ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_nvidia_api_key",
            return_value="nvidia-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.asyncio.sleep",
            new=AsyncMock(),
        ) as mock_sleep, patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
            new=AsyncMock(
                side_effect=[
                    _RetryableNvidiaError("upstream 504"),
                    {
                        "id": "msg_nvidia",
                        "type": "message",
                        "role": "assistant",
                        "model": "minimaxai/minimax-m2.7",
                        "content": [{"type": "text", "text": "model ok"}],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                        },
                    },
                ]
            ),
        ) as mock_completion_adapter:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["model"] == "minimaxai/minimax-m2.7"
        assert mock_completion_adapter.await_count == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_fake_streams_minimax_nvidia_model(self):
        request_body = {
            "model": "nvidia/minimaxai/minimax-m2.7",
            "max_tokens": 256,
            "stream": True,
            "messages": [{"role": "user", "content": "Say model ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_nvidia_api_key",
            return_value="nvidia-test-key",
        ), patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
            new=AsyncMock(
                return_value={
                    "id": "msg_nvidia",
                    "type": "message",
                    "role": "assistant",
                    "model": "minimaxai/minimax-m2.7",
                    "content": [{"type": "text", "text": "model ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                    },
                }
            ),
        ) as mock_completion_adapter:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert isinstance(result, StreamingResponse)
        call_kwargs = mock_completion_adapter.await_args.kwargs
        assert call_kwargs["stream"] is False
        streamed_chunks = []
        async for chunk in result.body_iterator:
            streamed_chunks.append(
                chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            )
        streamed_payload = "".join(streamed_chunks)
        assert "event: message_start" in streamed_payload
        assert '"model": "minimaxai/minimax-m2.7"' in streamed_payload
        assert '"text": "model ok"' in streamed_payload

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_keeps_native_streaming_for_non_minimax_nvidia_model(
        self,
    ):
        request_body = {
            "model": "nvidia/z-ai/glm4.7",
            "max_tokens": 256,
            "stream": True,
            "messages": [{"role": "user", "content": "Say model ok"}],
        }

        async def _anthropic_stream():
            yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"model ok"}}\n\n'
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_anthropic_adapter_nvidia_api_key",
            return_value="nvidia-test-key",
        ), patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
            new=AsyncMock(return_value=_anthropic_stream()),
        ) as mock_completion_adapter:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert isinstance(result, StreamingResponse)
        call_kwargs = mock_completion_adapter.await_args.kwargs
        assert call_kwargs["stream"] is True
        streamed_chunks = []
        async for chunk in result.body_iterator:
            streamed_chunks.append(
                chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            )
        streamed_payload = "".join(streamed_chunks)
        assert "event: message_start" in streamed_payload
        assert "model ok" in streamed_payload

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requested_model", "response_model"),
        [
            ("gpt-5.4", "gpt-5.4"),
            ("openai/gpt-5.5", "gpt-5.5"),
        ],
    )
    async def test_anthropic_proxy_route_adapts_allowlisted_openai_model_to_responses(
        self,
        requested_model: str,
        response_model: str,
    ):
        request_body = {
            "model": requested_model,
            "max_tokens": 256,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Say adapter ok"}],
        }

        async def _responses_stream():
            chunks = [
                b'event: response.created\ndata: {"type":"response.created"}\n\n',
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_123","output_index":0,"content_index":0,"delta":"adapter "}\n\n',
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_123","output_index":0,"content_index":0,"delta":"ok"}\n\n',
                f'event: response.completed\ndata: {{"type":"response.completed","response":{{"id":"resp_123","object":"response","created_at":1744974432,"model":"{response_model}","status":"completed","output":[],"usage":{{"input_tokens":12,"output_tokens":3,"total_tokens":15}}}}}}\n\n'.encode(
                    "utf-8"
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "x-pass-authorization": "Bearer codex-oauth-token",
            "x-pass-chatgpt-account-id": "acct_123",
            "x-pass-originator": "codex-cli",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=StreamingResponse(
                    _responses_stream(),
                    media_type="text/event-stream",
                    headers={"x-openai-upstream": "1"},
                )
            ),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert isinstance(result, Response)
        translated_body = json.loads(result.body.decode("utf-8"))
        assert translated_body["type"] == "message"
        assert translated_body["model"] == response_model
        assert translated_body["content"][0]["type"] == "text"
        assert translated_body["content"][0]["text"] == "adapter ok"
        assert translated_body["usage"]["input_tokens"] == 12
        assert translated_body["usage"]["output_tokens"] == 3
        assert result.headers["x-openai-upstream"] == "1"

        mock_create_route.assert_not_called()
        mock_pass_through_request.assert_awaited_once()
        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert (
            call_kwargs["target"]
            == "https://chatgpt.com/backend-api/codex/responses"
        )
        assert call_kwargs["forward_headers"] is True
        assert call_kwargs["allowed_forward_headers"] == [
            "authorization",
            "api-key",
            "chatgpt-account-id",
            "originator",
            "user-agent",
            "session_id",
            "session-id",
        ]
        assert call_kwargs["allowed_pass_through_prefixed_headers"] == [
            "authorization",
            "api-key",
            "chatgpt-account-id",
            "originator",
            "user-agent",
            "session_id",
            "session-id",
        ]
        assert call_kwargs["custom_headers"] == {}
        assert call_kwargs["custom_body"]["model"] == response_model
        assert call_kwargs["custom_body"]["stream"] is True
        assert "max_output_tokens" not in call_kwargs["custom_body"]
        assert "temperature" not in call_kwargs["custom_body"]
        assert "top_p" not in call_kwargs["custom_body"]
        assert call_kwargs["custom_body"]["include"] == [
            "reasoning.encrypted_content"
        ]
        assert call_kwargs["custom_body"]["instructions"] == "You are helpful."
        assert call_kwargs["custom_body"]["store"] is False
        assert call_kwargs["custom_body"]["input"][0]["role"] == "user"
        assert (
            call_kwargs["custom_body"]["litellm_metadata"][
                "passthrough_route_family"
            ]
            == "anthropic_openai_responses_adapter"
        )
        assert (
            "anthropic-openai-responses-adapter"
            in call_kwargs["custom_body"]["litellm_metadata"]["tags"]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("headers", "expected_tool_name", "expects_codex_alias_metadata"),
        [
            (
                {
                    "content-type": "application/json",
                    "x-pass-authorization": "Bearer openai-api-key",
                },
                "Bash",
                False,
            ),
            (
                {
                    "content-type": "application/json",
                    "x-pass-authorization": "Bearer codex-oauth-token",
                    "x-pass-chatgpt-account-id": "acct_123",
                    "x-pass-originator": "codex-cli",
                },
                "exec_command",
                True,
            ),
        ],
    )
    async def test_anthropic_proxy_route_scopes_codex_native_tool_aliases(
        self,
        headers: dict[str, str],
        expected_tool_name: str,
        expects_codex_alias_metadata: bool,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Run pwd with Bash."}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = headers
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_123",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "gpt-5.4",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        call_kwargs = mock_pass_through_request.await_args.kwargs
        custom_body = call_kwargs["custom_body"]
        assert custom_body["tools"][0]["name"] == expected_tool_name
        assert custom_body["tool_choice"] == {
            "type": "function",
            "name": expected_tool_name,
        }
        litellm_metadata = custom_body["litellm_metadata"]
        assert (
            "anthropic-openai-codex-native-tools" in litellm_metadata["tags"]
        ) == expects_codex_alias_metadata
        assert (
            litellm_metadata.get("anthropic_adapter_codex_native_tool_aliases")
            is True
        ) == expects_codex_alias_metadata

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_adds_minimal_adapter_instructions_without_system(
        self,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say adapter ok"}],
            "metadata": {"user_id": "claude-user-123"},
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "x-pass-authorization": "Bearer codex-oauth-token",
            "x-pass-chatgpt-account-id": "acct_123",
            "x-pass-originator": "codex-cli",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_123",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "gpt-5.4",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert call_kwargs["custom_body"]["instructions"] == "You are a helpful assistant."
        assert call_kwargs["custom_body"]["store"] is False
        assert call_kwargs["custom_body"]["stream"] is True
        assert "max_output_tokens" not in call_kwargs["custom_body"]
        assert "temperature" not in call_kwargs["custom_body"]
        assert "top_p" not in call_kwargs["custom_body"]
        assert call_kwargs["custom_body"]["include"] == [
            "reasoning.encrypted_content"
        ]

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_uses_local_codex_auth_for_claude_cli_request(
        self,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say adapter ok"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "user-agent": "claude-cli/2.1.114 (external, sdk-cli)",
            "x-claude-code-session-id": "session_123",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        local_codex_headers = {
            "Authorization": "Bearer codex-access-token",
            "ChatGPT-Account-Id": "acct_local",
            "originator": "codex_cli_rs",
            "user-agent": "codex_cli_rs/0.0.0",
            "session_id": "session_123",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_local_codex_auth_headers",
            return_value=local_codex_headers,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_123",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "gpt-5.4",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        call_kwargs = mock_pass_through_request.await_args.kwargs
        assert (
            call_kwargs["target"]
            == "https://chatgpt.com/backend-api/codex/responses"
        )
        assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.OPENAI.value
        assert mock_request.scope["path"] == "/anthropic/v1/messages"
        assert (
            mock_request.scope["query_string"]
            == b"beta=true -> chatgpt.com/backend-api/codex/responses"
        )
        assert call_kwargs["forward_headers"] is False
        assert call_kwargs["allowed_forward_headers"] == [
            "authorization",
            "api-key",
            "chatgpt-account-id",
            "originator",
            "user-agent",
            "session_id",
            "session-id",
        ]
        assert call_kwargs["allowed_pass_through_prefixed_headers"] == [
            "authorization",
            "api-key",
            "chatgpt-account-id",
            "originator",
            "user-agent",
            "session_id",
            "session-id",
        ]
        assert call_kwargs["custom_headers"] == local_codex_headers
        assert call_kwargs["custom_body"]["stream"] is True
        assert "max_output_tokens" not in call_kwargs["custom_body"]
        assert "temperature" not in call_kwargs["custom_body"]
        assert "top_p" not in call_kwargs["custom_body"]
        assert "user" not in call_kwargs["custom_body"]

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_rejects_raw_mcp_requests_on_adapted_route(
        self,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Roll 2d4+1"}],
            "mcp_servers": [
                {
                    "type": "url",
                    "url": "https://dmcp-server.deno.dev/sse",
                    "name": "dmcp",
                }
            ],
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "dmcp"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "user-agent": "claude-cli/2.1.114 (external, sdk-cli)",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        local_codex_headers = {
            "Authorization": "Bearer codex-access-token",
            "ChatGPT-Account-Id": "acct_local",
            "originator": "codex_cli_rs",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_local_codex_auth_headers",
            return_value=local_codex_headers,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(),
        ) as mock_pass_through_request:
            with pytest.raises(HTTPException) as exc_info:
                await anthropic_proxy_route(
                    endpoint="v1/messages",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

        assert exc_info.value.status_code == 400
        assert (
            exc_info.value.detail
            == "Anthropic adapter does not currently support raw MCP server/toolset requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools such as `mcp__...` or call the native OpenAI Responses API directly."
        )
        mock_pass_through_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_rejects_raw_mcp_requests_even_with_openai_credentials(
        self,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Roll 2d4+1"}],
            "mcp_servers": [
                {
                    "type": "url",
                    "url": "https://dmcp-server.deno.dev/sse",
                    "name": "dmcp",
                }
            ],
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "dmcp"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer anthropic-cli-token",
            "user-agent": "claude-cli/2.1.114 (external, sdk-cli)",
            "anthropic-version": "2023-06-01",
        }
        mock_request.url = httpx.URL(
            "http://127.0.0.1:4001/anthropic/v1/messages?beta=true"
        )
        mock_request.scope = {
            "path": "/anthropic/v1/messages",
            "query_string": b"beta=true",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        local_codex_headers = {
            "Authorization": "Bearer codex-access-token",
            "ChatGPT-Account-Id": "acct_local",
            "originator": "codex_cli_rs",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_local_codex_auth_headers",
            return_value=local_codex_headers,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="openai-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._assemble_headers",
            return_value={"Authorization": "Bearer openai-test-key"},
        ) as mock_assemble_headers, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=Response(
                    content=json.dumps(
                        {
                            "id": "resp_123",
                            "object": "response",
                            "created_at": 1744974432,
                            "model": "gpt-5.4",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode("utf-8"),
                    status_code=200,
                    media_type="application/json",
                )
            ),
        ) as mock_pass_through_request:
            with pytest.raises(HTTPException) as exc_info:
                await anthropic_proxy_route(
                    endpoint="v1/messages",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

        assert exc_info.value.status_code == 400
        assert (
            exc_info.value.detail
            == "Anthropic adapter does not currently support raw MCP server/toolset requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools such as `mcp__...` or call the native OpenAI Responses API directly."
        )
        mock_assemble_headers.assert_not_called()
        mock_pass_through_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_streams_allowlisted_openai_model_as_anthropic(
        self,
    ):
        request_body = {
            "model": "gpt-5.4",
            "max_tokens": 256,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hi"}],
        }

        async def _responses_stream():
            chunks = [
                b'event: response.created\ndata: {"type":"response.created"}\n\n',
                b'event: response.output_item.added\ndata: {"type":"response.output_item.added","item":{"type":"message","id":"msg_123"}}\n\n',
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_123","delta":"hi"}\n\n',
                b'event: response.output_item.done\ndata: {"type":"response.output_item.done","item":{"id":"msg_123"}}\n\n',
                b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":1},"output":[]}}\n\n',
            ]
            for chunk in chunks:
                yield chunk

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "x-pass-authorization": "Bearer codex-oauth-token",
            "x-pass-chatgpt-account-id": "acct_123",
            "x-pass-originator": "codex-cli",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(
                return_value=StreamingResponse(
                    _responses_stream(),
                    status_code=200,
                    headers={"x-openai-upstream": "1"},
                    media_type="text/event-stream",
                )
            ),
        ):
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert isinstance(result, StreamingResponse)
        chunks = [chunk async for chunk in result.body_iterator]
        payload = b"".join(chunks).decode("utf-8")
        assert payload.count("event: message_start") == 1
        assert "event: content_block_start" in payload
        assert '"type": "text_delta"' in payload
        assert '"text": "hi"' in payload
        assert "event: message_stop" in payload

    @pytest.mark.asyncio
    async def test_iterate_responses_sse_events_handles_split_utf8_bytes(self):
        payload = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_123","delta":"caf\xc3\xa9"}\n\n'

        async def _stream():
            yield payload[:-1]
            yield payload[-1:]

        events = []
        async for event in _iterate_responses_sse_events(_stream()):
            events.append(event)

        assert len(events) == 1
        assert getattr(events[0], "type", None) == "response.output_text.delta"
        assert getattr(events[0], "delta", None) == "café"

    @pytest.mark.asyncio
    async def test_collect_responses_stream_reconstructs_arguments_from_done(self):
        async def _responses_stream():
            chunks = [
                b'event: response.created\ndata: {"type":"response.created"}\n\n',
                b'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_pwd","id":"fc_pwd","name":"Bash","arguments":""}}\n\n',
                b'event: response.function_call_arguments.done\ndata: {"type":"response.function_call_arguments.done","item_id":"fc_pwd","output_index":0,"arguments":"{\\"command\\":\\"pwd\\"}"}\n\n',
                b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_codex","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":12,"output_tokens":4}}}\n\n',
            ]
            for chunk in chunks:
                yield chunk

        response = StreamingResponse(
            _responses_stream(),
            status_code=200,
            media_type="text/event-stream",
        )

        collected = await _collect_responses_response_from_stream(response)

        assert collected["output"] == [
            {
                "type": "function_call",
                "call_id": "call_pwd",
                "id": "fc_pwd",
                "name": "Bash",
                "arguments": '{"command":"pwd"}',
            }
        ]

    @pytest.mark.asyncio
    async def test_gemini_tool_use_stream_preserves_input_json_delta_when_starting_new_block(
        self,
    ):
        async def _stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id='call_1',
                                    function=SimpleNamespace(
                                        name='Bash',
                                        arguments='{"command":"date -u"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason='tool_calls',
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[],
                        ),
                    )
                ],
                usage=None,
            )

        wrapper = AnthropicStreamWrapper(_stream(), model='gemini-3-flash-preview')
        chunks = [chunk async for chunk in wrapper]

        tool_start = next(
            chunk for chunk in chunks if chunk.get('type') == 'content_block_start' and chunk.get('content_block', {}).get('type') == 'tool_use'
        )
        tool_delta = next(
            chunk for chunk in chunks if chunk.get('type') == 'content_block_delta' and chunk.get('delta', {}).get('type') == 'input_json_delta'
        )

        assert tool_start['content_block']['name'] == 'Bash'
        assert tool_delta['delta']['partial_json'] == '{"command":"date -u"}'

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_leaves_native_anthropic_models_unchanged(
        self,
    ):
        request_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_anthropic_request_body_for_passthrough",
            new=AsyncMock(return_value=(request_body, 0, set(), {})),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new=AsyncMock(return_value=False),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="anthropic-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            new=AsyncMock(),
        ) as mock_pass_through_request, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._load_local_codex_auth_headers"
        ) as mock_local_codex_auth_loader, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_create_route.return_value = AsyncMock(return_value={"id": "msg_123"})
            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        assert result == {"id": "msg_123"}
        mock_pass_through_request.assert_not_awaited()
        mock_local_codex_auth_loader.assert_not_called()
        mock_create_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_extracts_billing_header(self):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "metadata": {
                "user_id": {
                    "account_uuid": "claude-account-123",
                    "device_id": "claude-device-123",
                    "session_id": "claude-session-123",
                }
            },
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.101.a4a; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": "normal system text",
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        (
            updated_body,
            expanded_count,
            hooks,
            billing_header_fields,
        ) = await _prepare_anthropic_request_body_for_passthrough(mock_request, request_body)

        assert expanded_count == 0
        assert hooks == set()
        assert billing_header_fields == {
            "cc_version": "2.1.101.a4a",
            "cc_entrypoint": "cli",
            "cch": "42aab",
        }
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["anthropic_billing_header_present"] is True
        assert litellm_metadata["anthropic_billing_header_keys"] == [
            "cc_entrypoint",
            "cc_version",
            "cch",
        ]
        assert litellm_metadata["claude_thinking_type"] == "adaptive"
        assert litellm_metadata["claude_effort"] == "high"
        assert litellm_metadata["claude_context_edit_types"] == [
            "clear_thinking_20251015"
        ]
        assert litellm_metadata["claude_context_keep_values"] == ["all"]
        assert litellm_metadata["claude_context_edit_count"] == 1
        assert litellm_metadata["claude_account_uuid"] == "claude-account-123"
        assert litellm_metadata["claude_device_id"] == "claude-device-123"
        assert litellm_metadata["passthrough_route_family"] == "anthropic_messages"
        assert "route:anthropic_messages" in litellm_metadata["tags"]
        assert "claude-thinking-type:adaptive" in litellm_metadata["tags"]
        assert "thinking-type:adaptive" in litellm_metadata["tags"]
        assert "claude-effort:high" in litellm_metadata["tags"]
        assert "effort:high" in litellm_metadata["tags"]
        assert (
            "claude-context-edit:clear_thinking_20251015" in litellm_metadata["tags"]
        )
        assert "claude-context-keep:all" in litellm_metadata["tags"]
        assert litellm_metadata["session_id"] == "claude-session-123"

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_sanitizes_web_search_domain_lists(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_domains": [],
                    "blocked_domains": [],
                },
                {
                    "name": "regular_tool",
                    "input_schema": {"type": "object"},
                    "allowed_domains": [],
                },
            ],
            "messages": [{"role": "user", "content": "search"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        web_search_tool = updated_body["tools"][0]
        regular_tool = updated_body["tools"][1]
        assert web_search_tool["allowed_domains"] is None
        assert web_search_tool["blocked_domains"] is None
        assert regular_tool["allowed_domains"] == []

        litellm_metadata = updated_body["litellm_metadata"]
        assert (
            litellm_metadata["claude_web_search_domain_filter_sanitized_count"] == 2
        )
        assert (
            "claude-web-search-domain-filter-sanitized"
            in litellm_metadata["tags"]
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_extracts_session_from_stringified_user_id(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {
                "user_id": json.dumps(
                    {
                        "device_id": "device-123",
                        "account_uuid": "account-123",
                        "session_id": "claude-session-json-123",
                    }
                )
            },
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.101.a4a; cc_entrypoint=cli; cch=42aab;",
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["session_id"] == "claude-session-json-123"

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_replaces_claude_auto_memory_section(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.110.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": (
                        "Intro text before memory.\n\n"
                        "# auto memory\n\n"
                        "You have a persistent, file-based memory system at `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/memory/`. "
                        "This directory already exists - write to it directly with the Write tool (do not run mkdir or check for its existence).\n\n"
                        "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. "
                        "If they ask you to forget something, find and remove the relevant entry.\n\n"
                        "## Types of memory\n\n"
                        "There are several discrete types of memory that you can store in your memory system:\n\n"
                        "<types>\n"
                        "<type><name>user</name></type>\n"
                        "<type><name>feedback</name></type>\n"
                        "</types>\n\n"
                        "## What NOT to save in memory\n\n"
                        "- Old exclusion text.\n\n"
                        "## How to save memories\n\n"
                        "Old save flow.\n\n"
                        "## When to access memories\n\n"
                        "- Old access rules.\n\n"
                        "## Before recommending from memory\n\n"
                        "Verify the named file still exists.\n\n"
                        "## Memory and other forms of persistence\n"
                        "Memory is one of several persistence mechanisms.\n\n"
                        "# Environment\n"
                        "Other Claude system text.\n"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        updated_system_text = updated_body["system"][1]["text"]
        assert "persistent, file-based memory system at" not in updated_system_text
        assert "write to it directly with the Write tool" not in updated_system_text
        assert "memory_save(" in updated_system_text
        assert "memory_forget(source_ids=[...])" in updated_system_text
        assert "| `user` | Global |" in updated_system_text
        assert "| `feedback` | Agent-scoped |" in updated_system_text
        assert "## Staleness and verification" in updated_system_text
        assert "<types>\n<type><name>user</name></type>" not in updated_system_text
        assert "Old exclusion text" not in updated_system_text
        assert "Verify the named file still exists." not in updated_system_text
        assert "Memory is one of several persistence mechanisms." not in updated_system_text
        assert "# Environment\nOther Claude system text.\n" in updated_system_text

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_system_prompt_override_count"] == 1
        assert litellm_metadata["claude_system_prompt_override_ids"] == ["auto-memory"]
        assert litellm_metadata["claude_system_prompt_override_failure_ids"] == []
        assert litellm_metadata["claude_system_prompt_override_statuses"] == ["resolved"]
        assert litellm_metadata["claude_system_prompt_override_cc_versions"] == ["2.1.110.abc"]
        assert (
            "context-replacement/claude-code/2.1.110/auto-memory-replacement.md"
            in litellm_metadata["claude_system_prompt_override_template_paths"]
        )
        assert "claude-system-prompt-override" in litellm_metadata["tags"]
        assert "claude-system-prompt-override:auto-memory" in litellm_metadata["tags"]
        assert litellm_metadata["langfuse_spans"][0]["name"] == "claude.system_prompt_override"

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_skips_claude_auto_memory_override_for_unsupported_version(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        original_system_text = (
            "# auto memory\n\n"
            "You have a persistent, file-based memory system at `/tmp/memory/`.\n\n"
            "## Types of memory\n\n"
            "<types>\n<type><name>user</name></type>\n</types>\n\n"
            "## What NOT to save in memory\n\n"
            "- Old exclusion text.\n\n"
            "## Before recommending from memory\n\n"
            "Verify first.\n\n"
            "## Memory and other forms of persistence\n"
            "Old persistence text.\n\n"
            "# Environment\n"
        )
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.109.zzz; cc_entrypoint=cli; cch=42aab;",
                },
                {"type": "text", "text": original_system_text},
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        assert updated_body["system"][1]["text"] == original_system_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert "claude-system-prompt-override" not in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_applies_claude_auto_memory_override_for_newer_compatible_version(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.112.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": (
                        "# auto memory\n\n"
                        "You have a persistent, file-based memory system at `/tmp/memory/`.\n\n"
                        "## Types of memory\n\n"
                        "<types>\n<type><name>user</name></type>\n</types>\n\n"
                        "## What NOT to save in memory\n\n"
                        "- Old exclusion text.\n\n"
                        "## Before recommending from memory\n\n"
                        "Verify first.\n\n"
                        "## Memory and other forms of persistence\n"
                        "Old persistence text.\n\n"
                        "# Environment\n"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        updated_system_text = updated_body["system"][1]["text"]
        assert "persistent, file-based memory system at `/tmp/memory/`" not in updated_system_text
        assert "memory_save(" in updated_system_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_system_prompt_override_cc_versions"] == ["2.1.112.abc"]
        assert "claude-system-prompt-override" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_replaces_persistent_agent_memory_section(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-7",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.113.50e; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": (
                        "\n\n# Persistent Agent Memory\n\n"
                        "You have a persistent, file-based memory system at `/home/zepfu/projects/aawm/.claude/agent-memory-local/eyes/`. "
                        "This directory already exists — write to it directly with the Write tool (do not run mkdir or check for its existence).\n\n"
                        "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. "
                        "If they ask you to forget something, find and remove the relevant entry.\n\n"
                        "## Types of memory\n\n"
                        "There are several discrete types of memory that you can store in your memory system:\n\n"
                        "<types>\n"
                        "<type><name>user</name></type>\n"
                        "<type><name>feedback</name></type>\n"
                        "</types>\n\n"
                        "## What NOT to save in memory\n\n"
                        "- Old exclusion text.\n\n"
                        "## How to save memories\n\n"
                        "Old save flow.\n\n"
                        "## When to access memories\n\n"
                        "- Old access rules.\n\n"
                        "## Before recommending from memory\n\n"
                        "Verify the named file still exists.\n\n"
                        "## Memory and other forms of persistence\n"
                        "Memory is one of several persistence mechanisms.\n\n"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        updated_system_text = updated_body["system"][1]["text"]
        assert "# Persistent Agent Memory" in updated_system_text
        assert "persistent, file-based memory system at" not in updated_system_text
        assert "write to it directly with the Write tool" not in updated_system_text
        assert "memory_save(" in updated_system_text
        assert "memory_forget(source_ids=[...])" in updated_system_text
        assert "| `project` | Agent + project scoped |" in updated_system_text
        assert "## Memory vs plans/tasks" in updated_system_text
        assert "Old exclusion text" not in updated_system_text

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_system_prompt_override_ids"] == ["auto-memory"]
        assert litellm_metadata["claude_system_prompt_override_cc_versions"] == [
            "2.1.113.50e"
        ]
        assert litellm_metadata["claude_system_prompt_override_events"][0][
            "section_heading"
        ] == "# Persistent Agent Memory"
        assert "claude-system-prompt-override:auto-memory" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_compacts_claude_code_tool_advertisements(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        long_bash_description = "Run commands safely. " + ("Prefer dedicated tools. " * 80)
        long_schema_description = (
            "Clear, concise description of what this command does in active voice. "
            "Never use words like complex or risk in the description. "
            + ("Keep it brief. " * 30)
        )
        custom_tool = {
            "name": "custom_lookup",
            "description": "Lookup custom project data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short query text.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
        request_body = {
            "model": "claude-opus-4-7",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.113.50e; cc_entrypoint=cli; cch=42aab;",
                }
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": long_bash_description,
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            },
                            "description": {
                                "type": "string",
                                "description": long_schema_description,
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["default", "background"],
                                "description": "Execution mode.",
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
                custom_tool,
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        bash_tool = updated_body["tools"][0]
        assert bash_tool["name"] == "Bash"
        assert bash_tool["description"].startswith("Run a shell command.")
        assert len(bash_tool["description"]) < len(long_bash_description)
        assert "$schema" not in bash_tool["input_schema"]
        assert bash_tool["input_schema"]["type"] == "object"
        assert bash_tool["input_schema"]["required"] == ["command"]
        assert bash_tool["input_schema"]["additionalProperties"] is False
        assert (
            bash_tool["input_schema"]["properties"]["mode"]["enum"]
            == ["default", "background"]
        )
        compacted_property_description = bash_tool["input_schema"]["properties"][
            "description"
        ]["description"]
        assert len(compacted_property_description) < len(long_schema_description)
        assert "Never use words like complex or risk" in compacted_property_description
        assert updated_body["tools"][1] == custom_tool

        litellm_metadata = updated_body["litellm_metadata"]
        assert "claude-tool-advertisement-compaction" in litellm_metadata["tags"]
        assert litellm_metadata["claude_tool_advertisement_compaction_count"] == 1
        assert litellm_metadata[
            "claude_tool_advertisement_compaction_tool_names"
        ] == ["Bash"]
        assert litellm_metadata["claude_tool_advertisement_compaction_saved_chars"] > 0
        assert (
            litellm_metadata["claude_tool_advertisement_compaction_events"][0][
                "schema_dropped_key_count"
            ]
            == 1
        )
        assert any(
            span["name"] == "claude.tool_advertisement_compaction"
            for span in litellm_metadata["langfuse_spans"]
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_skips_tool_compaction_without_cc_version(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-7",
            "system": [{"type": "text", "text": "plain system prompt"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run commands safely. " + ("Prefer tools. " * 50),
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            }
                        },
                        "required": ["command"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        assert updated_body["tools"] == request_body["tools"]
        litellm_metadata = updated_body.get("litellm_metadata", {})
        assert "claude-tool-advertisement-compaction" not in litellm_metadata.get(
            "tags", []
        )
        assert "claude_tool_advertisement_compaction_count" not in litellm_metadata

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_applies_claude_prompt_patches(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        system_patch_before = (
            "IMPORTANT: Go straight to the point. Try the simplest approach first without going in circles. "
            "Do not overdo it. Be extra concise."
        )
        message_patch_before = (
            "NOTE: You are meant to be a fast agent that returns output as quickly as possible. "
            "In order to achieve this you must:\n"
            "- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations\n"
            "- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files\n\n"
            "Complete the user's search request efficiently and report your findings clearly."
        )
        security_patch_before = (
            "IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, "
            "and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass "
            "targeting, supply chain compromise, or detection evasion for malicious purposes. Dual-use "
            "security tools (C2 frameworks, credential testing, exploit development) require clear "
            "authorization context: pentesting engagements, CTF competitions, security research, or "
            "defensive use cases."
        )
        report_file_patch_before = (
            "Do NOT Write report/summary/findings/analysis .md files. Return findings directly as your final assistant message "
            "— the parent agent reads your text output, not files you create."
        )
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.112.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": f"Prelude\n\n{system_patch_before}\n\n{security_patch_before}\n\n{report_file_patch_before}\n",
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message_patch_before,
                        }
                    ],
                }
            ],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        updated_system_text = updated_body["system"][1]["text"]
        updated_message_text = updated_body["messages"][0]["content"][0]["text"]
        assert system_patch_before not in updated_system_text
        assert (
            "Choose the approach that correctly and completely solves the problem."
            in updated_system_text
        )
        assert security_patch_before not in updated_system_text
        assert (
            "IMPORTANT: Assist with authorized security testing, defensive security, incident response, security research, CTF challenges, malware analysis, and educational use."
            in updated_system_text
        )
        assert "Refuse requests for destructive techniques" not in updated_system_text
        assert report_file_patch_before not in updated_system_text
        assert (
            "Do NOT Write report/summary/findings/analysis .md files unless EXPLICITLY asked to do."
            in updated_system_text
        )
        assert (
            "Regardless of a file write-- you need to return findings directly as your final assistant message."
            in updated_system_text
        )
        assert message_patch_before not in updated_message_text
        assert "NOTE: Be thorough in your exploration." in updated_message_text
        assert (
            "Complete the user's search request thoroughly and report your findings clearly."
            in updated_message_text
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_prompt_patch_count"] == 4
        assert litellm_metadata["claude_prompt_patch_replacement_count"] == 4
        assert litellm_metadata["claude_prompt_patch_ids"] == [
            "explore-agent-speed-note",
            "output-efficiency-important-line",
            "security-authorized-use-instruction",
            "subagent-report-file-explicit-request",
        ]
        assert litellm_metadata["claude_prompt_patch_failure_ids"] == []
        assert litellm_metadata["claude_prompt_patch_statuses"] == [
            "resolved",
            "resolved",
            "resolved",
            "resolved",
        ]
        assert litellm_metadata["claude_prompt_patch_cc_versions"] == ["2.1.112.abc"]
        assert (
            "context-replacement/claude-code/prompt-patches/roman01la-2026-04-02.json"
            in litellm_metadata["claude_prompt_patch_manifest_paths"]
        )
        assert "claude-prompt-patch" in litellm_metadata["tags"]
        assert (
            "claude-prompt-patch:output-efficiency-important-line"
            in litellm_metadata["tags"]
        )
        assert "claude-prompt-patch:explore-agent-speed-note" in litellm_metadata["tags"]
        assert (
            "claude-prompt-patch:security-authorized-use-instruction"
            in litellm_metadata["tags"]
        )
        assert (
            "claude-prompt-patch:subagent-report-file-explicit-request"
            in litellm_metadata["tags"]
        )
        assert any(
            span.get("name") == "claude.prompt_patch"
            for span in litellm_metadata["langfuse_spans"]
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_skips_claude_prompt_patches_when_no_match(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.112.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": "System text without any patch candidates.",
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert "claude-prompt-patch" not in litellm_metadata["tags"]
        assert "claude_prompt_patch_ids" not in litellm_metadata

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_patches_report_file_instruction_in_plain_string_system(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        report_file_patch_before = (
            "Do NOT Write report/summary/findings/analysis .md files. Return findings directly as your final assistant message "
            "— the parent agent reads your text output, not files you create."
        )
        report_file_template_before = (
            "Do NOT ${$4} report/summary/findings/analysis .md files. Return findings directly as your final assistant message."
        )
        request_body = {
            "model": "claude-opus-4-6",
            "system": (
                "x-anthropic-billing-header: cc_version=2.1.119; cc_entrypoint=cli; cch=42aab;\n\n"
                f"{report_file_patch_before}\n\n{report_file_template_before}"
            ),
            "messages": [{"role": "user", "content": "write the review file"}],
        }

        updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
            mock_request, request_body
        )

        updated_system_text = updated_body["system"]
        assert report_file_patch_before not in updated_system_text
        assert report_file_template_before not in updated_system_text
        assert (
            updated_system_text.count(
                "Do NOT Write report/summary/findings/analysis .md files unless EXPLICITLY asked to do."
            )
            == 2
        )
        assert (
            updated_system_text.count(
                "Regardless of a file write-- you need to return findings directly as your final assistant message."
            )
            == 2
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_prompt_patch_ids"] == [
            "subagent-report-file-explicit-request",
        ]
        assert litellm_metadata["claude_prompt_patch_replacement_count"] == 2
        assert (
            "claude-prompt-patch:subagent-report-file-explicit-request"
            in litellm_metadata["tags"]
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_rewrites_commonmark_prompt_with_reference_identifiers(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_dynamic_injection_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-identifiers-1"}},
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.112.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": (
                        "You are 'eyes' and you are working on the 'aawm' project.\n"
                        "Your output will be displayed on a command line interface. "
                        "Your responses should be short and concise. "
                        "You can use Github-flavored markdown for formatting, and will be rendered in a "
                        "monospace font using the CommonMark specification.\n"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_reference_identifier_list",
            new=AsyncMock(return_value="api, cli, dal"),
        ) as mock_identifier_list:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        updated_system_text = updated_body["system"][1]["text"]
        assert (
            aawm_claude_control_plane._CLAUDE_COMMONMARK_PROMPT_SENTENCE
            not in updated_system_text
        )
        assert (
            "You can use Github-flavored markdown for formatting, and will be rendered in a monospace font "
            "using the CommonMark specification plus the following as a custom known list of technical "
            "identifiers: api, cli, dal."
        ) in updated_system_text

        litellm_metadata = updated_body["litellm_metadata"]
        assert "technical-identifiers-list" in litellm_metadata["claude_prompt_patch_ids"]
        technical_identifiers_event = next(
            event
            for event in litellm_metadata["claude_prompt_patch_events"]
            if event["id"] == "technical-identifiers-list"
        )
        assert technical_identifiers_event["status"] == "resolved"
        assert technical_identifiers_event["cache_status"] == "miss"
        assert technical_identifiers_event["identifier_count"] == 3
        mock_identifier_list.assert_awaited_once_with(
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_reuses_cached_reference_identifier_list(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_dynamic_injection_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-identifiers-2"}},
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.112.abc; cc_entrypoint=cli; cch=42aab;",
                },
                {
                    "type": "text",
                    "text": (
                        "You are 'eyes' and you are working on the 'aawm' project.\n"
                        "You can use Github-flavored markdown for formatting, and will be rendered in a "
                        "monospace font using the CommonMark specification.\n"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_reference_identifier_list",
            new=AsyncMock(return_value="api, cli, dal"),
        ) as mock_identifier_list:
            first_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )
            second_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        first_identifier_event = next(
            event
            for event in first_body["litellm_metadata"]["claude_prompt_patch_events"]
            if event["id"] == "technical-identifiers-list"
        )
        second_identifier_event = next(
            event
            for event in second_body["litellm_metadata"]["claude_prompt_patch_events"]
            if event["id"] == "technical-identifiers-list"
        )
        assert first_identifier_event["cache_status"] == "miss"
        assert second_identifier_event["cache_status"] == "hit"
        mock_identifier_list.assert_awaited_once_with(
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_expands_aawm_dynamic_injection(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "<!-- AAWM p=get_agent_memories ctx=agent,tenant -->\n"
                            ),
                        }
                    ],
                }
            ],
        }

        mock_get_agent_memories = AsyncMock(
            return_value="# Feedback Memories\n[id:12345678] Memory line"
        )
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=mock_get_agent_memories,
        ), patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_get_agent_memories",
            new=mock_get_agent_memories,
        ):
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "<!-- AAWM" not in injected_text
        assert "# Feedback Memories" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_count"] == 1
        assert litellm_metadata["aawm_dynamic_injection_procs"] == ["get_agent_memories"]
        assert litellm_metadata["aawm_dynamic_injection_failure_procs"] == []
        assert litellm_metadata["aawm_dynamic_injection_context_keys"] == ["agent", "tenant"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["resolved"]
        assert litellm_metadata["aawm_dynamic_injection_cache_hits"] == 0
        assert litellm_metadata["aawm_dynamic_injection_cache_misses"] == 1
        assert "aawm-dynamic-injection" in litellm_metadata["tags"]
        assert "aawm-proc:get_agent_memories" in litellm_metadata["tags"]
        assert litellm_metadata["langfuse_spans"][0]["name"] == "aawm.dynamic_injection"

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_reuses_cached_aawm_dynamic_injection(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_dynamic_injection_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-cache-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "<!-- AAWM p=get_agent_memories ctx=agent,tenant -->\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value="# Feedback Memories\n[id:12345678] Memory line"),
        ) as mock_get_agent_memories:
            first_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )
            second_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        first_metadata = first_body["litellm_metadata"]
        second_metadata = second_body["litellm_metadata"]
        assert first_metadata["aawm_dynamic_injection_cache_misses"] == 1
        assert first_metadata["aawm_dynamic_injection_cache_hits"] == 0
        assert second_metadata["aawm_dynamic_injection_cache_hits"] == 1
        assert second_metadata["aawm_dynamic_injection_cache_misses"] == 0
        assert second_metadata["aawm_dynamic_injection_cache_statuses"] == ["hit"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_expands_plaintext_aawm_dynamic_injection(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-dynamic-plain-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "AAWM p=get_agent_memories ctx=agent,tenant\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value="# Feedback Memories\n[id:87654321] Memory line"),
        ) as mock_get_agent_memories:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "\nAAWM p=get_agent_memories ctx=agent,tenant\n" not in injected_text
        assert "# Feedback Memories" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["resolved"]
        assert "aawm-dynamic-injection" in litellm_metadata["tags"]
        assert "aawm-proc:get_agent_memories" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_ignores_non_directive_aawm_text(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "AAWM maintains a managed fork of BerriAI/litellm for custom patches.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(),
        ) as mock_get_agent_memories:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert (
            "AAWM maintains a managed fork of BerriAI/litellm for custom patches."
            in injected_text
        )
        assert "litellm_metadata" not in updated_body or (
            "aawm-dynamic-injection"
            not in (updated_body["litellm_metadata"].get("tags") or [])
        )
        mock_get_agent_memories.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_expands_at_wrapped_aawm_dynamic_injection(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-dynamic-at-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "@@@ AAWM p=get_agent_memories ctx=agent,tenant @@@\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value="# Feedback Memories\n[id:99887766] Memory line"),
        ) as mock_get_agent_memories:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "@@@ AAWM p=get_agent_memories ctx=agent,tenant @@@" not in injected_text
        assert "# Feedback Memories" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["resolved"]
        assert "aawm-dynamic-injection" in litellm_metadata["tags"]
        assert "aawm-proc:get_agent_memories" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_call_aawm_context_grab_uses_tristore_search_exact_scope(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(
            return_value=[
                {"content": "Primary reference"},
                {"content": "Fallback reference"},
                {"content": "   "},
            ]
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._get_aawm_dynamic_injection_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await aawm_claude_control_plane._call_aawm_context_grab(
                name="alpha",
                tenant_id="aawm",
                agent_id="eyes",
            )

        assert result == "Primary reference\n\nFallback reference"
        mock_pool.fetch.assert_awaited_once_with(
            "SELECT content FROM tristore_search_exact($1, $2, $3)",
            "alpha",
            "aawm",
            "eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_expands_ctx_marker_appendix(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-ctx-marker-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Use :#alpha.ctx#: for this task.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value="Alpha context line"),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert ":#alpha.ctx#:" not in injected_text
        assert "Use alpha for this task." in injected_text
        assert "Alpha context line" in injected_text
        assert "~retrieved at: " in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_context_names"] == ["alpha"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["resolved"]
        assert litellm_metadata["aawm_dynamic_injection_cache_hits"] == 0
        assert litellm_metadata["aawm_dynamic_injection_cache_misses"] == 1
        mock_context_grab.assert_awaited_once_with(
            name="alpha",
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_preserves_escaped_ctx_marker_literal(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-ctx-escaped-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Use \\\\:#alpha.ctx#\\\\: as a literal marker.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value="Alpha context line"),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "\\\\:#alpha.ctx#\\\\:" not in injected_text
        assert "Use :#alpha.ctx#: as a literal marker." in injected_text
        assert "Alpha context line" not in injected_text
        assert "~retrieved at: " not in injected_text
        assert "aawm_dynamic_injection_count" not in updated_body.get(
            "litellm_metadata", {}
        )
        mock_context_grab.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_ctx_marker_dedupes_and_preserves_append_order(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-ctx-marker-2"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Use :#alpha.ctx#:, then :#beta.ctx#:, then :#alpha.ctx#: again.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        async def _mock_context_grab(
            *, name: str, tenant_id: str, agent_id: str
        ) -> str:
            return f"{name.title()} context line"

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(side_effect=_mock_context_grab),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert ":#alpha.ctx#:" not in injected_text
        assert ":#beta.ctx#:" not in injected_text
        assert "Use alpha, then beta, then alpha again." in injected_text
        assert injected_text.index("Alpha context line") < injected_text.index(
            "Beta context line"
        )
        assert mock_context_grab.await_count == 2
        assert [
            call.kwargs["name"] for call in mock_context_grab.await_args_list
        ] == ["alpha", "beta"]
        assert [
            (call.kwargs["tenant_id"], call.kwargs["agent_id"])
            for call in mock_context_grab.await_args_list
        ] == [("aawm", "eyes"), ("aawm", "eyes")]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_reuses_cached_ctx_marker_result(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_context_grab_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-ctx-cache-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Use :#alpha.ctx#: for this task.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value="Cached context line"),
        ) as mock_context_grab:
            first_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )
            second_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        first_metadata = first_body["litellm_metadata"]
        second_metadata = second_body["litellm_metadata"]
        assert first_metadata["aawm_dynamic_injection_cache_hits"] == 0
        assert first_metadata["aawm_dynamic_injection_cache_misses"] == 1
        assert second_metadata["aawm_dynamic_injection_cache_hits"] == 1
        assert second_metadata["aawm_dynamic_injection_cache_misses"] == 0
        assert second_metadata["aawm_dynamic_injection_cache_statuses"] == ["hit"]
        mock_context_grab.assert_awaited_once_with(
            name="alpha",
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_ctx_marker_appends_warning_for_no_results(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-ctx-empty-1"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Before :#missing.ctx#: after.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value=None),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert ":#missing.ctx#:" not in injected_text
        assert "Before missing after." in injected_text
        assert "~retrieved at: " not in injected_text
        assert (
            "IMPORTANT: context grab for missing returned no results. immediately inform the opperator."
            in injected_text
        )
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_context_names"] == ["missing"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["empty"]
        mock_context_grab.assert_awaited_once_with(
            name="missing",
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_dispatch_backticks_appendix_only_for_subagent_context(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_context_grab_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-backtick-1"}},
            "litellm_metadata": {"claude_persisted_output_hooks": ["subagentstart"]},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Check `dal`, then ```bash\n`ignored`\n```, then `cli`, then `api`, then `dal`.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        async def _mock_context_grab(
            *, name: str, tenant_id: str, agent_id: str
        ) -> str:
            return f"{name.title()} context line"

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(side_effect=_mock_context_grab),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "Check `dal`, then ```bash\n`ignored`\n```, then `cli`, then `api`, then `dal`." in injected_text
        assert injected_text.index("Dal context line") < injected_text.index(
            "Cli context line"
        )
        assert injected_text.index("Cli context line") < injected_text.index(
            "Api context line"
        )
        assert "`ignored`" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_context_names"] == ["api", "cli", "dal"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == [
            "resolved",
            "resolved",
            "resolved",
        ]
        assert [
            event["placeholder_type"]
            for event in litellm_metadata["aawm_dynamic_injection_events"]
        ] == [
            "dispatch_backtick",
            "dispatch_backtick",
            "dispatch_backtick",
        ]
        assert [
            call.kwargs["name"] for call in mock_context_grab.await_args_list
        ] == ["dal", "cli", "api"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_dispatch_acronyms_appendix_only_for_subagent_context(
        self,
    ):
        from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane

        aawm_claude_control_plane._aawm_context_grab_cache.clear()
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-acronym-1"}},
            "litellm_metadata": {"claude_persisted_output_hooks": ["subagentstart"]},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\nSubagentStart hook additional context: keep API hidden here.\n</system-reminder>\n"
                                "Focus on DAL, then API, then ```text\nCLI\n```, then CLI, then DAL.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        async def _mock_context_grab(
            *, name: str, tenant_id: str, agent_id: str
        ) -> str:
            return f"{name} context line"

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(side_effect=_mock_context_grab),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "Focus on DAL, then API, then ```text\nCLI\n```, then CLI, then DAL." in injected_text
        assert injected_text.index("DAL context line") < injected_text.index(
            "API context line"
        )
        assert injected_text.index("API context line") < injected_text.index(
            "CLI context line"
        )
        assert "keep API hidden here" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_context_names"] == ["API", "CLI", "DAL"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == [
            "resolved",
            "resolved",
            "resolved",
        ]
        assert [
            event["placeholder_type"]
            for event in litellm_metadata["aawm_dynamic_injection_events"]
        ] == [
            "dispatch_acronym",
            "dispatch_acronym",
            "dispatch_acronym",
        ]
        assert [
            call.kwargs["name"] for call in mock_context_grab.await_args_list
        ] == ["DAL", "API", "CLI"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_dispatch_backticks_are_silent_on_missing_context(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-backtick-2"}},
            "litellm_metadata": {"claude_persisted_output_hooks": ["subagentstart"]},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Keep `missing` literal.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value=None),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "Keep `missing` literal." in injected_text
        assert "~retrieved at: " not in injected_text
        assert (
            "IMPORTANT: context grab for missing returned no results. immediately inform the opperator."
            not in injected_text
        )
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_context_names"] == ["missing"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["empty"]
        assert litellm_metadata["aawm_dynamic_injection_events"][0]["placeholder_type"] == (
            "dispatch_backtick"
        )
        mock_context_grab.assert_awaited_once_with(
            name="missing",
            tenant_id="aawm",
            agent_id="eyes",
        )

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_ignores_dispatch_backticks_without_subagent_context(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "metadata": {"user_id": {"session_id": "session-backtick-3"}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'eyes' and you are working on the 'aawm' project.\n"
                                "Keep `alpha` unchanged.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane._call_aawm_context_grab",
            new=AsyncMock(return_value="Alpha context line"),
        ) as mock_context_grab:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "Keep `alpha` unchanged." in injected_text
        assert "~retrieved at: " not in injected_text
        assert updated_body.get("litellm_metadata", {}).get(
            "aawm_dynamic_injection_context_names"
        ) is None
        mock_context_grab.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_expands_aawm_dynamic_injection_to_no_memories_block(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'orchestrator' and you are working on the 'aawm' project.\n"
                                "Before\n<!-- AAWM p=get_agent_memories ctx=agent,tenant -->\nAfter\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value=None),
        ):
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "<!-- AAWM" not in injected_text
        assert 'AAWM "get_agent_memories" failed' not in injected_text
        assert "# Memory Injection" in injected_text
        assert "You have saved no memories as of yet." in injected_text
        assert "Before\n# Memory Injection" in injected_text
        assert "You have saved no memories as of yet.\n\nAfter" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_statuses"] == ["empty"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_tags_post_rewrite_context_files(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are 'orchestrator' and you are working on the 'litellm' project.\n"
                                "Contents of /home/zepfu/.claude/projects/-home-zepfu-projects-litellm/memory/MEMORY.md:\n"
                                "@@@ AAWM p=get_agent_memories ctx=agent,tenant @@@\n"
                                "Contents of /home/zepfu/projects/litellm/CLAUDE.md:\n"
                                "Project instructions.\n"
                            ),
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value="# Project Memories\nMemory line"),
        ):
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert "# Project Memories" in injected_text
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["claude_post_rewrite_context_files_present"] == [
            "MEMORY.md",
            "CLAUDE.md",
        ]
        assert litellm_metadata["claude_post_rewrite_context_file_count"] == 2
        assert "claude-post-rewrite-context-file-present" in litellm_metadata["tags"]
        assert "claude-post-rewrite-context-file:memory-md" in litellm_metadata["tags"]
        assert "claude-post-rewrite-context-file:claude-md" in litellm_metadata["tags"]

    @pytest.mark.asyncio
    async def test_prepare_anthropic_request_body_replaces_failed_aawm_dynamic_injection(
        self,
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        request_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<!-- AAWM p=get_agent_memories ctx=agent,tenant -->\n",
                        }
                    ],
                }
            ],
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._call_aawm_get_agent_memories",
            new=AsyncMock(return_value="# should not be called"),
        ) as mock_get_agent_memories:
            updated_body, _, _, _ = await _prepare_anthropic_request_body_for_passthrough(
                mock_request, request_body
            )

        injected_text = updated_body["messages"][0]["content"][0]["text"]
        assert 'AAWM "get_agent_memories" failed for this session.' in injected_text
        assert "Alert the user or session orchestrator." in injected_text
        mock_get_agent_memories.assert_not_awaited()
        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["aawm_dynamic_injection_failure_procs"] == ["get_agent_memories"]
        assert "aawm-dynamic-injection-failed" in litellm_metadata["tags"]

    def test_prepare_request_body_for_passthrough_observability_sets_environment_and_session(
        self, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-abc"}
        request_body = {"model": "gpt-5.4", "input": "hello"}

        updated_body = _prepare_request_body_for_passthrough_observability(
            mock_request,
            request_body,
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["session_id"] == "header-session-abc"
        assert litellm_metadata["trace_environment"] == "dev"

    def test_prepare_request_body_for_passthrough_observability_overrides_stale_environment(
        self, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"session_id": "header-session-abc"}
        request_body = {
            "model": "gpt-5.4",
            "input": "hello",
            "litellm_metadata": {"trace_environment": "prod"},
        }

        updated_body = _prepare_request_body_for_passthrough_observability(
            mock_request,
            request_body,
        )

        litellm_metadata = updated_body["litellm_metadata"]
        assert litellm_metadata["trace_environment"] == "dev"
        assert litellm_metadata["source_trace_environment"] == "prod"

    @pytest.mark.parametrize(
        ("endpoint", "expected_route_family"),
        [
            ("v1/chat/completions", "openai_chat_completions"),
            ("v1/responses", "openai_responses"),
            ("v1/models", "openai_passthrough"),
        ],
    )
    def test_openai_passthrough_route_family(
        self,
        endpoint,
        expected_route_family,
    ):
        assert _get_openai_passthrough_route_family(endpoint) == expected_route_family

    @pytest.mark.parametrize(
        ("endpoint", "expected_route_family"),
        [
            ("v1beta/models/gemini-2.5-flash:generateContent", "gemini_generate_content"),
            (
                "v1beta/models/gemini-2.5-flash:streamGenerateContent",
                "gemini_stream_generate_content",
            ),
            (
                "v1beta/models/veo-2.0-generate-001:predictLongRunning",
                "gemini_predict_long_running",
            ),
            ("v1internal:loadCodeAssist", None),
            ("v1internal:listExperiments", None),
            ("v1internal:retrieveUserQuota", None),
            ("v1internal:fetchAdminControls", None),
        ],
    )
    def test_gemini_passthrough_route_family(
        self,
        endpoint,
        expected_route_family,
    ):
        assert _get_gemini_passthrough_route_family(endpoint) == expected_route_family

    @pytest.mark.asyncio
    async def test_anthropic_proxy_route_extracts_billing_header_before_passthrough(
        self,
    ):
        request_body = {
            "model": "claude-opus-4-6",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.101.a4a; cc_entrypoint=cli; cch=42aab;",
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value=request_body),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new=AsyncMock(return_value=False),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ) as mock_set_parsed_body, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="anthropic-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(return_value={"id": "msg_123"})
            mock_create_route.return_value = mock_endpoint_func

            result = await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            assert result == {"id": "msg_123"}
            mock_set_parsed_body.assert_called_once()
            prepared_body = mock_set_parsed_body.call_args.args[1]
            litellm_metadata = prepared_body["litellm_metadata"]
            assert litellm_metadata["anthropic_billing_header_present"] is True
            assert "anthropic-billing-header" in litellm_metadata["tags"]
            assert (
                "anthropic-billing-header:cc_version=2.1.101.a4a"
                in litellm_metadata["tags"]
            )

    def test_append_openai_beta_header(self):
        print("\nTesting _append_openai_beta_header method...")

        # Create mock requests with different paths
        assistants_request = MagicMock(spec=Request)
        assistants_request.url = MagicMock()
        assistants_request.url.path = "/v1/threads/thread_123456/messages"

        non_assistants_request = MagicMock(spec=Request)
        non_assistants_request.url = MagicMock()
        non_assistants_request.url.path = "/v1/chat/completions"

        headers = {"authorization": "Bearer test_key"}

        # Test with assistants API request
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistants_request
        )
        print(f"Assistants API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"

        # Test with non-assistants API request
        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, non_assistants_request
        )
        print(f"Non-assistants API request: Headers: {result}")
        assert "OpenAI-Beta" not in result

        # Test with assistant in the path
        assistant_request = MagicMock(spec=Request)
        assistant_request.url = MagicMock()
        assistant_request.url.path = "/v1/assistants/asst_123456"

        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistant_request
        )
        print(f"Assistant API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"


@pytest.mark.asyncio
async def test_iterate_google_code_assist_unwrapped_stream_yields_text_chunks():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _iterate_google_code_assist_unwrapped_stream,
    )

    async def _body_iterator():
        yield b'data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\"role\":\"model\"}}]}}\n\n'

    iterator = _iterate_google_code_assist_unwrapped_stream(_body_iterator())
    chunk = await iterator.__anext__()

    assert isinstance(chunk, str)
    assert chunk.startswith('data: {\"candidates\"')


@pytest.mark.asyncio
async def test_iterate_google_code_assist_unwrapped_stream_yields_all_chunks_from_single_event_block():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _iterate_google_code_assist_unwrapped_stream,
    )

    async def _body_iterator():
        yield (
            b'data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Mon Apr 2\"}],\"role\":\"model\"}}]}}\n'
            b'data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"parts\":[{\"text\":\"0 22:02:44 UTC\"}],\"role\":\"model\"}}],\"usageMetadata\":{\"promptTokenCount\":1}}}\n\n'
        )

    iterator = _iterate_google_code_assist_unwrapped_stream(_body_iterator())
    first_chunk = await iterator.__anext__()
    second_chunk = await iterator.__anext__()

    assert isinstance(first_chunk, str)
    assert isinstance(second_chunk, str)
    assert '\"Mon Apr 2\"' in first_chunk
    assert '\"finishReason\": \"STOP\"' in second_chunk
    assert '\"0 22:02:44 UTC\"' in second_chunk

@pytest.mark.asyncio
async def test_iterate_google_code_assist_unwrapped_stream_arms_post_tool_cooldown(monkeypatch):
    import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as module

    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_POST_TOOL_COOLDOWN_SECONDS", "3")
    module._google_adapter_rate_limit_until_monotonic_by_key.clear()

    async def _body_iterator():
        yield (
            b'data: {"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"Bash","args":{"command":"date -u"}}}],"role":"model"}}, {"finishReason":"STOP"}]}}\n\n'
        )

    iterator = module._iterate_google_code_assist_unwrapped_stream(_body_iterator())
    first_chunk = await iterator.__anext__()

    assert 'functionCall' in first_chunk
    assert module._google_adapter_rate_limit_until_monotonic_by_key['__default__'] >= time.monotonic() + 2.5


def _decode_anthropic_sse_events(chunks: list[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    raw_stream = b"".join(chunks).decode("utf-8")
    for event_block in raw_stream.split("\n\n"):
        data_lines = [
            line.removeprefix("data: ")
            for line in event_block.splitlines()
            if line.startswith("data: ")
        ]
        if data_lines:
            events.append(json.loads("\n".join(data_lines)))
    return events


@pytest.mark.asyncio
async def test_google_code_assist_anthropic_stream_preserves_tool_use_and_usage_metadata():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _build_anthropic_streaming_response_from_google_code_assist_stream,
    )

    async def _body_iterator():
        yield (
            b'data: {"traceId":"trace-code-assist-1","response":{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"file_path":"/tmp/a.txt"}},"thoughtSignature":"sig-read"}]}}]}}\n\n'
        )
        yield (
            b'data: {"traceId":"trace-code-assist-1","response":{"candidates":[{"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":10,"trafficType":"ON_DEMAND"}}}\n\n'
        )

    google_stream = StreamingResponse(_body_iterator(), media_type="text/event-stream")
    anthropic_stream = _build_anthropic_streaming_response_from_google_code_assist_stream(
        response=google_stream,
        adapter_model="gemini-3-flash-preview",
        tool_name_mapping={"read_file": "Read"},
        gemini_optional_params={},
    )

    chunks = [chunk async for chunk in anthropic_stream.body_iterator]
    events = _decode_anthropic_sse_events(chunks)

    tool_start_events = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_start_events) == 1
    assert tool_start_events[0]["content_block"]["name"] == "Read"
    assert tool_start_events[0]["content_block"]["id"].startswith("call_")

    tool_delta_events = [
        event
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(tool_delta_events) == 1
    assert json.loads(tool_delta_events[0]["delta"]["partial_json"]) == {
        "file_path": "/tmp/a.txt"
    }

    message_delta_events = [
        event for event in events if event.get("type") == "message_delta"
    ]
    assert message_delta_events[-1]["delta"]["stop_reason"] == "tool_use"
    assert message_delta_events[-1]["usage"]["input_tokens"] == 7
    assert message_delta_events[-1]["usage"]["output_tokens"] == 3
    assert events[-1]["type"] == "message_stop"


@pytest.mark.asyncio
async def test_google_code_assist_anthropic_stream_buffers_parallel_tool_calls_across_chunks():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _build_anthropic_streaming_response_from_google_code_assist_stream,
    )

    async def _body_iterator():
        yield (
            b'data: {"traceId":"trace-code-assist-2","response":{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"file_path":"/tmp/a.txt"}}}]}}]}}\n\n'
        )
        yield (
            b'data: {"traceId":"trace-code-assist-2","response":{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"run_shell_command","args":{"command":"pwd"}}}]}}]}}\n\n'
        )
        yield (
            b'data: {"traceId":"trace-code-assist-2","response":{"candidates":[{"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,"totalTokenCount":13,"trafficType":"ON_DEMAND"}}}\n\n'
        )

    google_stream = StreamingResponse(_body_iterator(), media_type="text/event-stream")
    anthropic_stream = _build_anthropic_streaming_response_from_google_code_assist_stream(
        response=google_stream,
        adapter_model="gemini-3-flash-preview",
        tool_name_mapping={"read_file": "Read", "run_shell_command": "Bash"},
        gemini_optional_params={},
    )

    chunks = [chunk async for chunk in anthropic_stream.body_iterator]
    events = _decode_anthropic_sse_events(chunks)

    tool_start_events = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    assert [event["content_block"]["name"] for event in tool_start_events] == [
        "Read",
        "Bash",
    ]

    tool_delta_events = [
        event
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert [json.loads(event["delta"]["partial_json"]) for event in tool_delta_events] == [
        {"file_path": "/tmp/a.txt"},
        {"command": "pwd"},
    ]

    message_delta_events = [
        event for event in events if event.get("type") == "message_delta"
    ]
    assert message_delta_events[-1]["delta"]["stop_reason"] == "tool_use"
    assert message_delta_events[-1]["usage"]["input_tokens"] == 9
    assert message_delta_events[-1]["usage"]["output_tokens"] == 4
    assert events[-1]["type"] == "message_stop"


@pytest.mark.asyncio
async def test_google_code_assist_non_stream_preserves_tool_use_after_normalization():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _translate_google_code_assist_response_to_anthropic,
    )

    google_response = Response(
        content=json.dumps(
            {
                "traceId": "trace-code-assist-non-stream",
                "response": {
                    "candidates": [
                        {
                            "index": 0,
                            "finishReason": "STOP",
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "read_file",
                                            "args": {"file_path": "/tmp/a.txt"},
                                        },
                                        "thoughtSignature": "sig-read",
                                    }
                                ],
                            },
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 16,
                        "trafficType": "ON_DEMAND",
                    },
                },
            }
        ),
        media_type="application/json",
    )

    anthropic_response = await _translate_google_code_assist_response_to_anthropic(
        response=google_response,
        adapter_model="gemini-3-flash-preview",
        tool_name_mapping={"read_file": "Read"},
        completion_messages=[{"role": "user", "content": "read a file"}],
        gemini_optional_params={},
        litellm_params={},
        logging_obj=SimpleNamespace(post_call=lambda **_: None, optional_params={}),
    )
    payload = json.loads(anthropic_response.body.decode("utf-8"))

    assert payload["id"] == "trace-code-assist-non-stream"
    assert payload["stop_reason"] == "tool_use"
    assert payload["usage"] == {"input_tokens": 11, "output_tokens": 5}
    assert payload["content"] == [
        {
            "type": "tool_use",
            "id": payload["content"][0]["id"],
            "name": "Read",
            "input": {"file_path": "/tmp/a.txt"},
            "provider_specific_fields": {"signature": "sig-read"},
        }
    ]


@pytest.mark.asyncio
async def test_google_code_assist_round_trips_same_name_parallel_tool_results():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"session_id": "same-name-parallel-tools"}
    completion_kwargs = {
        "model": "google/gemini-3-flash-preview",
        "max_tokens": 32,
        "parallel_tool_calls": True,
        "tools": [
            {
                "name": "Read",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Read both files."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_a",
                        "name": "Read",
                        "input": {"file_path": "/tmp/a.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_read_b",
                        "name": "Read",
                        "input": {"file_path": "/tmp/b.txt"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_a",
                        "content": "alpha",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_b",
                        "content": "bravo",
                    },
                ],
            },
        ],
    }

    wrapped_request, tool_name_mapping, _, _, _, changes = (
        await _build_google_code_assist_request_from_completion_kwargs(
            completion_kwargs=completion_kwargs,
            adapter_model="gemini-3-flash-preview",
            project="test-project",
            request=mock_request,
        )
    )

    parts = [
        part
        for content in wrapped_request["request"]["contents"]
        for part in content.get("parts", [])
        if isinstance(part, dict)
    ]
    function_calls = [part["functionCall"] for part in parts if "functionCall" in part]
    function_responses = [
        part["functionResponse"] for part in parts if "functionResponse" in part
    ]

    assert function_calls == [
        {"name": "read_file", "args": {"file_path": "/tmp/a.txt"}},
        {"name": "read_file", "args": {"file_path": "/tmp/b.txt"}},
    ]
    assert function_responses == [
        {
            "name": "read_file",
            "response": {"output": "alpha", "tool_use_id": "toolu_read_a"},
        },
        {
            "name": "read_file",
            "response": {"output": "bravo", "tool_use_id": "toolu_read_b"},
        },
    ]
    assert tool_name_mapping["read_file"] == "Read"
    assert changes["google_adapter_annotated_duplicate_tool_response_count"] == 2


async def test_gemini_proxy_route_code_assist_oauth_passthrough_target():
    captured_call = {}

    def fake_create_pass_through_route(*args, **kwargs):
        captured_call.update(kwargs)

        async def _endpoint_func(request, fastapi_response, user_api_key_dict, **_):
            return {"ok": True}

        return _endpoint_func

    body = b'{"model":"gemini-3-flash-preview","contents":[{"role":"user","parts":[{"text":"hello"}]}]}'
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gemini/v1internal:generateContent",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer ya29.test-oauth-token"),
        ],
    }

    async def async_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope=scope, receive=async_receive)

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
        side_effect=fake_create_pass_through_route,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
        new=AsyncMock(return_value=MagicMock()),
    ):
        response = await gemini_proxy_route(
            endpoint="v1internal:generateContent",
            request=request,
            fastapi_response=Response(),
        )

    assert response == {"ok": True}
    assert (
        captured_call["target"]
        == "https://cloudcode-pa.googleapis.com/v1internal:generateContent"
    )
    assert captured_call["custom_llm_provider"] == "gemini"
    assert captured_call["_forward_headers"] is True
    assert captured_call["query_params"] == {}


@pytest.mark.asyncio
async def test_openai_passthrough_route_sets_repository_trace_environment_and_session(
    monkeypatch,
):
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.headers = {
        "session_id": "codex-session-123",
        "user-agent": "codex-cli/1.0",
        "x-aawm-repository": "git@github.com:zepfu/litellm.git",
    }
    mock_request.query_params = {}
    mock_response = MagicMock(spec=Response)
    mock_user_api_key_dict = MagicMock()

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        new=AsyncMock(
            return_value={
                "model": "gpt-5.4",
                "input": "hello",
                "reasoning": {"effort": "xhigh"},
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "prompt-cache-key-123",
            }
        ),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
    ) as mock_set_parsed_body, patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
        return_value=AsyncMock(return_value={"ok": True}),
    ):
        result = await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
            endpoint="/responses",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
            base_target_url="https://api.openai.com",
            api_key="test_api_key",
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        )

    assert result == {"ok": True}
    mock_set_parsed_body.assert_called_once()
    prepared_body = mock_set_parsed_body.call_args.args[1]
    litellm_metadata = prepared_body["litellm_metadata"]
    assert litellm_metadata["session_id"] == "codex-session-123"
    assert litellm_metadata["trace_environment"] == "dev"
    assert litellm_metadata["repository"] == "zepfu/litellm"
    assert litellm_metadata["passthrough_route_family"] == "codex_responses"
    assert "route:codex_responses" in litellm_metadata["tags"]


@pytest.mark.asyncio
async def test_gemini_proxy_route_sets_trace_environment_and_session(monkeypatch):
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")
    body = (
        b'{"model":"gemini-3-flash-preview","request":{"session_id":"gemini-session-123"},'
        b'"generationConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"HIGH","thinkingBudget":512}},'
        b'"tools":[{"googleSearch":{}}],"user_prompt_id":"prompt-123","project":"project-a",'
        b'"contents":[{"role":"user","parts":[{"text":"hello"}]}]}'
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gemini/v1internal:generateContent",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer ya29.test-oauth-token"),
            (b"x-aawm-repository", b"https://github.com/zepfu/litellm.git"),
        ],
    }

    async def async_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope=scope, receive=async_receive)

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
        return_value=AsyncMock(return_value={"ok": True}),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
        new=AsyncMock(return_value=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
    ) as mock_set_parsed_body:
        response = await gemini_proxy_route(
            endpoint="v1internal:generateContent",
            request=request,
            fastapi_response=Response(),
        )

    assert response == {"ok": True}
    mock_set_parsed_body.assert_called_once()
    prepared_body = mock_set_parsed_body.call_args.args[1]
    litellm_metadata = prepared_body["litellm_metadata"]
    assert litellm_metadata["session_id"] == "gemini-session-123"
    assert litellm_metadata["trace_environment"] == "dev"
    assert litellm_metadata["repository"] == "zepfu/litellm"
    assert litellm_metadata["gemini_thinking_config_present"] is True
    assert litellm_metadata["gemini_include_thoughts"] is True
    assert litellm_metadata["gemini_thinking_level"] == "high"
    assert litellm_metadata["gemini_thinking_budget"] == 512
    assert litellm_metadata["gemini_tools_present"] is True
    assert litellm_metadata["gemini_tool_count"] == 1
    assert litellm_metadata["gemini_user_prompt_id"] == "prompt-123"
    assert litellm_metadata["gemini_project"] == "project-a"
    assert litellm_metadata["passthrough_route_family"] == "gemini_generate_content"
    assert "route:gemini_generate_content" in litellm_metadata["tags"]
    assert "gemini-thinking-config-present" in litellm_metadata["tags"]
    assert "gemini-include-thoughts:true" in litellm_metadata["tags"]
    assert "include-thoughts:true" in litellm_metadata["tags"]
    assert "gemini-thinking-level:high" in litellm_metadata["tags"]
    assert "thinking-level:high" in litellm_metadata["tags"]
    assert "gemini-thinking-budget-configured" in litellm_metadata["tags"]
    assert "gemini-tools-present" in litellm_metadata["tags"]

    def test_assemble_headers(self):
        print("\nTesting _assemble_headers method...")

        # Mock request
        mock_request = MagicMock(spec=Request)
        api_key = "test_api_key"

        # Patch the _append_openai_beta_header method to avoid testing it again
        with patch.object(
            BaseOpenAIPassThroughHandler,
            "_append_openai_beta_header",
            return_value={
                "authorization": "Bearer test_api_key",
                "api-key": "test_api_key",
                "test-header": "value",
            },
        ):
            result = BaseOpenAIPassThroughHandler._assemble_headers(
                api_key, mock_request
            )
            print(f"Assembled headers: {result}")
            assert result["authorization"] == "Bearer test_api_key"
            assert result["api-key"] == "test_api_key"
            assert result["test-header"] == "value"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
    )
    async def test_base_openai_pass_through_handler(self, mock_create_pass_through):
        print("\nTesting _base_openai_pass_through_handler method...")

        # Mock dependencies
        mock_request = MagicMock(spec=Request)
        mock_request.query_params = {"model": "gpt-4"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock the endpoint function returned by create_pass_through_route
        mock_endpoint_func = AsyncMock(return_value={"result": "success"})
        mock_create_pass_through.return_value = mock_endpoint_func

        print("Testing standard endpoint pass-through...")
        # Test with standard endpoint
        result = await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
            base_target_url="https://api.openai.com",
            api_key="test_api_key",
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        )

        # Verify the result
        print(f"Result from handler: {result}")
        assert result == {"result": "success"}

        # Verify create_pass_through_route was called with correct parameters
        call_args = mock_create_pass_through.call_args[1]
        print(
            f"create_pass_through_route called with endpoint: {call_args['endpoint']}"
        )
        print(f"create_pass_through_route called with target: {call_args['target']}")
        assert call_args["endpoint"] == "/chat/completions"
        assert call_args["target"] == "https://api.openai.com/v1/chat/completions"

        # Verify endpoint_func was called with correct parameters
        print("Verifying endpoint_func call parameters...")
        mock_endpoint_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_base_openai_pass_through_handler_sets_trace_environment_and_session(
        self, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", "dev")

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "session_id": "codex-session-123",
            "user-agent": "codex-cli/1.0",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(
                return_value={
                    "model": "gpt-5.4",
                    "input": "hello",
                    "reasoning": {"effort": "xhigh"},
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "include": ["reasoning.encrypted_content"],
                    "prompt_cache_key": "prompt-cache-key-123",
                }
            ),
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ) as mock_set_parsed_body, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            return_value=AsyncMock(return_value={"ok": True}),
        ):
            result = await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/responses",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
                base_target_url="https://api.openai.com",
                api_key="test_api_key",
                custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            )

        assert result == {"ok": True}
        mock_set_parsed_body.assert_called_once()
        prepared_body = mock_set_parsed_body.call_args.args[1]
        litellm_metadata = prepared_body["litellm_metadata"]
        assert litellm_metadata["session_id"] == "codex-session-123"
        assert litellm_metadata["trace_environment"] == "dev"
        assert litellm_metadata["codex_reasoning_effort"] == "xhigh"
        assert litellm_metadata["codex_tool_choice"] == "auto"
        assert litellm_metadata["codex_parallel_tool_calls"] is True
        assert litellm_metadata["codex_include"] == ["reasoning.encrypted_content"]
        assert litellm_metadata["codex_prompt_cache_key_present"] is True
        assert litellm_metadata["passthrough_route_family"] == "codex_responses"
        assert "route:codex_responses" in litellm_metadata["tags"]
        assert "codex-effort:xhigh" in litellm_metadata["tags"]
        assert "effort:xhigh" in litellm_metadata["tags"]
        assert "codex-tool-choice:auto" in litellm_metadata["tags"]
        assert "codex-parallel-tools:true" in litellm_metadata["tags"]
        assert "codex-include:reasoning.encrypted_content" in litellm_metadata["tags"]


class TestVertexAIPassThroughHandler:
    """
    Case 1: User set passthrough credentials - confirm credentials used.

    Case 2: User set default credentials, no exact passthrough credentials - confirm default credentials used.

    Case 3: No default credentials, no mapped credentials - request passed through directly.
    """

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_credentials(self, monkeypatch):
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-1.5-flash:generateContent"

        # Mock request
        mock_request = Mock()
        mock_request.state = None  # Prevent Mock from returning a truthy _cached_headers
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-creds",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = vertex_credentials

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the vertex handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                f"https://{test_location}-aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value=f"https://{test_location}-aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            try:
                result = await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict={"api_key": "test-key"},
                )
            except Exception as e:
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{test_location}-aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {test_token}"},
                is_streaming_request=False,
            )

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_global_location(self, monkeypatch):
        """
        Test that when global location is used, it is correctly handled in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "global"
        vertex_credentials = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-1.5-flash:generateContent"

        # Mock request
        mock_request = Mock()
        mock_request.state = None  # Prevent Mock from returning a truthy _cached_headers
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-creds",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = vertex_credentials

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the vertex handler for global location
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                "https://aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value="https://aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            try:
                result = await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict={"api_key": "test-key"},
                )
            except Exception as e:
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {test_token}"},
                is_streaming_request=False,
            )

    @pytest.mark.parametrize(
        "initial_endpoint",
        [
            "publishers/google/models/gemini-1.5-flash:generateContent",
            "v1/projects/bad-project/locations/bad-location/publishers/google/models/gemini-1.5-flash:generateContent",
        ],
    )
    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_default_credentials(
        self, monkeypatch, initial_endpoint
    ):
        """
        Test that when no passthrough credentials are set, default credentials are used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        # Setup default credentials
        default_project = "default-project"
        default_location = "us-central1"
        default_credentials = "default-creds"

        pass_through_router = PassthroughEndpointRouter()
        pass_through_router.default_vertex_config = VertexPassThroughCredentials(
            vertex_project=default_project,
            vertex_location=default_location,
            vertex_credentials=default_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        # Use different project/location in request than the default
        endpoint = initial_endpoint

        mock_request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": f"/vertex_ai/{endpoint}",
                "headers": {},
            }
        )
        mock_response = Response()

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new_callable=AsyncMock,
        ) as mock_auth:
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = default_credentials

            mock_load_auth.return_value = (mock_credentials, default_project)
            mock_auth.return_value = MagicMock()

            # Mock the vertex handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                f"https://{default_location}-aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value=f"https://{default_location}-aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            try:
                await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")

            # Verify default credentials were used
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{default_location}-aiplatform.googleapis.com/v1/projects/{default_project}/locations/{default_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {default_credentials}"},
                is_streaming_request=False,
            )

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_no_default_credentials(self, monkeypatch):
        """
        Test that when no default credentials are set, the request fails
        """
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "my-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        test_project = "test-project"
        test_location = "test-location"
        test_token = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent"

        # Mock request
        mock_request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": endpoint,
                "headers": [
                    (b"authorization", b"Bearer test-creds"),
                ],
            }
        )

        # Mock response
        mock_response = Response()

        with mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
        ) as mock_ensure_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
        ) as mock_get_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new_callable=AsyncMock,
        ) as mock_auth:
            mock_ensure_token.return_value = ("test-auth-header", test_project)
            mock_get_token.return_value = (test_token, "")
            mock_auth.return_value = MagicMock()

            # Call the route
            try:
                await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{test_location}-aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"authorization": f"Bearer {test_token}"},
                is_streaming_request=False,
            )

    @pytest.mark.asyncio
    async def test_async_vertex_proxy_route_api_key_auth(self):
        """
        Critical

        This is how Vertex AI JS SDK will Auth to Litellm Proxy
        """
        # Mock dependencies
        mock_request = Mock()
        mock_request.headers = {"x-litellm-api-key": "test-key-123"}
        mock_request.method = "POST"
        mock_response = Mock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_auth:
            mock_auth.return_value = {"api_key": "test-key-123"}

            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_pass_through:
                mock_pass_through.return_value = AsyncMock(
                    return_value={"status": "success"}
                )

                # Call the function
                result = await vertex_proxy_route(
                    endpoint="v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
                    request=mock_request,
                    fastapi_response=mock_response,
                )

                # Verify user_api_key_auth was called with the correct Bearer token
                mock_auth.assert_called_once()
                call_args = mock_auth.call_args[1]
                assert call_args["api_key"] == "Bearer test-key-123"

    def test_vertex_passthrough_handler_multimodal_embedding_response(self):
        """
        Test that vertex_passthrough_handler correctly identifies and processes multimodal embedding responses
        """
        import datetime
        from unittest.mock import Mock

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Create mock multimodal embedding response data
        multimodal_response_data = {
            "predictions": [
                {
                    "textEmbedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "imageEmbedding": [0.6, 0.7, 0.8, 0.9, 1.0],
                },
                {
                    "videoEmbeddings": [
                        {
                            "embedding": [0.11, 0.22, 0.33, 0.44, 0.55],
                            "startOffsetSec": 0,
                            "endOffsetSec": 5,
                        }
                    ]
                },
            ]
        }

        # Create mock httpx.Response
        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = multimodal_response_data
        mock_httpx_response.status_code = 200

        # Create mock logging object
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.model_call_details = {}

        # Test URL with multimodal embedding model
        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/multimodalembedding@001:predict"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch(
            "litellm.llms.vertex_ai.multimodal_embeddings.transformation.VertexAIMultimodalEmbeddingConfig"
        ) as mock_multimodal_config:
            # Mock the multimodal config instance and its methods
            mock_config_instance = Mock()
            mock_multimodal_config.return_value = mock_config_instance

            # Create a mock embedding response that would be returned by the transformation
            from litellm.types.utils import Embedding, EmbeddingResponse, Usage

            mock_embedding_response = EmbeddingResponse(
                object="list",
                data=[
                    Embedding(
                        embedding=[0.1, 0.2, 0.3, 0.4, 0.5], index=0, object="embedding"
                    ),
                    Embedding(
                        embedding=[0.6, 0.7, 0.8, 0.9, 1.0], index=1, object="embedding"
                    ),
                ],
                model="multimodalembedding@001",
                usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=0),
            )
            mock_config_instance.transform_embedding_response.return_value = (
                mock_embedding_response
            )

            # Call the handler
            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

            # Verify multimodal embedding detection and processing
            assert result is not None
            assert "result" in result
            assert "kwargs" in result

            # Verify that the multimodal config was instantiated and used
            mock_multimodal_config.assert_called_once()
            mock_config_instance.transform_embedding_response.assert_called_once()

            # Verify the response is an EmbeddingResponse
            assert isinstance(result["result"], EmbeddingResponse)
            assert result["result"].model == "multimodalembedding@001"
            assert len(result["result"].data) == 2

    def test_vertex_passthrough_handler_multimodal_detection_method(self):
        """
        Test the _is_multimodal_embedding_response detection method specifically
        """
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Test case 1: Response with textEmbedding should be detected as multimodal
        response_with_text_embedding = {
            "predictions": [{"textEmbedding": [0.1, 0.2, 0.3]}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_text_embedding
            )
            is True
        )

        # Test case 2: Response with imageEmbedding should be detected as multimodal
        response_with_image_embedding = {
            "predictions": [{"imageEmbedding": [0.4, 0.5, 0.6]}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_image_embedding
            )
            is True
        )

        # Test case 3: Response with videoEmbeddings should be detected as multimodal
        response_with_video_embeddings = {
            "predictions": [
                {
                    "videoEmbeddings": [
                        {
                            "embedding": [0.7, 0.8, 0.9],
                            "startOffsetSec": 0,
                            "endOffsetSec": 5,
                        }
                    ]
                }
            ]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_video_embeddings
            )
            is True
        )

        # Test case 4: Regular text embedding response should NOT be detected as multimodal
        regular_embedding_response = {
            "predictions": [{"embeddings": {"values": [0.1, 0.2, 0.3]}}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                regular_embedding_response
            )
            is False
        )

        # Test case 5: Non-embedding response should NOT be detected as multimodal
        non_embedding_response = {
            "candidates": [{"content": {"parts": [{"text": "Hello world"}]}}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                non_embedding_response
            )
            is False
        )

        # Test case 6: Empty response should NOT be detected as multimodal
        empty_response = {}
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                empty_response
            )
            is False
        )

    def test_vertex_passthrough_handler_predict_cost_tracking(self):
        """
        Test that vertex_passthrough_handler correctly tracks costs for /predict endpoint
        """
        import datetime
        from unittest.mock import Mock, patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Create mock embedding response data
        embedding_response_data = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "statistics": {"token_count": 10},
                    }
                }
            ]
        }

        # Create mock httpx.Response
        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = embedding_response_data
        mock_httpx_response.status_code = 200

        # Create mock logging object
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.model_call_details = {}

        # Test URL with /predict endpoint
        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch("litellm.completion_cost") as mock_completion_cost:
            # Mock the completion cost calculation
            mock_completion_cost.return_value = 0.0001

            # Call the handler
            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

            # Verify cost tracking was implemented
            assert result is not None
            assert "result" in result
            assert "kwargs" in result

            # Verify cost calculation was called
            mock_completion_cost.assert_called_once()

            # Verify cost is set in kwargs
            assert "response_cost" in result["kwargs"]
            assert result["kwargs"]["response_cost"] == 0.0001

            # Verify cost is set in logging object
            assert "response_cost" in mock_logging_obj.model_call_details
            assert mock_logging_obj.model_call_details["response_cost"] == 0.0001

            # Verify model is set in kwargs
            assert "model" in result["kwargs"]
            assert result["kwargs"]["model"] == "textembedding-gecko@001"


class TestVertexAIDiscoveryPassThroughHandler:
    """
    Test cases for Vertex AI Discovery passthrough endpoint
    """

    @pytest.mark.asyncio
    async def test_vertex_discovery_passthrough_with_credentials(self, monkeypatch):
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"v1/projects/{vertex_project}/locations/{vertex_location}/dataStores/default/servingConfigs/default:search"

        # Mock request
        mock_request = Mock()
        mock_request.state = None  # Prevent Mock from returning a truthy _cached_headers
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = "test-auth-token"

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the discovery handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                "https://discoveryengine.googleapis.com"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value="https://discoveryengine.googleapis.com"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            result = await vertex_discovery_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
            )

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args
            assert call_args[1]["endpoint"] == endpoint
            assert test_project in call_args[1]["target"]
            assert test_location in call_args[1]["target"]
            assert "Authorization" in call_args[1]["custom_headers"]
            assert (
                call_args[1]["custom_headers"]["Authorization"]
                == f"Bearer {test_token}"
            )

    @pytest.mark.asyncio
    async def test_vertex_discovery_proxy_route_api_key_auth(self):
        """
        Test that the route correctly handles API key authentication
        """
        # Mock dependencies
        mock_request = Mock()
        mock_request.headers = {"x-litellm-api-key": "test-key-123"}
        mock_request.method = "POST"
        mock_response = Mock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_auth:
            mock_auth.return_value = {"api_key": "test-key-123"}

            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_pass_through:
                mock_pass_through.return_value = AsyncMock(
                    return_value={"status": "success"}
                )

                # Call the function
                result = await vertex_discovery_proxy_route(
                    endpoint="v1/projects/test-project/locations/us-central1/dataStores/default/servingConfigs/default:search",
                    request=mock_request,
                    fastapi_response=mock_response,
                )

                # Verify user_api_key_auth was called with the correct Bearer token
                mock_auth.assert_called_once()
                call_args = mock_auth.call_args[1]
                assert call_args["api_key"] == "Bearer test-key-123"


@pytest.mark.asyncio
async def test_is_streaming_request_fn():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        is_streaming_request_fn,
    )

    mock_request = Mock()
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.form = AsyncMock(return_value={"stream": "true"})
    assert await is_streaming_request_fn(mock_request) is True


class TestBedrockLLMProxyRoute:
    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_application_inference_profile(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value="success"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
        ):

            # Test application-inference-profile endpoint
            endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd/converse"

            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = (
                mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            )

            # For application-inference-profile, model should be "arn:aws:bedrock:us-east-1:026090525607:application-inference-profile/r742sbn2zckd"
            assert (
                call_kwargs["model"]
                == "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd"
            )
            assert result == "success"

    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_regular_model(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value="success"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
        ):

            # Test regular model endpoint
            endpoint = "model/anthropic.claude-3-sonnet-20240229-v1:0/converse"

            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = (
                mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            )

            # For regular models, model should be just the model ID
            assert call_kwargs["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert result == "success"

    @pytest.mark.asyncio
    async def test_bedrock_error_handling_returns_actual_error(self):
        """
        Test that when Bedrock API returns an error, it is properly propagated to the user
        instead of being returned as a generic "Internal Server Error".
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            handle_bedrock_passthrough_router_model,
        )

        bedrock_error_message = '{"message":"ContentBlock object at messages.0.content.0 must set one of the following keys: text, image, toolUse, toolResult, document, video."}'

        # Create a mock httpx.Response for the error
        mock_error_response = Mock(spec=httpx.Response)
        mock_error_response.status_code = 400
        mock_error_response.aread = AsyncMock(
            return_value=bedrock_error_message.encode("utf-8")
        )

        # Create the HTTPStatusError
        mock_http_error = httpx.HTTPStatusError(
            message="Bad Request",
            request=Mock(spec=httpx.Request),
            response=mock_error_response,
        )

        # Create mocks for all required parameters
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/bedrock/model/test-model/converse"

        mock_request_body = {
            "messages": [{"role": "user", "content": [{"textaaa": "Hello"}]}]
        }

        mock_llm_router = Mock()

        # Mock ProxyBaseLLMRequestProcessing to raise the httpx error
        with patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_passthrough_process_llm_request",
            new_callable=AsyncMock,
            side_effect=mock_http_error,
        ):
            mock_user_api_key_dict = Mock()
            mock_user_api_key_dict.api_key = "test-key"
            mock_user_api_key_dict.allowed_model_region = None

            mock_proxy_logging_obj = Mock()
            mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

            endpoint = "model/test-model/converse"
            model = "test-model"

            with pytest.raises(HTTPException) as exc_info:
                await handle_bedrock_passthrough_router_model(
                    model=model,
                    endpoint=endpoint,
                    request=mock_request,
                    request_body=mock_request_body,
                    llm_router=mock_llm_router,
                    user_api_key_dict=mock_user_api_key_dict,
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=None,
                    select_data_generator=None,
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    version=None,
                )

            assert exc_info.value.status_code == 400
            assert (
                "ContentBlock object at messages.0.content.0 must set one of the following keys"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_bedrock_passthrough_uses_model_specific_credentials(self):
        """
        Test that Bedrock passthrough endpoints use credentials from model configuration
        instead of environment variables when a router model is used.
        
        This test verifies the fix for the bug where passthrough endpoints were using
        environment variables instead of model-specific credentials from config.yaml.
        """
        from litellm import Router
        from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            handle_bedrock_passthrough_router_model,
        )

        # Model-specific credentials (different from env vars)
        model_access_key = "MODEL_SPECIFIC_ACCESS_KEY"
        model_secret_key = "MODEL_SPECIFIC_SECRET_KEY"
        model_region = "us-west-2"
        model_session_token = "MODEL_SESSION_TOKEN"

        # Environment variables (should NOT be used)
        env_access_key = "ENV_ACCESS_KEY"
        env_secret_key = "ENV_SECRET_KEY"
        env_region = "us-east-1"

        # Set environment variables to different values
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": env_access_key,
                "AWS_SECRET_ACCESS_KEY": env_secret_key,
                "AWS_REGION_NAME": env_region,
            },
        ):
            # Test 1: Verify get_litellm_params extracts AWS credentials from kwargs
            kwargs_with_creds = {
                "aws_access_key_id": model_access_key,
                "aws_secret_access_key": model_secret_key,
                "aws_region_name": model_region,
                "aws_session_token": model_session_token,
                "model": "bedrock/test-model",
            }
            litellm_params = get_litellm_params(**kwargs_with_creds)

            # Verify credentials are extracted
            assert litellm_params.get("aws_access_key_id") == model_access_key
            assert litellm_params.get("aws_secret_access_key") == model_secret_key
            assert litellm_params.get("aws_region_name") == model_region
            assert litellm_params.get("aws_session_token") == model_session_token

            # Test 2: Verify router passes model credentials to passthrough
            router = Router(
                model_list=[
                    {
                        "model_name": "claude-opus-4-1",
                        "litellm_params": {
                            "model": "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
                            "aws_access_key_id": model_access_key,
                            "aws_secret_access_key": model_secret_key,
                            "aws_region_name": model_region,
                            "aws_session_token": model_session_token,
                            "custom_llm_provider": "bedrock",
                        },
                    }
                ]
            )

            # Verify router has model-specific credentials
            deployments = router.get_model_list(model_name="claude-opus-4-1")
            assert len(deployments) > 0
            deployment = deployments[0]
            deployment_litellm_params = deployment.get("litellm_params", {})

            # Verify model-specific credentials are in the deployment
            assert deployment_litellm_params.get("aws_access_key_id") == model_access_key
            assert deployment_litellm_params.get("aws_secret_access_key") == model_secret_key
            assert deployment_litellm_params.get("aws_region_name") == model_region
            assert deployment_litellm_params.get("aws_session_token") == model_session_token

            # Verify environment variables are NOT in the deployment
            assert deployment_litellm_params.get("aws_access_key_id") != env_access_key
            assert deployment_litellm_params.get("aws_secret_access_key") != env_secret_key
            assert deployment_litellm_params.get("aws_region_name") != env_region

            # Test 3: Verify credentials are passed through the passthrough route
            # Mock the passthrough route to capture what credentials are used
            captured_kwargs = {}

            async def mock_llm_passthrough_route(**kwargs):
                captured_kwargs.update(kwargs)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.aread = AsyncMock(
                    return_value=b'{"content": [{"text": "Hello"}]}'
                )
                return mock_response

            mock_request = MagicMock(spec=Request)
            mock_request.method = "POST"
            mock_request.headers = {"content-type": "application/json"}
            mock_request.query_params = {}
            mock_request.url = MagicMock()
            mock_request.url.path = "/bedrock/model/claude-opus-4-1/converse"

            mock_request_body = {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}]
            }

            mock_user_api_key_dict = Mock()
            mock_user_api_key_dict.api_key = "test-key"
            mock_proxy_logging_obj = Mock()
            mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

            with patch(
                "litellm.passthrough.main.llm_passthrough_route",
                new_callable=AsyncMock,
                side_effect=mock_llm_passthrough_route,
            ), patch(
                "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_passthrough_process_llm_request",
                new_callable=AsyncMock,
            ) as mock_process:
                # Setup mock response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.aread = AsyncMock(
                    return_value=b'{"content": [{"text": "Hello"}]}'
                )
                mock_process.return_value = mock_response

                # Call the handler
                await handle_bedrock_passthrough_router_model(
                    model="claude-opus-4-1",
                    endpoint="model/claude-opus-4-1/converse",
                    request=mock_request,
                    request_body=mock_request_body,
                    llm_router=router,
                    user_api_key_dict=mock_user_api_key_dict,
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=None,
                    select_data_generator=None,
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    version=None,
                )

                # Verify that the router was called (which means credentials flow through)
                # The key verification is that get_litellm_params extracts the credentials
                # and they're available in the router's deployment
                assert mock_process.called


class TestLLMPassthroughFactoryProxyRoute:
    @pytest.mark.asyncio
    async def test_llm_passthrough_factory_proxy_route_success(self):
        from litellm.types.utils import LlmProviders

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.json = AsyncMock(return_value={"stream": False})
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_model_info"
        ) as mock_get_provider, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_creds, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_provider_config = MagicMock()
            mock_provider_config.get_api_base.return_value = "https://example.com/v1"
            mock_provider_config.validate_environment.return_value = {
                "x-api-key": "dummy"
            }
            mock_get_provider.return_value = mock_provider_config
            mock_get_creds.return_value = "dummy"

            mock_endpoint_func = AsyncMock(return_value="success")
            mock_create_route.return_value = mock_endpoint_func

            result = await llm_passthrough_factory_proxy_route(
                custom_llm_provider=LlmProviders.VLLM,
                endpoint="/chat/completions",
                request=mock_request,
                fastapi_response=mock_fastapi_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            assert result == "success"
            mock_get_provider.assert_called_once_with(
                provider=litellm.LlmProviders(LlmProviders.VLLM), model=None
            )
            mock_get_creds.assert_called_once_with(
                custom_llm_provider=LlmProviders.VLLM, region_name=None
            )
            mock_create_route.assert_called_once_with(
                endpoint="/chat/completions",
                target="https://example.com/v1/chat/completions",
                custom_headers={"x-api-key": "dummy"},
                is_streaming_request=False,
            )
            mock_endpoint_func.assert_awaited_once()


class TestVLLMProxyRoute:
    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        return_value={"model": "router-model", "stream": False},
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
        return_value=True,
    )
    @patch("litellm.proxy.proxy_server.llm_router")
    async def test_vllm_proxy_route_with_router_model(
        self, mock_llm_router, mock_is_router, mock_get_body
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_llm_router.allm_passthrough_route = AsyncMock(
            return_value=httpx.Response(200, json={"response": "success"})
        )

        await vllm_proxy_route(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_is_router.assert_called_once()
        mock_llm_router.allm_passthrough_route.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        return_value={"model": "other-model"},
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
        return_value=False,
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.llm_passthrough_factory_proxy_route"
    )
    async def test_vllm_proxy_route_fallback_to_factory(
        self, mock_factory_route, mock_is_router, mock_get_body
    ):
        mock_request = MagicMock(spec=Request)
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_factory_route.return_value = "factory_success"

        result = await vllm_proxy_route(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert result == "factory_success"
        mock_factory_route.assert_awaited_once()


class TestForwardHeaders:
    """
    Test cases for _forward_headers parameter in passthrough endpoints
    """

    @pytest.mark.asyncio
    async def test_pass_through_request_with_forward_headers_true(self):
        """
        Test that when forward_headers=True, user headers from the main request
        are forwarded to the target endpoint (except content-length and host)
        """
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        # Create a mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.state = None  # Prevent MagicMock from returning a truthy _cached_headers
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"

        # User headers that should be forwarded
        user_headers = {
            "x-custom-header": "custom-value",
            "x-api-key": "user-api-key",
            "authorization": "Bearer user-token",
            "user-agent": "test-client/1.0",
            "content-type": "application/json",
            # These should NOT be forwarded
            "content-length": "123",
            "host": "original-host.com",
        }
        mock_request.headers = user_headers
        mock_request.query_params = {}

        # Mock the request body
        mock_request_body = {"test": "data"}

        mock_user_api_key_dict = MagicMock()

        # Custom headers that should be merged with user headers
        custom_headers = {
            "x-litellm-header": "litellm-value",
        }

        target_url = "https://api.example.com/v1/test"

        # Mock the httpx client and response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            # Call pass_through_request with forward_headers=True
            result = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers=custom_headers,
                user_api_key_dict=mock_user_api_key_dict,
                forward_headers=True,  # Enable header forwarding
                stream=False,
            )

            # Verify the httpx client was called
            assert mock_client.request.called

            # Get the headers that were sent to the target
            call_args = mock_client.request.call_args
            sent_headers = call_args[1]["headers"]

            # Verify user headers were forwarded (except content-length and host)
            assert sent_headers["x-custom-header"] == "custom-value"
            assert sent_headers["x-api-key"] == "user-api-key"
            assert sent_headers["authorization"] == "Bearer user-token"
            assert sent_headers["user-agent"] == "test-client/1.0"
            assert sent_headers["content-type"] == "application/json"

            # Verify custom headers were included
            assert sent_headers["x-litellm-header"] == "litellm-value"

            # Verify content-length and host were NOT forwarded
            assert "content-length" not in sent_headers
            assert "host" not in sent_headers

    @pytest.mark.asyncio
    async def test_pass_through_request_with_forward_headers_false(self):
        """
        Test that when forward_headers=False (default), user headers are NOT forwarded,
        only custom_headers are sent
        """
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        # Create a mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"
        
        # User headers that should NOT be forwarded
        user_headers = {
            "x-custom-header": "custom-value",
            "x-api-key": "user-api-key",
            "authorization": "Bearer user-token",
        }
        mock_request.headers = user_headers
        mock_request.query_params = {}

        mock_request_body = {"test": "data"}
        mock_user_api_key_dict = MagicMock()

        # Only these custom headers should be sent
        custom_headers = {
            "x-litellm-header": "litellm-value",
            "authorization": "Bearer litellm-token",
        }

        target_url = "https://api.example.com/v1/test"

        # Mock the httpx client and response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()

            # Call pass_through_request with forward_headers=False (default)
            result = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers=custom_headers,
                user_api_key_dict=mock_user_api_key_dict,
                forward_headers=False,  # Explicitly set to False
                stream=False,
            )

            # Verify the httpx client was called
            assert mock_client.request.called

            # Get the headers that were sent to the target
            call_args = mock_client.request.call_args
            sent_headers = call_args[1]["headers"]

            # Verify only custom headers were sent
            assert sent_headers["x-litellm-header"] == "litellm-value"
            assert sent_headers["authorization"] == "Bearer litellm-token"

            # Verify user headers were NOT forwarded
            assert "x-custom-header" not in sent_headers
            assert "x-api-key" not in sent_headers
            # Authorization is present but should be from custom_headers, not user headers
            assert sent_headers["authorization"] == "Bearer litellm-token"

    @pytest.mark.asyncio
    async def test_llm_passthrough_factory_with_forward_headers(self):
        """
        Test that _forward_headers works correctly in llm_passthrough_factory_proxy_route
        which is used in the code snippet provided by the user
        """
        from litellm.types.utils import LlmProviders

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/openai/chat/completions"
        
        # User headers to be forwarded
        user_headers = {
            "x-custom-tracking-id": "tracking-123",
            "x-request-id": "req-456",
            "user-agent": "my-app/2.0",
        }
        mock_request.headers = user_headers
        mock_request.json = AsyncMock(return_value={"stream": False})
        
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock the httpx response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_model_info"
        ) as mock_get_provider, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_creds, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"messages": [{"role": "user", "content": "test"}]},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
            # Setup provider config
            mock_provider_config = MagicMock()
            mock_provider_config.get_api_base.return_value = "https://api.openai.com/v1"
            mock_provider_config.validate_environment.return_value = {
                "authorization": "Bearer sk-test"
            }
            mock_get_provider.return_value = mock_provider_config
            mock_get_creds.return_value = "sk-test"

            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"messages": [{"role": "user", "content": "test"}]}
            )
            mock_logging_obj.post_call_success_hook = AsyncMock()

            # This is the key part - when create_pass_through_route is called with _forward_headers=True
            # it should forward the user headers
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route:
                mock_endpoint_func = AsyncMock(return_value="success")
                mock_create_route.return_value = mock_endpoint_func

                result = await llm_passthrough_factory_proxy_route(
                    custom_llm_provider=LlmProviders.OPENAI,
                    endpoint="/chat/completions",
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

                # Verify create_pass_through_route was called
                mock_create_route.assert_called_once()
                
                # Get the call arguments to verify _forward_headers parameter
                call_kwargs = mock_create_route.call_args[1]
                
                # Note: The current implementation doesn't explicitly pass _forward_headers
                # This test documents the current behavior. If _forward_headers should be
                # configurable in llm_passthrough_factory_proxy_route, it would need to be added


class TestOpenAIAdapterClaudeContextCompaction:
    def test_compacts_large_subagent_and_claude_md_context_blocks(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP", "900")
        user_task = (
            "Emit exactly one assistant message containing exactly three tool calls."
        )
        subagent_context = (
            "<system-reminder>\n"
            "SubagentStart hook additional context: You are 'harness-gpt55-parallel-read-tools' "
            "and you are working on the 'litellm' project.\n"
            "# TriStore Inject [start:litellm:harness-gpt55-parallel-read-tools]\n"
            + ("project context line\n" * 500)
            + "</system-reminder>\n"
        )
        claude_md_context = (
            "<system-reminder>\n"
            "As you answer the user's questions, you can use the following context:\n"
            "# claudeMd\n"
            "Contents of /home/zepfu/projects/litellm/CLAUDE.md:\n"
            + ("developer instruction line\n" * 1500)
            + "</system-reminder>\n"
        )
        request_body = {
            "model": "openai/gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": subagent_context},
                        {"type": "text", "text": claude_md_context},
                        {"type": "text", "text": user_task},
                    ],
                }
            ],
        }

        updated_body, compacted_count, markers, metadata_items = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(
                request_body
            )
        )

        assert compacted_count == 2
        assert markers == {"claude-md", "subagentstart", "tristore-inject"}
        assert len(metadata_items) == 2
        updated_parts = updated_body["messages"][0]["content"]
        assert updated_parts[2]["text"] == user_task
        assert "OpenAI adapter compacted Claude Code context block" in updated_parts[0][
            "text"
        ]
        assert "OpenAI adapter compacted Claude Code context block" in updated_parts[1][
            "text"
        ]
        assert len(updated_parts[0]["text"]) < len(subagent_context)
        assert len(updated_parts[1]["text"]) < len(claude_md_context)

        litellm_metadata = updated_body["litellm_metadata"]
        assert "openai-adapter-claude-context-compacted" in litellm_metadata["tags"]
        assert (
            "openai-adapter-claude-context:claude-md"
            in litellm_metadata["tags"]
        )
        assert (
            litellm_metadata["openai_adapter_claude_context_compacted"]
            is True
        )
        assert (
            litellm_metadata["openai_adapter_claude_context_compacted_count"]
            == 2
        )
        assert litellm_metadata["openai_adapter_claude_context_saved_chars"] > 0
        assert any(
            span["name"] == "openai_adapter.claude_context_compaction"
            for span in litellm_metadata["langfuse_spans"]
        )

    def test_preserves_task_trailing_after_context_block(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP", "700")
        user_task = "Read TODO.md and return the first heading."
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>\n"
                        "SubagentStart hook additional context: cached context\n"
                        + ("context\n" * 800)
                        + "</system-reminder>\n\n"
                        + user_task
                    ),
                }
            ]
        }

        updated_body, compacted_count, _markers, _metadata_items = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(
                request_body
            )
        )

        assert compacted_count == 1
        updated_text = updated_body["messages"][0]["content"]
        assert updated_text.endswith(user_task)
        assert "OpenAI adapter compacted Claude Code context block" in updated_text

    def test_does_not_compact_small_context_blocks(self, monkeypatch):
        monkeypatch.setenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP", "2000")
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: short\n"
                                "</system-reminder>\n"
                            ),
                        },
                        {"type": "text", "text": "Return ok."},
                    ],
                }
            ]
        }

        updated_body, compacted_count, markers, metadata_items = (
            _compact_openai_adapter_claude_context_in_anthropic_request_body(
                request_body
            )
        )

        assert updated_body == request_body
        assert compacted_count == 0
        assert markers == set()
        assert metadata_items == []


class TestGooglePersistedOutputCompaction:
    def test_compacts_large_expanded_persisted_output_for_google_adapter(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", raising=False)
        large_text = "A" * 5000
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                f"{large_text}\n"
                                "</persisted-output>\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        assert compacted_count >= 1
        assert hooks == {"subagentstart"}
        compacted_text = updated_body["messages"][0]["content"][0]["text"]
        assert "Gemini adapter compacted persisted-output" in compacted_text
        assert len(compacted_text) < len(request_body["messages"][0]["content"][0]["text"])
        assert metadata_items[0]["original_chars"] == 5000
        assert metadata_items[0]["kept_chars"] <= 2000

    def test_compacts_large_persisted_output_preview_block_for_google_adapter(self, monkeypatch):
        monkeypatch.delenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", raising=False)
        preview_text = "B" * 5000
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                "Output too large (12.2KB). Full output saved to: /tmp/tool-results/example-additionalContext.txt\n\n"
                                "Preview (first 2KB):\n"
                                f"{preview_text}\n"
                                "</persisted-output>\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        compacted_text = updated_body["messages"][0]["content"][0]["text"]
        assert compacted_count >= 1
        assert hooks == {"subagentstart"}
        assert len(compacted_text) < 400
        assert "Full output saved to: /tmp/tool-results/example-additionalContext.txt" in compacted_text
        assert any(item.get("mode") == "preview_block_cap" for item in metadata_items)

    def test_does_not_compact_small_expanded_persisted_output_for_google_adapter(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", "2000")
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SessionStart hook additional context: <persisted-output>\n"
                                "short persisted output\n"
                                "</persisted-output>\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        assert compacted_count == 0
        assert hooks == set()
        assert metadata_items == []
        assert updated_body == request_body

    def test_compacts_oversized_auxiliary_context_block_after_persisted_output_trim(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", "2000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP", "1200")
        large_text = "A" * 5000
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                f"{large_text}\n"
                                "</persisted-output>\n"
                                "Supplemental reminder lines that keep the whole block large even after persisted-output compaction.\n"
                                "Second reminder line with additional routing details.\n"
                                "Third reminder line with even more context.\n"
                                "</system-reminder>\n"
                            ),
                        }
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        compacted_text = updated_body["messages"][0]["content"][0]["text"]
        assert compacted_count >= 2
        assert hooks == {"subagentstart"}
        assert len(compacted_text) < 1400
        assert any(item.get("mode") == "auxiliary_context_block_cap" for item in metadata_items)
        assert any(item.get("mode") == "fallback_text_cap" for item in metadata_items)

    def test_compacts_split_text_parts_for_google_adapter(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", "2000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP", "1200")
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>\n"},
                        {
                            "type": "text",
                            "text": "SubagentStart hook additional context: <persisted-output>\n" + ("A" * 5000),
                        },
                        {
                            "type": "text",
                            "text": "\n</persisted-output>\nSupplemental reminder line.\n</system-reminder>\n",
                        },
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        compacted_parts = updated_body["messages"][0]["content"]
        assert len(compacted_parts) == 1
        assert compacted_count >= 1
        assert hooks == {"subagentstart"}
        assert len(compacted_parts[0]["text"]) < 1400
        assert any(item.get("mode") in {"auxiliary_context_block_cap", "fallback_text_cap"} for item in metadata_items)

    def test_preserves_user_task_after_compacting_context_block(self, monkeypatch):
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP", "2000")
        monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP", "300")
        user_task = "Use Bash to run `date -u` exactly once and reply with exactly the timestamp."
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system-reminder>\n"
                                "SubagentStart hook additional context: <persisted-output>\n"
                                + ("A" * 5000)
                                + "\n</persisted-output>\n"
                                "</system-reminder>\n\n"
                                f"{user_task}"
                            ),
                        }
                    ],
                }
            ]
        }

        updated_body, compacted_count, hooks, metadata_items = (
            _compact_google_adapter_persisted_output_in_anthropic_request_body(request_body)
        )

        compacted_text = updated_body["messages"][0]["content"][0]["text"]
        assert compacted_count >= 1
        assert hooks == {"subagentstart"}
        assert user_task in compacted_text
        assert "fallback_text_cap" not in {item.get("mode") for item in metadata_items}



class TestMilvusProxyRoute:
    """
    Test cases for Milvus passthrough endpoint
    """

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_success(self):
        """
        Test successful Milvus proxy route with valid managed vector store index
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "dall-e-6"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"
        api_base = "http://localhost:19530"

        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.url = MagicMock()
        mock_request.url.path = "/milvus/vectors/search"

        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock vector store index object
        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        # Mock vector store
        mock_vector_store = {
            "litellm_params": {
                "api_base": api_base,
                "api_key": "test-milvus-key",
            }
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name, "data": [[0.1, 0.2]]},
        ) as mock_get_body, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ) as mock_is_allowed, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ) as mock_safe_set, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
            # Setup mocks
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {
                "headers": {"Authorization": "Bearer test-token"}
            }
            mock_provider_config.get_complete_url.return_value = api_base
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )

            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            mock_endpoint_func = AsyncMock(
                return_value={"results": [{"id": 1, "distance": 0.5}]}
            )
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            result = await milvus_proxy_route(
                endpoint="vectors/search",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify calls
            mock_get_body.assert_called_once()
            mock_index_registry.is_vector_store_index.assert_called_once_with(
                vector_store_index_name=collection_name
            )
            mock_is_allowed.assert_called_once()
            mock_safe_set.assert_called_once()

            # Verify collection name was updated to the actual index
            set_body_call_args = mock_safe_set.call_args[0]
            assert set_body_call_args[1]["collectionName"] == vector_store_index

            # Verify create_pass_through_route was called with correct URL
            mock_create_route.assert_called_once()
            create_route_args = mock_create_route.call_args[1]
            assert "vectors/search" in create_route_args["target"]
            assert create_route_args["custom_headers"] == {
                "Authorization": "Bearer test-token"
            }

            # Verify endpoint function was called
            mock_endpoint_func.assert_awaited_once()
            assert result == {"results": [{"id": 1, "distance": 0.5}]}

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_missing_collection_name(self):
        """
        Test that missing collection name raises HTTPException
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"data": [[0.1, 0.2]]},  # No collectionName
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config:
            mock_get_config.return_value = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 400
            assert "Collection name is required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_provider_config(self):
        """
        Test that missing provider config raises HTTPException
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 500
            assert "Unable to find Milvus vector store config" in str(
                exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_index_registry(self):
        """
        Test that missing index registry raises HTTPException
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch.object(
            litellm, "vector_store_index_registry", None
        ):
            mock_get_config.return_value = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 500
            assert "Unable to find Milvus vector store index registry" in str(
                exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_not_managed_index(self):
        """
        Test that non-managed vector store index raises HTTPException
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "unmanaged-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry", MagicMock()
        ):
            mock_get_config.return_value = MagicMock()
            mock_index_registry.is_vector_store_index.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 400
            assert (
                f"Collection {collection_name} is not a litellm managed vector store index"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_vector_store_not_found(self):
        """
        Test that missing vector store raises Exception
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"
        vector_store_name = "missing-store"
        vector_store_index = "collection_123"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
            mock_get_config.return_value = MagicMock()
            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                None
            )

            with pytest.raises(Exception) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert f"Vector store not found for {vector_store_name}" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_api_base(self):
        """
        Test that missing api_base raises Exception
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        mock_vector_store = {"litellm_params": {}}  # No api_base

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {"headers": {}}
            mock_provider_config.get_complete_url.return_value = None
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            with pytest.raises(Exception) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert (
                f"api_base not found in vector store configuration for {vector_store_name}"
                in str(exc_info.value)
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_endpoint_without_leading_slash(self):
        """
        Test that endpoint without leading slash is handled correctly
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"
        api_base = "http://localhost:19530"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        mock_vector_store = {"litellm_params": {"api_base": api_base}}

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {"headers": {}}
            mock_provider_config.get_complete_url.return_value = api_base
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call with endpoint without leading slash
            await milvus_proxy_route(
                endpoint="vectors/search",  # No leading slash
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify that the target URL has correct path
            create_route_args = mock_create_route.call_args[1]
            assert "/vectors/search" in create_route_args["target"]


class TestOpenAIPassthroughRoute:
    """
    Test cases for OpenAI passthrough endpoint (/openai_passthrough)
    """

    @pytest.mark.asyncio
    async def test_openai_passthrough_responses_api(self):
        """
        Test that /openai_passthrough endpoint correctly handles Responses API calls
        This verifies the fix for issue #18865 where /openai/v1/responses was being
        routed to LiteLLM's native implementation instead of passthrough
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        # Mock request for Responses API
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"id": "resp_123", "status": "completed"}
            )
            mock_create_route.return_value = mock_endpoint_func

            # Call the route with /v1/responses endpoint
            result = await openai_proxy_route(
                endpoint="v1/responses",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify create_pass_through_route was called with correct target
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            
            # Should route to OpenAI's responses API
            assert call_args["target"] == "https://api.openai.com/v1/responses"
            assert call_args["endpoint"] == "v1/responses"
            
            # Verify headers contain API key
            assert "authorization" in call_args["custom_headers"]
            assert "Bearer sk-test-key" in call_args["custom_headers"]["authorization"]
            
            # Verify result
            assert result == {"id": "resp_123", "status": "completed"}

    @pytest.mark.asyncio
    async def test_openai_passthrough_chat_completions(self):
        """
        Test that /openai_passthrough works for chat completions
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"id": "chatcmpl-123", "choices": []}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="v1/chat/completions",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify routing
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.openai.com/v1/chat/completions"
            
            # Verify result
            assert result == {"id": "chatcmpl-123", "choices": []}

    @pytest.mark.asyncio
    async def test_openai_passthrough_responses_api_preserves_client_auth(self):
        """
        Test that /openai_passthrough preserves inbound client auth for Responses API
        calls instead of requiring a server-side OPENAI_API_KEY.
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer codex-client-token",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_credentials, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"id": "resp_123", "status": "completed"}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="responses",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_get_credentials.assert_not_called()
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.openai.com/v1/responses"
            assert call_args["endpoint"] == "responses"
            assert call_args["custom_headers"] == {}
            assert call_args["_forward_headers"] is True
            assert result == {"id": "resp_123", "status": "completed"}

    @pytest.mark.asyncio
    async def test_openai_passthrough_codex_native_auth_targets_chatgpt_backend(self):
        """
        Codex-native OAuth traffic should preserve client auth and target the ChatGPT
        Codex backend instead of the public OpenAI Responses API.
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer codex-client-token",
            "chatgpt-account-id": "acct_123",
            "originator": "codex_exec",
            "session_id": "sess_123",
            "user-agent": "codex_exec/0.118.0",
        }
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_credentials, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"id": "resp_123", "status": "completed"}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="responses",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_get_credentials.assert_not_called()
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert (
                call_args["target"]
                == "https://chatgpt.com/backend-api/codex/responses"
            )
            assert call_args["endpoint"] == "responses"
            assert call_args["custom_headers"] == {}
            assert call_args["_forward_headers"] is True
            assert result == {"id": "resp_123", "status": "completed"}

    @pytest.mark.asyncio
    async def test_openai_passthrough_missing_api_key(self):
        """
        Test that missing OPENAI_API_KEY raises an exception
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ):
            with pytest.raises(Exception) as exc_info:
                await openai_proxy_route(
                    endpoint="v1/chat/completions",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert "Required 'OPENAI_API_KEY'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_passthrough_assistants_api(self):
        """
        Test that /openai_passthrough works for Assistants API endpoints
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/assistants"
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"id": "asst_123", "object": "assistant"}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="v1/assistants",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify routing
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.openai.com/v1/assistants"
            
            # Verify headers contain API key and OpenAI-Beta header
            assert "authorization" in call_args["custom_headers"]
            
            # Verify result
            assert result == {"id": "asst_123", "object": "assistant"}


class TestCursorProxyRoute:
    """Tests for the Cursor Cloud Agents pass-through route."""

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_creates_pass_through_with_basic_auth(self):
        """should create a pass-through route with Basic Auth header for Cursor API"""
        import base64

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        test_api_key = "test-cursor-api-key-123"

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=test_api_key,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={"agents": [], "nextCursor": None}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/agents",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/agents"

            expected_auth = base64.b64encode(
                f"{test_api_key}:".encode("utf-8")
            ).decode("ascii")
            assert call_args["custom_headers"]["Authorization"] == f"Basic {expected_auth}"

            assert result == {"agents": [], "nextCursor": None}

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_raises_on_missing_api_key(self):
        """should raise 401 when no Cursor API key is available"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.credential_list",
            [],
        ):
            with pytest.raises(Exception) as exc_info:
                await cursor_proxy_route(
                    endpoint="v0/agents",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_uses_ui_credential(self):
        """should use credentials added via UI (litellm.credential_list) when env var is not set"""
        from litellm.types.utils import CredentialItem

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        ui_credential = CredentialItem(
            credential_name="my-cursor-key",
            credential_values={"api_key": "crsr_ui_test_key", "api_base": "https://api.cursor.com"},
            credential_info={"custom_llm_provider": "cursor"},
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.credential_list",
            [ui_credential],
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(return_value={"models": []})
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/models",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/models"

            import base64
            expected_auth = base64.b64encode(b"crsr_ui_test_key:").decode("ascii")
            assert call_args["custom_headers"]["Authorization"] == f"Basic {expected_auth}"

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_custom_api_base(self):
        """should use CURSOR_API_BASE env var when set"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch.dict(
            os.environ, {"CURSOR_API_BASE": "https://custom-cursor.example.com"}
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(return_value={})
            mock_create_route.return_value = mock_endpoint_func

            await cursor_proxy_route(
                endpoint="v0/me",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://custom-cursor.example.com/v0/me"

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_launch_agent(self):
        """should handle POST to launch an agent through the pass-through"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_endpoint_func = AsyncMock(
                return_value={
                    "id": "bc_abc123",
                    "name": "Test Agent",
                    "status": "CREATING",
                }
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/agents",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/agents"
            assert result["id"] == "bc_abc123"
            assert result["status"] == "CREATING"
