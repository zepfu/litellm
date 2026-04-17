from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.aawm_agent_identity import (
    AawmAgentIdentity,
    _build_session_history_record_from_langfuse_trace_observation,
    _build_session_history_record_from_spend_log_row,
    _build_session_history_record,
    _derive_langfuse_trace_tags_from_langfuse_trace,
    _derive_langfuse_trace_tags_from_spend_log_row,
    _persist_session_history_records,
    _persist_session_history_record,
)


def _base_kwargs(trace_name: str = "claude-code") -> dict:
    return {
        "litellm_params": {"metadata": {"trace_name": trace_name}},
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {
            "request_body": {
                "messages": [
                    {
                        "role": "user",
                        "content": "You are 'engineer' and you are working on the 'aegis' project.",
                    }
                ]
            }
        },
    }


def test_aawm_agent_identity_enriches_trace_name() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs()

    updated_kwargs, result = logger.logging_hook(
        kwargs=kwargs,
        result={"choices": []},
        call_type="pass_through_endpoint",
    )

    assert result == {"choices": []}
    assert (
        updated_kwargs["litellm_params"]["metadata"]["trace_name"]
        == "claude-code.engineer"
    )
    assert updated_kwargs["standard_logging_object"]["metadata"]["trace_name"] == (
        "claude-code.engineer"
    )


def test_aawm_agent_identity_adds_claude_thinking_tags() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs()
    result = {
        "choices": [
            {
                "message": {
                    "content": "Ready.",
                    "role": "assistant",
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "EvMCCmwIDBgCKkAuqMZK8CbuNuz6LdJex7qr4ZB9T9OXQ6zOKvzTxK6SCYZUP3ageKCC1lr28wDIfpWyVJVPVGcFP+a5ScIJ6CsQMiJudW1iYXQtdjYtZWZmb3J0cy0yMC00MC04MC1hYi1wcm9kOAASDOBIjRsAoyR7Oe6UdhoMtmeDeP+RjpVIJjlYIjCq8O2tRhEP4C9HCx8SrqqE0v1cKQ6aiJBHzBOOdZZg92sNK/B/sL4ihlm+ptMA9NYqtAHTchkk3dthQhVBBayWWoOjP/IEZEhlzYHTdoTOzKkLUQNEeCmJQQst7E+ugv9gn+luB/SalmqboTM0FqmLYX8nWG5gMb8LI8ipTZwgLyYLoyvcg5NwaoWPqup1Wo4v85lJeoFam70xAyK7v2b1cDgNoYT+jVGRE4gUZy6W+ZOK7wxLdIkeObuEiAKjwKE6o8G6hfIB+AsW4mAOPymAOS8fm4JnYcz61kXO1MjvhtAqkjMNCPsYAQ==",
                            }
                        ]
                    },
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": " I'm ready and waiting for the user's question.",
                            "signature": "EvMCCmwIDBgCKkAuqMZK8CbuNuz6LdJex7qr4ZB9T9OXQ6zOKvzTxK6SCYZUP3ageKCC1lr28wDIfpWyVJVPVGcFP+a5ScIJ6CsQMiJudW1iYXQtdjYtZWZmb3J0cy0yMC00MC04MC1hYi1wcm9kOAASDOBIjRsAoyR7Oe6UdhoMtmeDeP+RjpVIJjlYIjCq8O2tRhEP4C9HCx8SrqqE0v1cKQ6aiJBHzBOOdZZg92sNK/B/sL4ihlm+ptMA9NYqtAHTchkk3dthQhVBBayWWoOjP/IEZEhlzYHTdoTOzKkLUQNEeCmJQQst7E+ugv9gn+luB/SalmqboTM0FqmLYX8nWG5gMb8LI8ipTZwgLyYLoyvcg5NwaoWPqup1Wo4v85lJeoFam70xAyK7v2b1cDgNoYT+jVGRE4gUZy6W+ZOK7wxLdIkeObuEiAKjwKE6o8G6hfIB+AsW4mAOPymAOS8fm4JnYcz61kXO1MjvhtAqkjMNCPsYAQ==",
                        }
                    ],
                    "reasoning_content": " I'm ready and waiting for the user's question.",
                }
            }
        ]
    }

    updated_kwargs, _ = logger.logging_hook(
        kwargs=kwargs,
        result=result,
        call_type="pass_through_endpoint",
    )

    metadata = updated_kwargs["litellm_params"]["metadata"]
    tags = metadata["tags"]
    assert metadata["trace_name"] == "claude-code.engineer"
    assert metadata["claude_thinking_signature_present"] is True
    assert metadata["claude_thinking_signature_count"] == 1
    assert metadata["claude_thinking_signature_decoded"] is True
    assert (
        metadata["claude_thinking_experiment_id"]
        == "numbat-v6-efforts-20-40-80-ab-prod"
    )
    assert metadata["claude_reasoning_content_present"] is True
    assert metadata["thinking_signature_present"] is True
    assert metadata["thinking_signature_decoded"] is True
    assert metadata["reasoning_content_present"] is True
    assert metadata["thinking_blocks_present"] is True
    assert "claude-thinking-signature" in tags
    assert "thinking-signature-present" in tags
    assert "thinking-signature-decoded" in tags
    assert "claude-thinking-decoded" in tags
    assert "claude-exp:numbat-v6-efforts-20-40-80-ab-prod" in tags
    assert "reasoning-present" in tags
    assert "thinking-blocks-present" in tags
    assert "claude-reasoning-present" in tags
    assert "claude-thinking-signature" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert "thinking-signature-present" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert (
        updated_kwargs["standard_logging_object"]["metadata"]["claude_thinking_signature_present"]
        is True
    )
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "claude.thinking_signature_decode" in span_names
    claude_span = next(
        span
        for span in metadata["langfuse_spans"]
        if isinstance(span, dict)
        and span.get("name") == "claude.thinking_signature_decode"
    )
    assert claude_span["metadata"]["signature_count"] == 1
    assert claude_span["metadata"]["decoded_signature_count"] == 1
    assert claude_span["metadata"]["reasoning_content_present"] is True
    assert "start_time" in claude_span
    assert "end_time" in claude_span
    assert "gemini_thought_signature_present" not in metadata


def test_aawm_agent_identity_adds_gemini_thought_signature_tags() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs(trace_name="gemini")
    result = {
        "choices": [
            {
                "message": {
                    "content": "gemini routed",
                    "role": "assistant",
                    "thinking_blocks": [],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            """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"""
                        ]
                    },
                }
            }
        ]
    }

    updated_kwargs, _ = logger.logging_hook(
        kwargs=kwargs,
        result=result,
        call_type="pass_through_endpoint",
    )

    metadata = updated_kwargs["litellm_params"]["metadata"]
    tags = metadata["tags"]
    assert metadata["trace_name"] == "gemini"
    assert metadata["gemini_thought_signature_present"] is True
    assert metadata["gemini_thought_signature_count"] == 1
    assert metadata["gemini_tsig_record_count"] == 9
    assert metadata["gemini_tsig_record_sizes"] == [
        36,
        104,
        124,
        194,
        156,
        280,
        276,
        328,
        112,
    ]
    assert metadata["gemini_tsig_0_record_0_size"] == 36
    assert metadata["gemini_tsig_marker_hex"] == "8f3d6b5f"
    assert len(metadata["gemini_tsig_marker_offsets"]) == 9
    assert metadata["gemini_reasoning_content_present"] is False
    assert metadata["thinking_signature_present"] is True
    assert metadata["thinking_signature_decoded"] is True
    assert metadata["reasoning_content_present"] is False
    assert metadata["thinking_blocks_present"] is False
    assert "gemini-thought-signature" in tags
    assert "thinking-signature-present" in tags
    assert "thinking-signature-decoded" in tags
    assert "gemini-thought-signature-decoded" in tags
    assert "gemini-tsig-records:9" in tags
    assert "reasoning-empty" in tags
    assert "thinking-blocks-empty" in tags
    assert "gemini-reasoning-empty" in tags
    assert any(tag.startswith("gemini-tsig-shape:") for tag in tags)
    assert "gemini-thought-signature" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert "thinking-signature-present" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert (
        updated_kwargs["standard_logging_object"]["metadata"]["gemini_thought_signature_present"]
        is True
    )
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "gemini.thought_signature_decode" in span_names
    gemini_span = next(
        span
        for span in metadata["langfuse_spans"]
        if isinstance(span, dict)
        and span.get("name") == "gemini.thought_signature_decode"
    )
    assert gemini_span["metadata"]["signature_count"] == 1
    assert gemini_span["metadata"]["decoded_signature_count"] == 1
    assert gemini_span["metadata"]["record_counts"] == [9]
    assert "start_time" in gemini_span
    assert "end_time" in gemini_span
    assert "claude_thinking_signature_present" not in metadata


def test_build_session_history_record_tracks_usage_reasoning_and_tools() -> None:
    kwargs = _base_kwargs()
    kwargs["model"] = "anthropic/claude-sonnet-4-6"
    kwargs["custom_llm_provider"] = "anthropic"
    kwargs["call_type"] = "pass_through_endpoint"
    kwargs["litellm_call_id"] = "call-123"
    kwargs["litellm_params"]["metadata"]["session_id"] = "session-123"
    kwargs["litellm_params"]["metadata"]["cc_version"] = "2.1.112"
    kwargs["standard_logging_object"]["request_tags"] = ["reasoning-present"]

    result = {
        "id": "provider-response-1",
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "prompt_tokens_details": {"cached_tokens": 11},
            "completion_tokens_details": {"reasoning_tokens": 9},
            "cache_creation_input_tokens": 7,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Working.",
                    "reasoning_content": "Need to inspect the current state before acting.",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
    }

    record = _build_session_history_record(
        kwargs=kwargs,
        result=result,
        start_time=None,
        end_time=None,
    )

    assert record is not None
    assert record["session_id"] == "session-123"
    assert record["model"] == "anthropic/claude-sonnet-4-6"
    assert record["provider"] == "anthropic"
    assert record["provider_response_id"] == "provider-response-1"
    assert record["agent_name"] == "engineer"
    assert record["tenant_id"] == "aegis"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 45
    assert record["total_tokens"] == 165
    assert record["cache_read_input_tokens"] == 11
    assert record["cache_creation_input_tokens"] == 7
    assert record["reasoning_tokens_reported"] == 9
    assert record["reasoning_tokens_estimated"] is None
    assert record["reasoning_tokens_source"] == "provider_reported"
    assert record["reasoning_present"] is True
    assert record["tool_call_count"] == 1
    assert record["tool_names"] == ["search"]
    assert record["metadata"]["request_tags"] == ["reasoning-present"]
    assert record["metadata"]["tenant_id"] == "aegis"
    assert record["metadata"]["cc_version"] == "2.1.112"


def test_aawm_agent_identity_adds_codex_usage_breakout_tags() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs(trace_name="codex")
    kwargs["model"] = "gpt-5.2-codex"
    kwargs["custom_llm_provider"] = "openai"
    kwargs["litellm_params"]["metadata"]["passthrough_route_family"] = (
        "codex_responses"
    )

    result = {
        "id": "resp-codex-1",
        "usage": {
            "input_tokens": 80,
            "output_tokens": 24,
            "total_tokens": 104,
            "input_tokens_details": {"cached_tokens": 31},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
        "output": [
            {
                "type": "function_call",
                "name": "apply_patch",
                "arguments": "{}",
                "call_id": "call_123",
            }
        ],
        "choices": [],
    }

    updated_kwargs, _ = logger.logging_hook(
        kwargs=kwargs,
        result=result,
        call_type="pass_through_endpoint",
    )

    metadata = updated_kwargs["litellm_params"]["metadata"]
    request_tags = updated_kwargs["standard_logging_object"]["request_tags"]

    assert metadata["usage_reasoning_tokens_reported"] == 12
    assert metadata["usage_reasoning_tokens_source"] == "provider_reported"
    assert metadata["usage_cache_read_input_tokens"] == 31
    assert metadata["usage_cache_creation_input_tokens"] == 0
    assert metadata["usage_tool_call_count"] == 1
    assert metadata["usage_tool_names"] == ["apply_patch"]
    assert metadata["codex_reasoning_tokens_reported"] == 12
    assert metadata["codex_cache_read_input_tokens"] == 31
    assert "codex-usage-breakout" in metadata["tags"]
    assert "codex-reasoning-tokens-reported" in metadata["tags"]
    assert "codex-cache-read-input-tokens" in metadata["tags"]
    assert "codex-tool-calls-present" in metadata["tags"]
    assert "reasoning-tokens-reported" in request_tags
    assert "cache-read-input-tokens" in request_tags
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "codex.usage_breakout" in span_names


def test_build_session_history_record_skips_without_session_id() -> None:
    kwargs = _base_kwargs()
    kwargs["model"] = "gemini/gemini-2.5-pro"
    result = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    assert (
        _build_session_history_record(
            kwargs=kwargs,
            result=result,
            start_time=None,
            end_time=None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_persist_session_history_record_executes_insert(monkeypatch) -> None:
    record = {
        "litellm_call_id": "call-123",
        "session_id": "session-123",
        "trace_id": "trace-123",
        "provider_response_id": "resp-123",
        "provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-6",
        "model_group": "claude-sonnet-4-6",
        "agent_name": "engineer",
        "tenant_id": "aegis",
        "call_type": "pass_through_endpoint",
        "start_time": None,
        "end_time": None,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens_reported": 3,
        "reasoning_tokens_estimated": None,
        "reasoning_tokens_source": "provider_reported",
        "reasoning_present": True,
        "thinking_signature_present": True,
        "tool_call_count": 1,
        "tool_names": ["search"],
        "response_cost_usd": 0.12,
        "metadata": {"request_tags": ["reasoning-present"]},
    }

    mock_pool = AsyncMock()
    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._get_aawm_session_history_pool",
        AsyncMock(return_value=mock_pool),
    )
    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._ensure_session_history_schema",
        AsyncMock(),
    )

    await _persist_session_history_record(record)

    mock_pool.execute.assert_awaited_once()
    executed_args = mock_pool.execute.await_args.args
    assert "INSERT INTO public.session_history" in executed_args[0]
    assert executed_args[1] == "call-123"
    assert executed_args[2] == "session-123"
    assert executed_args[6] == "anthropic/claude-sonnet-4-6"


def test_build_session_history_record_from_spend_log_row_recovers_real_session_id() -> None:
    spend_log_row = {
        "request_id": "req-123",
        "call_type": "pass_through_endpoint",
        "custom_llm_provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-6",
        "model_group": "claude-sonnet-4-6",
        "spend": 0.12,
        "prompt_tokens": 12,
        "completion_tokens": 9,
        "total_tokens": 21,
        "startTime": "2026-04-16T12:00:00+00:00",
        "endTime": "2026-04-16T12:00:01+00:00",
        "status": "success",
        "session_id": "trace-legacy-123",
        "metadata": {
            "cc_version": "2.1.112",
            "usage_object": {
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        },
        "request_tags": ["claude-thinking-signature"],
        "proxy_server_request": {
            "metadata": {
                "user_id": {
                    "session_id": "claude-session-456",
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": "You are 'eyes' and you are working on the 'litellm' project.",
                }
            ],
        },
        "messages": [
            {
                "role": "user",
                "content": "You are 'eyes' and you are working on the 'litellm' project.",
            }
        ],
        "response": {
            "id": "provider-response-123",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Inspect existing traces first.",
                        "tool_calls": [
                            {"function": {"name": "search"}},
                        ],
                    }
                }
            ],
        },
    }

    record = _build_session_history_record_from_spend_log_row(
        spend_log_row, backfill_run_id="run-1"
    )

    assert record is not None
    assert record["litellm_call_id"] == "req-123"
    assert record["session_id"] == "claude-session-456"
    assert record["trace_id"] == "trace-legacy-123"
    assert record["agent_name"] == "eyes"
    assert record["tenant_id"] == "litellm"
    assert record["reasoning_tokens_reported"] == 4
    assert record["tool_call_count"] == 1
    assert record["metadata"]["backfilled"] is True
    assert record["metadata"]["backfill_source"] == "LiteLLM_SpendLogs"
    assert record["metadata"]["backfill_run_id"] == "run-1"
    assert (
        record["metadata"]["session_id_source"]
        == "request_body.metadata.user_id.session_id"
    )
    assert record["metadata"]["trace_id_source"] == "legacy_spend_log_session_field"
    assert "claude-thinking-signature" in record["metadata"]["request_tags"]


def test_build_session_history_record_from_spend_log_row_falls_back_to_legacy_trace_id() -> None:
    spend_log_row = {
        "request_id": "req-456",
        "call_type": "pass_through_endpoint",
        "custom_llm_provider": "openai",
        "model": "openai/responses/gpt-5.4",
        "model_group": "gpt-5.4",
        "spend": 0.04,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "session_id": "trace-or-legacy-session-789",
        "status": "success",
        "request_tags": '["route:codex_responses"]',
        "metadata": '{"usage_object":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}',
        "proxy_server_request": "{}",
        "messages": "[]",
        "response": '{"id":"resp-456","usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30},"choices":[]}',
    }

    record = _build_session_history_record_from_spend_log_row(spend_log_row)

    assert record is not None
    assert record["session_id"] == "trace-or-legacy-session-789"
    assert record["trace_id"] == "trace-or-legacy-session-789"
    assert record["metadata"]["session_id_source"] == "legacy_spend_log_session_field"
    assert record["metadata"]["trace_id_source"] == "legacy_spend_log_session_field"
    assert record["metadata"]["request_tags"] == ["route:codex_responses"]


def test_build_session_history_record_from_spend_log_row_uses_responses_usage_details() -> None:
    spend_log_row = {
        "request_id": "req-codex-usage-1",
        "call_type": "responses",
        "custom_llm_provider": "openai",
        "model": "gpt-5.2-codex",
        "model_group": "gpt-5.2-codex",
        "spend": 0.09,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
        "session_id": "codex-session-1",
        "status": "success",
        "request_tags": ["route:codex_responses"],
        "metadata": {
            "usage_object": {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "input_tokens_details": {"cached_tokens": 19},
                "output_tokens_details": {"reasoning_tokens": 7},
            }
        },
        "proxy_server_request": {},
        "messages": [],
        "response": {
            "id": "resp-codex-usage-1",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "input_tokens_details": {"cached_tokens": 19},
                "output_tokens_details": {"reasoning_tokens": 7},
            },
            "output": [
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": "{}",
                    "call_id": "call_search",
                }
            ],
        },
    }

    record = _build_session_history_record_from_spend_log_row(spend_log_row)

    assert record is not None
    assert record["input_tokens"] == 50
    assert record["output_tokens"] == 20
    assert record["cache_read_input_tokens"] == 19
    assert record["reasoning_tokens_reported"] == 7
    assert record["reasoning_tokens_source"] == "provider_reported"
    assert record["tool_call_count"] == 1
    assert record["tool_names"] == ["search"]


def test_derive_langfuse_trace_tags_from_spend_log_row_merges_derived_tags() -> None:
    spend_log_row = {
        "request_id": "req-tags-1",
        "call_type": "pass_through_endpoint",
        "custom_llm_provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-6",
        "session_id": "trace-tags-123",
        "request_tags": ["claude-thinking-signature"],
        "metadata": {},
        "proxy_server_request": {
            "messages": [
                {
                    "role": "user",
                    "content": "You are 'grunt' and you are working on the 'litellm' project.",
                }
            ]
        },
        "messages": [
            {
                "role": "user",
                "content": "You are 'grunt' and you are working on the 'litellm' project.",
            }
        ],
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Need to inspect current usage first.",
                    }
                }
            ]
        },
    }

    trace_id, tags = _derive_langfuse_trace_tags_from_spend_log_row(spend_log_row)

    assert trace_id == "trace-tags-123"
    assert "claude-thinking-signature" in tags
    assert "claude-thinking-signature" in tags


def test_build_session_history_record_from_langfuse_trace_observation() -> None:
    trace = {
        "id": "trace-123",
        "name": "claude-code.orchestrator",
        "sessionId": "session-123",
        "environment": "dev",
        "totalCost": 0.55,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "You are 'orchestrator' and you are working on the 'litellm' project.",
                }
            ]
        },
    }
    observation = {
        "id": "call-123",
        "type": "GENERATION",
        "name": "litellm-pass_through_endpoint",
        "model": "claude-opus-4-7",
        "startTime": "2026-04-16T12:00:00Z",
        "endTime": "2026-04-16T12:00:03Z",
        "promptTokens": 120,
        "completionTokens": 45,
        "totalTokens": 165,
        "usageDetails": {
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 7,
        },
        "costDetails": {"total": 0.12},
        "output": {
            "id": "provider-response-1",
            "content": "Ready.",
            "reasoning_content": "Need to inspect the existing traces first.",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        "metadata": {
            "tags": ["route:anthropic_messages", "thinking-signature-present"],
            "cc_version": "2.1.112",
            "trace_name": "claude-code.orchestrator",
            "passthrough_route_family": "anthropic_messages",
            "anthropic_billing_header_fields": {"cc_version": "2.1.112"},
            "thinking_signature_present": True,
            "claude_effort": "max",
        },
    }

    record = _build_session_history_record_from_langfuse_trace_observation(
        trace,
        observation,
        backfill_run_id="run-1",
    )

    assert record is not None
    assert record["litellm_call_id"] == "call-123"
    assert record["session_id"] == "session-123"
    assert record["trace_id"] == "trace-123"
    assert record["provider_response_id"] == "provider-response-1"
    assert record["provider"] == "anthropic"
    assert record["agent_name"] == "orchestrator"
    assert record["tenant_id"] == "litellm"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 45
    assert record["total_tokens"] == 165
    assert record["cache_read_input_tokens"] == 11
    assert record["cache_creation_input_tokens"] == 7
    assert record["reasoning_present"] is True
    assert record["reasoning_tokens_source"] == "estimated_from_reasoning_text"
    assert record["tool_call_count"] == 1
    assert record["tool_names"] == ["search"]
    assert record["response_cost_usd"] == pytest.approx(0.12)
    assert record["metadata"]["backfilled"] is True
    assert record["metadata"]["backfill_source"] == "LangfuseTraces"
    assert record["metadata"]["backfill_run_id"] == "run-1"
    assert record["metadata"]["session_id_source"] == "trace.sessionId"
    assert record["metadata"]["trace_id_source"] == "trace.id"
    assert "route:anthropic_messages" in record["metadata"]["request_tags"]


def test_build_session_history_record_from_langfuse_trace_observation_uses_tool_name_fallback() -> None:
    trace = {
        "id": "trace-456",
        "name": "codex",
        "sessionId": "session-456",
        "input": {},
    }
    observation = {
        "id": "call-456",
        "type": "GENERATION",
        "name": "litellm-responses",
        "model": "openai/responses/gpt-5.4",
        "startTime": "2026-04-16T12:00:00Z",
        "endTime": "2026-04-16T12:00:02Z",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoningTokens": 2,
        },
        "output": {},
        "toolCallNames": ["search", "memory_save"],
        "metadata": {
            "passthrough_route_family": "codex_responses",
        },
    }

    record = _build_session_history_record_from_langfuse_trace_observation(
        trace,
        observation,
        backfill_run_id="run-2",
    )

    assert record is not None
    assert record["provider"] == "openai"
    assert record["tool_call_count"] == 2
    assert record["tool_names"] == ["search", "memory_save"]
    assert record["reasoning_tokens_reported"] == 2
    assert record["reasoning_tokens_source"] == "provider_reported"


def test_build_session_history_record_from_langfuse_trace_observation_uses_metadata_usage_object_for_gemini() -> None:
    trace = {
        "id": "trace-gemini-1",
        "name": "gemini",
        "sessionId": "session-gemini-1",
        "environment": "dev",
    }
    observation = {
        "id": "obs-gemini-1",
        "type": "GENERATION",
        "name": "litellm-pass_through_endpoint",
        "model": "gemini-2.5-pro",
        "startTime": "2026-04-17T14:00:00Z",
        "endTime": "2026-04-17T14:00:02Z",
        "usage": {
            "input": 120,
            "output": 40,
            "total": 160,
        },
        "costDetails": {"total": 0.03},
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "gemini result",
                        "provider_specific_fields": {
                            "thought_signatures": ["CiQBjz1rXzg04kJ2A8JC+Q=="]
                        },
                    }
                }
            ]
        },
        "metadata": {
            "passthrough_route_family": "gemini_generate_content",
            "usage_object": {
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
                "cachedContentTokenCount": 55,
                "thoughtsTokenCount": 18,
                "prompt_tokens_details": {"cached_tokens": 55},
            },
            "usage_tool_names": ["google_search"],
        },
    }

    record = _build_session_history_record_from_langfuse_trace_observation(
        trace,
        observation,
        backfill_run_id="run-gemini-1",
    )

    assert record is not None
    assert record["provider"] == "gemini"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 40
    assert record["cache_read_input_tokens"] == 55
    assert record["reasoning_tokens_reported"] == 18
    assert record["reasoning_tokens_source"] == "provider_reported"
    assert record["tool_call_count"] == 1
    assert record["tool_names"] == ["google_search"]


def test_derive_langfuse_trace_tags_from_langfuse_trace_merges_observation_metadata() -> None:
    trace = {
        "id": "trace-tags-123",
        "tags": ["existing-tag"],
        "observations": [
            {
                "type": "GENERATION",
                "metadata": {
                    "tags": ["route:anthropic_messages"],
                    "passthrough_route_family": "anthropic_messages",
                    "anthropic_billing_header_fields": {
                        "cc_version": "2.1.112",
                        "cc_entrypoint": "cli",
                    },
                    "thinking_signature_present": True,
                    "reasoning_content_present": False,
                    "thinking_blocks_present": True,
                    "claude_effort": "high",
                },
            },
            {"type": "SPAN", "metadata": {"tags": ["ignore-me"]}},
        ],
    }

    trace_id, tags = _derive_langfuse_trace_tags_from_langfuse_trace(trace)

    assert trace_id == "trace-tags-123"
    assert "existing-tag" in tags
    assert "route:anthropic_messages" in tags
    assert "anthropic-billing-header" in tags
    assert "anthropic-billing-header-key:cc_version" in tags
    assert "anthropic-billing-header:cc_version=2.1.112" in tags
    assert "thinking-signature-present" in tags
    assert "reasoning-empty" in tags
    assert "thinking-blocks-present" in tags
    assert "claude-effort:high" in tags
    assert "effort:high" in tags
    assert "ignore-me" not in tags


@pytest.mark.asyncio
async def test_persist_session_history_records_executes_batch_insert(monkeypatch) -> None:
    records = [
        {
            "litellm_call_id": "call-1",
            "session_id": "session-1",
            "trace_id": "trace-1",
            "provider_response_id": "resp-1",
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "model_group": "claude-sonnet-4-6",
            "agent_name": "eyes",
            "tenant_id": "litellm",
            "call_type": "pass_through_endpoint",
            "start_time": None,
            "end_time": None,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens_reported": 2,
            "reasoning_tokens_estimated": None,
            "reasoning_tokens_source": "provider_reported",
            "reasoning_present": True,
            "thinking_signature_present": True,
            "tool_call_count": 1,
            "tool_names": ["search"],
            "response_cost_usd": 0.01,
            "metadata": {"request_tags": ["reasoning-present"]},
        }
    ]

    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    acquire_context = AsyncMock()
    acquire_context.__aenter__.return_value = mock_conn
    mock_pool.acquire = MagicMock(return_value=acquire_context)

    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._get_aawm_session_history_pool",
        AsyncMock(return_value=mock_pool),
    )
    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._ensure_session_history_schema",
        AsyncMock(),
    )

    await _persist_session_history_records(records)

    mock_conn.executemany.assert_awaited_once()
    executed_args = mock_conn.executemany.await_args.args
    assert "INSERT INTO public.session_history" in executed_args[0]
    assert executed_args[1][0][0] == "call-1"
