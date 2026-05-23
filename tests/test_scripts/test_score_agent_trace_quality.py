from __future__ import annotations

import json

import scripts.score_agent_trace_quality as scorer


def _candidate(**overrides: object) -> scorer.SessionCandidate:
    values = {
        "row_id": 1,
        "created_at": "2026-05-17T19:50:00Z",
        "trace_id": "trace-1",
        "session_id": "session-1",
        "litellm_call_id": "obs-1",
        "source_observation_id": "obs-1",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "agent_name": "principal",
        "repository": "dashboard-shell",
        "tenant_id": "dashboard-shell",
        "input_tokens": 53874,
        "output_tokens": 3,
        "tool_call_count": 0,
        "invalid_tool_call_count": 0,
        "metadata": {},
    }
    values.update(overrides)
    return scorer.SessionCandidate(**values)  # type: ignore[arg-type]


def _payload(messages: list[dict[str, object]], output: object) -> scorer.ObservationPayload:
    anthropic_request = {"model": "claude-sonnet-4-6", "messages": messages}
    langfuse_input = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(anthropic_request),
            }
        ]
    }
    return scorer.ObservationPayload(
        observation_id="obs-1",
        trace_id="trace-1",
        body={
            "id": "obs-1",
            "traceId": "trace-1",
            "input": json.dumps(langfuse_input),
            "output": json.dumps(output),
        },
        source="minio",
        source_locator="observation/obs-1/blob.json",
    )


def _openai_payload(messages: list[dict[str, object]], output: object) -> scorer.ObservationPayload:
    return scorer.ObservationPayload(
        observation_id="obs-1",
        trace_id="trace-1",
        body={
            "id": "obs-1",
            "traceId": "trace-1",
            "input": json.dumps({"messages": messages}),
            "output": json.dumps(output),
        },
        source="minio",
        source_locator="observation/obs-1/blob.json",
    )


def test_should_extract_nested_anthropic_request_from_langfuse_input() -> None:
    request = {"messages": [{"role": "user", "content": "hello"}]}
    langfuse_input = {"messages": [{"role": "user", "content": json.dumps(request)}]}

    assert scorer._extract_anthropic_request_from_langfuse_input(
        json.dumps(langfuse_input)
    ) == request


def test_should_score_empty_completion_after_large_final_tool_result_as_failure() -> None:
    payload = _payload(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "name": "Read"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "a" * 195_820,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
        {"role": "assistant", "content": "", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 0.0
    assert evidence.empty_completion_failure is True
    assert evidence.large_tool_result_payload_risk is True
    assert evidence.final_tool_result_image_base64_max_bytes == 195_820
    assert evidence.reasons == ["empty_completion_after_large_final_tool_result"]


def test_should_not_fail_empty_completion_when_provider_error_exists() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "data": "a" * 150_000},
                            }
                        ],
                    }
                ],
            }
        ],
        {"role": "assistant", "content": "", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(),
        payload,
        provider_error_present=True,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 1.0
    assert evidence.empty_completion_failure is False
    assert evidence.reasons == [
        "empty_output_has_provider_error",
        "large_tool_result_payload_seen",
    ]


def test_should_not_fail_normal_nonempty_completion() -> None:
    payload = _payload(
        [{"role": "user", "content": "Please summarize"}],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 1.0
    assert evidence.empty_completion_failure is False
    assert evidence.large_tool_result_payload_risk is False
    assert evidence.reasons == []


def test_should_score_destructive_checkout_after_mutating_tool_use() -> None:
    payload = _payload(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": "src/App.tsx", "old_string": "a"},
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git checkout -- src/App.tsx"},
                    },
                ],
            }
        ],
        {"role": "assistant", "content": "stopped", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=10),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 0.0
    assert evidence.destructive_checkout_after_work is True
    assert evidence.mutating_tool_uses_before_destructive_checkout == 1
    assert evidence.destructive_checkout_command == "git checkout -- src/App.tsx"
    assert evidence.reasons == ["destructive_checkout_after_mutating_tool_use"]


def test_should_score_openai_tool_calls_with_destructive_checkout() -> None:
    payload = _openai_payload(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": json.dumps({"file_path": "src/App.tsx"}),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps(
                                {"cmd": "git restore --source=HEAD src/App.tsx"}
                            ),
                        },
                    },
                ],
            }
        ],
        {"role": "assistant", "content": "done", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(model="grok-build", output_tokens=10),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.message_count == 1
    assert evidence.trace_quality_score == 0.0
    assert evidence.destructive_checkout_after_work is True
    assert evidence.destructive_checkout_command == "git restore --source=HEAD src/App.tsx"
    assert evidence.reasons == ["destructive_checkout_after_mutating_tool_use"]


def test_should_keep_parse_errors_out_of_score_reasons() -> None:
    evidence = scorer.score_candidate(
        _candidate(),
        scorer.ObservationPayload(
            observation_id="obs-1",
            trace_id="trace-1",
            body={"input": "not a chat payload", "output": "{}"},
            source="clickhouse",
        ),
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.errors == ["missing_request_messages"]
    assert evidence.reasons == []
    assert evidence.trace_quality_score == 1.0


def test_should_not_score_safe_branch_checkout_as_destructive() -> None:
    payload = _payload(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Edit", "input": {}},
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git checkout -b feature/test"},
                    },
                ],
            }
        ],
        {"role": "assistant", "content": "done", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=10),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 1.0
    assert evidence.destructive_checkout_after_work is False
    assert evidence.reasons == []


def test_should_detect_anthropic_user_tool_result_error_payload() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "text",
                                "text": "<tool_use_error>InputValidationError: Bash failed due to the following issue: unexpected parameter `command`.",
                            }
                        ],
                    }
                ],
            }
        ],
        {"role": "assistant", "content": "Recovered from tool error.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.invalid_tool_call_error_count == 1
    assert evidence.invalid_tool_call_error_markers == [
        "InputValidationError",
        "tool_use_error",
        "unexpected_parameter",
    ]
    assert evidence.trace_quality_score == 1.0
    assert evidence.reasons == ["invalid_tool_call_error_seen"]


def test_should_detect_openai_tool_role_error_payload_and_preserve_quality() -> None:
    payload = _openai_payload(
        [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "InputValidationError: Bash failed due to invalid tool",
            }
        ],
        {"role": "assistant", "content": "Recovered."},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.invalid_tool_call_error_count == 1
    assert evidence.invalid_tool_call_error_markers == [
        "InputValidationError",
        "invalid_tool",
    ]
    assert evidence.trace_quality_score == 1.0
    assert evidence.reasons == ["invalid_tool_call_error_seen"]


def test_should_use_session_history_invalid_tool_count_when_payload_missing() -> None:
    evidence = scorer.score_candidate(
        _candidate(invalid_tool_call_count=2),
        None,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.invalid_tool_call_error_count == 2
    assert evidence.invalid_tool_call_error_markers == [
        "session_history_invalid_tool_call_count"
    ]
    assert evidence.reasons == ["invalid_tool_call_error_seen"]
    assert evidence.errors == ["missing_observation_payload", "missing_request_messages"]


def test_should_build_stable_langfuse_score_payloads() -> None:
    evidence = scorer.score_candidate(
        _candidate(),
        _payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "data": "a" * 120_000,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            {"role": "assistant", "content": "", "tool_calls": None},
        ),
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    payloads = [score.payload() for score in scorer.build_langfuse_scores(evidence)]

    assert {payload["name"] for payload in payloads} == {
        "aawm.agent.trace_quality",
        "aawm.agent.empty_completion_failure",
        "aawm.agent.large_tool_result_payload_risk",
        "aawm.agent.destructive_checkout_after_work",
        "aawm.agent.invalid_tool_call_error",
    }
    assert all(payload["traceId"] == "trace-1" for payload in payloads)
    assert all(payload["observationId"] == "obs-1" for payload in payloads)
    assert len({payload["id"] for payload in payloads}) == 5


def test_invalid_tool_call_error_score_includes_marker_count_comment() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "text",
                                "text": "unrecognized parameter `cmd`",
                            }
                        ],
                    }
                ],
            }
        ],
        {"role": "assistant", "content": "Recovered.", "tool_calls": None},
    )
    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )
    score_payloads = [score.payload() for score in scorer.build_langfuse_scores(evidence)]
    invalid_score = next(
        payload
        for payload in score_payloads
        if payload["name"] == "aawm.agent.invalid_tool_call_error"
    )
    assert invalid_score["value"] == 1.0
    assert evidence.to_json()["invalid_tool_call_error_count"] == 1
    assert "invalid_tool_call_errors=1" in invalid_score["comment"]
