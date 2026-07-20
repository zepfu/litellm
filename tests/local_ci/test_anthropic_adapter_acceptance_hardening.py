import importlib.util
import datetime as dt
import json
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"
ANTHROPIC_ADAPTER_CONFIG_PATH = (
    ROOT / "scripts" / "local-ci" / "anthropic_adapter_config.json"
)
SEQUENTIAL_CORE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Bash",
    "WebSearch",
    "WebFetch",
]
SEQUENTIAL_CASE_AGENTS = {
    "claude_adapter_gpt55_child_sequential_core_tools": (
        "harness-gpt55-sequential-core-tools"
    ),
}
PARALLEL_READ_TOOLS = ["Read", "Glob", "Grep"]
PARALLEL_CASE_AGENTS = {
    "claude_adapter_gpt55_child_parallel_read_tools": (
        "harness-gpt55-parallel-read-tools",
        "openai",
        "gpt-5.5",
        {"Read", "Glob", "Grep"},
    ),
    "claude_adapter_openrouter_nemotron_child_parallel_read_tools": (
        "harness-openrouter-nemotron-parallel-read-tools",
        "openrouter",
        "nvidia/nemotron-3-super-120b-a12b:free",
        {"Read", "Glob", "Grep"},
    ),
    "claude_adapter_nvidia_deepseek_child_parallel_read_tools": (
        "harness-nvidia-deepseek-parallel-read-tools",
        "nvidia_nim",
        "deepseek-ai/deepseek-v3.2",
        {"Read", "Glob", "Grep"},
    ),
}
D1251_PARALLEL_CASE_AGENTS = {
    "claude_adapter_gpt53_child_parallel_read_tools": (
        "harness-openai-gpt53-parallel-read-tools",
        "openai",
        "gpt-5.3-codex-spark",
        {"Read", "Glob", "Grep"},
        "openai/gpt-5.3-codex-spark",
    ),
    "claude_adapter_openrouter_deepseek_v4_flash_child_parallel_read_tools": (
        "harness-openrouter-deepseek-v4-flash-parallel-read-tools",
        "openrouter",
        "openrouter/deepseek/deepseek-v4-flash",
        {"Read", "Glob", "Grep"},
        "openrouter/deepseek/deepseek-v4-flash",
    ),
    "claude_adapter_anthropic_claude_haiku_4_5_20251001_child_parallel_read_tools": (
        "harness-anthropic-haiku-4-5-parallel-read-tools",
        "anthropic",
        "claude-haiku-4-5-20251001",
        {"Read", "Glob", "Grep"},
        "claude-haiku-4-5-20251001",
    ),
    "claude_adapter_anthropic_claude_opus_4_8_child_parallel_read_tools": (
        "harness-anthropic-opus-4-8-parallel-read-tools",
        "anthropic",
        "claude-opus-4-8",
        {"Read", "Glob", "Grep"},
        "claude-opus-4-8",
    ),
    "claude_adapter_antigravity_claude_sonnet_4_6_child_sequential_read_tools": (
        "harness-antigravity-claude-sonnet-4-6-sequential-read-tools",
        "antigravity",
        "claude-sonnet-4-6",
        {"read_file", "glob", "grep_search"},
        "antigravity/claude-sonnet-4-6",
    ),
    "claude_adapter_xai_grok_composer_child_parallel_read_tools": (
        "harness-xai-grok-composer-parallel-read-tools",
        "xai",
        "grok-composer-2.5-fast",
        {"Read", "Glob", "Grep"},
        "xai/grok-composer-2.5-fast",
    ),
    "claude_adapter_xai_oa_xai_grok_build_child_parallel_read_tools": (
        "harness-xai-oa-xai-grok-build-parallel-read-tools",
        "xai",
        "oa_xai/grok-build",
        {"Read", "Glob", "Grep"},
        "oa_xai/grok-build",
    ),
    "claude_adapter_anthropic_claude_sonnet_4_6_child_parallel_read_tools": (
        "harness-anthropic-sonnet-4-6-parallel-read-tools",
        "anthropic",
        "claude-sonnet-4-6",
        {"Read", "Glob", "Grep"},
        "claude-sonnet-4-6",
    ),
    "claude_adapter_opencode_zen_deepseek_v4_flash_child_parallel_read_tools": (
        "harness-opencode-zen-deepseek-v4-flash-parallel-read-tools",
        "opencode_zen",
        "deepseek-v4-flash",
        {"Read", "Glob", "Grep"},
        "opencode_zen/deepseek-v4-flash",
    ),
    "claude_adapter_opencode_zen_big_pickle_child_parallel_read_tools": (
        "harness-opencode-zen-big-pickle-parallel-read-tools",
        "opencode_zen",
        "big-pickle",
        {"Read", "Glob", "Grep"},
        "opencode_zen/big-pickle",
    ),
}
FORBIDDEN_D1251_GEMINI_CASE_MODELS = {
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
}
FORBIDDEN_D1251_GOOGLE_CHILD_PARALLEL_MODEL_PREFIXES = {
    "google/gemma",
    "openrouter/google/gemma",
}


def _is_forbidden_d1251_child_parallel_model(model: str) -> bool:
    model = model.lower()
    if model in FORBIDDEN_D1251_GEMINI_CASE_MODELS:
        return True
    return any(
        model.startswith(prefix)
        for prefix in FORBIDDEN_D1251_GOOGLE_CHILD_PARALLEL_MODEL_PREFIXES
    )


D1251_REQUIRED_TRACE_TAGS = {
    "openai": {"route:anthropic_openai_responses_adapter"},
    "openrouter": {"route:anthropic_openrouter_responses_adapter"},
    "anthropic": {"route:anthropic_messages"},
    "antigravity": {"route:anthropic_antigravity_completion_adapter"},
    "xai": {
        "route:anthropic_grok_native_responses_adapter",
        "route:anthropic_xai_oauth_responses_adapter",
    },
    "opencode_zen": {
        "route:anthropic_opencode_zen_responses_adapter",
        "route:anthropic_opencode_zen_completion_adapter",
    },
}
D1251_REQUIRED_TRACE_TAGS_BY_CASE = {
    "claude_adapter_xai_grok_composer_child_parallel_read_tools": {
        "route:anthropic_grok_native_responses_adapter",
    },
    "claude_adapter_xai_oa_xai_grok_build_child_parallel_read_tools": {
        "route:anthropic_xai_oauth_responses_adapter",
    },
}
D1251_DISALLOWED_TRACE_TAGS_BY_CASE = {
    "claude_adapter_xai_grok_composer_child_parallel_read_tools": {
        "route:anthropic_xai_oauth_responses_adapter",
    },
    "claude_adapter_xai_oa_xai_grok_build_child_parallel_read_tools": {
        "route:anthropic_grok_native_responses_adapter",
    },
}
D1251_SEQUENTIAL_TRANSCRIPT_CASES = {
    "claude_adapter_antigravity_claude_sonnet_4_6_child_sequential_read_tools",
}
D1251_OPENCODE_COMPLETION_CASES = {
    "claude_adapter_opencode_zen_big_pickle_child_parallel_read_tools",
}
D1256_ALIAS_REPLAY_CASE = (
    "claude_adapter_aawm_code_anthropic_alias_child_parallel_read_tools"
)
D1256_ALIAS_REPLAY_AGENT = "harness-aawm-code-anthropic-alias-parallel-read-tools"
D1256_AAWM_CODE_ANTHROPIC_DECLARED_PROVIDER_MODELS = {
    ("antigravity", "claude-sonnet-4-6"),
    ("openai", "gpt-5.3-codex-spark"),
    ("xai", "grok-composer-2.5-fast"),
    ("xai", "oa_xai/grok-build"),
    ("anthropic", "claude-sonnet-4-6"),
}
MS012_MOONSHOT_AGENTIC_CASE = "claude_adapter_aawm_sota_moonshot_agentic_tool_continuation"
MS012_MOONSHOT_BASH_TIME_CASE = "claude_adapter_aawm_sota_moonshot_child_bash_time"
MS012_MOONSHOT_STRESS_CASE = "claude_adapter_aawm_sota_moonshot_parallel_stress"
MS012_MOONSHOT_AGENT_PROFILE = "sota-moonshot"
MS012_MOONSHOT_TIME_AGENT_PROFILE = "sota-moonshot-time"
MS012_MOONSHOT_ALIAS = "aawm-sota-moonshot"
MOONSHOT_CODEX_BASH_TIME_CASE = (
    "native_openai_passthrough_responses_codex_aawm_sota_moonshot_bash_time"
)
MOONSHOT_CODEX_COLLABORATION_CASE = (
    "native_openai_passthrough_responses_codex_aawm_sota_moonshot_collaboration"
)
MS012_MOONSHOT_ADAPTER_PATH = "anthropic_kimi_chat_completions_adapter"
MS012_MOONSHOT_DECLARED_MODELS = {
    "kimi_code/k3-max",
    "kimi_code/k3-high",
}

D1322_LOW_ALIAS_REPLAY_CASES = (
    "claude_adapter_aawm_low_anthropic_alias_child_parallel_read_tools",
    "native_openai_passthrough_responses_codex_aawm_low_tool_activity",
)
D1322_OPENROUTER_COMPLETION_CASES = {
    "claude_adapter_openrouter_north_mini_completion_adapter_smoke": {
        "requested_model": "openrouter/cohere/north-mini-code:free",
        "adapter_model": "cohere/north-mini-code:free",
        "candidate_order": 1,
    },
    "claude_adapter_openrouter_owl_alpha_completion_adapter_smoke": {
        "requested_model": "openrouter/owl-alpha",
        "adapter_model": "owl-alpha",
        "candidate_order": 2,
    },
}
D1322_LOW_ANTHROPIC_ALIAS_REPLAY_CASE = (
    "claude_adapter_aawm_low_anthropic_alias_child_parallel_read_tools"
)
D1322_LOW_ANTHROPIC_ALIAS_REPLAY_AGENT = (
    "harness-aawm-low-anthropic-alias-parallel-read-tools"
)
D1322_CODEX_LOW_ALIAS_REPLAY_CASE = (
    "native_openai_passthrough_responses_codex_aawm_low_tool_activity"
)
D1322_AAWM_LOW_ANTHROPIC_DECLARED_PROVIDER_MODELS = {
    ("antigravity", "gemini-3.5-flash-low"),
    ("openrouter", "openrouter/cohere/north-mini-code:free"),
    ("openrouter", "openrouter/owl-alpha"),
    ("opencode_zen", "deepseek-v4-flash"),
    ("opencode_zen", "big-pickle"),
    ("anthropic", "claude-haiku-4-5-20251001"),
}
D1322_AAWM_LOW_CODEX_DECLARED_PROVIDER_MODELS = {
    ("antigravity", "gemini-3.5-flash-low"),
    ("openrouter", "openrouter/cohere/north-mini-code:free"),
    ("openrouter", "openrouter/owl-alpha"),
    ("opencode_zen", "deepseek-v4-flash"),
    ("opencode_zen", "big-pickle"),
    ("openai", "gpt-5.4-mini"),
}

REMOVED_GEMINI_HARNESS_CASES = {
    "native_gemini_passthrough_generate_content",
    "native_gemini_passthrough_stream_generate_content",
    "claude_adapter_gemini31_pro_read_tool_id_sanitizer",
    "claude_adapter_gemini31_pro_bash_then_read_stream_state",
    "claude_adapter_gemini31_pro_child_sequential_core_tools",
    "claude_adapter_gemini31_pro_child_parallel_read_tools",
    "claude_adapter_gemini3_flash_child_sequential_core_tools",
    "claude_adapter_gemini3_flash_child_parallel_read_tools",
    "claude_adapter_gemini_output_config_effort",
    "claude_adapter_gemini_output_config_minimal_effort",
    "claude_adapter_gemini_output_config_max_effort",
    "claude_adapter_gemini_output_config_minimal_effort_cache",
    "claude_adapter_gemini_output_config_max_effort_cache",
    "claude_adapter_gemini_fanout",
    "claude_adapter_gemini31_pro",
    "claude_adapter_gemini31_flash",
    "claude_adapter_gemma_31b",
    "claude_adapter_gemma_26b_a4b",
}
ACTIVE_ANTHROPIC_HARNESS_SURFACES = (
    HARNESS_PATH,
    ANTHROPIC_ADAPTER_CONFIG_PATH,
    ROOT / "scripts" / "local-ci" / "README.md",
)
FORBIDDEN_ACTIVE_GEMINI_HARNESS_SNIPPETS = (
    ".gemini",
    "@google/gemini-cli",
    "gemini_oauth",
    "litellm_gemini",
    "gemini_cli",
    '"cli_passthrough": "gemini"',
    "'cli_passthrough': 'gemini'",
    '"gemini", "prompt"',
    "google_code_assist",
    "google/gemini",
    "google/gemma",
)
DIRECT_ANTHROPIC_MODEL_PATTERN = re.compile(
    r"(?:^|[/:\s])anthropic(?:[/:\s]|$)|^aawm-.+-anthropic$",
    re.IGNORECASE,
)


def _load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "run_anthropic_adapter_acceptance_test_module",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warning_only_timeout_is_soft_by_default():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "warning_only_fixture",
        subprocess.TimeoutExpired(["claude"], 180),
        {"warning_only": True},
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["soft_failures"]
    assert result["warnings"]


def test_warning_only_timeout_can_be_hard_failed_by_config():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "warning_only_fixture",
        subprocess.TimeoutExpired(["claude"], 180),
        {
            "warning_only": True,
            "warning_only_hard_failure_substrings": ["timed out after"],
        },
    )

    assert result["passed"] is False
    assert result["failures"]
    assert result.get("soft_failures") in (None, [])


def test_empty_success_validation_hard_fails_zero_usage_success():
    harness = _load_harness_module()

    summary, failures = harness._validate_no_successful_empty_command_output(
        family="openrouter_empty_success_fixture",
        stdout=json.dumps(
            {
                "is_error": False,
                "result": "",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        ),
        stderr="",
        checks={"fail_empty_success": True},
    )

    assert summary["result_empty"] is True
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0
    assert any("successful empty command result" in failure for failure in failures)
    assert any("successful empty command usage" in failure for failure in failures)


def test_empty_success_validation_catches_adapter_diagnostic():
    harness = _load_harness_module()

    _, failures = harness._validate_no_successful_empty_command_output(
        family="openrouter_empty_success_fixture",
        stdout="",
        stderr=(
            "OpenRouter Responses adapter returned empty successful response: "
            '{"events":[]}'
        ),
        checks={"fail_empty_success": True},
    )

    assert failures == [
        "openrouter_empty_success_fixture successful empty OpenRouter adapter diagnostic surfaced"
    ]


def test_provider_unavailable_timeout_can_soft_fail_with_exact_log_signature(monkeypatch):
    harness = _load_harness_module()

    def fake_read_runtime_logs_since(**kwargs):
        return (
            {"docker_logs_exit_code": 0, "log_excerpt": "provider unavailable"},
            (
                "OpenRouter adapter upstream attempt 1/4\n"
                "failed with 503 (ProxyException, provider=OpenInference, "
                "raw=no healthy upstream)"
            ),
        )

    monkeypatch.setattr(
        harness,
        "_read_runtime_logs_since",
        fake_read_runtime_logs_since,
    )

    result = harness._provider_unavailable_timeout_error_result(
        "claude_adapter_gpt_oss_120b",
        subprocess.TimeoutExpired(["claude"], 180),
        {
            "soft_fail_timeout_runtime_log_check": {
                "required_substrings": [
                    "OpenRouter adapter upstream attempt",
                    "failed with 503",
                    "provider=OpenInference",
                    "raw=no healthy upstream",
                ]
            },
            "runtime_postconditions": {"docker_container_name": "aawm-litellm"},
        },
        started="2026-04-24T11:12:45+00:00",
    )

    assert result is not None
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["soft_failures"]
    assert result["warnings"]
    assert result["runtime_logs"]["matched_required_substrings"] == [
        "OpenRouter adapter upstream attempt",
        "failed with 503",
        "provider=OpenInference",
        "raw=no healthy upstream",
    ]


def test_generation_route_filter_excludes_anthropic_count_tokens_observations():
    harness = _load_harness_module()

    assert harness.RA._generation_observation_matches_allowed_route(
        {
            "metadata": {
                "user_api_key_request_route": "/anthropic/v1/messages/count_tokens",
                "passthrough_route_family": "anthropic_messages",
                "tags": ["route:anthropic_messages"],
            }
        },
        ["/anthropic/v1/messages"],
    ) is False

    assert harness.RA._generation_observation_matches_allowed_route(
        {
            "metadata": {
                "user_api_key_request_route": "/anthropic/v1/messages",
                "passthrough_route_family": "anthropic_messages",
                "tags": ["route:anthropic_messages"],
            }
        },
        ["/anthropic/v1/messages"],
    ) is True


def test_provider_unavailable_timeout_stays_hard_without_exact_log_signature(monkeypatch):
    harness = _load_harness_module()

    monkeypatch.setattr(
        harness,
        "_read_runtime_logs_since",
        lambda **kwargs: (
            {"docker_logs_exit_code": 0, "log_excerpt": "adapter traceback"},
            "OpenRouter adapter upstream attempt but local adapter failed",
        ),
    )

    result = harness._provider_unavailable_timeout_error_result(
        "claude_adapter_gpt_oss_120b",
        subprocess.TimeoutExpired(["claude"], 180),
        {
            "soft_fail_timeout_runtime_log_check": {
                "required_substrings": [
                    "OpenRouter adapter upstream attempt",
                    "failed with 503",
                    "provider=OpenInference",
                    "raw=no healthy upstream",
                ]
            },
            "runtime_postconditions": {"docker_container_name": "aawm-litellm"},
        },
        started="2026-04-24T11:12:45+00:00",
    )

    assert result is None


def test_command_json_validator_accepts_required_regex():
    harness = _load_harness_module()

    summary, failures = harness._validate_command_output_json(
        family="case",
        stdout=json.dumps({"is_error": False, "result": "2026-04-27\n"}),
        checks={"required_regex": {"result": r"^\d{4}-\d{2}-\d{2}\s*$"}},
    )

    assert failures == []
    assert summary["required_regex_hits"] == {"result": "2026-04-27\n"}


def test_command_json_validator_selects_terminal_claude_result_from_event_array():
    harness = _load_harness_module()
    stdout = json.dumps(
        [
            {"type": "system", "subtype": "init", "session_id": "session-1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "ok",
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        ]
    )

    summary, failures = harness._validate_command_output_json(
        family="case",
        stdout=stdout,
        checks={
            "required_equals": {"is_error": False, "result": "ok"},
            "required_minimums": {
                "usage.input_tokens": 1,
                "usage.output_tokens": 1,
                "total_cost_usd": 1e-06,
            },
        },
    )

    assert failures == []
    assert summary["parsed"]["type"] == "result"


def test_command_json_validator_rejects_required_regex_mismatch():
    harness = _load_harness_module()
    pattern = r"^\d{4}-\d{2}-\d{2}\s*$"

    summary, failures = harness._validate_command_output_json(
        family="case",
        stdout=json.dumps({"is_error": False, "result": "looked at files"}),
        checks={"required_regex": {"result": pattern}},
    )

    assert summary["required_regex_hits"] == {"result": "looked at files"}
    assert failures == [
        f"case command JSON regex mismatch for `result`: expected pattern {pattern!r}, got 'looked at files'"
    ]


def test_bash_stdout_report_validator_accepts_exact_codex_command_output():
    harness = _load_harness_module()
    timestamp = "2026-07-20T16:05:19-04:00"
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "date --iso-8601=seconds",
                        "aggregated_output": f"{timestamp}\n",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": timestamp,
                    },
                }
            ),
        ]
    )

    summary, failures = harness._validate_bash_stdout_report(
        family="case",
        stdout=stdout,
        checks={
            "expected_command": "date --iso-8601=seconds",
            "expected_regex": (
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                r"(?:Z|[+-]\d{2}:\d{2})$"
            ),
        },
        transcript_tool_use_summary={"agents": []},
    )

    assert failures == []
    assert summary["source"] == "codex_command_stdout"
    assert summary["bash_stdout"] == timestamp
    assert summary["parent_output"] == timestamp


def test_bash_stdout_report_validator_accepts_exact_claude_child_and_parent_output():
    harness = _load_harness_module()
    timestamp = "2026-07-20T16:06:21-04:00"

    summary, failures = harness._validate_bash_stdout_report(
        family="case",
        stdout=json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": timestamp,
            }
        ),
        checks={
            "expected_command": "date --iso-8601=seconds",
            "expected_regex": (
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                r"(?:Z|[+-]\d{2}:\d{2})$"
            ),
            "transcript_agent": "sota-moonshot-time",
        },
        transcript_tool_use_summary={
            "agents": [
                {
                    "expected_agent": "sota-moonshot-time",
                    "records": [
                        {
                            "tool_name": "Bash",
                            "input_preview": (
                                '{"command": "date --iso-8601=seconds"}'
                            ),
                            "tool_result_content_text": timestamp,
                        }
                    ],
                    "assistant_texts": [{"text": timestamp}],
                }
            ]
        },
    )

    assert failures == []
    assert summary["source"] == "claude_subagent_transcript"
    assert summary["bash_stdout"] == timestamp
    assert summary["child_output"] == timestamp
    assert summary["parent_output"] == timestamp


def test_bash_stdout_report_validator_rejects_rewritten_parent_output():
    harness = _load_harness_module()
    timestamp = "2026-07-20T16:07:22-04:00"
    reported = f"System time: {timestamp}"

    _summary, failures = harness._validate_bash_stdout_report(
        family="case",
        stdout=json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": reported,
            }
        ),
        checks={
            "expected_command": "date --iso-8601=seconds",
            "expected_regex": (
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                r"(?:Z|[+-]\d{2}:\d{2})$"
            ),
            "transcript_agent": "sota-moonshot-time",
        },
        transcript_tool_use_summary={
            "agents": [
                {
                    "expected_agent": "sota-moonshot-time",
                    "records": [
                        {
                            "tool_name": "Bash",
                            "input_preview": (
                                '{"command": "date --iso-8601=seconds"}'
                            ),
                            "tool_result_content_text": timestamp,
                        }
                    ],
                    "assistant_texts": [{"text": timestamp}],
                }
            ]
        },
    )

    assert failures == [
        "case parent response did not exactly report Bash stdout: "
        f"expected {timestamp!r}, got {reported!r}"
    ]


def test_generation_route_filter_accepts_route_tag_without_request_route(monkeypatch):
    harness = _load_harness_module()
    observation = {
        "id": "generation-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "startTime": "2026-04-28T02:15:23Z",
        "model": "gpt-5.5",
        "promptTokens": 10,
        "completionTokens": 2,
        "totalTokens": 12,
        "costDetails": {"total": 0.01},
        "calculatedTotalCost": 0.01,
        "metadata": {
            "passthrough_route_family": "anthropic_openai_responses_adapter",
            "tags": [
                "route:anthropic_messages",
                "route:anthropic_openai_responses_adapter",
            ],
        },
    }

    monkeypatch.setattr(
        harness.RA,
        "_recent_langfuse_generation_observations_for_trace_ids",
        lambda **_kwargs: [observation],
    )

    raw_observations, summaries, failures = harness.RA._validate_generation_observations(
        family="case",
        query_url="http://127.0.0.1:3000",
        public_key="pk",
        secret_key="sk",
        trace_ids=["trace-1"],
        start_time=dt.datetime(2026, 4, 28, 2, 15, tzinfo=dt.timezone.utc),
        allowed_request_routes=["/anthropic/v1/messages"],
    )

    assert failures == []
    assert raw_observations == [observation]
    assert summaries[0]["traceId"] == "trace-1"


def test_stream_tool_call_state_validation_accepts_recorded_responses_state():
    harness = _load_harness_module()
    observation = {
        "id": "generation-1",
        "metadata": {
            "responses_stream_event_types": [
                "response.output_item.added",
                "response.function_call_arguments.done",
                "response.output_item.done",
            ],
            "responses_stream_event_counts": {
                "response.output_item.added": 1,
                "response.function_call_arguments.done": 1,
                "response.output_item.done": 1,
            },
            "responses_stream_tool_state": [
                {
                    "type": "function_call",
                    "name": "Bash",
                    "call_id": "call_pwd",
                    "arguments": '{"command":"pwd"}',
                }
            ],
        },
    }

    summary, failures = harness._validate_stream_tool_call_state(
        family="case",
        observations=[observation],
        checks={
            "required_event_types": [
                "response.output_item.added",
                "response.output_item.done",
            ],
            "required_any_event_type_groups": [
                [
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                ]
            ],
            "expected_tools": [
                {
                    "tool_name": "Bash",
                    "tool_type": "function_call",
                    "arguments_required_substrings": ["pwd"],
                }
            ],
        },
    )

    assert failures == []
    assert summary["event_counts"]["response.output_item.added"] == 1
    assert summary["tool_names"] == ["Bash"]


def test_stream_tool_call_state_validation_rejects_missing_event_and_tool():
    harness = _load_harness_module()

    _summary, failures = harness._validate_stream_tool_call_state(
        family="case",
        observations=[
            {
                "id": "generation-1",
                "metadata": {
                    "responses_stream_event_types": ["response.created"],
                    "responses_stream_tool_state": [],
                },
            }
        ],
        checks={
            "required_event_types": ["response.output_item.added"],
            "required_any_event_type_groups": [
                [
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                ]
            ],
            "expected_tools": [
                {
                    "tool_name": "Bash",
                    "arguments_required_substrings": ["pwd"],
                }
            ],
        },
    )

    assert failures == [
        "case missing Responses stream event type `response.output_item.added`",
        "case missing any Responses stream event type from ['response.function_call_arguments.delta', 'response.function_call_arguments.done']",
        "case missing Responses stream tool state for {'tool_name': 'Bash', 'arguments_required_substrings': ['pwd']}; expected >= 1, got 0",
    ]


def test_stream_tool_call_state_uses_command_output_for_compacted_arguments():
    harness = _load_harness_module()
    observation = {
        "id": "generation-1",
        "metadata": {
            "responses_stream_event_types": [
                "response.output_item.added",
                "response.function_call_arguments.done",
                "response.output_item.done",
            ],
            "responses_stream_tool_state": {
                "type": "litellm_langfuse_metadata_compacted",
                "field": "responses_stream_tool_state",
                "tool_call_count": 1,
                "tool_names": ["exec_command"],
                "tool_type_counts": {"function_call": 1},
                "sample_tool_calls": [
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": "call-pwd",
                        "arguments_hash": "hash",
                    }
                ],
            },
        },
    }
    command_stdout = json.dumps(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-pwd",
                            "name": "Bash",
                            "input": {"command": "pwd"},
                        }
                    ]
                },
            }
        ]
    )

    summary, failures = harness._validate_stream_tool_call_state(
        family="case",
        observations=[observation],
        checks={
            "required_event_types": [
                "response.output_item.added",
                "response.output_item.done",
            ],
            "expected_tools": [
                {
                    "tool_name": "exec_command",
                    "tool_type": "function_call",
                    "arguments_required_substrings": ["pwd"],
                }
            ],
        },
        command_stdout=command_stdout,
    )

    assert failures == []
    assert summary["compacted_tool_state"] is True
    assert summary["tool_names"] == ["exec_command"]
    assert summary["command_tool_names"] == ["exec_command"]


def test_provider_unavailable_command_failure_can_soft_fail_with_exact_log_signature():
    harness = _load_harness_module()

    failures, soft_failures, warnings, runtime_logs = (
        harness._provider_unavailable_failure_soft_fail_result(
            failures=[
                "claude_adapter_gpt_oss_120b command failed",
                "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator",
            ],
            warnings=[],
            config={
                "soft_fail_timeout_runtime_log_check": {
                    "required_substrings": [
                        "OpenRouter adapter upstream attempt",
                        "failed with 503",
                        "provider=OpenInference",
                        "raw=no healthy upstream",
                    ]
                }
            },
            runtime_logs={
                "log_excerpt": (
                    "OpenRouter adapter upstream attempt 1/4\n"
                    "failed with 503 (ProxyException, provider=OpenInference, "
                    "raw=no healthy upstream)"
                )
            },
        )
    )

    # RR-082: only connectivity/timeout-class failures soft-fail; unrelated
    # validation failures (e.g. missing Langfuse traces) stay hard.
    assert soft_failures == [
        "claude_adapter_gpt_oss_120b command failed",
    ]
    assert failures == [
        "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator",
    ]
    assert warnings
    assert runtime_logs["matched_soft_fail_substrings"] == [
        "OpenRouter adapter upstream attempt",
        "failed with 503",
        "provider=OpenInference",
        "raw=no healthy upstream",
    ]


def test_provider_unavailable_command_failure_does_not_mask_runtime_log_failures():
    harness = _load_harness_module()

    failures, soft_failures, warnings, _ = (
        harness._provider_unavailable_failure_soft_fail_result(
            failures=[
                "claude_adapter_gpt_oss_120b runtime logs contained forbidden substring `KeyError: 'choices'`",
            ],
            warnings=[],
            config={
                "soft_fail_timeout_runtime_log_check": {
                    "required_substrings": [
                        "OpenRouter adapter upstream attempt",
                        "failed with 503",
                        "provider=OpenInference",
                        "raw=no healthy upstream",
                    ]
                }
            },
            runtime_logs={
                "log_excerpt": (
                    "OpenRouter adapter upstream attempt 1/4\n"
                    "failed with 503 (ProxyException, provider=OpenInference, "
                    "raw=no healthy upstream)"
                )
            },
        )
    )

    assert failures
    assert soft_failures == []
    assert warnings == []


def test_warning_only_noncritical_exception_remains_soft():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "optional_provider_case",
        RuntimeError("temporary upstream quality mismatch"),
        {"warning_only": True},
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["soft_failures"]
    assert result["warnings"]


def test_runtime_log_defaults_catch_async_and_content_length_errors(monkeypatch):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            "Task exception was never retrieved\n"
            "KeyError: 'choices'\n"
            "h11._util.LocalProtocolError: Too little data for declared Content-Length"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="openrouter_runtime_log_fixture",
        started="2026-04-23T14:06:00+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
    )

    assert warnings == []
    assert failures
    assert "Task exception was never retrieved" in summary["matched_forbidden_substrings"]
    assert "KeyError: 'choices'" in summary["matched_forbidden_substrings"]
    assert (
        "Too little data for declared Content-Length"
        in summary["matched_forbidden_substrings"]
    )
    assert "KeyError: 'choices'" in summary["matched_forbidden_contexts"]
    assert "KeyError: 'choices'" in summary["matched_forbidden_contexts"][
        "KeyError: 'choices'"
    ]


def test_runtime_log_read_is_bounded_by_until(monkeypatch):
    harness = _load_harness_module()
    commands = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, *args, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    started = dt.datetime(2026, 4, 23, 14, 6, tzinfo=dt.timezone.utc)
    until = dt.datetime(2026, 4, 23, 14, 7, tzinfo=dt.timezone.utc)

    summary, log_text = harness._read_runtime_logs_since(
        started=started,
        until=until,
        checks={"docker_container_name": "litellm-dev", "tail_lines": 25},
        runtime_postconditions={},
    )

    assert log_text == ""
    assert commands == [
        [
            "docker",
            "logs",
            "--since",
            started.isoformat(),
            "--until",
            until.isoformat(),
            "--tail",
            "25",
            "litellm-dev",
        ]
    ]
    assert summary["docker_logs_since"] == started.isoformat()
    assert summary["docker_logs_until"] == until.isoformat()


def test_runtime_log_ignores_unattributed_concurrent_auto_agent_traceback(monkeypatch):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            "litellm.exceptions.NotFoundError: OpenrouterException - "
            '{"error":{"message":"No endpoints found for '
            'deepseek/deepseek-v4-flash:free."}}\n'
            "ERROR:    Exception in ASGI application\n"
            + ("  File \"/app/noise.py\", line 1, in frame\n" * 40)
            + "  File \"/app/litellm/proxy/pass_through_endpoints/"
            "llm_passthrough_endpoints.py\", line 20899, in "
            "_base_openai_pass_through_handler\n"
            "    return await _handle_codex_auto_agent_alias_route(...)\n"
            "  File \"/app/litellm/proxy/pass_through_endpoints/"
            "llm_passthrough_endpoints.py\", line 20436, in "
            "_perform_codex_auto_agent_openrouter_completion_request\n"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_gpt54",
        started="2026-06-06T21:04:15+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
        attribution_substrings=[
            "claude_adapter_gpt54",
            "gpt-5.4",
            "active-session-id",
            "/anthropic/v1/messages",
        ],
    )

    assert failures == []
    assert warnings == [
        "claude_adapter_gpt54 ignored unattributed runtime log match "
        "`Exception in ASGI application` from unrelated concurrent container traffic"
    ]
    assert summary["matched_forbidden_substrings"] == []
    assert summary["ignored_unattributed_forbidden_substrings"] == [
        "Exception in ASGI application"
    ]
    assert "Exception in ASGI application" in summary[
        "ignored_unattributed_forbidden_contexts"
    ]
    assert (
        "_perform_codex_auto_agent_openrouter_completion_request"
        in summary["ignored_unattributed_forbidden_contexts"][
            "Exception in ASGI application"
        ]
    )


def test_runtime_log_keeps_attributed_exception_hard(monkeypatch):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            "model=gpt-5.4 session=active-session-id\n"
            "ERROR:    Exception in ASGI application\n"
            "Traceback for active request"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_gpt54",
        started="2026-06-06T21:04:15+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
        attribution_substrings=["gpt-5.4", "active-session-id"],
    )

    assert warnings == []
    assert failures == [
        "claude_adapter_gpt54 runtime logs contained forbidden substring "
        "`Exception in ASGI application`"
    ]
    assert summary["matched_forbidden_substrings"] == [
        "Exception in ASGI application"
    ]
    assert summary["ignored_unattributed_forbidden_substrings"] == []


def test_runtime_log_keeps_auto_agent_traceback_without_unrelated_error_hard(
    monkeypatch,
):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            "ERROR:    Exception in ASGI application\n"
            "  File \"/app/litellm/proxy/pass_through_endpoints/"
            "llm_passthrough_endpoints.py\", line 20899, in "
            "_base_openai_pass_through_handler\n"
            "    return await _handle_codex_auto_agent_alias_route(...)\n"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_gpt54",
        started="2026-06-06T21:04:15+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
        attribution_substrings=["gpt-5.4", "active-session-id"],
    )

    assert warnings == []
    assert failures == [
        "claude_adapter_gpt54 runtime logs contained forbidden substring "
        "`Exception in ASGI application`"
    ]
    assert summary["matched_forbidden_substrings"] == [
        "Exception in ASGI application"
    ]
    assert summary["ignored_unattributed_forbidden_substrings"] == []


def test_runtime_log_ignores_unattributed_foreign_model_passthrough_503(
    monkeypatch,
):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            'Langfuse warning: {"model": "gpt-5.5"}\n'
            "pass_through_endpoint(): Exception occured - 503: b'upstream "
            "connect error or disconnect/reset before headers. reset reason: "
            "connection timeout'\n"
            "https://chatgpt.com/backend-api/codex/responses\n"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_spark",
        started="2026-06-06T21:45:11+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
        attribution_substrings=[
            "claude_adapter_spark",
            "gpt-5.3-codex-spark",
            "active-spark-session",
        ],
    )

    assert failures == []
    assert warnings == [
        "claude_adapter_spark ignored unattributed runtime log match "
        "`pass_through_endpoint(): Exception occured - 503:` from unrelated "
        "concurrent container traffic"
    ]
    assert summary["matched_forbidden_substrings"] == []
    assert summary["ignored_unattributed_forbidden_substrings"] == [
        "pass_through_endpoint(): Exception occured - 503:"
    ]


def test_runtime_log_keeps_current_model_passthrough_503_hard(monkeypatch):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            'Langfuse warning: {"model": "gpt-5.5"}\n'
            "pass_through_endpoint(): Exception occured - 503: b'upstream "
            "connect error or disconnect/reset before headers. reset reason: "
            "connection timeout'\n"
            "https://chatgpt.com/backend-api/codex/responses\n"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_gpt55",
        started="2026-06-06T21:45:11+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
        attribution_substrings=["claude_adapter_gpt55", "gpt-5.5"],
    )

    assert warnings == []
    assert failures == [
        "claude_adapter_gpt55 runtime logs contained forbidden substring "
        "`pass_through_endpoint(): Exception occured - 503:`"
    ]
    assert summary["matched_forbidden_substrings"] == [
        "pass_through_endpoint(): Exception occured - 503:"
    ]
    assert summary["ignored_unattributed_forbidden_substrings"] == []


def test_target_profile_sets_session_history_runtime_identity_expectations(monkeypatch):
    monkeypatch.setenv("AAWM_CLAUDE_HARNESS_USER_ID", "litellm-harness-test")
    harness = _load_harness_module()

    config = {
        "cases": {
            "claude_adapter_gpt55": {
                "env": {"ANTHROPIC_BASE_URL": "placeholder"},
                "session_history_validation": {"expected_provider": "openai"},
            }
        }
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    validation = updated["cases"]["claude_adapter_gpt55"][
        "session_history_validation"
    ]
    case_env = updated["cases"]["claude_adapter_gpt55"]["env"]
    assert validation["expected_provider"] == "openai"
    assert validation["expected_litellm_environment"] == "dev"
    assert validation["expected_tenant_id"] == "litellm"
    assert validation["require_runtime_identity"] is True
    assert validation["metadata_required_equals"]["tenant_id"] == "litellm"
    assert (
        validation["metadata_required_equals"]["aawm_original_tenant_id"]
        == "adapter-harness-tenant"
    )
    assert validation["metadata_required_equals"]["aawm_harness_tenant_alias"] is True
    assert updated["cases"]["claude_adapter_gpt55"]["require_trace_user_id"] is True
    assert updated["cases"]["claude_adapter_gpt55"]["expected_user_ids"] == [
        "adapter-harness-tenant"
    ]
    assert case_env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "x-litellm-end-user-id: adapter-harness-tenant\n"
        "langfuse_trace_user_id: adapter-harness-tenant\n"
        "langfuse_trace_name: claude-code\n"
        "x-aawm-tenant-id: adapter-harness-tenant"
    )
    assert "harness_run_id" in updated["cases"]["claude_adapter_gpt55"]


def test_target_profile_overrides_stale_case_level_harness_tenant_expectations():
    harness = _load_harness_module()

    config = {
        "default_tenant_id": "adapter-harness-tenant",
        "cases": {
            "claude_adapter_openai_output_config_effort": {
                "http_request": {"path": "/anthropic/v1/messages"},
                "session_history_validation": {
                    "expected_tenant_id": "adapter-harness-tenant",
                    "metadata_required_equals": {
                        "tenant_id": "adapter-harness-tenant"
                    },
                },
            }
        },
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    validation = updated["cases"]["claude_adapter_openai_output_config_effort"][
        "session_history_validation"
    ]
    assert validation["expected_tenant_id"] == "litellm"
    assert validation["metadata_required_equals"]["tenant_id"] == "litellm"
    assert (
        validation["metadata_required_equals"]["aawm_original_tenant_id"]
        == "adapter-harness-tenant"
    )
    assert validation["metadata_required_equals"]["aawm_harness_tenant_alias"] is True


def test_target_profile_can_skip_trace_environment_validation():
    harness = _load_harness_module()

    config = {
        "cases": {
            "native_openrouter_free_daily_meter_chat": {
                "skip_trace_environment_validation": True,
                "http_request": {
                    "path": "/chat/completions",
                    "json": {"model": "openrouter/openai/gpt-oss-20b:free"},
                },
                "session_history_validation": {"expected_provider": "openrouter"},
            }
        }
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    case_config = updated["cases"]["native_openrouter_free_daily_meter_chat"]
    validation = case_config["session_history_validation"]
    assert "expected_trace_environment" not in case_config
    assert validation["metadata_required_equals"]["litellm_environment"] == "dev"
    assert "trace_environment" not in validation["metadata_required_equals"]


def test_target_profile_does_not_force_tenant_expectations_when_trace_user_id_is_disabled():
    harness = _load_harness_module()

    config = {
        "cases": {
            "native_grok_cli_passthrough_grok_build": {
                "require_trace_user_id": False,
                "session_history_validation": {
                    "expected_provider": "xai",
                    "metadata_required_equals": {
                        "passthrough_route_family": "grok_cli_chat_proxy"
                    },
                },
            }
        }
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    validation = updated["cases"]["native_grok_cli_passthrough_grok_build"][
        "session_history_validation"
    ]

    assert "expected_tenant_id" not in validation
    assert "tenant_id" not in validation["metadata_required_equals"]
    assert "tenant_id_source" not in validation["metadata_required_truthy"]


def test_http_request_retry_uses_status_code_for_api_error_status(monkeypatch):
    harness = _load_harness_module()
    runs = [
        {
            "exit_code": 1,
            "stdout": json.dumps({"status_code": 429, "is_error": True}),
        },
        {
            "exit_code": 0,
            "stdout": json.dumps({"status_code": 200, "is_error": False}),
        },
    ]

    monkeypatch.setattr(
        harness,
        "_run_http_request_with_repeat",
        lambda _config: runs.pop(0),
    )

    _started, final_run, attempts = harness._run_command_with_retry(
        config={
            "http_request": {"path": "/chat/completions"},
            "retry_on_api_error_statuses": [429],
            "retry_max_attempts": 2,
        }
    )

    assert final_run["exit_code"] == 0
    assert [attempt["api_error_status"] for attempt in attempts] == [429, None]


def test_target_profile_codex_cli_uses_pytest_classifier_harness_user_id(monkeypatch):
    monkeypatch.delenv("AAWM_HARNESS_USER_ID", raising=False)
    monkeypatch.delenv("PYTEST_CLASSIFIER_HARNESS_USER_ID", raising=False)
    monkeypatch.delenv("AAWM_CLAUDE_HARNESS_USER_ID", raising=False)
    monkeypatch.setenv("AAWM_OBSERVE_SERVICE_NAME", "pytest-classifier-scan")
    harness = _load_harness_module()

    config = {
        "cases": {
            "native_openai_passthrough_responses_codex": {
                "cli_passthrough": "codex",
                "command": [
                    "codex",
                    "exec",
                    "-p",
                    "{codex_profile}",
                    "--json",
                    "Reply ok",
                ],
                "session_history_validation": {"expected_provider": "openai"},
            }
        }
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    case_config = updated["cases"]["native_openai_passthrough_responses_codex"]
    command = case_config["command"]
    config_values = [
        str(command[index + 1])
        for index, item in enumerate(command[:-1])
        if item == "-c"
    ]

    assert case_config["expected_user_ids"] == ["litellm"]
    assert case_config["expected_trace_session_id"] == "pytest-classifier.session"
    assert (
        'model_providers.litellm-dev.http_headers.x-litellm-end-user-id="pytest-classifier"'
        in config_values
    )
    assert (
        'model_providers.litellm-dev.http_headers.langfuse_trace_user_id="pytest-classifier"'
        in config_values
    )
    assert (
        'model_providers.litellm-dev.http_headers.langfuse_trace_name="native_openai_passthrough_responses_codex"'
        in config_values
    )
    assert (
        'model_providers.litellm-dev.http_headers.session_id="pytest-classifier.session"'
        in config_values
    )
    assert case_config["match_trace_session_id_from_stdout"] is False
    assert (
        'model_providers.litellm-dev.http_headers.x-aawm-tenant-id="adapter-harness-tenant"'
        in config_values
    )
    assert (
        'model_providers.litellm-dev.http_headers.x-aawm-repository="zepfu/litellm"'
        in config_values
    )
    session_history_validation = case_config["session_history_validation"]
    assert session_history_validation["expected_tenant_id"] == "zepfu/litellm"
    assert session_history_validation["metadata_required_equals"][
        "tenant_id"
    ] == "zepfu/litellm"
    assert session_history_validation["metadata_required_equals"][
        "aawm_original_tenant_id"
    ] == "litellm"
    assert (
        "aawm_harness_tenant_alias"
        not in session_history_validation["metadata_required_equals"]
    )


def test_target_profile_ignores_unknown_cli_passthrough():
    harness = _load_harness_module()

    config = {
        "cases": {
            "native_retired_passthrough_removed": {
                "cli_passthrough": "retired-provider",
                "command": ["retired-provider", "prompt"],
            }
        }
    }

    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    case_config = updated["cases"]["native_retired_passthrough_removed"]
    assert case_config["env"] == {}
    assert case_config["command"] == ["retired-provider", "prompt"]
    assert "expected_user_ids" not in case_config
    assert "expected_trace_session_id" not in case_config


def test_target_profile_formats_claude_case_harness_run_id(monkeypatch):
    harness = _load_harness_module()

    class FixedUuid:
        hex = "abcdef1234567890"

    monkeypatch.setattr(harness.uuid, "uuid4", lambda: FixedUuid())

    config = {
        "cases": {
            "claude_adapter_gpt55_child_sequential_core_tools": {
                "command": [
                    "claude",
                    "-p",
                    "write /tmp/probe-{harness_run_id}.txt for {case_name}",
                ],
                "env": {"ANTHROPIC_BASE_URL": "placeholder"},
                "session_history_validation": {"expected_provider": "openai"},
            }
        }
    }

    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    case_config = updated["cases"]["claude_adapter_gpt55_child_sequential_core_tools"]
    assert (
        case_config["harness_run_id"]
        == "claude_adapter_gpt55_child_sequential_core_tools-abcdef123456"
    )
    assert case_config["command"][2] == (
        "write /tmp/probe-claude_adapter_gpt55_child_sequential_core_tools-abcdef123456.txt "
        "for claude_adapter_gpt55_child_sequential_core_tools"
    )


def test_target_profile_adds_native_cli_repository_headers(monkeypatch):
    harness = _load_harness_module()
    monkeypatch.setattr(
        harness.RA,
        "_git_value",
        lambda *args: "https://github.com/zepfu/litellm.git"
        if args == ("remote", "get-url", "origin")
        else "",
    )

    config = {
        "cases": {
            "native_codex": {
                "cli_passthrough": "codex",
                "command": ["codex", "exec", "--json", "hello"],
            }
        }
    }

    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    codex_command = updated["cases"]["native_codex"]["command"]
    assert (
        'model_providers.litellm-dev.http_headers.x-aawm-repository="zepfu/litellm"'
        in codex_command
    )


def _assert_session_history_validation_loads_provider_record(
    case_config,
    *,
    expected_provider,
    expected_model,
    expected_client_name,
):
    session_history_validation = case_config["session_history_validation"]
    assert session_history_validation["expected_provider"] == expected_provider
    assert session_history_validation["expected_model"] == expected_model
    assert session_history_validation["expected_client_name"] == expected_client_name
    assert session_history_validation["metadata_required_equals"][
        "client_name"
    ] == expected_client_name
    assert "request_tags" in session_history_validation["metadata_required_truthy"]
    assert "client_version" in session_history_validation["metadata_required_truthy"]
    assert session_history_validation["required_contains"]["repository"] == "litellm"
    assert session_history_validation["minimums"]["input_tokens"] == 1
    assert session_history_validation["minimums"]["output_tokens"] == 1
    assert session_history_validation["minimums"]["total_tokens"] == 1


def _assert_codex_rate_limit_validation(case_config):
    assert "codex_response_headers" in case_config[
        "required_generation_metadata_truthy"
    ]
    rate_limit_checks = case_config["rate_limit_observations_validation"]
    assert rate_limit_checks["allow_latest_snapshot_fallback"] is True
    expected_rows = rate_limit_checks["expected_rows"]
    assert {
        (row["quota_key"], row["required_equals"]["quota_period"])
        for row in expected_rows
    } == {
        ("codex:primary", "five_hour"),
        ("codex:secondary", "seven_day"),
    }
    for row in expected_rows:
        assert row["provider"] == "openai"
        assert row["client"] == "codex"
        assert row["source"] == "codex_response_headers"
        assert row["quota_type"] == "tokens"
        assert row["minimums"]["remaining_pct"] == 0
        assert row["maximums"]["remaining_pct"] == 100
        assert "expected_reset_at" in row["required_future_timestamps"]
        assert "expected_reset_at" in row["required_timestamp_after_observed"]


def _assert_openrouter_free_daily_rate_limit_validation(case_config):
    rate_limit_checks = case_config["rate_limit_observations_validation"]
    assert rate_limit_checks["allow_latest_snapshot_fallback"] is True
    [expected_row] = rate_limit_checks["expected_rows"]
    assert expected_row["provider"] == "openrouter"
    assert expected_row["client"] == "openrouter"
    assert expected_row["source"] == "openrouter_free_daily_local_meter"
    assert (
        expected_row["quota_key"]
        == "openrouter_free_daily_requests:requests"
    )
    assert expected_row["quota_type"] == "requests"
    assert expected_row["required_equals"]["quota_period"] == "daily"
    assert expected_row["minimums"]["remaining_pct"] == 0
    assert expected_row["maximums"]["remaining_pct"] == 100
    assert "expected_reset_at" in expected_row["required_future_timestamps"]
    assert "expected_reset_at" in expected_row[
        "required_timestamp_after_observed"
    ]


def test_grok_cli_cases_validate_session_history_config_flags():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_false_flags = {
        "changed_pre_commit_config": False,
        "changed_env_file": False,
        "changed_pyproject_toml": False,
        "changed_gitignore": False,
    }

    for case_name, expected_model in (
        ("native_grok_cli_passthrough_grok_build", "grok-build"),
        ("native_grok_cli_passthrough_grok_composer", "grok-composer-2.5-fast"),
    ):
        case_config = config["cases"][case_name]
        validation = case_config["session_history_validation"]

        assert case_config["cli_passthrough"] == "grok"
        assert "AAWM_ENABLE_GROK_CLI_HARNESS" in case_config["required_env"]
        assert case_config["allowed_generation_routes"] == ["/grok/v1/responses"]
        assert validation["expected_provider"] == "xai"
        assert validation["expected_model"] == expected_model
        assert validation["expected_client_name"] == "grok-build"
        assert validation["client_user_agent_contains"] == "grok-shell/"
        assert validation["metadata_required_equals"][
            "passthrough_route_family"
        ] == "grok_cli_chat_proxy"
        assert "grok_cli_chat_proxy" not in validation["metadata_required_equals"]
        assert "grok_model_override" not in validation["metadata_required_equals"]
        assert validation["required_equals"] == expected_false_flags


def test_codex_tool_activity_parity_cases_have_stream_state_gates():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    native_case = config["cases"][
        "native_openai_passthrough_responses_codex_tool_activity"
    ]
    claude_case = config["cases"]["claude_adapter_codex_tool_activity"]

    assert (
        "native_openai_passthrough_responses_codex_tool_activity"
        in config["default_excluded_cases"]
    )
    assert native_case["cli_passthrough"] == "codex"
    assert native_case["match_trace_session_id_from_stdout"] is False
    assert "pwd" in native_case["command"][-1]
    assert "responses_stream_tool_call_count" in native_case[
        "required_generation_metadata_minimums"
    ]
    assert native_case["stream_tool_call_state_validation"]["expected_tools"][0][
        "tool_name"
    ] == "exec_command"
    assert native_case["stream_tool_call_state_validation"]["expected_tools"][0][
        "tool_type"
    ] == "function_call"
    assert native_case["tool_activity_validation"]["expected_rows"][0][
        "tool_name"
    ] == "exec_command"
    assert native_case["tool_activity_validation"]["expected_rows"][0][
        "command_text_contains"
    ] == "pwd"
    _assert_session_history_validation_loads_provider_record(
        native_case,
        expected_provider="openai",
        expected_model="gpt-5.4-mini",
        expected_client_name="codex_exec",
    )
    _assert_codex_rate_limit_validation(native_case)

    claude_stream_gate = claude_case["stream_tool_call_state_validation"]
    assert "anthropic-openai-codex-native-tools" in claude_case[
        "required_trace_tags"
    ]
    assert "anthropic_adapter_codex_native_tool_aliases" in claude_case[
        "required_generation_metadata_truthy"
    ]
    assert claude_stream_gate["expected_tools"][0]["tool_name"] == "exec_command"
    assert claude_stream_gate["expected_tools"][0]["tool_type"] == "function_call"
    assert claude_case["tool_activity_validation"]["expected_rows"][0][
        "tool_name"
    ] == "exec_command"
    assert "Bash" in claude_case["command"]
    assert "response.function_call_arguments.done" in claude_stream_gate[
        "required_any_event_type_groups"
    ][0]
    assert "responses_stream_tool_call_count" in claude_case[
        "required_generation_metadata_minimums"
    ]


def test_native_codex_case_hard_gates_spawn_agent_tool_description_patch():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"]["native_openai_passthrough_responses_codex"]

    _assert_session_history_validation_loads_provider_record(
        case_config,
        expected_provider="openai",
        expected_model="gpt-5.4-mini",
        expected_client_name="codex_exec",
    )
    _assert_codex_rate_limit_validation(case_config)
    assert case_config["match_trace_session_id_from_stdout"] is False
    assert "codex-tool-description-patch" in case_config["required_trace_tags"]
    assert (
        "codex-tool-description-patch:spawn-agent-fanout-policy"
        in case_config["required_trace_tags"]
    )
    assert "codex_tool_description_patch_count" in case_config[
        "required_generation_metadata_truthy"
    ]

    request_text_checks = case_config["request_text_checks"]
    assert "Use subagents to parallelize independent work" in request_text_checks[
        "required_substrings"
    ]
    assert "latest frontier model" in request_text_checks["required_substrings"]
    assert "latest Codex model" in request_text_checks["required_substrings"]
    assert "mini-class agents" in request_text_checks["required_substrings"]
    assert "Only use `spawn_agent` if and only if" in request_text_checks[
        "forbidden_substrings"
    ]
    assert "Only use spawn_agent if and only if" in request_text_checks[
        "forbidden_substrings"
    ]
    assert "GPT-5.5" not in request_text_checks["forbidden_substrings"]

    session_history_validation = case_config["session_history_validation"]
    assert session_history_validation["metadata_required_equals"][
        "prompt_overhead_breakdown_source"
    ] == "request_body_estimate"
    assert session_history_validation["metadata_required_equals"][
        "prompt_overhead_counted_shape"
    ] == "openai_responses"
    assert session_history_validation["metadata_required_equals"][
        "prompt_overhead_classifier_version"
    ] == "deterministic-v2"
    for metadata_key in (
        "prompt_overhead_component_paths",
        "usage_input_system_tokens_estimated",
        "usage_input_tool_advertisement_tokens_estimated",
        "usage_input_conversation_tokens_estimated",
    ):
        assert metadata_key in session_history_validation["metadata_required_truthy"]
    for column_name in (
        "input_system_tokens_estimated",
        "input_tool_advertisement_tokens_estimated",
        "input_conversation_tokens_estimated",
    ):
        assert session_history_validation["minimums"][column_name] == 1


def test_moonshot_codex_collaboration_case_uses_production_harness_contract():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][MOONSHOT_CODEX_COLLABORATION_CASE]

    assert MOONSHOT_CODEX_COLLABORATION_CASE in config["default_excluded_cases"]
    assert case_config["cli_passthrough"] == "codex"
    assert case_config["verification_alias"] == MS012_MOONSHOT_ALIAS
    assert case_config["match_trace_session_id_from_stdout"] is False

    command = case_config["command"]
    assert command[:2] == ["codex", "exec"]
    assert "--ignore-user-config" in command
    assert command.count("--enable") == 2
    assert "multi_agent" in command
    assert "multi_agent_v2" in command
    assert command[command.index("-m") + 1] == MS012_MOONSHOT_ALIAS
    assert (
        'model_provider="{codex_profile}"'
        in command
    )
    assert (
        'model_providers.{codex_profile}.base_url="{litellm_base_url}/openai_passthrough"'
        in command
    )
    assert (
        'model_catalog_json="{repository_root}/scripts/local-ci/codex_moonshot_model_catalog.json"'
        in command
    )
    assert (
        'agents.moonshot.description="Moonshot production acceptance worker"'
        in command
    )
    assert 'agents.moonshot.config_file="{codex_home}/agents/moonshot.toml"' in command
    prompt = command[-1]
    assert 'agent_type="moonshot"' in prompt
    assert 'model="aawm-sota-moonshot"' in prompt
    assert 'fork_turns="none"' in prompt
    assert "complete self-contained plaintext message" in prompt
    assert "do not include the legacy fork_context field" in prompt
    assert "Do not rely on inherited context" in prompt
    assert "prohibit further agent spawning" in prompt
    assert "fork_context=false" not in prompt
    assert "omit the fork_turns field entirely" not in prompt
    assert "do not set fork_turns to none" not in prompt

    output_checks = case_config["command_output_text_checks"]
    assert output_checks["minimum_chars"] == 9800
    assert output_checks["maximum_chars"] == 10200
    assert output_checks["required_prefix"] == "CODEX_MOONSHOT_PROD_ACCEPTANCE_START"
    assert output_checks["required_suffix"] == "CODEX_MOONSHOT_PROD_ACCEPTANCE_END"
    assert set(output_checks["required_substrings"]) == {
        "CHILD_A_TWO_PARALLEL_BATCHES_PASSED",
        "CHILD_B_TWO_PARALLEL_BATCHES_PASSED",
    }

    assert case_config["codex_collaboration_validation"]["minimum_tool_counts"] == {
        "wait": 1,
    }
    spawn_row, command_row = case_config["tool_activity_validation"]["expected_rows"]
    assert spawn_row == {
        "provider": "kimi_code",
        "tool_name": "spawn_agent",
        "tool_kind": "other",
        "minimum_count": 2,
        "each_arguments_required_substrings": [
            '"agent_type": "moonshot"',
            '"model": "aawm-sota-moonshot"',
            '"fork_turns": "none"',
            '"message": "',
        ],
    }
    assert command_row == {
        "provider": "kimi_code",
        "tool_name": "exec_command",
        "tool_kind": "command",
        "minimum_count": 12,
    }

    [session_row] = case_config["session_history_validation"]["expected_rows"]
    assert session_row["required_one_of"] == {
        "provider": ["kimi_code"],
        "model": ["k3-max", "k3-high"],
    }
    assert session_row["metadata_required_equals"] == {
        "model_alias_label": MS012_MOONSHOT_ALIAS,
        "requested_model_alias": MS012_MOONSHOT_ALIAS,
        "codex_auto_agent_alias": MS012_MOONSHOT_ALIAS,
    }


def test_moonshot_codex_bash_time_case_requires_exact_stdout_reporting():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][MOONSHOT_CODEX_BASH_TIME_CASE]

    assert MOONSHOT_CODEX_BASH_TIME_CASE in config["default_excluded_cases"]
    assert case_config["cli_passthrough"] == "codex"
    assert case_config["verification_alias"] == MS012_MOONSHOT_ALIAS
    assert case_config["verification_candidate_label"] == "codex-bash-system-time"
    assert case_config["match_trace_session_id_from_stdout"] is False

    command = case_config["command"]
    assert command[:2] == ["codex", "exec"]
    assert "--ignore-user-config" in command
    assert command[command.index("-m") + 1] == MS012_MOONSHOT_ALIAS
    assert "Use exec_command exactly once" in command[-1]
    assert "date --iso-8601=seconds" in command[-1]
    assert "return only the exact stdout" in command[-1]

    bash_validation = case_config["bash_stdout_report_validation"]
    assert bash_validation["expected_command"] == "date --iso-8601=seconds"
    assert "transcript_agent" not in bash_validation

    [tool_row] = case_config["tool_activity_validation"]["expected_rows"]
    assert tool_row == {
        "provider": "kimi_code",
        "tool_name": "exec_command",
        "tool_kind": "command",
        "minimum_count": 1,
        "maximum_count": 1,
        "command_text_contains": "date --iso-8601=seconds",
    }


def test_command_text_checks_validate_codex_agent_message_and_stderr():
    harness = _load_harness_module()
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "START child-a child-b END",
                    },
                }
            ),
        ]
    )

    output_text = harness._extract_command_output_text(stdout)
    summary, failures = harness._validate_command_text_checks(
        family="moonshot",
        text=output_text,
        label="command output",
        checks={
            "required_prefix": "START",
            "required_suffix": "END",
            "required_substrings": ["child-a", "child-b"],
            "forbidden_substrings": ["Traceback"],
            "minimum_chars": 10,
            "maximum_chars": 100,
        },
    )

    assert failures == []
    assert summary["length"] == len("START child-a child-b END")

    _, stderr_failures = harness._validate_command_text_checks(
        family="moonshot",
        text="failed to refresh available models",
        label="command stderr",
        checks={"forbidden_substrings": ["failed to refresh available models"]},
    )
    assert stderr_failures == [
        "moonshot command stderr contained forbidden substring 'failed to refresh available models'"
    ]


def test_codex_collaboration_validation_counts_completed_calls_only():
    harness = _load_harness_module()
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"type": "collab_tool_call", "tool": "spawn_agent"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "collab_tool_call", "tool": "spawn_agent"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "collab_tool_call", "tool": "spawn_agent"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "collab_tool_call", "tool": "wait"},
                }
            ),
        ]
    )

    summary, failures = harness._validate_codex_collaboration_events(
        family="moonshot",
        stdout=stdout,
        checks={"minimum_tool_counts": {"spawn_agent": 2, "wait": 1}},
    )

    assert failures == []
    assert summary["tool_counts"] == {"spawn_agent": 2, "wait": 1}


def test_default_suite_keeps_peeromega_fanout_and_native_anthropic_rate_limit_gate_opt_in():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    assert "claude_adapter_peeromega_fanout" in config["default_excluded_cases"]
    assert "native_anthropic_passthrough_claude" in config[
        "default_excluded_cases"
    ]

    native_case = config["cases"]["native_anthropic_passthrough_claude"]
    assert "anthropic_response_headers" in native_case[
        "required_generation_metadata_truthy"
    ]
    rate_limit_checks = native_case["rate_limit_observations_validation"]
    assert rate_limit_checks["allow_latest_snapshot_fallback"] is True
    quota_keys = {
        row["quota_key"] for row in rate_limit_checks["expected_rows"]
    }
    assert "anthropic_unified_5h:5h" in quota_keys
    assert "anthropic_unified_7d:7d" in quota_keys


def test_anthropic_adapter_config_removes_gemini_harness_cases():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    assert REMOVED_GEMINI_HARNESS_CASES.isdisjoint(config["cases"])
    assert REMOVED_GEMINI_HARNESS_CASES.isdisjoint(config["default_excluded_cases"])
    cases_without_d1322_antigravity_low = {
        name: case_config
        for name, case_config in config["cases"].items()
        if name not in D1322_LOW_ALIAS_REPLAY_CASES
    }
    serialized_cases = json.dumps(cases_without_d1322_antigravity_low).lower()
    serialized_excluded_cases = json.dumps(config["default_excluded_cases"]).lower()
    for forbidden_selector in ("gemini", "google/gemma", "google_code_assist"):
        assert forbidden_selector not in serialized_cases
        assert forbidden_selector not in serialized_excluded_cases


def test_active_anthropic_adapter_harness_surfaces_do_not_include_gemini_paths():
    violations = []
    for path in ACTIVE_ANTHROPIC_HARNESS_SURFACES:
        text = path.read_text(encoding="utf-8").lower()
        if path == ANTHROPIC_ADAPTER_CONFIG_PATH:
            config = json.loads(path.read_text(encoding="utf-8"))
            config["cases"] = {
                name: case_config
                for name, case_config in config["cases"].items()
                if name not in D1322_LOW_ALIAS_REPLAY_CASES
            }
            text = json.dumps(config).lower()
        for snippet in FORBIDDEN_ACTIVE_GEMINI_HARNESS_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(ROOT)} contains {snippet}")

    assert violations == []


def _collect_codex_case_model_selectors(case_config):
    model_selectors = []
    command = case_config.get("command")
    if isinstance(command, list):
        for index, item in enumerate(command[:-1]):
            if item in {"-m", "--model"} and isinstance(command[index + 1], str):
                model_selectors.append(("command", command[index + 1]))

    direct_paths = (
        ("model",),
        ("target_model",),
        ("http_request", "json", "model"),
        ("session_history_validation", "expected_model"),
        ("command_json_checks", "required_equals", "model"),
    )
    for path in direct_paths:
        current = case_config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if isinstance(current, str):
                model_selectors.append((".".join(path), current))

    tool_activity_validation = case_config.get("tool_activity_validation")
    if isinstance(tool_activity_validation, dict):
        for index, expected_row in enumerate(
            tool_activity_validation.get("expected_rows", [])
        ):
            if isinstance(expected_row, dict) and isinstance(
                expected_row.get("model"), str
            ):
                model_selectors.append(
                    (
                        f"tool_activity_validation.expected_rows[{index}].model",
                        expected_row["model"],
                    )
                )

    return model_selectors


def _collect_child_parallel_case_models(case_config):
    models = set()
    for row in case_config.get("session_history_validation", {}).get("expected_rows", []):
        if isinstance(row, dict) and isinstance(row.get("model"), str):
            models.add(row["model"])

    for _, agent_config in case_config.get("claude_agents", {}).items():
        if isinstance(agent_config, dict) and isinstance(
            agent_config.get("model"),
            str,
        ):
            models.add(agent_config["model"])
    return models


def test_codex_harness_cases_do_not_directly_select_anthropic_models():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    violations = []

    for case_name, case_config in config["cases"].items():
        if (
            case_config.get("cli_passthrough") != "codex"
            and "codex" not in case_name.lower()
        ):
            continue

        case_text = json.dumps(case_config).lower()
        antigravity_mediated = "antigravity" in case_name.lower() or (
            "antigravity" in case_text
        )
        for source, model in _collect_codex_case_model_selectors(case_config):
            if DIRECT_ANTHROPIC_MODEL_PATTERN.search(model) and not (
                antigravity_mediated and "antigravity" in model.lower()
            ):
                violations.append(f"{case_name}:{source}={model}")

    assert violations == []


def test_openrouter_free_cases_validate_daily_rate_limit_observations():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name in (
        "claude_adapter_openrouter_free",
        "native_openrouter_free_daily_meter_chat",
        "claude_adapter_gpt_oss_20b",
    ):
        assert case_name in config["default_excluded_cases"]
    for case_name in (
        "native_openrouter_free_daily_meter_chat",
        "claude_adapter_gpt_oss_20b",
    ):
        _assert_openrouter_free_daily_rate_limit_validation(
            config["cases"][case_name]
        )
    assert config["cases"]["native_openrouter_free_daily_meter_chat"][
        "skip_trace_environment_validation"
    ] is True


def test_nvidia_hosted_tool_policy_case_gates_dropped_hosted_tools():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_name = "claude_adapter_nvidia_hosted_tool_policy"
    case_config = config["cases"][case_name]

    assert case_name in config["default_excluded_cases"]
    assert (
        case_config["http_request"]["json"]["model"]
        == "nvidia/deepseek-ai/deepseek-v3.2"
    )
    assert case_config["allowed_generation_routes"] == [
        "/anthropic/v1/messages"
    ]
    assert "route:anthropic_nvidia_completion_adapter" in case_config[
        "required_trace_tags"
    ]
    assert "anthropic.nvidia_completion_adapter" in case_config[
        "required_span_names"
    ]

    request_tools = case_config["http_request"]["json"]["tools"]
    assert {tool["name"] for tool in request_tools} == {
        "web_search",
        "bash",
        "get_weather",
    }
    assert case_config["http_request"]["json"]["tool_choice"] == {
        "type": "tool",
        "name": "bash",
    }

    assert "request_payload_checks" not in case_config

    for metadata_key in (
        "anthropic_adapter_unsupported_hosted_tools",
        "anthropic_adapter_unsupported_hosted_tool_choice",
    ):
        assert metadata_key in case_config["required_generation_metadata_truthy"]

    metadata_required_equals = case_config["session_history_validation"][
        "metadata_required_equals"
    ]
    assert metadata_required_equals["anthropic_adapter_unsupported_hosted_tools"] == [
        {"type": "bash_20250124", "name": "bash"}
    ]
    assert metadata_required_equals[
        "anthropic_adapter_unsupported_hosted_tool_choice"
    ] == {"type": "tool", "name": "bash"}
    assert case_config["session_history_validation"]["metadata_required_truthy"] == [
        "tenant_id_source"
    ]


def test_target_profile_appends_case_local_claude_agents(monkeypatch):
    harness = _load_harness_module()

    class FixedUuid:
        hex = "abcdef1234567890"

    monkeypatch.setattr(harness.uuid, "uuid4", lambda: FixedUuid())

    config = {
        "cases": {
            "claude_adapter_gpt55_child_sequential_core_tools": {
                "command": [
                    "claude",
                    "-p",
                    "Dispatch to harness-gpt55-sequential-core-tools.",
                    "--allowedTools",
                    "Agent",
                ],
                "claude_agents": {
                    "harness-gpt55-sequential-core-tools": {
                        "model": "openai/gpt-5.5",
                        "tools": SEQUENTIAL_CORE_TOOLS,
                    }
                },
                "env": {"ANTHROPIC_BASE_URL": "placeholder"},
                "session_history_validation": {"expected_provider": "openai"},
            }
        }
    }

    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    command = updated["cases"]["claude_adapter_gpt55_child_sequential_core_tools"][
        "command"
    ]
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "--tools" not in command
    assert command.count("--agents") == 1
    agents = json.loads(command[command.index("--agents") + 1])
    assert agents["harness-gpt55-sequential-core-tools"]["tools"] == (
        SEQUENTIAL_CORE_TOOLS
    )
    assert not any(
        tool.startswith("mcp__aawm__")
        for tool in agents["harness-gpt55-sequential-core-tools"]["tools"]
    )


def test_sequential_core_tool_prompts_use_neutral_fixture_and_harness_agents():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name, agent_name in SEQUENTIAL_CASE_AGENTS.items():
        case_config = config["cases"][case_name]
        command = case_config["command"]
        prompt = command[2]
        assert "scripts/local-ci/sequential_core_tools_fixture.txt" in prompt
        assert "sequential-core-tools-grep" in prompt
        assert "Use Read to read /home/zepfu/projects/litellm/TODO.md" not in prompt
        assert f"Dispatch to the {agent_name} agent" in prompt
        assert "A final response immediately after Bash is invalid" in prompt
        assert "the child must call WebSearch next" in prompt
        assert "the child must call WebFetch next" in prompt
        assert command[command.index("--allowedTools") + 1] == "Agent"
        assert "--tools" not in command

        assert set(case_config["claude_agents"]) == {agent_name}
        agent_config = case_config["claude_agents"][agent_name]
        assert agent_config["tools"] == SEQUENTIAL_CORE_TOOLS
        assert not any(
            tool.startswith("mcp__aawm__") for tool in agent_config["tools"]
        )
        assert "This is not a repository investigation" in agent_config["prompt"]
        assert (
            "You must complete exactly eight tool calls before final text"
            in agent_config["prompt"]
        )
        assert (
            "Never answer, summarize, or report failure after Bash"
            in agent_config["prompt"]
        )
        assert (
            "The task is incomplete until WebFetch returns"
            in agent_config["prompt"]
        )
        assert f"claude-code.{agent_name}" in case_config["required_trace_names"]
        assert case_config["expected_trace_user_ids_by_name"][
            f"claude-code.{agent_name}"
        ] == "adapter-harness-tenant"
        assert (
            case_config["transcript_tool_use_validation"]["expected_agents"][0][
                "agent_type"
            ]
            == agent_name
        )


def _assert_required_payload_paths(case_config, expected_paths):
    required_paths = case_config["request_payload_checks"]["required_paths"]
    for path in expected_paths:
        assert path in required_paths


def _assert_parallel_read_common_case(
    *,
    config,
    case_name,
    case_config,
    agent_name,
    provider,
    model,
    durable_tool_names,
    transcript_mode="parallel",
):
    command = case_config["command"]
    prompt = command[2]
    assert case_name in config["default_excluded_cases"]
    assert f"Dispatch to the {agent_name} agent" in prompt
    assert "exactly three tool calls" in prompt
    if transcript_mode == "parallel":
        assert "must not wait for any tool result" in prompt
    else:
        assert transcript_mode == "sequential"
        assert "waiting for each tool result before issuing the next tool" in prompt
    assert "sequential_core_tools_fixture.txt" in prompt
    assert "sequential-core-tools-grep" in prompt
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "--tools" not in command

    assert set(case_config["claude_agents"]) == {agent_name}
    agent_config = case_config["claude_agents"][agent_name]
    assert agent_config["tools"] == PARALLEL_READ_TOOLS
    if transcript_mode == "parallel":
        assert "first assistant message must contain exactly three tool_use blocks" in (
            agent_config["prompt"]
        )
        assert "Do not wait for any tool result" in agent_config["prompt"]
    else:
        assert "waiting for each tool result before issuing the next tool" in (
            agent_config["prompt"]
        )
    assert f"claude-code.{agent_name}" in case_config["required_trace_names"]
    assert case_config["expected_trace_user_ids_by_name"][
        f"claude-code.{agent_name}"
    ] == "adapter-harness-tenant"

    transcript_agent = case_config["transcript_tool_use_validation"][
        "expected_agents"
    ][0]
    assert transcript_agent["agent_type"] == agent_name
    assert transcript_agent["expected_tool_counts"] == {
        "Read": 1,
        "Glob": 1,
        "Grep": 1,
    }
    if transcript_mode == "parallel":
        assert transcript_agent["minimum_tools_in_single_assistant_message"] == 3
        assert transcript_agent["maximum_tool_uses_per_assistant_message"] == 3
        assert "require_tool_result_before_next_tool_use" not in transcript_agent
    else:
        assert transcript_mode == "sequential"
        assert "minimum_tools_in_single_assistant_message" not in transcript_agent
        assert transcript_agent["maximum_tool_uses_per_assistant_message"] == 1
        assert transcript_agent["require_tool_result_before_next_tool_use"] is True

    if provider is None and model is None:
        durable_rows = [
            row
            for row in case_config["tool_activity_validation"]["expected_rows"]
            if row.get("tool_kind") == "read"
            and "provider" not in row
            and "model" not in row
        ]
    else:
        durable_rows = [
            row
            for row in case_config["tool_activity_validation"]["expected_rows"]
            if row.get("provider") == provider
            and row.get("model") == model
            and row.get("tool_kind") != "other"
        ]
    assert {row["tool_name"] for row in durable_rows} == durable_tool_names


def _assert_openai_parallel_read_case(case_config):
    assert "claude-tool-advertisement-compaction" in case_config[
        "required_trace_tags"
    ]
    assert "claude-prompt-patch" in case_config["required_trace_tags"]
    assert "openai-adapter-claude-context-compacted" in case_config[
        "required_trace_tags"
    ]
    assert "openai-adapter-claude-context:claude-md" in case_config[
        "required_trace_tags"
    ]
    assert "openai-adapter-parallel-instruction-policy" in case_config[
        "required_trace_tags"
    ]
    _assert_required_payload_paths(
        case_config,
        (
            "model",
            "input",
            "instructions",
            "reasoning.effort",
            "stream",
            "tools",
            "litellm_metadata.openai_adapter_claude_context_compacted",
            "litellm_metadata.openai_adapter_claude_context_compaction_events",
            "litellm_metadata.openai_adapter_parallel_instruction_policy_applied",
        ),
    )
    required_equals = case_config["request_payload_checks"]["required_equals"]
    assert required_equals["parallel_tool_calls"] is True
    assert required_equals[
        "litellm_metadata.openai_adapter_claude_context_compacted"
    ] is True
    assert required_equals[
        "litellm_metadata.openai_adapter_parallel_instruction_policy_applied"
    ] is True


def _assert_openrouter_parallel_read_case(case_config):
    assert "route:anthropic_openrouter_responses_adapter" in case_config[
        "required_trace_tags"
    ]
    assert "openrouter-adapter-claude-context-compacted" in case_config[
        "required_trace_tags"
    ]
    assert "openrouter-adapter-parallel-instruction-policy" in case_config[
        "required_trace_tags"
    ]
    _assert_required_payload_paths(
        case_config,
        (
            "model",
            "input",
            "instructions",
            "stream",
            "tools",
            "litellm_metadata.openrouter_adapter_claude_context_compacted",
            "litellm_metadata.openrouter_adapter_claude_context_compaction_events",
            "litellm_metadata.openrouter_adapter_parallel_instruction_policy_applied",
        ),
    )
    required_equals = case_config["request_payload_checks"]["required_equals"]
    assert required_equals["parallel_tool_calls"] is True
    assert required_equals[
        "litellm_metadata.openrouter_adapter_claude_context_compacted"
    ] is True
    assert required_equals[
        "litellm_metadata.openrouter_adapter_parallel_instruction_policy_applied"
    ] is True


def _assert_nvidia_parallel_read_case(case_config):
    assert "route:anthropic_nvidia_completion_adapter" in case_config[
        "required_trace_tags"
    ]
    _assert_required_payload_paths(
        case_config,
        (
            "model",
            "messages",
            "max_tokens",
            "tools",
        ),
    )


def test_parallel_read_tool_prompts_use_harness_agents_and_parallel_gate():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name, (
        agent_name,
        provider,
        model,
        durable_tool_names,
    ) in PARALLEL_CASE_AGENTS.items():
        case_config = config["cases"][case_name]
        _assert_parallel_read_common_case(
            config=config,
            case_name=case_name,
            case_config=case_config,
            agent_name=agent_name,
            provider=provider,
            model=model,
            durable_tool_names=durable_tool_names,
        )
        if provider == "openai":
            _assert_openai_parallel_read_case(case_config)
        elif provider == "openrouter":
            _assert_openrouter_parallel_read_case(case_config)
        elif provider == "nvidia_nim":
            _assert_nvidia_parallel_read_case(case_config)


def test_d1251_parallel_read_cases_cover_expected_aawm_anthropic_target_matrix():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["cases"].keys() >= D1251_PARALLEL_CASE_AGENTS.keys()

    for case_name, (
        agent_name,
        provider,
        model,
        durable_tool_names,
        child_model_selector,
    ) in D1251_PARALLEL_CASE_AGENTS.items():
        case_config = config["cases"][case_name]
        _assert_parallel_read_common_case(
            config=config,
            case_name=case_name,
            case_config=case_config,
            agent_name=agent_name,
            provider=provider,
            model=model,
            durable_tool_names=durable_tool_names,
            transcript_mode=(
                "sequential"
                if case_name in D1251_SEQUENTIAL_TRANSCRIPT_CASES
                else "parallel"
            ),
        )
        assert case_config["claude_agents"][agent_name]["model"] == (
            child_model_selector
        )
        command_json_checks = case_config["command_json_checks"]
        assert command_json_checks["required_equals"] == {"is_error": False}
        expected_result_substring = command_json_checks["required_contains"][
            "result"
        ]
        assert expected_result_substring in case_config["command"][2]
        if provider == "xai":
            required_trace_tags = D1251_REQUIRED_TRACE_TAGS_BY_CASE.get(
                case_name,
                D1251_REQUIRED_TRACE_TAGS["xai"],
            )
            disallowed_trace_tags = D1251_DISALLOWED_TRACE_TAGS_BY_CASE.get(
                case_name,
                set(),
            )
        else:
            required_trace_tags = D1251_REQUIRED_TRACE_TAGS[provider]
            disallowed_trace_tags = set()
        assert required_trace_tags.intersection(
            set(case_config["required_trace_tags"])
        ), f"{case_name} is missing a D1-251 route tag for {provider}"
        assert not set(case_config["required_trace_tags"]).intersection(
            disallowed_trace_tags
        ), f"{case_name} is using a disallowed D1-251 route tag"
        if provider == "openrouter":
            assert case_config["allow_zero_cost"] is True
        if provider == "xai":
            request_paths = set(case_config["request_payload_checks"]["required_paths"])
            assert {"model", "input", "stream", "tools"} <= request_paths
            assert "instructions" not in request_paths
        if case_name in D1251_OPENCODE_COMPLETION_CASES:
            required_tags = set(case_config["required_trace_tags"])
            assert "route:anthropic_opencode_zen_completion_adapter" in required_tags
            assert (
                "anthropic-adapter-target:opencode_zen:/v1/chat/completions"
                in required_tags
            )
            request_paths = set(case_config["request_payload_checks"]["required_paths"])
            assert {"model", "messages", "stream", "tools"} <= request_paths
            assert "input" not in request_paths
            assert "instructions" not in request_paths
        elif provider == "opencode_zen":
            required_tags = set(case_config["required_trace_tags"])
            assert "route:anthropic_opencode_zen_responses_adapter" in required_tags
            assert (
                "anthropic-adapter-target:opencode_zen:/v1/responses"
                in required_tags
            )




def test_d1322_low_alias_replay_cases_are_opt_in_and_do_not_require_staging_env():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name in D1322_LOW_ALIAS_REPLAY_CASES:
        assert case_name in config["cases"]
        assert case_name in config["default_excluded_cases"]
        assert "required_env" not in config["cases"][case_name]


def test_d1322_openrouter_completion_adapter_cases_are_opt_in_and_target_chat_completions():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name, expected in D1322_OPENROUTER_COMPLETION_CASES.items():
        assert case_name in config["cases"]
        assert case_name in config["default_excluded_cases"]
        case_config = config["cases"][case_name]
        assert case_config["verification_alias"] == "aawm-low-anthropic"
        assert case_config["verification_candidate_order"] == expected[
            "candidate_order"
        ]
        assert (
            case_config["verification_candidate_label"]
            == "direct-openrouter-completion-adapter"
        )
        assert (
            case_config["http_request"]["json"]["model"]
            == expected["requested_model"]
        )
        assert set(case_config["required_trace_tags"]) >= {
            "route:anthropic_messages",
            "route:anthropic_openrouter_completion_adapter",
            "anthropic-openrouter-completion-adapter",
            f"anthropic-adapter-model:{expected['adapter_model']}",
            "anthropic-adapter-target:openrouter:/v1/chat/completions",
        }
        session_history = case_config["session_history_validation"]
        assert session_history["expected_provider"] == "openrouter"
        assert session_history["expected_model"] == expected["requested_model"]
        assert session_history["metadata_required_equals"] == {
            "tenant_id": "adapter-harness-tenant",
            "anthropic_adapter_model": expected["adapter_model"],
            "anthropic_adapter_original_model": expected["requested_model"],
            "anthropic_adapter_target_endpoint": "openrouter:/v1/chat/completions",
            "passthrough_route_family": "anthropic_openrouter_completion_adapter",
        }
        assert case_config["allow_zero_cost"] is True
        assert case_config["skip_generation_quality_checks"] is True


def test_d1322_low_anthropic_alias_replay_case_uses_aawm_low_anthropic_child_model_and_declared_targets():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][D1322_LOW_ANTHROPIC_ALIAS_REPLAY_CASE]

    _assert_parallel_read_common_case(
        config=config,
        case_name=D1322_LOW_ANTHROPIC_ALIAS_REPLAY_CASE,
        case_config=case_config,
        agent_name=D1322_LOW_ANTHROPIC_ALIAS_REPLAY_AGENT,
        provider="antigravity",
        model="gemini-3.5-flash-low",
        durable_tool_names={"read_file", "glob", "grep_search"},
    )

    agent_config = case_config["claude_agents"][D1322_LOW_ANTHROPIC_ALIAS_REPLAY_AGENT]
    assert agent_config["model"] == "aawm-low-anthropic"
    assert case_config["verification_alias"] == "aawm-low-anthropic"
    assert case_config["verification_candidate_order"] == -1
    assert case_config["verification_candidate_label"] == "alias-replay"
    declared_candidates = case_config["verification_declared_candidates"]
    assert {
        (row["provider"], row["model"]) for row in declared_candidates
    } == D1322_AAWM_LOW_ANTHROPIC_DECLARED_PROVIDER_MODELS
    assert [row["candidate_order"] for row in declared_candidates] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert set(case_config["required_trace_tags"]) >= {
        "route:anthropic_messages",
        "model-alias:aawm-low-anthropic",
        "anthropic-auto-agent-alias:aawm-low-anthropic",
    }
    assert "passthrough_route_family" in case_config[
        "required_generation_metadata_truthy"
    ]
    assert case_config["allow_zero_cost"] is True

    session_expected_rows = case_config["session_history_validation"]["expected_rows"]
    assert len(session_expected_rows) == 1
    assert session_expected_rows[0]["metadata_required_equals"] == {
        "model_alias_label": "aawm-low-anthropic",
        "requested_model_alias": "aawm-low-anthropic",
        "anthropic_auto_agent_alias": "aawm-low-anthropic",
    }
    assert set(session_expected_rows[0]["metadata_required_truthy"]) == {
        "anthropic_auto_agent_selected_provider",
        "anthropic_auto_agent_selected_model",
        "anthropic_auto_agent_selected_route_family",
        "aawm_alias_routing_audit_events",
    }
    declared_provider_values = set(
        session_expected_rows[0]["required_one_of"]["provider"]
    )
    declared_model_values = set(session_expected_rows[0]["required_one_of"]["model"])
    assert declared_provider_values == {
        provider for provider, _ in D1322_AAWM_LOW_ANTHROPIC_DECLARED_PROVIDER_MODELS
    }
    assert declared_model_values == {
        model for _, model in D1322_AAWM_LOW_ANTHROPIC_DECLARED_PROVIDER_MODELS
    }


def test_d1322_codex_low_alias_replay_case_uses_aawm_low_and_declared_targets():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_name = D1322_CODEX_LOW_ALIAS_REPLAY_CASE
    case_config = config["cases"][case_name]

    assert case_name in config["default_excluded_cases"]
    assert case_config["cli_passthrough"] == "codex"
    assert case_config["verification_alias"] == "aawm-low"
    assert case_config["verification_candidate_order"] == -1
    assert case_config["verification_candidate_label"] == "alias-replay"
    declared_candidates = case_config["verification_declared_candidates"]
    assert {
        (row["provider"], row["model"]) for row in declared_candidates
    } == D1322_AAWM_LOW_CODEX_DECLARED_PROVIDER_MODELS
    assert [row["candidate_order"] for row in declared_candidates] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    command = case_config["command"]
    model_index = command.index("-m") + 1
    assert command[model_index] == "aawm-low"
    assert "pwd" in command[-1]
    assert case_config["match_trace_session_id_from_stdout"] is False
    assert set(case_config["required_trace_tags"]) == {
        "model-alias:aawm-low",
        "codex-auto-agent-alias:aawm-low",
    }
    assert case_config["required_generation_metadata_truthy"] == [
        "passthrough_route_family"
    ]
    assert case_config["allow_zero_cost"] is True

    session_expected_rows = case_config["session_history_validation"]["expected_rows"]
    assert len(session_expected_rows) == 1
    assert session_expected_rows[0]["metadata_required_equals"] == {
        "model_alias_label": "aawm-low",
        "requested_model_alias": "aawm-low",
        "codex_auto_agent_alias": "aawm-low",
    }
    assert set(session_expected_rows[0]["metadata_required_truthy"]) == {
        "codex_auto_agent_selected_provider",
        "codex_auto_agent_selected_model",
        "codex_auto_agent_selected_route_family",
        "aawm_alias_routing_audit_events",
    }
    assert session_expected_rows[0]["required_contains"] == {"repository": "litellm"}
    declared_provider_values = set(
        session_expected_rows[0]["required_one_of"]["provider"]
    )
    declared_model_values = set(session_expected_rows[0]["required_one_of"]["model"])
    assert declared_provider_values == {
        provider for provider, _ in D1322_AAWM_LOW_CODEX_DECLARED_PROVIDER_MODELS
    }
    assert declared_model_values == {
        model for _, model in D1322_AAWM_LOW_CODEX_DECLARED_PROVIDER_MODELS
    }

    tool_row = case_config["tool_activity_validation"]["expected_rows"][0]
    assert tool_row["tool_name"] == "exec_command"
    assert tool_row["command_text_contains"] == "pwd"
    assert set(tool_row["required_one_of"]["provider"]) == declared_provider_values
    assert set(tool_row["required_one_of"]["model"]) == declared_model_values

def test_d1256_alias_replay_case_uses_aawm_code_anthropic_child_model_and_declared_targets():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][D1256_ALIAS_REPLAY_CASE]

    _assert_parallel_read_common_case(
        config=config,
        case_name=D1256_ALIAS_REPLAY_CASE,
        case_config=case_config,
        agent_name=D1256_ALIAS_REPLAY_AGENT,
        provider=None,
        model=None,
        durable_tool_names={"Read", "Glob", "Grep"},
    )

    agent_config = case_config["claude_agents"][D1256_ALIAS_REPLAY_AGENT]
    assert agent_config["model"] == "aawm-code-anthropic"
    assert case_config["verification_alias"] == "aawm-code-anthropic"
    assert case_config["verification_candidate_order"] == -1
    assert case_config["verification_candidate_label"] == "alias-replay"
    declared_candidates = case_config["verification_declared_candidates"]
    assert {
        (row["provider"], row["model"]) for row in declared_candidates
    } == D1256_AAWM_CODE_ANTHROPIC_DECLARED_PROVIDER_MODELS
    assert [row["candidate_order"] for row in declared_candidates] == [0, 1, 2, 3, 4]
    assert "route:anthropic_messages" in case_config["required_trace_tags"]
    assert "passthrough_route_family" in case_config[
        "required_generation_metadata_truthy"
    ]
    assert set(case_config["request_payload_checks"]["required_paths"]) == {
        "model",
        "messages",
        "max_tokens",
        "tools",
    }

    session_expected_rows = case_config["session_history_validation"][
        "expected_rows"
    ]
    assert len(session_expected_rows) == 1
    declared_provider_values = set(
        session_expected_rows[0]["required_one_of"]["provider"]
    )
    declared_model_values = set(
        session_expected_rows[0]["required_one_of"]["model"]
    )
    assert session_expected_rows[0]["metadata_required_equals"] == {
        "model_alias_label": "aawm-code-anthropic",
        "requested_model_alias": "aawm-code-anthropic",
        "anthropic_auto_agent_alias": "aawm-code-anthropic",
    }
    assert set(session_expected_rows[0]["metadata_required_truthy"]) == {
        "anthropic_auto_agent_selected_provider",
        "anthropic_auto_agent_selected_model",
        "anthropic_auto_agent_selected_route_family",
        "aawm_alias_routing_audit_events",
    }
    assert declared_provider_values == {
        provider
        for provider, _ in D1256_AAWM_CODE_ANTHROPIC_DECLARED_PROVIDER_MODELS
    }
    assert declared_model_values == {
        model for _, model in D1256_AAWM_CODE_ANTHROPIC_DECLARED_PROVIDER_MODELS
    }

    tool_rows = case_config["tool_activity_validation"]["expected_rows"]
    durable_rows = [
        row for row in tool_rows if row.get("tool_kind") == "read"
    ]
    assert {row["tool_name"] for row in durable_rows} == {"Read", "Glob", "Grep"}
    assert all("provider" not in row for row in durable_rows)
    assert all("model" not in row for row in durable_rows)
    assert all(row["maximum_count"] == 1 for row in durable_rows)
    assert all(row["minimum_count"] == 1 for row in durable_rows)


def test_ms012_moonshot_case_uses_the_canonical_alias_and_agentic_contract():
    harness = _load_harness_module()
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][MS012_MOONSHOT_AGENTIC_CASE]

    assert MS012_MOONSHOT_AGENTIC_CASE in config["default_excluded_cases"]
    assert case_config["moonshot_anthropic_agentic_only"] is True
    assert case_config["verification_alias"] == MS012_MOONSHOT_ALIAS
    assert case_config["verification_candidate_order"] == -1
    assert case_config["verification_candidate_label"] == "agentic-tool-continuation"
    assert case_config["allowed_generation_routes"] == ["/anthropic/v1/messages"]
    assert set(case_config["claude_agents"]) == {MS012_MOONSHOT_AGENT_PROFILE}

    command = case_config["command"]
    assert command[0] == "claude"
    assert "--model" not in command
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "Dispatch to the sota-moonshot agent" in command[2]
    assert "After the Read tool result" in command[2]
    assert "After the Grep tool result" in command[2]

    agent = case_config["claude_agents"][MS012_MOONSHOT_AGENT_PROFILE]
    assert agent["model"] == MS012_MOONSHOT_ALIAS
    assert agent["tools"] == ["Read", "Grep"]
    assert "After the Read tool result" in agent["prompt"]
    assert "After the Grep tool result" in agent["prompt"]

    assert {
        (row["provider"], row["model"], row["route_family"]) for row in case_config["verification_declared_candidates"]
    } == {("kimi_code", model, MS012_MOONSHOT_ADAPTER_PATH) for model in MS012_MOONSHOT_DECLARED_MODELS}
    assert set(case_config["required_trace_tags"]) >= {
        "route:anthropic_messages",
        f"route:{MS012_MOONSHOT_ADAPTER_PATH}",
        "anthropic-kimi-chat-completions-adapter",
        f"model-alias:{MS012_MOONSHOT_ALIAS}",
        f"anthropic-auto-agent-alias:{MS012_MOONSHOT_ALIAS}",
    }
    assert f"claude-code.{MS012_MOONSHOT_AGENT_PROFILE}" in case_config["required_trace_names"]

    transcript_agent = case_config["transcript_tool_use_validation"]["expected_agents"][0]
    assert transcript_agent == {
        "agent_type": MS012_MOONSHOT_AGENT_PROFILE,
        "expected_tool_counts": {"Read": 1, "Grep": 1},
        "expected_tool_sequence": ["Read", "Grep"],
        "minimum_total_tool_uses": 2,
        "maximum_total_tool_uses": 2,
        "maximum_tool_uses_per_assistant_message": 1,
        "require_tool_result_before_next_tool_use": True,
        "forbid_tool_result_errors": True,
    }
    session_row = case_config["session_history_validation"]["expected_rows"][0]
    assert session_row["required_one_of"] == {
        "provider": ["kimi_code"],
        "model": ["k3-max", "k3-high"],
    }
    assert session_row["metadata_required_equals"] == {
        "model_alias_label": MS012_MOONSHOT_ALIAS,
        "requested_model_alias": MS012_MOONSHOT_ALIAS,
        "anthropic_auto_agent_alias": MS012_MOONSHOT_ALIAS,
    }

    summary, failures = harness._validate_moonshot_anthropic_agentic_contract(
        family=MS012_MOONSHOT_AGENTIC_CASE,
        config=case_config,
    )

    assert failures == []
    assert summary == {
        "adapter_path": MS012_MOONSHOT_ADAPTER_PATH,
        "canonical_alias": MS012_MOONSHOT_ALIAS,
        "agent_profile": MS012_MOONSHOT_AGENT_PROFILE,
        "allowed_generation_routes": ["/anthropic/v1/messages"],
        "declared_models": sorted(MS012_MOONSHOT_DECLARED_MODELS),
    }
    assert "aawm-sota-moonshot-anthropic" not in json.dumps(config)


def test_ms012_moonshot_claude_stress_case_requires_two_parallel_child_batches():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][MS012_MOONSHOT_STRESS_CASE]

    assert MS012_MOONSHOT_STRESS_CASE in config["default_excluded_cases"]
    assert case_config["verification_alias"] == MS012_MOONSHOT_ALIAS
    assert case_config["verification_candidate_label"] == "claude-parallel-stress"
    assert case_config["allowed_generation_routes"] == ["/anthropic/v1/messages"]

    command = case_config["command"]
    assert command[0] == "claude"
    assert "--model" not in command
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "exactly two Agent tool calls" in command[2]
    assert "exactly two sequential batches of three parallel" in command[2]
    assert "between 9800 and 10200 characters" in command[2]

    expected_agents = {
        "sota-moonshot-stress-a",
        "sota-moonshot-stress-b",
    }
    assert set(case_config["claude_agents"]) == expected_agents
    for agent_name, agent in case_config["claude_agents"].items():
        assert agent["model"] == MS012_MOONSHOT_ALIAS
        assert agent["tools"] == ["Read", "Glob", "Grep"]
        assert "exactly two sequential parallel batches" in agent["prompt"]
        child_label = "A" if agent_name.endswith("-a") else "B"
        assert f"child {child_label}" in agent["prompt"]

    output_checks = case_config["command_output_text_checks"]
    assert output_checks["minimum_chars"] == 9800
    assert output_checks["maximum_chars"] == 10200
    assert output_checks["required_prefix"] == "CLAUDE_MOONSHOT_PROD_ACCEPTANCE_START"
    assert output_checks["required_suffix"] == "CLAUDE_MOONSHOT_PROD_ACCEPTANCE_END"
    assert set(output_checks["required_substrings"]) == {
        "CHILD_A_TWO_PARALLEL_BATCHES_PASSED",
        "CHILD_B_TWO_PARALLEL_BATCHES_PASSED",
    }

    transcript_agents = case_config["transcript_tool_use_validation"]["expected_agents"]
    assert {agent["agent_type"] for agent in transcript_agents} == expected_agents
    for agent in transcript_agents:
        assert agent["expected_tool_counts"] == {
            "Read": 2,
            "Glob": 2,
            "Grep": 2,
        }
        assert agent["minimum_total_tool_uses"] == 6
        assert agent["maximum_total_tool_uses"] == 6
        assert agent["minimum_parallel_tool_batches"] == 2
        assert agent["minimum_tools_per_parallel_batch"] == 3
        assert agent["maximum_tool_uses_per_assistant_message"] == 3

    tool_rows = case_config["tool_activity_validation"]["expected_rows"]
    assert tool_rows[0]["tool_name"] == "Agent"
    assert tool_rows[0]["minimum_count"] == 2
    assert {
        row["tool_name"]: (row["minimum_count"], row["maximum_count"])
        for row in tool_rows[1:]
    } == {
        "Read": (4, 4),
        "Glob": (4, 4),
        "Grep": (4, 4),
    }


def test_ms012_moonshot_claude_bash_time_case_requires_exact_stdout_reporting():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    case_config = config["cases"][MS012_MOONSHOT_BASH_TIME_CASE]

    assert MS012_MOONSHOT_BASH_TIME_CASE in config["default_excluded_cases"]
    assert case_config["verification_alias"] == MS012_MOONSHOT_ALIAS
    assert case_config["verification_candidate_label"] == "claude-bash-system-time"

    command = case_config["command"]
    assert command[0] == "claude"
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "Dispatch exactly one sota-moonshot-time child" in command[2]
    assert "date --iso-8601=seconds" in command[2]
    assert "return only the exact child text" in command[2]

    assert set(case_config["claude_agents"]) == {
        MS012_MOONSHOT_TIME_AGENT_PROFILE
    }
    agent = case_config["claude_agents"][MS012_MOONSHOT_TIME_AGENT_PROFILE]
    assert agent["model"] == MS012_MOONSHOT_ALIAS
    assert agent["tools"] == ["Bash"]
    assert "return only the exact command stdout" in agent["prompt"]

    bash_validation = case_config["bash_stdout_report_validation"]
    assert bash_validation == {
        "expected_command": "date --iso-8601=seconds",
        "expected_regex": (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:Z|[+-]\d{2}:\d{2})$"
        ),
        "transcript_agent": MS012_MOONSHOT_TIME_AGENT_PROFILE,
    }
    [transcript_agent] = case_config["transcript_tool_use_validation"][
        "expected_agents"
    ]
    assert transcript_agent["expected_tool_counts"] == {"Bash": 1}
    assert transcript_agent["expected_tool_sequence"] == ["Bash"]
    assert transcript_agent["maximum_total_tool_uses"] == 1


def test_ms012_moonshot_contract_rejects_a_raw_smoke_substitution():
    harness = _load_harness_module()
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_smoke = json.loads(json.dumps(config["cases"][MS012_MOONSHOT_AGENTIC_CASE]))
    raw_smoke["command"] = [
        "claude",
        "-p",
        "Reply with exactly two words: smoke check",
        "--output-format",
        "json",
        "--model",
        MS012_MOONSHOT_ALIAS,
        "--allowedTools",
        "",
    ]
    raw_smoke.pop("claude_agents")
    raw_smoke.pop("transcript_tool_use_validation")

    _, failures = harness._validate_moonshot_anthropic_agentic_contract(
        family=MS012_MOONSHOT_AGENTIC_CASE,
        config=raw_smoke,
    )

    assert any("must not use a direct --model selector" in failure for failure in failures)
    assert any("must require top-level Agent tool dispatch" in failure for failure in failures)
    assert any("must define exactly the sota-moonshot child profile" in failure for failure in failures)
    assert any("must validate the sota-moonshot child transcript" in failure for failure in failures)


def test_ms012_moonshot_transcript_requires_tool_result_continuation(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-moonshot.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": MS012_MOONSHOT_AGENT_PROFILE}),
        encoding="utf-8",
    )
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "moonshot",
                        "message": {
                            "id": "msg-read",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-read",
                                    "name": "Read",
                                    "input": {"file_path": ("scripts/local-ci/" "sequential_core_tools_fixture.txt")},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "agentId": "moonshot",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-read",
                                    "content": "fixture text",
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "moonshot",
                        "message": {
                            "id": "msg-grep",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-grep",
                                    "name": "Grep",
                                    "input": {
                                        "pattern": "sequential-core-tools-grep",
                                        "path": ("scripts/local-ci/" "sequential_core_tools_fixture.txt"),
                                        "output_mode": "content",
                                    },
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "agentId": "moonshot",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-grep",
                                    "content": "sequential-core-tools-grep",
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "moonshot",
                        "message": {
                            "id": "msg-final",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ("MOONSHOT ANTHROPIC AGENTIC TOOL " "CONTINUATION PASSED"),
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary, failures = harness._validate_transcript_tool_use(
        family=MS012_MOONSHOT_AGENTIC_CASE,
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": MS012_MOONSHOT_AGENT_PROFILE,
                    "expected_tool_counts": {"Read": 1, "Grep": 1},
                    "expected_tool_sequence": ["Read", "Grep"],
                    "minimum_total_tool_uses": 2,
                    "maximum_total_tool_uses": 2,
                    "maximum_tool_uses_per_assistant_message": 1,
                    "require_tool_result_before_next_tool_use": True,
                    "forbid_tool_result_errors": True,
                }
            ],
        },
    )

    assert failures == []
    records = summary["agents"][0]["records"]
    assert [record["tool_name"] for record in records] == ["Read", "Grep"]
    assert all(record["tool_result_line"] for record in records)


def test_d1251_parallel_read_cases_do_not_include_disallowed_gemini_models():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name, (_, _, persisted_model, _, child_model_selector) in (
        D1251_PARALLEL_CASE_AGENTS.items()
    ):
        for model in (persisted_model, child_model_selector):
            assert not _is_forbidden_d1251_child_parallel_model(model), (
                f"{case_name} uses forbidden Gemini/Google model {model}"
            )

    for case_name, case_config in config["cases"].items():
        if case_name.endswith("_child_parallel_read_tools"):
            for model in _collect_child_parallel_case_models(case_config):
                assert not _is_forbidden_d1251_child_parallel_model(model), (
                    f"{case_name} uses forbidden Gemini/Google model {model}"
                )


def test_claude_command_uses_settings_overlay_for_harness_headers(monkeypatch):
    harness = _load_harness_module()
    captured = {}

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(command, **kwargs):
        settings_path = pathlib.Path(command[command.index("--settings") + 1])
        captured["command"] = command
        captured["settings_path"] = settings_path
        captured["settings"] = json.loads(settings_path.read_text())
        captured["exists_during_run"] = settings_path.exists()
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(harness.RA.subprocess, "run", fake_run)

    result = harness.RA._run_command(
        ["claude", "-p", "hi"],
        extra_env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4001/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": (
                "x-litellm-end-user-id: litellm-harness-test\n"
                "langfuse_trace_user_id: litellm-harness-test\n"
                "langfuse_trace_name: claude-code"
            ),
            "AAWM_DB_PASSWORD": "not-written-to-settings",
        },
    )

    assert "--settings" in captured["command"]
    assert captured["exists_during_run"] is True
    assert captured["settings"] == {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4001/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": (
                "x-litellm-end-user-id: litellm-harness-test\n"
                "langfuse_trace_user_id: litellm-harness-test\n"
                "langfuse_trace_name: claude-code"
            ),
        }
    }
    assert "AAWM_DB_PASSWORD" not in captured["settings"]["env"]
    # RR-080 child-env scrubbing intentionally drops AAWM_DB_* secrets from the
    # subprocess env; only Anthropic overlay keys land in --settings JSON.
    assert "AAWM_DB_PASSWORD" not in captured["env"]
    assert not captured["settings_path"].exists()
    assert result["command"] == captured["command"]


def test_trace_user_id_validation_can_require_child_trace_users():
    harness = _load_harness_module()

    summary, failures = harness._validate_trace_user_ids_by_name(
        family="mixed fanout",
        traces=[
            {
                "name": "claude-code.gpt-5-4",
                "userId": "adapter-harness-tenant",
            },
            {
                "name": "claude-code.gpt-5-3-codex-spark",
                "userId": "wrong-user",
            },
        ],
        expected={
            "claude-code.gpt-5-4": "adapter-harness-tenant",
            "claude-code.gpt-5-3-codex-spark": "adapter-harness-tenant",
        },
    )

    assert summary["actual_by_name"]["claude-code.gpt-5-4"] == [
        "adapter-harness-tenant"
    ]
    assert failures == [
        "mixed fanout trace claude-code.gpt-5-3-codex-spark missing user id "
        "adapter-harness-tenant"
    ]


def test_trace_lookup_uses_expected_user_when_trace_name_user_checks_are_configured():
    harness = _load_harness_module()

    assert (
        harness._resolve_trace_lookup_user_id(
            ["adapter-harness-tenant"],
            {
                "claude-code.orchestrator": "adapter-harness-tenant",
                "claude-code.gpt-5-5": "adapter-harness-tenant",
            },
        )
        == "adapter-harness-tenant"
    )


def test_trace_lookup_derives_single_user_from_trace_name_user_checks():
    harness = _load_harness_module()

    assert (
        harness._resolve_trace_lookup_user_id(
            [],
            {
                "claude-code.orchestrator": "adapter-harness-tenant",
                "claude-code.gpt-5-5": "adapter-harness-tenant",
            },
        )
        == "adapter-harness-tenant"
    )


def test_trace_lookup_does_not_guess_when_trace_name_users_differ():
    harness = _load_harness_module()

    assert (
        harness._resolve_trace_lookup_user_id(
            [],
            {
                "claude-code.orchestrator": "tenant-a",
                "claude-code.gpt-5-5": "tenant-b",
            },
        )
        is None
    )


def test_http_request_repeat_runs_same_request_and_reports_each_pass(monkeypatch):
    harness = _load_harness_module()
    calls = []

    def fake_run_http_request(config):
        calls.append(config)
        pass_number = len(calls)
        return {
            "command": ["POST", "http://127.0.0.1:4001/v1/chat/completions"],
            "command_string": "POST http://127.0.0.1:4001/v1/chat/completions",
            "exit_code": 0,
            "duration_seconds": 0.1,
            "stdout": json.dumps(
                {
                    "session_id": "shared-session",
                    "status_code": 200,
                    "is_error": False,
                    "pass_seen": pass_number,
                }
            ),
            "stderr": "",
            "response_excerpt": "{}",
        }

    monkeypatch.setattr(harness, "_run_http_request", fake_run_http_request)

    result = harness._run_http_request_with_repeat(
        {
            "repeat_http_request": True,
            "http_request": {
                "path": "/v1/chat/completions",
                "session_id": "shared-session",
            },
        }
    )

    stdout = json.loads(result["stdout"])
    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert result["exit_code"] == 0
    assert result["http_request_repeat_count"] == 2
    assert stdout["session_id"] == "shared-session"
    assert stdout["http_request_repeat_count"] == 2
    assert [entry["stdout"]["pass_seen"] for entry in stdout["http_request_passes"]] == [
        1,
        2,
    ]


def test_http_request_repeat_count_can_be_explicit_and_preserves_single_default():
    harness = _load_harness_module()

    assert harness._http_request_repeat_count({"http_request": {}}) == 1
    assert (
        harness._http_request_repeat_count(
            {"http_request": {"http_request_repeat_count": 3}}
        )
        == 3
    )
    assert (
        harness._http_request_repeat_count(
            {"repeat_http_request": True, "http_request": {}}
        )
        == 2
    )


def test_repeat_text_fixture_expands_recursively():
    harness = _load_harness_module()

    expanded = harness._expand_repeat_text_fixtures(
        {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "repeat_text": "cacheable prefix",
                        "count": 3,
                        "separator": "\n",
                    },
                }
            ],
            "metadata": {"unchanged": {"repeat_text": "x"}},
        }
    )

    assert expanded["messages"][0]["content"] == (
        "cacheable prefix\ncacheable prefix\ncacheable prefix"
    )
    assert expanded["metadata"]["unchanged"] == {"repeat_text": "x"}


def test_http_header_env_placeholders_expand(monkeypatch):
    harness = _load_harness_module()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    assert harness._expand_env_placeholders("Bearer $OPENAI_API_KEY") == (
        "Bearer sk-test-key"
    )


def test_session_history_expected_rows_can_require_multiple_shared_session_rows():
    harness = _load_harness_module()

    records = [
        {
            "provider": "openai",
            "model": "gpt-5-mini",
            "tenant_id": "adapter-harness-tenant",
            "cache_read_input_tokens": 0,
            "input_system_tokens_estimated": 10,
            "input_tool_advertisement_tokens_estimated": 20,
            "input_conversation_tokens_estimated": 5,
            "metadata": {
                "prompt_overhead_counted_shape": "openai_responses",
                "prompt_overhead_component_paths": {"system": ["instructions"]},
            },
        },
        {
            "provider": "openai",
            "model": "gpt-5-mini",
            "tenant_id": "adapter-harness-tenant",
            "cache_read_input_tokens": 2048,
            "input_system_tokens_estimated": 12,
            "input_tool_advertisement_tokens_estimated": 30,
            "input_conversation_tokens_estimated": 8,
            "metadata": {
                "prompt_overhead_counted_shape": "openai_responses",
                "prompt_overhead_component_paths": {"system": ["instructions"]},
            },
        },
    ]

    matched_records, failures = harness._match_session_history_expected_rows(
        family="openai_prompt_cache",
        records=records,
        expected_rows=[
            {
                "provider": "openai",
                "model": "gpt-5-mini",
                "required_equals": {"tenant_id": "adapter-harness-tenant"},
                "minimums": {
                    "input_system_tokens_estimated": 1,
                    "input_tool_advertisement_tokens_estimated": 1,
                    "input_conversation_tokens_estimated": 1,
                },
                "metadata_required_equals": {
                    "prompt_overhead_counted_shape": "openai_responses",
                },
                "metadata_required_truthy": ["prompt_overhead_component_paths"],
                "minimum_count": 2,
            }
        ],
    )

    assert failures == []
    assert len(matched_records) == 2


def test_prompt_overhead_cost_share_report_groups_estimated_rows():
    harness = _load_harness_module()

    report = harness._build_prompt_overhead_cost_share_report(
        {
            "native_codex": {
                "session_history": {
                    "records": [
                        {
                            "client_name": "codex_exec",
                            "litellm_environment": "dev",
                            "provider": "openai",
                            "model": "gpt-5.4-mini",
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                            "response_cost_usd": 1.0,
                            "input_system_tokens_estimated": 20,
                            "input_tool_advertisement_tokens_estimated": 30,
                            "input_conversation_tokens_estimated": 40,
                            "input_other_tokens_estimated": 10,
                            "input_breakdown_residual_tokens": 10,
                            "system_behavior_tokens_estimated": 12,
                            "system_safety_tokens_estimated": 3,
                            "system_instructional_tokens_estimated": 4,
                            "system_unclassified_tokens_estimated": 1,
                            "metadata": {
                                "prompt_overhead_breakdown_source": (
                                    "request_body_estimate"
                                ),
                                "prompt_overhead_route_family": "codex_responses",
                                "prompt_overhead_counted_shape": (
                                    "openai_responses"
                                ),
                            },
                        }
                    ]
                }
            }
        }
    )

    assert report["totals"]["calls"] == 1
    assert report["totals"]["estimated_calls"] == 1
    assert report["totals"]["explicit_prompt_overhead_tokens_estimated"] == 50
    assert report["totals"]["prompt_overhead_plus_other_tokens_estimated"] == 60
    assert report["totals"]["explicit_prompt_overhead_input_share"] == 0.5
    assert report["totals"]["prompt_overhead_plus_other_input_share"] == 0.6
    assert report["totals"]["explicit_prompt_overhead_cost_usd_estimated"] == 0.5
    assert report["totals"]["prompt_overhead_plus_other_cost_usd_estimated"] == 0.6

    group = report["groups"][0]
    assert group["case_name"] == "native_codex"
    assert group["client_name"] == "codex_exec"
    assert group["route_family"] == "codex_responses"
    assert group["counted_shape"] == "openai_responses"
    assert group["provider"] == "openai"
    assert group["model"] == "gpt-5.4-mini"
    assert group["system_unclassified_tokens_estimated"] == 1


def test_prompt_overhead_cost_share_report_tracks_unestimated_rows():
    harness = _load_harness_module()

    report = harness._build_prompt_overhead_cost_share_report(
        {
            "native_anthropic": {
                "session_history": {
                    "all_records": [
                        {
                            "client_name": "claude-cli",
                            "litellm_environment": "dev",
                            "provider": "anthropic",
                            "model": "claude-opus-4-6",
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                            "response_cost_usd": 2.0,
                            "metadata": {"passthrough_route_family": "anthropic"},
                        },
                        {
                            "client_name": "claude-cli",
                            "litellm_environment": "dev",
                            "provider": "anthropic",
                            "model": "claude-opus-4-6",
                            "input_tokens": 0,
                            "output_tokens": 1,
                            "total_tokens": 1,
                            "response_cost_usd": 0.1,
                            "metadata": {
                                "prompt_overhead_breakdown_source": (
                                    "request_body_estimate"
                                ),
                                "prompt_overhead_route_family": "anthropic",
                                "prompt_overhead_counted_shape": (
                                    "anthropic_messages_semantic"
                                ),
                            },
                        },
                    ],
                    "records": [],
                }
            }
        }
    )

    assert report["totals"]["calls"] == 2
    assert report["totals"]["estimated_calls"] == 1
    assert report["totals"]["unestimated_calls"] == 1
    assert report["totals"]["input_tokens_with_breakdown"] == 0
    assert report["totals"]["breakdown_input_token_coverage_share"] == 0.0
    assert report["totals"]["explicit_prompt_overhead_input_share"] is None
    assert len(report["groups"]) == 2


def test_prompt_overhead_cost_share_report_prefers_selected_record_for_shared_sessions():
    harness = _load_harness_module()

    report = harness._build_prompt_overhead_cost_share_report(
        {
            "native_openai_responses": {
                "session_history": {
                    "record": {
                        "client_name": "openai-python",
                        "provider": "openai",
                        "model": "gpt-5.4-mini",
                        "input_tokens": 10,
                        "response_cost_usd": 0.01,
                        "metadata": {
                            "prompt_overhead_breakdown_source": (
                                "request_body_estimate"
                            ),
                            "prompt_overhead_route_family": "openai_responses",
                            "prompt_overhead_counted_shape": "openai_responses",
                        },
                    },
                    "records": [
                        {
                            "client_name": "openai-python",
                            "provider": "openai",
                            "model": "gpt-5.4-mini",
                            "input_tokens": 10,
                            "response_cost_usd": 0.01,
                            "metadata": {
                                "prompt_overhead_breakdown_source": (
                                    "request_body_estimate"
                                ),
                                "prompt_overhead_route_family": (
                                    "openai_responses"
                                ),
                                "prompt_overhead_counted_shape": (
                                    "openai_responses"
                                ),
                            },
                        },
                        {
                            "client_name": "openai-python",
                            "provider": "openai",
                            "model": "gpt-5.4-mini",
                            "input_tokens": 99,
                            "response_cost_usd": 0.99,
                            "metadata": {
                                "prompt_overhead_breakdown_source": (
                                    "request_body_estimate"
                                ),
                                "prompt_overhead_route_family": (
                                    "openai_chat_completions"
                                ),
                                "prompt_overhead_counted_shape": (
                                    "openai_chat_completions"
                                ),
                            },
                        },
                    ],
                }
            }
        }
    )

    assert report["totals"]["calls"] == 1
    assert report["totals"]["input_tokens"] == 10
    assert report["groups"][0]["route_family"] == "openai_responses"


def test_validation_db_connection_reuses_open_connection_and_closes(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    connections = []

    class FakeConnection:
        closed = False

        def close(self):
            self.closed = True

    def fake_connect(**kwargs):
        conn = FakeConnection()
        connections.append((kwargs, conn))
        return conn

    monkeypatch.setattr(harness.psycopg, "connect", fake_connect)

    settings = {
        "host": "127.0.0.1",
        "port": 5434,
        "dbname": "aawm_tristore",
        "user": "aawm",
        "password": "pw",
    }

    first = harness._validation_db_connection(settings)
    second = harness._validation_db_connection(settings)

    assert first is second
    assert len(connections) == 1
    assert connections[0][0]["autocommit"] is True
    assert connections[0][0]["connect_timeout"] == 10

    harness._close_validation_db_connections()

    assert connections[0][1].closed is True


def test_tool_activity_validation_rejects_forbidden_argument_substring(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            pass

        def fetchall(self):
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_index": 0,
                    "tool_name": "Read",
                    "tool_kind": "read",
                    "command_text": "",
                    "arguments": {"file_path": "/tmp/example.py", "pages": ""},
                    "metadata": {},
                    "created_at": None,
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())

    _, failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_name": "Read",
                    "tool_kind": "read",
                    "arguments_forbidden_substring": '"pages": ""',
                }
            ],
        },
    )

    assert failures == [
        "case tool_activity rows for provider='openai' model='gpt-5.5' tool_name='Read' included forbidden arguments substring '\"pages\": \"\"'"
    ]

    harness._close_validation_db_connections()


def test_tool_activity_validation_rejects_missing_required_argument_substring(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            pass

        def fetchall(self):
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_index": 0,
                    "tool_name": "WebSearch",
                    "tool_kind": "read",
                    "command_text": "",
                    "arguments": {"query": "litellm anthropic adapter"},
                    "metadata": {},
                    "created_at": None,
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())

    _, failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_name": "WebSearch",
                    "tool_kind": "read",
                    "arguments_required_substring": "IANA example domain",
                }
            ],
        },
    )

    assert failures == [
        "case tool_activity rows for provider='openai' model='gpt-5.5' tool_name='WebSearch' did not include arguments containing 'IANA example domain'"
    ]

    harness._close_validation_db_connections()


def test_tool_activity_validation_requires_argument_substrings_on_every_match(
    monkeypatch,
):
    harness = _load_harness_module()
    harness._close_validation_db_connections()

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            pass

        def fetchall(self):
            return [
                {
                    "provider": "kimi_code",
                    "model": "kimi_code/k3-max",
                    "tool_index": 0,
                    "tool_name": "spawn_agent",
                    "tool_kind": "other",
                    "command_text": "",
                    "arguments": {
                        "model": "aawm-sota-moonshot",
                        "fork_turns": "none",
                        "message": "Child A",
                    },
                    "metadata": {},
                    "created_at": None,
                },
                {
                    "provider": "kimi_code",
                    "model": "kimi_code/k3-max",
                    "tool_index": 1,
                    "tool_name": "spawn_agent",
                    "tool_kind": "other",
                    "command_text": "",
                    "arguments": {
                        "fork_turns": "none",
                        "message": "Child B",
                    },
                    "metadata": {},
                    "created_at": None,
                },
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())

    _, failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "kimi_code",
                    "tool_name": "spawn_agent",
                    "tool_kind": "other",
                    "minimum_count": 2,
                    "each_arguments_required_substrings": [
                        '"model": "aawm-sota-moonshot"',
                        '"fork_turns": "none"',
                        '"message": "',
                    ],
                }
            ],
        },
    )

    assert failures == [
        "case tool_activity rows for provider='kimi_code' model=None tool_name='spawn_agent' had 1 matching row(s) without arguments containing '\"model\": \"aawm-sota-moonshot\"'"
    ]

    harness._close_validation_db_connections()


def test_tool_activity_validation_rejects_too_many_and_forbidden_commands(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            pass

        def fetchall(self):
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_index": 0,
                    "tool_name": "Bash",
                    "tool_kind": "command",
                    "command_text": "date -u +%Y-%m-%d",
                    "arguments": {},
                    "metadata": {},
                    "created_at": None,
                },
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_index": 1,
                    "tool_name": "Bash",
                    "tool_kind": "command",
                    "command_text": "ls docs",
                    "arguments": {},
                    "metadata": {},
                    "created_at": None,
                },
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())

    _, failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_name": "Bash",
                    "tool_kind": "command",
                    "minimum_count": 1,
                    "maximum_count": 1,
                    "command_text_contains": "date -u +%Y-%m-%d",
                    "command_text_forbidden_substrings": ["ls"],
                }
            ],
        },
    )

    assert failures == [
        "case too many tool_activity rows for provider='openai' model='gpt-5.5' tool_name='Bash' tool_kind='command'; expected <= 1, got 2",
        "case tool_activity rows for provider='openai' model='gpt-5.5' tool_name='Bash' included forbidden command text substring 'ls'",
    ]

    harness._close_validation_db_connections()


def test_transcript_tool_use_validation_counts_child_tools(tmp_path):
    harness = _load_harness_module()
    subagents_dir = (
        tmp_path
        / "-home-zepfu-projects-litellm"
        / "session-1"
        / "subagents"
    )
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-abc.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "gpt-5-5"}),
        encoding="utf-8",
    )
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "timestamp": "2026-04-28T10:00:00Z",
                        "message": {
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-1",
                                    "name": "Read",
                                    "input": {"file_path": "TODO.md"},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "timestamp": "2026-04-28T10:00:01Z",
                        "message": {
                            "id": "msg-2",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-2",
                                    "name": "Bash",
                                    "input": {"command": "date -u"},
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "gpt-5-5",
                    "expected_tool_counts": {"Read": 1, "Bash": 1},
                    "expected_tool_sequence": ["Read", "Bash"],
                    "maximum_total_tool_uses": 2,
                    "maximum_tool_uses_per_assistant_message": 1,
                }
            ],
        },
    )

    assert failures == []
    agent_summary = summary["agents"][0]
    assert agent_summary["by_tool_name"] == {"Bash": 1, "Read": 1}
    assert agent_summary["max_tool_uses_in_single_assistant_message"] == 1
    assert len(agent_summary["records"]) == 2


def test_transcript_tool_use_validation_rejects_wrong_sequence(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-abc.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "gpt-5-5"}),
        encoding="utf-8",
    )
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-2",
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "tool-2", "name": "Read", "input": {}},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "gpt-5-5",
                    "expected_tool_sequence": ["Read", "Bash"],
                }
            ],
        },
    )

    assert failures == [
        "case transcript for agent='gpt-5-5' tool_use sequence mismatch; expected [\"Read\", \"Bash\"], got [\"Bash\", \"Read\"]"
    ]


def test_transcript_tool_use_validation_requires_result_before_next_tool(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-abc.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "gpt-5-5"}),
        encoding="utf-8",
    )
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-2",
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "tool-2", "name": "Bash", "input": {}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "agentId": "abc",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "gpt-5-5",
                    "require_tool_result_before_next_tool_use": True,
                }
            ],
        },
    )

    assert failures == [
        "case transcript for agent='gpt-5-5' did not record tool_result before next tool_use after 'Read'"
    ]


def test_transcript_tool_use_validation_rejects_tool_result_errors(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-abc.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "gpt-5-5"}),
        encoding="utf-8",
    )
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-1",
                                    "name": "WebFetch",
                                    "input": {"url": "https://example.com"},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "agentId": "abc",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-1",
                                    "is_error": True,
                                    "content": "Request failed with status code 403",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "gpt-5-5",
                    "expected_tool_counts": {"WebFetch": 1},
                    "forbid_tool_result_errors": True,
                }
            ],
        },
    )

    assert summary["agents"][0]["tool_result_errors"][0]["tool_use_id"] == "tool-1"
    assert len(failures) == 1
    assert "had tool_result errors" in failures[0]
    assert "Request failed with status code 403" in failures[0]


def test_transcript_tool_use_validation_rejects_batched_message(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-def.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "harness-gpt55-sequential-core-tools"}),
        encoding="utf-8",
    )
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "agentId": "def",
                "message": {
                    "id": "msg-parallel",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "tool-2", "name": "Bash", "input": {}},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "harness-gpt55-sequential-core-tools",
                    "expected_tool_counts": {"Read": 1, "Bash": 1},
                    "maximum_tool_uses_per_assistant_message": 1,
                }
            ],
        },
    )

    assert failures == [
        "case transcript for agent='harness-gpt55-sequential-core-tools' had 2 tool_use blocks in one assistant message; expected <= 1"
    ]


def test_transcript_tool_use_validation_accepts_required_parallel_message(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-def.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "harness-parallel-read-tools"}),
        encoding="utf-8",
    )
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "agentId": "def",
                "message": {
                    "id": "msg-parallel",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "tool-2", "name": "Glob", "input": {}},
                        {"type": "tool_use", "id": "tool-3", "name": "Grep", "input": {}},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "harness-parallel-read-tools",
                    "expected_tool_counts": {"Read": 1, "Glob": 1, "Grep": 1},
                    "minimum_total_tool_uses": 3,
                    "maximum_total_tool_uses": 3,
                    "minimum_tools_in_single_assistant_message": 3,
                    "maximum_tool_uses_per_assistant_message": 3,
                }
            ],
        },
    )

    assert failures == []
    agent_summary = summary["agents"][0]
    assert agent_summary["max_tool_uses_in_single_assistant_message"] == 3
    assert agent_summary["total_tool_uses"] == 3


def test_transcript_tool_use_validation_requires_multiple_parallel_batches(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-def.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "sota-moonshot-stress-a"}),
        encoding="utf-8",
    )
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "agentId": "def",
                "message": {
                    "id": "msg-parallel-1",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "tool-2", "name": "Glob", "input": {}},
                        {"type": "tool_use", "id": "tool-3", "name": "Grep", "input": {}},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [
                {
                    "agent_type": "sota-moonshot-stress-a",
                    "expected_tool_counts": {"Read": 1, "Glob": 1, "Grep": 1},
                    "minimum_parallel_tool_batches": 2,
                    "minimum_tools_per_parallel_batch": 3,
                }
            ],
        },
    )

    assert failures == [
        "case transcript for agent='sota-moonshot-stress-a' had 1 parallel tool batches with >= 3 tool_use blocks; expected >= 2"
    ]
    assert summary["agents"][0]["parallel_tool_batch_validation"] == {
        "minimum_batches": 2,
        "minimum_tools_per_batch": 3,
        "qualifying_batches": 1,
    }


def test_transcript_tool_use_validation_reports_missing_child_agent(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-def.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "harness-gpt55-sequential-core-tools"}),
        encoding="utf-8",
    )
    transcript.write_text("", encoding="utf-8")

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [{"agent_type": "harness-openrouter-nemotron-parallel-read-tools"}],
        },
    )

    assert summary["agents"][0]["candidate_transcripts"] == [
        {
            "path": str(transcript),
            "agent_type": "harness-gpt55-sequential-core-tools",
        }
    ]
    assert failures == [
        "case missing Claude subagent transcript for agent='harness-openrouter-nemotron-parallel-read-tools' session_id='session-1'"
    ]


def test_session_history_and_tool_activity_validators_share_db_connection(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    connections = []

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.query = query
            self.conn.executed.append((query, params))

        def fetchall(self):
            if "session_history_tool_activity" in self.query:
                return [
                    {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "tool_index": 0,
                        "tool_name": "Bash",
                        "tool_kind": "execute",
                        "command_text": "pwd",
                        "arguments": {},
                        "metadata": {},
                        "created_at": None,
                    }
                ]
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "session_id": "session-1",
                    "tenant_id": "adapter-harness-tenant",
                    "provider_cache_status": "not_attempted",
                    "provider_cache_miss": False,
                    "reasoning_tokens_source": "not_applicable",
                    "metadata": {},
                    "start_time": None,
                    "end_time": None,
                }
            ]

    class FakeConnection:
        closed = False

        def __init__(self):
            self.executed = []

        def cursor(self):
            return FakeCursor(self)

        def close(self):
            self.closed = True

    def fake_connect(**kwargs):
        conn = FakeConnection()
        connections.append((kwargs, conn))
        return conn

    monkeypatch.setattr(harness.psycopg, "connect", fake_connect)
    checks = {
        "db_password": "pw",
        "expected_provider": "openai",
        "expected_model": "gpt-5.4",
        "require_runtime_identity": False,
    }

    session_summary, session_failures = harness._validate_session_history(
        family="case",
        session_id="session-1",
        checks=checks,
    )
    tool_summary, tool_failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "tool_name": "Bash",
                    "tool_kind": "execute",
                }
            ],
        },
    )

    assert session_failures == []
    assert tool_failures == []
    assert session_summary["record"]["provider"] == "openai"
    assert tool_summary["record"]["tool_name"] == "Bash"
    assert len(connections) == 1
    assert len(connections[0][1].executed) == 2
    session_history_query = connections[0][1].executed[0][0]
    normalized_session_history_query = " ".join(
        session_history_query.lower().split()
    )
    session_history_where_clause = normalized_session_history_query.split(
        " where ", 1
    )[1].split(" order by ", 1)[0]
    assert session_history_where_clause == "session_id = %s"
    assert " or " not in session_history_where_clause
    assert "metadata->>" not in session_history_where_clause
    assert "d1_224_live_smoke" not in normalized_session_history_query
    for column_name in (
        "input_system_tokens_estimated",
        "input_tool_advertisement_tokens_estimated",
        "input_conversation_tokens_estimated",
        "input_other_tokens_estimated",
        "input_breakdown_residual_tokens",
        "system_behavior_tokens_estimated",
        "system_safety_tokens_estimated",
        "system_instructional_tokens_estimated",
        "system_unclassified_tokens_estimated",
        "changed_pre_commit_config",
        "changed_env_file",
        "changed_pyproject_toml",
        "changed_gitignore",
    ):
        assert column_name in session_history_query
    assert connections[0][0]["autocommit"] is True

    harness._close_validation_db_connections()


def test_session_history_validation_polls_until_expected_rows_are_visible(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    attempts = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            attempts.append((query, params))

        def fetchall(self):
            if len(attempts) == 1:
                return []
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "session_id": "session-1",
                    "tenant_id": "adapter-harness-tenant",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "provider_cache_status": "not_attempted",
                    "provider_cache_miss": False,
                    "reasoning_tokens_source": "not_applicable",
                    "tool_call_count": 1,
                    "tool_names": ["run_shell_command"],
                    "input_system_tokens_estimated": 14,
                    "input_tool_advertisement_tokens_estimated": 22,
                    "input_conversation_tokens_estimated": 6,
                    "metadata": {
                        "prompt_overhead_counted_shape": "openai_responses",
                        "prompt_overhead_component_paths": {
                            "system": ["instructions"]
                        },
                    },
                    "start_time": None,
                    "end_time": None,
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: None)

    summary, failures = harness._validate_session_history(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "require_runtime_identity": False,
            "poll_timeout_seconds": 1,
            "poll_interval_seconds": 0.1,
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "minimums": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "input_system_tokens_estimated": 1,
                        "input_tool_advertisement_tokens_estimated": 1,
                        "input_conversation_tokens_estimated": 1,
                    },
                    "metadata_required_equals": {
                        "prompt_overhead_counted_shape": "openai_responses"
                    },
                    "metadata_required_truthy": [
                        "prompt_overhead_component_paths"
                    ],
                }
            ],
        },
    )

    assert failures == []
    assert summary["record"]["model"] == "gpt-5.4-mini"
    assert len(attempts) == 2

    harness._close_validation_db_connections()


def test_session_history_expected_row_failure_reports_candidate_mismatch(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            pass

        def fetchall(self):
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "session_id": "session-1",
                    "tenant_id": "litellm",
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "total_tokens": 11,
                    "provider_cache_status": "miss",
                    "provider_cache_miss": True,
                    "provider_cache_miss_reason": "cache_attempted_without_hit",
                    "reasoning_tokens_source": "not_applicable",
                    "tool_call_count": 1,
                    "tool_names": ["run_shell_command"],
                    "input_system_tokens_estimated": 0,
                    "metadata": {
                        "tenant_id": "litellm",
                        "prompt_overhead_counted_shape": "anthropic_messages_semantic",
                    },
                    "start_time": None,
                    "end_time": None,
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())

    summary, failures = harness._validate_session_history(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "require_runtime_identity": False,
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "minimums": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "input_system_tokens_estimated": 1,
                    },
                    "required_equals": {"tenant_id": "adapter-harness-tenant"},
                    "metadata_required_equals": {
                        "prompt_overhead_counted_shape": "openai_responses"
                    },
                }
            ],
        },
    )

    assert summary["records"] == []
    assert summary["all_records"][0]["tenant_id"] == "litellm"
    assert len(failures) == 1
    assert "candidate rows" in failures[0]
    assert '"actual": "litellm"' in failures[0]
    assert '"expected": "adapter-harness-tenant"' in failures[0]
    assert '"input_system_tokens_estimated"' in failures[0]
    assert '"metadata.prompt_overhead_counted_shape"' in failures[0]

    harness._close_validation_db_connections()


def test_rate_limit_observations_validation_matches_session_rows(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    now = dt.datetime(2026, 5, 14, 18, 0, tzinfo=dt.timezone.utc)
    attempts = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            attempts.append((query, params))

        def fetchall(self):
            return [
                {
                    "observed_at": now,
                    "created_at": now,
                    "client": "claude",
                    "client_version": "1.2.3",
                    "account_hash": "acct",
                    "provider": "anthropic",
                    "model": "claude-opus-4-6",
                    "quota_key": "anthropic_unified_5h:5h",
                    "quota_period": "5h",
                    "quota_type": "tokens",
                    "expected_reset_at": now + dt.timedelta(hours=1),
                    "remaining_pct": 66.0,
                    "source": "anthropic_response_headers",
                    "session_id": "session-1",
                    "trace_id": "trace-1",
                    "litellm_call_id": "call-1",
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(harness.RA, "_utcnow", lambda: now)

    summary, failures, warnings = harness._validate_rate_limit_observations(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "anthropic",
                    "source": "anthropic_response_headers",
                    "quota_key": "anthropic_unified_5h:5h",
                    "minimums": {"remaining_pct": 0},
                    "maximums": {"remaining_pct": 100},
                    "required_future_timestamps": ["expected_reset_at"],
                    "required_timestamp_after_observed": ["expected_reset_at"],
                }
            ],
        },
    )

    assert failures == []
    assert warnings == []
    assert summary["match_source"] == "session"
    assert summary["matched_records"][0]["quota_key"] == "anthropic_unified_5h:5h"
    assert "from public.rate_limit_observations" in attempts[0][0]
    assert attempts[0][1] == ("session-1",)

    harness._close_validation_db_connections()


def test_rate_limit_observations_validation_matches_codex_rows(
    monkeypatch,
):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    now = dt.datetime(2026, 5, 14, 18, 0, tzinfo=dt.timezone.utc)

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "observed_at": now,
                    "created_at": now,
                    "client": "codex",
                    "client_version": "0.130.0",
                    "account_hash": "acct-openai-primary",
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "quota_key": "codex:primary",
                    "quota_period": "five_hour",
                    "quota_type": "tokens",
                    "expected_reset_at": now + dt.timedelta(hours=2),
                    "remaining_pct": 57.0,
                    "source": "codex_response_headers",
                    "session_id": "session-native",
                    "trace_id": "trace-codex-primary",
                    "litellm_call_id": "call-codex-primary",
                },
                {
                    "observed_at": now,
                    "created_at": now,
                    "client": "codex",
                    "client_version": "0.130.0",
                    "account_hash": "acct-openai-secondary",
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "quota_key": "codex:secondary",
                    "quota_period": "seven_day",
                    "quota_type": "tokens",
                    "expected_reset_at": now + dt.timedelta(days=2),
                    "remaining_pct": 82.0,
                    "source": "codex_response_headers",
                    "session_id": "session-native",
                    "trace_id": "trace-codex-secondary",
                    "litellm_call_id": "call-codex-secondary",
                },
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(harness.RA, "_utcnow", lambda: now)

    summary, failures, warnings = harness._validate_rate_limit_observations(
        family="native-provider-cases",
        session_id="session-native",
        checks={
            "db_password": "pw",
            "expected_rows": [
                {
                    "provider": "openai",
                    "client": "codex",
                    "source": "codex_response_headers",
                    "quota_key": "codex:primary",
                    "quota_type": "tokens",
                    "required_equals": {"quota_period": "five_hour"},
                    "minimums": {"remaining_pct": 0},
                    "maximums": {"remaining_pct": 100},
                    "required_future_timestamps": ["expected_reset_at"],
                    "required_timestamp_after_observed": ["expected_reset_at"],
                },
                {
                    "provider": "openai",
                    "client": "codex",
                    "model": "gpt-5.4",
                    "source": "codex_response_headers",
                    "quota_key": "codex:secondary",
                    "quota_type": "tokens",
                    "required_equals": {"quota_period": "seven_day"},
                    "minimums": {"remaining_pct": 0},
                    "maximums": {"remaining_pct": 100},
                    "required_future_timestamps": ["expected_reset_at"],
                    "required_timestamp_after_observed": ["expected_reset_at"],
                },
            ],
        },
    )

    assert failures == []
    assert warnings == []
    assert summary["match_source"] == "session"
    assert {
        row["quota_key"] for row in summary["matched_records"]
    } == {
        "codex:primary",
        "codex:secondary",
    }

    harness._close_validation_db_connections()


def test_rate_limit_observations_validation_can_fall_back_to_latest_snapshot(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    now = dt.datetime(2026, 5, 14, 18, 0, tzinfo=dt.timezone.utc)
    attempts = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            attempts.append((query, params))

        def fetchall(self):
            query = attempts[-1][0]
            if "where session_id = %s" in query:
                return []
            return [
                {
                    "observed_at": now - dt.timedelta(minutes=3),
                    "created_at": now - dt.timedelta(minutes=3),
                    "client": "claude",
                    "client_version": "1.2.3",
                    "account_hash": "acct",
                    "provider": "anthropic",
                    "model": "claude-opus-4-6",
                    "quota_key": "anthropic_unified_7d:7d",
                    "quota_period": "7d",
                    "quota_type": "tokens",
                    "expected_reset_at": now + dt.timedelta(days=1),
                    "remaining_pct": 99.0,
                    "source": "anthropic_response_headers",
                    "session_id": "older-session",
                    "trace_id": "trace-old",
                    "litellm_call_id": "call-old",
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(harness.RA, "_utcnow", lambda: now)

    summary, failures, warnings = harness._validate_rate_limit_observations(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "allow_latest_snapshot_fallback": True,
            "latest_snapshot_max_age_seconds": 21600,
            "expected_rows": [
                {
                    "provider": "anthropic",
                    "source": "anthropic_response_headers",
                    "quota_key": "anthropic_unified_7d:7d",
                    "minimums": {"remaining_pct": 0},
                    "maximums": {"remaining_pct": 100},
                    "required_future_timestamps": ["expected_reset_at"],
                    "required_timestamp_after_observed": ["expected_reset_at"],
                }
            ],
        },
    )

    assert failures == []
    assert summary["match_source"] == "latest_snapshot"
    assert summary["matched_records"][0]["session_id"] == "older-session"
    assert warnings == [
        "case rate_limit_observations matched latest current snapshots instead of session rows; unchanged duplicate snapshots may have been suppressed"
    ]

    harness._close_validation_db_connections()


def test_tool_activity_validation_polls_until_expected_rows_are_visible(monkeypatch):
    harness = _load_harness_module()
    harness._close_validation_db_connections()
    attempts = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            attempts.append((query, params))

        def fetchall(self):
            if len(attempts) == 1:
                return []
            return [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_index": 0,
                    "tool_name": "Bash",
                    "tool_kind": "command",
                    "command_text": "date -u +%Y-%m-%dT%H:%M:%S.%NZ",
                    "arguments": {},
                    "metadata": {},
                    "created_at": None,
                }
            ]

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness.psycopg, "connect", lambda **kwargs: FakeConnection())
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: None)

    summary, failures = harness._validate_tool_activity(
        family="case",
        session_id="session-1",
        checks={
            "db_password": "pw",
            "poll_timeout_seconds": 1,
            "poll_interval_seconds": 0.1,
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_name": "Bash",
                    "tool_kind": "command",
                    "command_text_contains": "date -u",
                }
            ],
        },
    )

    assert failures == []
    assert summary["record"]["tool_name"] == "Bash"
    assert len(attempts) == 2

    harness._close_validation_db_connections()


def test_run_command_with_retry_records_attempt_metadata_for_retried_api_errors(monkeypatch):
    harness = _load_harness_module()

    def fake_run_command(command, extra_env=None, timeout_seconds=300):
        fake_run_command.calls += 1
        if fake_run_command.calls == 1:
            return {
                "command": command,
                "exit_code": 200,
                "stdout": '{"status_code":503, "is_error":true}',
                "stderr": "",
                "duration_seconds": 0.0,
            }
        return {
            "command": command,
            "exit_code": 200,
            "stdout": '{"status_code":200, "is_error":false}',
            "stderr": "",
            "duration_seconds": 0.0,
        }

    fake_run_command.calls = 0
    monkeypatch.setattr(harness.RA, "_run_command", fake_run_command)

    _, final_run, attempts = harness._run_command_with_retry(
        config={
            "command": ["claude", "-p", "hello"],
            "retry_on_api_error_statuses": [503],
            "retry_max_attempts": 2,
            "retry_backoff_seconds": 0,
        }
    )

    assert len(attempts) == 2
    assert attempts[0]["attempt"] == 1
    assert attempts[0]["api_error_status"] == 503
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["api_error_status"] is None
    assert final_run["exit_code"] == 200
    assert final_run["stdout"] == '{"status_code":200, "is_error":false}'


def test_summarize_transcript_tool_uses_preserves_tool_use_id_tool_result_link(tmp_path):
    harness = _load_harness_module()
    transcript = tmp_path / "session-1" / "subagents" / "agent-abc.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "agentId": "abc",
                        "message": {
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool-1",
                                    "name": "Bash",
                                    "input": {"command": "pwd"},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "agentId": "abc",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-1",
                                    "content": "ok",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = harness._summarize_transcript_tool_uses([transcript])

    records = summary["records"]
    assert len(records) == 1
    assert records[0]["tool_use_id"] == "tool-1"
    assert records[0]["tool_name"] == "Bash"
    assert isinstance(records[0].get("tool_result_line"), int)


def test_build_case_verification_matrix_row_captures_status_provider_model_and_tool_activity():
    harness = _load_harness_module()

    case_config = {
        "session_history_validation": {
            "expected_provider": "openai",
            "expected_model": "gpt-5.5",
        },
        "required_trace_tags": ["route:anthropic_openai_responses_adapter"],
        "stream_tool_call_state_validation": {
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                }
            ]
        },
        "tool_activity_validation": {
            "expected_rows": [
                {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "tool_name": "Bash",
                    "arguments_required_substring": '"command": "pwd"',
                }
            ]
        },
        "transcript_tool_use_validation": {
            "expected_agents": [
                {
                    "agent_type": "harness-gpt55",
                    "require_tool_result_before_next_tool_use": True,
                }
            ]
        },
    }

    row_passed = harness._build_case_verification_matrix_row(
        alias="aawm-code-anthropic-openai",
        candidate_order=0,
        case_config=case_config,
        case_result={
            "passed": True,
            "exit_code": 0,
            "command_attempts": [
                {
                    "attempt": 1,
                    "api_error_status": None,
                    "exit_code": 0,
                }
            ],
            "session_history_passed": True,
            "stream_tool_call_state_passed": True,
            "tool_activity_passed": True,
            "transcript_tool_use_passed": True,
            "langfuse": {
                "command_session_id": "session-openai",
                "trace_ids": ["trace-openai"],
                "filtered_trace_ids": ["trace-openai"],
                "actual_trace_names": ["claude-code.aawm-code-anthropic-openai"],
                "actual_user_ids": ["adapter-harness-tenant"],
            },
            "transcript_tool_use": {
                "agents": [
                    {
                        "agent_type": "harness-gpt55",
                        "records": [
                            {
                                "tool_use_id": "tool-openai-1",
                                "tool_name": "Bash",
                                "tool_result_line": 12,
                            },
                            {
                                "tool_use_id": "tool-openai-2",
                                "tool_name": "Read",
                            },
                        ],
                    }
                ]
            },
        },
    )

    assert row_passed["alias"] == "aawm-code-anthropic-openai"
    assert row_passed["candidate_order"] == 0
    assert row_passed["provider"] == "openai"
    assert row_passed["model"] == "gpt-5.5"
    assert row_passed["route_family"] == "anthropic_openai_responses_adapter"
    assert row_passed["connectivity_passed"] is True
    assert row_passed["session_history_metadata_passed"] is True
    assert row_passed["stream_tool_call_state_passed"] is True
    assert row_passed["tool_call_emission_passed"] is True
    assert row_passed["tool_bearing_passed"] is True
    assert row_passed["required_tool_arguments_passed"] is True
    assert row_passed["tool_use_ids_passed"] is True
    assert row_passed["tool_result_replay_passed"] is True
    assert row_passed["multi_turn_tool_result_passed"] is True
    assert row_passed["status"] == "passed"
    assert row_passed["references"]["command_session_id"] == "session-openai"
    assert row_passed["references"]["command_attempts"] == [
        {"attempt": 1, "api_error_status": None, "exit_code": 0}
    ]
    assert row_passed["references"]["trace_ids"] == ["trace-openai"]


def test_verification_matrix_row_flags_missing_transcript_tool_use_ids_and_replay():
    harness = _load_harness_module()

    row = harness._build_case_verification_matrix_row(
        alias="aawm-code-anthropic-openai",
        candidate_order=1,
        case_config={
            "session_history_validation": {
                "expected_provider": "openai",
                "expected_model": "gpt-5.3-codex-spark",
            },
            "required_trace_tags": ["route:anthropic_openai_responses_adapter"],
            "tool_activity_validation": {
                "expected_rows": [
                    {
                        "tool_name": "Bash",
                        "arguments_required_substring": '"command"',
                    }
                ]
            },
            "transcript_tool_use_validation": {
                "expected_agents": [
                    {
                        "agent_type": "harness-openai",
                        "require_tool_result_before_next_tool_use": True,
                    }
                ]
            },
        },
        case_result={
            "passed": False,
            "exit_code": 0,
            "session_history_passed": True,
            "tool_activity_passed": False,
            "transcript_tool_use_passed": True,
            "langfuse": {},
            "transcript_tool_use": {
                "agents": [
                    {
                        "agent_type": "harness-openai",
                        "records": [
                            {"tool_use_id": "", "tool_name": "Bash"},
                            {"tool_use_id": "tool-openai-2", "tool_name": "Read"},
                        ],
                    }
                ]
            },
        },
    )

    assert row["provider"] == "openai"
    assert row["model"] == "gpt-5.3-codex-spark"
    assert row["candidate_order"] == 1
    assert row["status"] == "failed"
    assert row["tool_call_emission_passed"] is False
    assert row["required_tool_arguments_passed"] is False
    assert row["tool_use_ids_passed"] is False
    assert row["tool_result_replay_passed"] is False
    assert row["multi_turn_tool_result_passed"] is True


def test_verification_matrix_row_uses_runtime_session_history_for_alias_replay():
    harness = _load_harness_module()

    row = harness._build_case_verification_matrix_row(
        alias="aawm-code-anthropic",
        case_name=D1256_ALIAS_REPLAY_CASE,
        candidate_order=-1,
        case_config={
            "verification_candidate_label": "alias-replay",
            "verification_declared_candidates": [
                {
                    "candidate_order": 2,
                    "provider": "xai",
                    "model": "grok-composer-2.5-fast",
                    "route_family": "anthropic_grok_native_responses_adapter",
                },
                {
                    "candidate_order": 3,
                    "provider": "xai",
                    "model": "oa_xai/grok-build",
                    "route_family": "anthropic_xai_oauth_responses_adapter",
                },
            ],
            "session_history_validation": {
                "expected_rows": [
                    {
                        "required_one_of": {
                            "provider": ["xai", "anthropic"],
                            "model": ["oa_xai/grok-build", "claude-sonnet-4-6"],
                        }
                    }
                ]
            },
            "required_trace_tags": ["route:anthropic_messages"],
            "tool_activity_validation": {
                "expected_rows": [
                    {
                        "tool_name": "Read",
                        "arguments_required_substring": "fixture",
                    }
                ]
            },
            "transcript_tool_use_validation": {
                "expected_agents": [
                    {
                        "agent_type": D1256_ALIAS_REPLAY_AGENT,
                    }
                ]
            },
        },
        case_result={
            "passed": True,
            "exit_code": 0,
            "session_history_passed": True,
            "tool_activity_passed": True,
            "transcript_tool_use_passed": True,
            "session_history": {
                "record": {
                    "provider": "xai",
                    "model": "oa_xai/grok-build",
                    "metadata": {
                        "passthrough_route_family": (
                            "grok_cli_chat_proxy"
                        ),
                        "anthropic_auto_agent_selected_route_family": (
                            "anthropic_xai_oauth_responses_adapter"
                        )
                    },
                }
            },
            "langfuse": {},
            "transcript_tool_use": {
                "agents": [
                    {
                        "agent_type": D1256_ALIAS_REPLAY_AGENT,
                        "records": [{"tool_use_id": "tool-1", "tool_name": "Read"}],
                    }
                ]
            },
        },
    )

    assert row["alias"] == "aawm-code-anthropic"
    assert row["case_name"] == D1256_ALIAS_REPLAY_CASE
    assert row["candidate_order"] == -1
    assert row["candidate_label"] == "alias-replay"
    assert row["declared_candidates"] == [
        {
            "candidate_order": 2,
            "provider": "xai",
            "model": "grok-composer-2.5-fast",
            "route_family": "anthropic_grok_native_responses_adapter",
        },
        {
            "candidate_order": 3,
            "provider": "xai",
            "model": "oa_xai/grok-build",
            "route_family": "anthropic_xai_oauth_responses_adapter",
        },
    ]
    assert row["provider"] == "xai"
    assert row["model"] == "oa_xai/grok-build"
    assert row["route_family"] == "anthropic_xai_oauth_responses_adapter"
    assert row["tool_bearing_passed"] is True
    assert row["required_tool_arguments_passed"] is True
    assert row["tool_use_ids_passed"] is True


def test_verification_matrix_row_status_encodes_warning_soft_and_skip_states():
    harness = _load_harness_module()

    case_config = {
        "session_history_validation": {
            "expected_provider": "anthropic",
            "expected_model": "claude-opus-4-6",
        },
    }

    assert harness._build_case_verification_matrix_row(
        alias="candidate",
        candidate_order=1,
        case_config=case_config,
        case_result={"passed": True, "soft_failures": ["warn"], "langfuse": {}},
    )["status"] == "passed_with_soft_failures"

    assert harness._build_case_verification_matrix_row(
        alias="candidate",
        candidate_order=1,
        case_config=case_config,
        case_result={"passed": True, "skipped": True, "langfuse": {}},
    )["status"] == "skipped"

    assert harness._build_case_verification_matrix_row(
        alias="candidate",
        candidate_order=1,
        case_config=case_config,
        case_result={"passed": False, "langfuse": {}},
    )["status"] == "failed"


def test_main_writes_verification_matrix_and_outcome_records_by_candidate_order(tmp_path, monkeypatch):
    harness = _load_harness_module()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    config_path = tmp_path / "anthropic_adapter_config.json"
    artifact_path = tmp_path / "anthropic_adapter_results.json"
    config_path.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "cases": {
                    "aawm_code_anthropic_candidate_openai": {
                        "command": ["claude", "-p", "identity"],
                        "session_history_validation": {
                            "expected_provider": "openai",
                            "expected_model": "gpt-5.5",
                        },
                        "required_trace_tags": [
                            "route:anthropic_openai_responses_adapter"
                        ],
                    },
                    "aawm_code_anthropic_candidate_anthropic": {
                        "command": ["claude", "-p", "identity"],
                        "session_history_validation": {
                            "expected_provider": "anthropic",
                            "expected_model": "claude-opus-4-6",
                        },
                        "required_trace_tags": ["route:anthropic_messages"],
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_validate_case(name, config, query_url, public_key, secret_key, litellm_base_url):
        if name == "aawm_code_anthropic_candidate_openai":
            return {
                "passed": True,
                "skipped": False,
                "warning_only": False,
                "failures": [],
                "soft_failures": [],
                "warnings": [],
                "command_attempts": [
                    {
                        "attempt": 1,
                        "api_error_status": None,
                        "exit_code": 0,
                    }
                ],
                "tool_activity_passed": True,
                "transcript_tool_use_passed": True,
                "langfuse": {
                    "command_session_id": "session-openai",
                    "trace_ids": ["trace-openai"],
                    "filtered_trace_ids": ["trace-openai"],
                    "actual_trace_names": ["claude-code.aawm_code_anthropic_candidate_openai"],
                    "actual_user_ids": ["adapter-harness-tenant"],
                },
                "session_history": {
                    "record": {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "metadata": {
                            "provider": "openai",
                            "route_family": "anthropic_openai_responses_adapter",
                            "litellm_environment": "dev",
                        },
                    }
                },
                "tool_activity": {
                    "record": {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "tool_name": "Bash",
                        "tool_kind": "command",
                        "command_text": "pwd",
                        "arguments": {"command": "pwd"},
                    }
                },
                "transcript_tool_use": {
                    "agents": [
                        {
                            "agent_type": "harness-anthropic-openai",
                            "records": [
                                {
                                    "tool_use_id": "tool-openai-1",
                                    "tool_result_line": 12,
                                }
                            ],
                        }
                    ]
                },
            }

        return {
            "passed": False,
            "skipped": False,
            "warning_only": False,
            "failures": ["downstream candidate failed"],
            "soft_failures": [],
            "warnings": [],
            "command_attempts": [
                {
                    "attempt": 1,
                    "api_error_status": 503,
                    "exit_code": 0,
                },
                {
                    "attempt": 2,
                    "api_error_status": None,
                    "exit_code": 1,
                },
            ],
            "tool_activity_passed": False,
            "transcript_tool_use_passed": False,
            "langfuse": {
                "command_session_id": "session-anthropic",
                "trace_ids": [],
                "filtered_trace_ids": [],
                "actual_trace_names": [],
                "actual_user_ids": [],
            },
            "session_history": {
                "record": {
                    "provider": "anthropic",
                    "model": "claude-opus-4-6",
                    "metadata": {
                        "provider": "anthropic",
                        "route_family": "anthropic_messages",
                        "litellm_environment": "dev",
                    },
                }
            },
            "tool_activity": {"record": None},
            "transcript_tool_use": {"agents": []},
        }

    monkeypatch.setattr(harness, "_validate_case", fake_validate_case)
    monkeypatch.setattr(
        harness,
        "_docker_status_for_container",
        lambda container_name: "Up",
    )
    monkeypatch.setattr(
        harness.RA,
        "_git_value",
        lambda *args: "test",
    )
    monkeypatch.setattr(
        harness.sys,
        "argv",
        [
            "run-anthropic-adapter-acceptance",
            "--config",
            str(config_path),
            "--write-artifact",
            str(artifact_path),
            "--cases",
            "aawm_code_anthropic_candidate_openai,aawm_code_anthropic_candidate_anthropic",
        ],
    )

    assert harness.main() == 1

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    matrix = artifact["verification_matrix"]
    assert [row["alias"] for row in matrix] == [
        "aawm_code_anthropic_candidate_openai",
        "aawm_code_anthropic_candidate_anthropic",
    ]
    assert [row["candidate_order"] for row in matrix] == [0, 1]
    assert matrix[0]["provider"] == "openai"
    assert matrix[1]["provider"] == "anthropic"
    assert matrix[0]["status"] == "passed"
    assert matrix[1]["status"] == "failed"

    openai_result = artifact["results"]["aawm_code_anthropic_candidate_openai"]
    assert openai_result["command_attempts"] == [{"attempt": 1, "api_error_status": None, "exit_code": 0}]
    assert openai_result["session_history"]["record"]["metadata"]["litellm_environment"] == "dev"
    assert openai_result["tool_activity"]["record"]["tool_name"] == "Bash"
    assert openai_result["transcript_tool_use"]["agents"][0]["records"][0]["tool_result_line"] == 12

    anthropic_result = artifact["results"]["aawm_code_anthropic_candidate_anthropic"]
    assert anthropic_result["passed"] is False
    assert anthropic_result["command_attempts"][1]["attempt"] == 2
    assert anthropic_result["session_history"]["record"]["model"] == "claude-opus-4-6"
    assert artifact["summary"]["passed"] is False
