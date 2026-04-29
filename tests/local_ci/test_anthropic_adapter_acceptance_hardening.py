import importlib.util
import datetime as dt
import json
import pathlib
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
    "claude_adapter_gemini31_pro_child_sequential_core_tools": (
        "harness-gemini31-pro-sequential-core-tools"
    ),
    "claude_adapter_gemini3_flash_child_sequential_core_tools": (
        "harness-gemini3-flash-sequential-core-tools"
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
    "claude_adapter_nvidia_deepseek_child_parallel_read_tools": (
        "harness-nvidia-deepseek-parallel-read-tools",
        "nvidia_nim",
        "deepseek-ai/deepseek-v3.2",
        {"Read", "Glob", "Grep"},
    ),
    "claude_adapter_gemini31_pro_child_parallel_read_tools": (
        "harness-gemini31-pro-parallel-read-tools",
        "gemini",
        "gemini-3.1-pro-preview",
        {"read_file", "glob", "grep_search"},
    ),
    "claude_adapter_gemini3_flash_child_parallel_read_tools": (
        "harness-gemini3-flash-parallel-read-tools",
        "gemini",
        "gemini-3-flash-preview",
        {"read_file", "glob", "grep_search"},
    ),
}


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


def test_warning_only_timeout_still_fails_hard():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "warning_only_fixture",
        subprocess.TimeoutExpired(["claude"], 180),
        {"warning_only": True},
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

    assert failures == []
    assert soft_failures == [
        "claude_adapter_gpt_oss_120b command failed",
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
    assert validation["require_runtime_identity"] is True
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
            },
            "native_gemini": {
                "cli_passthrough": "gemini",
                "command": ["gemini", "--prompt", "hello"],
            },
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
    gemini_headers = updated["cases"]["native_gemini"]["env"][
        "GEMINI_CLI_CUSTOM_HEADERS"
    ]
    assert "x-aawm-repository: zepfu/litellm" in gemini_headers


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

    claude_stream_gate = claude_case["stream_tool_call_state_validation"]
    assert claude_stream_gate["expected_tools"][0]["tool_name"] == "Bash"
    assert claude_stream_gate["expected_tools"][0]["tool_type"] == "function_call"
    assert "response.function_call_arguments.done" in claude_stream_gate[
        "required_any_event_type_groups"
    ][0]
    assert "responses_stream_tool_call_count" in claude_case[
        "required_generation_metadata_minimums"
    ]


def test_target_profile_appends_case_local_claude_agents(monkeypatch):
    harness = _load_harness_module()

    class FixedUuid:
        hex = "abcdef1234567890"

    monkeypatch.setattr(harness.uuid, "uuid4", lambda: FixedUuid())

    config = {
        "cases": {
            "claude_adapter_gemini3_flash_child_sequential_core_tools": {
                "command": [
                    "claude",
                    "-p",
                    "Dispatch to harness-gemini3-flash-sequential-core-tools.",
                    "--allowedTools",
                    "Agent",
                ],
                "claude_agents": {
                    "harness-gemini3-flash-sequential-core-tools": {
                        "model": "google/gemini-3-flash-preview",
                        "tools": SEQUENTIAL_CORE_TOOLS,
                    }
                },
                "env": {"ANTHROPIC_BASE_URL": "placeholder"},
                "session_history_validation": {"expected_provider": "gemini"},
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

    command = updated["cases"][
        "claude_adapter_gemini3_flash_child_sequential_core_tools"
    ]["command"]
    assert command[command.index("--allowedTools") + 1] == "Agent"
    assert "--tools" not in command
    assert command.count("--agents") == 1
    agents = json.loads(command[command.index("--agents") + 1])
    assert agents["harness-gemini3-flash-sequential-core-tools"]["tools"] == (
        SEQUENTIAL_CORE_TOOLS
    )
    assert not any(
        tool.startswith("mcp__aawm__")
        for tool in agents["harness-gemini3-flash-sequential-core-tools"][
            "tools"
        ]
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


def test_parallel_read_tool_prompts_use_harness_agents_and_parallel_gate():
    config = json.loads(ANTHROPIC_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))

    for case_name, (
        agent_name,
        provider,
        model,
        durable_tool_names,
    ) in PARALLEL_CASE_AGENTS.items():
        case_config = config["cases"][case_name]
        command = case_config["command"]
        prompt = command[2]
        assert case_name in config["default_excluded_cases"]
        assert f"Dispatch to the {agent_name} agent" in prompt
        assert "exactly three tool calls" in prompt
        assert "must not wait for any tool result" in prompt
        assert "sequential_core_tools_fixture.txt" in prompt
        assert "sequential-core-tools-grep" in prompt
        assert command[command.index("--allowedTools") + 1] == "Agent"
        assert "--tools" not in command

        assert set(case_config["claude_agents"]) == {agent_name}
        agent_config = case_config["claude_agents"][agent_name]
        assert agent_config["tools"] == PARALLEL_READ_TOOLS
        assert "first assistant message must contain exactly three tool_use blocks" in (
            agent_config["prompt"]
        )
        assert "Do not wait for any tool result" in agent_config["prompt"]
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
        assert transcript_agent["minimum_tools_in_single_assistant_message"] == 3
        assert transcript_agent["maximum_tool_uses_per_assistant_message"] == 3
        assert "require_tool_result_before_next_tool_use" not in transcript_agent

        durable_rows = [
            row
            for row in case_config["tool_activity_validation"]["expected_rows"]
            if row.get("provider") == provider and row.get("model") == model
        ]
        assert {row["tool_name"] for row in durable_rows} == durable_tool_names

        if provider == "openai":
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
            required_paths = case_config["request_payload_checks"]["required_paths"]
            for path in (
                "model",
                "input",
                "instructions",
                "reasoning.effort",
                "stream",
                "tools",
                "litellm_metadata.openai_adapter_claude_context_compacted",
                "litellm_metadata.openai_adapter_claude_context_compaction_events",
                "litellm_metadata.openai_adapter_parallel_instruction_policy_applied",
            ):
                assert path in required_paths
            required_equals = case_config["request_payload_checks"][
                "required_equals"
            ]
            assert required_equals[
                "parallel_tool_calls"
            ] is True
            assert required_equals[
                "litellm_metadata.openai_adapter_claude_context_compacted"
            ] is True
            assert required_equals[
                "litellm_metadata.openai_adapter_parallel_instruction_policy_applied"
            ] is True
        elif provider == "openrouter":
            assert "route:anthropic_openrouter_responses_adapter" in case_config[
                "required_trace_tags"
            ]
            assert "openrouter-adapter-claude-context-compacted" in case_config[
                "required_trace_tags"
            ]
            assert "openrouter-adapter-parallel-instruction-policy" in case_config[
                "required_trace_tags"
            ]
            required_paths = case_config["request_payload_checks"]["required_paths"]
            for path in (
                "model",
                "input",
                "instructions",
                "stream",
                "tools",
                "litellm_metadata.openrouter_adapter_claude_context_compacted",
                "litellm_metadata.openrouter_adapter_claude_context_compaction_events",
                "litellm_metadata.openrouter_adapter_parallel_instruction_policy_applied",
            ):
                assert path in required_paths
            required_equals = case_config["request_payload_checks"][
                "required_equals"
            ]
            assert required_equals["parallel_tool_calls"] is True
            assert required_equals[
                "litellm_metadata.openrouter_adapter_claude_context_compacted"
            ] is True
            assert required_equals[
                "litellm_metadata.openrouter_adapter_parallel_instruction_policy_applied"
            ] is True
        elif provider == "nvidia_nim":
            assert "route:anthropic_nvidia_completion_adapter" in case_config[
                "required_trace_tags"
            ]
            required_paths = case_config["request_payload_checks"]["required_paths"]
            for path in (
                "model",
                "messages",
                "max_tokens",
                "tools",
            ):
                assert path in required_paths


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
    assert captured["env"]["AAWM_DB_PASSWORD"] == "not-written-to-settings"
    assert not captured["settings_path"].exists()
    assert result["command"] == captured["command"]


def test_trace_user_id_validation_can_require_child_trace_users():
    harness = _load_harness_module()

    summary, failures = harness._validate_trace_user_ids_by_name(
        family="gemini fanout",
        traces=[
            {
                "name": "claude-code.gemini-3-flash-preview",
                "userId": "adapter-harness-tenant",
            },
            {
                "name": "claude-code.gemini-3-1-pro-preview",
                "userId": "wrong-user",
            },
        ],
        expected={
            "claude-code.gemini-3-flash-preview": "adapter-harness-tenant",
            "claude-code.gemini-3-1-pro-preview": "adapter-harness-tenant",
        },
    )

    assert summary["actual_by_name"]["claude-code.gemini-3-flash-preview"] == [
        "adapter-harness-tenant"
    ]
    assert failures == [
        "gemini fanout trace claude-code.gemini-3-1-pro-preview missing user id "
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
        },
        {
            "provider": "openai",
            "model": "gpt-5-mini",
            "tenant_id": "adapter-harness-tenant",
            "cache_read_input_tokens": 2048,
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
                "minimum_count": 2,
            }
        ],
    )

    assert failures == []
    assert len(matched_records) == 2


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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "tool_index": 0,
                    "tool_name": "google_web_search",
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "tool_name": "google_web_search",
                    "tool_kind": "read",
                    "arguments_required_substring": "IANA example domain",
                }
            ],
        },
    )

    assert failures == [
        "case tool_activity rows for provider='gemini' model='gemini-3-flash-preview' tool_name='google_web_search' did not include arguments containing 'IANA example domain'"
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
                    "provider": "gemini",
                    "model": "gemini-3.1-pro-preview",
                    "tool_index": 0,
                    "tool_name": "run_shell_command",
                    "tool_kind": "command",
                    "command_text": "date -u +%Y-%m-%d",
                    "arguments": {},
                    "metadata": {},
                    "created_at": None,
                },
                {
                    "provider": "gemini",
                    "model": "gemini-3.1-pro-preview",
                    "tool_index": 1,
                    "tool_name": "run_shell_command",
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
                    "provider": "gemini",
                    "model": "gemini-3.1-pro-preview",
                    "tool_name": "run_shell_command",
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
        "case too many tool_activity rows for provider='gemini' model='gemini-3.1-pro-preview' tool_name='run_shell_command' tool_kind='command'; expected <= 1, got 2",
        "case tool_activity rows for provider='gemini' model='gemini-3.1-pro-preview' tool_name='run_shell_command' included forbidden command text substring 'ls'",
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
        json.dumps({"agentType": "gemini-3-1-pro-preview"}),
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
                    "agent_type": "gemini-3-1-pro-preview",
                    "expected_tool_counts": {"Read": 1, "Bash": 1},
                    "maximum_tool_uses_per_assistant_message": 1,
                }
            ],
        },
    )

    assert failures == [
        "case transcript for agent='gemini-3-1-pro-preview' had 2 tool_use blocks in one assistant message; expected <= 1"
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


def test_transcript_tool_use_validation_reports_missing_child_agent(tmp_path):
    harness = _load_harness_module()
    subagents_dir = tmp_path / "project" / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    transcript = subagents_dir / "agent-def.jsonl"
    transcript.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": "gemini-3-flash-preview"}),
        encoding="utf-8",
    )
    transcript.write_text("", encoding="utf-8")

    summary, failures = harness._validate_transcript_tool_use(
        family="case",
        session_id="session-1",
        checks={
            "claude_projects_root": str(tmp_path),
            "expected_agents": [{"agent_type": "gemini-3-1-pro-preview"}],
        },
    )

    assert summary["agents"][0]["candidate_transcripts"] == [
        {
            "path": str(transcript),
            "agent_type": "gemini-3-flash-preview",
        }
    ]
    assert failures == [
        "case missing Claude subagent transcript for agent='gemini-3-1-pro-preview' session_id='session-1'"
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
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
                    "metadata": {},
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "minimums": {"input_tokens": 1, "output_tokens": 1},
                }
            ],
        },
    )

    assert failures == []
    assert summary["record"]["model"] == "gemini-3-flash-preview"
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
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
                    "metadata": {"tenant_id": "litellm"},
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "minimums": {"input_tokens": 1, "output_tokens": 1},
                    "required_equals": {"tenant_id": "adapter-harness-tenant"},
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "tool_index": 0,
                    "tool_name": "run_shell_command",
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
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "tool_name": "run_shell_command",
                    "tool_kind": "command",
                    "command_text_contains": "date -u",
                }
            ],
        },
    )

    assert failures == []
    assert summary["record"]["tool_name"] == "run_shell_command"
    assert len(attempts) == 2

    harness._close_validation_db_connections()
