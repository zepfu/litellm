from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import scripts.score_agent_trace_quality as scorer
from litellm.integrations import aawm_agent_quality_rules


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
        "agent_id": None,
        "repository": "dashboard-shell",
        "tenant_id": "dashboard-shell",
        "input_tokens": 53874,
        "output_tokens": 3,
        "tool_call_count": 0,
        "invalid_tool_call_count": 0,
        "llm_upstream_elapsed_ms": None,
        "total_server_elapsed_ms": None,
        "ttft_ms": None,
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


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _init_ignored_path_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ignored-path-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / ".gitignore").write_text(".analysis/\n", encoding="utf-8")
    analysis_dir = repo / ".analysis"
    analysis_dir.mkdir()
    (analysis_dir / "todo.md").write_text("local planning\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", ".analysis/todo.md"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


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
    assert evidence.response_meaningfulness_score is None
    assert evidence.output_contract_compliance_score is None
    assert evidence.task_progress_score is None
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


def test_should_not_persist_missing_payload_booleans_as_false(monkeypatch) -> None:
    evidence = scorer.score_candidate(
        _candidate(invalid_tool_call_count=2),
        None,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )
    executed_params: list[dict[str, object]] = []

    class FakeCursor:
        rowcount = 1

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            if params is not None:
                executed_params.append(params)

        def fetchone(self) -> tuple[str]:
            return ("aawm_tristore",)

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    monkeypatch.setattr(scorer.psycopg, "connect", lambda dsn: FakeConnection())
    args = SimpleNamespace(
        pg_dsn="postgresql://example/aawm_tristore",
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
        require_target_database="aawm_tristore",
        ensure_session_history_score_schema=False,
    )

    assert scorer._update_session_history_scores(args, [evidence]) == 1

    params = executed_params[-1]
    assert params["trace_quality_score"] is None
    assert params["empty_completion_failure"] is None
    assert params["large_tool_result_payload_risk"] is None
    assert params["destructive_checkout_after_work"] is None
    assert params["invalid_tool_call_error"] is True
    assert params["answer_completeness_score"] is None
    assert params["evidence_fidelity_score"] is None
    assert params["tool_result_fidelity_score"] is None
    assert params["error_attribution_quality_score"] is None
    assert params["repetition_loop_risk_score"] is None
    assert params["context_retention_score"] is None
    metadata = json.loads(str(params["score_metadata"]))
    assert "usage_empty_completion_failure" not in metadata
    assert metadata["usage_invalid_tool_call_error"] is True


def test_should_score_read_only_policy_violation_for_mutating_command() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": "Do not edit files and do not run live DB/container commands.",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "timestamp": "2026-05-26T21:05:00Z",
                        "input": {
                            "cmd": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "docker logs litellm-dev"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "I checked it.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.read_only_policy_compliance_score == 0.0
    assert evidence.instruction_adherence_score == 0.0
    assert evidence.read_only_policy_violation_count == 2
    assert evidence.read_only_policy_violation_reasons == [
        "mutating_tool:Bash",
        "live_container_command",
    ]
    assert evidence.read_only_instruction_evidence == [
        {
            "marker": "do_not_edit",
            "message_index": 0,
            "role": "user",
            "instruction_snippet": (
                "Do not edit files and do not run live DB/container commands."
            ),
        },
        {
            "marker": "no_live_db",
            "message_index": 0,
            "role": "user",
            "instruction_snippet": (
                "Do not edit files and do not run live DB/container commands."
            ),
        },
        {
            "marker": "no_live_container",
            "message_index": 0,
            "role": "user",
            "instruction_snippet": (
                "Do not edit files and do not run live DB/container commands."
            ),
        },
    ]
    assert evidence.read_only_policy_violation_evidence[0] == {
        "reason": "mutating_tool:Bash",
        "tool_name": "Bash",
        "message_index": 1,
        "sequence_index": 0,
        "command_snippet": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY",
        "command_timestamp": "2026-05-26T21:05:00Z",
        "affected_paths": ["notes.txt"],
    }
    assert "read_only_policy_violation" in evidence.reasons


def test_should_not_score_read_only_policy_violation_for_local_reads() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": "Read-only investigation only; do not edit files.",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": 'rg -n "session_history" litellm tests'},
                    },
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "scripts/score_agent_trace_quality.py"},
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": "./.venv/bin/python -m pytest tests/test_scripts/test_score_agent_trace_quality.py -q"
                        },
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "No edits made.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.read_only_policy_compliance_score == 1.0
    assert evidence.instruction_adherence_score == 1.0
    assert evidence.read_only_policy_violation_count == 0
    assert "read_only_policy_violation" not in evidence.reasons


def test_should_score_forced_add_of_gitignored_path(tmp_path: Path) -> None:
    repo = _init_ignored_path_repo(tmp_path)
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "timestamp": "2026-05-29T12:00:00Z",
                        "input": {"cmd": "git add -f .analysis/todo.md"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 0.0
    assert evidence.ignored_path_tracking_policy_score == 0.0
    assert evidence.ignored_path_tracking_violation_count == 1
    assert "ignored_path_tracking_policy_violation" in evidence.reasons
    assert evidence.agent_score_reasons["ignored_path_tracking_policy"] == [
        "forced_tracking_ignored_path"
    ]
    assert evidence.ignored_path_tracking_evidence[0]["path"] == ".analysis/todo.md"
    assert evidence.ignored_path_tracking_evidence[0]["ignored_check"] == "tracked_ignored"

    payloads = {
        score["name"]: score for score in (s.payload() for s in scorer.build_langfuse_scores(evidence))
    }
    assert payloads["aawm.agent.ignored_path_tracking_policy"]["value"] == 0.0


def test_should_score_scripted_forced_add_of_gitignored_path(tmp_path: Path) -> None:
    repo = _init_ignored_path_repo(tmp_path)
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "exec_command",
                        "input": {
                            "cmd": (
                                "python - <<'PY'\n"
                                "import subprocess\n"
                                "subprocess.run(['git', 'add', '--force', '.analysis/todo.md'])\n"
                                "PY"
                            )
                        },
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.ignored_path_tracking_policy_score == 0.0
    assert evidence.ignored_path_tracking_violation_count == 1
    assert evidence.ignored_path_tracking_evidence[0]["path"] == ".analysis/todo.md"


def test_should_score_combined_short_force_add_of_gitignored_path(
    tmp_path: Path,
) -> None:
    repo = _init_ignored_path_repo(tmp_path)
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git add -Af .analysis/todo.md"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.ignored_path_tracking_policy_score == 0.0
    assert evidence.ignored_path_tracking_violation_count == 1
    assert evidence.ignored_path_tracking_evidence[0]["path"] == ".analysis/todo.md"


def test_should_score_command_only_pathspec_file_force_tracking() -> None:
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": "git add -f --pathspec-from-file /tmp/paths-to-add"
                        },
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=8),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.ignored_path_tracking_policy_score == 0.0
    assert evidence.ignored_path_tracking_violation_count == 1
    assert evidence.ignored_path_tracking_evidence[0]["evidence_mode"] == (
        "command_only_pathspec_file"
    )


def test_should_allow_explicitly_authorized_ignored_path_tracking(
    tmp_path: Path,
) -> None:
    repo = _init_ignored_path_repo(tmp_path)
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Please force-add the intentionally tracked ignored workflow "
                    "seed file .analysis/todo.md."
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git add -f .analysis/todo.md"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 1.0
    assert evidence.ignored_path_tracking_policy_score == 1.0
    assert evidence.ignored_path_tracking_violation_count == 0
    assert evidence.ignored_path_tracking_evidence == []


def test_should_score_baseline_deflection_attempt_without_incident() -> None:
    payload = _payload(
        [
            {"role": "user", "content": "Fix the classifier failure."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": "./.venv/bin/pytest-classifier scan --changed-file src/app.py"
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git show HEAD~1 -- src/app.py"},
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "apply_patch <<'PATCH'\n*** Begin Patch\nPATCH"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Actionable classifier findings: TS1 type_shape in src/app.py",
                    }
                ],
            },
        ],
        {
            "role": "assistant",
            "content": "This may have been pre-existing, but I patched the current failure.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=24, input_tokens=10_000),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.baseline_deflection_attempted_score == 1.0
    assert evidence.baseline_deflection_incident_score == 0.0
    assert evidence.baseline_deflection_tool_call_count == 1
    assert evidence.quality_gate_trigger_count >= 1
    assert evidence.quality_gate_fix_attempt_count == 1
    assert "baseline_deflection_attempted" in evidence.reasons
    assert "baseline_deflection_incident" not in evidence.reasons


def test_should_score_repeated_baseline_probe_as_incident() -> None:
    probe_blocks = [
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"cmd": command},
        }
        for command in (
            "git show HEAD~1 -- src/app.py",
            "git log -- src/app.py",
            "git blame src/app.py",
            "rg -n source_hash .pytest-classifier/cache",
            "sqlite3 source-analysis.sqlite3 'select schema_version from metadata'",
        )
    ]
    payload = _payload(
        [
            {"role": "user", "content": "Fix the classifier failure."},
            {"role": "assistant", "content": probe_blocks},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "pytest-classifier failed: Actionable classifier findings TS1 type_shape",
                    }
                ],
            },
        ],
        {
            "role": "assistant",
            "content": (
                "I need to prove whether this was already present in the baseline "
                "before deciding whether to fix it."
            ),
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=30, input_tokens=40_000),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.baseline_deflection_attempted_score == 1.0
    assert evidence.baseline_deflection_incident_score == 1.0
    assert evidence.baseline_deflection_tool_call_count == 5
    assert evidence.trace_quality_score == 0.0
    assert "baseline_deflection_incident" in evidence.reasons
    assert evidence.agent_score_reasons["baseline_deflection"] == [
        "baseline_deflection_incident"
    ]


def test_should_not_score_direct_quality_gate_fix_as_baseline_deflection() -> None:
    payload = _payload(
        [
            {"role": "user", "content": "Fix the classifier failure."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "./.venv/bin/pytest-classifier scan"},
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "apply_patch <<'PATCH'\n*** Begin Patch\nPATCH"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Actionable classifier findings: TS1 type_shape",
                    }
                ],
            },
        ],
        {"role": "assistant", "content": "Fixed and reran the gate.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.baseline_deflection_attempted_score == 0.0
    assert evidence.baseline_deflection_incident_score == 0.0
    assert "baseline_deflection_attempted" not in evidence.reasons


def test_should_not_score_fix_what_blocks_you_persona_guidance_as_baseline_deflection() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Fix-what-blocks-you. If a failure (test, lint, type-check, "
                    "build, gate, classifier, pre-commit hook) blocks your work, "
                    "fix it. Do not spend ANY turns investigating whether you "
                    "caused it, whether it is pre-existing, baseline, from a "
                    "prior commit, or not in scope. Attribution does not matter. "
                    "Apply the smallest fix that resolves the failure. Never "
                    "report this is pre-existing, this is baseline, not from my "
                    "changes, or similar as a completion state."
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "./.venv/bin/pytest-classifier scan"},
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "apply_patch <<'PATCH'\n*** Begin Patch\nPATCH"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Actionable classifier findings: TS1 type_shape",
                    }
                ],
            },
        ],
        {"role": "assistant", "content": "Fixed and reran the gate.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.baseline_deflection_attempted_score == 0.0
    assert evidence.baseline_deflection_incident_score == 0.0
    assert evidence.quality_gate_trigger_count >= 1
    assert evidence.quality_gate_fix_attempt_count == 1
    assert "baseline_deflection_attempted" not in evidence.reasons


def test_should_score_unsolicited_sleep_interruption() -> None:
    payload = _payload(
        [{"role": "user", "content": "Continue fixing the release blocker."}],
        {
            "role": "assistant",
            "content": "It's late. Go get some rest and we'll pick back up in the morning.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=18, input_tokens=12_000),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.sleep_wellness_interruption_attempted_score == 1.0
    assert evidence.sleep_wellness_interruption_incident_score == 1.0
    assert evidence.sleep_wellness_interruption_count >= 1
    assert evidence.trace_quality_score == 0.0
    assert "sleep_wellness_interruption_incident" in evidence.reasons


def test_should_score_sleep_interruption_after_user_pushback() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": "Stop telling me to sleep. Continue the deployment.",
            }
        ],
        {
            "role": "assistant",
            "content": "You need rest. Go to bed and continue tomorrow.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=18),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.sleep_wellness_interruption_attempted_score == 1.0
    assert evidence.sleep_wellness_interruption_incident_score == 1.0
    assert evidence.sleep_wellness_interruption_after_user_pushback_count == 1


def test_should_not_score_user_requested_sleep_advice() -> None:
    payload = _payload(
        [{"role": "user", "content": "I need sleep advice for a healthy schedule."}],
        {
            "role": "assistant",
            "content": "You should get some sleep and continue tomorrow.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=18),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.sleep_wellness_interruption_attempted_score == 0.0
    assert evidence.sleep_wellness_interruption_incident_score == 0.0
    assert "sleep_wellness_interruption_attempted" not in evidence.reasons


def test_should_not_score_sleep_phrase_catalog_discussion() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "We are designing scoring for sleep interruptions. "
                    "List common phrases models use."
                ),
            }
        ],
        {
            "role": "assistant",
            "content": "Common phrases include go to bed and get some rest.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=18),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.sleep_wellness_interruption_attempted_score == 0.0
    assert evidence.sleep_wellness_interruption_incident_score == 0.0


def test_should_score_missing_discovery_inventory_candidate() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Read `.analysis/seed-handoff.md` and any recent handoff "
                    "docs. Discovery inventory required: list the discovery "
                    "commands, list every candidate file, mark each candidate as "
                    "inspected, omitted, or unavailable, classify relevant "
                    "candidates, and call out any coverage gap."
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": "find .analysis -maxdepth 1 -name '*handoff*.md' -print"
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": (
                            ".analysis/seed-handoff.md\n"
                            ".analysis/recent-handoff.md\n"
                        ),
                    }
                ],
            },
        ],
        {
            "role": "assistant",
            "content": (
                "Discovery command: find .analysis -maxdepth 1 -name '*handoff*.md' "
                "-print. Candidates: .analysis/seed-handoff.md inspected, "
                "actionable. Coverage gaps: one discovered handoff was not "
                "reviewed yet."
            ),
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=64),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.discovery_inventory_coverage_score == 0.0
    assert evidence.discovery_inventory_missing_count == 1
    assert evidence.trace_quality_score == 0.0
    assert "discovery_inventory_coverage_failure" in evidence.reasons
    assert (
        "omitted_discovered_candidate:.analysis/recent-handoff.md"
        in evidence.agent_score_reasons["discovery_inventory_coverage"]
    )


def test_should_pass_complete_discovery_inventory() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Inspect `.analysis/seed-handoff.md` plus related docs. "
                    "Discovery inventory required: list the discovery command, "
                    "list every candidate, mark each candidate as inspected, "
                    "omitted, or unavailable, classify relevant candidates, and "
                    "call out any coverage gap."
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": "find .analysis -maxdepth 1 -name '*handoff*.md' -print"
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": (
                            ".analysis/seed-handoff.md\n"
                            ".analysis/recent-handoff.md\n"
                        ),
                    }
                ],
            },
        ],
        {
            "role": "assistant",
            "content": (
                "Discovery command: find .analysis -maxdepth 1 -name '*handoff*.md' "
                "-print. Candidates: .analysis/seed-handoff.md inspected "
                "actionable; .analysis/recent-handoff.md inspected stale. "
                "Coverage gaps: none."
            ),
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=64),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.discovery_inventory_coverage_score == 1.0
    assert evidence.discovery_inventory_missing_count == 0
    assert "discovery_inventory_coverage_failure" not in evidence.reasons


def test_should_leave_narrow_discovery_prompt_unscored() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "No broad discovery inventory is required; inspect only "
                    "`.analysis/seed-handoff.md`."
                ),
            }
        ],
        {
            "role": "assistant",
            "content": "Inspected .analysis/seed-handoff.md.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.discovery_inventory_coverage_score is None
    assert evidence.discovery_inventory_missing_count is None


def test_agent_quality_rule_catalog_reloads_and_keeps_last_valid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog_path = tmp_path / "rules.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "reload_ttl_seconds": 60,
                "sleep_wellness_interruption": {
                    "candidate_phrases": ["power down for tonight"]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AAWM_AGENT_QUALITY_RULES_PATH", str(catalog_path))
    aawm_agent_quality_rules.reset_agent_quality_rule_catalog_cache()

    catalog = aawm_agent_quality_rules.load_agent_quality_rule_catalog(now=0)
    assert catalog["version"] == "test-v1"
    assert "power down for tonight" in catalog["sleep_wellness_interruption"][
        "candidate_phrases"
    ]

    catalog_path.write_text("{not-json", encoding="utf-8")
    same_catalog = aawm_agent_quality_rules.load_agent_quality_rule_catalog(now=120)
    assert same_catalog["version"] == "test-v1"

    catalog_path.write_text(
        json.dumps(
            {
                "version": "test-v2",
                "reload_ttl_seconds": 60,
                "sleep_wellness_interruption": {
                    "candidate_phrases": ["put a pin in it until morning"]
                },
            }
        ),
        encoding="utf-8",
    )
    updated_catalog = aawm_agent_quality_rules.load_agent_quality_rule_catalog(now=181)
    assert updated_catalog["version"] == "test-v2"
    assert "put a pin in it until morning" in updated_catalog[
        "sleep_wellness_interruption"
    ]["candidate_phrases"]


def test_should_score_codex_transcript_exec_command_read_only_violation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    _write_jsonl(
        parent,
        [
            {
                "timestamp": "2026-05-27T01:04:21Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "call-spawn",
                    "arguments": json.dumps(
                        {
                            "message": (
                                "Read .analysis/todo.md first. Do not edit files "
                                "and do not run live DB/container commands."
                            )
                        }
                    ),
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-spawn",
                    "output": json.dumps(
                        {
                            "agent_id": "019e66f6-0ba8-7491-9e6a-d446cc1cab59",
                            "nickname": "Planck",
                        }
                    ),
                },
            },
        ],
    )
    _write_jsonl(
        child,
        [
            {
                "timestamp": "2026-05-27T01:04:22Z",
                "type": "session_meta",
                "payload": {
                    "id": "019e66f6-0ba8-7491-9e6a-d446cc1cab59",
                    "timestamp": "2026-05-27T01:04:21Z",
                    "cwd": "/home/zepfu/projects/aawm-tap",
                    "originator": "codex-tui",
                    "cli_version": "0.133.0",
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": (
                                    "019e4d93-b4d0-7fb0-9b04-0c57da041e0d"
                                )
                            }
                        }
                    },
                    "thread_source": "subagent",
                    "agent_nickname": "Planck",
                    "agent_role": "explorer",
                    "model_provider": "litellm",
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "turn_context",
                "payload": {
                    "turn_id": "019e66f6-1181-7b10-8457-1905ac7f1f08",
                    "cwd": "/home/zepfu/projects/aawm-tap",
                    "model": "aawm-codex-agent-auto",
                },
            },
            {
                "timestamp": "2026-05-27T01:05:31Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": (
                                "cat << 'EOF' > update_json.py\n"
                                "open('notes.txt', 'w').write('x')\n"
                                "EOF\npython update_json.py\nrm update_json.py"
                            )
                        }
                    ),
                },
            },
            {
                "timestamp": "2026-05-27T01:06:10Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "docker logs litellm-dev"}),
                },
            },
            {
                "timestamp": "2026-05-27T01:06:24Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "completed_at": 1779843984,
                    "duration_ms": 121958,
                    "last_agent_message": "I changed the files and tests pass.",
                },
            },
        ],
    )

    bundle = scorer._build_codex_transcript_bundle(child, parent_transcript=parent)
    evidence = scorer.score_candidate(
        bundle.candidate,
        bundle.payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert bundle.candidate.repository == "aawm-tap"
    assert bundle.candidate.agent_name == "Planck"
    assert bundle.candidate.tool_call_count == 2
    assert evidence.read_only_policy_compliance_score == 0.0
    assert evidence.instruction_adherence_score == 0.0
    assert evidence.destructive_action_policy_score == 0.0
    assert evidence.read_only_policy_violation_count == 2
    assert evidence.read_only_policy_violation_reasons == [
        "mutating_tool:exec_command",
        "live_container_command",
    ]
    assert set(evidence.read_only_policy_violation_evidence[0]["affected_paths"]) == {
        "notes.txt",
        "update_json.py",
    }


def test_should_not_score_codex_transcript_safe_exec_command_reads(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    _write_jsonl(
        parent,
        [
            {
                "timestamp": "2026-05-27T01:04:21Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "call-spawn",
                    "arguments": json.dumps(
                        {"message": "Read-only investigation only; do not edit files."}
                    ),
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-spawn",
                    "output": json.dumps({"agent_id": "child-session"}),
                },
            },
        ],
    )
    _write_jsonl(
        child,
        [
            {
                "timestamp": "2026-05-27T01:04:22Z",
                "type": "session_meta",
                "payload": {
                    "id": "child-session",
                    "timestamp": "2026-05-27T01:04:21Z",
                    "cwd": "/home/zepfu/projects/litellm",
                    "originator": "codex-tui",
                    "model_provider": "litellm",
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "turn_context",
                "payload": {"model": "aawm-codex-agent-auto"},
            },
            {
                "timestamp": "2026-05-27T01:05:31Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": (
                                "rg -n \"session_history\" scripts "
                                "tests/test_scripts/test_score_agent_trace_quality.py"
                            )
                        }
                    ),
                },
            },
            {
                "timestamp": "2026-05-27T01:06:24Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "completed_at": 1779843984,
                    "duration_ms": 1000,
                    "last_agent_message": "No edits were needed.",
                },
            },
        ],
    )

    bundle = scorer._build_codex_transcript_bundle(child, parent_transcript=parent)
    evidence = scorer.score_candidate(
        bundle.candidate,
        bundle.payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.read_only_policy_compliance_score == 1.0
    assert evidence.read_only_policy_violation_count == 0
    assert "read_only_policy_violation" not in evidence.reasons


def test_should_score_codex_transcript_null_final_with_partial_progress(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child-null-final.jsonl"
    _write_jsonl(
        child,
        [
            {
                "timestamp": "2026-05-27T01:04:22Z",
                "type": "session_meta",
                "payload": {
                    "id": "child-null-final",
                    "timestamp": "2026-05-27T01:04:21Z",
                    "cwd": "/home/zepfu/projects/litellm",
                    "originator": "codex-tui",
                    "model_provider": "litellm",
                    "agent_nickname": "Curie",
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "turn_context",
                "payload": {"model": "aawm-codex-agent-auto"},
            },
            {
                "timestamp": "2026-05-27T01:05:00Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I found the relevant callback file.",
                        }
                    ],
                },
            },
            {
                "timestamp": "2026-05-27T01:05:31Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "rg -n session_history scripts"}),
                },
            },
            {
                "timestamp": "2026-05-27T01:06:24Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "completed_at": 1779843984,
                    "duration_ms": 121958,
                    "last_agent_message": None,
                },
            },
        ],
    )

    bundle = scorer._build_codex_transcript_bundle(child)
    evidence = scorer.score_candidate(
        bundle.candidate,
        bundle.payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert bundle.candidate.metadata["codex_transcript_terminal_state"] == (
        "null_final_message"
    )
    assert bundle.candidate.metadata[
        "codex_transcript_non_empty_assistant_progress_count"
    ] == 1
    assert evidence.terminal_completion_score == 0.0
    assert evidence.response_meaningfulness_score == 0.0
    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.task_progress_score == 1.0
    assert evidence.trace_quality_score == 0.0
    assert evidence.agent_score_reasons["terminal_completion"] == [
        "null_final_message"
    ]


def test_should_score_codex_transcript_unsupported_tool_loop(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child-unsupported-tools.jsonl"
    _write_jsonl(
        child,
        [
            {
                "timestamp": "2026-05-27T01:04:22Z",
                "type": "session_meta",
                "payload": {
                    "id": "child-unsupported-tools",
                    "timestamp": "2026-05-27T01:04:21Z",
                    "cwd": "/home/zepfu/projects/litellm",
                    "originator": "codex-tui",
                    "model_provider": "litellm",
                },
            },
            {
                "timestamp": "2026-05-27T01:04:23Z",
                "type": "turn_context",
                "payload": {"model": "aawm-codex-agent-auto"},
            },
            {
                "timestamp": "2026-05-27T01:05:31Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "run_command",
                    "arguments": json.dumps({"command": "ls"}),
                },
            },
            {
                "timestamp": "2026-05-27T01:05:35Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "list_dir",
                    "arguments": json.dumps({"path": "."}),
                },
            },
            {
                "timestamp": "2026-05-27T01:06:24Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "completed_at": 1779843984,
                    "duration_ms": 1000,
                    "last_agent_message": None,
                },
            },
        ],
    )

    bundle = scorer._build_codex_transcript_bundle(child)
    evidence = scorer.score_candidate(
        bundle.candidate,
        bundle.payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert bundle.candidate.invalid_tool_call_count == 2
    assert bundle.candidate.metadata["codex_transcript_unsupported_tool_names"] == [
        "list_dir",
        "run_command",
    ]
    assert evidence.tool_use_validity_score == 0.0
    assert evidence.tool_error_recovery_score == 0.0
    assert "unsupported_tool_call:list_dir" in evidence.invalid_tool_call_error_markers
    assert "unsupported_tool_call:run_command" in evidence.invalid_tool_call_error_markers


def test_should_upsert_codex_transcript_session_history_for_missing_native_row(
    monkeypatch,
) -> None:
    candidate = _candidate(
        row_id=-1,
        trace_id="child-session",
        session_id="child-session",
        litellm_call_id="codex-transcript:child-session",
        source_observation_id="codex-transcript:child-session",
        provider="litellm",
        model="aawm-codex-agent-auto",
        agent_name="Planck",
        agent_id="019e66f6-0ba8-7491-9e6a-d446cc1cab59",
        repository="aawm-tap",
        tenant_id="aawm-tap",
        output_tokens=12,
        tool_call_count=1,
        metadata={
            "source": "codex_transcript",
            "codex_transcript_path": "/tmp/child.jsonl",
            "originator": "codex-tui",
            "cli_version": "0.133.0",
        },
    )
    payload = _payload(
        [
            {"role": "user", "content": "Do not edit files."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "exec_command",
                        "input": {"cmd": "sed -i 's/a/b/' config/example.json"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Updated it.", "tool_calls": None},
    )
    evidence = scorer.score_candidate(
        candidate,
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )
    executed: list[tuple[str, dict[str, object] | None]] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            executed.append((sql, params))

        def fetchone(self) -> tuple[object, ...]:
            if len(executed) == 1:
                return ("aawm_tristore",)
            return (123,)

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    monkeypatch.setattr(scorer.psycopg, "connect", lambda dsn: FakeConnection())
    args = SimpleNamespace(
        pg_dsn="postgresql://example/aawm_tristore",
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
        require_target_database="aawm_tristore",
        ensure_session_history_score_schema=False,
    )

    assert scorer._upsert_codex_transcript_session_history(
        args,
        [(candidate, evidence)],
    ) == 1

    sql, params = executed[-1]
    assert "INSERT INTO public.session_history" in sql
    assert "ON CONFLICT (litellm_call_id) DO UPDATE" in sql
    assert params is not None
    assert params["litellm_call_id"] == "codex-transcript:child-session"
    assert params["session_id"] == "child-session"
    assert params["repository"] == "aawm-tap"
    assert params["agent_id"] == "019e66f6-0ba8-7491-9e6a-d446cc1cab59"
    assert params["provider"] is None
    assert params["model"] == "codex-transcript"
    assert params["model_group"] is None
    assert params["read_only_policy_compliance_score"] == 0.0
    metadata = json.loads(str(params["metadata"]))
    assert metadata["synthetic"] is True
    assert metadata["session_history_usage_record"] is False
    assert metadata["session_history_reporting_excluded"] is True
    assert (
        metadata["session_history_reporting_exclusion_reason"]
        == "synthetic_codex_transcript"
    )
    assert (
        metadata["session_history_repair_source"]
        == "d1_159_codex_transcript_policy_scoring"
    )
    assert metadata["source"] == "codex_transcript"
    assert metadata["agent_id"] == "019e66f6-0ba8-7491-9e6a-d446cc1cab59"
    assert metadata["agent_id_source"] == "codex_transcript.session_meta.id"
    assert metadata["usage_read_only_policy_compliance_score"] == 0.0


def test_should_keep_legitimate_one_token_text_meaningful() -> None:
    payload = _payload(
        [{"role": "user", "content": "Answer yes or no: did the smoke pass?"}],
        {"role": "assistant", "content": "yes", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=1),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.response_meaningfulness_score == 1.0
    assert evidence.output_contract_compliance_score == 1.0
    assert evidence.task_progress_score == 1.0


def test_should_score_one_token_noop_response_as_not_meaningful() -> None:
    payload = _payload(
        [{"role": "user", "content": "Do the task."}],
        {"role": "assistant", "content": "", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=1),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.response_meaningfulness_score == 0.0
    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.task_progress_score == 0.0
    assert evidence.agent_score_reasons["response_meaningfulness"] == [
        "no_meaningful_output"
    ]


def test_should_score_one_token_ack_text_as_not_meaningful() -> None:
    payload = _payload(
        [{"role": "user", "content": "Do the task."}],
        {"role": "assistant", "content": "ok", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=1),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.response_meaningfulness_score == 0.0
    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.task_progress_score == 0.0


def test_should_score_malformed_success_output_shape_as_contract_failure() -> None:
    payload = _payload(
        [{"role": "user", "content": "Return a normal answer."}],
        {"choices": []},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=3),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.agent_score_reasons["output_contract_compliance"] == [
        "malformed_response_shape"
    ]


def test_should_score_missing_required_final_phrase_as_contract_failure() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Read-only task. Do not edit files.\n\n"
                    'Your final answer must truthfully include: "No files were modified."'
                ),
            }
        ],
        {
            "role": "assistant",
            "content": "I inspected the scorer and found the insertion point.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 0.0
    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.output_contract_required_final_phrase == "No files were modified."
    assert evidence.output_contract_required_final_phrase_present is False
    assert evidence.output_contract_failure_class == "missing_required_final_phrase"
    assert evidence.output_contract_failure_count == 1
    assert evidence.agent_score_reasons["output_contract_compliance"] == [
        "missing_required_final_phrase"
    ]
    _, score_metadata = scorer._session_history_score_values(evidence)
    assert (
        score_metadata["usage_output_contract_required_final_phrase"]
        == "No files were modified."
    )
    assert (
        score_metadata["usage_output_contract_required_final_phrase_present"] is False
    )
    assert (
        score_metadata["usage_output_contract_failure_class"]
        == "missing_required_final_phrase"
    )


def test_should_record_required_final_phrase_present() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": (
                    "Read-only task. Do not edit files.\n\n"
                    'Your final answer must truthfully include: "No files were modified."'
                ),
            }
        ],
        {
            "role": "assistant",
            "content": "Inspected the scorer. No files were modified.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=12),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 1.0
    assert evidence.output_contract_compliance_score == 1.0
    assert evidence.output_contract_required_final_phrase == "No files were modified."
    assert evidence.output_contract_required_final_phrase_present is True
    assert evidence.output_contract_failure_class is None


def test_should_score_setup_only_completion_as_contract_failure() -> None:
    payload = _payload(
        [{"role": "user", "content": "Read-only task. Inspect the callback path."}],
        {
            "role": "assistant",
            "content": "I will inspect the callback path now.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=9),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.trace_quality_score == 0.0
    assert evidence.response_meaningfulness_score == 1.0
    assert evidence.output_contract_compliance_score == 0.0
    assert evidence.output_contract_failure_class == "setup_only_completion"
    assert evidence.output_contract_setup_only_detected is True
    assert "i will inspect" in evidence.output_contract_setup_only_markers
    assert evidence.agent_score_reasons["output_contract_compliance"] == [
        "setup_only_completion"
    ]


def test_should_score_scope_control_escape_paths() -> None:
    payload = _payload(
        [
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "sed -n '1,10p' ../other-repo/app.py"},
                    }
                ],
            },
        ],
        {"role": "assistant", "content": "Found it.", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=8, repository="zepfu/litellm"),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.scope_control_score == 0.0
    assert evidence.agent_score_reasons["scope_control"] == [
        "path_scope_escape:../other-repo/app.py"
    ]


def test_should_score_long_low_output_completion_as_stall_risk() -> None:
    payload = _payload(
        [{"role": "user", "content": "Do the task."}],
        {"role": "assistant", "content": "", "tool_calls": None},
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=1, llm_upstream_elapsed_ms=31 * 60 * 1000),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.stall_risk_score == 1.0
    assert evidence.agent_score_reasons["stall_risk"] == ["long_elapsed_low_output"]


def test_should_score_additional_content_quality_dimensions() -> None:
    payload = _payload(
        [
            {
                "role": "user",
                "content": "Do not edit files. Investigate the provider error and explain the root cause.",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "timestamp": "2026-05-26T21:05:00Z",
                        "input": {
                            "cmd": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "timestamp": "2026-05-26T21:06:00Z",
                        "input": {
                            "cmd": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
                        },
                    },
                ],
            },
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
            },
        ],
        {
            "role": "assistant",
            "content": "Recovered from the tool error and verified the provider issue.",
            "tool_calls": None,
        },
    )

    evidence = scorer.score_candidate(
        _candidate(output_tokens=18),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert evidence.answer_completeness_score == 1.0
    assert evidence.evidence_fidelity_score == 1.0
    assert evidence.tool_result_fidelity_score == 1.0
    assert evidence.error_attribution_quality_score == 1.0
    assert evidence.repetition_loop_risk_score == 1.0
    assert evidence.context_retention_score == 0.0
    assert evidence.agent_score_reasons["repetition_loop_risk"] == [
        "repeated_tool_command:bash:python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
    ]
    assert evidence.agent_score_reasons["context_retention"] == [
        "lost_active_constraints"
    ]

    payloads = {score["name"]: score for score in (s.payload() for s in scorer.build_langfuse_scores(evidence))}
    assert payloads["aawm.agent.answer_completeness"]["value"] == 1.0
    assert payloads["aawm.agent.evidence_fidelity"]["value"] == 1.0
    assert payloads["aawm.agent.tool_result_fidelity"]["value"] == 1.0
    assert payloads["aawm.agent.error_attribution_quality"]["value"] == 1.0
    assert payloads["aawm.agent.repetition_loop_risk"]["value"] == 1.0
    assert payloads["aawm.agent.context_retention"]["value"] == 0.0


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
        "aawm.agent.answer_completeness",
        "aawm.agent.response_meaningfulness",
        "aawm.agent.evidence_fidelity",
        "aawm.agent.tool_result_fidelity",
        "aawm.agent.tool_use_validity",
        "aawm.agent.tool_error_recovery",
        "aawm.agent.output_contract_compliance",
        "aawm.agent.task_progress",
        "aawm.agent.scope_control",
        "aawm.agent.ignored_path_tracking_policy",
        "aawm.agent.baseline_deflection_attempted",
        "aawm.agent.baseline_deflection_incident",
        "aawm.agent.sleep_wellness_interruption_attempted",
        "aawm.agent.sleep_wellness_interruption_incident",
    }
    assert all(payload["traceId"] == "trace-1" for payload in payloads)
    assert all(payload["observationId"] == "obs-1" for payload in payloads)
    assert len({payload["id"] for payload in payloads}) == 19


def test_should_persist_additional_content_quality_scores_to_session_history(
    monkeypatch,
) -> None:
    evidence = scorer.score_candidate(
        _candidate(output_tokens=18),
        _payload(
            [
                {
                    "role": "user",
                    "content": "Do not edit files. Investigate the provider error and explain the root cause.",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "timestamp": "2026-05-26T21:05:00Z",
                            "input": {
                                "cmd": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
                            },
                        },
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "timestamp": "2026-05-26T21:06:00Z",
                            "input": {
                                "cmd": "python - <<'PY'\nopen('notes.txt', 'w').write('x')\nPY"
                            },
                        },
                    ],
                },
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
                },
            ],
            {
                "role": "assistant",
                "content": "Recovered from the tool error and verified the provider issue.",
                "tool_calls": None,
            },
        ),
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )
    executed_params: list[dict[str, object]] = []

    class FakeCursor:
        rowcount = 1

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            if params is not None:
                executed_params.append(params)

        def fetchone(self) -> tuple[str]:
            return ("aawm_tristore",)

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    monkeypatch.setattr(scorer.psycopg, "connect", lambda dsn: FakeConnection())
    args = SimpleNamespace(
        pg_dsn="postgresql://example/aawm_tristore",
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
        require_target_database="aawm_tristore",
        ensure_session_history_score_schema=False,
    )

    assert scorer._update_session_history_scores(args, [evidence]) == 1

    params = executed_params[-1]
    assert params["answer_completeness_score"] == 1.0
    assert params["evidence_fidelity_score"] == 1.0
    assert params["tool_result_fidelity_score"] == 1.0
    assert params["error_attribution_quality_score"] == 1.0
    assert params["repetition_loop_risk_score"] == 1.0
    assert params["context_retention_score"] == 0.0
    metadata = json.loads(str(params["score_metadata"]))
    assert metadata["usage_answer_completeness_score"] == 1.0
    assert metadata["usage_evidence_fidelity_score"] == 1.0
    assert metadata["usage_tool_result_fidelity_score"] == 1.0
    assert metadata["usage_error_attribution_quality_score"] == 1.0
    assert metadata["usage_repetition_loop_risk_score"] == 1.0
    assert metadata["usage_context_retention_score"] == 0.0


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
