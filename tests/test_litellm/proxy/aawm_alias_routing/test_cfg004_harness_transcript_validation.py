"""CFG-004 harness transcript + OAuth descriptor validation tests.

Covers:
- OAuth account descriptor extraction (secret-safe, fail-closed).
- Exact thread-ID transcript correlation.
- Real Codex 0.146 rollout-shaped transcript parsing.
- Transcript collaboration validation with actual event shapes.
- Non-vacuous parallel proof (same turn, calls before outputs).
- Pass-3-like zero-child-command failure.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib


# ---------------------------------------------------------------------------
# Load the harness module by path (same pattern as sibling tests).
# ---------------------------------------------------------------------------

_HARNESS_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "scripts"
    / "local-ci"
    / "run_anthropic_adapter_acceptance.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "run_anthropic_adapter_acceptance", _HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HARNESS = _load_harness()


# ---------------------------------------------------------------------------
# Helpers: real Codex 0.146 rollout-shaped fixture builders
# ---------------------------------------------------------------------------


def _write_jsonl(path: pathlib.Path, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in objects:
            fh.write(json.dumps(obj) + "\n")


_EXACT_CHILD_PROMPT = (
    "You are a deterministic acceptance harness child. Do not spawn additional agents. "
    "Your first tool response must contain exactly three exec_command calls in one "
    "parallel batch with no text: pwd, git rev-parse --show-toplevel, git status --short. "
    "All commands must run in the repository root. Do not wait for one result before "
    "issuing the other calls. After all three results succeed, use no more tools and "
    "return exactly BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED with no other text, no "
    "Markdown, no code fences, and no explanation."
)


def _make_parent_transcript(
    thread_id: str,
    *,
    spawn: bool = True,
    wait: bool = True,
    child_thread_id: str | None = "child-thread-001",
    wait_timed_out: bool = False,
    parent_marker: str | None = None,
    spawn_message: str = _EXACT_CHILD_PROMPT,
) -> list[dict]:
    """Build a parent rollout transcript using real Codex 0.146 shapes."""
    turn_id = f"turn-{thread_id}"
    lines: list[dict] = [
        {
            "timestamp": "2026-08-02T05:57:50.003Z",
            "type": "session_meta",
            "payload": {
                "session_id": thread_id,
                "id": thread_id,
                "timestamp": "2026-08-02T05:57:49.632Z",
                "cwd": "/home/zepfu/projects/litellm",
                "originator": "codex_exec",
                "cli_version": "0.146.0",
                "source": "exec",
                "thread_source": "user",
            },
        },
        {
            "timestamp": "2026-08-02T05:57:50.009Z",
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": turn_id,
                "started_at": 1785650269,
            },
        },
    ]
    if spawn:
        lines.append({
            "timestamp": "2026-08-02T05:57:58.615Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc_spawn_1",
                "name": "spawn_agent",
                "namespace": "collaboration",
                "arguments": json.dumps({
                    "task_name": "basic_alias_child",
                    "agent_type": "opencode",
                    "model": "basic",
                    "fork_turns": "none",
                    "message": spawn_message,
                }),
                "call_id": "call_spawn_1",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
        # sub_agent_activity event correlates spawn call_id to child thread.
        if child_thread_id:
            lines.append({
                "timestamp": "2026-08-02T05:57:58.814Z",
                "type": "event_msg",
                "payload": {
                    "type": "sub_agent_activity",
                    "event_id": "call_spawn_1",
                    "occurred_at_ms": 1785650278814,
                    "agent_thread_id": child_thread_id,
                    "agent_path": "/root/basic_alias_child",
                    "kind": "started",
                },
            })
        lines.append({
            "timestamp": "2026-08-02T05:57:58.817Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "id": "fco_spawn_1",
                "call_id": "call_spawn_1",
                "output": json.dumps(
                    {"task_name": "/root/basic_alias_child"}
                ),
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
    if wait:
        lines.append({
            "timestamp": "2026-08-02T05:58:01.252Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc_wait_1",
                "name": "wait_agent",
                "namespace": "collaboration",
                "arguments": json.dumps({"timeout_ms": 3600000}),
                "call_id": "call_wait_1",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
        lines.append({
            "timestamp": "2026-08-02T05:58:04.746Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "id": "fco_wait_1",
                "call_id": "call_wait_1",
                "output": json.dumps({
                    "message": "Wait completed.",
                    "timed_out": wait_timed_out,
                }),
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
    # Inbound inter-agent message (must NOT count as terminal evidence).
    lines.append({
        "timestamp": "2026-08-02T05:58:04.753Z",
        "type": "response_item",
        "payload": {
            "type": "agent_message",
            "id": "amsg_inbound_1",
            "author": "/root/basic_alias_child",
            "recipient": "/root",
            "content": [
                {
                    "type": "input_text",
                    "text": "Message Type: FINAL_ANSWER\nTask name: /root\n"
                    "Sender: /root/basic_alias_child\nPayload:\nCOMPLETE",
                }
            ],
            "internal_chat_message_metadata_passthrough": {
                "turn_id": turn_id,
            },
        },
    })
    if parent_marker:
        # Outbound terminal: event_msg.agent_message
        lines.append({
            "timestamp": "2026-08-02T05:58:09.857Z",
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": parent_marker,
                "phase": "final_answer",
                "memory_citation": None,
            },
        })
        # Outbound terminal: response_item.message(role=assistant)
        lines.append({
            "timestamp": "2026-08-02T05:58:09.861Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg_final_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": parent_marker}],
                "phase": "final_answer",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
        # task_complete with last_agent_message
        lines.append({
            "timestamp": "2026-08-02T05:58:09.887Z",
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": parent_marker,
                "started_at": 1785650269,
                "completed_at": 1785650289,
                "duration_ms": 19908,
            },
        })
    return lines


def _make_child_transcript(
    child_id: str,
    parent_thread_id: str,
    *,
    command_list: list[str] | None = None,
    exit_codes: list[int] | None = None,
    multi_turn: bool = False,
    terminal_marker: str | None = None,
    sequential_call_output: bool = False,
    inbound_prompt: str | None = None,
    encrypted_inbound: bool = False,
    include_turn_context: bool = True,
    turn_context_overrides: dict | None = None,
) -> list[dict]:
    """Build a child rollout transcript using real Codex 0.146 shapes.

    Commands are represented as correlated function_call(exec_command)
    plus function_call_output pairs with exit code in output text.
    """
    turn_id = f"turn-{child_id}"
    lines: list[dict] = [
        {
            "timestamp": "2026-08-02T05:57:58.823Z",
            "type": "session_meta",
            "payload": {
                "session_id": parent_thread_id,
                "id": child_id,
                "parent_thread_id": parent_thread_id,
                "timestamp": "2026-08-02T05:57:58.635Z",
                "cwd": "/home/zepfu/projects/litellm",
                "originator": "codex_exec",
                "cli_version": "0.146.0",
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": parent_thread_id,
                            "depth": 1,
                            "agent_path": "/root/basic_alias_child",
                            "agent_nickname": "Boyle",
                            "agent_role": "opencode",
                        }
                    }
                },
                "thread_source": "subagent",
            },
        },
        {
            "timestamp": "2026-08-02T05:57:59.610Z",
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": turn_id,
                "started_at": 1785650278,
            },
        },
    ]
    if include_turn_context:
        # Real Codex 0.146 top-level turn_context record.
        _tc_inner: dict = {
            "turn_id": turn_id,
            "model": "opencode_zen/big-pickle",
            "sandbox_policy": {"type": "read-only"},
        }
        if turn_context_overrides:
            # Allow overrides to set model directly and sandbox via
            # sandbox_policy.type or flat "sandbox" key.
            for _k, _v in turn_context_overrides.items():
                if _k == "sandbox":
                    _tc_inner["sandbox_policy"] = {"type": _v}
                else:
                    _tc_inner[_k] = _v
        lines.append({
            "timestamp": "2026-08-02T05:57:59.615Z",
            "type": "turn_context",
            "payload": _tc_inner,
        })
    # Inbound NEW_TASK assignment message.
    _inbound_text = inbound_prompt if inbound_prompt is not None else _EXACT_CHILD_PROMPT
    if encrypted_inbound:
        _inbound_text = "gAAAAABmEncryptedPayloadHere=="
    lines.append({
        "timestamp": "2026-08-02T05:57:59.620Z",
        "type": "response_item",
        "payload": {
            "type": "agent_message",
            "id": "amsg_newtask_1",
            "author": "/root",
            "recipient": "/root/basic_alias_child",
            "content": [
                {
                    "type": "input_text",
                    "text": "Message Type: NEW_TASK\n"
                    "Task name: /root/basic_alias_child\n"
                    "Sender: /root\nPayload:\n" + _inbound_text,
                }
            ],
            "internal_chat_message_metadata_passthrough": {
                "turn_id": turn_id,
            },
        },
    })
    actual_commands = command_list if command_list is not None else []
    exit_codes = exit_codes or []

    if sequential_call_output:
        # Sequential: each call immediately followed by its output.
        for i, cmd in enumerate(actual_commands):
            ec = exit_codes[i] if i < len(exit_codes) else 0
            tid = turn_id if not multi_turn else f"turn-{child_id}-{i}"
            if multi_turn and i > 0:
                lines.append({
                    "timestamp": f"2026-08-02T05:58:0{i}.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": tid,
                        "started_at": 1785650280 + i,
                    },
                })
            lines.append({
                "timestamp": f"2026-08-02T05:58:0{i}.100Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": f"fc_exec_{i}",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": cmd}),
                    "call_id": f"call_exec_{i}",
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": tid,
                    },
                },
            })
            lines.append({
                "timestamp": f"2026-08-02T05:58:0{i}.200Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "id": f"fco_exec_{i}",
                    "call_id": f"call_exec_{i}",
                    "output": (
                        f"Chunk ID: abc{i}\nWall time: 0.0001 seconds\n"
                        f"Process exited with code {ec}\n"
                        f"Original token count: 1\nOutput:\nok\n"
                    ),
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": tid,
                    },
                },
            })
    else:
        # Parallel: all calls first, then all outputs.
        for i, cmd in enumerate(actual_commands):
            tid = turn_id if not multi_turn else f"turn-{child_id}-{i}"
            if multi_turn and i > 0:
                lines.append({
                    "timestamp": f"2026-08-02T05:58:0{i}.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": tid,
                        "started_at": 1785650280 + i,
                    },
                })
            lines.append({
                "timestamp": f"2026-08-02T05:58:00.{100+i}Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": f"fc_exec_{i}",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": cmd}),
                    "call_id": f"call_exec_{i}",
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": tid,
                    },
                },
            })
        for i, cmd in enumerate(actual_commands):
            ec = exit_codes[i] if i < len(exit_codes) else 0
            tid = turn_id if not multi_turn else f"turn-{child_id}-{i}"
            lines.append({
                "timestamp": f"2026-08-02T05:58:01.{100+i}Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "id": f"fco_exec_{i}",
                    "call_id": f"call_exec_{i}",
                    "output": (
                        f"Chunk ID: abc{i}\nWall time: 0.0001 seconds\n"
                        f"Process exited with code {ec}\n"
                        f"Original token count: 1\nOutput:\nok\n"
                    ),
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": tid,
                    },
                },
            })

    if terminal_marker:
        # Outbound terminal: event_msg.agent_message
        lines.append({
            "timestamp": "2026-08-02T05:58:04.738Z",
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": terminal_marker,
                "phase": None,
                "memory_citation": None,
            },
        })
        # Outbound terminal: response_item.message(role=assistant)
        lines.append({
            "timestamp": "2026-08-02T05:58:04.738Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg_child_final",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": terminal_marker}
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
        })
        # task_complete
        lines.append({
            "timestamp": "2026-08-02T05:58:04.746Z",
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": terminal_marker,
                "started_at": 1785650278,
                "completed_at": 1785650284,
                "duration_ms": 5929,
            },
        })
    return lines


# ---------------------------------------------------------------------------
# OAuth account descriptor tests
# ---------------------------------------------------------------------------


class TestOAuthAccountDescriptor:
    def test_valid_account_id(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {
                "account_id": "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5",
                "access_token": "SECRET_TOKEN",
                "refresh_token": "SECRET_REFRESH",
            }
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert failures == []
        assert account_id == "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5"

    def test_uppercase_uuid_normalized(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {
                "account_id": "019FC10C-CDB2-7333-A3EC-6F1B1C6E8DD5",
            }
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert failures == []
        assert account_id == "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5"

    def test_non_uuid_rejected(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {"account_id": "user-abc_123.xyz"}
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any("canonical lowercase UUID" in f for f in failures)

    def test_missing_file(self, tmp_path: pathlib.Path):
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=tmp_path / "nonexistent.json",
        )
        assert account_id is None
        assert len(failures) == 1
        assert "not found" in failures[0]

    def test_invalid_json(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{invalid json")
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any("invalid JSON" in f for f in failures)

    def test_missing_tokens(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"other": "data"}))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any("tokens" in f for f in failures)

    def test_empty_account_id(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"tokens": {"account_id": "  "}}))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any("absent or empty" in f for f in failures)

    def test_oversized_account_id(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {"account_id": "a" * 300}
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any(
            "canonical lowercase UUID" in f or "exceeds max length" in f
            for f in failures
        )

    def test_bad_characters(self, tmp_path: pathlib.Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {"account_id": "user@evil.com; rm -rf /"}
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id is None
        assert any("canonical lowercase UUID" in f for f in failures)

    def test_never_returns_secrets(self, tmp_path: pathlib.Path):
        """Ensure secret fields are never in the returned value or failures."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {
                "account_id": "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5",
                "access_token": "sk-super-secret",
                "refresh_token": "rt-super-secret",
                "id_token": "id-super-secret",
            }
        }))
        account_id, failures = HARNESS._cfg004_read_oauth_account_descriptor(
            auth_json_path=auth_file,
        )
        assert account_id == "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5"
        all_text = json.dumps(failures)
        assert "sk-super-secret" not in all_text
        assert "rt-super-secret" not in all_text
        assert "id-super-secret" not in all_text


# ---------------------------------------------------------------------------
# Transcript correlation tests
# ---------------------------------------------------------------------------


class TestTranscriptCorrelation:
    def test_exact_thread_id_match(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "019fc10c-cdb2-7333-a3ec-6f1b1c6e8dd5"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-57-49-{thread_id}.jsonl",
            _make_parent_transcript(thread_id),
        )
        path, failures = HARNESS._cfg004_locate_transcript_by_thread_id(
            thread_id, sessions_dir=sessions,
        )
        assert failures == []
        assert path is not None
        assert thread_id in path.name

    def test_no_match_fails_closed(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        (sessions / "2026" / "08" / "02").mkdir(parents=True)
        path, failures = HARNESS._cfg004_locate_transcript_by_thread_id(
            "nonexistent-thread-id", sessions_dir=sessions,
        )
        assert path is None
        assert len(failures) == 1
        assert "no transcript found" in failures[0]

    def test_multiple_matches_fails_closed(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "dup-thread-id"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T02-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id),
        )
        path, failures = HARNESS._cfg004_locate_transcript_by_thread_id(
            thread_id, sessions_dir=sessions,
        )
        assert path is None
        assert any("multiple" in f for f in failures)

    def test_missing_sessions_dir(self, tmp_path: pathlib.Path):
        path, failures = HARNESS._cfg004_locate_transcript_by_thread_id(
            "some-id", sessions_dir=tmp_path / "nonexistent",
        )
        assert path is None
        assert any("not found" in f for f in failures)


# ---------------------------------------------------------------------------
# Transcript parsing unit tests (real Codex 0.146 shapes)
# ---------------------------------------------------------------------------


class TestTranscriptParsing:
    def test_parse_parent_collaboration_events(self, tmp_path: pathlib.Path):
        thread_id = "parse-test-thread"
        child_id = "parse-child-thread"
        path = tmp_path / "rollout.jsonl"
        _write_jsonl(
            path,
            _make_parent_transcript(thread_id, child_thread_id=child_id),
        )

        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            path
        )
        assert failures == []
        assert summary["session_id"] == thread_id
        assert len(summary["spawn_calls"]) == 1
        assert summary["spawn_calls"][0]["task_name"] == "/root/basic_alias_child"
        assert summary["spawn_calls"][0]["child_thread_id"] == child_id
        assert len(summary["wait_calls"]) == 1
        assert summary["wait_calls"][0]["timed_out"] is False
        assert summary["command_count"] == 0

    def test_parse_child_with_commands(self, tmp_path: pathlib.Path):
        path = tmp_path / "child.jsonl"
        _write_jsonl(
            path,
            _make_child_transcript(
                "child-1",
                "parent-1",
                command_list=["pwd", "git status --short"],
            ),
        )

        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            path
        )
        assert failures == []
        assert summary["parent_thread_id"] == "parent-1"
        assert summary["command_count"] == 2
        assert summary["commands"][0]["command"] == "pwd"
        assert summary["commands"][0]["exit_code"] == 0
        assert summary["commands"][0]["has_output"] is True

    def test_parse_unreadable_file(self, tmp_path: pathlib.Path):
        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            tmp_path / "nonexistent.jsonl",
        )
        assert summary == {}
        assert any("unreadable" in f for f in failures)

    def test_malformed_json_lines_skipped(self, tmp_path: pathlib.Path):
        path = tmp_path / "mixed.jsonl"
        turn_id = "turn-mixed"
        lines = [
            json.dumps({
                "type": "session_meta",
                "payload": {"id": "t1", "session_id": "t1"},
            }),
            "NOT VALID JSON",
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "ls"}),
                    "call_id": "call_ls",
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": turn_id,
                    },
                },
            }),
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "Process exited with code 0\nOutput:\nfile.txt",
                },
            }),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            path
        )
        assert failures == []
        assert summary["command_count"] == 1
        assert summary["commands"][0]["exit_code"] == 0

    def test_inbound_agent_message_excluded_from_terminal(self, tmp_path):
        """Inbound response_item.agent_message must not be terminal evidence."""
        path = tmp_path / "inbound.jsonl"
        _write_jsonl(path, [
            {
                "type": "session_meta",
                "payload": {"id": "t1", "session_id": "t1"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "agent_message",
                    "id": "amsg_1",
                    "author": "/root/child",
                    "recipient": "/root",
                    "content": [
                        {"type": "input_text", "text": "FAKE_MARKER"}
                    ],
                },
            },
        ])
        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            path
        )
        assert failures == []
        assert "FAKE_MARKER" not in summary["outbound_terminal_messages"]

    def test_encrypted_payload_excluded(self, tmp_path: pathlib.Path):
        """Encrypted reasoning content must not produce terminal evidence."""
        path = tmp_path / "encrypted.jsonl"
        _write_jsonl(path, [
            {
                "type": "session_meta",
                "payload": {"id": "t1", "session_id": "t1"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "gAAAAAB_encrypted_blob",
                },
            },
        ])
        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(
            path
        )
        assert failures == []
        assert summary["outbound_terminal_messages"] == []


# ---------------------------------------------------------------------------
# Transcript + stdout merge / collaboration validation
# ---------------------------------------------------------------------------


class TestTranscriptCollaborationValidation:
    def test_disabled_when_no_checks(self):
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id="any", checks={},
        )
        assert summary == {"enabled": False}
        assert failures == []

    def test_no_thread_id_fails_closed(self):
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=None, checks={"require_spawn_and_wait": True},
        )
        assert summary["enabled"] is True
        assert any("no thread_id" in f for f in failures)

    def test_absent_transcript_fails_closed(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        (sessions / "2026" / "08" / "02").mkdir(parents=True)
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id="missing-thread",
            checks={"require_spawn_and_wait": True},
            sessions_dir=sessions,
        )
        assert summary["enabled"] is True
        assert any("no transcript found" in f for f in failures)

    def test_successful_parent_with_child_commands(self, tmp_path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "parent-thread-abc"
        child_id = "child-thread-def"

        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id,
                thread_id,
                command_list=[
                    "pwd",
                    "git rev-parse --show-toplevel",
                    "git status --short",
                ],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )

        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
                "exact_child_commands": [
                    "pwd",
                    "git rev-parse --show-toplevel",
                    "git status --short",
                ],
                "require_parallel_batch": True,
                "child_terminal_marker": (
                    "BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED"
                ),
                "parent_terminal_marker": (
                    "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
                ),
            },
            sessions_dir=sessions,
        )
        assert failures == [], f"Unexpected failures: {failures}"
        assert summary["enabled"] is True
        assert summary["total_child_commands"] == 3
        assert summary["child_thread_id"] == child_id
        parent_t = summary["parent_transcript"]
        assert len(parent_t["spawn_calls"]) == 1
        assert len(parent_t["wait_calls"]) == 1

    def test_pass3_zero_child_commands_fails(self, tmp_path: pathlib.Path):
        """Reproduce pass-3 failure: parent has spawn+wait but child made
        zero commands.  The validator must fail, not claim success."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "pass3-parent-thread"
        child_id = "pass3-child-thread"

        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-57-49-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id, child_thread_id=child_id,
            ),
        )
        # Child transcript with zero commands (like pass 3).
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-57-58-{child_id}.jsonl",
            _make_child_transcript(child_id, thread_id, command_list=[]),
        )

        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
            },
            sessions_dir=sessions,
        )
        assert summary["enabled"] is True
        assert summary["total_child_commands"] == 0
        assert any("executed 0 commands" in f for f in failures), (
            f"Expected zero-child-command failure, got: {failures}"
        )

    def test_no_child_thread_id_fails(self, tmp_path: pathlib.Path):
        """Spawn without sub_agent_activity cannot locate child."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-child-id-parent"

        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id, child_thread_id=None,
            ),
        )

        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
            },
            sessions_dir=sessions,
        )
        assert any("sub_agent_activity" in f for f in failures)

    def test_parent_missing_spawn_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-spawn-parent"

        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id, spawn=False),
        )

        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={"require_spawn_and_wait": True},
            sessions_dir=sessions,
        )
        assert any("no spawn_agent" in f for f in failures)

    def test_parent_missing_wait_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-wait-parent"

        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id, wait=False),
        )

        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={"require_spawn_and_wait": True},
            sessions_dir=sessions,
        )
        assert any("no wait_agent" in f for f in failures)


# ---------------------------------------------------------------------------
# Focused negative tests for enhanced child contract validation
# ---------------------------------------------------------------------------

_FULL_CHECKS = {
    "require_spawn_and_wait": True,
    "minimum_child_commands": 1,
    "exact_child_commands": [
        "pwd",
        "git rev-parse --show-toplevel",
        "git status --short",
    ],
    "require_parallel_batch": True,
    "child_terminal_marker": "BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
    "parent_terminal_marker": "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
}


class TestTranscriptNegativeCases:
    """Focused negative tests: each isolates one contract violation."""

    def _setup(
        self,
        tmp_path: pathlib.Path,
        *,
        parent_kwargs: dict | None = None,
        child_kwargs: dict | None = None,
        checks: dict | None = None,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "neg-parent-thread"
        child_id = "neg-child-thread"
        pk = dict(parent_kwargs or {})
        ck = dict(child_kwargs or {})
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker=pk.pop(
                    "parent_marker",
                    "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                ),
                **pk,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id,
                thread_id,
                command_list=ck.pop("command_list", [
                    "pwd",
                    "git rev-parse --show-toplevel",
                    "git status --short",
                ]),
                terminal_marker=ck.pop(
                    "terminal_marker",
                    "BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                ),
                **ck,
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=checks or dict(_FULL_CHECKS),
            sessions_dir=sessions,
        )

    def test_zero_commands_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            child_kwargs={"command_list": []},
        )
        assert any("executed 0 commands" in f for f in failures)

    def test_wrong_command_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            child_kwargs={
                "command_list": ["pwd", "ls -la", "git status --short"],
            },
        )
        assert any("do not match expected" in f for f in failures)

    def test_nonzero_exit_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            child_kwargs={"exit_codes": [0, 1, 0]},
        )
        assert any("exit_code" in f and "!= 0" in f for f in failures)

    def test_sequential_call_output_fails_parallel_proof(
        self, tmp_path: pathlib.Path,
    ):
        """Sequential call/output pairs must fail parallel proof."""
        _, failures = self._setup(
            tmp_path,
            child_kwargs={"sequential_call_output": True},
        )
        assert any("parallel proof failed" in f for f in failures)

    def test_multi_turn_fails_parallel_proof(self, tmp_path: pathlib.Path):
        """Commands spanning multiple turns must fail parallel proof."""
        _, failures = self._setup(
            tmp_path,
            child_kwargs={"multi_turn": True},
        )
        assert any("parallel proof failed" in f for f in failures)

    def test_missing_child_marker_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            child_kwargs={"terminal_marker": None},
        )
        assert any("child missing terminal marker" in f for f in failures)

    def test_timed_out_wait_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            parent_kwargs={"wait_timed_out": True},
        )
        assert any("timed out" in f for f in failures)

    def test_mismatched_thread_id_fails(self, tmp_path: pathlib.Path):
        """Parent session_meta id does not match the emitted thread_id."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        file_thread = "file-thread-id"
        meta_thread = "different-meta-id"
        lines = _make_parent_transcript(meta_thread)
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{file_thread}.jsonl",
            lines,
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=file_thread,
            checks={"require_spawn_and_wait": True},
            sessions_dir=sessions,
        )
        assert any("does not match emitted thread_id" in f for f in failures)

    def test_missing_parent_marker_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path,
            parent_kwargs={"parent_marker": None},
        )
        assert any("parent missing terminal marker" in f for f in failures)

    def test_inbound_marker_not_counted_as_parent_terminal(
        self, tmp_path: pathlib.Path,
    ):
        """Inbound FINAL_ANSWER with marker text must not satisfy parent
        terminal marker requirement."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "inbound-marker-parent"
        child_id = "inbound-marker-child"
        # Build parent with no outbound marker but inbound agent_message
        # containing the marker text.
        parent_lines = _make_parent_transcript(
            thread_id,
            child_thread_id=child_id,
            parent_marker=None,
        )
        # The default fixture already includes an inbound agent_message
        # with "COMPLETE". Add one with the exact parent marker text.
        parent_lines.append({
            "type": "response_item",
            "payload": {
                "type": "agent_message",
                "id": "amsg_fake",
                "author": "/root/child",
                "recipient": "/root",
                "content": [{
                    "type": "input_text",
                    "text": "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                }],
            },
        })
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            parent_lines,
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id,
                thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
                "parent_terminal_marker": (
                    "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED"
                ),
            },
            sessions_dir=sessions,
        )
        assert any(
            "parent missing terminal marker" in f for f in failures
        ), f"Expected parent marker failure, got: {failures}"


# ---------------------------------------------------------------------------
# Exactly-one-wait contract tests
# ---------------------------------------------------------------------------


class TestExactlyOneWait:
    """Validator must require exactly one successful wait call/output."""

    def _setup_parent_with_waits(
        self,
        tmp_path: pathlib.Path,
        *,
        wait_count: int = 1,
        wait_timed_out: bool = False,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wait-count-parent"
        child_id = "wait-count-child"

        # Build parent with N wait calls.
        parent_lines = _make_parent_transcript(
            thread_id,
            child_thread_id=child_id,
            wait=False,  # We add waits manually
        )
        turn_id = f"turn-{thread_id}"
        for i in range(wait_count):
            parent_lines.append({
                "timestamp": f"2026-08-02T05:58:0{i}.252Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": f"fc_wait_{i}",
                    "name": "wait_agent",
                    "namespace": "collaboration",
                    "arguments": json.dumps({"timeout_ms": 3600000}),
                    "call_id": f"call_wait_{i}",
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": turn_id,
                    },
                },
            })
            parent_lines.append({
                "timestamp": f"2026-08-02T05:58:0{i}.746Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "id": f"fco_wait_{i}",
                    "call_id": f"call_wait_{i}",
                    "output": json.dumps({
                        "message": "Wait completed.",
                        "timed_out": wait_timed_out,
                    }),
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": turn_id,
                    },
                },
            })
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            parent_lines,
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id,
                thread_id,
                command_list=["pwd"],
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
            },
            sessions_dir=sessions,
        )

    def test_exactly_one_successful_wait_passes(self, tmp_path: pathlib.Path):
        summary, failures = self._setup_parent_with_waits(
            tmp_path, wait_count=1,
        )
        assert not any("successful wait" in f for f in failures), (
            f"Single wait should pass, got: {failures}"
        )

    def test_duplicate_waits_fail(self, tmp_path: pathlib.Path):
        summary, failures = self._setup_parent_with_waits(
            tmp_path, wait_count=2,
        )
        assert any(
            "expected exactly 1 successful wait" in f for f in failures
        ), f"Duplicate waits must fail, got: {failures}"

    def test_zero_waits_fail(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "zero-wait-parent"
        child_id = "zero-wait-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id, child_thread_id=child_id, wait=False,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(child_id, thread_id, command_list=["pwd"]),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={"require_spawn_and_wait": True, "minimum_child_commands": 1},
            sessions_dir=sessions,
        )
        assert any("no wait_agent" in f for f in failures)


# ---------------------------------------------------------------------------
# Artifact key agreement tests
# ---------------------------------------------------------------------------


class TestArtifactKeyAgreement:
    """Validator result and artifact assembly must agree on child key."""

    def test_child_transcript_key_is_singular_dict(self, tmp_path):
        """Validator returns child_transcript (singular dict), not
        child_transcripts (plural list)."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "artifact-key-parent"
        child_id = "artifact-key-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id, child_thread_id=child_id),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id, command_list=["pwd"],
            ),
        )
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
            },
            sessions_dir=sessions,
        )
        # Key must be singular "child_transcript" containing a dict.
        assert "child_transcript" in summary
        assert isinstance(summary["child_transcript"], dict)
        assert summary["child_transcript"].get("session_id") == child_id
        # Plural key must NOT exist (old bug).
        assert "child_transcripts" not in summary

    def test_child_transcript_path_extractable(self, tmp_path):
        """child_transcript_paths artifact extraction must find the path."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "path-extract-parent"
        child_id = "path-extract-child"
        child_filename = f"rollout-2026-08-02T01-00-10-{child_id}.jsonl"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(thread_id, child_thread_id=child_id),
        )
        _write_jsonl(
            date_dir / child_filename,
            _make_child_transcript(
                child_id, thread_id, command_list=["pwd"],
            ),
        )
        summary, _ = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks={
                "require_spawn_and_wait": True,
                "minimum_child_commands": 1,
            },
            sessions_dir=sessions,
        )
        # Simulate the artifact assembly logic (post-fix).
        child_summary = summary.get("child_transcript")
        child_paths = [
            child_summary["transcript_path"],
        ] if isinstance(child_summary, dict) and child_summary.get("transcript_path") else []
        assert len(child_paths) == 1
        assert child_id in child_paths[0]
        # The summary exclusion filter must remove the large child dict.
        filtered = {
            k: v
            for k, v in summary.items()
            if k not in ("parent_transcript", "child_transcript")
        }
        assert "child_transcript" not in filtered
        assert "parent_transcript" not in filtered
        # Scalar fields remain.
        assert filtered["child_thread_id"] == child_id
        assert filtered["total_child_commands"] == 1


# ---------------------------------------------------------------------------
# Spawn args, identity, task_complete, and message validation tests
# ---------------------------------------------------------------------------

_FULL_CHECKS_V2 = {
    **_FULL_CHECKS,
    "expected_spawn_args": {
        "agent_type": "opencode",
        "model": "basic",
        "fork_turns": "none",
    },
    "exact_child_prompt": _EXACT_CHILD_PROMPT,
    "reject_encrypted_message_prefix": "gAAAA",
    "expected_child_identity": {
        "agent_role": "opencode",
        "requested_alias": "basic",
        "resolved_model": "opencode_zen/big-pickle",
        "sandbox": "read-only",
        "cli_version": "nonempty",
        "originator": "codex_exec",
    },
    "require_child_task_complete_before_wait": True,
    "require_wait_child_correlation": True,
    "require_parent_task_complete": True,
}


class TestSpawnArgsValidation:
    """Validate exact spawn_agent arguments from parent transcript."""

    def _setup(
        self,
        tmp_path: pathlib.Path,
        *,
        spawn_message: str = _EXACT_CHILD_PROMPT,
        checks: dict | None = None,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "spawn-args-parent"
        child_id = "spawn-args-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                spawn_message=spawn_message,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id,
                thread_id,
                command_list=[
                    "pwd",
                    "git rev-parse --show-toplevel",
                    "git status --short",
                ],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=checks or dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )

    def test_exact_spawn_args_pass(self, tmp_path: pathlib.Path):
        summary, failures = self._setup(tmp_path)
        spawn_failures = [f for f in failures if "spawn arg" in f]
        assert spawn_failures == [], f"Unexpected spawn arg failures: {spawn_failures}"

    def test_wrong_agent_type_fails(self, tmp_path: pathlib.Path):
        """Spawn with wrong agent_type must fail."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-agent-type-parent"
        child_id = "wrong-agent-type-child"
        # Build parent with wrong agent_type in spawn args.
        parent_lines = _make_parent_transcript(
            thread_id,
            child_thread_id=child_id,
            parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
        )
        # Patch the spawn arguments to have wrong agent_type.
        for line in parent_lines:
            if (
                line.get("type") == "response_item"
                and isinstance(line.get("payload"), dict)
                and line["payload"].get("name") == "spawn_agent"
            ):
                args = json.loads(line["payload"]["arguments"])
                args["agent_type"] = "alibaba"
                line["payload"]["arguments"] = json.dumps(args)
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            parent_lines,
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("spawn arg" in f and "agent_type" in f for f in failures)

    def test_wrong_model_fails(self, tmp_path: pathlib.Path):
        """Spawn with wrong model must fail."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-model-parent"
        child_id = "wrong-model-child"
        parent_lines = _make_parent_transcript(
            thread_id,
            child_thread_id=child_id,
            parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
        )
        for line in parent_lines:
            if (
                line.get("type") == "response_item"
                and isinstance(line.get("payload"), dict)
                and line["payload"].get("name") == "spawn_agent"
            ):
                args = json.loads(line["payload"]["arguments"])
                args["model"] = "gpt-5.4-mini"
                line["payload"]["arguments"] = json.dumps(args)
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            parent_lines,
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("spawn arg" in f and "model" in f for f in failures)


class TestMessageRejection:
    """Validate empty, encrypted, and wrong prompt rejection."""

    def _setup(
        self,
        tmp_path: pathlib.Path,
        *,
        spawn_message: str = _EXACT_CHILD_PROMPT,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "msg-reject-parent"
        child_id = "msg-reject-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                spawn_message=spawn_message,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )

    def test_exact_message_passes(self, tmp_path: pathlib.Path):
        _, failures = self._setup(tmp_path)
        msg_failures = [f for f in failures if "message" in f.lower()]
        assert msg_failures == [], f"Unexpected message failures: {msg_failures}"

    def test_empty_message_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(tmp_path, spawn_message="")
        assert any("empty or absent" in f for f in failures)

    def test_encrypted_message_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path, spawn_message="gAAAAABmEncryptedPayload=="
        )
        assert any("opaque/encrypted" in f for f in failures)

    def test_wrong_message_fails(self, tmp_path: pathlib.Path):
        _, failures = self._setup(
            tmp_path, spawn_message="Do something else entirely"
        )
        assert any("does not match" in f and "exact child prompt" in f for f in failures)


class TestChildIdentityValidation:
    """Validate child identity evidence from session_meta."""

    def test_correct_identity_passes(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "identity-parent"
        child_id = "identity-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        identity_failures = [f for f in failures if "child identity" in f]
        assert identity_failures == [], f"Unexpected identity failures: {identity_failures}"

    def test_wrong_agent_role_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-role-parent"
        child_id = "wrong-role-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        # Build child with wrong agent_role.
        child_lines = _make_child_transcript(
            child_id, thread_id,
            command_list=["pwd"],
            terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
        )
        for line in child_lines:
            if line.get("type") == "session_meta":
                line["payload"]["source"]["subagent"]["thread_spawn"]["agent_role"] = "alibaba"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            child_lines,
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("child identity" in f and "agent_role" in f for f in failures)


class TestTaskCompleteOrdering:
    """Validate child task_complete before wait, parent task_complete."""

    def test_child_task_complete_present(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "tc-order-parent"
        child_id = "tc-order-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        tc_failures = [f for f in failures if "task_complete" in f]
        assert tc_failures == [], f"Unexpected task_complete failures: {tc_failures}"

    def test_child_missing_task_complete_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-tc-parent"
        child_id = "no-tc-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        # Child with terminal_marker=None means no task_complete event.
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker=None,
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("child has no explicit task_complete" in f for f in failures)

    def test_parent_missing_task_complete_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-parent-tc"
        child_id = "no-parent-tc-child"
        # parent_marker=None means no task_complete event.
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker=None,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("parent has no explicit task_complete" in f for f in failures)


class TestChildInboundPromptValidation:
    """Validate child inbound NEW_TASK payload matches exact prompt."""

    def test_correct_inbound_prompt_passes(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "inbound-prompt-parent"
        child_id = "inbound-prompt-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                inbound_prompt=_EXACT_CHILD_PROMPT,
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        inbound_failures = [f for f in failures if "inbound" in f.lower()]
        assert inbound_failures == [], f"Unexpected inbound failures: {inbound_failures}"

    def test_encrypted_inbound_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "encrypted-inbound-parent"
        child_id = "encrypted-inbound-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                encrypted_inbound=True,
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("encrypted/opaque" in f for f in failures)

    def test_wrong_inbound_prompt_fails(self, tmp_path: pathlib.Path):
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-inbound-parent"
        child_id = "wrong-inbound-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                inbound_prompt="Completely different prompt text",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("does not" in f and "exact child prompt" in f for f in failures)

    def test_encrypted_content_plaintext_inbound_passes(
        self, tmp_path: pathlib.Path
    ):
        """Saved pass-3 two-part shape: part 1 input_text carries only the
        NEW_TASK envelope and part 2 type=encrypted_content carries the
        exact plaintext prompt (OpenCode materializes it before provider
        conversion). Inbound prompt validation must pass."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "enccontent-plaintext-parent"
        child_id = "enccontent-plaintext-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        child_lines = _make_child_transcript(
            child_id, thread_id,
            command_list=["pwd"],
            terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
        )
        for _line in child_lines:
            _payload = _line.get("payload") or {}
            if _payload.get("type") == "agent_message":
                _payload["content"] = [
                    {
                        "type": "input_text",
                        "text": "Message Type: NEW_TASK\n"
                        "Task name: /root/basic_alias_child\n"
                        "Sender: /root\nPayload:\n",
                    },
                    {
                        "type": "encrypted_content",
                        "encrypted_content": _EXACT_CHILD_PROMPT,
                    },
                ]
                break
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            child_lines,
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        inbound_failures = [f for f in failures if "inbound" in f.lower()]
        assert inbound_failures == [], (
            f"Unexpected inbound failures: {inbound_failures}"
        )


class TestAliasVsResolvedModel:
    """Validate alias-vs-resolved-model distinction is preserved."""

    def test_spawn_model_basic_vs_child_resolved(self, tmp_path: pathlib.Path):
        """spawn_agent model=basic (public alias) while child resolves to
        opencode_zen/big-pickle from codex_opencode_agent.toml."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "alias-resolved-parent"
        child_id = "alias-resolved-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        summary, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        # Verify spawn args preserve alias "basic".
        parent_t = summary.get("parent_transcript") or {}
        spawn_args_list = parent_t.get("spawn_arguments") or []
        assert len(spawn_args_list) == 1
        assert spawn_args_list[0]["model"] == "basic"
        assert spawn_args_list[0]["agent_type"] == "opencode"
        # Child identity preserves role and alias/upstream distinction.
        child_t = summary.get("child_transcript") or {}
        child_ident = child_t.get("session_identity") or {}
        assert child_ident.get("agent_role") == "opencode"
        assert child_ident.get("requested_alias") == "basic"
        assert child_ident.get("resolved_model") == "opencode_zen/big-pickle"
        assert child_ident.get("sandbox") == "read-only"
        # No identity failures.
        identity_failures = [f for f in failures if "child identity" in f]
        assert identity_failures == []

    def test_wrong_resolved_model_fails(self, tmp_path: pathlib.Path):
        """Child resolving to a different model must fail identity check."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-resolved-parent"
        child_id = "wrong-resolved-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                turn_context_overrides={"model": "openai/gpt-5.4-mini"},
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any(
            "child identity" in f and "resolved_model" in f
            for f in failures
        )

    def test_wrong_sandbox_fails(self, tmp_path: pathlib.Path):
        """Child with non-read-only sandbox must fail identity check."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wrong-sandbox-parent"
        child_id = "wrong-sandbox-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                turn_context_overrides={"sandbox": "danger-full-access"},
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any(
            "child identity" in f and "sandbox" in f
            for f in failures
        )

    def test_missing_turn_context_fails_identity(self, tmp_path: pathlib.Path):
        """Child without turn_context must fail resolved_model/sandbox checks."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "no-tc-ctx-parent"
        child_id = "no-tc-ctx-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                include_turn_context=False,
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any(
            "child identity" in f and "resolved_model" in f
            for f in failures
        )


class TestWaitChildCorrelation:
    """Validate wait-to-spawn correlation and ordering."""

    def _setup(
        self,
        tmp_path: pathlib.Path,
        *,
        parent_kwargs: dict | None = None,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "wait-corr-parent"
        child_id = "wait-corr-child"
        pk = dict(parent_kwargs or {})
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker=pk.pop(
                    "parent_marker",
                    "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                ),
                **pk,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )

    def test_wait_correlated_to_spawned_child(self, tmp_path: pathlib.Path):
        """Successful wait correlated to the exact spawned child passes."""
        summary, failures = self._setup(tmp_path)
        wait_failures = [f for f in failures if "wait" in f.lower()]
        assert wait_failures == [], f"Unexpected wait failures: {wait_failures}"
        assert summary["child_thread_id"] == "wait-corr-child"

    def test_wait_timed_out_fails(self, tmp_path: pathlib.Path):
        """Timed-out wait must fail."""
        _, failures = self._setup(
            tmp_path, parent_kwargs={"wait_timed_out": True}
        )
        assert any("timed out" in f for f in failures)

    def test_no_wait_fails(self, tmp_path: pathlib.Path):
        """Missing wait must fail when require_spawn_and_wait is set."""
        _, failures = self._setup(
            tmp_path, parent_kwargs={"wait": False}
        )
        assert any("no wait_agent" in f for f in failures)


class TestDirectParserChildTranscript:
    """Direct parser test against a pass-3-shaped child transcript.

    Identity parses correctly, but the pass still fails on
    encrypted/empty prompt and zero commands.
    """

    def _write_child(self, tmp_path: pathlib.Path, **kwargs) -> pathlib.Path:
        child_id = "direct-parse-child"
        parent_id = "direct-parse-parent"
        lines = _make_child_transcript(child_id, parent_id, **kwargs)
        path = tmp_path / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl"
        _write_jsonl(path, lines)
        return path

    def test_identity_parses_from_turn_context_and_session_meta(
        self, tmp_path: pathlib.Path
    ):
        """turn_context.model -> resolved_model, sandbox_policy.type ->
        sandbox; session_meta -> agent_role, cli_version, originator."""
        path = self._write_child(
            tmp_path,
            command_list=["pwd"],
            terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
        )
        summary, failures = HARNESS._cfg004_parse_transcript_collaboration(path)
        assert failures == []
        ident = summary["session_identity"]
        assert ident["resolved_model"] == "opencode_zen/big-pickle"
        assert ident["sandbox"] == "read-only"
        assert ident["agent_role"] == "opencode"
        assert ident["cli_version"] == "0.146.0"
        assert ident["originator"] == "codex_exec"

    def test_encrypted_prompt_fails_validation(self, tmp_path: pathlib.Path):
        """Encrypted inbound prompt causes failure even with valid identity."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "enc-direct-parent"
        child_id = "enc-direct-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                encrypted_inbound=True,
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("encrypted/opaque" in f for f in failures)

    def test_zero_commands_fails_validation(self, tmp_path: pathlib.Path):
        """Zero child commands causes failure even with valid identity."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "zero-cmd-parent"
        child_id = "zero-cmd-child"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=[],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any("commands" in f and "expected" in f for f in failures)


class TestOrderingPassFail:
    """Validate correct and incorrect chronology ordering."""

    def _setup_full(
        self,
        tmp_path: pathlib.Path,
        *,
        parent_kwargs: dict | None = None,
        child_kwargs: dict | None = None,
    ) -> tuple[dict, list[str]]:
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "ordering-parent"
        child_id = "ordering-child"
        pk = dict(parent_kwargs or {})
        ck = dict(child_kwargs or {})
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker=pk.pop(
                    "parent_marker",
                    "CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
                ),
                **pk,
            ),
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=ck.pop("command_list", ["pwd"]),
                terminal_marker=ck.pop(
                    "terminal_marker",
                    "BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
                ),
                **ck,
            ),
        )
        return HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )

    def test_correct_chronology_passes(self, tmp_path: pathlib.Path):
        """spawn -> wait call -> child task_complete -> wait output ->
        child FINAL_ANSWER -> parent task_complete."""
        summary, failures = self._setup_full(tmp_path)
        ordering_failures = [
            f for f in failures
            if "task_complete" in f or "wait" in f.lower()
        ]
        assert ordering_failures == [], f"Ordering failures: {ordering_failures}"

    def test_child_task_complete_after_wait_output_fails(
        self, tmp_path: pathlib.Path
    ):
        """Child task_complete timestamp after wait output must fail."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "late-tc-parent"
        child_id = "late-tc-child"
        # Parent: wait output at 05:58:04.746Z.
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            _make_parent_transcript(
                thread_id,
                child_thread_id=child_id,
                parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
            ),
        )
        # Child: task_complete at 05:58:04.746Z (same as wait output)
        # passes; push it later to fail.
        child_lines = _make_child_transcript(
            child_id, thread_id,
            command_list=["pwd"],
            terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
        )
        for line in child_lines:
            if (
                line.get("type") == "event_msg"
                and isinstance(line.get("payload"), dict)
                and line["payload"].get("type") == "task_complete"
            ):
                line["timestamp"] = "2026-08-02T05:58:05.000Z"
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            child_lines,
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any(
            "child task_complete occurs after wait" in f for f in failures
        )

    def test_parent_task_complete_before_wait_output_fails(
        self, tmp_path: pathlib.Path
    ):
        """Parent task_complete before wait output must fail."""
        sessions = tmp_path / "sessions"
        date_dir = sessions / "2026" / "08" / "02"
        thread_id = "early-ptc-parent"
        child_id = "early-ptc-child"
        parent_lines = _make_parent_transcript(
            thread_id,
            child_thread_id=child_id,
            parent_marker="CODEX_BASIC_ALIAS_PARALLEL_TOOLS_PASSED",
        )
        # Move parent task_complete before wait output by rewriting
        # line order: insert task_complete right after spawn output.
        # Easiest: change the task_complete timestamp to be before wait
        # output and reorder lines.
        tc_idx = None
        for i, line in enumerate(parent_lines):
            if (
                line.get("type") == "event_msg"
                and isinstance(line.get("payload"), dict)
                and line["payload"].get("type") == "task_complete"
            ):
                tc_idx = i
                break
        assert tc_idx is not None
        # Move task_complete to right after spawn output (index 4).
        tc_line = parent_lines.pop(tc_idx)
        parent_lines.insert(4, tc_line)
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-00-{thread_id}.jsonl",
            parent_lines,
        )
        _write_jsonl(
            date_dir / f"rollout-2026-08-02T01-00-10-{child_id}.jsonl",
            _make_child_transcript(
                child_id, thread_id,
                command_list=["pwd"],
                terminal_marker="BASIC_ALIAS_CHILD_PARALLEL_TOOLS_PASSED",
            ),
        )
        _, failures = HARNESS._cfg004_validate_transcript_collaboration(
            thread_id=thread_id,
            checks=dict(_FULL_CHECKS_V2),
            sessions_dir=sessions,
        )
        assert any(
            "parent task_complete occurs before wait output" in f
            for f in failures
        )
