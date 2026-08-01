#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import hashlib
import http.client
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

import psycopg

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'scripts' / 'local-ci' / 'anthropic_adapter_config.json'
RUN_ACCEPTANCE_PATH = ROOT / 'scripts' / 'local-ci' / 'run_acceptance.py'
BUILT_IN_TARGET_PROFILES: dict[str, dict[str, str]] = {
    'dev': {
        'litellm_base_url': 'http://127.0.0.1:4001',
        'anthropic_base_url': 'http://127.0.0.1:4001/anthropic',
        'docker_container_name': 'litellm-dev',
        'expected_trace_environment': 'dev',
        'expected_runtime_environment': 'litellm-dev',
    },
    'prod': {
        'litellm_base_url': 'http://127.0.0.1:4000',
        'anthropic_base_url': 'http://127.0.0.1:4000/anthropic',
        'docker_container_name': 'aawm-litellm',
        'expected_trace_environment': 'prod',
        'expected_runtime_environment': 'prod',
    },
}
DEFAULT_RUNTIME_LOG_FORBIDDEN_SUBSTRINGS = [
    'Task exception was never retrieved',
    'Exception in ASGI application',
    "KeyError: 'choices'",
    'h11._util.LocalProtocolError',
    'Too little data for declared Content-Length',
]
DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS = [
    'pass_through_endpoint(): Exception occured - 429:',
    'pass_through_endpoint(): Exception occured - 500:',
    'pass_through_endpoint(): Exception occured - 502:',
    'pass_through_endpoint(): Exception occured - 503:',
    'pass_through_endpoint(): Exception occured - 504:',
]
DEFAULT_WARNING_ONLY_HARD_FAILURE_SUBSTRINGS = [
    'runtime logs contained forbidden substring',
    'successful empty',
]
MOONSHOT_ANTHROPIC_AGENTIC_CASE = "claude_adapter_aawm_sota_moonshot_agentic_tool_continuation"
MOONSHOT_ANTHROPIC_AGENTIC_FLAG = "moonshot_anthropic_agentic_only"
MOONSHOT_ANTHROPIC_ADAPTER_PATH = "anthropic_kimi_chat_completions_adapter"
MOONSHOT_CANONICAL_ALIAS = "aawm-sota-moonshot"
MOONSHOT_AGENT_PROFILE = "sota-moonshot"
MOONSHOT_SELECTED_MODELS = {"kimi_code/k3-max", "kimi_code/k3-high"}
ATTRIBUTION_SCOPED_RUNTIME_LOG_SUBSTRINGS = {
    *DEFAULT_RUNTIME_LOG_FORBIDDEN_SUBSTRINGS,
    *DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS,
}
UNRELATED_AUTO_AGENT_RUNTIME_LOG_CONTEXT_MARKERS = [
    '_handle_codex_auto_agent_alias_route',
    '_perform_codex_auto_agent_openrouter_completion_request',
    'codex_auto_agent_alias',
]
UNRELATED_PASSTHROUGH_RUNTIME_LOG_CONTEXT_MARKERS = [
    'chatgpt.com/backend-api/codex/responses',
]
UNRELATED_RUNTIME_LOG_ERROR_SIGNATURES = [
    'deepseek/deepseek-v4-flash:free',
    'reset reason: connection timeout',
]
# Bound docker CLI calls so an unresponsive daemon/logs cannot hang the suite.
DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS = 30

_CONTAINER_ENV_CACHE: dict[tuple[str, str], str | None] = {}


def _resolve_container_env_value(container_name: str, env_name: str) -> str | None:
    """Retrieve a single named env var from a running container via docker exec.

    Results are cached per (container, env_name) pair.  The value is never
    printed, logged, or stored in artifacts.  Returns ``None`` on any failure.
    """
    cache_key = (container_name, env_name)
    if cache_key in _CONTAINER_ENV_CACHE:
        return _CONTAINER_ENV_CACHE[cache_key]
    value: str | None = None
    try:
        result = subprocess.run(
            ['docker', 'exec', container_name, 'printenv', env_name],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            value = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    _CONTAINER_ENV_CACHE[cache_key] = value
    return value

# Tight window for positive "unrelated concurrent traffic" evidence around a match.
UNRELATED_RUNTIME_LOG_LOCAL_CONTEXT_CHARS = 800
# Failures that may be soft-failed when provider-unavailable log evidence is present.
PROVIDER_UNAVAILABLE_SOFT_FAILABLE_FAILURE_MARKERS = (
    'command failed',
    'timed out after',
    'TimeoutExpired',
    'runtime healthcheck failed',
    'runtime container `',
    'provider unavailable',
    'provider-unavailable',
    'connection refused',
    'ConnectError',
    'APIConnectionError',
    'APITimeoutError',
    'ReadTimeout',
    'ConnectTimeout',
)
RUNTIME_LOG_MODEL_FIELD_RE = re.compile(r'"model"\s*:\s*"([^"]+)"')
_VALIDATION_DB_CONNECTIONS: dict[tuple[str, int, str, str, str], Any] = {}


def _close_validation_db_connections() -> None:
    while _VALIDATION_DB_CONNECTIONS:
        _, conn = _VALIDATION_DB_CONNECTIONS.popitem()
        try:
            conn.close()
        except Exception:
            pass


atexit.register(_close_validation_db_connections)


def _load_run_acceptance_module() -> Any:
    spec = importlib.util.spec_from_file_location('run_acceptance_module', RUN_ACCEPTANCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'unable to load run_acceptance helpers from {RUN_ACCEPTANCE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RA = _load_run_acceptance_module()


def _emit_stdout(message: str, *, flush: bool = False) -> None:
    """Write operator-facing stdout (avoids Ruff T201 print ban)."""
    sys.stdout.write(message if message.endswith('\n') else f'{message}\n')
    if flush:
        sys.stdout.flush()


def _emit_stderr(message: str, *, flush: bool = False) -> None:
    """Write operator-facing stderr (avoids Ruff T201 print ban)."""
    sys.stderr.write(message if message.endswith('\n') else f'{message}\n')
    if flush:
        sys.stderr.flush()


def _extract_path_value(value: Any, path: str) -> Any | None:
    current = value
    segments = path.split('.')
    index = 0
    while index < len(segments):
        if not isinstance(current, dict):
            return None

        matched_key = None
        matched_end = None
        for end in range(len(segments), index, -1):
            candidate = '.'.join(segments[index:end])
            if candidate in current:
                matched_key = candidate
                matched_end = end
                break

        if matched_key is None:
            return None

        current = current.get(matched_key)
        index = matched_end if matched_end is not None else len(segments)

    return current


def _parse_command_output_json(stdout: str) -> dict[str, Any] | None:
    objects = RA._parse_stdout_json_objects(stdout)
    if not objects:
        return None
    for obj in reversed(objects):
        if obj.get('type') == 'result':
            return obj
    return _sanitize_turn_failed_output(objects[-1])


_TURN_FAILED_MAX_MESSAGE_CHARS = 4096
_TURN_FAILED_SECRET_PATTERNS = (
    'sk-', 'Bearer ', 'Authorization:', 'api-key', 'api_key',
    'raw_body', 'raw_response_body', 'raw_provider_body',
    'provider_response_body', 'account_id', 'account_identifier',
    'account_email',
)
_TURN_FAILED_EMAIL_PATTERN = re.compile(
    r'(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}'
    r'(?![A-Z0-9._%+-])'
)


def _turn_failed_value_contains_secret(value: Any) -> bool:
    if isinstance(value, str):
        value_lower = value.lower()
        return (
            any(
                pattern.lower() in value_lower
                for pattern in _TURN_FAILED_SECRET_PATTERNS
            )
            or _TURN_FAILED_EMAIL_PATTERN.search(value) is not None
        )
    if isinstance(value, dict):
        return any(
            _turn_failed_value_contains_secret(key)
            or _turn_failed_value_contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_turn_failed_value_contains_secret(item) for item in value)
    return False


def _extract_turn_failed_code(obj: dict[str, Any]) -> int | None:
    """Extract a valid 400/429 status code from a parsed error object."""
    error = obj.get('error')
    if not isinstance(error, dict):
        return None
    code = error.get('code')
    if isinstance(code, bool):
        return None
    if isinstance(code, int) and code in (400, 429):
        return code
    if isinstance(code, str) and code in ('400', '429'):
        return int(code)
    return None


def _sanitize_turn_failed_output(
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """D1-574: sanitize a turn.failed command output object.

    Parses error.message nested JSON at most two layers, accepts only
    error.code integer/string 400 or 429 (not bool/float/other), rejects
    malformed/unexpected/oversized or secret-bearing content, and returns
    sanitized is_error=True, api_error_status, status_code without copying
    raw message/body/secrets.
    """
    if not isinstance(parsed, dict) or parsed.get('type') != 'turn.failed':
        return parsed

    sanitized: dict[str, Any] = {'type': 'turn.failed', 'is_error': True}

    error = parsed.get('error')
    if not isinstance(error, dict):
        return sanitized

    message = error.get('message')
    if not isinstance(message, str):
        return sanitized

    if len(message) > _TURN_FAILED_MAX_MESSAGE_CHARS:
        return sanitized

    if _turn_failed_value_contains_secret(message):
        return sanitized

    # Layer 1: parse error.message as JSON
    try:
        layer1 = json.loads(message)
    except (json.JSONDecodeError, ValueError):
        return sanitized

    if not isinstance(layer1, dict):
        return sanitized
    if _turn_failed_value_contains_secret(layer1):
        return sanitized

    code = _extract_turn_failed_code(layer1)
    if code is not None:
        sanitized['api_error_status'] = code
        sanitized['status_code'] = code
        return sanitized

    # Layer 2: if layer1 has error.message as string, parse again
    inner_error = layer1.get('error')
    if isinstance(inner_error, dict):
        inner_message = inner_error.get('message')
        if (
            isinstance(inner_message, str)
            and len(inner_message) <= _TURN_FAILED_MAX_MESSAGE_CHARS
        ):
            if not _turn_failed_value_contains_secret(inner_message):
                try:
                    layer2 = json.loads(inner_message)
                except (json.JSONDecodeError, ValueError):
                    return sanitized
                if isinstance(layer2, dict):
                    if _turn_failed_value_contains_secret(layer2):
                        return sanitized
                    code = _extract_turn_failed_code(layer2)
                    if code is not None:
                        sanitized['api_error_status'] = code
                        sanitized['status_code'] = code

    return sanitized


def _extract_command_output_text(stdout: str) -> str:
    objects = RA._parse_stdout_json_objects(stdout)
    for obj in reversed(objects):
        if obj.get("type") == "item.completed":
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    return text
        if obj.get("type") == "result":
            result = obj.get("result")
            if isinstance(result, str):
                return result
    return stdout


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        content = value.get("content")
        if content is not None:
            return _content_text(content)
    return ""


def _collect_codex_command_execution_results(stdout: str) -> list[dict[str, str]]:
    collected_by_id: dict[str, dict[str, str]] = {}
    collected_without_id: list[dict[str, str]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "command_execution":
                command = value.get("command")
                output = value.get("aggregated_output")
                if not isinstance(output, str):
                    output = value.get("stdout")
                if not isinstance(output, str):
                    output = value.get("output")
                if isinstance(command, str) and isinstance(output, str):
                    record = {"command": command, "output": output}
                    execution_id = value.get("id")
                    if isinstance(execution_id, str) and execution_id:
                        existing = collected_by_id.get(execution_id)
                        if (
                            existing is None
                            or (not existing["output"] and output)
                            or value.get("status") == "completed"
                        ):
                            collected_by_id[execution_id] = record
                    else:
                        collected_without_id.append(record)
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    for obj in RA._parse_stdout_json_objects(stdout):
        _walk(obj)
    return [*collected_by_id.values(), *collected_without_id]


def _validate_command_text_checks(
    *,
    family: str,
    text: str,
    checks: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    if not checks:
        return {"enabled": False}, []

    failures: list[str] = []
    required_prefix = checks.get("required_prefix")
    required_suffix = checks.get("required_suffix")
    required_substrings = _as_string_list(checks.get("required_substrings"))
    forbidden_substrings = _as_string_list(checks.get("forbidden_substrings"))
    forbidden_regex = _as_string_list(checks.get("forbidden_regex"))
    minimum_chars = checks.get("minimum_chars")
    maximum_chars = checks.get("maximum_chars")

    if isinstance(required_prefix, str) and not text.startswith(required_prefix):
        failures.append(
            f"{family} {label} did not start with {required_prefix!r}"
        )
    if isinstance(required_suffix, str) and not text.endswith(required_suffix):
        failures.append(
            f"{family} {label} did not end with {required_suffix!r}"
        )
    for substring in required_substrings:
        if substring not in text:
            failures.append(
                f"{family} {label} missing required substring {substring!r}"
            )
    for substring in forbidden_substrings:
        if substring in text:
            failures.append(
                f"{family} {label} contained forbidden substring {substring!r}"
            )
    for pattern in forbidden_regex:
        try:
            matched = re.search(pattern, text) is not None
        except re.error as exc:
            failures.append(
                f"{family} {label} has invalid forbidden regex {pattern!r}: {exc}"
            )
            continue
        if matched:
            failures.append(
                f"{family} {label} matched forbidden regex {pattern!r}"
            )
    if minimum_chars is not None and len(text) < int(minimum_chars):
        failures.append(
            f"{family} {label} below minimum length: expected >= {int(minimum_chars)}, got {len(text)}"
        )
    if maximum_chars is not None and len(text) > int(maximum_chars):
        failures.append(
            f"{family} {label} above maximum length: expected <= {int(maximum_chars)}, got {len(text)}"
        )

    return {
        "enabled": True,
        "length": len(text),
        "required_prefix": required_prefix,
        "required_suffix": required_suffix,
        "required_substrings": required_substrings,
        "forbidden_substrings": forbidden_substrings,
        "forbidden_regex": forbidden_regex,
    }, failures


_CODEX_TOOL_NAME_CANONICAL_MAP: dict[str, str] = {
    "functions.collaboration.spawn_agent": "spawn_agent",
    "functions.collaboration.wait": "wait",
    "functions.exec": "exec_command",
    "functions.exec_command": "exec_command",
}


def _normalize_codex_tool_name(raw: str) -> str:
    """Normalize Codex 0.146.0+ fully-qualified tool names to canonical short names.

    Codex 0.146.0 emits names like ``functions.collaboration.spawn_agent`` and
    ``functions.exec`` in collab_tool_call items.  The harness contract uses the
    canonical short names ``spawn_agent``, ``wait``, ``exec_command``.
    """
    return _CODEX_TOOL_NAME_CANONICAL_MAP.get(raw, raw)


def _validate_codex_collaboration_events(  # noqa: PLR0915
    *,
    family: str,
    stdout: str,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not checks:
        return {"enabled": False}, []

    command_execution_checks = checks.get("command_execution_validation") or {}
    expected_commands = [
        str(command)
        for command in command_execution_checks.get("exact_commands") or []
        if isinstance(command, str) and command
    ]
    expected_command_set = set(expected_commands)
    tool_counts: dict[str, int] = {}
    completed_tool_items: dict[str, list[dict[str, Any]]] = {}
    command_starts: list[dict[str, Any]] = []
    command_completions: list[dict[str, Any]] = []
    active_expected_command_ids: set[str] = set()
    maximum_parallel_expected_commands = 0
    turn_index = -1
    for obj in RA._parse_stdout_json_objects(stdout):
        if obj.get("type") == "turn.started":
            turn_index += 1
            continue
        if obj.get("type") not in {"item.started", "item.completed"}:
            continue
        item = obj.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") == "collab_tool_call" and obj.get("type") == "item.completed":
            tool = item.get("tool")
            if isinstance(tool, str) and tool:
                canonical = _normalize_codex_tool_name(tool)
                tool_counts[canonical] = tool_counts.get(canonical, 0) + 1
                completed_tool_items.setdefault(canonical, []).append(item)
            continue
        if (
            item.get("type") != "command_execution"
            or not command_execution_checks
        ):
            continue
        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        command = command.strip()
        item_id = item.get("id")
        record = {
            "id": item_id,
            "command": command,
            "status": item.get("status"),
            "exit_code": item.get("exit_code"),
            "error": item.get("error"),
            "turn_index": turn_index,
        }
        if obj.get("type") == "item.started":
            command_starts.append(record)
            if command in expected_command_set and isinstance(item_id, str) and item_id:
                active_expected_command_ids.add(item_id)
                maximum_parallel_expected_commands = max(
                    maximum_parallel_expected_commands,
                    len(active_expected_command_ids),
                )
        else:
            command_completions.append(record)
            if isinstance(item_id, str):
                active_expected_command_ids.discard(item_id)

    failures: list[str] = []
    minimum_tool_counts = checks.get("minimum_tool_counts") or {}
    for tool, minimum in minimum_tool_counts.items():
        actual = tool_counts.get(str(tool), 0)
        if actual < int(minimum):
            failures.append(
                f"{family} missing completed Codex collaboration calls for {tool!r}: expected >= {int(minimum)}, got {actual}"
            )

    maximum_tool_counts = checks.get("maximum_tool_counts") or {}
    for tool, maximum in maximum_tool_counts.items():
        actual = tool_counts.get(str(tool), 0)
        if actual > int(maximum):
            failures.append(
                f"{family} excess completed Codex collaboration calls for {tool!r}: expected <= {int(maximum)}, got {actual}"
            )

    required_successful_tools = {
        str(tool) for tool in checks.get("required_successful_tools") or []
    }
    spawned_agent_ids: set[str] = set()
    waited_agent_statuses: dict[str, Any] = {}
    for tool in required_successful_tools:
        for item in completed_tool_items.get(tool, []):
            if item.get("status") != "completed":
                failures.append(
                    f"{family} Codex collaboration {tool!r} did not complete successfully: status={item.get('status')!r}"
                )
            if item.get("error") not in (None, "", False, {}):
                failures.append(
                    f"{family} Codex collaboration {tool!r} recorded an error"
                )
            result = item.get("result")
            if not isinstance(result, dict):
                result = {}
            if tool == "spawn_agent":
                receiver_agent_ids = item.get("receiver_agent_ids")
                if not isinstance(receiver_agent_ids, list):
                    receiver_agent_ids = result.get("receiver_agent_ids")
                valid_receiver_agent_ids = {
                    agent_id
                    for agent_id in receiver_agent_ids or []
                    if isinstance(agent_id, str) and agent_id
                }
                if not valid_receiver_agent_ids:
                    failures.append(
                        f"{family} successful Codex spawn_agent did not record receiver_agent_ids"
                    )
                spawned_agent_ids.update(valid_receiver_agent_ids)
            elif tool == "wait":
                agents_states = item.get("agents_states")
                if not isinstance(agents_states, dict):
                    agents_states = result.get("agents_states")
                if not isinstance(agents_states, dict) or not agents_states:
                    failures.append(
                        f"{family} successful Codex wait did not record agents_states"
                    )
                    continue
                for agent_id, state in agents_states.items():
                    status = state.get("status") if isinstance(state, dict) else state
                    waited_agent_statuses[str(agent_id)] = status
                    if status != "completed":
                        failures.append(
                            f"{family} Codex wait agent {agent_id!r} did not terminate cleanly: status={status!r}"
                        )

    if checks.get("require_wait_for_spawned_agents"):
        missing_wait_agent_ids = sorted(
            spawned_agent_ids - set(waited_agent_statuses)
        )
        if missing_wait_agent_ids:
            failures.append(
                f"{family} Codex wait did not report spawned agents: {missing_wait_agent_ids!r}"
            )

    command_execution_summary: dict[str, Any] = {"enabled": False}
    if command_execution_checks:
        started_commands = [record["command"] for record in command_starts]
        completed_commands = [record["command"] for record in command_completions]
        if sorted(started_commands) != sorted(expected_commands):
            failures.append(
                f"{family} Codex command executions started unexpected commands: expected {sorted(expected_commands)!r}, got {sorted(started_commands)!r}"
            )
        if sorted(completed_commands) != sorted(expected_commands):
            failures.append(
                f"{family} Codex command executions completed unexpected commands: expected {sorted(expected_commands)!r}, got {sorted(completed_commands)!r}"
            )

        started_commands_by_id = {
            record["id"]: record["command"]
            for record in command_starts
            if isinstance(record.get("id"), str) and record["id"]
        }
        completed_commands_by_id = {
            record["id"]: record["command"]
            for record in command_completions
            if isinstance(record.get("id"), str) and record["id"]
        }
        if (
            len(started_commands_by_id) != len(command_starts)
            or len(completed_commands_by_id) != len(command_completions)
            or completed_commands_by_id != started_commands_by_id
        ):
            failures.append(
                f"{family} Codex command execution start/completion ID-to-command pairs did not match exactly"
            )

        required_status = command_execution_checks.get(
            "required_status", "completed"
        )
        required_exit_code = command_execution_checks.get("required_exit_code", 0)
        for record in command_completions:
            if record["status"] != required_status:
                failures.append(
                    f"{family} Codex command {record['command']!r} status was {record['status']!r}, expected {required_status!r}"
                )
            if (
                isinstance(record["exit_code"], bool)
                or record["exit_code"] != required_exit_code
            ):
                failures.append(
                    f"{family} Codex command {record['command']!r} exit_code was {record['exit_code']!r}, expected {required_exit_code!r}"
                )
            if record["error"] not in (None, "", False, {}):
                failures.append(
                    f"{family} Codex command {record['command']!r} recorded an error"
                )

        if command_execution_checks.get("require_same_turn"):
            command_turn_indexes = {
                record["turn_index"] for record in command_starts
            }
            if len(command_turn_indexes) != 1 or -1 in command_turn_indexes:
                failures.append(
                    f"{family} Codex command executions were not recorded in one turn: {sorted(command_turn_indexes)!r}"
                )

        minimum_parallel_count = int(
            command_execution_checks.get("minimum_parallel_count") or 0
        )
        if maximum_parallel_expected_commands < minimum_parallel_count:
            failures.append(
                f"{family} Codex command executions did not overlap as one parallel batch: expected >= {minimum_parallel_count}, got {maximum_parallel_expected_commands}"
            )

        command_execution_summary = {
            "enabled": True,
            "expected_commands": expected_commands,
            "started": command_starts,
            "completed": command_completions,
            "maximum_parallel_expected_commands": (
                maximum_parallel_expected_commands
            ),
        }

    return {
        "enabled": True,
        "tool_counts": tool_counts,
        "minimum_tool_counts": minimum_tool_counts,
        "required_successful_tools": sorted(required_successful_tools),
        "spawned_agent_ids": sorted(spawned_agent_ids),
        "waited_agent_statuses": waited_agent_statuses,
        "command_execution": command_execution_summary,
    }, failures


def _extract_command_session_id(stdout: str) -> str | None:
    direct = RA._extract_command_session_id(stdout)
    if direct:
        return direct

    def _walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ('session_id', 'sessionId'):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for child in value.values():
                found = _walk(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = _walk(child)
                if found:
                    return found
        return None

    for obj in RA._parse_stdout_json_objects(stdout):
        found = _walk(obj)
        if found:
            return found
    return None


def _extract_command_thread_id(stdout: str) -> str | None:
    """Extract thread_id from a top-level ``thread.started`` JSONL event.

    Only accepts events whose ``type`` is exactly ``thread.started`` and
    that carry a top-level ``thread_id`` or ``threadId`` string.  This
    prevents conflation with unrelated nested objects or session IDs.
    """
    for obj in RA._parse_stdout_json_objects(stdout):
        if obj.get("type") != "thread.started":
            continue
        for key in ("thread_id", "threadId"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_env_placeholders(child) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_env_placeholders(child) for child in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_dotenv_into_environment(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, raw_value = line.split('=', 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        os.environ[key] = os.path.expandvars(value)


def _format_harness_template(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _format_harness_template(child, context)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_format_harness_template(child, context) for child in value]
    if isinstance(value, str):
        try:
            return value.format(**context)
        except (KeyError, ValueError, IndexError):
            return value
    return value


_UNRESOLVED_PLACEHOLDER_RE = re.compile(r'\{[a-z_][a-z0-9_]*\}')


def _contains_unresolved_placeholder(value: Any) -> bool:
    """Return True if a string value contains an unresolved {placeholder}."""
    if isinstance(value, str):
        return bool(_UNRESOLVED_PLACEHOLDER_RE.search(value))
    if isinstance(value, list):
        return any(_contains_unresolved_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_unresolved_placeholder(k) or _contains_unresolved_placeholder(v)
            for k, v in value.items()
        )
    return False


def _is_template_placeholder(value: str) -> bool:
    """Return True if a string is entirely an unresolved template placeholder."""
    return bool(re.fullmatch(r'\{[a-z_][a-z0-9_]*\}', value.strip()))

def _append_claude_agents_arg(command: list[Any], agents: Any) -> list[Any]:
    if not isinstance(agents, dict) or not agents:
        return command
    if any(str(item) == '--agents' for item in command):
        return command
    return [
        *command,
        '--agents',
        json.dumps(agents, sort_keys=True, separators=(',', ':')),
    ]


def _validate_moonshot_anthropic_agentic_contract(  # noqa: PLR0915
    *,
    family: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Reject a Moonshot case that regresses to a raw completion smoke test."""
    is_moonshot_case = family == MOONSHOT_ANTHROPIC_AGENTIC_CASE or bool(config.get(MOONSHOT_ANTHROPIC_AGENTIC_FLAG))
    if not is_moonshot_case:
        return {}, []

    context = (
        "Moonshot Anthropic Messages adapter contract "
        f"({MOONSHOT_ANTHROPIC_ADAPTER_PATH}, alias {MOONSHOT_CANONICAL_ALIAS})"
    )
    failures: list[str] = []

    def _require(condition: bool, requirement: str) -> None:
        if not condition:
            failures.append(f"{context} {requirement}")

    command = config.get("command")
    if not isinstance(command, list):
        command = []
    command_values = [str(value) for value in command]
    command_prompt = ""
    if "-p" in command_values:
        prompt_index = command_values.index("-p") + 1
        if prompt_index < len(command_values):
            command_prompt = command_values[prompt_index]
    allowed_tools = None
    if "--allowedTools" in command_values:
        tools_index = command_values.index("--allowedTools") + 1
        if tools_index < len(command_values):
            allowed_tools = command_values[tools_index]

    _require(
        config.get(MOONSHOT_ANTHROPIC_AGENTIC_FLAG) is True,
        "must set moonshot_anthropic_agentic_only=true",
    )
    _require(
        not isinstance(config.get("http_request"), dict),
        "must launch the Claude agent runtime, not an HTTP smoke request",
    )
    _require(
        command_values[:1] == ["claude"],
        "must launch the Claude CLI for Anthropic Messages ingress",
    )
    _require(
        "--model" not in command_values and "-m" not in command_values,
        "must not use a direct --model selector in place of the child profile",
    )
    _require(
        allowed_tools == "Agent",
        "must require top-level Agent tool dispatch",
    )
    _require(
        "Dispatch to the sota-moonshot agent" in command_prompt,
        "must dispatch the sota-moonshot agent profile",
    )
    _require(
        "tool result" in command_prompt.lower(),
        "must require continuation after tool results",
    )

    agents = config.get("claude_agents")
    _require(
        isinstance(agents, dict) and set(agents) == {MOONSHOT_AGENT_PROFILE},
        "must define exactly the sota-moonshot child profile",
    )
    agent_config = agents.get(MOONSHOT_AGENT_PROFILE) if isinstance(agents, dict) else None
    if not isinstance(agent_config, dict):
        agent_config = {}
    _require(
        agent_config.get("model") == MOONSHOT_CANONICAL_ALIAS,
        "must select the canonical aawm-sota-moonshot alias",
    )
    _require(
        agent_config.get("tools") == ["Read", "Grep"],
        "must expose the deterministic Read then Grep tool sequence",
    )
    agent_prompt = agent_config.get("prompt")
    _require(
        isinstance(agent_prompt, str)
        and "After the Read tool result" in agent_prompt
        and "After the Grep tool result" in agent_prompt,
        "must require tool-result continuation before the child final answer",
    )

    declared_candidates = config.get("verification_declared_candidates")
    declared_pairs = (
        {
            (candidate.get("model"), candidate.get("route_family"))
            for candidate in declared_candidates
            if isinstance(candidate, dict)
        }
        if isinstance(declared_candidates, list)
        else set()
    )
    _require(
        config.get("verification_alias") == MOONSHOT_CANONICAL_ALIAS,
        "must report the canonical Moonshot alias",
    )
    _require(
        declared_pairs == {(model, MOONSHOT_ANTHROPIC_ADAPTER_PATH) for model in MOONSHOT_SELECTED_MODELS},
        "must declare only k3-max and k3-high through the Moonshot adapter",
    )

    _require(
        config.get("allowed_generation_routes") == ["/anthropic/v1/messages"],
        "must validate Anthropic Messages ingress only",
    )
    required_trace_tags = set(config.get("required_trace_tags") or [])
    _require(
        {
            "route:anthropic_messages",
            f"route:{MOONSHOT_ANTHROPIC_ADAPTER_PATH}",
            "anthropic-kimi-chat-completions-adapter",
            f"model-alias:{MOONSHOT_CANONICAL_ALIAS}",
            f"anthropic-auto-agent-alias:{MOONSHOT_CANONICAL_ALIAS}",
        }
        <= required_trace_tags,
        "must require Moonshot adapter route and alias trace tags",
    )
    _require(
        f"claude-code.{MOONSHOT_AGENT_PROFILE}" in set(config.get("required_trace_names") or []),
        "must require the sota-moonshot child trace",
    )

    transcript_validation = config.get("transcript_tool_use_validation")
    expected_agents = transcript_validation.get("expected_agents") if isinstance(transcript_validation, dict) else None
    expected_agent = (
        expected_agents[0]
        if isinstance(expected_agents, list) and len(expected_agents) == 1 and isinstance(expected_agents[0], dict)
        else {}
    )
    _require(
        expected_agent.get("agent_type") == MOONSHOT_AGENT_PROFILE,
        "must validate the sota-moonshot child transcript",
    )
    _require(
        expected_agent.get("expected_tool_counts") == {"Read": 1, "Grep": 1}
        and expected_agent.get("expected_tool_sequence") == ["Read", "Grep"]
        and expected_agent.get("require_tool_result_before_next_tool_use") is True
        and expected_agent.get("maximum_tool_uses_per_assistant_message") == 1,
        "must require sequential tool use with tool-result replay",
    )

    command_json_checks = config.get("command_json_checks")
    required_result = (
        command_json_checks.get("required_contains", {}).get("result")
        if isinstance(command_json_checks, dict) and isinstance(command_json_checks.get("required_contains"), dict)
        else None
    )
    _require(
        required_result == "MOONSHOT ANTHROPIC AGENTIC TOOL CONTINUATION PASSED",
        "must require the final agentic completion marker",
    )

    session_validation = config.get("session_history_validation")
    expected_rows = session_validation.get("expected_rows") if isinstance(session_validation, dict) else None
    session_row = (
        expected_rows[0]
        if isinstance(expected_rows, list) and len(expected_rows) == 1 and isinstance(expected_rows[0], dict)
        else {}
    )
    required_one_of = session_row.get("required_one_of")
    metadata_required_equals = session_row.get("metadata_required_equals")
    metadata_required_truthy = set(session_row.get("metadata_required_truthy") or [])
    _require(
        isinstance(required_one_of, dict)
        and set(required_one_of.get("provider") or []) == {"kimi_code"}
        and set(required_one_of.get("model") or [])
        == {"kimi_code/k3-max", "kimi_code/k3-high"},
        "must require provider-prefixed Kimi Code k3-max or k3-high session metadata",
    )
    _require(
        isinstance(metadata_required_equals, dict)
        and metadata_required_equals.get("model_alias_label") == MOONSHOT_CANONICAL_ALIAS
        and metadata_required_equals.get("requested_model_alias") == MOONSHOT_CANONICAL_ALIAS
        and metadata_required_equals.get("anthropic_auto_agent_alias") == MOONSHOT_CANONICAL_ALIAS,
        "must require canonical alias metadata in session history",
    )
    _require(
        {
            "anthropic_auto_agent_selected_provider",
            "anthropic_auto_agent_selected_model",
            "anthropic_auto_agent_selected_route_family",
            "aawm_alias_routing_audit_events",
        }
        <= metadata_required_truthy,
        "must require selected Moonshot adapter metadata in session history",
    )
    _require(
        not bool(config.get("warning_only")) and not bool(config.get("skip_generation_quality_checks")),
        "must not downgrade the agentic acceptance path to a warning or smoke check",
    )

    return {
        "adapter_path": MOONSHOT_ANTHROPIC_ADAPTER_PATH,
        "canonical_alias": MOONSHOT_CANONICAL_ALIAS,
        "agent_profile": MOONSHOT_AGENT_PROFILE,
        "allowed_generation_routes": config.get("allowed_generation_routes"),
        "declared_models": sorted(
            model
            for model, route_family in declared_pairs
            if route_family == MOONSHOT_ANTHROPIC_ADAPTER_PATH and isinstance(model, str)
        ),
    }, failures


def _docker_status_for_container(container_name: str) -> str:
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name=^{container_name}$', '--format', '{{.Status}}'],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return ''
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def _target_profile_settings(
    *,
    config: dict[str, Any],
    target: str,
    litellm_base_url: str | None = None,
    anthropic_base_url: str | None = None,
    docker_container_name: str | None = None,
    expected_trace_environment: str | None = None,
) -> dict[str, str]:
    configured_profiles = config.get('target_profiles')
    profiles = dict(BUILT_IN_TARGET_PROFILES)
    if isinstance(configured_profiles, dict):
        for name, profile in configured_profiles.items():
            if isinstance(name, str) and isinstance(profile, dict):
                profiles[name] = {
                    key: str(value)
                    for key, value in profile.items()
                    if isinstance(value, (str, int, float))
                }

    if target not in profiles:
        valid = ', '.join(sorted(profiles))
        raise SystemExit(f'Unknown adapter target `{target}`. Valid targets: {valid}')

    profile = dict(profiles[target])
    if litellm_base_url:
        profile['litellm_base_url'] = litellm_base_url.rstrip('/')
    if anthropic_base_url:
        profile['anthropic_base_url'] = anthropic_base_url.rstrip('/')
    if docker_container_name:
        profile['docker_container_name'] = docker_container_name
    if expected_trace_environment:
        profile['expected_trace_environment'] = expected_trace_environment

    profile.setdefault('litellm_base_url', config.get('litellm_base_url', 'http://127.0.0.1:4001'))
    profile.setdefault('anthropic_base_url', f"{profile['litellm_base_url'].rstrip('/')}/anthropic")
    profile.setdefault('docker_container_name', 'litellm-dev')
    profile.setdefault('expected_trace_environment', target)
    profile.setdefault('expected_runtime_environment', profile['docker_container_name'])
    profile['litellm_base_url'] = profile['litellm_base_url'].rstrip('/')
    profile['anthropic_base_url'] = profile['anthropic_base_url'].rstrip('/')
    return profile


def _apply_target_profile_to_config(  # noqa: PLR0915
    config: dict[str, Any],
    *,
    target: str,
    profile: dict[str, str],
) -> dict[str, Any]:
    updated_config = dict(config)
    updated_config['target_profile'] = target
    updated_config['litellm_base_url'] = profile['litellm_base_url']
    updated_config['expected_trace_environment'] = profile['expected_trace_environment']

    cases = updated_config.get('cases') or {}
    updated_cases: dict[str, Any] = {}
    for case_name, case_config in cases.items():
        if not isinstance(case_config, dict):
            updated_cases[case_name] = case_config
            continue

        updated_case = dict(case_config)
        case_env = dict(updated_case.get('env') or {})
        if 'ANTHROPIC_BASE_URL' in case_env:
            case_env['ANTHROPIC_BASE_URL'] = profile['anthropic_base_url']
        updated_case['env'] = case_env

        runtime_postconditions = dict(updated_case.get('runtime_postconditions') or {})
        runtime_postconditions['healthcheck_url'] = (
            f"{profile['litellm_base_url']}/health/liveliness"
        )
        runtime_postconditions['docker_container_name'] = profile['docker_container_name']
        updated_case['runtime_postconditions'] = runtime_postconditions
        skip_trace_environment_validation = bool(
            updated_case.get('skip_trace_environment_validation')
        )
        if skip_trace_environment_validation:
            updated_case.pop("expected_trace_environment", None)
        else:
            updated_case['expected_trace_environment'] = profile[
                'expected_trace_environment'
            ]
        updated_case.setdefault('require_trace_user_id', True)
        updated_case['target_profile'] = target
        updated_case['case_name'] = case_name
        tenant_id = _resolve_harness_tenant_id(updated_config, updated_case)
        cli_kind = str(updated_case.get('cli_passthrough') or '').strip().lower()
        repository_tenant_id = (
            _resolve_harness_repository() if cli_kind == 'codex' else None
        )
        expected_session_history_tenant_id = (
            _resolve_expected_session_history_tenant_id(
                tenant_id,
                repository=repository_tenant_id,
            )
        )
        expected_original_tenant_id = tenant_id
        updated_case['tenant_id'] = tenant_id
        harness_run_id = str(
            updated_case.get('harness_run_id')
            or f'{case_name}-{uuid.uuid4().hex[:12]}'
        )
        updated_case['harness_run_id'] = harness_run_id
        template_context = {
            'target': target,
            'case_name': case_name,
            'tenant_id': tenant_id,
            'harness_run_id': harness_run_id,
            'repository_root': str(ROOT),
            'litellm_base_url': profile['litellm_base_url'],
            'anthropic_base_url': profile['anthropic_base_url'],
        }
        updated_case = _format_harness_template(updated_case, template_context)
        command = updated_case.get('command')
        if isinstance(command, list):
            updated_case['command'] = _append_claude_agents_arg(
                command,
                updated_case.get('claude_agents'),
            )
        session_history_validation = dict(
            updated_case.get('session_history_validation') or {}
        )
        session_history_validation.setdefault(
            'expected_litellm_environment',
            profile.get('expected_runtime_environment', profile['expected_trace_environment']),
        )
        metadata_required_equals = dict(
            session_history_validation.get('metadata_required_equals') or {}
        )
        if not skip_trace_environment_validation:
            metadata_required_equals['trace_environment'] = profile[
                'expected_trace_environment'
            ]
        metadata_required_equals['litellm_environment'] = profile.get(
            'expected_runtime_environment', profile['expected_trace_environment']
        )
        require_trace_user_id = (
            updated_case.get('require_trace_user_id', True) is not False
        )
        if require_trace_user_id:
            metadata_required_equals['tenant_id'] = (
                expected_session_history_tenant_id
            )
            metadata_required_equals.pop('aawm_original_tenant_id', None)
            metadata_required_equals.pop('aawm_harness_tenant_alias', None)
            if expected_session_history_tenant_id != expected_original_tenant_id:
                metadata_required_equals['aawm_original_tenant_id'] = (
                    expected_original_tenant_id
                )
                if _is_harness_tenant_alias(expected_original_tenant_id):
                    metadata_required_equals['aawm_harness_tenant_alias'] = True
        session_history_validation['metadata_required_equals'] = (
            metadata_required_equals
        )
        metadata_required_truthy = list(
            session_history_validation.get('metadata_required_truthy') or []
        )
        if require_trace_user_id and 'tenant_id_source' not in metadata_required_truthy:
            metadata_required_truthy.append('tenant_id_source')
        session_history_validation['metadata_required_truthy'] = metadata_required_truthy
        expected_rows = session_history_validation.get('expected_rows')
        has_expected_rows = isinstance(expected_rows, list) and bool(expected_rows)
        if has_expected_rows:
            session_history_validation['expected_rows'] = [
                _with_expected_row_tenant(row, expected_session_history_tenant_id)
                for row in expected_rows
            ]
        elif require_trace_user_id:
            session_history_validation['expected_tenant_id'] = (
                expected_session_history_tenant_id
            )
        session_history_validation.setdefault('require_runtime_identity', True)
        updated_case['session_history_validation'] = session_history_validation
        if isinstance(updated_case.get('http_request'), dict):
            updated_case = _ensure_http_harness_context(
                updated_case,
                profile=profile,
                target=target,
                case_name=case_name,
            )
        elif isinstance(updated_case.get('cli_passthrough'), str):
            updated_case = _ensure_cli_harness_context(
                updated_case,
                profile=profile,
                target=target,
                case_name=case_name,
            )
        else:
            updated_case.setdefault('expected_user_ids', [tenant_id])
            updated_case = _ensure_claude_tenant_header(updated_case, tenant_id)
            updated_case = RA._ensure_claude_harness_headers(
                updated_case,
                target=target,
                case_name=case_name,
            )
        _apply_profile_validation_db_overrides(updated_case, profile)
        updated_cases[case_name] = updated_case

    updated_config['cases'] = updated_cases
    return updated_config


_DB_VALIDATION_KEYS = (
    'session_history_validation',
    'rate_limit_observations_validation',
    'provider_error_observations_validation',
    'tool_activity_validation',
)


def _apply_profile_validation_db_overrides(
    case_config: dict[str, Any],
    profile: dict[str, str],
) -> None:
    """Override DB connection settings in all DB-backed validation blocks.

    Only applies when the target profile carries ``validation_db_*`` keys.
    Profiles without them (e.g. prod) leave case-level settings untouched.
    The password is never stored in the config; only the container name and
    env-var name are recorded so ``_validation_db_settings`` can resolve it
    at runtime via ``_resolve_container_env_value``.
    """
    db_host = profile.get('validation_db_host')
    if not db_host:
        return
    container_name = str(
        profile.get('validation_db_password_container')
        or profile.get('docker_container_name')
        or ''
    )
    password_container_env = profile.get('validation_db_password_container_env', '')
    for key in _DB_VALIDATION_KEYS:
        block = case_config.get(key)
        if not isinstance(block, dict):
            continue
        block['db_host'] = db_host
        if profile.get('validation_db_port'):
            block['db_port'] = int(profile['validation_db_port'])
        if profile.get('validation_db_name'):
            block['db_name'] = profile['validation_db_name']
        if profile.get('validation_db_user'):
            block['db_user'] = profile['validation_db_user']
        if password_container_env and container_name:
            block['db_password_container'] = container_name
            block['db_password_container_env'] = password_container_env


def _resolve_harness_tenant_id(
    suite_config: dict[str, Any],
    case_config: dict[str, Any],
) -> str:
    value = case_config.get('tenant_id', suite_config.get('default_tenant_id'))
    if isinstance(value, (str, int, float)) and str(value).strip():
        return str(value).strip()
    return 'adapter-harness-tenant'


def _is_harness_tenant_alias(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized) and (
        'harness' in normalized or 'validation' in normalized
    )


def _resolve_expected_session_history_tenant_id(
    tenant_id: str,
    *,
    repository: str | None = None,
) -> str:
    if _is_harness_tenant_alias(tenant_id):
        if isinstance(repository, str) and repository.strip():
            return repository.strip()
        return 'litellm'
    return tenant_id


def _with_expected_row_tenant(row: Any, tenant_id: str) -> Any:
    if not isinstance(row, dict):
        return row
    updated_row = dict(row)
    required_equals = dict(updated_row.get('required_equals') or {})
    required_equals.setdefault('tenant_id', tenant_id)
    updated_row['required_equals'] = required_equals
    return updated_row


def _ensure_claude_tenant_header(config: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    updated = dict(config)
    env = dict(updated.get('env') or {})
    headers = RA._parse_claude_custom_header_lines(env.get('ANTHROPIC_CUSTOM_HEADERS'))
    if not any(key.lower() == 'x-aawm-tenant-id' for key, _ in headers):
        headers.append(('x-aawm-tenant-id', tenant_id))
    env['ANTHROPIC_CUSTOM_HEADERS'] = RA._format_claude_custom_header_lines(headers)
    updated['env'] = env
    return updated


def _append_codex_tenant_config_arg(command: list[Any], tenant_id: str) -> list[Any]:
    return _append_codex_header_config_arg(
        command,
        "x-aawm-tenant-id",
        tenant_id,
    )


def _append_codex_header_config_arg(
    command: list[Any],
    header_name: str,
    header_value: str,
) -> list[Any]:
    header_config = f'model_providers.{{codex_profile}}.http_headers.{header_name}="{header_value}"'
    updated = list(command)
    header_path = f'.http_headers.{header_name}='
    for index, item in enumerate(updated):
        item_text = str(item)
        if item_text.startswith('model_providers.') and header_path in item_text:
            updated[index] = header_config
            return updated
    try:
        insert_at = updated.index('--json')
    except ValueError:
        insert_at = max(0, len(updated) - 1)
    updated[insert_at:insert_at] = ['-c', header_config]
    return updated


def _normalize_harness_repository(value: str) -> str:
    repository = value.strip()
    if repository.startswith('git@') and ':' in repository:
        repository = repository.split(':', 1)[1]
    elif 'github.com/' in repository:
        repository = repository.split('github.com/', 1)[1]
    repository = repository.strip().strip('/')
    if repository.endswith('.git'):
        repository = repository[:-4]
    return repository or ROOT.name


def _resolve_harness_repository() -> str:
    for args in (('remote', 'get-url', 'origin'), ('rev-parse', '--show-toplevel')):
        value = RA._git_value(*args)
        if value:
            return _normalize_harness_repository(value)
    return ROOT.name


def _ensure_http_harness_context(
    config: dict[str, Any],
    *,
    profile: dict[str, str],
    target: str,
    case_name: str,
) -> dict[str, Any]:
    updated = dict(config)
    request_config = dict(updated.get('http_request') or {})
    headers = dict(request_config.get('headers') or {})
    tenant_id = str(updated.get('tenant_id') or 'adapter-harness-tenant')
    expected_user_ids = [
        str(value).strip()
        for value in (updated.get('expected_user_ids') or [])
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    # Fix 1: template placeholders like {harness_user_id} must not become the
    # effective ID.  Only concrete (non-placeholder) values are preserved.
    concrete_user_ids = [
        uid for uid in expected_user_ids if not _is_template_placeholder(uid)
    ]
    harness_user_id = (
        concrete_user_ids[0]
        if concrete_user_ids
        else RA._build_claude_harness_user_id(target=target, case_name=case_name)
    )
    # Resolve session ID with cross-field fallback: prefer a concrete
    # request session_id, then a concrete expected_trace_session_id, and
    # only derive <user>.session when neither provides a concrete value.
    _req_sid = str(request_config.get('session_id') or '').strip()
    _exp_sid = str(updated.get('expected_trace_session_id') or '').strip()
    if _req_sid and not _is_template_placeholder(_req_sid):
        session_id = _req_sid
    elif _exp_sid and not _is_template_placeholder(_exp_sid):
        session_id = _exp_sid
    else:
        session_id = f'{harness_user_id}.session'
    if request_config.get('add_default_authorization') is not False:
        headers.setdefault('authorization', 'Bearer sk-1234')
    # Controlled headers: replace unresolved placeholder values rather than
    # preserving them via setdefault.  Concrete explicit values remain
    # authoritative.
    _controlled_http = {
        'x-litellm-end-user-id': harness_user_id,
        'langfuse_trace_user_id': harness_user_id,
        'langfuse_trace_name': case_name,
        'session_id': session_id,
        'x-aawm-tenant-id': tenant_id,
    }
    for hdr_name, hdr_value in _controlled_http.items():
        existing = headers.get(hdr_name)
        if existing is None or _contains_unresolved_placeholder(existing):
            headers[hdr_name] = hdr_value
    headers.setdefault('user-agent', 'AAWMNativePassthroughHarness/0.1')

    request_config['headers'] = headers
    request_config['session_id'] = session_id
    request_config['litellm_base_url'] = profile['litellm_base_url']
    updated['http_request'] = request_config
    updated['expected_user_ids'] = [harness_user_id]
    updated['expected_trace_session_id'] = session_id
    if updated.get('match_trace_session_id_from_stdout') is None:
        updated['match_trace_session_id_from_stdout'] = True
    return updated


def _ensure_cli_harness_context(
    config: dict[str, Any],
    *,
    profile: dict[str, str],
    target: str,
    case_name: str,
) -> dict[str, Any]:
    updated = dict(config)
    cli_kind = str(updated.get('cli_passthrough') or '').strip().lower()
    if cli_kind not in {'codex', 'grok'}:
        return updated

    tenant_id = str(updated.get('tenant_id') or 'adapter-harness-tenant')
    expected_user_ids = [
        str(value).strip()
        for value in (updated.get('expected_user_ids') or [])
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    # Fix 1: template placeholders like {harness_user_id} must not become the
    # effective ID.  Only concrete (non-placeholder) values are preserved.
    concrete_user_ids = [
        uid for uid in expected_user_ids if not _is_template_placeholder(uid)
    ]
    harness_user_id = (
        concrete_user_ids[0]
        if concrete_user_ids
        else RA._build_claude_harness_user_id(target=target, case_name=case_name)
    )
    raw_session_id = str(
        updated.get('expected_trace_session_id') or ''
    ).strip()
    # Derive a concrete session ID when the configured value is a placeholder.
    session_id = (
        raw_session_id
        if raw_session_id and not _is_template_placeholder(raw_session_id)
        else f'{harness_user_id}.session'
    )
    repository = _resolve_harness_repository()
    codex_profile = 'litellm' if target == 'prod' else 'litellm-dev'
    context = {
        'target': target,
        'case_name': case_name,
        'harness_user_id': harness_user_id,
        'session_id': session_id,
        'repository': repository,
        'repository_root': str(ROOT),
        'codex_home': str(pathlib.Path.home() / '.codex'),
        'litellm_base_url': profile['litellm_base_url'],
        'anthropic_base_url': profile['anthropic_base_url'],
        'codex_profile': codex_profile,
    }

    updated = _format_harness_template(updated, context)
    env = dict(updated.get('env') or {})
    controlled_headers = [
        ('x-litellm-end-user-id', harness_user_id),
        ('langfuse_trace_user_id', harness_user_id),
        ('langfuse_trace_name', case_name),
        ('x-aawm-tenant-id', tenant_id),
        ('x-aawm-repository', repository),
    ]
    if cli_kind == 'codex':
        controlled_headers.append(('session_id', session_id))
        command = updated.get('command')
        if isinstance(command, list):
            for header_name, header_value in controlled_headers:
                command = _append_codex_header_config_arg(
                    command,
                    header_name,
                    header_value,
                )
            updated['command'] = _format_harness_template(
                command,
                context,
            )
    if cli_kind == 'grok':
        env['GROK_CLI_CHAT_PROXY_BASE_URL'] = (
            f"{profile['litellm_base_url']}/grok/v1"
        )
        env.setdefault('GROK_DISABLE_UPDATE_CHECK', '1')
        env.setdefault('GROK_SANDBOX', 'workspace')
    updated['env'] = env
    if cli_kind == 'grok':
        updated['expected_user_ids'] = []
        updated['require_trace_user_id'] = False
    else:
        updated['expected_user_ids'] = [harness_user_id]
    if cli_kind == 'codex':
        updated['expected_trace_session_id'] = session_id
    else:
        updated.pop('expected_trace_session_id', None)
    if updated.get('match_trace_session_id_from_stdout') is None:
        updated['match_trace_session_id_from_stdout'] = cli_kind != 'codex'
    updated.setdefault('require_trace_user_id', True)
    return updated


def _missing_required_env(config: dict[str, Any]) -> list[str]:
    required_env = config.get('required_env') or []
    if not isinstance(required_env, list):
        return []
    return [
        value
        for value in required_env
        if isinstance(value, str) and value and not os.environ.get(value)
    ]


def _normalize_expected_trace_user_ids_by_name(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for trace_name, user_id in value.items():
        if not isinstance(trace_name, str):
            continue
        trace_name = trace_name.strip()
        if not trace_name:
            continue
        if not isinstance(user_id, (str, int, float)):
            continue
        user_id = str(user_id).strip()
        if user_id:
            normalized[trace_name] = user_id
    return normalized


def _resolve_trace_lookup_user_id(
    expected_user_ids: list[str],
    expected_trace_user_ids_by_name: dict[str, str],
) -> str | None:
    if expected_user_ids:
        return expected_user_ids[0]

    expected_user_ids_from_trace_names = sorted(
        {
            user_id
            for user_id in expected_trace_user_ids_by_name.values()
            if isinstance(user_id, str) and user_id
        }
    )
    if len(expected_user_ids_from_trace_names) == 1:
        return expected_user_ids_from_trace_names[0]
    return None


def _validate_trace_user_ids_by_name(
    *,
    family: str,
    traces: list[dict[str, Any]],
    expected: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    actual_by_name: dict[str, list[str]] = {}
    for trace in traces:
        trace_name = trace.get('name')
        user_id = trace.get('userId')
        if not isinstance(trace_name, str) or not trace_name:
            continue
        if not isinstance(user_id, str) or not user_id:
            continue
        actual_by_name.setdefault(trace_name, [])
        if user_id not in actual_by_name[trace_name]:
            actual_by_name[trace_name].append(user_id)

    failures: list[str] = []
    for trace_name, expected_user_id in expected.items():
        actual_user_ids = actual_by_name.get(trace_name, [])
        if expected_user_id not in actual_user_ids:
            failures.append(
                f'{family} trace {trace_name} missing user id {expected_user_id}'
            )

    summary = {
        'expected': expected,
        'actual_by_name': {
            trace_name: sorted(user_ids)
            for trace_name, user_ids in sorted(actual_by_name.items())
        },
    }
    return summary, failures


def _validate_command_output_json(*, family: str, stdout: str, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    parsed = _parse_command_output_json(stdout)
    if parsed is None:
        return {'parsed': None}, [f'{family} command stdout did not contain JSON']

    required_equals = checks.get('required_equals', {}) or {}
    required_contains = checks.get('required_contains', {}) or {}
    required_regex = checks.get('required_regex', {}) or {}
    required_minimums = checks.get('required_minimums', {}) or {}

    equals_hits: dict[str, Any] = {}
    contains_hits: dict[str, Any] = {}
    regex_hits: dict[str, Any] = {}
    minimum_hits: dict[str, Any] = {}

    for path, expected in required_equals.items():
        actual = _extract_path_value(parsed, path)
        equals_hits[path] = actual
        if actual != expected:
            failures.append(f'{family} command JSON mismatch for `{path}`: expected {expected!r}, got {actual!r}')

    for path, expected_substring in required_contains.items():
        actual = _extract_path_value(parsed, path)
        contains_hits[path] = actual
        if not isinstance(actual, str) or not isinstance(expected_substring, str) or expected_substring not in actual:
            failures.append(
                f'{family} command JSON missing substring for `{path}`: expected to contain {expected_substring!r}, got {actual!r}'
            )

    for path, expected_pattern in required_regex.items():
        actual = _extract_path_value(parsed, path)
        regex_hits[path] = actual
        if not isinstance(actual, str) or not isinstance(expected_pattern, str):
            failures.append(
                f'{family} command JSON regex mismatch for `{path}`: expected pattern {expected_pattern!r}, got {actual!r}'
            )
            continue
        try:
            matched = re.search(expected_pattern, actual) is not None
        except re.error as exc:
            failures.append(
                f'{family} command JSON invalid regex for `{path}`: {expected_pattern!r} ({exc})'
            )
            continue
        if not matched:
            failures.append(
                f'{family} command JSON regex mismatch for `{path}`: expected pattern {expected_pattern!r}, got {actual!r}'
            )

    for path, minimum in required_minimums.items():
        actual = _extract_path_value(parsed, path)
        minimum_hits[path] = actual
        if not isinstance(actual, (int, float)) or actual < minimum:
            failures.append(f'{family} command JSON below minimum for `{path}`: expected >= {minimum!r}, got {actual!r}')

    return {
        'parsed': parsed,
        'required_equals_hits': equals_hits,
        'required_contains_hits': contains_hits,
        'required_regex_hits': regex_hits,
        'required_minimum_hits': minimum_hits,
    }, failures


def _validate_no_successful_empty_command_output(
    *,
    family: str,
    stdout: str,
    stderr: str,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not bool(checks.get('fail_empty_success')):
        return {'enabled': False}, []

    parsed = _parse_command_output_json(stdout)
    combined_output = f'{stdout}\n{stderr}'
    adapter_diagnostic_hit = (
        'OpenRouter Responses adapter returned empty successful response'
        in combined_output
    )
    summary: dict[str, Any] = {
        'enabled': True,
        'adapter_diagnostic_hit': adapter_diagnostic_hit,
        'parsed': parsed,
    }
    failures: list[str] = []
    if adapter_diagnostic_hit:
        failures.append(
            f'{family} successful empty OpenRouter adapter diagnostic surfaced'
        )

    if not isinstance(parsed, dict):
        return summary, failures

    is_error = parsed.get('is_error')
    result = parsed.get('result')
    input_tokens = _extract_path_value(parsed, 'usage.input_tokens')
    output_tokens = _extract_path_value(parsed, 'usage.output_tokens')
    result_empty = not isinstance(result, str) or not result.strip()
    input_zero = isinstance(input_tokens, (int, float)) and input_tokens <= 0
    output_zero = isinstance(output_tokens, (int, float)) and output_tokens <= 0
    summary.update(
        {
            'is_error': is_error,
            'result_empty': result_empty,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }
    )

    if is_error is False and result_empty:
        failures.append(f'{family} successful empty command result')
    if is_error is False and input_zero and output_zero:
        failures.append(
            f'{family} successful empty command usage: input_tokens={input_tokens!r}, output_tokens={output_tokens!r}'
        )

    return summary, failures



def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _extract_request_payload_path_values(value: Any, path: str) -> list[Any]:
    segments = [segment for segment in path.split('.') if segment]
    if not segments:
        return [value]

    def _walk(current: Any, remaining: list[str]) -> list[Any]:
        if not remaining:
            return [current]

        segment = remaining[0]
        rest = remaining[1:]
        if segment == '**':
            matches = _walk(current, rest)
            if isinstance(current, dict):
                for child in current.values():
                    matches.extend(_walk(child, remaining))
            elif isinstance(current, list):
                for child in current:
                    matches.extend(_walk(child, remaining))
            return matches

        if isinstance(current, dict):
            if segment not in current:
                return []
            return _walk(current[segment], rest)

        if isinstance(current, list):
            if segment.isdigit():
                index = int(segment)
                if 0 <= index < len(current):
                    return _walk(current[index], rest)
                return []
            matches: list[Any] = []
            for child in current:
                matches.extend(_walk(child, remaining))
            return matches

        return []

    return _walk(value, segments)


def _preview_request_payload_value(value: Any) -> str:
    return RA._preview_request_body_path_value(value)


def _validate_logged_request_payload_checks(  # noqa: PLR0915
    *,
    family: str,
    observations: list[dict[str, Any]],
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    required_paths = _as_string_list(checks.get('required_paths'))
    warning_present_paths = _as_string_list(checks.get('warning_present_paths'))
    forbidden_paths = _as_string_list(checks.get('forbidden_paths'))
    required_equals = checks.get('required_equals') or {}
    required_one_of = checks.get('required_one_of') or {}
    if not isinstance(required_equals, dict):
        required_equals = {}
    if not isinstance(required_one_of, dict):
        required_one_of = {}

    required_path_found: dict[str, bool] = {path: False for path in required_paths}
    required_path_values: dict[str, list[str]] = {path: [] for path in required_paths}
    warning_path_hits: dict[str, list[dict[str, str]]] = {
        path: [] for path in warning_present_paths
    }
    forbidden_path_hits: dict[str, list[dict[str, str]]] = {
        path: [] for path in forbidden_paths
    }
    required_equals_found: dict[str, bool] = {
        str(path): False for path in required_equals
    }
    required_equals_observed: dict[str, list[str]] = {
        str(path): [] for path in required_equals
    }
    required_one_of_found: dict[str, bool] = {str(path): False for path in required_one_of}
    required_one_of_observed: dict[str, list[str]] = {
        str(path): [] for path in required_one_of
    }

    for observation in observations:
        request_body = RA._extract_logged_request_body(observation)
        if request_body is None:
            continue

        observation_id = str(observation.get('id'))
        for path in required_paths:
            values = _extract_request_payload_path_values(request_body, path)
            if not values:
                continue
            required_path_found[path] = True
            for value in values:
                preview = _preview_request_payload_value(value)
                if preview not in required_path_values[path]:
                    required_path_values[path].append(preview)

        for path in warning_present_paths:
            for value in _extract_request_payload_path_values(request_body, path):
                warning_path_hits[path].append(
                    {
                        'observation_id': observation_id,
                        'value': _preview_request_payload_value(value),
                    }
                )

        for path in forbidden_paths:
            for value in _extract_request_payload_path_values(request_body, path):
                forbidden_path_hits[path].append(
                    {
                        'observation_id': observation_id,
                        'value': _preview_request_payload_value(value),
                    }
                )

        for raw_path, expected in required_equals.items():
            path = str(raw_path)
            for value in _extract_request_payload_path_values(request_body, path):
                preview = _preview_request_payload_value(value)
                if preview not in required_equals_observed[path]:
                    required_equals_observed[path].append(preview)
                if value == expected:
                    required_equals_found[path] = True

        for raw_path, allowed_values in required_one_of.items():
            path = str(raw_path)
            allowed_list = allowed_values if isinstance(allowed_values, list) else []
            for value in _extract_request_payload_path_values(request_body, path):
                preview = _preview_request_payload_value(value)
                if preview not in required_one_of_observed[path]:
                    required_one_of_observed[path].append(preview)
                if any(value == allowed for allowed in allowed_list):
                    required_one_of_found[path] = True

    for path, found in required_path_found.items():
        if not found:
            failures.append(f'{family} missing request payload path: {path}')

    for path, found in required_equals_found.items():
        if found:
            continue
        observed = required_equals_observed.get(path) or ['<missing>']
        failures.append(
            f'{family} request payload `{path}` did not equal '
            f'{required_equals[path]!r}; observed: {", ".join(observed)}'
        )

    for path, found in required_one_of_found.items():
        if found:
            continue
        observed = required_one_of_observed.get(path) or ['<missing>']
        failures.append(
            f'{family} request payload `{path}` was not one of '
            f'{required_one_of[path]!r}; observed: {", ".join(observed)}'
        )

    for path, hits in forbidden_path_hits.items():
        if not hits:
            continue
        observed_values = sorted({hit['value'] for hit in hits})
        failures.append(
            f'{family} request payload includes forbidden path `{path}` '
            f'with value(s): {", ".join(observed_values)}'
        )

    for path, hits in warning_path_hits.items():
        if not hits:
            continue
        observed_values = sorted({hit['value'] for hit in hits})
        warnings.append(
            f'{family} request payload includes warning path `{path}` with value(s): '
            + ', '.join(observed_values)
        )

    summary = {
        'required_paths_found': required_path_found,
        'required_path_values': required_path_values,
        'required_equals_found': required_equals_found,
        'required_equals_observed': required_equals_observed,
        'required_one_of_found': required_one_of_found,
        'required_one_of_observed': required_one_of_observed,
        'forbidden_path_hits': forbidden_path_hits,
        'warning_present_path_hits': warning_path_hits,
    }
    return summary, failures, warnings


def _stream_tool_state_from_output_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get('type')
    if item_type not in {
        'function_call',
        'local_shell_call',
        'apply_patch_call',
        'custom_tool_call',
        'mcp_call',
    }:
        return None
    tool_name = item.get('name')
    if not isinstance(tool_name, str) or not tool_name.strip():
        tool_name = item_type
    arguments = None
    for key in ('arguments', 'input', 'action', 'patch'):
        if item.get(key) is not None:
            arguments = item.get(key)
            break
    if isinstance(arguments, str):
        arguments_text = arguments
    elif arguments is None:
        arguments_text = ''
    else:
        try:
            arguments_text = json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            arguments_text = str(arguments)
    return {
        'type': item_type,
        'name': tool_name,
        'call_id': item.get('call_id') or item.get('id'),
        'arguments': arguments_text,
    }


def _collect_stream_tool_call_state(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    event_types: list[str] = []
    event_counts: dict[str, int] = {}
    tool_state: list[dict[str, Any]] = []
    compacted_tool_state = False

    for observation in observations:
        metadata = observation.get('metadata')
        if isinstance(metadata, dict):
            metadata_event_types = metadata.get('responses_stream_event_types')
            if isinstance(metadata_event_types, list):
                for event_type in metadata_event_types:
                    if isinstance(event_type, str) and event_type not in event_types:
                        event_types.append(event_type)
            metadata_event_counts = metadata.get('responses_stream_event_counts')
            if isinstance(metadata_event_counts, dict):
                for event_type, count in metadata_event_counts.items():
                    if not isinstance(event_type, str):
                        continue
                    if isinstance(count, (int, float)):
                        event_counts[event_type] = event_counts.get(event_type, 0) + int(count)
            metadata_tool_state = metadata.get('responses_stream_tool_state')
            if isinstance(metadata_tool_state, list):
                for item in metadata_tool_state:
                    if isinstance(item, dict):
                        tool_state.append(dict(item))
            elif (
                isinstance(metadata_tool_state, dict)
                and metadata_tool_state.get('type')
                == 'litellm_langfuse_metadata_compacted'
            ):
                compacted_tool_state = True
                for item in metadata_tool_state.get('sample_tool_calls') or []:
                    if not isinstance(item, dict):
                        continue
                    tool_state.append(
                        {
                            'type': item.get('type'),
                            'name': item.get('name'),
                            'call_id': item.get('call_id'),
                            'arguments': '',
                            'arguments_compacted': True,
                        }
                    )

        output = observation.get('output')
        if isinstance(output, dict):
            for path in (
                '_hidden_params.responses_output',
                'hidden_params.responses_output',
                'output',
            ):
                output_items = _extract_path_value(output, path)
                if not isinstance(output_items, list):
                    continue
                for item in output_items:
                    if not isinstance(item, dict):
                        continue
                    state_item = _stream_tool_state_from_output_item(item)
                    if state_item is not None:
                        tool_state.append(state_item)

    deduped_tool_state: list[dict[str, Any]] = []
    seen_tools: set[tuple[str, str, str]] = set()
    for item in tool_state:
        key = (
            str(item.get('type') or ''),
            str(item.get('name') or ''),
            str(item.get('arguments') or ''),
        )
        if key in seen_tools:
            continue
        seen_tools.add(key)
        deduped_tool_state.append(item)

    return {
        'event_types': event_types,
        'event_counts': event_counts,
        'tool_state': deduped_tool_state,
        'tool_names': [
            item.get('name')
            for item in deduped_tool_state
            if isinstance(item.get('name'), str) and item.get('name')
        ],
        'compacted_tool_state': compacted_tool_state,
    }


def _command_tool_state_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get('type')
    if item_type == 'command_execution':
        command = item.get('command')
        if not isinstance(command, str) or not command:
            return None
        return {
            'type': 'function_call',
            'name': 'exec_command',
            'call_id': item.get('id'),
            'arguments': json.dumps({'cmd': command}, sort_keys=True),
            'source': 'command_stdout',
        }
    if item_type != 'tool_use':
        return None

    tool_name = item.get('name')
    if not isinstance(tool_name, str) or not tool_name:
        return None
    tool_input = item.get('input')
    if tool_name == 'Bash':
        tool_name = 'exec_command'
        if isinstance(tool_input, dict):
            normalized_input = dict(tool_input)
            if 'command' in normalized_input:
                normalized_input['cmd'] = normalized_input.pop('command')
            tool_input = normalized_input
    try:
        arguments = json.dumps(tool_input, sort_keys=True)
    except (TypeError, ValueError):
        arguments = str(tool_input or '')
    return {
        'type': 'function_call',
        'name': tool_name,
        'call_id': item.get('id'),
        'arguments': arguments,
        'source': 'command_stdout',
    }


def _collect_command_tool_call_state(stdout: str) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            state_item = _command_tool_state_from_item(value)
            if state_item is not None:
                collected.append(state_item)
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    for obj in RA._parse_stdout_json_objects(stdout):
        _walk(obj)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in collected:
        key = (
            str(item.get('type') or ''),
            str(item.get('name') or ''),
            str(item.get('arguments') or ''),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _stream_tool_state_matches_expected(
    item: dict[str, Any],
    expected: dict[str, Any],
    *,
    check_arguments: bool = True,
) -> bool:
    expected_name = expected.get('tool_name')
    if expected_name is not None and item.get('name') != expected_name:
        return False
    name_one_of = expected.get('tool_name_one_of')
    if isinstance(name_one_of, list) and name_one_of:
        if item.get('name') not in set(name_one_of):
            return False

    expected_type = expected.get('tool_type')
    if expected_type is not None and item.get('type') != expected_type:
        return False
    type_one_of = expected.get('tool_type_one_of')
    if isinstance(type_one_of, list) and type_one_of:
        if item.get('type') not in set(type_one_of):
            return False

    if not check_arguments:
        return True

    argument_text = str(item.get('arguments') or '')
    required_substrings = []
    configured_substring = expected.get('arguments_required_substring')
    if isinstance(configured_substring, str) and configured_substring:
        required_substrings.append(configured_substring)
    configured_substrings = expected.get('arguments_required_substrings')
    if isinstance(configured_substrings, list):
        required_substrings.extend(
            value
            for value in configured_substrings
            if isinstance(value, str) and value
        )
    return all(substring in argument_text for substring in required_substrings)


def _validate_stream_tool_call_state(
    *,
    family: str,
    observations: list[dict[str, Any]],
    checks: dict[str, Any],
    command_stdout: str = '',
) -> tuple[dict[str, Any], list[str]]:
    if not checks:
        return {}, []

    summary = _collect_stream_tool_call_state(observations)
    failures: list[str] = []
    observed_event_types = set(summary.get('event_types') or [])

    for event_type in _as_string_list(checks.get('required_event_types')):
        if event_type not in observed_event_types:
            failures.append(
                f'{family} missing Responses stream event type `{event_type}`'
            )

    required_any_groups = checks.get('required_any_event_type_groups') or []
    if isinstance(required_any_groups, list):
        for group in required_any_groups:
            group_values = _as_string_list(group)
            if not group_values:
                continue
            if not any(event_type in observed_event_types for event_type in group_values):
                failures.append(
                    f'{family} missing any Responses stream event type from {group_values!r}'
                )

    tool_state = [
        item for item in summary.get('tool_state') or [] if isinstance(item, dict)
    ]
    command_tool_state = _collect_command_tool_call_state(command_stdout)
    summary['command_tool_state'] = command_tool_state
    summary['command_tool_names'] = [
        item.get('name')
        for item in command_tool_state
        if isinstance(item.get('name'), str) and item.get('name')
    ]
    for expected in checks.get('expected_tools') or []:
        if not isinstance(expected, dict):
            continue
        try:
            minimum_count = max(1, int(expected.get('minimum_count') or 1))
        except (TypeError, ValueError):
            minimum_count = 1
        matches = [
            item
            for item in tool_state
            if _stream_tool_state_matches_expected(item, expected)
        ]
        compacted_matches = [
            item
            for item in tool_state
            if item.get('arguments_compacted') is True
            and _stream_tool_state_matches_expected(
                item,
                expected,
                check_arguments=False,
            )
        ]
        command_matches = [
            item
            for item in command_tool_state
            if _stream_tool_state_matches_expected(item, expected)
        ]
        matched_count = len(matches) + min(
            len(compacted_matches),
            len(command_matches),
        )
        if matched_count < minimum_count:
            failures.append(
                f'{family} missing Responses stream tool state for {expected!r}; expected >= {minimum_count}, got {matched_count}'
            )

    return summary, failures



def _validate_runtime_postcondition(*, family: str, litellm_base_url: str, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    health_url = str(checks.get('healthcheck_url') or f"{litellm_base_url.rstrip('/')}/health/liveliness")
    container_name = checks.get('docker_container_name') or 'litellm-dev'

    summary: dict[str, Any] = {
        'healthcheck_url': health_url,
        'docker_container_name': container_name,
        'healthcheck_ok': False,
        'docker_status': None,
    }
    failures: list[str] = []

    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            body = response.read().decode('utf-8', errors='replace')
        summary['healthcheck_status'] = getattr(response, 'status', 200)
        summary['healthcheck_body'] = body
        summary['healthcheck_ok'] = 200 <= int(summary['healthcheck_status']) < 300
    except Exception as exc:
        summary['healthcheck_error'] = str(exc)
        failures.append(f'{family} runtime healthcheck failed: {exc}')

    if container_name:
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name=^{container_name}$', '--format', '{{.Status}}'],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
                timeout=DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS,
            )
            docker_status = result.stdout.strip() if result.returncode == 0 else ''
        except subprocess.TimeoutExpired:
            docker_status = ''
        summary['docker_status'] = docker_status
        if not docker_status:
            failures.append(f'{family} runtime container `{container_name}` not found')
        elif docker_status.lower().startswith('exited') or docker_status.lower().startswith('dead'):
            failures.append(f'{family} runtime container `{container_name}` is down: {docker_status}')

    return summary, failures


def _read_runtime_logs_since(
    *,
    started: Any,
    until: Any | None = None,
    checks: dict[str, Any],
    runtime_postconditions: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    container_name = (
        checks.get('docker_container_name')
        or runtime_postconditions.get('docker_container_name')
        or 'litellm-dev'
    )
    tail_lines = int(checks.get('tail_lines') or 400)
    summary: dict[str, Any] = {
        'docker_container_name': container_name,
        'tail_lines': tail_lines,
        'docker_logs_exit_code': None,
        'docker_logs_since': None,
        'docker_logs_until': None,
        'log_structural': {'line_count': 0, 'char_count': 0, 'sha256': ''},
    }
    if not container_name:
        return summary, ''

    since_value = started.isoformat() if hasattr(started, 'isoformat') else str(started)
    until_value = until.isoformat() if hasattr(until, 'isoformat') else (
        str(until) if until is not None else None
    )
    command = ['docker', 'logs', '--since', since_value]
    if until_value:
        command.extend(['--until', until_value])
    command.extend(['--tail', str(tail_lines), container_name])
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS,
        )
        log_text = '\n'.join(
            value for value in (result.stdout, result.stderr) if isinstance(value, str) and value
        )
        summary['docker_logs_exit_code'] = result.returncode
    except subprocess.TimeoutExpired:
        log_text = ''
        summary['docker_logs_exit_code'] = 124
        summary['docker_logs_error'] = (
            f'docker logs timed out after {DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS}s'
        )
    summary['docker_logs_since'] = since_value
    summary['docker_logs_until'] = until_value
    summary['log_structural'] = {
        'line_count': log_text.count('\n') + (1 if log_text and not log_text.endswith('\n') else 0) if log_text else 0,
        'char_count': len(log_text),
        'sha256': hashlib.sha256(log_text.encode()).hexdigest()[:16] if log_text else '',
    }
    return summary, log_text


def _runtime_log_match_contexts(
    *,
    log_text: str,
    substrings: list[str],
    context_chars: int = 4000,
) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for substring in substrings:
        match_index = log_text.find(substring)
        if match_index < 0:
            continue
        start_index = max(0, match_index - context_chars)
        end_index = min(
            len(log_text),
            match_index + len(substring) + context_chars,
        )
        contexts[substring] = log_text[start_index:end_index]
    return contexts



def _digest_log_context(context: str) -> dict[str, Any]:
    """Return structural/digest-only evidence for a runtime log context.

    Never persists raw runtime text.  The sha256 digest plus bounded
    structural counts prove that a specific context was captured for the
    matched substring without leaking prompt/tool/log text into artifacts.
    """
    return {
        'sha256': hashlib.sha256(context.encode('utf-8', errors='replace')).hexdigest(),
        'char_count': len(context),
        'line_count': context.count('\n') + (1 if context else 0),
    }


def _command_model_name(config: dict[str, Any]) -> str | None:
    command = config.get('command')
    if not isinstance(command, list):
        return None
    for index, value in enumerate(command):
        if value == '--model' and index + 1 < len(command):
            model = command[index + 1]
            return model if isinstance(model, str) and model else None
    return None


def _runtime_log_attribution_substrings(
    *,
    family: str,
    config: dict[str, Any],
    session_id: str | None,
) -> list[str]:
    values: set[str] = {family}
    if session_id:
        values.add(session_id)

    command_model = _command_model_name(config)
    if command_model:
        values.add(command_model)

    session_history_validation = config.get('session_history_validation')
    if isinstance(session_history_validation, dict):
        expected_model = session_history_validation.get('expected_model')
        if isinstance(expected_model, str) and expected_model:
            values.add(expected_model)

    for key in ('allowed_generation_routes', 'required_trace_tags'):
        configured_values = config.get(key)
        if isinstance(configured_values, list):
            values.update(
                value for value in configured_values if isinstance(value, str) and value
            )

    return sorted(values)


def _local_runtime_log_match_window(
    context: str,
    substring: str,
    *,
    radius: int = UNRELATED_RUNTIME_LOG_LOCAL_CONTEXT_CHARS,
) -> str:
    """Return a tight window around the first occurrence of substring in context."""
    if not context or not substring:
        return context
    match_index = context.find(substring)
    if match_index < 0:
        return context
    start_index = max(0, match_index - radius)
    end_index = min(len(context), match_index + len(substring) + radius)
    return context[start_index:end_index]


def _foreign_models_in_runtime_log_context(
    context: str,
    attribution_substrings: list[str],
) -> list[str]:
    foreign: list[str] = []
    for model in RUNTIME_LOG_MODEL_FIELD_RE.findall(context):
        if not model:
            continue
        if any(model in value for value in attribution_substrings):
            continue
        foreign.append(model)
    return foreign


def _is_unrelated_runtime_log_match(
    *,
    substring: str,
    context: str,
    attribution_substrings: list[str],
) -> bool:
    """Ignore a forbidden match only with positive foreign-traffic evidence.

    Absence of attribution alone is never enough. An unrelated provider
    signature must appear near the match. Upstream 429/5xx signatures
    additionally require a foreign model field near the match so interleaved
    concurrent logs cannot silently mask a genuine regression.
    """
    if substring not in ATTRIBUTION_SCOPED_RUNTIME_LOG_SUBSTRINGS:
        return False
    if not context or not attribution_substrings:
        return False
    if any(value in context for value in attribution_substrings):
        return False
    local_context = _local_runtime_log_match_window(context, substring)
    if not any(
        signature in local_context for signature in UNRELATED_RUNTIME_LOG_ERROR_SIGNATURES
    ):
        return False
    has_auto_agent = any(
        marker in context for marker in UNRELATED_AUTO_AGENT_RUNTIME_LOG_CONTEXT_MARKERS
    )
    has_passthrough = any(
        marker in context for marker in UNRELATED_PASSTHROUGH_RUNTIME_LOG_CONTEXT_MARKERS
    )
    foreign_models_local = _foreign_models_in_runtime_log_context(
        local_context,
        attribution_substrings,
    )
    if substring in DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS:
        return bool(foreign_models_local) and (has_passthrough or has_auto_agent)
    if has_auto_agent:
        return True
    if not has_passthrough:
        return False
    return bool(foreign_models_local)


def _validate_runtime_logs(
    *,
    family: str,
    started: Any,
    checks: dict[str, Any],
    runtime_postconditions: dict[str, Any],
    attribution_substrings: list[str] | None = None,
    require_evidence: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    container_name = (
        checks.get('docker_container_name')
        or runtime_postconditions.get('docker_container_name')
        or 'litellm-dev'
    )
    tail_lines = int(checks.get('tail_lines') or 400)
    default_upstream_error_substrings = list(
        DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS
    )
    if bool(checks.get('disable_default_429_traceback_check')):
        default_upstream_error_substrings = [
            substring
            for substring in default_upstream_error_substrings
            if 'Exception occured - 429:' not in substring
        ]
    forbidden_substrings = [
        *DEFAULT_RUNTIME_LOG_FORBIDDEN_SUBSTRINGS,
        *default_upstream_error_substrings,
        *list(checks.get('forbidden_substrings') or []),
    ]
    if bool(checks.get('disable_default_error_signature_check')):
        configured_substrings = list(checks.get('forbidden_substrings') or [])
        forbidden_substrings = [
            *configured_substrings,
            *default_upstream_error_substrings,
        ]
    forbidden_substrings = sorted(set(forbidden_substrings))

    summary: dict[str, Any] = {
        'docker_container_name': container_name,
        'tail_lines': tail_lines,
        'forbidden_substrings': forbidden_substrings,
        'matched_forbidden_substrings': [],
        'matched_forbidden_contexts': {},
        'ignored_unattributed_forbidden_substrings': [],
        'ignored_unattributed_forbidden_contexts': {},
        'attribution_substrings': attribution_substrings or [],
    }
    failures: list[str] = []
    warnings: list[str] = []

    if not container_name or not forbidden_substrings:
        return summary, failures, warnings

    log_summary, log_text = _read_runtime_logs_since(
        started=started,
        until=RA._utcnow(),
        checks={'docker_container_name': container_name, 'tail_lines': tail_lines},
        runtime_postconditions=runtime_postconditions,
    )
    summary['docker_logs_exit_code'] = log_summary.get('docker_logs_exit_code')
    summary['docker_logs_since'] = log_summary.get('docker_logs_since')
    summary['docker_logs_until'] = log_summary.get('docker_logs_until')
    summary['log_structural'] = log_summary.get(
        'log_structural', {'line_count': 0, 'char_count': 0, 'sha256': ''}
    )
    summary['log_evidence_read'] = bool(log_text)
    # Private key for in-process soft-fail matching; redacted by _write_artifact.
    summary['_log_text'] = log_text

    if summary['docker_logs_exit_code'] != 0:
        if require_evidence:
            failures.append(
                f'{family} runtime log evidence mandatory but docker logs unreadable for `{container_name}` (exit {summary["docker_logs_exit_code"]})'
            )
        else:
            warnings.append(
                f'{family} runtime log check could not read docker logs for `{container_name}` (exit {summary["docker_logs_exit_code"]})'
            )
        return summary, failures, warnings

    matched = [
        substring for substring in forbidden_substrings if substring and substring in log_text
    ]
    match_contexts = _runtime_log_match_contexts(
        log_text=log_text,
        substrings=matched,
    )
    failing_matches: list[str] = []
    ignored_matches: list[str] = []
    ignored_contexts: dict[str, str] = {}
    for substring in matched:
        context = match_contexts.get(substring, '')
        if _is_unrelated_runtime_log_match(
            substring=substring,
            context=context,
            attribution_substrings=attribution_substrings or [],
        ):
            ignored_matches.append(substring)
            ignored_contexts[substring] = context
            warnings.append(
                f'{family} ignored unattributed runtime log match `{substring}` from unrelated concurrent container traffic'
            )
            continue
        failing_matches.append(substring)
        failures.append(
            f'{family} runtime logs contained forbidden substring `{substring}`'
        )

    summary['matched_forbidden_substrings'] = failing_matches
    summary['matched_forbidden_contexts'] = {
        substring: _digest_log_context(match_contexts[substring])
        for substring in failing_matches
        if substring in match_contexts
    }
    summary['ignored_unattributed_forbidden_substrings'] = ignored_matches
    summary['ignored_unattributed_forbidden_contexts'] = {
        substring: _digest_log_context(context)
        for substring, context in ignored_contexts.items()
    }

    return summary, failures, warnings


def _validate_session_history(*, family: str, session_id: str | None, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:  # noqa: PLR0915
    if not session_id:
        return {'record': None}, [f'{family} missing command session_id for session_history validation']

    db_settings, db_failures = _validation_db_settings(
        family=family,
        checks=checks,
        validation_name='session_history',
    )
    if db_settings is None:
        return {'record': None}, db_failures

    expected_provider = checks.get('expected_provider')
    expected_model = checks.get('expected_model')
    expected_tenant_id = checks.get('expected_tenant_id')
    expected_rows = checks.get('expected_rows') or []
    expected_litellm_environment = checks.get('expected_litellm_environment')
    require_runtime_identity = checks.get('require_runtime_identity', True) is not False

    # Finding 1 (round 7): phase_start_time freshness enforcement.
    # When injected via checks, historical rows older than the phase start
    # are excluded even if session_id/provider/model match.
    phase_start_time = checks.get('phase_start_time')
    phase_start_clause = ""
    query_params: list[Any] = [session_id]
    if isinstance(phase_start_time, str) and phase_start_time.strip():
        phase_start_clause = " AND start_time >= %s"
        query_params.append(phase_start_time.strip())

    query = """
        select provider, model, session_id, tenant_id, repository,
               input_tokens, output_tokens, total_tokens,
               cache_read_input_tokens, cache_creation_input_tokens,
               provider_cache_attempted, provider_cache_status,
               provider_cache_miss, provider_cache_miss_reason,
               provider_cache_miss_token_count, provider_cache_miss_cost_usd,
               reasoning_tokens_reported, reasoning_tokens_estimated,
               reasoning_tokens_source, tool_call_count, tool_names,
               file_read_count, file_modified_count,
               changed_pre_commit_config, changed_env_file,
               changed_pyproject_toml, changed_gitignore,
               git_commit_count, git_push_count,
               response_cost_usd,
               litellm_environment, litellm_version, litellm_fork_version,
               litellm_wheel_versions, client_name, client_version, client_user_agent,
               input_system_tokens_estimated,
               input_tool_advertisement_tokens_estimated,
               input_conversation_tokens_estimated,
               input_other_tokens_estimated,
               input_breakdown_residual_tokens,
               system_behavior_tokens_estimated,
               system_safety_tokens_estimated,
               system_instructional_tokens_estimated,
               system_unclassified_tokens_estimated,
               metadata, start_time, end_time
        from public.session_history
        where session_id = %s{phase_start_clause}
        order by start_time desc
    """.format(phase_start_clause=phase_start_clause)
    conn = _validation_db_connection(db_settings)
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    while True:
        with conn.cursor() as cur:
            cur.execute(query, tuple(query_params))
            records = cur.fetchall()

        if records:
            if not expected_rows:
                break
            _, expected_row_failures = _match_session_history_expected_rows(
                family=family,
                records=records,
                expected_rows=expected_rows,
            )
            if not expected_row_failures:
                break

        if time.monotonic() >= poll_deadline:
            break
        time.sleep(poll_interval_seconds)

    if not records:
        return {'record': None, 'records': []}, [f'{family} missing session_history row for session_id `{session_id}`']

    failures: list[str] = []

    for row in records:
        row_provider = row.get('provider')
        if not isinstance(row_provider, str) or not row_provider.strip():
            failures.append(
                f'{family} session_history row model={row.get("model")!r} has null/empty `provider`'
            )
        if row_provider in {'anthropic', 'openai', 'openrouter'}:
            cache_status = row.get('provider_cache_status')
            if not isinstance(cache_status, str) or not cache_status.strip():
                failures.append(
                    f'{family} session_history row provider={row_provider!r} model={row.get("model")!r} has null/empty `provider_cache_status`'
                )
            if row.get('provider_cache_miss'):
                miss_reason = row.get('provider_cache_miss_reason')
                if not isinstance(miss_reason, str) or not miss_reason.strip():
                    failures.append(
                        f'{family} session_history row provider={row_provider!r} model={row.get("model")!r} has `provider_cache_miss=true` with null/empty `provider_cache_miss_reason`'
                    )

        source = row.get('reasoning_tokens_source')
        if not isinstance(source, str) or not source.strip():
            failures.append(
                f'{family} session_history row provider={row.get("provider")!r} model={row.get("model")!r} has null/empty `reasoning_tokens_source`'
            )
            continue
        if source == 'provider_reported':
            reported = row.get('reasoning_tokens_reported')
            if not isinstance(reported, (int, float)) or reported <= 0:
                failures.append(
                    f'{family} session_history row provider={row.get("provider")!r} model={row.get("model")!r} has `reasoning_tokens_source=provider_reported` with non-positive `reasoning_tokens_reported`={reported!r}'
                )

        if expected_litellm_environment is not None and row.get('litellm_environment') != expected_litellm_environment:
            failures.append(
                f'{family} session_history row provider={row.get("provider")!r} model={row.get("model")!r} has `litellm_environment`={row.get("litellm_environment")!r}; expected {expected_litellm_environment!r}'
            )

        if require_runtime_identity:
            for key in (
                'litellm_environment',
                'litellm_version',
                'litellm_fork_version',
                'client_name',
                'client_version',
            ):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    failures.append(
                        f'{family} session_history row provider={row.get("provider")!r} model={row.get("model")!r} has null/empty `{key}`'
                    )
            wheel_versions = row.get('litellm_wheel_versions')
            if not isinstance(wheel_versions, dict) or not wheel_versions:
                failures.append(
                    f'{family} session_history row provider={row.get("provider")!r} model={row.get("model")!r} has null/empty `litellm_wheel_versions`'
                )

    def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.isoformat() if hasattr(value, 'isoformat') else value)
            for key, value in row.items()
        }

    if expected_rows:
        matched_records, expected_row_failures = _match_session_history_expected_rows(
            family=family,
            records=records,
            expected_rows=expected_rows,
        )
        failures.extend(expected_row_failures)

        return {
            'record': matched_records[0] if matched_records else None,
            'records': matched_records,
            'all_records': [_normalize_record(row) for row in records],
        }, failures

    filtered_records = [
        row for row in records
        if (expected_provider is None or row.get('provider') == expected_provider)
        and (expected_model is None or row.get('model') == expected_model)
    ]
    record = filtered_records[0] if filtered_records else None

    if record is None:
        return {'record': None, 'records': [_normalize_record(row) for row in records]}, [f'{family} missing session_history row for session_id `{session_id}`']

    normalized_record = _normalize_record(record)

    if expected_provider is not None and record.get('provider') != expected_provider:
        failures.append(f'{family} session_history provider mismatch: expected `{expected_provider}`, got `{record.get("provider")}`')

    if expected_model is not None and record.get('model') != expected_model:
        failures.append(f'{family} session_history model mismatch: expected `{expected_model}`, got `{record.get("model")}`')

    if expected_tenant_id is not None and record.get('tenant_id') != expected_tenant_id:
        failures.append(
            f'{family} session_history tenant_id mismatch: expected `{expected_tenant_id}`, got `{record.get("tenant_id")}`'
        )

    expected_client_name = checks.get('expected_client_name')
    if expected_client_name is not None and record.get('client_name') != expected_client_name:
        failures.append(
            f'{family} session_history client_name mismatch: expected `{expected_client_name}`, got `{record.get("client_name")}`'
        )

    expected_client_version = checks.get('expected_client_version')
    if expected_client_version is not None and record.get('client_version') != expected_client_version:
        failures.append(
            f'{family} session_history client_version mismatch: expected `{expected_client_version}`, got `{record.get("client_version")}`'
        )

    client_user_agent_contains = checks.get('client_user_agent_contains')
    if client_user_agent_contains is not None:
        actual_user_agent = record.get('client_user_agent')
        if not isinstance(actual_user_agent, str) or str(client_user_agent_contains) not in actual_user_agent:
            failures.append(
                f'{family} session_history client_user_agent missing substring `{client_user_agent_contains}`: got `{actual_user_agent}`'
            )

    metadata = record.get('metadata')
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        failures.append(
            f'{family} session_history metadata is not an object: got `{type(metadata).__name__}`'
        )
        metadata = {}

    for key, expected in (checks.get('metadata_required_equals') or {}).items():
        actual = metadata.get(key)
        if actual != expected:
            failures.append(
                f'{family} session_history metadata `{key}` mismatch: expected `{expected}`, got `{actual}`'
            )

    for key in checks.get('metadata_required_truthy') or []:
        if not metadata.get(key):
            failures.append(f'{family} session_history metadata `{key}` is not truthy')

    for key, expected_substring in (checks.get('metadata_required_contains') or {}).items():
        actual = metadata.get(key)
        if not isinstance(actual, str) or str(expected_substring) not in actual:
            failures.append(
                f'{family} session_history metadata `{key}` missing substring `{expected_substring}`: got `{actual}`'
            )

    for key, expected in (checks.get('required_equals') or {}).items():
        actual = record.get(key)
        if actual != expected:
            failures.append(
                f'{family} session_history `{key}` mismatch: expected `{expected}`, got `{actual}`'
            )

    for key, allowed_values in (checks.get('required_one_of') or {}).items():
        actual = record.get(key)
        allowed_list = allowed_values if isinstance(allowed_values, list) else []
        if not any(actual == allowed for allowed in allowed_list):
            failures.append(
                f'{family} session_history `{key}` expected one of {allowed_values!r}, got `{actual}`'
            )

    for key in checks.get('required_truthy') or []:
        if not record.get(key):
            failures.append(f'{family} session_history `{key}` is not truthy')

    for key, expected_substring in (checks.get('required_contains') or {}).items():
        actual = record.get(key)
        if not isinstance(actual, str) or str(expected_substring) not in actual:
            failures.append(
                f'{family} session_history `{key}` missing substring `{expected_substring}`: got `{actual}`'
            )

    for key, minimum in (checks.get('minimums') or {}).items():
        actual = record.get(key)
        if not isinstance(actual, (int, float)) or actual < minimum:
            failures.append(f'{family} session_history `{key}` below minimum: expected >= {minimum!r}, got {actual!r}')

    minimum_record_count = checks.get('minimum_record_count')
    if minimum_record_count is not None:
        try:
            minimum_record_count_int = int(minimum_record_count)
        except (TypeError, ValueError):
            minimum_record_count_int = 0
        if minimum_record_count_int > 0 and len(filtered_records) < minimum_record_count_int:
            failures.append(
                f'{family} session_history rows below minimum: expected >= {minimum_record_count_int}, got {len(filtered_records)}'
            )

    return {'record': normalized_record, 'records': [_normalize_record(row) for row in records]}, failures


def _validation_db_settings(
    *,
    family: str,
    checks: dict[str, Any],
    validation_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    db_host = str(checks.get('db_host') or '127.0.0.1')
    db_port = int(checks.get('db_port') or 5434)
    db_name = str(checks.get('db_name') or 'aawm_tristore')
    db_user = str(checks.get('db_user') or 'aawm')
    db_password = None
    if isinstance(checks.get('db_password_container_env'), str):
        container_name = str(checks.get('db_password_container') or '').strip()
        env_name = str(checks['db_password_container_env']).strip()
        if not container_name or not env_name:
            return None, [
                f'{family} missing target-owned DB credential configuration for {validation_name} validation'
            ]
        db_password = _resolve_container_env_value(container_name, env_name)
        if db_password is None:
            return None, [
                f'{family} could not retrieve target-owned DB credential for {validation_name} validation'
            ]
    elif isinstance(checks.get('db_password_env'), str):
        db_password = os.environ.get(str(checks['db_password_env']))
    if db_password is None and isinstance(checks.get('db_password'), str):
        db_password = str(checks['db_password'])
    if db_password is None:
        return None, [f'{family} missing DB password for {validation_name} validation']
    return {
        'host': db_host,
        'port': db_port,
        'dbname': db_name,
        'user': db_user,
        'password': db_password,
    }, []


def _validation_db_connection(settings: dict[str, Any]) -> Any:
    key = (
        str(settings['host']),
        int(settings['port']),
        str(settings['dbname']),
        str(settings['user']),
        str(settings['password']),
    )
    conn = _VALIDATION_DB_CONNECTIONS.get(key)
    if conn is not None and not bool(getattr(conn, 'closed', False)):
        return conn

    if conn is not None:
        _VALIDATION_DB_CONNECTIONS.pop(key, None)

    conn = psycopg.connect(
        host=key[0],
        port=key[1],
        dbname=key[2],
        user=key[3],
        password=key[4],
        connect_timeout=10,
        autocommit=True,
        row_factory=psycopg.rows.dict_row,
    )
    _VALIDATION_DB_CONNECTIONS[key] = conn
    return conn


def _session_history_record_matches_expected(
    row: dict[str, Any],
    expected_row: dict[str, Any],
) -> bool:
    row_provider = expected_row.get('provider')
    row_model = expected_row.get('model')
    if row_provider is not None and row.get('provider') != row_provider:
        return False
    if row_model is not None and row.get('model') != row_model:
        return False
    for key, expected in (expected_row.get('required_equals') or {}).items():
        if row.get(key) != expected:
            return False
    for key, allowed_values in (expected_row.get('required_one_of') or {}).items():
        if row.get(key) not in set(allowed_values or []):
            return False
    # Correlated candidate triples: the row's provider+model+route_family must
    # match one of the allowed triples (same-row correlation, not independent
    # allowlists).  route_family is read from metadata selected_route_family /
    # passthrough_route_family.
    correlated_triples = expected_row.get('correlated_candidate_triples')
    if isinstance(correlated_triples, list) and correlated_triples:
        row_metadata = row.get('metadata')
        if not isinstance(row_metadata, dict):
            row_metadata = {}
        row_route_family = None
        for rf_key in (
            'codex_auto_agent_selected_route_family',
            'anthropic_auto_agent_selected_route_family',
            'passthrough_route_family',
        ):
            rf_val = row_metadata.get(rf_key)
            if isinstance(rf_val, str) and rf_val.strip():
                row_route_family = rf_val.strip()
                break
        triple_matched = any(
            isinstance(triple, dict)
            and row.get('provider') == triple.get('provider')
            and row.get('model') == triple.get('model')
            and row_route_family == triple.get('route_family')
            for triple in correlated_triples
        )
        if not triple_matched:
            return False
    for key in expected_row.get('required_truthy') or []:
        if not row.get(key):
            return False
    for key, expected_substring in (
        expected_row.get('required_contains') or {}
    ).items():
        actual = row.get(key)
        if not isinstance(actual, str) or expected_substring not in actual:
            return False
    metadata = row.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}
    for key, expected in (expected_row.get('metadata_required_equals') or {}).items():
        if metadata.get(key) != expected:
            return False
    for key in expected_row.get('metadata_required_truthy') or []:
        if not metadata.get(key):
            return False
    for key, expected_substring in (
        expected_row.get('metadata_required_contains') or {}
    ).items():
        actual = metadata.get(key)
        if not isinstance(actual, str) or expected_substring not in actual:
            return False
    for key, minimum in (expected_row.get('minimums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual < minimum:
            return False
    return True


def _session_history_candidate_summary(
    row: dict[str, Any],
    expected_row: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'provider': row.get('provider'),
        'model': row.get('model'),
        'tenant_id': row.get('tenant_id'),
        'input_tokens': row.get('input_tokens'),
        'input_system_tokens_estimated': row.get('input_system_tokens_estimated'),
        'input_tool_advertisement_tokens_estimated': row.get('input_tool_advertisement_tokens_estimated'),
        'input_conversation_tokens_estimated': row.get('input_conversation_tokens_estimated'),
        'input_other_tokens_estimated': row.get('input_other_tokens_estimated'),
        'input_breakdown_residual_tokens': row.get('input_breakdown_residual_tokens'),
        'output_tokens': row.get('output_tokens'),
        'response_cost_usd': row.get('response_cost_usd'),
    }
    metadata = row.get('metadata')
    if isinstance(metadata, dict) and metadata.get('tenant_id') is not None:
        summary['metadata.tenant_id'] = metadata.get('tenant_id')
    if isinstance(metadata, dict):
        for key in (
            'prompt_overhead_breakdown_source',
            'prompt_overhead_counted_shape',
            'prompt_overhead_classifier_version',
        ):
            if metadata.get(key) is not None:
                summary[f'metadata.{key}'] = metadata.get(key)

    mismatches: dict[str, Any] = {}
    for key, expected in (expected_row.get('required_equals') or {}).items():
        actual = row.get(key)
        if actual != expected:
            mismatches[key] = {'expected': expected, 'actual': actual}
    if isinstance(metadata, dict):
        for key, expected in (
            expected_row.get('metadata_required_equals') or {}
        ).items():
            actual = metadata.get(key)
            if actual != expected:
                mismatches[f'metadata.{key}'] = {
                    'expected': expected,
                    'actual': actual,
                }
        for key in expected_row.get('metadata_required_truthy') or []:
            actual = metadata.get(key)
            if not actual:
                mismatches[f'metadata.{key}'] = {
                    'expected': 'truthy',
                    'actual': actual,
                }
    else:
        for key in expected_row.get('metadata_required_equals') or {}:
            mismatches[f'metadata.{key}'] = {
                'expected': expected_row['metadata_required_equals'][key],
                'actual': None,
            }
        for key in expected_row.get('metadata_required_truthy') or []:
            mismatches[f'metadata.{key}'] = {
                'expected': 'truthy',
                'actual': None,
            }
    for key, minimum in (expected_row.get('minimums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual < minimum:
            mismatches[key] = {'minimum': minimum, 'actual': actual}
    if mismatches:
        summary['mismatches'] = mismatches
    return summary


def _match_session_history_expected_rows(
    *,
    family: str,
    records: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.isoformat() if hasattr(value, 'isoformat') else value)
            for key, value in row.items()
        }

    failures: list[str] = []
    matched_records: list[dict[str, Any]] = []
    used_record_indexes: set[int] = set()
    for expected_row in expected_rows:
        row_provider = expected_row.get('provider')
        row_model = expected_row.get('model')
        try:
            minimum_count = max(1, int(expected_row.get('minimum_count') or 1))
        except (TypeError, ValueError):
            minimum_count = 1
        matches: list[tuple[int, dict[str, Any]]] = [
            (index, row)
            for index, row in enumerate(records)
            if index not in used_record_indexes
            and _session_history_record_matches_expected(row, expected_row)
        ]
        if len(matches) < minimum_count:
            candidate_rows = [
                row
                for row in records
                if (row_provider is None or row.get('provider') == row_provider)
                and (row_model is None or row.get('model') == row_model)
            ]
            candidate_summary = [
                _session_history_candidate_summary(row, expected_row)
                for row in candidate_rows[:5]
            ]
            detail = ''
            if candidate_summary:
                detail = (
                    '; candidate rows: '
                    + json.dumps(candidate_summary, sort_keys=True, default=str)
                )
            failures.append(
                f'{family} missing session_history rows for provider={row_provider!r} model={row_model!r}; expected >= {minimum_count}, got {len(matches)}{detail}'
            )
            continue
        selected_matches = matches[:minimum_count]
        used_record_indexes.update(index for index, _ in selected_matches)
        matched_records.extend(_normalize_record(row) for _, row in selected_matches)

    return matched_records, failures


def _normalize_db_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.isoformat() if hasattr(value, 'isoformat') else value)
        for key, value in row.items()
    }


def _rate_limit_observation_record_matches_expected(
    row: dict[str, Any],
    expected_row: dict[str, Any],
) -> bool:
    for key in ('provider', 'model', 'source', 'quota_key', 'quota_type', 'client'):
        expected = expected_row.get(key)
        if expected is not None and row.get(key) != expected:
            return False
    for key, expected in (expected_row.get('required_equals') or {}).items():
        if row.get(key) != expected:
            return False
    for key, allowed_values in (expected_row.get('required_one_of') or {}).items():
        if row.get(key) not in set(allowed_values or []):
            return False
    # Correlated candidate triples: the row's provider+model+route_family must
    # match one of the allowed triples (same-row correlation, not independent
    # allowlists).  route_family is read from metadata selected_route_family /
    # passthrough_route_family.
    correlated_triples = expected_row.get('correlated_candidate_triples')
    if isinstance(correlated_triples, list) and correlated_triples:
        row_metadata = row.get('metadata')
        if not isinstance(row_metadata, dict):
            row_metadata = {}
        row_route_family = None
        for rf_key in (
            'codex_auto_agent_selected_route_family',
            'anthropic_auto_agent_selected_route_family',
            'passthrough_route_family',
        ):
            rf_val = row_metadata.get(rf_key)
            if isinstance(rf_val, str) and rf_val.strip():
                row_route_family = rf_val.strip()
                break
        triple_matched = any(
            isinstance(triple, dict)
            and row.get('provider') == triple.get('provider')
            and row.get('model') == triple.get('model')
            and row_route_family == triple.get('route_family')
            for triple in correlated_triples
        )
        if not triple_matched:
            return False
    for key in expected_row.get('required_truthy') or []:
        if not row.get(key):
            return False
    for key, minimum in (expected_row.get('minimums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual < minimum:
            return False
    for key, maximum in (expected_row.get('maximums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual > maximum:
            return False
    for key in expected_row.get('required_future_timestamps') or []:
        actual = row.get(key)
        if not hasattr(actual, 'timestamp') or actual <= RA._utcnow():
            return False
    for key in expected_row.get('required_timestamp_after_observed') or []:
        actual = row.get(key)
        observed_at = row.get('observed_at')
        if (
            not hasattr(actual, 'timestamp')
            or not hasattr(observed_at, 'timestamp')
            or actual <= observed_at
        ):
            return False
    return True


def _rate_limit_observation_candidate_summary(
    row: dict[str, Any],
    expected_row: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'observed_at': row.get('observed_at'),
        'provider': row.get('provider'),
        'model': row.get('model'),
        'quota_key': row.get('quota_key'),
        'quota_type': row.get('quota_type'),
        'remaining_pct': row.get('remaining_pct'),
        'quota_limit': row.get('quota_limit'),
        'quota_used': row.get('quota_used'),
        'quota_remaining': row.get('quota_remaining'),
        'expected_reset_at': row.get('expected_reset_at'),
        'billing_period_start_at': row.get('billing_period_start_at'),
        'billing_period_end_at': row.get('billing_period_end_at'),
        'source': row.get('source'),
        'session_id': row.get('session_id'),
    }
    mismatches: dict[str, Any] = {}
    for key in ('provider', 'model', 'source', 'quota_key', 'quota_type', 'client'):
        expected = expected_row.get(key)
        actual = row.get(key)
        if expected is not None and actual != expected:
            mismatches[key] = {'expected': expected, 'actual': actual}
    for key, expected in (expected_row.get('required_equals') or {}).items():
        actual = row.get(key)
        if actual != expected:
            mismatches[key] = {'expected': expected, 'actual': actual}
    for key, allowed_values in (expected_row.get('required_one_of') or {}).items():
        actual = row.get(key)
        if actual not in set(allowed_values or []):
            mismatches[key] = {'expected_one_of': allowed_values, 'actual': actual}
    for key in expected_row.get('required_truthy') or []:
        actual = row.get(key)
        if not actual:
            mismatches[key] = {'expected': 'truthy', 'actual': actual}
    for key, minimum in (expected_row.get('minimums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual < minimum:
            mismatches[key] = {'minimum': minimum, 'actual': actual}
    for key, maximum in (expected_row.get('maximums') or {}).items():
        actual = row.get(key)
        if not isinstance(actual, (int, float)) or actual > maximum:
            mismatches[key] = {'maximum': maximum, 'actual': actual}
    for key in expected_row.get('required_future_timestamps') or []:
        actual = row.get(key)
        if not hasattr(actual, 'timestamp') or actual <= RA._utcnow():
            mismatches[key] = {'expected': 'future timestamp', 'actual': actual}
    for key in expected_row.get('required_timestamp_after_observed') or []:
        actual = row.get(key)
        observed_at = row.get('observed_at')
        if (
            not hasattr(actual, 'timestamp')
            or not hasattr(observed_at, 'timestamp')
            or actual <= observed_at
        ):
            mismatches[key] = {
                'expected': 'timestamp after observed_at',
                'actual': actual,
                'observed_at': observed_at,
            }
    if mismatches:
        summary['mismatches'] = mismatches
    return summary


def _match_rate_limit_observation_expected_rows(
    *,
    family: str,
    records: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    matched_records: list[dict[str, Any]] = []
    used_record_indexes: set[int] = set()
    for expected_row in expected_rows:
        row_provider = expected_row.get('provider')
        row_quota_key = expected_row.get('quota_key')
        row_source = expected_row.get('source')
        try:
            minimum_count = max(1, int(expected_row.get('minimum_count') or 1))
        except (TypeError, ValueError):
            minimum_count = 1
        matches: list[tuple[int, dict[str, Any]]] = [
            (index, row)
            for index, row in enumerate(records)
            if index not in used_record_indexes
            and _rate_limit_observation_record_matches_expected(row, expected_row)
        ]
        if len(matches) < minimum_count:
            candidate_rows = [
                row
                for row in records
                if (row_provider is None or row.get('provider') == row_provider)
                and (row_quota_key is None or row.get('quota_key') == row_quota_key)
                and (row_source is None or row.get('source') == row_source)
            ]
            candidate_summary = [
                _rate_limit_observation_candidate_summary(row, expected_row)
                for row in candidate_rows[:5]
            ]
            detail = ''
            if candidate_summary:
                detail = (
                    '; candidate rows: '
                    + json.dumps(candidate_summary, sort_keys=True, default=str)
                )
            failures.append(
                f'{family} missing rate_limit_observations rows for provider={row_provider!r} quota_key={row_quota_key!r} source={row_source!r}; expected >= {minimum_count}, got {len(matches)}{detail}'
            )
            continue
        selected_matches = matches[:minimum_count]
        used_record_indexes.update(index for index, _ in selected_matches)
        matched_records.extend(_normalize_db_record(row) for _, row in selected_matches)

    return matched_records, failures


def _validate_rate_limit_observations(
    *,
    family: str,
    session_id: str | None,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not checks:
        return {'records': [], 'matched_records': []}, [], []

    db_settings, db_failures = _validation_db_settings(
        family=family,
        checks=checks,
        validation_name='rate_limit_observations',
    )
    if db_settings is None:
        return {'records': [], 'matched_records': []}, db_failures, []

    expected_rows = checks.get('expected_rows') or []
    if not isinstance(expected_rows, list) or not expected_rows:
        return {'records': [], 'matched_records': []}, [], []

    conn = _validation_db_connection(db_settings)
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    allow_latest_snapshot_fallback = bool(checks.get('allow_latest_snapshot_fallback'))
    max_snapshot_age_seconds = max(
        1.0,
        float(checks.get('latest_snapshot_max_age_seconds') or 21600),
    )
    latest_cutoff = RA._utcnow() - RA.dt.timedelta(seconds=max_snapshot_age_seconds)
    session_query = '''
        select observed_at, created_at, client, client_version, account_hash,
               provider, model, quota_key, quota_period, quota_type,
               expected_reset_at, remaining_pct, quota_limit, quota_used,
               quota_remaining, billing_period_start_at, billing_period_end_at,
               raw_provider_fields, evidence, source, session_id, trace_id,
               litellm_call_id
        from public.rate_limit_observations
        where session_id = %s
        order by observed_at desc, id desc
    '''
    latest_query = '''
        select observed_at, created_at, client, client_version, account_hash,
               provider, model, quota_key, quota_period, quota_type,
               expected_reset_at, remaining_pct, quota_limit, quota_used,
               quota_remaining, billing_period_start_at, billing_period_end_at,
               raw_provider_fields, evidence, source, session_id, trace_id,
               litellm_call_id
        from public.rate_limit_observations
        where observed_at >= %s
        order by observed_at desc, id desc
        limit 500
    '''

    session_records: list[dict[str, Any]] = []
    latest_records: list[dict[str, Any]] = []
    records_for_matching: list[dict[str, Any]] = []
    matched_records: list[dict[str, Any]] = []
    match_failures: list[str] = []
    match_source = 'session'
    poll_deadline = time.monotonic() + poll_timeout_seconds
    while True:
        session_records = []
        if session_id:
            with conn.cursor() as cur:
                cur.execute(session_query, (session_id,))
                session_records = cur.fetchall()

        records_for_matching = session_records
        matched_records, match_failures = _match_rate_limit_observation_expected_rows(
            family=family,
            records=records_for_matching,
            expected_rows=expected_rows,
        )
        match_source = 'session'

        if match_failures and allow_latest_snapshot_fallback:
            with conn.cursor() as cur:
                cur.execute(latest_query, (latest_cutoff,))
                latest_records = cur.fetchall()
            matched_records, match_failures = (
                _match_rate_limit_observation_expected_rows(
                    family=family,
                    records=latest_records,
                    expected_rows=expected_rows,
                )
            )
            records_for_matching = latest_records
            match_source = 'latest_snapshot'

        if not match_failures:
            break
        if time.monotonic() >= poll_deadline:
            break
        time.sleep(poll_interval_seconds)

    warnings: list[str] = []
    if not match_failures and match_source == 'latest_snapshot':
        warnings.append(
            f'{family} rate_limit_observations matched latest current snapshots instead of session rows; unchanged duplicate snapshots may have been suppressed'
        )

    return {
        'match_source': match_source,
        'records': [_normalize_db_record(row) for row in records_for_matching],
        'session_records': [_normalize_db_record(row) for row in session_records],
        'latest_snapshot_records': [_normalize_db_record(row) for row in latest_records],
        'matched_records': matched_records,
        'latest_snapshot_cutoff': latest_cutoff.isoformat(),
    }, match_failures, warnings


_EMAIL_LIKE_RE = re.compile(
    r'(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9._%+-])'
)


def _mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_mapping_keys(nested))
    return keys


def _provider_error_record_failures(
    *,
    family: str,
    record: dict[str, Any],
    checks: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for key in checks.get('required_truthy') or []:
        if not record.get(key):
            failures.append(
                f'{family} provider_error_observations `{key}` is not truthy'
            )

    retry_after = record.get('retry_after_seconds')
    if retry_after is not None:
        if isinstance(retry_after, bool) or not isinstance(retry_after, (int, float)):
            failures.append(
                f'{family} provider_error_observations retry_after_seconds is not numeric'
            )
        else:
            minimum = checks.get('retry_after_seconds_minimum')
            maximum = checks.get('retry_after_seconds_maximum')
            if minimum is not None and retry_after < float(minimum):
                failures.append(
                    f'{family} provider_error_observations retry_after_seconds below minimum'
                )
            if maximum is not None and retry_after > float(maximum):
                failures.append(
                    f'{family} provider_error_observations retry_after_seconds above maximum'
                )

    metadata = record.get('metadata')
    metadata_text = json.dumps(metadata or {}, sort_keys=True, default=str)
    maximum_metadata_chars = checks.get('maximum_metadata_chars')
    if (
        maximum_metadata_chars is not None
        and len(metadata_text) > int(maximum_metadata_chars)
    ):
        failures.append(
            f'{family} provider_error_observations metadata exceeds maximum length'
        )

    normalized_keys = {key.strip().lower() for key in _mapping_keys(metadata)}
    for forbidden_key in checks.get('forbidden_metadata_keys') or []:
        if str(forbidden_key).strip().lower() in normalized_keys:
            failures.append(
                f'{family} provider_error_observations metadata contained forbidden key'
            )
    metadata_text_lower = metadata_text.lower()
    for substring in checks.get('forbidden_substrings') or []:
        if str(substring).lower() in metadata_text_lower:
            failures.append(
                f'{family} provider_error_observations metadata contained forbidden text'
            )
    if checks.get('forbid_email_like') and _EMAIL_LIKE_RE.search(metadata_text):
        failures.append(
            f'{family} provider_error_observations metadata contained email-like text'
        )
    return failures


def _validate_provider_error_records(
    *,
    family: str,
    records: list[dict[str, Any]],
    checks: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    expected_rows = checks.get('expected_rows') or []
    matched_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for expected_row in expected_rows:
        matches = [
            record
            for record in records
            if all(
                record.get(key) == expected
                for key, expected in (expected_row.get('required_equals') or {}).items()
            )
        ]
        if not matches:
            failures.append(
                f'{family} missing correlated provider_error_observations row'
            )
            continue
        record = matches[0]
        failures.extend(
            _provider_error_record_failures(
                family=family,
                record=record,
                checks=expected_row,
            )
        )
        matched_records.append(_normalize_db_record(record))
    return matched_records, failures


def _validate_provider_error_observations(
    *,
    family: str,
    session_id: str | None,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not checks:
        return {'records': [], 'matched_records': []}, []
    if not session_id:
        return {'records': [], 'matched_records': []}, [
            f'{family} missing command session_id for provider_error_observations validation'
        ]

    db_settings, db_failures = _validation_db_settings(
        family=family,
        checks=checks,
        validation_name='provider_error_observations',
    )
    if db_settings is None:
        return {'records': [], 'matched_records': []}, db_failures

    query = '''
        select observed_at, created_at, environment, provider, model,
               model_group, route_family, status_code, error_type, error_code,
               error_class, retry_after_seconds, expected_reset_at, session_id,
               trace_id, litellm_call_id, metadata
        from public.provider_error_observations
        where session_id = %s
        order by observed_at desc, id desc
    '''
    conn = _validation_db_connection(db_settings)
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    records: list[dict[str, Any]] = []
    matched_records: list[dict[str, Any]] = []
    failures: list[str] = []
    while True:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            records = cur.fetchall()
        matched_records, failures = _validate_provider_error_records(
            family=family,
            records=records,
            checks=checks,
        )
        if not failures or time.monotonic() >= poll_deadline:
            break
        time.sleep(poll_interval_seconds)
    return {
        'records': [_normalize_db_record(record) for record in records],
        'matched_records': matched_records,
    }, failures


def _tool_activity_name_matches(actual: Any, expected: str | None) -> bool:
    """Compare tool_activity tool_name, canonicalizing namespaced Codex names.

    The DB may store fully-qualified names like
    ``functions.collaboration.spawn_agent`` while the harness config uses
    canonical short names like ``spawn_agent``.  Canonicalize the DB value
    before comparing so both forms match.
    """
    if expected is None:
        return True
    actual_str = str(actual or '')
    return _normalize_codex_tool_name(actual_str) == expected


def _validate_tool_activity(*, family: str, session_id: str | None, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:  # noqa: PLR0915
    if not session_id:
        return {'record': None, 'records': []}, [f'{family} missing command session_id for tool_activity validation']

    db_settings, db_failures = _validation_db_settings(
        family=family,
        checks=checks,
        validation_name='tool_activity',
    )
    if db_settings is None:
        return {'record': None, 'records': []}, db_failures

    # Finding 1 (round 7): phase_start_time freshness enforcement for
    # tool_activity.  Correlate to the current phase via created_at lower bound.
    phase_start_time = checks.get('phase_start_time')
    ta_phase_clause = ""
    ta_query_params: list[Any] = [session_id]
    if isinstance(phase_start_time, str) and phase_start_time.strip():
        ta_phase_clause = " AND created_at >= %s"
        ta_query_params.append(phase_start_time.strip())

    query = '''
        select litellm_call_id, tool_call_id, provider, model, tool_index,
               tool_name, tool_kind, command_text, arguments, metadata, created_at
        from public.session_history_tool_activity
        where session_id = %s{ta_phase_clause}
        order by created_at asc, tool_index asc
    '''.format(ta_phase_clause=ta_phase_clause)
    conn = _validation_db_connection(db_settings)
    expected_rows = checks.get('expected_rows') or []
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    while True:
        with conn.cursor() as cur:
            cur.execute(query, tuple(ta_query_params))
            records = cur.fetchall()

        missing_expected_rows = False
        for expected_row in expected_rows:
            row_provider = expected_row.get('provider')
            row_model = expected_row.get('model')
            row_tool_name = expected_row.get('tool_name')
            row_tool_kind = expected_row.get('tool_kind')
            matches = [
                row
                for row in records
                if (row_provider is None or row.get("provider") == row_provider)
                and (row_model is None or row.get("model") == row_model)
                and _tool_activity_name_matches(row.get("tool_name"), row_tool_name)
                and (row_tool_kind is None or row.get("tool_kind") == row_tool_kind)
            ]
            minimum_count = int(expected_row.get('minimum_count') or 1)
            if len(matches) < minimum_count:
                missing_expected_rows = True
                break

        if records and not missing_expected_rows:
            break
        if time.monotonic() >= poll_deadline:
            break
        time.sleep(poll_interval_seconds)

    def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.isoformat() if hasattr(value, 'isoformat') else value)
            for key, value in row.items()
        }

    failures: list[str] = []
    matched_records: list[dict[str, Any]] = []
    for expected_row in expected_rows:
        row_provider = expected_row.get('provider')
        row_model = expected_row.get('model')
        row_tool_name = expected_row.get('tool_name')
        row_tool_kind = expected_row.get('tool_kind')
        matches = [
            row
            for row in records
            if (row_provider is None or row.get('provider') == row_provider)
            and (row_model is None or row.get('model') == row_model)
            and _tool_activity_name_matches(row.get('tool_name'), row_tool_name)
            and (row_tool_kind is None or row.get('tool_kind') == row_tool_kind)
        ]
        minimum_count = int(expected_row.get('minimum_count') or 1)
        if len(matches) < minimum_count:
            failures.append(
                f'{family} missing tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} tool_kind={row_tool_kind!r}; expected >= {minimum_count}, got {len(matches)}'
            )
            continue
        maximum_count = expected_row.get('maximum_count')
        if maximum_count is not None:
            maximum_count_int = int(maximum_count)
            if len(matches) > maximum_count_int:
                failures.append(
                    f'{family} too many tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} tool_kind={row_tool_kind!r}; expected <= {maximum_count_int}, got {len(matches)}'
                )
        exact_command_texts = expected_row.get('exact_command_texts')
        if isinstance(exact_command_texts, list):
            expected_command_texts = sorted(
                value
                for value in exact_command_texts
                if isinstance(value, str)
            )
            actual_command_texts = sorted(
                str(row.get('command_text') or '') for row in matches
            )
            if actual_command_texts != expected_command_texts:
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} did not match exact commands; expected {expected_command_texts!r}, got {actual_command_texts!r}'
                )
        expected_tool_indexes = expected_row.get('expected_tool_indexes')
        if isinstance(expected_tool_indexes, list):
            actual_tool_indexes = sorted(row.get('tool_index') for row in matches)
            if actual_tool_indexes != sorted(expected_tool_indexes):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} had unexpected tool indexes; expected {sorted(expected_tool_indexes)!r}, got {actual_tool_indexes!r}'
                )
        if expected_row.get('require_single_litellm_call_id'):
            litellm_call_ids = {
                row.get('litellm_call_id')
                for row in matches
                if isinstance(row.get('litellm_call_id'), str)
                and row.get('litellm_call_id')
            }
            if len(litellm_call_ids) != 1 or any(
                not isinstance(row.get('litellm_call_id'), str)
                or not row.get('litellm_call_id')
                for row in matches
            ):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} did not share one nonempty litellm_call_id'
                )
        command_text_contains = expected_row.get('command_text_contains')
        if isinstance(command_text_contains, str) and command_text_contains:
            if not any(command_text_contains in str(row.get('command_text') or '') for row in matches):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} did not include command text containing {command_text_contains!r}'
                )
        required_argument_substrings = []
        configured_required_argument = expected_row.get(
            'arguments_required_substring'
        )
        if (
            isinstance(configured_required_argument, str)
            and configured_required_argument
        ):
            required_argument_substrings.append(configured_required_argument)
        configured_required_arguments = expected_row.get(
            'arguments_required_substrings'
        )
        if isinstance(configured_required_arguments, list):
            required_argument_substrings.extend(
                value
                for value in configured_required_arguments
                if isinstance(value, str) and value
            )
        for required_argument_substring in required_argument_substrings:
            if not any(
                required_argument_substring
                in json.dumps(row.get('arguments'), sort_keys=True)
                for row in matches
            ):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} did not include arguments containing {required_argument_substring!r}'
                )
        each_required_argument_substrings = expected_row.get(
            'each_arguments_required_substrings'
        )
        if isinstance(each_required_argument_substrings, list):
            for required_argument_substring in each_required_argument_substrings:
                if not (
                    isinstance(required_argument_substring, str)
                    and required_argument_substring
                ):
                    continue
                missing_match_count = sum(
                    required_argument_substring
                    not in json.dumps(row.get('arguments'), sort_keys=True)
                    for row in matches
                )
                if missing_match_count:
                    failures.append(
                        f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} had {missing_match_count} matching row(s) without arguments containing {required_argument_substring!r}'
                    )
        forbidden_command_substrings = []
        configured_forbidden_command = expected_row.get(
            'command_text_forbidden_substring'
        )
        if (
            isinstance(configured_forbidden_command, str)
            and configured_forbidden_command
        ):
            forbidden_command_substrings.append(configured_forbidden_command)
        configured_forbidden_commands = expected_row.get(
            'command_text_forbidden_substrings'
        )
        if isinstance(configured_forbidden_commands, list):
            forbidden_command_substrings.extend(
                value
                for value in configured_forbidden_commands
                if isinstance(value, str) and value
            )
        for forbidden_command_substring in forbidden_command_substrings:
            if any(
                forbidden_command_substring in str(row.get('command_text') or '')
                for row in matches
            ):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} included forbidden command text substring {forbidden_command_substring!r}'
                )
        forbidden_argument_substrings = []
        configured_forbidden_argument = expected_row.get(
            'arguments_forbidden_substring'
        )
        if (
            isinstance(configured_forbidden_argument, str)
            and configured_forbidden_argument
        ):
            forbidden_argument_substrings.append(configured_forbidden_argument)
        configured_forbidden_arguments = expected_row.get(
            'arguments_forbidden_substrings'
        )
        if isinstance(configured_forbidden_arguments, list):
            forbidden_argument_substrings.extend(
                value
                for value in configured_forbidden_arguments
                if isinstance(value, str) and value
            )
        for forbidden_argument_substring in forbidden_argument_substrings:
            if any(
                forbidden_argument_substring
                in json.dumps(row.get('arguments'), sort_keys=True)
                for row in matches
            ):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} included forbidden arguments substring {forbidden_argument_substring!r}'
                )
        matched_records.extend(_normalize_record(row) for row in matches[:minimum_count])

    return {
        'record': matched_records[0] if matched_records else None,
        'records': [_normalize_record(row) for row in records],
        'matched_records': matched_records,
    }, failures


def _claude_projects_root(checks: dict[str, Any]) -> pathlib.Path:
    configured = (
        checks.get('claude_projects_root')
        or checks.get('projects_root')
        or os.environ.get('CLAUDE_PROJECTS_ROOT')
        or os.environ.get('CLAUDE_PROJECTS_DIR')
    )
    if configured:
        return pathlib.Path(str(configured)).expanduser()
    return pathlib.Path.home() / '.claude' / 'projects'


def _iter_claude_jsonl(path: pathlib.Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return records
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append((line_number, parsed))
    return records


def _preview_json(value: Any, *, max_chars: int = 300) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    text = text.replace('\n', '\\n')
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + '...'


def _assistant_message_text(content: list[Any]) -> str:
    return _content_text(
        [
            block
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        ]
    ).strip()


def _append_assistant_text(
    assistant_texts: list[dict[str, Any]],
    *,
    path: pathlib.Path,
    line_number: int,
    message_id: str,
    content: list[Any],
) -> None:
    message_text = _assistant_message_text(content)
    if message_text:
        assistant_texts.append(
            {
                'path': str(path),
                'line': line_number,
                'message_id': message_id,
                'text': message_text,
            }
        )


def _transcript_agent_type(path: pathlib.Path) -> str | None:
    meta_path = path.with_suffix('.meta.json')
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        meta = {}
    if isinstance(meta, dict):
        for key in ('agentType', 'agent_type', 'name'):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for _, record in _iter_claude_jsonl(path)[:5]:
        attachment = record.get('attachment')
        if not isinstance(attachment, dict):
            continue
        content = attachment.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, str):
                continue
            match = re.search(r"You are '([^']+)'", item)
            if match:
                return match.group(1).strip()
    return None


def _find_claude_subagent_transcripts(
    *,
    session_id: str,
    projects_root: pathlib.Path,
    expected_agent: str | None,
) -> tuple[list[pathlib.Path], list[dict[str, Any]]]:
    pattern = f'*/{session_id}/subagents/*.jsonl'
    candidate_paths = sorted(projects_root.glob(pattern))
    candidates: list[dict[str, Any]] = []
    matches: list[pathlib.Path] = []
    expected = expected_agent.strip() if isinstance(expected_agent, str) else None
    for path in candidate_paths:
        if path.name.startswith('agent-acompact-'):
            continue
        agent_type = _transcript_agent_type(path)
        candidate = {
            'path': str(path),
            'agent_type': agent_type,
        }
        candidates.append(candidate)
        if expected is None or agent_type == expected:
            matches.append(path)
    return matches, candidates


def _summarize_transcript_tool_uses(paths: list[pathlib.Path]) -> dict[str, Any]:  # noqa: PLR0915
    by_tool_name: dict[str, int] = {}
    by_assistant_message_id: dict[str, dict[str, int]] = {}
    assistant_texts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    records_by_tool_use_id: dict[str, dict[str, Any]] = {}
    tool_result_errors: list[dict[str, Any]] = []
    transcript_summaries: list[dict[str, Any]] = []
    for path in paths:
        transcript_tool_count = 0
        # Turn index increments on each user record so that genuinely
        # separate assistant messages reusing the same message.id (with
        # tool_results between them) are NOT merged, while fragmented
        # JSONL lines from one logical assistant message (consecutive
        # assistant records, same message.id, no intervening user record)
        # ARE grouped together.
        turn_index = 0
        for line_number, record in _iter_claude_jsonl(path):
            message = record.get('message')
            if not isinstance(message, dict):
                continue
            content = message.get('content')
            if not isinstance(content, list):
                continue
            if message.get('role') == 'user':
                turn_index += 1
                for block in content:
                    if not isinstance(block, dict) or block.get('type') != 'tool_result':
                        continue
                    tool_use_id = str(block.get('tool_use_id') or '')
                    is_error = block.get('is_error') is True
                    tool_result_record = {
                        'path': str(path),
                        'line': line_number,
                        'timestamp': record.get('timestamp'),
                        'agent_id': record.get('agentId'),
                        'entry_uuid': record.get('uuid'),
                        'tool_use_id': tool_use_id,
                        'is_error': is_error,
                        'content_preview': _preview_json(block.get('content')),
                    }
                    if tool_use_id in records_by_tool_use_id:
                        records_by_tool_use_id[tool_use_id][
                            'tool_result_is_error'
                        ] = is_error
                        records_by_tool_use_id[tool_use_id][
                            'tool_result_line'
                        ] = line_number
                        records_by_tool_use_id[tool_use_id][
                            'tool_result_content_preview'
                        ] = tool_result_record['content_preview']
                        records_by_tool_use_id[tool_use_id][
                            'tool_result_content_text'
                        ] = _content_text(block.get('content')).strip()
                    if is_error:
                        tool_result_errors.append(tool_result_record)
                continue
            if message.get('role') != 'assistant':
                continue
            message_id = str(message.get('id') or '')
            # Grouping key: message identity + path + turn discriminator.
            # This merges fragmented JSONL lines from one logical assistant
            # message while keeping separate messages (separated by user
            # tool_result records) distinct even if they reuse an ID.
            logical_message_key = f"{path}:{message_id}:turn{turn_index}" if message_id else f"{path}:{line_number}"
            _append_assistant_text(
                assistant_texts,
                path=path,
                line_number=line_number,
                message_id=message_id,
                content=content,
            )
            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'tool_use':
                    continue
                tool_name = str(block.get('name') or '')
                if not tool_name:
                    continue
                by_tool_name[tool_name] = by_tool_name.get(tool_name, 0) + 1
                message_counts = by_assistant_message_id.setdefault(logical_message_key, {})
                message_counts[tool_name] = message_counts.get(tool_name, 0) + 1
                transcript_tool_count += 1
                tool_use_id = str(block.get('id') or '')
                tool_record = {
                    'path': str(path),
                    'line': line_number,
                    'timestamp': record.get('timestamp'),
                    'agent_id': record.get('agentId'),
                    'message_id': message_id,
                    'entry_uuid': record.get('uuid'),
                    'tool_use_id': tool_use_id,
                    'tool_name': tool_name,
                    'input_preview': _preview_json(block.get('input')),
                }
                records.append(tool_record)
                if tool_use_id:
                    records_by_tool_use_id[tool_use_id] = tool_record
        transcript_summaries.append({
            'path': str(path),
            'agent_type': _transcript_agent_type(path),
            'tool_use_count': transcript_tool_count,
        })

    message_totals = {
        message_id: sum(tool_counts.values())
        for message_id, tool_counts in by_assistant_message_id.items()
    }
    max_tools_in_message = max(message_totals.values(), default=0)
    return {
        'transcripts': transcript_summaries,
        'by_tool_name': dict(sorted(by_tool_name.items())),
        'by_assistant_message_id': {
            message_id: dict(sorted(tool_counts.items()))
            for message_id, tool_counts in sorted(by_assistant_message_id.items())
        },
        'assistant_message_tool_use_totals': dict(sorted(message_totals.items())),
        'max_tool_uses_in_single_assistant_message': max_tools_in_message,
        'total_tool_uses': len(records),
        'tool_result_errors': tool_result_errors,
        'assistant_texts': assistant_texts,
        'records': records,
    }


def _normalize_transcript_agent_checks(checks: dict[str, Any]) -> list[dict[str, Any]]:
    configured_agents = checks.get('expected_agents')
    if isinstance(configured_agents, list) and configured_agents:
        return [agent for agent in configured_agents if isinstance(agent, dict)]

    expected_agent = checks.get('expected_child_agent', checks.get('expected_agent'))
    if expected_agent is None and not (
        checks.get('expected_tool_counts') or checks.get('tool_counts')
    ):
        return []
    return [{
        'agent_type': expected_agent,
        'expected_tool_counts': checks.get('expected_tool_counts')
        or checks.get('tool_counts')
        or {},
        'minimum_total_tool_uses': checks.get('minimum_total_tool_uses'),
        'maximum_total_tool_uses': checks.get('maximum_total_tool_uses'),
        'maximum_tool_uses_per_assistant_message': checks.get(
            'maximum_tool_uses_per_assistant_message'
        ),
        'minimum_tools_in_single_assistant_message': checks.get(
            'minimum_tools_in_single_assistant_message'
        ),
        'minimum_parallel_tool_batches': checks.get(
            'minimum_parallel_tool_batches'
        ),
        'minimum_tools_per_parallel_batch': checks.get(
            'minimum_tools_per_parallel_batch'
        ),
        'forbid_tool_result_errors': checks.get('forbid_tool_result_errors'),
        'expected_tool_sequence': checks.get('expected_tool_sequence'),
        'require_tool_result_before_next_tool_use': checks.get(
            'require_tool_result_before_next_tool_use'
        ),
        'require_all_tool_results': checks.get('require_all_tool_results'),
        'require_child_terminal_response': checks.get(
            'require_child_terminal_response'
        ),
        'require_explicit_completion': checks.get('require_explicit_completion'),
        'child_output_max_chars': checks.get('child_output_max_chars'),
    }]


def _tool_count_bounds(expected: Any) -> tuple[int, int | None]:
    if isinstance(expected, dict):
        raw_minimum = expected.get('minimum_count', expected.get('min', 1))
        raw_maximum = expected.get('maximum_count', expected.get('max'))
    else:
        raw_minimum = expected
        raw_maximum = expected
    try:
        minimum = int(raw_minimum)
    except (TypeError, ValueError):
        minimum = 1
    maximum: int | None
    try:
        maximum = int(raw_maximum) if raw_maximum is not None else None
    except (TypeError, ValueError):
        maximum = None
    return minimum, maximum


def _validate_transcript_agent_tool_uses(  # noqa: PLR0915
    *,
    family: str,
    session_id: str,
    projects_root: pathlib.Path,
    agent_checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    expected_agent = agent_checks.get('agent_type') or agent_checks.get('agent')
    expected_agent_text = (
        str(expected_agent).strip()
        if isinstance(expected_agent, (str, int, float))
        and str(expected_agent).strip()
        else None
    )
    paths, candidates = _find_claude_subagent_transcripts(
        session_id=session_id,
        projects_root=projects_root,
        expected_agent=expected_agent_text,
    )
    summary = {
        'expected_agent': expected_agent_text,
        'candidate_transcripts': candidates,
        **_summarize_transcript_tool_uses(paths),
    }
    failures: list[str] = []
    if not paths:
        failures.append(
            f'{family} missing Claude subagent transcript for agent={expected_agent_text!r} session_id={session_id!r}'
        )
        return summary, failures

    expected_tool_counts = (
        agent_checks.get('expected_tool_counts')
        or agent_checks.get('tool_counts')
        or {}
    )
    if isinstance(expected_tool_counts, dict):
        for tool_name, expected_count in expected_tool_counts.items():
            minimum, maximum = _tool_count_bounds(expected_count)
            actual = int(summary['by_tool_name'].get(str(tool_name), 0))
            if actual < minimum:
                failures.append(
                    f'{family} transcript for agent={expected_agent_text!r} missing tool_use {str(tool_name)!r}; expected >= {minimum}, got {actual}'
                )
            if maximum is not None and actual > maximum:
                failures.append(
                    f'{family} transcript for agent={expected_agent_text!r} had too many tool_use {str(tool_name)!r}; expected <= {maximum}, got {actual}'
                )

    total_tool_uses = int(summary['total_tool_uses'])
    for key, label, comparator in (
        ('minimum_total_tool_uses', 'total tool_use count', '>='),
        ('maximum_total_tool_uses', 'total tool_use count', '<='),
    ):
        raw_expected = agent_checks.get(key)
        if raw_expected is None:
            continue
        expected_int = int(raw_expected)
        if comparator == '>=' and total_tool_uses < expected_int:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} {label} expected >= {expected_int}, got {total_tool_uses}'
            )
        if comparator == '<=' and total_tool_uses > expected_int:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} {label} expected <= {expected_int}, got {total_tool_uses}'
            )

    expected_tool_sequence = agent_checks.get('expected_tool_sequence')
    if isinstance(expected_tool_sequence, list):
        expected_sequence = [str(tool_name) for tool_name in expected_tool_sequence]
        actual_sequence = [
            str(record.get('tool_name') or '')
            for record in (summary.get('records') or [])
        ]
        if actual_sequence != expected_sequence:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} tool_use sequence mismatch; expected {json.dumps(expected_sequence)}, got {json.dumps(actual_sequence)}'
            )

    if agent_checks.get('require_tool_result_before_next_tool_use') is True:
        records = [
            record for record in (summary.get('records') or [])
            if isinstance(record, dict)
        ]
        for previous_record, next_record in zip(records, records[1:]):
            result_line = previous_record.get('tool_result_line')
            previous_path = previous_record.get('path')
            next_path = next_record.get('path')
            try:
                result_line_int = int(result_line)
            except (TypeError, ValueError):
                result_line_int = 0
            try:
                next_line_int = int(next_record.get('line') or 0)
            except (TypeError, ValueError):
                next_line_int = 0
            if previous_path != next_path:
                failures.append(
                    f'{family} transcript for agent={expected_agent_text!r} cannot prove tool_result before next tool_use across transcripts after {previous_record.get("tool_name")!r}'
                )
                continue
            if result_line_int <= 0 or result_line_int >= next_line_int:
                failures.append(
                    f'{family} transcript for agent={expected_agent_text!r} did not record tool_result before next tool_use after {previous_record.get("tool_name")!r}'
                )

    max_per_message = int(summary['max_tool_uses_in_single_assistant_message'])
    raw_max_per_message = agent_checks.get('maximum_tool_uses_per_assistant_message')
    if raw_max_per_message is not None:
        expected_max = int(raw_max_per_message)
        if max_per_message > expected_max:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} had {max_per_message} tool_use blocks in one assistant message; expected <= {expected_max}'
            )
    raw_min_parallel = agent_checks.get('minimum_tools_in_single_assistant_message')
    if raw_min_parallel is not None:
        expected_min = int(raw_min_parallel)
        if max_per_message < expected_min:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} never had >= {expected_min} tool_use blocks in one assistant message; max was {max_per_message}'
            )

    raw_min_parallel_batches = agent_checks.get('minimum_parallel_tool_batches')
    if raw_min_parallel_batches is not None:
        expected_batches = int(raw_min_parallel_batches)
        minimum_tools_per_batch = int(
            agent_checks.get('minimum_tools_per_parallel_batch') or 2
        )
        message_tool_totals = summary.get('assistant_message_tool_use_totals') or {}
        qualifying_batches = sum(
            1
            for total in message_tool_totals.values()
            if int(total) >= minimum_tools_per_batch
        )
        summary['parallel_tool_batch_validation'] = {
            'minimum_batches': expected_batches,
            'minimum_tools_per_batch': minimum_tools_per_batch,
            'qualifying_batches': qualifying_batches,
        }
        if qualifying_batches < expected_batches:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} had {qualifying_batches} parallel tool batches with >= {minimum_tools_per_batch} tool_use blocks; expected >= {expected_batches}'
            )

    if agent_checks.get('forbid_tool_result_errors') is True:
        tool_result_errors = summary.get('tool_result_errors') or []
        if tool_result_errors:
            previews = [
                {
                    'line': error.get('line'),
                    'tool_use_id': error.get('tool_use_id'),
                    'content_preview': error.get('content_preview'),
                }
                for error in tool_result_errors[:5]
            ]
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} had tool_result errors: {json.dumps(previews, sort_keys=True)}'
            )

    # Child proof: every tool_use must have a corresponding successful tool_result.
    if agent_checks.get('require_all_tool_results') is True:
        records = summary.get('records') or []
        missing_results: list[str] = []
        errored_results: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            tool_use_id = str(record.get('tool_use_id') or '')
            tool_name = str(record.get('tool_name') or '')
            if not record.get('tool_result_line'):
                missing_results.append(f'{tool_name}(id={tool_use_id})')
            elif record.get('tool_result_is_error') is True:
                errored_results.append(f'{tool_name}(id={tool_use_id})')
        if missing_results:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} tool_use without tool_result: {", ".join(missing_results[:10])}'
            )
        if errored_results:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} tool_use with failed tool_result: {", ".join(errored_results[:10])}'
            )

    # Child proof: child must emit a required exact terminal response.
    required_terminal = agent_checks.get('require_child_terminal_response')
    if isinstance(required_terminal, str) and required_terminal:
        assistant_texts = summary.get('assistant_texts') or []
        final_text = ''
        for item in reversed(assistant_texts):
            if isinstance(item, dict) and isinstance(item.get('text'), str):
                final_text = item['text'].strip()
                break
        if not final_text:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} missing child terminal response text'
            )
        elif final_text != required_terminal:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} child terminal response mismatch: expected {required_terminal!r}, got {final_text[:200]!r}'
            )

    # Child proof: child must explicitly complete/terminate (at least one
    # assistant text after all tool_use blocks).
    if agent_checks.get('require_explicit_completion') is True:
        assistant_texts = summary.get('assistant_texts') or []
        records = summary.get('records') or []
        last_tool_line = 0
        for record in records:
            if isinstance(record, dict):
                try:
                    last_tool_line = max(last_tool_line, int(record.get('line') or 0))
                except (TypeError, ValueError):
                    pass
        has_terminal_text = any(
            isinstance(item, dict)
            and isinstance(item.get('text'), str)
            and item['text'].strip()
            and int(item.get('line') or 0) > last_tool_line
            for item in assistant_texts
        )
        if not has_terminal_text:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} no explicit completion text after final tool_use'
            )

    # Child proof: child output must remain bounded.
    child_output_max = agent_checks.get('child_output_max_chars')
    if child_output_max is not None:
        max_chars = int(child_output_max)
        assistant_texts = summary.get('assistant_texts') or []
        total_chars = sum(
            len(item.get('text') or '')
            for item in assistant_texts
            if isinstance(item, dict)
        )
        if total_chars > max_chars:
            failures.append(
                f'{family} transcript for agent={expected_agent_text!r} child output {total_chars} chars exceeds bound {max_chars}'
            )

    return summary, failures


def _validate_transcript_tool_use(
    *,
    family: str,
    session_id: str | None,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not session_id:
        return {'agents': []}, [f'{family} missing command session_id for transcript tool_use validation']

    projects_root = _claude_projects_root(checks)
    agent_checks = _normalize_transcript_agent_checks(checks)
    if not agent_checks:
        return {'projects_root': str(projects_root), 'agents': []}, []

    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    final_summary: dict[str, Any] = {'projects_root': str(projects_root), 'agents': []}
    final_failures: list[str] = []
    while True:
        summaries: list[dict[str, Any]] = []
        failures: list[str] = []
        for one_agent_checks in agent_checks:
            summary, agent_failures = _validate_transcript_agent_tool_uses(
                family=family,
                session_id=session_id,
                projects_root=projects_root,
                agent_checks=one_agent_checks,
            )
            summaries.append(summary)
            failures.extend(agent_failures)

        final_summary = {
            'projects_root': str(projects_root),
            'session_id': session_id,
            'agents': summaries,
        }
        final_failures = failures
        if not failures or time.monotonic() >= poll_deadline:
            break
        time.sleep(poll_interval_seconds)

    return final_summary, final_failures


def _validate_bash_stdout_report(
    *,
    family: str,
    stdout: str,
    checks: dict[str, Any],
    transcript_tool_use_summary: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not checks:
        return {"enabled": False}, []

    expected_command = str(checks.get("expected_command") or "").strip()
    expected_pattern = str(checks.get("expected_regex") or "").strip()
    expected_agent = str(checks.get("transcript_agent") or "").strip()
    final_output = _extract_command_output_text(stdout).strip()
    failures: list[str] = []
    bash_stdout = ""
    child_output = ""
    source = "codex_command_stdout"

    if expected_agent:
        source = "claude_subagent_transcript"
        agent_summaries = [
            summary
            for summary in transcript_tool_use_summary.get("agents") or []
            if isinstance(summary, dict)
            and summary.get("expected_agent") == expected_agent
        ]
        if len(agent_summaries) != 1:
            failures.append(
                f"{family} Bash stdout report expected one transcript summary for agent={expected_agent!r}, got {len(agent_summaries)}"
            )
        else:
            agent_summary = agent_summaries[0]
            bash_records = [
                record
                for record in agent_summary.get("records") or []
                if isinstance(record, dict)
                and record.get("tool_name") == "Bash"
                and expected_command in str(record.get("input_preview") or "")
            ]
            if len(bash_records) != 1:
                failures.append(
                    f"{family} Bash stdout report expected one `{expected_command}` tool result for agent={expected_agent!r}, got {len(bash_records)}"
                )
            else:
                bash_stdout = str(
                    bash_records[0].get("tool_result_content_text") or ""
                ).strip()
            assistant_texts = [
                item
                for item in agent_summary.get("assistant_texts") or []
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if assistant_texts:
                child_output = str(assistant_texts[-1]["text"]).strip()
            else:
                failures.append(
                    f"{family} Bash stdout report missing final child text for agent={expected_agent!r}"
                )
    else:
        matches = [
            record
            for record in _collect_codex_command_execution_results(stdout)
            if expected_command in record["command"]
        ]
        if len(matches) != 1:
            failures.append(
                f"{family} Bash stdout report expected one `{expected_command}` command execution, got {len(matches)}"
            )
        else:
            bash_stdout = matches[0]["output"].strip()

    if expected_pattern:
        try:
            timestamp_matches = re.fullmatch(expected_pattern, bash_stdout) is not None
        except re.error as exc:
            failures.append(
                f"{family} Bash stdout report has invalid expected regex {expected_pattern!r}: {exc}"
            )
        else:
            if not timestamp_matches:
                failures.append(
                    f"{family} Bash stdout {bash_stdout!r} did not match {expected_pattern!r}"
                )

    if expected_agent and bash_stdout and child_output != bash_stdout:
        failures.append(
            f"{family} child response did not exactly report Bash stdout: expected {bash_stdout!r}, got {child_output!r}"
        )
    if bash_stdout and final_output != bash_stdout:
        failures.append(
            f"{family} parent response did not exactly report Bash stdout: expected {bash_stdout!r}, got {final_output!r}"
        )

    return {
        "enabled": True,
        "source": source,
        "expected_command": expected_command,
        "bash_stdout": bash_stdout,
        "child_output": child_output or None,
        "parent_output": final_output,
    }, failures


def _downgrade_configured_failures_to_warnings(
    *,
    failures: list[str],
    config: dict[str, Any],
    command_json_summary: dict[str, Any],
) -> tuple[list[str], list[str]]:
    rules = config.get('downgrade_failures_to_warnings') or []
    if not rules or not failures:
        return failures, []

    parsed_command = command_json_summary.get('parsed')
    command_result_text = (
        parsed_command.get('result')
        if isinstance(parsed_command, dict)
        and isinstance(parsed_command.get('result'), str)
        else ''
    )

    remaining_failures: list[str] = []
    warning_messages: list[str] = []
    for failure in failures:
        downgraded = False
        for rule in rules:
            failure_contains = rule.get('failure_contains')
            result_contains = rule.get('if_command_result_contains')
            if not isinstance(failure_contains, str) or failure_contains not in failure:
                continue
            if isinstance(result_contains, str) and result_contains not in command_result_text:
                continue
            warning_messages.append(f'downgraded failure: {failure}')
            downgraded = True
            break
        if not downgraded:
            remaining_failures.append(failure)

    return remaining_failures, warning_messages


def _split_warning_only_failures(
    *,
    failures: list[str],
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    hard_substrings = [
        *DEFAULT_WARNING_ONLY_HARD_FAILURE_SUBSTRINGS,
        *list(config.get('warning_only_hard_failure_substrings') or []),
    ]
    if bool(config.get('warning_only_allow_timeouts')):
        hard_substrings = [
            value for value in hard_substrings if value != 'timed out after'
        ]

    hard_failures: list[str] = []
    soft_failures: list[str] = []
    for failure in failures:
        if any(substring and substring in failure for substring in hard_substrings):
            hard_failures.append(failure)
        else:
            soft_failures.append(failure)
    return hard_failures, soft_failures


def _is_warning_only_hard_exception(
    *,
    exc: Exception,
    config: dict[str, Any],
) -> bool:
    hard_substrings = [
        *DEFAULT_WARNING_ONLY_HARD_FAILURE_SUBSTRINGS,
        *list(config.get('warning_only_hard_failure_substrings') or []),
    ]
    if bool(config.get('warning_only_allow_timeouts')):
        hard_substrings = [
            value for value in hard_substrings if value != 'timed out after'
        ]
    error_text = str(exc)
    return any(substring and substring in error_text for substring in hard_substrings)


def _inject_http_litellm_metadata(
    body: Any,
    *,
    session_id: str,
    trace_name: str,
) -> Any:
    if not isinstance(body, dict):
        return body
    updated = dict(body)
    metadata = dict(updated.get('litellm_metadata') or {})
    metadata.setdefault('session_id', session_id)
    metadata.setdefault('trace_name', trace_name)
    updated['litellm_metadata'] = metadata
    return updated


def _expand_repeat_text_fixtures(value: Any) -> Any:
    if isinstance(value, dict):
        if 'repeat_text' in value and 'count' in value:
            repeat_text = value.get('repeat_text')
            separator = value.get('separator', '')
            try:
                count = int(value.get('count'))
            except (TypeError, ValueError):
                count = 0
            if isinstance(repeat_text, str) and isinstance(separator, str) and count >= 0:
                return separator.join([repeat_text] * count)
        return {
            key: _expand_repeat_text_fixtures(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_expand_repeat_text_fixtures(child) for child in value]
    return value


def _expand_env_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _summarize_http_response_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    usage = payload.get('usage')
    if not isinstance(usage, dict):
        usage = payload.get('usageMetadata')
    summary: dict[str, Any] = {}
    if isinstance(usage, dict):
        summary['usage'] = usage
    if isinstance(payload.get('id'), str):
        summary['id'] = payload['id']
    if isinstance(payload.get('model'), str):
        summary['model'] = payload['model']
    return summary


def _http_request_repeat_count(config: dict[str, Any]) -> int:
    request_config = dict(config.get('http_request') or {})
    raw_value = request_config.get(
        'repeat_count',
        request_config.get(
            'http_request_repeat_count',
            config.get(
                'http_request_repeat_count',
                2 if bool(config.get('repeat_http_request')) else 1,
            ),
        ),
    )
    try:
        repeat_count = int(raw_value)
    except (TypeError, ValueError):
        repeat_count = 1
    return max(1, repeat_count)


def _http_request_repeat_delay_seconds(config: dict[str, Any]) -> float:
    request_config = dict(config.get('http_request') or {})
    raw_value = request_config.get(
        'repeat_delay_seconds',
        request_config.get(
            'http_request_repeat_delay_seconds',
            config.get('http_request_repeat_delay_seconds', 0),
        ),
    )
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 0.0


def _run_http_request(config: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    request_config = dict(config.get('http_request') or {})
    method = str(request_config.get('method') or 'POST').upper()
    base_url = str(request_config.get('litellm_base_url') or '').rstrip('/')
    path = str(request_config.get('path') or '')
    if not base_url or not path.startswith('/'):
        raise RuntimeError('http_request requires litellm_base_url and absolute path')

    query = dict(request_config.get('query') or {})
    if request_config.get('auth_query_param'):
        auth_query_value = request_config.get('auth_query_param_value')
        auth_query_env = request_config.get('auth_query_param_env')
        if auth_query_value is None and isinstance(auth_query_env, str):
            auth_query_value = os.environ.get(auth_query_env)
        query.setdefault(
            str(request_config['auth_query_param']),
            str(auth_query_value or 'sk-1234'),
        )
    url = f'{base_url}{path}'
    if query:
        separator = '&' if '?' in url else '?'
        url = f'{url}{separator}{urllib.parse.urlencode(query)}'

    headers = {
        str(key): _expand_env_placeholders(str(value))
        for key, value in (request_config.get('headers') or {}).items()
    }
    body = request_config.get('json')
    session_id = str(request_config.get('session_id') or '')
    if session_id:
        body = _inject_http_litellm_metadata(
            body,
            session_id=session_id,
            trace_name=str(headers.get('langfuse_trace_name') or config.get('case_name') or 'native-passthrough'),
        )
    body = _expand_repeat_text_fixtures(body)

    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers.setdefault('content-type', 'application/json')

    started = time.time()
    status_code: int | None = None
    response_text = ''
    parsed_response: Any = None
    error_text: str | None = None
    try:
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(
            request,
            timeout=int(request_config.get('timeout_seconds') or config.get('timeout_seconds') or 300),
        ) as response:
            status_code = int(response.status)
            response_text = response.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        response_text = exc.read().decode('utf-8', errors='replace')
        error_text = str(exc)
    except urllib.error.URLError as exc:
        error_text = str(exc)
    if response_text:
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            parsed_response = None

    is_error = error_text is not None or status_code is None or status_code >= 400
    stdout_payload = {
        'session_id': session_id,
        'status_code': status_code,
        'is_error': is_error,
        'url': url,
        **_summarize_http_response_payload(parsed_response),
    }
    if error_text:
        stdout_payload['error'] = error_text
    if parsed_response is None and response_text:
        stdout_payload['response_excerpt'] = response_text[:1000]

    return {
        'command': [method, url],
        'command_string': f'{method} {url}',
        'exit_code': 1 if is_error else 0,
        'duration_seconds': round(time.time() - started, 3),
        'stdout': json.dumps(stdout_payload),
        'stderr': error_text or '',
        'response_excerpt': response_text[:300],
    }


def _run_http_request_with_repeat(config: dict[str, Any]) -> dict[str, Any]:
    repeat_count = _http_request_repeat_count(config)
    if repeat_count <= 1:
        return _run_http_request(config)

    delay_seconds = _http_request_repeat_delay_seconds(config)
    pass_results: list[dict[str, Any]] = []
    final_run: dict[str, Any] | None = None
    started = time.time()
    for pass_index in range(1, repeat_count + 1):
        run = _run_http_request(config)
        parsed_stdout = _parse_command_output_json(run.get('stdout', ''))
        pass_summary = {
            'pass': pass_index,
            'exit_code': run.get('exit_code'),
            'duration_seconds': run.get('duration_seconds'),
            'command': run.get('command'),
            'command_string': run.get('command_string'),
            'stderr': run.get('stderr'),
            'response_excerpt': run.get('response_excerpt'),
            'stdout': parsed_stdout if parsed_stdout is not None else run.get('stdout'),
        }
        pass_results.append(pass_summary)
        final_run = run
        if pass_index < repeat_count and delay_seconds > 0:
            time.sleep(delay_seconds)

    if final_run is None:
        raise RuntimeError('http_request repeat loop produced no run result')

    final_stdout = _parse_command_output_json(final_run.get('stdout', '')) or {}
    if not isinstance(final_stdout, dict):
        final_stdout = {}
    stdout_payload = {
        **final_stdout,
        'http_request_repeat_count': repeat_count,
        'http_request_passes': [
            {
                key: value
                for key, value in pass_result.items()
                if key
                in {
                    'pass',
                    'exit_code',
                    'duration_seconds',
                    'stdout',
                    'stderr',
                }
            }
            for pass_result in pass_results
        ],
    }
    return {
        **final_run,
        'exit_code': 1 if any(pass_result.get('exit_code') != 0 for pass_result in pass_results) else 0,
        'duration_seconds': round(time.time() - started, 3),
        'stdout': json.dumps(stdout_payload),
        'stderr': '\n'.join(
            str(pass_result.get('stderr') or '')
            for pass_result in pass_results
            if pass_result.get('stderr')
        ),
        'http_request_repeat_count': repeat_count,
        'http_request_passes': pass_results,
    }


def _run_command_with_retry(*, config: dict[str, Any]) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    retry_statuses = {int(value) for value in (config.get('retry_on_api_error_statuses') or [])}
    max_attempts = max(1, int(config.get('retry_max_attempts', 1) or 1))
    base_backoff_seconds = float(config.get('retry_backoff_seconds', 0) or 0)

    attempts: list[dict[str, Any]] = []
    final_started = RA._utcnow()
    final_run: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        started = RA._utcnow()
        if isinstance(config.get('http_request'), dict):
            run = _run_http_request_with_repeat(config)
        else:
            run = RA._run_command(
                config['command'],
                extra_env=config.get('env'),
                timeout_seconds=int(config.get('timeout_seconds', 300)),
            )
        parsed = _parse_command_output_json(run['stdout'])
        api_error_status = None
        is_error = None
        if isinstance(parsed, dict):
            api_error_status = parsed.get('api_error_status')
            is_error = parsed.get('is_error')
            if (
                api_error_status is None
                and is_error is True
                and isinstance(parsed.get('status_code'), int)
            ):
                api_error_status = parsed.get('status_code')
        attempts.append({
            'attempt': attempt,
            'started_at': RA._isoformat(started),
            'exit_code': run.get('exit_code'),
            'api_error_status': api_error_status,
            'is_error': is_error,
        })
        final_started = started
        final_run = run

        should_retry = (
            attempt < max_attempts
            and isinstance(api_error_status, int)
            and api_error_status in retry_statuses
        )
        if not should_retry:
            break
        sleep_seconds = base_backoff_seconds * attempt if base_backoff_seconds > 0 else 0
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if final_run is None:
        raise RuntimeError('command retry loop produced no run result')
    return final_started, final_run, attempts


def _command_api_error_status(
    *,
    run: dict[str, Any],
    command_attempts: list[dict[str, Any]],
) -> int | None:
    if command_attempts:
        status = command_attempts[-1].get('api_error_status')
        if isinstance(status, int):
            return status
    parsed_stdout = _parse_command_output_json(str(run.get('stdout') or ''))
    if isinstance(parsed_stdout, dict):
        status = parsed_stdout.get('api_error_status')
        if isinstance(status, int):
            return status
        if parsed_stdout.get('is_error') is True:
            status = parsed_stdout.get('status_code')
            if isinstance(status, int):
                return status
    return None


def _validate_case(name: str, config: dict[str, Any], *, query_url: str, public_key: str, secret_key: str, litellm_base_url: str, cfg003_transactional: bool = False) -> dict[str, Any]:  # noqa: PLR0915
    started, run, command_attempts = _run_command_with_retry(config=config)
    observed_api_error_status = _command_api_error_status(
        run=run,
        command_attempts=command_attempts,
    )
    expected_api_error_status = config.get('expected_api_error_status')
    expected_api_error_matched = (
        run.get('exit_code') != 0
        and isinstance(expected_api_error_status, int)
        and observed_api_error_status == expected_api_error_status
    )
    use_failure_observability = (
        expected_api_error_matched
        and bool(config.get('provider_error_observations_validation'))
    )
    use_expected_429_generation_policy = (
        use_failure_observability
        and expected_api_error_status == 429
    )
    command_session_id = _extract_command_session_id(run['stdout'])
    command_thread_id = _extract_command_thread_id(run['stdout'])
    if (
        not config.get('match_trace_session_id_from_stdout')
        and isinstance(config.get('expected_trace_session_id'), str)
        and str(config.get('expected_trace_session_id')).strip()
    ):
        command_session_id = str(config['expected_trace_session_id']).strip()
    post_run_wait_seconds = float(config.get('post_run_wait_seconds', 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)

    expected_trace_names = config.get('required_trace_names', [])
    expected_user_ids = config.get('expected_user_ids', [])
    expected_trace_user_ids_by_name = _normalize_expected_trace_user_ids_by_name(
        config.get('expected_trace_user_ids_by_name')
    )
    lookup_user_id = _resolve_trace_lookup_user_id(
        expected_user_ids,
        expected_trace_user_ids_by_name,
    )
    use_session_trace_lookup = bool(config.get('use_session_trace_lookup', True))
    can_session_trace_lookup = (
        use_session_trace_lookup
        and isinstance(command_session_id, str)
        and command_session_id.strip()
    )
    expected_error_generation_poll_timeout_seconds = max(
        float(
            config.get(
                'expected_api_error_langfuse_poll_timeout_seconds',
                10,
            )
            or 0
        ),
        0.0,
    )
    expected_error_generation_poll_interval_seconds = max(
        float(
            config.get(
                'expected_api_error_langfuse_poll_interval_seconds',
                1,
            )
            or 0
        ),
        0.0,
    )
    observed_error_generations: list[dict[str, Any]] = []
    expected_error_generation_poll_attempts = 0
    expected_error_generation_lookup_error: str | None = None
    expected_error_generation_missing_context = False
    if use_failure_observability and not use_expected_429_generation_policy:
        traces, lookup_error = [], None
    elif use_expected_429_generation_policy:
        traces, lookup_error = [], None
        poll_deadline = (
            time.monotonic() + expected_error_generation_poll_timeout_seconds
        )
        expected_error_generation_missing_context = not (
            expected_trace_names or can_session_trace_lookup
        )
        while True:
            remaining_seconds = poll_deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            if expected_trace_names:
                try:
                    traces = RA._recent_langfuse_required_name_traces(
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        names=expected_trace_names,
                        user_id=lookup_user_id,
                        start_time=started,
                        limit=100,
                        deadline=poll_deadline,
                    )
                    lookup_error = None
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    http.client.RemoteDisconnected,
                    ConnectionResetError,
                    TimeoutError,
                ) as exc:
                    traces = []
                    lookup_error = str(exc)
            elif can_session_trace_lookup:
                try:
                    traces = RA._recent_langfuse_all_traces(
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        user_id=lookup_user_id,
                        start_time=started,
                        session_id=command_session_id.strip(),
                        limit=100,
                        deadline=poll_deadline,
                    )
                    lookup_error = None
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    http.client.RemoteDisconnected,
                    ConnectionResetError,
                    TimeoutError,
                ) as exc:
                    traces = []
                    lookup_error = str(exc)
            else:
                break

            expected_error_generation_poll_attempts += 1
            trace_ids_for_probe = [
                trace.get('id')
                for trace in traces
                if isinstance(trace.get('id'), str)
            ]
            if trace_ids_for_probe:
                try:
                    observed_error_generations = (
                        RA._recent_langfuse_generation_observations_for_trace_ids(
                            query_url=query_url,
                            public_key=public_key,
                            secret_key=secret_key,
                            trace_ids=trace_ids_for_probe,
                            start_time=started,
                            deadline=poll_deadline,
                        )
                    )
                    expected_error_generation_lookup_error = None
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    http.client.RemoteDisconnected,
                    ConnectionResetError,
                    TimeoutError,
                ) as exc:
                    observed_error_generations = []
                    expected_error_generation_lookup_error = str(exc)
            if observed_error_generations:
                break

            remaining_seconds = poll_deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            sleep_seconds = min(
                expected_error_generation_poll_interval_seconds,
                remaining_seconds,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    elif expected_trace_names:
        # Prefer name-based lookup when the suite already knows which traces should exist.
        # This avoids spending the full session lookup timeout on providers that log a
        # null trace.sessionId; trace-context validation below still enforces sessionId
        # requirements for the cases that care about them.
        traces, lookup_error = RA._poll_langfuse_required_name_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            names=expected_trace_names,
            user_id=lookup_user_id,
            start_time=started,
            limit=100,
            timeout_seconds=int(config.get("langfuse_poll_timeout_seconds", 60)),
        )
        # Fix 5: When name-based lookup finds nothing but a session ID is
        # available, fall back to session-based lookup.  The proxy may not
        # propagate langfuse_trace_name to the Langfuse trace name field.
        if not traces and can_session_trace_lookup:
            traces, lookup_error = RA._poll_langfuse_session_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=None,  # session discovery must not filter by expected user_id
                start_time=started,
                session_id=command_session_id.strip(),
                timeout_seconds=int(config.get('langfuse_poll_timeout_seconds', 60)),
            )
    elif can_session_trace_lookup:
        traces, lookup_error = RA._poll_langfuse_session_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            user_id=lookup_user_id,
            start_time=started,
            session_id=command_session_id.strip(),
            timeout_seconds=int(config.get('langfuse_poll_timeout_seconds', 60)),
        )
    else:
        traces = []
        lookup_error = None

    actual_trace_names = sorted({trace.get('name') for trace in traces if trace.get('name')})
    actual_user_ids = sorted({trace.get('userId') for trace in traces if trace.get('userId')})
    trace_ids = [trace.get('id') for trace in traces if trace.get('id')]

    # Fix 2: When generation_trace_names is configured, scope generation,
    # request-payload, and alias-route checks to only those trace rows
    # (typically alias-child traces), excluding native parent traffic.
    generation_trace_names = config.get('generation_trace_names')
    if isinstance(generation_trace_names, list) and generation_trace_names:
        generation_trace_name_set = set(generation_trace_names)
        generation_trace_ids = [
            trace.get('id')
            for trace in traces
            if trace.get('id') and trace.get('name') in generation_trace_name_set
        ]
    else:
        generation_trace_ids = trace_ids

    # Fix 5: When Langfuse returns zero traces without a query error, emit
    # one explicit required observability/correlation failure and skip
    # dependent payload/tag/selection assertions.  Do not substitute
    # session_history for required Langfuse proof or fabricate trace IDs.
    langfuse_zero_trace_correlation_failure = (
        not traces
        and not lookup_error
        and not use_failure_observability
    )

    failures: list[str] = []
    warnings: list[str] = []
    if run['exit_code'] != 0:
        if expected_api_error_matched:
            warnings.append(
                f'{name} command exited nonzero with expected API error status {expected_api_error_status}'
            )
        else:
            failures.append(f'{name} command failed')
    elif isinstance(expected_api_error_status, int):
        failures.append(
            f'{name} command succeeded but expected API error status {expected_api_error_status}'
        )
    if lookup_error:
        warnings.append(f'{name} Langfuse lookup warning: {lookup_error}')
    if expected_error_generation_lookup_error:
        warnings.append(
            f'{name} expected API error generation lookup warning: '
            f'{expected_error_generation_lookup_error}'
        )
    if use_failure_observability:
        trace_user_ids_by_name_summary = {
            'skipped': 'expected_api_error',
        }
        trace_user_ids_by_name_failures = []
    elif langfuse_zero_trace_correlation_failure:
        failures.append(
            f'{name} required Langfuse observability correlation failed: zero '
            'traces returned without a lookup error; dependent payload/tag/'
            'selection assertions skipped (session_history is not a substitute '
            'for required Langfuse proof)'
        )
        trace_user_ids_by_name_summary = {
            'skipped': 'langfuse_zero_trace_correlation',
        }
        trace_user_ids_by_name_failures = []
    else:
        for trace_name in expected_trace_names:
            if trace_name not in actual_trace_names:
                failures.append(f'missing {name} trace name: {trace_name}')
        for user_id in expected_user_ids:
            if user_id not in actual_user_ids:
                failures.append(f'missing {name} user id: {user_id}')
        trace_user_ids_by_name_summary, trace_user_ids_by_name_failures = (
            _validate_trace_user_ids_by_name(
                family=name,
                traces=traces,
                expected=expected_trace_user_ids_by_name,
            )
        )
    failures.extend(trace_user_ids_by_name_failures)
    if (
        not use_failure_observability
        and bool(config.get('require_trace_user_id'))
        and traces
        and not actual_user_ids
    ):
        failures.append(f'{name} traces did not include a Langfuse userId')

    if use_failure_observability and not use_expected_429_generation_policy:
        raw_generation_observations = []
        generation_observations = []
        generation_failures = []
        generation_validation_summary = {'skipped': 'expected_api_error'}
    elif use_expected_429_generation_policy:
        if expected_error_generation_missing_context:
            raw_generation_observations = []
            generation_observations = []
            generation_failures = [
                f'{name} missing Langfuse trace or session lookup context for '
                'expected API error generation validation'
            ]
            generation_validation_summary = {
                'failed': 'missing_lookup_context',
                'poll_attempts': 0,
                'poll_timeout_seconds': (
                    expected_error_generation_poll_timeout_seconds
                ),
            }
        elif expected_error_generation_poll_attempts == 0:
            raw_generation_observations = []
            generation_observations = []
            generation_failures = [
                f'{name} expected API error generation validation completed '
                'without a Langfuse lookup attempt'
            ]
            generation_validation_summary = {
                'failed': 'no_lookup_attempts',
                'poll_attempts': 0,
                'poll_timeout_seconds': (
                    expected_error_generation_poll_timeout_seconds
                ),
            }
        elif observed_error_generations:
            (
                raw_generation_observations,
                generation_observations,
                generation_failures,
            ) = RA._validate_generation_observations(
                family=name,
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                trace_ids=trace_ids,
                start_time=started,
                allowed_request_routes=config.get('allowed_generation_routes'),
                skip_quality_checks=bool(
                    config.get('skip_generation_quality_checks')
                ),
                allow_zero_cost=bool(config.get('allow_zero_cost')),
                allow_reference_cost_when_invoice_unknown=bool(
                    config.get('allow_reference_cost_when_invoice_unknown')
                ),
                allow_unknown_cost_when_invoice_unknown=bool(
                    config.get('allow_unknown_cost_when_invoice_unknown')
                ),
                preloaded_observations=observed_error_generations,
            )
            generation_validation_summary = {
                'validated': 'expected_api_error',
                'poll_attempts': expected_error_generation_poll_attempts,
                'poll_timeout_seconds': (
                    expected_error_generation_poll_timeout_seconds
                ),
            }
        else:
            raw_generation_observations = []
            generation_observations = []
            generation_failures = []
            if lookup_error or expected_error_generation_lookup_error:
                generation_failures.append(
                    f'{name} expected API error generation absence could not '
                    'be established after Langfuse lookup failure'
                )
                generation_validation_summary = {
                    'failed': 'langfuse_lookup',
                    'poll_attempts': expected_error_generation_poll_attempts,
                    'poll_timeout_seconds': (
                        expected_error_generation_poll_timeout_seconds
                    ),
                }
            else:
                generation_validation_summary = {
                    'skipped': 'expected_api_error',
                    'poll_attempts': expected_error_generation_poll_attempts,
                    'poll_timeout_seconds': (
                        expected_error_generation_poll_timeout_seconds
                    ),
                }
    elif langfuse_zero_trace_correlation_failure:
        raw_generation_observations = []
        generation_observations = []
        generation_failures = []
        generation_validation_summary = {
            'skipped': 'langfuse_zero_trace_correlation',
        }
    else:
        (
            raw_generation_observations,
            generation_observations,
            generation_failures,
        ) = RA._validate_generation_observations(
            family=name,
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            trace_ids=generation_trace_ids,
            start_time=started,
            allowed_request_routes=config.get('allowed_generation_routes'),
            skip_quality_checks=bool(config.get('skip_generation_quality_checks')),
            allow_zero_cost=bool(config.get('allow_zero_cost')),
            allow_reference_cost_when_invoice_unknown=bool(
                config.get('allow_reference_cost_when_invoice_unknown')
            ),
            allow_unknown_cost_when_invoice_unknown=bool(
                config.get('allow_unknown_cost_when_invoice_unknown')
            ),
        )
        generation_validation_summary = {'validated': True}
    failures.extend(generation_failures)

    filtered_trace_ids = sorted(
        {
            observation.get('traceId')
            for observation in raw_generation_observations
            if isinstance(observation.get('traceId'), str)
        }
    )
    filtered_traces = [trace for trace in traces if trace.get('id') in set(filtered_trace_ids)]

    if use_failure_observability:
        trace_enrichment_summary = {'skipped': 'expected_api_error'}
        trace_enrichment_failures = []
        trace_enrichment_warnings = []
    elif langfuse_zero_trace_correlation_failure:
        trace_enrichment_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        trace_enrichment_failures = []
        trace_enrichment_warnings = []
    else:
        (
            trace_enrichment_summary,
            trace_enrichment_failures,
            trace_enrichment_warnings,
        ) = RA._validate_trace_enrichment(
            family=name,
            traces=filtered_traces,
            required_tags=config.get('required_trace_tags'),
            required_tag_prefixes=config.get('required_trace_tag_prefixes'),
            warning_tag_prefixes=config.get('warning_trace_tag_prefixes'),
        )
    failures.extend(trace_enrichment_failures)
    warnings.extend(trace_enrichment_warnings)

    if use_failure_observability:
        trace_context_summary = {'skipped': 'expected_api_error'}
        trace_context_failures = []
    elif langfuse_zero_trace_correlation_failure:
        trace_context_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        trace_context_failures = []
    else:
        trace_context_summary, trace_context_failures = RA._validate_trace_context(
            family=name,
            traces=filtered_traces,
            expected_environment=config.get('expected_trace_environment'),
            require_trace_session_id=bool(config.get('require_trace_session_id')),
            expected_trace_session_id=(command_session_id if config.get('match_trace_session_id_from_stdout') else config.get('expected_trace_session_id')),
            require_trace_ids_distinct_from_session_ids=bool(config.get('require_trace_ids_distinct_from_session_ids')),
        )
    failures.extend(trace_context_failures)

    if use_failure_observability:
        generation_metadata_summary = {'skipped': 'expected_api_error'}
        generation_metadata_failures = []
    elif langfuse_zero_trace_correlation_failure:
        generation_metadata_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        generation_metadata_failures = []
    else:
        (
            generation_metadata_summary,
            generation_metadata_failures,
        ) = RA._validate_generation_metadata(
            family=name,
            observations=raw_generation_observations,
            required_metadata_truthy=config.get('required_generation_metadata_truthy'),
            required_metadata_minimums=config.get('required_generation_metadata_minimums'),
        )
    failures.extend(generation_metadata_failures)

    if langfuse_zero_trace_correlation_failure:
        request_payload_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        request_payload_failures: list[str] = []
        request_payload_warnings: list[str] = []
        request_text_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        request_text_failures: list[str] = []
        request_text_warnings: list[str] = []
        stream_tool_call_state_summary = {'skipped': 'langfuse_zero_trace_correlation'}
        stream_tool_call_state_failures: list[str] = []
    else:
        request_payload_summary, request_payload_failures, request_payload_warnings = _validate_logged_request_payload_checks(
            family=name,
            observations=raw_generation_observations,
            checks=config.get('request_payload_checks') or {},
        )
        request_text_summary, request_text_failures, request_text_warnings = RA._validate_logged_request_text_checks(
            family=name,
            observations=raw_generation_observations,
            required_substrings=(config.get('request_text_checks') or {}).get('required_substrings'),
            forbidden_substrings=(config.get('request_text_checks') or {}).get('forbidden_substrings'),
            warning_required_substrings=(config.get('request_text_checks') or {}).get('warning_required_substrings'),
        )
        stream_tool_call_state_summary, stream_tool_call_state_failures = _validate_stream_tool_call_state(
            family=name,
            observations=raw_generation_observations,
            checks=config.get('stream_tool_call_state_validation') or {},
            command_stdout=run.get('stdout', ''),
        )
    failures.extend(request_payload_failures)
    warnings.extend(request_payload_warnings)
    failures.extend(request_text_failures)
    warnings.extend(request_text_warnings)
    failures.extend(stream_tool_call_state_failures)
    stream_tool_call_state_passed = (
        not bool(stream_tool_call_state_failures)
        if config.get('stream_tool_call_state_validation')
        else True
    )

    aawm_dynamic_injection_summary = None
    aawm_dynamic_injection_config = config.get('aawm_dynamic_injection')
    if langfuse_zero_trace_correlation_failure:
        aawm_dynamic_injection_summary = {'skipped': 'langfuse_zero_trace_correlation'}
    elif isinstance(aawm_dynamic_injection_config, dict):
        (
            aawm_dynamic_injection_summary,
            aawm_dynamic_injection_failures,
            aawm_dynamic_injection_warnings,
        ) = RA._validate_aawm_dynamic_injection(
            family=name,
            observations=raw_generation_observations,
            required_proc=aawm_dynamic_injection_config.get(
                'required_proc', 'get_agent_memories'
            ),
            required_context_keys=aawm_dynamic_injection_config.get(
                'required_context_keys'
            ),
            acceptable_statuses=aawm_dynamic_injection_config.get(
                'acceptable_statuses'
            ),
            warning_statuses=aawm_dynamic_injection_config.get('warning_statuses'),
            no_memory_required_substrings=aawm_dynamic_injection_config.get(
                'no_memory_required_substrings'
            ),
        )
        failures.extend(aawm_dynamic_injection_failures)
        warnings.extend(aawm_dynamic_injection_warnings)

    if use_failure_observability:
        span_observations, span_failures = {'skipped': 'expected_api_error'}, []
    elif langfuse_zero_trace_correlation_failure:
        span_observations, span_failures = {'skipped': 'langfuse_zero_trace_correlation'}, []
    else:
        _, span_observations, span_failures = RA._validate_span_observations(
            family=name,
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            trace_ids=filtered_trace_ids,
            start_time=started,
            required_names=config.get('required_span_names'),
        )
    failures.extend(span_failures)

    command_json_summary, command_json_failures = _validate_command_output_json(
        family=name,
        stdout=run['stdout'],
        checks=config.get('command_json_checks') or {},
    )
    failures.extend(command_json_failures)
    command_output_text_summary, command_output_text_failures = (
        _validate_command_text_checks(
            family=name,
            text=_extract_command_output_text(run["stdout"]),
            checks=config.get("command_output_text_checks") or {},
            label="command output",
        )
    )
    failures.extend(command_output_text_failures)
    command_stdout_text_summary, command_stdout_text_failures = (
        _validate_command_text_checks(
            family=name,
            text=run["stdout"],
            checks=config.get("command_stdout_text_checks") or {},
            label="command stdout",
        )
    )
    failures.extend(command_stdout_text_failures)
    command_stderr_text_summary, command_stderr_text_failures = (
        _validate_command_text_checks(
            family=name,
            text=run["stderr"],
            checks=config.get("command_stderr_text_checks") or {},
            label="command stderr",
        )
    )
    failures.extend(command_stderr_text_failures)
    codex_collaboration_summary, codex_collaboration_failures = (
        _validate_codex_collaboration_events(
            family=name,
            stdout=run["stdout"],
            checks=config.get("codex_collaboration_validation") or {},
        )
    )
    failures.extend(codex_collaboration_failures)
    empty_success_summary, empty_success_failures = (
        _validate_no_successful_empty_command_output(
            family=name,
            stdout=run['stdout'],
            stderr=run['stderr'],
            checks=config,
        )
    )
    failures.extend(empty_success_failures)

    if use_failure_observability:
        if config.get('session_history_validation'):
            session_history_summary, session_history_failures = _validate_session_history(
                family=name,
                session_id=command_session_id,
                checks=config.get('session_history_validation') or {},
            )
        else:
            session_history_summary = {
                'record': None,
                'records': [],
                'skipped': 'expected_api_error',
            }
            session_history_failures = []
    else:
        session_history_summary, session_history_failures = _validate_session_history(
            family=name,
            session_id=command_session_id,
            checks=config.get('session_history_validation') or {},
        )
    failures.extend(session_history_failures)
    session_history_passed = (
        not bool(session_history_failures)
        if config.get('session_history_validation')
        else True
    )

    # Fix 1: validate the normalized session-history identity independently
    # from raw Langfuse trace names.
    session_history_identity_summary, session_history_identity_failures = (
        _validate_session_history_identity(
            family=name,
            session_history_summary=session_history_summary,
            checks=config.get('session_history_identity') or {},
        )
    )
    failures.extend(session_history_identity_failures)
    if session_history_identity_failures:
        session_history_passed = False

    (
        rate_limit_observations_summary,
        rate_limit_observations_failures,
        rate_limit_observations_warnings,
    ) = _validate_rate_limit_observations(
        family=name,
        session_id=command_session_id,
        checks=config.get('rate_limit_observations_validation') or {},
    )
    failures.extend(rate_limit_observations_failures)
    warnings.extend(rate_limit_observations_warnings)
    if use_failure_observability:
        (
            provider_error_observations_summary,
            provider_error_observations_failures,
        ) = _validate_provider_error_observations(
            family=name,
            session_id=command_session_id,
            checks=config.get('provider_error_observations_validation') or {},
        )
    else:
        provider_error_observations_summary = {
            'record': None,
            'records': [],
        }
        if config.get('provider_error_observations_validation'):
            provider_error_observations_summary['skipped'] = (
                'expected_api_error_not_matched'
            )
        provider_error_observations_failures = []
    failures.extend(provider_error_observations_failures)
    tool_activity_summary, tool_activity_failures = _validate_tool_activity(
        family=name,
        session_id=command_session_id,
        checks=config.get('tool_activity_validation') or {},
    ) if config.get('tool_activity_validation') else ({'record': None, 'records': []}, [])
    failures.extend(tool_activity_failures)
    tool_activity_passed = not bool(tool_activity_failures) if config.get('tool_activity_validation') else True
    transcript_tool_use_summary, transcript_tool_use_failures = _validate_transcript_tool_use(
        family=name,
        session_id=command_session_id,
        checks=config.get('transcript_tool_use_validation') or {},
    ) if config.get('transcript_tool_use_validation') else ({'agents': []}, [])
    failures.extend(transcript_tool_use_failures)
    transcript_tool_use_passed = (
        not bool(transcript_tool_use_failures)
        if config.get('transcript_tool_use_validation')
        else True
    )
    bash_stdout_report_summary, bash_stdout_report_failures = (
        _validate_bash_stdout_report(
            family=name,
            stdout=run['stdout'],
            checks=config.get('bash_stdout_report_validation') or {},
            transcript_tool_use_summary=transcript_tool_use_summary,
        )
    )
    failures.extend(bash_stdout_report_failures)

    runtime_summary, runtime_failures = _validate_runtime_postcondition(
        family=name,
        litellm_base_url=litellm_base_url,
        checks=config.get('runtime_postconditions') or {},
    )
    failures.extend(runtime_failures)
    runtime_log_summary, runtime_log_failures, runtime_log_warnings = _validate_runtime_logs(
        family=name,
        started=started,
        checks=config.get('runtime_log_checks') or {},
        runtime_postconditions=runtime_summary,
        attribution_substrings=_runtime_log_attribution_substrings(
            family=name,
            config=config,
            session_id=command_session_id,
        ),
        require_evidence=cfg003_transactional,
    )
    failures.extend(runtime_log_failures)
    warnings.extend(runtime_log_warnings)
    failures, downgraded_warnings = _downgrade_configured_failures_to_warnings(
        failures=failures,
        config=config,
        command_json_summary=command_json_summary,
    )
    warnings.extend(downgraded_warnings)

    failure_context = config.get('failure_context')
    if isinstance(failure_context, str) and failure_context:
        failures = [f'{failure_context}: {failure}' for failure in failures]
        warnings = [f'{failure_context}: {warning}' for warning in warnings]

    unique_failures = sorted(set(failures))
    unique_warnings = sorted(set(warnings))
    warning_only = bool(config.get('warning_only'))
    hard_failures: list[str] = unique_failures
    soft_failures: list[str] = []
    (
        hard_failures,
        soft_failures,
        unique_warnings,
        runtime_log_summary,
    ) = _provider_unavailable_failure_soft_fail_result(
        failures=hard_failures,
        warnings=unique_warnings,
        config=config,
        runtime_logs=runtime_log_summary,
    )
    if warning_only and not soft_failures:
        hard_failures, soft_failures = _split_warning_only_failures(
            failures=hard_failures,
            config=config,
        )
    if warning_only and soft_failures:
        unique_warnings.extend(
            f'warning-only failure: {failure}' for failure in soft_failures
        )
        unique_warnings = sorted(set(unique_warnings))

    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "warning_only": warning_only,
        "command_attempts": command_attempts,
        "langfuse": {
            "required_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "expected_trace_user_ids_by_name": expected_trace_user_ids_by_name,
            "trace_user_ids_by_name": trace_user_ids_by_name_summary,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "lookup_error": lookup_error,
            "filtered_trace_ids": filtered_trace_ids,
            "command_session_id": command_session_id,
            "command_thread_id": command_thread_id,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_validation": generation_validation_summary,
            "generation_metadata": generation_metadata_summary,
            "request_payload_checks": request_payload_summary,
            "request_text_checks": request_text_summary,
            "stream_tool_call_state": stream_tool_call_state_summary,
            "aawm_dynamic_injection": aawm_dynamic_injection_summary,
            "span_observations": span_observations,
            "generation_observations": generation_observations,
        },
        "command_json": command_json_summary,
        "command_output_text": command_output_text_summary,
        "command_stdout_text": command_stdout_text_summary,
        "command_stderr_text": command_stderr_text_summary,
        "codex_collaboration": codex_collaboration_summary,
        "empty_success": empty_success_summary,
        "session_history": session_history_summary,
        "session_history_passed": session_history_passed,
        "session_history_identity": session_history_identity_summary,
        "rate_limit_observations": rate_limit_observations_summary,
        "provider_error_observations": provider_error_observations_summary,
        "tool_activity": tool_activity_summary,
        "transcript_tool_use": transcript_tool_use_summary,
        "bash_stdout_report": bash_stdout_report_summary,
        "tool_activity_passed": tool_activity_passed,
        "transcript_tool_use_passed": transcript_tool_use_passed,
        "stream_tool_call_state_passed": stream_tool_call_state_passed,
        "runtime_postconditions": runtime_summary,
        "runtime_logs": runtime_log_summary,
        "passed": not hard_failures,
        "failures": hard_failures,
        "soft_failures": soft_failures,
        "warnings": unique_warnings,
    }


def _session_history_rows_for_prompt_overhead_report(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    session_history = result.get('session_history')
    if not isinstance(session_history, dict):
        return []

    records = session_history.get('records')
    all_records = session_history.get('all_records')
    if isinstance(all_records, list):
        if isinstance(records, list) and records:
            return [row for row in records if isinstance(row, dict)]
        return [row for row in all_records if isinstance(row, dict)]

    record = session_history.get('record')
    if isinstance(record, dict) and record:
        return [record]
    if isinstance(records, list):
        return [row for row in records if isinstance(row, dict)]
    return []


def _prompt_report_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _prompt_report_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _prompt_report_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get('metadata')
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _prompt_report_value(*values: Any, default: str = 'unknown') -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _new_prompt_overhead_group(
    *,
    case_name: str,
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        'case_name': case_name,
        'client_name': _prompt_report_value(row.get('client_name')),
        'route_family': _prompt_report_value(
            metadata.get('prompt_overhead_route_family'),
            metadata.get('passthrough_route_family'),
            metadata.get('adapter_route_family'),
            metadata.get('route_family'),
        ),
        'counted_shape': _prompt_report_value(
            metadata.get('prompt_overhead_counted_shape')
        ),
        'litellm_environment': _prompt_report_value(row.get('litellm_environment')),
        'provider': _prompt_report_value(row.get('provider')),
        'model': _prompt_report_value(row.get('model')),
        'calls': 0,
        'estimated_calls': 0,
        'unestimated_calls': 0,
        'input_tokens': 0,
        'input_tokens_with_breakdown': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'response_cost_usd': 0.0,
        'response_cost_usd_with_breakdown': 0.0,
        'input_system_tokens_estimated': 0,
        'input_tool_advertisement_tokens_estimated': 0,
        'input_conversation_tokens_estimated': 0,
        'input_other_tokens_estimated': 0,
        'input_breakdown_residual_tokens': 0,
        'input_opaque_state_tokens_estimated': 0,
        'system_behavior_tokens_estimated': 0,
        'system_safety_tokens_estimated': 0,
        'system_instructional_tokens_estimated': 0,
        'system_unclassified_tokens_estimated': 0,
        'explicit_prompt_overhead_tokens_estimated': 0,
        'prompt_overhead_plus_other_tokens_estimated': 0,
        'explicit_prompt_overhead_cost_usd_estimated': 0.0,
        'prompt_overhead_plus_other_cost_usd_estimated': 0.0,
    }


def _prompt_overhead_group_key(
    *,
    case_name: str,
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    return (
        case_name,
        _prompt_report_value(row.get('client_name')),
        _prompt_report_value(
            metadata.get('prompt_overhead_route_family'),
            metadata.get('passthrough_route_family'),
            metadata.get('adapter_route_family'),
            metadata.get('route_family'),
        ),
        _prompt_report_value(metadata.get('prompt_overhead_counted_shape')),
        _prompt_report_value(row.get('litellm_environment')),
        _prompt_report_value(row.get('provider')),
        _prompt_report_value(row.get('model')),
    )


def _add_prompt_overhead_row(
    group: dict[str, Any],
    *,
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    input_tokens = _prompt_report_int(row.get('input_tokens'))
    output_tokens = _prompt_report_int(row.get('output_tokens'))
    total_tokens = _prompt_report_int(row.get('total_tokens'))
    response_cost_usd = _prompt_report_float(row.get('response_cost_usd'))

    system_tokens = _prompt_report_int(row.get('input_system_tokens_estimated'))
    tool_tokens = _prompt_report_int(
        row.get('input_tool_advertisement_tokens_estimated')
    )
    other_tokens = _prompt_report_int(row.get('input_other_tokens_estimated'))
    explicit_overhead_tokens = system_tokens + tool_tokens
    overhead_plus_other_tokens = explicit_overhead_tokens + other_tokens
    has_breakdown = (
        metadata.get('prompt_overhead_breakdown_source') == 'request_body_estimate'
    )

    group['calls'] += 1
    group['input_tokens'] += input_tokens
    group['output_tokens'] += output_tokens
    group['total_tokens'] += total_tokens
    group['response_cost_usd'] += response_cost_usd

    if has_breakdown:
        group['estimated_calls'] += 1
        group['input_tokens_with_breakdown'] += input_tokens
        group['response_cost_usd_with_breakdown'] += response_cost_usd
    else:
        group['unestimated_calls'] += 1

    for key in (
        'input_system_tokens_estimated',
        'input_tool_advertisement_tokens_estimated',
        'input_conversation_tokens_estimated',
        'input_other_tokens_estimated',
        'input_breakdown_residual_tokens',
        'system_behavior_tokens_estimated',
        'system_safety_tokens_estimated',
        'system_instructional_tokens_estimated',
        'system_unclassified_tokens_estimated',
    ):
        group[key] += _prompt_report_int(row.get(key))

    group['explicit_prompt_overhead_tokens_estimated'] += explicit_overhead_tokens
    group['prompt_overhead_plus_other_tokens_estimated'] += overhead_plus_other_tokens
    group['input_opaque_state_tokens_estimated'] += _prompt_report_int(
        metadata.get('usage_input_opaque_state_tokens_estimated')
    )
    if input_tokens > 0 and has_breakdown:
        group['explicit_prompt_overhead_cost_usd_estimated'] += (
            response_cost_usd * explicit_overhead_tokens / input_tokens
        )
        group['prompt_overhead_plus_other_cost_usd_estimated'] += (
            response_cost_usd * overhead_plus_other_tokens / input_tokens
        )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _finalize_prompt_overhead_group(group: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(group)
    finalized['breakdown_input_token_coverage_share'] = _ratio(
        finalized['input_tokens_with_breakdown'],
        finalized['input_tokens'],
    )
    finalized['explicit_prompt_overhead_input_share'] = _ratio(
        finalized['explicit_prompt_overhead_tokens_estimated'],
        finalized['input_tokens_with_breakdown'],
    )
    finalized['prompt_overhead_plus_other_input_share'] = _ratio(
        finalized['prompt_overhead_plus_other_tokens_estimated'],
        finalized['input_tokens_with_breakdown'],
    )
    finalized['opaque_state_to_input_token_ratio'] = _ratio(
        finalized['input_opaque_state_tokens_estimated'],
        finalized['input_tokens_with_breakdown'],
    )
    for key in (
        'response_cost_usd',
        'response_cost_usd_with_breakdown',
        'explicit_prompt_overhead_cost_usd_estimated',
        'prompt_overhead_plus_other_cost_usd_estimated',
    ):
        finalized[key] = round(float(finalized[key]), 12)
    return finalized


def _build_prompt_overhead_cost_share_report(
    results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    totals = _new_prompt_overhead_group(
        case_name='__all__',
        row={},
        metadata={},
    )
    totals['case_name'] = 'all'

    for case_name, result in results.items():
        for row in _session_history_rows_for_prompt_overhead_report(result):
            metadata = _prompt_report_metadata(row)
            key = _prompt_overhead_group_key(
                case_name=case_name,
                row=row,
                metadata=metadata,
            )
            group = groups.get(key)
            if group is None:
                group = _new_prompt_overhead_group(
                    case_name=case_name,
                    row=row,
                    metadata=metadata,
                )
                groups[key] = group
            _add_prompt_overhead_row(group, row=row, metadata=metadata)
            _add_prompt_overhead_row(totals, row=row, metadata=metadata)

    finalized_groups = [_finalize_prompt_overhead_group(group) for group in groups.values()]
    finalized_groups.sort(
        key=lambda group: (
            -float(group['prompt_overhead_plus_other_cost_usd_estimated']),
            -int(group['prompt_overhead_plus_other_tokens_estimated']),
            str(group['case_name']),
            str(group['provider']),
            str(group['model']),
        )
    )

    return {
        'cost_allocation_basis': (
            'estimated from response_cost_usd weighted by each row prompt-overhead '
            'input-token share; session_history does not yet store exact input cost; '
            'opaque response-state tokens are reported separately and not allocated '
            'as prompt-overhead cost'
        ),
        'group_by': [
            'case_name',
            'client_name',
            'route_family',
            'counted_shape',
            'litellm_environment',
            'provider',
            'model',
        ],
        'totals': _finalize_prompt_overhead_group(totals),
        'groups': finalized_groups,
    }


def _collect_rows_by_key(container: Any, key: str) -> list[dict[str, Any]]:
    values = container.get(key) if isinstance(container, dict) else None
    if not isinstance(values, list):
        return []
    return [row for row in values if isinstance(row, dict)]


def _case_result_check_passed(
    *,
    case_config: dict[str, Any],
    case_result: dict[str, Any],
    config_key: str,
    result_key: str,
) -> bool | None:
    if not case_config.get(config_key):
        return None
    if result_key in case_result:
        return bool(case_result.get(result_key))
    return bool(case_result.get('passed'))


def _case_command_connectivity_passed(case_result: dict[str, Any]) -> bool | None:
    if case_result.get('skipped') is True:
        return None
    exit_code = case_result.get('exit_code')
    if exit_code is not None and exit_code != 0:
        return False

    attempts = [
        attempt
        for attempt in case_result.get('command_attempts') or []
        if isinstance(attempt, dict)
    ]
    if attempts:
        final_attempt = attempts[-1]
        if isinstance(final_attempt.get('api_error_status'), int):
            return False
        if final_attempt.get('is_error') is True:
            return False

    parsed_stdout = _parse_command_output_json(str(case_result.get('stdout') or ''))
    if isinstance(parsed_stdout, dict):
        if parsed_stdout.get('is_error') is True:
            return False
        status_code = parsed_stdout.get('status_code')
        if isinstance(status_code, int) and status_code >= 400:
            return False

    if exit_code is None and not attempts and parsed_stdout is None:
        return None
    return True


def _tool_activity_requires_arguments(case_config: dict[str, Any]) -> bool:
    for expected_row in _collect_rows_by_key(
        case_config.get('tool_activity_validation') or {},
        key='expected_rows',
    ):
        for key in ('arguments_required_substring', 'arguments_required_substrings'):
            value = expected_row.get(key)
            if isinstance(value, str) and value:
                return True
            if isinstance(value, list) and any(
                isinstance(item, str) and item for item in value
            ):
                return True
    return False


def _transcript_tool_use_records(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    transcript_tool_use = case_result.get('transcript_tool_use')
    if not isinstance(transcript_tool_use, dict):
        return []
    records: list[dict[str, Any]] = []
    for agent in transcript_tool_use.get('agents') or []:
        if not isinstance(agent, dict):
            continue
        records.extend(
            record
            for record in agent.get('records') or []
            if isinstance(record, dict)
        )
    return records


def _case_tool_use_ids_passed(
    *,
    case_config: dict[str, Any],
    case_result: dict[str, Any],
    transcript_passed: bool,
) -> bool | None:
    if not case_config.get('transcript_tool_use_validation'):
        return None
    if not transcript_passed:
        return False
    records = _transcript_tool_use_records(case_result)
    if not records:
        return False
    return all(bool(record.get('tool_use_id')) for record in records)


def _case_tool_result_replay_passed(
    *,
    case_config: dict[str, Any],
    case_result: dict[str, Any],
    transcript_passed: bool,
) -> bool | None:
    if not _case_requires_multi_turn_tool_results(case_config):
        return None
    if not transcript_passed:
        return False
    records = _transcript_tool_use_records(case_result)
    if len(records) < 2:
        return False
    return all(bool(record.get('tool_result_line')) for record in records[:-1])


def _first_string_value_from_rows(
    rows: list[dict[str, Any]],
    key: str,
) -> str | None:
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_provider_model_from_expected_rows(
    case_config: dict[str, Any],
) -> tuple[str | None, str | None]:
    session_history_validation = case_config.get('session_history_validation')
    if not isinstance(session_history_validation, dict):
        session_history_validation = {}

    provider = session_history_validation.get('expected_provider')
    model = session_history_validation.get('expected_model')
    if not isinstance(provider, str) or not provider.strip():
        provider = _first_string_value_from_rows(
            _collect_rows_by_key(session_history_validation, key='expected_rows'),
            'provider',
        )
    if not isinstance(model, str) or not model.strip():
        model = _first_string_value_from_rows(
            _collect_rows_by_key(session_history_validation, key='expected_rows'),
            'model',
        )

    tool_activity_rows = _collect_rows_by_key(
        case_config.get('tool_activity_validation') or {},
        key='expected_rows',
    )
    if not isinstance(provider, str) or not provider.strip():
        provider = _first_string_value_from_rows(tool_activity_rows, 'provider')
    if not isinstance(model, str) or not model.strip():
        model = _first_string_value_from_rows(tool_activity_rows, 'model')

    return (
        provider.strip() if isinstance(provider, str) and provider.strip() else None,
        model.strip() if isinstance(model, str) and model.strip() else None,
    )


def _case_result_session_history_record(
    case_result: dict[str, Any],
) -> dict[str, Any]:
    session_history = case_result.get('session_history')
    if not isinstance(session_history, dict):
        return {}
    record = session_history.get('record')
    if isinstance(record, dict):
        return record
    records = session_history.get('records')
    if isinstance(records, list):
        for candidate in records:
            if isinstance(candidate, dict):
                return candidate
    return {}


def _validate_session_history_identity(
    *,
    family: str,
    session_history_summary: dict[str, Any],
    checks: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the normalized session-history identity from the DB row's own
    metadata.trace_name field.

    Fix 1: This is independent from raw Langfuse trace names
    (required_trace_names validates those separately).  The observed alias
    session_history record carries metadata.trace_name == "orchestrator"
    regardless of which raw Langfuse trace name the row correlates with.
    """
    if not isinstance(checks, dict) or not checks:
        return {'skipped': 'not_configured'}, []

    expected_trace_name = checks.get('expected_metadata_trace_name')
    if not isinstance(expected_trace_name, str) or not expected_trace_name.strip():
        return {'skipped': 'no_expected_metadata_trace_name'}, []

    expected_trace_name = expected_trace_name.strip()
    sh_record = _case_result_session_history_record(
        {'session_history': session_history_summary}
    )
    sh_metadata = sh_record.get('metadata')
    actual_trace_name = (
        sh_metadata.get('trace_name')
        if isinstance(sh_metadata, dict)
        else None
    )

    if actual_trace_name == expected_trace_name:
        return {
            'validated': True,
            'expected_metadata_trace_name': expected_trace_name,
            'actual_metadata_trace_name': actual_trace_name,
        }, []

    return {
        'validated': False,
        'expected_metadata_trace_name': expected_trace_name,
        'actual_metadata_trace_name': actual_trace_name,
    }, [
        f'{family} session_history_identity: metadata.trace_name '
        f'mismatch: expected {expected_trace_name!r}, '
        f'got {actual_trace_name!r}'
    ]


def _candidate_provider_model_from_case_result(
    case_result: dict[str, Any],
) -> tuple[str | None, str | None]:
    record = _case_result_session_history_record(case_result)
    provider = record.get('provider')
    model = record.get('model')
    return (
        provider.strip() if isinstance(provider, str) and provider.strip() else None,
        model.strip() if isinstance(model, str) and model.strip() else None,
    )


def _candidate_model_from_http_request(case_config: dict[str, Any]) -> str | None:
    http_request = case_config.get('http_request')
    if not isinstance(http_request, dict):
        return None
    json_payload = http_request.get('json')
    if not isinstance(json_payload, dict):
        return None
    model = json_payload.get('model')
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _candidate_provider_from_trace_tags(case_config: dict[str, Any]) -> str | None:
    for tag in case_config.get('required_trace_tags', []):
        if not isinstance(tag, str):
            continue
        if tag.startswith('anthropic-adapter-target:'):
            target = tag.removeprefix('anthropic-adapter-target:').strip()
            candidate = target.split(':', 1)[0]
            if candidate:
                return candidate
        if tag.startswith('provider:'):
            value = tag.split(':', 1)[1].strip()
            if value:
                return value
    return None


def _candidate_model_provider_from_trace_tags(
    case_config: dict[str, Any],
) -> tuple[str | None, str | None]:
    if not isinstance(case_config.get('required_trace_tags'), list):
        return None, None
    for tag in case_config['required_trace_tags']:
        if not isinstance(tag, str):
            continue
        if tag.startswith('anthropic-adapter-model:'):
            value = tag.split(':', 1)[1].strip()
            if value:
                return value, 'anthropic'
        if tag.startswith('openai-adapter-model:'):
            value = tag.split(':', 1)[1].strip()
            if value:
                return value, 'openai'
    return None, None


def _infer_provider_from_model(model: str | None) -> str | None:
    if model is None:
        return None
    if '/' in model:
        return model.split('/', 1)[0]
    if model.startswith(('claude-', 'anthropic-')):
        return 'anthropic'
    return None


def _extract_case_provider_and_model(
    case_config: dict[str, Any],
    case_result: dict[str, Any],
) -> tuple[str | None, str | None]:
    provider, model = _candidate_provider_model_from_expected_rows(case_config)
    if provider is None or model is None:
        result_provider, result_model = _candidate_provider_model_from_case_result(
            case_result
        )
        provider = provider or result_provider
        model = model or result_model
    if model is None:
        model = _candidate_model_from_http_request(case_config)
    if provider is None:
        provider = _candidate_provider_from_trace_tags(case_config)
    if provider is None:
        provider = _infer_provider_from_model(model)
    if model is None:
        tag_model, tag_provider = _candidate_model_provider_from_trace_tags(case_config)
        model = tag_model
        provider = provider or tag_provider

    return provider or None, model or None


def _extract_case_route_family(
    *,
    case_config: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    tags = case_config.get('required_trace_tags')
    if isinstance(tags, list):
        for tag in tags:
            if (
                isinstance(tag, str)
                and tag.startswith('route:')
                and tag != 'route:anthropic_messages'
            ):
                route_family = tag.removeprefix('route:').strip()
                if route_family:
                    return route_family

    observations = result.get('langfuse', {}).get('generation_observations', [])
    for observation in observations:
        metadata = observation.get('metadata')
        if not isinstance(metadata, dict):
            continue
        for metadata_key in (
            'anthropic_auto_agent_selected_route_family',
            'prompt_overhead_route_family',
            'passthrough_route_family',
            'adapter_route_family',
            'route_family',
        ):
            value = metadata.get(metadata_key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    session_history_record = _case_result_session_history_record(result)
    metadata = session_history_record.get('metadata')
    if isinstance(metadata, dict):
        for metadata_key in (
            'anthropic_auto_agent_selected_route_family',
            'prompt_overhead_route_family',
            'passthrough_route_family',
            'adapter_route_family',
            'route_family',
        ):
            value = metadata.get(metadata_key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith('route:'):
                route_family = tag.removeprefix('route:').strip()
                if route_family:
                    return route_family

    allowed_routes = case_config.get('allowed_generation_routes')
    if isinstance(allowed_routes, list):
        for route in allowed_routes:
            if not isinstance(route, str):
                continue
            candidates = RA._route_family_candidates_for_request_route(route)
            for candidate in sorted(candidates):
                if candidate == 'anthropic_messages':
                    continue
                return candidate
            if candidates:
                return next(iter(sorted(candidates)))
    return None


def _extract_case_tool_mode(case_config: dict[str, Any]) -> str:
    transcript_validation = case_config.get('transcript_tool_use_validation')
    if isinstance(transcript_validation, dict):
        expected_agents = transcript_validation.get('expected_agents')
        if isinstance(expected_agents, list):
            for expected_agent in expected_agents:
                if not isinstance(expected_agent, dict):
                    continue
                if expected_agent.get('require_tool_result_before_next_tool_use') is True:
                    return 'sequential'
                minimum_tools_in_single_message = (
                    expected_agent.get('minimum_tools_in_single_assistant_message')
                )
                if (
                    isinstance(minimum_tools_in_single_message, int)
                    and minimum_tools_in_single_message > 1
                ):
                    return 'parallel'
                max_tools = expected_agent.get('maximum_tool_uses_per_assistant_message')
                if isinstance(max_tools, int) and max_tools > 1:
                    return 'parallel'

    required_rows = _collect_rows_by_key(
        case_config.get('tool_activity_validation') or {},
        'expected_rows',
    )
    if not required_rows:
        return 'unknown'

    distinct_tool_names = {
        row.get('tool_name')
        for row in required_rows
        if isinstance(row.get('tool_name'), str)
    }
    if len(distinct_tool_names) > 1:
        return 'parallel'
    return 'single'


def _case_requires_multi_turn_tool_results(case_config: dict[str, Any]) -> bool:
    transcript_validation = case_config.get('transcript_tool_use_validation')
    if not isinstance(transcript_validation, dict):
        return False
    for expected_agent in _collect_rows_by_key(transcript_validation, 'expected_agents'):
        if expected_agent.get('require_tool_result_before_next_tool_use') is True:
            return True
    return False


def _declared_candidates_for_case(case_config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = case_config.get('verification_declared_candidates')
    if not isinstance(candidates, list):
        return []

    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        row: dict[str, Any] = {}
        for key in ('candidate_order', 'provider', 'model', 'route_family'):
            value = candidate.get(key)
            if value is not None:
                row[key] = value
        if row:
            normalized.append(row)
    return normalized


def _verification_status_for_case(result: dict[str, Any]) -> str:
    if result.get('skipped') is True:
        # fail_on_skip records skipped=True with failures; treat that as failed.
        if result.get('failures') or result.get('passed') is False:
            return 'failed'
        return 'skipped'
    if not result.get('passed'):
        return 'failed'
    if result.get('warning_only'):
        return 'warning_only'
    if result.get('soft_failures'):
        return 'passed_with_soft_failures'
    return 'passed'


def _build_case_verification_matrix_row(
    *,
    alias: str,
    case_name: str | None = None,
    candidate_order: int,
    case_config: dict[str, Any],
    case_result: dict[str, Any],
) -> dict[str, Any]:
    provider, model = _extract_case_provider_and_model(case_config, case_result)
    route_family = _extract_case_route_family(
        case_config=case_config,
        result=case_result,
    )
    tool_mode = _extract_case_tool_mode(case_config)
    multi_turn_required = _case_requires_multi_turn_tool_results(case_config)

    transcript_passed = bool(case_result.get('transcript_tool_use_passed', True))
    tool_activity_passed = bool(case_result.get('tool_activity_passed', True))
    tool_bearing_configured = bool(
        case_config.get('tool_activity_validation')
        or case_config.get('transcript_tool_use_validation')
    )
    if tool_bearing_configured:
        tool_bearing_passed = tool_activity_passed and transcript_passed
    else:
        tool_bearing_passed = None

    if not multi_turn_required:
        multi_turn_tool_result_passed = None
    elif not isinstance(case_config.get('transcript_tool_use_validation'), dict):
        multi_turn_tool_result_passed = False
    else:
        multi_turn_tool_result_passed = transcript_passed

    session_history_passed = _case_result_check_passed(
        case_config=case_config,
        case_result=case_result,
        config_key='session_history_validation',
        result_key='session_history_passed',
    )
    stream_tool_call_state_passed = _case_result_check_passed(
        case_config=case_config,
        case_result=case_result,
        config_key='stream_tool_call_state_validation',
        result_key='stream_tool_call_state_passed',
    )
    required_tool_arguments_passed = (
        tool_activity_passed if _tool_activity_requires_arguments(case_config) else None
    )
    tool_use_ids_passed = _case_tool_use_ids_passed(
        case_config=case_config,
        case_result=case_result,
        transcript_passed=transcript_passed,
    )
    tool_result_replay_passed = _case_tool_result_replay_passed(
        case_config=case_config,
        case_result=case_result,
        transcript_passed=transcript_passed,
    )

    langfuse = case_result.get('langfuse')
    if not isinstance(langfuse, dict):
        langfuse = {}
    command_attempts = [
        attempt
        for attempt in case_result.get('command_attempts') or []
        if isinstance(attempt, dict)
    ]

    return {
        'case_name': case_name or alias,
        'alias': alias,
        'candidate_order': candidate_order,
        'candidate_label': case_config.get('verification_candidate_label'),
        'declared_candidates': _declared_candidates_for_case(case_config),
        'provider': provider,
        'model': model,
        'route_family': route_family,
        'connectivity_passed': _case_command_connectivity_passed(case_result),
        'session_history_metadata_passed': session_history_passed,
        'stream_tool_call_state_passed': stream_tool_call_state_passed,
        'tool_mode': tool_mode,
        'tool_call_emission_passed': tool_bearing_passed,
        'tool_bearing_passed': tool_bearing_passed,
        'required_tool_arguments_passed': required_tool_arguments_passed,
        'tool_use_ids_passed': tool_use_ids_passed,
        'tool_result_replay_passed': tool_result_replay_passed,
        'multi_turn_tool_result_passed': multi_turn_tool_result_passed,
        'status': _verification_status_for_case(case_result),
        'references': {
            'command_session_id': langfuse.get('command_session_id'),
            'command_attempts': command_attempts,
            'trace_ids': langfuse.get('trace_ids') or [],
            'filtered_trace_ids': langfuse.get('filtered_trace_ids') or [],
            'actual_trace_names': langfuse.get('actual_trace_names') or [],
            'actual_user_ids': langfuse.get('actual_user_ids') or [],
        },
    }


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    skipped_cases: list[str] = []
    for family, result in results.items():
        if result.get('skipped') is True:
            skipped_cases.append(family)
        for failure in result.get('failures', []):
            failures.append(f'{family}: {failure}')
        for warning in result.get('warnings', []):
            warnings.append(f'{family}: {warning}')
    skipped_cases = sorted(skipped_cases)
    return {
        'passed': not failures,
        'failures': failures,
        'warnings': warnings,
        'skipped_count': len(skipped_cases),
        'skipped_cases': skipped_cases,
        'prompt_overhead_cost_share': _build_prompt_overhead_cost_share_report(
            results
        ),
    }


def _warning_only_error_result(
    family: str,
    exc: Exception,
    config: dict[str, Any],
) -> dict[str, Any]:
    if _is_warning_only_hard_exception(exc=exc, config=config):
        return RA._family_error_result(family, exc)

    base = RA._family_error_result(family, exc)
    failures = list(base.get('failures', []))
    return {
        **base,
        'warning_only': True,
        'passed': True,
        'failures': [],
        'soft_failures': failures,
        'warnings': [f'warning-only failure: {failure}' for failure in failures],
    }


def _provider_unavailable_timeout_error_result(
    family: str,
    exc: Exception,
    config: dict[str, Any],
    *,
    started: Any,
) -> dict[str, Any] | None:
    soft_timeout_config = config.get('soft_fail_timeout_runtime_log_check')
    if not isinstance(soft_timeout_config, dict):
        return None
    if not isinstance(exc, subprocess.TimeoutExpired):
        return None

    required_substrings = [
        value
        for value in soft_timeout_config.get('required_substrings', [])
        if isinstance(value, str) and value
    ]
    if not required_substrings:
        return None

    runtime_postconditions = dict(config.get('runtime_postconditions') or {})
    runtime_logs, log_text = _read_runtime_logs_since(
        started=started,
        until=RA._utcnow(),
        checks={
            'docker_container_name': (
                soft_timeout_config.get('docker_container_name')
                or runtime_postconditions.get('docker_container_name')
            ),
            'tail_lines': soft_timeout_config.get('tail_lines', 800),
        },
        runtime_postconditions=runtime_postconditions,
    )
    matched_substrings = [
        substring for substring in required_substrings if substring in log_text
    ]
    runtime_logs['required_substrings'] = required_substrings
    runtime_logs['matched_required_substrings'] = matched_substrings
    if runtime_logs.get('docker_logs_exit_code') != 0:
        return None
    if len(matched_substrings) != len(required_substrings):
        return None

    base = RA._family_error_result(family, exc)
    failures = list(base.get('failures', []))
    return {
        **base,
        'passed': True,
        'failures': [],
        'soft_failures': failures,
        'warnings': [
            f'provider-unavailable timeout soft-fail: {failure}'
            for failure in failures
        ],
        'runtime_logs': runtime_logs,
    }


def _is_provider_unavailable_soft_failable_failure(failure: str) -> bool:
    """Only connectivity/timeout-class failures may be provider-unavailable soft-fails."""
    if not isinstance(failure, str) or not failure:
        return False
    if 'runtime logs contained forbidden substring' in failure:
        return False
    return any(
        marker in failure
        for marker in PROVIDER_UNAVAILABLE_SOFT_FAILABLE_FAILURE_MARKERS
    )


def _provider_unavailable_failure_soft_fail_result(
    *,
    failures: list[str],
    warnings: list[str],
    config: dict[str, Any],
    runtime_logs: dict[str, Any],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    soft_timeout_config = config.get('soft_fail_timeout_runtime_log_check')
    if not isinstance(soft_timeout_config, dict) or not failures:
        return failures, [], warnings, runtime_logs
    if any('runtime logs contained forbidden substring' in failure for failure in failures):
        return failures, [], warnings, runtime_logs

    required_substrings = [
        value
        for value in soft_timeout_config.get('required_substrings', [])
        if isinstance(value, str) and value
    ]
    if not required_substrings:
        return failures, [], warnings, runtime_logs

    log_text = runtime_logs.get('_log_text')
    if not isinstance(log_text, str) or not log_text:
        return failures, [], warnings, runtime_logs

    matched_substrings = [
        substring for substring in required_substrings if substring in log_text
    ]
    runtime_logs = {
        **runtime_logs,
        'required_soft_fail_substrings': required_substrings,
        'matched_soft_fail_substrings': matched_substrings,
    }
    if len(matched_substrings) != len(required_substrings):
        return failures, [], warnings, runtime_logs

    soft_failures = [
        failure
        for failure in failures
        if _is_provider_unavailable_soft_failable_failure(failure)
    ]
    if not soft_failures:
        return failures, [], warnings, runtime_logs
    remaining_failures = [
        failure for failure in failures if failure not in soft_failures
    ]
    soft_warnings = [
        f'provider-unavailable soft-fail: {failure}' for failure in soft_failures
    ]
    return (
        remaining_failures,
        soft_failures,
        sorted(set([*warnings, *soft_warnings])),
        runtime_logs,
    )


def _write_artifact(path: pathlib.Path, artifact: dict[str, Any]) -> None:
    sanitized = RA._redact_sensitive_artifact_fields(artifact)
    path.write_text(json.dumps(sanitized, indent=2) + '\n', encoding='utf-8')


def _parse_selected_cases(
    raw: str | None,
    available: list[str],
    *,
    default_excluded_cases: list[str] | None = None,
) -> list[str]:
    preferred_order = [
        'claude_adapter_gpt54',
        'claude_adapter_gpt55',
        'claude_adapter_gpt54_mini',
        'claude_adapter_ctx_marker',
        'claude_adapter_ctx_marker_escaped',
        'claude_adapter_codex_tool_activity',
        'claude_adapter_peeromega_fanout',
    ]
    if not raw:
        excluded = {
            value
            for value in (default_excluded_cases or [])
            if isinstance(value, str) and value
        }
        default_available = [name for name in available if name not in excluded]
        priority = {name: index for index, name in enumerate(preferred_order)}
        return sorted(
            default_available,
            key=lambda name: (
                priority.get(name, len(preferred_order)),
                available.index(name),
            ),
        )
    requested = [value.strip() for value in raw.split(',') if value.strip()]
    invalid = [value for value in requested if value not in available]
    if invalid:
        raise SystemExit(f'Unknown adapter case(s): {", ".join(invalid)}')
    return requested


def _missing_env_case_result(
    *,
    case_config: dict[str, Any],
    suite_config: dict[str, Any],
    missing_required_env: list[str],
) -> dict[str, Any]:
    missing_env_message = (
        f'missing required env: {", ".join(sorted(missing_required_env))}'
    )
    fail_on_skip = bool(
        case_config.get('fail_on_skip')
        if case_config.get('fail_on_skip') is not None
        else suite_config.get('fail_on_skip')
    )
    if fail_on_skip:
        return {
            'passed': False,
            'skipped': True,
            'failures': [missing_env_message],
            'soft_failures': [],
            'warnings': [],
            'skip_reason': missing_env_message,
        }
    return {
        'passed': True,
        'skipped': True,
        'failures': [],
        'soft_failures': [],
        'warnings': [missing_env_message],
        'skip_reason': missing_env_message,
    }


def _record_case_artifact_result(
    *,
    artifact: dict[str, Any],
    artifact_path: pathlib.Path,
    case_name: str,
    case_config: dict[str, Any],
    case_result: dict[str, Any],
    selected_case_order: int,
) -> None:
    artifact['results'][case_name] = case_result
    artifact['summary'] = _build_summary(artifact['results'])
    verification_alias = str(case_config.get('verification_alias') or case_name)
    matrix_candidate_order = int(
        case_config.get('verification_candidate_order', selected_case_order)
    )
    artifact['verification_matrix'] = [
        row
        for row in artifact['verification_matrix']
        if row.get('case_name') != case_name
    ] + [
        _build_case_verification_matrix_row(
            alias=verification_alias,
            case_name=case_name,
            candidate_order=matrix_candidate_order,
            case_config=case_config,
            case_result=case_result,
        )
    ]
    _write_artifact(artifact_path, artifact)
    _emit_stderr(
        f"[done] {case_name} passed={case_result.get('passed')} "
        f"skipped={case_result.get('skipped', False)} "
        f"failures={len(case_result.get('failures', []))} "
        f"warnings={len(case_result.get('warnings', []))}",
        flush=True,
    )


def _resolve_main_credentials(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    profile: dict[str, str] | None = None,
) -> tuple[str, str, str, str, str] | int:
    public_key_env = config.get('langfuse_public_key_env', 'LANGFUSE_PUBLIC_KEY')
    secret_key_env = config.get('langfuse_secret_key_env', 'LANGFUSE_SECRET_KEY')
    public_key = ''
    secret_key = ''
    # Fix 6: explicit langfuse_credential_container takes precedence over
    # docker_container_name, preserving backward-compatible fallback.
    container_name = str(
        (profile or {}).get('langfuse_credential_container')
        or (profile or {}).get('docker_container_name')
        or ''
    )
    pk_container_env = (profile or {}).get('langfuse_public_key_container_env', '')
    sk_container_env = (profile or {}).get('langfuse_secret_key_container_env', '')
    container_owned_credentials = bool(pk_container_env or sk_container_env)
    if container_owned_credentials:
        if not container_name or not pk_container_env or not sk_container_env:
            _emit_stderr('Missing target-owned Langfuse credential configuration')
            return 2
        public_key = _resolve_container_env_value(container_name, pk_container_env) or ''
        secret_key = _resolve_container_env_value(container_name, sk_container_env) or ''
        if not public_key or not secret_key:
            _emit_stderr('Could not retrieve target-owned Langfuse credentials')
            return 2
    else:
        public_key = os.environ.get(public_key_env, '')
        secret_key = os.environ.get(secret_key_env, '')
    query_url = (
        args.langfuse_query_url
        or os.environ.get('LANGFUSE_QUERY_URL')
        or config.get('langfuse_query_url', 'http://127.0.0.1:3000')
    )
    if not public_key or not secret_key:
        _emit_stderr(
            f'Missing Langfuse credentials in env vars {public_key_env}/{secret_key_env}'
        )
        return 2
    return public_key, secret_key, query_url, public_key_env, secret_key_env


def _build_initial_artifact(
    *,
    config: dict[str, Any],
    profile: dict[str, str],
    target: str,
    litellm_base_url: str,
    query_url: str,
    public_key_env: str,
    secret_key_env: str,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        'suite_version': config.get('suite_version', 1),
        'timestamp': RA._isoformat(RA._utcnow()),
        'git_commit': RA._git_value('rev-parse', 'HEAD'),
        'git_branch': RA._git_value('branch', '--show-current'),
        'environment': {
            'target_profile': target,
            'litellm_base_url': litellm_base_url,
            'anthropic_base_url': profile['anthropic_base_url'],
            'langfuse_query_url': query_url,
            'langfuse_public_key_env': public_key_env,
            'langfuse_secret_key_env': secret_key_env,
            'expected_trace_environment': profile['expected_trace_environment'],
            'docker_container_name': profile['docker_container_name'],
            'docker_container_status': _docker_status_for_container(
                profile['docker_container_name']
            ),
        },
        'results': {},
        'verification_matrix': [],
        'summary': {},
    }
    artifact['summary'] = _build_summary(artifact['results'])
    return artifact


def _run_selected_case(
    *,
    case_name: str,
    case_config: dict[str, Any],
    suite_config: dict[str, Any],
    query_url: str,
    public_key: str,
    secret_key: str,
    litellm_base_url: str,
    cfg003_transactional: bool = False,
) -> dict[str, Any]:
    agentic_contract, contract_failures = (
        _validate_moonshot_anthropic_agentic_contract(
            family=case_name,
            config=case_config,
        )
    )
    if contract_failures:
        return {
            'passed': False,
            'skipped': False,
            'failures': contract_failures,
            'soft_failures': [],
            'warnings': [],
            'agentic_contract': agentic_contract,
        }

    case_started = RA._utcnow()
    missing_required_env = _missing_required_env(case_config)
    if missing_required_env:
        return _missing_env_case_result(
            case_config=case_config,
            suite_config=suite_config,
            missing_required_env=missing_required_env,
        )
    try:
        result = _validate_case(
            case_name,
            case_config,
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            litellm_base_url=litellm_base_url,
            cfg003_transactional=(
                cfg003_transactional
                and _cfg003_case_requires_runtime_evidence(case_name, case_config)
            ),
        )
        result['agentic_contract'] = agentic_contract
        return result
    except Exception as exc:
        provider_unavailable_timeout = _provider_unavailable_timeout_error_result(
            case_name,
            exc,
            case_config,
            started=case_started,
        )
        if provider_unavailable_timeout is not None:
            return provider_unavailable_timeout
        if bool(case_config.get("warning_only")):
            return _warning_only_error_result(case_name, exc, case_config)
        return RA._family_error_result(case_name, exc)


# ---------------------------------------------------------------------------
# CFG-003: Transactional live priority-swap refresh orchestration
# ---------------------------------------------------------------------------

_CFG003_INVALID_YAML = "aliases:\n  - name: read\n    candidates: not_a_list\n"

_CFG003_CODEX_PROOF_CASE = (
    "native_openai_passthrough_responses_codex_read_alias_collaboration"
)
_CFG003_CLAUDE_PROOF_CASE = (
    "claude_adapter_read_alias_child_parallel_read_tools"
)

# Targets that may run the transactional refresh test (dev only).
_CFG003_ALLOWED_TARGETS = frozenset({"dev"})

# Canonical dev profile values that MUST all match for CFG-003 to proceed.
_CFG003_CANONICAL_DEV_PROFILE = {
    "litellm_base_url": "http://127.0.0.1:4001",
    "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
    "docker_container_name": "litellm-dev",
    "expected_trace_environment": "dev",
}


def _cfg003_validate_canonical_dev_profile(
    *,
    target: str,
    profile: dict[str, str],
) -> tuple[bool, list[str]]:
    """Validate that the resolved target is EXACTLY the canonical checked-in
    dev profile.  CLI overrides or a dev-labelled profile pointing at prod,
    port 4000, or aawm-litellm must fail closed.

    Returns (valid, failures).
    """
    failures: list[str] = []
    if target != "dev":
        failures.append(
            f"cfg003 canonical dev gate: target must be 'dev', got {target!r}"
        )
    for key, expected in sorted(_CFG003_CANONICAL_DEV_PROFILE.items()):
        actual = str(profile.get(key, "")).rstrip("/")
        if actual != expected:
            failures.append(
                f"cfg003 canonical dev gate: {key} must be {expected!r}, "
                f"got {actual!r}"
            )
    # Explicit rejection of known production signatures even if target is dev.
    litellm_url = str(profile.get("litellm_base_url", "")).rstrip("/")
    if ":4000" in litellm_url:
        failures.append(
            "cfg003 canonical dev gate: litellm_base_url contains port 4000 "
            "(production signature)"
        )
    container = str(profile.get("docker_container_name", ""))
    if container == "aawm-litellm":
        failures.append(
            "cfg003 canonical dev gate: docker_container_name is 'aawm-litellm' "
            "(production container)"
        )
    return not failures, failures




def _cfg003_raw_config_canonical_preflight(
    *,
    config_path: pathlib.Path,
    target_override: str | None,
    litellm_base_url_override: str | None,
    anthropic_base_url_override: str | None,
    docker_container_name_override: str | None,
    expected_trace_environment_override: str | None,
) -> tuple[bool, list[str]]:
    """Finding 3 (round 8): Canonical transactional target rejection BEFORE
    dotenv loading.  Reads only the raw config JSON and CLI overrides to build
    the target profile, then validates it against the canonical dev profile.

    This prevents an invalid transactional target from triggering dotenv
    loading or credential resolution.  The resolved-profile validation after
    profile resolution is retained as a second gate.

    Returns (valid, failures).
    """
    try:
        raw_config = RA._load_json(config_path)
    except Exception as exc:  # noqa: BLE001
        return False, [f"cfg003 raw preflight: cannot load config: {exc}"]
    target = target_override or str(raw_config.get("default_target_profile") or "dev")
    profile = _target_profile_settings(
        config=raw_config,
        target=target,
        litellm_base_url=litellm_base_url_override,
        anthropic_base_url=anthropic_base_url_override,
        docker_container_name=docker_container_name_override,
        expected_trace_environment=expected_trace_environment_override,
    )
    return _cfg003_validate_canonical_dev_profile(target=target, profile=profile)

def _cfg003_readiness_check(
    litellm_base_url: str,
    *,
    expected_hash: str,
    expected_version: str,
    phase_label: str,
) -> tuple[bool, list[str]]:
    """Finding 2 (round 7): Authoritative readiness hash/version check.

    Queries /health/readiness and requires the active config_hash and
    config_version to match the expected values exactly.  Returns
    (ok, failures).
    """
    readiness_url = f"{litellm_base_url}{RA._HEALTH_READINESS_PATH}"
    try:
        status, body = RA._http_get_json_plain(readiness_url, timeout=15.0)
    except Exception as exc:  # noqa: BLE001
        return False, [f"{phase_label} readiness check exception: {exc}"]
    if status != 200:
        return False, [f"{phase_label} readiness unavailable: status={status}"]
    alias_section = {}
    if isinstance(body, dict):
        alias_section = body.get("aawm_alias_config") or {}
    actual_hash = str(alias_section.get("config_hash", ""))
    actual_version = str(alias_section.get("config_version", ""))
    failures: list[str] = []
    if actual_hash != expected_hash:
        failures.append(
            f"{phase_label} readiness hash mismatch: expected {expected_hash!r}, "
            f"got {actual_hash!r}"
        )
    if actual_version != expected_version:
        failures.append(
            f"{phase_label} readiness version mismatch: expected {expected_version!r}, "
            f"got {actual_version!r}"
        )
    return not failures, failures


class _Cfg003InsufficientCandidates(Exception):
    """Internal control-flow signal: not enough eligible candidates."""


def _cfg003_query_active_inventory(  # noqa: PLR0915
    litellm_base_url: str,
) -> dict[str, Any]:
    """Query readiness and build the authoritative alias inventory using
    CFG-002 compile_directory.

    Fail-closed: readiness unavailable, non-active, missing hash, source-file
    mismatch, alias mismatch (readiness vs compiled), zero candidates/ingresses,
    or missing local alias all produce healthy=False with inventory_failures.
    """
    readiness_url = f"{litellm_base_url}{RA._HEALTH_READINESS_PATH}"
    status, body = RA._http_get_json_plain(readiness_url, timeout=15.0)

    alias_config_section = {}
    if isinstance(body, dict):
        alias_config_section = body.get("aawm_alias_config") or {}

    config_hash = str(alias_config_section.get("config_hash", ""))
    config_version = str(alias_config_section.get("config_version", ""))
    source_files = sorted(alias_config_section.get("files", []))
    active_aliases = sorted(alias_config_section.get("aliases", []))
    state = str(alias_config_section.get("state", "unknown"))

    result: dict[str, Any] = {
        "readiness_status": status,
        "readiness_state": state,
        "config_hash": config_hash,
        "config_version": config_version,
        "source_files": source_files,
        "active_aliases": active_aliases,
        "alias_inventory": [],
        "healthy": False,
        "inventory_failures": [],
    }

    if status != 200:
        result["inventory_failures"].append(f"readiness unavailable: status={status}")
        return result
    if state != "active":
        result["inventory_failures"].append(f"alias config not active: state={state!r}")
        return result
    if not config_hash or not config_version:
        result["inventory_failures"].append("readiness missing semantic config_hash/config_version")
        return result

    # Use CFG-002 authoritative compile_directory.
    try:
        auth = RA._load_authoritative_startup_config()
    except Exception as exc:  # noqa: BLE001
        result["inventory_failures"].append(f"CFG-002 compile_directory failed: {exc}")
        return result

    compiled_hash = auth["config_hash"]
    compiled_aliases = auth["aliases"]
    compiled_files = auth["file_names"]

    # Source files must match.
    if compiled_files != source_files:
        result["inventory_failures"].append(
            f"source file mismatch: readiness={source_files} compiled={compiled_files}"
        )
        return result

    # Active aliases must exactly equal compiled aliases.
    if compiled_aliases != active_aliases:
        result["inventory_failures"].append(
            f"alias mismatch: readiness={active_aliases} compiled={compiled_aliases}"
        )
        return result

    # Semantic hash must match.
    if compiled_hash != config_hash:
        result["inventory_failures"].append(
            f"semantic hash mismatch: readiness={config_hash} compiled={compiled_hash}"
        )
        return result

    snapshot = auth["snapshot"]
    alias_inventory: list[dict[str, Any]] = []
    for alias_name in compiled_aliases:
        ingresses = RA._derive_ingresses_from_snapshot(snapshot, alias_name)
        eligible = RA._derive_eligible_candidates_from_snapshot(
            snapshot, alias_name=alias_name, excluded_providers=frozenset()
        )
        if not ingresses:
            result["inventory_failures"].append(
                f"alias {alias_name!r} has zero supported ingresses"
            )
            return result
        if not eligible:
            result["inventory_failures"].append(
                f"alias {alias_name!r} has zero candidates"
            )
            return result
        alias_inventory.append({
            "alias": alias_name,
            "config_hash": config_hash,
            "config_version": config_version,
            "source_files": source_files,
            "supported_ingresses": ingresses,
            "candidate_count": len(eligible),
        })

    if not alias_inventory:
        result["inventory_failures"].append("no active aliases after compilation")
        return result

    result["alias_inventory"] = alias_inventory
    result["healthy"] = True
    return result


def _cfg003_extract_observed_selection(
    case_result: dict[str, Any],
) -> dict[str, Any]:
    """Extract the OBSERVED selected provider/model/route from a case result's
    session-history record.  Based on observed correlation, not configured
    independent allowlists.

    Fix 3: When multiple session_history records exist, prefer alias-child
    records (those carrying alias metadata) over native parent rows so that
    selection evidence reflects the scoped alias route, not parent traffic.
    """
    record = _cfg003_prefer_alias_child_session_record(case_result)
    provider = record.get("provider")
    model = record.get("model")
    metadata = record.get("metadata")
    route_family = None
    if isinstance(metadata, dict):
        for key in (
            "codex_auto_agent_selected_route_family",
            "anthropic_auto_agent_selected_route_family",
            "passthrough_route_family",
        ):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip():
                route_family = val.strip()
                break
    return {
        "provider": provider if isinstance(provider, str) and provider else None,
        "model": model if isinstance(model, str) and model else None,
        "route_family": route_family,
    }


def _cfg003_prefer_alias_child_session_record(
    case_result: dict[str, Any],
) -> dict[str, Any]:
    """Return the best session_history record for selection extraction.

    Prefers records whose metadata contains alias-child markers
    (model_alias_label, anthropic_auto_agent_alias, or
    requested_model_alias) AND carry a usable full identity (non-empty
    provider, model, and an accepted route-family metadata field).
    Falls back to searching all records for any with full identity.
    Returns ``{}`` only when no usable record exists.
    """
    _ALIAS_METADATA_KEYS = (
        "model_alias_label",
        "anthropic_auto_agent_alias",
        "requested_model_alias",
    )
    session_history = case_result.get("session_history")
    if not isinstance(session_history, dict):
        return {}
    records = session_history.get("records")
    if isinstance(records, list) and len(records) > 1:
        for candidate in records:
            if not isinstance(candidate, dict):
                continue
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if any(
                isinstance(metadata.get(k), str) and metadata[k].strip()
                for k in _ALIAS_METADATA_KEYS
            ):
                if _cfg003_record_has_full_identity(candidate):
                    return candidate
    # Fallback: search all records for any with full identity rather than
    # blindly returning session_history.record / first record.
    return _cfg003_fallback_usable_record(session_history)


_CFG003_ROUTE_FAMILY_KEYS = (
    "codex_auto_agent_selected_route_family",
    "anthropic_auto_agent_selected_route_family",
    "passthrough_route_family",
)


def _cfg003_record_has_full_identity(record: dict[str, Any]) -> bool:
    """True when *record* carries provider, model, AND a route-family field."""
    prov = record.get("provider")
    mod = record.get("model")
    if not (isinstance(prov, str) and prov.strip()):
        return False
    if not (isinstance(mod, str) and mod.strip()):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return any(
        isinstance(metadata.get(k), str) and metadata[k].strip()
        for k in _CFG003_ROUTE_FAMILY_KEYS
    )


def _cfg003_fallback_usable_record(
    session_history: dict[str, Any],
) -> dict[str, Any]:
    """Return the first record with full identity, or ``{}``."""
    # Prefer the canonical single-record slot when it is usable.
    record = session_history.get("record")
    if isinstance(record, dict) and _cfg003_record_has_full_identity(record):
        return record
    records = session_history.get("records")
    if isinstance(records, list):
        for candidate in records:
            if isinstance(candidate, dict) and _cfg003_record_has_full_identity(candidate):
                return candidate
    return {}


def _cfg003_run_proof_case(
    *,
    case_name: str,
    case_config_key: str,
    cases: dict[str, dict[str, Any]],
    suite_config: dict[str, Any],
    query_url: str,
    public_key: str,
    secret_key: str,
    litellm_base_url: str,
) -> dict[str, Any]:
    """Run one real TUI proof case with a FRESH phase-specific session identity.

    Finding 4: each proof phase receives a unique session ID so selection
    evidence is attributable to that exact phase, not a reused session.
    """
    phase_session_id = str(uuid.uuid4())
    phase_start_time = dt.datetime.now(dt.timezone.utc)
    # Create a phase-specific copy of the case config with a fresh session.
    case_config = dict(cases[case_config_key])
    case_config["expected_trace_session_id"] = phase_session_id
    # Finding 1 (round 8): inject phase_start_time into nested validation
    # dicts so historical DB rows cannot satisfy a new proof phase.
    phase_start_iso = phase_start_time.isoformat()
    shv = case_config.get("session_history_validation")
    if isinstance(shv, dict):
        shv = dict(shv)
        shv["phase_start_time"] = phase_start_iso
        case_config["session_history_validation"] = shv
    tav = case_config.get("tool_activity_validation")
    if isinstance(tav, dict):
        tav = dict(tav)
        tav["phase_start_time"] = phase_start_iso
        case_config["tool_activity_validation"] = tav
    result = _run_selected_case(
        case_name=case_name,
        case_config=case_config,
        suite_config=suite_config,
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        litellm_base_url=litellm_base_url,
        cfg003_transactional=True,
    )
    selection = _cfg003_extract_observed_selection(result)
    return {
        "result": result,
        "selection": selection,
        "phase_session_id": phase_session_id,
        "phase_start_time": phase_start_time.isoformat(),
    }


def _cfg003_proof_correlation_ids(proof: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (session_id, trace_id) correlation ids from a proof result for
    error-intake attribution.  Reads the Langfuse section of the case result."""
    result = proof.get("result") if isinstance(proof, dict) else None
    if not isinstance(result, dict):
        return None, None
    langfuse = result.get("langfuse")
    if not isinstance(langfuse, dict):
        return None, None
    session_id = langfuse.get("command_session_id")
    trace_ids = langfuse.get("trace_ids") or langfuse.get("filtered_trace_ids") or []
    trace_id = trace_ids[0] if isinstance(trace_ids, list) and trace_ids else None
    return (
        str(session_id) if session_id else None,
        str(trace_id) if trace_id else None,
    )




def _cfg003_case_requires_runtime_evidence(
    case_name: str, case_config: dict[str, Any]
) -> bool:
    """Derive whether runtime-log evidence is mandatory for this case.

    Required only for transactional alias/TUI coverage cases (those with
    verification_alias) and exact CFG-003 proof case identities.
    Ordinary non-alias cases never require runtime-log evidence, even when
    the global --cfg003-transactional-refresh flag is active.
    """
    if case_config.get("verification_alias"):
        return True
    # Proof cases use suffixed names (e.g. ...__cfg003_baseline).
    for proof_id in (_CFG003_CODEX_PROOF_CASE, _CFG003_CLAUDE_PROOF_CASE):
        if case_name == proof_id or case_name.startswith(proof_id + "__"):
            return True
    return False


def _cfg003_case_correlation_ids(case_result: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (session_id, trace_id) correlation ids from a plain case result
    for per-case error-intake attribution (finding 2, round 9)."""
    if not isinstance(case_result, dict):
        return None, None
    langfuse = case_result.get("langfuse")
    if not isinstance(langfuse, dict):
        return None, None
    session_id = langfuse.get("command_session_id")
    trace_ids = langfuse.get("trace_ids") or langfuse.get("filtered_trace_ids") or []
    trace_id = trace_ids[0] if isinstance(trace_ids, list) and trace_ids else None
    return (
        str(session_id) if session_id else None,
        str(trace_id) if trace_id else None,
    )

def _cfg003_derive_terminal_marker(case_config: dict[str, Any]) -> str:
    """Finding 5 (round 7): Derive the exact validated terminal marker from
    the case's configured exact-output contract.

    For Codex: command_output_text_checks.required_prefix/suffix exact marker.
    For Claude: command_json_checks.required_equals.result required result.

    Returns the expected marker string, or empty string if contract cannot
    derive it (fail closed).
    """
    # Codex: exact prefix/suffix from command_output_text_checks.
    text_checks = case_config.get("command_output_text_checks")
    if isinstance(text_checks, dict):
        prefix = text_checks.get("required_prefix")
        suffix = text_checks.get("required_suffix")
        if isinstance(prefix, str) and prefix.strip():
            return prefix.strip()
        if isinstance(suffix, str) and suffix.strip():
            return suffix.strip()

    # Claude: required result from command_json_checks.
    json_checks = case_config.get("command_json_checks")
    if isinstance(json_checks, dict):
        required_equals = json_checks.get("required_equals")
        if isinstance(required_equals, dict):
            result = required_equals.get("result")
            if isinstance(result, str) and result.strip():
                return result.strip()

    return ""


def _cfg003_build_phase_evidence(
    *,
    phase_name: str,
    case_name: str,
    proof: dict[str, Any],
    case_config: dict[str, Any],
    active_hash: str | None = None,
    active_version: str | None = None,
    active_order: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a sanitized evidence record for one proof phase.

    Finding 5: includes case name, TUI executable/ingress, phase-specific
    session/trace, selected provider/model/route, active hash/version/order,
    terminal success marker, configured agent names/profiles, prompt identity
    as SHA-256+length (never raw), and bounded tool/child outcome summaries.
    Never persists raw prompts, authorization values, command/tool arguments,
    stdout/stderr, or raw provider bodies.
    """
    result = proof.get("result", {})
    selection = proof.get("selection", {})
    session_id = proof.get("phase_session_id")
    phase_start = proof.get("phase_start_time")

    # Extract correlation IDs.
    corr_session, corr_trace = _cfg003_proof_correlation_ids(proof)

    # Finding 4 (round 7): Executable-aware prompt identity extraction.
    # For Claude, -p is the prompt.  For Codex `codex exec`, -p is the
    # provider/profile and the actual prompt is the final positional argument.
    # Hash the actual prompt and length, never raw prompt.
    raw_prompt = ""
    command = case_config.get("command")
    if isinstance(command, list) and command:
        executable = str(command[0])
        if "codex" in executable:
            # Codex: final positional argument is the prompt.
            # Skip flags and their values; the last non-flag argument is prompt.
            positional_args: list[str] = []
            skip_next = False
            for arg in command[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if arg.startswith("-"):
                    # Flags that take a value argument.
                    if arg in ("-p", "-m", "-c", "--model", "--profile"):
                        skip_next = True
                    continue
                positional_args.append(str(arg))
            if positional_args:
                raw_prompt = positional_args[-1]
        else:
            # Claude and others: -p is the prompt.
            for i, arg in enumerate(command):
                if arg == "-p" and i + 1 < len(command):
                    raw_prompt = str(command[i + 1])
                    break
    prompt_sha256 = hashlib.sha256(raw_prompt.encode()).hexdigest() if raw_prompt else ""
    prompt_length = len(raw_prompt)

    # TUI executable.
    tui_executable = ""
    if isinstance(command, list) and command:
        tui_executable = str(command[0])

    # Bounded tool/child outcome summaries from session history.
    record = _case_result_session_history_record(result)
    metadata = record.get("metadata") if isinstance(record, dict) else None
    tool_summary = None
    child_summary = None
    if isinstance(metadata, dict):
        tool_count = metadata.get("tool_call_count")
        if tool_count is not None:
            tool_summary = {"tool_call_count": tool_count}
        child_agent = metadata.get("child_agent_name")
        if child_agent:
            child_summary = {"child_agent_name": str(child_agent)}

    return {
        "phase": phase_name,
        "case_name": case_name,
        "tui_executable": tui_executable,
        "verification_ingress": case_config.get("verification_ingress", ""),
        "phase_session_id": session_id,
        "phase_start_time": phase_start,
        "correlated_session_id": corr_session,
        "correlated_trace_id": corr_trace,
        "selected_provider": selection.get("provider"),
        "selected_model": selection.get("model"),
        "selected_route_family": selection.get("route_family"),
        "active_config_hash": active_hash,
        "active_config_version": active_version,
        "active_candidate_order": active_order,
        "passed": bool(result.get("passed")),
        "terminal_marker": _cfg003_derive_terminal_marker(case_config) or result.get("terminal_marker", ""),
        "terminal_marker_source": "derived_from_contract" if _cfg003_derive_terminal_marker(case_config) else "result_fallback",
        "expected_parent_agent_name": case_config.get("expected_parent_agent_name", ""),
        "expected_child_agent_name": case_config.get("expected_child_agent_name", ""),
        "agent_profile": case_config.get("agent_profile", ""),
        "prompt_sha256": prompt_sha256,
        "prompt_length": prompt_length,
        "tool_summary": tool_summary,
        "child_summary": child_summary,
    }


def _cfg003_selection_matches_candidate(
    selection: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    """Require observed provider, model, AND route_family all match."""
    return (
        selection.get("provider") == candidate["provider"]
        and selection.get("model") == candidate["model"]
        and selection.get("route_family") == candidate["route_family"]
    )


def _cfg003_collect_availability_evidence(
    candidates: list[dict[str, Any]],
    *,
    db_settings: dict[str, Any] | None = None,
    environment: str = "dev",
) -> dict[str, Any]:
    """Collect POSITIVE availability evidence from rate_limit_observations DB.

    Requires an explicit fresh non-exhausted quota row (remaining_pct > 0)
    for each candidate (provider, model) identity.  Missing/unknown/stale/
    exhausted evidence does NOT count as available.

    Returns {evidence: {(provider, model): {available, evidence, observed_at,
             environment, environment_binding}},
             available_identities: [{provider, model}],
             evidence_records: [sanitized JSON-safe list],
             source: "rate_limit_observations"}.
    """
    if db_settings is None:
        # No DB available -- no positive evidence possible.
        return {
            "evidence": {},
            "available_identities": [],
            "evidence_records": [],
            "source": "none",
            "note": "no DB settings; positive availability cannot be established",
        }

    availability = RA._query_positive_availability_evidence(
        db_settings=db_settings,
        candidates=candidates,
        environment=environment,
    )
    available_identities = sorted(
        ({"provider": key[0], "model": key[1]}
         for key, info in availability.items()
         if info.get("available") is True),
        key=lambda d: (d["provider"], d["model"]),
    )
    evidence_records = RA._serialize_availability_evidence(availability)
    return {
        "evidence": availability,
        "available_identities": available_identities,
        "evidence_records": evidence_records,
        "source": "rate_limit_observations",
    }


# ---------------------------------------------------------------------------
# CFG-003: Operator-asserted availability identities
# ---------------------------------------------------------------------------

_CFG003_ASSERTION_IDENTITY_RE = re.compile(
    r"^(?P<provider>[A-Za-z0-9_]+)=(?P<model>.+)$"
)


def _cfg003_parse_operator_assertions(
    raw_values: list[str] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse repeatable ``provider=model`` CLI tokens into exact identities.

    The ``=`` separator is unambiguous because provider names are
    ``[A-Za-z0-9_]+`` while model strings may contain slashes, colons, dots,
    and hyphens.  Only the FIRST ``=`` splits provider from model.

    Returns (parsed_identities, failures).  Any malformed token produces a
    failure and the identity is excluded.
    """
    if not raw_values:
        return [], []
    identities: list[tuple[str, str]] = []
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_values:
        raw = raw.strip()
        if not raw:
            failures.append("empty assertion token")
            continue
        m = _CFG003_ASSERTION_IDENTITY_RE.match(raw)
        if m is None:
            failures.append(
                f"malformed assertion (expected provider=model): {raw!r}"
            )
            continue
        provider = m.group("provider")
        model = m.group("model").strip()
        if not model:
            failures.append(f"empty model in assertion: {raw!r}")
            continue
        identity = (provider, model)
        if identity in seen:
            failures.append(f"duplicate assertion: {provider}={model}")
            continue
        seen.add(identity)
        identities.append(identity)
    return identities, failures


def _cfg003_validate_operator_assertions(
    identities: list[tuple[str, str]],
    *,
    eligible_snapshot: list[dict[str, Any]],
) -> list[str]:
    """Validate every asserted identity belongs to the current schedule-eligible
    active read-alias snapshot.

    ``eligible_snapshot`` is the list of candidate dicts produced by
    ``RA._derive_eligible_candidates_from_snapshot`` (already filtered by
    schedule windows and excluded providers).  An identity not present in this
    snapshot is rejected (unknown, schedule-expired, or non-read-alias).

    Returns a list of failure strings (empty means all valid).
    """
    eligible_set: set[tuple[str, str]] = {
        (c["provider"], c["model"]) for c in eligible_snapshot
    }
    failures: list[str] = []
    for provider, model in identities:
        if (provider, model) not in eligible_set:
            failures.append(
                f"asserted identity ({provider}, {model}) is not in the "
                f"current schedule-eligible active read-alias snapshot"
            )
    return failures


def _cfg003_bind_asserted_candidates(
    identities: list[tuple[str, str]],
    *,
    eligible_snapshot: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the candidate dicts for exactly the asserted identities, ordered
    by their position in the active schedule-eligible snapshot.

    This binds the transactional swap candidate set to exactly the asserted
    identities so that other positive DB evidence cannot displace the intended
    exact pair.  Caller must have already validated the identities via
    ``_cfg003_validate_operator_assertions``.
    """
    by_id = {(c["provider"], c["model"]): c for c in eligible_snapshot}
    ordered = [by_id[key] for key in identities if key in by_id]
    ordered.sort(key=lambda c: c["priority"], reverse=True)
    return ordered


def _cfg003_build_operator_assertion_evidence(
    identities: list[tuple[str, str]],
    *,
    environment: str = "dev",
    route_context: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Create boundary-valid in-memory availability evidence for operator-asserted
    identities.  Distinct from DB evidence: source is ``operator_assertion``.

    Each record satisfies ``_availability_record_is_valid`` boundary checks
    (available=True, provider/model/environment/environment_binding present,
    fresh observed_at).  No credentials or raw prompts are recorded.
    """
    now = dt.datetime.now(dt.timezone.utc)
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for provider, model in identities:
        evidence[(provider, model)] = {
            "available": True,
            "provider": provider,
            "model": model,
            "environment": environment,
            "environment_binding": "target_db_profile",
            "observed_at": now.isoformat(),
            "source": "operator_assertion",
            "assertion_timestamp": now.isoformat(),
            "route_context": route_context or [],
        }
    return evidence


def _cfg003_merge_availability_evidence(
    db_evidence: dict[tuple[str, str], dict[str, Any]],
    assertion_evidence: dict[tuple[str, str], dict[str, Any]],
    *,
    environment: str = "dev",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Merge DB and operator-assertion evidence by exact (provider, model) identity.

    A DB record is preserved ONLY when it is boundary-valid positive evidence
    for its exact provider/model/environment (``_availability_record_is_valid``).
    A keyed DB record that is absent, ``available=False``, stale, malformed, or
    otherwise invalid is dropped and the exact operator assertion replaces it in
    the in-memory effective evidence.  A boundary-valid positive DB record is
    never overridden by an assertion (observed DB data is preserved).  No DB
    rows are inserted or mutated; this is purely in-memory.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for key, record in db_evidence.items():
        provider, model = key
        if RA._availability_record_is_valid(
            record, provider=provider, model=model, environment=environment
        ):
            merged[key] = record
    for key, record in assertion_evidence.items():
        if key not in merged:
            merged[key] = record
    return merged


def _cfg003_db_settings(
    config: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build DB settings for rate_limit_observations from config/env.

    Resolution order for the password (finding 7):
    1. Container-owned credential via ``validation_db_password_container_env``
       resolved through ``_resolve_container_env_value`` (target profile).
    2. Local environment variable from ``db_password_env``.

    The password value is never logged.  Returns None when no password can be
    resolved (positive availability then cannot be established and the
    transaction fails closed on < 2 candidates).
    """
    checks: dict[str, Any] = {}
    for section in ("session_history_validation", "transcript_tool_use_validation"):
        candidate = config.get(section)
        if isinstance(candidate, dict) and (
            candidate.get("db_password_env") or candidate.get("db_password_container_env")
        ):
            checks = candidate
            break
    if not checks:
        for case_cfg in (config.get("cases") or {}).values():
            if not isinstance(case_cfg, dict):
                continue
            for sub in case_cfg.values():
                if isinstance(sub, dict) and (
                    sub.get("db_password_env") or sub.get("db_password_container_env")
                ):
                    checks = sub
                    break
            if checks:
                break

    # Profile-level container-owned credential (finding 7).
    if profile is None:
        profile = {}
    container_env_name = str(
        checks.get("db_password_container_env")
        or profile.get("validation_db_password_container_env")
        or ""
    ).strip()
    container_name = str(
        checks.get("db_password_container")
        or profile.get("validation_db_password_container")
        or profile.get("docker_container_name")
        or ""
    ).strip()

    db_host = str(checks.get("db_host") or profile.get("validation_db_host") or os.environ.get("AAWM_DB_HOST") or "127.0.0.1")
    db_port = int(checks.get("db_port") or profile.get("validation_db_port") or os.environ.get("AAWM_DB_PORT") or 5434)
    db_name = str(checks.get("db_name") or profile.get("validation_db_name") or os.environ.get("AAWM_DB_NAME") or "aawm_tristore")
    db_user = str(checks.get("db_user") or profile.get("validation_db_user") or os.environ.get("AAWM_DB_USER") or "aawm")

    db_password: str | None = None
    # 1. Container-owned credential (preferred).
    if container_env_name and container_name:
        db_password = _resolve_container_env_value(container_name, container_env_name)
    # 2. Local environment fallback.
    if db_password is None:
        password_env = str(checks.get("db_password_env") or "AAWM_DB_PASSWORD")
        db_password = os.environ.get(password_env)
    if db_password is None:
        return None
    return {
        "host": db_host,
        "port": db_port,
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
    }


def _cfg003_verify_source_files_unchanged(
    original_per_file_hashes: dict[str, str],
) -> tuple[bool, list[str]]:
    """Prove the checked-in source files remain byte-identical."""
    failures: list[str] = []
    config_dir = RA._AAWM_ALIAS_CONFIG_DIR
    for filename, expected_hash in sorted(original_per_file_hashes.items()):
        filepath = config_dir / filename
        try:
            actual_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"cannot read {filename}: {exc}")
            continue
        if actual_hash != expected_hash:
            failures.append(
                f"source file {filename} changed: expected={expected_hash[:12]} "
                f"actual={actual_hash[:12]}"
            )
    return not failures, failures


def _cfg003_phase_error_intake(
    phase_baseline: dict[str, dict[str, Any]],
    *,
    initiation_time: dt.datetime,
    environment: str,
    container: str,
    case_name: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    strict_correlation: bool = False,
    analysis_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Collect one phase's error-intake delta against a per-phase baseline.

    Returns a sanitized record containing baseline/current/delta snapshot
    summaries plus attributed events and failures, and the advanced baseline
    (current snapshot) for the next phase.  Finding 4: each phase uses a fresh
    baseline advanced from the prior phase so attribution is scoped to the
    phase's own window.

    Finding 3: a single authoritative current snapshot is taken once and reused
    for both delta collection and baseline advancement, eliminating the prior
    two-snapshot race.  Finding 2/9: ``strict_correlation`` requires exact
    session/trace identity for transactional alias/TUI cases.
    """
    # Strict correlation requires BOTH session_id and trace_id.  Missing
    # either is a hard failure: the caller cannot attribute events and the
    # phase must not silently pass with unverifiable evidence.
    if strict_correlation and (not session_id or not trace_id):
        current = RA._snapshot_error_intake(analysis_dir)
        return {
            "baseline_summary": RA._summarize_error_intake_snapshot(phase_baseline),
            "current_summary": RA._summarize_error_intake_snapshot(current),
            "delta_summary": RA._delta_error_intake_summary(phase_baseline, current),
            "attributed_events": [],
            "attributed_count": 0,
            "failures": [
                f"strict correlation: missing required correlation IDs "
                f"(session={session_id!r}, trace={trace_id!r})"
            ],
            "advanced_baseline": current,
        }
    # Finding 3: one authoritative current snapshot reused below.
    current = RA._snapshot_error_intake(analysis_dir)
    events, failures = RA._collect_error_intake_delta(
        phase_baseline,
        initiation_time=initiation_time,
        environment=environment,
        container=container,
        case_name=case_name,
        session_id=session_id,
        trace_id=trace_id,
        strict_correlation=strict_correlation,
        current_snapshot=current,
        analysis_dir=analysis_dir,
    )
    return {
        "baseline_summary": RA._summarize_error_intake_snapshot(phase_baseline),
        "current_summary": RA._summarize_error_intake_snapshot(current),
        "delta_summary": RA._delta_error_intake_summary(phase_baseline, current),
        "attributed_events": events,
        "attributed_count": len(events),
        "failures": failures,
        "advanced_baseline": current,
    }


def _cfg003_transactional_refresh_test(  # noqa: PLR0915
    *,
    litellm_base_url: str,
    cases: dict[str, dict[str, Any]],
    suite_config: dict[str, Any],
    query_url: str,
    public_key: str,
    secret_key: str,
    db_settings: dict[str, Any] | None = None,
    environment: str = "dev",
    container_name: str = "litellm-dev",
    operator_assertions: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the full CFG-003 transactional priority-swap refresh test.

    Uses CFG-002 compile_directory for authoritative startup config.
    Requires two POSITIVELY evidenced-available candidates (rate_limit_observations).
    All proofs require observed provider+model+route_family match.
    Restoration is unconditional and always promoted to primary failure.
    """
    refresh_url = f"{litellm_base_url}{RA._AAWM_ALIAS_CONFIG_REFRESH_PATH}"
    evidence: dict[str, Any] = {
        "test_name": "cfg003_transactional_priority_swap",
        "phases": {},
        "passed": False,
        "failures": [],
    }
    raw_source_text: str | None = None
    original_per_file_hashes: dict[str, str] = {}
    original_semantic_hash: str | None = None
    original_semantic_version: str | None = None
    source_inventory_before: dict[str, str] | None = None
    error_intake_baseline: dict[str, dict[str, Any]] | None = None
    restoration_intake_baseline: dict[str, dict[str, Any]] | None = None
    initiation_time: dt.datetime | None = None
    original_eligible: list[dict[str, Any]] = []
    original_full_order: list[dict[str, Any]] = []
    original_first: dict[str, Any] | None = None
    original_second: dict[str, Any] | None = None
    restoration_error: str | None = None
    restore_proof_session: str | None = None
    restore_proof_trace: str | None = None
    mutation_attempted: bool = False
    expected_swapped_hash: str = ""
    expected_swapped_version: str = ""
    expected_swapped_full_order: list[dict[str, Any]] = []
    proof_case = _CFG003_CODEX_PROOF_CASE
    have_proof_case = proof_case in cases

    try:
        # Error intake baseline snapshot (finding 1).
        error_intake_baseline = RA._snapshot_error_intake()
        initiation_time = dt.datetime.now(dt.timezone.utc)

        # Phase load: CFG-002 authoritative startup config.
        auth = RA._load_authoritative_startup_config()
        original_per_file_hashes = auth["per_file_hashes"]
        original_semantic_hash = auth["config_hash"]
        original_semantic_version = auth["config_version"]
        snapshot = auth["snapshot"]

        # Finding 1: Derive the authoritative RECURSIVE YAML source inventory
        # from the CFG-002 discovery path and require exactly one supported
        # source before any transaction/proof/POST.  Nested or multi-file
        # source sets hard-fail before egress.
        recursive_source_inventory = RA._recursive_yaml_source_inventory()
        if len(recursive_source_inventory) != 1:
            evidence["failures"].append(
                f"require exactly one recursive YAML source for raw-byte restore, "
                f"got {len(recursive_source_inventory)}: {sorted(recursive_source_inventory)}"
            )
            raise _Cfg003InsufficientCandidates()
        single_filename = next(iter(recursive_source_inventory))
        single_filepath = RA._AAWM_ALIAS_CONFIG_DIR / single_filename
        raw_source_text = single_filepath.read_bytes().decode("utf-8")

        # Source inventory snapshot for post-run verification.
        source_inventory_before = RA._snapshot_source_inventory()

        # Capture the COMPLETE compiled candidate order directly from the
        # snapshot, independent of provider exclusions, availability, and
        # schedule windows (finding 3 round 5).
        original_full_order = RA._derive_full_order_from_snapshot(
            snapshot, alias_name="read"
        )

        # Collect POSITIVE availability evidence from rate_limit_observations.
        all_eligible = RA._derive_eligible_candidates_from_snapshot(
            snapshot, alias_name="read"
        )
        avail = _cfg003_collect_availability_evidence(
            all_eligible, db_settings=db_settings, environment=environment
        )
        availability_evidence = avail["evidence"]

        # Operator-asserted availability: validate against schedule-eligible
        # snapshot, build in-memory evidence, merge with DB evidence.
        if operator_assertions:
            # Exactly two assertion identities are required for transactional
            # assertion mode (the intended swap pair).
            if len(operator_assertions) != 2:
                evidence["failures"].append(
                    f"operator assertion mode requires exactly 2 identities, "
                    f"got {len(operator_assertions)}"
                )
                evidence["phases"]["operator_assertion_gate"] = {
                    "passed": False,
                    "failures": evidence["failures"],
                }
                raise _Cfg003InsufficientCandidates()
            _assertion_validation_failures = _cfg003_validate_operator_assertions(
                operator_assertions,
                eligible_snapshot=all_eligible,
            )
            if _assertion_validation_failures:
                evidence["failures"].extend(_assertion_validation_failures)
                evidence["phases"]["operator_assertion_gate"] = {
                    "passed": False,
                    "failures": _assertion_validation_failures,
                }
                raise _Cfg003InsufficientCandidates()
            _route_context = [
                {
                    "provider": c["provider"],
                    "model": c["model"],
                    "route_family": c["route_family"],
                    "priority": c["priority"],
                }
                for c in all_eligible
            ]
            _assertion_evidence = _cfg003_build_operator_assertion_evidence(
                operator_assertions,
                environment=environment,
                route_context=_route_context,
            )
            availability_evidence = _cfg003_merge_availability_evidence(
                availability_evidence, _assertion_evidence,
                environment=environment,
            )
            evidence["phases"]["operator_assertion_gate"] = {
                "passed": True,
                "asserted_identities": [
                    {"provider": p, "model": m} for p, m in operator_assertions
                ],
                "assertion_evidence_records": RA._serialize_availability_evidence(
                    _assertion_evidence
                ),
            }

        # Filter to positively-available candidates only.
        original_eligible = RA._filter_candidates_by_positive_availability(
            all_eligible, availability_evidence
        )

        # When operator assertions are active, bind the swap candidate set to
        # exactly the asserted identities (ordered by snapshot priority) so
        # other positive DB evidence cannot displace the intended exact pair.
        if operator_assertions:
            original_eligible = _cfg003_bind_asserted_candidates(
                operator_assertions, eligible_snapshot=all_eligible
            )
        evidence["phases"]["load"] = {
            "original_semantic_hash": original_semantic_hash,
            "original_semantic_version": original_semantic_version,
            "per_file_hashes": original_per_file_hashes,
            "single_source_file": single_filename,
            "recursive_source_inventory": recursive_source_inventory,
            "availability_evidence": avail["evidence_records"],
            "available_identities": avail["available_identities"],
            "availability_source": avail["source"],
            "eligible_count": len(original_eligible),
            "eligible_order": [
                {
                    "provider": c["provider"],
                    "model": c["model"],
                    "route_family": c["route_family"],
                    "priority": c["priority"],
                }
                for c in original_eligible
            ],
        }

        if len(original_eligible) < 2:
            evidence["failures"].append(
                f"need >= 2 evidenced-available eligible candidates, "
                f"got {len(original_eligible)}"
            )
            raise _Cfg003InsufficientCandidates()

        original_first = original_eligible[0]
        original_second = original_eligible[1]
        evidence["phases"]["load"]["original_first"] = {
            "provider": original_first["provider"],
            "model": original_first["model"],
            "route_family": original_first["route_family"],
            "priority": original_first["priority"],
        }
        evidence["phases"]["load"]["original_second"] = {
            "provider": original_second["provider"],
            "model": original_second["model"],
            "route_family": original_second["route_family"],
            "priority": original_second["priority"],
        }

        # Finding 2 (round 7): Authoritative readiness check BEFORE baseline proof.
        # Require the active runtime hash/version to match the locally compiled
        # original state exactly.
        pre_baseline_ok, pre_baseline_failures = _cfg003_readiness_check(
            litellm_base_url,
            expected_hash=original_semantic_hash,
            expected_version=original_semantic_version,
            phase_label="pre_baseline",
        )
        evidence["phases"]["pre_baseline_readiness"] = {
            "passed": pre_baseline_ok,
            "expected_hash": original_semantic_hash,
            "expected_version": original_semantic_version,
            "failures": pre_baseline_failures,
        }
        if not pre_baseline_ok:
            evidence["failures"].extend(pre_baseline_failures)
            # Finding 2 (round 8): fail closed before baseline proof and all
            # later swap work.  No TUI/POST calls may proceed.
            raise _Cfg003InsufficientCandidates()

        # Phase baseline: real TUI proof selecting original first.
        if not have_proof_case:
            evidence["failures"].append(f"missing real TUI proof case {proof_case!r}")
        else:
            baseline = _cfg003_run_proof_case(
                case_name=f"{proof_case}__cfg003_baseline",
                case_config_key=proof_case,
                cases=cases,
                suite_config=suite_config,
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                litellm_base_url=litellm_base_url,
            )
            baseline_sel = baseline["selection"]
            baseline_ok = (
                bool(baseline["result"].get("passed"))
                and _cfg003_selection_matches_candidate(baseline_sel, original_first)
            )
            evidence["phases"]["baseline"] = {
                "case": f"{proof_case}__cfg003_baseline",
                "passed": bool(baseline["result"].get("passed")),
                "selection": baseline_sel,
                "selected_original_first": baseline_ok,
                "phase_evidence": _cfg003_build_phase_evidence(
                    phase_name="baseline",
                    case_name=f"{proof_case}__cfg003_baseline",
                    proof=baseline,
                    case_config=cases[proof_case],
                    active_hash=original_semantic_hash,
                    active_version=original_semantic_version,
                    active_order=original_full_order,
                ),
            }
            if not baseline_ok:
                evidence["failures"].append(
                    f"baseline did not select original first "
                    f"{original_first['provider']}/{original_first['model']}"
                    f"/{original_first['route_family']}: got {baseline_sel}"
                )

            # Error intake delta after baseline (finding 4: per-phase baseline,
            # case/session/trace context, persisted summaries).
            if error_intake_baseline is not None and initiation_time is not None:
                b_session, b_trace = _cfg003_proof_correlation_ids(baseline)
                intake = _cfg003_phase_error_intake(
                    error_intake_baseline,
                    initiation_time=initiation_time,
                    environment=environment,
                    container=container_name,
                    case_name=f"{proof_case}__cfg003_baseline",
                    session_id=b_session,
                    trace_id=b_trace,
                    strict_correlation=True,
                )
                evidence["phases"]["baseline"]["error_intake"] = {
                    "baseline_summary": intake["baseline_summary"],
                    "current_summary": intake["current_summary"],
                    "delta_summary": intake["delta_summary"],
                    "attributed_events": intake["attributed_events"],
                    "attributed_count": intake["attributed_count"],
                }
                # Advance the phase baseline for the next phase.
                error_intake_baseline = intake["advanced_baseline"]
                if intake["failures"]:
                    evidence["failures"].extend(intake["failures"])
                if intake["attributed_events"]:
                    evidence["failures"].append(
                        f"error intake: {intake['attributed_count']} new attributable "
                        f"error(s) after baseline"
                    )

        # Finding 2 (round 7): Authoritative readiness check AFTER baseline proof.
        post_baseline_ok, post_baseline_failures = _cfg003_readiness_check(
            litellm_base_url,
            expected_hash=original_semantic_hash,
            expected_version=original_semantic_version,
            phase_label="post_baseline",
        )
        evidence["phases"]["post_baseline_readiness"] = {
            "passed": post_baseline_ok,
            "failures": post_baseline_failures,
        }
        if not post_baseline_ok:
            evidence["failures"].extend(post_baseline_failures)

        # Phase unchanged_control (pre-swap, original-active).
        # Finding 2: controls run while original config is active, BEFORE any
        # swap mutation, so restore remains the last config mutation.
        # Finding 1 (round 9): mark mutation_attempted BEFORE the first POST
        # so restoration fires unconditionally after any refresh attempt.
        mutation_attempted = True
        try:
            unchanged_status, unchanged_response = RA._http_post_json(
                refresh_url, {"yaml": raw_source_text}
            )
            unchanged_changed = bool(unchanged_response.get("changed"))
            unchanged_hash = RA._extract_refresh_response_hash(unchanged_response)
            unchanged_version = RA._extract_refresh_response_version(unchanged_response)
            evidence["phases"]["unchanged_control"] = {
                "status_code": unchanged_status,
                "changed": unchanged_changed,
                "semantic_hash": unchanged_hash,
                "version": unchanged_version,
                "hash_matches_original": unchanged_hash == original_semantic_hash,
                "version_matches_original": unchanged_version == original_semantic_version,
            }
            if unchanged_status != 200 or unchanged_changed:
                evidence["failures"].append(
                    f"unchanged control: expected 200/changed=false, "
                    f"got status={unchanged_status} changed={unchanged_changed}"
                )
            if unchanged_hash != original_semantic_hash:
                evidence["failures"].append(
                    f"unchanged control: hash mismatch: {unchanged_hash} "
                    f"!= {original_semantic_hash}"
                )
            if unchanged_version != original_semantic_version:
                evidence["failures"].append(
                    f"unchanged control: version mismatch: {unchanged_version} "
                    f"!= {original_semantic_version}"
                )
        except Exception as uc_exc:  # noqa: BLE001
            evidence["phases"]["unchanged_control"] = {"error": str(uc_exc)}
            evidence["failures"].append(f"unchanged control exception: {uc_exc}")

        # Phase invalid_control (pre-swap, original-active).
        # Finding 2: invalid YAML rejected while original config is active.
        try:
            invalid_status, invalid_response = RA._http_post_json(
                refresh_url, {"yaml": _CFG003_INVALID_YAML}
            )
            invalid_hash = RA._extract_refresh_response_hash(invalid_response)
            evidence["phases"]["invalid_control"] = {
                "status_code": invalid_status,
                "lkg_semantic_hash": invalid_hash,
                "lkg_preserved": invalid_hash == original_semantic_hash,
                "rejected": invalid_status == 400,
            }
            if invalid_status != 400:
                evidence["failures"].append(
                    f"invalid control: expected 400, got {invalid_status}"
                )
            if invalid_hash != original_semantic_hash:
                evidence["failures"].append(
                    "invalid control: LKG semantic hash changed after invalid refresh"
                )
        except Exception as inv_exc:  # noqa: BLE001
            evidence["phases"]["invalid_control"] = {"error": str(inv_exc)}
            evidence["failures"].append(f"invalid control exception: {inv_exc}")

        # Phase swap_build: use the EXACT evidenced pair, not first-two raw.
        # Finding 3: wire _build_exact_pair_priority_swap_yaml with the two
        # availability-evidenced (provider, model) identities.  Any helper
        # error, identity mismatch, or returned order mismatch fails BEFORE
        # the swap POST.
        evidenced_pair = (
            (original_first["provider"], original_first["model"]),
            (original_second["provider"], original_second["model"]),
        )
        try:
            swapped_yaml, _orig, swapped_eligible = (
                RA._build_exact_pair_priority_swap_yaml(
                    raw_source_text, pair=evidenced_pair, alias_name="read"
                )
            )
        except (ValueError, KeyError) as swap_build_exc:
            evidence["failures"].append(
                f"exact-pair swap build failed before POST: {swap_build_exc}"
            )
            raise _Cfg003InsufficientCandidates() from swap_build_exc

        # Finding 5: validate the swap by identities and relative positions,
        # NOT by requiring the pair to occupy positions zero and one.  When
        # unavailable candidates sit between the two available ones in the
        # full order, the swapped eligible list may have other candidates
        # around the pair.  Prove: (a) A and C priorities exchanged,
        # (b) C now precedes A in the eligible order, (c) all other
        # candidates unchanged.
        _orig_by_id = {
            (c["provider"], c["model"]): c for c in _orig
        }
        _swap_by_id = {
            (c["provider"], c["model"]): c for c in swapped_eligible
        }
        _id_a = (original_first["provider"], original_first["model"])
        _id_c = (original_second["provider"], original_second["model"])
        _swap_a = _swap_by_id.get(_id_a)
        _swap_c = _swap_by_id.get(_id_c)
        _orig_a = _orig_by_id.get(_id_a)
        _orig_c = _orig_by_id.get(_id_c)
        # (a) Priorities exchanged.
        _priorities_swapped = (
            _swap_a is not None and _swap_c is not None
            and _orig_a is not None and _orig_c is not None
            and _swap_a["priority"] == _orig_c["priority"]
            and _swap_c["priority"] == _orig_a["priority"]
        )
        # (b) Relative position: C now before A in the eligible order.
        _swap_eligible_ids = [
            (c["provider"], c["model"]) for c in swapped_eligible
        ]
        _pos_a = _swap_eligible_ids.index(_id_a) if _id_a in _swap_eligible_ids else -1
        _pos_c = _swap_eligible_ids.index(_id_c) if _id_c in _swap_eligible_ids else -1
        _relative_swapped = _pos_c >= 0 and _pos_a >= 0 and _pos_c < _pos_a
        # (c) All other candidates unchanged.
        _others_unchanged = all(
            _swap_by_id.get((_o["provider"], _o["model"]), {}).get("priority") == _o["priority"]
            for _o in _orig
            if (_o["provider"], _o["model"]) not in (_id_a, _id_c)
        )
        swap_exact = _priorities_swapped and _relative_swapped and _others_unchanged
        evidence["phases"]["swap_build"] = {
            "evidenced_pair": [
                {"provider": p, "model": m} for p, m in evidenced_pair
            ],
            "swapped_first": (
                {
                    "provider": swapped_eligible[0]["provider"],
                    "model": swapped_eligible[0]["model"],
                    "route_family": swapped_eligible[0]["route_family"],
                    "priority": swapped_eligible[0]["priority"],
                }
                if swapped_eligible
                else None
            ),
            "swap_is_exact": swap_exact,
            "priorities_swapped": _priorities_swapped,
            "relative_position_swapped": _relative_swapped,
            "other_candidates_unchanged": _others_unchanged,
        }
        if not swap_exact:
            evidence["failures"].append(
                "exact-pair swap returned order mismatch vs evidenced pair"
            )
            raise _Cfg003InsufficientCandidates()

        # Finding 2 (round 7): Locally compile the exact-pair swapped YAML to
        # obtain expected semantic hash/version/full complete order.
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
            compile_yaml as _compile_yaml_for_swap,
        )
        swapped_snapshot = _compile_yaml_for_swap(swapped_yaml)
        expected_swapped_hash = swapped_snapshot.config_hash
        expected_swapped_version = swapped_snapshot.config_version
        expected_swapped_full_order = RA._derive_full_order_from_snapshot(
            swapped_snapshot, alias_name="read"
        )
        evidence["phases"]["swap_build"]["expected_swapped_hash"] = expected_swapped_hash
        evidence["phases"]["swap_build"]["expected_swapped_version"] = expected_swapped_version

        # Phase swap_refresh.
        swap_status, swap_response = RA._http_post_json(
            refresh_url, {"yaml": swapped_yaml}
        )
        swap_changed = bool(swap_response.get("changed"))
        swap_new_hash = RA._extract_refresh_response_hash(swap_response)
        swap_new_version = RA._extract_refresh_response_version(swap_response)
        swap_active_order = swap_response.get("active_candidate_order")
        swap_read_order = None
        if isinstance(swap_active_order, dict):
            swap_read_order = swap_active_order.get("read")
        # Finding 2 (round 7): require exact hash, version, and full order match.
        swap_hash_matches = swap_new_hash == expected_swapped_hash
        swap_version_matches = swap_new_version == expected_swapped_version
        swap_order_matches = RA._candidate_order_matches(
            swap_read_order, expected_swapped_full_order
        )
        evidence["phases"]["swap_refresh"] = {
            "status_code": swap_status,
            "changed": swap_changed,
            "new_semantic_hash": swap_new_hash,
            "expected_swapped_hash": expected_swapped_hash,
            "hash_matches_expected": swap_hash_matches,
            "new_version": swap_new_version,
            "expected_swapped_version": expected_swapped_version,
            "version_matches_expected": swap_version_matches,
            "hash_differs_from_original": (
                bool(swap_new_hash) and swap_new_hash != original_semantic_hash
            ),
            "active_candidate_order": swap_read_order,
            "order_matches_expected": swap_order_matches,
        }
        if swap_status != 200 or not swap_changed:
            evidence["failures"].append(
                f"swap refresh failed: status={swap_status} changed={swap_changed}"
            )
        elif not swap_new_hash:
            evidence["failures"].append("swap refresh returned empty semantic hash")
        elif swap_new_hash == original_semantic_hash:
            evidence["failures"].append("swap refresh did not change semantic config_hash")
        if swap_status == 200 and swap_changed:
            if not swap_hash_matches:
                evidence["failures"].append(
                    f"swap refresh hash mismatch: expected {expected_swapped_hash!r}, "
                    f"got {swap_new_hash!r}"
                )
            if not swap_version_matches:
                evidence["failures"].append(
                    f"swap refresh version mismatch: expected {expected_swapped_version!r}, "
                    f"got {swap_new_version!r}"
                )
            if not swap_order_matches:
                evidence["failures"].append(
                    "swap refresh active_candidate_order does not match "
                    "locally compiled swapped full order"
                )

        # Finding 2 (round 7): Readiness check BEFORE swap_proof.
        pre_swap_proof_ok = False
        if swap_status == 200 and swap_changed:
            pre_swap_proof_ok, pre_swap_proof_failures = _cfg003_readiness_check(
                litellm_base_url,
                expected_hash=expected_swapped_hash,
                expected_version=expected_swapped_version,
                phase_label="pre_swap_proof",
            )
            evidence["phases"]["pre_swap_proof_readiness"] = {
                "passed": pre_swap_proof_ok,
                "expected_hash": expected_swapped_hash,
                "expected_version": expected_swapped_version,
                "failures": pre_swap_proof_failures,
            }
            if not pre_swap_proof_ok:
                evidence["failures"].extend(pre_swap_proof_failures)
            # Finding 2 (round 8): preserve the actually observed order in
            # readiness evidence for diagnostic clarity.
            evidence["phases"]["pre_swap_proof_readiness"]["observed_active_order"] = swap_read_order

        # Phase swap_proof.
        # Finding 2 (round 8): swap_proof runs ONLY when readiness, hash,
        # version, and full order all match.  Wrong observed state prevents
        # the proof TUI call while restoration remains unconditional.
        swap_proof_gate = (
            swap_status == 200 and swap_changed
            and swap_hash_matches and swap_version_matches and swap_order_matches
            and pre_swap_proof_ok
        )
        if have_proof_case and swap_proof_gate:
            swap_proof = _cfg003_run_proof_case(
                case_name=f"{proof_case}__cfg003_swap_proof",
                case_config_key=proof_case,
                cases=cases,
                suite_config=suite_config,
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                litellm_base_url=litellm_base_url,
            )
            swap_sel = swap_proof["selection"]
            swap_proof_ok = (
                bool(swap_proof["result"].get("passed"))
                and _cfg003_selection_matches_candidate(swap_sel, original_second)
            )
            evidence["phases"]["swap_proof"] = {
                "case": f"{proof_case}__cfg003_swap_proof",
                "passed": bool(swap_proof["result"].get("passed")),
                "selection": swap_sel,
                "selected_original_second": swap_proof_ok,
                "phase_evidence": _cfg003_build_phase_evidence(
                    phase_name="swap_proof",
                    case_name=f"{proof_case}__cfg003_swap_proof",
                    proof=swap_proof,
                    case_config=cases[proof_case],
                    active_hash=swap_new_hash,
                    active_version=expected_swapped_version,
                    active_order=expected_swapped_full_order,
                ),
            }
            if not swap_proof_ok:
                evidence["failures"].append(
                    f"swap proof did not select original second "
                    f"{original_second['provider']}/{original_second['model']}"
                    f"/{original_second['route_family']}: got {swap_sel}"
                )

            # Error intake delta after swap proof (finding 4: per-phase baseline).
            if error_intake_baseline is not None and initiation_time is not None:
                s_session, s_trace = _cfg003_proof_correlation_ids(swap_proof)
                intake = _cfg003_phase_error_intake(
                    error_intake_baseline,
                    initiation_time=initiation_time,
                    environment=environment,
                    container=container_name,
                    case_name=f"{proof_case}__cfg003_swap_proof",
                    session_id=s_session,
                    trace_id=s_trace,
                    strict_correlation=True,
                )
                evidence["phases"]["swap_proof"]["error_intake"] = {
                    "baseline_summary": intake["baseline_summary"],
                    "current_summary": intake["current_summary"],
                    "delta_summary": intake["delta_summary"],
                    "attributed_events": intake["attributed_events"],
                    "attributed_count": intake["attributed_count"],
                }
                error_intake_baseline = intake["advanced_baseline"]
                if intake["failures"]:
                    evidence["failures"].extend(intake["failures"])
                if intake["attributed_events"]:
                    evidence["failures"].append(
                        f"error intake: {intake['attributed_count']} new attributable "
                        f"error(s) after swap proof"
                    )

        # Finding 2: distinct restoration baseline advanced after swap so the
        # restoration-phase attribution is scoped to the restoration window.
        restoration_intake_baseline = RA._snapshot_error_intake()

    except _Cfg003InsufficientCandidates:
        pass
    except Exception as exc:  # noqa: BLE001
        evidence["failures"].append(f"pre-restoration exception: {exc}")

    finally:
        # Phase restoration: UNCONDITIONAL repost of exact original raw bytes.
        # Finding 1 (round 9): restoration fires only after a mutation/refresh
        # POST was attempted.  Pre-baseline readiness rejection (before any
        # POST) must cause exactly zero POSTs and zero TUI calls.
        if raw_source_text is not None and mutation_attempted:
            try:
                restore_status, restore_response = RA._http_post_json(
                    refresh_url, {"yaml": raw_source_text}
                )
                restore_hash = RA._extract_refresh_response_hash(restore_response)
                restore_version = RA._extract_refresh_response_version(restore_response)
                restore_ok = (
                    restore_status == 200
                    and bool(restore_hash)
                    and restore_hash == original_semantic_hash
                    and restore_version == original_semantic_version
                )
                # Finding 3: Prove active restored full candidate order from the
                # authoritative refresh response (not a locally inferred order).
                restored_order = restore_response.get("active_candidate_order")
                restored_read_order = None
                if isinstance(restored_order, dict):
                    restored_read_order = restored_order.get("read")
                # Finding 5: compare the EXACT complete ordered list against the
                # full compiled candidate order -- no prefix acceptance, no
                # extra tail, including anthropic_route_family and last_resort.
                order_matches = RA._candidate_order_matches(
                    restored_read_order, original_full_order
                )
                restore_ok = restore_ok and order_matches
                evidence["phases"]["restoration"] = {
                    "status_code": restore_status,
                    "restored_semantic_hash": restore_hash,
                    "restored_version": restore_version,
                    "hash_matches_original": restore_hash == original_semantic_hash,
                    "version_matches_original": restore_version == original_semantic_version,
                    "restored_candidate_order": restored_read_order,
                    "expected_full_order": [
                        {
                            "provider": c["provider"],
                            "model": c["model"],
                            "route_family": c["route_family"],
                            "anthropic_route_family": c.get("anthropic_route_family", ""),
                            "priority": c["priority"],
                        }
                        for c in original_full_order
                    ],
                    "order_matches_original": order_matches,
                }
                if not restore_ok:
                    restoration_error = (
                        f"RESTORATION FAILED: status={restore_status} "
                        f"hash={restore_hash} version={restore_version} "
                        f"expected_hash={original_semantic_hash} "
                        f"expected_version={original_semantic_version} "
                        f"order_matches={order_matches}"
                    )
            except Exception as restore_exc:  # noqa: BLE001
                restoration_error = f"RESTORATION EXCEPTION: {restore_exc}"
                evidence["phases"]["restoration"] = {"error": str(restore_exc)}

            # Finding 1 (item 7): each cleanup verifier is wrapped so its
            # exception can never escape or mask the restoration failure.
            # Failures are recorded as redacted evidence; restoration_error
            # remains primary.
            try:
                src_ok, src_failures = _cfg003_verify_source_files_unchanged(
                    original_per_file_hashes
                )
                evidence["phases"]["source_files_unchanged"] = {
                    "passed": src_ok,
                    "failures": src_failures,
                }
                if not src_ok:
                    if restoration_error is None:
                        restoration_error = f"SOURCE FILES CHANGED: {src_failures}"
                    evidence["failures"].extend(src_failures)
            except Exception as src_exc:  # noqa: BLE001
                evidence["phases"]["source_files_unchanged"] = {
                    "passed": False,
                    "failures": [f"source file verifier exception: {src_exc}"],
                }
                evidence["failures"].append(f"source file verifier exception: {src_exc}")

            try:
                if source_inventory_before is not None:
                    source_inventory_after = RA._snapshot_source_inventory()
                    inventory_ok = source_inventory_before == source_inventory_after
                    evidence["phases"]["source_inventory_unchanged"] = {
                        "passed": inventory_ok,
                        "before_count": len(source_inventory_before),
                        "after_count": len(source_inventory_after),
                    }
                    if not inventory_ok:
                        if restoration_error is None:
                            restoration_error = "SOURCE INVENTORY CHANGED: path set or raw hashes differ"
                        evidence["failures"].append("source inventory changed after run")
            except Exception as inv_exc:  # noqa: BLE001
                evidence["phases"]["source_inventory_unchanged"] = {
                    "passed": False,
                    "failures": [f"source inventory verifier exception: {inv_exc}"],
                }
                evidence["failures"].append(f"source inventory verifier exception: {inv_exc}")

            try:
                if restoration_error is None:
                    post_restore_ok, post_restore_failures = _cfg003_readiness_check(
                        litellm_base_url,
                        expected_hash=original_semantic_hash,
                        expected_version=original_semantic_version,
                        phase_label="post_restoration",
                    )
                    evidence["phases"]["post_restoration_readiness"] = {
                        "passed": post_restore_ok,
                        "expected_hash": original_semantic_hash,
                        "expected_version": original_semantic_version,
                        "failures": post_restore_failures,
                    }
                    if not post_restore_ok:
                        evidence["failures"].extend(post_restore_failures)
                        if restoration_error is None:
                            restoration_error = (
                                f"POST-RESTORATION READINESS FAILED: {post_restore_failures}"
                            )
            except Exception as ready_exc:  # noqa: BLE001
                evidence["phases"]["post_restoration_readiness"] = {
                    "passed": False,
                    "failures": [f"post-restoration readiness exception: {ready_exc}"],
                }
                evidence["failures"].append(f"post-restoration readiness exception: {ready_exc}")

            # Phase restore_proof.
            if restoration_error is None and have_proof_case and original_first is not None:
                try:
                    restore_proof = _cfg003_run_proof_case(
                        case_name=f"{proof_case}__cfg003_restore_proof",
                        case_config_key=proof_case,
                        cases=cases,
                        suite_config=suite_config,
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        litellm_base_url=litellm_base_url,
                    )
                    restore_sel = restore_proof["selection"]
                    restore_proof_ok = (
                        bool(restore_proof["result"].get("passed"))
                        and _cfg003_selection_matches_candidate(restore_sel, original_first)
                    )
                    # Finding 2: capture restore-proof correlation IDs for final
                    # phase attribution.
                    restore_proof_session, restore_proof_trace = _cfg003_proof_correlation_ids(restore_proof)
                    evidence["phases"]["restore_proof"] = {
                        "case": f"{proof_case}__cfg003_restore_proof",
                        "passed": bool(restore_proof["result"].get("passed")),
                        "selection": restore_sel,
                        "selected_original_first": restore_proof_ok,
                        "session_id": restore_proof_session,
                        "trace_id": restore_proof_trace,
                        "phase_evidence": _cfg003_build_phase_evidence(
                            phase_name="restore_proof",
                            case_name=f"{proof_case}__cfg003_restore_proof",
                            proof=restore_proof,
                            case_config=cases[proof_case],
                            active_hash=original_semantic_hash,
                            active_version=original_semantic_version,
                            active_order=original_full_order,
                        ),
                    }
                    if not restore_proof_ok:
                        evidence["failures"].append(
                            f"restore proof did not reselect original first: got {restore_sel}"
                        )
                        # Finding 4: restore_proof failure IS a restoration failure.
                        restoration_error = (
                            f"RESTORATION PROOF FAILED: did not reselect original first "
                            f"{original_first['provider']}/{original_first['model']}"
                            f"/{original_first['route_family']}: got {restore_sel}"
                        )
                except Exception as proof_exc:  # noqa: BLE001
                    evidence["phases"]["restore_proof"] = {"error": str(proof_exc)}
                    evidence["failures"].append(f"restore proof exception: {proof_exc}")
                    # Finding 4: exceptional restore proof IS a restoration failure.
                    restoration_error = f"RESTORATION PROOF EXCEPTION: {proof_exc}"

            # Finding 2: unchanged_control and invalid_control moved to
            # pre-swap phase (original-active).  No config mutation POST
            # occurs after the unconditional restoration.

            try:
                if restoration_intake_baseline is not None and initiation_time is not None:
                    intake = _cfg003_phase_error_intake(
                        restoration_intake_baseline,
                        initiation_time=initiation_time,
                        environment=environment,
                        container=container_name,
                        case_name=f"{proof_case}__cfg003_restore_proof",
                        session_id=restore_proof_session,
                        trace_id=restore_proof_trace,
                        strict_correlation=True,
                    )
                    evidence["phases"]["error_intake_final"] = {
                        "baseline_summary": intake["baseline_summary"],
                        "current_summary": intake["current_summary"],
                        "delta_summary": intake["delta_summary"],
                        "attributed_events": intake["attributed_events"],
                        "attributed_count": intake["attributed_count"],
                        "failures": intake["failures"],
                    }
                    if intake["failures"]:
                        evidence["failures"].extend(intake["failures"])
                    if intake["attributed_events"]:
                        evidence["failures"].append(
                            f"error intake: {intake['attributed_count']} new attributable "
                            f"error(s) after restoration"
                        )
            except Exception as intake_exc:  # noqa: BLE001
                evidence["phases"]["error_intake_final"] = {
                    "passed": False,
                    "failures": [f"error intake verifier exception: {intake_exc}"],
                }
                evidence["failures"].append(f"error intake verifier exception: {intake_exc}")

    # Restoration failure is ALWAYS the primary hard failure.
    if restoration_error is not None:
        evidence["restoration_failure"] = restoration_error
        evidence["failures"] = [f for f in evidence["failures"] if f != restoration_error]
        evidence["failures"].insert(0, restoration_error)
    # Finding 1 (item 7): ALWAYS emit the recovery artifact so operators can
    # manually restore original state even when a cleanup verifier exception
    # was swallowed and no explicit restoration_error was recorded.
    evidence["recovery_artifact"] = RA._redact_sensitive_artifact_fields({
        "original_semantic_hash": original_semantic_hash,
        "original_semantic_version": original_semantic_version,
        "per_file_hashes": original_per_file_hashes,
        "original_eligible_order": [
            {"provider": c["provider"], "model": c["model"]}
            for c in original_eligible
        ],
        "original_full_order": [
            {"provider": c["provider"], "model": c["model"],
             "route_family": c["route_family"], "priority": c["priority"]}
            for c in original_full_order
        ],
        "phases": evidence.get("phases", {}),
    })

    evidence["passed"] = not evidence["failures"]
    return RA._redact_sensitive_artifact_fields(evidence)

def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Run real-Claude Anthropic adapter acceptance checks through a target LiteLLM instance."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to adapter suite config JSON.")
    parser.add_argument('--write-artifact', required=True, help='Where to write the JSON artifact.')
    parser.add_argument('--langfuse-query-url', default=None, help='Override Langfuse query URL.')
    parser.add_argument('--cases', default=None, help='Comma-separated subset of adapter cases to run.')
    parser.add_argument(
        "--target",
        default=os.environ.get("AAWM_ADAPTER_TARGET", None),
        help="Target profile to test. Built-ins: dev (:4001/litellm-dev), prod (:4000/aawm-litellm).",
    )
    parser.add_argument('--litellm-base-url', default=None, help='Override the target LiteLLM base URL.')
    parser.add_argument('--anthropic-base-url', default=None, help='Override ANTHROPIC_BASE_URL passed to Claude CLI.')
    parser.add_argument('--docker-container-name', default=None, help='Override the Docker container used for health/log checks.')
    parser.add_argument('--expected-trace-environment', default=None, help='Override expected Langfuse trace environment.')
    parser.add_argument(
        '--cfg003-transactional-refresh',
        action='store_true',
        default=False,
        help='Run the CFG-003 transactional priority-swap refresh test (dev only).',
    )
    parser.add_argument(
        '--cfg003-assert-availability',
        action='append',
        default=None,
        metavar='PROVIDER=MODEL',
        help=(
            'Operator-asserted exact availability identity (repeatable). '
            'Only valid with --cfg003-transactional-refresh on canonical dev. '
            'Syntax: provider=model (e.g. openrouter=openrouter/cohere/north-mini-code:free).'
        ),
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    artifact_path = pathlib.Path(args.write_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # CFG-003: Operator-asserted availability -- early parse/gate.
    # Assertions are only valid with --cfg003-transactional-refresh.
    _cfg003_assertion_identities: list[tuple[str, str]] = []
    if args.cfg003_assert_availability:
        _early_assertion_failures: list[str] = []
        if not args.cfg003_transactional_refresh:
            _early_assertion_failures.append(
                "--cfg003-assert-availability requires --cfg003-transactional-refresh"
            )
        else:
            _cfg003_assertion_identities, _assertion_parse_failures = (
                _cfg003_parse_operator_assertions(args.cfg003_assert_availability)
            )
            _early_assertion_failures.extend(_assertion_parse_failures)
        if _early_assertion_failures:
            _emit_stderr(
                f"[cfg003] operator assertion gate FAILED: "
                f"{_early_assertion_failures}",
                flush=True,
            )
            _assert_gate_artifact: dict[str, Any] = {
                "suite_version": 1,
                "timestamp": RA._isoformat(RA._utcnow()),
                "git_commit": RA._git_value("rev-parse", "HEAD"),
                "git_branch": RA._git_value("branch", "--show-current"),
                "environment": {"target_profile": args.target or "dev"},
                "results": {
                    "cfg003_assertion_gate": {
                        "passed": False,
                        "skipped": False,
                        "failures": _early_assertion_failures,
                        "warnings": [],
                    },
                },
                "verification_matrix": [],
                "summary": {},
            }
            _assert_gate_artifact["summary"] = _build_summary(
                _assert_gate_artifact["results"]
            )
            _write_artifact(artifact_path, _assert_gate_artifact)
            _emit_stdout(json.dumps(_assert_gate_artifact["summary"], indent=2))
            return 1
    # Finding 3 (round 8): raw-config canonical preflight BEFORE dotenv.
    # An invalid transactional target must not trigger dotenv loading or
    # credential resolution.  The resolved-profile gate after profile
    # resolution is retained as a second check.
    if args.cfg003_transactional_refresh:
        _raw_ok, _raw_failures = _cfg003_raw_config_canonical_preflight(
            config_path=config_path,
            target_override=args.target,
            litellm_base_url_override=args.litellm_base_url,
            anthropic_base_url_override=args.anthropic_base_url,
            docker_container_name_override=args.docker_container_name,
            expected_trace_environment_override=args.expected_trace_environment,
        )
        if not _raw_ok:
            _emit_stderr(
                f"[cfg003] raw-config canonical preflight FAILED (before dotenv): "
                f"{_raw_failures}",
                flush=True,
            )
            _raw_target = args.target or "dev"
            gate_artifact: dict[str, Any] = {
                "suite_version": 1,
                "timestamp": RA._isoformat(RA._utcnow()),
                "git_commit": RA._git_value("rev-parse", "HEAD"),
                "git_branch": RA._git_value("branch", "--show-current"),
                "environment": {"target_profile": _raw_target},
                "results": {
                    "cfg003_raw_config_gate": {
                        "passed": False,
                        "skipped": False,
                        "failures": _raw_failures,
                        "warnings": [],
                    },
                },
                "verification_matrix": [],
                "summary": {},
            }
            gate_artifact["summary"] = _build_summary(gate_artifact["results"])
            _write_artifact(artifact_path, gate_artifact)
            _emit_stdout(json.dumps(gate_artifact["summary"], indent=2))
            return 1
    _load_dotenv_into_environment(ROOT / '.env')
    config = _resolve_env_placeholders(RA._load_json(config_path))
    target = args.target or str(config.get('default_target_profile') or 'dev')
    profile = _target_profile_settings(
        config=config,
        target=target,
        litellm_base_url=args.litellm_base_url,
        anthropic_base_url=args.anthropic_base_url,
        docker_container_name=args.docker_container_name,
        expected_trace_environment=args.expected_trace_environment,
    )
    config = _apply_target_profile_to_config(
        config,
        target=target,
        profile=profile,
    )

    # Finding 3 (round 7): Canonical dev isolation gate BEFORE credential
    # resolution.  A dev-labelled prod/aawm-litellm override must not trigger
    # docker exec or read target-owned credentials.
    if args.cfg003_transactional_refresh:
        _cfg003_canonical_ok, _cfg003_canonical_failures = (
            _cfg003_validate_canonical_dev_profile(target=target, profile=profile)
        )
        if not _cfg003_canonical_ok:
            _emit_stderr(
                f"[cfg003] canonical dev gate FAILED: {_cfg003_canonical_failures}",
                flush=True,
            )
            # Build a minimal artifact for the gate failure without credentials.
            litellm_base_url = config.get('litellm_base_url', profile['litellm_base_url'])
            gate_artifact: dict[str, Any] = {
                'suite_version': config.get('suite_version', 1),
                'timestamp': RA._isoformat(RA._utcnow()),
                'git_commit': RA._git_value('rev-parse', 'HEAD'),
                'git_branch': RA._git_value('branch', '--show-current'),
                'environment': {
                    'target_profile': target,
                    'litellm_base_url': litellm_base_url,
                },
                'results': {
                    'cfg003_target_gate': {
                        'passed': False,
                        'skipped': False,
                        'failures': _cfg003_canonical_failures,
                        'warnings': [],
                    },
                },
                'verification_matrix': [],
                'summary': {},
            }
            gate_artifact['summary'] = _build_summary(gate_artifact['results'])
            _write_artifact(artifact_path, gate_artifact)
            _emit_stdout(json.dumps(gate_artifact['summary'], indent=2))
            return 1

    credentials = _resolve_main_credentials(config=config, args=args, profile=profile)
    if isinstance(credentials, int):
        return credentials
    public_key, secret_key, query_url, public_key_env, secret_key_env = credentials

    cases = config.get('cases') or {}
    available_cases = list(cases.keys())
    selected_cases = _parse_selected_cases(
        args.cases,
        available_cases,
        default_excluded_cases=config.get('default_excluded_cases'),
    )

    litellm_base_url = config.get('litellm_base_url', profile['litellm_base_url'])
    artifact = _build_initial_artifact(
        config=config,
        profile=profile,
        target=target,
        litellm_base_url=litellm_base_url,
        query_url=query_url,
        public_key_env=public_key_env,
        secret_key_env=secret_key_env,
    )
    _write_artifact(artifact_path, artifact)

    # CFG-003: Active alias/ingress inventory (always captured for artifact).
    cfg003_inventory = _cfg003_query_active_inventory(litellm_base_url)
    artifact['cfg003_alias_inventory'] = RA._redact_sensitive_artifact_fields(
        cfg003_inventory
    )
    _write_artifact(artifact_path, artifact)

    # CFG-003: Validate the complete configured case map for every active
    # alias/ingress during ALL runs (ordinary and transactional).  This
    # ensures the configured map is coherent without requiring every case
    # to be selected.
    # Finding 4: complete-coverage enforcement is gated to transactional mode
    # so ordinary non-CFG003 runs retain prior behavior.
    if args.cfg003_transactional_refresh and cfg003_inventory.get('healthy'):
        cfg003_map_passed, cfg003_map_failures = RA._validate_complete_coverage_map(
            alias_inventory=cfg003_inventory.get('alias_inventory', []),
            cases=cases,
        )
        artifact['cfg003_coverage_map'] = {
            'passed': cfg003_map_passed,
            'failures': cfg003_map_failures,
        }
        if not cfg003_map_passed:
            _emit_stderr(
                f"[cfg003] configured coverage map INVALID: {cfg003_map_failures}",
                flush=True,
            )
            artifact['results']['cfg003_coverage_map'] = {
                'passed': False,
                'skipped': False,
                'failures': cfg003_map_failures,
                'warnings': [],
            }
            artifact['summary'] = _build_summary(artifact['results'])
            _write_artifact(artifact_path, artifact)
            _emit_stdout(json.dumps(artifact['summary'], indent=2))
            return 1
        _write_artifact(artifact_path, artifact)

    # Finding 3/6: Unhealthy inventory blocks ALL selected egress cases before
    # execution -- real TUI commands, http_request, cli_passthrough, and any
    # other provider path -- not only Codex/Claude commands.  Ordinary and
    # transactional runs fail before _run_selected_case.
    # Finding 4: fail-closed inventory/egress enforcement is gated to
    # transactional mode so ordinary non-CFG003 runs retain prior behavior.
    if args.cfg003_transactional_refresh:
        _selected_egress = [
            c for c in selected_cases
            if isinstance(cases.get(c), dict) and RA._is_egress_case(cases[c])
        ]
        if _selected_egress and not cfg003_inventory.get('healthy'):
            _inv_failures = cfg003_inventory.get('inventory_failures', []) or ['inventory not healthy']
            _emit_stderr(
                f"[cfg003] inventory FAILED (fail-closed before egress): {_inv_failures}",
                flush=True,
            )
            artifact['results']['cfg003_inventory_gate'] = {
                'passed': False,
                'skipped': False,
                'failures': _inv_failures,
                'warnings': [],
            }
            artifact['summary'] = _build_summary(artifact['results'])
            _write_artifact(artifact_path, artifact)
            _emit_stdout(json.dumps(artifact['summary'], indent=2))
            return 1

    # Coverage gate: enforced ONLY when --cfg003-transactional-refresh is
    # active.  When the transactional test IS selected, fail-closed: readiness
    # must be healthy and every active alias/ingress must have exactly one
    # real TUI case among the selected cases.
    if args.cfg003_transactional_refresh:
        cfg003_inventory_failures = cfg003_inventory.get('inventory_failures', [])
        if cfg003_inventory_failures or not cfg003_inventory.get('healthy'):
            _emit_stderr(
                f"[cfg003] inventory FAILED (fail-closed before egress): "
                f"{cfg003_inventory_failures}",
                flush=True,
            )
            artifact['results']['cfg003_inventory_gate'] = {
                'passed': False,
                'skipped': False,
                'failures': cfg003_inventory_failures or ['inventory not healthy'],
                'warnings': [],
            }
            artifact['summary'] = _build_summary(artifact['results'])
            _write_artifact(artifact_path, artifact)
            _emit_stdout(json.dumps(artifact['summary'], indent=2))
            return 1

        cfg003_coverage_passed, cfg003_coverage_failures = (
            RA._validate_alias_ingress_coverage(
                alias_inventory=cfg003_inventory.get('alias_inventory', []),
                cases=cases,
                selected_cases=selected_cases,
            )
        )
        artifact['cfg003_coverage_gate'] = {
            'passed': cfg003_coverage_passed,
            'failures': cfg003_coverage_failures,
        }
        if not cfg003_coverage_passed:
            _emit_stderr(
                f"[cfg003] coverage gate FAILED: {cfg003_coverage_failures}",
                flush=True,
            )
            artifact['results']['cfg003_coverage_gate'] = {
                'passed': False,
                'skipped': False,
                'failures': cfg003_coverage_failures,
                'warnings': [],
            }
            artifact['summary'] = _build_summary(artifact['results'])
            _write_artifact(artifact_path, artifact)
            _emit_stdout(json.dumps(artifact['summary'], indent=2))
            return 1
        _write_artifact(artifact_path, artifact)

    # CFG-003: Pre-TUI snapshot validation of operator assertions.
    # Validate asserted identities against the authoritative schedule-eligible
    # read snapshot BEFORE any selected TUI case or refresh mutation.
    if args.cfg003_transactional_refresh and _cfg003_assertion_identities:
        try:
            _pre_tui_auth = RA._load_authoritative_startup_config()
            _pre_tui_eligible = RA._derive_eligible_candidates_from_snapshot(
                _pre_tui_auth["snapshot"], alias_name="read"
            )
        except Exception as _pre_tui_exc:  # noqa: BLE001
            _pre_tui_eligible = []
            _pre_tui_snapshot_failures = [
                f"pre-TUI snapshot load failed: {_pre_tui_exc}"
            ]
        else:
            _pre_tui_snapshot_failures = []
        _pre_tui_assertion_failures = _cfg003_validate_operator_assertions(
            _cfg003_assertion_identities,
            eligible_snapshot=_pre_tui_eligible,
        )
        _pre_tui_all_failures = _pre_tui_snapshot_failures + _pre_tui_assertion_failures
        if _pre_tui_all_failures:
            _emit_stderr(
                f"[cfg003] pre-TUI assertion snapshot gate FAILED: "
                f"{_pre_tui_all_failures}",
                flush=True,
            )
            artifact["results"]["cfg003_assertion_gate"] = {
                "passed": False,
                "skipped": False,
                "failures": _pre_tui_all_failures,
                "warnings": [],
            }
            artifact["summary"] = _build_summary(artifact["results"])
            _write_artifact(artifact_path, artifact)
            _emit_stdout(json.dumps(artifact["summary"], indent=2))
            return 1

    # Finding 2 (round 9): per-case error-intake validation during CFG-003.
    # Snapshot before the first selected alias case; collect and advance a
    # per-case delta after each case using case/session/trace correlation.
    # Ordinary non-CFG003 runs are unaffected.
    _cfg003_case_intake_baseline: dict[str, dict[str, Any]] | None = None
    _cfg003_case_intake_initiation: dt.datetime | None = None
    if args.cfg003_transactional_refresh:
        _cfg003_case_intake_baseline = RA._snapshot_error_intake()
        _cfg003_case_intake_initiation = dt.datetime.now(dt.timezone.utc)
    for selected_case_order, case_name in enumerate(selected_cases):
        _emit_stderr(f'[start] {case_name}', flush=True)
        case_result = _run_selected_case(
            case_name=case_name,
            case_config=cases[case_name],
            suite_config=config,
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            litellm_base_url=litellm_base_url,
            cfg003_transactional=args.cfg003_transactional_refresh,
        )
        # Finding 2 (round 9): per-case error-intake delta (CFG-003 only).
        if (
            _cfg003_case_intake_baseline is not None
            and _cfg003_case_intake_initiation is not None
        ):
            _case_session, _case_trace = _cfg003_case_correlation_ids(case_result)
            _case_environment = profile.get("expected_trace_environment", "dev")
            _case_container = profile.get("docker_container_name", "litellm-dev")
            # Finding 2/9: strict correlation for transactional alias/TUI
            # cases.  Missing required session/trace IDs must fail the case;
            # events cannot qualify only by environment/container/time.
            _is_alias_tui = bool(cases[case_name].get("verification_alias"))
            if _is_alias_tui and (not _case_session or not _case_trace):
                case_result.setdefault("failures", []).append(
                    f"strict correlation: transactional alias/TUI case missing "
                    f"required correlation IDs "
                    f"(session={_case_session!r}, trace={_case_trace!r})"
                )
                case_result["passed"] = False
            try:
                _case_intake = _cfg003_phase_error_intake(
                    _cfg003_case_intake_baseline,
                    initiation_time=_cfg003_case_intake_initiation,
                    environment=_case_environment,
                    container=_case_container,
                    case_name=case_name,
                    session_id=_case_session,
                    trace_id=_case_trace,
                    strict_correlation=True,
                )
                case_result["error_intake"] = RA._redact_sensitive_artifact_fields({
                    "baseline_summary": _case_intake["baseline_summary"],
                    "current_summary": _case_intake["current_summary"],
                    "delta_summary": _case_intake["delta_summary"],
                    "attributed_events": _case_intake["attributed_events"],
                    "attributed_count": _case_intake["attributed_count"],
                })
                _cfg003_case_intake_baseline = _case_intake["advanced_baseline"]
                if _case_intake["failures"]:
                    case_result.setdefault("failures", []).extend(
                        _case_intake["failures"]
                    )
                    case_result["passed"] = False
                if _case_intake["attributed_events"]:
                    case_result.setdefault("failures", []).append(
                        f"error intake: {_case_intake['attributed_count']} "
                        f"new attributable error(s) during case"
                    )
                    case_result["passed"] = False
            except Exception as _intake_exc:  # noqa: BLE001
                case_result.setdefault("failures", []).append(
                    f"error intake collection failure: {_intake_exc}"
                )
                case_result["passed"] = False
                _cfg003_case_intake_baseline = RA._snapshot_error_intake()
        _record_case_artifact_result(
            artifact=artifact,
            artifact_path=artifact_path,
            case_name=case_name,
            case_config=cases[case_name],
            case_result=case_result,
            selected_case_order=selected_case_order,
        )

    # CFG-003: Transactional priority-swap refresh test (opt-in, dev only).
    if args.cfg003_transactional_refresh:
        _emit_stderr('[cfg003] starting transactional priority-swap refresh test', flush=True)
        cfg003_result = _cfg003_transactional_refresh_test(
            litellm_base_url=litellm_base_url,
            cases=cases,
            suite_config=config,
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            db_settings=_cfg003_db_settings(config, profile=profile),
            environment=profile.get('expected_trace_environment', 'dev'),
            container_name=profile.get('docker_container_name', 'litellm-dev'),
            operator_assertions=_cfg003_assertion_identities or None,
        )
        artifact['cfg003_transactional_refresh'] = cfg003_result
        artifact['results']['cfg003_transactional_refresh'] = {
            'passed': cfg003_result.get('passed', False),
            'skipped': False,
            'failures': cfg003_result.get('failures', []),
            'warnings': [],
        }
        _write_artifact(artifact_path, artifact)
        _emit_stderr(
            f"[cfg003] transactional refresh passed={cfg003_result.get('passed')}",
            flush=True,
        )

    artifact['summary'] = _build_summary(artifact['results'])
    _write_artifact(artifact_path, artifact)
    _emit_stdout(json.dumps(artifact['summary'], indent=2))
    skipped_count = int(artifact['summary'].get('skipped_count') or 0)
    if skipped_count:
        skipped_cases = artifact['summary'].get('skipped_cases') or []
        _emit_stderr(
            f"[summary] skipped_cases={skipped_count}: {', '.join(skipped_cases)}",
            flush=True,
        )
    return 0 if artifact['summary']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
