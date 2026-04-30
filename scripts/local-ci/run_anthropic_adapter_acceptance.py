#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
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
    },
    'prod': {
        'litellm_base_url': 'http://127.0.0.1:4000',
        'anthropic_base_url': 'http://127.0.0.1:4000/anthropic',
        'docker_container_name': 'aawm-litellm',
        'expected_trace_environment': 'prod',
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
    'timed out after',
    'runtime logs contained forbidden substring',
    'successful empty',
]
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
    for obj in RA._parse_stdout_json_objects(stdout):
        if isinstance(obj, dict):
            return obj
    return None


def _extract_command_session_id(stdout: str) -> str | None:
    direct = RA._extract_command_session_id(stdout)
    if direct:
        return direct

    def _walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ('session_id', 'sessionId', 'thread_id', 'threadId'):
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


def _append_comma_headers(existing: str | None, headers: list[tuple[str, str]]) -> str:
    parts = [existing.strip()] if isinstance(existing, str) and existing.strip() else []
    parts.extend(f'{key}: {value}' for key, value in headers)
    return ', '.join(parts)


def _docker_status_for_container(container_name: str) -> str:
    result = subprocess.run(
        ['docker', 'ps', '--filter', f'name=^{container_name}$', '--format', '{{.Status}}'],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
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
    profile['litellm_base_url'] = profile['litellm_base_url'].rstrip('/')
    profile['anthropic_base_url'] = profile['anthropic_base_url'].rstrip('/')
    return profile


def _apply_target_profile_to_config(
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
        updated_case['expected_trace_environment'] = profile['expected_trace_environment']
        updated_case.setdefault('require_trace_user_id', True)
        updated_case['target_profile'] = target
        updated_case['case_name'] = case_name
        tenant_id = _resolve_harness_tenant_id(updated_config, updated_case)
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
            profile['expected_trace_environment'],
        )
        metadata_required_equals = dict(
            session_history_validation.get('metadata_required_equals') or {}
        )
        metadata_required_equals['trace_environment'] = profile[
            'expected_trace_environment'
        ]
        metadata_required_equals['litellm_environment'] = profile[
            'expected_trace_environment'
        ]
        metadata_required_equals.setdefault('tenant_id', tenant_id)
        session_history_validation['metadata_required_equals'] = (
            metadata_required_equals
        )
        metadata_required_truthy = list(
            session_history_validation.get('metadata_required_truthy') or []
        )
        if 'tenant_id_source' not in metadata_required_truthy:
            metadata_required_truthy.append('tenant_id_source')
        session_history_validation['metadata_required_truthy'] = metadata_required_truthy
        expected_rows = session_history_validation.get('expected_rows')
        has_expected_rows = isinstance(expected_rows, list) and bool(expected_rows)
        if has_expected_rows:
            session_history_validation['expected_rows'] = [
                _with_expected_row_tenant(row, tenant_id)
                for row in expected_rows
            ]
        else:
            session_history_validation.setdefault('expected_tenant_id', tenant_id)
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
        updated_cases[case_name] = updated_case

    updated_config['cases'] = updated_cases
    return updated_config


def _resolve_harness_tenant_id(
    suite_config: dict[str, Any],
    case_config: dict[str, Any],
) -> str:
    value = case_config.get('tenant_id', suite_config.get('default_tenant_id'))
    if isinstance(value, (str, int, float)) and str(value).strip():
        return str(value).strip()
    return 'adapter-harness-tenant'


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
    harness_user_id = (
        expected_user_ids[0]
        if expected_user_ids
        else RA._build_claude_harness_user_id(target=target, case_name=case_name)
    )
    session_id = str(
        request_config.get('session_id')
        or updated.get('expected_trace_session_id')
        or f'{harness_user_id}.session'
    )
    if request_config.get('add_default_authorization') is not False:
        headers.setdefault('authorization', 'Bearer sk-1234')
    headers.setdefault('x-litellm-end-user-id', harness_user_id)
    headers.setdefault('langfuse_trace_user_id', harness_user_id)
    headers.setdefault('langfuse_trace_name', case_name)
    headers.setdefault('session_id', session_id)
    headers.setdefault('x-aawm-tenant-id', tenant_id)
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
    if cli_kind not in {'codex', 'gemini'}:
        return updated

    tenant_id = str(updated.get('tenant_id') or 'adapter-harness-tenant')
    expected_user_ids = [
        str(value).strip()
        for value in (updated.get('expected_user_ids') or [])
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    harness_user_id = (
        expected_user_ids[0]
        if expected_user_ids
        else RA._build_claude_harness_user_id(target=target, case_name=case_name)
    )
    session_id = str(
        updated.get('expected_trace_session_id') or f'{harness_user_id}.session'
    )
    repository = _resolve_harness_repository()
    codex_profile = 'litellm' if target == 'prod' else 'litellm-dev'
    context = {
        'target': target,
        'case_name': case_name,
        'harness_user_id': harness_user_id,
        'session_id': session_id,
        'repository': repository,
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
    if cli_kind == 'gemini':
        env['CODE_ASSIST_ENDPOINT'] = f"{profile['litellm_base_url']}/gemini"
        env['GOOGLE_GENAI_USE_GCA'] = 'true'
        env['GEMINI_CLI_CUSTOM_HEADERS'] = _append_comma_headers(
            env.get('GEMINI_CLI_CUSTOM_HEADERS'),
            controlled_headers,
        )
    updated['env'] = env
    updated['expected_user_ids'] = [harness_user_id]
    if cli_kind == 'codex':
        updated['expected_trace_session_id'] = session_id
    else:
        updated.pop('expected_trace_session_id', None)
    if updated.get('match_trace_session_id_from_stdout') is None:
        updated['match_trace_session_id_from_stdout'] = True
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


def _validate_logged_request_payload_checks(
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
    required_one_of_found: dict[str, bool] = {
        str(path): False for path in required_one_of
    }
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
    }


def _stream_tool_state_matches_expected(
    item: dict[str, Any],
    expected: dict[str, Any],
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
        if len(matches) < minimum_count:
            failures.append(
                f'{family} missing Responses stream tool state for {expected!r}; expected >= {minimum_count}, got {len(matches)}'
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
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'name=^{container_name}$', '--format', '{{.Status}}'],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        docker_status = result.stdout.strip() if result.returncode == 0 else ''
        summary['docker_status'] = docker_status
        if not docker_status:
            failures.append(f'{family} runtime container `{container_name}` not found')
        elif docker_status.lower().startswith('exited') or docker_status.lower().startswith('dead'):
            failures.append(f'{family} runtime container `{container_name}` is down: {docker_status}')

    return summary, failures


def _read_runtime_logs_since(
    *,
    started: Any,
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
        'log_excerpt': '',
    }
    if not container_name:
        return summary, ''

    since_value = started.isoformat() if hasattr(started, 'isoformat') else str(started)
    result = subprocess.run(
        ['docker', 'logs', '--since', since_value, '--tail', str(tail_lines), container_name],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = '\n'.join(
        value for value in (result.stdout, result.stderr) if isinstance(value, str) and value
    )
    summary['docker_logs_exit_code'] = result.returncode
    summary['log_excerpt'] = log_text[-4000:] if log_text else ''
    return summary, log_text


def _validate_runtime_logs(
    *,
    family: str,
    started: Any,
    checks: dict[str, Any],
    runtime_postconditions: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    container_name = (
        checks.get('docker_container_name')
        or runtime_postconditions.get('docker_container_name')
        or 'litellm-dev'
    )
    tail_lines = int(checks.get('tail_lines') or 400)
    forbidden_substrings = [
        *DEFAULT_RUNTIME_LOG_FORBIDDEN_SUBSTRINGS,
        *list(checks.get('forbidden_substrings') or []),
    ]
    if not bool(checks.get('disable_default_429_traceback_check')):
        forbidden_substrings.extend(DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS)
    if bool(checks.get('disable_default_error_signature_check')):
        configured_substrings = list(checks.get('forbidden_substrings') or [])
        forbidden_substrings = configured_substrings
        if not bool(checks.get('disable_default_429_traceback_check')):
            forbidden_substrings.extend(DEFAULT_RUNTIME_LOG_UPSTREAM_ERROR_SUBSTRINGS)
    forbidden_substrings = sorted(set(forbidden_substrings))

    summary: dict[str, Any] = {
        'docker_container_name': container_name,
        'tail_lines': tail_lines,
        'forbidden_substrings': forbidden_substrings,
        'matched_forbidden_substrings': [],
    }
    failures: list[str] = []
    warnings: list[str] = []

    if not container_name or not forbidden_substrings:
        return summary, failures, warnings

    log_summary, log_text = _read_runtime_logs_since(
        started=started,
        checks={'docker_container_name': container_name, 'tail_lines': tail_lines},
        runtime_postconditions=runtime_postconditions,
    )
    summary['docker_logs_exit_code'] = log_summary.get('docker_logs_exit_code')
    summary['log_excerpt'] = log_summary.get('log_excerpt', '')

    if summary['docker_logs_exit_code'] != 0:
        warnings.append(
            f'{family} runtime log check could not read docker logs for `{container_name}` (exit {summary["docker_logs_exit_code"]})'
        )
        return summary, failures, warnings

    matched = [
        substring for substring in forbidden_substrings if substring and substring in log_text
    ]
    summary['matched_forbidden_substrings'] = matched
    for substring in matched:
        failures.append(
            f'{family} runtime logs contained forbidden substring `{substring}`'
        )

    return summary, failures, warnings


def _validate_session_history(*, family: str, session_id: str | None, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
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

    query = '''
        select provider, model, session_id, tenant_id, repository,
               input_tokens, output_tokens, total_tokens,
               cache_read_input_tokens, cache_creation_input_tokens,
               provider_cache_attempted, provider_cache_status,
               provider_cache_miss, provider_cache_miss_reason,
               provider_cache_miss_token_count, provider_cache_miss_cost_usd,
               reasoning_tokens_reported, reasoning_tokens_estimated,
               reasoning_tokens_source, tool_call_count, tool_names,
               file_read_count, file_modified_count, git_commit_count, git_push_count,
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
        where session_id = %s
        order by start_time desc
    '''
    conn = _validation_db_connection(db_settings)
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    while True:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
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
        if row_provider in {'anthropic', 'openai', 'openrouter', 'gemini'}:
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
    if isinstance(checks.get('db_password_env'), str):
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


def _validate_tool_activity(*, family: str, session_id: str | None, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not session_id:
        return {'record': None, 'records': []}, [f'{family} missing command session_id for tool_activity validation']

    db_settings, db_failures = _validation_db_settings(
        family=family,
        checks=checks,
        validation_name='tool_activity',
    )
    if db_settings is None:
        return {'record': None, 'records': []}, db_failures

    query = '''
        select provider, model, tool_index, tool_name, tool_kind, command_text,
               arguments, metadata, created_at
        from public.session_history_tool_activity
        where session_id = %s
        order by created_at asc, tool_index asc
    '''
    conn = _validation_db_connection(db_settings)
    expected_rows = checks.get('expected_rows') or []
    poll_timeout_seconds = max(0.0, float(checks.get('poll_timeout_seconds') or 0))
    poll_interval_seconds = max(0.1, float(checks.get('poll_interval_seconds') or 1))
    poll_deadline = time.monotonic() + poll_timeout_seconds
    while True:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
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
                if (row_provider is None or row.get('provider') == row_provider)
                and (row_model is None or row.get('model') == row_model)
                and (row_tool_name is None or row.get('tool_name') == row_tool_name)
                and (row_tool_kind is None or row.get('tool_kind') == row_tool_kind)
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
            and (row_tool_name is None or row.get('tool_name') == row_tool_name)
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
        or '/home/zepfu/.claude/projects'
    )
    return pathlib.Path(str(configured)).expanduser()


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


def _summarize_transcript_tool_uses(paths: list[pathlib.Path]) -> dict[str, Any]:
    by_tool_name: dict[str, int] = {}
    by_assistant_message_id: dict[str, dict[str, int]] = {}
    records: list[dict[str, Any]] = []
    records_by_tool_use_id: dict[str, dict[str, Any]] = {}
    tool_result_errors: list[dict[str, Any]] = []
    transcript_summaries: list[dict[str, Any]] = []
    for path in paths:
        transcript_tool_count = 0
        for line_number, record in _iter_claude_jsonl(path):
            message = record.get('message')
            if not isinstance(message, dict):
                continue
            content = message.get('content')
            if not isinstance(content, list):
                continue
            if message.get('role') == 'user':
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
                    if is_error:
                        tool_result_errors.append(tool_result_record)
                continue
            if message.get('role') != 'assistant':
                continue
            message_id = str(message.get('id') or '')
            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'tool_use':
                    continue
                tool_name = str(block.get('name') or '')
                if not tool_name:
                    continue
                by_tool_name[tool_name] = by_tool_name.get(tool_name, 0) + 1
                message_counts = by_assistant_message_id.setdefault(message_id, {})
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
        'max_tool_uses_in_single_assistant_message': max_tools_in_message,
        'total_tool_uses': len(records),
        'tool_result_errors': tool_result_errors,
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
        'forbid_tool_result_errors': checks.get('forbid_tool_result_errors'),
        'expected_tool_sequence': checks.get('expected_tool_sequence'),
        'require_tool_result_before_next_tool_use': checks.get(
            'require_tool_result_before_next_tool_use'
        ),
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


def _validate_transcript_agent_tool_uses(
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


def _run_http_request(config: dict[str, Any]) -> dict[str, Any]:
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


def _validate_case(name: str, config: dict[str, Any], *, query_url: str, public_key: str, secret_key: str, litellm_base_url: str) -> dict[str, Any]:
    started, run, command_attempts = _run_command_with_retry(config=config)
    command_session_id = _extract_command_session_id(run['stdout'])
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
    if expected_trace_names:
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

    failures: list[str] = []
    warnings: list[str] = []
    if run['exit_code'] != 0:
        failures.append(f'{name} command failed')
    if lookup_error:
        warnings.append(f'{name} Langfuse lookup warning: {lookup_error}')
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
    if bool(config.get('require_trace_user_id')) and traces and not actual_user_ids:
        failures.append(f'{name} traces did not include a Langfuse userId')

    raw_generation_observations, generation_observations, generation_failures = RA._validate_generation_observations(
        family=name,
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=config.get('allowed_generation_routes'),
        skip_quality_checks=bool(config.get('skip_generation_quality_checks')),
        allow_zero_cost=bool(config.get('allow_zero_cost')),
    )
    failures.extend(generation_failures)

    filtered_trace_ids = sorted(
        {
            observation.get('traceId')
            for observation in raw_generation_observations
            if isinstance(observation.get('traceId'), str)
        }
    )
    filtered_traces = [trace for trace in traces if trace.get('id') in set(filtered_trace_ids)]

    trace_enrichment_summary, trace_enrichment_failures, trace_enrichment_warnings = RA._validate_trace_enrichment(
        family=name,
        traces=filtered_traces,
        required_tags=config.get('required_trace_tags'),
        required_tag_prefixes=config.get('required_trace_tag_prefixes'),
        warning_tag_prefixes=config.get('warning_trace_tag_prefixes'),
    )
    failures.extend(trace_enrichment_failures)
    warnings.extend(trace_enrichment_warnings)

    trace_context_summary, trace_context_failures = RA._validate_trace_context(
        family=name,
        traces=filtered_traces,
        expected_environment=config.get('expected_trace_environment'),
        require_trace_session_id=bool(config.get('require_trace_session_id')),
        expected_trace_session_id=(command_session_id if config.get('match_trace_session_id_from_stdout') else config.get('expected_trace_session_id')),
        require_trace_ids_distinct_from_session_ids=bool(config.get('require_trace_ids_distinct_from_session_ids')),
    )
    failures.extend(trace_context_failures)

    generation_metadata_summary, generation_metadata_failures = RA._validate_generation_metadata(
        family=name,
        observations=raw_generation_observations,
        required_metadata_truthy=config.get('required_generation_metadata_truthy'),
        required_metadata_minimums=config.get('required_generation_metadata_minimums'),
    )
    failures.extend(generation_metadata_failures)

    request_payload_summary, request_payload_failures, request_payload_warnings = _validate_logged_request_payload_checks(
        family=name,
        observations=raw_generation_observations,
        checks=config.get('request_payload_checks') or {},
    )
    failures.extend(request_payload_failures)
    warnings.extend(request_payload_warnings)

    request_text_summary, request_text_failures, request_text_warnings = RA._validate_logged_request_text_checks(
        family=name,
        observations=raw_generation_observations,
        required_substrings=(config.get('request_text_checks') or {}).get('required_substrings'),
        forbidden_substrings=(config.get('request_text_checks') or {}).get('forbidden_substrings'),
        warning_required_substrings=(config.get('request_text_checks') or {}).get('warning_required_substrings'),
    )
    failures.extend(request_text_failures)
    warnings.extend(request_text_warnings)

    stream_tool_call_state_summary, stream_tool_call_state_failures = _validate_stream_tool_call_state(
        family=name,
        observations=raw_generation_observations,
        checks=config.get('stream_tool_call_state_validation') or {},
    )
    failures.extend(stream_tool_call_state_failures)

    aawm_dynamic_injection_summary = None
    aawm_dynamic_injection_config = config.get('aawm_dynamic_injection')
    if isinstance(aawm_dynamic_injection_config, dict):
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
    empty_success_summary, empty_success_failures = (
        _validate_no_successful_empty_command_output(
            family=name,
            stdout=run['stdout'],
            stderr=run['stderr'],
            checks=config,
        )
    )
    failures.extend(empty_success_failures)

    session_history_summary, session_history_failures = _validate_session_history(
        family=name,
        session_id=command_session_id,
        checks=config.get('session_history_validation') or {},
    )
    failures.extend(session_history_failures)
    tool_activity_summary, tool_activity_failures = _validate_tool_activity(
        family=name,
        session_id=command_session_id,
        checks=config.get('tool_activity_validation') or {},
    ) if config.get('tool_activity_validation') else ({'record': None, 'records': []}, [])
    failures.extend(tool_activity_failures)
    transcript_tool_use_summary, transcript_tool_use_failures = _validate_transcript_tool_use(
        family=name,
        session_id=command_session_id,
        checks=config.get('transcript_tool_use_validation') or {},
    ) if config.get('transcript_tool_use_validation') else ({'agents': []}, [])
    failures.extend(transcript_tool_use_failures)

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
    )
    failures.extend(runtime_log_failures)
    warnings.extend(runtime_log_warnings)
    failures, downgraded_warnings = _downgrade_configured_failures_to_warnings(
        failures=failures,
        config=config,
        command_json_summary=command_json_summary,
    )
    warnings.extend(downgraded_warnings)

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
        'streaming_checked': config.get('streaming_checked', False),
        'warning_only': warning_only,
        'command_attempts': command_attempts,
        'langfuse': {
            'required_trace_names': expected_trace_names,
            'actual_trace_names': actual_trace_names,
            'expected_user_ids': expected_user_ids,
            'actual_user_ids': actual_user_ids,
            'expected_trace_user_ids_by_name': expected_trace_user_ids_by_name,
            'trace_user_ids_by_name': trace_user_ids_by_name_summary,
            'trace_ids': trace_ids,
            'trace_count': len(traces),
            'lookup_error': lookup_error,
            'filtered_trace_ids': filtered_trace_ids,
            'command_session_id': command_session_id,
            'trace_context': trace_context_summary,
            'trace_enrichment': trace_enrichment_summary,
            'generation_metadata': generation_metadata_summary,
            'request_payload_checks': request_payload_summary,
            'request_text_checks': request_text_summary,
            'stream_tool_call_state': stream_tool_call_state_summary,
            'aawm_dynamic_injection': aawm_dynamic_injection_summary,
            'span_observations': span_observations,
            'generation_observations': generation_observations,
        },
        'command_json': command_json_summary,
        'empty_success': empty_success_summary,
        'session_history': session_history_summary,
        'tool_activity': tool_activity_summary,
        'transcript_tool_use': transcript_tool_use_summary,
        'runtime_postconditions': runtime_summary,
        'runtime_logs': runtime_log_summary,
        'passed': not hard_failures,
        'failures': hard_failures,
        'soft_failures': soft_failures,
        'warnings': unique_warnings,
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
            'input-token share; session_history does not yet store exact input cost'
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


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for family, result in results.items():
        for failure in result.get('failures', []):
            failures.append(f'{family}: {failure}')
        for warning in result.get('warnings', []):
            warnings.append(f'{family}: {warning}')
    return {
        'passed': not failures,
        'failures': failures,
        'warnings': warnings,
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

    log_excerpt = runtime_logs.get('log_excerpt')
    if not isinstance(log_excerpt, str) or not log_excerpt:
        return failures, [], warnings, runtime_logs

    matched_substrings = [
        substring for substring in required_substrings if substring in log_excerpt
    ]
    runtime_logs = {
        **runtime_logs,
        'required_soft_fail_substrings': required_substrings,
        'matched_soft_fail_substrings': matched_substrings,
    }
    if len(matched_substrings) != len(required_substrings):
        return failures, [], warnings, runtime_logs

    soft_failures = list(failures)
    soft_warnings = [
        f'provider-unavailable soft-fail: {failure}' for failure in soft_failures
    ]
    return [], soft_failures, sorted(set([*warnings, *soft_warnings])), runtime_logs


def _write_artifact(path: pathlib.Path, artifact: dict[str, Any]) -> None:
    path.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')


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
        'claude_adapter_gemini_fanout',
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


def main() -> int:
    parser = argparse.ArgumentParser(description='Run real-Claude Anthropic adapter acceptance checks through a target LiteLLM instance.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Path to adapter suite config JSON.')
    parser.add_argument('--write-artifact', required=True, help='Where to write the JSON artifact.')
    parser.add_argument('--langfuse-query-url', default=None, help='Override Langfuse query URL.')
    parser.add_argument('--cases', default=None, help='Comma-separated subset of adapter cases to run.')
    parser.add_argument(
        '--target',
        default=os.environ.get('AAWM_ADAPTER_TARGET', None),
        help='Target profile to test. Built-ins: dev (:4001/litellm-dev), prod (:4000/aawm-litellm).',
    )
    parser.add_argument('--litellm-base-url', default=None, help='Override the target LiteLLM base URL.')
    parser.add_argument('--anthropic-base-url', default=None, help='Override ANTHROPIC_BASE_URL passed to Claude CLI.')
    parser.add_argument('--docker-container-name', default=None, help='Override the Docker container used for health/log checks.')
    parser.add_argument('--expected-trace-environment', default=None, help='Override expected Langfuse trace environment.')
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    artifact_path = pathlib.Path(args.write_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
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

    public_key_env = config.get('langfuse_public_key_env', 'LANGFUSE_PUBLIC_KEY')
    secret_key_env = config.get('langfuse_secret_key_env', 'LANGFUSE_SECRET_KEY')
    public_key = os.environ.get(public_key_env, '')
    secret_key = os.environ.get(secret_key_env, '')
    query_url = args.langfuse_query_url or os.environ.get('LANGFUSE_QUERY_URL') or config.get('langfuse_query_url', 'http://127.0.0.1:3000')

    if not public_key or not secret_key:
        print(f'Missing Langfuse credentials in env vars {public_key_env}/{secret_key_env}', file=sys.stderr)
        return 2

    cases = config.get('cases') or {}
    available_cases = list(cases.keys())
    selected_cases = _parse_selected_cases(
        args.cases,
        available_cases,
        default_excluded_cases=config.get('default_excluded_cases'),
    )

    litellm_base_url = config.get('litellm_base_url', profile['litellm_base_url'])

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
        'summary': {},
    }
    artifact['summary'] = _build_summary(artifact['results'])
    _write_artifact(artifact_path, artifact)

    for case_name in selected_cases:
        print(f'[start] {case_name}', file=sys.stderr, flush=True)
        case_started = RA._utcnow()
        try:
            missing_required_env = _missing_required_env(cases[case_name])
            if missing_required_env:
                artifact['results'][case_name] = {
                    'passed': True,
                    'skipped': True,
                    'failures': [],
                    'soft_failures': [],
                    'warnings': [
                        f'missing required env: {", ".join(sorted(missing_required_env))}'
                    ],
                }
                continue
            artifact['results'][case_name] = _validate_case(
                case_name,
                cases[case_name],
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                litellm_base_url=litellm_base_url,
            )
        except Exception as exc:
            provider_unavailable_timeout = _provider_unavailable_timeout_error_result(
                case_name,
                exc,
                cases[case_name],
                started=case_started,
            )
            if provider_unavailable_timeout is not None:
                artifact['results'][case_name] = provider_unavailable_timeout
            elif bool(cases[case_name].get('warning_only')):
                artifact['results'][case_name] = _warning_only_error_result(
                    case_name,
                    exc,
                    cases[case_name],
                )
            else:
                artifact['results'][case_name] = RA._family_error_result(
                    case_name, exc
                )
        finally:
            artifact['summary'] = _build_summary(artifact['results'])
            _write_artifact(artifact_path, artifact)
            case_result = artifact['results'].get(case_name, {})
            print(
                f"[done] {case_name} passed={case_result.get('passed')} "
                f"skipped={case_result.get('skipped', False)} "
                f"failures={len(case_result.get('failures', []))} "
                f"warnings={len(case_result.get('warnings', []))}",
                file=sys.stderr,
                flush=True,
            )

    artifact['summary'] = _build_summary(artifact['results'])
    _write_artifact(artifact_path, artifact)
    print(json.dumps(artifact['summary'], indent=2))
    return 0 if artifact['summary']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
