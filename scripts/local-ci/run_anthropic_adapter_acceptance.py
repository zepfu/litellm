#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
]


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
        except (KeyError, ValueError):
            return value
    return value


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
    tenant_config = f'model_providers.{{codex_profile}}.http_headers.x-aawm-tenant-id="{tenant_id}"'
    if any(str(item) == tenant_config for item in command):
        return command
    updated = list(command)
    try:
        insert_at = updated.index('--json')
    except ValueError:
        insert_at = max(0, len(updated) - 1)
    updated[insert_at:insert_at] = ['-c', tenant_config]
    return updated


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
    codex_profile = 'litellm' if target == 'prod' else 'litellm-dev'
    context = {
        'target': target,
        'case_name': case_name,
        'harness_user_id': harness_user_id,
        'session_id': session_id,
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
    ]
    if cli_kind == 'codex':
        controlled_headers.append(('session_id', session_id))
        command = updated.get('command')
        if isinstance(command, list):
            updated['command'] = _format_harness_template(
                _append_codex_tenant_config_arg(command, tenant_id),
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


def _validate_command_output_json(*, family: str, stdout: str, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    parsed = _parse_command_output_json(stdout)
    if parsed is None:
        return {'parsed': None}, [f'{family} command stdout did not contain JSON']

    required_equals = checks.get('required_equals', {}) or {}
    required_contains = checks.get('required_contains', {}) or {}
    required_minimums = checks.get('required_minimums', {}) or {}

    equals_hits: dict[str, Any] = {}
    contains_hits: dict[str, Any] = {}
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

    for path, minimum in required_minimums.items():
        actual = _extract_path_value(parsed, path)
        minimum_hits[path] = actual
        if not isinstance(actual, (int, float)) or actual < minimum:
            failures.append(f'{family} command JSON below minimum for `{path}`: expected >= {minimum!r}, got {actual!r}')

    return {
        'parsed': parsed,
        'required_equals_hits': equals_hits,
        'required_contains_hits': contains_hits,
        'required_minimum_hits': minimum_hits,
    }, failures



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
        return {'record': None}, [f'{family} missing DB password for session_history validation']

    expected_provider = checks.get('expected_provider')
    expected_model = checks.get('expected_model')
    expected_tenant_id = checks.get('expected_tenant_id')
    expected_rows = checks.get('expected_rows') or []
    expected_litellm_environment = checks.get('expected_litellm_environment')
    require_runtime_identity = checks.get('require_runtime_identity', True) is not False

    query = '''
        select provider, model, session_id, tenant_id,
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
               metadata, start_time, end_time
        from public.session_history
        where session_id = %s
        order by start_time desc
    '''
    with psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        connect_timeout=10,
        row_factory=psycopg.rows.dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            records = cur.fetchall()

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
        matched_records: list[dict[str, Any]] = []

        def _record_matches_expected(
            row: dict[str, Any], expected_row: dict[str, Any]
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
            for key, minimum in (expected_row.get('minimums') or {}).items():
                actual = row.get(key)
                if not isinstance(actual, (int, float)) or actual < minimum:
                    return False
            return True

        for expected_row in expected_rows:
            row_provider = expected_row.get('provider')
            row_model = expected_row.get('model')
            match = next(
                (
                    row
                    for row in records
                    if _record_matches_expected(row, expected_row)
                ),
                None,
            )
            if match is None:
                failures.append(
                    f'{family} missing session_history row for provider={row_provider!r} model={row_model!r}'
                )
                continue
            matched_records.append(_normalize_record(match))
            for key, minimum in (expected_row.get('minimums') or {}).items():
                actual = match.get(key)
                if not isinstance(actual, (int, float)) or actual < minimum:
                    failures.append(
                        f'{family} session_history row provider={row_provider!r} model={row_model!r} `{key}` below minimum: expected >= {minimum!r}, got {actual!r}'
                    )

        return {'record': matched_records[0] if matched_records else None, 'records': matched_records}, failures

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

    return {'record': normalized_record, 'records': [_normalize_record(row) for row in records]}, failures


def _validate_tool_activity(*, family: str, session_id: str | None, checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not session_id:
        return {'record': None, 'records': []}, [f'{family} missing command session_id for tool_activity validation']

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
        return {'record': None, 'records': []}, [f'{family} missing DB password for tool_activity validation']

    query = '''
        select provider, model, tool_index, tool_name, tool_kind, command_text,
               arguments, metadata, created_at
        from public.session_history_tool_activity
        where session_id = %s
        order by created_at asc, tool_index asc
    '''
    with psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        connect_timeout=10,
        row_factory=psycopg.rows.dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            records = cur.fetchall()

    def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.isoformat() if hasattr(value, 'isoformat') else value)
            for key, value in row.items()
        }

    failures: list[str] = []
    expected_rows = checks.get('expected_rows') or []
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
        command_text_contains = expected_row.get('command_text_contains')
        if isinstance(command_text_contains, str) and command_text_contains:
            if not any(command_text_contains in str(row.get('command_text') or '') for row in matches):
                failures.append(
                    f'{family} tool_activity rows for provider={row_provider!r} model={row_model!r} tool_name={row_tool_name!r} did not include command text containing {command_text_contains!r}'
                )
        matched_records.extend(_normalize_record(row) for row in matches[:minimum_count])

    return {
        'record': matched_records[0] if matched_records else None,
        'records': [_normalize_record(row) for row in records],
        'matched_records': matched_records,
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

    headers = {str(key): str(value) for key, value in (request_config.get('headers') or {}).items()}
    body = request_config.get('json')
    session_id = str(request_config.get('session_id') or '')
    if session_id:
        body = _inject_http_litellm_metadata(
            body,
            session_id=session_id,
            trace_name=str(headers.get('langfuse_trace_name') or config.get('case_name') or 'native-passthrough'),
        )

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
            run = _run_http_request(config)
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
            user_id=expected_user_ids[0] if expected_user_ids else None,
            start_time=started,
            limit=100,
            timeout_seconds=int(config.get('langfuse_poll_timeout_seconds', 60)),
        )
    elif can_session_trace_lookup:
        traces, lookup_error = RA._poll_langfuse_session_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            user_id=expected_user_ids[0] if expected_user_ids else None,
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
            'aawm_dynamic_injection': aawm_dynamic_injection_summary,
            'span_observations': span_observations,
            'generation_observations': generation_observations,
        },
        'command_json': command_json_summary,
        'session_history': session_history_summary,
        'tool_activity': tool_activity_summary,
        'runtime_postconditions': runtime_summary,
        'runtime_logs': runtime_log_summary,
        'passed': not hard_failures,
        'failures': hard_failures,
        'soft_failures': soft_failures,
        'warnings': unique_warnings,
    }


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for family, result in results.items():
        for failure in result.get('failures', []):
            failures.append(f'{family}: {failure}')
        for warning in result.get('warnings', []):
            warnings.append(f'{family}: {warning}')
    return {'passed': not failures, 'failures': failures, 'warnings': warnings}


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
