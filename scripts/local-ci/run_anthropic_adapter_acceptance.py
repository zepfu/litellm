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
import urllib.request
from typing import Any

import psycopg

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'scripts' / 'local-ci' / 'anthropic_adapter_config.json'
RUN_ACCEPTANCE_PATH = ROOT / 'scripts' / 'local-ci' / 'run_acceptance.py'


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
    forbidden_substrings = list(checks.get('forbidden_substrings') or [])
    if not bool(checks.get('disable_default_429_traceback_check')):
        forbidden_substrings.extend(
            [
                'pass_through_endpoint(): Exception occured - 429:',
                'pass_through_endpoint(): Exception occured - 500:',
                'pass_through_endpoint(): Exception occured - 502:',
                'pass_through_endpoint(): Exception occured - 503:',
                'pass_through_endpoint(): Exception occured - 504:',
            ]
        )

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

    if result.returncode != 0:
        warnings.append(
            f'{family} runtime log check could not read docker logs for `{container_name}` (exit {result.returncode})'
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
    expected_rows = checks.get('expected_rows') or []

    query = '''
        select provider, model, session_id, input_tokens, output_tokens, total_tokens,
               cache_read_input_tokens, cache_creation_input_tokens,
               provider_cache_attempted, provider_cache_status,
               provider_cache_miss, provider_cache_miss_reason,
               provider_cache_miss_token_count, provider_cache_miss_cost_usd,
               reasoning_tokens_reported, reasoning_tokens_estimated,
               reasoning_tokens_source, tool_call_count, tool_names,
               file_read_count, file_modified_count, git_commit_count, git_push_count,
               response_cost_usd,
               start_time, end_time
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

    def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.isoformat() if hasattr(value, 'isoformat') else value)
            for key, value in row.items()
        }

    if expected_rows:
        matched_records: list[dict[str, Any]] = []
        for expected_row in expected_rows:
            row_provider = expected_row.get('provider')
            row_model = expected_row.get('model')
            match = next(
                (
                    row
                    for row in records
                    if (row_provider is None or row.get('provider') == row_provider)
                    and (row_model is None or row.get('model') == row_model)
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



def _run_command_with_retry(*, config: dict[str, Any]) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    retry_statuses = {int(value) for value in (config.get('retry_on_api_error_statuses') or [])}
    max_attempts = max(1, int(config.get('retry_max_attempts', 1) or 1))
    base_backoff_seconds = float(config.get('retry_backoff_seconds', 0) or 0)

    attempts: list[dict[str, Any]] = []
    final_started = RA._utcnow()
    final_run: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        started = RA._utcnow()
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
    command_session_id = RA._extract_command_session_id(run['stdout'])
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

    request_payload_summary, request_payload_failures, request_payload_warnings = RA._validate_logged_request_payload_checks(
        family=name,
        observations=raw_generation_observations,
        required_paths=(config.get('request_payload_checks') or {}).get('required_paths'),
        warning_present_paths=(config.get('request_payload_checks') or {}).get('warning_present_paths'),
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

    warning_only = bool(config.get('warning_only'))
    unique_failures = sorted(set(failures))
    unique_warnings = sorted(set(warnings))
    soft_failures = unique_failures if warning_only else []
    if warning_only and unique_failures:
        unique_warnings.extend(
            f'warning-only failure: {failure}' for failure in unique_failures
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
        'passed': (not unique_failures) or warning_only,
        'failures': [] if warning_only else unique_failures,
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


def _warning_only_error_result(family: str, exc: Exception) -> dict[str, Any]:
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
        'claude_adapter_gpt54_mini',
        'claude_adapter_ctx_marker',
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
    parser = argparse.ArgumentParser(description='Run real-Claude Anthropic adapter acceptance checks through litellm-dev.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Path to adapter suite config JSON.')
    parser.add_argument('--write-artifact', required=True, help='Where to write the JSON artifact.')
    parser.add_argument('--langfuse-query-url', default=None, help='Override Langfuse query URL.')
    parser.add_argument('--cases', default=None, help='Comma-separated subset of adapter cases to run.')
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    artifact_path = pathlib.Path(args.write_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    config = _resolve_env_placeholders(RA._load_json(config_path))

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

    litellm_base_url = config.get('litellm_base_url', 'http://127.0.0.1:4001')

    artifact: dict[str, Any] = {
        'suite_version': config.get('suite_version', 1),
        'timestamp': RA._isoformat(RA._utcnow()),
        'git_commit': RA._git_value('rev-parse', 'HEAD'),
        'git_branch': RA._git_value('branch', '--show-current'),
        'environment': {
            'litellm_base_url': litellm_base_url,
            'langfuse_query_url': query_url,
            'docker_litellm_dev_status': RA._docker_status(),
        },
        'results': {},
        'summary': {},
    }
    artifact['summary'] = _build_summary(artifact['results'])
    _write_artifact(artifact_path, artifact)

    for case_name in selected_cases:
        print(f'[start] {case_name}', file=sys.stderr, flush=True)
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
            if bool(cases[case_name].get('warning_only')):
                artifact['results'][case_name] = _warning_only_error_result(
                    case_name, exc
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
