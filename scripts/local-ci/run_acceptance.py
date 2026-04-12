#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts" / "local-ci" / "config.json"
_CLAUDE_AGENT_NAME_RE = re.compile(r"You are '([^']+)' and you are working")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _isoformat(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _http_get_json(url: str, public_key: str, secret_key: str, timeout: float = 20.0) -> dict[str, Any]:
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
        },
        method="GET",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            raise
        except urllib.error.HTTPError as exc:
            last_error = exc
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("unexpected langfuse query failure")


def _parse_langfuse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _recent_langfuse_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    name: str | None,
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
) -> list[dict[str, Any]]:
    params = {
        "limit": str(limit),
        "fields": "core",
        "orderBy": "timestamp.desc",
        "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
    }
    if name:
        params["name"] = name
    if user_id:
        params["userId"] = user_id
    url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, public_key, secret_key)
    traces = payload.get("data", [])
    recent: list[dict[str, Any]] = []
    floor = start_time - dt.timedelta(seconds=5)
    for trace in traces:
        timestamp = _parse_langfuse_timestamp(
            trace.get("timestamp") or trace.get("createdAt") or trace.get("updatedAt")
        )
        if timestamp is None or timestamp < floor:
            continue
        recent.append(trace)
    return recent


def _recent_langfuse_all_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    user_id: str | None,
    start_time: dt.datetime,
    session_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    params = {
        "limit": str(limit),
        "fields": "core",
        "orderBy": "timestamp.desc",
        "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
    }
    if user_id:
        params["userId"] = user_id
    if session_id:
        params["sessionId"] = session_id
    url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, public_key, secret_key)
    traces = payload.get("data", [])
    recent: list[dict[str, Any]] = []
    floor = start_time - dt.timedelta(seconds=5)
    for trace in traces:
        timestamp = _parse_langfuse_timestamp(
            trace.get("timestamp") or trace.get("createdAt") or trace.get("updatedAt")
        )
        if timestamp is None or timestamp < floor:
            continue
        recent.append(trace)
    return recent


def _poll_langfuse_session_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    user_id: str | None,
    start_time: dt.datetime,
    session_id: str,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], str | None]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    last_error: str | None = None
    while True:
        try:
            traces = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                session_id=session_id,
                limit=100,
            )
            last_error = None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as exc:
            traces = []
            last_error = str(exc)
        if traces:
            return traces, last_error
        if time.time() >= deadline:
            return traces, last_error
        time.sleep(interval_seconds)


def _recent_langfuse_generation_observations_for_trace_ids(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    limit_per_trace: int = 10,
) -> list[dict[str, Any]]:
    floor = start_time - dt.timedelta(seconds=5)
    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for trace_id in trace_ids:
        params = {
            "traceId": trace_id,
            "type": "GENERATION",
            "limit": str(limit_per_trace),
            "orderBy": "startTime.desc",
            "fields": "core",
        }
        url = f"{query_url.rstrip('/')}/api/public/observations?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, public_key, secret_key)
        for observation in payload.get("data", []):
            observation_id = observation.get("id")
            if isinstance(observation_id, str) and observation_id in seen_ids:
                continue
            timestamp = _parse_langfuse_timestamp(
                observation.get("startTime")
                or observation.get("createdAt")
                or observation.get("updatedAt")
            )
            if timestamp is None or timestamp < floor:
                continue
            if isinstance(observation_id, str):
                seen_ids.add(observation_id)
            observations.append(observation)
    return observations


def _recent_langfuse_span_observations_for_trace_ids(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    limit_per_trace: int = 25,
) -> list[dict[str, Any]]:
    floor = start_time - dt.timedelta(seconds=5)
    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for trace_id in trace_ids:
        params = {
            "traceId": trace_id,
            "type": "SPAN",
            "limit": str(limit_per_trace),
            "orderBy": "startTime.desc",
            "fields": "core",
        }
        url = f"{query_url.rstrip('/')}/api/public/observations?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, public_key, secret_key)
        for observation in payload.get("data", []):
            observation_id = observation.get("id")
            if isinstance(observation_id, str) and observation_id in seen_ids:
                continue
            timestamp = _parse_langfuse_timestamp(
                observation.get("startTime")
                or observation.get("createdAt")
                or observation.get("updatedAt")
            )
            if timestamp is None or timestamp < floor:
                continue
            if isinstance(observation_id, str):
                seen_ids.add(observation_id)
            observations.append(observation)
    return observations


def _extract_generation_metric(
    observation: dict[str, Any], *path: str
) -> Any:
    current: Any = observation
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _validate_generation_observations(
    *,
    family: str,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    allowed_request_routes: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    if not trace_ids:
        return [], [], [f"{family} missing trace ids for generation validation"]

    try:
        observations = _recent_langfuse_generation_observations_for_trace_ids(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            trace_ids=trace_ids,
            start_time=start_time,
        )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        ConnectionResetError,
        TimeoutError,
    ) as exc:
        return [], [], [f"{family} generation lookup failed: {exc}"]

    if not observations:
        return [], [], [f"{family} missing generation observations"]

    route_filtered_observations = observations
    allowed_request_routes = allowed_request_routes or []
    if allowed_request_routes:
        route_filtered_observations = [
            observation
            for observation in observations
            if (observation.get("metadata") or {}).get("user_api_key_request_route")
            in allowed_request_routes
        ]
        if not route_filtered_observations:
            return [], [], [
                f"{family} missing generation observations for routes: {', '.join(allowed_request_routes)}"
            ]

    summaries: list[dict[str, Any]] = []
    for observation in route_filtered_observations:
        model = observation.get("model")
        prompt_tokens = observation.get("promptTokens")
        completion_tokens = observation.get("completionTokens")
        total_tokens = observation.get("totalTokens")
        cost_total = _extract_generation_metric(observation, "costDetails", "total")
        calculated_total_cost = observation.get("calculatedTotalCost")
        summary = {
            "id": observation.get("id"),
            "traceId": observation.get("traceId"),
            "name": observation.get("name"),
            "model": model,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "costDetails.total": cost_total,
            "calculatedTotalCost": calculated_total_cost,
        }
        summaries.append(summary)

        if not isinstance(model, str) or not model.strip():
            failures.append(f"{family} generation missing model")
        elif model.strip().lower() == "unknown":
            failures.append(f"{family} generation model resolved to unknown")

        if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
            failures.append(f"{family} generation missing promptTokens")
        if not isinstance(completion_tokens, (int, float)) or completion_tokens <= 0:
            failures.append(f"{family} generation missing completionTokens")
        if not isinstance(total_tokens, (int, float)) or total_tokens <= 0:
            failures.append(f"{family} generation missing totalTokens")
        if not isinstance(cost_total, (int, float)) or cost_total <= 0:
            failures.append(f"{family} generation missing costDetails.total")
        if (
            not isinstance(calculated_total_cost, (int, float))
            or calculated_total_cost <= 0
        ):
            failures.append(f"{family} generation missing calculatedTotalCost")

    return route_filtered_observations, summaries, sorted(set(failures))


def _validate_span_observations(
    *,
    family: str,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    required_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    required_names = required_names or []
    if not required_names:
        return [], [], []
    if not trace_ids:
        return [], [], [f"{family} missing trace ids for span validation"]

    try:
        observations = _recent_langfuse_span_observations_for_trace_ids(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            trace_ids=trace_ids,
            start_time=start_time,
        )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        ConnectionResetError,
        TimeoutError,
    ) as exc:
        return [], [], [f"{family} span lookup failed: {exc}"]

    if not observations:
        return [], [], [f"{family} missing span observations"]

    observed_names = sorted(
        {
            str(observation.get("name")).strip()
            for observation in observations
            if isinstance(observation.get("name"), str) and observation.get("name").strip()
        }
    )
    failures: list[str] = []
    for name in required_names:
        if name not in observed_names:
            failures.append(f"{family} missing span observation: {name}")

    summaries = [
        {
            "id": observation.get("id"),
            "traceId": observation.get("traceId"),
            "name": observation.get("name"),
            "startTime": observation.get("startTime"),
            "endTime": observation.get("endTime"),
        }
        for observation in observations
    ]
    return observations, summaries, sorted(set(failures))


def _collect_trace_tags(traces: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for trace in traces:
        tags = trace.get("tags") or []
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag not in seen:
                seen.add(tag)
                ordered.append(tag)
    return ordered


def _collect_trace_metadata(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_items: list[dict[str, Any]] = []
    for trace in traces:
        metadata = trace.get("metadata") or {}
        if isinstance(metadata, dict):
            metadata_items.append(metadata)
    return metadata_items


def _parse_stdout_json_objects(stdout: str) -> list[dict[str, Any]]:
    stripped = stdout.strip()
    if not stripped:
        return []

    objects: list[dict[str, Any]] = []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed_line = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_line, dict):
            objects.append(parsed_line)

    return objects


def _extract_command_session_id(stdout: str) -> str | None:
    for obj in _parse_stdout_json_objects(stdout):
        for key in ("session_id", "sessionId"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_trace_environment(trace: dict[str, Any]) -> str | None:
    value = trace.get("environment")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_trace_session_id(trace: dict[str, Any]) -> str | None:
    for key in ("sessionId", "session_id"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validate_trace_context(
    *,
    family: str,
    traces: list[dict[str, Any]],
    expected_environment: str | None = None,
    require_trace_session_id: bool = False,
    expected_trace_session_id: str | None = None,
    require_trace_ids_distinct_from_session_ids: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    trace_ids = sorted(
        {str(trace.get("id")) for trace in traces if isinstance(trace.get("id"), str)}
    )
    environments = sorted(
        {
            environment
            for trace in traces
            if (environment := _extract_trace_environment(trace)) is not None
        }
    )
    session_ids = sorted(
        {
            session_id
            for trace in traces
            if (session_id := _extract_trace_session_id(trace)) is not None
        }
    )
    missing_environment_trace_ids = [
        str(trace.get("id"))
        for trace in traces
        if _extract_trace_environment(trace) is None
    ]
    missing_session_id_trace_ids = [
        str(trace.get("id"))
        for trace in traces
        if _extract_trace_session_id(trace) is None
    ]

    if expected_environment is not None:
        unexpected_environment_trace_ids = [
            str(trace.get("id"))
            for trace in traces
            if _extract_trace_environment(trace) != expected_environment
        ]
        if unexpected_environment_trace_ids:
            failures.append(
                f"{family} trace environment mismatch: expected `{expected_environment}`"
            )
    else:
        unexpected_environment_trace_ids = []

    if require_trace_session_id and missing_session_id_trace_ids:
        failures.append(f"{family} missing trace sessionId")

    if expected_trace_session_id is not None:
        mismatched_session_trace_ids = [
            str(trace.get("id"))
            for trace in traces
            if _extract_trace_session_id(trace) != expected_trace_session_id
        ]
        if mismatched_session_trace_ids:
            failures.append(
                f"{family} trace sessionId mismatch: expected `{expected_trace_session_id}`"
            )
    else:
        mismatched_session_trace_ids = []

    overlapping_trace_session_ids = sorted(set(trace_ids).intersection(session_ids))
    if require_trace_ids_distinct_from_session_ids and overlapping_trace_session_ids:
        failures.append(f"{family} trace ids collapsed into session ids")

    summary = {
        "trace_ids": trace_ids,
        "trace_environments": environments,
        "trace_session_ids": session_ids,
        "missing_environment_trace_ids": missing_environment_trace_ids,
        "missing_session_id_trace_ids": missing_session_id_trace_ids,
        "unexpected_environment_trace_ids": unexpected_environment_trace_ids,
        "mismatched_session_trace_ids": mismatched_session_trace_ids,
        "overlapping_trace_session_ids": overlapping_trace_session_ids,
    }
    return summary, failures


def _validate_trace_enrichment(
    *,
    family: str,
    traces: list[dict[str, Any]],
    required_tags: list[str] | None = None,
    required_tag_prefixes: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    all_tags = _collect_trace_tags(traces)
    metadata_items = _collect_trace_metadata(traces)
    metadata_keys = sorted({key for metadata in metadata_items for key in metadata.keys()})

    for tag in required_tags or []:
        if tag not in all_tags:
            failures.append(f"{family} missing trace tag: {tag}")

    for prefix in required_tag_prefixes or []:
        if not any(tag.startswith(prefix) for tag in all_tags):
            failures.append(f"{family} missing trace tag prefix: {prefix}")

    summary = {
        "trace_tags": all_tags,
        "trace_metadata_keys": metadata_keys,
    }
    return summary, failures


def _validate_generation_metadata(
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_metadata_truthy: list[str] | None = None,
    required_metadata_minimums: dict[str, int | float] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    metadata_items = [
        observation.get("metadata")
        for observation in observations
        if isinstance(observation.get("metadata"), dict)
    ]
    metadata_items = [metadata for metadata in metadata_items if isinstance(metadata, dict)]
    metadata_keys = sorted({key for metadata in metadata_items for key in metadata.keys()})

    truthy_hits: dict[str, bool] = {}
    for key in required_metadata_truthy or []:
        truthy_hit = any(bool(metadata.get(key)) for metadata in metadata_items)
        truthy_hits[key] = truthy_hit
        if not truthy_hit:
            failures.append(f"{family} missing truthy generation metadata: {key}")

    minimum_hits: dict[str, Any] = {}
    for key, minimum in (required_metadata_minimums or {}).items():
        hit = None
        for metadata in metadata_items:
            value = metadata.get(key)
            if isinstance(value, (int, float)) and value >= minimum:
                hit = value
                break
        minimum_hits[key] = hit
        if hit is None:
            failures.append(f"{family} missing generation metadata >= {minimum}: {key}")

    summary = {
        "generation_metadata_keys": metadata_keys,
        "generation_metadata_truthy_hits": truthy_hits,
        "generation_metadata_minimum_hits": minimum_hits,
    }
    return summary, failures


def _extract_logged_request_body(observation: dict[str, Any]) -> dict[str, Any] | None:
    input_payload = observation.get("input")
    if not isinstance(input_payload, dict):
        return None
    messages = input_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    first_message = messages[0]
    if not isinstance(first_message, dict):
        return None
    content = first_message.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_claude_agent_name_from_observation(observation: dict[str, Any]) -> str | None:
    request_body = _extract_logged_request_body(observation)
    if not isinstance(request_body, dict):
        return None
    match = _CLAUDE_AGENT_NAME_RE.search(json.dumps(request_body))
    if not match:
        return None
    agent_name = match.group(1).strip()
    return agent_name or None


def _collect_text_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            fragments.append(value["text"])
        for child in value.values():
            fragments.extend(_collect_text_fragments(child))
    elif isinstance(value, list):
        for child in value:
            fragments.extend(_collect_text_fragments(child))
    elif isinstance(value, str):
        fragments.append(value)
    return fragments


def _validate_logged_request_text_checks(
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_substrings: list[str] | None = None,
    forbidden_substrings: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    required_substrings = required_substrings or []
    forbidden_substrings = forbidden_substrings or []
    matched_observation_id: str | None = None
    matched_required: dict[str, bool] = {value: False for value in required_substrings}
    forbidden_hits: dict[str, list[str]] = {value: [] for value in forbidden_substrings}

    for observation in observations:
        request_body = _extract_logged_request_body(observation)
        if request_body is None:
            continue
        request_text = "\n".join(_collect_text_fragments(request_body))
        if not request_text:
            continue

        for value in required_substrings:
            if value in request_text:
                matched_required[value] = True

        current_forbidden_hits = [
            value for value in forbidden_substrings if value in request_text
        ]
        for value in current_forbidden_hits:
            forbidden_hits[value].append(str(observation.get("id")))

        if all(value in request_text for value in required_substrings) and not current_forbidden_hits:
            matched_observation_id = str(observation.get("id"))

    for value, matched in matched_required.items():
        if not matched:
            failures.append(f"{family} missing request substring: {value}")
    for value, observation_ids in forbidden_hits.items():
        if observation_ids:
            failures.append(
                f"{family} request still contains forbidden substring `{value}` in {len(observation_ids)} observation(s)"
            )
    if required_substrings and matched_observation_id is None and not failures:
        failures.append(f"{family} missing request observation satisfying text checks")

    summary = {
        "matched_observation_id": matched_observation_id,
        "required_substrings_found": matched_required,
        "forbidden_substring_hits": forbidden_hits,
    }
    return summary, failures


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_logged_request_source_files(
    *,
    family: str,
    observations: list[dict[str, Any]],
    source_paths_key: str,
    source_hashes_key: str | None = None,
    source_bytes_key: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    checked_observation_ids: list[str] = []
    checked_source_paths: list[str] = []
    content_mismatch_paths: list[str] = []
    metadata_hash_mismatch_paths: list[str] = []
    metadata_bytes_mismatch_paths: list[str] = []
    unreadable_source_paths: list[str] = []

    for observation in observations:
        metadata = observation.get("metadata")
        if not isinstance(metadata, dict):
            continue
        source_paths = metadata.get(source_paths_key)
        if not isinstance(source_paths, list) or not any(
            isinstance(path, str) and path.strip() for path in source_paths
        ):
            continue

        request_body = _extract_logged_request_body(observation)
        if request_body is None:
            failures.append(f"{family} missing logged request body for source verification")
            continue
        request_text = "\n".join(_collect_text_fragments(request_body))
        if not request_text:
            failures.append(f"{family} missing logged request text for source verification")
            continue

        checked_observation_ids.append(str(observation.get("id")))
        source_hashes = metadata.get(source_hashes_key) if source_hashes_key else None
        source_bytes = metadata.get(source_bytes_key) if source_bytes_key else None

        for index, source_path_value in enumerate(source_paths):
            if not isinstance(source_path_value, str) or not source_path_value.strip():
                continue
            source_path = pathlib.Path(source_path_value)
            checked_source_paths.append(str(source_path))
            try:
                file_text = source_path.read_text(encoding="utf-8", errors="replace").rstrip(
                    "\n"
                )
            except Exception:
                unreadable_source_paths.append(str(source_path))
                continue

            actual_hash = _sha256_text(file_text)
            actual_bytes = len(file_text.encode("utf-8"))

            if file_text not in request_text:
                content_mismatch_paths.append(str(source_path))

            if isinstance(source_hashes, list) and index < len(source_hashes):
                metadata_hash = source_hashes[index]
                if isinstance(metadata_hash, str) and metadata_hash != actual_hash:
                    metadata_hash_mismatch_paths.append(str(source_path))

            if isinstance(source_bytes, list) and index < len(source_bytes):
                metadata_size = source_bytes[index]
                if isinstance(metadata_size, int) and metadata_size != actual_bytes:
                    metadata_bytes_mismatch_paths.append(str(source_path))

    if not checked_observation_ids:
        failures.append(f"{family} missing source-file metadata for request verification")
    if unreadable_source_paths:
        failures.append(f"{family} unreadable persisted-output source files")
    if content_mismatch_paths:
        failures.append(f"{family} logged request missing full persisted-output file contents")
    if metadata_hash_mismatch_paths:
        failures.append(f"{family} persisted-output content hash metadata mismatch")
    if metadata_bytes_mismatch_paths:
        failures.append(f"{family} persisted-output byte metadata mismatch")

    summary = {
        "checked_observation_ids": checked_observation_ids,
        "checked_source_paths": checked_source_paths,
        "content_mismatch_paths": content_mismatch_paths,
        "metadata_hash_mismatch_paths": metadata_hash_mismatch_paths,
        "metadata_bytes_mismatch_paths": metadata_bytes_mismatch_paths,
        "unreadable_source_paths": unreadable_source_paths,
    }
    return summary, failures


def _observation_has_gemini_thought_signature(observation: dict[str, Any]) -> bool:
    output = observation.get("output")
    if not isinstance(output, dict):
        return False
    provider_specific_fields = output.get("provider_specific_fields")
    if not isinstance(provider_specific_fields, dict):
        return False
    thought_signatures = provider_specific_fields.get("thought_signatures")
    return isinstance(thought_signatures, list) and any(
        isinstance(value, str) and value.strip() for value in thought_signatures
    )


def _extract_claude_thinking_blocks(observation: dict[str, Any]) -> list[dict[str, Any]]:
    output = observation.get("output")
    if not isinstance(output, dict):
        return []
    thinking_blocks = output.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        return [block for block in thinking_blocks if isinstance(block, dict)]
    provider_specific_fields = output.get("provider_specific_fields")
    if not isinstance(provider_specific_fields, dict):
        return []
    provider_blocks = provider_specific_fields.get("thinking_blocks")
    if isinstance(provider_blocks, list):
        return [block for block in provider_blocks if isinstance(block, dict)]
    return []


def _observation_has_claude_thinking_signature(observation: dict[str, Any]) -> bool:
    for block in _extract_claude_thinking_blocks(observation):
        signature = block.get("signature")
        if isinstance(signature, str) and signature.strip():
            return True
    return False


def _recent_langfuse_required_name_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
) -> list[dict[str, Any]]:
    recent_by_id: dict[str, dict[str, Any]] = {}
    for name in names:
        params = {
            "limit": str(limit),
            "name": name,
            "fields": "core",
            "orderBy": "timestamp.desc",
            "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
        }
        if user_id:
            params["userId"] = user_id
        url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
        try:
            payload = _http_get_json(url, public_key, secret_key)
            traces = payload.get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.RemoteDisconnected):
            traces = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                limit=max(limit, 100),
            )
        for trace in traces:
            trace_name = trace.get("name")
            if trace_name != name:
                continue
            timestamp = _parse_langfuse_timestamp(
                trace.get("timestamp") or trace.get("createdAt") or trace.get("updatedAt")
            )
            trace_id = trace.get("id")
            if isinstance(trace_id, str):
                recent_by_id[trace_id] = trace
    return list(recent_by_id.values())


def _poll_langfuse_required_name_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], str | None]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    last_error: str | None = None
    while True:
        try:
            traces = _recent_langfuse_required_name_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                names=names,
                user_id=user_id,
                start_time=start_time,
                limit=limit,
            )
            last_error = None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as exc:
            traces = []
            last_error = str(exc)
        actual_names = {trace.get("name") for trace in traces if trace.get("name")}
        if all(name in actual_names for name in names):
            return traces, last_error
        if time.time() >= deadline:
            return traces, last_error
        time.sleep(interval_seconds)


def _poll_langfuse_named_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    while True:
        traces = []
        try:
            for name in names:
                traces.extend(
                    _recent_langfuse_traces(
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        name=name,
                        user_id=user_id,
                        start_time=start_time,
                        limit=limit,
                    )
                )
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.RemoteDisconnected):
            all_recent = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                limit=max(limit, 100),
            )
            traces = [
                trace
                for trace in all_recent
                if trace.get("name") in names
            ]
        unique = {
            trace.get("id"): trace
            for trace in traces
            if isinstance(trace.get("id"), str)
        }
        traces = list(unique.values())
        actual_names = {trace.get("name") for trace in traces if trace.get("name")}
        if all(name in actual_names for name in names):
            return traces
        if time.time() >= deadline:
            return traces
        time.sleep(interval_seconds)


def _run_command(
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration = round(time.time() - started, 3)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return {
        "command": command,
        "command_string": " ".join(shlex.quote(part) for part in command),
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "stdout": stdout,
        "stderr": stderr,
        "response_excerpt": _response_excerpt(stdout, stderr),
    }


def _response_excerpt(stdout: str, stderr: str, limit: int = 300) -> str:
    text = stdout or stderr
    text = text.replace("\n", " ").strip()
    return text[:limit]


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _docker_status() -> str:
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=litellm-dev", "--format", "{{.Status}}"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _validate_codex(
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
) -> dict[str, Any]:
    started = _utcnow()
    run = _run_command(config["command"], timeout_seconds=int(config.get("timeout_seconds", 300)))
    command_session_id = _extract_command_session_id(run["stdout"])
    post_run_wait_seconds = float(config.get("post_run_wait_seconds", 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)
    expected_trace_names = config.get("expected_trace_names", [])
    expected_user_ids = config.get("expected_user_ids", [])
    traces = _poll_langfuse_named_traces(
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        names=expected_trace_names,
        user_id=expected_user_ids[0] if expected_user_ids else None,
        start_time=started,
        timeout_seconds=int(config.get("langfuse_poll_timeout_seconds", 45)),
    )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    trace_ids = [trace.get("id") for trace in traces if trace.get("id")]
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("codex command failed")
    for name in expected_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Codex trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Codex user id: {user_id}")
    (
        _raw_generation_observations,
        generation_observations,
        generation_failures,
    ) = _validate_generation_observations(
        family="codex",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=config.get("allowed_generation_routes"),
    )
    failures.extend(generation_failures)
    trace_enrichment_summary, trace_enrichment_failures = _validate_trace_enrichment(
        family="codex",
        traces=traces,
        required_tags=config.get("required_trace_tags"),
        required_tag_prefixes=config.get("required_trace_tag_prefixes"),
    )
    failures.extend(trace_enrichment_failures)
    trace_context_summary, trace_context_failures = _validate_trace_context(
        family="codex",
        traces=traces,
        expected_environment=config.get("expected_trace_environment"),
        require_trace_session_id=bool(config.get("require_trace_session_id")),
        expected_trace_session_id=config.get("expected_trace_session_id"),
        require_trace_ids_distinct_from_session_ids=bool(
            config.get("require_trace_ids_distinct_from_session_ids")
        ),
    )
    failures.extend(trace_context_failures)
    generation_metadata_summary, generation_metadata_failures = _validate_generation_metadata(
        family="codex",
        observations=_raw_generation_observations,
        required_metadata_truthy=config.get("required_generation_metadata_truthy"),
        required_metadata_minimums=config.get("required_generation_metadata_minimums"),
    )
    failures.extend(generation_metadata_failures)
    _, span_observations, span_failures = _validate_span_observations(
        family="codex",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        required_names=config.get("required_span_names"),
    )
    failures.extend(span_failures)
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "expected_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "command_session_id": command_session_id,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_metadata": generation_metadata_summary,
            "span_observations": span_observations,
            "generation_observations": generation_observations,
        },
        "passed": not failures,
        "failures": sorted(set(failures)),
    }


def _validate_gemini(
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
) -> dict[str, Any]:
    started = _utcnow()
    run = _run_command(config["command"], timeout_seconds=int(config.get("timeout_seconds", 300)))
    command_session_id = _extract_command_session_id(run["stdout"])
    post_run_wait_seconds = float(config.get("post_run_wait_seconds", 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)
    expected_trace_names = config.get("expected_trace_names", [])
    expected_user_ids = config.get("expected_user_ids", [])
    traces = _poll_langfuse_named_traces(
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        names=expected_trace_names,
        user_id=expected_user_ids[0] if expected_user_ids else None,
        start_time=started,
        limit=100,
        timeout_seconds=int(config.get("langfuse_poll_timeout_seconds", 45)),
    )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    trace_ids = [trace.get("id") for trace in traces if trace.get("id")]
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("gemini command failed")
    for name in expected_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Gemini trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Gemini user id: {user_id}")
    (
        raw_generation_observations,
        generation_observations,
        generation_failures,
    ) = _validate_generation_observations(
        family="gemini",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=config.get("allowed_generation_routes"),
    )
    failures.extend(generation_failures)
    filtered_trace_ids = sorted(
        {
            observation.get("traceId")
            for observation in raw_generation_observations
            if isinstance(observation.get("traceId"), str)
        }
    )
    filtered_traces = [
        trace for trace in traces if trace.get("id") in set(filtered_trace_ids)
    ]
    trace_enrichment_summary, trace_enrichment_failures = _validate_trace_enrichment(
        family="gemini",
        traces=filtered_traces,
        required_tags=config.get("required_trace_tags"),
        required_tag_prefixes=config.get("required_trace_tag_prefixes"),
    )
    failures.extend(trace_enrichment_failures)
    trace_context_summary, trace_context_failures = _validate_trace_context(
        family="gemini",
        traces=filtered_traces,
        expected_environment=config.get("expected_trace_environment"),
        require_trace_session_id=bool(config.get("require_trace_session_id")),
        expected_trace_session_id=(
            command_session_id
            if config.get("match_trace_session_id_from_stdout")
            else config.get("expected_trace_session_id")
        ),
        require_trace_ids_distinct_from_session_ids=bool(
            config.get("require_trace_ids_distinct_from_session_ids")
        ),
    )
    failures.extend(trace_context_failures)
    generation_metadata_summary, generation_metadata_failures = _validate_generation_metadata(
        family="gemini",
        observations=raw_generation_observations,
        required_metadata_truthy=config.get("required_generation_metadata_truthy"),
        required_metadata_minimums=config.get("required_generation_metadata_minimums"),
    )
    failures.extend(generation_metadata_failures)
    _, span_observations, span_failures = _validate_span_observations(
        family="gemini",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=filtered_trace_ids,
        start_time=started,
        required_names=config.get("required_span_names"),
    )
    failures.extend(span_failures)
    gemini_signature_observed = any(
        _observation_has_gemini_thought_signature(observation)
        for observation in raw_generation_observations
    )
    if not gemini_signature_observed:
        failures.append("gemini missing thought_signatures in logged generation output")
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "expected_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "filtered_trace_ids": filtered_trace_ids,
            "command_session_id": command_session_id,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_metadata": generation_metadata_summary,
            "span_observations": span_observations,
            "thought_signature_observed": gemini_signature_observed,
            "generation_observations": generation_observations,
        },
        "passed": not failures,
        "failures": sorted(set(failures)),
    }


def _validate_claude(
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    fanout_mode: str = "minimal",
) -> dict[str, Any]:
    fanout_modes = config.get("fanout_modes", {})
    selected_mode = fanout_modes.get(fanout_mode, {})
    effective_config = dict(config)
    if selected_mode:
        effective_config.update(selected_mode)

    started = _utcnow()
    run = _run_command(
        effective_config["command"],
        extra_env=effective_config.get("env"),
        timeout_seconds=int(effective_config.get("timeout_seconds", 300)),
    )
    command_session_id = _extract_command_session_id(run["stdout"])
    post_run_wait_seconds = float(effective_config.get("post_run_wait_seconds", 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)
    required_trace_names = effective_config.get("required_trace_names", [])
    expected_user_ids = effective_config.get("expected_user_ids", [])
    if isinstance(command_session_id, str) and command_session_id.strip():
        traces, lookup_error = _poll_langfuse_session_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            user_id=expected_user_ids[0] if expected_user_ids else None,
            start_time=started,
            session_id=command_session_id.strip(),
            timeout_seconds=int(
                effective_config.get("langfuse_poll_timeout_seconds", 60)
            ),
        )
    else:
        traces, lookup_error = _poll_langfuse_required_name_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            names=required_trace_names,
            user_id=expected_user_ids[0] if expected_user_ids else None,
            start_time=started,
            limit=100,
            timeout_seconds=int(
                effective_config.get("langfuse_poll_timeout_seconds", 60)
            ),
        )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    trace_ids = [trace.get("id") for trace in traces if trace.get("id")]
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("claude command failed")
    if lookup_error:
        failures.append(f"Claude Langfuse lookup warning: {lookup_error}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Claude user id: {user_id}")
    (
        raw_generation_observations,
        generation_observations,
        generation_failures,
    ) = _validate_generation_observations(
        family="claude",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=effective_config.get("allowed_generation_routes"),
    )
    failures.extend(generation_failures)
    observed_agents = sorted(
        {
            agent_name
            for observation in raw_generation_observations
            if (agent_name := _extract_claude_agent_name_from_observation(observation))
        }
    )
    required_agent_names = sorted(
        {
            name.removeprefix("claude-code.")
            for name in required_trace_names
            if isinstance(name, str) and name.strip()
        }
    )
    for agent_name in required_agent_names:
        if agent_name not in observed_agents:
            failures.append(f"missing Claude agent observation: {agent_name}")
    if "orchestrator" not in observed_agents:
        failures.append("missing Claude orchestrator observation")
    if len([name for name in observed_agents if name != "orchestrator"]) == 0:
        failures.append("missing Claude persona/subagent observations")
    filtered_trace_ids = sorted(
        {
            observation.get("traceId")
            for observation in raw_generation_observations
            if isinstance(observation.get("traceId"), str)
        }
    )
    filtered_traces = [
        trace for trace in traces if trace.get("id") in set(filtered_trace_ids)
    ]
    trace_enrichment_summary, trace_enrichment_failures = _validate_trace_enrichment(
        family="claude",
        traces=filtered_traces,
        required_tags=effective_config.get("required_trace_tags"),
        required_tag_prefixes=effective_config.get("required_trace_tag_prefixes"),
    )
    failures.extend(trace_enrichment_failures)
    trace_context_summary, trace_context_failures = _validate_trace_context(
        family="claude",
        traces=filtered_traces,
        expected_environment=effective_config.get("expected_trace_environment"),
        require_trace_session_id=bool(effective_config.get("require_trace_session_id")),
        expected_trace_session_id=(
            command_session_id
            if effective_config.get("match_trace_session_id_from_stdout")
            else effective_config.get("expected_trace_session_id")
        ),
        require_trace_ids_distinct_from_session_ids=bool(
            effective_config.get("require_trace_ids_distinct_from_session_ids")
        ),
    )
    failures.extend(trace_context_failures)
    generation_metadata_summary, generation_metadata_failures = _validate_generation_metadata(
        family="claude",
        observations=raw_generation_observations,
        required_metadata_truthy=effective_config.get("required_generation_metadata_truthy"),
        required_metadata_minimums=effective_config.get("required_generation_metadata_minimums"),
    )
    failures.extend(generation_metadata_failures)
    request_text_checks = effective_config.get("request_text_checks", {})
    request_text_summary, request_text_failures = _validate_logged_request_text_checks(
        family="claude",
        observations=raw_generation_observations,
        required_substrings=request_text_checks.get("required_substrings"),
        forbidden_substrings=request_text_checks.get("forbidden_substrings"),
    )
    failures.extend(request_text_failures)
    source_file_verification_config = effective_config.get(
        "request_source_file_verification", {}
    )
    source_file_summary, source_file_failures = _validate_logged_request_source_files(
        family="claude",
        observations=raw_generation_observations,
        source_paths_key=source_file_verification_config.get(
            "source_paths_key", "claude_persisted_output_source_paths"
        ),
        source_hashes_key=source_file_verification_config.get(
            "source_hashes_key", "claude_persisted_output_source_content_hashes"
        ),
        source_bytes_key=source_file_verification_config.get(
            "source_bytes_key", "claude_persisted_output_source_bytes"
        ),
    )
    failures.extend(source_file_failures)
    _, span_observations, span_failures = _validate_span_observations(
        family="claude",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=filtered_trace_ids,
        start_time=started,
        required_names=effective_config.get("required_span_names"),
    )
    failures.extend(span_failures)
    claude_signature_observed = any(
        _observation_has_claude_thinking_signature(observation)
        for observation in raw_generation_observations
    )
    return {
        **run,
        "streaming_checked": effective_config.get("streaming_checked", False),
        "langfuse": {
            "fanout_mode": fanout_mode,
            "required_trace_names": required_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "lookup_error": lookup_error,
            "filtered_trace_ids": filtered_trace_ids,
            "command_session_id": command_session_id,
            "observed_agents": observed_agents,
            "required_agent_names": required_agent_names,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_metadata": generation_metadata_summary,
            "request_text_checks": request_text_summary,
            "request_source_file_verification": source_file_summary,
            "span_observations": span_observations,
            "thought_signature_observed": claude_signature_observed,
            "generation_observations": generation_observations,
        },
        "passed": not failures,
        "failures": sorted(set(failures)),
    }


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    for family, result in results.items():
        for failure in result.get("failures", []):
            failures.append(f"{family}: {failure}")
    return {
        "passed": not failures,
        "failures": failures,
    }


def _family_error_result(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "command": [],
        "command_string": "",
        "exit_code": 1,
        "duration_seconds": 0,
        "stdout": "",
        "stderr": "",
        "response_excerpt": "",
        "streaming_checked": False,
        "langfuse": {
            "expected_trace_names": [],
            "actual_trace_names": [],
            "expected_user_ids": [],
            "actual_user_ids": [],
            "trace_ids": [],
            "trace_count": 0,
        },
        "passed": False,
        "failures": [f"{name} validator error: {exc}"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local CLI acceptance checks through litellm-dev.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to suite config JSON.")
    parser.add_argument("--write-artifact", required=True, help="Where to write the JSON artifact.")
    parser.add_argument("--langfuse-query-url", default=None, help="Override Langfuse query URL.")
    parser.add_argument(
        "--claude-fanout-mode",
        choices=("minimal", "full"),
        default="minimal",
        help="Claude fan-out validation depth.",
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    artifact_path = pathlib.Path(args.write_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    config = _load_json(config_path)

    public_key_env = config.get("langfuse_public_key_env", "LANGFUSE_PUBLIC_KEY")
    secret_key_env = config.get("langfuse_secret_key_env", "LANGFUSE_SECRET_KEY")
    public_key = os.environ.get(public_key_env, "")
    secret_key = os.environ.get(secret_key_env, "")
    query_url = args.langfuse_query_url or os.environ.get("LANGFUSE_QUERY_URL") or config.get(
        "langfuse_query_url", "http://127.0.0.1:3000"
    )

    if not public_key or not secret_key:
        print(
            f"Missing Langfuse credentials in env vars {public_key_env}/{secret_key_env}",
            file=sys.stderr,
        )
        return 2

    artifact: dict[str, Any] = {
        "suite_version": config.get("suite_version", 1),
        "timestamp": _isoformat(_utcnow()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "environment": {
            "litellm_base_url": config.get("litellm_base_url", "http://127.0.0.1:4001"),
            "langfuse_query_url": query_url,
            "docker_litellm_dev_status": _docker_status(),
        },
        "results": {},
        "summary": {},
    }

    try:
        artifact["results"]["codex"] = _validate_codex(
            config["codex"],
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
        )
    except Exception as exc:
        artifact["results"]["codex"] = _family_error_result("codex", exc)

    try:
        artifact["results"]["gemini"] = _validate_gemini(
            config["gemini"],
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
        )
    except Exception as exc:
        artifact["results"]["gemini"] = _family_error_result("gemini", exc)

    try:
        artifact["results"]["claude"] = _validate_claude(
            config["claude"],
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            fanout_mode=args.claude_fanout_mode,
        )
    except Exception as exc:
        artifact["results"]["claude"] = _family_error_result("claude", exc)

    artifact["summary"] = _build_summary(artifact["results"])
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    return 0 if artifact["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
