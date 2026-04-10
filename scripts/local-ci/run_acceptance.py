#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import http.client
import json
import os
import pathlib
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
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("codex command failed")
    for name in expected_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Codex trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Codex user id: {user_id}")
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "expected_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": [trace.get("id") for trace in traces if trace.get("id")],
            "trace_count": len(traces),
        },
        "passed": not failures,
        "failures": failures,
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
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("gemini command failed")
    for name in expected_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Gemini trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Gemini user id: {user_id}")
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "expected_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": [trace.get("id") for trace in traces if trace.get("id")],
            "trace_count": len(traces),
        },
        "passed": not failures,
        "failures": failures,
    }


def _validate_claude(
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
) -> dict[str, Any]:
    started = _utcnow()
    run = _run_command(
        config["command"],
        extra_env=config.get("env"),
        timeout_seconds=int(config.get("timeout_seconds", 300)),
    )
    required_trace_names = config.get("required_trace_names", [])
    expected_user_ids = config.get("expected_user_ids", [])
    traces, lookup_error = _poll_langfuse_required_name_traces(
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        names=required_trace_names,
        user_id=expected_user_ids[0] if expected_user_ids else None,
        start_time=started,
        limit=100,
        timeout_seconds=int(config.get("langfuse_poll_timeout_seconds", 60)),
    )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    minimum_trace_count = int(config.get("minimum_trace_count", len(required_trace_names) or 1))
    failures: list[str] = []
    if run["exit_code"] != 0:
        failures.append("claude command failed")
    if lookup_error:
        failures.append(f"Claude Langfuse lookup warning: {lookup_error}")
    for name in required_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Claude trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Claude user id: {user_id}")
    if len(actual_trace_names) < minimum_trace_count:
        failures.append(
            f"expected at least {minimum_trace_count} distinct Claude trace names, found {len(actual_trace_names)}"
        )
    if "claude-code.orchestrator" not in actual_trace_names:
        failures.append("missing Claude orchestrator trace")
    if len([name for name in actual_trace_names if name != "claude-code.orchestrator"]) == 0:
        failures.append("missing Claude persona/subagent traces")
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "required_trace_names": required_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": [trace.get("id") for trace in traces if trace.get("id")],
            "trace_count": len(traces),
            "lookup_error": lookup_error,
        },
        "passed": not failures,
        "failures": failures,
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
        )
    except Exception as exc:
        artifact["results"]["claude"] = _family_error_result("claude", exc)

    artifact["summary"] = _build_summary(artifact["results"])
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    return 0 if artifact["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
