"""RR-068 residuals for docker-log provider-error backfill.

#1 log_signature stability + in-batch/DB-existing idempotency filtering
#2 indexed nearby lookups + sparse retention (no full O(N) scan / full idle log load)
#3 EXCEPTION_RE requires LiteLLM/proxy prefix; ACCESS_RE is line-anchored (reject bare echoed substrings)
#4 --limit / max_observations bounds parse + correlation work early
Also covers source/provenance metadata and positive --limit argument validation.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_provider_error_observations_from_docker_logs.py"


def _load() -> ModuleType:
    name = "backfill_provider_error_observations_from_docker_logs_rr068"
    # Always reload so residual patches are visible during iterative runs.
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ts(i: int) -> str:
    # docker timestamps with nanosecond-ish fraction
    base = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
    return base.strftime("%Y-%m-%dT%H:%M:%S.") + f"{base.microsecond:06d}000Z"


def _line(i: int, message: str) -> str:
    return f"{_ts(i)} {message}"


def test_exception_re_requires_proxy_prefix() -> None:
    mod = _load()
    # Bare echoed text (model/tool output) must not match.
    bare = 'Exception occured - 500: {"error":{"message":"boom"}}'
    assert mod.EXCEPTION_RE.search(bare) is None

    # Integration layer echoes that are not proxy request handlers must not match.
    assert (
        mod.EXCEPTION_RE.search(
            "prometheus Layer Error(): Exception occured - 500: not provider"
        )
        is None
    )

    # Known pass-through proxy shape must match.
    proxy = (
        "pass_through_endpoint(): Exception occured - 429: "
        '{"error":{"type":"usage_limit_reached","message":"rate limited"}}'
    )
    match = mod.EXCEPTION_RE.search(proxy)
    assert match is not None
    assert match.group("status") == "429"
    assert "usage_limit_reached" in match.group("detail")

    # Fully-qualified pass_through shape must match (real docker logger prefix).
    proxy_fq = (
        "17:17:21 - LiteLLM Proxy:ERROR: pass_through_endpoints.py:1668 - "
        "litellm.proxy.proxy_server.pass_through_endpoint(): Exception occured - "
        "503: upstream unavailable"
    )
    assert mod.EXCEPTION_RE.search(proxy_fq) is not None

    # litellm.proxy...(): shape must match.
    proxy2 = (
        "litellm.proxy.proxy_server.anthropic_response(): Exception occured - "
        "503: upstream unavailable"
    )
    assert mod.EXCEPTION_RE.search(proxy2) is not None

    # Double-colon hook shapes under litellm. must match.
    hook = (
        "litellm.proxy.hooks.dynamic_rate_limiter.py::"
        "async_post_call_success_hook(): Exception occured - 429: limited"
    )
    assert mod.EXCEPTION_RE.search(hook) is not None

    # Router-style prefix must match.
    router = (
        "litellm.router.Router::deployment_callback_on_success(): Exception "
        "occured - 500: boom"
    )
    assert mod.EXCEPTION_RE.search(router) is not None

    # 2xx/3xx statuses are not provider-error observations.
    assert (
        mod.EXCEPTION_RE.search(
            "litellm.proxy.proxy_server.completion(): Exception occured - "
            "200: not an error status"
        )
        is None
    )


def test_access_re_requires_line_start() -> None:
    """RR-068 #3: mid-line access-log shapes must not become observations."""
    mod = _load()
    real = (
        'INFO: 1.2.3.4:9 - "POST /v1/chat/completions HTTP/1.1" '
        "500 Internal Server Error"
    )
    real_padded = (
        'INFO:     172.18.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" '
        "429 Too Many Requests"
    )
    echoed = (
        'tool stdout: INFO: 1.2.3.4:9 - "POST /v1/chat/completions HTTP/1.1" '
        "500 Internal Server Error"
    )
    assert mod.ACCESS_RE.search(real) is not None
    assert mod.ACCESS_RE.search(real_padded) is not None
    assert mod.ACCESS_RE.search(echoed) is None

    lines = [
        _line(0, echoed),
        _line(1, real),
        _line(
            2,
            "pass_through_endpoint(): Exception occured - 500: "
            '{"error":{"message":"real"}}',
        ),
    ]
    parsed = mod._parse_logs(lines)
    assert len(parsed.access_logs) == 1
    assert parsed.access_logs[0].status_code == 500
    assert all("tool stdout" not in a.entry.clean_content for a in parsed.access_logs)


def test_parse_logs_skips_bare_exception_substring() -> None:
    mod = _load()
    lines = [
        _line(0, 'INFO: 1.2.3.4:9 - "POST /v1/chat/completions HTTP/1.1" 200 OK'),
        _line(
            1,
            # Simulated model/tool stdout echoed into container logs
            'tool stdout: Exception occured - 500: {"error":{"message":"not real"}}',
        ),
        _line(
            2,
            'pass_through_endpoint(): Exception occured - 503: {"error":{"message":"real"}}',
        ),
        _line(
            3,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" 503 Service Unavailable',
        ),
        _line(4, "INFO: idle keepalive"),
        _line(5, "ERROR: nearby ConnectionResetError for correlation"),
    ]
    parsed = mod._parse_logs(lines)
    assert len(parsed.exception_logs) == 1
    assert parsed.exception_logs[0].status_code == 503
    assert "real" in parsed.exception_logs[0].detail
    assert len(parsed.access_logs) == 1
    assert parsed.access_logs[0].status_code == 503
    # Sparse retention: idle/2xx/bare-exception lines are not kept in entries.
    retained = {entry.clean_content for entry in parsed.entries}
    assert any("real" in text for text in retained)
    assert any("503 Service Unavailable" in text for text in retained)
    assert any("ConnectionResetError" in text for text in retained)
    assert not any("idle keepalive" in text for text in retained)
    assert not any("200 OK" in text for text in retained)
    # Bare "Exception occured" text may still be retained as a context-marker line
    # because it contains "Exception", but it must never become exception_logs.
    assert all("not real" not in exc.detail for exc in parsed.exception_logs)


def test_parse_logs_max_exception_logs_stops_early() -> None:
    mod = _load()
    lines: list[str] = []
    for i in range(10):
        lines.append(
            _line(
                i,
                "pass_through_endpoint(): Exception occured - 503: "
                f'{{"error":{{"message":"e{i}"}}}}',
            )
        )
    parsed = mod._parse_logs(lines, max_exception_logs=3)
    assert len(parsed.exception_logs) == 3
    assert "e0" in parsed.exception_logs[0].detail
    assert "e2" in parsed.exception_logs[2].detail


def test_parse_logs_limit_drains_access_tail() -> None:
    """--limit must still capture the co-timed access line after the last exception."""
    mod = _load()
    lines = [
        _line(
            0,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"e0"}}',
        ),
        _line(
            1,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
            "503 Service Unavailable",
        ),
        _line(2, "ERROR: nearby ConnectionResetError"),
        _line(
            3,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"e1-should-not-be-accepted"}}',
        ),
    ]
    parsed = mod._parse_logs(lines, max_exception_logs=1)
    assert len(parsed.exception_logs) == 1
    assert "e0" in parsed.exception_logs[0].detail
    assert len(parsed.access_logs) == 1
    assert parsed.access_logs[0].status_code == 503
    retained = "\n".join(entry.clean_content for entry in parsed.entries)
    assert "ConnectionResetError" in retained
    assert "e1-should-not-be-accepted" not in retained


def test_find_nearby_context_uses_time_buckets_not_full_scan() -> None:
    mod = _load()
    base = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)
    # Build a large sparse log: many far-away ERROR lines + one nearby.
    entries = []
    for i in range(5000):
        observed = base + timedelta(seconds=i)
        content = "INFO: idle"
        if i == 9:
            content = "ERROR: nearby traceback ConnectionResetError"
        elif i % 100 == 0 and i != 0:
            content = "ERROR: far away noise"
        entries.append(
            mod.LogEntry(
                line_index=i + 1,
                timestamp_text=observed.isoformat(),
                observed_at=observed,
                content=content,
                clean_content=content,
            )
        )
    parsed = mod.ParsedLogs(entries=entries, access_logs=[], exception_logs=[])
    index = mod._build_log_index(parsed)

    nearby = mod._find_nearby_context(
        entries,
        observed_at=base + timedelta(seconds=10),
        line_index=11,
        seconds=2.0,
        log_index=index,
    )
    assert any("nearby traceback" in e.clean_content for e in nearby)
    # Far-away ERROR markers must not be pulled in when using the time index.
    assert not any("far away noise" in e.clean_content for e in nearby)
    # Indexed path should only consider buckets within the seconds radius.
    center = base + timedelta(seconds=10)
    for entry in nearby:
        assert abs((entry.observed_at - center).total_seconds()) <= 2.0


def test_nearest_access_log_uses_time_buckets() -> None:
    mod = _load()
    base = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)

    def make_access(i: int, status: int) -> object:
        observed = base + timedelta(seconds=i)
        entry = mod.LogEntry(
            line_index=i + 1,
            timestamp_text=observed.isoformat(),
            observed_at=observed,
            content=f"access {status}",
            clean_content=f'INFO: 1.1.1.1:1 - "POST /v1/x HTTP/1.1" {status} X',
        )
        return mod.AccessLog(
            entry=entry,
            method="POST",
            path="/v1/x",
            status_code=status,
            phrase="X",
        )

    # Many same-status access logs far from the exception plus one nearby.
    access_logs = [make_access(i, 429) for i in range(0, 200)]
    access_logs.append(make_access(49, 429))
    exception_entry = mod.LogEntry(
        line_index=50,
        timestamp_text=(base + timedelta(seconds=49)).isoformat(),
        observed_at=base + timedelta(seconds=49),
        content="exc",
        clean_content="pass_through_endpoint(): Exception occured - 429: limited",
    )
    exception_log = mod.ExceptionLog(
        entry=exception_entry, status_code=429, detail="limited"
    )
    parsed = mod.ParsedLogs(entries=[], access_logs=access_logs, exception_logs=[])
    index = mod._build_log_index(parsed)
    assert not hasattr(index, "access_by_status")
    center_bucket = int(exception_entry.observed_at.timestamp())
    assert center_bucket in index.access_by_second
    # Indexed path should only consult nearby second buckets, not the full list.
    considered_buckets = set(mod._iter_second_buckets(exception_entry.observed_at, 5.0))
    assert all(abs(bucket - center_bucket) <= 6 for bucket in considered_buckets)

    nearest = mod._nearest_access_log(
        exception_log,
        access_logs,
        used_access_indexes=set(),
        max_delta_seconds=5.0,
        log_index=index,
    )
    assert nearest is not None
    assert nearest.status_code == 429
    assert (
        abs((nearest.entry.observed_at - exception_entry.observed_at).total_seconds())
        <= 5.0
    )
    # Nearest should be the co-timed access around second 49, not second 0.
    assert (
        abs(
            nearest.entry.observed_at.timestamp()
            - exception_entry.observed_at.timestamp()
        )
        < 1.0
    )


def test_max_observations_limits_correlation_work() -> None:
    mod = _load()
    lines: list[str] = []
    # Many real proxy exception + access pairs.
    for i in range(0, 40, 2):
        lines.append(
            _line(
                i,
                "pass_through_endpoint(): Exception occured - 503: "
                '{"error":{"message":"upstream"}}',
            )
        )
        lines.append(
            _line(
                i + 1,
                'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
                "503 Service Unavailable",
            )
        )
    parsed = mod._parse_logs(lines)
    assert len(parsed.exception_logs) == 20

    limited = mod._build_observations(
        parsed_logs=parsed,
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
        max_observations=3,
    )
    assert len(limited) == 3

    unlimited = mod._build_observations(
        parsed_logs=parsed,
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
        max_observations=None,
    )
    assert len(unlimited) >= 3
    assert len(unlimited) > len(limited)


def test_limit_arg_help_mentions_early_stop() -> None:
    mod = _load()
    parser = mod._build_parser()
    action = next(a for a in parser._actions if "--limit" in a.option_strings)
    assert action.help is not None
    help_text = action.help.lower()
    assert (
        "correlation" in help_text or "producing" in help_text or "parse" in help_text
    )
    assert "exception" in help_text


def test_docker_logs_is_streamed_not_fully_buffered() -> None:
    """_iter_docker_log_lines must stream lines (RR-068 #2 streaming direction)."""
    mod = _load()
    source = inspect.getsource(mod._iter_docker_log_lines)
    assert "Popen" in source
    assert "communicate" not in source
    assert "readlines()" not in source
    # stdout must be iterated, not materialised as one list up front.
    assert "for line in process.stdout" in source or "for line in handle" in source


def test_log_signature_still_stable_across_line_indexes() -> None:
    """Cross-check residual #1 remains fixed alongside residual work."""
    mod = _load()
    sig = mod._log_signature(
        container="c",
        timestamp_text="2026-07-17T00:00:00.000000000Z",
        content="same",
    )
    assert len(sig) == 64
    assert "line_index" not in inspect.signature(mod._log_signature).parameters


def test_parse_logs_max_exception_logs_zero_accepts_none() -> None:
    """Programmatic max_exception_logs<=0 must not keep the first exception."""
    mod = _load()
    lines = [
        _line(
            0,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"e0"}}',
        ),
        _line(
            1,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
            "503 Service Unavailable",
        ),
    ]
    parsed = mod._parse_logs(lines, max_exception_logs=0)
    assert parsed.exception_logs == []
    # Access lines after a zero-exception limit should not be scanned either
    # (limit reached with no tail budget when max is non-positive).
    assert parsed.access_logs == []


def test_limit_arg_rejects_non_positive() -> None:
    mod = _load()
    parser = mod._build_parser()
    for bad in ("0", "-1"):
        try:
            parser.parse_args(
                ["--container", "c", "--environment", "e", "--limit", bad]
            )
            raise AssertionError(f"expected SystemExit for --limit {bad}")
        except SystemExit as exc:
            assert exc.code != 0
    args = parser.parse_args(
        ["--container", "c", "--environment", "e", "--limit", "2"]
    )
    assert args.limit == 2


def test_observation_provenance_metadata() -> None:
    """Source/provenance fields must identify docker-log backfill rows."""
    mod = _load()
    lines = [
        _line(
            0,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"real"}}',
        ),
        _line(
            1,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
            "503 Service Unavailable",
        ),
    ]
    parsed = mod._parse_logs(lines)
    observations = mod._build_observations(
        parsed_logs=parsed,
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
    )
    assert observations
    metadata = observations[0]["metadata"]
    assert metadata["source"] == "docker_log_backfill"
    assert metadata["parser"] == (
        "backfill_provider_error_observations_from_docker_logs"
    )
    assert metadata["parser_version"] == 1
    assert metadata["container"] == "litellm-dev"
    assert metadata["environment"] == "dev"
    assert metadata["source_kind"] in {"exception_log", "access_log"}
    assert isinstance(metadata["log_signature"], str)
    assert len(metadata["log_signature"]) == 64
    assert "docker_timestamp" in metadata
    assert "raw_log_excerpt" in metadata


def test_filter_new_observations_dedupes_existing_and_in_batch() -> None:
    """Idempotency: skip DB-existing and in-batch duplicate log_signature values."""
    mod = _load()
    lines = [
        _line(
            0,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"real"}}',
        ),
        _line(
            1,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
            "503 Service Unavailable",
        ),
        # Duplicate of the same logical exception/access pair later in the stream
        # is unlikely, but in-batch signature dedupe must still collapse clones.
    ]
    parsed = mod._parse_logs(lines)
    observations = mod._build_observations(
        parsed_logs=parsed,
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
    )
    assert observations
    # Clone the first observation to simulate an in-batch duplicate signature.
    cloned = dict(observations[0])
    cloned["metadata"] = dict(observations[0]["metadata"])
    with_dupes = observations + [cloned]
    existing = {observations[0]["metadata"]["log_signature"]}
    filtered_existing = mod._filter_new_observations(
        with_dupes, existing_signatures=existing
    )
    assert filtered_existing == []

    filtered_batch = mod._filter_new_observations(
        with_dupes, existing_signatures=set()
    )
    # One unique signature from the clone pair, plus any other distinct rows.
    signatures = [
        row["metadata"]["log_signature"]
        for row in filtered_batch
        if isinstance(row.get("metadata"), dict)
    ]
    assert len(signatures) == len(set(signatures))
    assert observations[0]["metadata"]["log_signature"] in signatures


def test_log_signature_stable_across_shifted_parse_windows() -> None:
    """Same timestamp+content must keep signature even when stream indexes shift."""
    mod = _load()
    core = [
        _line(
            5,
            "pass_through_endpoint(): Exception occured - 503: "
            '{"error":{"message":"real"}}',
        ),
        _line(
            6,
            'INFO: 1.2.3.4:9 - "POST /anthropic/v1/messages HTTP/1.1" '
            "503 Service Unavailable",
        ),
    ]
    noise = [_line(i, "INFO: idle keepalive") for i in range(5)]
    obs_a = mod._build_observations(
        parsed_logs=mod._parse_logs(core),
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
    )
    obs_b = mod._build_observations(
        parsed_logs=mod._parse_logs(noise + core),
        container="litellm-dev",
        environment="dev",
        max_correlation_seconds=20.0,
    )
    sigs_a = sorted(row["metadata"]["log_signature"] for row in obs_a)
    sigs_b = sorted(row["metadata"]["log_signature"] for row in obs_b)
    assert sigs_a == sigs_b
    assert all(
        a["metadata"]["docker_line_index"] != b["metadata"]["docker_line_index"]
        for a, b in zip(obs_a, obs_b)
    ) or len(obs_a) == 0  # indexes differ when noise shifts enumerate()
    # Explicit index shift check for the exception source line.
    assert obs_a[0]["metadata"]["docker_line_index"] != obs_b[0][
        "metadata"
    ]["docker_line_index"]
    assert (
        obs_a[0]["metadata"]["log_signature"]
        == obs_b[0]["metadata"]["log_signature"]
    )
