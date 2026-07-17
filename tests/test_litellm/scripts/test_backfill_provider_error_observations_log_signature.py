"""RR-068: docker log backfill signatures must be stable across windows."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_provider_error_observations_from_docker_logs.py"


def _load_backfill_module() -> ModuleType:
    module_name = "backfill_provider_error_observations_from_docker_logs"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_log_signature_ignores_stream_relative_line_index() -> None:
    mod = _load_backfill_module()
    container = "litellm-dev"
    timestamp_text = "2026-07-17T12:34:56.789012345Z"
    content = (
        'INFO: 172.18.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" '
        "429 Too Many Requests"
    )

    sig_a = mod._log_signature(
        container=container,
        timestamp_text=timestamp_text,
        content=content,
    )
    sig_b = mod._log_signature(
        container=container,
        timestamp_text=timestamp_text,
        content=content,
    )

    assert sig_a == sig_b
    assert len(sig_a) == 64

    # Same logical line with different stream-relative indexes (as would happen
    # when --since/--until shifts the docker logs window) must still match.
    entry_window_a = mod.LogEntry(
        line_index=10,
        timestamp_text=timestamp_text,
        observed_at=datetime(2026, 7, 17, 12, 34, 56, 789012, tzinfo=timezone.utc),
        content=content,
        clean_content=content,
    )
    entry_window_b = mod.LogEntry(
        line_index=9999,
        timestamp_text=timestamp_text,
        observed_at=datetime(2026, 7, 17, 12, 34, 56, 789012, tzinfo=timezone.utc),
        content=content,
        clean_content=content,
    )
    meta_a = mod._build_metadata(
        container=container,
        environment="dev",
        source_entry=entry_window_a,
        source_kind="access_log",
        raw_excerpt=content,
        normalized_error_text=content,
        request_id=None,
        access_log=None,
        context_lines=[],
        error_origin="provider_exception",
    )
    meta_b = mod._build_metadata(
        container=container,
        environment="dev",
        source_entry=entry_window_b,
        source_kind="access_log",
        raw_excerpt=content,
        normalized_error_text=content,
        request_id=None,
        access_log=None,
        context_lines=[],
        error_origin="provider_exception",
    )

    assert meta_a["log_signature"] == meta_b["log_signature"]
    assert meta_a["docker_line_index"] != meta_b["docker_line_index"]
    assert meta_a["log_signature"] == sig_a


def test_log_signature_changes_when_stable_fields_change() -> None:
    mod = _load_backfill_module()
    base = {
        "container": "litellm-dev",
        "timestamp_text": "2026-07-17T12:34:56.789012345Z",
        "content": "Exception occured - 503: upstream unavailable",
    }
    base_sig = mod._log_signature(**base)

    assert (
        mod._log_signature(
            container="other-container",
            timestamp_text=base["timestamp_text"],
            content=base["content"],
        )
        != base_sig
    )
    assert (
        mod._log_signature(
            container=base["container"],
            timestamp_text="2026-07-17T12:34:57.000000000Z",
            content=base["content"],
        )
        != base_sig
    )
    assert (
        mod._log_signature(
            container=base["container"],
            timestamp_text=base["timestamp_text"],
            content="Exception occured - 500: different error",
        )
        != base_sig
    )


def test_log_signature_does_not_accept_line_index_kwarg() -> None:
    """Regression guard: stream-relative index must not be part of the API."""
    mod = _load_backfill_module()
    import inspect

    params = inspect.signature(mod._log_signature).parameters
    assert "line_index" not in params
    assert set(params) == {"container", "timestamp_text", "content"}
