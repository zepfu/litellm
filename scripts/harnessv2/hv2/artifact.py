"""JSON artifact writer. Redact secrets; stamp git identity."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hv2.load_config import as_str_list, repo_root_from_harness_dir


def git_stamp(repo: Path | None = None) -> dict[str, str]:
    root = repo or repo_root_from_harness_dir(Path(__file__).resolve().parents[1])
    stamp = {"commit": "", "branch": "", "dirty": "unknown"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if commit.returncode == 0:
            stamp["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if branch.returncode == 0:
            stamp["branch"] = branch.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if dirty.returncode == 0:
            stamp["dirty"] = "true" if dirty.stdout.strip() else "false"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return stamp


def redact_mapping(value: Any, redact_keys: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            if key_s.lower() in redact_keys or "authorization" in key_s.lower():
                out[key_s] = "<redacted>"
            else:
                out[key_s] = redact_mapping(item, redact_keys)
        return out
    if isinstance(value, list):
        return [redact_mapping(item, redact_keys) for item in value]
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_artifact(path: Path, payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    redact = {item.lower() for item in as_str_list((config.get("artifact") or {}).get("redact_headers"))}
    body = redact_mapping(dict(payload), redact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_SHA_DRIFT_WARNING = (
    "STRONG WARNING: git commit or dirty state changed mid-run "
    "(start {start} dirty={start_dirty} → end {end} dirty={end_dirty}). "
    "Results may not consistently represent the current checkout state. "
    "The run is not invalidated solely because of this drift."
)


def durable_dir(config: Mapping[str, Any], repo: Path | None = None) -> Path:
    spec = config.get("artifact") if isinstance(config.get("artifact"), dict) else {}
    raw = str(spec.get("durable_dir") or "{repo}/.analysis/harnessv2")
    root = repo or repo_root_from_harness_dir(Path(__file__).resolve().parents[1])
    return Path(raw.replace("{repo}", str(root)))


def durable_jsonl_path(
    config: Mapping[str, Any],
    *,
    started: str,
    kind: str,
    container: str,
    commit: str,
    repo: Path | None = None,
) -> Path:
    stamp = started.replace(":", "").replace("-", "")
    short = (commit or "unknown")[:12]
    safe_kind = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in kind) or "run"
    safe_container = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in container) or "instance"
    return durable_dir(config, repo) / f"{stamp}-{safe_kind}-{safe_container}-{short}.jsonl"


def append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event), sort_keys=True, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha_drift_warning(start: Mapping[str, str], end: Mapping[str, str]) -> str | None:
    start_commit = str(start.get("commit") or "")
    end_commit = str(end.get("commit") or "")
    start_dirty = str(start.get("dirty") or "")
    end_dirty = str(end.get("dirty") or "")
    drifted = bool(start_commit and end_commit and start_commit != end_commit)
    dirty_flipped = start_dirty == "false" and end_dirty == "true"
    if not drifted and not dirty_flipped:
        return None
    return _SHA_DRIFT_WARNING.format(
        start=start_commit or "unknown",
        start_dirty=start_dirty or "unknown",
        end=end_commit or "unknown",
        end_dirty=end_dirty or "unknown",
    )


def step_is_halt(payload: Mapping[str, Any]) -> bool:
    """True when a step hit a logging-regression stop signal."""

    needles = (
        "Traceback (most recent call last)",
        "Exception in ASGI application",
        "leftover uvicorn access line",
        "runtime logs contained forbidden substring",
    )
    blobs: list[str] = []
    for item in payload.get("failures") or []:
        blobs.append(str(item))
    for hit in payload.get("forbidden_hits") or []:
        if isinstance(hit, dict):
            blobs.append(str(hit.get("kind") or ""))
            blobs.append(str(hit.get("substring") or ""))
        else:
            blobs.append(str(hit))
    text = "\n".join(blobs)
    return any(needle in text for needle in needles)


def bounded_step_detail(payload: Mapping[str, Any], *, limit: int = 800) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for key in (
        "status",
        "url",
        "path",
        "name",
        "bytes",
        "ok",
        "halted",
        "skipped",
        "reason",
    ):
        if key in payload:
            detail[key] = payload[key]
    failures = [str(item)[:limit] for item in (payload.get("failures") or [])]
    if failures:
        detail["failures"] = failures[:20]
    hits = payload.get("forbidden_hits") or []
    if hits:
        detail["forbidden_hits"] = hits[:10]
    preview = payload.get("pane_preview") or payload.get("text")
    if isinstance(preview, str) and preview:
        detail["preview"] = preview[-limit:]
    return detail
