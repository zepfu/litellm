"""Scan the AAWM error JSONL file. Path lives in YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from hv2.checks.soft_fail import matching_signatures
from hv2.load_config import expand_string


def jsonl_path(config: Mapping[str, Any]) -> Path:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    spec = checks.get("error_jsonl") if isinstance(checks.get("error_jsonl"), dict) else {}
    raw = str(spec.get("path") or "{repo}/.analysis/alpha-error.jsonl")
    harness_dir = Path(__file__).resolve().parents[2]
    repo = harness_dir.parents[1]
    expanded = expand_string(raw, {"repo": str(repo), "home": str(Path.home())})
    return Path(expanded)


def snapshot_cursor(path: Path) -> int:
    if not path.is_file():
        return 0
    return path.stat().st_size


def scan_new_rows(
    config: Mapping[str, Any],
    *,
    before_size: int,
) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    spec = checks.get("error_jsonl") if isinstance(checks.get("error_jsonl"), dict) else {}
    path = jsonl_path(config)
    warnings: list[str] = []
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    soft_fail_matches: list[dict[str, Any]] = []
    empty = {
        "ok": True,
        "failures": [],
        "warnings": [],
        "rows": [],
        "soft_fail_matches": [],
        "path": str(path),
    }
    if not path.is_file():
        return empty
    data = path.read_bytes()
    new = data[before_size:] if before_size <= len(data) else data
    text = new.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"malformed JSONL row in {path}")
            continue
        if not isinstance(parsed, dict):
            continue
        rows.append(parsed)
        traceback = parsed.get("traceback")
        traceback_empty = traceback in (None, "", [], {})
        label = parsed.get("failure_kind") or parsed.get("message")
        matches = matching_signatures(
            config,
            text=_row_text(parsed),
            model=_row_models(parsed),
        )
        # Signature match + null traceback is a YAML-tunable warning (owl-alpha / MS-037).
        if matches and traceback_empty:
            soft_fail_matches.extend(matches)
            for match in matches:
                warnings.append(f"soft-fail ({match.get('name')}): {label}")
            continue
        if not traceback_empty:
            failures.append(f"new error JSONL row has traceback: {label}")
        elif spec.get("traceback_null_is_warning", True):
            warnings.append(f"structured terminal JSONL (traceback null): {label}")
    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "rows": rows,
        "soft_fail_matches": soft_fail_matches,
        "path": str(path),
    }


def _row_text(parsed: Mapping[str, Any]) -> str:
    parts = [
        str(parsed.get("message") or ""),
        str(parsed.get("failure_kind") or ""),
        str(parsed.get("error") or ""),
        str(parsed.get("error_code") or ""),
        str(parsed.get("reason") or ""),
        str(parsed.get("raw_text") or ""),
    ]
    context = parsed.get("context")
    if isinstance(context, dict):
        parts.append(str(context.get("failure_kind") or ""))
        parts.append(str(context.get("error_code") or ""))
    return "\n".join(part for part in parts if part)


def _row_models(parsed: Mapping[str, Any]) -> list[str]:
    keys = ("model", "model_id", "model_alias", "alias_model")
    found: list[str] = []
    seen: set[str] = set()
    for source in (parsed, parsed.get("context")):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if not value:
                continue
            token = str(value)
            if token not in seen:
                seen.add(token)
                found.append(token)
    return found
