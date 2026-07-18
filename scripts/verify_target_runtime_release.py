#!/usr/bin/env python3
"""Target-aware runtime/session_history verifier for AAWM LiteLLM releases."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0.0"
SUPPORTED_TARGETS = ("dev", "prod")
SUPPORTED_LANES = ("target-manifest", "session-history")
LANE_ALLOWLIST = list(SUPPORTED_LANES)

# Session-history marker scan bounds (fail-closed if no match within window).
DEFAULT_SESSION_HISTORY_ROW_LIMIT = 200
MAX_SESSION_HISTORY_ROW_LIMIT = 5000
DEFAULT_SESSION_HISTORY_LOOKBACK_HOURS = 168.0  # 7 days

TARGET_PROFILES: Dict[str, Dict[str, Any]] = {
    "dev": {
        "compose_project": "litellm",
        "container_name": "litellm-dev",
        "base_url": "http://127.0.0.1:4001",
        "db_name": "aawm_tristore",
        "source_mode": "bind_mount",
        "allowed_actions": [
            "read_only_http",
            "read_only_sql",
            "docker_inspect",
            "session_history_probe",
        ],
    },
    "prod": {
        "compose_project": None,
        "container_name": "aawm-litellm",
        "base_url": "http://127.0.0.1:4000",
        "db_name": "aawm_tristore",
        "source_mode": "released_image",
        "allowed_actions": ["read_only"],
    },
}

PROD_REQUIRED_CHECKPOINTS = (
    "release_runbook",
    "image_tag",
    "callback_wheel",
    "db_name",
    "source_mode_released_image",
)

_REDACT_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(Basic\s+)[A-Za-z0-9+/=]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}\b"), "sk-[REDACTED]"),
    (
        re.compile(
            r"(?i)((?:api[_-]?key|api[_-]?token|password|secret|token)\s*[=:]\s*)['\"]?"
            r"[^'\"\s&]+['\"]?"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"//([^:@/\s]+):([^@/\s]+)@"),
        r"//[REDACTED]:[REDACTED]@",
    ),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
]

_PROJECTS_ROOT_RE = re.compile(r"^/home/zepfu/projects/([^/]+)/")


def redact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): redact_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_structure(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return redact_text(value)
        return value
    return redact_text(value)


def _git_value(args: Sequence[str], workspace_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def _active_repository(workspace_root: Path) -> str:
    return workspace_root.name


def _referenced_artifact_owners(paths: Sequence[str]) -> List[Dict[str, str]]:
    owners: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in paths:
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        match = _PROJECTS_ROOT_RE.match(cleaned)
        if not match:
            continue
        repo = match.group(1)
        if repo in seen:
            continue
        seen.add(repo)
        owners.append({"path": cleaned, "referenced_repository": repo})
    return owners


def _resolve_db_dsn(cli_dsn: Optional[str]) -> Tuple[Optional[str], str]:
    if cli_dsn and cli_dsn.strip():
        return cli_dsn.strip(), "cli_db_dsn"
    for env_name in (
        "AAWM_DATABASE_URL",
        "AAWM_DIRECT_DATABASE_URL",
    ):
        val = os.getenv(env_name, "").strip()
        if val:
            return val, env_name
    host = os.getenv("AAWM_DB_HOST", "").strip()
    port = os.getenv("AAWM_DB_PORT", "5432").strip() or "5432"
    user = os.getenv("AAWM_DB_USER", "").strip()
    password = os.getenv("AAWM_DB_PASSWORD", "") or os.getenv("AAWM_DB_PWD", "")
    db_name = os.getenv("AAWM_DB_NAME", "").strip()
    if host and user and db_name:
        auth = f"{user}:{password}@" if password else f"{user}@"
        dsn = f"postgresql://{auth}{host}:{port}/{db_name}"
        return dsn, "AAWM_DB_*"
    return None, "missing"


def _extract_route_family(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for key in (
        "passthrough_route_family",
        "openai_passthrough_route_family",
        "route_family",
    ):
        val = metadata.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _marker_sources_for_row(row: Dict[str, Any], marker_id: str) -> List[str]:
    marker = str(marker_id or "").strip()
    if not marker:
        return []

    sources: List[str] = []
    for field in ("session_id", "trace_id", "litellm_call_id"):
        val = row.get(field)
        if val is not None and str(val).strip() == marker:
            sources.append(field)

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key, val in metadata.items():
            if isinstance(val, (str, int, float, bool)) and str(val).strip() == marker:
                sources.append(f"metadata.{key}")
    return sources


def _release_runbook_path(raw: Optional[str]) -> Optional[Path]:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    return Path(cleaned).expanduser()


def _release_runbook_is_file(raw: Optional[str]) -> bool:
    path = _release_runbook_path(raw)
    return path is not None and path.is_file()


def _prod_checkpoint_missing(args: argparse.Namespace) -> List[str]:
    missing: List[str] = []
    if not (args.release_runbook or "").strip():
        missing.append("release_runbook")
    elif not _release_runbook_is_file(args.release_runbook):
        # Fail closed: non-empty path that is missing or not a regular file
        # does not satisfy the prod checkpoint gate.
        missing.append("release_runbook_file")
    if not (args.image_tag or "").strip():
        missing.append("image_tag")
    if not (args.callback_wheel or "").strip():
        missing.append("callback_wheel")
    if not (args.db_name or "").strip():
        missing.append("db_name")
    profile = TARGET_PROFILES["prod"]
    if profile.get("source_mode") != "released_image":
        missing.append("source_mode_released_image")
    return missing


def _build_manifest(
    *,
    target: str,
    lanes_requested: Sequence[str],
    lanes_executed: Sequence[str],
    lanes_skipped: Sequence[str],
    workspace_root: Path,
    referenced_artifacts: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = dict(TARGET_PROFILES[target])
    db_name = (args.db_name or "").strip() or profile["db_name"]
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "compose_project": profile.get("compose_project"),
        "container_name": profile["container_name"],
        "base_url": profile["base_url"],
        "db_name": db_name,
        "source_mode": profile["source_mode"],
        "image_tag": (args.image_tag or "").strip() or None,
        "callback_wheel": (args.callback_wheel or "").strip() or None,
        "allowed_actions": list(profile["allowed_actions"]),
        "lane_allowlist": list(LANE_ALLOWLIST),
        "lanes_requested": list(lanes_requested),
        "lanes_executed": list(lanes_executed),
        "skipped_lanes": list(lanes_skipped),
        "redaction_policy": "mask_bearer_basic_sk_tokens_credentials_urls_ips",
        "provenance": {
            "workspace_root": str(workspace_root.resolve()),
            "active_repository": _active_repository(workspace_root),
            "referenced_artifact_owners": list(referenced_artifacts),
            "git_commit": _git_value(["rev-parse", "HEAD"], workspace_root),
            "git_branch": _git_value(
                ["rev-parse", "--abbrev-ref", "HEAD"], workspace_root
            ),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    runbook_path = _release_runbook_path(args.release_runbook)
    if runbook_path is not None:
        # Prefer resolved path when the file exists; otherwise keep expanded path
        # for diagnostics (prod gate rejects non-files before success paths).
        if runbook_path.is_file():
            manifest["release_runbook"] = str(runbook_path.resolve())
            manifest["release_runbook_exists"] = True
        else:
            manifest["release_runbook"] = str(runbook_path)
            manifest["release_runbook_exists"] = False
    if extra:
        manifest.update(extra)
    return manifest


def _failure_evidence(
    *,
    target: str,
    reason: str,
    missing: Optional[Sequence[str]] = None,
    workspace_root: Optional[Path] = None,
    lanes_requested: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "reason": redact_text(reason),
        "prod_read_only": target == "prod",
    }
    if missing:
        evidence["missing_checkpoint_fields"] = list(missing)
    if workspace_root is not None:
        evidence["provenance"] = {
            "workspace_root": str(workspace_root.resolve()),
            "active_repository": _active_repository(workspace_root),
        }
    if lanes_requested is not None:
        evidence["lanes_requested"] = list(lanes_requested)
    return evidence


def _emit_outputs(
    evidence: Dict[str, Any],
    *,
    emit_json: Optional[str],
    emit_md: Optional[str],
    quiet: bool,
) -> None:
    safe = redact_structure(evidence)
    if not quiet:
        print(json.dumps(safe, indent=2, sort_keys=True, default=str))  # noqa: T201
    if emit_json:
        path = Path(emit_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(safe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if emit_md:
        path = Path(emit_md).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Target runtime verification", ""]
        lines.append(f"- ok: `{safe.get('ok')}`")
        lines.append(f"- target: `{safe.get('target')}`")
        if safe.get("reason"):
            lines.append(f"- reason: {safe.get('reason')}")
        if safe.get("missing_checkpoint_fields"):
            lines.append(
                "- missing_checkpoint_fields: "
                + ", ".join(safe["missing_checkpoint_fields"])
            )
        if safe.get("manifest"):
            m = safe["manifest"]
            lines.append(f"- container: `{m.get('container_name')}`")
            lines.append(f"- db_name: `{m.get('db_name')}`")
            lines.append(f"- source_mode: `{m.get('source_mode')}`")
        if safe.get("session_history"):
            sh = safe["session_history"]
            lines.append(f"- persisted_row_id: `{sh.get('persisted_row_id')}`")
            lines.append(f"- marker_id: `{sh.get('marker_id')}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _session_history_row_limit(raw: Optional[int]) -> int:
    if raw is None:
        return DEFAULT_SESSION_HISTORY_ROW_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SESSION_HISTORY_ROW_LIMIT
    if value < 1:
        return 1
    if value > MAX_SESSION_HISTORY_ROW_LIMIT:
        return MAX_SESSION_HISTORY_ROW_LIMIT
    return value


def _session_history_created_after(
    *,
    created_after: Optional[str],
    lookback_hours: Optional[float],
) -> Optional[datetime]:
    """Resolve a lower bound for session_history.created_at (UTC, inclusive)."""
    cleaned = (created_after or "").strip()
    if cleaned:
        # Accept trailing Z for ISO-8601 UTC.
        normalized = cleaned.replace("Z", "+00:00") if cleaned.endswith("Z") else cleaned
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if lookback_hours is None:
        return None
    hours = float(lookback_hours)
    if hours <= 0:
        return None
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _session_history_lookups(
    session_id: Optional[str],
    trace_id: Optional[str],
    litellm_call_id: Optional[str],
) -> List[Tuple[str, str]]:
    lookups: List[Tuple[str, str]] = []
    if session_id:
        lookups.append(("session_id", session_id))
    if trace_id:
        lookups.append(("trace_id", trace_id))
    if litellm_call_id:
        lookups.append(("litellm_call_id", litellm_call_id))
    return lookups


def _fetch_session_history_candidates(
    *,
    dsn: str,
    lookups: Sequence[Tuple[str, str]],
    row_limit: int,
    created_after: Optional[datetime],
) -> List[Dict[str, Any]]:
    import psycopg
    import psycopg.rows

    limit = _session_history_row_limit(row_limit)
    select_sql = """
        SELECT
            id,
            session_id,
            trace_id,
            litellm_call_id,
            inbound_model_alias,
            provider,
            model,
            repository,
            tenant_id,
            created_at,
            metadata
        FROM public.session_history
    """
    # Bound by time window (when provided) plus a configurable LIMIT so long-lived
    # shared session_ids are not truncated by an arbitrary top-10 DESC scan.
    order_sql = f"ORDER BY created_at DESC LIMIT {limit}"
    rows: List[Dict[str, Any]] = []
    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO public")
            for field, value in lookups:
                if created_after is not None:
                    cur.execute(
                        f"{select_sql} WHERE {field} = %s AND created_at >= %s {order_sql}",
                        [value, created_after],
                    )
                else:
                    cur.execute(
                        f"{select_sql} WHERE {field} = %s {order_sql}",
                        [value],
                    )
                rows.extend(dict(row) for row in cur.fetchall())
    return rows


def _select_marker_session_history_row(
    rows: Sequence[Dict[str, Any]],
    marker_id: str,
    db_name: str,
) -> Optional[Dict[str, Any]]:
    for candidate in rows:
        candidate_row = dict(candidate)
        marker_sources = _marker_sources_for_row(candidate_row, marker_id)
        if not marker_sources:
            continue
        candidate_row["target_db_name"] = db_name
        candidate_row["metadata_redacted"] = True
        candidate_row["marker_match_sources"] = marker_sources
        meta = candidate_row.get("metadata")
        candidate_row["route_family"] = _extract_route_family(meta)
        candidate_row.pop("metadata", None)
        return candidate_row
    return None


def query_session_history_row(
    *,
    dsn: str,
    db_name: str,
    marker_id: str,
    session_id: Optional[str],
    trace_id: Optional[str],
    litellm_call_id: Optional[str],
    row_limit: int = DEFAULT_SESSION_HISTORY_ROW_LIMIT,
    created_after: Optional[datetime] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    lookups = _session_history_lookups(session_id, trace_id, litellm_call_id)
    if not lookups:
        return None, db_name
    rows = _fetch_session_history_candidates(
        dsn=dsn,
        lookups=lookups,
        row_limit=row_limit,
        created_after=created_after,
    )
    return _select_marker_session_history_row(rows, marker_id, db_name), db_name


def _session_history_scan_bounds(
    args: argparse.Namespace,
) -> Tuple[int, Optional[datetime]]:
    row_limit = _session_history_row_limit(
        getattr(args, "session_history_row_limit", None)
    )
    created_after = _session_history_created_after(
        created_after=getattr(args, "session_history_created_after", None),
        lookback_hours=getattr(args, "session_history_lookback_hours", None),
    )
    return row_limit, created_after


def _session_history_bound_fields(
    *,
    row_limit: int,
    created_after: Optional[datetime],
) -> Dict[str, Any]:
    return {
        "session_history_row_limit": row_limit,
        "session_history_created_after": (
            created_after.isoformat() if created_after is not None else None
        ),
    }


def _collect_session_history_field_mismatches(
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    mismatches: List[str] = []
    checks = (
        ("expected_inbound_alias", "inbound_model_alias"),
        ("expected_repository", "repository"),
        ("expected_tenant_id", "tenant_id"),
    )
    for arg_name, field in checks:
        expected_raw = (getattr(args, arg_name) or "").strip()
        if not expected_raw:
            continue
        actual = row.get(field)
        if actual != expected_raw:
            mismatches.append(f"{field} expected {expected_raw!r} got {actual!r}")
    return mismatches


def _run_session_history_lane(
    *,
    target: str,
    args: argparse.Namespace,
    workspace_root: Path,
    referenced_artifacts: Sequence[Dict[str, str]],
    lanes_requested: Sequence[str],
) -> Tuple[int, Dict[str, Any]]:
    profile = TARGET_PROFILES[target]
    marker_id = (args.marker_id or "").strip()
    if not marker_id:
        ev = _failure_evidence(
            target=target,
            reason="session-history lane requires --marker-id",
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        return 2, ev

    has_corr = any(
        (getattr(args, name) or "").strip()
        for name in ("session_id", "trace_id", "litellm_call_id")
    )
    if not has_corr:
        ev = _failure_evidence(
            target=target,
            reason=(
                "session-history lane requires --marker-id and at least one of "
                "--session-id, --trace-id, --litellm-call-id"
            ),
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        return 2, ev

    dsn, dsn_source = _resolve_db_dsn(args.db_dsn)
    db_name = (args.db_name or "").strip() or profile["db_name"]
    if not dsn:
        ev = _failure_evidence(
            target=target,
            reason="session-history lane requires --db-dsn or AAWM database env",
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        ev["db_name"] = db_name
        return 2, ev

    try:
        row_limit, created_after = _session_history_scan_bounds(args)
    except ValueError as exc:
        ev = _failure_evidence(
            target=target,
            reason=(
                "session-history lane invalid --session-history-created-after: "
                f"{redact_text(str(exc))}"
            ),
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        ev["db_name"] = db_name
        return 2, ev
    bound_fields = _session_history_bound_fields(
        row_limit=row_limit, created_after=created_after
    )

    try:
        row, _ = query_session_history_row(
            dsn=dsn,
            db_name=db_name,
            marker_id=marker_id,
            session_id=(args.session_id or "").strip() or None,
            trace_id=(args.trace_id or "").strip() or None,
            litellm_call_id=(args.litellm_call_id or "").strip() or None,
            row_limit=row_limit,
            created_after=created_after,
        )
    except Exception as exc:  # noqa: BLE001 — surface redacted DB failures only
        # Never print raw DSN (may embed password from AAWM_DB_*).
        redacted = redact_text(str(exc))
        # Also strip any accidental embedding of the full DSN string.
        if dsn:
            redacted = redacted.replace(dsn, "<redacted-dsn>")
        ev = _failure_evidence(
            target=target,
            reason=f"session-history query failed ({type(exc).__name__}): {redacted}",
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        ev.update(
            {
                "db_name": db_name,
                "dsn_source": dsn_source,
                "marker_id": marker_id,
                "http_success_not_accepted": True,
                **bound_fields,
            }
        )
        return 2, ev
    if row is None:
        ev = _failure_evidence(
            target=target,
            reason="no persisted session_history row matched marker and correlation ids",
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        ev.update(
            {
                "db_name": db_name,
                "dsn_source": dsn_source,
                "marker_id": marker_id,
                "session_id": (args.session_id or "").strip() or None,
                "trace_id": (args.trace_id or "").strip() or None,
                "litellm_call_id": (args.litellm_call_id or "").strip() or None,
                "http_success_not_accepted": True,
                **bound_fields,
            }
        )
        ev["manifest"] = _build_manifest(
            target=target,
            lanes_requested=lanes_requested,
            lanes_executed=["session-history"],
            lanes_skipped=[],
            workspace_root=workspace_root,
            referenced_artifacts=referenced_artifacts,
            args=args,
        )
        return 3, ev

    mismatches = _collect_session_history_field_mismatches(row, args)
    if mismatches:
        ev = _failure_evidence(
            target=target,
            reason="persisted row failed expected field checks: " + "; ".join(mismatches),
            workspace_root=workspace_root,
            lanes_requested=lanes_requested,
        )
        ev["field_mismatches"] = mismatches
        ev["manifest"] = _build_manifest(
            target=target,
            lanes_requested=lanes_requested,
            lanes_executed=["session-history"],
            lanes_skipped=[],
            workspace_root=workspace_root,
            referenced_artifacts=referenced_artifacts,
            args=args,
        )
        ev["session_history"] = {
            "persisted_row_id": row.get("id"),
            "marker_id": marker_id,
            "repository": row.get("repository"),
            "tenant_id": row.get("tenant_id"),
        }
        return 4, ev

    session_evidence = {
        "target": target,
        "container_name": profile["container_name"],
        "profile_source_mode": profile["source_mode"],
        "database": db_name,
        "dsn_source": dsn_source,
        "marker_id": marker_id,
        **bound_fields,
        "marker_match_sources": row.get("marker_match_sources"),
        "persisted_row_id": row.get("id"),
        "session_id": row.get("session_id"),
        "trace_id": row.get("trace_id"),
        "litellm_call_id": row.get("litellm_call_id"),
        "inbound_model_alias": row.get("inbound_model_alias"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "route_family": row.get("route_family"),
        "repository": row.get("repository"),
        "tenant_id": row.get("tenant_id"),
        "row_created_at": (
            row.get("created_at").isoformat()
            if hasattr(row.get("created_at"), "isoformat")
            else row.get("created_at")
        ),
        "metadata_redacted": True,
        "raw_metadata_omitted": True,
    }
    manifest = _build_manifest(
        target=target,
        lanes_requested=lanes_requested,
        lanes_executed=["session-history"],
        lanes_skipped=[],
        workspace_root=workspace_root,
        referenced_artifacts=referenced_artifacts,
        args=args,
    )
    evidence = {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "lanes_requested": list(lanes_requested),
        "manifest": manifest,
        "session_history": session_evidence,
        "provenance": manifest["provenance"],
    }
    return 0, evidence


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify AAWM target runtime profile and optional session_history persistence.",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=SUPPORTED_TARGETS,
        help="Deployment target profile (dev or prod).",
    )
    parser.add_argument(
        "--lane",
        action="append",
        dest="lanes",
        help=f"Verification lane; repeat. Supported: {", ".join(SUPPORTED_LANES)}. Default: target-manifest only.",
    )
    parser.add_argument("--emit-json", default=None, help="Optional path for JSON evidence.")
    parser.add_argument("--emit-md", default=None, help="Optional path for Markdown summary.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout JSON.")
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Active workspace root (default: git root or cwd).",
    )
    parser.add_argument(
        "--referenced-artifact-path",
        action="append",
        default=[],
        dest="referenced_artifact_paths",
        help="Referenced artifact path; repeatable. Does not override active repository.",
    )
    parser.add_argument("--release-runbook", default=None, help="Prod checkpoint: runbook path.")
    parser.add_argument("--image-tag", default=None, help="Released image tag (manifest/prod).")
    parser.add_argument("--callback-wheel", default=None, help="Callback wheel id/path (prod).")
    parser.add_argument("--db-name", default=None, help="Database name override.")
    parser.add_argument("--db-dsn", default=None, help="PostgreSQL DSN for session-history lane.")
    parser.add_argument("--marker-id", default=None, help="Correlation marker in session_history metadata.")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--trace-id", default=None)
    parser.add_argument("--litellm-call-id", default=None)
    parser.add_argument("--expected-inbound-alias", default=None)
    parser.add_argument("--expected-repository", default=None)
    parser.add_argument("--expected-tenant-id", default=None)
    parser.add_argument(
        "--session-history-row-limit",
        type=int,
        default=DEFAULT_SESSION_HISTORY_ROW_LIMIT,
        help=(
            "Max session_history rows to scan per correlation field "
            f"(default {DEFAULT_SESSION_HISTORY_ROW_LIMIT}, "
            f"hard cap {MAX_SESSION_HISTORY_ROW_LIMIT})."
        ),
    )
    parser.add_argument(
        "--session-history-lookback-hours",
        type=float,
        default=DEFAULT_SESSION_HISTORY_LOOKBACK_HOURS,
        help=(
            "Only consider session_history rows with created_at within this many "
            f"hours (default {DEFAULT_SESSION_HISTORY_LOOKBACK_HOURS}). "
            "Set <=0 to disable the time window."
        ),
    )
    parser.add_argument(
        "--session-history-created-after",
        default=None,
        help=(
            "Absolute lower bound for session_history.created_at (ISO-8601). "
            "Overrides --session-history-lookback-hours when set."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: PLR0915
    args = _parse_args(argv)
    target = args.target
    lanes = args.lanes if args.lanes else ["target-manifest"]
    unknown = [lane for lane in lanes if lane not in SUPPORTED_LANES]
    if unknown:
        ev = _failure_evidence(
            target=target,
            reason=f"unsupported lane(s): {', '.join(unknown)}",
            lanes_requested=lanes,
        )
        _emit_outputs(ev, emit_json=args.emit_json, emit_md=args.emit_md, quiet=args.quiet)
        return 2

    if target == "prod":
        missing = _prod_checkpoint_missing(args)
        if missing:
            reason = (
                "prod target refused before probes: prod is read-only and requires "
                "checkpoint fields: existing --release-runbook file, --image-tag, "
                "--callback-wheel, --db-name, and source_mode released_image"
            )
            ev = _failure_evidence(
                target=target,
                reason=reason,
                missing=missing,
                lanes_requested=lanes,
            )
            ev["message"] = "prod mutation is never allowed; verification refused fail-closed"
            _emit_outputs(
                ev,
                emit_json=args.emit_json,
                emit_md=args.emit_md,
                quiet=args.quiet,
            )
            return 1

    workspace_root = Path(
        args.workspace_root or os.getcwd()
    ).expanduser()
    if args.workspace_root is None:
        git_root = _git_value(["rev-parse", "--show-toplevel"], Path(os.getcwd()))
        if git_root:
            workspace_root = Path(git_root)
    referenced = _referenced_artifact_owners(args.referenced_artifact_paths or [])

    lanes_set = list(dict.fromkeys(lanes))
    executed: List[str] = []
    skipped: List[str] = []
    evidence: Dict[str, Any] = {}
    exit_code = 0

    if "target-manifest" in lanes_set:
        executed.append("target-manifest")
        skipped = [lane for lane in LANE_ALLOWLIST if lane not in lanes_set]
        manifest = _build_manifest(
            target=target,
            lanes_requested=lanes_set,
            lanes_executed=executed.copy(),
            lanes_skipped=skipped,
            workspace_root=workspace_root,
            referenced_artifacts=referenced,
            args=args,
        )
        evidence = {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "target": target,
            "lanes_requested": lanes_set,
            "manifest": manifest,
            "provenance": manifest["provenance"],
        }

    if "session-history" in lanes_set:
        code, sh_evidence = _run_session_history_lane(
            target=target,
            args=args,
            workspace_root=workspace_root,
            referenced_artifacts=referenced,
            lanes_requested=lanes_set,
        )
        if "target-manifest" not in lanes_set:
            evidence = sh_evidence
            exit_code = code
        else:
            if sh_evidence.get("ok"):
                evidence["session_history"] = sh_evidence.get("session_history")
                evidence["lanes_executed"] = list(
                    dict.fromkeys(executed + ["session-history"])
                )
            else:
                evidence = sh_evidence
                _emit_outputs(
                    evidence,
                    emit_json=args.emit_json,
                    emit_md=args.emit_md,
                    quiet=args.quiet,
                )
                return code
        if "session-history" not in (evidence.get("lanes_executed") or executed):
            executed.append("session-history")

    if evidence.get("ok") and "lanes_executed" not in evidence:
        evidence["lanes_executed"] = executed
    if evidence.get("ok") and evidence.get("manifest"):
        evidence["manifest"]["lanes_executed"] = evidence.get(
            "lanes_executed", executed
        )
        evidence["manifest"]["skipped_lanes"] = [
            lane for lane in LANE_ALLOWLIST if lane not in lanes_set
        ]

    _emit_outputs(
        evidence,
        emit_json=args.emit_json,
        emit_md=args.emit_md,
        quiet=args.quiet,
    )
    if exit_code != 0:
        return exit_code
    return 0 if evidence.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
