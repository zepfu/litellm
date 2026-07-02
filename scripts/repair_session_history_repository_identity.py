#!/usr/bin/env python3
"""Repair malformed repository identity values in public.session_history."""

import argparse
from collections import Counter
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psycopg
import psycopg.rows

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_repo_dotenv() -> None:
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned_value = value.strip()
        if (
            len(cleaned_value) >= 2
            and cleaned_value[0] == cleaned_value[-1]
            and cleaned_value[0] in {'"', "'"}
        ):
            cleaned_value = cleaned_value[1:-1]
        os.environ[key] = cleaned_value


_load_repo_dotenv()

from litellm.integrations.aawm_agent_identity import (  # noqa: E402
    _AAWM_REPOSITORY_AGENT_ID_RE,
    _AAWM_REPOSITORY_AGENT_ROLE_VALUES,
    _AAWM_REPOSITORY_PLACEHOLDER_VALUES,
    _AAWM_REPOSITORY_WAVE_AGENT_RE,
    _CODEX_MEMORY_REPOSITORY_SUFFIX,
    _build_aawm_dsn,
    _extract_repository_identity_from_metadata_sources,
    _normalize_repository_identity,
)


def _build_aawm_admin_dsn() -> Optional[str]:
    direct_dsn = os.getenv("AAWM_DIRECT_DATABASE_URL")
    if direct_dsn and direct_dsn.strip():
        return direct_dsn.strip()
    return _build_aawm_dsn()


_VALID_IDENTITY_SQL = r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?( \(memory\))?$"
_TRANSCRIPT_ARTIFACT_SQL = r"^(rollout-[0-9]{4}(-[A-Za-z0-9_.-]*)?|.*\.jsonl?)( \(memory\))?$"
_AGENT_IDENTITY_SQL = _AAWM_REPOSITORY_AGENT_ID_RE.pattern
_WAVE_AGENT_IDENTITY_SQL = _AAWM_REPOSITORY_WAVE_AGENT_RE.pattern
_DISALLOWED_IDENTITY_VALUES = tuple(
    sorted(_AAWM_REPOSITORY_PLACEHOLDER_VALUES | _AAWM_REPOSITORY_AGENT_ROLE_VALUES)
)
_METADATA_IDENTITY_KEYS = (
    "repository",
    "source_repository",
    "aawm_repository",
    "repo",
    "repo_name",
    "repository_name",
    "git_repository",
    "vcs_repository",
    "workspace_root",
    "workspaceRoot",
    "project_root",
    "projectRoot",
    "root_path",
    "rootPath",
    "working_directory",
    "workingDirectory",
    "cwd_path",
    "cwdPath",
    "cwd_uri",
    "cwdUri",
    "aawm_claude_project",
    "aawm_d1_452_referenced_artifact_owner",
    "tenant_id",
    "aawm_tenant_id",
)
_ROW_METADATA_REPOSITORY_PRIORITY = (
    "aawm_d1_452_referenced_artifact_owner",
    "aawm_claude_project",
    "aawm_repository",
    "repository",
    "source_repository",
    "repo",
    "repo_name",
    "repository_name",
    "git_repository",
    "vcs_repository",
    "workspace_root",
    "workspaceRoot",
    "project_root",
    "projectRoot",
    "root_path",
    "rootPath",
    "working_directory",
    "workingDirectory",
    "cwd_path",
    "cwdPath",
    "cwd_uri",
    "cwdUri",
)
_ROW_METADATA_TENANT_PRIORITY = (
    "aawm_tenant_id",
    "tenant_id",
)
_NOISY_REPO_PREFIX_PATTERNS = (
    re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+)\s+(?:all|commits|files)="),
    re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+)(?:\\\\n|\\n|\n)+"),
)
_REPOSITORY_UNRESOLVED_KEYS = (
    "session_history_repository_unresolved",
    "session_history_repository_unresolved_reason",
    "repository_identity_classified_at",
    "repository_identity_classification_source",
    "repository_tenant_fallback_skipped",
    "trace_user_tenant_fallback_skipped",
)


def _safe_json_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _load_known_repositories(projects_dir: Path) -> set[str]:
    if not projects_dir.exists():
        return {REPO_ROOT.name}
    return {
        path.name
        for path in projects_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def _repository_from_cwd(cwd: str, known_repositories: set[str]) -> Optional[str]:
    cleaned = str(cwd or "").strip().rstrip("/")
    if not cleaned:
        return None
    name = cleaned.rsplit("/", 1)[-1]
    if name in known_repositories:
        return name
    return None


def _load_rollout_repository_map(
    memories_dir: Path,
    known_repositories: set[str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    memory_file = memories_dir / "MEMORY.md"
    if memory_file.exists():
        for line in memory_file.read_text(errors="replace").splitlines():
            rollout_match = re.search(r"rollout_path=([^,\)]+)", line)
            thread_match = re.search(r"thread_id=([0-9a-f-]{12,})", line)
            cwd_match = re.search(r"cwd=([^,\)]+)", line)
            if not cwd_match:
                continue
            repository = _repository_from_cwd(
                cwd_match.group(1).strip(),
                known_repositories,
            )
            if not repository:
                continue
            if rollout_match:
                rollout_path = rollout_match.group(1).strip()
                mapping[Path(rollout_path).name] = repository
                mapping[Path(rollout_path).with_suffix(".json").name] = repository
            if thread_match:
                mapping[thread_match.group(1)] = repository

    summaries_dir = memories_dir / "rollout_summaries"
    if summaries_dir.exists():
        for summary_path in summaries_dir.glob("*.md"):
            try:
                text = summary_path.read_text(errors="replace")
            except OSError:
                continue
            header = "\n".join(text.splitlines()[:8])
            thread_match = re.search(r"^thread_id:\s*([0-9a-f-]{12,})", header, re.M)
            rollout_match = re.search(r"^rollout_path:\s*(\S+)", header, re.M)
            cwd_match = re.search(r"^cwd:\s*(\S+)", header, re.M)
            if not cwd_match:
                continue
            repository = _repository_from_cwd(cwd_match.group(1), known_repositories)
            if not repository:
                continue
            if thread_match:
                mapping[thread_match.group(1)] = repository
            if rollout_match:
                rollout_path = rollout_match.group(1)
                mapping[Path(rollout_path).name] = repository
                mapping[Path(rollout_path).with_suffix(".json").name] = repository
    return mapping


def _repo_base(identity: str) -> str:
    if identity.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        return identity[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]
    return identity


def _has_memory_suffix(value: Any) -> bool:
    return isinstance(value, str) and value.strip().endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX)


def _rollout_repository_candidate(
    value: Any,
    rollout_repository_map: Dict[str, str],
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().strip("`'\"").strip("/")
    if text.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        text = text[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]
    if not text:
        return None
    names = {text, Path(text).name}
    for name in list(names):
        if name.endswith(".json"):
            names.add(f"{name}l")
        elif name.endswith(".jsonl"):
            names.add(name[:-1])
    thread_match = re.search(r"([0-9a-f]{8}-[0-9a-f-]{20,})", text, re.I)
    if thread_match:
        names.add(thread_match.group(1))
    for name in names:
        repository = rollout_repository_map.get(name)
        if repository:
            if _has_memory_suffix(value):
                return f"{repository}{_CODEX_MEMORY_REPOSITORY_SUFFIX}"
            return repository
    return None


def _is_known_repository(identity: Optional[str], known_repositories: set[str]) -> bool:
    if not identity:
        return False
    base = _repo_base(identity)
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    return base in known_repositories


def _repair_noisy_identity(
    value: Any, known_repositories: set[str]
) -> Optional[str]:
    normalized = _normalize_repository_identity(value)
    if normalized and _is_known_repository(normalized, known_repositories):
        return normalized
    if not isinstance(value, str):
        return None

    text = value.strip()
    memory_suffix = _CODEX_MEMORY_REPOSITORY_SUFFIX in text
    for pattern in _NOISY_REPO_PREFIX_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        candidate = match.group("repo")
        if candidate in known_repositories:
            if memory_suffix:
                return f"{candidate}{_CODEX_MEMORY_REPOSITORY_SUFFIX}"
            return candidate
    return None


def _is_bad_repository_fragment(value: Any, known_repositories: set[str]) -> bool:
    if not isinstance(value, str):
        return False
    if _is_known_repository(value, known_repositories):
        return False
    text = value.strip()
    if not text:
        return False
    base = _repo_base(text).strip().strip("/").lower()
    if base in _DISALLOWED_IDENTITY_VALUES:
        return True
    if re.fullmatch(_TRANSCRIPT_ARTIFACT_SQL, text, re.IGNORECASE):
        return True
    return _normalize_repository_identity(text) is None


def _known_repository_candidate(
    value: Any,
    known_repositories: set[str],
) -> Optional[str]:
    repaired = _repair_noisy_identity(value, known_repositories)
    if repaired and _is_known_repository(repaired, known_repositories):
        return repaired
    return None


def _best_row_repository_candidate(
    row: Dict[str, Any],
    known_repositories: set[str],
) -> Optional[Tuple[str, str, int]]:
    metadata = _safe_json_metadata(row.get("metadata"))

    for priority, key in enumerate(_ROW_METADATA_REPOSITORY_PRIORITY):
        repository = _known_repository_candidate(metadata.get(key), known_repositories)
        if repository:
            return repository, f"session_metadata.{key}", priority

    repository = _extract_repository_identity_from_metadata_sources(
        ("session_history.metadata", metadata)
    )
    if _is_known_repository(repository, known_repositories):
        return repository, "session_metadata.recursive_repository", 40

    repository = _known_repository_candidate(row.get("repository"), known_repositories)
    if repository:
        return repository, "session_history.repository", 50

    for offset, key in enumerate(_ROW_METADATA_TENANT_PRIORITY):
        repository = _known_repository_candidate(metadata.get(key), known_repositories)
        if repository:
            return repository, f"session_metadata.{key}", 60 + offset

    repository = _known_repository_candidate(row.get("tenant_id"), known_repositories)
    if repository:
        return repository, "session_history.tenant_id", 70

    return None


def _is_grok_row(row: Dict[str, Any]) -> bool:
    provider = str(row.get("provider") or "").lower()
    model = str(row.get("model") or "").lower()
    return provider == "xai" or "grok" in model


def _build_session_repository_map(
    rows: list[Dict[str, Any]],
    known_repositories: set[str],
) -> Dict[str, Dict[str, Any]]:
    best_by_session: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        session_id = row.get("session_id")
        if not session_id:
            continue
        candidate = _best_row_repository_candidate(row, known_repositories)
        if candidate is None:
            continue
        repository, source, priority = candidate
        existing = best_by_session.get(session_id)
        if existing is not None and int(existing["priority"]) <= priority:
            continue
        best_by_session[session_id] = {
            "repository": repository,
            "source": source,
            "priority": priority,
            "source_row_id": row.get("id"),
        }
    return best_by_session


def _build_unique_session_repository_map(
    rows: list[Dict[str, Any]],
    known_repositories: set[str],
) -> Dict[str, Dict[str, Any]]:
    candidates_by_session: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        session_id = row.get("session_id")
        if not session_id:
            continue
        candidate = _best_row_repository_candidate(row, known_repositories)
        if candidate is None:
            continue
        repository, source, priority = candidate
        session_key = str(session_id)
        session_candidates = candidates_by_session.setdefault(session_key, {})
        existing = session_candidates.get(repository)
        if existing is not None and int(existing["priority"]) <= priority:
            continue
        session_candidates[repository] = {
            "repository": repository,
            "source": source,
            "priority": priority,
            "source_row_id": row.get("id"),
        }

    unique_by_session: Dict[str, Dict[str, Any]] = {}
    for session_id, candidates in candidates_by_session.items():
        if len(candidates) != 1:
            continue
        unique_by_session[session_id] = next(iter(candidates.values()))
    return unique_by_session


def _build_repaired_row(  # noqa: PLR0915
    row: Dict[str, Any],
    known_repositories: set[str],
    session_repositories: Optional[Dict[str, Dict[str, Any]]] = None,
    grok_repository: Optional[str] = None,
    rollout_repository_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    session_repositories = session_repositories or {}
    rollout_repository_map = rollout_repository_map or {}
    metadata = _safe_json_metadata(row.get("metadata"))
    original_metadata = _safe_json_metadata(row.get("metadata"))
    original_repository = row.get("repository")
    original_tenant_id = row.get("tenant_id")

    repository = _repair_noisy_identity(original_repository, known_repositories)
    metadata_repository = _repair_noisy_identity(
        metadata.get("repository"), known_repositories
    )
    referenced_artifact_owner = _repair_noisy_identity(
        metadata.get("aawm_d1_452_referenced_artifact_owner"),
        known_repositories,
    )
    source_repository = _repair_noisy_identity(
        metadata.get("source_repository"), known_repositories
    )
    tenant_id = _repair_noisy_identity(original_tenant_id, known_repositories)
    repaired_tenant_id = tenant_id
    repair_source = "row_identity_normalization"

    if grok_repository and _is_grok_row(row):
        repository = grok_repository
        repair_source = "grok_repository_override"
    elif repository is None and (
        rollout_repository := _rollout_repository_candidate(
            original_repository,
            rollout_repository_map,
        )
    ) is not None:
        repository = rollout_repository
        repair_source = "rollout_memory_registry"
    elif repository is None and (
        rollout_repository := _rollout_repository_candidate(
            metadata.get("repository"),
            rollout_repository_map,
        )
    ) is not None:
        repository = rollout_repository
        repair_source = "rollout_memory_registry"
    elif repository is None and (
        rollout_repository := _rollout_repository_candidate(
            metadata.get("source_repository"),
            rollout_repository_map,
        )
    ) is not None:
        repository = rollout_repository
        repair_source = "rollout_memory_registry"
    elif repository is None:
        if referenced_artifact_owner is not None:
            repository = referenced_artifact_owner
            repair_source = "row_metadata.aawm_d1_452_referenced_artifact_owner"
        elif metadata_repository is not None:
            repository = metadata_repository
            repair_source = "row_metadata.repository"
        elif source_repository is not None:
            repository = source_repository
            repair_source = "row_metadata.source_repository"
        elif (
            session_identity := session_repositories.get(str(row.get("session_id")))
        ) is not None:
            repository = str(session_identity["repository"])
            repair_source = f"same_session.{session_identity['source']}"
        elif _is_known_repository(tenant_id, known_repositories):
            repository = tenant_id
            repair_source = "row_tenant_id"
        elif _is_known_repository(repaired_tenant_id, known_repositories):
            repository = repaired_tenant_id
            repair_source = "row_tenant_id_noisy"

    if tenant_id is None:
        tenant_id = repaired_tenant_id or repository

    if repository is not None:
        metadata["repository"] = repository
        if repository.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
            metadata["source_repository"] = _repo_base(repository)
    else:
        metadata.pop("repository", None)
        if _is_bad_repository_fragment(
            metadata.get("source_repository"),
            known_repositories,
        ):
            metadata.pop("source_repository", None)

    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    else:
        metadata.pop("tenant_id", None)
        metadata.pop("tenant_id_source", None)
    trace_user_id = metadata.get("trace_user_id")
    if trace_user_id in (
        original_repository,
        original_tenant_id,
        original_metadata.get("repository"),
    ):
        if repository is not None:
            metadata["trace_user_id"] = repository
        else:
            metadata.pop("trace_user_id", None)

    changed = (
        repository != original_repository
        or tenant_id != original_tenant_id
        or metadata != original_metadata
    )
    if not changed:
        return None

    if repository is None and tenant_id is None:
        metadata["session_history_repository_status"] = "unresolved"
        metadata["session_history_repository_unresolved"] = True
        metadata["session_history_repository_unresolved_reason"] = (
            _unresolved_repository_reason(row, metadata, known_repositories)
        )
        metadata["repository_identity_classified_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        metadata["repository_identity_classification_source"] = (
            "repair_session_history_repository_identity.identity_cleanup"
        )
        repair_source = "unresolved_identity_cleanup"
    else:
        metadata["session_history_repository_status"] = "repaired"
        metadata["session_history_repository_status_source"] = repair_source
        for key in _REPOSITORY_UNRESOLVED_KEYS:
            metadata.pop(key, None)
    metadata["repository_identity_repaired_at"] = datetime.now(timezone.utc).isoformat()
    metadata["repository_identity_repair_source"] = repair_source
    if repository != original_repository:
        metadata["repository_identity_previous_repository"] = original_repository
    if tenant_id != original_tenant_id:
        metadata["repository_identity_previous_tenant_id"] = original_tenant_id
    if tenant_id == repository and tenant_id is not None:
        metadata["tenant_id_source"] = "repository_repair"

    return {
        "id": row["id"],
        "repository": repository,
        "tenant_id": tenant_id,
        "metadata": metadata,
        "previous_repository": original_repository,
        "previous_tenant_id": original_tenant_id,
        "repair_source": repair_source,
    }


def _metadata_flag_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _unresolved_repository_reason(
    row: Dict[str, Any],
    metadata: Dict[str, Any],
    known_repositories: set[str],
) -> str:
    original_candidate = metadata.get("aawm_d1_452_original_repository")
    if original_candidate is not None:
        if _is_bad_repository_fragment(original_candidate, known_repositories):
            return "untrusted_file_like_repository_candidate"
        return "untrusted_repository_candidate"

    provider = str(row.get("provider") or "").lower()
    client_name = str(metadata.get("client_name") or "").lower()
    if provider == "anthropic" or "claude" in client_name:
        return "no_trusted_claude_project_signal"
    if _is_grok_row(row):
        return "no_trusted_grok_project_signal"
    return "no_trusted_repository_signal"


def _build_unresolved_classification_row(
    row: Dict[str, Any],
    known_repositories: set[str],
) -> Optional[Dict[str, Any]]:
    metadata = _safe_json_metadata(row.get("metadata"))
    if _metadata_flag_is_true(metadata.get("session_history_reporting_excluded")):
        return None

    original_metadata = dict(metadata)
    original_repository = row.get("repository")
    original_tenant_id = row.get("tenant_id")
    reason = _unresolved_repository_reason(row, metadata, known_repositories)
    if (
        metadata.get("session_history_repository_status") == "unresolved"
        and metadata.get("session_history_repository_unresolved") is True
        and metadata.get("session_history_repository_unresolved_reason") == reason
        and metadata.get("repository_identity_classification_source")
    ):
        return None

    metadata["session_history_repository_status"] = "unresolved"
    metadata["session_history_repository_unresolved"] = True
    metadata["session_history_repository_unresolved_reason"] = reason
    metadata["repository_identity_classified_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    metadata["repository_identity_classification_source"] = (
        "repair_session_history_repository_identity.null_repository_since"
    )
    if original_repository is not None:
        metadata["repository_identity_previous_repository"] = original_repository
    if original_tenant_id is not None:
        metadata["repository_identity_previous_tenant_id"] = original_tenant_id

    if metadata == original_metadata:
        return None

    return {
        "id": row["id"],
        "repository": original_repository,
        "tenant_id": original_tenant_id,
        "metadata": metadata,
        "previous_repository": original_repository,
        "previous_tenant_id": original_tenant_id,
        "classification_reason": reason,
        "classification_source": "unresolved_repository",
    }


def _fetch_repository_value_params(
    cursor_id: int,
    repository_values: list[str],
    batch_size: int,
    max_id: Optional[int],
) -> Tuple[Any, ...]:
    params: list[Any] = [cursor_id]
    if max_id is not None:
        params.append(max_id)
    params.extend([repository_values, repository_values, batch_size])
    return tuple(params)


def _fetch_null_repository_since_params(
    cursor_id: int,
    null_repository_since: str,
    batch_size: int,
    max_id: Optional[int],
) -> Tuple[Any, ...]:
    params: list[Any] = [cursor_id]
    if max_id is not None:
        params.append(max_id)
    params.extend([null_repository_since, batch_size])
    return tuple(params)


def _fetch_default_candidate_params(
    *,
    cursor_id: int,
    batch_size: int,
    include_all_grok: bool,
    null_repository_since: Optional[str],
    max_id: Optional[int],
) -> Tuple[Any, ...]:
    params: list[Any] = [cursor_id]
    if max_id is not None:
        params.append(max_id)
    params.extend(
        [
            include_all_grok,
            "%grok%",
            _VALID_IDENTITY_SQL,
            list(_DISALLOWED_IDENTITY_VALUES),
            list(_DISALLOWED_IDENTITY_VALUES),
            _AGENT_IDENTITY_SQL,
            _WAVE_AGENT_IDENTITY_SQL,
            _TRANSCRIPT_ARTIFACT_SQL,
            _VALID_IDENTITY_SQL,
            list(_DISALLOWED_IDENTITY_VALUES),
            list(_DISALLOWED_IDENTITY_VALUES),
            _AGENT_IDENTITY_SQL,
            _WAVE_AGENT_IDENTITY_SQL,
            _TRANSCRIPT_ARTIFACT_SQL,
            list(_METADATA_IDENTITY_KEYS),
            _VALID_IDENTITY_SQL,
            list(_DISALLOWED_IDENTITY_VALUES),
            list(_DISALLOWED_IDENTITY_VALUES),
            _AGENT_IDENTITY_SQL,
            _WAVE_AGENT_IDENTITY_SQL,
            _TRANSCRIPT_ARTIFACT_SQL,
            "%grok%",
            null_repository_since,
            null_repository_since,
            batch_size,
        ]
    )
    return tuple(params)


def _fetch_candidate_rows(
    conn: psycopg.Connection,
    *,
    cursor_id: int,
    batch_size: int,
    include_all_grok: bool = False,
    null_repository_since: Optional[str] = None,
    repository_values: Optional[list[str]] = None,
    max_id: Optional[int] = None,
) -> list[Dict[str, Any]]:
    max_id_clause = "AND id <= %s" if max_id is not None else ""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        if repository_values:
            cur.execute(
                f"""
                SELECT id, session_id, provider, model, repository, tenant_id, metadata, created_at
                FROM public.session_history
                WHERE id > %s
                  {max_id_clause}
                  AND (
                        repository = ANY(%s::text[])
                     OR tenant_id = ANY(%s::text[])
                  )
                ORDER BY id ASC
                LIMIT %s
                """,
                _fetch_repository_value_params(
                    cursor_id, repository_values, batch_size, max_id
                ),
            )
            return [dict(row) for row in cur.fetchall()]

        if null_repository_since and not include_all_grok:
            cur.execute(
                f"""
                SELECT id, session_id, provider, model, repository, tenant_id, metadata, created_at
                FROM public.session_history
                WHERE id > %s
                  {max_id_clause}
                  AND repository IS NULL
                  AND created_at >= %s::timestamptz
                ORDER BY id ASC
                LIMIT %s
                """,
                _fetch_null_repository_since_params(
                    cursor_id, null_repository_since, batch_size, max_id
                ),
            )
            return [dict(row) for row in cur.fetchall()]

        cur.execute(
            f"""
            SELECT id, session_id, provider, model, repository, tenant_id, metadata, created_at
            FROM public.session_history
            WHERE id > %s
              {max_id_clause}
              AND (
                    (
                        %s
                    AND (provider = 'xai' OR model ILIKE %s)
                    )
                 OR (
                        repository IS NOT NULL
                    AND (
                            repository !~ %s
                         OR lower(repository) = ANY(%s::text[])
                         OR lower(regexp_replace(repository, ' \\(memory\\)$', '')) = ANY(%s::text[])
                         OR repository ~* %s
                         OR repository ~* %s
                         OR repository ~* %s
                        )
                    )
                 OR (
                        tenant_id IS NOT NULL
                    AND (
                            tenant_id !~ %s
                         OR lower(tenant_id) = ANY(%s::text[])
                         OR lower(regexp_replace(tenant_id, ' \\(memory\\)$', '')) = ANY(%s::text[])
                         OR tenant_id ~* %s
                         OR tenant_id ~* %s
                         OR tenant_id ~* %s
                        )
                    )
                 OR (
                        jsonb_typeof(metadata) = 'object'
                    AND EXISTS (
                            SELECT 1
                            FROM jsonb_each_text(metadata) AS metadata_entry(key, value)
                            WHERE metadata_entry.key = ANY(%s::text[])
                              AND metadata_entry.value IS NOT NULL
                              AND (
                                    metadata_entry.value !~ %s
                                 OR lower(metadata_entry.value) = ANY(%s::text[])
                                 OR lower(regexp_replace(metadata_entry.value, ' \\(memory\\)$', '')) = ANY(%s::text[])
                                 OR metadata_entry.value ~* %s
                                 OR metadata_entry.value ~* %s
                                 OR metadata_entry.value ~* %s
                              )
                        )
                    )
                 OR (repository IS NULL AND metadata ? 'aawm_claude_project')
                 OR (repository IS NULL AND metadata ? 'aawm_d1_452_referenced_artifact_owner')
                 OR (repository IS NULL AND (provider = 'xai' OR model ILIKE %s))
                 OR (
                        %s::timestamptz IS NOT NULL
                    AND repository IS NULL
                    AND created_at >= %s::timestamptz
                    )
              )
            ORDER BY id ASC
            LIMIT %s
            """,
            _fetch_default_candidate_params(
                cursor_id=cursor_id,
                batch_size=batch_size,
                include_all_grok=include_all_grok,
                null_repository_since=null_repository_since,
                max_id=max_id,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def _fetch_session_identity_rows(
    conn: psycopg.Connection,
    session_ids: set[str],
) -> list[Dict[str, Any]]:
    if not session_ids:
        return []

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT id, session_id, repository, tenant_id, metadata
            FROM public.session_history
            WHERE session_id = ANY(%s::text[])
              AND (
                    repository IS NOT NULL
                 OR tenant_id IS NOT NULL
                 OR (
                        jsonb_typeof(metadata) = 'object'
                    AND metadata ?| %s::text[]
                    )
              )
            """,
            (
                list(session_ids),
                list(_METADATA_IDENTITY_KEYS),
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def _apply_repairs(
    conn: psycopg.Connection,
    repairs: list[Dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE public.session_history
            SET repository = %s,
                tenant_id = %s,
                metadata = %s::jsonb
            WHERE id = %s
            """,
            [
                (
                    repair["repository"],
                    repair["tenant_id"],
                    json.dumps(repair["metadata"], sort_keys=True),
                    repair["id"],
                )
                for repair in repairs
            ],
        )


def repair_repository_identities(args: argparse.Namespace) -> int:  # noqa: PLR0915
    dsn = args.dsn or _build_aawm_admin_dsn()
    if not dsn:
        raise SystemExit("No database DSN found. Set AAWM_DB_* or pass --dsn.")

    known_repositories = _load_known_repositories(Path(args.projects_dir))
    rollout_repository_map = _load_rollout_repository_map(
        Path(args.memories_dir),
        known_repositories,
    )
    grok_repository = None
    if args.grok_repository:
        grok_repository = _normalize_repository_identity(args.grok_repository)
        if not _is_known_repository(grok_repository, known_repositories):
            raise SystemExit(
                f"--grok-repository must resolve to a known repository: {args.grok_repository!r}"
            )

    total_seen = 0
    total_repaired = 0
    total_classified_unresolved = 0
    repair_sources: Counter[str] = Counter()
    classification_reasons: Counter[str] = Counter()
    repair_groups: Counter[str] = Counter()
    unresolved_groups: Counter[str] = Counter()
    preview: list[Dict[str, Any]] = []
    classification_preview: list[Dict[str, Any]] = []
    cursor_id = args.cursor_id

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            database_name = cur.fetchone()[0]

        while True:
            rows = _fetch_candidate_rows(
                conn,
                cursor_id=cursor_id,
                batch_size=args.batch_size,
                include_all_grok=grok_repository is not None,
                null_repository_since=args.null_repository_since,
                repository_values=args.repository_value,
                max_id=args.max_id,
            )
            if not rows:
                break
            total_seen += len(rows)
            cursor_id = max(int(row["id"]) for row in rows)
            session_ids = {
                str(row["session_id"])
                for row in rows
                if row.get("session_id")
            }
            session_identity_rows = (
                []
                if args.skip_session_evidence
                else _fetch_session_identity_rows(conn, session_ids)
            )
            if args.null_repository_since:
                session_repositories = _build_unique_session_repository_map(
                    session_identity_rows,
                    known_repositories,
                )
            else:
                session_repositories = _build_session_repository_map(
                    session_identity_rows,
                    known_repositories,
                )
            repairs = [
                repair
                for row in rows
                if (
                    repair := _build_repaired_row(
                        row,
                        known_repositories,
                        session_repositories,
                        grok_repository,
                        rollout_repository_map,
                    )
                )
                is not None
            ]
            repaired_ids = {int(repair["id"]) for repair in repairs}
            classifications: list[Dict[str, Any]] = []
            if args.null_repository_since:
                for row in rows:
                    provider = str(row.get("provider") or "unknown")
                    model = str(row.get("model") or "unknown")
                    group_key = f"{provider}|{model}"
                    if int(row["id"]) in repaired_ids:
                        repair_groups[group_key] += 1
                    elif not _metadata_flag_is_true(
                        _safe_json_metadata(row.get("metadata")).get(
                            "session_history_reporting_excluded"
                        )
                    ):
                        unresolved_groups[group_key] += 1
                        if args.classify_unresolved:
                            classification = _build_unresolved_classification_row(
                                row,
                                known_repositories,
                            )
                            if classification is not None:
                                classifications.append(classification)
                    else:
                        unresolved_groups[f"{group_key}|reporting_excluded"] += 1
            updates = repairs + classifications
            if repairs:
                total_repaired += len(repairs)
                repair_sources.update(str(repair["repair_source"]) for repair in repairs)
                preview.extend(repairs[: max(0, args.preview_limit - len(preview))])
            if classifications:
                total_classified_unresolved += len(classifications)
                classification_reasons.update(
                    str(classification["classification_reason"])
                    for classification in classifications
                )
                classification_preview.extend(
                    classifications[
                        : max(0, args.preview_limit - len(classification_preview))
                    ]
                )
            if updates and args.apply:
                _apply_repairs(conn, updates)
                conn.commit()
            elif args.apply:
                conn.commit()

        if not args.apply:
            conn.rollback()

    print(f"database={database_name}")  # noqa: T201
    print(f"candidate_rows={total_seen}")  # noqa: T201
    print(f"repairable_rows={total_repaired}")  # noqa: T201
    print(f"classified_unresolved_rows={total_classified_unresolved}")  # noqa: T201
    print(f"applied={str(bool(args.apply)).lower()}")  # noqa: T201
    for source, count in sorted(repair_sources.items()):
        print(f"repair_source {source}={count}")  # noqa: T201
    for reason, count in sorted(classification_reasons.items()):
        print(f"classification_reason {reason}={count}")  # noqa: T201
    for group, count in repair_groups.most_common(20):
        print(f"repair_group {group}={count}")  # noqa: T201
    for group, count in unresolved_groups.most_common(20):
        print(f"unresolved_group {group}={count}")  # noqa: T201
    for repair in preview:
        print(  # noqa: T201
            "preview "
            f"id={repair['id']} "
            f"repository={repair['previous_repository']!r}->{repair['repository']!r} "
            f"tenant_id={repair['previous_tenant_id']!r}->{repair['tenant_id']!r}"
        )
    for classification in classification_preview:
        print(  # noqa: T201
            "preview_classification "
            f"id={classification['id']} "
            f"reason={classification['classification_reason']!r}"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair malformed public.session_history repository identities."
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN. Defaults to AAWM_DB_* env.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--cursor-id", type=int, default=0)
    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help=(
            "Optional upper bound on session_history.id. When set, candidate fetches "
            "only include rows with id <= this value."
        ),
    )
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--projects-dir", default="/home/zepfu/projects")
    parser.add_argument("--memories-dir", default="/home/zepfu/.codex/memories")
    parser.add_argument(
        "--repository-value",
        action="append",
        help=(
            "Limit repair candidates to an exact repository/tenant/metadata value. "
            "May be passed multiple times for focused cleanup runs."
        ),
    )
    parser.add_argument(
        "--null-repository-since",
        help=(
            "Include rows with repository IS NULL and created_at at or after this "
            "timestamptz. Uses strict single-repository session evidence."
        ),
    )
    parser.add_argument(
        "--grok-repository",
        help=(
            "Explicit repository to stamp onto Grok/xAI rows. "
            "Use only when the Grok dataset is known to belong to one repository."
        ),
    )
    parser.add_argument(
        "--classify-unresolved",
        action="store_true",
        help=(
            "For --null-repository-since runs: when a row cannot be repaired to a known repository, "
            "stamp durable classification metadata instead of leaving it blank. "
            "Requires --apply to persist. Does not set repository or tenant for unresolved rows."
        ),
    )
    parser.add_argument(
        "--skip-session-evidence",
        action="store_true",
        help=(
            "Skip same-session identity expansion. Use for exact known repository "
            "repair passes where each candidate row already contains enough evidence."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(repair_repository_identities(_parse_args()))
