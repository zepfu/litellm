#!/usr/bin/env python3
"""
Backfill public.session_history from local CLI transcript history.

The importer is dry-run first. It streams historical records from local Claude,
Codex, Gemini, and Grok CLI state, derives aggregate session_history-compatible
rows, and skips any transcript day that already has rows in public.session_history.

Raw transcript text, auth files, and unbounded tool output are intentionally not
stored. Tool arguments are redacted/truncated before they are included in
session_history_tool_activity metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import date, datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

try:
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - exercised only on thin hosts.
    psycopg = None  # type: ignore[assignment]


IMPORT_MARKER = "local_cli_history_2026_05_19"
PARSER_VERSION = "local-cli-history-v1"
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_TARGET_DB = "aawm_tristore"
CLIENT_NAME_BY_SOURCE = {
    "claude": "claude-code",
    "codex": "codex_tui",
    "gemini": "gemini_cli",
    "grok": "grok_build",
}

SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|cookie|credential|oauth|password|"
    r"refresh[_-]?token|secret|session[_-]?token|token)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(api[_-]?key\s*[=:]\s*)[A-Za-z0-9._~+/=-]+|"
    r"(token\s*[=:]\s*)[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)
GIT_COMMAND_RE = re.compile(r"(?<!\S)git\s+(?P<verb>commit|push)\b", re.IGNORECASE)
SENSITIVE_CONFIG_CHANGE_FIELDS = (
    "changed_pre_commit_config",
    "changed_env_file",
    "changed_pyproject_toml",
    "changed_gitignore",
)

READ_TOOL_NAMES = {
    "cat",
    "fetch",
    "glob",
    "grep",
    "list_files",
    "listdir",
    "ls",
    "notebookread",
    "read",
    "read_file",
    "read_many_files",
    "search",
    "view",
    "web_fetch",
    "webfetch",
    "websearch",
}
MODIFY_TOOL_NAMES = {
    "apply_patch",
    "applypatch",
    "edit",
    "multiedit",
    "notebookedit",
    "notebookwrite",
    "replace",
    "replacement",
    "write",
    "write_file",
}
COMMAND_TOOL_NAMES = {
    "bash",
    "exec",
    "exec_command",
    "run",
    "run_terminal_command",
    "shell",
    "terminal",
}
READ_PATH_KEYS = {
    "file",
    "file_path",
    "filename",
    "glob",
    "path",
    "paths",
    "pattern",
    "query",
    "target_file",
}
WRITE_PATH_KEYS = {
    "file",
    "file_path",
    "filename",
    "new_path",
    "notebook_path",
    "path",
    "target_file",
}
SKIP_PATH_KEYS = {
    "cmd",
    "command",
    "content",
    "description",
    "new_str",
    "old_str",
    "output",
    "patch",
    "reasoning",
    "replacement",
    "result",
    "text",
    "thinking",
}


@dataclass
class ScanStats:
    files_seen: Counter[str] = field(default_factory=Counter)
    files_skipped: Counter[str] = field(default_factory=Counter)
    parse_errors: Counter[str] = field(default_factory=Counter)
    records_seen: Counter[str] = field(default_factory=Counter)
    records_skipped_existing_day: Counter[str] = field(default_factory=Counter)
    records_after_day_filter: Counter[str] = field(default_factory=Counter)


@dataclass
class DryRunSummary:
    records: int = 0
    tool_rows: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    response_cost_usd: float = 0.0
    response_cost_rows: int = 0
    provider_cache_miss_rows: int = 0
    provider_cache_miss_token_count: int = 0
    provider_cache_miss_cost_usd: float = 0.0
    file_read_count: int = 0
    file_modified_count: int = 0
    git_commit_count: int = 0
    git_push_count: int = 0
    no_token_records: int = 0
    by_client: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    by_date: dict[date, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    by_model: Counter[str] = field(default_factory=Counter)
    by_tool: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any], day: date) -> None:
        client = str(record.get("client_name") or "unknown")
        model = str(record.get("model") or "unknown")
        tool_activity = record.get("tool_activity") or []
        tool_rows = len(tool_activity) if isinstance(tool_activity, list) else 0
        input_tokens = _safe_int(record.get("input_tokens")) or 0
        output_tokens = _safe_int(record.get("output_tokens")) or 0
        cache_read = _safe_int(record.get("cache_read_input_tokens")) or 0
        cache_creation = _safe_int(record.get("cache_creation_input_tokens")) or 0
        reasoning = (
            _safe_int(record.get("reasoning_tokens_reported"))
            or _safe_int(record.get("reasoning_tokens_estimated"))
            or 0
        )
        response_cost = _safe_float(record.get("response_cost_usd"))
        cache_miss = bool(record.get("provider_cache_miss"))
        cache_miss_tokens = _safe_int(record.get("provider_cache_miss_token_count")) or 0
        cache_miss_cost = _safe_float(record.get("provider_cache_miss_cost_usd"))
        file_read_count = _safe_int(record.get("file_read_count")) or 0
        file_modified_count = _safe_int(record.get("file_modified_count")) or 0
        git_commit_count = _safe_int(record.get("git_commit_count")) or 0
        git_push_count = _safe_int(record.get("git_push_count")) or 0

        self.records += 1
        self.tool_rows += tool_rows
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_input_tokens += cache_read
        self.cache_creation_input_tokens += cache_creation
        self.reasoning_tokens += reasoning
        if response_cost is not None:
            self.response_cost_rows += 1
            self.response_cost_usd += response_cost
        if cache_miss:
            self.provider_cache_miss_rows += 1
            self.provider_cache_miss_token_count += cache_miss_tokens
            if cache_miss_cost is not None:
                self.provider_cache_miss_cost_usd += cache_miss_cost
        self.file_read_count += file_read_count
        self.file_modified_count += file_modified_count
        self.git_commit_count += git_commit_count
        self.git_push_count += git_push_count
        if input_tokens + output_tokens + cache_read + cache_creation + reasoning == 0:
            self.no_token_records += 1

        self.by_client[client]["records"] += 1
        self.by_client[client]["tool_rows"] += tool_rows
        self.by_client[client]["input_tokens"] += input_tokens
        self.by_client[client]["output_tokens"] += output_tokens
        self.by_client[client]["cache_read_input_tokens"] += cache_read
        self.by_client[client]["cache_creation_input_tokens"] += cache_creation
        self.by_client[client]["reasoning_tokens"] += reasoning
        if response_cost is not None:
            self.by_client[client]["response_cost_rows"] += 1
            self.by_client[client]["response_cost_usd"] += response_cost
        if cache_miss:
            self.by_client[client]["provider_cache_miss_rows"] += 1
            self.by_client[client]["provider_cache_miss_token_count"] += cache_miss_tokens
            if cache_miss_cost is not None:
                self.by_client[client]["provider_cache_miss_cost_usd"] += cache_miss_cost
        self.by_client[client]["file_read_count"] += file_read_count
        self.by_client[client]["file_modified_count"] += file_modified_count
        self.by_client[client]["git_commit_count"] += git_commit_count
        self.by_client[client]["git_push_count"] += git_push_count
        if input_tokens + output_tokens + cache_read + cache_creation + reasoning == 0:
            self.by_client[client]["no_token_records"] += 1

        self.by_date[day]["records"] += 1
        self.by_date[day]["tool_rows"] += tool_rows
        self.by_date[day]["input_tokens"] += input_tokens
        self.by_date[day]["output_tokens"] += output_tokens
        self.by_date[day]["cache_read_input_tokens"] += cache_read
        self.by_date[day]["cache_creation_input_tokens"] += cache_creation
        self.by_date[day]["reasoning_tokens"] += reasoning
        if response_cost is not None:
            self.by_date[day]["response_cost_rows"] += 1
            self.by_date[day]["response_cost_usd"] += response_cost
        if cache_miss:
            self.by_date[day]["provider_cache_miss_rows"] += 1
            self.by_date[day]["provider_cache_miss_token_count"] += cache_miss_tokens
            if cache_miss_cost is not None:
                self.by_date[day]["provider_cache_miss_cost_usd"] += cache_miss_cost
        self.by_date[day]["file_read_count"] += file_read_count
        self.by_date[day]["file_modified_count"] += file_modified_count
        self.by_date[day]["git_commit_count"] += git_commit_count
        self.by_date[day]["git_push_count"] += git_push_count

        self.by_model[f"{client}:{model}"] += 1
        for tool in tool_activity:
            if isinstance(tool, dict):
                name = str(tool.get("tool_name") or "unknown")
                self.by_tool[f"{client}:{name}"] += 1

        if len(self.samples) < 12:
            self.samples.append(_sample_record(record, day))


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        dt = datetime.fromtimestamp(number, tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return _parse_datetime(int(text))
        text = text.replace("Z", "+00:00")
        text = re.sub(
            r"\.(\d{6})\d+([+-]\d\d:\d\d)$",
            lambda match: f".{match.group(1)}{match.group(2)}",
            text,
        )
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_bucket(value: datetime, tz: ZoneInfo) -> date:
    return value.astimezone(tz).date()


def _sha(value: str, length: int = 12) -> str:
    return sha1(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def _iter_jsonl(path: Path, stats: ScanStats, client: str) -> Iterator[tuple[int, dict[str, Any]]]:
    stats.files_seen[client] += 1
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw_line in enumerate(handle):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    stats.parse_errors[f"{client}:jsonl"] += 1
                    continue
                if isinstance(obj, dict):
                    yield index, obj
    except OSError:
        stats.files_skipped[f"{client}:open_error"] += 1


def _redact_text(value: str, max_len: int = 500) -> str:
    redacted = SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}<redacted>", value)
    if len(redacted) > max_len:
        return redacted[:max_len] + "...<truncated>"
    return redacted


def _redact_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if SENSITIVE_KEY_RE.search(key):
        return "<redacted>"
    if depth > 5:
        return "<truncated>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_key_text = str(child_key)
            if child_key_text.lower() in {"result", "response", "output"}:
                result[child_key_text] = "<redacted:tool-output>"
                continue
            result[child_key_text] = _redact_value(
                child_value,
                key=child_key_text,
                depth=depth + 1,
            )
        return result
    if isinstance(value, list):
        limited = value[:20]
        redacted_list = [_redact_value(item, depth=depth + 1) for item in limited]
        if len(value) > len(limited):
            redacted_list.append("<truncated>")
        return redacted_list
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _extract_paths_from_value(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            paths.append(_redact_text(text, max_len=300))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_extract_paths_from_value(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in SKIP_PATH_KEYS:
                continue
            paths.extend(_extract_paths_from_value(item))
    return _dedupe(paths)


def _extract_paths(arguments: Any, keys: set[str]) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    paths: list[str] = []
    for key, value in arguments.items():
        key_text = str(key).lower()
        if key_text in keys or key_text.endswith("_path") or key_text.endswith("_paths"):
            paths.extend(_extract_paths_from_value(value))
    return _dedupe(paths)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _changed_file_basename(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().strip("'\"").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        return None
    return normalized.rstrip("/").rsplit("/", 1)[-1]


def _sensitive_config_change_flags_from_paths(paths: Iterable[str]) -> dict[str, bool]:
    flags = {field: False for field in SENSITIVE_CONFIG_CHANGE_FIELDS}
    for path in _dedupe(str(path) for path in paths):
        basename = _changed_file_basename(path)
        if not basename:
            continue
        basename_lower = basename.lower()
        if basename_lower in {".pre-commit-config.yaml", ".pre-commit-config.yml"}:
            flags["changed_pre_commit_config"] = True
        if basename_lower.startswith(".env"):
            flags["changed_env_file"] = True
        if basename_lower == "pyproject.toml":
            flags["changed_pyproject_toml"] = True
        if basename_lower == ".gitignore":
            flags["changed_gitignore"] = True
    return flags


def _tool_kind(tool_name: str) -> str:
    normalized = tool_name.lower().replace("-", "_")
    if normalized in READ_TOOL_NAMES:
        return "read"
    if normalized in MODIFY_TOOL_NAMES:
        return "modify"
    if normalized in COMMAND_TOOL_NAMES:
        return "command"
    return "tool"


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return {"input": stripped}
        if isinstance(decoded, dict):
            return decoded
        return {"input": decoded}
    return {}


def _command_text(tool_name: str, arguments: dict[str, Any]) -> Optional[str]:
    normalized = tool_name.lower().replace("-", "_")
    if normalized not in COMMAND_TOOL_NAMES:
        return None
    for key in ("cmd", "command", "input", "script", "shell", "bash", "code"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _redact_text(value.strip(), max_len=1000)
    return None


def _build_tool_activity(
    *,
    tool_index: int,
    tool_name: str,
    tool_call_id: Optional[str],
    arguments: Any,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    parsed_args = _parse_tool_arguments(arguments)
    kind = _tool_kind(tool_name)
    read_paths = _extract_paths(parsed_args, READ_PATH_KEYS) if kind in {"read", "tool"} else []
    modified_paths = (
        _extract_paths(parsed_args, WRITE_PATH_KEYS) if kind in {"modify", "tool"} else []
    )
    command = _command_text(tool_name, parsed_args)
    commit_count = 0
    push_count = 0
    if command:
        for match in GIT_COMMAND_RE.finditer(command):
            if match.group("verb").lower() == "commit":
                commit_count += 1
            elif match.group("verb").lower() == "push":
                push_count += 1
    return {
        "tool_index": tool_index,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name or "unknown",
        "tool_kind": kind,
        "file_paths_read": read_paths,
        "file_paths_modified": modified_paths,
        "git_commit_count": commit_count,
        "git_push_count": push_count,
        "command_text": command,
        "arguments": _redact_value(parsed_args),
        "metadata": metadata or {},
    }


def _tool_summary(tool_activity: list[dict[str, Any]]) -> dict[str, Any]:
    read_paths: list[str] = []
    modified_paths: list[str] = []
    git_commit_count = 0
    git_push_count = 0
    names: list[str] = []
    for item in tool_activity:
        names.append(str(item.get("tool_name") or "unknown"))
        read_paths.extend(str(path) for path in item.get("file_paths_read") or [])
        modified_paths.extend(str(path) for path in item.get("file_paths_modified") or [])
        git_commit_count += _safe_int(item.get("git_commit_count")) or 0
        git_push_count += _safe_int(item.get("git_push_count")) or 0
    return {
        "tool_names": _dedupe(names),
        "file_read_count": len(_dedupe(read_paths)),
        "file_modified_count": len(_dedupe(modified_paths)),
        **_sensitive_config_change_flags_from_paths(modified_paths),
        "git_commit_count": git_commit_count,
        "git_push_count": git_push_count,
    }


def _estimate_tokens_from_text(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _provider_from_model(model: Optional[str], fallback: Optional[str] = None) -> str:
    cleaned = (model or "").lower()
    fallback_cleaned = (fallback or "").lower()
    if "claude" in cleaned:
        return "anthropic"
    if "gemini" in cleaned:
        return "gemini"
    if "grok" in cleaned:
        return "xai"
    if cleaned.startswith(("gpt-", "o1", "o3", "o4", "o5")) or "codex" in cleaned:
        return "openai"
    if fallback_cleaned and fallback_cleaned not in {"default", "unknown"}:
        return fallback_cleaned
    return "unknown"


def _storage_client_name(source_client: str) -> str:
    return CLIENT_NAME_BY_SOURCE.get(source_client, source_client)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, Decimal):
            return float(value)
    return None


def _model_cost_info(model: str, provider: str) -> Optional[dict[str, Any]]:
    if not model or model == "unknown" or model == "<synthetic>":
        return None
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
    except Exception:
        return None
    candidates = []
    if provider and provider != "unknown":
        candidates.append(f"{provider}/{model}")
    candidates.append(model)
    for candidate in candidates:
        info = litellm.model_cost.get(candidate)
        if isinstance(info, dict):
            return info
    return None


def _cost_value(model_info: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _safe_float(model_info.get(key))
        if value is not None:
            return value
    return None


def _cache_creation_write_cost(
    *,
    model_info: dict[str, Any],
    cache_creation_input_tokens: int,
    usage_obj: Optional[dict[str, Any]],
) -> float:
    if cache_creation_input_tokens <= 0:
        return 0.0
    base_cost = _cost_value(
        model_info,
        "cache_creation_input_token_cost",
        "input_cost_per_token",
    )
    if base_cost is None:
        return 0.0
    above_1hr_cost = _cost_value(
        model_info,
        "cache_creation_input_token_cost_above_1hr",
    )
    creation_detail = (
        usage_obj.get("cache_creation")
        if isinstance(usage_obj, dict) and isinstance(usage_obj.get("cache_creation"), dict)
        else {}
    )
    one_hour_tokens = _safe_int(creation_detail.get("ephemeral_1h_input_tokens")) or 0
    five_min_tokens = _safe_int(creation_detail.get("ephemeral_5m_input_tokens")) or 0
    if one_hour_tokens or five_min_tokens:
        remaining = max(cache_creation_input_tokens - one_hour_tokens - five_min_tokens, 0)
        one_hour_effective_cost = above_1hr_cost if above_1hr_cost is not None else base_cost
        return (
            (one_hour_tokens * one_hour_effective_cost)
            + (five_min_tokens * base_cost)
            + (remaining * base_cost)
        )
    return cache_creation_input_tokens * base_cost


def _calculate_response_cost_usd(
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
    usage_obj: Optional[dict[str, Any]],
) -> Optional[float]:
    model_info = _model_cost_info(model, provider)
    if not model_info:
        return None
    input_cost = _cost_value(model_info, "input_cost_per_token")
    output_cost = _cost_value(model_info, "output_cost_per_token")
    cache_read_cost = _cost_value(
        model_info,
        "cache_read_input_token_cost",
        "input_cost_per_token",
    )
    if input_cost is None and output_cost is None and cache_read_cost is None:
        return None
    cost = 0.0
    if input_cost is not None:
        cost += input_tokens * input_cost
    if output_cost is not None:
        cost += output_tokens * output_cost
    if cache_read_cost is not None:
        cost += cache_read_input_tokens * cache_read_cost
    cost += _cache_creation_write_cost(
        model_info=model_info,
        cache_creation_input_tokens=cache_creation_input_tokens,
        usage_obj=usage_obj,
    )
    return max(cost, 0.0)


def _provider_cache_state_for_record(
    *,
    model: str,
    provider: str,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
    usage_obj: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if cache_read_input_tokens > 0:
        return {
            "provider_cache_attempted": True,
            "provider_cache_status": "hit",
            "provider_cache_miss": False,
            "provider_cache_miss_reason": None,
            "provider_cache_miss_token_count": None,
            "provider_cache_miss_cost_usd": None,
            "provider_cache_miss_cost_basis": None,
        }
    if cache_creation_input_tokens <= 0:
        return {
            "provider_cache_attempted": False,
            "provider_cache_status": None,
            "provider_cache_miss": False,
            "provider_cache_miss_reason": None,
            "provider_cache_miss_token_count": None,
            "provider_cache_miss_cost_usd": None,
            "provider_cache_miss_cost_basis": None,
        }

    miss_cost = None
    model_info = _model_cost_info(model, provider)
    if model_info:
        write_cost = _cache_creation_write_cost(
            model_info=model_info,
            cache_creation_input_tokens=cache_creation_input_tokens,
            usage_obj=usage_obj,
        )
        cache_read_cost = _cost_value(model_info, "cache_read_input_token_cost") or 0.0
        miss_cost = max(
            write_cost - (cache_creation_input_tokens * cache_read_cost),
            0.0,
        )
    return {
        "provider_cache_attempted": True,
        "provider_cache_status": "write",
        "provider_cache_miss": True,
        "provider_cache_miss_reason": "cache_write_only",
        "provider_cache_miss_token_count": cache_creation_input_tokens,
        "provider_cache_miss_cost_usd": miss_cost,
        "provider_cache_miss_cost_basis": (
            "write_vs_read_delta" if miss_cost is not None else None
        ),
    }


def _base_record(
    *,
    source_client_name: str,
    created_at: datetime,
    litellm_call_id: str,
    session_id: str,
    provider: str,
    model: str,
    repository: Optional[str],
    metadata: dict[str, Any],
    client_version: Optional[str] = None,
    agent_name: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: Optional[int] = None,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_tokens_reported: Optional[int] = None,
    reasoning_tokens_estimated: Optional[int] = None,
    reasoning_tokens_source: Optional[str] = None,
    reasoning_present: bool = False,
    thinking_signature_present: bool = False,
    tool_activity: Optional[list[dict[str, Any]]] = None,
    usage_obj: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    tools = tool_activity or []
    summary = _tool_summary(tools)
    storage_client_name = _storage_client_name(source_client_name)
    resolved_total = (
        total_tokens
        if total_tokens is not None
        else input_tokens + output_tokens
    )
    response_cost_usd = _calculate_response_cost_usd(
        model=model or "unknown",
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        usage_obj=usage_obj,
    )
    cache_state = _provider_cache_state_for_record(
        model=model or "unknown",
        provider=provider,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        usage_obj=usage_obj,
    )
    record = {
        "created_at": created_at,
        "litellm_call_id": litellm_call_id,
        "session_id": session_id,
        "trace_id": session_id,
        "provider_response_id": None,
        "provider": provider,
        "model": model or "unknown",
        "model_group": model or "unknown",
        "agent_name": agent_name,
        "tenant_id": "aawm",
        "call_type": "local_cli_history",
        "start_time": created_at,
        "end_time": created_at,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": resolved_total or 0,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reasoning_tokens_reported,
        "reasoning_tokens_estimated": reasoning_tokens_estimated,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": bool(
            reasoning_present
            or (reasoning_tokens_reported or 0) > 0
            or (reasoning_tokens_estimated or 0) > 0
        ),
        "thinking_signature_present": thinking_signature_present,
        **cache_state,
        "tool_call_count": len(tools),
        "invalid_tool_call_count": 0,
        "tool_names": summary["tool_names"],
        "file_read_count": summary["file_read_count"],
        "file_modified_count": summary["file_modified_count"],
        "changed_pre_commit_config": summary["changed_pre_commit_config"],
        "changed_env_file": summary["changed_env_file"],
        "changed_pyproject_toml": summary["changed_pyproject_toml"],
        "changed_gitignore": summary["changed_gitignore"],
        "git_commit_count": summary["git_commit_count"],
        "git_push_count": summary["git_push_count"],
        "response_cost_usd": response_cost_usd,
        "litellm_environment": "local_history",
        "client_name": storage_client_name,
        "client_version": None,
        "client_user_agent": "backfill",
        "token_permission_input": 0,
        "token_permission_output": 0,
        "permission_usd_cost": 0.0,
        "repository": repository,
        "tool_activity": tools,
        "metadata": {
            "source_import": IMPORT_MARKER,
            "parser_version": PARSER_VERSION,
            "source_client": source_client_name,
            "source_client_version": client_version,
            "usage_provider_cache_status": cache_state["provider_cache_status"],
            "usage_provider_cache_miss": cache_state["provider_cache_miss"],
            "usage_provider_cache_miss_reason": cache_state[
                "provider_cache_miss_reason"
            ],
            "usage_provider_cache_miss_token_count": cache_state[
                "provider_cache_miss_token_count"
            ],
            "usage_provider_cache_miss_cost_usd": cache_state[
                "provider_cache_miss_cost_usd"
            ],
            "usage_provider_cache_miss_cost_basis": cache_state[
                "provider_cache_miss_cost_basis"
            ],
            "usage_response_cost_source": (
                "bundled_model_cost_map" if response_cost_usd is not None else None
            ),
            **metadata,
        },
    }
    return record


def _extract_claude_reasoning(content: Any) -> tuple[bool, Optional[int], bool]:
    if not isinstance(content, list):
        return False, None, False
    reasoning_text_parts: list[str] = []
    signature_present = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        signature_present = signature_present or bool(
            block.get("signature") or block.get("encrypted") or block.get("data")
        )
        text = block.get("thinking") or block.get("text")
        if isinstance(text, str):
            reasoning_text_parts.append(text)
    reasoning_text = "\n".join(reasoning_text_parts)
    estimated = _estimate_tokens_from_text(reasoning_text) if reasoning_text else None
    return bool(reasoning_text_parts or signature_present), estimated, signature_present


def _iter_claude_records(root: Path, stats: ScanStats) -> Iterator[dict[str, Any]]:
    projects_root = root / ".claude" / "projects"
    if not projects_root.exists():
        stats.files_skipped["claude:missing_root"] += 1
        return
    for path in projects_root.rglob("*.jsonl"):
        if path.name == "hook-events.jsonl":
            stats.files_skipped["claude:hook_events"] += 1
            continue
        path_hash = _sha(str(path))
        grouped: dict[str, dict[str, Any]] = {}
        group_order: list[str] = []
        for line_index, obj in _iter_jsonl(path, stats, "claude"):
            if obj.get("type") != "assistant":
                continue
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            created_at = _parse_datetime(obj.get("timestamp"))
            if created_at is None:
                continue
            model = _safe_str(message.get("model")) or "unknown"
            session_id = _safe_str(obj.get("sessionId")) or path.stem
            row_id = (
                _safe_str(message.get("id"))
                or _safe_str(obj.get("requestId"))
                or _safe_str(obj.get("uuid"))
                or str(line_index)
            )
            group_key = f"{session_id}:{row_id}"
            if group_key not in grouped:
                grouped[group_key] = {
                    "created_at": created_at,
                    "line_index": line_index,
                    "source_lines": [],
                    "source_uuids": [],
                    "obj": obj,
                    "message": message,
                    "content": [],
                    "tool_activity": [],
                    "tool_ids": set(),
                }
                group_order.append(group_key)
            group = grouped[group_key]
            group["source_lines"].append(line_index)
            if obj.get("uuid"):
                group["source_uuids"].append(obj.get("uuid"))
            if created_at < group["created_at"]:
                group["created_at"] = created_at
                group["line_index"] = line_index
                group["obj"] = obj
                group["message"] = message
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    group["content"].append(block)
                    if block.get("type") != "tool_use":
                        continue
                    tool_id = _safe_str(block.get("id")) or f"line-{line_index}:tool-{len(group['tool_activity'])}"
                    if tool_id in group["tool_ids"]:
                        continue
                    group["tool_ids"].add(tool_id)
                    group["tool_activity"].append(
                        _build_tool_activity(
                            tool_index=len(group["tool_activity"]),
                            tool_name=_safe_str(block.get("name")) or "unknown",
                            tool_call_id=_safe_str(block.get("id")),
                            arguments=block.get("input") or {},
                            metadata={"source": "claude_message_content"},
                        )
                    )
        for group_key in group_order:
            group = grouped[group_key]
            obj = group["obj"]
            message = group["message"]
            usage = message.get("usage") if isinstance(message, dict) else {}
            if not isinstance(usage, dict):
                continue
            created_at = group["created_at"]
            model = _safe_str(message.get("model")) or "unknown"
            session_id = _safe_str(obj.get("sessionId")) or path.stem
            row_id = (
                _safe_str(message.get("id"))
                or _safe_str(obj.get("requestId"))
                or _safe_str(obj.get("uuid"))
                or str(group["line_index"])
            )
            content = group["content"]
            reasoning_present, reasoning_estimated, signature_present = (
                _extract_claude_reasoning(content)
            )
            input_tokens = _safe_int(usage.get("input_tokens")) or 0
            output_tokens = _safe_int(usage.get("output_tokens")) or 0
            cache_read = _safe_int(usage.get("cache_read_input_tokens")) or 0
            cache_creation = _safe_int(usage.get("cache_creation_input_tokens")) or 0
            metadata = {
                "source_path": str(path),
                "source_path_hash": path_hash,
                "source_line": group["line_index"],
                "source_lines": group["source_lines"],
                "source_request_id": obj.get("requestId"),
                "source_uuid": obj.get("uuid"),
                "source_uuids": group["source_uuids"],
                "source_message_id": message.get("id"),
                "source_usage": _redact_value(usage),
                "is_sidechain": bool(obj.get("isSidechain")),
                "parent_uuid": obj.get("parentUuid"),
                "git_branch": obj.get("gitBranch"),
                "entrypoint": obj.get("entrypoint"),
                "coalesced_claude_rows": len(group["source_lines"]),
            }
            yield _base_record(
                source_client_name="claude",
                client_version=_safe_str(obj.get("version")),
                created_at=created_at,
                litellm_call_id=f"local-cli:claude:{session_id}:{path_hash}:{row_id}",
                session_id=session_id,
                provider="anthropic",
                model=model,
                repository=_safe_str(obj.get("cwd")),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
                reasoning_tokens_estimated=reasoning_estimated,
                reasoning_tokens_source=(
                    "estimated_from_thinking_blocks" if reasoning_estimated else None
                ),
                reasoning_present=reasoning_present,
                thinking_signature_present=signature_present,
                tool_activity=group["tool_activity"],
                usage_obj=usage,
                metadata=metadata,
            )


def _load_codex_threads(codex_root: Path, stats: ScanStats) -> dict[str, dict[str, Any]]:
    db_candidates = sorted(codex_root.glob("state_*.sqlite"))
    if not db_candidates:
        stats.files_skipped["codex:missing_state_db"] += 1
        return {}
    db_path = db_candidates[-1]
    stats.files_seen["codex_state_db"] += 1
    rows: dict[str, dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT * FROM threads"):
                data = dict(row)
                rollout_path = _safe_str(data.get("rollout_path"))
                if rollout_path:
                    rows[rollout_path] = data
        finally:
            conn.close()
    except sqlite3.Error:
        stats.files_skipped["codex:state_db_open_error"] += 1
    return rows


def _codex_thread_time(row: dict[str, Any], key: str) -> Optional[datetime]:
    return _parse_datetime(row.get(f"{key}_at_ms") or row.get(f"{key}_at"))


def _iter_codex_records(root: Path, stats: ScanStats) -> Iterator[dict[str, Any]]:
    codex_root = root / ".codex"
    sessions_root = codex_root / "sessions"
    if not sessions_root.exists():
        stats.files_skipped["codex:missing_sessions"] += 1
        return
    thread_by_rollout = _load_codex_threads(codex_root, stats)
    for path in sessions_root.rglob("*.jsonl"):
        thread = thread_by_rollout.get(str(path), {})
        thread_id = _safe_str(thread.get("id")) or path.stem.rsplit("-", 1)[-1]
        path_hash = _sha(str(path))
        model = _safe_str(thread.get("model")) or _safe_str(thread.get("model_provider")) or "unknown"
        provider = _provider_from_model(model, _safe_str(thread.get("model_provider")))
        repository = _safe_str(thread.get("cwd"))
        client_version = _safe_str(thread.get("cli_version"))
        agent_name = _safe_str(thread.get("agent_role")) or _safe_str(thread.get("agent_nickname"))
        pending_tools: list[dict[str, Any]] = []
        last_usage_signature: Optional[tuple[int, int, int, int, int]] = None
        for line_index, obj in _iter_jsonl(path, stats, "codex"):
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = obj.get("type")
            payload_type = payload.get("type")
            if event_type == "session_meta":
                meta_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
                if isinstance(meta_payload, dict):
                    model = _safe_str(meta_payload.get("model")) or model
                    repository = _safe_str(meta_payload.get("cwd")) or repository
                continue
            if event_type == "response_item" and payload_type in {
                "function_call",
                "custom_tool_call",
            }:
                arguments = payload.get("arguments")
                if arguments is None:
                    arguments = payload.get("input")
                pending_tools.append(
                    _build_tool_activity(
                        tool_index=len(pending_tools),
                        tool_name=_safe_str(payload.get("name")) or "unknown",
                        tool_call_id=_safe_str(payload.get("call_id") or payload.get("id")),
                        arguments=arguments or {},
                        metadata={"source": "codex_response_item"},
                    )
                )
                continue
            if event_type != "event_msg" or payload_type != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            last_usage = info.get("last_token_usage")
            if not isinstance(last_usage, dict):
                continue
            input_tokens = _safe_int(last_usage.get("input_tokens")) or 0
            cache_read = _safe_int(last_usage.get("cached_input_tokens")) or 0
            output_tokens = _safe_int(last_usage.get("output_tokens")) or 0
            reasoning_tokens = _safe_int(last_usage.get("reasoning_output_tokens")) or 0
            total_tokens = (
                _safe_int(last_usage.get("total_tokens"))
                or input_tokens + output_tokens
            )
            signature = (
                input_tokens,
                cache_read,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            )
            if signature == last_usage_signature and not pending_tools:
                continue
            last_usage_signature = signature
            created_at = _parse_datetime(obj.get("timestamp")) or _codex_thread_time(
                thread,
                "created",
            )
            if created_at is None:
                continue
            rate_limits = payload.get("rate_limits")
            metadata = {
                "source_path": str(path),
                "source_path_hash": path_hash,
                "source_line": line_index,
                "thread_source": thread.get("source"),
                "thread_title": _redact_text(str(thread.get("title") or ""), max_len=200),
                "thread_created_at": thread.get("created_at_ms")
                or thread.get("created_at"),
                "thread_updated_at": thread.get("updated_at_ms")
                or thread.get("updated_at"),
                "thread_tokens_used": thread.get("tokens_used"),
                "model_context_window": info.get("model_context_window"),
                "total_token_usage": _redact_value(info.get("total_token_usage") or {}),
                "last_token_usage": _redact_value(last_usage),
                "rate_limits": _redact_value(rate_limits or {}),
                "sandbox_policy": thread.get("sandbox_policy"),
                "approval_mode": thread.get("approval_mode"),
                "git_branch": thread.get("git_branch"),
                "git_sha": thread.get("git_sha"),
                "agent_nickname": thread.get("agent_nickname"),
                "agent_role": thread.get("agent_role"),
            }
            yield _base_record(
                source_client_name="codex",
                client_version=client_version,
                created_at=created_at,
                litellm_call_id=f"local-cli:codex:{thread_id}:{path_hash}:{line_index}",
                session_id=thread_id,
                provider=provider,
                model=model,
                repository=repository,
                agent_name=agent_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read,
                reasoning_tokens_reported=reasoning_tokens or None,
                reasoning_tokens_source=(
                    "codex_token_count.last_token_usage.reasoning_output_tokens"
                    if reasoning_tokens
                    else None
                ),
                reasoning_present=reasoning_tokens > 0,
                tool_activity=pending_tools,
                usage_obj=last_usage,
                metadata=metadata,
            )
            pending_tools = []


def _load_gemini_project_roots(gemini_root: Path) -> dict[str, str]:
    projects_path = gemini_root / "projects.json"
    if not projects_path.exists():
        return {}
    try:
        data = json.loads(projects_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return {}
    return {str(project): str(path) for path, project in projects.items()}


def _read_gemini_messages(path: Path, stats: ScanStats) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats.files_seen["gemini"] += 1
    if path.suffix == ".jsonl":
        header: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        stats.parse_errors["gemini:jsonl"] += 1
                        continue
                    if not isinstance(obj, dict) or "$set" in obj:
                        continue
                    if not header and "type" not in obj and "role" not in obj:
                        header = obj
                    elif "type" in obj or "role" in obj:
                        messages.append(obj)
        except OSError:
            stats.files_skipped["gemini:open_error"] += 1
        return header, messages
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        stats.files_skipped["gemini:json_error"] += 1
        return {}, []
    messages = data.get("messages") if isinstance(data, dict) else []
    return data if isinstance(data, dict) else {}, [
        item for item in messages if isinstance(item, dict)
    ] if isinstance(messages, list) else []


def _gemini_message_is_model(message: dict[str, Any]) -> bool:
    message_type = str(message.get("type") or "").lower()
    role = str(message.get("role") or "").lower()
    return (
        message_type in {"gemini", "model", "assistant"}
        or role in {"model", "assistant"}
        or bool(message.get("tokens") and message.get("model"))
    )


def _iter_gemini_records(root: Path, stats: ScanStats) -> Iterator[dict[str, Any]]:
    gemini_root = root / ".gemini"
    tmp_root = gemini_root / "tmp"
    if not tmp_root.exists():
        stats.files_skipped["gemini:missing_tmp"] += 1
        return
    project_roots = _load_gemini_project_roots(gemini_root)
    seen_messages: set[tuple[str, str, str, str]] = set()
    for path in sorted(list(tmp_root.rglob("chats/*.jsonl")) + list(tmp_root.rglob("chats/*.json"))):
        header, messages = _read_gemini_messages(path, stats)
        try:
            project_name = path.relative_to(tmp_root).parts[0]
        except (IndexError, ValueError):
            project_name = ""
        repository = project_roots.get(project_name) or project_name or None
        session_id = _safe_str(header.get("sessionId")) or path.stem
        path_hash = _sha(str(path))
        for message_index, message in enumerate(messages):
            if not _gemini_message_is_model(message):
                continue
            created_at = _parse_datetime(message.get("timestamp")) or _parse_datetime(
                header.get("lastUpdated") or header.get("startTime")
            )
            if created_at is None:
                continue
            model = _safe_str(message.get("model")) or "unknown"
            message_id = _safe_str(message.get("id")) or str(message_index)
            dedupe_key = (
                session_id,
                message_id,
                str(message.get("timestamp") or ""),
                model,
            )
            if dedupe_key in seen_messages:
                continue
            seen_messages.add(dedupe_key)
            tokens = message.get("tokens") if isinstance(message.get("tokens"), dict) else {}
            input_tokens = _safe_int(tokens.get("input")) or _safe_int(tokens.get("input_tokens")) or 0
            output_tokens = _safe_int(tokens.get("output")) or _safe_int(tokens.get("output_tokens")) or 0
            cache_read = _safe_int(tokens.get("cached")) or _safe_int(tokens.get("cache_read_input_tokens")) or 0
            reasoning_tokens = _safe_int(tokens.get("thoughts")) or _safe_int(tokens.get("reasoning")) or 0
            total_tokens = _safe_int(tokens.get("total")) or input_tokens + output_tokens
            tool_activity: list[dict[str, Any]] = []
            tool_calls = message.get("toolCalls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_activity.append(
                        _build_tool_activity(
                            tool_index=len(tool_activity),
                            tool_name=_safe_str(tool_call.get("name")) or "unknown",
                            tool_call_id=_safe_str(tool_call.get("id")),
                            arguments=tool_call.get("args") or {},
                            metadata={"source": "gemini_toolCalls"},
                        )
                    )
            thoughts = message.get("thoughts") if isinstance(message.get("thoughts"), list) else []
            metadata = {
                "source_path": str(path),
                "source_path_hash": path_hash,
                "source_message_index": message_index,
                "source_message_id": message_id,
                "project_name": project_name,
                "project_hash": header.get("projectHash"),
                "session_kind": header.get("kind"),
                "session_start_time": header.get("startTime"),
                "session_last_updated": header.get("lastUpdated"),
                "source_tokens": _redact_value(tokens),
                "thought_count": len(thoughts),
            }
            yield _base_record(
                source_client_name="gemini",
                created_at=created_at,
                litellm_call_id=f"local-cli:gemini:{session_id}:{message_id}:{path_hash}",
                session_id=session_id,
                provider="gemini",
                model=model,
                repository=repository,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read,
                reasoning_tokens_reported=reasoning_tokens or None,
                reasoning_tokens_source=(
                    "gemini_message.tokens.thoughts" if reasoning_tokens else None
                ),
                reasoning_present=reasoning_tokens > 0 or bool(thoughts),
                tool_activity=tool_activity,
                usage_obj=tokens,
                metadata=metadata,
            )


def _iter_grok_session_dirs(sessions_root: Path) -> Iterator[Path]:
    for cwd_dir in sessions_root.iterdir() if sessions_root.exists() else []:
        if not cwd_dir.is_dir():
            continue
        for session_dir in cwd_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "summary.json").exists():
                yield session_dir


def _grok_cwd_from_session_dir(session_dir: Path) -> Optional[str]:
    try:
        return unquote(session_dir.parent.name)
    except Exception:
        return None


def _grok_reasoning_state(obj: dict[str, Any]) -> tuple[bool, Optional[int], bool]:
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, dict):
        return False, None, False
    text = reasoning.get("text")
    encrypted = reasoning.get("encrypted")
    estimated = _estimate_tokens_from_text(text) if isinstance(text, str) else None
    return bool(text or encrypted), estimated, bool(encrypted)


def _iter_grok_records(root: Path, stats: ScanStats) -> Iterator[dict[str, Any]]:
    sessions_root = root / ".grok" / "sessions"
    if not sessions_root.exists():
        stats.files_skipped["grok:missing_sessions"] += 1
        return
    for session_dir in _iter_grok_session_dirs(sessions_root):
        stats.files_seen["grok_summary"] += 1
        try:
            summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats.files_skipped["grok:summary_error"] += 1
            continue
        info = summary.get("info") if isinstance(summary.get("info"), dict) else {}
        session_id = _safe_str(info.get("id")) or session_dir.name
        repository = _safe_str(info.get("cwd")) or _grok_cwd_from_session_dir(session_dir)
        model = _safe_str(summary.get("current_model_id")) or "grok-build"
        created_base = (
            _parse_datetime(summary.get("created_at"))
            or _parse_datetime(summary.get("last_active_at"))
            or _parse_datetime(summary.get("updated_at"))
        )
        if created_base is None:
            continue
        path = session_dir / "chat_history.jsonl"
        for line_index, obj in _iter_jsonl(path, stats, "grok"):
            if obj.get("type") != "assistant":
                continue
            row_time = created_base + timedelta(milliseconds=line_index)
            tool_activity: list[dict[str, Any]] = []
            tool_calls = obj.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_activity.append(
                        _build_tool_activity(
                            tool_index=len(tool_activity),
                            tool_name=_safe_str(tool_call.get("name")) or "unknown",
                            tool_call_id=_safe_str(tool_call.get("id")),
                            arguments=tool_call.get("arguments")
                            or tool_call.get("args")
                            or tool_call.get("input")
                            or {},
                            metadata={"source": "grok_tool_calls"},
                        )
                    )
            reasoning_present, reasoning_estimated, signature_present = (
                _grok_reasoning_state(obj)
            )
            metadata = {
                "source_path": str(path),
                "source_path_hash": _sha(str(path)),
                "source_line": line_index,
                "session_request_id": summary.get("request_id"),
                "session_created_at": summary.get("created_at"),
                "session_updated_at": summary.get("updated_at"),
                "session_last_active_at": summary.get("last_active_at"),
                "session_kind": summary.get("session_kind"),
                "git_branch": summary.get("head_branch"),
                "head_commit": summary.get("head_commit"),
                "num_messages": summary.get("num_messages"),
                "num_chat_messages": summary.get("num_chat_messages"),
                "model_fingerprint": obj.get("model_fingerprint"),
            }
            yield _base_record(
                source_client_name="grok",
                created_at=row_time,
                litellm_call_id=f"local-cli:grok:{session_id}:{line_index}",
                session_id=session_id,
                provider="xai",
                model=model,
                repository=repository,
                reasoning_tokens_estimated=reasoning_estimated,
                reasoning_tokens_source=(
                    "estimated_from_grok_reasoning_text" if reasoning_estimated else None
                ),
                reasoning_present=reasoning_present,
                thinking_signature_present=signature_present,
                tool_activity=tool_activity,
                metadata=metadata,
            )


def _iter_records(root: Path, clients: set[str], stats: ScanStats) -> Iterator[dict[str, Any]]:
    if "claude" in clients:
        yield from _iter_claude_records(root, stats)
    if "codex" in clients:
        yield from _iter_codex_records(root, stats)
    if "gemini" in clients:
        yield from _iter_gemini_records(root, stats)
    if "grok" in clients:
        yield from _iter_grok_records(root, stats)


def _postgres_dsn_from_args(args: argparse.Namespace) -> str:
    if args.pg_dsn:
        return args.pg_dsn
    has_component_override = any(
        (args.pg_host, args.pg_port, args.pg_user, args.pg_password)
    )
    direct_dsn = os.getenv("AAWM_DIRECT_DATABASE_URL")
    if not has_component_override and direct_dsn and direct_dsn.strip():
        return direct_dsn.strip()
    host = args.pg_host or os.getenv("AAWM_DB_HOST") or "127.0.0.1"
    port = args.pg_port or os.getenv("AAWM_DB_PORT") or "5434"
    user = args.pg_user or os.getenv("AAWM_DB_USER") or "aawm"
    password = args.pg_password or os.getenv("AAWM_DB_PASSWORD") or "aawm_dev"
    return f"postgresql://{quote(user)}:{quote(password)}@{host}:{port}/{quote(args.target_db_name)}"


def _fetch_existing_days(args: argparse.Namespace, tz_name: str) -> set[date]:
    if psycopg is None:
        raise RuntimeError("psycopg is required for the session_history day check")
    dsn = _postgres_dsn_from_args(args)
    with psycopg.connect(dsn, connect_timeout=args.pg_connect_timeout) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            [current_database] = cur.fetchone()
            if current_database != args.target_db_name:
                raise RuntimeError(
                    f"Refusing to use database {current_database!r}; expected "
                    f"{args.target_db_name!r}"
                )
            cur.execute(
                """
                SELECT DISTINCT (COALESCE(start_time, created_at) AT TIME ZONE %s)::date
                FROM public.session_history
                WHERE COALESCE(start_time, created_at) IS NOT NULL
                """,
                (tz_name,),
            )
            return {row[0] for row in cur.fetchall() if row[0] is not None}


SESSION_HISTORY_INSERT_SQL = """
INSERT INTO public.session_history (
    created_at,
    litellm_call_id,
    session_id,
    trace_id,
    provider_response_id,
    provider,
    model,
    model_group,
    agent_name,
    tenant_id,
    call_type,
    start_time,
    end_time,
    input_tokens,
    output_tokens,
    total_tokens,
    cache_read_input_tokens,
    cache_creation_input_tokens,
    reasoning_tokens_reported,
    reasoning_tokens_estimated,
    reasoning_tokens_source,
    reasoning_present,
    thinking_signature_present,
    provider_cache_attempted,
    provider_cache_status,
    provider_cache_miss,
    provider_cache_miss_reason,
    provider_cache_miss_token_count,
    provider_cache_miss_cost_usd,
    tool_call_count,
    invalid_tool_call_count,
    tool_names,
    file_read_count,
    file_modified_count,
    changed_pre_commit_config,
    changed_env_file,
    changed_pyproject_toml,
    changed_gitignore,
    git_commit_count,
    git_push_count,
    response_cost_usd,
    litellm_environment,
    client_name,
    client_version,
    client_user_agent,
    token_permission_input,
    token_permission_output,
    permission_usd_cost,
    metadata,
    repository
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
)
ON CONFLICT (litellm_call_id) DO NOTHING
"""

TOOL_ACTIVITY_INSERT_SQL = """
INSERT INTO public.session_history_tool_activity (
    created_at,
    litellm_call_id,
    session_id,
    trace_id,
    provider,
    model,
    agent_name,
    tool_index,
    tool_call_id,
    tool_name,
    tool_kind,
    file_paths_read,
    file_paths_modified,
    git_commit_count,
    git_push_count,
    command_text,
    arguments,
    metadata
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb
)
ON CONFLICT (litellm_call_id, tool_index) DO NOTHING
"""


def _history_payload(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["created_at"],
        record["litellm_call_id"],
        record["session_id"],
        record.get("trace_id"),
        record.get("provider_response_id"),
        record.get("provider"),
        record["model"],
        record.get("model_group"),
        record.get("agent_name"),
        record.get("tenant_id"),
        record.get("call_type"),
        record.get("start_time"),
        record.get("end_time"),
        record.get("input_tokens") or 0,
        record.get("output_tokens") or 0,
        record.get("total_tokens") or 0,
        record.get("cache_read_input_tokens") or 0,
        record.get("cache_creation_input_tokens") or 0,
        record.get("reasoning_tokens_reported"),
        record.get("reasoning_tokens_estimated"),
        record.get("reasoning_tokens_source"),
        bool(record.get("reasoning_present")),
        bool(record.get("thinking_signature_present")),
        bool(record.get("provider_cache_attempted")),
        record.get("provider_cache_status"),
        bool(record.get("provider_cache_miss")),
        record.get("provider_cache_miss_reason"),
        record.get("provider_cache_miss_token_count"),
        record.get("provider_cache_miss_cost_usd"),
        record.get("tool_call_count") or 0,
        record.get("invalid_tool_call_count") or 0,
        json.dumps(record.get("tool_names") or []),
        record.get("file_read_count") or 0,
        record.get("file_modified_count") or 0,
        record.get("changed_pre_commit_config"),
        record.get("changed_env_file"),
        record.get("changed_pyproject_toml"),
        record.get("changed_gitignore"),
        record.get("git_commit_count") or 0,
        record.get("git_push_count") or 0,
        record.get("response_cost_usd"),
        record.get("litellm_environment"),
        record.get("client_name"),
        record.get("client_version"),
        record.get("client_user_agent"),
        record.get("token_permission_input") or 0,
        record.get("token_permission_output") or 0,
        record.get("permission_usd_cost") or 0.0,
        json.dumps(record.get("metadata") or {}),
        record.get("repository"),
    )


def _tool_payloads(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    payloads: list[tuple[Any, ...]] = []
    record_metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for index, tool in enumerate(record.get("tool_activity") or []):
        if not isinstance(tool, dict):
            continue
        tool_metadata = tool.get("metadata") if isinstance(tool.get("metadata"), dict) else {}
        stored_tool_metadata = {
            **tool_metadata,
            "source_import": record_metadata.get("source_import"),
            "parser_version": record_metadata.get("parser_version"),
            "source_client": record_metadata.get("source_client"),
            "client_name": record.get("client_name"),
            "client_user_agent": record.get("client_user_agent"),
        }
        payloads.append(
            (
                record["created_at"],
                record["litellm_call_id"],
                record["session_id"],
                record.get("trace_id"),
                record.get("provider"),
                record["model"],
                record.get("agent_name"),
                _safe_int(tool.get("tool_index")) if _safe_int(tool.get("tool_index")) is not None else index,
                tool.get("tool_call_id"),
                tool.get("tool_name") or "unknown",
                tool.get("tool_kind"),
                json.dumps(tool.get("file_paths_read") or []),
                json.dumps(tool.get("file_paths_modified") or []),
                tool.get("git_commit_count") or 0,
                tool.get("git_push_count") or 0,
                tool.get("command_text"),
                json.dumps(tool.get("arguments") or {}),
                json.dumps(stored_tool_metadata),
            )
        )
    return payloads


def _apply_batch(conn: Any, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with conn.cursor() as cur:
        cur.executemany(SESSION_HISTORY_INSERT_SQL, [_history_payload(item) for item in records])
        tool_payloads: list[tuple[Any, ...]] = []
        for record in records:
            tool_payloads.extend(_tool_payloads(record))
        if tool_payloads:
            cur.executemany(TOOL_ACTIVITY_INSERT_SQL, tool_payloads)


def _sample_record(record: dict[str, Any], day: date) -> dict[str, Any]:
    return {
        "day": day.isoformat(),
        "client_name": record.get("client_name"),
        "model": record.get("model"),
        "model_group": record.get("model_group"),
        "provider": record.get("provider"),
        "repository": record.get("repository"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "cache_read_input_tokens": record.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": record.get("cache_creation_input_tokens"),
        "provider_cache_status": record.get("provider_cache_status"),
        "provider_cache_miss": record.get("provider_cache_miss"),
        "provider_cache_miss_reason": record.get("provider_cache_miss_reason"),
        "provider_cache_miss_token_count": record.get("provider_cache_miss_token_count"),
        "provider_cache_miss_cost_usd": record.get("provider_cache_miss_cost_usd"),
        "reasoning_tokens": record.get("reasoning_tokens_reported")
        or record.get("reasoning_tokens_estimated"),
        "response_cost_usd": record.get("response_cost_usd"),
        "client_user_agent": record.get("client_user_agent"),
        "tool_call_count": record.get("tool_call_count"),
        "tool_names": record.get("tool_names"),
        "litellm_call_id": record.get("litellm_call_id"),
    }


def _print_counter(title: str, counter: Counter[str], limit: int = 20) -> None:
    print(f"\n{title}")
    if not counter:
        print("  none")
        return
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")
    if len(counter) > limit:
        print(f"  ... {len(counter) - limit} more")


def _print_summary(
    *,
    summary: DryRunSummary,
    stats: ScanStats,
    existing_days: set[date],
    args: argparse.Namespace,
) -> None:
    print("Local CLI session_history backfill dry run")
    print(f"target_database: {args.target_db_name}")
    print(f"day_timezone: {args.day_timezone}")
    print(f"source_import: {IMPORT_MARKER}")
    if existing_days:
        print(
            "existing_session_history_days: "
            f"{len(existing_days)} ({min(existing_days).isoformat()}..{max(existing_days).isoformat()})"
        )
    else:
        print("existing_session_history_days: 0")
    print(f"records_that_would_be_added: {summary.records}")
    print(f"tool_activity_rows_that_would_be_added: {summary.tool_rows}")
    print(
        "stored_token_fields_that_would_be_added: "
        f"input={summary.input_tokens} "
        f"output={summary.output_tokens} "
        f"cache_read={summary.cache_read_input_tokens} "
        f"cache_creation={summary.cache_creation_input_tokens} "
        f"reasoning={summary.reasoning_tokens}"
    )
    print(
        "prompt_tokens_including_cache_that_would_be_added: "
        f"{summary.input_tokens + summary.cache_read_input_tokens + summary.cache_creation_input_tokens}"
    )
    print(
        "cost_fields_that_would_be_added: "
        f"rows_with_response_cost={summary.response_cost_rows} "
        f"response_cost_usd={summary.response_cost_usd:.6f} "
        f"cache_miss_rows={summary.provider_cache_miss_rows} "
        f"cache_miss_tokens={summary.provider_cache_miss_token_count} "
        f"cache_miss_cost_usd={summary.provider_cache_miss_cost_usd:.6f}"
    )
    print(
        "derived_activity_that_would_be_added: "
        f"file_reads={summary.file_read_count} "
        f"file_modifications={summary.file_modified_count} "
        f"git_commits={summary.git_commit_count} "
        f"git_pushes={summary.git_push_count}"
    )
    print(f"records_without_token_detail: {summary.no_token_records}")

    if summary.by_date:
        dates = sorted(summary.by_date)
        print(f"candidate_date_range: {dates[0].isoformat()}..{dates[-1].isoformat()}")
        print("\nCandidate days")
        for day in dates:
            row = summary.by_date[day]
            print(
                "  "
                f"{day.isoformat()}: records={row['records']} tools={row['tool_rows']} "
                f"in={row['input_tokens']} out={row['output_tokens']} "
                f"cache_read={row['cache_read_input_tokens']} "
                f"cache_create={row['cache_creation_input_tokens']} "
                f"reasoning={row['reasoning_tokens']} "
                f"response_cost_usd={row['response_cost_usd']:.6f} "
                f"cache_miss_rows={row['provider_cache_miss_rows']} "
                f"cache_miss_tokens={row['provider_cache_miss_token_count']} "
                f"cache_miss_cost_usd={row['provider_cache_miss_cost_usd']:.6f} "
                f"file_reads={row['file_read_count']} "
                f"file_mods={row['file_modified_count']} "
                f"git_commits={row['git_commit_count']} "
                f"git_pushes={row['git_push_count']}"
            )
    else:
        print("candidate_date_range: none")

    print("\nBy client")
    if not summary.by_client:
        print("  none")
    for client in sorted(summary.by_client):
        row = summary.by_client[client]
        print(
            "  "
            f"{client}: records={row['records']} tools={row['tool_rows']} "
            f"in={row['input_tokens']} out={row['output_tokens']} "
            f"cache_read={row['cache_read_input_tokens']} "
            f"cache_create={row['cache_creation_input_tokens']} "
            f"reasoning={row['reasoning_tokens']} "
            f"response_cost_rows={row['response_cost_rows']} "
            f"response_cost_usd={row['response_cost_usd']:.6f} "
            f"cache_miss_rows={row['provider_cache_miss_rows']} "
            f"cache_miss_tokens={row['provider_cache_miss_token_count']} "
            f"cache_miss_cost_usd={row['provider_cache_miss_cost_usd']:.6f} "
            f"file_reads={row['file_read_count']} "
            f"file_mods={row['file_modified_count']} "
            f"git_commits={row['git_commit_count']} "
            f"git_pushes={row['git_push_count']} "
            f"no_tokens={row['no_token_records']}"
        )

    _print_counter("Top client/model buckets", summary.by_model)
    _print_counter("Top tool buckets", summary.by_tool)
    _print_counter("Files seen", stats.files_seen)
    _print_counter("Files skipped", stats.files_skipped)
    _print_counter("Parse errors", stats.parse_errors)
    _print_counter("Records seen before day skip", stats.records_seen)
    _print_counter("Records skipped because day already exists", stats.records_skipped_existing_day)

    print("\nRedacted sample candidate rows")
    if not summary.samples:
        print("  none")
    for sample in summary.samples:
        print("  " + json.dumps(sample, sort_keys=True))


def _parse_clients(value: str) -> set[str]:
    clients = {part.strip().lower() for part in value.split(",") if part.strip()}
    valid = {"claude", "codex", "gemini", "grok"}
    unknown = clients - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown clients: {', '.join(sorted(unknown))}")
    return clients or valid


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run/apply local CLI transcript backfill into session_history."
    )
    parser.add_argument("--home", default=str(Path.home()), help="Home directory containing CLI state")
    parser.add_argument("--clients", type=_parse_clients, default=_parse_clients("claude,codex,gemini,grok"))
    parser.add_argument("--day-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--from-date", help="Only include candidate days on or after YYYY-MM-DD")
    parser.add_argument("--to-date", help="Only include candidate days on or before YYYY-MM-DD")
    parser.add_argument("--apply", action="store_true", help="Write rows. Default is dry-run only.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--target-db-name", default=DEFAULT_TARGET_DB)
    parser.add_argument("--pg-dsn")
    parser.add_argument("--pg-host")
    parser.add_argument("--pg-port")
    parser.add_argument("--pg-user")
    parser.add_argument("--pg-password")
    parser.add_argument("--pg-connect-timeout", type=int, default=5)
    return parser.parse_args(argv)


def _optional_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.home).expanduser()
    tz = ZoneInfo(args.day_timezone)
    from_date = _optional_date(args.from_date)
    to_date = _optional_date(args.to_date)
    existing_days = _fetch_existing_days(args, args.day_timezone)
    stats = ScanStats()
    summary = DryRunSummary()

    conn = None
    batch: list[dict[str, Any]] = []
    if args.apply:
        if psycopg is None:
            raise RuntimeError("psycopg is required for --apply")
        conn = psycopg.connect(
            _postgres_dsn_from_args(args),
            connect_timeout=args.pg_connect_timeout,
        )
        conn.autocommit = False

    try:
        for record in _iter_records(root, args.clients, stats):
            client = str(record.get("client_name") or "unknown")
            stats.records_seen[client] += 1
            created_at = record.get("created_at")
            if not isinstance(created_at, datetime):
                continue
            day = _date_bucket(created_at, tz)
            if from_date and day < from_date:
                continue
            if to_date and day > to_date:
                continue
            if day in existing_days:
                stats.records_skipped_existing_day[client] += 1
                continue
            stats.records_after_day_filter[client] += 1
            summary.add(record, day)
            if args.apply and conn is not None:
                batch.append(record)
                if len(batch) >= args.batch_size:
                    _apply_batch(conn, batch)
                    conn.commit()
                    batch.clear()
        if args.apply and conn is not None and batch:
            _apply_batch(conn, batch)
            conn.commit()
            batch.clear()
    finally:
        if conn is not None:
            conn.close()

    _print_summary(summary=summary, stats=stats, existing_days=existing_days, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
