"""Bounded deterministic agent-quality rule matching for AAWM callbacks.

This module intentionally uses literal phrase matching and capped input sizes.
The LiteLLM callback path must not inspect local repositories, run subprocesses,
or depend on expensive parsing to derive these scores.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_DEFAULT_RELOAD_TTL_SECONDS = 7200
_DEFAULT_LIMITS = {
    "max_file_bytes": 65536,
    "max_phrases_per_group": 160,
    "max_phrase_chars": 180,
    "max_text_bytes": 100000,
    "max_commands": 400,
    "max_command_chars": 1200,
}
_CATALOG_ENV = "AAWM_AGENT_QUALITY_RULES_PATH"
_DEFAULT_CATALOG_PATH = Path(__file__).with_name("aawm_agent_quality_rules.json")
_DISCOVERY_CONTRACT_MARKERS = (
    "discovery inventory required",
    "list the discovery command",
    "list every candidate",
    "mark each candidate as inspected",
    "classify relevant candidates",
    "call out any coverage gap",
)
_DISCOVERY_NO_CONTRACT_MARKERS = (
    "no broad discovery inventory is required",
    "no discovery inventory is required",
)
_DISCOVERY_BROAD_SCOPE_MARKERS = (
    "any recent",
    "handoff",
    "contract",
    "investigate-",
    "glob",
    "similar files",
    "related docs",
    "candidate file",
    "candidate item",
)
_DISCOVERY_CLASSIFICATION_MARKERS = (
    "actionable",
    "stale",
    "context-only",
    "context only",
    "not relevant",
)
_DISCOVERY_ACCOUNTING_MARKERS = (
    "inspected",
    "omitted",
    "unavailable",
)
_OUTPUT_CONTRACT_TASK_MARKERS = (
    "read-only task",
    "read only task",
    "do not edit files",
    "do not modify files",
    "no files were modified",
    "read-only audit",
    "readonly audit",
    "scout",
    "audit",
    "inspect",
)
_OUTPUT_CONTRACT_SETUP_PREFIXES = (
    "i will ",
    "i'll ",
    "i am going to ",
    "i'm going to ",
    "i plan to ",
    "plan:",
    "next i will ",
    "next i'll ",
)
_OUTPUT_CONTRACT_SETUP_MARKERS = (
    "i will inspect",
    "i'll inspect",
    "i will review",
    "i'll review",
    "i will audit",
    "i'll audit",
    "i will check",
    "i'll check",
    "i will look",
    "i'll look",
    "i am going to inspect",
    "i'm going to inspect",
    "i am going to review",
    "i'm going to review",
)
_OUTPUT_CONTRACT_COMPLETION_MARKERS = (
    "found",
    "inspected",
    "verified",
    "confirmed",
    "evidence",
    "recommend",
    "result",
    "passed",
    "failed",
    "issue",
    "line ",
    ".py",
    ".md",
)
_COMPOSER_CALL_TEXT_MARKERS = (
    re.compile(r"^\s*composer_call\s*$", re.IGNORECASE),
    re.compile(r"\bcomposer_call\s*\(", re.IGNORECASE),
    re.compile(r"['\"]name['\"]\s*:\s*['\"]composer_call['\"]", re.IGNORECASE),
)
_COMPOSER_CALL_NAME = "composer_call"
_COMPOSER_CALL_TRANSCRIPT_FIELD_RE = re.compile(
    r"(?im)^(?:name|arguments):\s*",
)
_COMPOSER_CALL_CALL_ID_LINE_RE = re.compile(
    r"(?im)^call id:\s*[^\n]*composer_call",
)
_COMPOSER_CALL_SAME_LINE_TRANSCRIPT_RE = re.compile(
    r"(?is)"
    r"(?:^|\b)name:\s*\S+.*?"
    r"call\s*id:\s*[^\s\n]*composer_call[^\s\n]*.*?"
    r"arguments:\s*",
)
_COMPOSER_CALL_TOOL_CALL_MARKERS = (
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
)


def is_malformed_composer_call_literal_text(value: str) -> bool:
    """Detect non-executing composer_call text without flagging benign prose."""
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = _normalize_contract_text(value)
    if "composer_call" not in normalized:
        return False
    if any(pattern.search(value) for pattern in _COMPOSER_CALL_TEXT_MARKERS):
        return True
    if _COMPOSER_CALL_SAME_LINE_TRANSCRIPT_RE.search(value):
        return True
    if any(marker in value for marker in _COMPOSER_CALL_TOOL_CALL_MARKERS):
        return True
    if _COMPOSER_CALL_CALL_ID_LINE_RE.search(value) and (
        _COMPOSER_CALL_TRANSCRIPT_FIELD_RE.search(value)
        or "previous tool call" in normalized
        or "arguments:" in normalized
    ):
        return True
    return False


_DISCOVERY_PATH_RE = re.compile(
    r"(?<![\w@:/.-])(?:\./|\.\./|/|[A-Za-z0-9_.-]+/)"
    r"[A-Za-z0-9_./{}@%+=,:~#-]*[A-Za-z0-9_./{}@%+=~#-]"
)
_DISCOVERY_PATH_SUFFIXES = (
    ".md",
    ".py",
    ".json",
    ".jsonl",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".sql",
    ".sh",
    ".tsx",
    ".ts",
    ".js",
    ".jsx",
)


@dataclass(frozen=True)
class AgentQualityCommand:
    name: str = ""
    command: str = ""
    timestamp: Optional[str] = None
    affected_paths: Tuple[str, ...] = ()


@dataclass
class RuleCatalogCacheEntry:
    catalog: Dict[str, Any]
    loaded_at: float
    path: Optional[str]
    mtime_ns: Optional[int]
    error: Optional[str] = None


@dataclass
class AgentQualityRuleResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    reasons: Dict[str, Any] = field(default_factory=dict)


_CATALOG_CACHE: Dict[str, RuleCatalogCacheEntry] = {}


def reset_agent_quality_rule_catalog_cache() -> None:
    _CATALOG_CACHE.clear()


def _default_catalog() -> Dict[str, Any]:
    path = _DEFAULT_CATALOG_PATH
    if not path.exists():
        return {
            "version": "builtin-empty",
            "reload_ttl_seconds": _DEFAULT_RELOAD_TTL_SECONDS,
            "limits": dict(_DEFAULT_LIMITS),
            "ignored_path_tracking": {},
            "baseline_deflection": {},
            "sleep_wellness_interruption": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return _validate_catalog(data)


def _catalog_path(path: Optional[Path | str] = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv(_CATALOG_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return _DEFAULT_CATALOG_PATH


def _int_limit(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _phrase_list(value: Any, *, max_count: int, max_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    phrases: List[str] = []
    for item in value[:max_count]:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        phrases.append(cleaned[:max_chars])
    return phrases


def _validate_catalog(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("catalog root must be an object")
    raw_limits_value = raw.get("limits")
    raw_limits: Dict[str, Any] = (
        raw_limits_value if isinstance(raw_limits_value, dict) else {}
    )
    limits = {
        "max_file_bytes": _int_limit(
            raw_limits.get("max_file_bytes"),
            _DEFAULT_LIMITS["max_file_bytes"],
            minimum=4096,
            maximum=262144,
        ),
        "max_phrases_per_group": _int_limit(
            raw_limits.get("max_phrases_per_group"),
            _DEFAULT_LIMITS["max_phrases_per_group"],
            minimum=1,
            maximum=500,
        ),
        "max_phrase_chars": _int_limit(
            raw_limits.get("max_phrase_chars"),
            _DEFAULT_LIMITS["max_phrase_chars"],
            minimum=8,
            maximum=512,
        ),
        "max_text_bytes": _int_limit(
            raw_limits.get("max_text_bytes"),
            _DEFAULT_LIMITS["max_text_bytes"],
            minimum=1024,
            maximum=500000,
        ),
        "max_commands": _int_limit(
            raw_limits.get("max_commands"),
            _DEFAULT_LIMITS["max_commands"],
            minimum=1,
            maximum=1000,
        ),
        "max_command_chars": _int_limit(
            raw_limits.get("max_command_chars"),
            _DEFAULT_LIMITS["max_command_chars"],
            minimum=80,
            maximum=5000,
        ),
    }
    max_count = limits["max_phrases_per_group"]
    max_chars = limits["max_phrase_chars"]

    def section(name: str, keys: Sequence[str]) -> Dict[str, Any]:
        raw_section = raw.get(name)
        if not isinstance(raw_section, dict):
            raw_section = {}
        validated: Dict[str, Any] = {}
        for key in keys:
            validated[key] = _phrase_list(
                raw_section.get(key),
                max_count=max_count,
                max_chars=max_chars,
            )
        thresholds = raw_section.get("thresholds")
        if isinstance(thresholds, dict):
            validated["thresholds"] = {
                str(key): _int_limit(value, 0, minimum=0, maximum=10_000_000)
                for key, value in thresholds.items()
                if isinstance(key, str)
            }
        else:
            validated["thresholds"] = {}
        return validated

    ttl = _int_limit(
        raw.get("reload_ttl_seconds"),
        _DEFAULT_RELOAD_TTL_SECONDS,
        minimum=60,
        maximum=86400,
    )
    return {
        "version": str(raw.get("version") or "unknown")[:80],
        "reload_ttl_seconds": ttl,
        "limits": limits,
        "ignored_path_tracking": section(
            "ignored_path_tracking",
            (
                "authorization_phrases",
                "prohibition_phrases",
                "common_ignored_prefixes",
            ),
        ),
        "baseline_deflection": section(
            "baseline_deflection",
            (
                "quality_gate_triggers",
                "framing_phrases",
                "defer_phrases",
                "provenance_probe_phrases",
                "fix_command_phrases",
            ),
        ),
        "sleep_wellness_interruption": section(
            "sleep_wellness_interruption",
            (
                "candidate_phrases",
                "task_interruption_phrases",
                "suppression_user_phrases",
                "pushback_user_phrases",
            ),
        ),
    }


def load_agent_quality_rule_catalog(
    *,
    path: Optional[Path | str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    catalog_path = _catalog_path(path)
    cache_key = str(catalog_path)
    cached = _CATALOG_CACHE.get(cache_key)
    ttl = (
        cached.catalog.get("reload_ttl_seconds", _DEFAULT_RELOAD_TTL_SECONDS)
        if cached is not None
        else _DEFAULT_RELOAD_TTL_SECONDS
    )
    if cached is not None and now - cached.loaded_at < ttl:
        return cached.catalog

    try:
        stat = catalog_path.stat()
        if stat.st_size > _DEFAULT_LIMITS["max_file_bytes"]:
            raise ValueError("catalog exceeds maximum file size")
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = _validate_catalog(raw)
        limits = catalog["limits"]
        if stat.st_size > limits["max_file_bytes"]:
            raise ValueError("catalog exceeds configured maximum file size")
        _CATALOG_CACHE[cache_key] = RuleCatalogCacheEntry(
            catalog=catalog,
            loaded_at=now,
            path=str(catalog_path),
            mtime_ns=stat.st_mtime_ns,
        )
        return catalog
    except Exception as exc:
        if cached is not None:
            cached.loaded_at = now
            cached.error = str(exc)
            return cached.catalog
        fallback = _default_catalog()
        _CATALOG_CACHE[cache_key] = RuleCatalogCacheEntry(
            catalog=fallback,
            loaded_at=now,
            path=str(catalog_path),
            mtime_ns=None,
            error=str(exc),
        )
        return fallback


def _clip_text(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _joined_text(values: Iterable[str], *, max_bytes: int) -> str:
    parts: List[str] = []
    remaining = max_bytes
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        clipped = _clip_text(value, remaining)
        if clipped:
            parts.append(clipped)
            remaining -= len(clipped.encode("utf-8", errors="ignore"))
        if remaining <= 0:
            break
    return "\n".join(parts).lower()


def _joined_text_preserve_case(values: Iterable[str], *, max_bytes: int) -> str:
    parts: List[str] = []
    remaining = max_bytes
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        clipped = _clip_text(value, remaining)
        if clipped:
            parts.append(clipped)
            remaining -= len(clipped.encode("utf-8", errors="ignore"))
        if remaining <= 0:
            break
    return "\n".join(parts)


def _normalize_contract_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _last_non_empty_text(values: Sequence[str]) -> str:
    for value in reversed(values):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_required_final_phrase(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    if not user_text:
        return None, None
    patterns = (
        (
            "final_answer_must_include",
            r"(?:final answer|final response|final output)[^\n]{0,100}"
            r"must[^\n]{0,80}include[^\n]{0,30}([\"'`])(?P<phrase>[^\"'`\n]{1,180})\1",
        ),
        (
            "must_include_exact_phrase",
            r"must[^\n]{0,80}include[^\n]{0,30}(?:exact phrase|phrase)?"
            r"[^\n]{0,20}([\"'`])(?P<phrase>[^\"'`\n]{1,180})\1",
        ),
    )
    for source, pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        phrase = (match.group("phrase") or "").strip()
        if phrase:
            return phrase, source
    return None, None


def _contract_phrase_present(final_text: str, phrase: str) -> bool:
    normalized_final = _normalize_contract_text(final_text)
    normalized_phrase = _normalize_contract_text(phrase)
    return bool(normalized_phrase and normalized_phrase in normalized_final)


def _setup_only_markers(final_text: str, user_text: str) -> List[str]:
    normalized_final = _normalize_contract_text(final_text)
    if not normalized_final or len(normalized_final) > 500:
        return []
    normalized_user = _normalize_contract_text(user_text)
    if not any(marker in normalized_user for marker in _OUTPUT_CONTRACT_TASK_MARKERS):
        return []
    if any(marker in normalized_final for marker in _OUTPUT_CONTRACT_COMPLETION_MARKERS):
        return []

    markers: List[str] = []
    for prefix in _OUTPUT_CONTRACT_SETUP_PREFIXES:
        if normalized_final.startswith(prefix):
            markers.append(prefix.strip())
    for marker in _OUTPUT_CONTRACT_SETUP_MARKERS:
        if marker in normalized_final:
            markers.append(marker)
    return sorted(set(markers))


def _score_output_contract(
    *,
    user_text: str,
    assistant_texts: Sequence[str],
    tool_call_names: Sequence[str] = (),
) -> Dict[str, Any]:
    final_text = _last_non_empty_text(assistant_texts)
    required_phrase, required_phrase_source = _extract_required_final_phrase(user_text)
    required_phrase_present: Optional[bool] = None
    if required_phrase is not None:
        required_phrase_present = _contract_phrase_present(final_text, required_phrase)

    setup_markers = _setup_only_markers(final_text, user_text)
    failure_class: Optional[str] = None
    if required_phrase_present is False:
        failure_class = "missing_required_final_phrase"
    if setup_markers:
        failure_class = "setup_only_completion"
    composer_call_markers = [
        name
        for name in tool_call_names
        if isinstance(name, str)
        and name.strip().lower() == _COMPOSER_CALL_NAME
    ]
    if not composer_call_markers and final_text:
        if is_malformed_composer_call_literal_text(final_text):
            failure_class = failure_class or "literal_tool_call_text"
    elif composer_call_markers:
        failure_class = failure_class or "malformed_tool_call_text"

    contract_observed = (
        required_phrase is not None
        or bool(setup_markers)
        or bool(failure_class)
    )
    compliance_score: Optional[float] = None
    if contract_observed:
        compliance_score = 0.0 if failure_class else 1.0

    return {
        "output_contract_compliance_score": compliance_score,
        "output_contract_required_final_phrase": required_phrase,
        "output_contract_required_final_phrase_present": required_phrase_present,
        "output_contract_required_final_phrase_source": required_phrase_source,
        "output_contract_failure_class": failure_class,
        "output_contract_failure_count": (
            1 if failure_class else 0 if contract_observed else None
        ),
        "output_contract_setup_only_detected": (
            bool(setup_markers) if contract_observed else None
        ),
        "output_contract_setup_only_markers": setup_markers,
        "output_contract_final_text_chars": (
            len(final_text) if contract_observed else None
        ),
    }


def _matched_phrases(text: str, phrases: Sequence[str]) -> List[str]:
    return [phrase for phrase in phrases if phrase and phrase in text]


def _snippet(value: str, limit: int = 220) -> str:
    cleaned = " ".join(str(value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _normalize_discovery_candidate_path(value: str) -> Optional[str]:
    cleaned = value.strip().strip("`'\"[](){}<>,;:")
    cleaned = cleaned.replace("\\", "/")
    if not cleaned or "://" in cleaned or "*" in cleaned:
        return None
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    lowered = cleaned.lower()
    if not (
        "/" in cleaned
        or lowered.startswith(".analysis/")
        or lowered.startswith("analysis/")
    ):
        return None
    if not (
        lowered.endswith(_DISCOVERY_PATH_SUFFIXES)
        or "/investigate-" in lowered
        or lowered.startswith(".analysis/")
        or lowered.startswith("analysis/")
    ):
        return None
    return cleaned


def _discovery_candidate_paths_from_text(text: str) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()
    for match in _DISCOVERY_PATH_RE.finditer(text or ""):
        normalized = _normalize_discovery_candidate_path(match.group(0))
        if normalized is None:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)
    return candidates


def _discovery_candidate_paths_from_commands(
    commands: Sequence[AgentQualityCommand],
) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()
    for command in commands:
        for path in command.affected_paths:
            normalized = _normalize_discovery_candidate_path(path)
            if normalized is None:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(normalized)
    return candidates


def _path_accounted_for(path: str, text: str) -> bool:
    lowered = text.lower()
    path_lower = path.lower()
    variants = {
        path_lower,
        path_lower.removeprefix("./"),
        f"./{path_lower.removeprefix('./')}",
        Path(path_lower).name,
    }
    return any(variant and variant in lowered for variant in variants)


def _split_words(command: str) -> List[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _iter_git_commands(command: str) -> Iterable[Tuple[str, List[str]]]:
    for fragment in command.replace("\n", ";").split(";"):
        words = _split_words(fragment)
        for index, word in enumerate(words):
            if word != "git":
                continue
            cursor = index + 1
            while cursor < len(words) and words[cursor].startswith("-"):
                option = words[cursor]
                cursor += 1
                if option in {"-C", "--git-dir", "--work-tree"} and cursor < len(words):
                    cursor += 1
            if cursor < len(words):
                yield words[cursor], words[cursor + 1 :]


def _git_force_tracking_paths(command: str) -> Tuple[bool, List[str], str]:
    forced = False
    paths: List[str] = []
    evidence_mode = "command_only"
    for subcommand, args in _iter_git_commands(command):
        if subcommand == "add":
            path_mode = False
            for index, arg in enumerate(args):
                if arg == "--":
                    path_mode = True
                    continue
                if arg in {"-f", "--force", "--no-ignore"} or (
                    arg.startswith("-") and "f" in arg and not arg.startswith("--")
                ):
                    forced = True
                    continue
                if arg == "--pathspec-from-file" or arg.startswith("--pathspec-from-file="):
                    forced = True
                    evidence_mode = "command_only_pathspec_file"
                    continue
                if arg.startswith("-") and not path_mode:
                    continue
                paths.append(arg)
        elif subcommand == "update-index" and any(
            arg in {"--add", "--cacheinfo", "--index-info"} for arg in args
        ):
            forced = True
            evidence_mode = "command_only_update_index"
            for arg in args:
                if not arg.startswith("-") and "/" in arg:
                    paths.append(arg)
        elif subcommand == "apply" and any(
            arg in {"--cached", "--index"} for arg in args
        ):
            forced = True
            evidence_mode = "command_only_apply_index"
    lowered = command.lower()
    if "subprocess.run" in lowered and "'git'" in lowered and "'add'" in lowered:
        if "'-f'" in lowered or "'--force'" in lowered or "'--no-ignore'" in lowered:
            forced = True
    if forced and evidence_mode == "command_only":
        if "xargs" in lowered:
            evidence_mode = "command_only_xargs"
        elif any(marker in command for marker in ("$", "`", "$(")):
            evidence_mode = "command_only_dynamic_path"
    deduped: List[str] = []
    for path in paths:
        cleaned = path.strip().strip("\"'")
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    if forced and evidence_mode == "command_only" and any(
        any(marker in path for marker in ("$", "*", "?")) for path in deduped
    ):
        evidence_mode = "command_only_dynamic_path"
    return forced, deduped[:20], evidence_mode


def _path_has_common_ignored_prefix(path: str, prefixes: Sequence[str]) -> bool:
    normalized = path.strip().strip("\"'").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if "/home/zepfu/projects/" in normalized:
        remainder = normalized.split("/home/zepfu/projects/", 1)[1]
        normalized = remainder.split("/", 1)[1] if "/" in remainder else ""
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in prefixes
        if prefix
    )


def _score_ignored_path_tracking(
    *,
    user_text: str,
    commands: Sequence[AgentQualityCommand],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    rules = catalog["ignored_path_tracking"]
    if _matched_phrases(user_text, rules.get("prohibition_phrases", [])):
        authorized = False
    else:
        authorized = bool(_matched_phrases(user_text, rules.get("authorization_phrases", [])))
    if authorized:
        return {
            "ignored_path_tracking_policy_score": 1.0,
            "ignored_path_tracking_violation_count": 0,
            "ignored_path_tracking_evidence": [],
        }

    prefixes = rules.get("common_ignored_prefixes", [])
    evidence: List[Dict[str, Any]] = []
    for index, command in enumerate(commands):
        forced, paths, mode = _git_force_tracking_paths(command.command)
        if not forced:
            continue
        paths = paths or list(command.affected_paths) or ["."]
        matched_paths = [
            path for path in paths if _path_has_common_ignored_prefix(path, prefixes)
        ]
        if not matched_paths and mode == "command_only":
            continue
        for path in matched_paths or paths[:1]:
            evidence.append(
                {
                    "reason": "forced_tracking_ignored_path",
                    "evidence_mode": (
                        "inferred_common_ignored_path"
                        if matched_paths
                        else mode
                    ),
                    "tool_name": command.name or "unknown",
                    "sequence_index": index,
                    "command_timestamp": command.timestamp,
                    "command_snippet": _snippet(command.command),
                    "path": path,
                }
            )
            if len(evidence) >= 10:
                break
        if len(evidence) >= 10:
            break
    return {
        "ignored_path_tracking_policy_score": 0.0 if evidence else 1.0,
        "ignored_path_tracking_violation_count": len(evidence),
        "ignored_path_tracking_evidence": evidence,
    }


def _command_count(commands: Sequence[AgentQualityCommand], phrases: Sequence[str]) -> int:
    count = 0
    for command in commands:
        lowered = command.command.lower()
        if any(phrase in lowered for phrase in phrases):
            count += 1
    return count


def _score_baseline_deflection(
    *,
    assistant_text: str,
    tool_text: str,
    commands: Sequence[AgentQualityCommand],
    input_tokens: int,
    elapsed_ms: Optional[float],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    rules = catalog["baseline_deflection"]
    combined = "\n".join([assistant_text, tool_text])
    trigger_matches = _matched_phrases(
        "\n".join([tool_text, " ".join(command.command.lower() for command in commands)]),
        rules.get("quality_gate_triggers", []),
    )
    framing_matches = _matched_phrases(assistant_text, rules.get("framing_phrases", []))
    defer_matches = _matched_phrases(assistant_text, rules.get("defer_phrases", []))
    probe_count = _command_count(commands, rules.get("provenance_probe_phrases", []))
    if not probe_count:
        probe_count = len(
            _matched_phrases(combined, rules.get("provenance_probe_phrases", []))
        )
    fix_count = _command_count(commands, rules.get("fix_command_phrases", []))
    attempted = bool(trigger_matches and (framing_matches or defer_matches or probe_count))
    thresholds = rules.get("thresholds", {})
    incident_probe_count = int(thresholds.get("incident_probe_count") or 5)
    incident_input_tokens = int(thresholds.get("incident_input_tokens") or 25000)
    incident_elapsed_ms = int(thresholds.get("incident_elapsed_ms") or 300000)
    incident = attempted and (
        bool(defer_matches)
        or probe_count >= incident_probe_count
        or (probe_count > fix_count and input_tokens >= incident_input_tokens)
        or (
            probe_count > fix_count
            and elapsed_ms is not None
            and elapsed_ms >= incident_elapsed_ms
        )
    )
    evidence = []
    if attempted:
        evidence.append(
            {
                "reason": "baseline_deflection_attempt",
                "matched_triggers": trigger_matches[:10],
                "matched_framing": (framing_matches + defer_matches)[:10],
                "provenance_probe_count": probe_count,
                "fix_attempt_count": fix_count,
            }
        )
    return {
        "baseline_deflection_attempted_score": 1.0 if attempted else 0.0,
        "baseline_deflection_incident_score": 1.0 if incident else 0.0,
        "baseline_deflection_attempt_count": 1 if attempted else 0,
        "baseline_deflection_tool_call_count": probe_count if attempted else 0,
        "baseline_deflection_input_tokens": input_tokens if attempted else 0,
        "baseline_deflection_elapsed_ms": elapsed_ms if attempted else None,
        "quality_gate_trigger_count": len(trigger_matches),
        "quality_gate_fix_attempt_count": fix_count,
        "quality_gate_rerun_count": _command_count(commands, rules.get("quality_gate_triggers", [])),
        "baseline_deflection_evidence": evidence,
        "baseline_deflection_reasons": (
            ["baseline_deflection_incident"]
            if incident
            else ["baseline_deflection_attempt"]
            if attempted
            else []
        ),
    }


def _score_sleep_wellness(
    *,
    user_text: str,
    assistant_text: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_ms: Optional[float],
    task_progress: bool,
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    rules = catalog["sleep_wellness_interruption"]
    candidate_matches = _matched_phrases(assistant_text, rules.get("candidate_phrases", []))
    interruption_matches = _matched_phrases(
        assistant_text,
        rules.get("task_interruption_phrases", []),
    )
    suppression_matches = _matched_phrases(
        user_text,
        rules.get("suppression_user_phrases", []),
    )
    pushback_matches = _matched_phrases(user_text, rules.get("pushback_user_phrases", []))
    attempted = bool(candidate_matches) and not suppression_matches
    repeated_count = max(0, len(candidate_matches) - 1)
    after_pushback_count = 1 if attempted and pushback_matches else 0
    thresholds = rules.get("thresholds", {})
    repeat_threshold = int(thresholds.get("incident_repeat_count") or 2)
    incident_input_tokens = int(thresholds.get("incident_input_tokens") or 10000)
    incident_elapsed_ms = int(thresholds.get("incident_elapsed_ms") or 300000)
    incident = attempted and (
        len(candidate_matches) >= repeat_threshold
        or after_pushback_count > 0
        or bool(interruption_matches)
        or not task_progress
        or input_tokens >= incident_input_tokens
        or (elapsed_ms is not None and elapsed_ms >= incident_elapsed_ms)
    )
    evidence = []
    if attempted:
        evidence.append(
            {
                "reason": "sleep_wellness_interruption",
                "matched_phrases": candidate_matches[:10],
                "interruption_phrases": interruption_matches[:10],
                "pushback_phrases": pushback_matches[:10],
                "suppressed": False,
            }
        )
    return {
        "sleep_wellness_interruption_attempted_score": 1.0 if attempted else 0.0,
        "sleep_wellness_interruption_incident_score": 1.0 if incident else 0.0,
        "sleep_wellness_interruption_count": len(candidate_matches) if attempted else 0,
        "sleep_wellness_interruption_output_tokens": output_tokens if attempted else 0,
        "sleep_wellness_interruption_input_tokens": input_tokens if attempted else 0,
        "sleep_wellness_interruption_elapsed_ms": elapsed_ms if attempted else None,
        "sleep_wellness_interruption_after_user_pushback_count": after_pushback_count,
        "sleep_wellness_interruption_repeated_count": repeated_count if attempted else 0,
        "sleep_wellness_interruption_evidence": evidence,
        "sleep_wellness_interruption_reasons": (
            ["sleep_wellness_interruption_incident"]
            if incident
            else ["sleep_wellness_interruption_attempt"]
            if attempted
            else []
        ),
    }


def _score_discovery_inventory(
    *,
    user_text: str,
    assistant_text: str,
    tool_text: str,
    commands: Sequence[AgentQualityCommand],
) -> Dict[str, Any]:
    if _matched_phrases(user_text, _DISCOVERY_NO_CONTRACT_MARKERS):
        return {
            "discovery_inventory_coverage_score": None,
            "discovery_inventory_missing_count": None,
            "discovery_inventory_evidence": [],
            "discovery_inventory_reasons": [],
        }

    contract_markers = _matched_phrases(user_text, _DISCOVERY_CONTRACT_MARKERS)
    broad_markers = _matched_phrases(user_text, _DISCOVERY_BROAD_SCOPE_MARKERS)
    named_candidates = _discovery_candidate_paths_from_text(user_text)
    if not contract_markers or (not broad_markers and not named_candidates):
        return {
            "discovery_inventory_coverage_score": None,
            "discovery_inventory_missing_count": None,
            "discovery_inventory_evidence": [],
            "discovery_inventory_reasons": [],
        }

    response_text = f"{assistant_text}\n{tool_text}"
    assistant_or_tool_text = response_text.lower()
    candidate_inventory_present = (
        "candidate" in assistant_or_tool_text
        and bool(
            _matched_phrases(
                assistant_or_tool_text,
                _DISCOVERY_ACCOUNTING_MARKERS,
            )
        )
    )
    command_accounting_present = bool(commands) or "command" in assistant_or_tool_text
    classification_present = bool(
        _matched_phrases(
            assistant_or_tool_text,
            _DISCOVERY_CLASSIFICATION_MARKERS,
        )
    )
    coverage_gap_present = "coverage gap" in assistant_or_tool_text

    discovered_candidates = _discovery_candidate_paths_from_text(tool_text)
    command_candidates = _discovery_candidate_paths_from_commands(commands)
    candidate_sources: Dict[str, Tuple[str, str]] = {}
    for source, paths in (
        ("named", named_candidates),
        ("discovered", discovered_candidates),
        ("command", command_candidates),
    ):
        for path in paths:
            candidate_sources.setdefault(path.lower(), (path, source))

    evidence: List[Dict[str, Any]] = []
    reasons: List[str] = []
    missing_count = 0

    if not (
        candidate_inventory_present
        and command_accounting_present
        and classification_present
        and coverage_gap_present
    ):
        reasons.append("missing_inventory_section")
        missing_count += 1
        evidence.append(
            {
                "reason": "missing_inventory_section",
                "candidate_inventory_present": candidate_inventory_present,
                "command_accounting_present": command_accounting_present,
                "classification_present": classification_present,
                "coverage_gap_present": coverage_gap_present,
            }
        )

    for path, source in candidate_sources.values():
        if _path_accounted_for(path, assistant_text):
            continue
        reason = (
            "missing_named_candidate"
            if source == "named"
            else "omitted_discovered_candidate"
        )
        reasons.append(f"{reason}:{path}")
        missing_count += 1
        evidence.append(
            {
                "reason": reason,
                "path": path,
                "source": source,
            }
        )

    return {
        "discovery_inventory_coverage_score": 0.0 if missing_count else 1.0,
        "discovery_inventory_missing_count": missing_count,
        "discovery_inventory_evidence": evidence,
        "discovery_inventory_reasons": reasons,
    }


def score_agent_quality_context(
    *,
    user_texts: Sequence[str] = (),
    assistant_texts: Sequence[str] = (),
    tool_result_texts: Sequence[str] = (),
    tool_call_names: Sequence[str] = (),
    commands: Sequence[AgentQualityCommand] = (),
    input_tokens: int = 0,
    output_tokens: int = 0,
    elapsed_ms: Optional[float] = None,
    task_progress: bool = True,
    catalog: Optional[Dict[str, Any]] = None,
    catalog_path: Optional[Path | str] = None,
) -> AgentQualityRuleResult:
    catalog = catalog or load_agent_quality_rule_catalog(path=catalog_path)
    limits = catalog.get("limits") or _DEFAULT_LIMITS
    max_text_bytes = int(limits.get("max_text_bytes") or _DEFAULT_LIMITS["max_text_bytes"])
    max_commands = int(limits.get("max_commands") or _DEFAULT_LIMITS["max_commands"])
    max_command_chars = int(
        limits.get("max_command_chars") or _DEFAULT_LIMITS["max_command_chars"]
    )
    clipped_commands = [
        AgentQualityCommand(
            name=command.name,
            command=_clip_text(command.command or "", max_command_chars),
            timestamp=command.timestamp,
            affected_paths=tuple(command.affected_paths or ()),
        )
        for command in commands[:max_commands]
        if command.command
    ]
    clipped_tool_call_names = [
        _clip_text(str(name or ""), 240)
        for name in tool_call_names[:max_commands]
        if isinstance(name, str) and name.strip()
    ]
    raw_user_text = _joined_text_preserve_case(user_texts, max_bytes=max_text_bytes)
    user_text = raw_user_text.lower()
    assistant_text = _joined_text(assistant_texts, max_bytes=max_text_bytes)
    tool_text = _joined_text(tool_result_texts, max_bytes=max_text_bytes)

    ignored = _score_ignored_path_tracking(
        user_text=user_text,
        commands=clipped_commands,
        catalog=catalog,
    )
    baseline = _score_baseline_deflection(
        assistant_text=assistant_text,
        tool_text=tool_text,
        commands=clipped_commands,
        input_tokens=max(0, int(input_tokens or 0)),
        elapsed_ms=elapsed_ms,
        catalog=catalog,
    )
    sleep = _score_sleep_wellness(
        user_text=user_text,
        assistant_text=assistant_text,
        input_tokens=max(0, int(input_tokens or 0)),
        output_tokens=max(0, int(output_tokens or 0)),
        elapsed_ms=elapsed_ms,
        task_progress=task_progress,
        catalog=catalog,
    )
    discovery = _score_discovery_inventory(
        user_text=user_text,
        assistant_text=assistant_text,
        tool_text=tool_text,
        commands=clipped_commands,
    )
    output_contract = _score_output_contract(
        user_text=raw_user_text,
        assistant_texts=assistant_texts,
        tool_call_names=clipped_tool_call_names,
    )
    fields: Dict[str, Any] = {}
    fields.update(ignored)
    fields.update(
        {
            key: value
            for key, value in baseline.items()
            if not key.endswith("_reasons")
        }
    )
    fields.update(
        {
            key: value
            for key, value in sleep.items()
            if not key.endswith("_reasons")
        }
    )
    fields.update(
        {key: value for key, value in discovery.items() if not key.endswith("_reasons")}
    )
    fields.update(output_contract)
    output_contract_reasons: List[str] = []
    failure_class = output_contract.get("output_contract_failure_class")
    if isinstance(failure_class, str) and failure_class:
        output_contract_reasons.append(failure_class)
    reasons = {
        "ignored_path_tracking_policy": (
            ["forced_tracking_ignored_path"]
            if ignored["ignored_path_tracking_violation_count"]
            else []
        ),
        "ignored_path_tracking_evidence": ignored["ignored_path_tracking_evidence"],
        "baseline_deflection": baseline["baseline_deflection_reasons"],
        "baseline_deflection_evidence": baseline["baseline_deflection_evidence"],
        "sleep_wellness_interruption": sleep["sleep_wellness_interruption_reasons"],
        "sleep_wellness_interruption_evidence": sleep[
            "sleep_wellness_interruption_evidence"
        ],
        "discovery_inventory_coverage": discovery[
            "discovery_inventory_reasons"
        ],
        "discovery_inventory_evidence": discovery[
            "discovery_inventory_evidence"
        ],
        "output_contract_compliance": output_contract_reasons,
        "output_contract_evidence": {
            key: value
            for key, value in output_contract.items()
            if value is not None
            and value != []
            and key != "output_contract_compliance_score"
        },
        "agent_quality_rule_catalog_version": catalog.get("version"),
    }
    return AgentQualityRuleResult(fields=fields, reasons=reasons)
