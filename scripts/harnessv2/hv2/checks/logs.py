"""Docker log forbidden-string and leftover-uvicorn checks. Patterns live in YAML."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from hv2.load_config import as_str_list

_UVICORN_ACCESS_PATH = re.compile(
    r'"(?:GET|POST|PUT|PATCH|DELETE|HEAD)\s+(\S+)',
    re.IGNORECASE,
)

# Ohmypi rollup identity: <repository>#Ohmypi[<version>]@<host>, e.g.
# litellm#Ohmypi[17.3.8]@thoth. Bare client labels like Bun[...]* or Oh@host
# lack the repository/name/version triple and are rejected for Ohmypi runs.
_OHMYPI_ROLLUP_IDENTITY = re.compile(
    r"\S+#Ohmypi\[[^\]\s]+\]@\S+",
)
# Repo token is the \S+ immediately before @ after the rollup timestamp.
_ROLLUP_REPO_BEFORE_AT = re.compile(
    r"^\d{8} \d{2}:\d{2}:\d{2}(?: \[EARLY\])? (\S+)@"
)


def _logs_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    logs = checks.get("logs") if isinstance(checks.get("logs"), dict) else {}
    return logs


def _leftover_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    leftover = _logs_spec(config).get("leftover_uvicorn")
    return leftover if isinstance(leftover, dict) else {}


def leftover_uvicorn_regex(config: Mapping[str, Any]) -> re.Pattern[str] | None:
    leftover = _leftover_spec(config)
    if leftover.get("enabled") is False:
        return None
    template = str(leftover.get("regex") or "")
    if not template:
        return None
    if "{paths}" in template:
        paths = as_str_list(leftover.get("replaced_route_paths"))
        if not paths:
            return None
        template = template.format(paths="|".join(re.escape(path) for path in paths))
    try:
        return re.compile(template)
    except re.error:
        return None


def leftover_uvicorn_allow_paths(config: Mapping[str, Any]) -> set[str]:
    leftover = _leftover_spec(config)
    return {item for item in as_str_list(leftover.get("allow_paths")) if item}


def _is_concurrent_workspace_rollup(hit: str, repos: set[str]) -> bool:
    """True for <repo>@host /path headers from other AAWM workspaces, not litellm."""
    if not repos:
        return False
    match = _ROLLUP_REPO_BEFORE_AT.search(hit)
    if match is None:
        return False
    repo = match.group(1)
    return repo != "litellm" and repo in repos


def _line_at(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end]


def _uvicorn_access_path(line: str) -> str:
    match = _UVICORN_ACCESS_PATH.search(line)
    if match is None:
        return ""
    return match.group(1).split("?", 1)[0]


def scan_log_text(
    text: str,
    config: Mapping[str, Any],
    *,
    attribution_substrings: Sequence[str] | None = None,
    require_rollup: bool = False,
    plan_models: Sequence[str] | None = None,
    tui: str | None = None,
) -> dict[str, Any]:
    logs = _logs_spec(config)
    forbidden = as_str_list(logs.get("forbidden_substrings"))
    traceback_needles = as_str_list(logs.get("traceback_substrings"))
    hits: list[dict[str, str]] = []
    warnings: list[str] = []
    failures: list[str] = []

    for needle in forbidden:
        if not needle or needle not in text:
            continue
        unexplained = False
        for start in _all_indexes(text, needle):
            local = _window_at(text, start, len(needle), _local_chars(config, needle))
            if _is_unrelated_local(
                needle, text, local, config, attribution_substrings or ()
            ):
                warnings.append(f"ignored unrelated log match: {needle}")
                continue
            expected = _expected_signature_at(needle, local, config)
            if expected:
                warnings.append(f"expected log signature ({expected}): {needle}")
                continue
            unexplained = True
        if unexplained:
            hits.append({"kind": "forbidden", "substring": needle})
            failures.append(f"runtime logs contained forbidden substring: {needle}")

    leftover_re = leftover_uvicorn_regex(config)
    if leftover_re is not None:
        allow_paths = leftover_uvicorn_allow_paths(config)
        for match in leftover_re.finditer(text):
            line = _line_at(text, match.start(), match.end())
            if _uvicorn_access_path(line) in allow_paths:
                continue
            hits.append({"kind": "leftover_uvicorn", "substring": line})
            failures.append(f"leftover uvicorn access line: {line}")

    rollup = logs.get("rollup") if isinstance(logs.get("rollup"), dict) else {}
    header_regex = str(rollup.get("header_regex") or "")
    rollup_hits: list[str] = []
    if header_regex:
        try:
            compiled = re.compile(header_regex, re.MULTILINE)
            rollup_hits = compiled.findall(text) or [
                m.group(0) for m in compiled.finditer(text)
            ]
        except re.error:
            failures.append("invalid rollup.header_regex in checks.yaml")
    if require_rollup and header_regex and not rollup_hits:
        failures.append("expected AAWM route-rollup header was not found in docker logs")
    if require_rollup and rollup_hits and tui == "ohmypi":
        concurrent_repos = {
            repo
            for repo in as_str_list(rollup.get("concurrent_workspace_repos"))
            if repo and repo != "litellm"
        }
        has_ohmypi_identity = any(
            _OHMYPI_ROLLUP_IDENTITY.search(hit) for hit in rollup_hits
        )
        unidentified = [
            hit
            for hit in rollup_hits
            if not _OHMYPI_ROLLUP_IDENTITY.search(hit)
            and not _is_concurrent_workspace_rollup(hit, concurrent_repos)
        ]
        if not has_ohmypi_identity or unidentified:
            suffix = f": {'; '.join(unidentified)}" if unidentified else ""
            failures.append(
                "Ohmypi rollup headers lacked a client identity "
                "(repository#Ohmypi[version]@host)" + suffix
            )

    traceback_hits = [needle for needle in traceback_needles if needle and needle in text]
    _apply_model_regressions(text, config, plan_models, hits, failures)
    return {
        "forbidden_hits": hits,
        "warnings": warnings,
        "failures": failures,
        "rollup_hits": rollup_hits,
        "traceback_hits": traceback_hits,
        "ok": not failures,
    }


def _apply_model_regressions(
    text: str,
    config: Mapping[str, Any],
    plan_models: Sequence[str] | None,
    hits: list[dict[str, str]],
    failures: list[str],
) -> None:
    logs = _logs_spec(config)
    rows = logs.get("model_regressions")
    if not isinstance(rows, list):
        return
    plan_set = {item for item in as_str_list(plan_models) if item}
    if not plan_set:
        return
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        needle = str(row.get("substring") or "")
        if not needle or needle not in text:
            continue
        models = as_str_list(row.get("models"))
        if not plan_set.intersection(models):
            continue
        hits.append({"kind": "model_regression", "substring": needle})
        label = str(row.get("name") or f"model_regressions[{index}]")
        scoped = ",".join(models) if models else "any"
        failures.append(
            f"runtime logs contained model regression ({label}): {needle} [{scoped}]"
        )


def _all_indexes(text: str, substring: str) -> list[int]:
    indexes: list[int] = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx < 0:
            return indexes
        indexes.append(idx)
        start = idx + max(len(substring), 1)


def _window_at(text: str, start: int, length: int, chars: int) -> str:
    begin = max(0, start - chars)
    end = min(len(text), start + length + chars)
    return text[begin:end]


def _local_chars(config: Mapping[str, Any], substring: str) -> int:
    logs = _logs_spec(config)
    unrelated = logs.get("unrelated") if isinstance(logs.get("unrelated"), dict) else {}
    chars = int(unrelated.get("local_context_chars") or 800)
    rows = logs.get("expected_signatures")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("substring") or "") == substring:
                chars = max(chars, int(row.get("local_context_chars") or chars))
    return chars


def _expected_signature_at(
    substring: str,
    local: str,
    config: Mapping[str, Any],
) -> str | None:
    logs = _logs_spec(config)
    rows = logs.get("expected_signatures")
    if not isinstance(rows, list):
        return None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        needle = str(row.get("substring") or "")
        if needle != substring:
            continue
        nearby = as_str_list(row.get("nearby"))
        if nearby and not any(token in local for token in nearby):
            continue
        return str(row.get("name") or f"expected_signatures[{index}]")
    return None


def _is_unrelated_local(
    substring: str,
    text: str,
    local: str,
    config: Mapping[str, Any],
    attribution_substrings: Sequence[str],
) -> bool:
    logs = _logs_spec(config)
    scoped = set(as_str_list(logs.get("attribution_scoped_substrings")))
    if substring not in scoped:
        return False
    if not attribution_substrings:
        return False
    if any(value in text for value in attribution_substrings):
        return False
    unrelated = logs.get("unrelated") if isinstance(logs.get("unrelated"), dict) else {}
    signatures = as_str_list(unrelated.get("error_signatures"))
    if not any(sig in local for sig in signatures):
        return False
    has_auto = any(m in text for m in as_str_list(unrelated.get("auto_agent_markers")))
    has_pass = any(m in text for m in as_str_list(unrelated.get("passthrough_markers")))
    upstream = substring.startswith("pass_through_endpoint(): Exception occured")
    if upstream:
        return bool(has_pass or has_auto)
    if has_auto:
        return True
    return bool(has_pass)
