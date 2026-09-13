"""Orchestration child-spawn evidence. Recap-only is not a pass."""

from __future__ import annotations

import base64
import heapq
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hv2.pane import _pane_scan_start


_UNKNOWN_AGENT_MARKERS = (
    "Unknown agent",
    "failed preflight",
    "unavailable. The spawn tool reported available agents",
)
_TASK_RESULT_AGENT = re.compile(
    r'<task-result\b[^>]*\bagent="([^"]+)"[^>]*\bstatus="completed"',
    re.IGNORECASE,
)
_PANE_CHILD_DATE = re.compile(
    r"(?m)^[ \t]*-[ \t]+([\w-]+)\n[ \t]+-[ \t]+date:",
)
_HUB_IDLE_PEER = re.compile(
    r"([\w-]+)Date \[([\w-]+) · sub · idle\]",
)
_COMPLETION_TOOLS = {"task", "hub", "bash", "yield"}
_JSONL_SCAN_CAP = 64
_JSONL_MAX_BYTES = 2 * 1024 * 1024
_JSONL_MAX_LINES = 20000

class _BoundedJSONLRead:
    def __init__(self, records: list[dict[str, Any]], truncated: bool) -> None:
        self.records = records
        self.truncated = truncated

    def __iter__(self) -> Iterable[dict[str, Any]]:
        return iter(self.records)
_OPERATIONAL_FALLBACK_ALIASES = frozenset(
    {
        "basic",
        "work",
        "work-other",
        "expert",
        "sota",
        "sota-openai",
        "sota-xai",
        "sota-alibaba",
        "sota-moonshot",
        "sota-deepseek",
        "sota-zai",
        "auto-review",
        "codex-auto-review",
    }
)


def _session_jsonl_paths(session_dir: Path, *, since_mtime: float | None) -> list[Path]:
    if not session_dir.is_dir():
        return []
    rows: list[Path] = []
    mtimes: dict[Path, float] = {}
    for path in (*session_dir.glob("*.jsonl"), *session_dir.glob("*/*.jsonl")):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since_mtime is not None and mtime < (since_mtime - 2):
            continue
        rows.append(path)
        mtimes[path] = mtime
    # Capture mtimes before sorting so disappearing files cannot raise here.
    rows.sort(key=lambda item: (mtimes[item], str(item)), reverse=True)
    return rows[:_JSONL_SCAN_CAP]


def _iter_jsonl_objects(path: Path) -> Iterable[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    truncated = False
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return _BoundedJSONLRead(records, truncated)
    with handle:
        consumed = 0
        for index, line in enumerate(handle):
            if index >= _JSONL_MAX_LINES:
                truncated = True
                break
            line_bytes = len(line.encode("utf-8", errors="replace"))
            if line_bytes > _JSONL_MAX_BYTES or consumed + line_bytes > _JSONL_MAX_BYTES:
                truncated = True
                break
            consumed += line_bytes
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return _BoundedJSONLRead(records, truncated)


def _message_payload(obj: Mapping[str, Any]) -> dict[str, Any]:
    message = obj.get("message")
    if isinstance(message, dict):
        return message
    return dict(obj)


def _content_text(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
    return "\n".join(parts)


def _task_result_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    details = payload.get("details")
    if not isinstance(details, dict):
        return []
    results = details.get("results")
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, dict)]


def _agent_from_result(row: Mapping[str, Any]) -> str:
    for key in ("agent", "agentName", "name", "displayName"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    resolved = row.get("resolvedModel")
    if isinstance(resolved, str) and "/" in resolved:
        return resolved.rsplit("/", 1)[-1].strip()
    blob = str(row.get("resultText") or "")
    match = _TASK_RESULT_AGENT.search(blob)
    if match:
        return match.group(1).strip()
    return ""


def _alias_tail(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    if "/" in text:
        return text.rsplit("/", 1)[-1].strip()
    return text


def _provider_id_from_alias(alias: str) -> str:
    if alias.startswith("provider-"):
        return alias[len("provider-") :]
    return ""


def _wanted_from_model(value: Any, wanted: Sequence[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    if text in wanted:
        return text
    if "/" in text:
        tail = text.rsplit("/", 1)[-1].strip()
        if tail in wanted:
            return tail
    return ""


def _is_alias_identity_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return True
    if "model_id:none" in lowered.replace(" ", ""):
        return True
    return text.startswith("provider-")


def _coerce_route_identity(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    provider = str(value.get("producer_provider") or "").strip()
    model = str(value.get("producer_model") or "").strip()
    route = str(value.get("producer_route_family") or "").strip()
    if not provider or not model or not route:
        return None
    if _is_alias_identity_value(provider) or _is_alias_identity_value(model):
        return None
    return {
        "producer_provider": provider,
        "producer_model": model,
        "producer_route_family": route,
    }


def _walk_named_identity(obj: Any, field: str) -> dict[str, str] | None:
    if isinstance(obj, Mapping):
        if field in obj:
            found = _coerce_route_identity(obj.get(field))
            if found:
                return found
        for nested in obj.values():
            found = _walk_named_identity(nested, field)
            if found:
                return found
    elif isinstance(obj, list):
        for nested in obj:
            found = _walk_named_identity(nested, field)
            if found:
                return found
    return None


def extract_child_route_identity(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, str] | None:
    """Read observed producer identity from nested Ohmypi records.

    Last valid record wins across the append-only transcript. Within one
    record, ``aawm_route_identity`` is preferred over
    ``aawm_encrypted_reasoning_provenance``. Alias prefix, ``model_id:None``,
    and ``provider-*`` as model are never a pass.
    """

    latest: dict[str, str] | None = None
    for obj in records:
        if not isinstance(obj, Mapping):
            continue
        stamp = _walk_named_identity(obj, "aawm_route_identity")
        provenance = _walk_named_identity(
            obj, "aawm_encrypted_reasoning_provenance"
        )
        found = stamp or provenance
        if found:
            latest = found
    return latest


def _result_looks_successful(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    if status in {"error", "failed", "fail", "preflight", "blocked", "running"}:
        return False
    if row.get("isError") is True:
        return False
    text = json.dumps(row, default=str)
    if any(marker in text for marker in _UNKNOWN_AGENT_MARKERS):
        return False
    return True


def _hub_job_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    details = payload.get("details")
    if not isinstance(details, dict):
        return []
    jobs = details.get("jobs")
    if not isinstance(jobs, list):
        return []
    return [row for row in jobs if isinstance(row, dict)]


def _agents_from_text(text: str, wanted: Sequence[str]) -> set[str]:
    found: set[str] = set()
    wanted_set = set(wanted)
    for match in _TASK_RESULT_AGENT.finditer(text):
        agent = match.group(1).strip()
        if agent in wanted_set:
            found.add(agent)
    for match in _HUB_IDLE_PEER.finditer(text):
        agent = match.group(2).strip()
        if agent in wanted_set:
            found.add(agent)
    return found


def _collect_successful_agents(
    payload: Mapping[str, Any],
    text: str,
    wanted: Sequence[str],
) -> set[str]:
    found = _agents_from_text(text, wanted)
    wanted_set = set(wanted)
    for row in (*_task_result_rows(payload), *_hub_job_rows(payload)):
        agent = _agent_from_result(row)
        if agent in wanted_set and _result_looks_successful(row):
            found.add(agent)
    return found


def _nested_child_route_row(path: Path, wanted: Sequence[str]) -> dict[str, Any] | None:
    """Classify one nested Ohmypi child transcript.

    A completed PONG/date/fanout after Ohmypi auto-retried onto an
    operational alias (``sota``, ``basic``, …) is a provider-specific
    failure, not a pass. In-alias model fallback is allowed. Provider
    aliases require observed ``producer_provider`` / ``producer_model`` /
    ``producer_route_family`` fields; the alias prefix is not a route.
    """

    wanted_set = set(wanted)
    requested = ""
    selected_models: list[str] = []
    records: list[Mapping[str, Any]] = []
    tools: set[str] = set()
    fallback = False
    error = ""
    completed = False
    bounded_read = _iter_jsonl_objects(path)
    for obj in bounded_read:
        records.append(obj)
        payload = _message_payload(obj)
        obj_type = str(obj.get("type") or "")
        tool_name = str(payload.get("toolName") or obj.get("toolName") or "")
        text = _content_text(payload)
        if obj_type in {"session_init", "model_change"} or payload.get("agent"):
            candidate = _wanted_from_model(
                payload.get("agent") or obj.get("agent") or "",
                wanted,
            ) or _wanted_from_model(
                payload.get("resolvedModel")
                or obj.get("resolvedModel")
                or obj.get("model")
                or payload.get("model")
                or "",
                wanted,
            )
            if candidate and not requested:
                requested = candidate
            tail = _alias_tail(
                obj.get("model") or payload.get("model") or obj.get("resolvedModel")
            )
            if tail:
                selected_models.append(tail)
            if obj.get("role") == "fallback" or obj.get("resolvedModelIsFallback") is True:
                fallback = True
        if tool_name:
            tools.add(tool_name)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        if data.get("toolName"):
            tools.add(str(data.get("toolName")))
        err = payload.get("errorMessage")
        if isinstance(err, str) and err.strip() and not error:
            error = err.strip()
        # Nested bash while the parent is still waiting is not a completed
        # child. Count the nested transcript only after a successful yield
        # or a completed task-result.
        if (
            tool_name == "yield"
            and str(payload.get("role") or "") == "toolResult"
            and payload.get("isError") is not True
        ) or _TASK_RESULT_AGENT.search(text):
            completed = True
    if not requested and path.stem in wanted_set:
        requested = path.stem
    if not requested:
        return None
    unique_models = list(dict.fromkeys(selected_models))
    escaped = [
        name
        for name in unique_models
        if name in _OPERATIONAL_FALLBACK_ALIASES and name != requested
    ]
    expected_provider = _provider_id_from_alias(requested)
    identity = extract_child_route_identity(records)
    selected_provider = identity["producer_provider"] if identity else ""
    observed_model = identity["producer_model"] if identity else ""
    observed_route = identity["producer_route_family"] if identity else ""
    observed_identity = identity is not None
    requires_observed_identity = bool(expected_provider)
    cross_provider = bool(
        expected_provider
        and selected_provider
        and selected_provider != expected_provider
    )
    disposition = "completed"
    if escaped or (fallback and escaped) or cross_provider:
        disposition = "fallback_operational"
    elif error:
        disposition = "unavailable"
    elif not completed:
        disposition = "incomplete"
    elif requires_observed_identity and not observed_identity:
        disposition = "identity_unobserved"
    ok = (
        completed
        and not escaped
        and not cross_provider
        and not error
        and (observed_identity if requires_observed_identity else True)
    )
    if bounded_read.truncated:
        ok = False
        error = "child transcript exceeded bounded evidence read"
    return {
        "requested_alias": requested,
        "selected_provider": selected_provider,
        "model": observed_model,
        "route_family": observed_route,
        "endpoint_family": "openai_passthrough/v1/responses",
        "terminal_disposition": disposition,
        "fallback": fallback,
        "escaped_aliases": escaped,
        "tools": sorted(tools),
        "error": error,
        "ok": ok,
    }


def _record_looks_like_completion(payload: Mapping[str, Any], text: str) -> bool:
    tool_name = str(payload.get("toolName") or "")
    role = str(payload.get("role") or "")
    custom_type = str(payload.get("customType") or "")
    if _TASK_RESULT_AGENT.search(text) or _HUB_IDLE_PEER.search(text):
        return True
    if role == "toolResult" and (
        tool_name in _COMPLETION_TOOLS or _hub_job_rows(payload) or _task_result_rows(payload)
    ):
        return True
    if custom_type == "async-result":
        return True
    return False


def child_spawn_evidence(
    *,
    children: Sequence[str],
    pane: str = "",
    session_dir: str | None = None,
    since_mtime: float | None = None,
    prompt: str | None = None,
    after_echo_index: int | None = None,
) -> dict[str, Any]:
    """Return whether Ohmypi actually spawned the requested child profiles.

    A parent recap (`※ recap:`) is not evidence. Preflight `Unknown agent`
    and empty Ohmypi `task` / `hub` job completions are failures. A
    `Spawned N background agents using …` line is spawn intent, not a
    completed child result.
    """

    wanted = [str(item) for item in children if str(item).strip()]
    wanted_set = set(wanted)
    failures: list[str] = []
    pane_text = pane or ""
    if prompt is not None and after_echo_index is not None:
        scan_start = _pane_scan_start(pane_text, prompt, after_echo_index=after_echo_index)
        pane_lines = pane_text.splitlines()[scan_start:]
        pane_text = "\n".join(pane_lines)
    combined = pane_text
    session_paths: list[str] = []
    successful_agents: set[str] = set()
    failed_agents: set[str] = set()
    routes: dict[str, dict[str, Any]] = {}
    unknown_agents: set[str] = set()
    saw_task_result = False
    project_agents_dir: str | None = None

    successful_agents.update(_agents_from_text(pane_text, wanted))
    successful_agents.update(
        name for name in _PANE_CHILD_DATE.findall(pane_text) if name in wanted_set
    )

    root = Path(session_dir) if session_dir else None
    jsonl_files = _session_jsonl_paths(root, since_mtime=since_mtime) if root else []
    for path in jsonl_files[:_JSONL_SCAN_CAP]:
        session_paths.append(str(path))
        file_agent: str | None = None
        file_completed = False
        bounded_read = _iter_jsonl_objects(path)
        for obj in bounded_read:
            payload = _message_payload(obj)
            tool_name = str(payload.get("toolName") or obj.get("toolName") or "")
            text = _content_text(payload)
            combined = f"{combined}\n{text}"
            obj_type = str(obj.get("type") or "")
            custom_type = str(obj.get("customType") or payload.get("customType") or "")
            if obj_type in {"session_init", "model_change"} or payload.get("agent"):
                candidate = _wanted_from_model(
                    payload.get("agent") or obj.get("agent") or "",
                    wanted,
                ) or _wanted_from_model(
                    payload.get("resolvedModel")
                    or obj.get("resolvedModel")
                    or obj.get("model")
                    or payload.get("model")
                    or "",
                    wanted,
                )
                if candidate:
                    file_agent = candidate
            if _record_looks_like_completion(payload, text) or custom_type == "async-result":
                saw_task_result = True
            details = payload.get("details")
            if isinstance(details, dict):
                raw_dir = details.get("projectAgentsDir")
                if isinstance(raw_dir, str) and raw_dir.strip():
                    project_agents_dir = raw_dir
            found = _collect_successful_agents(payload, text, wanted)
            successful_agents.update(found)
            # Nested bash `date` while the parent is still waiting is not a
            # completed child result. Count the nested transcript only after
            # a successful `yield` (or a `<task-result status="completed">`).
            if found or (
                tool_name == "yield"
                and str(payload.get("role") or "") == "toolResult"
                and payload.get("isError") is not True
            ):
                file_completed = True
        if bounded_read.truncated:
            failures.append(
                f"Ohmypi transcript exceeded bounded evidence read: {path.name}"
            )
        if root is not None and path.parent != root:
            nested_route = _nested_child_route_row(path, wanted)
            if nested_route:
                alias = str(nested_route.get("requested_alias") or file_agent or "")
                if alias in wanted_set:
                    routes[alias] = nested_route
                    if nested_route.get("ok") is True:
                        successful_agents.add(alias)
                    elif nested_route.get("terminal_disposition") in {
                        "fallback_operational",
                        "identity_unobserved",
                        "unavailable",
                    }:
                        failed_agents.add(alias)
                        successful_agents.discard(alias)
                    continue
        if file_agent and file_completed and file_agent not in failed_agents:
            successful_agents.add(file_agent)

    for child in wanted:
        if f'Unknown agent "{child}"' in combined or f"Unknown agent '{child}'" in combined:
            unknown_agents.add(child)

    unresolved_unknown = [child for child in wanted if child in unknown_agents]
    if unresolved_unknown and not set(wanted).issubset(successful_agents):
        failures.append(
            "orchestration spawn preflight rejected child agents "
            f"(unknown={sorted(unknown_agents) or 'see pane'})"
        )
    for alias, row in routes.items():
        if row.get("ok") is True:
            successful_agents.add(alias)
            failed_agents.discard(alias)
        elif row.get("terminal_disposition") in {
            "fallback_operational",
            "identity_unobserved",
            "unavailable",
        }:
            failed_agents.add(alias)
            successful_agents.discard(alias)
    escaped = [
        child
        for child, row in routes.items()
        if child in wanted_set and row.get("terminal_disposition") == "fallback_operational"
    ]
    if escaped:
        failures.append(
            "orchestration child fell back onto an operational alias or other "
            f"provider (not in-alias fallback): {escaped}"
        )
    unobserved = [
        child
        for child, row in routes.items()
        if child in wanted_set and row.get("terminal_disposition") == "identity_unobserved"
    ]
    if unobserved:
        failures.append(
            "orchestration child completed without observed producer identity "
            "(selected_provider, model, and route_family must come from "
            f"transcript producer fields, not the alias prefix): {unobserved}"
        )
    unavailable = [
        child
        for child, row in routes.items()
        if child in wanted_set and row.get("terminal_disposition") == "unavailable"
    ]
    if unavailable:
        failures.append(
            "orchestration child failed with an in-alias provider error "
            f"(not a pass): {unavailable}"
        )
    missing = [child for child in wanted if child not in successful_agents]
    if missing:
        failures.append(
            "orchestration session is missing successful Ohmypi task results "
            f"for {missing}; recap-only is not child-spawn evidence"
        )
    if wanted and not saw_task_result and not successful_agents:
        failures.append("orchestration session has no Ohmypi `task` tool result")

    return {
        "ok": not failures,
        "failures": failures,
        "children": wanted,
        "successful_agents": sorted(successful_agents),
        "failed_agents": sorted(failed_agents),
        "unknown_agents": sorted(unknown_agents),
        "routes": routes,
        "session_jsonl": session_paths[:4],
        "project_agents_dir": project_agents_dir,
        "saw_task_result": saw_task_result,
    }


_GROK_SPAWN_CHROME_RE = re.compile(
    r"(?:◆\s*)?(?:Subagent|spawn_subagent)\b",
    re.IGNORECASE,
)
_GROK_RUN_ROW_RE = re.compile(r"◆\s+Run\b")
_GROK_PWD_ROW_RE = re.compile(
    r"◆\s+Run\b.*\bpwd\b",
    re.IGNORECASE,
)
_GROK_UNAME_ROW_RE = re.compile(
    r"◆\s+Run\b.*\buname\b",
    re.IGNORECASE,
)
_GROK_TOOL_LABEL_DUMP_RE = re.compile(r"(?im)^Tool label:\s*\S")
_GROK_SESSION_UPDATE_KEYS = frozenset(
    {"sessionUpdate", "session_update", "type", "customType"}
)


def _grok_current_turn_text(
    pane: str,
    prompt: str | None,
    after_echo_index: int | None,
) -> str:
    pane_text = pane or ""
    if prompt is not None:
        scan_start = _pane_scan_start(
            pane_text, prompt, after_echo_index=after_echo_index
        )
        pane_text = "\n".join(pane_text.splitlines()[scan_start:])
    return pane_text


def grok_workspace_session_root(cwd: str | None = None) -> Path:
    """Return ``~/.grok/sessions/<urlencoded cwd>`` for Grok Build JSONL."""

    workspace = Path(cwd or "/tmp/hv2-grok-workspace")
    encoded = "".join(
        ch if ch.isalnum() or ch in "._-" else f"%{ord(ch):02X}"
        for ch in str(workspace)
    )
    return Path.home() / ".grok" / "sessions" / encoded


def _grok_workspace_session_roots(session_dir: str | None) -> list[Path]:
    """Scan only the provided session directory and its child session dirs."""

    roots: list[Path] = []
    if not session_dir:
        return roots
    root = Path(session_dir)
    roots.append(root)
    if root.is_dir():
        newest = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        roots.extend(newest[:8])
    return roots


def _grok_session_tool_records(
    session_dir: str | None,
    *,
    since_mtime: float | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    truncated = False
    seen: set[str] = set()
    for root in _grok_workspace_session_roots(session_dir):
        for path in _session_jsonl_paths(root, since_mtime=since_mtime)[
            :_JSONL_SCAN_CAP
        ]:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            bounded_read = _iter_jsonl_objects(path)
            for obj in bounded_read:
                records.append(obj)
            truncated = truncated or bounded_read.truncated
    return records, truncated


def _grok_nested_update(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    params = obj.get("params")
    if isinstance(params, Mapping):
        update = params.get("update")
        if isinstance(update, Mapping):
            return update
    update = obj.get("update")
    if isinstance(update, Mapping):
        return update
    return {}


def _grok_xai_tool_name(source: Mapping[str, Any]) -> str:
    meta = source.get("_meta")
    if not isinstance(meta, Mapping):
        return ""
    xai_tool = meta.get("x.ai/tool")
    if not isinstance(xai_tool, Mapping):
        return ""
    name = xai_tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _grok_record_tool_name(obj: Mapping[str, Any]) -> str:
    """Prefer structured xAI tool name over TUI display titles."""

    update = _grok_nested_update(obj)
    for source in (update, obj):
        named = _grok_xai_tool_name(source)
        if named:
            return named
        for key in ("tool_name", "name", "toolName"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for source in (update, obj):
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    calls = obj.get("tool_calls")
    if isinstance(calls, list):
        for item in calls:
            if isinstance(item, Mapping):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return ""


def _grok_command_from_mapping(source: Mapping[str, Any]) -> str:
    for key in ("rawInput", "rawOutput"):
        raw = source.get(key)
        if not isinstance(raw, Mapping):
            continue
        command = raw.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()
        nested = raw.get("input")
        if isinstance(nested, Mapping):
            command = nested.get("command")
            if isinstance(command, str) and command.strip():
                return command.strip()
    meta = source.get("_meta")
    if isinstance(meta, Mapping):
        xai_tool = meta.get("x.ai/tool")
        if isinstance(xai_tool, Mapping):
            nested = xai_tool.get("input")
            if isinstance(nested, Mapping):
                command = nested.get("command")
                if isinstance(command, str) and command.strip():
                    return command.strip()
    return ""


def _grok_record_command(obj: Mapping[str, Any]) -> str:
    update = _grok_nested_update(obj)
    for source in (update, obj):
        command = _grok_command_from_mapping(source)
        if command:
            return command
    return ""


def _grok_record_is_tool_call(obj: Mapping[str, Any]) -> bool:
    update = _grok_nested_update(obj)
    session_update = str(update.get("sessionUpdate") or obj.get("sessionUpdate") or "")
    obj_type = str(obj.get("type") or "")
    if session_update in {
        "tool_call",
        "tool_call_update",
        "subagent_spawned",
        "subagent_finished",
    }:
        return True
    if obj_type in {"tool_started", "tool_completed"}:
        return True
    if obj.get("tool_calls"):
        return True
    name = _grok_record_tool_name(obj).lower()
    return name in {
        "spawn_subagent",
        "run_terminal_command",
        "pwd",
        "uname",
        "get_command_or_subagent_output",
    }


def grok_spawn_tool_evidence(
    *,
    pane: str = "",
    prompt: str | None = None,
    after_echo_index: int | None = None,
    session_dir: str | None = None,
    since_mtime: float | None = None,
) -> dict[str, Any]:
    """Grok Build spawn + parallel tool evidence. Not Ohmypi recap.

    After the current-turn prompt echo, require spawn chrome that is not the
    sent prompt, plus two distinct tool rows for parallel ``pwd`` and
    ``uname``. Fail closed on prompt-echo-only spawn, ``Tool label:`` dumps,
    or a wrap-token / ``/tmp`` recap with no spawn chrome.
    """

    failures: list[str] = []
    sent = (prompt or "").strip()
    current = _grok_current_turn_text(pane, prompt, after_echo_index)
    session_records, session_truncated = _grok_session_tool_records(
        session_dir, since_mtime=since_mtime
    )
    if session_truncated:
        failures.append("Grok transcript exceeded bounded evidence read")
    session_tool_calls = [
        obj for obj in session_records if _grok_record_is_tool_call(obj)
    ]
    session_blob = json.dumps(session_tool_calls, default=str).lower()
    tool_names = {
        _grok_record_tool_name(obj).lower() for obj in session_tool_calls
    }

    dump = bool(_GROK_TOOL_LABEL_DUMP_RE.search(current))
    spawn_in_prompt = "spawn_subagent" in sent.lower()
    spawn_chrome = False
    for match in _GROK_SPAWN_CHROME_RE.finditer(current):
        line = current[max(0, match.start() - 80) : match.end() + 80]
        if spawn_in_prompt and "Call spawn_subagent" in line:
            continue
        spawn_chrome = True
        break
    if not spawn_chrome:
        spawn_chrome = (
            "spawn_subagent" in tool_names or "subagent_spawned" in session_blob
        )

    run_rows = list(_GROK_RUN_ROW_RE.findall(current))
    pwd_row = bool(_GROK_PWD_ROW_RE.search(current))
    uname_row = bool(_GROK_UNAME_ROW_RE.search(current))
    session_run_commands: list[str] = []
    seen_commands: set[str] = set()
    for obj in session_tool_calls:
        name = _grok_record_tool_name(obj).lower()
        command = _grok_record_command(obj).lower()
        if not command and name in {"pwd", "uname"}:
            command = name
        if not command:
            continue
        is_run = name in {"run_terminal_command", "pwd", "uname"} or command in {
            "pwd",
            "uname",
        } or command.startswith("pwd ") or command.startswith("uname")
        if not is_run:
            continue
        if command in seen_commands:
            continue
        seen_commands.add(command)
        session_run_commands.append(command)
    if session_run_commands:
        pwd_row = pwd_row or any(
            command == "pwd" or command.startswith("pwd ")
            for command in session_run_commands
        )
        uname_row = uname_row or any(
            command == "uname" or command.startswith("uname")
            for command in session_run_commands
        )
        if len(session_run_commands) >= 2:
            run_rows = run_rows or ["session"] * len(session_run_commands)

    if dump:
        failures.append(
            "Grok orchestration dumped native-text Tool label: output; "
            "structured spawn/tool launch is required"
        )
    if not spawn_chrome:
        failures.append(
            "Grok orchestration is missing current-turn spawn chrome "
            "(Subagent / spawn_subagent tool row, or sessionUpdate=tool_call); "
            "prompt-echo-only spawn is not evidence"
        )
    if len(run_rows) < 2 and not (pwd_row and uname_row):
        failures.append(
            "Grok orchestration is missing two distinct current-turn tool rows "
            "for parallel pwd and uname; a single parent uname/recap is not "
            "parallel tool evidence"
        )
    elif not (pwd_row and uname_row):
        failures.append(
            "Grok orchestration tool rows do not show both pwd and uname"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "spawn_chrome": spawn_chrome,
        "pwd_row": pwd_row,
        "uname_row": uname_row,
        "run_rows": len(run_rows),
        "tool_label_dump": dump,
        "session_tool_calls": len(session_tool_calls),
        "kind": "grok_spawn_tool",
    }


_MUSE_SPAWN_TOOLS = frozenset(
    {
        "subagent_spawn",
        "muse.subagent_spawn",
    }
)
_MUSE_RESULT_TOOLS = frozenset(
    {
        "subagent_read_result",
        "muse.subagent_read_result",
        "subagent_wait",
        "muse.subagent_wait",
    }
)
_MUSE_SPAWN_EVENTS = frozenset(
    {
        "subagent.control.spawn_accepted",
        "subagent.control.child_session_bound",
        "subagent.control.start_attested",
        "spawn_accepted",
        "child_session_bound",
        "start_attested",
    }
)
_MUSE_RESULT_EVENTS = frozenset(
    {
        "subagent.control.result_ready",
        "subagent.control.closed",
        "result_ready",
        "closed",
    }
)
_MUSE_SPAWN_CHROME_RE = re.compile(
    r"(?:subagent_spawn|Spawn accepted|Child starting|Child running)\b",
    re.IGNORECASE,
)
_MUSE_RESULT_CHROME_RE = re.compile(
    r"(?:Result envelope ready|Reading result for|Child closed)\b",
    re.IGNORECASE,
)
_MUSE_PROMPT_SPAWN_RE = re.compile(r"Call subagent_spawn", re.IGNORECASE)


def muse_workspace_session_root(session_dir: str | None = None) -> Path:
    """Return the harness Muse session root (XDG_DATA_HOME overlay)."""

    return Path(session_dir or "/tmp/hv2-muse-sessions")


def _muse_current_turn_text(
    pane: str,
    prompt: str | None,
    after_echo_index: int | None,
) -> str:
    pane_text = pane or ""
    if prompt is not None:
        scan_start = _pane_scan_start(
            pane_text, prompt, after_echo_index=after_echo_index
        )
        pane_text = "\n".join(pane_text.splitlines()[scan_start:])
    return pane_text


def _muse_session_jsonl_paths(
    session_dir: str | None,
    *,
    since_mtime: float | None = None,
) -> list[Path]:
    root = Path(session_dir) if session_dir else None
    if root is None or not root.is_dir():
        return []
    rows: list[Path] = []
    mtimes: dict[Path, float] = {}
    for path in root.rglob("session.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since_mtime is not None and mtime < (since_mtime - 2):
            continue
        rows.append(path)
        mtimes[path] = mtime
    rows.sort(key=lambda item: (mtimes[item], str(item)), reverse=True)
    return rows[:_JSONL_SCAN_CAP]


def _muse_event_kind(obj: Mapping[str, Any]) -> str:
    payload = obj.get("payload")
    if isinstance(payload, Mapping):
        event = payload.get("event")
        if isinstance(event, Mapping):
            for key in ("kind", "action", "operation", "event"):
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        payload_type = obj.get("payload_type")
        if isinstance(payload_type, str) and payload_type.strip():
            return payload_type.strip()
    for key in ("kind", "type", "payload_type"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _muse_tool_names(obj: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    blob_sources: list[Any] = [obj]
    payload = obj.get("payload")
    if isinstance(payload, Mapping):
        blob_sources.append(payload)
        event = payload.get("event")
        if isinstance(event, Mapping):
            blob_sources.append(event)
            calls = event.get("tool_calls")
            if isinstance(calls, list):
                for item in calls:
                    if isinstance(item, Mapping):
                        name = item.get("name")
                        if isinstance(name, str) and name.strip():
                            names.add(name.strip())
            task_kind = event.get("task_kind")
            if isinstance(task_kind, str) and task_kind.startswith("tool."):
                names.add(task_kind[len("tool.") :])
    for source in blob_sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("tool", "tool_name", "toolName", "name"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


def muse_spawn_tool_evidence(
    *,
    pane: str = "",
    prompt: str | None = None,
    after_echo_index: int | None = None,
    session_dir: str | None = None,
    since_mtime: float | None = None,
) -> dict[str, Any]:
    """Muse-owned spawn + child completion. Wrap-token recap is not a pass.

    After the current-turn prompt echo, require ``subagent_spawn`` chrome that
    is not the sent prompt, plus session JSONL spawn/result events when the
    harness session dir is available. Fail closed on prompt-echo-only spawn
    or a wrap-token recap with no spawn chrome.
    """

    failures: list[str] = []
    sent = (prompt or "").strip()
    current = _muse_current_turn_text(pane, prompt, after_echo_index)
    session_paths = _muse_session_jsonl_paths(session_dir, since_mtime=since_mtime)
    tool_names: set[str] = set()
    event_kinds: set[str] = set()
    result_ready = False
    session_truncated = False
    for path in session_paths:
        bounded_read = _iter_jsonl_objects(path)
        for obj in bounded_read:
            kind = _muse_event_kind(obj)
            if kind:
                event_kinds.add(kind)
            names = _muse_tool_names(obj)
            tool_names.update(names)
            if kind in _MUSE_RESULT_EVENTS or names & _MUSE_RESULT_TOOLS:
                result_ready = True
        session_truncated = session_truncated or bounded_read.truncated
    if session_truncated:
        failures.append("Muse transcript exceeded bounded evidence read")

    spawn_chrome = False
    for match in _MUSE_SPAWN_CHROME_RE.finditer(current):
        line = current[max(0, match.start() - 80) : match.end() + 80]
        if _MUSE_PROMPT_SPAWN_RE.search(line) and "Call subagent_spawn" in sent:
            continue
        spawn_chrome = True
        break
    if not spawn_chrome:
        spawn_chrome = bool(tool_names & _MUSE_SPAWN_TOOLS) or bool(
            event_kinds & _MUSE_SPAWN_EVENTS
        )
    child_completed = bool(result_ready or _MUSE_RESULT_CHROME_RE.search(current))

    if not spawn_chrome:
        failures.append(
            "Muse orchestration is missing current-turn spawn chrome "
            "(subagent_spawn / Spawn accepted, or session JSONL "
            "subagent.control.spawn_accepted); prompt-echo-only spawn is "
            "not evidence"
        )
    if spawn_chrome and not child_completed:
        failures.append(
            "Muse orchestration spawned a child but has no completion "
            "evidence (subagent.control.result_ready / Reading result for); "
            "wrap-token recap is not child-spawn evidence"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "spawn_chrome": spawn_chrome,
        "child_completed": child_completed,
        "tool_names": sorted(tool_names),
        "event_kinds": sorted(event_kinds),
        "session_jsonl": [str(path) for path in session_paths[:4]],
        "kind": "muse_spawn_tool",
    }


def codex_workspace_session_root() -> Path:
    """Return ``~/.codex/sessions`` for Codex rollout JSONL."""

    return Path.home() / ".codex" / "sessions"


def _codex_jsonl_paths(
    session_dir: str | None,
    *,
    since_mtime: float | None = None,
) -> list[Path]:
    root = Path(session_dir) if session_dir else None
    if root is None or not root.is_dir():
        return []
    rows: list[tuple[float, str, Path]] = []
    for path in root.rglob("*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since_mtime is not None and mtime < (since_mtime - 2):
            continue
        rows.append((mtime, str(path), path))
    newest = heapq.nlargest(
        _JSONL_SCAN_CAP,
        rows,
        key=lambda item: (item[0], item[1]),
    )
    return [item[2] for item in newest]


def _codex_session_meta(path: Path) -> dict[str, Any]:
    bounded_read = _iter_jsonl_objects(path)
    found: dict[str, Any] = {}
    for obj in bounded_read:
        if obj.get("type") == "session_meta" and isinstance(obj.get("payload"), dict):
            found = dict(obj["payload"])
    if bounded_read.truncated:
        found["_bounded_read_truncated"] = True
    return found


def _codex_function_call_name(payload: Mapping[str, Any]) -> str:
    name = payload.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _codex_function_call_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("arguments")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _codex_spawn_targets(payload: Mapping[str, Any]) -> set[str]:
    args = _codex_function_call_args(payload)
    targets: set[str] = set()
    for key in ("model", "agent_type", "agent", "task_name"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            targets.add(value.strip())
    return targets


def _codex_nested_sources(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _codex_nested_sources(child)
    elif isinstance(value, list):
        for child in value:
            yield from _codex_nested_sources(child)


def _codex_child_contract(
    path: Path,
    *,
    workspace: str,
    expected_alias: str,
) -> dict[str, Any]:
    """Validate one child transcript against the two-command acceptance contract."""

    bounded_read = _iter_jsonl_objects(path)
    records = list(bounded_read)
    calls: list[dict[str, Any]] = []
    outputs: dict[str, Mapping[str, Any]] = {}
    command_events: dict[str, Mapping[str, Any]] = {}
    effective_aliases: set[str] = set()
    providers: set[str] = set()
    selected_models: set[str] = set()
    route_families: set[str] = set()
    provenance_tuples: set[tuple[str, str, str, str]] = set()
    task_complete = False
    final_answer = False

    for index, obj in enumerate(records):
        payload = obj.get("payload")
        if not isinstance(payload, Mapping):
            continue
        for source in _codex_nested_sources(obj):
            alias_value = next(
                (
                    str(source.get(key)).strip()
                    for key in (
                        "alias_model",
                        "effective_alias_model",
                        "model_alias",
                        "requested_model_alias",
                        "codex_auto_agent_alias",
                    )
                    if isinstance(source.get(key), str)
                    and source.get(key).strip()
                ),
                "",
            )
            for key in (
                "alias_model",
                "effective_alias_model",
                "model_alias",
                "requested_model_alias",
                "codex_auto_agent_alias",
            ):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    effective_aliases.add(value.strip())
            for key in (
                "selected_provider",
                "producer_provider",
                "codex_auto_agent_selected_provider",
                "anthropic_auto_agent_selected_provider",
            ):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    providers.add(value.strip())
            for key in (
                "selected_model",
                "producer_model",
                "codex_auto_agent_selected_model",
                "anthropic_auto_agent_selected_model",
            ):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    selected_models.add(value.strip())
            for key in (
                "selected_route_family",
                "producer_route_family",
                "codex_auto_agent_selected_route_family",
                "anthropic_auto_agent_selected_route_family",
            ):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    route_families.add(value.strip())
            provider_value = next(
                (
                    str(source.get(key)).strip()
                    for key in (
                        "producer_provider",
                        "selected_provider",
                        "codex_auto_agent_selected_provider",
                        "anthropic_auto_agent_selected_provider",
                    )
                    if isinstance(source.get(key), str)
                    and source.get(key).strip()
                ),
                "",
            )
            model_value = next(
                (
                    str(source.get(key)).strip()
                    for key in (
                        "producer_model",
                        "selected_model",
                        "codex_auto_agent_selected_model",
                        "anthropic_auto_agent_selected_model",
                    )
                    if isinstance(source.get(key), str)
                    and source.get(key).strip()
                ),
                "",
            )
            route_value = next(
                (
                    str(source.get(key)).strip()
                    for key in (
                        "producer_route_family",
                        "selected_route_family",
                        "codex_auto_agent_selected_route_family",
                        "anthropic_auto_agent_selected_route_family",
                    )
                    if isinstance(source.get(key), str)
                    and source.get(key).strip()
                ),
                "",
            )
            if alias_value and provider_value and model_value and route_value:
                provenance_tuples.add(
                    (alias_value, provider_value, model_value, route_value)
                )
            encrypted = source.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted.startswith("aawm_erp:"):
                encoded = encrypted[len("aawm_erp:") :].split(";", 1)[0]
                try:
                    decoded = base64.urlsafe_b64decode(
                        encoded + "=" * (-len(encoded) % 4)
                    )
                    envelope = json.loads(decoded.decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    envelope = {}
                if isinstance(envelope, Mapping):
                    alias_value = next(
                        (
                            str(envelope.get(key)).strip()
                            for key in (
                                "alias_model",
                                "effective_alias_model",
                                "alias_family",
                            )
                            if isinstance(envelope.get(key), str)
                            and envelope.get(key).strip()
                        ),
                        "",
                    )
                    for key in ("alias_model", "effective_alias_model", "alias_family"):
                        value = envelope.get(key)
                        if isinstance(value, str) and value.strip():
                            effective_aliases.add(value.strip())
                    envelope_provider = next(
                        (
                            str(envelope.get(key)).strip()
                            for key in ("producer_provider", "selected_provider")
                            if isinstance(envelope.get(key), str)
                            and envelope.get(key).strip()
                        ),
                        "",
                    )
                    envelope_model = next(
                        (
                            str(envelope.get(key)).strip()
                            for key in ("producer_model", "selected_model")
                            if isinstance(envelope.get(key), str)
                            and envelope.get(key).strip()
                        ),
                        "",
                    )
                    envelope_route = next(
                        (
                            str(envelope.get(key)).strip()
                            for key in (
                                "producer_route_family",
                                "selected_route_family",
                            )
                            if isinstance(envelope.get(key), str)
                            and envelope.get(key).strip()
                        ),
                        "",
                    )
                    for key in ("producer_provider", "selected_provider"):
                        value = envelope.get(key)
                        if isinstance(value, str) and value.strip():
                            providers.add(value.strip())
                    for key in ("producer_model", "selected_model"):
                        value = envelope.get(key)
                        if isinstance(value, str) and value.strip():
                            selected_models.add(value.strip())
                    for key in (
                        "producer_route_family",
                        "selected_route_family",
                    ):
                        value = envelope.get(key)
                        if isinstance(value, str) and value.strip():
                            route_families.add(value.strip())
                    if alias_value and envelope_provider and envelope_model and envelope_route:
                        provenance_tuples.add(
                            (
                                alias_value,
                                envelope_provider,
                                envelope_model,
                                envelope_route,
                            )
                        )
        kind = str(payload.get("type") or "")
        if kind == "function_call":
            name = _codex_function_call_name(payload)
            if name == "exec_command":
                args = _codex_function_call_args(payload)
                call_id = payload.get("call_id") or payload.get("id")
                calls.append(
                    {
                        "index": index,
                        "call_id": str(call_id or ""),
                        "cmd": str(args.get("cmd") or "").strip(),
                    }
                )
        elif kind == "function_call_output":
            call_id = payload.get("call_id")
            if isinstance(call_id, str) and call_id:
                outputs[call_id] = payload

        item = payload.get("item")
        if isinstance(item, Mapping) and str(item.get("type") or "") == "CommandExecution":
            event_id = item.get("id")
            if isinstance(event_id, str) and event_id:
                command_events[event_id] = item
            continue

        if kind == "task_complete":
            task_complete = True
        if (
            isinstance(item, Mapping)
            and str(item.get("type") or "") == "AgentMessage"
            and str(item.get("phase") or "") == "final_answer"
        ):
            final_answer = True
        if (
            kind == "message"
            and str(payload.get("phase") or "") == "final_answer"
        ):
            final_answer = True

    failures: list[str] = []
    if bounded_read.truncated:
        failures.append("child transcript exceeded bounded evidence read")
    expected = {"pwd": f"{workspace}\n", "uname -s": "Linux\n"}
    call_indices = [call["index"] for call in calls]
    if len(calls) != 2:
        failures.append(
            f"child must issue exactly two exec_command calls (found={len(calls)})"
        )
    elif call_indices[1] != call_indices[0] + 1:
        failures.append("child exec_command calls were not emitted concurrently")

    commands = [call["cmd"] for call in calls]
    if sorted(commands) != sorted(expected):
        failures.append(
            "child commands must be exactly `pwd` and `uname -s` "
            f"(found={commands!r})"
        )

    for call in calls:
        cmd = call["cmd"]
        call_id = call["call_id"]
        expected_stdout = expected.get(cmd)
        event = command_events.get(call_id)
        output = outputs.get(call_id)
        if not call_id:
            failures.append(f"child {cmd or 'exec_command'} record has no call_id")
            continue
        if event is None:
            failures.append(f"child {cmd} has no matching CommandExecution record")
            continue
        if str(event.get("status") or "") != "completed" or event.get("exit_code") != 0:
            failures.append(f"child {cmd} did not complete successfully")
        stdout = event.get("stdout")
        if expected_stdout is not None and stdout != expected_stdout:
            failures.append(
                f"child {cmd} stdout mismatch "
                f"(expected={expected_stdout!r}, found={stdout!r})"
            )
        if output is None:
            failures.append(f"child {cmd} has no matching function_call_output")
        elif expected_stdout is not None:
            rendered = str(output.get("output") or "")
            marker = "Output:\n"
            if (
                marker not in rendered
                or rendered.split(marker, 1)[1].strip()
                != expected_stdout.strip()
            ):
                failures.append(
                    f"child {cmd} function_call_output lacks exact stdout"
                )

    if not final_answer:
        failures.append("child has no final_answer record")
    if not task_complete:
        failures.append("child has no task_complete record")
    expected_identity = {
        "basic": ("openai", "gpt-5.6-luna", "codex_responses"),
        "work": ("openai", "gpt-5.6-luna", "codex_responses"),
        "expert": ("openai", "gpt-6-astra", "codex_responses"),
    }.get(expected_alias)
    if expected_alias not in effective_aliases:
        failures.append(
            f"child effective alias does not include requested `{expected_alias}`"
        )
    matching_tuples = {
        item for item in provenance_tuples if item[0] == expected_alias
    }
    if expected_identity is not None:
        matching_tuples = {
            item for item in matching_tuples if item[1:] == expected_identity
        }
    if not matching_tuples:
        failures.append(
            "child has no correlated expected producer identity "
            f"for alias `{expected_alias}`"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "commands": commands,
        "call_count": len(calls),
        "parallel_streak": 2 if len(calls) == 2 and not failures else 0,
        "final_answer": final_answer,
        "task_complete": task_complete,
        "effective_aliases": sorted(effective_aliases),
        "providers": sorted(providers),
        "selected_models": sorted(selected_models),
        "route_families": sorted(route_families),
        "provenance_tuples": [list(item) for item in sorted(provenance_tuples)],
    }


def codex_spawn_tool_evidence(
    *,
    children: Sequence[str],
    pane: str = "",
    session_dir: str | None = None,
    since_mtime: float | None = None,
    prompt: str | None = None,
    after_echo_index: int | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Require each planned Codex child to complete the two-command contract."""

    wanted = [str(item) for item in children if str(item).strip()]
    failures: list[str] = []
    paths = _codex_jsonl_paths(session_dir, since_mtime=since_mtime)
    workspace_cwd = str(workspace or "").rstrip("/")
    session_paths: list[str] = []
    spawn_requests: list[dict[str, Any]] = []
    activity_by_call_id: dict[str, str] = {}
    child_paths_by_id: dict[str, Path] = {}

    for path in paths:
        meta = _codex_session_meta(path)
        cwd = str(meta.get("cwd") or "").rstrip("/")
        if workspace_cwd and cwd and cwd != workspace_cwd:
            continue
        if meta.get("_bounded_read_truncated"):
            failures.append(
                f"Codex transcript exceeded bounded evidence read: {path.name}"
            )
        session_paths.append(str(path))
        session_id = str(meta.get("id") or meta.get("session_id") or "").strip()
        if session_id:
            child_paths_by_id[session_id] = path
        for obj in _iter_jsonl_objects(path):
            payload = obj.get("payload")
            if not isinstance(payload, Mapping):
                continue
            kind = str(payload.get("type") or "")
            if kind == "function_call" and _codex_function_call_name(payload) == "spawn_agent":
                call_id = payload.get("call_id") or payload.get("id")
                spawn_requests.append(
                    {
                        "call_id": str(call_id or ""),
                        "targets": _codex_spawn_targets(payload),
                    }
                )
            item = payload.get("item")
            if (
                isinstance(item, Mapping)
                and str(item.get("type") or "") == "SubAgentActivity"
            ):
                thread_id = item.get("agent_thread_id")
                if (
                    str(item.get("kind") or "") == "started"
                    and isinstance(thread_id, str)
                    and thread_id.strip()
                ):
                    activity_id = item.get("id")
                    if isinstance(activity_id, str) and activity_id.strip():
                        activity_by_call_id[activity_id] = thread_id.strip()

    remaining_requests = list(spawn_requests)
    child_evidence: list[dict[str, Any]] = []
    spawned_targets: list[str] = []
    for target in wanted:
        match_index = next(
            (
                index
                for index, request in enumerate(remaining_requests)
                if target in request["targets"]
                and request["call_id"] in activity_by_call_id
            ),
            None,
        )
        if match_index is None:
            failures.append(f"Codex orchestration is missing spawn_agent target {target}")
            child_evidence.append(
                {"target": target, "ok": False, "failures": ["spawn_agent missing"]}
            )
            continue
        request = remaining_requests.pop(match_index)
        spawned_targets.append(target)
        thread_id = activity_by_call_id.get(request["call_id"])
        if not thread_id:
            message = f"spawn_agent target {target} has no started SubAgentActivity"
            failures.append(message)
            child_evidence.append({"target": target, "ok": False, "failures": [message]})
            continue
        path = child_paths_by_id.get(thread_id)
        if path is None:
            message = f"spawn_agent target {target} has no child rollout JSONL"
            failures.append(message)
            child_evidence.append(
                {
                    "target": target,
                    "thread_id": thread_id,
                    "ok": False,
                    "failures": [message],
                }
            )
            continue
        contract = _codex_child_contract(
            path,
            workspace=workspace_cwd,
            expected_alias=target,
        )
        evidence = {
            "target": target,
            "thread_id": thread_id,
            "jsonl": str(path),
            **contract,
        }
        child_evidence.append(evidence)
        failures.extend(f"{target}: {failure}" for failure in contract["failures"])

    parallel_streak = min(
        (
            int(evidence.get("parallel_streak") or 0)
            for evidence in child_evidence
            if evidence.get("target") in wanted
        ),
        default=0,
    )

    return {
        "ok": not failures,
        "failures": failures,
        "children": wanted,
        "spawned_targets": sorted(spawned_targets),
        "child_thread_ids": sorted(activity_by_call_id.values()),
        "parallel_streak": parallel_streak,
        "child_evidence": child_evidence,
        "session_jsonl": session_paths[:4],
        "child_jsonl": [
            str(evidence["jsonl"])
            for evidence in child_evidence
            if evidence.get("jsonl")
        ][:4],
        "kind": "codex_parallel_child_tools",
    }
