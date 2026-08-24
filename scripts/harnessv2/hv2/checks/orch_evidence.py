"""Orchestration child-spawn evidence. Recap-only is not a pass."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
    for path in (*session_dir.glob("*.jsonl"), *session_dir.glob("*/*.jsonl")):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since_mtime is not None and mtime < (since_mtime - 2):
            continue
        rows.append(path)
    return sorted(rows, key=lambda item: item.stat().st_mtime, reverse=True)


def _iter_jsonl_objects(path: Path) -> Iterable[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


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
    failure, not a pass. In-alias model fallback is allowed.
    """

    wanted_set = set(wanted)
    requested = ""
    selected_models: list[str] = []
    producer_providers: list[str] = []
    producer_models: list[str] = []
    producer_routes: list[str] = []
    tools: set[str] = set()
    fallback = False
    error = ""
    completed = False
    for obj in _iter_jsonl_objects(path):
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
        blob = json.dumps(obj, default=str)
        if "producer_provider" in blob:
            match = re.search(r'"producer_provider"\s*:\s*"([^"]+)"', blob)
            if match:
                producer_providers.append(match.group(1))
            match = re.search(r'"producer_model"\s*:\s*"([^"]+)"', blob)
            if match:
                producer_models.append(match.group(1))
            match = re.search(r'"producer_route_family"\s*:\s*"([^"]+)"', blob)
            if match:
                producer_routes.append(match.group(1))
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
    selected_provider = ""
    unique_producers = list(dict.fromkeys(producer_providers))
    if unique_producers:
        selected_provider = unique_producers[-1]
    elif expected_provider and not escaped:
        selected_provider = expected_provider
    cross_provider = bool(
        expected_provider
        and selected_provider
        and selected_provider != expected_provider
    )
    disposition = "completed"
    if escaped or (fallback and escaped) or cross_provider:
        disposition = "fallback_operational"
    elif error and not completed:
        disposition = "unavailable"
    elif not completed:
        disposition = "incomplete"
    ok = completed and not escaped and not cross_provider
    return {
        "requested_alias": requested,
        "selected_provider": selected_provider,
        "model": (producer_models[-1] if producer_models else (unique_models[-1] if unique_models else "")),
        "route_family": producer_routes[-1] if producer_routes else "",
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
        for obj in _iter_jsonl_objects(path):
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
        if root is not None and path.parent != root:
            nested_route = _nested_child_route_row(path, wanted)
            if nested_route:
                alias = str(nested_route.get("requested_alias") or file_agent or "")
                if alias in wanted_set:
                    routes[alias] = nested_route
                    if nested_route.get("ok") is True:
                        successful_agents.add(alias)
                    elif nested_route.get("terminal_disposition") == "fallback_operational":
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
        elif row.get("terminal_disposition") == "fallback_operational":
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
