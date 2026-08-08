import asyncio
import hashlib
import json
import re
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from litellm.proxy.pass_through_endpoints import (
    aawm_context_query as _context_query,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    claude_prompt_replacement as _prompt_replacement,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    observability_metadata as _observability_metadata,
)

_CLAUDE_AUTO_MEMORY_TEMPLATE_LOGICAL_PATH = (
    _prompt_replacement._CLAUDE_AUTO_MEMORY_TEMPLATE_LOGICAL_PATH
)
_CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH = (
    _prompt_replacement._CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH
)
_CLAUDE_COMMONMARK_PROMPT_SENTENCE = (
    "You can use Github-flavored markdown for formatting, and will be rendered in a "
    "monospace font using the CommonMark specification."
)
_CLAUDE_COMMONMARK_PROMPT_IDENTIFIER_TEMPLATE = (
    "You can use Github-flavored markdown for formatting, and will be rendered in a "
    "monospace font using the CommonMark specification plus the following as a custom "
    "known list of technical identifiers: {identifiers}."
)
_CLAUDE_TOOL_DESCRIPTION_MAX_CHARS = 360
_CLAUDE_TOOL_SCHEMA_DESCRIPTION_MAX_CHARS = 160
_CLAUDE_TOOL_SCHEMA_DROP_KEYS = {"$schema"}
_CLAUDE_TOOL_DESCRIPTION_PRESERVE_NAMES = {"Agent"}
_CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_NAME = (
    "claude-tool-advertisement-compaction"
)
_CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_VERSION = "2026-06-23.1"
_CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_CACHE_MAX_ENTRIES = 256
_CLAUDE_KNOWN_TOOL_DESCRIPTIONS = {
    "Bash": (
        "Run a shell command. Prefer dedicated tools for search/read/edit/write. "
        "Use absolute paths when practical, provide a concise `description`, and "
        "use `run_in_background` for long tasks. Avoid destructive git, force, "
        "no-verify, commit, or push actions unless explicitly requested."
    ),
    "Glob": "Find files by glob pattern. Provide `pattern`; optionally set `path`.",
    "Grep": (
        "Search file contents with ripgrep syntax. Provide `pattern`; optionally set "
        "`path`, `glob`, `type`, output mode, context, and case-sensitivity options."
    ),
    "Read": "Read a file by absolute path. Use offsets/limits for long text reads.",
    "Edit": (
        "Edit an existing file by exact string replacement after reading it. "
        "Preserve indentation; use `replace_all` only for intentional broad changes."
    ),
    "Write": (
        "Create or overwrite a file. Read existing files first and prefer `Edit` "
        "for modifications. Do not create docs unless explicitly requested."
    ),
    "NotebookEdit": "Edit a Jupyter notebook cell after reading the notebook.",
    "WebFetch": "Fetch and summarize a web page from a URL.",
    "WebSearch": "Search the web for current information.",
    "TodoWrite": "Create or update the current conversation task list for active work.",
    "Task": "Launch a subagent for bounded parallel work with a clear task.",
    "Skill": "Load and follow a named local skill when it applies.",
    "ExitPlanMode": "Leave plan mode after the user approves the plan.",
    "BashOutput": "Read output from a background bash command.",
    "KillBash": "Stop a background bash command.",
    "EnterWorktree": "Enter an existing worktree for isolated task work.",
    "ExitWorktree": "Exit and optionally clean up an isolated worktree.",
}
_AAWM_REFERENCE_IDENTIFIER_PATCH_ID = "technical-identifiers-list"
_AAWM_REFERENCE_IDENTIFIER_CACHE_KEY = "reference-identifiers"
_AAWM_DYNAMIC_DIRECTIVE_PATTERN = re.compile(
    r"<!--\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<html_attrs>.*?)\s*-->"
    r"|@@@\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<at_attrs>.*?)\s*@@@"
    r"|^[ \t]*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<line_attrs>[^\r\n]+?)\s*$",
    re.DOTALL | re.MULTILINE,
)
_AAWM_CONTEXT_MARKER_PATTERN = re.compile(r":#(?P<name>[^#\r\n]+?)\.ctx#:")
_AAWM_ESCAPED_CONTEXT_MARKER_PATTERN = re.compile(
    r"\\+:#(?P<name>[^#\r\n]+?)\.ctx#\\+:"
)
_AAWM_ESCAPED_CONTEXT_MARKER_PLACEHOLDER = "@@AAWM_ESCAPED_CTX_MARKER_{index}@@"
_AAWM_DISPATCH_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"(?<![\\`])`(?P<backtick>[^`\r\n]+?)`(?!`)" r"|(?P<acronym>\b[A-Z][A-Z0-9]{1,}\b)"
)
_AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)="
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s]+))'
)
_AAWM_AGENT_MEMORY_PROC_NAME = "get_agent_memories"
_AAWM_DYNAMIC_PROC_ALIASES = {"get_agent_memory": _AAWM_AGENT_MEMORY_PROC_NAME}
_AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS: dict[str, tuple[str, ...]] = {
    _AAWM_AGENT_MEMORY_PROC_NAME: ("agent", "tenant"),
}
_AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE = (
    "## AAWM Injection Status\n\n"
    'AAWM "{proc_name}" failed for this session.\n'
    "Alert the user or session orchestrator.\n"
)
_AAWM_CONTEXT_GRAB_FAILURE_TEMPLATE = "IMPORTANT: context grab for {name} returned no results. immediately inform the opperator."
_AAWM_SUBAGENTSTART_CONTEXT_MARKERS = (
    "SubagentStart hook additional context:",
    "SubAgentStart hook additional context:",
)
_AAWM_SYSTEM_REMINDER_BLOCK_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>\n*",
    re.DOTALL,
)
_AAWM_NO_MEMORIES_TEMPLATE = (
    "# Memory Injection\n" "You have saved no memories as of yet.\n"
)
_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE = (
    _context_query._AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE
)
# Cap distinct dispatch backtick/acronym lookups per text node (High/RR-053 #1).
_AAWM_DISPATCH_CONTEXT_REFERENCE_MAX = 24
# Cap total dispatch lookups across an entire request (all trusted text blocks).
# Per-node caps alone still multiply when system + first user have many blocks.
_AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX = 48
# Common all-caps tokens that are noise for dispatch context grabs.
_AAWM_DISPATCH_ACRONYM_STOPWORDS = frozenset(
    {
        "AAWM",
        "API",
        "AWS",
        "CPU",
        "CSS",
        "CSV",
        "DB",
        "DNS",
        "ENV",
        "EOF",
        "GET",
        "GPU",
        "HTML",
        "HTTP",
        "HTTPS",
        "ID",
        "JSON",
        "JWT",
        "LLM",
        "OK",
        "OS",
        "PDF",
        "POST",
        "PUT",
        "RAM",
        "REST",
        "SDK",
        "SQL",
        "SSH",
        "SSL",
        "TCP",
        "TLS",
        "TODO",
        "TTL",
        "UI",
        "UID",
        "URL",
        "URI",
        "UUID",
        "UTF",
        "XML",
        "YAML",
        "YML",
    }
)
_aawm_dynamic_injection_cache = _context_query._aawm_dynamic_injection_cache
_aawm_context_grab_cache = _context_query._aawm_context_grab_cache
_claude_context_replacement_template_cache = (
    _prompt_replacement._claude_context_replacement_template_cache
)
_claude_prompt_patch_manifest_cache = (
    _prompt_replacement._claude_prompt_patch_manifest_cache
)
_claude_tool_advertisement_compaction_cache: dict[
    str, tuple[dict[str, Any], dict[str, Any]]
] = {}


@dataclass(frozen=True, slots=True)
class ClaudeControlPlaneServices:
    prompt: _prompt_replacement.ClaudePromptReplacementServices
    context_query: _context_query.ContextQueryServices
    now_utc: Callable[[], datetime]
    merge_metadata: Callable[..., dict[str, Any]]
    build_span: Callable[..., dict[str, Any]]
    format_span_timestamp: Callable[[datetime], str]
    add_context_file_metadata: Callable[[dict[str, Any]], dict[str, Any]]


_active_services: ContextVar[Optional[ClaudeControlPlaneServices]] = ContextVar(
    "aawm_claude_control_plane_services",
    default=None,
)
_standalone_rewriter: Optional["ClaudeControlPlaneRewriter"] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_standalone_services() -> ClaudeControlPlaneServices:
    prompt = _prompt_replacement.ClaudePromptReplacementServices(
        resolve_auto_memory_template_path=(
            lambda cc_version: _resolve_claude_auto_memory_template_path(cc_version)
        ),
        resolve_prompt_patch_manifest_path=(
            lambda: _resolve_claude_prompt_patch_manifest_path()
        ),
        load_prompt_patch_manifest=(
            lambda path: _load_claude_prompt_patch_manifest(path)
        ),
        replace_auto_memory_section=(
            lambda text, version: _replace_claude_auto_memory_section_in_text(
                text, version
            )
        ),
        apply_prompt_patch_manifest=(
            lambda text, **kwargs: _apply_claude_prompt_patch_manifest_to_text(
                text, **kwargs
            )
        ),
        add_override_metadata=(
            lambda body, events: _add_claude_system_prompt_override_logging_metadata(
                body, events
            )
        ),
        add_patch_metadata=(
            lambda body, events: _add_claude_prompt_patch_logging_metadata(
                body, events
            )
        ),
    )
    context_query = _context_query.build_context_query_services(
        get_agent_memories=(
            lambda **kwargs: _call_aawm_get_agent_memories(**kwargs)
        ),
        get_context=lambda **kwargs: _call_aawm_context_grab(**kwargs),
        get_reference_identifiers=(
            lambda **kwargs: _call_aawm_reference_identifier_list(**kwargs)
        ),
    )
    return ClaudeControlPlaneServices(
        prompt=prompt,
        context_query=context_query,
        now_utc=_utc_now,
        merge_metadata=_observability_metadata._merge_litellm_metadata,
        build_span=_observability_metadata._build_langfuse_span_descriptor,
        format_span_timestamp=(
            _observability_metadata._format_langfuse_span_timestamp
        ),
        add_context_file_metadata=(
            _observability_metadata._add_claude_post_rewrite_context_file_logging_metadata
        ),
    )


def build_claude_control_plane_services(
    *,
    context_query: _context_query.ContextQueryServices,
    now_utc: Callable[[], datetime],
    merge_metadata: Callable[..., dict[str, Any]],
    build_span: Callable[..., dict[str, Any]],
    format_span_timestamp: Callable[[datetime], str],
    add_context_file_metadata: Callable[[dict[str, Any]], dict[str, Any]],
    prompt: Optional[
        _prompt_replacement.ClaudePromptReplacementServices
    ] = None,
) -> ClaudeControlPlaneServices:
    return ClaudeControlPlaneServices(
        prompt=prompt
        or _prompt_replacement.build_claude_prompt_replacement_services(),
        context_query=context_query,
        now_utc=now_utc,
        merge_metadata=merge_metadata,
        build_span=build_span,
        format_span_timestamp=format_span_timestamp,
        add_context_file_metadata=add_context_file_metadata,
    )


class ClaudeControlPlaneRewriter:
    def __init__(self, services: ClaudeControlPlaneServices) -> None:
        self.services = services

    async def apply_rewrites(
        self,
        request_body: dict[str, Any],
        billing_header_fields: dict[str, str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        token = _active_services.set(self.services)
        try:
            return await _apply_claude_control_plane_rewrites_impl(
                request_body,
                billing_header_fields,
            )
        finally:
            _active_services.reset(token)

    async def expand_dynamic_context(
        self,
        request_body: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        token = _active_services.set(self.services)
        try:
            return await _expand_aawm_dynamic_directives_impl(request_body)
        finally:
            _active_services.reset(token)

    def add_post_rewrite_context_file_metadata(
        self,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return self.services.add_context_file_metadata(request_body)


def compose_claude_control_plane(
    services: ClaudeControlPlaneServices,
) -> ClaudeControlPlaneRewriter:
    if services.prompt is None:
        raise ValueError("Claude control-plane prompt services are required")
    if services.context_query is None:
        raise ValueError("Claude control-plane context-query services are required")
    return ClaudeControlPlaneRewriter(services)


def _get_standalone_rewriter() -> ClaudeControlPlaneRewriter:
    global _standalone_rewriter
    if _standalone_rewriter is None:
        _standalone_rewriter = ClaudeControlPlaneRewriter(
            _build_standalone_services()
        )
    return _standalone_rewriter


def _get_services() -> ClaudeControlPlaneServices:
    active = _active_services.get()
    if active is not None:
        return active
    return _get_standalone_rewriter().services


def _get_aawm_dynamic_injection_cache_ttl_seconds() -> float:
    return _context_query._get_aawm_dynamic_injection_cache_ttl_seconds()


async def _get_cached_aawm_dynamic_injection_result(
    cache_key: tuple[str, str, str, str],
) -> tuple[bool, Optional[str]]:
    return await _get_services().context_query.get_cached_dynamic_result(cache_key)


async def _set_cached_aawm_dynamic_injection_result(
    cache_key: tuple[str, str, str, str],
    injected_text: Optional[str],
) -> None:
    await _get_services().context_query.set_cached_dynamic_result(
        cache_key,
        injected_text,
    )


async def _get_cached_aawm_context_grab_result(
    cache_key: tuple[str, str, str, str, str],
) -> tuple[bool, Optional[dict[str, str]]]:
    return await _get_services().context_query.get_cached_context_result(cache_key)


async def _set_cached_aawm_context_grab_result(
    cache_key: tuple[str, str, str, str, str],
    cached_payload: dict[str, str],
) -> None:
    await _get_services().context_query.set_cached_context_result(
        cache_key,
        cached_payload,
    )


_clean_secret_string = _context_query._clean_secret_string
_get_first_secret_value = _context_query._get_first_secret_value
_normalize_aawm_sslmode = _context_query._normalize_aawm_sslmode
_parse_claude_code_version = _prompt_replacement._parse_claude_code_version
_resolve_claude_auto_memory_template_path = (
    _prompt_replacement._resolve_claude_auto_memory_template_path
)
_load_claude_context_replacement_template = (
    _prompt_replacement._load_claude_context_replacement_template
)
_resolve_claude_prompt_patch_manifest_path = (
    _prompt_replacement._resolve_claude_prompt_patch_manifest_path
)
_load_claude_prompt_patch_manifest = (
    _prompt_replacement._load_claude_prompt_patch_manifest
)
_extract_markdown_section = _prompt_replacement._extract_markdown_section
_render_claude_auto_memory_replacement = (
    _prompt_replacement._render_claude_auto_memory_replacement
)
_replace_claude_auto_memory_section_in_text = (
    _prompt_replacement._replace_claude_auto_memory_section_in_text
)
_replace_claude_system_prompt_override_in_value = (
    _prompt_replacement._replace_claude_system_prompt_override_in_value
)
_add_claude_system_prompt_override_logging_metadata = (
    _prompt_replacement._add_claude_system_prompt_override_logging_metadata
)


def _json_compact_char_count(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return 0


def _collapse_tool_description_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate_tool_description_text(value: str, max_chars: int) -> str:
    collapsed = _collapse_tool_description_text(value)
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 3)].rstrip() + "..."


def _get_claude_tool_name(tool: dict[str, Any]) -> Optional[str]:
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    function_block = tool.get("function")
    if isinstance(function_block, dict):
        function_name = function_block.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def _clone_claude_tool_advertisement_value(value: Any) -> Any:
    return deepcopy(value)


def _get_claude_tool_advertisement_compaction_policy() -> tuple[str, str]:
    return (
        _CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_NAME,
        _CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_VERSION,
    )


def _build_claude_tool_advertisement_compaction_fingerprint(
    *,
    tool: dict[str, Any],
    tool_name: str,
    cc_version: str,
) -> str:
    policy_name, policy_version = _get_claude_tool_advertisement_compaction_policy()
    fingerprint_payload = {
        "cc_version": cc_version,
        "tool_name": tool_name,
        "tool": tool,
        "compaction_policy_name": policy_name,
        "compaction_policy_version": policy_version,
        "schema_description_max_chars": _CLAUDE_TOOL_SCHEMA_DESCRIPTION_MAX_CHARS,
        "tool_description_max_chars": _CLAUDE_TOOL_DESCRIPTION_MAX_CHARS,
        "schema_drop_keys": sorted(_CLAUDE_TOOL_SCHEMA_DROP_KEYS),
        "preserve_tool_names": sorted(_CLAUDE_TOOL_DESCRIPTION_PRESERVE_NAMES),
    }
    canonical_payload = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _remember_claude_tool_advertisement_compaction(
    compaction_fingerprint: str,
    updated_tool: dict[str, Any],
    compaction_event: dict[str, Any],
) -> None:
    _claude_tool_advertisement_compaction_cache.pop(compaction_fingerprint, None)
    _claude_tool_advertisement_compaction_cache[compaction_fingerprint] = (
        _clone_claude_tool_advertisement_value(updated_tool),
        _clone_claude_tool_advertisement_value(compaction_event),
    )
    while (
        len(_claude_tool_advertisement_compaction_cache)
        > _CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_CACHE_MAX_ENTRIES
    ):
        oldest_key = next(iter(_claude_tool_advertisement_compaction_cache))
        _claude_tool_advertisement_compaction_cache.pop(oldest_key, None)


def _compact_claude_tool_schema_value(
    value: Any,
) -> tuple[Any, int, int]:
    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        changed = False
        description_count = 0
        dropped_key_count = 0

        for key, child in value.items():
            if key in _CLAUDE_TOOL_SCHEMA_DROP_KEYS:
                changed = True
                dropped_key_count += 1
                continue

            if key == "description" and isinstance(child, str):
                compacted_child = _truncate_tool_description_text(
                    child,
                    _CLAUDE_TOOL_SCHEMA_DESCRIPTION_MAX_CHARS,
                )
                updated_dict[key] = compacted_child
                if compacted_child != child:
                    changed = True
                    description_count += 1
                continue

            (
                compacted_child,
                child_description_count,
                child_dropped_key_count,
            ) = _compact_claude_tool_schema_value(child)
            updated_dict[key] = compacted_child
            description_count += child_description_count
            dropped_key_count += child_dropped_key_count
            if compacted_child is not child:
                changed = True

        return (
            updated_dict if changed else value,
            description_count,
            dropped_key_count,
        )

    if isinstance(value, list):
        updated_list = []
        changed = False
        description_count = 0
        dropped_key_count = 0
        for child in value:
            (
                compacted_child,
                child_description_count,
                child_dropped_key_count,
            ) = _compact_claude_tool_schema_value(child)
            updated_list.append(compacted_child)
            description_count += child_description_count
            dropped_key_count += child_dropped_key_count
            if compacted_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            description_count,
            dropped_key_count,
        )

    return value, 0, 0


def _compact_claude_tool_advertisement(
    tool: Any,
    *,
    cc_version: str,
) -> tuple[Any, Optional[dict[str, Any]]]:
    if not isinstance(tool, dict):
        return tool, None

    tool_name = _get_claude_tool_name(tool) or "unknown"
    description = tool.get("description")
    input_schema = tool.get("input_schema")
    compaction_fingerprint = _build_claude_tool_advertisement_compaction_fingerprint(
        tool=tool,
        tool_name=tool_name,
        cc_version=cc_version,
    )
    cached_entry = _claude_tool_advertisement_compaction_cache.get(
        compaction_fingerprint
    )
    if cached_entry is not None:
        cached_tool, cached_event = cached_entry
        _claude_tool_advertisement_compaction_cache.pop(compaction_fingerprint, None)
        _claude_tool_advertisement_compaction_cache[compaction_fingerprint] = (
            cached_tool,
            cached_event,
        )
        event = _clone_claude_tool_advertisement_value(cached_event)
        event["cc_version"] = cc_version
        event["compaction_cache_status"] = "hit"
        event["compaction_schema_fingerprint"] = compaction_fingerprint
        return (
            _clone_claude_tool_advertisement_value(cached_tool),
            event,
        )

    original_chars = _json_compact_char_count(tool)
    updated_tool = dict(tool)
    changed = False
    top_level_description_compacted = False
    schema_description_count = 0
    schema_dropped_key_count = 0

    if (
        isinstance(description, str)
        and tool_name not in _CLAUDE_TOOL_DESCRIPTION_PRESERVE_NAMES
    ):
        known_description = _CLAUDE_KNOWN_TOOL_DESCRIPTIONS.get(tool_name)
        compacted_description = (
            known_description
            if known_description is not None
            else _truncate_tool_description_text(
                description,
                _CLAUDE_TOOL_DESCRIPTION_MAX_CHARS,
            )
        )
        if compacted_description != description:
            updated_tool["description"] = compacted_description
            changed = True
            top_level_description_compacted = True

    if isinstance(input_schema, dict):
        (
            compacted_schema,
            schema_description_count,
            schema_dropped_key_count,
        ) = _compact_claude_tool_schema_value(input_schema)
        if compacted_schema is not input_schema:
            updated_tool["input_schema"] = compacted_schema
            changed = True

    if not changed:
        return tool, None

    compacted_chars = _json_compact_char_count(updated_tool)
    compaction_event = {
        "id": "tool-advertisement",
        "status": "resolved",
        "cc_version": cc_version,
        "tool_name": tool_name,
        "original_chars": original_chars,
        "compacted_chars": compacted_chars,
        "saved_chars": max(0, original_chars - compacted_chars),
        "top_level_description_compacted": top_level_description_compacted,
        "schema_description_compaction_count": schema_description_count,
        "schema_dropped_key_count": schema_dropped_key_count,
        "compaction_policy_name": _CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_NAME,
        "compaction_policy_version": _CLAUDE_TOOL_ADVERTISEMENT_COMPACTION_POLICY_VERSION,
        "compaction_cache_status": "miss",
        "compaction_schema_fingerprint": compaction_fingerprint,
    }
    _remember_claude_tool_advertisement_compaction(
        compaction_fingerprint,
        updated_tool,
        compaction_event,
    )
    return updated_tool, compaction_event


def _compact_claude_tool_advertisements_in_request_body(
    request_body: dict[str, Any],
    *,
    cc_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = request_body.get("tools")
    if not isinstance(tools, list) or not tools:
        return request_body, []

    updated_tools = []
    events: list[dict[str, Any]] = []
    changed = False
    for tool in tools:
        updated_tool, event = _compact_claude_tool_advertisement(
            tool,
            cc_version=cc_version,
        )
        updated_tools.append(updated_tool)
        if event is not None:
            events.append(event)
        if updated_tool is not tool:
            changed = True

    if not changed:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = updated_tools
    return updated_body, events


def _add_claude_tool_advertisement_compaction_logging_metadata(
    request_body: dict[str, Any],
    compaction_events: list[dict[str, Any]],
) -> dict[str, Any]:
    services = _get_services()
    tool_names = sorted(
        {
            event["tool_name"]
            for event in compaction_events
            if isinstance(event.get("tool_name"), str) and event["tool_name"]
        }
    )
    statuses = [
        event["status"]
        for event in compaction_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in compaction_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    original_chars = sum(
        event["original_chars"]
        for event in compaction_events
        if isinstance(event.get("original_chars"), int)
    )
    compacted_chars = sum(
        event["compacted_chars"]
        for event in compaction_events
        if isinstance(event.get("compacted_chars"), int)
    )
    saved_chars = sum(
        event["saved_chars"]
        for event in compaction_events
        if isinstance(event.get("saved_chars"), int)
    )
    (
        compaction_policy_name,
        compaction_policy_version,
    ) = _get_claude_tool_advertisement_compaction_policy()
    compaction_cache_hits = sum(
        1
        for event in compaction_events
        if event.get("compaction_cache_status") == "hit"
    )
    compaction_cache_misses = sum(
        1
        for event in compaction_events
        if event.get("compaction_cache_status") == "miss"
    )

    span_metadata: dict[str, Any] = {
        "tool_count": len(compaction_events),
        "original_chars": original_chars,
        "compacted_chars": compacted_chars,
        "saved_chars": saved_chars,
        "compaction_policy_name": compaction_policy_name,
        "compaction_policy_version": compaction_policy_version,
    }
    if tool_names:
        span_metadata["tool_names"] = tool_names
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions
    if compaction_cache_hits:
        span_metadata["compaction_cache_hits"] = compaction_cache_hits
    if compaction_cache_misses:
        span_metadata["compaction_cache_misses"] = compaction_cache_misses

    return services.merge_metadata(
        request_body,
        tags_to_add=["claude-tool-advertisement-compaction"],
        extra_fields={
            "claude_tool_advertisement_compaction_count": len(compaction_events),
            "claude_tool_advertisement_compaction_tool_names": tool_names,
            "claude_tool_advertisement_compaction_statuses": statuses,
            "claude_tool_advertisement_compaction_cc_versions": cc_versions,
            "claude_tool_advertisement_compaction_original_chars": original_chars,
            "claude_tool_advertisement_compaction_compacted_chars": compacted_chars,
            "claude_tool_advertisement_compaction_saved_chars": saved_chars,
            "claude_tool_advertisement_compaction_policy_name": compaction_policy_name,
            "claude_tool_advertisement_compaction_policy_version": (
                compaction_policy_version
            ),
            "claude_tool_advertisement_compaction_cache_hits": compaction_cache_hits,
            "claude_tool_advertisement_compaction_cache_misses": compaction_cache_misses,
            "claude_tool_advertisement_compaction_events": compaction_events,
            "langfuse_spans": [
                services.build_span(
                    name="claude.tool_advertisement_compaction",
                    metadata=span_metadata,
                )
            ],
        },
    )


_apply_claude_prompt_patch_manifest_to_text = (
    _prompt_replacement._apply_claude_prompt_patch_manifest_to_text
)
_add_claude_prompt_patch_logging_metadata = (
    _prompt_replacement._add_claude_prompt_patch_logging_metadata
)


async def _rewrite_claude_control_plane_text(
    text: str,
    *,
    cc_version: str,
    manifest: dict[str, Any],
    available_context: dict[str, str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    services = _get_services()
    updated_text = text
    override_events: list[dict[str, Any]] = []
    patch_events: list[dict[str, Any]] = []

    if (
        "# auto memory" in updated_text or "# Persistent Agent Memory" in updated_text
    ) and services.prompt.resolve_auto_memory_template_path(cc_version) is not None:
        try:
            updated_text, override_event = services.prompt.replace_auto_memory_section(
                updated_text,
                cc_version,
            )
        except Exception as exc:
            override_events.append(
                {
                    "id": "auto-memory",
                    "status": "failed",
                    "cc_version": cc_version,
                    "error": exc.__class__.__name__,
                }
            )
        else:
            if override_event is not None:
                override_events.append(override_event)

    try:
        updated_text, patch_events = services.prompt.apply_prompt_patch_manifest(
            updated_text,
            cc_version=cc_version,
            manifest=manifest,
        )
    except Exception as exc:
        patch_events.append(
            {
                "id": "manifest-load",
                "status": "failed",
                "cc_version": cc_version,
                "error": exc.__class__.__name__,
            }
        )

    if _CLAUDE_COMMONMARK_PROMPT_SENTENCE in updated_text:
        cache_key = (
            _AAWM_REFERENCE_IDENTIFIER_CACHE_KEY,
            available_context.get("session_id", ""),
            available_context.get("tenant", ""),
            available_context.get("agent", ""),
        )
        cache_hit, identifier_list = await _get_cached_aawm_dynamic_injection_result(
            cache_key
        )
        patch_event: dict[str, Any] = {
            "id": _AAWM_REFERENCE_IDENTIFIER_PATCH_ID,
            "cc_version": cc_version,
            "manifest_path": _CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH,
            "cache_status": "hit" if cache_hit else "miss",
            "context_keys": [
                context_key
                for context_key in ("session_id", "tenant", "agent")
                if available_context.get(context_key)
            ],
        }
        if not cache_hit:
            identifier_list = await services.context_query.get_reference_identifiers(
                tenant_id=available_context.get("tenant"),
                agent_id=available_context.get("agent"),
            )
            await _set_cached_aawm_dynamic_injection_result(cache_key, identifier_list)

        occurrences = updated_text.count(_CLAUDE_COMMONMARK_PROMPT_SENTENCE)
        replacement_identifiers = identifier_list or "none"
        updated_text = updated_text.replace(
            _CLAUDE_COMMONMARK_PROMPT_SENTENCE,
            _CLAUDE_COMMONMARK_PROMPT_IDENTIFIER_TEMPLATE.format(
                identifiers=replacement_identifiers
            ),
        )
        patch_event["status"] = "resolved" if identifier_list else "empty"
        patch_event["occurrences"] = occurrences
        patch_event["identifier_count"] = (
            len([name for name in replacement_identifiers.split(", ") if name])
            if identifier_list
            else 0
        )
        patch_events.append(patch_event)

    return updated_text, override_events, patch_events


async def _rewrite_claude_control_plane_in_value(
    value: Any,
    *,
    cc_version: str,
    manifest: dict[str, Any],
    available_context: dict[str, str],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                updated_text,
                override_events,
                patch_events,
            ) = await _rewrite_claude_control_plane_text(
                value["text"],
                cc_version=cc_version,
                manifest=manifest,
                available_context=available_context,
            )
            if not override_events and not patch_events:
                return value, [], []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, override_events, patch_events

        updated_dict: dict[str, Any] = {}
        combined_override_events: list[dict[str, Any]] = []
        combined_patch_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_override_events,
                child_patch_events,
            ) = await _rewrite_claude_control_plane_in_value(
                child,
                cc_version=cc_version,
                manifest=manifest,
                available_context=available_context,
            )
            updated_dict[key] = updated_child
            combined_override_events.extend(child_override_events)
            combined_patch_events.extend(child_patch_events)
            if updated_child is not child:
                changed = True
        return (
            updated_dict if changed else value,
            combined_override_events,
            combined_patch_events,
        )

    if isinstance(value, list):
        updated_list = []
        list_override_events: list[dict[str, Any]] = []
        list_patch_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_override_events,
                child_patch_events,
            ) = await _rewrite_claude_control_plane_in_value(
                child,
                cc_version=cc_version,
                manifest=manifest,
                available_context=available_context,
            )
            updated_list.append(updated_child)
            list_override_events.extend(child_override_events)
            list_patch_events.extend(child_patch_events)
            if updated_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            list_override_events,
            list_patch_events,
        )

    if isinstance(value, str):
        (
            updated_text,
            override_events,
            patch_events,
        ) = await _rewrite_claude_control_plane_text(
            value,
            cc_version=cc_version,
            manifest=manifest,
            available_context=available_context,
        )
        if not override_events and not patch_events:
            return value, [], []
        return updated_text, override_events, patch_events

    return value, [], []


async def _apply_claude_control_plane_rewrites_impl(  # noqa: PLR0915
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    services = _get_services()
    cc_version = billing_header_fields.get("cc_version")
    if not cc_version:
        return request_body, [], []

    span_started_at = services.now_utc()
    manifest_path = services.prompt.resolve_prompt_patch_manifest_path()
    manifest = services.prompt.load_prompt_patch_manifest(manifest_path)
    available_context = _build_aawm_context_for_anthropic_request(request_body)

    # RR-053 #4: rewrites target stable control-plane surfaces (system + first
    # user message). Full history is resent every turn; re-scanning all messages
    # is wasted CPU once early prompts have been rewritten.
    updated_body = dict(request_body)
    override_events: list[dict[str, Any]] = []
    patch_events: list[dict[str, Any]] = []
    changed = False

    if "system" in request_body:
        (
            updated_system,
            sys_overrides,
            sys_patches,
        ) = await _rewrite_claude_control_plane_in_value(
            request_body["system"],
            cc_version=cc_version,
            manifest=manifest,
            available_context=available_context,
        )
        if updated_system is not request_body["system"]:
            updated_body["system"] = updated_system
            changed = True
        override_events.extend(sys_overrides)
        patch_events.extend(sys_patches)

    messages = request_body.get("messages")
    if isinstance(messages, list) and messages:
        # Prefer the first user message even if a non-user item precedes it.
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            (
                updated_first,
                first_overrides,
                first_patches,
            ) = await _rewrite_claude_control_plane_in_value(
                message,
                cc_version=cc_version,
                manifest=manifest,
                available_context=available_context,
            )
            if updated_first is not message:
                updated_messages = list(messages)
                updated_messages[message_index] = updated_first
                updated_body["messages"] = updated_messages
                changed = True
            override_events.extend(first_overrides)
            patch_events.extend(first_patches)
            break

    if not changed and not override_events and not patch_events:
        # Still allow tool advertisement compaction below on original body.
        updated_body = request_body

    (
        updated_body,
        compaction_events,
    ) = _compact_claude_tool_advertisements_in_request_body(
        updated_body,
        cc_version=cc_version,
    )
    if not override_events and not patch_events and not compaction_events:
        return request_body, [], []

    if override_events:
        updated_body = services.prompt.add_override_metadata(
            updated_body,
            override_events,
        )
    if patch_events:
        updated_body = services.prompt.add_patch_metadata(
            updated_body,
            patch_events,
        )
    if compaction_events:
        updated_body = _add_claude_tool_advertisement_compaction_logging_metadata(
            updated_body,
            compaction_events,
        )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if not isinstance(span_descriptor, dict):
                    continue
                if span_descriptor.get("name") in {
                    "claude.system_prompt_override",
                    "claude.prompt_patch",
                    "claude.tool_advertisement_compaction",
                }:
                    span_descriptor["start_time"] = services.format_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = services.format_span_timestamp(
                        services.now_utc()
                    )

    return updated_body, override_events, patch_events


async def apply_claude_control_plane_rewrites_to_anthropic_request_body(
    request_body: dict[str, Any],
    billing_header_fields: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return await _get_standalone_rewriter().apply_rewrites(
        request_body,
        billing_header_fields,
    )


def _parse_aawm_directive_attributes(attrs_text: str) -> dict[str, str]:
    parsed_attrs: dict[str, str] = {}
    for match in _AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN.finditer(attrs_text):
        value = (
            match.group("double") or match.group("single") or match.group("bare") or ""
        ).strip()
        if value:
            parsed_attrs[match.group("key")] = value
    return parsed_attrs


def _get_aawm_directive_attrs_text(match: re.Match[str]) -> str:
    return (
        (
            match.group("html_attrs")
            or match.group("at_attrs")
            or match.group("line_attrs")
            or ""
        )
    ).strip()


def _get_nested_str_value(source: Any, path: tuple[str, ...]) -> Optional[str]:
    current = source
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


_iter_anthropic_text_fragments = _observability_metadata._iter_anthropic_text_fragments
_extract_claude_agent_and_tenant_from_request_body = (
    _observability_metadata._extract_claude_agent_and_tenant_from_request_body
)
_detect_claude_post_rewrite_context_files = (
    _observability_metadata._detect_claude_post_rewrite_context_files
)


def _extract_aawm_session_id_from_request_body(
    request_body: dict[str, Any]
) -> Optional[str]:
    for path in (
        ("litellm_metadata", "session_id"),
        ("metadata", "user_id", "session_id"),
        ("metadata", "session_id"),
        ("request", "session_id"),
        ("session_id",),
    ):
        value = _get_nested_str_value(request_body, path)
        if value:
            return value
    return None


def _build_aawm_context_for_anthropic_request(
    request_body: dict[str, Any]
) -> dict[str, str]:
    context: dict[str, str] = {}
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if agent:
        context["agent"] = agent
    if tenant:
        context["tenant"] = tenant
    session_id = _extract_aawm_session_id_from_request_body(request_body)
    if session_id:
        context["session_id"] = session_id
    return context


def add_claude_post_rewrite_context_file_logging_metadata(
    request_body: dict[str, Any]
) -> dict[str, Any]:
    return _get_services().add_context_file_metadata(request_body)


def _build_aawm_dynamic_injection_failure_text(proc_name: str) -> str:
    return _AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE.format(
        proc_name=proc_name or "unknown"
    )


def _build_aawm_context_grab_failure_text(name: str) -> str:
    return _AAWM_CONTEXT_GRAB_FAILURE_TEMPLATE.format(name=name or "unknown")


_append_aawm_dynamic_injection_dsn_query_params = (
    _context_query._append_aawm_dynamic_injection_dsn_query_params
)
close_aawm_dynamic_injection_pool = (
    _context_query.close_aawm_dynamic_injection_pool
)


def _get_aawm_dynamic_injection_application_name() -> str:
    return _context_query._get_aawm_dynamic_injection_application_name(
        get_first_secret_value=lambda names: _get_first_secret_value(names)
    )


def _get_aawm_dynamic_injection_server_settings() -> dict[str, str]:
    return _context_query._get_aawm_dynamic_injection_server_settings(
        get_application_name=_get_aawm_dynamic_injection_application_name
    )


async def _initialize_aawm_dynamic_injection_connection(conn: Any) -> None:
    await _context_query._initialize_aawm_dynamic_injection_connection(
        conn,
        get_application_name=_get_aawm_dynamic_injection_application_name,
    )


def _build_aawm_dynamic_injection_dsn() -> Optional[str]:
    return _context_query._build_aawm_dynamic_injection_dsn(
        get_first_secret_value=lambda names: _get_first_secret_value(names),
        normalize_sslmode=lambda value: _normalize_aawm_sslmode(value),
        get_application_name=_get_aawm_dynamic_injection_application_name,
    )


async def _get_aawm_dynamic_injection_pool() -> Any:
    return await _context_query._get_aawm_dynamic_injection_pool(
        build_dsn=_build_aawm_dynamic_injection_dsn,
        get_server_settings=_get_aawm_dynamic_injection_server_settings,
        initialize_connection=_initialize_aawm_dynamic_injection_connection,
    )


def _aawm_dynamic_injection_acquire_timeout_seconds() -> float:
    return _context_query._aawm_dynamic_injection_acquire_timeout_seconds()


async def _aawm_pool_fetch(pool: Any, query: str, *args: Any) -> Any:
    return await _context_query._aawm_pool_fetch(
        pool,
        query,
        *args,
        get_timeout=_aawm_dynamic_injection_acquire_timeout_seconds,
    )


async def _aawm_pool_fetchval(pool: Any, query: str, *args: Any) -> Any:
    return await _context_query._aawm_pool_fetchval(
        pool,
        query,
        *args,
        get_timeout=_aawm_dynamic_injection_acquire_timeout_seconds,
    )


async def _call_aawm_get_agent_memories(
    *, agent_name: str, tenant_id: str
) -> Optional[str]:
    return await _context_query._call_aawm_get_agent_memories(
        agent_name=agent_name,
        tenant_id=tenant_id,
        get_pool=_get_aawm_dynamic_injection_pool,
        pool_fetchval=_aawm_pool_fetchval,
    )


def _get_aawm_context_grab_proc_name() -> str:
    return _context_query._get_aawm_context_grab_proc_name()


def _get_aawm_context_grab_proc_name_for_logging() -> str:
    return _context_query._get_aawm_context_grab_proc_name_for_logging()


def _format_aawm_context_retrieved_at(retrieved_at: datetime) -> str:
    return (
        retrieved_at.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


async def _call_aawm_context_grab(
    *, name: str, tenant_id: Optional[str], agent_id: Optional[str]
) -> Optional[str]:
    return await _context_query._call_aawm_context_grab(
        name=name,
        tenant_id=tenant_id,
        agent_id=agent_id,
        get_pool=_get_aawm_dynamic_injection_pool,
        pool_fetch=_aawm_pool_fetch,
        get_proc_name=_get_aawm_context_grab_proc_name,
    )


async def _call_aawm_reference_identifier_list(
    *, tenant_id: Optional[str], agent_id: Optional[str]
) -> Optional[str]:
    return await _context_query._call_aawm_reference_identifier_list(
        tenant_id=tenant_id,
        agent_id=agent_id,
        get_pool=_get_aawm_dynamic_injection_pool,
        pool_fetch=_aawm_pool_fetch,
    )


def _resolve_aawm_dynamic_context_fields(
    proc_name: str, directive_attrs: dict[str, str]
) -> tuple[str, ...]:
    raw_ctx = directive_attrs.get("ctx")
    if raw_ctx:
        ctx_fields = tuple(
            field.strip() for field in raw_ctx.split(",") if field.strip()
        )
    else:
        ctx_fields = _AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS.get(proc_name, ())

    if not ctx_fields:
        raise ValueError("No AAWM context fields were provided")

    return tuple(dict.fromkeys(ctx_fields))


def _select_aawm_dynamic_context(
    *, context_fields: tuple[str, ...], available_context: dict[str, str]
) -> dict[str, str]:
    selected_context: dict[str, str] = {}
    for field_name in context_fields:
        if field_name not in {"agent", "tenant"}:
            raise ValueError(f"Unsupported AAWM context field: {field_name}")

        field_value = available_context.get(field_name)
        if not field_value:
            raise ValueError(f"Missing AAWM context field: {field_name}")
        selected_context[field_name] = field_value
    return selected_context


async def _resolve_aawm_dynamic_directive(
    directive_attrs: dict[str, str],
    available_context: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    services = _get_services()
    raw_proc_name = directive_attrs.get("p") or directive_attrs.get("proc") or "unknown"
    proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(raw_proc_name, raw_proc_name)
    context_fields = _resolve_aawm_dynamic_context_fields(proc_name, directive_attrs)
    selected_context = _select_aawm_dynamic_context(
        context_fields=context_fields,
        available_context=available_context,
    )

    event: dict[str, Any] = {
        "proc": proc_name,
        "context_keys": list(context_fields),
    }
    version = directive_attrs.get("v") or directive_attrs.get("version")
    if version:
        event["version"] = version
    scope = directive_attrs.get("s") or directive_attrs.get("scope")
    if scope:
        event["scope"] = scope

    if proc_name == _AAWM_AGENT_MEMORY_PROC_NAME:
        cache_key = (
            proc_name,
            selected_context.get("session_id", ""),
            selected_context["agent"],
            selected_context["tenant"],
        )
        cache_hit, injected_text = await _get_cached_aawm_dynamic_injection_result(
            cache_key
        )
        event["cache_status"] = "hit" if cache_hit else "miss"
        if not cache_hit:
            injected_text = await services.context_query.get_agent_memories(
                agent_name=selected_context["agent"],
                tenant_id=selected_context["tenant"],
            )
            await _set_cached_aawm_dynamic_injection_result(cache_key, injected_text)

        if injected_text is None:
            event["status"] = "empty"
            return _AAWM_NO_MEMORIES_TEMPLATE, event

        event["status"] = "resolved"
        event["output_chars"] = len(injected_text)
        return injected_text, event

    raise ValueError(f"Unsupported AAWM proc: {proc_name}")


async def _resolve_aawm_context_marker(
    name: str, available_context: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    appendix_entry, event = await _resolve_aawm_context_reference(
        name,
        available_context,
        placeholder_type="ctx_marker",
    )
    if appendix_entry is not None:
        return appendix_entry, event
    return _build_aawm_context_grab_failure_text(name), event


async def _resolve_aawm_context_reference(
    name: str,
    available_context: dict[str, str],
    *,
    placeholder_type: str,
) -> tuple[Optional[str], dict[str, Any]]:
    services = _get_services()
    proc_name = services.context_query.get_context_proc_name()
    context_keys = [
        context_key
        for context_key in ("session_id", "tenant", "agent")
        if available_context.get(context_key)
    ]
    cache_key = (
        proc_name,
        available_context.get("session_id", ""),
        available_context.get("tenant", ""),
        available_context.get("agent", ""),
        name,
    )
    event: dict[str, Any] = {
        "proc": proc_name,
        "status": "failed",
        "context_keys": context_keys,
        "context_name": name,
        "placeholder_type": placeholder_type,
    }
    cache_hit, cached_payload = await _get_cached_aawm_context_grab_result(cache_key)
    event["cache_status"] = "hit" if cache_hit else "miss"
    if not cache_hit:
        retrieved_at = _format_aawm_context_retrieved_at(services.now_utc())
        content = await services.context_query.get_context(
            name=name,
            tenant_id=available_context.get("tenant"),
            agent_id=available_context.get("agent"),
        )
        cached_payload = {
            "status": "empty",
            "retrieved_at": retrieved_at,
        }
        if content is not None:
            cached_payload["status"] = "resolved"
            cached_payload["text"] = content
        await _set_cached_aawm_context_grab_result(cache_key, cached_payload)

    if cached_payload is None:
        raise RuntimeError("AAWM context grab cache returned no payload")

    event["status"] = cached_payload.get("status", "failed")
    event["retrieved_at"] = cached_payload.get("retrieved_at")
    resolved_text = cached_payload.get("text")
    if event["status"] == "resolved" and resolved_text:
        event["output_chars"] = len(resolved_text)
        return (
            f"{resolved_text}\n~retrieved at: {cached_payload['retrieved_at']}",
            event,
        )

    return None, event


def _append_aawm_context_entries_to_text(text: str, entries: list[str]) -> str:
    if not entries:
        return text

    if not text:
        separator = ""
    else:
        trailing_newlines = len(text) - len(text.rstrip("\n"))
        if trailing_newlines >= 2:
            separator = ""
        elif trailing_newlines == 1:
            separator = "\n"
        else:
            separator = "\n\n"

    return text + separator + "\n\n".join(entries)


def _protect_escaped_aawm_context_markers(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        placeholder = _AAWM_ESCAPED_CONTEXT_MARKER_PLACEHOLDER.format(
            index=len(replacements)
        )
        replacements[placeholder] = f":#{match.group('name')}.ctx#:"
        return placeholder

    return _AAWM_ESCAPED_CONTEXT_MARKER_PATTERN.sub(_replace, text), replacements


def _restore_escaped_aawm_context_markers(
    text: str, replacements: dict[str, str]
) -> str:
    for placeholder, restored_marker in replacements.items():
        text = text.replace(placeholder, restored_marker)
    return text


async def _expand_aawm_context_markers_in_text(
    text: str, available_context: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    protected_text, escaped_markers = _protect_escaped_aawm_context_markers(text)
    matches = list(_AAWM_CONTEXT_MARKER_PATTERN.finditer(protected_text))
    if not matches:
        return (
            _restore_escaped_aawm_context_markers(protected_text, escaped_markers),
            [],
        )

    rebuilt_parts: list[str] = []
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    cursor = 0

    for match in matches:
        rebuilt_parts.append(protected_text[cursor : match.start()])
        name = match.group("name").strip()
        rebuilt_parts.append(name)
        if name and name not in seen_names:
            seen_names.add(name)
            ordered_names.append(name)
        cursor = match.end()

    rebuilt_parts.append(protected_text[cursor:])
    updated_text = _restore_escaped_aawm_context_markers(
        "".join(rebuilt_parts),
        escaped_markers,
    )
    if not ordered_names:
        return updated_text, []

    semaphore = asyncio.Semaphore(_get_services().context_query.max_parallel_queries)

    async def _resolve_with_limit(
        name: str,
    ) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                return await _resolve_aawm_context_marker(name, available_context)
            except Exception as exc:
                return (
                    _build_aawm_context_grab_failure_text(name),
                    {
                        "proc": _get_services().context_query.get_context_proc_name_for_logging(),
                        "status": "failed",
                        "error": exc.__class__.__name__,
                        "context_keys": [
                            context_key
                            for context_key in ("session_id", "tenant", "agent")
                            if available_context.get(context_key)
                        ],
                        "context_name": name,
                        "placeholder_type": "ctx_marker",
                    },
                )

    resolved_entries = await asyncio.gather(
        *(_resolve_with_limit(name) for name in ordered_names)
    )
    appendix_entries: list[str] = []
    context_events: list[dict[str, Any]] = []
    for appendix_entry, event in resolved_entries:
        appendix_entries.append(appendix_entry)
        context_events.append(event)

    return (
        _append_aawm_context_entries_to_text(updated_text, appendix_entries),
        context_events,
    )


class _AawmDispatchLookupBudget:
    """Request-wide budget for dispatch backtick/acronym context grabs (RR-053 #1).

    Per-text-node caps alone still multiply across many trusted text blocks in a
    single request. This counter tracks remaining lookup slots for the whole
    request walk so fan-out stays bounded even when system + first user contain
    many distinct references.
    """

    __slots__ = ("remaining", "seen_names")

    def __init__(self, max_lookups: int = _AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX):
        self.remaining = max(0, int(max_lookups))
        self.seen_names: set[str] = set()

    def reserve_slots(self, count: int) -> int:
        """Consume up to ``count`` slots; return how many were granted."""
        if count <= 0 or self.remaining <= 0:
            return 0
        granted = min(int(count), self.remaining)
        self.remaining -= granted
        return granted


def _extract_aawm_dispatch_context_references(
    text: str,
    *,
    max_references: int = _AAWM_DISPATCH_CONTEXT_REFERENCE_MAX,
) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        return []

    scan_text = (
        _AAWM_SYSTEM_REMINDER_BLOCK_PATTERN.sub("\n", text)
        if "</system-reminder>" in text
        else text
    )
    if "`" not in scan_text and re.search(r"\b[A-Z][A-Z0-9]{1,}\b", scan_text) is None:
        return []

    ordered_references: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    limit = max(0, int(max_references))
    for index, segment in enumerate(scan_text.split("```")):
        if index % 2 == 1:
            continue
        for match in _AAWM_DISPATCH_CONTEXT_REFERENCE_PATTERN.finditer(segment):
            if match.group("backtick") is not None:
                name = match.group("backtick").strip()
                placeholder_type = "dispatch_backtick"
            else:
                name = (match.group("acronym") or "").strip()
                placeholder_type = "dispatch_acronym"
                # Skip ubiquitous all-caps tokens that would thrash the pool.
                if name.upper() in _AAWM_DISPATCH_ACRONYM_STOPWORDS:
                    continue
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            ordered_references.append((name, placeholder_type))
            if len(ordered_references) >= limit:
                return ordered_references
    return ordered_references


async def _expand_aawm_dispatch_context_references_in_text(
    text: str,
    available_context: dict[str, str],
    *,
    request_lookup_budget: Optional[_AawmDispatchLookupBudget] = None,
) -> tuple[str, list[dict[str, Any]]]:
    # Request-wide budget (RR-053 #1) bounds total DB fan-out across all text
    # blocks; per-node max still applies when extracting from this node.
    per_node_max = _AAWM_DISPATCH_CONTEXT_REFERENCE_MAX
    if request_lookup_budget is not None:
        if request_lookup_budget.remaining <= 0:
            return text, []
        per_node_max = min(per_node_max, request_lookup_budget.remaining)

    ordered_references = _extract_aawm_dispatch_context_references(
        text,
        max_references=per_node_max,
    )
    if not ordered_references:
        return text, []

    # Deduplicate against names already resolved earlier in this request so we
    # do not spend budget on the same identifier twice across text nodes.
    if request_lookup_budget is not None:
        filtered: list[tuple[str, str]] = []
        for name, placeholder_type in ordered_references:
            if name in request_lookup_budget.seen_names:
                continue
            filtered.append((name, placeholder_type))
        ordered_references = filtered
        if not ordered_references:
            return text, []
        granted = request_lookup_budget.reserve_slots(len(ordered_references))
        if granted <= 0:
            return text, []
        ordered_references = ordered_references[:granted]
        for name, _placeholder_type in ordered_references:
            request_lookup_budget.seen_names.add(name)

    semaphore = asyncio.Semaphore(_get_services().context_query.max_parallel_queries)

    async def _resolve_with_limit(
        name: str,
        placeholder_type: str,
    ) -> tuple[Optional[str], dict[str, Any]]:
        async with semaphore:
            try:
                return await _resolve_aawm_context_reference(
                    name,
                    available_context,
                    placeholder_type=placeholder_type,
                )
            except Exception as exc:
                return (
                    None,
                    {
                        "proc": _get_services().context_query.get_context_proc_name_for_logging(),
                        "status": "failed",
                        "error": exc.__class__.__name__,
                        "context_keys": [
                            context_key
                            for context_key in ("session_id", "tenant", "agent")
                            if available_context.get(context_key)
                        ],
                        "context_name": name,
                        "placeholder_type": placeholder_type,
                    },
                )

    resolved_entries = await asyncio.gather(
        *(
            _resolve_with_limit(name, placeholder_type)
            for name, placeholder_type in ordered_references
        )
    )
    appendix_entries: list[str] = []
    context_events: list[dict[str, Any]] = []
    for appendix_entry, event in resolved_entries:
        if appendix_entry:
            appendix_entries.append(appendix_entry)
        context_events.append(event)

    return _append_aawm_context_entries_to_text(text, appendix_entries), context_events


async def _expand_aawm_dynamic_directives_in_text(
    text: str, available_context: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_AAWM_DYNAMIC_DIRECTIVE_PATTERN.finditer(text))
    if not matches:
        return text, []

    rebuilt_parts: list[str] = []
    injection_events: list[dict[str, Any]] = []
    cursor = 0

    for match in matches:
        rebuilt_parts.append(text[cursor : match.start()])
        directive_attrs = _parse_aawm_directive_attributes(
            _get_aawm_directive_attrs_text(match)
        )
        proc_name = directive_attrs.get("p") or directive_attrs.get("proc") or "unknown"
        try:
            replacement_text, event = await _resolve_aawm_dynamic_directive(
                directive_attrs,
                available_context,
            )
        except Exception as exc:
            normalized_proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(proc_name, proc_name)
            replacement_text = _build_aawm_dynamic_injection_failure_text(
                normalized_proc_name
            )
            requested_context_fields = []
            raw_ctx = directive_attrs.get("ctx")
            if raw_ctx:
                requested_context_fields = [
                    field.strip() for field in raw_ctx.split(",") if field.strip()
                ]
            event = {
                "proc": normalized_proc_name,
                "status": "failed",
                "error": exc.__class__.__name__,
                "context_keys": requested_context_fields
                or list(available_context.keys()),
            }
        rebuilt_parts.append(replacement_text)
        injection_events.append(event)
        cursor = match.end()

    rebuilt_parts.append(text[cursor:])
    return "".join(rebuilt_parts), injection_events


def _aawm_surface_is_trusted_for_injection(
    *,
    trusted_surface: bool,
    messages_trusted_first_user_only: bool,
    first_user_message_seen: bool,
    role: Optional[str],
) -> tuple[bool, bool]:
    """Return (surface_is_trusted, is_first_user_message)."""
    is_first_user_message = (
        messages_trusted_first_user_only
        and not first_user_message_seen
        and role == "user"
    )
    return bool(trusted_surface) or is_first_user_message, is_first_user_message


async def _expand_aawm_dynamic_directives_in_value(
    value: Any,
    available_context: dict[str, str],
    *,
    enable_dispatch_backtick_context: bool = False,
    enable_context_markers: bool = True,
    trusted_surface: bool = False,
    messages_trusted_first_user_only: bool = False,
    request_lookup_budget: Optional[_AawmDispatchLookupBudget] = None,
    _message_index: Optional[int] = None,
    _first_user_message_seen: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    allow_untrusted_injection = bool(trusted_surface)

    if isinstance(value, dict):
        # When walking messages[], only the first user message is a trusted
        # surface for marker/dispatch expansion (tool/web later turns are not).
        role = value.get("role") if isinstance(value.get("role"), str) else None
        (
            surface_is_trusted,
            is_first_user_message,
        ) = _aawm_surface_is_trusted_for_injection(
            trusted_surface=allow_untrusted_injection,
            messages_trusted_first_user_only=messages_trusted_first_user_only,
            first_user_message_seen=_first_user_message_seen,
            role=role,
        )

        if value.get("type") == "text" and isinstance(value.get("text"), str):
            original_text = value["text"]
            (
                updated_text,
                injection_events,
            ) = await _expand_aawm_dynamic_directives_in_text(
                original_text,
                available_context,
            )
            combined_events = list(injection_events)
            if enable_context_markers and surface_is_trusted:
                (
                    updated_text,
                    context_events,
                ) = await _expand_aawm_context_markers_in_text(
                    updated_text,
                    available_context,
                )
                combined_events.extend(context_events)
            if enable_dispatch_backtick_context and surface_is_trusted:
                (
                    updated_text,
                    dispatch_context_events,
                ) = await _expand_aawm_dispatch_context_references_in_text(
                    updated_text,
                    available_context,
                    request_lookup_budget=request_lookup_budget,
                )
                combined_events.extend(dispatch_context_events)
            if combined_events or updated_text != original_text:
                updated_value = dict(value)
                updated_value["text"] = updated_text
                return updated_value, combined_events
            return value, []

        updated_dict: dict[str, Any] = {}
        dict_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_events,
            ) = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
                enable_dispatch_backtick_context=enable_dispatch_backtick_context,
                enable_context_markers=enable_context_markers,
                trusted_surface=surface_is_trusted
                if key == "content"
                else trusted_surface,
                messages_trusted_first_user_only=False,
                request_lookup_budget=request_lookup_budget,
                _message_index=_message_index,
                _first_user_message_seen=_first_user_message_seen
                or is_first_user_message,
            )
            updated_dict[key] = updated_child
            dict_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), dict_events

    if isinstance(value, list):
        updated_list = []
        list_events: list[dict[str, Any]] = []
        changed = False
        first_user_message_seen = _first_user_message_seen
        for index, child in enumerate(value):
            child_role = (
                child.get("role")
                if isinstance(child, dict) and isinstance(child.get("role"), str)
                else None
            )
            (
                updated_child,
                child_events,
            ) = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
                enable_dispatch_backtick_context=enable_dispatch_backtick_context,
                enable_context_markers=enable_context_markers,
                trusted_surface=trusted_surface,
                messages_trusted_first_user_only=messages_trusted_first_user_only,
                request_lookup_budget=request_lookup_budget,
                _message_index=index
                if messages_trusted_first_user_only
                else _message_index,
                _first_user_message_seen=first_user_message_seen,
            )
            if (
                messages_trusted_first_user_only
                and not first_user_message_seen
                and child_role == "user"
            ):
                first_user_message_seen = True
            updated_list.append(updated_child)
            list_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), list_events

    if isinstance(value, str):
        # Plain system/user text fields (not content-block dicts).
        surface_is_trusted = allow_untrusted_injection
        original_text = value
        updated_text, injection_events = await _expand_aawm_dynamic_directives_in_text(
            original_text,
            available_context,
        )
        combined_events = list(injection_events)
        if enable_context_markers and surface_is_trusted:
            updated_text, context_events = await _expand_aawm_context_markers_in_text(
                updated_text,
                available_context,
            )
            combined_events.extend(context_events)
        if enable_dispatch_backtick_context and surface_is_trusted:
            (
                updated_text,
                dispatch_context_events,
            ) = await _expand_aawm_dispatch_context_references_in_text(
                updated_text,
                available_context,
                request_lookup_budget=request_lookup_budget,
            )
            combined_events.extend(dispatch_context_events)
        if combined_events or updated_text != original_text:
            return updated_text, combined_events
        return value, []

    return value, []


def _add_aawm_dynamic_injection_logging_metadata(
    request_body: dict[str, Any], injection_events: list[dict[str, Any]]
) -> dict[str, Any]:
    services = _get_services()
    proc_names = sorted(
        {
            event["proc"]
            for event in injection_events
            if isinstance(event.get("proc"), str) and event["proc"]
        }
    )
    failure_procs = sorted(
        {
            event["proc"]
            for event in injection_events
            if event.get("status") == "failed"
            and isinstance(event.get("proc"), str)
            and event["proc"]
        }
    )
    context_keys = sorted(
        {
            context_key
            for event in injection_events
            for context_key in event.get("context_keys", [])
            if isinstance(context_key, str) and context_key
        }
    )
    context_names = sorted(
        {
            context_name
            for event in injection_events
            for context_name in [event.get("context_name")]
            if isinstance(context_name, str) and context_name
        }
    )
    status_values = [
        event["status"]
        for event in injection_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cache_status_values = [
        event["cache_status"]
        for event in injection_events
        if isinstance(event.get("cache_status"), str) and event["cache_status"]
    ]
    cache_hit_count = sum(1 for status in cache_status_values if status == "hit")
    cache_miss_count = sum(1 for status in cache_status_values if status == "miss")

    tags_to_add = ["aawm-dynamic-injection"]
    tags_to_add.extend(f"aawm-proc:{proc_name}" for proc_name in proc_names)
    if failure_procs:
        tags_to_add.append("aawm-dynamic-injection-failed")

    span_metadata: dict[str, Any] = {
        "injection_count": len(injection_events),
        "failure_count": len(failure_procs),
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
    }
    if proc_names:
        span_metadata["procs"] = proc_names
    if context_keys:
        span_metadata["context_keys"] = context_keys
    if context_names:
        span_metadata["context_names"] = context_names

    return services.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "aawm_dynamic_injection_count": len(injection_events),
            "aawm_dynamic_injection_procs": proc_names,
            "aawm_dynamic_injection_failure_procs": failure_procs,
            "aawm_dynamic_injection_context_keys": context_keys,
            "aawm_dynamic_injection_context_names": context_names,
            "aawm_dynamic_injection_statuses": status_values,
            "aawm_dynamic_injection_cache_statuses": cache_status_values,
            "aawm_dynamic_injection_cache_hits": cache_hit_count,
            "aawm_dynamic_injection_cache_misses": cache_miss_count,
            "aawm_dynamic_injection_events": injection_events,
            "langfuse_spans": [
                services.build_span(
                    name="aawm.dynamic_injection",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _request_uses_aawm_dispatch_backtick_context(request_body: dict[str, Any]) -> bool:
    litellm_metadata = request_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        hooks = litellm_metadata.get("claude_persisted_output_hooks")
        if isinstance(hooks, list):
            for hook in hooks:
                if isinstance(hook, str) and hook.strip().lower() == "subagentstart":
                    return True

    for top_level_key in ("system", "messages"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            if any(
                marker in fragment for marker in _AAWM_SUBAGENTSTART_CONTEXT_MARKERS
            ):
                return True

    return False


async def _expand_aawm_dynamic_directives_impl(
    request_body: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    services = _get_services()
    available_context = _build_aawm_context_for_anthropic_request(request_body)
    enable_dispatch_backtick_context = _request_uses_aawm_dispatch_backtick_context(
        request_body
    )
    span_started_at = services.now_utc()
    updated_body = dict(request_body)
    injection_events: list[dict[str, Any]] = []
    changed = False
    # Shared across all trusted text blocks so RR-053 #1 bounds total fan-out
    # for the request, not merely per node.
    request_lookup_budget = _AawmDispatchLookupBudget(
        _AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX
    )

    # RR-053 #5 trust boundary:
    # - HTML/@@@ AAWM directives may appear anywhere (explicit, operator-authored).
    # - :#name.ctx#: markers and dispatch backtick/acronym grabs are restricted to
    #   trusted surfaces (system + first user message) so tool/web output cannot
    #   trigger arbitrary same-tenant content-store reads.
    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        updated_value, value_events = await _expand_aawm_dynamic_directives_in_value(
            request_body[top_level_key],
            available_context,
            enable_dispatch_backtick_context=enable_dispatch_backtick_context,
            enable_context_markers=True,
            trusted_surface=(top_level_key == "system"),
            messages_trusted_first_user_only=(top_level_key == "messages"),
            request_lookup_budget=request_lookup_budget,
        )
        if updated_value is not request_body[top_level_key]:
            updated_body[top_level_key] = updated_value
            changed = True
        if value_events:
            injection_events.extend(value_events)

    if not injection_events:
        return (updated_body if changed else request_body), []

    updated_body = _add_aawm_dynamic_injection_logging_metadata(
        updated_body,
        injection_events,
    )
    if changed:
        litellm_metadata = updated_body.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            langfuse_spans = litellm_metadata.get("langfuse_spans")
            if isinstance(langfuse_spans, list):
                for span_descriptor in langfuse_spans:
                    if (
                        isinstance(span_descriptor, dict)
                        and span_descriptor.get("name") == "aawm.dynamic_injection"
                    ):
                        span_descriptor[
                            "start_time"
                        ] = services.format_span_timestamp(span_started_at)
                        span_descriptor[
                            "end_time"
                        ] = services.format_span_timestamp(
                            services.now_utc()
                        )
    return updated_body, injection_events


async def expand_aawm_dynamic_directives_in_anthropic_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return await _get_standalone_rewriter().expand_dynamic_context(request_body)
