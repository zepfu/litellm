import asyncio
import importlib
import json
import re
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

if False:  # pragma: no cover
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints


_REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parents[3] / "context-replacement" / "claude-code"
)
_PACKAGED_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parent
    / "aawm_claude_control_plane_data"
    / "claude-code"
)
_CLAUDE_AUTO_MEMORY_TEMPLATE_LOGICAL_PATH = (
    "context-replacement/claude-code/2.1.110/auto-memory-replacement.md"
)
_CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH = (
    "context-replacement/claude-code/prompt-patches/roman01la-2026-04-02.json"
)
_CLAUDE_REPORT_FILE_PATCH_ID = "subagent-report-file-explicit-request"
_CLAUDE_REPORT_FILE_INSTRUCTION_PATTERN = re.compile(
    r"Do NOT\s+(?:Write|\$\{[^}\r\n]+\})\s+"
    r"report/summary/findings/analysis\s+\.md\s+files\.\s+"
    r"Return findings directly as your final assistant message"
    r"(?:\s+[—-]\s+the parent agent reads your text output,\s+not files you create)?"
    r"\.?",
)
_CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION = (2, 1, 110)
_CLAUDE_MEMORY_SECTION_PATTERN = re.compile(
    r"(?ms)^(?P<section_heading># (?:auto memory|Persistent Agent Memory))\n"
    r".*?(?=^# [^\n]+\n|\Z)"
)
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
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
_AAWM_REFERENCE_IDENTIFIER_LIST_QUERY = """
SELECT DISTINCT rc.name
FROM ag_catalog.raw_content rc
WHERE rc.role = 'reference'
  AND rc.valid_to IS NULL
  AND ($1::text IS NULL OR rc.tenant_id IS NULL OR rc.tenant_id = $1::text)
  AND ($2::text IS NULL OR rc.agent_id IS NULL OR rc.agent_id = $2::text)
  AND rc.name NOT IN (SELECT name FROM public.agents)
  AND rc.name NOT IN (SELECT name || '-instructions' FROM public.agents)
ORDER BY rc.name
"""
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
    r"(?<![\\`])`(?P<backtick>[^`\r\n]+?)`(?!`)"
    r"|(?P<acronym>\b[A-Z][A-Z0-9]{1,}\b)"
)
_AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_-]*)='
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s]+))'
)
_AAWM_SQL_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_CLAUDE_AGENT_TENANT_PATTERN = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project\b"
)
_CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MEMORY.md", "memory-md"),
    ("CLAUDE.md", "claude-md"),
)
_AAWM_AGENT_MEMORY_PROC_NAME = "get_agent_memories"
_AAWM_DYNAMIC_PROC_ALIASES = {"get_agent_memory": _AAWM_AGENT_MEMORY_PROC_NAME}
_AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS: dict[str, tuple[str, ...]] = {
    _AAWM_AGENT_MEMORY_PROC_NAME: ("agent", "tenant"),
}
_AAWM_CONTEXT_GRAB_PROC_NAME_ENV_VARS = (
    "AAWM_CONTEXT_GRAB_PROC_NAME",
    "AAWM_DYNAMIC_CONTEXT_GRAB_PROC_NAME",
)
_AAWM_CONTEXT_GRAB_DEFAULT_PROC_NAME = "tristore_search_exact"
_AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE = (
    "## AAWM Injection Status\n\n"
    'AAWM "{proc_name}" failed for this session.\n'
    "Alert the user or session orchestrator.\n"
)
_AAWM_CONTEXT_GRAB_FAILURE_TEMPLATE = (
    "IMPORTANT: context grab for {name} returned no results. immediately inform the opperator."
)
_AAWM_SUBAGENTSTART_CONTEXT_MARKERS = (
    "SubagentStart hook additional context:",
    "SubAgentStart hook additional context:",
)
_AAWM_SYSTEM_REMINDER_BLOCK_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>\n*",
    re.DOTALL,
)
_AAWM_NO_MEMORIES_TEMPLATE = (
    "# Memory Injection\n"
    "You have saved no memories as of yet.\n"
)
_AAWM_DB_HOST_ENV_VARS = (
    "AAWM_DB_HOST",
    "AAWM_POSTGRES_SERVER",
    "POSTGRES_SERVER",
    "PGHOST",
)
_AAWM_DB_PORT_ENV_VARS = (
    "AAWM_DB_PORT",
    "AAWM_POSTGRES_PORT",
    "POSTGRES_PORT",
    "PGPORT",
)
_AAWM_DB_USER_ENV_VARS = (
    "AAWM_DB_USER",
    "AAWM_POSTGRES_USER",
    "POSTGRES_USER",
    "PGUSER",
)
_AAWM_DB_PASSWORD_ENV_VARS = (
    "AAWM_DB_PASSWORD",
    "AAWM_DB_PWD",
    "AAWM_POSTGRES_PASSWORD",
    "AAWM_POSTGRES_PWD",
    "POSTGRES_PASSWORD",
    "POSTGRES_PWD",
    "PGPASSWORD",
)
_AAWM_DB_NAME_ENV_VARS = (
    "AAWM_DB_NAME",
    "AAWM_POSTGRES_DATABASE",
    "POSTGRES_DATABASE",
    "PGDATABASE",
)
_AAWM_DB_SSLMODE_ENV_VARS = (
    "AAWM_DB_SSLMODE",
    "AAWM_POSTGRES_SSLMODE",
    "POSTGRES_SSLMODE",
    "PGSSLMODE",
)
_AAWM_DB_SSL_BOOL_ENV_VARS = (
    "AAWM_DB_SSL",
    "AAWM_POSTGRES_SSL",
    "POSTGRES_SSL",
)
_AAWM_DB_URL_ENV_VARS = (
    "AAWM_DB_URL",
    "AAWM_DATABASE_URL",
    "AAWM_POSTGRES_URL",
)
_AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS = 15.0
_AAWM_DYNAMIC_INJECTION_POOL_MIN_SIZE = 1
_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE = 4
_AAWM_DYNAMIC_INJECTION_COMMAND_TIMEOUT_SECONDS = 10
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_aawm_dynamic_injection_cache: dict[tuple[str, str, str, str], tuple[float, Optional[str]]] = {}
_aawm_dynamic_injection_cache_lock = asyncio.Lock()
_aawm_context_grab_cache: dict[
    tuple[str, str, str, str, str], tuple[float, dict[str, str]]
] = {}
_aawm_context_grab_cache_lock = asyncio.Lock()
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}


def _get_aawm_dynamic_injection_cache_ttl_seconds() -> float:
    raw_value = _clean_secret_string(_lp().get_secret_str("AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS"))
    if not raw_value:
        return _AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return _AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS


async def _get_cached_aawm_dynamic_injection_result(
    cache_key: tuple[str, str, str, str],
) -> tuple[bool, Optional[str]]:
    async with _aawm_dynamic_injection_cache_lock:
        cached_entry = _aawm_dynamic_injection_cache.get(cache_key)
        if cached_entry is None:
            return False, None

        expires_at, cached_value = cached_entry
        if expires_at < monotonic():
            _aawm_dynamic_injection_cache.pop(cache_key, None)
            return False, None
        return True, cached_value


async def _set_cached_aawm_dynamic_injection_result(
    cache_key: tuple[str, str, str, str],
    injected_text: Optional[str],
) -> None:
    ttl_seconds = _get_aawm_dynamic_injection_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return

    async with _aawm_dynamic_injection_cache_lock:
        _aawm_dynamic_injection_cache[cache_key] = (
            monotonic() + ttl_seconds,
            injected_text,
        )


async def _get_cached_aawm_context_grab_result(
    cache_key: tuple[str, str, str, str, str],
) -> tuple[bool, Optional[dict[str, str]]]:
    async with _aawm_context_grab_cache_lock:
        cached_entry = _aawm_context_grab_cache.get(cache_key)
        if cached_entry is None:
            return False, None

        expires_at, cached_value = cached_entry
        if expires_at < monotonic():
            _aawm_context_grab_cache.pop(cache_key, None)
            return False, None
        return True, dict(cached_value)


async def _set_cached_aawm_context_grab_result(
    cache_key: tuple[str, str, str, str, str],
    cached_payload: dict[str, str],
) -> None:
    ttl_seconds = _get_aawm_dynamic_injection_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return

    async with _aawm_context_grab_cache_lock:
        _aawm_context_grab_cache[cache_key] = (
            monotonic() + ttl_seconds,
            dict(cached_payload),
        )


def _lp():
    return importlib.import_module(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
    )


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    get_secret_str = _lp().get_secret_str
    for secret_name in secret_names:
        value = _clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None


def _normalize_aawm_sslmode(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_secret_string(value)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "require"
    if lowered in {"0", "false", "no", "off"}:
        return "disable"
    return cleaned


def _candidate_context_replacement_dirs() -> tuple[Path, ...]:
    return tuple(
        directory
        for directory in (
            _REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
            _PACKAGED_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
        )
        if directory.exists()
    )


def _resolve_context_replacement_file(relative_parts: tuple[str, ...]) -> Optional[Path]:
    for base_dir in _candidate_context_replacement_dirs():
        candidate = base_dir.joinpath(*relative_parts)
        if candidate.exists():
            return candidate
    return None


def _parse_claude_code_version(cc_version: Optional[str]) -> Optional[tuple[int, int, int]]:
    if not cc_version:
        return None

    match = _CLAUDE_CC_VERSION_PATTERN.match(cc_version.strip())
    if match is None:
        return None

    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _resolve_claude_auto_memory_template_path(cc_version: Optional[str]) -> Optional[Path]:
    parsed_version = _parse_claude_code_version(cc_version)
    if parsed_version is None:
        return None

    major, minor, patch = parsed_version
    min_major, min_minor, min_patch = _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION

    if (major, minor) != (min_major, min_minor):
        return None
    if patch < min_patch:
        return None

    return _resolve_context_replacement_file(("2.1.110", "auto-memory-replacement.md"))


def _load_claude_context_replacement_template(template_path: Path) -> str:
    cached_template = _claude_context_replacement_template_cache.get(template_path)
    if cached_template is not None:
        return cached_template

    template_text = template_path.read_text(encoding="utf-8").strip()
    if not template_text:
        raise ValueError(f"Claude context replacement template is empty: {template_path}")

    cached_template = template_text + "\n"
    _claude_context_replacement_template_cache[template_path] = cached_template
    return cached_template


def _resolve_claude_prompt_patch_manifest_path() -> Path:
    manifest_path = _resolve_context_replacement_file(
        ("prompt-patches", "roman01la-2026-04-02.json")
    )
    if manifest_path is None:
        raise ValueError("Claude prompt patch manifest is missing")
    return manifest_path


def _load_claude_prompt_patch_manifest(template_path: Path) -> dict[str, Any]:
    cached_manifest = _claude_prompt_patch_manifest_cache.get(template_path)
    if cached_manifest is not None:
        return cached_manifest

    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid Claude prompt patch manifest: {template_path}")

    patches = manifest.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError(f"Claude prompt patch manifest has no patches: {template_path}")

    normalized_patches: list[dict[str, str]] = []
    for patch_descriptor in patches:
        if not isinstance(patch_descriptor, dict):
            raise ValueError(
                f"Invalid Claude prompt patch descriptor in {template_path}"
            )
        patch_id = patch_descriptor.get("id")
        before_text = patch_descriptor.get("before")
        after_text = patch_descriptor.get("after")
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError(
                f"Claude prompt patch manifest is missing patch id in {template_path}"
            )
        if not isinstance(before_text, str) or not before_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing before text for {patch_id}"
            )
        if not isinstance(after_text, str) or not after_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing after text for {patch_id}"
            )
        normalized_patches.append(
            {
                "id": patch_id,
                "before": before_text,
                "after": after_text,
            }
        )

    normalized_manifest = {
        "source": manifest.get("source"),
        "patches": normalized_patches,
    }
    _claude_prompt_patch_manifest_cache[template_path] = normalized_manifest
    return normalized_manifest


def _extract_markdown_section(markdown_text: str, heading: str) -> str:
    section_pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)"
    )
    match = section_pattern.search(markdown_text)
    if match is None:
        raise ValueError(f"Missing Claude auto-memory section: {heading}")
    return match.group(0).rstrip()


def _render_claude_auto_memory_replacement(
    memory_section: str, cc_version: str, section_heading: str
) -> tuple[str, str]:
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None:
        raise ValueError(
            f"Unsupported Claude Code version for auto-memory override: {cc_version}"
        )

    template_text = _load_claude_context_replacement_template(template_path)
    rendered_text = template_text
    if "{{TYPES_XML_BLOCK}}" in rendered_text:
        types_match = _CLAUDE_TYPES_XML_BLOCK_PATTERN.search(memory_section)
        if types_match is None:
            raise ValueError("Missing Claude auto-memory <types> block")
        rendered_text = rendered_text.replace(
            "{{TYPES_XML_BLOCK}}", types_match.group(0).rstrip()
        )

    section_placeholders = {
        "{{WHAT_NOT_TO_SAVE_SECTION}}": "What NOT to save in memory",
        "{{BEFORE_RECOMMENDING_SECTION}}": "Before recommending from memory",
        "{{MEMORY_AND_PERSISTENCE_SECTION}}": "Memory and other forms of persistence",
    }
    for placeholder, heading in section_placeholders.items():
        if placeholder in rendered_text:
            rendered_text = rendered_text.replace(
                placeholder, _extract_markdown_section(memory_section, heading)
            )

    unresolved_placeholders = _CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN.findall(
        rendered_text
    )
    if unresolved_placeholders:
        raise ValueError(
            "Unresolved Claude context replacement placeholders: "
            + ", ".join(sorted(unresolved_placeholders))
        )

    if section_heading != "# auto memory":
        rendered_text = rendered_text.replace("# auto memory", section_heading, 1)

    return rendered_text.rstrip() + "\n", _CLAUDE_AUTO_MEMORY_TEMPLATE_LOGICAL_PATH


def _replace_claude_auto_memory_section_in_text(
    text: str, cc_version: str
) -> tuple[str, Optional[dict[str, Any]]]:
    if "# auto memory" not in text and "# Persistent Agent Memory" not in text:
        return text, None

    section_match = _CLAUDE_MEMORY_SECTION_PATTERN.search(text)
    if section_match is None:
        return text, None

    section_heading = section_match.group("section_heading")
    replacement_text, logical_path = _render_claude_auto_memory_replacement(
        section_match.group(0),
        cc_version,
        section_heading,
    )
    replacement_event: dict[str, Any] = {
        "id": "auto-memory",
        "status": "resolved",
        "cc_version": cc_version,
        "template_path": logical_path,
        "section_heading": section_heading,
        "output_chars": len(replacement_text),
    }
    return (
        text[: section_match.start()] + replacement_text + text[section_match.end() :],
        replacement_event,
    )


def _replace_claude_system_prompt_override_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            if (
                "# auto memory" not in value["text"]
                and "# Persistent Agent Memory" not in value["text"]
            ):
                return value, []
            try:
                updated_text, event = _replace_claude_auto_memory_section_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "auto-memory",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]

            if event is None:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, [event]

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    return value, []


def _add_claude_system_prompt_override_logging_metadata(
    request_body: dict[str, Any], override_events: list[dict[str, Any]]
) -> dict[str, Any]:
    lp = _lp()
    override_ids = sorted(
        {
            event["id"]
            for event in override_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in override_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in override_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in override_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    template_paths = sorted(
        {
            event["template_path"]
            for event in override_events
            if isinstance(event.get("template_path"), str) and event["template_path"]
        }
    )

    tags_to_add = ["claude-system-prompt-override"]
    tags_to_add.extend(
        f"claude-system-prompt-override:{override_id}" for override_id in override_ids
    )
    if failure_ids:
        tags_to_add.append("claude-system-prompt-override-failed")

    span_metadata: dict[str, Any] = {
        "override_count": len(override_events),
        "failure_count": len(failure_ids),
    }
    if override_ids:
        span_metadata["override_ids"] = override_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return lp._merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_system_prompt_override_count": len(override_events),
            "claude_system_prompt_override_ids": override_ids,
            "claude_system_prompt_override_failure_ids": failure_ids,
            "claude_system_prompt_override_statuses": statuses,
            "claude_system_prompt_override_cc_versions": cc_versions,
            "claude_system_prompt_override_template_paths": template_paths,
            "claude_system_prompt_override_events": override_events,
            "langfuse_spans": [
                lp._build_langfuse_span_descriptor(
                    name="claude.system_prompt_override",
                    metadata=span_metadata,
                )
            ],
        },
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

            compacted_child, child_description_count, child_dropped_key_count = (
                _compact_claude_tool_schema_value(child)
            )
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
            compacted_child, child_description_count, child_dropped_key_count = (
                _compact_claude_tool_schema_value(child)
            )
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
    original_chars = _json_compact_char_count(tool)
    updated_tool = dict(tool)
    changed = False
    top_level_description_compacted = False
    schema_description_count = 0
    schema_dropped_key_count = 0

    description = tool.get("description")
    if isinstance(description, str):
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

    input_schema = tool.get("input_schema")
    if isinstance(input_schema, dict):
        compacted_schema, schema_description_count, schema_dropped_key_count = (
            _compact_claude_tool_schema_value(input_schema)
        )
        if compacted_schema is not input_schema:
            updated_tool["input_schema"] = compacted_schema
            changed = True

    if not changed:
        return tool, None

    compacted_chars = _json_compact_char_count(updated_tool)
    return updated_tool, {
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
    }


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
    lp = _lp()
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

    span_metadata: dict[str, Any] = {
        "tool_count": len(compaction_events),
        "original_chars": original_chars,
        "compacted_chars": compacted_chars,
        "saved_chars": saved_chars,
    }
    if tool_names:
        span_metadata["tool_names"] = tool_names
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return lp._merge_litellm_metadata(
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
            "claude_tool_advertisement_compaction_events": compaction_events,
            "langfuse_spans": [
                lp._build_langfuse_span_descriptor(
                    name="claude.tool_advertisement_compaction",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _apply_claude_prompt_patch_manifest_to_text(
    text: str,
    *,
    cc_version: str,
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    updated_text = text
    patch_events: list[dict[str, Any]] = []

    for patch_descriptor in manifest["patches"]:
        patch_id = patch_descriptor["id"]
        before_text = patch_descriptor["before"]
        after_text = patch_descriptor["after"]
        occurrences = updated_text.count(before_text)
        match_types: list[str] = []
        if occurrences:
            updated_text = updated_text.replace(before_text, after_text)
            match_types.append("exact")

        if patch_id == _CLAUDE_REPORT_FILE_PATCH_ID:
            updated_text, pattern_occurrences = (
                _CLAUDE_REPORT_FILE_INSTRUCTION_PATTERN.subn(
                    after_text,
                    updated_text,
                )
            )
            if pattern_occurrences:
                occurrences += pattern_occurrences
                match_types.append("pattern")

        if not occurrences:
            continue

        event: dict[str, Any] = {
            "id": patch_id,
            "status": "resolved",
            "cc_version": cc_version,
            "manifest_path": _CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH,
            "occurrences": occurrences,
        }
        if match_types:
            event["match_types"] = match_types
        patch_events.append(event)

    return updated_text, patch_events


async def _rewrite_claude_control_plane_text(
    text: str,
    *,
    cc_version: str,
    manifest: dict[str, Any],
    available_context: dict[str, str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    updated_text = text
    override_events: list[dict[str, Any]] = []
    patch_events: list[dict[str, Any]] = []

    if (
        ("# auto memory" in updated_text or "# Persistent Agent Memory" in updated_text)
        and _resolve_claude_auto_memory_template_path(cc_version) is not None
    ):
        try:
            updated_text, override_event = _replace_claude_auto_memory_section_in_text(
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
        updated_text, patch_events = _apply_claude_prompt_patch_manifest_to_text(
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
            resolver = getattr(
                _lp(),
                "_call_aawm_reference_identifier_list",
                _call_aawm_reference_identifier_list,
            )
            identifier_list = await resolver(
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
            updated_text, override_events, patch_events = await _rewrite_claude_control_plane_text(
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
            updated_child, child_override_events, child_patch_events = await _rewrite_claude_control_plane_in_value(
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
        combined_override_events: list[dict[str, Any]] = []
        combined_patch_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_override_events, child_patch_events = await _rewrite_claude_control_plane_in_value(
                child,
                cc_version=cc_version,
                manifest=manifest,
                available_context=available_context,
            )
            updated_list.append(updated_child)
            combined_override_events.extend(child_override_events)
            combined_patch_events.extend(child_patch_events)
            if updated_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            combined_override_events,
            combined_patch_events,
        )

    if isinstance(value, str):
        updated_text, override_events, patch_events = await _rewrite_claude_control_plane_text(
            value,
            cc_version=cc_version,
            manifest=manifest,
            available_context=available_context,
        )
        if not override_events and not patch_events:
            return value, [], []
        return updated_text, override_events, patch_events

    return value, [], []



async def apply_claude_control_plane_rewrites_to_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lp = _lp()
    cc_version = billing_header_fields.get("cc_version")
    if not cc_version:
        return request_body, [], []

    span_started_at = datetime.now(timezone.utc)
    manifest_path = _resolve_claude_prompt_patch_manifest_path()
    manifest = _load_claude_prompt_patch_manifest(manifest_path)
    available_context = _build_aawm_context_for_anthropic_request(request_body)
    updated_body, override_events, patch_events = await _rewrite_claude_control_plane_in_value(
        request_body,
        cc_version=cc_version,
        manifest=manifest,
        available_context=available_context,
    )

    if not isinstance(updated_body, dict):
        return request_body, [], []

    updated_body, compaction_events = _compact_claude_tool_advertisements_in_request_body(
        updated_body,
        cc_version=cc_version,
    )
    if not override_events and not patch_events and not compaction_events:
        return request_body, [], []

    if override_events:
        updated_body = _add_claude_system_prompt_override_logging_metadata(
            updated_body,
            override_events,
        )
    if patch_events:
        updated_body = _add_claude_prompt_patch_logging_metadata(
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
                    span_descriptor["start_time"] = lp._format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = lp._format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )

    return updated_body, override_events, patch_events


def replace_claude_system_prompt_in_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lp = _lp()
    cc_version = billing_header_fields.get("cc_version")
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None or "system" not in request_body:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    updated_system, override_events = _replace_claude_system_prompt_override_in_value(
        request_body["system"],
        cc_version,
    )
    if not override_events:
        return request_body, []

    updated_body["system"] = updated_system
    updated_body = _add_claude_system_prompt_override_logging_metadata(
        updated_body,
        override_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.system_prompt_override"
                ):
                    span_descriptor["start_time"] = lp._format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = lp._format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, override_events


def _apply_claude_prompt_patches_in_text(
    text: str, cc_version: str
) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = _resolve_claude_prompt_patch_manifest_path()
    manifest = _load_claude_prompt_patch_manifest(manifest_path)
    return _apply_claude_prompt_patch_manifest_to_text(
        text,
        cc_version=cc_version,
        manifest=manifest,
    )


def _replace_claude_prompt_patches_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            try:
                updated_text, patch_events = _apply_claude_prompt_patches_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "manifest-load",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]
            if not patch_events:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, patch_events

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    if isinstance(value, str):
        try:
            updated_text, patch_events = _apply_claude_prompt_patches_in_text(
                value, cc_version
            )
        except Exception as exc:
            return value, [
                {
                    "id": "manifest-load",
                    "status": "failed",
                    "cc_version": cc_version,
                    "error": exc.__class__.__name__,
                }
            ]
        if not patch_events:
            return value, []
        return updated_text, patch_events

    return value, []


def _add_claude_prompt_patch_logging_metadata(
    request_body: dict[str, Any], patch_events: list[dict[str, Any]]
) -> dict[str, Any]:
    lp = _lp()
    patch_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in patch_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in patch_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    manifest_paths = sorted(
        {
            event["manifest_path"]
            for event in patch_events
            if isinstance(event.get("manifest_path"), str) and event["manifest_path"]
        }
    )
    total_occurrences = sum(
        event["occurrences"]
        for event in patch_events
        if isinstance(event.get("occurrences"), int)
    )

    tags_to_add = ["claude-prompt-patch"]
    tags_to_add.extend(f"claude-prompt-patch:{patch_id}" for patch_id in patch_ids)
    if failure_ids:
        tags_to_add.append("claude-prompt-patch-failed")

    span_metadata: dict[str, Any] = {
        "patch_count": len(patch_events),
        "replacement_count": total_occurrences,
        "failure_count": len(failure_ids),
    }
    if patch_ids:
        span_metadata["patch_ids"] = patch_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return lp._merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_prompt_patch_count": len(patch_events),
            "claude_prompt_patch_replacement_count": total_occurrences,
            "claude_prompt_patch_ids": patch_ids,
            "claude_prompt_patch_failure_ids": failure_ids,
            "claude_prompt_patch_statuses": statuses,
            "claude_prompt_patch_cc_versions": cc_versions,
            "claude_prompt_patch_manifest_paths": manifest_paths,
            "claude_prompt_patch_events": patch_events,
            "langfuse_spans": [
                lp._build_langfuse_span_descriptor(
                    name="claude.prompt_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def apply_claude_prompt_patches_to_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lp = _lp()
    cc_version = billing_header_fields.get("cc_version")
    if not cc_version:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body, patch_events = _replace_claude_prompt_patches_in_value(
        request_body,
        cc_version,
    )
    if not patch_events:
        return request_body, []

    if not isinstance(updated_body, dict):
        return request_body, []

    updated_body = _add_claude_prompt_patch_logging_metadata(
        updated_body,
        patch_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.prompt_patch"
                ):
                    span_descriptor["start_time"] = lp._format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = lp._format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, patch_events


def _parse_aawm_directive_attributes(attrs_text: str) -> dict[str, str]:
    parsed_attrs: dict[str, str] = {}
    for match in _AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN.finditer(attrs_text):
        value = (
            match.group("double")
            or match.group("single")
            or match.group("bare")
            or ""
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


def _iter_anthropic_text_fragments(value: Any):
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            yield value["text"]
            return
        for child in value.values():
            yield from _iter_anthropic_text_fragments(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from _iter_anthropic_text_fragments(child)


def _extract_claude_agent_and_tenant_from_request_body(
    request_body: dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    for top_level_key in ("messages", "system"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            match = _CLAUDE_AGENT_TENANT_PATTERN.search(fragment)
            if match is None:
                continue
            agent = match.group("agent").strip()
            tenant = match.group("tenant").strip()
            if agent and tenant:
                return agent, tenant
    return None, None


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


def _detect_claude_post_rewrite_context_files(
    request_body: dict[str, Any]
) -> list[str]:
    present_files: list[str] = []
    seen_files: set[str] = set()

    for top_level_key in ("system", "messages"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            for marker, _tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
                if marker in seen_files:
                    continue
                if marker in fragment:
                    present_files.append(marker)
                    seen_files.add(marker)

    return present_files


def add_claude_post_rewrite_context_file_logging_metadata(
    request_body: dict[str, Any]
) -> dict[str, Any]:
    lp = _lp()
    present_files = _detect_claude_post_rewrite_context_files(request_body)
    if not present_files:
        return request_body

    tags_to_add = ["claude-post-rewrite-context-file-present"]
    for marker, tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
        if marker in present_files:
            tags_to_add.append(f"claude-post-rewrite-context-file:{tag_suffix}")

    return lp._merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_post_rewrite_context_files_present": present_files,
            "claude_post_rewrite_context_file_count": len(present_files),
        },
    )


def _build_aawm_dynamic_injection_failure_text(proc_name: str) -> str:
    return _AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE.format(
        proc_name=proc_name or "unknown"
    )


def _build_aawm_context_grab_failure_text(name: str) -> str:
    return _AAWM_CONTEXT_GRAB_FAILURE_TEMPLATE.format(name=name or "unknown")


def _build_aawm_dynamic_injection_dsn() -> Optional[str]:
    host = _get_first_secret_value(_AAWM_DB_HOST_ENV_VARS)
    port = _get_first_secret_value(_AAWM_DB_PORT_ENV_VARS)
    user = _get_first_secret_value(_AAWM_DB_USER_ENV_VARS)
    password = _get_first_secret_value(_AAWM_DB_PASSWORD_ENV_VARS)
    database = _get_first_secret_value(_AAWM_DB_NAME_ENV_VARS)
    sslmode = _normalize_aawm_sslmode(
        _get_first_secret_value(_AAWM_DB_SSLMODE_ENV_VARS)
        or _get_first_secret_value(_AAWM_DB_SSL_BOOL_ENV_VARS)
    )

    has_component_config = any((host, port, user, password, database, sslmode))
    if has_component_config:
        if not host or not user or not database:
            return None

        credentials = quote(user, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        dsn = (
            f"postgresql://{credentials}@{host}:{port or '5432'}/"
            f"{quote(database, safe='')}"
        )
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return dsn

    return _get_first_secret_value(_AAWM_DB_URL_ENV_VARS)


async def _get_aawm_dynamic_injection_pool() -> Any:
    global _aawm_dynamic_injection_pool

    if _aawm_dynamic_injection_pool is not None:
        return _aawm_dynamic_injection_pool

    async with _aawm_dynamic_injection_pool_lock:
        if _aawm_dynamic_injection_pool is not None:
            return _aawm_dynamic_injection_pool

        dsn = _build_aawm_dynamic_injection_dsn()
        if not dsn:
            raise RuntimeError("AAWM dynamic injection database configuration is missing")

        try:
            asyncpg = importlib.import_module("asyncpg")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AAWM dynamic injection requires asyncpg to be installed"
            ) from exc

        _aawm_dynamic_injection_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=_AAWM_DYNAMIC_INJECTION_POOL_MIN_SIZE,
            max_size=_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE,
            command_timeout=_AAWM_DYNAMIC_INJECTION_COMMAND_TIMEOUT_SECONDS,
        )
        return _aawm_dynamic_injection_pool


async def _call_aawm_get_agent_memories(
    *, agent_name: str, tenant_id: str
) -> Optional[str]:
    pool = await _get_aawm_dynamic_injection_pool()
    result = await pool.fetchval(
        "SELECT get_agent_memories($1, $2)",
        agent_name,
        tenant_id,
    )
    if isinstance(result, str):
        stripped_result = result.strip()
        if stripped_result:
            return stripped_result
    return None


def _get_aawm_context_grab_proc_name() -> str:
    proc_name = (
        _get_first_secret_value(_AAWM_CONTEXT_GRAB_PROC_NAME_ENV_VARS)
        or _AAWM_CONTEXT_GRAB_DEFAULT_PROC_NAME
    )
    if _AAWM_SQL_IDENTIFIER_PATTERN.fullmatch(proc_name) is None:
        raise RuntimeError("AAWM context grab proc name is invalid")
    return proc_name


def _get_aawm_context_grab_proc_name_for_logging() -> str:
    try:
        return _get_aawm_context_grab_proc_name()
    except Exception:
        return "unknown"


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
    proc_name = _get_aawm_context_grab_proc_name()
    pool = await _get_aawm_dynamic_injection_pool()
    rows = await pool.fetch(
        f"SELECT content FROM {proc_name}($1, $2, $3)",
        name,
        tenant_id,
        agent_id,
    )
    contents: list[str] = []
    for row in rows:
        content: Optional[str] = None
        if isinstance(row, dict):
            content = row.get("content")
        elif hasattr(row, "get"):
            content = row.get("content")
        if isinstance(content, str):
            stripped_content = content.strip()
            if stripped_content:
                contents.append(stripped_content)
    if contents:
        return "\n\n".join(contents)
    return None


async def _call_aawm_reference_identifier_list(
    *, tenant_id: Optional[str], agent_id: Optional[str]
) -> Optional[str]:
    pool = await _get_aawm_dynamic_injection_pool()
    rows = await pool.fetch(
        _AAWM_REFERENCE_IDENTIFIER_LIST_QUERY,
        tenant_id,
        agent_id,
    )
    identifier_names: list[str] = []
    for row in rows:
        identifier_name: Optional[str] = None
        if isinstance(row, dict):
            identifier_name = row.get("name")
        elif hasattr(row, "get"):
            identifier_name = row.get("name")
        if isinstance(identifier_name, str):
            stripped_identifier_name = identifier_name.strip()
            if stripped_identifier_name:
                identifier_names.append(stripped_identifier_name)
    if identifier_names:
        return ", ".join(identifier_names)
    return None


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
    lp = _lp()
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
            resolver = getattr(lp, "_call_aawm_get_agent_memories", _call_aawm_get_agent_memories)
            injected_text = await resolver(
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
    lp = _lp()
    proc_name = _get_aawm_context_grab_proc_name()
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
        retrieved_at = _format_aawm_context_retrieved_at(datetime.now(timezone.utc))
        resolver = getattr(lp, "_call_aawm_context_grab", _call_aawm_context_grab)
        content = await resolver(
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
        return _restore_escaped_aawm_context_markers(protected_text, escaped_markers), []

    rebuilt_parts: list[str] = []
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    cursor = 0

    for match in matches:
        rebuilt_parts.append(protected_text[cursor:match.start()])
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

    semaphore = asyncio.Semaphore(_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE)

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
                        "proc": _get_aawm_context_grab_proc_name_for_logging(),
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

    return _append_aawm_context_entries_to_text(updated_text, appendix_entries), context_events


def _extract_aawm_dispatch_context_references(
    text: str,
) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        return []

    scan_text = _AAWM_SYSTEM_REMINDER_BLOCK_PATTERN.sub("\n", text)
    if "`" not in scan_text and re.search(r"\b[A-Z][A-Z0-9]{1,}\b", scan_text) is None:
        return []

    ordered_references: list[tuple[str, str]] = []
    seen_names: set[str] = set()
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
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            ordered_references.append((name, placeholder_type))
    return ordered_references


async def _expand_aawm_dispatch_context_references_in_text(
    text: str, available_context: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    ordered_references = _extract_aawm_dispatch_context_references(text)
    if not ordered_references:
        return text, []

    semaphore = asyncio.Semaphore(_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE)

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
                        "proc": _get_aawm_context_grab_proc_name_for_logging(),
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
        rebuilt_parts.append(text[cursor:match.start()])
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
                "context_keys": requested_context_fields or list(available_context.keys()),
            }
        rebuilt_parts.append(replacement_text)
        injection_events.append(event)
        cursor = match.end()

    rebuilt_parts.append(text[cursor:])
    return "".join(rebuilt_parts), injection_events


async def _expand_aawm_dynamic_directives_in_value(
    value: Any,
    available_context: dict[str, str],
    *,
    enable_dispatch_backtick_context: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            original_text = value["text"]
            updated_text, injection_events = await _expand_aawm_dynamic_directives_in_text(
                original_text,
                available_context,
            )
            updated_text, context_events = await _expand_aawm_context_markers_in_text(
                updated_text,
                available_context,
            )
            combined_events = injection_events + context_events
            if enable_dispatch_backtick_context:
                (
                    updated_text,
                    dispatch_context_events,
                ) = await _expand_aawm_dispatch_context_references_in_text(
                    updated_text,
                    available_context,
                )
                combined_events.extend(dispatch_context_events)
            if combined_events or updated_text != original_text:
                updated_value = dict(value)
                updated_value["text"] = updated_text
                return updated_value, combined_events
            return value, []

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
                enable_dispatch_backtick_context=enable_dispatch_backtick_context,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
                enable_dispatch_backtick_context=enable_dispatch_backtick_context,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    return value, []


def _add_aawm_dynamic_injection_logging_metadata(
    request_body: dict[str, Any], injection_events: list[dict[str, Any]]
) -> dict[str, Any]:
    lp = _lp()
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

    return lp._merge_litellm_metadata(
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
                lp._build_langfuse_span_descriptor(
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
            if any(marker in fragment for marker in _AAWM_SUBAGENTSTART_CONTEXT_MARKERS):
                return True

    return False


async def expand_aawm_dynamic_directives_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lp = _lp()
    available_context = _build_aawm_context_for_anthropic_request(request_body)
    enable_dispatch_backtick_context = _request_uses_aawm_dispatch_backtick_context(
        request_body
    )
    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    injection_events: list[dict[str, Any]] = []
    changed = False

    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        updated_value, value_events = await _expand_aawm_dynamic_directives_in_value(
            request_body[top_level_key],
            available_context,
            enable_dispatch_backtick_context=enable_dispatch_backtick_context,
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
                        span_descriptor["start_time"] = lp._format_langfuse_span_timestamp(
                            span_started_at
                        )
                        span_descriptor["end_time"] = lp._format_langfuse_span_timestamp(
                            datetime.now(timezone.utc)
                        )
    return updated_body, injection_events
