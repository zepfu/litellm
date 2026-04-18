import asyncio
import importlib
import json
import re
from datetime import datetime, timezone
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
_CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION = (2, 1, 110)
_CLAUDE_MEMORY_SECTION_PATTERN = re.compile(
    r"(?ms)^(?P<section_heading># (?:auto memory|Persistent Agent Memory))\n"
    r".*?(?=^# [^\n]+\n|\Z)"
)
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
_AAWM_DYNAMIC_DIRECTIVE_PATTERN = re.compile(
    r"<!--\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<html_attrs>.*?)\s*-->"
    r"|@@@\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<at_attrs>.*?)\s*@@@"
    r"|^[ \t]*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<line_attrs>[^\r\n]+?)\s*$",
    re.DOTALL | re.MULTILINE,
)
_AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_-]*)='
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s]+))'
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
_AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE = (
    "## AAWM Injection Status\n\n"
    'AAWM "{proc_name}" failed for this session.\n'
    "Alert the user or session orchestrator.\n"
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
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}


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
    types_match = _CLAUDE_TYPES_XML_BLOCK_PATTERN.search(memory_section)
    if types_match is None:
        raise ValueError("Missing Claude auto-memory <types> block")

    rendered_text = template_text
    rendered_text = rendered_text.replace(
        "{{TYPES_XML_BLOCK}}", types_match.group(0).rstrip()
    )
    rendered_text = rendered_text.replace(
        "{{WHAT_NOT_TO_SAVE_SECTION}}",
        _extract_markdown_section(memory_section, "What NOT to save in memory"),
    )
    rendered_text = rendered_text.replace(
        "{{BEFORE_RECOMMENDING_SECTION}}",
        _extract_markdown_section(memory_section, "Before recommending from memory"),
    )
    rendered_text = rendered_text.replace(
        "{{MEMORY_AND_PERSISTENCE_SECTION}}",
        _extract_markdown_section(memory_section, "Memory and other forms of persistence"),
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
    updated_text = text
    patch_events: list[dict[str, Any]] = []

    for patch_descriptor in manifest["patches"]:
        before_text = patch_descriptor["before"]
        if before_text not in updated_text:
            continue

        after_text = patch_descriptor["after"]
        occurrences = updated_text.count(before_text)
        updated_text = updated_text.replace(before_text, after_text)
        patch_events.append(
            {
                "id": patch_descriptor["id"],
                "status": "resolved",
                "cc_version": cc_version,
                "manifest_path": _CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH,
                "occurrences": occurrences,
            }
        )

    return updated_text, patch_events


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


def _build_aawm_context_for_anthropic_request(
    request_body: dict[str, Any]
) -> dict[str, str]:
    context: dict[str, str] = {}
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if agent:
        context["agent"] = agent
    if tenant:
        context["tenant"] = tenant
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
            min_size=1,
            max_size=4,
            command_timeout=10,
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
        resolver = getattr(lp, "_call_aawm_get_agent_memories", _call_aawm_get_agent_memories)
        injected_text = await resolver(
            agent_name=selected_context["agent"],
            tenant_id=selected_context["tenant"],
        )
        if injected_text is None:
            event["status"] = "empty"
            return _AAWM_NO_MEMORIES_TEMPLATE, event

        event["status"] = "resolved"
        event["output_chars"] = len(injected_text)
        return injected_text, event

    raise ValueError(f"Unsupported AAWM proc: {proc_name}")


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
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            updated_text, injection_events = await _expand_aawm_dynamic_directives_in_text(
                value["text"],
                available_context,
            )
            if injection_events:
                updated_value = dict(value)
                updated_value["text"] = updated_text
                return updated_value, injection_events
            return value, []

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
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
    status_values = [
        event["status"]
        for event in injection_events
        if isinstance(event.get("status"), str) and event["status"]
    ]

    tags_to_add = ["aawm-dynamic-injection"]
    tags_to_add.extend(f"aawm-proc:{proc_name}" for proc_name in proc_names)
    if failure_procs:
        tags_to_add.append("aawm-dynamic-injection-failed")

    span_metadata: dict[str, Any] = {
        "injection_count": len(injection_events),
        "failure_count": len(failure_procs),
    }
    if proc_names:
        span_metadata["procs"] = proc_names
    if context_keys:
        span_metadata["context_keys"] = context_keys

    return lp._merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "aawm_dynamic_injection_count": len(injection_events),
            "aawm_dynamic_injection_procs": proc_names,
            "aawm_dynamic_injection_failure_procs": failure_procs,
            "aawm_dynamic_injection_context_keys": context_keys,
            "aawm_dynamic_injection_statuses": status_values,
            "aawm_dynamic_injection_events": injection_events,
            "langfuse_spans": [
                lp._build_langfuse_span_descriptor(
                    name="aawm.dynamic_injection",
                    metadata=span_metadata,
                )
            ],
        },
    )


async def expand_aawm_dynamic_directives_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lp = _lp()
    available_context = _build_aawm_context_for_anthropic_request(request_body)
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
        )
        if value_events:
            updated_body[top_level_key] = updated_value
            injection_events.extend(value_events)
            changed = True

    if not injection_events:
        return request_body, []

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
