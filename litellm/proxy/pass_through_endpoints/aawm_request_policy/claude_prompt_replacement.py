"""Wave 6E extraction: Claude prompt-replacement transformations.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``.
Do not import ``llm_passthrough_endpoints`` at module scope.

This module owns:
- Claude auto-memory section replacement (context replacement templates)
- Claude prompt-patch manifest loading and application (delegated to
  ``prompt_replacement_engine`` for the shared matcher/loader/cache pipeline
  required by CFG-073)
- Metadata/logging helpers for both replacement families

It uses the Wave 6D ``observability_metadata`` module for shared metadata
primitives via explicit imports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _build_langfuse_span_descriptor,
    _format_langfuse_span_timestamp,
    _merge_litellm_metadata,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (  # CFG-073 shared engine
    apply_rules_to_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parents[4] / "context-replacement" / "claude-code"
)
_PACKAGED_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "aawm_claude_control_plane_data"
    / "claude-code"
)
_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = _REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR
_CLAUDE_AUTO_MEMORY_TEMPLATE_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "2.1.110" / "auto-memory-replacement.md"
)
_CLAUDE_PROMPT_PATCH_MANIFEST_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "prompt-patches" / "roman01la-2026-04-02.json"
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
_CLAUDE_AUTO_MEMORY_SECTION_PATTERN = _CLAUDE_MEMORY_SECTION_PATTERN
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
)


@dataclass(frozen=True, slots=True)
class ClaudePromptReplacementServices:
    resolve_auto_memory_template_path: Callable[[Optional[str]], Optional[Path]]
    resolve_prompt_patch_manifest_path: Callable[[], Path]
    load_prompt_patch_manifest: Callable[[Path], dict[str, Any]]
    replace_auto_memory_section: Callable[
        [str, str], tuple[str, Optional[dict[str, Any]]]
    ]
    apply_prompt_patch_manifest: Callable[
        ..., tuple[str, list[dict[str, Any]]]
    ]
    add_override_metadata: Callable[
        [dict[str, Any], list[dict[str, Any]]], dict[str, Any]
    ]
    add_patch_metadata: Callable[
        [dict[str, Any], list[dict[str, Any]]], dict[str, Any]
    ]

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

# Auto-memory template loading uses shared load_text_asset / _text_asset_cache;
# the local cache alias is kept for the legacy re-export contract.
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Version parsing / template resolution
# ---------------------------------------------------------------------------
# _CLAUDE_CC_VERSION_PATTERN is kept for the legacy re-export contract
# (prompt_replacement_legacy.py); actual version parsing is delegated to the
# shared _parse_cc_version in prompt_replacement_engine.


def _parse_claude_code_version(
    cc_version: Optional[str],
) -> Optional[tuple[int, int, int]]:
    from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (
        _parse_cc_version,
    )
    return _parse_cc_version(cc_version)


def _candidate_context_replacement_dirs() -> tuple[Path, ...]:
    return tuple(
        directory
        for directory in (
            _REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
            _PACKAGED_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
        )
        if directory.exists()
    )


def _resolve_context_replacement_file(
    relative_parts: tuple[str, ...],
) -> Optional[Path]:
    for base_dir in _candidate_context_replacement_dirs():
        candidate = base_dir.joinpath(*relative_parts)
        if candidate.exists():
            return candidate
    return None


def _resolve_claude_auto_memory_template_path(
    cc_version: Optional[str],
) -> Optional[Path]:
    parsed_version = _parse_claude_code_version(cc_version)
    if parsed_version is None:
        return None

    major, minor, patch = parsed_version
    min_major, min_minor, min_patch = _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION

    if (major, minor) != (min_major, min_minor):
        return None
    if patch < min_patch:
        return None

    return _resolve_context_replacement_file(
        ("2.1.110", "auto-memory-replacement.md")
    )


def _resolve_claude_prompt_patch_manifest_path() -> Path:
    manifest_path = _resolve_context_replacement_file(
        ("prompt-patches", "roman01la-2026-04-02.json")
    )
    if manifest_path is None:
        raise ValueError("Claude prompt patch manifest is missing")
    return manifest_path


# ---------------------------------------------------------------------------
# Template / manifest loading
# ---------------------------------------------------------------------------


def _load_claude_context_replacement_template(template_path: Path) -> str:
    from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (
        _text_asset_cache,
        load_text_asset,
    )
    cached_template = _claude_context_replacement_template_cache.get(template_path)
    if cached_template is not None:
        return cached_template

    try:
        template_text = load_text_asset(template_path, cache=_text_asset_cache)
    except ValueError as exc:
        if "empty" in str(exc):
            raise ValueError(
                f"Claude context replacement template is empty: {template_path}"
            ) from exc
        raise
    _claude_context_replacement_template_cache[template_path] = template_text
    return template_text


def _load_claude_prompt_patch_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate the Claude prompt-patch manifest.

    Returns the legacy dict contract.  Shared-engine loading, validation,
    and caching are delegated to ``_parse_rules_from_manifest`` (the
    already-parsed ``raw`` dict is fed directly to avoid a double file
    read) / ``_DEFAULT_MANIFEST_CACHE``.
    """
    cached_manifest = _claude_prompt_patch_manifest_cache.get(manifest_path)
    if cached_manifest is not None:
        return cached_manifest

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Invalid Claude prompt patch manifest: {manifest_path}"
        )
    patches_raw = raw.get("patches")
    if not isinstance(patches_raw, list) or not patches_raw:
        raise ValueError(
            f"Claude prompt patch manifest has no patches: {manifest_path}"
        )
    for pd in patches_raw:
        if not isinstance(pd, dict):
            raise ValueError(
                f"Invalid Claude prompt patch descriptor in {manifest_path}"
            )
        patch_id = pd.get("id")
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError(
                f"Claude prompt patch manifest is missing patch id in {manifest_path}"
            )
        before_text = pd.get("before")
        if not isinstance(before_text, str) or not before_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing before text for {patch_id}"
            )
        after_text = pd.get("after")
        if not isinstance(after_text, str) or not after_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing after text for {patch_id}"
            )

    from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (
        _DEFAULT_MANIFEST_CACHE,
        _parse_rules_from_manifest,
    )
    rules = _parse_rules_from_manifest(raw)
    _DEFAULT_MANIFEST_CACHE[manifest_path] = rules
    patches: list[dict[str, str]] = []
    for rule in rules:
        patches.append({
            "id": rule.id,
            "before": rule.before,
            "after": rule.after,
        })
    normalized_manifest: dict[str, Any] = {
        "patches": patches,
        "source": raw.get("source"),
    }
    _claude_prompt_patch_manifest_cache[manifest_path] = normalized_manifest
    return normalized_manifest


# ---------------------------------------------------------------------------
# Auto-memory replacement
# ---------------------------------------------------------------------------


def _extract_markdown_section(markdown_text: str, heading: str) -> str:
    section_pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)"
    )
    match = section_pattern.search(markdown_text)
    if match is None:
        raise ValueError(f"Missing Claude auto-memory section: {heading}")
    return match.group(0).rstrip()


def _render_claude_auto_memory_replacement(
    memory_section: str,
    cc_version: str,
    section_heading: str = "# auto memory",
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

    unresolved_placeholders = (
        _CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN.findall(rendered_text)
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
            (
                updated_child,
                child_events,
            ) = _replace_claude_system_prompt_override_in_value(
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
        list_combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_events,
            ) = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            list_combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), list_combined_events

    return value, []


def _add_claude_system_prompt_override_logging_metadata(
    request_body: dict[str, Any], override_events: list[dict[str, Any]]
) -> dict[str, Any]:
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

    return _merge_litellm_metadata(
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
                _build_langfuse_span_descriptor(
                    name="claude.system_prompt_override",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _replace_claude_system_prompt_in_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cc_version = billing_header_fields.get("cc_version")
    if not isinstance(cc_version, str) or not cc_version:
        return request_body, []
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
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, override_events


# ---------------------------------------------------------------------------
# Prompt-patch replacement  (CFG-073: routed through shared engine)
# ---------------------------------------------------------------------------


def _claude_manifest_to_rules(manifest: dict[str, Any]) -> list[Any]:
    """Convert a legacy Claude patch manifest dict to ReplacementRule list.

    Enriches the report-file rule with its pre-compiled regex.
    Uses the shared ``_DEFAULT_MANIFEST_CACHE`` to retrieve already-loaded
    engine rules and re-enriches per Claude-specific contract.
    """
    from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (
        ReplacementRule,
    )
    replacement_rules: list[Any] = []
    for patch in manifest["patches"]:
        before_pattern = None
        if patch["id"] == _CLAUDE_REPORT_FILE_PATCH_ID:
            before_pattern = _CLAUDE_REPORT_FILE_INSTRUCTION_PATTERN
        replacement_rules.append(
            ReplacementRule(
                id=patch["id"],
                before=patch["before"],
                after=patch["after"],
                before_pattern=before_pattern,
            )
        )
    return replacement_rules


def _apply_claude_prompt_patch_manifest_to_text(
    text: str,
    *,
    cc_version: str,
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply Claude prompt-patch rules via the shared CFG-073 engine.

    Accepts the legacy ``manifest`` dict and converts internally.
    """
    rules = _claude_manifest_to_rules(manifest)
    updated_text, result = apply_rules_to_text(
        text,
        rules,
        cc_version=cc_version,
    )

    patch_events: list[dict[str, Any]] = []
    for match in result.matched:
        event: dict[str, Any] = {
            "id": match.rule_id,
            "status": "resolved",
            "cc_version": cc_version,
            "manifest_path": _CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH,
            "occurrences": match.occurrences,
        }
        if match.match_types:
            event["match_types"] = match.match_types
        patch_events.append(event)

    for failure in result.failures:
        patch_events.append(failure)

    return updated_text, patch_events


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
        list_combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            list_combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), list_combined_events

    return value, []


def _add_claude_prompt_patch_logging_metadata(
    request_body: dict[str, Any], patch_events: list[dict[str, Any]]
) -> dict[str, Any]:
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

    return _merge_litellm_metadata(
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
                _build_langfuse_span_descriptor(
                    name="claude.prompt_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _apply_claude_prompt_patches_to_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, patch_events


# ---------------------------------------------------------------------------
# Build services
# ---------------------------------------------------------------------------


def build_claude_prompt_replacement_services() -> ClaudePromptReplacementServices:
    return ClaudePromptReplacementServices(
        resolve_auto_memory_template_path=_resolve_claude_auto_memory_template_path,
        resolve_prompt_patch_manifest_path=(
            _resolve_claude_prompt_patch_manifest_path
        ),
        load_prompt_patch_manifest=_load_claude_prompt_patch_manifest,
        replace_auto_memory_section=_replace_claude_auto_memory_section_in_text,
        apply_prompt_patch_manifest=_apply_claude_prompt_patch_manifest_to_text,
        add_override_metadata=_add_claude_system_prompt_override_logging_metadata,
        add_patch_metadata=_add_claude_prompt_patch_logging_metadata,
    )
