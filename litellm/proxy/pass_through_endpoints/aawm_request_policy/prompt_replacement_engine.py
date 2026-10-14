"""CFG-073 shared prompt-replacement engine.

Provides rule loading, eligibility filtering, text replacement, and
per-API-surface adapters for all destination providers and source TUIs.

Matcher concepts (exact + regex replacement), manifest caching, and
version parsing are generalized from the existing Claude prompt-patch
implementation in ``claude_prompt_replacement.py``.

Rules are loaded from JSON manifest files.  Each rule specifies text to
find and replace, with optional filters for source TUI, minimum CC
version, and prompt field.  Global rules (no filters) are always
eligible.  Idempotent under repeated invocation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplacementRule:
    """A single replacement rule parsed from a manifest."""

    id: str
    before: str = ""  # exact text to find; empty if pattern-only
    after: str = ""  # replacement text
    before_pattern: Optional[re.Pattern] = None  # regex alternative
    source_tuis: Optional[tuple[str, ...]] = None  # e.g. ("codex", "claude-code")
    cc_version_min: Optional[tuple[int, int, int]] = None
    prompt_fields: Optional[tuple[str, ...]] = None  # e.g. ("system", "developer")


@dataclass(frozen=True, slots=True)
class ReplacementMatch:
    """Outcome for a single matched rule."""

    rule_id: str
    occurrences: int
    match_types: list[str]  # "exact", "pattern"


@dataclass
class ReplacementResult:
    """Aggregated outcome of applying rules."""

    matched: list[ReplacementMatch] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def matched_ids(self) -> list[str]:
        return [m.rule_id for m in self.matched]

    @property
    def total_replacements(self) -> int:
        return sum(m.occurrences for m in self.matched)


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

_CC_VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
)


def _parse_cc_version(
    cc_version: Optional[str],
) -> Optional[tuple[int, int, int]]:
    """Parse a dotted version string into (major, minor, patch).

    Returns None for missing, empty, or unparseable input.
    """
    if not cc_version:
        return None
    match = _CC_VERSION_PATTERN.match(cc_version.strip())
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _cc_version_gte(
    version: Optional[tuple[int, int, int]],
    minimum: Optional[tuple[int, int, int]],
) -> bool:
    """Return True when *version* >= *minimum*, or *minimum* is None."""
    if minimum is None:
        return True
    if version is None:
        return False
    return version >= minimum


# ---------------------------------------------------------------------------
# Rule eligibility
# ---------------------------------------------------------------------------


def _rule_is_eligible(
    rule: ReplacementRule,
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[tuple[int, int, int]] = None,
    prompt_field: Optional[str] = None,
) -> bool:
    """A rule is eligible when every set filter matches the request context.

    - ``source_tuis``: request TUI must be in the set (or rule has no filter).
    - ``cc_version_min``: request version >= minimum (or rule has no filter).
    - ``prompt_fields``: current field must be in the set (or rule has no filter).
    """
    if rule.source_tuis is not None:
        if source_tui is None or source_tui not in rule.source_tuis:
            return False
    if not _cc_version_gte(cc_version, rule.cc_version_min):
        return False
    if rule.prompt_fields is not None:
        if prompt_field is None or prompt_field not in rule.prompt_fields:
            return False
    return True


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _validate_legacy_manifest(raw: dict[str, Any], legacy_label: str) -> None:
    patches = raw.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError(f"Claude prompt patch manifest has no patches: {legacy_label}")
    for descriptor in patches:
        if not isinstance(descriptor, dict):
            raise ValueError(f"Invalid Claude prompt patch descriptor in {legacy_label}")
        rule_id = descriptor.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(f"Claude prompt patch manifest is missing patch id in {legacy_label}")
        for key in ("before", "after"):
            value = descriptor.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Claude prompt patch manifest is missing {key} text for {rule_id}")


def _parse_rules_from_manifest(
    raw: dict[str, Any],
    *,
    legacy_label: Optional[str] = None,
    patterns: Optional[dict[str, re.Pattern[str]]] = None,
) -> list[ReplacementRule]:
    """Parse a raw manifest dict into ReplacementRule objects.

    Accepts either ``"rules"`` (new shared format) or ``"patches"``
    (legacy Claude prompt-patch format) at the top level.
    """
    if legacy_label is not None:
        _validate_legacy_manifest(raw, legacy_label)
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list):
        raw_rules = raw.get("patches")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("Manifest has no 'rules' or 'patches' list")

    rules: list[ReplacementRule] = []
    for descriptor in raw_rules:
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid rule descriptor in manifest")

        rule_id = descriptor.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError("Manifest rule missing id")

        before_text = descriptor.get("before")
        if not isinstance(before_text, str):
            before_text = ""

        if "after" not in descriptor:
            raise ValueError(
                f"Rule {rule_id} missing 'after' key"
            )
        after_text = descriptor["after"]
        if not isinstance(after_text, str):
            raise ValueError(
                f"Rule {rule_id} 'after' must be a string,"
                f" got {type(after_text).__name__}"
            )

        before_pattern_raw = descriptor.get("before_pattern")
        before_pattern: Optional[re.Pattern] = None
        if isinstance(before_pattern_raw, str) and before_pattern_raw:
            before_pattern = re.compile(before_pattern_raw)

        if not after_text and not before_text and not before_pattern:
            raise ValueError(
                f"Rule {rule_id} needs at least one of"
                f" 'before', 'before_pattern', or non-empty 'after'"
            )
        if not before_text and not before_pattern:
            raise ValueError(
                f"Rule {rule_id} needs at least one of 'before' or 'before_pattern'"
            )

        source_tuis_raw = descriptor.get("source_tuis")
        source_tuis: Optional[tuple[str, ...]] = None
        if isinstance(source_tuis_raw, list):
            filtered = tuple(t for t in source_tuis_raw if isinstance(t, str) and t)
            if filtered:
                source_tuis = filtered

        cc_version_min_raw = descriptor.get("cc_version_min")
        cc_version_min: Optional[tuple[int, int, int]] = None
        if isinstance(cc_version_min_raw, str) and cc_version_min_raw:
            cc_version_min = _parse_cc_version(cc_version_min_raw)

        prompt_fields_raw = descriptor.get("prompt_fields")
        prompt_fields: Optional[tuple[str, ...]] = None
        if isinstance(prompt_fields_raw, list):
            filtered = tuple(
                f for f in prompt_fields_raw if isinstance(f, str) and f
            )
            if filtered:
                prompt_fields = filtered

        if patterns and rule_id in patterns:
            before_pattern = patterns[rule_id]

        rules.append(
            ReplacementRule(
                id=rule_id,
                before=before_text,
                after=after_text,
                before_pattern=before_pattern,
                source_tuis=source_tuis,
                cc_version_min=cc_version_min,
                prompt_fields=prompt_fields,
            )
        )

    return rules


class LoadedManifest(dict[str, Any]):
    """Legacy-compatible document and its single compiled rule representation."""

    def __init__(self, document: dict[str, Any], rules: list[ReplacementRule]):
        super().__init__(document)
        self.replacement_rules = rules


_DEFAULT_MANIFEST_CACHE: dict[Path, dict[str, Any]] = {}
_text_asset_cache: dict[Path, str] = {}


def rules_from_manifest(
    document: dict[str, Any],
    *,
    patterns: Optional[dict[str, re.Pattern[str]]] = None,
) -> list[ReplacementRule]:
    if isinstance(document, LoadedManifest):
        return document.replacement_rules
    return _parse_rules_from_manifest(document, patterns=patterns)


def load_manifest_document(
    manifest_path: Path,
    *,
    legacy: bool = False,
    patterns: Optional[dict[str, re.Pattern[str]]] = None,
    cache: Optional[dict[Path, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Read, validate and compile each manifest once, with one cache owner."""
    effective_cache = _DEFAULT_MANIFEST_CACHE if cache is None else cache
    cached = effective_cache.get(manifest_path)
    if cached is not None:
        return cached
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        label = "Claude prompt patch manifest" if legacy else "manifest"
        raise ValueError(f"Invalid {label}: {manifest_path}")
    rules = _parse_rules_from_manifest(
        raw, legacy_label=str(manifest_path) if legacy else None, patterns=patterns,
    )
    if legacy:
        document = {
            "patches": [{"id": r.id, "before": r.before, "after": r.after} for r in rules],
            "source": raw.get("source"),
        }
    else:
        document = raw
    loaded = LoadedManifest(document, rules)
    effective_cache[manifest_path] = loaded
    return loaded


def load_manifest(manifest_path: Path) -> list[ReplacementRule]:
    """Load a manifest through the common document and compiled-rule cache."""
    return rules_from_manifest(load_manifest_document(manifest_path))


def load_text_asset(
    asset_path: Path,
    *,
    cache: Optional[dict[Path, str]] = None,
) -> str:
    """Load and cache a plain-text asset file.

    Returns the file content (rstripped + trailing newline).
    Raises ValueError for empty files.
    """
    effective_cache = cache if cache is not None else _text_asset_cache
    cached = effective_cache.get(asset_path)
    if cached is not None:
        return cached
    text = asset_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text asset is empty: {asset_path}")
    cached_text = text + "\n"
    effective_cache[asset_path] = cached_text
    return cached_text


def load_manifests(
    manifest_paths: list[Path],
    *,
    cache: Optional[dict[Path, dict[str, Any]]] = None,
) -> list[ReplacementRule]:
    """Load rules through the same cache used by the legacy Claude facade."""
    return [
        rule
        for path in manifest_paths
        for rule in rules_from_manifest(load_manifest_document(path, cache=cache))
    ]


def extract_markdown_section(markdown_text: str, heading: str) -> str:
    match = re.search(rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)", markdown_text)
    if match is None:
        raise ValueError(f"Missing Claude auto-memory section: {heading}")
    return match.group(0).rstrip()


def render_section_template(
    section: str,
    template: str,
    *,
    blocks: dict[str, re.Pattern[str]],
    sections: dict[str, str],
    placeholder_pattern: re.Pattern[str],
) -> str:
    """Render a section template while retaining captured source sections."""
    for placeholder, pattern in blocks.items():
        if placeholder in template:
            match = pattern.search(section)
            if match is None:
                raise ValueError("Missing Claude auto-memory <types> block")
            template = template.replace(placeholder, match.group(0).rstrip())
    for placeholder, heading in sections.items():
        if placeholder in template:
            template = template.replace(placeholder, extract_markdown_section(section, heading))
    unresolved = placeholder_pattern.findall(template)
    if unresolved:
        raise ValueError("Unresolved Claude context replacement placeholders: " + ", ".join(sorted(unresolved)))
    return template


def replace_template_section(
    text: str,
    pattern: re.Pattern[str],
    render: Callable[[re.Match[str]], str],
) -> tuple[str, Optional[re.Match[str]], Optional[str]]:
    """Match and substitute one eligible section without touching its neighbors."""
    match = pattern.search(text)
    if match is None:
        return text, None, None
    replacement = render(match)
    return text[:match.start()] + replacement + text[match.end():], match, replacement


# ---------------------------------------------------------------------------
# Core engine: single text field
# ---------------------------------------------------------------------------


def _contained_in_output(
    text: str, start: int, end: int, output: str,
) -> bool:
    """Protect a match only when a complete replacement contains that match."""
    if not output:
        return False
    position = text.find(output, max(0, end - len(output)), start + len(output))
    while position >= 0 and position <= start:
        if end <= position + len(output):
            return True
        position = text.find(output, position + 1, start + len(output))
    return False


def _replace_exact_idempotently(
    text: str, *, before: str, after: str,
) -> tuple[str, int]:
    if before == after:
        return text, 0
    parts: list[str] = []
    position = 0
    occurrences = 0
    while True:
        start = text.find(before, position)
        if start < 0:
            break
        end = start + len(before)
        parts.append(text[position:start])
        if _contained_in_output(text, start, end, after):
            parts.append(text[start:end])
        else:
            parts.append(after)
            occurrences += 1
        position = end
    parts.append(text[position:])
    return "".join(parts), occurrences


def _replace_pattern_idempotently(
    text: str, *, pattern: re.Pattern[str], replacement: str,
) -> tuple[str, int]:
    parts: list[str] = []
    position = 0
    occurrences = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        output = match.expand(replacement)
        parts.append(text[position:start])
        if text[start:end] == output or _contained_in_output(text, start, end, output):
            parts.append(text[start:end])
        else:
            parts.append(output)
            occurrences += 1
        position = end
    parts.append(text[position:])
    return "".join(parts), occurrences


def apply_rules_to_text(
    text: str,
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
    prompt_field: Optional[str] = None,
) -> tuple[str, ReplacementResult]:
    """Apply eligible rules to a single text field.

    Returns ``(updated_text, result)``.  Unmatched text is unchanged.
    Idempotent: running again on the updated text produces no additional
    matches.
    """
    parsed_cc_version = _parse_cc_version(cc_version)

    updated_text = text
    result = ReplacementResult()

    for rule in rules:
        if not _rule_is_eligible(
            rule,
            source_tui=source_tui,
            cc_version=parsed_cc_version,
            prompt_field=prompt_field,
        ):
            continue

        try:
            occurrences = 0
            match_types: list[str] = []

            if rule.before:
                updated_text, count = _replace_exact_idempotently(
                    updated_text,
                    before=rule.before,
                    after=rule.after,
                )
                if count:
                    occurrences += count
                    match_types.append("exact")

            if rule.before_pattern:
                updated_text, pattern_count = _replace_pattern_idempotently(
                    updated_text,
                    pattern=rule.before_pattern,
                    replacement=rule.after,
                )
                if pattern_count:
                    occurrences += pattern_count
                    match_types.append("pattern")

            if occurrences:
                result.matched.append(
                    ReplacementMatch(
                        rule_id=rule.id,
                        occurrences=occurrences,
                        match_types=match_types,
                    )
                )
        except Exception as exc:
            result.failures.append(
                {
                    "id": rule.id,
                    "status": "failed",
                    "error": exc.__class__.__name__,
                }
            )

    return updated_text, result


# ---------------------------------------------------------------------------
# Per-API-surface adapters
# ---------------------------------------------------------------------------

_RESPONSES_SYSTEM_DEVELOPER_ROLES = frozenset({"system", "developer"})


def apply_rules_to_responses_request(
    body: dict[str, Any],
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
) -> tuple[dict[str, Any], ReplacementResult]:
    """Apply rules to a Responses API request body.

    Touches ``instructions`` (str) and ``input`` messages whose role is
    ``system`` or ``developer`` (str content or ``input_text`` parts).
    """
    updated_body = dict(body)
    combined = ReplacementResult()

    # -- instructions --------------------------------------------------
    instructions = updated_body.get("instructions")
    if isinstance(instructions, str):
        updated_instructions, result = apply_rules_to_text(
            instructions,
            rules,
            source_tui=source_tui,
            cc_version=cc_version,
            prompt_field="instructions",
        )
        if result.matched or result.failures:
            updated_body["instructions"] = updated_instructions
            combined.matched.extend(result.matched)
            combined.failures.extend(result.failures)

    # -- input messages ------------------------------------------------
    input_messages = updated_body.get("input")
    if not isinstance(input_messages, list):
        return updated_body, combined

    updated_input: list[dict[str, Any]] = []
    for msg in input_messages:
        if not isinstance(msg, dict):
            updated_input.append(msg)
            continue
        role = msg.get("role")
        if role not in _RESPONSES_SYSTEM_DEVELOPER_ROLES:
            updated_input.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            updated_content, result = apply_rules_to_text(
                content, rules,
                source_tui=source_tui, cc_version=cc_version,
                prompt_field=role,
            )
            if result.matched or result.failures:
                updated_msg = dict(msg)
                updated_msg["content"] = updated_content
                updated_input.append(updated_msg)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
            else:
                updated_input.append(msg)
        elif isinstance(content, list):
            updated_parts, result = _apply_rules_to_content_parts(
                content, rules,
                source_tui=source_tui, cc_version=cc_version,
                prompt_field=role, text_key="input_text",
            )
            if result.matched or result.failures:
                updated_msg = dict(msg)
                updated_msg["content"] = updated_parts
                updated_input.append(updated_msg)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
            else:
                updated_input.append(msg)
        else:
            updated_input.append(msg)

    updated_body["input"] = updated_input
    return updated_body, combined


# --- Shared content-part processors -----------------------------------


def _apply_rules_to_content_parts(
    content_parts: list[dict[str, Any]],
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
    prompt_field: str = "system",
    text_key: str = "text",
) -> tuple[list[dict[str, Any]], ReplacementResult]:
    """Apply rules to content parts with a ``text_key`` (e.g. ``text`` or
    ``input_text``), returning ``(updated_parts, result)``."""
    combined = ReplacementResult()
    updated_parts: list[dict[str, Any]] = []
    changed = False
    for part in content_parts:
        if (
            isinstance(part, dict)
            and part.get("type") == text_key
            and isinstance(part.get("text"), str)
        ):
            updated_text, result = apply_rules_to_text(
                part["text"],
                rules,
                source_tui=source_tui,
                cc_version=cc_version,
                prompt_field=prompt_field,
            )
            if result.matched or result.failures:
                updated_part = dict(part)
                updated_part["text"] = updated_text
                updated_parts.append(updated_part)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
                changed = True
            else:
                updated_parts.append(part)
        else:
            updated_parts.append(part)
    return (updated_parts if changed else content_parts), combined


# --- Chat Completions adapter -------------------------------------------

_CHAT_SYSTEM_DEVELOPER_ROLES = frozenset({"system", "developer"})


def apply_rules_to_chat_completions_messages(
    messages: list[dict[str, Any]],
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
) -> tuple[list[dict[str, Any]], ReplacementResult]:
    """Apply rules to Chat Completions messages.

    Touches messages with role ``system`` or ``developer``.
    Content may be a plain string or a list of text parts.
    """
    combined = ReplacementResult()
    updated_messages: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            updated_messages.append(msg)
            continue
        role = msg.get("role")
        if role not in _CHAT_SYSTEM_DEVELOPER_ROLES:
            updated_messages.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            updated_content, result = apply_rules_to_text(
                content, rules,
                source_tui=source_tui, cc_version=cc_version,
                prompt_field=role,
            )
            if result.matched or result.failures:
                updated_msg = dict(msg)
                updated_msg["content"] = updated_content
                updated_messages.append(updated_msg)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
            else:
                updated_messages.append(msg)
        elif isinstance(content, list):
            updated_parts, result = _apply_rules_to_content_parts(
                content, rules,
                source_tui=source_tui, cc_version=cc_version,
                prompt_field=role,
            )
            if result.matched or result.failures:
                updated_msg = dict(msg)
                updated_msg["content"] = updated_parts
                updated_messages.append(updated_msg)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
            else:
                updated_messages.append(msg)
        else:
            updated_messages.append(msg)

    return updated_messages, combined


# --- Anthropic Messages adapter -----------------------------------------


def apply_rules_to_anthropic_request(
    body: dict[str, Any],
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
) -> tuple[dict[str, Any], ReplacementResult]:
    """Apply rules to an Anthropic Messages request body.

    Touches ``system`` (str or list of ``text`` content blocks).
    """
    updated_body = dict(body)
    combined = ReplacementResult()

    system = updated_body.get("system")
    if system is None:
        return updated_body, combined

    if isinstance(system, str):
        updated_system, result = apply_rules_to_text(
            system,
            rules,
            source_tui=source_tui,
            cc_version=cc_version,
            prompt_field="system",
        )
        if result.matched or result.failures:
            updated_body["system"] = updated_system
            combined.matched.extend(result.matched)
            combined.failures.extend(result.failures)
    elif isinstance(system, list):
        updated_system: list[dict[str, Any]] = []
        changed = False
        for block in system:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                updated_text, result = apply_rules_to_text(
                    block["text"],
                    rules,
                    source_tui=source_tui,
                    cc_version=cc_version,
                    prompt_field="system",
                )
                if result.matched or result.failures:
                    updated_block = dict(block)
                    updated_block["text"] = updated_text
                    updated_system.append(updated_block)
                    combined.matched.extend(result.matched)
                    combined.failures.extend(result.failures)
                    changed = True
                else:
                    updated_system.append(block)
            else:
                updated_system.append(block)
        if changed:
            updated_body["system"] = updated_system

    return updated_body, combined


# --- Gemini adapter -----------------------------------------------------


def apply_rules_to_gemini_request(
    body: dict[str, Any],
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
) -> tuple[dict[str, Any], ReplacementResult]:
    """Apply rules to a Gemini ``generateContent`` request body.

    Touches ``systemInstruction.parts[*].text``.
    """
    updated_body = dict(body)
    combined = ReplacementResult()

    nested = body.get("custom_body")
    if isinstance(nested, dict):
        updated_nested, result = apply_rules_to_gemini_request(
            nested, rules, source_tui=source_tui, cc_version=cc_version,
        )
        if result.matched or result.failures:
            updated_body["custom_body"] = updated_nested
        return updated_body, result

    system_instruction = updated_body.get("systemInstruction")
    if not isinstance(system_instruction, dict):
        return updated_body, combined

    parts = system_instruction.get("parts")
    if not isinstance(parts, list):
        return updated_body, combined

    updated_parts: list[dict[str, Any]] = []
    changed = False
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            updated_text, result = apply_rules_to_text(
                part["text"],
                rules,
                source_tui=source_tui,
                cc_version=cc_version,
                prompt_field="system",
            )
            if result.matched or result.failures:
                updated_part = dict(part)
                updated_part["text"] = updated_text
                updated_parts.append(updated_part)
                combined.matched.extend(result.matched)
                combined.failures.extend(result.failures)
                changed = True
            else:
                updated_parts.append(part)
        else:
            updated_parts.append(part)

    if changed:
        updated_si = dict(system_instruction)
        updated_si["parts"] = updated_parts
        updated_body["systemInstruction"] = updated_si

    return updated_body, combined


# ---------------------------------------------------------------------------
# Observability metadata helper
# ---------------------------------------------------------------------------


def build_replacement_metadata(
    result: ReplacementResult,
    *,
    manifest_paths: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the ``cf073_prompt_replacement`` metadata dict for logging."""
    return {
        "matched_rule_ids": result.matched_ids,
        "total_replacements": result.total_replacements,
        "failure_count": len(result.failures),
        "failures": result.failures,
        "manifest_paths": manifest_paths or [],
    }


# ---------------------------------------------------------------------------
# Symbol inventory for host publication
# ---------------------------------------------------------------------------

OWNED_SYMBOLS: tuple[str, ...] = (
    "ReplacementRule",
    "ReplacementMatch",
    "ReplacementResult",
    "_parse_cc_version",
    "load_manifest",
    "load_manifests",
    "load_text_asset",
    "apply_rules_to_text",
    "apply_rules_to_responses_request",
    "apply_rules_to_chat_completions_messages",
    "apply_rules_to_anthropic_request",
    "apply_rules_to_gemini_request",
    "build_replacement_metadata",
    "_DEFAULT_MANIFEST_CACHE",
    "_text_asset_cache",
)
