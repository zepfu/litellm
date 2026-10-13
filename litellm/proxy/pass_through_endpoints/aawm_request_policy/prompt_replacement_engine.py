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
from typing import Any, Optional

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


def _parse_rules_from_manifest(raw: dict[str, Any]) -> list[ReplacementRule]:
    """Parse a raw manifest dict into ReplacementRule objects.

    Accepts either ``"rules"`` (new shared format) or ``"patches"``
    (legacy Claude prompt-patch format) at the top level.
    """
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

        after_text = descriptor.get("after")
        if not isinstance(after_text, str):
            after_text = ""

        before_pattern_raw = descriptor.get("before_pattern")
        before_pattern: Optional[re.Pattern] = None
        if isinstance(before_pattern_raw, str) and before_pattern_raw:
            before_pattern = re.compile(before_pattern_raw)

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


def load_manifest(manifest_path: Path) -> list[ReplacementRule]:
    """Load and parse a single prompt-replacement manifest."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    return _parse_rules_from_manifest(raw)


def load_manifests(
    manifest_paths: list[Path],
    *,
    cache: Optional[dict[Path, list[ReplacementRule]]] = None,
) -> list[ReplacementRule]:
    """Load rules from multiple manifests, using an optional per-path cache."""
    all_rules: list[ReplacementRule] = []
    for path in manifest_paths:
        if cache is not None and path in cache:
            all_rules.extend(cache[path])
            continue
        rules = load_manifest(path)
        if cache is not None:
            cache[path] = rules
        all_rules.extend(rules)
    return all_rules


# ---------------------------------------------------------------------------
# Core engine: single text field
# ---------------------------------------------------------------------------


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
                count = updated_text.count(rule.before)
                if count:
                    updated_text = updated_text.replace(rule.before, rule.after)
                    occurrences += count
                    match_types.append("exact")

            if rule.before_pattern:
                updated_text, pattern_count = rule.before_pattern.subn(
                    rule.after, updated_text
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
# Recursive dict/list walker (used by adapters)
# ---------------------------------------------------------------------------


def _replace_rules_in_value(
    value: Any,
    rules: list[ReplacementRule],
    *,
    source_tui: Optional[str] = None,
    cc_version: Optional[str] = None,
    prompt_field: Optional[str] = None,
) -> tuple[Any, ReplacementResult]:
    """Recursively apply rules to text leaves in a dict/list tree.

    Only recurses into dicts and lists; scalar leaves are passed to
    ``apply_rules_to_text`` if they are strings.
    """
    if isinstance(value, str):
        return apply_rules_to_text(
            value,
            rules,
            source_tui=source_tui,
            cc_version=cc_version,
            prompt_field=prompt_field,
        )

    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        combined = ReplacementResult()
        changed = False
        for key, child in value.items():
            updated_child, child_result = _replace_rules_in_value(
                child,
                rules,
                source_tui=source_tui,
                cc_version=cc_version,
                prompt_field=prompt_field,
            )
            updated_dict[key] = updated_child
            combined.matched.extend(child_result.matched)
            combined.failures.extend(child_result.failures)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined

    if isinstance(value, list):
        updated_list: list[Any] = []
        combined = ReplacementResult()
        changed = False
        for child in value:
            updated_child, child_result = _replace_rules_in_value(
                child,
                rules,
                source_tui=source_tui,
                cc_version=cc_version,
                prompt_field=prompt_field,
            )
            updated_list.append(updated_child)
            combined.matched.extend(child_result.matched)
            combined.failures.extend(child_result.failures)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined

    return value, ReplacementResult()


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
    "apply_rules_to_text",
    "apply_rules_to_responses_request",
    "apply_rules_to_chat_completions_messages",
    "apply_rules_to_anthropic_request",
    "apply_rules_to_gemini_request",
    "build_replacement_metadata",
)
