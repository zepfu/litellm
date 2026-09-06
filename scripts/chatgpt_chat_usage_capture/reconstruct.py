"""Reconstruct generation attempts from sanitized Chat history messages."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Mapping, Optional, Sequence

from .models import AttemptRecord, MessageRecord
from .privacy import SURFACE_CHAT


PROVISIONAL_PREFIX = "provisional:"


def reconstruct_attempts(
    messages: Sequence[MessageRecord],
    *,
    mapping_version: str,
    mapping_rules: Sequence[Mapping[str, object]] = (),
    conversation_id: str,
) -> list[AttemptRecord]:
    """Group message nodes into user-initiated generation attempts."""
    by_id = {item.message_id: item for item in messages}
    attempts: list[AttemptRecord] = []
    seen_ids: set[str] = set()
    for user in _user_prompts(messages):
        children = [by_id[child_id] for child_id in user.children if child_id in by_id]
        if not children:
            children = [
                item
                for item in messages
                if item.parent_id in {user.message_id, user.node_id}
                and item.role == "assistant"
            ]
        groups = _group_assistant_messages(children)
        if not groups:
            continue
        for group in groups:
            attempt = _attempt_from_group(
                user=user,
                group=group,
                mapping_version=mapping_version,
                mapping_rules=mapping_rules,
                conversation_id=conversation_id,
            )
            if attempt.attempt_id in seen_ids:
                continue
            seen_ids.add(attempt.attempt_id)
            attempts.append(attempt)
    orphan_assistants = [
        item
        for item in messages
        if item.role == "assistant"
        and item.message_id not in {mid for attempt in attempts for mid in attempt.evidence_message_ids}
        and item.channel not in {"tool", "commentary"}
    ]
    for group in _group_assistant_messages(orphan_assistants):
        attempt = _attempt_from_group(
            user=None,
            group=group,
            mapping_version=mapping_version,
            mapping_rules=mapping_rules,
            conversation_id=conversation_id,
        )
        if attempt.attempt_id not in seen_ids:
            seen_ids.add(attempt.attempt_id)
            attempts.append(attempt)
    return attempts


def _user_prompts(messages: Sequence[MessageRecord]) -> list[MessageRecord]:
    return [item for item in messages if item.role == "user"]


def _group_assistant_messages(messages: Sequence[MessageRecord]) -> list[list[MessageRecord]]:
    grouped: dict[str, list[MessageRecord]] = {}
    ungrouped: list[MessageRecord] = []
    for item in messages:
        if item.role not in {"assistant", "tool"}:
            continue
        if item.generation_id:
            grouped.setdefault(f"generation:{item.generation_id}", []).append(item)
        elif item.request_id:
            grouped.setdefault(f"request:{item.request_id}", []).append(item)
        else:
            ungrouped.append(item)
    groups = list(grouped.values())
    for item in ungrouped:
        groups.append([item])
    return groups


def _attempt_from_group(
    *,
    user: Optional[MessageRecord],
    group: Sequence[MessageRecord],
    mapping_version: str,
    mapping_rules: Sequence[Mapping[str, object]],
    conversation_id: str,
) -> AttemptRecord:
    final = _final_answer(group)
    generation_ids = [item.generation_id for item in group if item.generation_id]
    request_ids = [item.request_id for item in group if item.request_id]
    if generation_ids:
        identity_basis = "generation"
        attempt_key = generation_ids[0]
    elif request_ids and user is not None:
        identity_basis = "request"
        attempt_key = f"{conversation_id}:{request_ids[0]}:{user.message_id}"
    elif final is not None and user is not None:
        identity_basis = "provisional"
        attempt_key = f"{conversation_id}:{user.message_id}:{final.message_id}"
    elif final is not None:
        identity_basis = "unresolved"
        attempt_key = f"{conversation_id}:{final.message_id}"
    else:
        identity_basis = "unresolved"
        attempt_key = f"{conversation_id}:{group[0].message_id}"
    attempt_id = _stable_id(conversation_id, attempt_key)
    requested_model = user.requested_model_raw if user is not None else None
    requested_mode = user.requested_mode_raw if user is not None else None
    requested_effort = user.requested_reasoning_effort_raw if user is not None else None
    recorded_final = final.recorded_final_model_raw if final is not None else None
    attempt_time, time_basis, earliest, latest = _attempt_timing(user, group, final)
    outcome = _outcome(group, final)
    surface = _surface([user] if user is not None else [], group)
    origin = _origin(user, group)
    aliases = []
    for value in generation_ids:
        aliases.append(("generation", value))
    for value in request_ids:
        aliases.append(("request", f"{conversation_id}:{value}"))
    if user is not None:
        aliases.append(("prompt", f"{conversation_id}:{user.message_id}:{generation_ids[0] if generation_ids else final.message_id if final else 'none'}"))
    warnings: list[str] = []
    if identity_basis == "unresolved":
        warnings.append("unresolved_linkage")
    if origin in {"shared", "imported", "copied"}:
        warnings.append(f"origin:{origin}")
    if surface != SURFACE_CHAT:
        warnings.append(f"surface:{surface}")
    requested_family = map_family(requested_model, requested_mode, requested_effort, mapping_rules)
    recorded_family = map_family(recorded_final, None, None, mapping_rules)
    return AttemptRecord(
        attempt_id=attempt_id,
        conversation_id=conversation_id,
        identity_basis=identity_basis,
        time_basis=time_basis,
        attempt_time=attempt_time,
        earliest_possible_at=earliest,
        latest_possible_at=latest,
        requested_model_raw=requested_model,
        requested_mode_raw=requested_mode,
        requested_reasoning_effort_raw=requested_effort,
        recorded_final_model_raw=recorded_final,
        resolved_model_raw=None,
        requested_family=requested_family,
        recorded_final_family=recorded_family,
        resolved_family=None,
        mapping_version=mapping_version,
        outcome=outcome,
        completed_answer=final is not None and outcome in {"completed", "completed_inferred"},
        generation_started=any(
            item.role == "assistant" or item.status in {"in_progress", "finished_successfully"}
            for item in group
        ),
        surface=surface,
        origin=origin,
        aliases=tuple(aliases),
        evidence_message_ids=tuple(item.message_id for item in ([user] if user else []) + list(group)),
        revision=1,
        warnings=tuple(warnings),
    )


def _final_answer(group: Sequence[MessageRecord]) -> Optional[MessageRecord]:
    candidates = [
        item
        for item in group
        if item.role == "assistant"
        and item.channel not in {"tool", "commentary", "reasoning"}
        and (item.end_turn is True or item.status in {"finished_successfully", "completed", None})
    ]
    if not candidates:
        candidates = [item for item in group if item.role == "assistant" and item.channel not in {"tool"}]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.created_at or datetime.min)[-1]


def _attempt_timing(
    user: Optional[MessageRecord],
    group: Sequence[MessageRecord],
    final: Optional[MessageRecord],
) -> tuple[Optional[datetime], str, Optional[datetime], Optional[datetime]]:
    times = [item.created_at for item in group if item.created_at]
    if user is not None and user.created_at is not None:
        return user.created_at, "user_message", user.created_at, final.created_at if final and final.created_at else user.created_at
    if times:
        earliest = min(times)
        latest = max(times)
        if final is not None and final.created_at is not None and user is None:
            return None, "bounded_interval", earliest, latest
        return earliest, "assistant_bracket", earliest, latest
    return None, "unknown", None, None


def _outcome(group: Sequence[MessageRecord], final: Optional[MessageRecord]) -> str:
    statuses = {item.status for item in group if item.status}
    if "rejected" in statuses or "moderation_blocked" in statuses:
        return "rejected_before_start"
    if any(item.status in {"cancelled", "interrupted"} for item in group) and final is None:
        return "cancelled_after_start"
    if any(item.status in {"error", "failed"} for item in group) and final is None:
        return "failed_after_start"
    if final is not None:
        return "completed"
    if any(item.role == "assistant" for item in group):
        return "completion_unknown"
    return "unresolved"


def _surface(user_messages: Sequence[MessageRecord], group: Sequence[MessageRecord]) -> str:
    surfaces = [
        item.surface
        for item in list(user_messages) + list(group)
        if item is not None and item.surface
    ]
    if SURFACE_CHAT in surfaces and all(item in {SURFACE_CHAT, "unknown"} for item in surfaces):
        return SURFACE_CHAT
    non_unknown = [item for item in surfaces if item != "unknown"]
    if len(set(non_unknown)) == 1:
        return non_unknown[0]
    if non_unknown:
        return "unknown"
    return "unknown"


def _origin(user: Optional[MessageRecord], group: Sequence[MessageRecord]) -> Optional[str]:
    values = []
    if user is not None and user.origin:
        values.append(user.origin)
    values.extend(item.origin for item in group if item.origin)
    if not values:
        return None
    if "imported" in values:
        return "imported"
    if "shared" in values or "true" in values:
        return "shared"
    if "copied" in values:
        return "copied"
    return values[0]


def map_family(
    slug: Optional[str],
    mode: Optional[str],
    effort: Optional[str],
    rules: Sequence[Mapping[str, object]],
) -> Optional[str]:
    if not slug:
        return None
    for rule in rules:
        rule_slug = str(rule.get("slug") or "")
        if rule_slug and rule_slug != slug:
            continue
        if rule.get("mode") and rule.get("mode") != mode:
            continue
        if rule.get("reasoning_effort") and rule.get("reasoning_effort") != effort:
            continue
        family = rule.get("family")
        if family:
            return str(family)
    lowered = slug.lower()
    if "codex" in lowered:
        return None
    return None


def _stable_id(conversation_id: str, attempt_key: str) -> str:
    digest = hashlib.sha256(f"{conversation_id}|{attempt_key}".encode("utf-8")).hexdigest()
    return digest
