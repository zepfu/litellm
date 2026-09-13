"""Stable identities for request-local skipped-candidate observations."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _clean_skip_identity_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_skip_identity_value(
    candidate: Mapping[str, Any],
    *fields: str,
) -> str:
    for field in fields:
        value = _clean_skip_identity_value(candidate.get(field))
        if value:
            return value
    return ""


def _skip_identity_presence_value(
    candidate: Mapping[str, Any],
    field: str,
) -> tuple[str, str]:
    """Preserve whether an optional occurrence field was supplied."""
    if field not in candidate or candidate[field] is None:
        return ("missing", "")
    return ("present", _clean_skip_identity_value(candidate[field]))


def _auto_agent_alias_skip_identity(
    candidate: Mapping[str, Any],
    *,
    alias_family: Optional[Any] = None,
    alias_model: Optional[Any] = None,
    skip_reason: Optional[Any] = None,
    cooldown_scope: Optional[Any] = None,
) -> tuple[Any, ...]:
    """Return the stable identity for one skipped candidate occurrence.

    The identity is scoped to the owning alias and resolved candidate route.
    Compiled occurrence identity, account/lane and cooldown identities
    distinguish otherwise equal routes; semantic skip reason/scope distinguish
    different decisions for that occurrence. Volatile source labels, remaining
    durations, and attempt-list positions are intentionally excluded.
    """

    resolved_skip_reason = _clean_skip_identity_value(skip_reason)
    if not resolved_skip_reason:
        resolved_skip_reason = _first_skip_identity_value(
            candidate,
            "selection_reason",
            "skip_reason",
            "reason",
        )
    if not resolved_skip_reason:
        event_type = _clean_skip_identity_value(candidate.get("event_type"))
        if event_type.startswith("candidate_skipped_"):
            resolved_skip_reason = event_type.removeprefix("candidate_skipped_")
    if not resolved_skip_reason:
        resolved_skip_reason = "unavailable"

    resolved_scope = _clean_skip_identity_value(cooldown_scope)
    if not resolved_scope:
        resolved_scope = _first_skip_identity_value(candidate, "cooldown_scope")

    return (
        "aawm_alias_skip_identity_v1",
        _clean_skip_identity_value(alias_family)
        or _clean_skip_identity_value(candidate.get("alias_family")),
        _clean_skip_identity_value(alias_model)
        or _clean_skip_identity_value(candidate.get("alias_model")),
        _first_skip_identity_value(candidate, "provider"),
        _first_skip_identity_value(candidate, "model"),
        _first_skip_identity_value(candidate, "route_family"),
        _skip_identity_presence_value(candidate, "selection_priority"),
        _skip_identity_presence_value(candidate, "last_resort"),
        _first_skip_identity_value(candidate, "resolved_alias"),
        _first_skip_identity_value(candidate, "cooldown_identity_tag"),
        _first_skip_identity_value(
            candidate,
            "account_hash",
            "codex_oauth_account_hash",
            "xai_oauth_account_hash",
        ),
        _first_skip_identity_value(
            candidate,
            "account_lane",
            "codex_oauth_lane_key",
            "xai_oauth_lane_key",
        ),
        _first_skip_identity_value(candidate, "lane_key"),
        _first_skip_identity_value(
            candidate,
            "cooldown_key",
            "logical_cooldown_key",
        ),
        resolved_skip_reason,
        resolved_scope,
        _first_skip_identity_value(
            candidate,
            "candidate_semantic_ineligibility_reason",
        ),
    )


__all__ = ["_auto_agent_alias_skip_identity"]
