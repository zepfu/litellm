"""Map Cursor Dashboard GetCurrentPeriodUsage into rate_limit_observations.

Monthly source is Connect
`POST /aiserver.v1.DashboardService/GetCurrentPeriodUsage` on
`https://api2.cursor.sh`. This is not Cloud Agents `GET /v0/me`.

Weekly Cursor Grok Bot used/limit/reset has no stable Dashboard or
Connect source. Reevaluate only when AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE
names a verified RPC. Do not invent a weekly quota_key. Do not treat
xAI Grok Build weekly credits or BugBot license RPCs as Grok Bot.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .usage_client import CURSOR_AGENT_PROVIDER

CURSOR_AGENT_USAGE_SOURCE = "cursor_agent_usage"
CURSOR_AGENT_USAGE_PARSER_VERSION = "cursor_agent_usage_v1"
CURSOR_AGENT_USAGE_CLIENT = "cursor-agent"
CURSOR_AGENT_USAGE_MODEL = "cursor-agent"
CURSOR_AGENT_USAGE_QUOTA_TYPE = "cents"
CURSOR_AGENT_USAGE_QUOTA_PERIOD = "monthly"
CURSOR_AGENT_MONTHLY_QUOTA_KEY = "cursor_agent_monthly:cents"
CURSOR_AGENT_GROK_BOT_USAGE_SOURCE_ENV = "AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE"
CURSOR_AGENT_GROK_BOT_STATUS = "unknown"

_PLAN_USAGE_KEYS = ("planUsage", "plan_usage")
_INCLUDED_SPEND_KEYS = ("includedSpend", "included_spend")
_LIMIT_KEYS = ("limit",)
_REMAINING_KEYS = ("remaining",)
_TOTAL_SPEND_KEYS = ("totalSpend", "total_spend")
_AUTO_SPEND_KEYS = ("autoSpend", "auto_spend")
_API_SPEND_KEYS = ("apiSpend", "api_spend")
_TOTAL_PERCENT_KEYS = ("totalPercentUsed", "total_percent_used")
_AUTO_PERCENT_KEYS = ("autoPercentUsed", "auto_percent_used")
_API_PERCENT_KEYS = ("apiPercentUsed", "api_percent_used")
_BILLING_START_KEYS = ("billingCycleStart", "billing_cycle_start")
_BILLING_END_KEYS = ("billingCycleEnd", "billing_cycle_end")
_ACCOUNT_IDENTITY_KEYS = (
    "userId",
    "user_id",
    "accountId",
    "account_id",
    "membershipId",
    "membership_id",
    "teamId",
    "team_id",
)


def grok_bot_reevaluation_checkpoint() -> Dict[str, Any]:
    """Return the truthful-unknown weekly Grok Bot checkpoint.

    A configured source is only a review signal. It does not invent a
    weekly quota_key or authorize treating Grok Build / BugBot as Grok Bot.
    """
    source = (os.getenv(CURSOR_AGENT_GROK_BOT_USAGE_SOURCE_ENV) or "").strip()
    return {
        "status": CURSOR_AGENT_GROK_BOT_STATUS,
        "quota_key": None,
        "reevaluation_source": source or None,
        "reevaluation_ready": bool(source),
    }


def _first_present(
    mapping: Mapping[str, Any], names: Tuple[str, ...]
) -> Tuple[bool, Any]:
    for name in names:
        if name in mapping:
            return True, mapping[name]
    return False, None


def _parse_usage_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:
        return None
    return number


def _parse_usage_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:
            return None
        if numeric > 10_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _mapping_nodes(
    payload: Mapping[str, Any],
) -> List[Tuple[Tuple[str, ...], Mapping[str, Any]]]:
    nodes: List[Tuple[Tuple[str, ...], Mapping[str, Any]]] = []
    pending: List[Tuple[Tuple[str, ...], Any]] = [((), payload)]
    while pending:
        path, value = pending.pop(0)
        if len(path) > 5:
            continue
        if isinstance(value, Mapping):
            nodes.append((path, value))
            for key, nested in value.items():
                pending.append((path + (str(key),), nested))
        elif isinstance(value, list):
            for index, nested in enumerate(value[:50]):
                pending.append((path + (str(index),), nested))
    return nodes


def hash_cursor_agent_account_identity(
    payload: Mapping[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """Hash account identity fields. Never return raw tokens or account ids."""
    identity_parts: List[str] = []
    identity_fields: List[str] = []
    for path, mapping in _mapping_nodes(payload):
        for field_name in _ACCOUNT_IDENTITY_KEYS:
            value = mapping.get(field_name)
            if isinstance(value, (str, int)) and str(value).strip():
                prefix = ".".join(path + (field_name,)) if path else field_name
                identity_parts.append(f"{prefix}={value}")
                identity_fields.append(prefix)
        for container_name in ("user", "account", "membership", "team"):
            nested = mapping.get(container_name)
            if not isinstance(nested, Mapping):
                continue
            value = nested.get("id")
            if isinstance(value, (str, int)) and str(value).strip():
                prefix = (
                    ".".join(path + (container_name, "id"))
                    if path
                    else f"{container_name}.id"
                )
                identity_parts.append(f"{prefix}={value}")
                identity_fields.append(prefix)
    if not identity_parts:
        return None, []
    seen = set()
    unique_parts: List[str] = []
    unique_fields: List[str] = []
    for part, field_name in zip(identity_parts, identity_fields):
        if part in seen:
            continue
        seen.add(part)
        unique_parts.append(part)
        unique_fields.append(field_name)
    material = ("cursor-agent-account|" + "|".join(unique_parts)).encode("utf-8")
    return hashlib.sha256(material).hexdigest(), unique_fields


def parse_current_period_usage(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse a GetCurrentPeriodUsage JSON body into a monthly snapshot.

    The 2026-08-12 dashboard dump was camelCase JSON with spend in USD
    cents. Accept the proto snake_case aliases as well.

    Trustworthy included fraction is includedSpend / limit. Do not treat
    totalPercentUsed / autoPercentUsed / apiPercentUsed as that fraction.
    """
    grok_bot = grok_bot_reevaluation_checkpoint()
    account_hash, account_identity_fields = hash_cursor_agent_account_identity(
        payload
    )
    present, plan_usage = _first_present(payload, _PLAN_USAGE_KEYS)
    if not present or not isinstance(plan_usage, Mapping):
        return {
            "state": "absent",
            "quota_used": None,
            "quota_limit": None,
            "quota_remaining": None,
            "remaining_pct": None,
            "billing_period_start_at": None,
            "billing_period_end_at": None,
            "account_hash": account_hash,
            "account_identity_fields": account_identity_fields,
            "raw_provider_fields": {
                "parser_version": CURSOR_AGENT_USAGE_PARSER_VERSION,
                "window": CURSOR_AGENT_USAGE_QUOTA_PERIOD,
                "quota_unit": CURSOR_AGENT_USAGE_QUOTA_TYPE,
                "percent_fields_are_not_total_over_limit": True,
            },
            "evidence": {
                "signals": ["cursor_agent_get_current_period_usage"],
                "parser_version": CURSOR_AGENT_USAGE_PARSER_VERSION,
                "telemetry_status": "absent",
                "weekly_grok_bot": grok_bot["status"],
                "weekly_grok_bot_quota_key": grok_bot["quota_key"],
                "account_identity_fields": account_identity_fields,
            },
            "grok_bot": grok_bot,
        }

    included_present, included_raw = _first_present(plan_usage, _INCLUDED_SPEND_KEYS)
    limit_present, limit_raw = _first_present(plan_usage, _LIMIT_KEYS)
    remaining_present, remaining_raw = _first_present(plan_usage, _REMAINING_KEYS)
    quota_used = _parse_usage_number(included_raw) if included_present else None
    quota_limit = _parse_usage_number(limit_raw) if limit_present else None
    quota_remaining = (
        _parse_usage_number(remaining_raw) if remaining_present else None
    )
    if included_present and quota_used is None:
        state = "malformed"
    elif limit_present and quota_limit is None:
        state = "malformed"
    elif remaining_present and quota_remaining is None:
        state = "malformed"
    elif quota_used is None and quota_limit is None and quota_remaining is None:
        state = "malformed"
    elif quota_used == 0 or (
        quota_used is None
        and all(
            value == 0
            for value in (quota_limit, quota_remaining)
            if value is not None
        )
    ):
        state = "valid_zero"
    else:
        state = "valid_nonzero"

    if (
        quota_remaining is None
        and quota_limit is not None
        and quota_used is not None
    ):
        quota_remaining = max(0.0, quota_limit - quota_used)

    remaining_pct: Optional[float] = None
    if quota_limit is not None and quota_limit > 0 and quota_used is not None:
        remaining_pct = max(
            0.0, min(100.0, 100.0 - (quota_used / quota_limit * 100.0))
        )
    elif (
        quota_limit is not None
        and quota_limit > 0
        and quota_remaining is not None
    ):
        remaining_pct = max(
            0.0, min(100.0, quota_remaining / quota_limit * 100.0)
        )

    start_present, start_raw = _first_present(payload, _BILLING_START_KEYS)
    end_present, end_raw = _first_present(payload, _BILLING_END_KEYS)
    billing_period_start_at = (
        _parse_usage_timestamp(start_raw) if start_present else None
    )
    billing_period_end_at = (
        _parse_usage_timestamp(end_raw) if end_present else None
    )

    _, total_spend = _first_present(plan_usage, _TOTAL_SPEND_KEYS)
    _, auto_spend = _first_present(plan_usage, _AUTO_SPEND_KEYS)
    _, api_spend = _first_present(plan_usage, _API_SPEND_KEYS)
    _, total_percent = _first_present(plan_usage, _TOTAL_PERCENT_KEYS)
    _, auto_percent = _first_present(plan_usage, _AUTO_PERCENT_KEYS)
    _, api_percent = _first_present(plan_usage, _API_PERCENT_KEYS)

    raw_provider_fields: Dict[str, Any] = {
        "parser_version": CURSOR_AGENT_USAGE_PARSER_VERSION,
        "window": CURSOR_AGENT_USAGE_QUOTA_PERIOD,
        "quota_unit": CURSOR_AGENT_USAGE_QUOTA_TYPE,
        "quota_used": quota_used,
        "quota_limit": quota_limit,
        "quota_remaining": quota_remaining,
        "included_spend_cents": quota_used,
        "percent_fields_are_not_total_over_limit": True,
        "included_fraction_source": "includedSpend / limit",
    }
    for key, raw_value in (
        ("total_spend_cents", total_spend),
        ("auto_spend_cents", auto_spend),
        ("api_spend_cents", api_spend),
        ("total_percent_used", total_percent),
        ("auto_percent_used", auto_percent),
        ("api_percent_used", api_percent),
    ):
        parsed = _parse_usage_number(raw_value)
        if parsed is not None:
            raw_provider_fields[key] = parsed

    evidence = {
        "signals": [
            "cursor_agent_get_current_period_usage",
            "cursor_agent_monthly_included_spend",
        ],
        "parser_version": CURSOR_AGENT_USAGE_PARSER_VERSION,
        "telemetry_status": "valid" if state.startswith("valid") else state,
        "window_state": state,
        "weekly_grok_bot": grok_bot["status"],
        "weekly_grok_bot_quota_key": grok_bot["quota_key"],
        "account_identity_fields": account_identity_fields,
        "percent_fields_are_not_total_over_limit": True,
        "unit_note": (
            "Cursor Dashboard planUsage spend values are USD cents. "
            "quota_used is includedSpend, not totalSpend."
        ),
    }
    if (
        quota_limit is not None
        and quota_used is not None
        and quota_used > quota_limit
    ):
        evidence["signals"].append("cursor_agent_usage_exceeds_limit")

    return {
        "state": state,
        "quota_used": quota_used,
        "quota_limit": quota_limit,
        "quota_remaining": quota_remaining,
        "remaining_pct": remaining_pct,
        "billing_period_start_at": billing_period_start_at,
        "billing_period_end_at": billing_period_end_at,
        "account_hash": account_hash,
        "account_identity_fields": account_identity_fields,
        "raw_provider_fields": raw_provider_fields,
        "evidence": evidence,
        "grok_bot": grok_bot,
        "provider": CURSOR_AGENT_PROVIDER,
        "quota_key": CURSOR_AGENT_MONTHLY_QUOTA_KEY,
        "quota_period": CURSOR_AGENT_USAGE_QUOTA_PERIOD,
        "quota_type": CURSOR_AGENT_USAGE_QUOTA_TYPE,
    }
