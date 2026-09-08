"""Managed xAI OAuth account binding.

Only server-selected inventory records reach request state. Candidate bodies,
client metadata, and bearer material are never accepted as an account binding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from fastapi import HTTPException, Request

from litellm.secret_managers.xai_oauth_inventory import (
    XAI_OAUTH_INVENTORY_ENV,
    XaiOAuthAccountRecord,
    XaiOAuthIdentityMismatchError,
    XaiOAuthInventoryError,
    legacy_xai_oauth_account_record,
    load_xai_oauth_inventory,
    xai_oauth_record_identity,
    xai_oauth_scope_identity,
)

_XAI_OAUTH_SELECTED_ACCOUNT_STATE = "aawm_xai_oauth_selected_account"
_XAI_OAUTH_MANAGED_ROUTE_FAMILIES = frozenset(
    {
        "codex_xai_oauth_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "codex_auto_agent_xai_oauth_responses",
    }
)


@dataclass(frozen=True)
class XaiOAuthSelectedAccount:
    """A request-bound managed xAI OAuth record with no credential contents."""

    record: XaiOAuthAccountRecord = field(repr=False)
    account_hash: str
    scope_identity: str
    lane_key: str

    @property
    def label(self) -> str:
        return self.record.label


def is_managed_xai_oauth_candidate(candidate: Any) -> bool:
    """Return whether a compiled candidate uses managed xAI OAuth."""

    return bool(
        isinstance(candidate, Mapping)
        and candidate.get("provider") == "xai"
        and candidate.get("route_family") in _XAI_OAUTH_MANAGED_ROUTE_FAMILIES
    )


def xai_oauth_account_lane_key(record: XaiOAuthAccountRecord) -> str:
    """Return a stable server-owned lane for one configured record."""

    return f"xai-oauth:{record.label}:{xai_oauth_record_identity(record)}"


def configured_xai_oauth_records() -> tuple[XaiOAuthAccountRecord, ...]:
    """Return inventory records, retaining one-file compatibility when absent."""

    raw_inventory = os.getenv(XAI_OAUTH_INVENTORY_ENV)
    if not isinstance(raw_inventory, str) or not raw_inventory.strip():
        return (legacy_xai_oauth_account_record(),)
    return load_xai_oauth_inventory(raw_inventory).ordered_records(
        enabled_only=True
    )


def select_xai_oauth_account_record(
    *,
    label: Optional[str] = None,
) -> XaiOAuthAccountRecord:
    """Select exactly one enabled inventory record or legacy default."""

    raw_inventory = os.getenv(XAI_OAUTH_INVENTORY_ENV)
    if not isinstance(raw_inventory, str) or not raw_inventory.strip():
        record = legacy_xai_oauth_account_record()
        if label is not None and label != record.label:
            raise XaiOAuthInventoryError(
                "Selected xAI OAuth account label is not configured."
            )
        return record
    return load_xai_oauth_inventory(raw_inventory).select_record(label=label)


def build_xai_oauth_selected_account(
    record: XaiOAuthAccountRecord,
) -> XaiOAuthSelectedAccount:
    """Build safe account identity for a selected record."""

    return XaiOAuthSelectedAccount(
        record=record,
        account_hash=xai_oauth_record_identity(record),
        scope_identity=xai_oauth_scope_identity(record),
        lane_key=xai_oauth_account_lane_key(record),
    )


def _clean_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _candidate_selected_account(
    candidate: Mapping[str, Any],
) -> Optional[XaiOAuthSelectedAccount]:
    label = _clean_string(candidate.get("xai_oauth_account_label"))
    account_hash = _clean_string(candidate.get("xai_oauth_account_hash"))
    scope_identity = _clean_string(candidate.get("xai_oauth_scope_identity"))
    lane_key = _clean_string(candidate.get("xai_oauth_lane_key"))
    present_count = sum(
        value is not None
        for value in (label, account_hash, scope_identity, lane_key)
    )
    if present_count == 0:
        return None
    if present_count != 4:
        raise HTTPException(
            status_code=500,
            detail="Selected xAI OAuth account context is incomplete.",
        )
    assert label is not None
    assert account_hash is not None
    assert scope_identity is not None
    assert lane_key is not None

    try:
        record = select_xai_oauth_account_record(label=label)
    except XaiOAuthInventoryError as exc:
        raise HTTPException(
            status_code=500,
            detail="Selected xAI OAuth account is not configured.",
        ) from exc
    selected = build_xai_oauth_selected_account(record)
    if (
        selected.account_hash != account_hash
        or selected.scope_identity != scope_identity
        or selected.lane_key != lane_key
    ):
        raise HTTPException(
            status_code=500,
            detail="Selected xAI OAuth account identity is invalid.",
        )
    return selected


def bind_xai_oauth_candidate_to_request(
    request: Request,
    candidate: Mapping[str, Any],
) -> Optional[XaiOAuthSelectedAccount]:
    """Bind only a server-selected managed xAI account to request state."""

    if not is_managed_xai_oauth_candidate(candidate):
        setattr(request.state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, None)
        return None
    selected = _candidate_selected_account(candidate)
    if selected is None:
        raise HTTPException(
            status_code=500,
            detail="Managed xAI OAuth candidate has no selected account.",
        )
    setattr(request.state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, selected)
    return selected


def get_bound_xai_oauth_selected_account(
    request: Any,
) -> Optional[XaiOAuthSelectedAccount]:
    """Return the validated server binding currently held in request state."""

    state = getattr(request, "state", None)
    selected = getattr(state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, None)
    if not isinstance(selected, XaiOAuthSelectedAccount):
        return None
    try:
        current_record = select_xai_oauth_account_record(label=selected.label)
    except XaiOAuthInventoryError:
        return None
    expected = build_xai_oauth_selected_account(current_record)
    if (
        expected.account_hash != selected.account_hash
        or expected.scope_identity != selected.scope_identity
        or expected.lane_key != selected.lane_key
    ):
        return None
    return selected


def get_or_bind_xai_oauth_selected_account(
    request: Request,
) -> XaiOAuthSelectedAccount:
    """Return an existing binding or select the configured primary record."""

    selected = get_bound_xai_oauth_selected_account(request)
    if selected is not None:
        return selected
    try:
        selected = build_xai_oauth_selected_account(
            select_xai_oauth_account_record()
        )
    except XaiOAuthInventoryError as exc:
        raise ValueError(
            "Managed xAI OAuth account inventory is unavailable."
        ) from exc
    setattr(request.state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, selected)
    return selected


async def get_xai_oauth_snapshot_for_selected_account(
    selected: XaiOAuthSelectedAccount,
) -> Any:
    """Load the exact selected record and prove its configured identity pin."""

    from litellm.llms.xai.oauth import get_xai_oauth_snapshot_for_record

    snapshot = await get_xai_oauth_snapshot_for_record(selected.record)
    expected_identity = selected.record.expected_account_identity
    actual_identity = getattr(snapshot, "account_identity", None)
    if expected_identity is not None and actual_identity != expected_identity:
        raise XaiOAuthIdentityMismatchError(
            "Managed xAI OAuth credential identity does not match the configured "
            f"record '{selected.label}'."
        )
    return snapshot


def xai_oauth_selected_account_metadata(
    selected: XaiOAuthSelectedAccount,
) -> dict[str, str | bool]:
    """Return the bounded server-derived metadata allowed into observations."""

    return {
        "xai_oauth_server_account_binding": True,
        "xai_oauth_account_label": selected.label,
        "xai_oauth_account_hash": selected.account_hash,
        "xai_oauth_lane_key": selected.lane_key,
        "xai_oauth_record_identity": selected.account_hash,
        "xai_oauth_scope_identity": selected.scope_identity,
    }


def validated_xai_oauth_server_account_metadata(
    metadata: Mapping[str, Any],
) -> Optional[dict[str, str | bool]]:
    """Return inventory-proven observation metadata or reject the payload."""

    if metadata.get("xai_oauth_server_account_binding") is not True:
        return None
    label = _clean_string(metadata.get("xai_oauth_account_label"))
    account_hash = _clean_string(metadata.get("xai_oauth_account_hash"))
    scope_identity = _clean_string(metadata.get("xai_oauth_scope_identity"))
    lane_key = _clean_string(metadata.get("xai_oauth_lane_key"))
    record_identity = _clean_string(metadata.get("xai_oauth_record_identity"))
    if not all(
        (
            label,
            account_hash,
            scope_identity,
            lane_key,
            record_identity,
        )
    ):
        return None
    assert label is not None
    try:
        selected = build_xai_oauth_selected_account(
            select_xai_oauth_account_record(label=label)
        )
    except XaiOAuthInventoryError:
        return None
    expected = xai_oauth_selected_account_metadata(selected)
    if (
        account_hash != expected["xai_oauth_account_hash"]
        or scope_identity != expected["xai_oauth_scope_identity"]
        or lane_key != expected["xai_oauth_lane_key"]
        or record_identity != expected["xai_oauth_record_identity"]
    ):
        return None
    return expected


__all__ = [
    "XaiOAuthSelectedAccount",
    "bind_xai_oauth_candidate_to_request",
    "build_xai_oauth_selected_account",
    "configured_xai_oauth_records",
    "get_bound_xai_oauth_selected_account",
    "get_or_bind_xai_oauth_selected_account",
    "get_xai_oauth_snapshot_for_selected_account",
    "is_managed_xai_oauth_candidate",
    "select_xai_oauth_account_record",
    "validated_xai_oauth_server_account_metadata",
    "xai_oauth_account_lane_key",
    "xai_oauth_selected_account_metadata",
]
