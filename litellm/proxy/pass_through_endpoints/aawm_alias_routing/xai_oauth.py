"""Managed xAI OAuth account binding.

Only server-selected inventory records reach request state. Candidate bodies,
client metadata, and bearer material are never accepted as an account binding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
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
_XAI_OAUTH_SELECTED_ACCOUNT_CONTEXTS_STATE = (
    "aawm_xai_oauth_selected_account_contexts"
)
_XAI_OAUTH_CANDIDATE_IDENTITY_FIELDS = (
    "provider",
    "model",
    "route_family",
    "xai_oauth_account_label",
    "xai_oauth_account_hash",
    "xai_oauth_scope_identity",
    "xai_oauth_lane_key",
)
_XAI_OAUTH_MANAGED_ROUTE_FAMILIES = frozenset(
    {
        "codex_xai_oauth_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "codex_auto_agent_xai_oauth_responses",
    }
)
_XAI_OAUTH_DIRECT_OWNER_ROUTE_FAMILIES = frozenset(
    {
        "xai_oauth",
        "xai_oauth_api",
    }
)
_XAI_OAUTH_CONTINUATION_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "item_reference",
        "mcp_approval_request",
        "mcp_approval_response",
        "mcp_call",
        "reasoning",
        "tool_result",
        "tool_use",
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
    """Build a server-owned identity for the selected credential record."""

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


def _xai_oauth_candidate_context_key(
    candidate: Mapping[str, Any],
) -> Optional[tuple[str, ...]]:
    values = tuple(
        _clean_string(candidate.get(field))
        for field in _XAI_OAUTH_CANDIDATE_IDENTITY_FIELDS
    )
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _selected_account_matches_candidate(
    candidate: Mapping[str, Any],
    selected: XaiOAuthSelectedAccount,
) -> bool:
    return all(
        (
            _clean_string(candidate.get("xai_oauth_account_label"))
            == selected.label,
            _clean_string(candidate.get("xai_oauth_account_hash"))
            == selected.account_hash,
            _clean_string(candidate.get("xai_oauth_scope_identity"))
            == selected.scope_identity,
            _clean_string(candidate.get("xai_oauth_lane_key"))
            == selected.lane_key,
        )
    )


def _same_xai_oauth_selected_account(
    left: XaiOAuthSelectedAccount,
    right: XaiOAuthSelectedAccount,
) -> bool:
    return (
        left.label == right.label
        and left.account_hash == right.account_hash
        and left.scope_identity == right.scope_identity
        and left.lane_key == right.lane_key
        and left.record.auth_path == right.record.auth_path
        and left.record.scope == right.record.scope
        and left.record.expected_account_identity
        == right.record.expected_account_identity
    )


def _xai_oauth_snapshot_matches_selected_account(
    snapshot: Any,
    selected: XaiOAuthSelectedAccount,
) -> bool:
    """Return whether a request snapshot is trusted for the selected record."""

    if (
        getattr(snapshot, "credential_family", None) != "xai_oauth"
        or getattr(snapshot, "auth_file", None) != selected.record.auth_path
        or getattr(snapshot, "scope", None) != selected.record.scope
    ):
        return False
    expected_identity = _clean_string(
        getattr(selected.record, "expected_account_identity", None)
    )
    if expected_identity is None:
        # Legacy files need no added account metadata; this binds only the
        # configured record, not a proven upstream account.
        return selected.record.legacy
    snapshot_identity = _clean_string(
        getattr(snapshot, "account_identity", None)
    )
    return snapshot_identity == expected_identity


def preserve_xai_oauth_candidate_context(
    request: Any,
    candidate: Mapping[str, Any],
    selected: XaiOAuthSelectedAccount,
    snapshot: Any = None,
) -> None:
    """Keep server-owned account state out of public candidate dictionaries."""

    key = _xai_oauth_candidate_context_key(candidate)
    state = getattr(request, "state", None)
    if key is None or state is None:
        return
    contexts = getattr(state, _XAI_OAUTH_SELECTED_ACCOUNT_CONTEXTS_STATE, None)
    if not isinstance(contexts, dict):
        contexts = {}
        setattr(state, _XAI_OAUTH_SELECTED_ACCOUNT_CONTEXTS_STATE, contexts)
    contexts[key] = (selected, snapshot)


def preserve_xai_oauth_snapshot_refresh_context(
    request: Any,
    snapshot: Any,
) -> None:
    """Persist a trusted reread snapshot for all matching request candidates."""

    selected = get_bound_xai_oauth_selected_account(request)
    if selected is None or not _xai_oauth_snapshot_matches_selected_account(
        snapshot,
        selected,
    ):
        return
    state = getattr(request, "state", None)
    contexts = getattr(state, _XAI_OAUTH_SELECTED_ACCOUNT_CONTEXTS_STATE, None)
    if not isinstance(contexts, dict):
        return
    for key, context in tuple(contexts.items()):
        if not isinstance(context, tuple) or len(context) != 2:
            continue
        context_selected, _context_snapshot = context
        if not isinstance(context_selected, XaiOAuthSelectedAccount):
            continue
        if _same_xai_oauth_selected_account(context_selected, selected):
            contexts[key] = (context_selected, snapshot)


def _get_preserved_xai_oauth_candidate_context(
    request: Any,
    candidate: Mapping[str, Any],
) -> Optional[tuple[XaiOAuthSelectedAccount, Any]]:
    key = _xai_oauth_candidate_context_key(candidate)
    state = getattr(request, "state", None)
    contexts = getattr(state, _XAI_OAUTH_SELECTED_ACCOUNT_CONTEXTS_STATE, None)
    if key is None or not isinstance(contexts, Mapping):
        return None
    context = contexts.get(key)
    if not isinstance(context, tuple) or len(context) != 2:
        return None
    selected, snapshot = context
    if not isinstance(selected, XaiOAuthSelectedAccount):
        return None
    return selected, snapshot


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
    preserved_context = _get_preserved_xai_oauth_candidate_context(
        request,
        candidate,
    )
    if preserved_context is not None:
        selected, snapshot = preserved_context
        if not _selected_account_matches_candidate(candidate, selected):
            raise HTTPException(
                status_code=500,
                detail="Selected xAI OAuth account identity is invalid.",
            )
        if snapshot is not None:
            if not _xai_oauth_snapshot_matches_selected_account(
                snapshot,
                selected,
            ):
                raise HTTPException(
                    status_code=500,
                    detail="Selected xAI OAuth snapshot identity is invalid.",
                )
            from litellm.llms.xai.oauth import bind_xai_oauth_snapshot_to_request

            bind_xai_oauth_snapshot_to_request(request, snapshot)
    else:
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
    if (
        selected.record.expected_account_identity is None
        and not selected.record.legacy
    ):
        return None
    try:
        current_record = select_xai_oauth_account_record(label=selected.label)
    except XaiOAuthInventoryError:
        return None
    if (
        current_record.auth_path != selected.record.auth_path
        or current_record.scope != selected.record.scope
        or current_record.legacy != selected.record.legacy
        or (
            current_record.expected_account_identity is not None
            and current_record.expected_account_identity
            != selected.record.expected_account_identity
        )
    ):
        return None
    expected = build_xai_oauth_selected_account(selected.record)
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
    """Load the exact selected record and enforce any configured identity pin."""

    from litellm.llms.xai.oauth import get_xai_oauth_snapshot_for_record

    snapshot = await get_xai_oauth_snapshot_for_record(selected.record)
    if not _xai_oauth_snapshot_matches_selected_account(snapshot, selected):
        raise XaiOAuthIdentityMismatchError(
            "Managed xAI OAuth credential identity does not match the configured "
            f"record '{selected.label}'."
        )
    return snapshot


async def get_or_bind_xai_oauth_selected_account_and_snapshot(
    request: Request,
    *,
    selected_account: Optional[XaiOAuthSelectedAccount] = None,
) -> tuple[XaiOAuthSelectedAccount, Any]:
    """Resolve and bind one account together with its exact credential snapshot."""

    if selected_account is None:
        selected_account = get_or_bind_xai_oauth_selected_account(request)
    if not isinstance(selected_account, XaiOAuthSelectedAccount):
        raise ValueError("Managed xAI OAuth account selection is invalid.")

    from litellm.llms.xai.oauth import (
        bind_xai_oauth_snapshot_to_request,
        clear_xai_oauth_snapshot_from_request,
        get_xai_oauth_snapshot_from_request,
    )

    snapshot = get_xai_oauth_snapshot_from_request(request)
    if (
        snapshot is not None
        and not _xai_oauth_snapshot_matches_selected_account(
            snapshot, selected_account
        )
    ):
        clear_xai_oauth_snapshot_from_request(request)
        snapshot = None

    if snapshot is None:
        snapshot = await get_xai_oauth_snapshot_for_selected_account(
            selected_account
        )

    selected_account = await resolve_xai_oauth_selected_account_identity(
        selected_account,
        snapshot=snapshot,
    )
    setattr(request.state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, selected_account)
    bind_xai_oauth_snapshot_to_request(request, snapshot)
    return selected_account, snapshot


async def resolve_xai_oauth_selected_account_identity(
    selected: XaiOAuthSelectedAccount,
    *,
    snapshot: Any = None,
) -> XaiOAuthSelectedAccount:
    """Use existing account evidence without requiring it in legacy records."""

    loaded_snapshot = (
        snapshot
        if snapshot is not None
        else await get_xai_oauth_snapshot_for_selected_account(selected)
    )
    if not _xai_oauth_snapshot_matches_selected_account(
        loaded_snapshot, selected
    ):
        raise XaiOAuthIdentityMismatchError(
            "Managed xAI OAuth credential identity does not match the configured "
            f"record '{selected.label}'."
        )
    actual_identity = _clean_string(
        getattr(loaded_snapshot, "account_identity", None)
    )
    if (
        selected.record.expected_account_identity is not None
        or actual_identity is None
    ):
        return selected
    return build_xai_oauth_selected_account(
        replace(selected.record, expected_account_identity=actual_identity)
    )


def _xai_oauth_request_has_continuation_state(
    value: Any,
    _seen: Optional[set[int]] = None,
) -> bool:
    """Return whether a direct Responses request carries provider state."""

    if isinstance(value, (dict, list)):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return False
        _seen.add(value_id)

    if isinstance(value, dict):
        for key in (
            "previous_response_id",
            "call_id",
            "tool_call_id",
            "item_id",
        ):
            if value.get(key):
                return True
        item_type = value.get("type")
        if (
            isinstance(item_type, str)
            and item_type.strip().casefold() in _XAI_OAUTH_CONTINUATION_ITEM_TYPES
        ):
            return True
        if value.get("role") == "tool" or value.get("tool_calls"):
            return True
        return any(
            _xai_oauth_request_has_continuation_state(child, _seen)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _xai_oauth_request_has_continuation_state(item, _seen)
            for item in value
        )
    return False


def _raise_xai_oauth_direct_continuation_redispatch(
    *,
    request: Request,
    session_identity: Optional[str],
    cache_key: Optional[str],
    owner_record: Optional[Mapping[str, Any]],
    failure_phase: str,
    mismatch_reason: str,
) -> None:
    """Raise the common 409 owner response without permitting provider I/O."""

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity,
    )

    owner_id = (
        owner_record.get("owner")
        if isinstance(owner_record, Mapping)
        else None
    )
    owner_attributes = (
        session_affinity._owner_attributes(owner_record)
        if isinstance(owner_record, Mapping)
        else {}
    )
    session_affinity.raise_session_owner_redispatch_required(
        session_identity=session_identity,
        candidate={
            "provider": owner_attributes.get("provider"),
            "route_family": owner_attributes.get("route_family"),
            "xai_oauth_account_label": owner_attributes.get("account_label"),
            "xai_oauth_account_hash": owner_attributes.get("account_hash"),
            "xai_oauth_scope_identity": owner_attributes.get("account_scope"),
            "xai_oauth_lane_key": owner_attributes.get("account_lane"),
        },
        failure_phase=failure_phase,
        guard=session_affinity.SessionOwnerGuardResult(
            decision=session_affinity.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=session_identity,
            cache_key=cache_key,
            owner_record=(
                dict(owner_record)
                if isinstance(owner_record, Mapping)
                else None
            ),
            owner_id=owner_id if isinstance(owner_id, str) else None,
            mismatch_reason=mismatch_reason,
            provenance=session_affinity.build_session_owner_provenance(
                session_identity=session_identity,
                decision="redispatch_required",
                owner_record=owner_record,
                owner_id=owner_id if isinstance(owner_id, str) else None,
                mismatch_reason=mismatch_reason,
                cache_key=cache_key,
            ),
        ),
        request=request,
        attempted_provider_call=False,
    )


async def resolve_xai_oauth_direct_continuation_account(
    request: Request,
    request_body: Mapping[str, Any],
) -> Optional[XaiOAuthSelectedAccount]:
    """Require an established server association before direct continuation use."""

    if not _xai_oauth_request_has_continuation_state(request_body):
        return None

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity,
    )

    session_identity = session_affinity.resolve_canonical_session_identity(
        request,
        request_body,
    )
    if session_identity is None:
        _raise_xai_oauth_direct_continuation_redispatch(
            request=request,
            session_identity=None,
            cache_key=None,
            owner_record=None,
            failure_phase="xai_direct_continuation_session_identity_missing",
            mismatch_reason="session_owner: missing canonical session identity",
        )

    owner_record, cache_key, owner_error = (
        await session_affinity.get_session_owner_record(
            session_identity=session_identity,
            request=request,
            wait_for_foreign_reservation=False,
        )
    )
    if owner_error is not None:
        _raise_xai_oauth_direct_continuation_redispatch(
            request=request,
            session_identity=session_identity,
            cache_key=cache_key,
            owner_record=owner_record,
            failure_phase="xai_direct_continuation_owner_redis_unavailable",
            mismatch_reason=owner_error,
        )
    if owner_record is None:
        _raise_xai_oauth_direct_continuation_redispatch(
            request=request,
            session_identity=session_identity,
            cache_key=cache_key,
            owner_record=None,
            failure_phase="xai_direct_continuation_owner_missing",
            mismatch_reason="session_owner: durable owner record is missing",
        )

    owner_state = session_affinity._record_state(owner_record)
    if owner_state != "owned":
        failure_phase = (
            "xai_direct_continuation_owner_reserved"
            if owner_state == "reserved"
            else "xai_direct_continuation_owner_malformed"
        )
        mismatch_reason = (
            "session_owner: session has an active competing reservation"
            if owner_state == "reserved"
            else "session_owner: malformed ownership state"
        )
        _raise_xai_oauth_direct_continuation_redispatch(
            request=request,
            session_identity=session_identity,
            cache_key=cache_key,
            owner_record=owner_record,
            failure_phase=failure_phase,
            mismatch_reason=mismatch_reason,
        )

    owner_attributes = session_affinity._owner_attributes(owner_record)
    expected_route = _clean_string(owner_attributes.get("route_family"))
    if (
        _clean_string(owner_attributes.get("provider")) != "xai"
        or expected_route not in _XAI_OAUTH_DIRECT_OWNER_ROUTE_FAMILIES
        or _clean_string(owner_attributes.get("endpoint_contract"))
        != "openai_responses"
        or _clean_string(owner_attributes.get("state_format"))
        != "openai_responses"
    ):
        _raise_xai_oauth_direct_continuation_redispatch(
            request=request,
            session_identity=session_identity,
            cache_key=cache_key,
            owner_record=owner_record,
            failure_phase="xai_direct_continuation_owner_route_mismatch",
            mismatch_reason=(
                "session_owner: durable owner is not the managed xAI OAuth "
                "Responses route"
            ),
        )

    # Session ownership is keyed by canonical session and currently carries no
    # exact submitted continuation identifier. It cannot prove that this
    # provider-owned response belongs to the durable xAI account, so do not
    # select inventory or load credentials as a fallback.
    _raise_xai_oauth_direct_continuation_redispatch(
        request=request,
        session_identity=session_identity,
        cache_key=cache_key,
        owner_record=owner_record,
        failure_phase="xai_direct_continuation_owner_association_missing",
        mismatch_reason=(
            "session_owner: durable xAI OAuth owner is not associated with "
            "the submitted continuation"
        ),
    )


def xai_oauth_selected_account_metadata(
    selected: XaiOAuthSelectedAccount,
) -> dict[str, str | bool]:
    """Return the bounded server-derived metadata allowed into observations."""

    metadata: dict[str, str | bool] = {
        "xai_oauth_server_account_binding": True,
        "xai_oauth_account_label": selected.label,
        "xai_oauth_account_hash": selected.account_hash,
        "xai_oauth_lane_key": selected.lane_key,
        "xai_oauth_record_identity": selected.account_hash,
        "xai_oauth_scope_identity": selected.scope_identity,
    }
    return metadata


def validated_xai_oauth_server_account_metadata(
    metadata: Mapping[str, Any],
    *,
    request: Any = None,
) -> Optional[dict[str, str | bool]]:
    """Return request-bound observation metadata or reject the payload."""

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
    if request is None:
        return None
    from litellm.llms.xai.oauth import get_xai_oauth_snapshot_from_request

    selected = get_bound_xai_oauth_selected_account(request)
    snapshot = get_xai_oauth_snapshot_from_request(request)
    if selected is None or snapshot is None:
        return None
    if (
        not _xai_oauth_snapshot_matches_selected_account(snapshot, selected)
        or label != selected.label
    ):
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
    "get_or_bind_xai_oauth_selected_account_and_snapshot",
    "get_xai_oauth_snapshot_for_selected_account",
    "is_managed_xai_oauth_candidate",
    "preserve_xai_oauth_candidate_context",
    "preserve_xai_oauth_snapshot_refresh_context",
    "resolve_xai_oauth_direct_continuation_account",
    "resolve_xai_oauth_selected_account_identity",
    "select_xai_oauth_account_record",
    "validated_xai_oauth_server_account_metadata",
    "xai_oauth_account_lane_key",
    "xai_oauth_selected_account_metadata",
]
