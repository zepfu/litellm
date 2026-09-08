"""Managed xAI OAuth account binding.

Only server-selected inventory records reach request state. Candidate bodies,
client metadata, and bearer material are never accepted as an account binding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

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
_XAI_OAUTH_DIRECT_ACCOUNT_COOLDOWN_SECONDS = 3 * 60 * 60.0


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


@dataclass
class XaiOAuthDirectAccountTraversal:
    """Bounded account state for one direct managed xAI request."""

    selected_account: XaiOAuthSelectedAccount
    accounts: tuple[XaiOAuthSelectedAccount, ...]
    cooldown_family: Literal["codex", "anthropic"]
    attempted_account_hashes: set[str] = field(default_factory=set)
    generation_reread_account_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.attempted_account_hashes.add(self.selected_account.account_hash)

    @property
    def max_retry_attempts(self) -> int:
        """Allow one same-account reread and one provider attempt per account."""

        return max(1, (2 * len(self.accounts)) - 1)

    def claim_same_account_generation_reread(self) -> bool:
        """Permit at most one changed-generation reread for the active record."""

        account_hash = self.selected_account.account_hash
        if account_hash in self.generation_reread_account_hashes:
            return False
        self.generation_reread_account_hashes.add(account_hash)
        return True

    async def advance(self) -> Optional[XaiOAuthSelectedAccount]:
        """Select the next untraversed account without an active cooldown."""

        for account in self.accounts:
            if account.account_hash in self.attempted_account_hashes:
                continue
            if not await _xai_oauth_direct_account_is_eligible(
                account,
                cooldown_family=self.cooldown_family,
            ):
                continue
            self.attempted_account_hashes.add(account.account_hash)
            self.selected_account = account
            return account
        return None


@dataclass(frozen=True)
class XaiOAuthDirectRetryRecovery:
    """One direct managed xAI recovery action."""

    refreshed_snapshot: Any = field(default=None, repr=False)
    selected_account: Optional[XaiOAuthSelectedAccount] = None


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


def _xai_oauth_direct_cooldown_candidate(
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> dict[str, str]:
    if cooldown_family == "codex":
        route_family = "codex_xai_oauth_responses_adapter"
    elif cooldown_family == "anthropic":
        route_family = "anthropic_xai_oauth_responses_adapter"
    else:
        raise ValueError("Unsupported managed xAI OAuth cooldown family.")
    return {
        "provider": "xai",
        "model": "managed-xai-oauth",
        "route_family": route_family,
        "cooldown_identity_tag": (
            f"direct-managed-xai-oauth:{cooldown_family}"
        ),
    }


def _xai_oauth_direct_account_cooldown_key(
    selected: XaiOAuthSelectedAccount,
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> str:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
        _get_codex_auto_agent_grok_account_quota_lane_cooldown_key,
    )

    cooldown_key = (
        _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(
            _xai_oauth_direct_cooldown_candidate(
                cooldown_family=cooldown_family
            ),
            selected.lane_key,
        )
    )
    if not isinstance(cooldown_key, str) or not cooldown_key:
        raise RuntimeError(
            "Managed xAI OAuth account cooldown key could not be resolved."
        )
    return cooldown_key


async def _xai_oauth_direct_account_cooldown_seconds(
    selected: XaiOAuthSelectedAccount,
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> float:
    cooldown_key = _xai_oauth_direct_account_cooldown_key(
        selected,
        cooldown_family=cooldown_family,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        _get_anthropic_auto_agent_active_cooldown_state,
        _get_codex_auto_agent_active_cooldown_state,
    )

    if cooldown_family == "codex":
        seconds, _source = (
            await _get_codex_auto_agent_active_cooldown_state(cooldown_key)
        )
    elif cooldown_family == "anthropic":
        seconds, _source = (
            await _get_anthropic_auto_agent_active_cooldown_state(cooldown_key)
        )
    else:
        raise ValueError("Unsupported managed xAI OAuth cooldown family.")
    return max(0.0, float(seconds))


async def _xai_oauth_direct_account_is_eligible(
    selected: XaiOAuthSelectedAccount,
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> bool:
    return (
        await _xai_oauth_direct_account_cooldown_seconds(
            selected,
            cooldown_family=cooldown_family,
        )
        <= 0.0
    )


async def _select_xai_oauth_direct_eligible_account(
    accounts: tuple[XaiOAuthSelectedAccount, ...],
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> XaiOAuthSelectedAccount:
    for account in accounts:
        if await _xai_oauth_direct_account_is_eligible(
            account,
            cooldown_family=cooldown_family,
        ):
            return account
    raise ValueError(
        "No enabled managed xAI OAuth account is eligible while account "
        "quota cooldowns are active."
    )


async def build_xai_oauth_direct_account_traversal(
    *,
    cooldown_family: Literal["codex", "anthropic"],
    request: Optional[Request] = None,
) -> XaiOAuthDirectAccountTraversal:
    """Select an eligible direct record and retain bounded inventory order."""

    accounts = tuple(
        build_xai_oauth_selected_account(record)
        for record in configured_xai_oauth_records()
    )
    selected = (
        get_bound_xai_oauth_selected_account(request)
        if request is not None
        else None
    )
    if selected is None:
        selected = await _select_xai_oauth_direct_eligible_account(
            accounts,
            cooldown_family=cooldown_family,
        )
        if request is not None:
            bind_xai_oauth_selected_account_to_request(request, selected)
    if not any(
        account.account_hash == selected.account_hash for account in accounts
    ):
        raise ValueError("Selected xAI OAuth account is not in the inventory.")
    return XaiOAuthDirectAccountTraversal(
        selected_account=selected,
        accounts=accounts,
        cooldown_family=cooldown_family,
    )


def bind_xai_oauth_selected_account_to_request(
    request: Request,
    selected: XaiOAuthSelectedAccount,
) -> None:
    """Replace request state only with an inventory-proven selected account."""

    expected = build_xai_oauth_selected_account(
        select_xai_oauth_account_record(label=selected.label)
    )
    if expected != selected:
        raise ValueError("Selected xAI OAuth account identity is invalid.")
    setattr(request.state, _XAI_OAUTH_SELECTED_ACCOUNT_STATE, selected)


def _direct_xai_oauth_rollover_body_is_fresh(
    request_body: Mapping[str, Any],
) -> bool:
    """Reject continuation, reasoning, and account-bound request state."""

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
        _aawm_auto_agent_audit_request_has_account_bound_state,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity import (
        is_replay_safe_session_owner_redispatch_body,
    )

    return bool(
        is_replay_safe_session_owner_redispatch_body(request_body)
        and not _aawm_auto_agent_audit_request_has_account_bound_state(
            request_body
        )
    )


async def _publish_xai_oauth_direct_account_quota_cooldown(
    selected: XaiOAuthSelectedAccount,
    *,
    cooldown_family: Literal["codex", "anthropic"],
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
        _persist_anthropic_cooldown_durable,
        _persist_codex_cooldown_durable,
        execute_cooldown_publication_transaction,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        _publish_anthropic_cooldown_memory,
        _publish_codex_cooldown_memory,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
        CooldownPublicationPlan,
    )

    cooldown_key = _xai_oauth_direct_account_cooldown_key(
        selected,
        cooldown_family=cooldown_family,
    )
    plan = CooldownPublicationPlan(
        memory_keys=(cooldown_key,),
        durable_keys=(cooldown_key,),
        duration_seconds=_XAI_OAUTH_DIRECT_ACCOUNT_COOLDOWN_SECONDS,
        applied_scope="candidate",
        grok_account_quota_exhausted=True,
    )
    if cooldown_family == "codex":
        publish_cooldown_memory_fn = _publish_codex_cooldown_memory
        persist_cooldown_fn = _persist_codex_cooldown_durable
    elif cooldown_family == "anthropic":
        publish_cooldown_memory_fn = _publish_anthropic_cooldown_memory
        persist_cooldown_fn = _persist_anthropic_cooldown_durable
    else:
        raise ValueError("Unsupported managed xAI OAuth cooldown family.")
    await execute_cooldown_publication_transaction(
        alias_family=cooldown_family,
        candidate=_xai_oauth_direct_cooldown_candidate(
            cooldown_family=cooldown_family
        ),
        plan=plan,
        publish_cooldown_memory_fn=publish_cooldown_memory_fn,
        persist_cooldown_fn=persist_cooldown_fn,
    )


async def recover_xai_oauth_direct_request(
    *,
    traversal: XaiOAuthDirectAccountTraversal,
    request_body: Mapping[str, Any],
    exc: BaseException,
    snapshot: Any,
    api_base: Optional[str],
) -> Optional[XaiOAuthDirectRetryRecovery]:
    """Choose one same-account reread or fresh-request account rollover."""

    from litellm.llms.xai.oauth import (
        get_xai_oauth_exception_status_code,
        is_xai_oauth_direct_account_quota_failure,
        is_xai_oauth_direct_rollover_failure,
        is_xai_oauth_precommit_provider_401,
        reread_xai_oauth_snapshot_after_provider_401,
    )

    status_code = get_xai_oauth_exception_status_code(exc)
    if (
        status_code == 401
        and snapshot is not None
        and is_xai_oauth_precommit_provider_401(exc, api_base=api_base)
        and traversal.claim_same_account_generation_reread()
    ):
        refreshed_snapshot = await reread_xai_oauth_snapshot_after_provider_401(
            snapshot,
            exc,
            api_base=api_base,
        )
        if refreshed_snapshot is not None:
            return XaiOAuthDirectRetryRecovery(
                refreshed_snapshot=refreshed_snapshot
            )

    if is_xai_oauth_direct_account_quota_failure(
        exc,
        api_base=api_base,
    ):
        await _publish_xai_oauth_direct_account_quota_cooldown(
            traversal.selected_account,
            cooldown_family=traversal.cooldown_family,
        )

    if not (
        is_xai_oauth_direct_rollover_failure(exc, api_base=api_base)
        and _direct_xai_oauth_rollover_body_is_fresh(request_body)
    ):
        return None
    selected_account = await traversal.advance()
    if selected_account is None:
        return None
    return XaiOAuthDirectRetryRecovery(selected_account=selected_account)


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
    "XaiOAuthDirectAccountTraversal",
    "XaiOAuthDirectRetryRecovery",
    "XaiOAuthSelectedAccount",
    "bind_xai_oauth_selected_account_to_request",
    "bind_xai_oauth_candidate_to_request",
    "build_xai_oauth_direct_account_traversal",
    "build_xai_oauth_selected_account",
    "configured_xai_oauth_records",
    "get_bound_xai_oauth_selected_account",
    "get_or_bind_xai_oauth_selected_account",
    "get_xai_oauth_snapshot_for_selected_account",
    "is_managed_xai_oauth_candidate",
    "select_xai_oauth_account_record",
    "recover_xai_oauth_direct_request",
    "validated_xai_oauth_server_account_metadata",
    "xai_oauth_account_lane_key",
    "xai_oauth_selected_account_metadata",
]
