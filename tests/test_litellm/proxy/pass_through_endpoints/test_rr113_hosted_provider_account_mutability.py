"""RR-113: account mutability is OpenAI-only, not every hosted_provider match.

OPENAI-020 keeps OpenAI account/model continuation compatible. xAI (and other
non-OpenAI hosted providers) must still treat account A vs B as a mismatch in
both exact and relaxed compatibility modes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    session_affinity as sa,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod

from test_openai020_session_owner_model_switch import (
    SID,
    _FakeRedisCache,
    _own_on_request,
    _openai_attrs,
    _patch_dual,
    _provider_attrs,
    _request,
)


def _owned_record(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": "owned",
        "owner": sa.build_session_owner_id(attributes=attributes),
        "attributes": attributes,
    }


def test_rr113_openai_same_hosted_provider_accounts_remain_compatible_as_openai020() -> None:
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _openai_attrs(model="gpt-5.4", account="account2")
    assert sa._hosted_providers_match(owner, requested)
    assert sa._attributes_exactly_equal(left=owner, right=requested)
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=True,
    ) is None
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=False,
    ) is None
    assert sa.build_session_owner_id(attributes=owner) == sa.build_session_owner_id(
        attributes=requested
    )


def test_rr113_xai_account_a_vs_b_is_not_equal_just_because_hosted_provider_matches() -> None:
    owner = _provider_attrs(provider="xai", model="grok-4", account="account-a")
    requested = _provider_attrs(provider="xai", model="grok-4", account="account-b")
    assert sa._hosted_provider_from_attributes(owner) == "xai"
    assert sa._hosted_providers_match(owner, requested)
    assert not sa._attributes_exactly_equal(left=owner, right=requested)
    exact_reason = sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=True,
    )
    relaxed_reason = sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=False,
    )
    assert exact_reason is not None
    assert relaxed_reason is not None
    assert "account" in (relaxed_reason or "").lower()


def test_rr113_moonshot_account_a_vs_b_is_not_equal_just_because_hosted_provider_matches() -> None:
    owner = _provider_attrs(
        provider="moonshot",
        model="kimi-k2.5",
        account="account-a",
        route_family="moonshot_openai",
        endpoint_contract="openai_chat",
        state_format="openai_chat",
    )
    requested = _provider_attrs(
        provider="moonshot",
        model="kimi-k2.5",
        account="account-b",
        route_family="moonshot_openai",
        endpoint_contract="openai_chat",
        state_format="openai_chat",
    )
    assert sa._hosted_provider_from_attributes(owner) == "moonshot"
    assert sa._hosted_providers_match(owner, requested)
    assert not sa._attributes_exactly_equal(left=owner, right=requested)
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=True,
    ) is not None
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(owner),
        requested_attributes=requested,
        require_exact_attributes=False,
    ) is not None


@pytest.mark.asyncio
async def test_rr113_xai_account_switch_requires_redispatch_in_strict_and_relaxed_modes() -> None:
    redis = _FakeRedisCache()
    owner = _provider_attrs(provider="xai", model="grok-4", account="account-a")
    requested = _provider_attrs(provider="xai", model="grok-4", account="account-b")
    request = _request()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "grok-4"},
        )
        for exact in (True, False):
            resume = _request()
            second = await sa.ensure_session_owner_guard_for_request(
                request=resume,
                request_body={"model": "grok-4", "previous_response_id": "resp_prev"},
                session_identity=SID,
                requested_attributes=requested,
                require_exact_attributes=exact,
                raise_on_redispatch=False,
            )
            assert second.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED, (
                f"xAI A vs B must redispatch when require_exact_attributes={exact}"
            )


@pytest.mark.asyncio
async def test_rr113_openai_account_switch_stays_compatible_on_canonical_identity() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _openai_attrs(model="gpt-5.4", account="account2")
    request = _request()
    body = {"model": "gpt-5.4", "previous_response_id": "resp_prev"}
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.4"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body=body,
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
        assert second.session_identity == SID
