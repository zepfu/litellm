"""Sidecar preparation callbacks for the unverified native history contract.

Synthetic adapter fixtures and the structural feasibility observer do not
establish authenticated identity, detail, or pagination contracts. These
callbacks therefore acquire no browser resources and return no history data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .pg_ledger import LedgerScope
from .ts_bridge import BridgeOperationContext, BridgeProtocolError


def bootstrap_native_history_binding(
    scope: LedgerScope,
    seen_at: datetime,
    operation: BridgeOperationContext,
) -> LedgerScope:
    """Do not initialize a ledger binding from configured identity alone."""

    operation.check()
    raise BridgeProtocolError(
        "native session identity contract is not verified",
        code="history_contract_unavailable",
        retryable=True,
    )


def prepare_native_history(
    scope: LedgerScope,
    collector_account_id: str,
    profile_id: str,
    operation: BridgeOperationContext,
) -> Mapping[str, Any]:
    """Keep counting unavailable until identity-bound history is observed."""

    operation.check()
    raise BridgeProtocolError(
        "native history index, detail, and pagination contracts are not verified",
        code="history_contract_unavailable",
        retryable=True,
    )
