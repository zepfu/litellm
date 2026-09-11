"""Persist reconstructed ChatGPT history attempts into PostgreSQL.

The sidecar uses this adapter after native observation returns metadata-only
index/detail/message pages. Reconstructed attempts are not treated as provider
quota charges. Cookies, tokens, titles, prompts, and raw headers never enter
persisted rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from .models import AttemptRecord, MessageRecord
from .pg_ledger import (
    IngestContext,
    LedgerBinding,
    LedgerScope,
    PgLedger,
    UsageCounts,
)
from .privacy import ADAPTER_VERSION, SURFACE_CHAT, assert_no_secrets
from .reconstruct import reconstruct_attempts
from .timeutil import parse_datetime


DEFAULT_MAPPING_VERSION = "native-history-v1"
DEFAULT_MAPPING_RULES: tuple[dict[str, object], ...] = (
    {"slug": "gpt-5.6-astra-pro", "family": "astra_pro"},
    {"slug": "gpt-6-sol-pro", "family": "sol_pro"},
    {"slug": "gpt-6-pro", "family": "astra_pro"},
    {"slug": "gpt-test", "family": "astra_pro"},
)


class NativeHistoryIngestError(RuntimeError):
    """Raised when history-backed ledger ingest cannot proceed honestly."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def message_record_from_projection(item: Mapping[str, Any]) -> Optional[MessageRecord]:
    conversation_id = str(item.get("conversation_id") or "").strip()
    message_id = str(item.get("message_id") or "").strip()
    if not conversation_id or not message_id:
        return None
    children_raw = item.get("children") or ()
    children = tuple(
        str(child)
        for child in children_raw
        if isinstance(child, str) and child.strip()
    )
    return MessageRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        node_id=item.get("node_id") if isinstance(item.get("node_id"), str) else message_id,
        parent_id=item.get("parent_id") if isinstance(item.get("parent_id"), str) else None,
        children=children,
        role=item.get("role") if isinstance(item.get("role"), str) else None,
        channel=item.get("channel") if isinstance(item.get("channel"), str) else None,
        created_at=parse_datetime(item.get("created_at")),
        status=item.get("status") if isinstance(item.get("status"), str) else None,
        end_turn=item.get("end_turn") if isinstance(item.get("end_turn"), bool) else None,
        requested_model_raw=item.get("requested_model_raw")
        if isinstance(item.get("requested_model_raw"), str)
        else None,
        requested_mode_raw=item.get("requested_mode_raw")
        if isinstance(item.get("requested_mode_raw"), str)
        else None,
        requested_reasoning_effort_raw=item.get("requested_reasoning_effort_raw")
        if isinstance(item.get("requested_reasoning_effort_raw"), str)
        else None,
        recorded_final_model_raw=item.get("recorded_final_model_raw")
        if isinstance(item.get("recorded_final_model_raw"), str)
        else None,
        generation_id=item.get("generation_id")
        if isinstance(item.get("generation_id"), str)
        else None,
        request_id=item.get("request_id") if isinstance(item.get("request_id"), str) else None,
        surface=str(item.get("surface") or SURFACE_CHAT),
        origin=item.get("origin") if isinstance(item.get("origin"), str) else None,
        metadata={},
    )


def messages_from_history_pages(
    pages: Sequence[Mapping[str, Any]],
) -> list[MessageRecord]:
    records: dict[str, MessageRecord] = {}
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        conversation_id = page.get("conversation_id")
        for item in page.get("messages") or ():
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            if conversation_id and not payload.get("conversation_id"):
                payload["conversation_id"] = conversation_id
            record = message_record_from_projection(payload)
            if record is None:
                continue
            records[record.message_id] = record
    return list(records.values())


def reconstruct_attempts_from_history_pages(
    pages: Sequence[Mapping[str, Any]],
    *,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
    mapping_rules: Sequence[Mapping[str, object]] = DEFAULT_MAPPING_RULES,
) -> list[AttemptRecord]:
    by_conversation: dict[str, list[MessageRecord]] = {}
    for record in messages_from_history_pages(pages):
        by_conversation.setdefault(record.conversation_id, []).append(record)
    attempts: list[AttemptRecord] = []
    for conversation_id, messages in by_conversation.items():
        attempts.extend(
            reconstruct_attempts(
                messages,
                mapping_version=mapping_version,
                mapping_rules=mapping_rules,
                conversation_id=conversation_id,
            )
        )
    return attempts


def ledger_scope_for_account(
    collector_account_id: str,
    *,
    provider_user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    quota_owner_id: Optional[str] = None,
    surface: str = SURFACE_CHAT,
) -> LedgerScope:
    account_token = collector_account_id.strip()
    return LedgerScope(
        collector_account_id=account_token,
        provider="chatgpt",
        provider_user_id=provider_user_id or account_token,
        workspace_id=workspace_id or account_token,
        quota_owner_id=quota_owner_id or account_token,
        surface=surface,
    )


def persist_reconstructed_attempts(
    ledger: PgLedger,
    *,
    collector_account_id: str,
    pages: Sequence[Mapping[str, Any]],
    observed_at: Optional[datetime] = None,
    run_id: Optional[str] = None,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
    mapping_rules: Sequence[Mapping[str, object]] = DEFAULT_MAPPING_RULES,
    expected_binding: Optional[LedgerBinding] = None,
    provider_user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    quota_owner_id: Optional[str] = None,
) -> dict[str, Any]:
    """Reconstruct attempts from native pages and upsert them idempotently."""

    seen_at = observed_at or _utc_now()
    ingest_run_id = run_id or str(uuid4())
    scope = ledger_scope_for_account(
        collector_account_id,
        provider_user_id=provider_user_id,
        workspace_id=workspace_id,
        quota_owner_id=quota_owner_id,
    )
    attempts = reconstruct_attempts_from_history_pages(
        pages,
        mapping_version=mapping_version,
        mapping_rules=mapping_rules,
    )
    assert_no_secrets(
        {
            "collector_account_id": collector_account_id,
            "attempts": [
                {
                    "attempt_id": attempt.attempt_id,
                    "requested_model_raw": attempt.requested_model_raw,
                    "recorded_final_model_raw": attempt.recorded_final_model_raw,
                    "surface": attempt.surface,
                }
                for attempt in attempts
            ],
        }
    )
    statuses: list[str] = []
    binding = expected_binding
    if binding is None:
        binding = ledger.initialize_binding(scope, seen_at=seen_at)
    for attempt in attempts:
        context = IngestContext(
            run_id=ingest_run_id,
            observed_at=seen_at,
            source_kind="native_history",
            source_id=attempt.attempt_id,
            schema_version=ADAPTER_VERSION,
            provenance={
                "collector": "chatgpt_native_history",
                "adapter_version": ADAPTER_VERSION,
                "surface": SURFACE_CHAT,
            },
        )
        result = ledger.upsert_attempt(
            scope,
            attempt,
            context,
            expected_binding=binding,
        )
        statuses.append(result.status)
        binding = ledger.capture_binding(scope, seen_at=seen_at)
    counts = ledger.count_attempts(collector_account_id)
    return {
        "collector_account_id": collector_account_id,
        "attempt_count": len(attempts),
        "inserted": sum(1 for status in statuses if status == "inserted"),
        "deduplicated": sum(1 for status in statuses if status == "deduplicated"),
        "statuses": statuses,
        "counts": counts,
        "quota_charge": False,
    }


def persist_coverage_gap(
    ledger: PgLedger,
    *,
    collector_account_id: str,
    reason: str,
    source_id: str,
    details: Optional[Mapping[str, Any]] = None,
    observed_at: Optional[datetime] = None,
    source_kind: str = "native_history",
) -> dict[str, Any]:
    """Record an explicit ledger coverage gap for one collector account."""

    seen_at = observed_at or _utc_now()
    scope = ledger_scope_for_account(collector_account_id)
    assert_no_secrets(
        {
            "collector_account_id": collector_account_id,
            "reason": reason,
            "source_id": source_id,
            "source_kind": source_kind,
        }
    )
    binding = ledger.initialize_binding(scope, seen_at=seen_at)
    gap_id = ledger.record_coverage_gap(
        scope,
        source_kind=source_kind,
        source_id=source_id,
        reason=reason,
        state="open",
        details=details,
        seen_at=seen_at,
        expected_binding=binding,
    )
    return {
        "collector_account_id": collector_account_id,
        "reason": reason,
        "source_id": source_id,
        "gap_id": gap_id,
        "quota_charge": False,
    }


def count_attempts_for_account(
    ledger: PgLedger,
    collector_account_id: str,
    *,
    model_family: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
) -> UsageCounts:
    return ledger.count_attempts(
        collector_account_id,
        model_family=model_family,
        window_start=window_start,
        window_end=window_end,
    )
